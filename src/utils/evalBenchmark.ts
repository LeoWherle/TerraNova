/**
 * evalBenchmark.ts — Compare JS Worker vs Rust IPC evaluation performance.
 *
 * Usage (from browser dev console):
 *
 *   import { runEvalBenchmark } from "@/utils/evalBenchmark";
 *   await runEvalBenchmark();              // default: 5 iterations, 128 resolution
 *   await runEvalBenchmark({ iterations: 10, resolutions: [64, 128, 256] });
 *   await runEvalBenchmark({ includeVolume: true });
 *
 * Or call from a React component / debug panel:
 *
 *   const results = await runEvalBenchmark({ resolutions: [64, 128] });
 *   console.table(results.summary);
 *
 * This benchmark measures the FULL end-to-end time for each path:
 *   - JS Worker: postMessage → worker evaluates → onmessage with Float32Array
 *   - Rust IPC:  invoke() → Tauri serializes request JSON → Rust deserializes →
 *                evaluates → Rust serializes response JSON → Tauri deserializes →
 *                JS converts number[] to Float32Array
 *
 * Both paths include all serialization, deserialization, and data conversion
 * overhead that occurs in real usage.
 */

import type { Node, Edge } from "@xyflow/react";
import { evaluateInWorker, cancelEvaluation } from "@/utils/densityWorkerClient";
import { evaluateGrid, evaluateVolume } from "@/utils/ipc";

// ── Benchmark graph factories ─────────────────────────────────────

/** Minimal node shape matching what both evaluators expect. */
function makeNode(id: string, densityType: string, fields: Record<string, unknown>): Node {
  return {
    id,
    type: "densityNode",
    position: { x: 0, y: 0 },
    data: { type: densityType, fields },
  };
}

function makeEdge(source: string, target: string, targetHandle: string): Edge {
  return {
    id: `${source}->${target}`,
    source,
    target,
    targetHandle,
  };
}

interface BenchGraph {
  label: string;
  nodes: Node[];
  edges: Edge[];
  rootNodeId: string;
}

function graphConstant(): BenchGraph {
  return {
    label: "constant",
    nodes: [makeNode("c", "Constant", { Value: 42 })],
    edges: [],
    rootNodeId: "c",
  };
}

function graphNoise2D(): BenchGraph {
  return {
    label: "simplex_2d",
    nodes: [makeNode("n", "SimplexNoise2D", { Frequency: 0.01, Seed: 12345 })],
    edges: [],
    rootNodeId: "n",
  };
}

function graphFractal2D(): BenchGraph {
  return {
    label: "fractal_2d_4oct",
    nodes: [
      makeNode("fn", "FractalNoise2D", {
        Frequency: 0.005,
        Octaves: 4,
        Lacunarity: 2.0,
        Gain: 0.5,
        Seed: 42,
      }),
    ],
    edges: [],
    rootNodeId: "fn",
  };
}

function graphFractal2D6Oct(): BenchGraph {
  return {
    label: "fractal_2d_6oct",
    nodes: [
      makeNode("fn", "FractalNoise2D", {
        Frequency: 0.005,
        Octaves: 6,
        Lacunarity: 2.0,
        Gain: 0.5,
        Seed: 999,
      }),
    ],
    edges: [],
    rootNodeId: "fn",
  };
}

function graphRealisticTerrain(): BenchGraph {
  const nodes: Node[] = [
    makeNode("yg", "YGradient", {
      FromY: 0,
      ToY: 256,
      FromValue: 1,
      ToValue: -1,
    }),
    makeNode("fn", "FractalNoise2D", {
      Frequency: 0.01,
      Octaves: 4,
      Lacunarity: 2.0,
      Gain: 0.5,
      Seed: 42,
    }),
    makeNode("amp", "AmplitudeConstant", { Value: 30 }),
    makeNode("sum", "Sum", {}),
    makeNode("clamp", "Clamp", { Min: -1, Max: 1 }),
    makeNode("lt", "LinearTransform", {
      InputMin: -1,
      InputMax: 1,
      OutputMin: 0,
      OutputMax: 1,
    }),
  ];
  const edges: Edge[] = [
    makeEdge("fn", "amp", "Input"),
    makeEdge("yg", "sum", "Inputs[0]"),
    makeEdge("amp", "sum", "Inputs[1]"),
    makeEdge("sum", "clamp", "Input"),
    makeEdge("clamp", "lt", "Input"),
  ];
  return { label: "realistic_terrain", nodes, edges, rootNodeId: "lt" };
}

function graphComplexMultiNoise(): BenchGraph {
  const nodes: Node[] = [
    makeNode("n1", "FractalNoise2D", {
      Frequency: 0.005,
      Octaves: 5,
      Lacunarity: 2.0,
      Gain: 0.5,
      Seed: 100,
    }),
    makeNode("n2", "SimplexNoise2D", { Frequency: 0.05, Seed: 200 }),
    makeNode("n3", "SimplexNoise3D", { Frequency: 0.03, Seed: 300 }),
    makeNode("yg", "YGradient", {
      FromY: 0,
      ToY: 256,
      FromValue: 1,
      ToValue: -1,
    }),
    makeNode("amp1", "AmplitudeConstant", { Value: 40 }),
    makeNode("amp2", "AmplitudeConstant", { Value: 8 }),
    makeNode("amp3", "AmplitudeConstant", { Value: 0.5 }),
    makeNode("sum_terrain", "Sum", {}),
    makeNode("sum_detail", "Sum", {}),
    makeNode("cave_clamp", "Clamp", { Min: -0.3, Max: 0.3 }),
    makeNode("blend_factor", "Constant", { Value: 0.3 }),
    makeNode("blend", "Blend", {}),
    makeNode("product_cave", "Product", {}),
    makeNode("sum_final", "Sum", {}),
    makeNode("final_clamp", "Clamp", { Min: -1, Max: 1 }),
  ];
  const edges: Edge[] = [
    makeEdge("n1", "amp1", "Input"),
    makeEdge("yg", "sum_terrain", "Inputs[0]"),
    makeEdge("amp1", "sum_terrain", "Inputs[1]"),
    makeEdge("n2", "amp2", "Input"),
    makeEdge("sum_terrain", "sum_detail", "Inputs[0]"),
    makeEdge("amp2", "sum_detail", "Inputs[1]"),
    makeEdge("n3", "amp3", "Input"),
    makeEdge("amp3", "cave_clamp", "Input"),
    makeEdge("sum_detail", "blend", "InputA"),
    makeEdge("cave_clamp", "blend", "InputB"),
    makeEdge("blend_factor", "blend", "Factor"),
    makeEdge("blend", "product_cave", "Inputs[0]"),
    makeEdge("product_cave", "sum_final", "Inputs[0]"),
    makeEdge("sum_final", "final_clamp", "Input"),
  ];
  return { label: "complex_multi_noise", nodes, edges, rootNodeId: "final_clamp" };
}

const ALL_GRAPHS: (() => BenchGraph)[] = [
  graphConstant,
  graphNoise2D,
  graphFractal2D,
  graphFractal2D6Oct,
  graphRealisticTerrain,
  graphComplexMultiNoise,
];

// ── Types ─────────────────────────────────────────────────────────

export interface BenchmarkOptions {
  /** Number of iterations per test case (default: 5). */
  iterations?: number;
  /** Grid resolutions to test (default: [64, 128]). */
  resolutions?: number[];
  /** Also benchmark volume evaluation (default: false). */
  includeVolume?: boolean;
  /** Volume configs: [resolution, ySlices][] (default: [[16,16],[32,32]]). */
  volumeConfigs?: [number, number][];
  /** Which graphs to test (default: all). */
  graphs?: string[];
  /** Print progress to console (default: true). */
  verbose?: boolean;
}

export interface TimingResult {
  graph: string;
  mode: string;
  resolution: string;
  backend: "js_worker" | "rust_ipc";
  samples: number;
  meanMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
  stddevMs: number;
  samplesPerSec: number;
  totalSamples: number;
}

export interface PairComparison {
  graph: string;
  mode: string;
  resolution: string;
  jsMedianMs: number;
  rustMedianMs: number;
  speedup: string;
  ipcOverheadEstMs: number | null;
}

export interface BenchmarkResult {
  timings: TimingResult[];
  comparisons: PairComparison[];
  summary: PairComparison[];
}

// ── Stats helpers ─────────────────────────────────────────────────

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function mean(arr: number[]): number {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function stddev(arr: number[], avg: number): number {
  if (arr.length < 2) return 0;
  const variance = arr.reduce((s, v) => s + (v - avg) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(variance);
}

function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`;
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// ── Evaluation wrappers ───────────────────────────────────────────

async function timeJsWorkerGrid(
  graph: BenchGraph,
  resolution: number,
  rangeMin: number,
  rangeMax: number,
  yLevel: number,
): Promise<number> {
  const t0 = performance.now();
  const result = await evaluateInWorker({
    nodes: graph.nodes,
    edges: graph.edges,
    resolution,
    rangeMin,
    rangeMax,
    yLevel,
    rootNodeId: graph.rootNodeId,
    options: {},
  });
  const elapsed = performance.now() - t0;
  // Ensure the result is materialized (prevent lazy optimization)
  if (result.values.length !== resolution * resolution) {
    throw new Error(`Unexpected JS result length: ${result.values.length}`);
  }
  return elapsed;
}

async function timeRustIpcGrid(
  graph: BenchGraph,
  resolution: number,
  rangeMin: number,
  rangeMax: number,
  yLevel: number,
): Promise<number> {
  const t0 = performance.now();
  const result = await evaluateGrid({
    nodes: graph.nodes.map((n) => ({
      id: n.id,
      type: n.type,
      data: n.data,
    })),
    edges: graph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      targetHandle: e.targetHandle,
    })),
    resolution,
    range_min: rangeMin,
    range_max: rangeMax,
    y_level: yLevel,
    root_node_id: graph.rootNodeId,
    content_fields: {},
  });
  // Include Float32Array conversion in the measurement — this is real overhead
  const values = new Float32Array(result.values);
  const elapsed = performance.now() - t0;
  if (values.length !== resolution * resolution) {
    throw new Error(`Unexpected Rust result length: ${values.length}`);
  }
  return elapsed;
}

async function timeRustIpcVolume(
  graph: BenchGraph,
  resolution: number,
  ySlices: number,
  rangeMin: number,
  rangeMax: number,
  yMin: number,
  yMax: number,
): Promise<number> {
  const t0 = performance.now();
  const result = await evaluateVolume({
    nodes: graph.nodes.map((n) => ({
      id: n.id,
      type: n.type,
      data: n.data,
    })),
    edges: graph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      targetHandle: e.targetHandle,
    })),
    resolution,
    range_min: rangeMin,
    range_max: rangeMax,
    y_min: yMin,
    y_max: yMax,
    y_slices: ySlices,
    root_node_id: graph.rootNodeId,
    content_fields: {},
  });
  const densities = new Float32Array(result.densities);
  const elapsed = performance.now() - t0;
  if (densities.length !== resolution * resolution * ySlices) {
    throw new Error(`Unexpected Rust volume length: ${densities.length}`);
  }
  return elapsed;
}

// ── Parity check ──────────────────────────────────────────────────

/**
 * Verify that JS Worker and Rust IPC produce the same grid output
 * (within floating-point tolerance).
 */
export async function checkGridParity(
  graph: BenchGraph,
  resolution: number = 16,
  tolerance: number = 0.01,
): Promise<{ match: boolean; maxDiff: number; diffCount: number; total: number }> {
  const rangeMin = -64;
  const rangeMax = 64;
  const yLevel = 64;

  const jsResult = await evaluateInWorker({
    nodes: graph.nodes,
    edges: graph.edges,
    resolution,
    rangeMin,
    rangeMax,
    yLevel,
    rootNodeId: graph.rootNodeId,
    options: {},
  });

  const rustResult = await evaluateGrid({
    nodes: graph.nodes.map((n) => ({ id: n.id, type: n.type, data: n.data })),
    edges: graph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      targetHandle: e.targetHandle,
    })),
    resolution,
    range_min: rangeMin,
    range_max: rangeMax,
    y_level: yLevel,
    root_node_id: graph.rootNodeId,
    content_fields: {},
  });

  const rustValues = new Float32Array(rustResult.values);
  const total = resolution * resolution;
  let maxDiff = 0;
  let diffCount = 0;

  for (let i = 0; i < total; i++) {
    const diff = Math.abs(jsResult.values[i] - rustValues[i]);
    if (diff > maxDiff) maxDiff = diff;
    if (diff > tolerance) diffCount++;
  }

  return { match: diffCount === 0, maxDiff, diffCount, total };
}

// ── IPC overhead estimation ───────────────────────────────────────

/**
 * Estimate IPC overhead by comparing a trivial graph (Constant) at
 * minimal resolution. The evaluation itself is ~0, so the measured
 * time is almost entirely IPC overhead.
 */
async function estimateIpcOverhead(iterations: number): Promise<number> {
  const graph = graphConstant();
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    const result = await evaluateGrid({
      nodes: graph.nodes.map((n) => ({ id: n.id, type: n.type, data: n.data })),
      edges: [],
      resolution: 1,
      range_min: 0,
      range_max: 1,
      y_level: 0,
      root_node_id: graph.rootNodeId,
      content_fields: {},
    });
    // Force materialization
    void new Float32Array(result.values);
    times.push(performance.now() - t0);
  }
  return median(times);
}

// ── Main benchmark runner ─────────────────────────────────────────

export async function runEvalBenchmark(
  options: BenchmarkOptions = {},
): Promise<BenchmarkResult> {
  const {
    iterations = 5,
    resolutions = [64, 128],
    includeVolume = false,
    volumeConfigs = [[16, 16], [32, 32]],
    graphs: graphFilter,
    verbose = true,
  } = options;

  const log = verbose ? console.log.bind(console) : () => { };

  log(
    "%c═══ TerraNova Evaluation Benchmark ═══",
    "font-weight:bold;font-size:14px;color:#4fc3f7",
  );
  log(`Iterations: ${iterations} | Resolutions: ${resolutions.join(", ")}`);
  if (includeVolume) {
    log(`Volume configs: ${volumeConfigs.map(([r, y]) => `${r}×${y}×${r}`).join(", ")}`);
  }
  log("");

  // Estimate baseline IPC overhead
  log("Estimating IPC overhead...");
  let ipcOverheadMs: number | null = null;
  try {
    ipcOverheadMs = await estimateIpcOverhead(Math.max(iterations, 5));
    log(`  IPC baseline (1×1 constant): ${formatMs(ipcOverheadMs)}`);
  } catch {
    log("  (IPC overhead estimation failed — Tauri not available?)");
  }
  log("");

  const selectedGraphs = ALL_GRAPHS
    .map((fn) => fn())
    .filter((g) => !graphFilter || graphFilter.includes(g.label));

  const timings: TimingResult[] = [];
  const comparisons: PairComparison[] = [];

  // ── Grid benchmarks ──
  for (const graph of selectedGraphs) {
    for (const res of resolutions) {
      const rangeMin = -64;
      const rangeMax = 64;
      const yLevel = 64;
      const totalSamples = res * res;
      const resLabel = `${res}×${res}`;

      log(`▸ ${graph.label} @ ${resLabel} grid`);

      // -- Warmup --
      try {
        await timeJsWorkerGrid(graph, res, rangeMin, rangeMax, yLevel);
      } catch {
        log("  JS Worker warmup failed, skipping JS path");
      }

      // -- JS Worker --
      const jsTimes: number[] = [];
      for (let i = 0; i < iterations; i++) {
        try {
          // Cancel any lingering worker state
          cancelEvaluation();
          const t = await timeJsWorkerGrid(graph, res, rangeMin, rangeMax, yLevel);
          jsTimes.push(t);
        } catch (err) {
          log(`  JS iteration ${i} failed: ${err}`);
        }
      }

      if (jsTimes.length > 0) {
        const m = mean(jsTimes);
        const jsResult: TimingResult = {
          graph: graph.label,
          mode: "grid",
          resolution: resLabel,
          backend: "js_worker",
          samples: jsTimes.length,
          meanMs: m,
          medianMs: median(jsTimes),
          minMs: Math.min(...jsTimes),
          maxMs: Math.max(...jsTimes),
          stddevMs: stddev(jsTimes, m),
          samplesPerSec: totalSamples / (m / 1000),
          totalSamples,
        };
        timings.push(jsResult);
        log(
          `    JS Worker:  median=${formatMs(jsResult.medianMs)} ` +
          `min=${formatMs(jsResult.minMs)} max=${formatMs(jsResult.maxMs)} ` +
          `(${jsTimes.length} runs)`,
        );
      }

      // -- Rust IPC --
      const rustTimes: number[] = [];
      for (let i = 0; i < iterations; i++) {
        try {
          const t = await timeRustIpcGrid(graph, res, rangeMin, rangeMax, yLevel);
          rustTimes.push(t);
        } catch (err) {
          if (i === 0) {
            log(`  Rust IPC not available: ${err}`);
            break;
          }
        }
      }

      if (rustTimes.length > 0) {
        const m = mean(rustTimes);
        const rustResult: TimingResult = {
          graph: graph.label,
          mode: "grid",
          resolution: resLabel,
          backend: "rust_ipc",
          samples: rustTimes.length,
          meanMs: m,
          medianMs: median(rustTimes),
          minMs: Math.min(...rustTimes),
          maxMs: Math.max(...rustTimes),
          stddevMs: stddev(rustTimes, m),
          samplesPerSec: totalSamples / (m / 1000),
          totalSamples,
        };
        timings.push(rustResult);
        log(
          `    Rust IPC:   median=${formatMs(rustResult.medianMs)} ` +
          `min=${formatMs(rustResult.minMs)} max=${formatMs(rustResult.maxMs)} ` +
          `(${rustTimes.length} runs)`,
        );
      }

      // -- Comparison --
      if (jsTimes.length > 0 && rustTimes.length > 0) {
        const jsMedian = median(jsTimes);
        const rustMedian = median(rustTimes);
        const speedup = jsMedian / rustMedian;
        const comp: PairComparison = {
          graph: graph.label,
          mode: "grid",
          resolution: resLabel,
          jsMedianMs: jsMedian,
          rustMedianMs: rustMedian,
          speedup: `${speedup.toFixed(2)}×`,
          ipcOverheadEstMs: ipcOverheadMs,
        };
        comparisons.push(comp);

        const arrow = speedup >= 1 ? "🟢" : "🔴";
        log(`    ${arrow} Speedup: ${speedup.toFixed(2)}×`);
        if (ipcOverheadMs !== null) {
          const evalOnlyEst = Math.max(0, rustMedian - ipcOverheadMs);
          const evalSpeedup = evalOnlyEst > 0 ? jsMedian / evalOnlyEst : Infinity;
          log(
            `    📊 IPC overhead: ~${formatMs(ipcOverheadMs)} | ` +
            `Pure eval est: ~${formatMs(evalOnlyEst)} (${evalSpeedup.toFixed(1)}× vs JS)`,
          );
        }
      }
      log("");
    }
  }

  // ── Volume benchmarks ──
  if (includeVolume) {
    log("%c── Volume Benchmarks ──", "font-weight:bold;color:#81c784");

    for (const graph of selectedGraphs) {
      for (const [res, ySlices] of volumeConfigs) {
        const resLabel = `${res}×${ySlices}×${res}`;
        const totalSamples = res * res * ySlices;
        log(`▸ ${graph.label} @ ${resLabel} volume`);

        // Rust IPC only for volume (JS volume worker uses a different API)
        const rustTimes: number[] = [];
        for (let i = 0; i < iterations; i++) {
          try {
            const t = await timeRustIpcVolume(graph, res, ySlices, -64, 64, 0, 256);
            rustTimes.push(t);
          } catch (err) {
            if (i === 0) {
              log(`  Rust IPC not available: ${err}`);
              break;
            }
          }
        }

        if (rustTimes.length > 0) {
          const m = mean(rustTimes);
          const rustResult: TimingResult = {
            graph: graph.label,
            mode: "volume",
            resolution: resLabel,
            backend: "rust_ipc",
            samples: rustTimes.length,
            meanMs: m,
            medianMs: median(rustTimes),
            minMs: Math.min(...rustTimes),
            maxMs: Math.max(...rustTimes),
            stddevMs: stddev(rustTimes, m),
            samplesPerSec: totalSamples / (m / 1000),
            totalSamples,
          };
          timings.push(rustResult);
          log(
            `    Rust IPC:   median=${formatMs(rustResult.medianMs)} ` +
            `min=${formatMs(rustResult.minMs)} max=${formatMs(rustResult.maxMs)} ` +
            `(${rustTimes.length} runs)`,
          );
          log(
            `    Throughput: ${(rustResult.samplesPerSec / 1e6).toFixed(2)}M samples/s`,
          );
        }
        log("");
      }
    }
  }

  // ── Summary table ──
  log("%c═══ Summary ═══", "font-weight:bold;font-size:14px;color:#4fc3f7");
  if (comparisons.length > 0) {
    console.table(
      comparisons.map((c) => ({
        Graph: c.graph,
        Resolution: c.resolution,
        "JS (ms)": c.jsMedianMs.toFixed(2),
        "Rust (ms)": c.rustMedianMs.toFixed(2),
        Speedup: c.speedup,
        "IPC overhead (ms)": c.ipcOverheadEstMs?.toFixed(2) ?? "N/A",
      })),
    );
  } else {
    log("No comparisons available (need both JS and Rust results)");
  }

  return {
    timings,
    comparisons,
    summary: comparisons,
  };
}

/**
 * Quick parity check across all benchmark graphs.
 * Logs whether JS and Rust produce matching results.
 */
export async function runParityCheck(
  resolution: number = 16,
  tolerance: number = 0.01,
): Promise<void> {
  console.log(
    "%c═══ JS/Rust Parity Check ═══",
    "font-weight:bold;font-size:14px;color:#ffb74d",
  );
  console.log(`Resolution: ${resolution}×${resolution} | Tolerance: ${tolerance}`);
  console.log("");

  for (const factory of ALL_GRAPHS) {
    const graph = factory();
    try {
      const result = await checkGridParity(graph, resolution, tolerance);
      const icon = result.match ? "✅" : "❌";
      console.log(
        `${icon} ${graph.label}: maxDiff=${result.maxDiff.toExponential(3)} ` +
        `diffs=${result.diffCount}/${result.total}`,
      );
    } catch (err) {
      console.log(`⚠️  ${graph.label}: ${err}`);
    }
  }
}

// Export graph factories for ad-hoc testing
export const benchmarkGraphs = {
  constant: graphConstant,
  noise2D: graphNoise2D,
  fractal2D: graphFractal2D,
  fractal2D6Oct: graphFractal2D6Oct,
  realisticTerrain: graphRealisticTerrain,
  complexMultiNoise: graphComplexMultiNoise,
};
