/**
 * Parity check utility — compares JS and Rust evaluation results.
 *
 * Dev-only: not shipped in production builds.
 * Used during the Rust evaluation migration to verify that the Rust
 * evaluator produces identical results to the TypeScript evaluator.
 */

import type { Node, Edge } from "@xyflow/react";
import { createEvaluationContext } from "../densityEvaluator";

// NOTE: `invoke` is only available in Tauri runtime context.
// This utility is intended for interactive dev-mode testing, not CI.
// We lazily resolve it on first use via dynamic import to avoid breaking
// non-Tauri environments (e.g. vitest).

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _invoke: ((cmd: string, args?: Record<string, unknown>) => Promise<any>) | null | undefined;

async function getInvoke() {
  if (_invoke !== undefined) return _invoke;
  try {
    const tauri = await import("@tauri-apps/api/core");
    _invoke = tauri.invoke;
  } catch {
    _invoke = null;
  }
  return _invoke;
}

export interface ParityPoint {
  x: number;
  y: number;
  z: number;
}

export interface ParityDiff {
  point: ParityPoint;
  js: number;
  rust: number;
  delta: number;
}

export interface ParityResult {
  match: boolean;
  totalPoints: number;
  mismatches: number;
  diffs: ParityDiff[];
  /** If set, the Rust side returned an error instead of values */
  rustError?: string;
}

/**
 * Serialize a React Flow node for IPC transmission to the Rust backend.
 * Strips UI-only fields (position, measured, etc.) to reduce payload size.
 */
function serializeNode(node: Node): Record<string, unknown> {
  return {
    id: node.id,
    type: node.type,
    data: node.data,
  };
}

/**
 * Serialize a React Flow edge for IPC transmission.
 */
function serializeEdge(edge: Edge): Record<string, unknown> {
  return {
    source: edge.source,
    target: edge.target,
    targetHandle: edge.targetHandle ?? null,
  };
}

/**
 * Compare JS and Rust evaluation of a graph at the given sample points.
 *
 * @param nodes       React Flow nodes
 * @param edges       React Flow edges
 * @param samplePoints Array of (x, y, z) coordinates to evaluate
 * @param tolerance   Maximum allowed absolute difference (default 1e-4)
 * @param rootNodeId  Optional explicit root node ID
 * @param contentFields Optional content fields map
 * @returns Parity comparison result
 */
export async function checkParity(
  nodes: Node[],
  edges: Edge[],
  samplePoints: ParityPoint[],
  tolerance: number = 1e-4,
  rootNodeId?: string,
  contentFields?: Record<string, number>,
): Promise<ParityResult> {
  const invoke = await getInvoke();
  if (!invoke) {
    return {
      match: false,
      totalPoints: samplePoints.length,
      mismatches: samplePoints.length,
      diffs: [],
      rustError: "Tauri invoke not available — not running in Tauri context",
    };
  }

  // ── JS evaluation ──────────────────────────────────────────────────
  const ctx = createEvaluationContext(nodes, edges, rootNodeId, contentFields);
  const jsValues = samplePoints.map((p) => {
    if (!ctx) return 0;
    return ctx.evaluate(ctx.rootId, p.x, p.y, p.z);
  });

  // ── Rust evaluation ────────────────────────────────────────────────
  let rustValues: number[];
  try {
    rustValues = await invoke("evaluate_points", {
      nodes: nodes.map(serializeNode),
      edges: edges.map(serializeEdge),
      points: samplePoints.map((p) => [p.x, p.y, p.z]),
      rootNodeId: rootNodeId ?? null,
      contentFields: contentFields ?? null,
    });
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    return {
      match: false,
      totalPoints: samplePoints.length,
      mismatches: samplePoints.length,
      diffs: [],
      rustError: errorMsg,
    };
  }

  // ── Compare ────────────────────────────────────────────────────────
  const diffs: ParityDiff[] = [];
  for (let i = 0; i < samplePoints.length; i++) {
    const js = jsValues[i];
    const rust = rustValues[i];
    const delta = Math.abs(js - rust);
    if (delta > tolerance) {
      diffs.push({ point: samplePoints[i], js, rust, delta });
    }
  }

  return {
    match: diffs.length === 0,
    totalPoints: samplePoints.length,
    mismatches: diffs.length,
    diffs,
  };
}

/**
 * Generate random sample points within a range.
 * Useful for quick parity checks during development.
 */
export function generateRandomPoints(
  count: number,
  rangeMin: number = -128,
  rangeMax: number = 128,
  yMin: number = 0,
  yMax: number = 256,
): ParityPoint[] {
  const points: ParityPoint[] = [];
  for (let i = 0; i < count; i++) {
    points.push({
      x: rangeMin + Math.random() * (rangeMax - rangeMin),
      y: yMin + Math.random() * (yMax - yMin),
      z: rangeMin + Math.random() * (rangeMax - rangeMin),
    });
  }
  return points;
}

/**
 * Run parity check and log results to console.
 * Intended for quick interactive testing from browser devtools.
 */
export async function logParity(
  nodes: Node[],
  edges: Edge[],
  pointCount: number = 50,
  tolerance: number = 1e-4,
  rootNodeId?: string,
): Promise<void> {
  const points = generateRandomPoints(pointCount);
  const result = await checkParity(nodes, edges, points, tolerance, rootNodeId);

  if (result.rustError) {
    console.warn("[Parity] Rust evaluation error:", result.rustError);
    return;
  }

  if (result.match) {
    console.log(
      `[Parity] ✅ All ${result.totalPoints} points match within tolerance ${tolerance}`,
    );
  } else {
    console.warn(
      `[Parity] ❌ ${result.mismatches}/${result.totalPoints} points differ beyond tolerance ${tolerance}`,
    );
    console.table(
      result.diffs.slice(0, 20).map((d) => ({
        x: d.point.x.toFixed(2),
        y: d.point.y.toFixed(2),
        z: d.point.z.toFixed(2),
        js: d.js,
        rust: d.rust,
        delta: d.delta.toExponential(3),
      })),
    );
  }
}
