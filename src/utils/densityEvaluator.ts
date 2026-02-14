import type { Node, Edge } from "@xyflow/react";
import { getCurveEvaluator, normalizePoints, catmullRomInterpolate } from "./curveEvaluators";
import { evaluateVectorProvider, vec3Normalize, vec3Length } from "./vectorEvaluator";
import {
  createHytaleNoise2D, createHytaleNoise3D,
  createHytaleNoise2DWithGradient, createHytaleNoise3DWithGradient,
  seedToInt,
} from "./hytaleNoise";
import { EvalStatus } from "@/schema/types";
import { HASH_PRIME_A, HASH_PRIME_B, HASH_PRIME_C, DEFAULT_WORLD_HEIGHT } from "@/constants";

/* ── Types ────────────────────────────────────────────────────────── */

export interface DensityGridResult {
  values: Float32Array;
  minValue: number;
  maxValue: number;
}

export interface EvaluationOptions {
  /** Content field values from WorldStructures, e.g. { "Base": 100, "Water": 100, "Bedrock": 0 } */
  contentFields?: Record<string, number>;
}

/* ── Seeded PRNG ──────────────────────────────────────────────────── */

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashSeed(seed: number | string | undefined): number {
  return seedToInt(seed);
}

/* ── Voronoi noise ────────────────────────────────────────────────── */

function voronoiNoise2D(
  _rng: () => number,
  cellType: string = "Euclidean",
  jitter: number = 1.0,
): (x: number, y: number) => number {
  return (x: number, y: number) => {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    let d1 = Infinity;
    let d2 = Infinity;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const cx = ix + dx;
        const cy = iy + dy;
        const s = hashSeed(cx * HASH_PRIME_A + cy * HASH_PRIME_B);
        const cellRng = mulberry32(s);
        const px = cx + cellRng() * jitter;
        const py = cy + cellRng() * jitter;
        const dist = Math.sqrt((x - px) ** 2 + (y - py) ** 2);
        if (dist < d1) { d2 = d1; d1 = dist; }
        else if (dist < d2) { d2 = dist; }
      }
    }
    if (cellType === "Distance2Div") return d2 > 0 ? (d1 / d2) * 2 - 1 : 0;
    if (cellType === "Distance2Sub") return (d2 - d1) * 2 - 1;
    return d1 * 2 - 1; // Euclidean default
  };
}

function voronoiNoise3D(
  _rng: () => number,
  cellType: string = "Euclidean",
  jitter: number = 1.0,
): (x: number, y: number, z: number) => number {
  return (x: number, y: number, z: number) => {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    const iz = Math.floor(z);
    let d1 = Infinity;
    let d2 = Infinity;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const cx = ix + dx;
          const cy = iy + dy;
          const cz = iz + dz;
          const s = hashSeed(cx * HASH_PRIME_A + cy * HASH_PRIME_B + cz * HASH_PRIME_C);
          const cellRng = mulberry32(s);
          const px = cx + cellRng() * jitter;
          const py = cy + cellRng() * jitter;
          const pz = cz + cellRng() * jitter;
          const dist = Math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2);
          if (dist < d1) { d2 = d1; d1 = dist; }
          else if (dist < d2) { d2 = dist; }
        }
      }
    }
    if (cellType === "Distance2Div") return d2 > 0 ? (d1 / d2) * 2 - 1 : 0;
    if (cellType === "Distance2Sub") return (d2 - d1) * 2 - 1;
    return d1 * 2 - 1; // Euclidean default
  };
}

/* ── Smooth math helpers ─────────────────────────────────────────── */

function smoothMin(a: number, b: number, k: number): number {
  if (k <= 0) return Math.min(a, b);
  const h = Math.max(0, Math.min(1, 0.5 + 0.5 * (b - a) / k));
  return b + (a - b) * h - k * h * (1 - h);
}

function smoothMax(a: number, b: number, k: number): number {
  return -smoothMin(-a, -b, k);
}

/* ── Rotation helpers (Rodrigues' formula) ────────────────────────── */

type Mat3 = [number, number, number, number, number, number, number, number, number];

const IDENTITY_MAT3: Mat3 = [1,0,0, 0,1,0, 0,0,1];

/**
 * Build a rotation matrix that maps the Y axis [0,1,0] to `newYAxis`
 * and then applies a spin rotation around the new Y axis.
 * Uses Rodrigues' rotation formula.
 * Reference: WorldGenV2/docs/math/coordinate-transforms.md
 */
function buildRotationMatrix(
  newYAxis: { x?: number; y?: number; z?: number } | undefined,
  spinAngle: number,
): Mat3 {
  // Default: no rotation
  if (!newYAxis) return IDENTITY_MAT3;

  let nx = Number(newYAxis.x ?? 0);
  let ny = Number(newYAxis.y ?? 1);
  let nz = Number(newYAxis.z ?? 0);

  // Normalize the target Y axis
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (len < 1e-10) return IDENTITY_MAT3;
  nx /= len; ny /= len; nz /= len;

  // Step 1: Rotate standard Y [0,1,0] to newYAxis
  // Rotation axis = cross(up, newY) = (-nz, 0, nx)
  const ax = -nz;
  const ay = 0;
  const az = nx;
  const axisLen = Math.sqrt(ax * ax + ay * ay + az * az);

  let r1: Mat3;
  if (axisLen < 1e-10) {
    // newY is parallel to Y
    if (ny > 0) {
      r1 = IDENTITY_MAT3;
    } else {
      // 180-degree rotation around X axis
      r1 = [1,0,0, 0,-1,0, 0,0,-1];
    }
  } else {
    // Normalize rotation axis
    const kx = ax / axisLen;
    const ky = ay / axisLen;
    const kz = az / axisLen;
    const cosT = ny; // dot(up, newY)
    const sinT = axisLen;
    r1 = rodriguesMatrix(kx, ky, kz, cosT, sinT);
  }

  // Step 2: Apply spin rotation around newYAxis
  if (Math.abs(spinAngle) < 1e-10) return r1;

  const cosS = Math.cos(spinAngle);
  const sinS = Math.sin(spinAngle);
  const r2 = rodriguesMatrix(nx, ny, nz, cosS, sinS);

  // Combined: r2 * r1
  return mat3Multiply(r2, r1);
}

/**
 * Rodrigues' rotation formula as a 3x3 matrix.
 * Given unit axis (kx,ky,kz) and angle with known cos and sin.
 */
function rodriguesMatrix(
  kx: number, ky: number, kz: number,
  cosT: number, sinT: number,
): Mat3 {
  const oneMinusCos = 1.0 - cosT;
  return [
    cosT + kx * kx * oneMinusCos,         kx * ky * oneMinusCos - kz * sinT,   kx * kz * oneMinusCos + ky * sinT,
    ky * kx * oneMinusCos + kz * sinT,     cosT + ky * ky * oneMinusCos,         ky * kz * oneMinusCos - kx * sinT,
    kz * kx * oneMinusCos - ky * sinT,     kz * ky * oneMinusCos + kx * sinT,   cosT + kz * kz * oneMinusCos,
  ];
}

function mat3Multiply(a: Mat3, b: Mat3): Mat3 {
  return [
    a[0]*b[0]+a[1]*b[3]+a[2]*b[6], a[0]*b[1]+a[1]*b[4]+a[2]*b[7], a[0]*b[2]+a[1]*b[5]+a[2]*b[8],
    a[3]*b[0]+a[4]*b[3]+a[5]*b[6], a[3]*b[1]+a[4]*b[4]+a[5]*b[7], a[3]*b[2]+a[4]*b[5]+a[5]*b[8],
    a[6]*b[0]+a[7]*b[3]+a[8]*b[6], a[6]*b[1]+a[7]*b[4]+a[8]*b[7], a[6]*b[2]+a[7]*b[5]+a[8]*b[8],
  ];
}

/** Apply rotation matrix to a vector. */
function mat3Apply(m: Mat3, x: number, y: number, z: number): [number, number, number] {
  return [
    m[0] * x + m[1] * y + m[2] * z,
    m[3] * x + m[4] * y + m[5] * z,
    m[6] * x + m[7] * y + m[8] * z,
  ];
}

/* ── Density type constants ───────────────────────────────────────── */

const UNSUPPORTED_TYPES = new Set([
  "HeightAboveSurface",
  "SurfaceDensity",
  "TerrainBoolean",
  "TerrainMask",
  "BeardDensity",
  "ColumnDensity",
  "CaveDensity",
  // Context-dependent types (require server data)
  "Terrain",
  // CellWallDistance is now implemented
  "DistanceToBiomeEdge",
  "Pipeline",
]);

/** Types with approximate or stub evaluators (yellow badge) */
const APPROXIMATED_TYPES = new Set([
  "PositionsCellNoise", // Approximated as voronoi
  "Positions3D",        // Approximated as voronoi
  "PositionsPinch",     // Simplified pinch transform
  "PositionsTwist",     // Simplified twist transform
]);

/**
 * Get the evaluation status for a given density type.
 * - Full: accurately evaluated with proper math
 * - Approximated: uses a simplified or stub evaluator
 * - Unsupported: context-dependent, returns 0
 */
export function getEvalStatus(type: string): EvalStatus {
  if (UNSUPPORTED_TYPES.has(type)) return EvalStatus.Unsupported;
  if (APPROXIMATED_TYPES.has(type)) return EvalStatus.Approximated;
  return EvalStatus.Full;
}

/* ── Root-finding ─────────────────────────────────────────────────── */

const DENSITY_TYPES = new Set([
  "SimplexNoise2D", "SimplexNoise3D", "SimplexRidgeNoise2D", "SimplexRidgeNoise3D",
  "VoronoiNoise2D", "VoronoiNoise3D", "FractalNoise2D", "FractalNoise3D",
  "DomainWarp2D", "DomainWarp3D",
  "Sum", "SumSelf", "WeightedSum", "Product",
  "Negate", "Abs", "SquareRoot", "CubeRoot", "Square", "Cube", "Inverse", "Modulo",
  "Constant", "ImportedValue", "Zero", "One",
  "Clamp", "ClampToIndex", "Normalizer", "DoubleNormalizer", "RangeChoice",
  "LinearTransform", "Interpolate",
  "CoordinateX", "CoordinateY", "CoordinateZ",
  "DistanceFromOrigin", "DistanceFromAxis", "DistanceFromPoint",
  "AngleFromOrigin", "AngleFromPoint", "HeightAboveSurface",
  "CurveFunction", "SplineFunction", "FlatCache",
  "Conditional", "Switch", "Blend", "BlendCurve",
  "MinFunction", "MaxFunction", "AverageFunction",
  "CacheOnce", "Wrap",
  "TranslatedPosition", "ScaledPosition", "RotatedPosition", "MirroredPosition", "QuantizedPosition",
  "SurfaceDensity", "TerrainBoolean", "TerrainMask", "GradientDensity",
  "BeardDensity", "ColumnDensity", "CaveDensity",
  "Debug", "YGradient", "Passthrough",
  "BaseHeight", "Floor", "Ceiling",
  "SmoothClamp", "SmoothFloor", "SmoothMin", "SmoothMax",
  "YOverride", "Anchor", "Exported", "Offset", "Distance",
  "PositionsCellNoise",
  "AmplitudeConstant", "Pow",
  "XOverride", "ZOverride", "SmoothCeiling", "Gradient",
  "Amplitude", "YSampled", "SwitchState", "MultiMix",
  "Positions3D", "PositionsPinch", "PositionsTwist",
  "GradientWarp", "FastGradientWarp", "VectorWarp",
  "Terrain", "CellWallDistance", "DistanceToBiomeEdge", "Pipeline",
  // Shape SDFs
  "Ellipsoid", "Cuboid", "Cylinder", "Plane", "Shell",
]);

function getNodeType(node: Node): string {
  const data = node.data as Record<string, unknown>;
  return (data.type as string) ?? "";
}

function findDensityRoot(nodes: Node[], edges: Edge[]): Node | null {
  // Strategy 0: user-designated output node
  const outputNode = nodes.find(
    (n) => (n.data as Record<string, unknown>)._outputNode === true,
  );
  if (outputNode) return outputNode;

  // Strategy 1: explicit biome field tag
  const tagged = nodes.find(
    (n) => (n.data as Record<string, unknown>)._biomeField === "Terrain",
  );
  if (tagged) return tagged;

  // Strategy 2: terminal nodes (no outgoing edges) with density types
  const sourcesWithOutgoing = new Set(edges.map((e) => e.source));
  const terminals = nodes.filter((n) => !sourcesWithOutgoing.has(n.id));

  const terminalDensity = terminals.find((n) => DENSITY_TYPES.has(getNodeType(n)));
  if (terminalDensity) return terminalDensity;

  // Strategy 3: any node with a density type
  return nodes.find((n) => DENSITY_TYPES.has(getNodeType(n))) ?? null;
}

/* ── Evaluation Context Factory ───────────────────────────────────── */

export interface EvaluationContext {
  rootId: string;
  evaluate: (nodeId: string, x: number, y: number, z: number) => number;
  clearMemo: () => void;
}

/**
 * Create a reusable evaluation context from a node graph.
 * This extracts all the noise caches, memoization, and evaluation logic
 * so it can be shared between 2D grid evaluation and 3D volume evaluation.
 */
export function createEvaluationContext(
  nodes: Node[],
  edges: Edge[],
  rootNodeId?: string,
  options?: EvaluationOptions,
): EvaluationContext | null {
  if (nodes.length === 0) return null;

  let root: Node | null = null;
  if (rootNodeId) {
    root = nodes.find((nd) => nd.id === rootNodeId) ?? null;
  }
  if (!root) {
    root = findDensityRoot(nodes, edges);
  }
  if (!root) return null;

  // Build inputEdges: targetId → Map<handleId, sourceId>
  const inputEdges = new Map<string, Map<string, string>>();
  for (const e of edges) {
    const map = inputEdges.get(e.target) ?? new Map<string, string>();
    const handle = e.targetHandle ?? "Input";
    map.set(handle, e.source);
    inputEdges.set(e.target, map);
  }

  const nodeById = new Map(nodes.map((nd) => [nd.id, nd]));

  // Content fields from WorldStructure (e.g. { "Base": 100, "Water": 100 })
  const contentFields = options?.contentFields ?? {};

  // Noise caches — keyed by seed to reuse noise functions
  const noise2DCache = new Map<number, (x: number, y: number) => number>();
  const noise3DCache = new Map<number, (x: number, y: number, z: number) => number>();
  const voronoi2DCache = new Map<string, (x: number, y: number) => number>();
  const voronoi3DCache = new Map<string, (x: number, y: number, z: number) => number>();

  function getNoise2D(seed: number): (x: number, y: number) => number {
    let fn = noise2DCache.get(seed);
    if (!fn) {
      fn = createHytaleNoise2D(seed);
      noise2DCache.set(seed, fn);
    }
    return fn;
  }

  function getNoise3D(seed: number): (x: number, y: number, z: number) => number {
    let fn = noise3DCache.get(seed);
    if (!fn) {
      fn = createHytaleNoise3D(seed);
      noise3DCache.set(seed, fn);
    }
    return fn;
  }

  function getVoronoi2D(seed: number, cellType: string = "Euclidean", jitter: number = 1.0): (x: number, y: number) => number {
    const key = `${seed}:${cellType}:${jitter}`;
    let fn = voronoi2DCache.get(key);
    if (!fn) {
      const rng = mulberry32(seed);
      fn = voronoiNoise2D(rng, cellType, jitter);
      voronoi2DCache.set(key, fn);
    }
    return fn;
  }

  function getVoronoi3D(seed: number, cellType: string = "Euclidean", jitter: number = 1.0): (x: number, y: number, z: number) => number {
    const key = `${seed}:${cellType}:${jitter}`;
    let fn = voronoi3DCache.get(key);
    if (!fn) {
      const rng = mulberry32(seed);
      fn = voronoiNoise3D(rng, cellType, jitter);
      voronoi3DCache.set(key, fn);
    }
    return fn;
  }

  // Memoization for CacheOnce / FlatCache nodes
  const memoCache = new Map<string, number>();

  // Cycle detection per evaluation path
  const visiting = new Set<string>();

  // Evaluation context state (mutable, propagated through the graph)
  // densityAnchor: set by position providers / Anchor nodes
  let ctxAnchorX = 0, ctxAnchorY = 0, ctxAnchorZ = 0;
  let ctxAnchorSet = false;
  // switchState: set by SwitchState nodes, read by Switch nodes
  let ctxSwitchState = 0;
  // distanceFromCellWall: set by PositionsCellNoise, read by CellWallDistance
  let ctxCellWallDist = Infinity;

  /** Apply a curve node to an input value. Shared by CurveFunction, BlendCurve, SplineFunction. */
  function applyCurve(curveHandleName: string, inputVal: number, inputs: Map<string, string>): number {
    const curveNodeId = inputs.get(curveHandleName);
    if (!curveNodeId) return inputVal;
    const curveNode = nodeById.get(curveNodeId);
    if (!curveNode) return inputVal;
    const curveData = curveNode.data as Record<string, unknown>;
    const curveType = ((curveData.type as string) ?? "").replace(/^Curve:/, "");
    const curveFields = (curveData.fields as Record<string, unknown>) ?? {};

    if (curveType === "Manual") {
      const rawPoints = curveFields.Points as unknown[] | undefined;
      if (rawPoints && rawPoints.length >= 2) {
        const pts = normalizePoints(rawPoints);
        const sortedPts = pts.sort((a, b) => a.x - b.x);
        const sampled = catmullRomInterpolate(sortedPts, 32);
        const xMin = sortedPts[0].x;
        const xMax = sortedPts[sortedPts.length - 1].x;
        const clampedInput = Math.max(xMin, Math.min(xMax, inputVal));
        let lo = 0;
        let hi = sampled.length - 1;
        while (lo < hi - 1) {
          const mid = (lo + hi) >> 1;
          if (sampled[mid].x <= clampedInput) lo = mid;
          else hi = mid;
        }
        const p0 = sampled[lo];
        const p1 = sampled[hi];
        const dx = p1.x - p0.x;
        const t = dx === 0 ? 0 : (clampedInput - p0.x) / dx;
        return p0.y + (p1.y - p0.y) * t;
      }
      return inputVal;
    }

    const curveFn = getCurveEvaluator(curveType, curveFields);
    return curveFn ? curveFn(inputVal) : inputVal;
  }

  /** Apply spline interpolation using Points field. */
  function applySpline(fields: Record<string, unknown>, inputVal: number): number {
    const rawPoints = fields.Points as unknown[] | undefined;
    if (rawPoints && rawPoints.length >= 2) {
      const pts = normalizePoints(rawPoints);
      const sortedPts = pts.sort((a, b) => a.x - b.x);
      const sampled = catmullRomInterpolate(sortedPts, 32);
      const xMin = sortedPts[0].x;
      const xMax = sortedPts[sortedPts.length - 1].x;
      const clampedInput = Math.max(xMin, Math.min(xMax, inputVal));
      let lo = 0;
      let hi = sampled.length - 1;
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (sampled[mid].x <= clampedInput) lo = mid;
        else hi = mid;
      }
      const p0 = sampled[lo];
      const p1 = sampled[hi];
      const dx = p1.x - p0.x;
      const t = dx === 0 ? 0 : (clampedInput - p0.x) / dx;
      return p0.y + (p1.y - p0.y) * t;
    }
    return inputVal;
  }

  /** Evaluate a single node at sample position (x, y, z). */
  function evaluate(nodeId: string, x: number, y: number, z: number): number {
    if (visiting.has(nodeId)) return 0; // cycle guard
    visiting.add(nodeId);

    const node = nodeById.get(nodeId);
    if (!node) {
      visiting.delete(nodeId);
      return 0;
    }

    const data = node.data as Record<string, unknown>;
    const type = getNodeType(node);
    const fields = (data.fields as Record<string, unknown>) ?? {};
    const inputs = inputEdges.get(nodeId) ?? new Map<string, string>();

    let result = 0;

    switch (type) {
      // ── Noise ──────────────────────────────────────────────────────

      case "SimplexNoise2D": {
        const freq = Number(fields.Frequency ?? 0.01);
        const amp = Number(fields.Amplitude ?? 1.0);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const octaves = Math.max(1, Number(fields.Octaves ?? 1));
        const lacunarity = Number(fields.Lacunarity ?? 2.0);
        const gain = Number(fields.Gain ?? 0.5);
        const noise = getNoise2D(seed);
        result = fbm2D(noise, x, z, freq, octaves, lacunarity, gain) * amp;
        break;
      }

      case "SimplexNoise3D": {
        const freq = Number(fields.Frequency ?? 0.01);
        const amp = Number(fields.Amplitude ?? 1.0);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const octaves = Math.max(1, Number(fields.Octaves ?? 1));
        const lacunarity = Number(fields.Lacunarity ?? 2.0);
        const gain = Number(fields.Gain ?? 0.5);
        const noise = getNoise3D(seed);
        result = fbm3D(noise, x, y, z, freq, octaves, lacunarity, gain) * amp;
        break;
      }

      case "SimplexRidgeNoise2D": {
        const freq = Number(fields.Frequency ?? 0.01);
        const amp = Number(fields.Amplitude ?? 1.0);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const octaves = Math.max(1, Number(fields.Octaves ?? 1));
        const noise = getNoise2D(seed);
        result = ridgeFbm2D(noise, x, z, freq, octaves) * amp;
        break;
      }

      case "SimplexRidgeNoise3D": {
        const freq = Number(fields.Frequency ?? 0.01);
        const amp = Number(fields.Amplitude ?? 1.0);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const octaves = Math.max(1, Number(fields.Octaves ?? 1));
        const noise = getNoise3D(seed);
        result = ridgeFbm3D(noise, x, y, z, freq, octaves) * amp;
        break;
      }

      case "VoronoiNoise2D": {
        const freq = Number(fields.Frequency ?? 0.01);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const cellType = (fields.CellType as string) ?? "Euclidean";
        const jitter = Number(fields.Jitter ?? 1.0);
        const octaves = Math.max(1, Number(fields.Octaves ?? 1));
        const lacunarity = Number(fields.Lacunarity ?? 2.0);
        const gain = Number(fields.Gain ?? 0.5);
        const noise = getVoronoi2D(seed, cellType, jitter);
        result = octaves > 1
          ? voronoiFbm2D(noise, x, z, freq, octaves, lacunarity, gain)
          : noise(x * freq, z * freq);
        break;
      }

      case "VoronoiNoise3D": {
        const freq = Number(fields.Frequency ?? 0.01);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const cellType = (fields.CellType as string) ?? "Euclidean";
        const jitter = Number(fields.Jitter ?? 1.0);
        const octaves = Math.max(1, Number(fields.Octaves ?? 1));
        const lacunarity = Number(fields.Lacunarity ?? 2.0);
        const gain = Number(fields.Gain ?? 0.5);
        const noise = getVoronoi3D(seed, cellType, jitter);
        result = octaves > 1
          ? voronoiFbm3D(noise, x, y, z, freq, octaves, lacunarity, gain)
          : noise(x * freq, y * freq, z * freq);
        break;
      }

      case "FractalNoise2D": {
        const freq = Number(fields.Frequency ?? 0.01);
        const octaves = Math.max(1, Number(fields.Octaves ?? 4));
        const lacunarity = Number(fields.Lacunarity ?? 2.0);
        const gain = Number(fields.Gain ?? 0.5);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const noise = getNoise2D(seed);
        result = fbm2D(noise, x, z, freq, octaves, lacunarity, gain);
        break;
      }

      case "FractalNoise3D": {
        const freq = Number(fields.Frequency ?? 0.01);
        const octaves = Math.max(1, Number(fields.Octaves ?? 4));
        const lacunarity = Number(fields.Lacunarity ?? 2.0);
        const gain = Number(fields.Gain ?? 0.5);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const noise = getNoise3D(seed);
        result = fbm3D(noise, x, y, z, freq, octaves, lacunarity, gain);
        break;
      }

      // ── Constants ──────────────────────────────────────────────────

      case "Constant":
        result = Number(fields.Value ?? 0);
        break;

      case "Zero":
        result = 0;
        break;

      case "One":
        result = 1;
        break;

      // ── Arithmetic (single input) ─────────────────────────────────

      case "Negate":
        result = -getInput(inputs, "Input", x, y, z);
        break;

      case "Abs":
        result = Math.abs(getInput(inputs, "Input", x, y, z));
        break;

      case "SquareRoot":
        result = Math.sqrt(Math.abs(getInput(inputs, "Input", x, y, z)));
        break;

      case "CubeRoot":
        result = Math.cbrt(getInput(inputs, "Input", x, y, z));
        break;

      case "Square": {
        const v = getInput(inputs, "Input", x, y, z);
        result = v * v;
        break;
      }

      case "Cube": {
        const v = getInput(inputs, "Input", x, y, z);
        result = v * v * v;
        break;
      }

      case "Inverse": {
        const v = getInput(inputs, "Input", x, y, z);
        result = v === 0 ? 0 : 1 / v;
        break;
      }

      case "SumSelf": {
        const count = Math.max(1, Number(fields.Count ?? 2));
        result = getInput(inputs, "Input", x, y, z) * count;
        break;
      }

      case "Modulo": {
        const v = getInput(inputs, "Input", x, y, z);
        const divisor = Number(fields.Divisor ?? 1.0);
        result = divisor === 0 ? 0 : v % divisor;
        break;
      }

      case "AmplitudeConstant": {
        const v = getInput(inputs, "Input", x, y, z);
        const amp = Number(fields.Value ?? 1);
        result = v * amp;
        break;
      }

      case "Pow": {
        const v = getInput(inputs, "Input", x, y, z);
        const exp = Number(fields.Exponent ?? 2);
        result = Math.pow(Math.abs(v), exp) * Math.sign(v);
        break;
      }

      // ── Arithmetic (dual input) ───────────────────────────────────

      case "Sum":
        result = getInput(inputs, "InputA", x, y, z) + getInput(inputs, "InputB", x, y, z);
        break;

      case "WeightedSum": {
        const a = getInput(inputs, "Inputs[0]", x, y, z);
        const b = getInput(inputs, "Inputs[1]", x, y, z);
        const weights = (fields.Weights as number[]) ?? [1, 1];
        result = a * (weights[0] ?? 1) + b * (weights[1] ?? 1);
        break;
      }

      case "Product":
        result = getInput(inputs, "Inputs[0]", x, y, z) * getInput(inputs, "Inputs[1]", x, y, z);
        break;

      // ── Clamping & range ──────────────────────────────────────────

      case "Clamp": {
        const v = getInput(inputs, "Input", x, y, z);
        const min = Number(fields.Min ?? 0);
        const max = Number(fields.Max ?? 1);
        result = Math.max(min, Math.min(max, v));
        break;
      }

      case "ClampToIndex": {
        const v = getInput(inputs, "Input", x, y, z);
        const min = Number(fields.Min ?? 0);
        const max = Number(fields.Max ?? 255);
        result = Math.max(min, Math.min(max, Math.floor(v)));
        break;
      }

      case "Normalizer": {
        const v = getInput(inputs, "Input", x, y, z);
        const src = fields.SourceRange as { Min?: number; Max?: number } | undefined;
        const tgt = fields.TargetRange as { Min?: number; Max?: number } | undefined;
        const srcMin = Number(src?.Min ?? -1);
        const srcMax = Number(src?.Max ?? 1);
        const tgtMin = Number(tgt?.Min ?? 0);
        const tgtMax = Number(tgt?.Max ?? 1);
        const range = srcMax - srcMin;
        const t = range === 0 ? 0 : (v - srcMin) / range;
        result = tgtMin + t * (tgtMax - tgtMin);
        break;
      }

      case "DoubleNormalizer": {
        const v = getInput(inputs, "Input", x, y, z);
        const srcA = fields.SourceRangeA as { Min?: number; Max?: number } | undefined;
        const tgtA = fields.TargetRangeA as { Min?: number; Max?: number } | undefined;
        const srcB = fields.SourceRangeB as { Min?: number; Max?: number } | undefined;
        const tgtB = fields.TargetRangeB as { Min?: number; Max?: number } | undefined;
        if (v < 0) {
          const srcMin = Number(srcA?.Min ?? -1);
          const srcMax = Number(srcA?.Max ?? 0);
          const tgtMin = Number(tgtA?.Min ?? 0);
          const tgtMax = Number(tgtA?.Max ?? 0.5);
          const range = srcMax - srcMin;
          const t = range === 0 ? 0 : (v - srcMin) / range;
          result = tgtMin + t * (tgtMax - tgtMin);
        } else {
          const srcMin = Number(srcB?.Min ?? 0);
          const srcMax = Number(srcB?.Max ?? 1);
          const tgtMin = Number(tgtB?.Min ?? 0.5);
          const tgtMax = Number(tgtB?.Max ?? 1);
          const range = srcMax - srcMin;
          const t = range === 0 ? 0 : (v - srcMin) / range;
          result = tgtMin + t * (tgtMax - tgtMin);
        }
        break;
      }

      case "LinearTransform": {
        const v = getInput(inputs, "Input", x, y, z);
        const scale = Number(fields.Scale ?? 1);
        const offset = Number(fields.Offset ?? 0);
        result = v * scale + offset;
        break;
      }

      case "RangeChoice": {
        const cond = getInput(inputs, "Condition", x, y, z);
        const threshold = Number(fields.Threshold ?? 0.5);
        result = cond >= threshold
          ? getInput(inputs, "TrueInput", x, y, z)
          : getInput(inputs, "FalseInput", x, y, z);
        break;
      }

      case "Interpolate": {
        const a = getInput(inputs, "InputA", x, y, z);
        const b = getInput(inputs, "InputB", x, y, z);
        const f = getInput(inputs, "Factor", x, y, z);
        result = a + (b - a) * f;
        break;
      }

      // ── Position-based ────────────────────────────────────────────

      case "CoordinateX":
        result = x;
        break;

      case "CoordinateY":
        result = y;
        break;

      case "CoordinateZ":
        result = z;
        break;

      case "DistanceFromOrigin":
        result = Math.sqrt(x * x + y * y + z * z);
        break;

      case "DistanceFromAxis": {
        const axis = String(fields.Axis ?? "Y");
        if (axis === "X") result = Math.sqrt(y * y + z * z);
        else if (axis === "Z") result = Math.sqrt(x * x + y * y);
        else result = Math.sqrt(x * x + z * z); // Y axis
        break;
      }

      case "DistanceFromPoint": {
        const pt = fields.Point as { x?: number; y?: number; z?: number } | undefined;
        const px = Number(pt?.x ?? 0);
        const py = Number(pt?.y ?? 0);
        const pz = Number(pt?.z ?? 0);
        result = Math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2);
        break;
      }

      case "AngleFromOrigin":
        result = Math.atan2(z, x);
        break;

      case "AngleFromPoint": {
        const pt = fields.Point as { x?: number; y?: number; z?: number } | undefined;
        const px = Number(pt?.x ?? 0);
        const pz = Number(pt?.z ?? 0);
        result = Math.atan2(z - pz, x - px);
        break;
      }

      case "YGradient": {
        const fromY = Number(fields.FromY ?? 0);
        const toY = Number(fields.ToY ?? DEFAULT_WORLD_HEIGHT);
        const range = toY - fromY;
        result = range === 0 ? 0 : (y - fromY) / range;
        break;
      }

      case "GradientDensity": {
        const fromY = Number(fields.FromY ?? 0);
        const toY = Number(fields.ToY ?? DEFAULT_WORLD_HEIGHT);
        const range = toY - fromY;
        result = range === 0 ? 0 : (y - fromY) / range;
        break;
      }

      // ── BaseHeight ─────────────────────────────────────────────────

      case "BaseHeight": {
        const name = (fields.BaseHeightName as string) ?? "Base";
        const baseY = contentFields[name] ?? 100;
        const distance = fields.Distance === true;
        result = distance ? (y - baseY) : baseY;
        break;
      }

      // ── Floor / Ceiling ────────────────────────────────────────────

      case "Floor":
        result = Math.floor(getInput(inputs, "Input", x, y, z));
        break;

      case "Ceiling":
        result = Math.ceil(getInput(inputs, "Input", x, y, z));
        break;

      // ── Smooth operations ──────────────────────────────────────────

      case "SmoothClamp": {
        const v = getInput(inputs, "Input", x, y, z);
        const minV = Number(fields.Min ?? 0);
        const maxV = Number(fields.Max ?? 1);
        const k = Number(fields.Smoothness ?? 0.1);
        result = smoothMax(smoothMin(v, maxV, k), minV, k);
        break;
      }

      case "SmoothFloor": {
        const v = getInput(inputs, "Input", x, y, z);
        const threshold = Number(fields.Threshold ?? 0);
        const k = Number(fields.Smoothness ?? 0.1);
        result = smoothMax(v, threshold, k);
        break;
      }

      case "SmoothMin": {
        const a = getInput(inputs, "Inputs[0]", x, y, z);
        const b = getInput(inputs, "Inputs[1]", x, y, z);
        const k = Number(fields.Smoothness ?? 0.1);
        result = smoothMin(a, b, k);
        break;
      }

      case "SmoothMax": {
        const a = getInput(inputs, "Inputs[0]", x, y, z);
        const b = getInput(inputs, "Inputs[1]", x, y, z);
        const k = Number(fields.Smoothness ?? 0.1);
        result = smoothMax(a, b, k);
        break;
      }

      // ── Position overrides ─────────────────────────────────────────

      case "YOverride": {
        const overrideY = Number(fields.OverrideY ?? fields.Y ?? 0);
        result = getInput(inputs, "Input", x, overrideY, z);
        break;
      }

      case "Anchor": {
        const isReversed = fields.Reversed === true;
        if (ctxAnchorSet) {
          if (isReversed) {
            // Reversed: local-to-world — add anchor to position
            result = getInput(inputs, "Input", x + ctxAnchorX, y + ctxAnchorY, z + ctxAnchorZ);
          } else {
            // Normal: world-to-local — subtract anchor from position
            result = getInput(inputs, "Input", x - ctxAnchorX, y - ctxAnchorY, z - ctxAnchorZ);
          }
        } else {
          // No anchor set — passthrough
          result = getInput(inputs, "Input", x, y, z);
        }
        break;
      }

      // ── Passthrough / tagging ──────────────────────────────────────

      case "Exported":
      case "ImportedValue":
        result = getInput(inputs, "Input", x, y, z);
        break;

      // ── Offset (density variant) ───────────────────────────────────

      case "Offset":
        result = getInput(inputs, "Input", x, y, z) + getInput(inputs, "Offset", x, y, z);
        break;

      // ── Distance with curve ────────────────────────────────────────

      case "Distance": {
        const dist = Math.sqrt(x * x + y * y + z * z);
        result = applyCurve("Curve", dist, inputs);
        break;
      }

      // ── PositionsCellNoise (approximated as voronoi) ───────────────

      case "PositionsCellNoise": {
        const maxDist = Number(fields.MaxDistance ?? 0);
        const freq = maxDist > 0 ? 1 / maxDist : Number(fields.Frequency ?? 0.01);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const returnType = (fields.ReturnType as string) ?? "Distance";
        const distFn = (fields.DistanceFunction as string) ?? "Euclidean";
        const noise = getVoronoi2D(seed, distFn, 1.0);
        let raw = noise(x * freq, z * freq);
        if (returnType === "Distance2Div") {
          raw = Math.abs(raw);
        }
        result = raw;
        break;
      }

      case "CellWallDistance": {
        // Distance to nearest Voronoi cell wall: (F2 - F1) / 2
        // If context has it set, use that. Otherwise compute from voronoi.
        if (ctxCellWallDist < Infinity) {
          result = ctxCellWallDist;
        } else {
          // Compute directly — approximate with voronoi Distance2Sub
          const freq = Number(fields.Frequency ?? 0.01);
          const seed = hashSeed(fields.Seed as string | number | undefined);
          const noise = getVoronoi2D(seed, "Distance2Sub", 1.0);
          result = Math.max(0, noise(x * freq, z * freq));
        }
        break;
      }

      // ── Additional density types ──

      case "XOverride": {
        const overrideX = Number(fields.OverrideX ?? 0);
        result = getInput(inputs, "Input", overrideX, y, z);
        break;
      }

      case "ZOverride": {
        const overrideZ = Number(fields.OverrideZ ?? 0);
        result = getInput(inputs, "Input", x, y, overrideZ);
        break;
      }

      case "SmoothCeiling": {
        const v = getInput(inputs, "Input", x, y, z);
        const threshold = Number(fields.Threshold ?? 1.0);
        const k = Number(fields.Smoothness ?? 0.1);
        result = smoothMin(v, threshold, k);
        break;
      }

      case "Gradient": {
        const fromY = Number(fields.FromY ?? 0);
        const toY = Number(fields.ToY ?? DEFAULT_WORLD_HEIGHT);
        const range = toY - fromY;
        result = range === 0 ? 0 : (y - fromY) / range;
        break;
      }

      case "Amplitude": {
        const v = getInput(inputs, "Input", x, y, z);
        const amp = getInput(inputs, "Amplitude", x, y, z);
        result = v * amp;
        break;
      }

      case "YSampled": {
        const sampleDist = Number(fields.SampleDistance ?? 4.0);
        const sampleOffset = Number(fields.SampleOffset ?? 0.0);
        const snappedY = sampleDist > 0
          ? Math.round((y - sampleOffset) / sampleDist) * sampleDist + sampleOffset
          : y;
        result = getInput(inputs, "Input", x, snappedY, z);
        break;
      }

      case "SwitchState": {
        // Set the context switchState and evaluate child
        const prevState = ctxSwitchState;
        ctxSwitchState = hashSeed(fields.State as string | number | undefined);
        result = getInput(inputs, "Input", x, y, z);
        ctxSwitchState = prevState; // restore
        break;
      }

      case "Positions3D": {
        const freq = Number(fields.Frequency ?? 0.01);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const noise = getVoronoi3D(seed, "Euclidean", 1.0);
        result = noise(x * freq, y * freq, z * freq);
        break;
      }

      case "PositionsPinch": {
        const strength = Number(fields.Strength ?? 1.0);
        const dist = Math.sqrt(x * x + z * z);
        const pinchFactor = dist > 0 ? Math.pow(dist, strength) / dist : 1;
        result = getInput(inputs, "Input", x * pinchFactor, y, z * pinchFactor);
        break;
      }

      case "PositionsTwist": {
        const angle = Number(fields.Angle ?? 0);
        const rad = (angle * Math.PI / 180) * y;
        const cos = Math.cos(rad);
        const sin = Math.sin(rad);
        result = getInput(inputs, "Input", x * cos - z * sin, y, x * sin + z * cos);
        break;
      }

      case "GradientWarp": {
        const warpFactor = Number(fields.WarpFactor ?? fields.WarpScale ?? 1.0);
        const eps = Number(fields.SampleRange ?? 1.0);
        const is2D = fields.Is2D === true;
        const yFor2D = Number(fields.YFor2D ?? 0.0);
        const inv2e = 1.0 / (2.0 * eps);
        const sampleY = is2D ? yFor2D : y;

        // Central finite differences for gradient estimation
        const dfdx = (getInput(inputs, "WarpSource", x + eps, sampleY, z)
                    - getInput(inputs, "WarpSource", x - eps, sampleY, z)) * inv2e;
        const dfdz = (getInput(inputs, "WarpSource", x, sampleY, z + eps)
                    - getInput(inputs, "WarpSource", x, sampleY, z - eps)) * inv2e;

        let wx = x + warpFactor * dfdx;
        let wy = y;
        let wz = z + warpFactor * dfdz;

        if (!is2D) {
          // 3D mode: also warp Y
          const dfdy = (getInput(inputs, "WarpSource", x, sampleY + eps, z)
                      - getInput(inputs, "WarpSource", x, sampleY - eps, z)) * inv2e;
          wy = y + warpFactor * dfdy;
        }

        result = getInput(inputs, "Input", wx, wy, wz);
        break;
      }

      case "VectorWarp": {
        const warpFactor = Number(fields.WarpFactor ?? 1.0);
        // Get direction vector from the connected vector provider
        const dirNodeId = inputs.get("Direction") ?? inputs.get("WarpVector");
        const dir = dirNodeId
          ? evaluateVectorProvider(dirNodeId, x, y, z, nodeById, inputEdges, evaluate)
          : { x: 0, y: 0, z: 0 };
        const dirNorm = vec3Normalize(dir);
        const dirLen = vec3Length(dir);

        if (dirLen < 1e-10) {
          // Zero direction: no warp
          result = getInput(inputs, "Input", x, y, z);
        } else {
          // Get magnitude from connected density
          const magnitude = getInput(inputs, "Magnitude", x, y, z);
          const displacement = magnitude * warpFactor;
          result = getInput(inputs, "Input",
            x + dirNorm.x * displacement,
            y + dirNorm.y * displacement,
            z + dirNorm.z * displacement,
          );
        }
        break;
      }

      case "FastGradientWarp": {
        const warpFactor = Number(fields.WarpFactor ?? 1.0);
        const warpSeed = hashSeed(fields.WarpSeed as string | number | undefined);
        const warpScale = Number(fields.WarpScale ?? 0.01);
        const warpOctaves = Math.max(1, Number(fields.WarpOctaves ?? 3));
        const warpLacunarity = Number(fields.WarpLacunarity ?? 2.0);
        const warpPersistence = Number(fields.WarpPersistence ?? 0.5);
        const is2D = fields.Is2D === true;

        // Accumulate analytic gradient across octaves
        let gx = 0, gy = 0, gz = 0;

        if (is2D) {
          let amp = 1.0;
          let freq = warpScale;
          for (let i = 0; i < warpOctaves; i++) {
            const noiseFn = createHytaleNoise2DWithGradient(warpSeed + i);
            const r = noiseFn(x * freq, z * freq);
            gx += amp * r.dx * freq;
            gz += amp * r.dy * freq; // dy in 2D noise maps to the z component
            amp *= warpPersistence;
            freq *= warpLacunarity;
          }
          result = getInput(inputs, "Input",
            x + warpFactor * gx,
            y,
            z + warpFactor * gz,
          );
        } else {
          let amp = 1.0;
          let freq = warpScale;
          for (let i = 0; i < warpOctaves; i++) {
            const noiseFn = createHytaleNoise3DWithGradient(warpSeed + i);
            const r = noiseFn(x * freq, y * freq, z * freq);
            gx += amp * r.dx * freq;
            gy += amp * r.dy * freq;
            gz += amp * r.dz * freq;
            amp *= warpPersistence;
            freq *= warpLacunarity;
          }
          result = getInput(inputs, "Input",
            x + warpFactor * gx,
            y + warpFactor * gy,
            z + warpFactor * gz,
          );
        }
        break;
      }

      // ── Shape SDFs ────────────────────────────────────────────────

      case "Ellipsoid": {
        const scale = fields.Scale as { x?: number; y?: number; z?: number } | undefined
                   ?? fields.Radius as { x?: number; y?: number; z?: number } | undefined;
        const sx = Number(scale?.x ?? 1) || 1;
        const sy = Number(scale?.y ?? 1) || 1;
        const sz = Number(scale?.z ?? 1) || 1;
        const rot = buildRotationMatrix(
          fields.NewYAxis as { x?: number; y?: number; z?: number } | undefined,
          Number(fields.SpinAngle ?? 0),
        );
        // Apply rotation to query point, then scale to unit sphere
        const [erx, ery, erz] = mat3Apply(rot, x, y, z);
        const esx = erx / sx, esy = ery / sy, esz = erz / sz;
        result = Math.sqrt(esx * esx + esy * esy + esz * esz) - 1;
        break;
      }

      case "Cuboid": {
        const scale = fields.Scale as { x?: number; y?: number; z?: number } | undefined
                   ?? fields.Size as { x?: number; y?: number; z?: number } | undefined;
        const sx = Number(scale?.x ?? 1) || 1;
        const sy = Number(scale?.y ?? 1) || 1;
        const sz = Number(scale?.z ?? 1) || 1;
        const rot = buildRotationMatrix(
          fields.NewYAxis as { x?: number; y?: number; z?: number } | undefined,
          Number(fields.SpinAngle ?? 0),
        );
        const [crx, cry, crz] = mat3Apply(rot, x, y, z);
        // Scale to unit box, then standard box SDF
        const csx = crx / sx, csy = cry / sy, csz = crz / sz;
        const qx = Math.abs(csx) - 1.0;
        const qy = Math.abs(csy) - 1.0;
        const qz = Math.abs(csz) - 1.0;
        const outside = Math.sqrt(
          Math.max(qx, 0) ** 2 + Math.max(qy, 0) ** 2 + Math.max(qz, 0) ** 2,
        );
        const inside = Math.min(Math.max(qx, qy, qz), 0);
        result = outside + inside;
        break;
      }

      case "Cylinder": {
        const rot = buildRotationMatrix(
          fields.NewYAxis as { x?: number; y?: number; z?: number } | undefined,
          Number(fields.SpinAngle ?? 0),
        );
        const [cylrx, cylry, cylrz] = mat3Apply(rot, x, y, z);
        // Cylinder SDF along local Y axis
        const radius = Number(fields.Radius ?? 1) || 1;
        const height = Number(fields.Height ?? 2);
        const halfH = height / 2;
        const dRadial = Math.sqrt(cylrx * cylrx + cylrz * cylrz) - radius;
        const dVertical = Math.abs(cylry) - halfH;
        const outsideR = Math.max(dRadial, 0);
        const outsideV = Math.max(dVertical, 0);
        result = Math.sqrt(outsideR ** 2 + outsideV ** 2) + Math.min(Math.max(dRadial, dVertical), 0);
        break;
      }

      case "Plane": {
        const n = fields.Normal as { x?: number; y?: number; z?: number } | undefined;
        let nx = Number(n?.x ?? 0);
        let ny = Number(n?.y ?? 1);
        let nz = Number(n?.z ?? 0);
        // Normalize the normal vector
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        nx /= len; ny /= len; nz /= len;
        const d = Number(fields.Distance ?? 0);
        result = nx * x + ny * y + nz * z - d;
        break;
      }

      case "Shell": {
        // Shell uses angle × distance curve multiplication (ShellDensity.java)
        const shellAxis = fields.Axis as { x?: number; y?: number; z?: number } | undefined;
        let sax = Number(shellAxis?.x ?? 0);
        let say = Number(shellAxis?.y ?? 1);
        let saz = Number(shellAxis?.z ?? 0);
        const saLen = Math.sqrt(sax * sax + say * say + saz * saz);
        if (saLen > 1e-9) { sax /= saLen; say /= saLen; saz /= saLen; }
        const shellMirror = fields.Mirror === true;

        const shellDist = Math.sqrt(x * x + y * y + z * z);
        // Evaluate DistanceCurve on distance
        const distAmplitude = applyCurve("DistanceCurve", shellDist, inputs);
        if (Math.abs(distAmplitude) < 1e-12) { result = 0; break; }

        if (shellDist <= 1e-9) {
          // At origin, angle is undefined — return amplitude directly
          result = distAmplitude;
          break;
        }

        // Angle between position vector and axis (in degrees, 0-180)
        const posLen = shellDist;
        const dotP = (x * sax + y * say + z * saz) / posLen;
        let angleDeg = Math.acos(Math.max(-1, Math.min(1, dotP))) * (180 / Math.PI);
        if (shellMirror && angleDeg > 90) angleDeg = 180 - angleDeg;

        // Evaluate AngleCurve on angle
        const angleVal = applyCurve("AngleCurve", angleDeg, inputs);
        result = distAmplitude * angleVal;
        break;
      }

      // ── Combinators ───────────────────────────────────────────────

      case "Conditional": {
        const cond = getInput(inputs, "Condition", x, y, z);
        const threshold = Number(fields.Threshold ?? 0);
        result = cond >= threshold
          ? getInput(inputs, "TrueInput", x, y, z)
          : getInput(inputs, "FalseInput", x, y, z);
        break;
      }

      case "MinFunction":
        result = Math.min(
          getInput(inputs, "Inputs[0]", x, y, z),
          getInput(inputs, "Inputs[1]", x, y, z),
        );
        break;

      case "MaxFunction":
        result = Math.max(
          getInput(inputs, "Inputs[0]", x, y, z),
          getInput(inputs, "Inputs[1]", x, y, z),
        );
        break;

      case "AverageFunction":
        result = (getInput(inputs, "Inputs[0]", x, y, z) + getInput(inputs, "Inputs[1]", x, y, z)) / 2;
        break;

      case "Blend": {
        const a = getInput(inputs, "InputA", x, y, z);
        const b = getInput(inputs, "InputB", x, y, z);
        const hasFactor = inputs.has("Factor");
        const f = hasFactor ? getInput(inputs, "Factor", x, y, z) : 0.5;
        result = a + (b - a) * f;
        break;
      }

      case "Switch": {
        // Try matching against context switchState first
        const switchStates = fields.SwitchStates as (string | number)[] | undefined;
        if (switchStates && switchStates.length > 0) {
          let matched = false;
          for (let i = 0; i < switchStates.length; i++) {
            if (hashSeed(switchStates[i] as string | number | undefined) === ctxSwitchState) {
              const src = inputs.get(`Inputs[${i}]`);
              result = src ? evaluate(src, x, y, z) : 0;
              matched = true;
              break;
            }
          }
          if (!matched) result = 0;
        } else {
          // Fallback: use Selector field as index
          const selector = Math.max(0, Math.floor(Number(fields.Selector ?? 0)));
          const src = inputs.get(`Inputs[${selector}]`);
          result = src ? evaluate(src, x, y, z) : 0;
        }
        break;
      }

      case "BlendCurve": {
        const a = getInput(inputs, "InputA", x, y, z);
        const b = getInput(inputs, "InputB", x, y, z);
        const rawFactor = getInput(inputs, "Factor", x, y, z);
        const curvedFactor = applyCurve("Curve", rawFactor, inputs);
        result = a + (b - a) * curvedFactor;
        break;
      }

      case "MultiMix": {
        const keys = (fields.Keys as number[]) ?? [];
        if (keys.length === 0) { result = 0; break; }
        const selector = getInput(inputs, "Selector", x, y, z);
        // Find the bracketing key indices
        let lo = 0;
        for (let i = 1; i < keys.length; i++) {
          if (keys[i] <= selector) lo = i;
        }
        const hi = Math.min(lo + 1, keys.length - 1);
        if (lo === hi) {
          result = getInput(inputs, `Densities[${lo}]`, x, y, z);
        } else {
          const t = Math.max(0, Math.min(1, (selector - keys[lo]) / (keys[hi] - keys[lo])));
          const a = getInput(inputs, `Densities[${lo}]`, x, y, z);
          const b = getInput(inputs, `Densities[${hi}]`, x, y, z);
          result = a + (b - a) * t;
        }
        break;
      }

      // ── Curve ─────────────────────────────────────────────────────

      case "CurveFunction": {
        const inputVal = getInput(inputs, "Input", x, y, z);
        result = applyCurve("Curve", inputVal, inputs);
        break;
      }

      case "SplineFunction": {
        const inputVal = getInput(inputs, "Input", x, y, z);
        result = applySpline(fields, inputVal);
        break;
      }

      // ── Position transforms ───────────────────────────────────────

      case "TranslatedPosition": {
        const tr = fields.Translation as { x?: number; y?: number; z?: number } | undefined;
        const dx = Number(tr?.x ?? 0);
        const dy = Number(tr?.y ?? 0);
        const dz = Number(tr?.z ?? 0);
        result = getInput(inputs, "Input", x - dx, y - dy, z - dz);
        break;
      }

      case "ScaledPosition": {
        const sc = fields.Scale as { x?: number; y?: number; z?: number } | undefined;
        const sx = Number(sc?.x ?? 1) || 1;
        const sy = Number(sc?.y ?? 1) || 1;
        const sz = Number(sc?.z ?? 1) || 1;
        result = getInput(inputs, "Input", x / sx, y / sy, z / sz);
        break;
      }

      case "RotatedPosition": {
        const angleDeg = Number(fields.AngleDegrees ?? 0);
        const rad = (angleDeg * Math.PI) / 180;
        const cos = Math.cos(rad);
        const sin = Math.sin(rad);
        // Rotate around Y axis
        const rx = x * cos + z * sin;
        const rz = -x * sin + z * cos;
        result = getInput(inputs, "Input", rx, y, rz);
        break;
      }

      case "MirroredPosition": {
        const axis = String(fields.Axis ?? "X");
        const mx = axis === "X" ? Math.abs(x) : x;
        const my = axis === "Y" ? Math.abs(y) : y;
        const mz = axis === "Z" ? Math.abs(z) : z;
        result = getInput(inputs, "Input", mx, my, mz);
        break;
      }

      case "QuantizedPosition": {
        const step = Number(fields.StepSize ?? 1) || 1;
        result = getInput(inputs, "Input", Math.floor(x / step) * step, Math.floor(y / step) * step, Math.floor(z / step) * step);
        break;
      }

      // ── Domain warp ───────────────────────────────────────────────

      case "DomainWarp2D": {
        const amp = Number(fields.Amplitude ?? 1.0);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const freq = Number(fields.Frequency ?? 0.01);
        const noiseX = getNoise2D(seed);
        const noiseZ = getNoise2D(seed + 1);
        const warpX = noiseX(x * freq, z * freq) * amp;
        const warpZ = noiseZ(x * freq, z * freq) * amp;
        result = getInput(inputs, "Input", x + warpX, y, z + warpZ);
        break;
      }

      case "DomainWarp3D": {
        const amp = Number(fields.Amplitude ?? 1.0);
        const seed = hashSeed(fields.Seed as string | number | undefined);
        const freq = Number(fields.Frequency ?? 0.01);
        const noiseX = getNoise3D(seed);
        const noiseY = getNoise3D(seed + 1);
        const noiseZ = getNoise3D(seed + 2);
        const warpX = noiseX(x * freq, y * freq, z * freq) * amp;
        const warpY = noiseY(x * freq, y * freq, z * freq) * amp;
        const warpZ = noiseZ(x * freq, y * freq, z * freq) * amp;
        result = getInput(inputs, "Input", x + warpX, y + warpY, z + warpZ);
        break;
      }

      // ── Passthrough / caching ─────────────────────────────────────

      case "CacheOnce": {
        const key = `${nodeId}:${x}:${y}:${z}`;
        const cached = memoCache.get(key);
        if (cached !== undefined) {
          result = cached;
        } else {
          result = getInput(inputs, "Input", x, y, z);
          memoCache.set(key, result);
        }
        break;
      }

      case "FlatCache":
      case "Wrap":
      case "Passthrough":
      case "Debug":
        result = getInput(inputs, "Input", x, y, z);
        break;

      // ── Unsupported ───────────────────────────────────────────────

      default:
        if (UNSUPPORTED_TYPES.has(type)) {
          result = 0;
        } else {
          // Unknown type — try to follow Input handle
          const src = inputs.get("Input") ?? inputs.get("Inputs[0]");
          result = src ? evaluate(src, x, y, z) : 0;
        }
        break;
    }

    visiting.delete(nodeId);
    return result;
  }

  function getInput(inputs: Map<string, string>, handle: string, x: number, y: number, z: number): number {
    const src = inputs.get(handle);
    if (!src) return 0;
    return evaluate(src, x, y, z);
  }

  return {
    rootId: root.id,
    evaluate,
    clearMemo: () => memoCache.clear(),
  };
}

/* ── Main export ──────────────────────────────────────────────────── */

/**
 * Evaluate a density node graph over a 2D grid.
 * Samples at (x, yLevel, z) for each grid cell, where x and z vary over [rangeMin, rangeMax].
 */
export function evaluateDensityGrid(
  nodes: Node[],
  edges: Edge[],
  resolution: number,
  rangeMin: number,
  rangeMax: number,
  yLevel: number,
  rootNodeId?: string,
  options?: EvaluationOptions,
): DensityGridResult {
  const n = Math.max(1, resolution);
  const values = new Float32Array(n * n);

  const ctx = createEvaluationContext(nodes, edges, rootNodeId, options);
  if (!ctx) {
    return { values, minValue: 0, maxValue: 0 };
  }

  // Evaluate grid
  const step = (rangeMax - rangeMin) / n;
  let minVal = Infinity;
  let maxVal = -Infinity;

  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const sx = rangeMin + col * step;
      const sz = rangeMin + row * step;

      // Clear per-sample state
      ctx.clearMemo();

      const val = ctx.evaluate(ctx.rootId, sx, yLevel, sz);
      values[row * n + col] = val;
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }
  }

  if (!isFinite(minVal)) minVal = 0;
  if (!isFinite(maxVal)) maxVal = 0;

  return { values, minValue: minVal, maxValue: maxVal };
}

/* ── Fractal Brownian Motion helpers ──────────────────────────────── */

function fbm2D(
  noise: (x: number, y: number) => number,
  x: number, z: number,
  freq: number, octaves: number, lacunarity: number, gain: number,
): number {
  let sum = 0;
  let amp = 1;
  let f = freq;
  for (let i = 0; i < octaves; i++) {
    sum += noise(x * f, z * f) * amp;
    f *= lacunarity;
    amp *= gain;
  }
  return sum;
}

function fbm3D(
  noise: (x: number, y: number, z: number) => number,
  x: number, y: number, z: number,
  freq: number, octaves: number, lacunarity: number, gain: number,
): number {
  let sum = 0;
  let amp = 1;
  let f = freq;
  for (let i = 0; i < octaves; i++) {
    sum += noise(x * f, y * f, z * f) * amp;
    f *= lacunarity;
    amp *= gain;
  }
  return sum;
}

function ridgeFbm2D(
  noise: (x: number, y: number) => number,
  x: number, z: number,
  freq: number, octaves: number,
): number {
  let sum = 0;
  let amp = 1;
  let f = freq;
  for (let i = 0; i < octaves; i++) {
    const n = 1 - Math.abs(noise(x * f, z * f));
    sum += n * n * amp;
    f *= 2;
    amp *= 0.5;
  }
  return sum * 2 - 1;
}

function ridgeFbm3D(
  noise: (x: number, y: number, z: number) => number,
  x: number, y: number, z: number,
  freq: number, octaves: number,
): number {
  let sum = 0;
  let amp = 1;
  let f = freq;
  for (let i = 0; i < octaves; i++) {
    const n = 1 - Math.abs(noise(x * f, y * f, z * f));
    sum += n * n * amp;
    f *= 2;
    amp *= 0.5;
  }
  return sum * 2 - 1;
}

function voronoiFbm2D(
  noise: (x: number, y: number) => number,
  x: number, z: number,
  freq: number, octaves: number, lacunarity: number, gain: number,
): number {
  let sum = 0;
  let amp = 1;
  let f = freq;
  for (let i = 0; i < octaves; i++) {
    sum += noise(x * f, z * f) * amp;
    f *= lacunarity;
    amp *= gain;
  }
  return sum;
}

function voronoiFbm3D(
  noise: (x: number, y: number, z: number) => number,
  x: number, y: number, z: number,
  freq: number, octaves: number, lacunarity: number, gain: number,
): number {
  let sum = 0;
  let amp = 1;
  let f = freq;
  for (let i = 0; i < octaves; i++) {
    sum += noise(x * f, y * f, z * f) * amp;
    f *= lacunarity;
    amp *= gain;
  }
  return sum;
}
