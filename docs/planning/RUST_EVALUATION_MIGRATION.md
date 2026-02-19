# Rust Evaluation Pipeline Migration Plan

> Migrate density, volume, material, and voxel evaluation from TypeScript Web Workers to the Rust backend.
> Each phase is independently shippable and testable. No phase breaks the existing application.

---

## Table of Contents

- [Background & Motivation](#background--motivation)
- [Current Architecture](#current-architecture)
- [x] [Target Architecture](#target-architecture)
- [x] [Phase 0: Infrastructure & Testing Harness](#phase-0-infrastructure--testing-harness)
- [x] [Phase 1: Rust Graph Evaluator — Core Types](#phase-1-rust-graph-evaluator--core-types)
- [x] [Phase 2: Rust Graph Evaluator — All 68 Density Types](#phase-2-rust-graph-evaluator--all-68-density-types)
- [x] [Phase 3: Grid & Volume Evaluation Commands](#phase-3-grid--volume-evaluation-commands)
- [ ] [Phase 4: Frontend Integration with Feature Flag](#phase-4-frontend-integration-with-feature-flag)
- [ ] [Phase 5: Material Evaluation in Rust](#phase-5-material-evaluation-in-rust)
- [ ] [Phase 6: Voxel Extraction & Mesh Building](#phase-6-voxel-extraction--mesh-building)
- [ ] [Phase 7: Progressive Streaming & Caching](#phase-7-progressive-streaming--caching)
- [ ] [Phase 8: Cleanup & Deprecation](#phase-8-cleanup--deprecation)
- [Appendix A: Node Type Inventory](#appendix-a-node-type-inventory)
- [Appendix B: Data Format Contracts](#appendix-b-data-format-contracts)
- [Appendix C: Noise Parity Reference](#appendix-c-noise-parity-reference)

---

## Background & Motivation

### Why the current IPC concern is wrong

ADR-3 in `CONTEXT.md` states:

> "Avoids IPC overhead for per-pixel evaluation (thousands of calls per frame)"

This assumes **one IPC call per pixel**. In practice, both the current Web Worker path and the proposed Rust path make **one call per grid** — the entire NxN or NxNxY evaluation happens inside a single invocation. The data returned is a single `Float32Array`.

**Actual overhead comparison (128×128 grid = 16,384 points):**

| Step                | JS Web Worker | Rust via IPC | Delta   |
|---------------------|---------------|--------------|---------|
| Serialize request   | ~0.5ms        | ~1ms         | +0.5ms  |
| Evaluate all points | ~50ms         | ~5ms         | **-45ms** |
| Serialize response  | ~0.5ms        | ~1ms         | +0.5ms  |
| **Total**           | **~51ms**     | **~7ms**     | **-44ms (7× faster)** |

At 128³ voxel resolution the gap widens to 13-16×. The IPC overhead (~2ms round trip) is negligible compared to the evaluation time saved by native code.

### What we gain

1. **7-100× faster preview** depending on resolution
2. **Single source of truth** — one evaluator to maintain instead of two
3. **rayon parallelism** — multi-threaded grid evaluation for free
4. **SIMD potential** — future `f32x4`/`f32x8` vectorization
5. **~2,500 lines of complex TypeScript removed** from the frontend bundle
6. **Smaller WASM/JS bundle** — drop `simplex-noise` dependency

### What we keep in the frontend

- React Flow graph editor (UI interactions)
- Three.js / React Three Fiber rendering
- Canvas pan/zoom/interaction
- Zustand UI state management
- Preview controls and settings

---

## Current Architecture

```
User edits graph
       │
       ▼
┌──────────────────┐
│  editorStore.ts  │  (nodes, edges, contentFields)
└────────┬─────────┘
         │
         ▼
┌────────────────────────────┐     ┌─────────────────────────────────┐
│ usePreviewEvaluation.ts    │     │ useVoxelEvaluation.ts           │
│  - debounce (configurable) │     │  - debounce                     │
│  - calls evaluateInWorker  │     │  - calls evaluateVolumeInWorker │
└────────┬───────────────────┘     │  - extractSurfaceVoxels (JS)    │
         │                         │  - evaluateMaterialGraph (JS)   │
         ▼                         │  - buildVoxelMeshes (JS)        │
┌────────────────────────────┐     └────────┬────────────────────────┘
│ densityWorkerClient.ts     │              │
│  - spawns Web Worker       │              ▼
│  - postMessage / onmessage │     ┌────────────────────────────────┐
│  - 30s timeout + fallback  │     │ volumeWorkerClient.ts          │
└────────┬───────────────────┘     │  - same pattern                │
         │                         └────────┬───────────────────────┘
         ▼                                  ▼
┌────────────────────────────┐     ┌────────────────────────────────┐
│ densityWorker.ts           │     │ volumeWorker.ts                │
│  (Web Worker thread)       │     │  (Web Worker thread)           │
│  - evaluateDensityGrid()   │     │  - evaluateDensityVolume()     │
│  - 1,600 lines of eval     │     │  - reuses densityEvaluator     │
│  - Float32Array transfer   │     │  - Float32Array transfer       │
└────────────────────────────┘     └────────────────────────────────┘

Supporting files:
  - densityEvaluator.ts    (1,600 lines — 68 density node types)
  - volumeEvaluator.ts     (80 lines — 3D grid loop)
  - hytaleNoise.ts         (300 lines — simplex noise impl)
  - curveEvaluators.ts     (250 lines — 19 curve types)
  - vectorEvaluator.ts     (100 lines — vector providers)
  - materialEvaluator.ts   (700 lines — material graph evaluation)
  - materialResolver.ts    (400 lines — depth-based fallback materials)
  - voxelExtractor.ts      (300 lines — surface voxel extraction)
  - voxelMeshBuilder.ts    (400 lines — greedy meshing + AO)
  - positionEvaluator.ts   (250 lines — position providers)
```

### Rust backend (current — barely used)

```
src-tauri/src/noise/
  evaluator.rs   — 118 lines, parses V2 JSON, supports 5 types only
  nodes.rs       — 140 lines, ConstantNode, SimplexNoise2DNode, SumNode, ClampNode, NormalizerNode

src-tauri/src/commands/
  preview.rs     — evaluate_density command, called with V2 JSON (not React Flow graph)
                   Not used by the frontend preview pipeline at all.
```

---

## Target Architecture

```
User edits graph
       │
       ▼
┌──────────────────┐
│  editorStore.ts  │  (nodes, edges, contentFields)
└────────┬─────────┘
         │
         ▼
┌────────────────────────────┐
│ usePreviewEvaluation.ts    │
│  - debounce (configurable) │
│  - invoke("evaluate_grid") │◄── single Tauri IPC call
│  - receive Float32Array    │
└────────┬───────────────────┘
         │
         ▼  (Tauri IPC — ~1-2ms overhead)
┌═══════════════════════════════════════════════════════════════╗
║  RUST BACKEND                                                ║
║                                                              ║
║  ┌────────────────────────┐   ┌────────────────────────────┐ ║
║  │ commands/preview.rs    │   │ commands/voxel.rs          │ ║
║  │  evaluate_grid()       │   │  evaluate_voxel_preview()  │ ║
║  │  evaluate_volume()     │   │  - volume eval             │ ║
║  │                        │   │  - material eval           │ ║
║  └────────┬───────────────┘   │  - surface extraction      │ ║
║           │                   │  - mesh building           │ ║
║           ▼                   └────────┬───────────────────┘ ║
║  ┌────────────────────────┐            │                     ║
║  │ eval/graph.rs          │◄───────────┘                     ║
║  │  - parse React Flow    │                                  ║
║  │    nodes + edges into  │   ┌────────────────────────────┐ ║
║  │    evaluable graph     │   │ eval/cache.rs              │ ║
║  │                        │   │  - LRU content-hash cache  │ ║
║  └────────┬───────────────┘   │  - skip re-eval on same    │ ║
║           │                   │    graph + params           │ ║
║           ▼                   └────────────────────────────┘ ║
║  ┌────────────────────────┐                                  ║
║  │ eval/nodes.rs          │                                  ║
║  │  - 68 density types    │   ┌────────────────────────────┐ ║
║  │  - trait NodeEval      │   │ eval/noise.rs              │ ║
║  │  - all math ops        │   │  - Hytale-parity simplex   │ ║
║  │  - curves, warps       │◄──│  - permutation table       │ ║
║  │  - coordinate access   │   │  - FBM, ridge, voronoi     │ ║
║  └────────────────────────┘   └────────────────────────────┘ ║
║                                                              ║
║  Grid evaluation uses rayon::par_iter for multi-core.        ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Phase 0: Infrastructure & Testing Harness

**Goal:** Set up the Rust module structure, cross-language test fixtures, and a comparison tool so every subsequent phase can be verified automatically.

**Duration:** 1-2 days

### 0.1 Create Rust module skeleton

Create the new evaluation module alongside the existing `noise/` module (which will be replaced incrementally):

```
src-tauri/src/
├── eval/                    # NEW — graph-based evaluator
│   ├── mod.rs               # pub mod declarations
│   ├── graph.rs             # React Flow graph → evaluable tree
│   ├── nodes.rs             # NodeEval trait + all node implementations
│   ├── noise.rs             # Hytale-parity noise functions
│   ├── curves.rs            # Curve evaluators (19 types)
│   ├── vectors.rs           # Vector provider evaluation
│   ├── grid.rs              # 2D grid evaluation loop (+ rayon)
│   ├── volume.rs            # 3D volume evaluation loop (+ rayon)
│   └── tests.rs             # Unit tests
├── noise/                   # EXISTING — keep until Phase 8
│   ├── evaluator.rs
│   └── nodes.rs
```

Update `lib.rs`:

```rust
mod eval;  // add this
```

Add `rayon` to `Cargo.toml` (do NOT add it yet — Phase 3 will add it when grid evaluation is implemented. For now, sequential evaluation is sufficient for parity testing):

```toml
# Phase 3 addition:
# rayon = "1.10"
```

**Test:** `cargo build` succeeds with empty module files.

### 0.2 Create cross-language test fixtures

Create a shared test fixture file that both Rust and TypeScript tests consume. This ensures both evaluators are tested against the exact same inputs and expected outputs.

Create `tests/fixtures/eval_cases.json`:

```json
[
  {
    "name": "constant_42",
    "nodes": [
      { "id": "c", "data": { "type": "Constant", "fields": { "Value": 42 } } }
    ],
    "edges": [],
    "sample_points": [
      { "x": 0, "y": 64, "z": 0, "expected": 42.0 },
      { "x": 100, "y": 0, "z": -50, "expected": 42.0 }
    ]
  },
  {
    "name": "sum_of_constants",
    "nodes": [
      { "id": "a", "data": { "type": "Constant", "fields": { "Value": 10 } } },
      { "id": "b", "data": { "type": "Constant", "fields": { "Value": 20 } } },
      { "id": "s", "data": { "type": "Sum", "fields": {} } }
    ],
    "edges": [
      { "source": "a", "target": "s", "targetHandle": "Input[0]" },
      { "source": "b", "target": "s", "targetHandle": "Input[1]" }
    ],
    "sample_points": [
      { "x": 0, "y": 0, "z": 0, "expected": 30.0 }
    ]
  }
]
```

This file will grow as each node type is implemented. The format matches the React Flow node/edge shape so both sides can parse it directly.

**Test:** Both `vitest` and `cargo test` load and parse the fixture file.

### 0.3 Parity comparison command

Add a Tauri command that evaluates a graph at specific points using the Rust evaluator, returning raw values. The frontend can call this alongside its own JS evaluation and compare results.

```rust
// src-tauri/src/commands/preview.rs — add alongside existing evaluate_density

#[tauri::command]
pub fn evaluate_points(
    nodes: Vec<serde_json::Value>,
    edges: Vec<serde_json::Value>,
    points: Vec<[f64; 3]>,        // [[x, y, z], ...]
    root_node_id: Option<String>,
    content_fields: Option<HashMap<String, f64>>,
) -> Result<Vec<f64>, String> {
    // Phase 1 will implement this
    Err("Not yet implemented".into())
}
```

**Test:** Command is registered and callable (returns error until Phase 1).

### 0.4 Frontend parity test utility

Create a dev-only utility that runs both JS and Rust evaluators and compares results:

```typescript
// src/utils/__tests__/parityCheck.ts (dev-only, not shipped)

export async function checkParity(
  nodes: Node[],
  edges: Edge[],
  samplePoints: Array<{ x: number; y: number; z: number }>,
  tolerance: number = 1e-4,
): Promise<{ match: boolean; diffs: Array<{ point: any; js: number; rust: number }> }> {
  // JS evaluation
  const ctx = createEvaluationContext(nodes, edges);
  const jsValues = samplePoints.map(p => ctx!.evaluate(ctx!.rootId, p.x, p.y, p.z));

  // Rust evaluation
  const rustValues = await invoke<number[]>('evaluate_points', {
    nodes: nodes.map(serializeNode),
    edges: edges.map(serializeEdge),
    points: samplePoints.map(p => [p.x, p.y, p.z]),
  });

  // Compare
  const diffs = [];
  for (let i = 0; i < samplePoints.length; i++) {
    if (Math.abs(jsValues[i] - rustValues[i]) > tolerance) {
      diffs.push({ point: samplePoints[i], js: jsValues[i], rust: rustValues[i] });
    }
  }

  return { match: diffs.length === 0, diffs };
}
```

**Test:** Utility compiles, returns mismatch results (Rust returns error).

### Deliverables — Phase 0

- [ ] `src-tauri/src/eval/mod.rs` exists with stub submodules
- [ ] `tests/fixtures/eval_cases.json` with ≥5 test cases
- [ ] `evaluate_points` command registered in `lib.rs`
- [ ] Frontend parity utility exists
- [ ] `cargo build` and `pnpm build` both succeed
- [ ] All existing tests pass (no regressions)

---

## Phase 1: Rust Graph Evaluator — Core Types

**Goal:** Implement a Rust evaluator that accepts React Flow graph format (nodes + edges) and evaluates the 15 most common density types. Achieve parity with the TypeScript evaluator for these types.

**Duration:** 3-5 days

### Why React Flow format, not V2 JSON?

The existing Rust evaluator (`noise/evaluator.rs`) accepts V2 JSON (Hytale's nested format). But the frontend operates on React Flow's flat graph format (nodes array + edges array). Converting to V2 JSON first would require running `graphToJson.ts` → `internalToHytale.ts` on every preview update — an expensive and lossy round-trip.

The new evaluator accepts the same `{ nodes, edges }` shape that the frontend already has in memory. Field values are read from `node.data.fields.*`. Edges encode connections as `{ source, target, targetHandle }`.

### 1.1 Graph representation

```rust
// src-tauri/src/eval/graph.rs

use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

/// A node from the React Flow graph, as sent over IPC.
/// We only deserialize the fields we need — id and data.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub data: NodeData,
    #[serde(rename = "type")]
    pub node_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeData {
    #[serde(rename = "type", default)]
    pub density_type: Option<String>,
    #[serde(default)]
    pub fields: HashMap<String, Value>,
    /// User-designated output node flag
    #[serde(rename = "_outputNode", default)]
    pub is_output: bool,
    /// Biome field tag (e.g. "Terrain")
    #[serde(rename = "_biomeField", default)]
    pub biome_field: Option<String>,
}

/// An edge from the React Flow graph.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    #[serde(rename = "targetHandle")]
    pub target_handle: Option<String>,
}

/// Parsed, indexed graph ready for evaluation.
pub struct EvalGraph {
    pub nodes: HashMap<String, GraphNode>,
    /// target_id → { handle_name → source_id }
    pub inputs: HashMap<String, HashMap<String, String>>,
    pub root_id: String,
}

impl EvalGraph {
    pub fn from_raw(
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        root_node_id: Option<&str>,
    ) -> Result<Self, String> {
        // Build node map
        let node_map: HashMap<String, GraphNode> =
            nodes.into_iter().map(|n| (n.id.clone(), n)).collect();

        // Build input adjacency: target → { handle → source }
        let mut inputs: HashMap<String, HashMap<String, String>> = HashMap::new();
        for edge in &edges {
            let handle = edge.target_handle.clone().unwrap_or_else(|| "Input".into());
            inputs
                .entry(edge.target.clone())
                .or_default()
                .insert(handle, edge.source.clone());
        }

        // Find root (same strategy as TS: outputNode → tagged → terminal → first density)
        let root_id = Self::find_root(&node_map, &edges, root_node_id)?;

        Ok(EvalGraph {
            nodes: node_map,
            inputs,
            root_id,
        })
    }

    fn find_root(
        nodes: &HashMap<String, GraphNode>,
        edges: &[GraphEdge],
        explicit_id: Option<&str>,
    ) -> Result<String, String> {
        // ... root-finding logic matching TS findDensityRoot()
        // Strategy 0: explicit root_node_id parameter
        // Strategy 1: _outputNode === true
        // Strategy 2: _biomeField === "Terrain"
        // Strategy 3: terminal nodes (no outgoing edges) with density type
        // Strategy 4: any density node
        todo!()
    }
}
```

**Test:** Parse a fixture graph, verify root node is found correctly.

### 1.2 Noise implementation with Hytale parity

This is the most critical parity requirement. The TypeScript implementation in `hytaleNoise.ts` uses:

1. **Java `String.hashCode()`** for seed → integer conversion
2. **Fisher-Yates shuffle** with mulberry32 PRNG for permutation table
3. **Standard simplex noise** with specific gradient vectors and skew constants
4. **Scale factor 70.0** (2D) and **32.0** (3D)

The Rust implementation must produce **identical output** for the same inputs. The existing `noise/nodes.rs` uses `fastnoise-lite` which does NOT match Hytale's implementation — it uses OpenSimplex2 with different gradient tables.

```rust
// src-tauri/src/eval/noise.rs

/// Java-compatible String.hashCode() — must match hytaleNoise.ts
pub fn java_string_hash_code(s: &str) -> i32 {
    let mut hash: i32 = 0;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as i32);
    }
    hash
}

/// Convert a seed value to integer (matching TS seedToInt)
pub fn seed_to_int(seed: &Value) -> i32 {
    match seed {
        Value::Number(n) => n.as_i64().unwrap_or(0) as i32,
        Value::String(s) => java_string_hash_code(s),
        _ => 0,
    }
}

/// Mulberry32 PRNG — must produce same sequence as TS version
pub struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    pub fn new(seed: i32) -> Self {
        Self { state: seed as u32 }
    }

    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut t = self.state ^ (self.state >> 15);
        t = t.wrapping_mul(1 | self.state);
        t = t.wrapping_add(t.wrapping_mul(61 | (t ^ (t >> 7)))) ^ t;
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

/// Build a 512-entry permutation table using Fisher-Yates shuffle.
/// Must match hytaleNoise.ts buildPermutationTable().
pub fn build_perm_table(seed: i32) -> [u8; 512] { ... }

/// 2D simplex noise — must match createHytaleNoise2D() exactly.
/// Returns value in approximately [-1, 1].
pub fn simplex_2d(perm: &[u8; 512], x: f64, y: f64) -> f64 { ... }

/// 3D simplex noise — must match createHytaleNoise3D() exactly.
pub fn simplex_3d(perm: &[u8; 512], x: f64, y: f64, z: f64) -> f64 { ... }

/// FBM (fractal Brownian motion) 2D
pub fn fbm_2d(perm: &[u8; 512], x: f64, z: f64, freq: f64, octaves: u32, lac: f64, gain: f64) -> f64 { ... }

/// FBM 3D
pub fn fbm_3d(perm: &[u8; 512], x: f64, y: f64, z: f64, freq: f64, octaves: u32, lac: f64, gain: f64) -> f64 { ... }

/// Ridge noise 2D
pub fn ridge_fbm_2d(perm: &[u8; 512], x: f64, z: f64, freq: f64, octaves: u32) -> f64 { ... }

/// Ridge noise 3D
pub fn ridge_fbm_3d(perm: &[u8; 512], x: f64, y: f64, z: f64, freq: f64, octaves: u32) -> f64 { ... }
```

**Test:** For 100 random (x, z) coordinates and 5 different seeds, verify that `simplex_2d` produces values within `1e-10` of the TypeScript implementation. This is the single most important parity test in the entire migration.

### 1.3 NodeEval trait and first 15 types

```rust
// src-tauri/src/eval/nodes.rs

/// Result of evaluating a node. Most return f64, but we use this
/// to allow future expansion (e.g. vector results).
pub type EvalResult = f64;

/// Context passed to every evaluation call.
pub struct EvalContext<'a> {
    pub graph: &'a EvalGraph,
    /// Per-sample memoization cache (cleared between samples)
    pub memo: HashMap<String, f64>,
    /// Cycle detection
    pub visiting: HashSet<String>,
    /// Noise function caches (persist across samples)
    pub noise_2d_cache: HashMap<i32, Box<dyn Fn(f64, f64) -> f64>>,
    pub noise_3d_cache: HashMap<i32, Box<dyn Fn(f64, f64, f64) -> f64>>,
    /// Content fields (from WorldStructure)
    pub content_fields: HashMap<String, f64>,
}

/// Evaluate a single node at position (x, y, z).
/// Recursive — follows edges to evaluate input nodes.
pub fn evaluate(ctx: &mut EvalContext, node_id: &str, x: f64, y: f64, z: f64) -> f64 {
    // Cycle guard
    if ctx.visiting.contains(node_id) { return 0.0; }

    // Memo check
    if let Some(&cached) = ctx.memo.get(node_id) { return cached; }

    ctx.visiting.insert(node_id.to_string());

    let result = evaluate_inner(ctx, node_id, x, y, z);

    ctx.visiting.remove(node_id);
    ctx.memo.insert(node_id.to_string(), result);

    result
}
```

**Phase 1 node types (15 types — covers ~80% of real-world graphs):**

| # | Type | Category | Complexity | Notes |
|---|------|----------|------------|-------|
| 1 | `Constant` | Constant | Trivial | `fields.Value` |
| 2 | `Zero` | Constant | Trivial | Always 0 |
| 3 | `One` | Constant | Trivial | Always 1 |
| 4 | `SimplexNoise2D` | Noise | Medium | Must match `hytaleNoise.ts` exactly |
| 5 | `SimplexNoise3D` | Noise | Medium | Must match `hytaleNoise.ts` exactly |
| 6 | `Sum` | Arithmetic | Easy | Array input handle `Input[0]`, `Input[1]`, ... |
| 7 | `Negate` | Arithmetic | Trivial | Single input |
| 8 | `Clamp` | Range | Easy | `fields.Min`, `fields.Max` |
| 9 | `Normalizer` | Range | Easy | Linear remap |
| 10 | `LinearTransform` | Arithmetic | Easy | `Scale * input + Offset` |
| 11 | `CoordinateX` | Position | Trivial | Returns x |
| 12 | `CoordinateY` | Position | Trivial | Returns y |
| 13 | `CoordinateZ` | Position | Trivial | Returns z |
| 14 | `Abs` | Arithmetic | Trivial | |
| 15 | `Product` | Arithmetic | Easy | Array input handle, short-circuit on 0 |

For each type, the implementation reads fields from `node.data.fields` and input edges from `graph.inputs[node_id]`, exactly mirroring the TypeScript `evaluate()` switch statement in `densityEvaluator.ts` (lines 492-1595).

**Array handles:** Some nodes (Sum, Product, WeightedSum) accept an array of inputs. In React Flow, these are encoded as edges with target handles `Input[0]`, `Input[1]`, `Input[2]`, etc. The evaluator collects all edges matching the pattern `Input[N]` and evaluates them in order.

```rust
/// Collect array inputs matching "HandleName[0]", "HandleName[1]", etc.
fn get_array_inputs(
    ctx: &mut EvalContext,
    node_id: &str,
    base_handle: &str,
    x: f64, y: f64, z: f64,
) -> Vec<f64> {
    let inputs = match ctx.graph.inputs.get(node_id) {
        Some(m) => m,
        None => return vec![],
    };

    let mut indexed: Vec<(usize, f64)> = vec![];
    for (handle, source_id) in inputs {
        if let Some(rest) = handle.strip_prefix(base_handle) {
            if let Some(idx_str) = rest.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    let val = evaluate(ctx, source_id, x, y, z);
                    indexed.push((idx, val));
                }
            }
        }
        // Also accept bare handle name as index 0
        if handle == base_handle {
            let val = evaluate(ctx, source_id, x, y, z);
            indexed.push((0, val));
        }
    }

    indexed.sort_by_key(|(i, _)| *i);
    indexed.into_iter().map(|(_, v)| v).collect()
}
```

### 1.4 Wire up the `evaluate_points` command

```rust
// src-tauri/src/commands/preview.rs

#[tauri::command]
pub fn evaluate_points(
    nodes: Vec<serde_json::Value>,
    edges: Vec<serde_json::Value>,
    points: Vec<[f64; 3]>,
    root_node_id: Option<String>,
    content_fields: Option<HashMap<String, f64>>,
) -> Result<Vec<f64>, String> {
    // Deserialize into our graph types
    let graph_nodes: Vec<GraphNode> = nodes
        .into_iter()
        .map(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    let graph_edges: Vec<GraphEdge> = edges
        .into_iter()
        .map(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    let graph = EvalGraph::from_raw(
        graph_nodes,
        graph_edges,
        root_node_id.as_deref(),
    )?;

    let mut ctx = EvalContext::new(&graph, content_fields.unwrap_or_default());

    let results: Vec<f64> = points
        .iter()
        .map(|[x, y, z]| {
            ctx.clear_memo();
            evaluate(&mut ctx, &graph.root_id, *x, *y, *z)
        })
        .collect();

    Ok(results)
}
```

### Deliverables — Phase 1

- [ ] `eval/noise.rs` passes parity tests against TypeScript for all noise types
- [ ] `eval/graph.rs` parses React Flow graph format and finds root node
- [ ] `eval/nodes.rs` evaluates 15 density types with correct results
- [ ] `evaluate_points` command returns correct values
- [ ] Fixture tests pass for all 15 types
- [ ] Parity check: for each type, Rust output matches JS output within `1e-6`
- [ ] No regressions — all existing tests pass

### How to test Phase 1

1. **Rust unit tests** (`cargo test`):
   - Noise parity: sample 100 points, compare against hardcoded expected values generated by running the TS noise implementation
   - Each node type: fixture-driven tests from `eval_cases.json`

2. **Frontend parity** (dev mode only):
   - Add a temporary "Check Parity" button in the preview panel (behind `import.meta.env.DEV` guard)
   - For the current graph, evaluate 50 random points with both JS and Rust
   - Log any differences to console
   - This is manual verification during development, not automated CI

3. **Cross-reference with existing test suite**:
   - Run `pnpm test` — all existing `densityEvaluator.test.ts` tests still pass
   - The Rust evaluator does not replace anything yet — it runs alongside

---

## Phase 2: Rust Graph Evaluator — All 68 Density Types

**Goal:** Implement the remaining 53 density node types. After this phase, the Rust evaluator handles every node type the TypeScript evaluator supports.

**Duration:** 5-8 days (large but mechanical — each type is a self-contained switch arm)

### Implementation order

Types are grouped by dependency. Each group can be tested independently. Each group is a separate commit.

#### Group A: Simple Arithmetic (8 types)

These have no special dependencies — just single-input math.

| Type | Formula | TS line ref |
|------|---------|-------------|
| `SquareRoot` | `√|input|` | L661 |
| `CubeRoot` | `∛input` | L667 |
| `Square` | `input²` | L655 |
| `CubeMath` | `input³` | L661 |
| `Inverse` | `1/input` | L667 |
| `SumSelf` | `input × Count` | L673 |
| `Modulo` | `input % Divisor` | L679 |
| `AmplitudeConstant` | `input × Value` | L686 |

**Test:** Each type with known inputs → expected outputs.

#### Group B: Dual-Input & Aggregation (8 types)

| Type | Notes | TS line ref |
|------|-------|-------------|
| `Pow` | `|input|^Exponent × sign(input)` | L693 |
| `WeightedSum` | `Σ(weight_i × input_i) / Σweight_i` | L711 |
| `Interpolate` | `lerp(A, B, Factor)` | L805 |
| `MinFunction` | `min(Input[0], Input[1], ...)` | L721 (as Product-like) |
| `MaxFunction` | `max(...)` | similar |
| `AverageFunction` | `mean(...)` | similar |
| `SmoothMin` | `smin(a, b, k)` | L935 |
| `SmoothMax` | `smax(a, b, k)` | L944 |

**Test:** Dual-input with known values, edge cases (division by zero in WeightedSum).

#### Group C: Range Operations (8 types)

| Type | Notes |
|------|-------|
| `ClampToIndex` | Clamp + floor to integer |
| `DoubleNormalizer` | Two-segment linear remap |
| `RangeChoice` | Conditional based on range |
| `SmoothClamp` | Smooth clamping with steepness |
| `SmoothFloor` | Smooth floor |
| `SmoothCeiling` | Smooth ceiling |
| `Wrap` | Wrap value into range |
| `Passthrough` | Identity (for debugging) |

**Test:** Boundary conditions for clamp/wrap operations.

#### Group D: Noise Types (6 types)

These depend on Phase 1's noise infrastructure.

| Type | Notes |
|------|-------|
| `SimplexRidgeNoise2D` | Ridge FBM 2D |
| `SimplexRidgeNoise3D` | Ridge FBM 3D |
| `VoronoiNoise2D` | Needs voronoi implementation |
| `VoronoiNoise3D` | Needs voronoi implementation |
| `FractalNoise2D` | FBM without amplitude |
| `FractalNoise3D` | FBM without amplitude |

**Voronoi:** The TypeScript implementation uses a custom cell noise implementation with configurable cell type (Euclidean, Manhattan, Chebyshev) and jitter. This is ~30 lines per dimension. Translate directly from `densityEvaluator.ts` lines 43-104.

**Test:** Noise parity tests — 100 sample points per type per seed.

#### Group E: Curves & Splines (3 types)

| Type | Notes |
|------|-------|
| `CurveFunction` | Follows edge to Curve node, samples its evaluator |
| `SplineFunction` | Catmull-Rom or linear interpolation of control points |
| `FlatCache` | Evaluate input once per column (x,z), cache for all y |

**CurveFunction** requires following an edge to a curve node and evaluating the curve. The curve types (Manual, Power, StepFunction, etc.) must also be implemented. See `curveEvaluators.ts` — 19 curve types, each a simple formula.

```rust
// src-tauri/src/eval/curves.rs

pub fn evaluate_curve(curve_type: &str, fields: &HashMap<String, Value>, input: f64) -> f64 {
    match curve_type {
        "Manual" => { /* linear interpolation of sorted points */ }
        "Constant" => { /* fields.Value */ }
        "Power" => { /* input.pow(fields.Exponent) */ }
        "StepFunction" => { /* floor(input * steps) / steps */ }
        "Threshold" => { /* if input >= threshold { 1 } else { 0 } */ }
        "SmoothStep" => { /* hermite interpolation */ }
        "DistanceExponential" => { /* 1 - t^exp */ }
        "DistanceS" => { /* exp(-(|x-offset|/width)^exp * steepness) */ }
        "Inverter" => { -input }
        "Not" => { 1.0 - input }
        "Clamp" => { input.clamp(min, max) }
        "LinearRemap" => { /* affine remap */ }
        _ => input, // passthrough for unknown types
    }
}
```

**Test:** Curve evaluation at 20 sample points per curve type, compare with `curveEvaluators.test.ts` expected values.

#### Group F: Conditionals & Blending (6 types)

| Type | Notes |
|------|-------|
| `Conditional` | `if condition > threshold then A else B` |
| `Switch` | Multi-way select based on SwitchState |
| `Blend` | `lerp(A, B, factor)` with optional curve |
| `BlendCurve` | Blend with curve-modified factor |
| `MultiMix` | N-way blending with selector |
| `SwitchState` | Sets context state, evaluates child |

**Test:** Each conditional path exercised with known inputs.

#### Group G: Position & Distance (10 types)

| Type | Notes |
|------|-------|
| `DistanceFromOrigin` | `√(x² + y² + z²)` |
| `DistanceFromAxis` | Distance from specified axis |
| `DistanceFromPoint` | Distance from (px, py, pz) |
| `AngleFromOrigin` | `atan2(z, x)` in degrees |
| `AngleFromPoint` | Angle relative to a point |
| `TranslatedPosition` | Offset (x,y,z) then evaluate child |
| `ScaledPosition` | Scale (x,y,z) then evaluate child |
| `RotatedPosition` | Rodrigues rotation matrix then evaluate child |
| `MirroredPosition` | Mirror across axis |
| `QuantizedPosition` | Snap to grid |

**RotatedPosition** is the most complex — it builds a Rodrigues rotation matrix from axis + angle fields. Translate from `densityEvaluator.ts` lines 1248-1260 (the `buildRotationMatrix` and `rodriguesMatrix` helpers at lines 130-214).

**Test:** For each position transform, verify that evaluating `Constant(1)` through the transform produces the correct coordinate mapping at known points.

#### Group H: Shape SDFs (5 types)

| Type | Notes |
|------|-------|
| `Ellipsoid` | SDF with scale/rotation |
| `Cuboid` | Box SDF with rounded edges |
| `Cylinder` | Cylinder SDF |
| `Plane` | Half-space SDF |
| `Shell` | Hollow sphere (approximated in TS) |

**Test:** SDF values at points inside, outside, and on the surface.

#### Group I: Warps & Advanced (9 types)

| Type | Notes |
|------|-------|
| `DomainWarp2D` | Warp x,z using noise gradient |
| `DomainWarp3D` | Warp x,y,z using noise gradient |
| `GradientWarp` | Warp using analytic gradient |
| `FastGradientWarp` | Optimized gradient warp |
| `VectorWarp` | Warp using vector provider |
| `GradientDensity` | Y-axis linear gradient |
| `YGradient` | Simple Y gradient |
| `YOverride` | Replace Y coordinate |
| `XOverride` / `ZOverride` | Replace X/Z coordinate |

**Gradient warps** require the noise-with-gradient functions. Translate from `hytaleNoise.ts` `createHytaleNoise2DWithGradient` and `createHytaleNoise3DWithGradient` (lines 232-370).

**Test:** Warp functions with known noise seeds, verify output matches TS at specific coordinates.

#### Group J: Context & Special (6 types)

| Type | Notes |
|------|-------|
| `Anchor` | Set anchor point, evaluate child |
| `Exported` / `Imported` | Follow input edge (passthrough) |
| `CacheOnce` | Memoize per-sample |
| `Debug` | Passthrough with logging |
| `ImportedValue` | Read from contentFields map |
| `CellWallDistance` | Voronoi cell wall distance |

**Unsupported types** (return 0, matching TS behavior):
`HeightAboveSurface`, `SurfaceDensity`, `TerrainBoolean`, `TerrainMask`, `BeardDensity`, `ColumnDensity`, `CaveDensity`, `Terrain`, `DistanceToBiomeEdge`, `Pipeline`

These require server-side terrain context and cannot be evaluated offline.

### 1.5 Vector provider support

Some density nodes (VectorWarp, Angle) reference vector provider nodes. Translate `vectorEvaluator.ts` (~100 lines):

```rust
// src-tauri/src/eval/vectors.rs

pub fn evaluate_vector(
    ctx: &mut EvalContext,
    node_id: &str,
    x: f64, y: f64, z: f64,
) -> [f64; 3] {
    // Constant: read fields.Value.{x,y,z}
    // DensityGradient: finite differences of connected density
    // Cache/Exported/Imported: follow input edge
    // Default: [0, 1, 0]
}
```

### Deliverables — Phase 2

- [ ] All 68 density types implemented (or explicitly marked unsupported)
- [ ] All 19 curve types implemented
- [ ] Vector provider evaluation implemented
- [ ] ≥200 fixture test cases in `eval_cases.json`
- [ ] Parity check passes for all bundled templates (void, forest-hills, shattered-archipelago, tropical-pirate-islands, eldritch-spirelands)
- [ ] `cargo test` passes all Rust tests
- [ ] No regressions in existing TS tests

### How to test Phase 2

1. **Per-group Rust unit tests**: Each group gets its own `#[test]` module with fixture-driven tests.

2. **Template parity test** (key integration test):
   - Load each bundled template's biome JSON files
   - Parse them through `hytaleToInternal.ts` → `jsonToGraph.ts` to get React Flow nodes/edges
   - Evaluate at 100 sample points with both JS and Rust
   - All must match within `1e-4` (floating-point tolerance for different operation ordering)

3. **Manual visual test**: Open each bundled template in the app, toggle between JS and Rust evaluator, visually confirm identical heatmaps.

---

## Phase 3: Grid & Volume Evaluation Commands

**Goal:** Add Tauri commands for bulk evaluation (2D grids and 3D volumes) that the frontend can call instead of Web Workers. Add rayon parallelism. This is the performance inflection point.

**Duration:** 2-3 days

### 3.1 Add rayon dependency

```toml
# Cargo.toml
[dependencies]
rayon = "1.10"
```

### 3.2 Grid evaluation command

```rust
// src-tauri/src/eval/grid.rs

use rayon::prelude::*;

#[derive(Serialize)]
pub struct GridResult {
    /// Row-major f32 values (matches TS Float32Array layout)
    pub values: Vec<f32>,
    pub resolution: u32,
    pub min_value: f32,
    pub max_value: f32,
}

/// Evaluate a density graph over an NxN grid.
/// Each row is evaluated in parallel using rayon.
pub fn evaluate_grid(
    graph: &EvalGraph,
    resolution: u32,
    range_min: f64,
    range_max: f64,
    y_level: f64,
    content_fields: &HashMap<String, f64>,
) -> GridResult {
    let n = resolution as usize;
    let step = (range_max - range_min) / n as f64;

    // Evaluate rows in parallel
    let row_results: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|z_idx| {
            // Each thread gets its own EvalContext (noise caches are per-thread)
            let mut ctx = EvalContext::new(graph, content_fields.clone());
            let z = range_min + (z_idx as f64 + 0.5) * step;

            (0..n)
                .map(|x_idx| {
                    let x = range_min + (x_idx as f64 + 0.5) * step;
                    ctx.clear_memo();
                    evaluate(&mut ctx, &graph.root_id, x, y_level, z) as f32
                })
                .collect()
        })
        .collect();

    // Flatten + compute min/max
    let mut values = Vec::with_capacity(n * n);
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for row in row_results {
        for val in row {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            values.push(val);
        }
    }

    GridResult { values, resolution, min_value: min_val, max_value: max_val }
}
```

### 3.3 Volume evaluation command

```rust
// src-tauri/src/eval/volume.rs

#[derive(Serialize)]
pub struct VolumeResult {
    /// Y-major layout: densities[y * n * n + z * n + x]
    pub densities: Vec<f32>,
    pub resolution: u32,
    pub y_slices: u32,
    pub min_value: f32,
    pub max_value: f32,
}

/// Evaluate a density graph over a 3D volume.
/// Y-slices are evaluated in parallel.
pub fn evaluate_volume(
    graph: &EvalGraph,
    resolution: u32,
    range_min: f64,
    range_max: f64,
    y_min: f64,
    y_max: f64,
    y_slices: u32,
    content_fields: &HashMap<String, f64>,
) -> VolumeResult {
    let n = resolution as usize;
    let ys = y_slices as usize;
    let step_xz = (range_max - range_min) / n as f64;
    let step_y = if ys > 1 { (y_max - y_min) / (ys as f64 - 1.0) } else { 0.0 };

    // Evaluate Y-slices in parallel
    let slice_results: Vec<(Vec<f32>, f32, f32)> = (0..ys)
        .into_par_iter()
        .map(|yi| {
            let mut ctx = EvalContext::new(graph, content_fields.clone());
            let wy = y_min + yi as f64 * step_y;
            let mut slice = Vec::with_capacity(n * n);
            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;

            for zi in 0..n {
                let wz = range_min + zi as f64 * step_xz;
                for xi in 0..n {
                    let wx = range_min + xi as f64 * step_xz;
                    ctx.clear_memo();
                    let val = evaluate(&mut ctx, &graph.root_id, wx, wy, wz) as f32;
                    min_v = min_v.min(val);
                    max_v = max_v.max(val);
                    slice.push(val);
                }
            }
            (slice, min_v, max_v)
        })
        .collect();

    // Assemble into Y-major layout
    let total = n * n * ys;
    let mut densities = Vec::with_capacity(total);
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for (slice, s_min, s_max) in slice_results {
        min_val = min_val.min(s_min);
        max_val = max_val.max(s_max);
        densities.extend(slice);
    }

    VolumeResult { densities, resolution, y_slices, min_value: min_val, max_value: max_val }
}
```

### 3.4 Register new Tauri commands

```rust
// src-tauri/src/commands/preview.rs — add new commands

#[derive(Deserialize)]
pub struct GridRequest {
    pub nodes: Vec<serde_json::Value>,
    pub edges: Vec<serde_json::Value>,
    pub resolution: u32,
    pub range_min: f64,
    pub range_max: f64,
    pub y_level: f64,
    pub root_node_id: Option<String>,
    pub content_fields: Option<HashMap<String, f64>>,
}

#[tauri::command]
pub fn evaluate_grid(request: GridRequest) -> Result<GridResult, String> {
    let graph = parse_graph(
        request.nodes,
        request.edges,
        request.root_node_id.as_deref(),
    )?;
    Ok(eval::grid::evaluate_grid(
        &graph,
        request.resolution,
        request.range_min,
        request.range_max,
        request.y_level,
        &request.content_fields.unwrap_or_default(),
    ))
}

#[tauri::command]
pub fn evaluate_volume(request: VolumeRequest) -> Result<VolumeResult, String> {
    // Similar pattern
}
```

Add to `lib.rs` invoke handler:

```rust
preview::evaluate_grid,
preview::evaluate_volume,
preview::evaluate_points,
```

### 3.5 IPC wrapper on frontend

```typescript
// src/utils/ipc.ts — add

export interface GridRequest {
  nodes: unknown[];
  edges: unknown[];
  resolution: number;
  range_min: number;
  range_max: number;
  y_level: number;
  root_node_id?: string;
  content_fields?: Record<string, number>;
}

export interface GridResponse {
  values: number[];
  resolution: number;
  min_value: number;
  max_value: number;
}

export async function evaluateGrid(request: GridRequest): Promise<GridResponse> {
  return invoke<GridResponse>("evaluate_grid", { request });
}

export interface VolumeRequest {
  nodes: unknown[];
  edges: unknown[];
  resolution: number;
  range_min: number;
  range_max: number;
  y_min: number;
  y_max: number;
  y_slices: number;
  root_node_id?: string;
  content_fields?: Record<string, number>;
}

export interface VolumeResponse {
  densities: number[];
  resolution: number;
  y_slices: number;
  min_value: number;
  max_value: number;
}

export async function evaluateVolume(request: VolumeRequest): Promise<VolumeResponse> {
  return invoke<VolumeResponse>("evaluate_volume", { request });
}
```

### Deliverables — Phase 3

- [ ] `evaluate_grid` command works and returns correct values
- [ ] `evaluate_volume` command works and returns correct values
- [ ] rayon parallelism enabled — benchmark shows multi-core scaling
- [ ] IPC wrappers added to `ipc.ts`
- [ ] Benchmark: 128×128 grid in <10ms on 4-core machine

### How to test Phase 3

1. **Benchmark test** (Rust):
   ```rust
   #[bench]
   fn bench_grid_128(b: &mut Bencher) {
       let graph = load_template_graph("forest-hills");
       b.iter(|| evaluate_grid(&graph, 128, -64.0, 64.0, 64.0, &HashMap::new()));
   }
   ```

2. **Grid parity test**: Evaluate a 32×32 grid with both JS and Rust, compare all 1024 values within `1e-4`.

3. **Volume parity test**: Evaluate a 16×16×16 volume with both, compare all 4096 values.

4. **Performance logging**: Add `console.time` around both JS worker and Rust IPC calls in dev mode to measure real-world difference.

---

## Phase 4: Frontend Integration with Feature Flag

**Goal:** Wire the Rust evaluation commands into the existing preview pipeline behind a feature flag. Users can opt-in. Both paths coexist.

**Duration:** 2-3 days

### 4.1 Feature flag in settings

Add to `settingsStore.ts`:

```typescript
// Settings store addition
useRustEvaluator: getStoredBool("tn-use-rust-eval", false),
setUseRustEvaluator: (v: boolean) => {
  localStorage.setItem("tn-use-rust-eval", v ? "true" : "false");
  set({ useRustEvaluator: v });
},
```

Add a toggle in the Configuration dialog (accessible from the toolbar) in the "CPU" section:

```
☐ Use Rust evaluator (experimental)
   Native evaluation via Tauri backend. Faster but may have
   minor differences from the JS evaluator.
```

### 4.2 Modify `usePreviewEvaluation.ts`

The key change: when the Rust evaluator is enabled, call `evaluateGrid` from `ipc.ts` instead of `evaluateInWorker` from `densityWorkerClient.ts`.

```typescript
// src/hooks/usePreviewEvaluation.ts — modified

import { evaluateGrid as evaluateGridRust } from "@/utils/ipc";

// Inside the useEffect:
if (useSettingsStore.getState().useRustEvaluator) {
  // Rust path — single IPC call, no worker management
  const result = await evaluateGridRust({
    nodes: nodes.map(n => ({
      id: n.id,
      type: n.type,
      data: n.data,
    })),
    edges: edges.map(e => ({
      source: e.source,
      target: e.target,
      targetHandle: e.targetHandle,
    })),
    resolution,
    range_min: rangeMin,
    range_max: rangeMax,
    y_level: yLevel,
    root_node_id: selectedPreviewNodeId ?? outputNodeId ?? undefined,
    content_fields: contentFields,
  });

  if (evalId === evalIdRef.current) {
    const values = new Float32Array(result.values);
    setValues(values, result.min_value, result.max_value);
  }
} else {
  // Existing JS Worker path — unchanged
  const result = await evaluateInWorker({ ... });
  // ...
}
```

### 4.3 Modify `useVoxelEvaluation.ts`

Same pattern: when flag is enabled, call `evaluateVolume` from `ipc.ts` instead of `evaluateVolumeInWorker`. The post-processing pipeline (material evaluation, surface extraction, mesh building) stays in JS for now — only the density evaluation moves to Rust in this phase.

```typescript
// Replace the evaluateVolumeInWorker call:
if (useSettingsStore.getState().useRustEvaluator) {
  const volumeResult = await evaluateVolumeRust({
    nodes: ...,
    edges: ...,
    resolution: res,
    range_min: rangeMin,
    range_max: rangeMax,
    y_min: voxelYMin,
    y_max: voxelYMax,
    y_slices: ySlices,
    root_node_id: ...,
    content_fields: contentFields,
  });
  result = {
    densities: new Float32Array(volumeResult.densities),
    resolution: volumeResult.resolution,
    ySlices: volumeResult.y_slices,
    minValue: volumeResult.min_value,
    maxValue: volumeResult.max_value,
  };
} else {
  result = await evaluateVolumeInWorker({ ... });
}

// Everything below (material eval, surface extract, mesh build) is unchanged
```

### 4.4 Comparison view support

`useComparisonEvaluation.ts` runs two evaluations in parallel. Both should respect the feature flag independently.

### Deliverables — Phase 4

- [ ] Feature flag toggle in Configuration dialog
- [ ] 2D preview works with Rust evaluator enabled
- [ ] 3D voxel preview works with Rust evaluator enabled
- [ ] Comparison view works with Rust evaluator
- [ ] Toggling the flag switches evaluator without restart
- [ ] Performance improvement is measurable (dev tools console.time)
- [ ] No regressions when flag is OFF (default)

### How to test Phase 4

1. **Manual workflow test**:
   - Open app, load forest-hills template
   - Open preview panel, verify 2D heatmap renders
   - Open Configuration → enable "Use Rust evaluator"
   - Verify heatmap looks identical
   - Switch to 3D voxel mode, verify it renders
   - Modify a node (change noise frequency), verify preview updates
   - Toggle flag OFF, verify preview still works

2. **All bundled templates**: Load each template, toggle between evaluators, verify visual match.

3. **Performance comparison**: Log timings for both paths. Expected: 5-10× speedup for 128×128.

4. **Edge cases**: Empty graph, single constant node, very deep graph (50+ nodes), cyclic reference.

---

## Phase 5: Material Evaluation in Rust

**Goal:** Move the material graph evaluator (`materialEvaluator.ts`, ~700 lines) to Rust. After this phase, the entire density+material pipeline runs natively.

**Duration:** 3-5 days

### Why this matters

Material evaluation runs AFTER density evaluation — it iterates every solid voxel in the 3D volume and determines which material (stone, dirt, grass, etc.) to assign based on a graph of material provider nodes. At 64³ resolution this means up to ~260,000 voxels evaluated through a separate graph traversal. Moving this to Rust provides the same ~10× speedup.

### 5.1 Material evaluator module

```rust
// src-tauri/src/eval/material.rs

/// Result of material graph evaluation
#[derive(Serialize)]
pub struct MaterialResult {
    /// Per-voxel material index (indexes into palette)
    pub material_ids: Vec<u8>,
    /// Material palette
    pub palette: Vec<MaterialEntry>,
}

#[derive(Serialize, Clone)]
pub struct MaterialEntry {
    pub name: String,
    pub color: String,
    pub roughness: f32,
    pub metalness: f32,
    pub emissive: String,
    pub emissive_intensity: f32,
}

/// Evaluate the material graph for a 3D volume.
///
/// Material nodes are identified by React Flow type prefix "Material:"
/// and data.type values from the MATERIAL_TYPES set.
pub fn evaluate_material_graph(
    nodes: &[GraphNode],
    edges: &[GraphEdge],
    densities: &[f32],
    resolution: u32,
    y_slices: u32,
    range_min: f64,
    range_max: f64,
    y_min: f64,
    y_max: f64,
    density_ctx: Option<&EvalGraph>,
) -> Option<MaterialResult> {
    // 1. Find material root node (same strategy as TS findMaterialRoot)
    // 2. Build column contexts (surface Y, downward depth, etc.)
    // 3. For each solid voxel, evaluate material graph → material name
    // 4. Map material names to palette indices
    // ...
}
```

### 5.2 Material types to implement

From `materialEvaluator.ts` (lines 183-195):

| Type | Notes |
|------|-------|
| `Constant` | Fixed material name |
| `FieldFunction` | Material based on density value ranges |
| `SpaceAndDepth` | Material based on depth from surface |
| `Conditional` | Condition-based material selection |
| `DensityBased` | Material from density threshold |
| `Striped` | Alternating materials by depth |
| `WeightedRandom` | Random material selection with weights |
| `HeightGradient` | Material by Y coordinate |
| `SurfaceDepth` | Material by distance from surface |
| `ConditionalChain` | Multi-condition material chain |
| `LayerStack` | Ordered material layers |
| `Queue` | Priority queue of material providers |
| `Cache` | Memoized material |
| `Exported` / `Imported` | Passthrough |

### 5.3 Integrate into voxel pipeline

Add a combined `evaluate_voxel_preview` command that runs the full pipeline:

```rust
#[tauri::command]
pub fn evaluate_voxel_preview(request: VoxelRequest) -> Result<VoxelPreviewResult, String> {
    // 1. Evaluate density volume (reuse eval/volume.rs)
    let volume = evaluate_volume(...);

    // 2. Evaluate materials (if material nodes present)
    let materials = evaluate_material_graph(...);

    // 3. Return both
    Ok(VoxelPreviewResult {
        densities: volume.densities,
        resolution: volume.resolution,
        y_slices: volume.y_slices,
        min_value: volume.min_value,
        max_value: volume.max_value,
        material_ids: materials.map(|m| m.material_ids),
        palette: materials.map(|m| m.palette),
    })
}
```

### Deliverables — Phase 5

- [ ] Material graph evaluator implemented in Rust
- [ ] All 14 material provider types supported
- [ ] `evaluate_voxel_preview` command combines density + material evaluation
- [ ] Frontend calls combined command when Rust evaluator is enabled
- [ ] Material colors in voxel preview match between JS and Rust
- [ ] Parity tests pass for all templates with material providers

### How to test Phase 5

1. **Material parity tests**: For each material type, create a fixture with known voxel positions and verify assigned materials match.

2. **Visual test**: Load eldritch-spirelands template (has complex materials), compare voxel preview colors between JS and Rust.

3. **Benchmark**: Measure combined density+material evaluation time.

---

## Phase 6: Voxel Extraction & Mesh Building

**Goal:** Move surface voxel extraction (`voxelExtractor.ts`) and mesh building (`voxelMeshBuilder.ts`) to Rust. After this phase, the frontend receives ready-to-render mesh data from a single IPC call.

**Duration:** 3-4 days

### Why this matters

Currently, after density evaluation returns, the frontend runs:
1. `extractSurfaceVoxels()` — iterate all voxels, find surface ones (~300 lines)
2. `buildVoxelMeshes()` — greedy meshing with AO computation (~400 lines)

These are CPU-intensive loops over large arrays. Moving them to Rust means the frontend receives final vertex/index buffers that can be uploaded directly to the GPU.

### 6.1 Surface extraction in Rust

```rust
// src-tauri/src/eval/voxel.rs

pub const SOLID_THRESHOLD: f32 = 0.0;

/// Check if a voxel is on the surface (has at least one air neighbor)
fn is_surface(densities: &[f32], x: usize, y: usize, z: usize, n: usize, ys: usize) -> bool {
    // 6-neighbor check — same logic as TS isSurface()
}

/// Extract surface voxels from a density volume
pub fn extract_surface_voxels(
    densities: &[f32],
    resolution: u32,
    y_slices: u32,
    material_ids: Option<&[u8]>,
    palette: &[MaterialEntry],
    fluid_config: Option<&FluidConfig>,
) -> SurfaceVoxels { ... }
```

### 6.2 Mesh building in Rust

```rust
// src-tauri/src/eval/mesh.rs

#[derive(Serialize)]
pub struct MeshData {
    pub material_index: u32,
    pub color: String,
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub colors: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Build per-material meshes with AO and face shading.
/// Same algorithm as voxelMeshBuilder.ts but parallelized per-material.
pub fn build_voxel_meshes(
    voxels: &SurfaceVoxels,
    densities: &[f32],
    resolution: u32,
    y_slices: u32,
    scale: [f32; 3],
    offset: [f32; 3],
) -> Vec<MeshData> { ... }
```

### 6.3 Full pipeline command

```rust
#[tauri::command]
pub fn evaluate_full_voxel_preview(request: FullVoxelRequest) -> Result<FullVoxelResult, String> {
    // 1. Density volume
    // 2. Material evaluation
    // 3. Surface extraction
    // 4. Mesh building
    // Returns ready-to-render mesh data
}
```

The frontend can then upload the returned `positions`, `normals`, `colors`, `indices` directly to Three.js `BufferGeometry` attributes without any further processing.

### 6.4 Frontend mesh consumption

```typescript
// In VoxelPreview3D.tsx — when Rust evaluator is enabled:
const result = await invoke<FullVoxelResult>('evaluate_full_voxel_preview', { ... });

for (const mesh of result.meshes) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(mesh.positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(mesh.normals, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(mesh.colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(mesh.indices), 1));
  // ... add to scene
}
```

### Deliverables — Phase 6

- [ ] Surface extraction in Rust matches TS output
- [ ] Mesh building in Rust produces identical geometry
- [ ] Full pipeline command: one IPC call → ready-to-render meshes
- [ ] Frontend uploads mesh data directly to Three.js
- [ ] Fluid plane support (lava/water)
- [ ] Significant performance improvement (full pipeline in Rust)

### How to test Phase 6

1. **Surface voxel count test**: For a known density volume, verify exact same number of surface voxels.

2. **Mesh vertex test**: For a small volume (8³), verify vertex positions and normals match between JS and Rust.

3. **Visual test**: Render both JS-built and Rust-built meshes, screenshot compare.

4. **Performance benchmark**: Full pipeline time (density → mesh) for 64³ volume.

---

## Phase 7: Progressive Streaming & Caching

**Goal:** Re-implement progressive resolution rendering using Tauri event streaming, and add LRU caching to avoid re-evaluating unchanged graphs.

**Duration:** 2-3 days

### 7.1 Progressive streaming via Tauri events

Replace the current progressive setTimeout cascade with Rust-side streaming:

```rust
#[tauri::command]
pub async fn evaluate_grid_progressive(
    app: tauri::AppHandle,
    request: GridRequest,
) -> Result<(), String> {
    let graph = parse_graph(request.nodes, request.edges, ...)?;

    // Progressive steps: 16 → 32 → 64 → target
    let steps: Vec<u32> = vec![16, 32, 64, request.resolution]
        .into_iter()
        .filter(|&r| r <= request.resolution)
        .collect();

    for resolution in steps {
        let result = eval::grid::evaluate_grid(&graph, resolution, ...);
        app.emit("preview_progressive", &result)
           .map_err(|e| e.to_string())?;
    }

    Ok(())
}
```

Frontend:

```typescript
// Listen for progressive updates
const unlisten = await listen<GridResponse>('preview_progressive', (event) => {
  const result = event.payload;
  setValues(new Float32Array(result.values), result.min_value, result.max_value);
});

// Trigger progressive evaluation
await invoke('evaluate_grid_progressive', { ... });
unlisten();
```

### 7.2 Evaluation cache

```rust
// src-tauri/src/eval/cache.rs

use std::collections::HashMap;
use std::sync::Mutex;

/// Content-addressable cache for evaluation results.
/// Key = hash(graph_structure + params), Value = result.
pub struct EvalCache {
    grid_cache: Mutex<lru::LruCache<u64, GridResult>>,
    volume_cache: Mutex<lru::LruCache<u64, VolumeResult>>,
}

impl EvalCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            grid_cache: Mutex::new(lru::LruCache::new(capacity.try_into().unwrap())),
            volume_cache: Mutex::new(lru::LruCache::new(capacity.try_into().unwrap())),
        }
    }

    pub fn get_grid(&self, hash: u64) -> Option<GridResult> {
        self.grid_cache.lock().unwrap().get(&hash).cloned()
    }

    pub fn put_grid(&self, hash: u64, result: GridResult) {
        self.grid_cache.lock().unwrap().put(hash, result);
    }
}
```

Add `lru` crate:

```toml
[dependencies]
lru = "0.12"
```

Register as Tauri managed state:

```rust
.manage(EvalCache::new(32))
```

### 7.3 Graph hashing

```rust
/// Hash the graph structure + evaluation parameters.
/// Must be deterministic — same graph + params = same hash.
pub fn hash_eval_request(
    nodes: &[GraphNode],
    edges: &[GraphEdge],
    resolution: u32,
    range_min: f64,
    range_max: f64,
    y_level: f64,
) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    // Hash node count, each node's id + type + fields
    // Hash edge count, each edge's source + target + handle
    // Hash resolution, range, y_level
    // ...
    hasher.finish()
}
```

### Deliverables — Phase 7

- [ ] Progressive streaming works via Tauri events
- [ ] LRU cache skips re-evaluation when graph hasn't changed
- [ ] Cache invalidation works when nodes/edges change
- [ ] Cache state managed as Tauri state (thread-safe)

### How to test Phase 7

1. **Progressive test**: Verify that the frontend receives multiple results of increasing resolution.

2. **Cache hit test**: Evaluate the same graph twice, verify the second call returns instantly (no re-evaluation). Log cache hit/miss in dev mode.

3. **Cache invalidation test**: Change a node field, verify cache misses and re-evaluates.

---

## Phase 8: Cleanup & Deprecation

**Goal:** Make the Rust evaluator the default, remove the feature flag, and deprecate the TypeScript evaluation code.

**Duration:** 2-3 days

**Prerequisites:** Phases 1-7 are complete, the Rust evaluator has been used in production by real users for at least 2 weeks with no reported parity issues.

### 8.1 Make Rust the default

```typescript
// settingsStore.ts
useRustEvaluator: getStoredBool("tn-use-rust-eval", true),  // default: true
```

### 8.2 Keep JS fallback for unsupported types

The Rust evaluator will return a status indicating which node types were evaluated vs. returned as 0 (unsupported context-dependent types). The frontend can fall back to JS for graphs containing unsupported types.

```rust
#[derive(Serialize)]
pub struct GridResult {
    pub values: Vec<f32>,
    pub resolution: u32,
    pub min_value: f32,
    pub max_value: f32,
    /// If true, all nodes were fully evaluated
    pub fully_evaluated: bool,
    /// List of unsupported node types encountered
    pub unsupported_types: Vec<String>,
}
```

### 8.3 Remove deprecated code (do NOT do this hastily)

Only after the Rust evaluator has been the default for a full release cycle:

1. Remove `src/workers/densityWorker.ts`
2. Remove `src/workers/volumeWorker.ts`
3. Remove `src/utils/densityWorkerClient.ts`
4. Remove `src/utils/volumeWorkerClient.ts`
5. Remove `src/utils/densityEvaluator.ts` (keep as reference in `docs/` if desired)
6. Remove `src/utils/volumeEvaluator.ts`
7. Remove `simplex-noise` from `package.json` dependencies
8. Remove old `src-tauri/src/noise/` module (replaced by `eval/`)
9. Update ADR-3 in `CONTEXT.md`

**Lines removed:** ~3,500 TypeScript, ~250 Rust (old noise module)
**Lines added:** ~2,500 Rust (new eval module — more compact due to Rust's expressiveness)
**Net:** ~1,250 fewer lines of evaluation code

### 8.4 Update documentation

- Update `CONTEXT.md` Architecture section and ADR-3
- Update `README.md` performance section with new benchmarks
- Add a note to `CHANGELOG.md`

### Deliverables — Phase 8

- [ ] Rust evaluator is the default
- [ ] JS fallback works for unsupported types
- [ ] Old evaluation code removed (after bake-in period)
- [ ] Documentation updated
- [ ] Bundle size reduced

---

## Appendix A: Node Type Inventory

Complete list of all 68 density types, their implementation phase, and the TypeScript source line reference in `densityEvaluator.ts`.

| # | Type | Phase | TS Lines | Category | Inputs |
|---|------|-------|----------|----------|--------|
| 1 | `Constant` | 1 | 649 | Const | — |
| 2 | `Zero` | 1 | 651 | Const | — |
| 3 | `One` | 1 | 653 | Const | — |
| 4 | `ImportedValue` | 2-J | 1473 | Const | contentFields |
| 5 | `SimplexNoise2D` | 1 | 512-520 | Noise | — |
| 6 | `SimplexNoise3D` | 1 | 524-532 | Noise | — |
| 7 | `SimplexRidgeNoise2D` | 2-D | 536-542 | Noise | — |
| 8 | `SimplexRidgeNoise3D` | 2-D | 546-552 | Noise | — |
| 9 | `VoronoiNoise2D` | 2-D | 556-576 | Noise | ReturnCurve, ReturnDensity |
| 10 | `VoronoiNoise3D` | 2-D | 578-598 | Noise | ReturnCurve, ReturnDensity |
| 11 | `FractalNoise2D` | 2-D | 600-607 | Noise | — |
| 12 | `FractalNoise3D` | 2-D | 611-618 | Noise | — |
| 13 | `Sum` | 1 | 700-704 | Arith | Input[N] |
| 14 | `Product` | 1 | 719-727 | Arith | Input[N] |
| 15 | `Negate` | 1 | 655 | Arith | Input |
| 16 | `Abs` | 1 | 657 | Arith | Input |
| 17 | `SquareRoot` | 2-A | 661 | Arith | Input |
| 18 | `CubeRoot` | 2-A | 663 | Arith | Input |
| 19 | `Square` | 2-A | 655 | Arith | Input |
| 20 | `CubeMath` | 2-A | 661 | Arith | Input |
| 21 | `Inverse` | 2-A | 667 | Arith | Input |
| 22 | `SumSelf` | 2-A | 673 | Arith | Input |
| 23 | `Modulo` | 2-A | 679 | Arith | Input |
| 24 | `AmplitudeConstant` | 2-A | 686 | Arith | Input |
| 25 | `Pow` | 2-B | 693 | Arith | Input |
| 26 | `WeightedSum` | 2-B | 711 | Arith | Input[N] + Weight[N] |
| 27 | `Interpolate` | 2-B | 805 | Arith | InputA, InputB, Factor |
| 28 | `MinFunction` | 2-B | — | Arith | Input[N] |
| 29 | `MaxFunction` | 2-B | — | Arith | Input[N] |
| 30 | `AverageFunction` | 2-B | 1398 | Arith | Input[N] |
| 31 | `SmoothMin` | 2-B | 935 | Arith | InputA, InputB |
| 32 | `SmoothMax` | 2-B | 944 | Arith | InputA, InputB |
| 33 | `Clamp` | 1 | 730-735 | Range | Input |
| 34 | `ClampToIndex` | 2-C | 740 | Range | Input |
| 35 | `Normalizer` | 1 | 748-757 | Range | Input |
| 36 | `DoubleNormalizer` | 2-C | 762-784 | Range | Input |
| 37 | `RangeChoice` | 2-C | 796 | Range | Input |
| 38 | `LinearTransform` | 1 | 788 | Range | Input |
| 39 | `SmoothClamp` | 2-C | 952 | Range | Input |
| 40 | `SmoothFloor` | 2-C | 944 | Range | Input |
| 41 | `SmoothCeiling` | 2-C | — | Range | Input |
| 42 | `Wrap` | 2-C | — | Range | Input |
| 43 | `Passthrough` | 2-C | — | Range | Input |
| 44 | `CurveFunction` | 2-E | 427-463 | Curve | ReturnCurve |
| 45 | `SplineFunction` | 2-E | 466-489 | Curve | — |
| 46 | `FlatCache` | 2-E | 1562 | Cache | Input |
| 47 | `CacheOnce` | 2-J | — | Cache | Input |
| 48 | `Conditional` | 2-F | 1373 | Logic | Condition, InputA, InputB |
| 49 | `Switch` | 2-F | 1419 | Logic | State[N] |
| 50 | `Blend` | 2-F | 1409 | Logic | InputA, InputB, Factor |
| 51 | `BlendCurve` | 2-F | 1441 | Logic | InputA, InputB, Factor |
| 52 | `MultiMix` | 2-F | 1450 | Logic | Input[N], Selector |
| 53 | `SwitchState` | 2-F | 1108 | Logic | Input |
| 54 | `CoordinateX` | 1 | 831 | Pos | — |
| 55 | `CoordinateY` | 1 | 831 | Pos | — |
| 56 | `CoordinateZ` | 1 | 831 | Pos | — |
| 57 | `DistanceFromOrigin` | 2-G | 839 | Pos | — |
| 58 | `DistanceFromAxis` | 2-G | 852 | Pos | — |
| 59 | `DistanceFromPoint` | 2-G | 860 | Pos | VectorProvider |
| 60 | `AngleFromOrigin` | 2-G | 1359 | Pos | — |
| 61 | `AngleFromPoint` | 2-G | 1365 | Pos | VectorProvider |
| 62 | `TranslatedPosition` | 2-G | 1487 | Transform | Input |
| 63 | `ScaledPosition` | 2-G | 1496 | Transform | Input |
| 64 | `RotatedPosition` | 2-G | 1505 | Transform | Input |
| 65 | `MirroredPosition` | 2-G | 1517 | Transform | Input |
| 66 | `QuantizedPosition` | 2-G | 1526 | Transform | Input |
| 67 | `GradientDensity` | 2-I | 897 | Gradient | — |
| 68 | `YGradient` | 2-I | 905 | Gradient | — |

**Unsupported (return 0):** HeightAboveSurface, SurfaceDensity, TerrainBoolean, TerrainMask, BeardDensity, ColumnDensity, CaveDensity, Terrain, DistanceToBiomeEdge, Pipeline

**Approximated:** PositionsCellNoise, Positions3D, PositionsPinch, PositionsTwist, VectorWarp, Shell

---

## Appendix B: Data Format Contracts

### React Flow Node (IPC format)

Only the fields the Rust evaluator needs:

```json
{
  "id": "node_abc123",
  "type": "Density:SimplexNoise2D",
  "data": {
    "type": "SimplexNoise2D",
    "fields": {
      "Frequency": 0.01,
      "Amplitude": 1.0,
      "Seed": "main",
      "Octaves": 4,
      "Lacunarity": 2.0,
      "Gain": 0.5
    },
    "_outputNode": false,
    "_biomeField": null
  }
}
```

### React Flow Edge (IPC format)

```json
{
  "source": "noise_1",
  "target": "sum_1",
  "targetHandle": "Input[0]"
}
```

### Grid Response

```json
{
  "values": [0.0, 0.1, 0.2, ...],
  "resolution": 128,
  "min_value": -0.95,
  "max_value": 0.87
}
```

`values` is a row-major flat array of `f32` (length = resolution²). The frontend wraps it in `new Float32Array(response.values)`.

### Volume Response

```json
{
  "densities": [0.0, 0.1, ...],
  "resolution": 64,
  "y_slices": 64,
  "min_value": -1.2,
  "max_value": 0.9
}
```

`densities` is Y-major: `densities[y * n * n + z * n + x]` (length = resolution² × y_slices).

---

## Appendix C: Noise Parity Reference

### Critical parity requirements

The TypeScript noise implementation in `hytaleNoise.ts` was specifically written to match Hytale's V2 Java runtime. The Rust implementation must produce **bit-identical results** (within f64 precision) for the same inputs.

### Key algorithms to match exactly

1. **`javaStringHashCode(s)`** — Java's `String.hashCode()` algorithm
   - `hash = hash * 31 + charCode` with 32-bit wrapping
   - TS: `hytaleNoise.ts` lines 49-54
   - Test: `javaStringHashCode("main")` must return the same integer in both

2. **`mulberry32(seed)`** — PRNG sequence
   - State update: `s = (s + 0x6d2b79f5) | 0`
   - TS: `hytaleNoise.ts` lines 63-70
   - Test: For seed 42, first 10 outputs must match exactly

3. **`buildPermutationTable(seed)`** — Fisher-Yates shuffle
   - 256-entry identity shuffled with mulberry32, doubled to 512
   - TS: `hytaleNoise.ts` lines 80-99
   - Test: For seed 0, `perm[0..10]` must match exactly

4. **Simplex noise constants**
   - 2D: `F2 = (√3 - 1) / 2`, `G2 = (3 - √3) / 6`, scale = 70.0
   - 3D: `F3 = 1/3`, `G3 = 1/6`, scale = 32.0
   - 2D gradients: `[[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]]`
   - 3D gradients: 12 edge directions of a cube

5. **Gradient-with-derivative** (for FastGradientWarp)
   - TS: `hytaleNoise.ts` lines 232-370
   - Returns `{ value, dx, dy }` (2D) or `{ value, dx, dy, dz }` (3D)
   - Uses analytic derivative of the simplex kernel

### Parity test procedure

For each noise function:
1. Generate 100 deterministic sample points: `[(i * 7.3 + 1.1, i * 11.7 - 3.2)]`
2. Evaluate in TypeScript, save expected values to fixture file
3. Evaluate in Rust, compare within `1e-10` tolerance
4. Test with seeds: `0`, `42`, `"main"`, `"A"`, `"terrain_base"`

### fastnoise-lite note

The existing Rust code uses `fastnoise-lite` which implements **OpenSimplex2**, not standard simplex noise. OpenSimplex2 uses different gradient tables and skew factors, producing different output for the same inputs. This is why the existing `evaluate_density` command cannot be used for preview — it doesn't match Hytale's noise.

The new `eval/noise.rs` must use the **exact same algorithm** as `hytaleNoise.ts`, which is standard simplex noise matching Hytale's Java implementation. The `fastnoise-lite` dependency can be removed in Phase 8 since nothing else uses it.
