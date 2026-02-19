// eval/nodes.rs — Density node evaluation (all 68+ types)
//
// Implements the core evaluation loop and ALL density node types,
// matching the TypeScript `evaluate()` function in `densityEvaluator.ts`.

use crate::eval::curves;
use crate::eval::graph::EvalGraph;
use crate::eval::noise;
use crate::eval::vectors;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Result of evaluating a node. Most return f64.
pub type EvalResult = f64;

/// Default world height matching TS `DEFAULT_WORLD_HEIGHT`.
const DEFAULT_WORLD_HEIGHT: f64 = 320.0;

// ── Smooth math helpers ─────────────────────────────────────────────

/// Polynomial smooth minimum. Matches `smoothMin` in `densityEvaluator.ts`.
fn smooth_min(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).max(0.0).min(1.0);
    b + (a - b) * h - k * h * (1.0 - h)
}

/// Polynomial smooth maximum. Matches `smoothMax` in `densityEvaluator.ts`.
fn smooth_max(a: f64, b: f64, k: f64) -> f64 {
    -smooth_min(-a, -b, k)
}

// ── Rotation matrix helpers ─────────────────────────────────────────

type Mat3 = [f64; 9];

const IDENTITY_MAT3: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// Rodrigues' rotation formula as a 3x3 matrix.
/// Given unit axis (kx,ky,kz) and angle with known cos and sin.
fn rodrigues_matrix(kx: f64, ky: f64, kz: f64, cos_t: f64, sin_t: f64) -> Mat3 {
    let omc = 1.0 - cos_t;
    [
        cos_t + kx * kx * omc,
        kx * ky * omc - kz * sin_t,
        kx * kz * omc + ky * sin_t,
        ky * kx * omc + kz * sin_t,
        cos_t + ky * ky * omc,
        ky * kz * omc - kx * sin_t,
        kz * kx * omc - ky * sin_t,
        kz * ky * omc + kx * sin_t,
        cos_t + kz * kz * omc,
    ]
}

fn mat3_multiply(a: &Mat3, b: &Mat3) -> Mat3 {
    [
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    ]
}

fn mat3_apply(m: &Mat3, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    (
        m[0] * x + m[1] * y + m[2] * z,
        m[3] * x + m[4] * y + m[5] * z,
        m[6] * x + m[7] * y + m[8] * z,
    )
}

/// Build a rotation matrix that maps Y axis [0,1,0] to `new_y_axis`
/// and then applies a spin rotation around the new Y axis.
/// Matches `buildRotationMatrix` in `densityEvaluator.ts`.
fn build_rotation_matrix(new_y_axis: Option<&Value>, spin_angle: f64) -> Mat3 {
    let (mut nx, mut ny, mut nz) = match new_y_axis {
        Some(Value::Object(obj)) => {
            let x = obj.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let y = obj.get("y").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let z = obj.get("z").and_then(|v| v.as_f64()).unwrap_or(0.0);
            (x, y, z)
        }
        _ => return IDENTITY_MAT3,
    };

    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-10 {
        return IDENTITY_MAT3;
    }
    nx /= len;
    ny /= len;
    nz /= len;

    // Step 1: Rotate standard Y [0,1,0] to newYAxis
    // Rotation axis = cross(up, newY) = (-nz, 0, nx)
    let ax = -nz;
    let ay = 0.0;
    let az = nx;
    let axis_len = (ax * ax + ay * ay + az * az).sqrt();

    let r1: Mat3 = if axis_len < 1e-10 {
        if ny > 0.0 {
            IDENTITY_MAT3
        } else {
            // 180-degree rotation around X axis
            [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]
        }
    } else {
        let kx = ax / axis_len;
        let ky = ay / axis_len;
        let kz = az / axis_len;
        let cos_t = ny; // dot(up, newY)
        let sin_t = axis_len;
        rodrigues_matrix(kx, ky, kz, cos_t, sin_t)
    };

    // Step 2: Apply spin rotation around newYAxis
    if spin_angle.abs() < 1e-10 {
        return r1;
    }

    let cos_s = spin_angle.cos();
    let sin_s = spin_angle.sin();
    let r2 = rodrigues_matrix(nx, ny, nz, cos_s, sin_s);

    mat3_multiply(&r2, &r1)
}

// ── Context ─────────────────────────────────────────────────────────

/// Context passed to every evaluation call.
///
/// Holds the parsed graph, per-sample memoization cache, cycle detection,
/// noise caches (persist across samples), content fields, and mutable
/// evaluation state (anchor, switch state, cell wall distance).
pub struct EvalContext<'a> {
    pub graph: &'a EvalGraph,
    /// Per-sample memoization cache (cleared between samples)
    pub memo: HashMap<String, f64>,
    /// Cycle detection — tracks nodes currently being evaluated
    pub visiting: HashSet<String>,
    /// Noise permutation table cache — keyed by seed, persists across samples
    pub perm_cache: HashMap<i32, [u8; 512]>,
    /// Content fields (from WorldStructure, e.g. {"Base": 100, "Water": 100})
    pub content_fields: HashMap<String, f64>,
    /// Anchor position — set by Anchor nodes
    pub anchor: [f64; 3],
    /// Whether an anchor has been set
    pub anchor_set: bool,
    /// Switch state — set by SwitchState nodes, read by Switch nodes
    pub switch_state: i32,
    /// Cell wall distance — set by PositionsCellNoise, read by CellWallDistance
    pub cell_wall_dist: f64,
}

impl<'a> EvalContext<'a> {
    pub fn new(graph: &'a EvalGraph, content_fields: HashMap<String, f64>) -> Self {
        EvalContext {
            graph,
            memo: HashMap::new(),
            visiting: HashSet::new(),
            perm_cache: HashMap::new(),
            content_fields,
            anchor: [0.0, 0.0, 0.0],
            anchor_set: false,
            switch_state: 0,
            cell_wall_dist: f64::INFINITY,
        }
    }

    /// Clear the per-sample memo cache. Call between evaluation points.
    pub fn clear_memo(&mut self) {
        self.memo.clear();
        self.visiting.clear();
    }

    /// Get or build a permutation table for the given seed.
    pub fn get_perm_table(&mut self, seed: i32) -> [u8; 512] {
        if let Some(perm) = self.perm_cache.get(&seed) {
            return *perm;
        }
        let perm = noise::build_perm_table(seed);
        self.perm_cache.insert(seed, perm);
        perm
    }
}

// ── Helper: read a field as f64 ─────────────────────────────────────

fn field_f64(fields: &HashMap<String, Value>, key: &str, default: f64) -> f64 {
    fields.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

fn field_i32(fields: &HashMap<String, Value>, key: &str, default: i32) -> i32 {
    fields
        .get(key)
        .and_then(|v| v.as_i64())
        .map(|n| n as i32)
        .unwrap_or(default)
}

fn field_str<'a>(fields: &'a HashMap<String, Value>, key: &str, default: &'a str) -> &'a str {
    fields.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

fn field_bool(fields: &HashMap<String, Value>, key: &str, default: bool) -> bool {
    fields.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

fn field_vec3(
    fields: &HashMap<String, Value>,
    key: &str,
    dx: f64,
    dy: f64,
    dz: f64,
) -> (f64, f64, f64) {
    match fields.get(key) {
        Some(Value::Object(obj)) => {
            let x = obj.get("x").and_then(|v| v.as_f64()).unwrap_or(dx);
            let y = obj.get("y").and_then(|v| v.as_f64()).unwrap_or(dy);
            let z = obj.get("z").and_then(|v| v.as_f64()).unwrap_or(dz);
            (x, y, z)
        }
        _ => (dx, dy, dz),
    }
}

// ── Helper: get a single input value via edge handle ────────────────

fn get_input(ctx: &mut EvalContext, node_id: &str, handle: &str, x: f64, y: f64, z: f64) -> f64 {
    let source_id = ctx
        .graph
        .inputs
        .get(node_id)
        .and_then(|m| m.get(handle))
        .cloned();
    match source_id {
        Some(src) => evaluate(ctx, &src, x, y, z),
        None => 0.0,
    }
}

fn has_input(ctx: &EvalContext, node_id: &str, handle: &str) -> bool {
    ctx.graph
        .inputs
        .get(node_id)
        .map(|m| m.contains_key(handle))
        .unwrap_or(false)
}

fn get_input_source(ctx: &EvalContext, node_id: &str, handle: &str) -> Option<String> {
    ctx.graph
        .inputs
        .get(node_id)
        .and_then(|m| m.get(handle))
        .cloned()
}

// ── Helper: collect array inputs ────────────────────────────────────

/// Collect array inputs matching "HandleName[0]", "HandleName[1]", etc.
/// Also accepts the bare handle name as index 0.
/// Returns values sorted by index.
fn get_array_inputs(
    ctx: &mut EvalContext,
    node_id: &str,
    base_handle: &str,
    x: f64,
    y: f64,
    z: f64,
) -> Vec<f64> {
    let inputs = match ctx.graph.inputs.get(node_id) {
        Some(m) => m.clone(),
        None => return vec![],
    };

    let mut indexed: Vec<(usize, String)> = vec![];
    for (handle, source_id) in &inputs {
        if let Some(rest) = handle.strip_prefix(base_handle) {
            if let Some(idx_str) = rest.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    indexed.push((idx, source_id.clone()));
                }
            }
        }
        // Also accept bare handle name as index 0
        if handle == base_handle {
            indexed.push((0, source_id.clone()));
        }
    }

    indexed.sort_by_key(|(i, _)| *i);

    indexed
        .into_iter()
        .map(|(_, src)| evaluate(ctx, &src, x, y, z))
        .collect()
}

/// Product of array inputs (no short-circuit — matches TS behavior).
fn get_array_inputs_product(
    ctx: &mut EvalContext,
    node_id: &str,
    base_handle: &str,
    x: f64,
    y: f64,
    z: f64,
) -> f64 {
    let inputs = match ctx.graph.inputs.get(node_id) {
        Some(m) => m.clone(),
        None => return 1.0,
    };

    let mut indexed: Vec<(usize, String)> = vec![];
    for (handle, source_id) in &inputs {
        if let Some(rest) = handle.strip_prefix(base_handle) {
            if let Some(idx_str) = rest.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    indexed.push((idx, source_id.clone()));
                }
            }
        }
        if handle == base_handle {
            indexed.push((0, source_id.clone()));
        }
    }

    indexed.sort_by_key(|(i, _)| *i);

    if indexed.is_empty() {
        return 1.0;
    }

    let mut prod = 1.0;
    for (_, src) in indexed {
        prod *= evaluate(ctx, &src, x, y, z);
    }
    prod
}

/// Count how many array inputs exist for a given handle base name.
fn count_array_inputs(ctx: &EvalContext, node_id: &str, base_handle: &str) -> usize {
    let inputs = match ctx.graph.inputs.get(node_id) {
        Some(m) => m,
        None => return 0,
    };

    let mut count = 0;
    for handle in inputs.keys() {
        if let Some(rest) = handle.strip_prefix(base_handle) {
            if rest
                .strip_prefix('[')
                .and_then(|s| s.strip_suffix(']'))
                .and_then(|s| s.parse::<usize>().ok())
                .is_some()
            {
                count += 1;
            }
        }
    }
    count
}

// ── Helper: apply curve from a connected curve node ─────────────────

/// Follow a curve handle edge and apply the curve to the input value.
/// Matches the inner `applyCurve` function in `densityEvaluator.ts`.
fn apply_curve_from_handle(
    ctx: &EvalContext,
    node_id: &str,
    curve_handle: &str,
    input_val: f64,
) -> f64 {
    let curve_node_id = match ctx
        .graph
        .inputs
        .get(node_id)
        .and_then(|m| m.get(curve_handle))
    {
        Some(id) => id.clone(),
        None => return input_val,
    };

    let curve_node = match ctx.graph.nodes.get(&curve_node_id) {
        Some(n) => n.clone(),
        None => return input_val,
    };

    let raw_type = curve_node.data.density_type.as_deref().unwrap_or("");
    let curve_type = raw_type.strip_prefix("Curve:").unwrap_or(raw_type);
    let curve_fields = &curve_node.data.fields;

    curves::apply_curve(curve_type, curve_fields, input_val)
}

// ── Main evaluation function ────────────────────────────────────────

/// Evaluate a single node at position (x, y, z).
/// Recursive — follows edges to evaluate input nodes.
pub fn evaluate(ctx: &mut EvalContext, node_id: &str, x: f64, y: f64, z: f64) -> f64 {
    // Cycle guard
    if ctx.visiting.contains(node_id) {
        return 0.0;
    }

    // Memo check
    if let Some(&cached) = ctx.memo.get(node_id) {
        return cached;
    }

    ctx.visiting.insert(node_id.to_string());

    let result = evaluate_inner(ctx, node_id, x, y, z);

    ctx.visiting.remove(node_id);
    ctx.memo.insert(node_id.to_string(), result);

    result
}

/// Inner evaluation — dispatches by node density type.
fn evaluate_inner(ctx: &mut EvalContext, node_id: &str, x: f64, y: f64, z: f64) -> f64 {
    // Look up the node
    let node = match ctx.graph.nodes.get(node_id) {
        Some(n) => n.clone(),
        None => return 0.0,
    };

    let density_type = match &node.data.density_type {
        Some(t) => t.clone(),
        None => return 0.0,
    };

    let fields = node.data.fields.clone();

    match density_type.as_str() {
        // ══════════════════════════════════════════════════════════════
        // Constants
        // ══════════════════════════════════════════════════════════════
        "Constant" => field_f64(&fields, "Value", 0.0),

        "Zero" => 0.0,

        "One" => 1.0,

        // ══════════════════════════════════════════════════════════════
        // Coordinates
        // ══════════════════════════════════════════════════════════════
        "CoordinateX" => x,

        "CoordinateY" => y,

        "CoordinateZ" => z,

        // ══════════════════════════════════════════════════════════════
        // Arithmetic — single input
        // ══════════════════════════════════════════════════════════════
        "Negate" => -get_input(ctx, node_id, "Input", x, y, z),

        "Abs" => get_input(ctx, node_id, "Input", x, y, z).abs(),

        "SquareRoot" => get_input(ctx, node_id, "Input", x, y, z).abs().sqrt(),

        "CubeRoot" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            v.cbrt()
        }

        "Square" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            v * v
        }

        "CubeMath" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            v * v * v
        }

        "Inverse" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            if v == 0.0 {
                0.0
            } else {
                1.0 / v
            }
        }

        "SumSelf" => {
            let count = field_f64(&fields, "Count", 2.0).max(1.0);
            get_input(ctx, node_id, "Input", x, y, z) * count
        }

        "Modulo" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let divisor = field_f64(&fields, "Divisor", 1.0);
            if divisor == 0.0 {
                0.0
            } else {
                v % divisor
            }
        }

        "AmplitudeConstant" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let amp = field_f64(&fields, "Value", 1.0);
            v * amp
        }

        "Pow" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let exp = field_f64(&fields, "Exponent", 2.0);
            v.abs().powf(exp) * v.signum()
        }

        "LinearTransform" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let scale = field_f64(&fields, "Scale", 1.0);
            let offset = field_f64(&fields, "Offset", 0.0);
            v * scale + offset
        }

        "Floor" => get_input(ctx, node_id, "Input", x, y, z).floor(),

        "Ceiling" => get_input(ctx, node_id, "Input", x, y, z).ceil(),

        // ══════════════════════════════════════════════════════════════
        // Arithmetic — array input
        // ══════════════════════════════════════════════════════════════
        "Sum" => {
            let vals = get_array_inputs(ctx, node_id, "Inputs", x, y, z);
            vals.iter().sum()
        }

        "Product" => get_array_inputs_product(ctx, node_id, "Inputs", x, y, z),

        "WeightedSum" => {
            let weights: Vec<f64> = match fields.get("Weights") {
                Some(Value::Array(arr)) => arr.iter().map(|v| v.as_f64().unwrap_or(1.0)).collect(),
                _ => vec![],
            };
            let inputs_map = ctx.graph.inputs.get(node_id).cloned().unwrap_or_default();
            let mut indexed: Vec<(usize, String)> = vec![];
            for (handle, source_id) in &inputs_map {
                if let Some(rest) = handle.strip_prefix("Inputs") {
                    if let Some(idx_str) = rest.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
                    {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            indexed.push((idx, source_id.clone()));
                        }
                    }
                }
            }
            indexed.sort_by_key(|(i, _)| *i);
            let mut wsum = 0.0;
            for (idx, src) in indexed {
                let val = evaluate(ctx, &src, x, y, z);
                let w = weights.get(idx).copied().unwrap_or(1.0);
                wsum += val * w;
            }
            wsum
        }

        "MinFunction" => {
            let vals = get_array_inputs(ctx, node_id, "Inputs", x, y, z);
            if vals.is_empty() {
                0.0
            } else {
                let mut result = vals[0];
                for &v in &vals[1..] {
                    result = result.min(v);
                }
                result
            }
        }

        "MaxFunction" => {
            let vals = get_array_inputs(ctx, node_id, "Inputs", x, y, z);
            if vals.is_empty() {
                0.0
            } else {
                let mut result = vals[0];
                for &v in &vals[1..] {
                    result = result.max(v);
                }
                result
            }
        }

        "AverageFunction" => {
            let vals = get_array_inputs(ctx, node_id, "Inputs", x, y, z);
            if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            }
        }

        "Interpolate" => {
            let a = get_input(ctx, node_id, "InputA", x, y, z);
            let b = get_input(ctx, node_id, "InputB", x, y, z);
            let f = get_input(ctx, node_id, "Factor", x, y, z);
            a + (b - a) * f
        }

        "Offset" => {
            get_input(ctx, node_id, "Input", x, y, z) + get_input(ctx, node_id, "Offset", x, y, z)
        }

        "Amplitude" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let amp = get_input(ctx, node_id, "Amplitude", x, y, z);
            v * amp
        }

        // ══════════════════════════════════════════════════════════════
        // Smooth operations
        // ══════════════════════════════════════════════════════════════
        "SmoothMin" => {
            let k = field_f64(&fields, "Smoothness", 0.1);
            let vals = get_array_inputs(ctx, node_id, "Inputs", x, y, z);
            if vals.is_empty() {
                0.0
            } else {
                let mut result = vals[0];
                for &v in &vals[1..] {
                    result = smooth_min(result, v, k);
                }
                result
            }
        }

        "SmoothMax" => {
            let k = field_f64(&fields, "Smoothness", 0.1);
            let vals = get_array_inputs(ctx, node_id, "Inputs", x, y, z);
            if vals.is_empty() {
                0.0
            } else {
                let mut result = vals[0];
                for &v in &vals[1..] {
                    result = smooth_max(result, v, k);
                }
                result
            }
        }

        "SmoothClamp" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let min_v = field_f64(&fields, "Min", 0.0);
            let max_v = field_f64(&fields, "Max", 1.0);
            let k = field_f64(&fields, "Smoothness", 0.1);
            smooth_max(smooth_min(v, max_v, k), min_v, k)
        }

        "SmoothFloor" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let threshold = field_f64(&fields, "Threshold", 0.0);
            let k = field_f64(&fields, "Smoothness", 0.1);
            smooth_max(v, threshold, k)
        }

        "SmoothCeiling" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let threshold = field_f64(&fields, "Threshold", 1.0);
            let k = field_f64(&fields, "Smoothness", 0.1);
            smooth_min(v, threshold, k)
        }

        // ══════════════════════════════════════════════════════════════
        // Clamping & Range
        // ══════════════════════════════════════════════════════════════
        "Clamp" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let min = field_f64(&fields, "Min", 0.0);
            let max = field_f64(&fields, "Max", 1.0);
            v.max(min).min(max)
        }

        "ClampToIndex" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let min = field_f64(&fields, "Min", 0.0);
            let max = field_f64(&fields, "Max", 255.0);
            v.floor().max(min).min(max)
        }

        "Normalizer" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            let (src_min, src_max) = match fields.get("SourceRange") {
                Some(Value::Object(obj)) => {
                    let smin = obj.get("Min").and_then(|v| v.as_f64()).unwrap_or(-1.0);
                    let smax = obj.get("Max").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    (smin, smax)
                }
                _ => (
                    field_f64(&fields, "FromMin", -1.0),
                    field_f64(&fields, "FromMax", 1.0),
                ),
            };
            let (tgt_min, tgt_max) = match fields.get("TargetRange") {
                Some(Value::Object(obj)) => {
                    let tmin = obj.get("Min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let tmax = obj.get("Max").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    (tmin, tmax)
                }
                _ => (
                    field_f64(&fields, "ToMin", 0.0),
                    field_f64(&fields, "ToMax", 1.0),
                ),
            };
            let range = src_max - src_min;
            let t = if range == 0.0 {
                0.0
            } else {
                (v - src_min) / range
            };
            tgt_min + t * (tgt_max - tgt_min)
        }

        "DoubleNormalizer" => {
            let v = get_input(ctx, node_id, "Input", x, y, z);
            if v < 0.0 {
                let (src_min, src_max) = match fields.get("SourceRangeA") {
                    Some(Value::Object(obj)) => (
                        obj.get("Min").and_then(|v| v.as_f64()).unwrap_or(-1.0),
                        obj.get("Max").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    ),
                    _ => (-1.0, 0.0),
                };
                let (tgt_min, tgt_max) = match fields.get("TargetRangeA") {
                    Some(Value::Object(obj)) => (
                        obj.get("Min").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        obj.get("Max").and_then(|v| v.as_f64()).unwrap_or(0.5),
                    ),
                    _ => (0.0, 0.5),
                };
                let range = src_max - src_min;
                let t = if range == 0.0 {
                    0.0
                } else {
                    (v - src_min) / range
                };
                tgt_min + t * (tgt_max - tgt_min)
            } else {
                let (src_min, src_max) = match fields.get("SourceRangeB") {
                    Some(Value::Object(obj)) => (
                        obj.get("Min").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        obj.get("Max").and_then(|v| v.as_f64()).unwrap_or(1.0),
                    ),
                    _ => (0.0, 1.0),
                };
                let (tgt_min, tgt_max) = match fields.get("TargetRangeB") {
                    Some(Value::Object(obj)) => (
                        obj.get("Min").and_then(|v| v.as_f64()).unwrap_or(0.5),
                        obj.get("Max").and_then(|v| v.as_f64()).unwrap_or(1.0),
                    ),
                    _ => (0.5, 1.0),
                };
                let range = src_max - src_min;
                let t = if range == 0.0 {
                    0.0
                } else {
                    (v - src_min) / range
                };
                tgt_min + t * (tgt_max - tgt_min)
            }
        }

        "RangeChoice" => {
            let cond = get_input(ctx, node_id, "Condition", x, y, z);
            let threshold = field_f64(&fields, "Threshold", 0.5);
            if cond >= threshold {
                get_input(ctx, node_id, "TrueInput", x, y, z)
            } else {
                get_input(ctx, node_id, "FalseInput", x, y, z)
            }
        }

        "Wrap" | "Passthrough" | "Debug" | "FlatCache" => get_input(ctx, node_id, "Input", x, y, z),

        // ══════════════════════════════════════════════════════════════
        // Noise
        // ══════════════════════════════════════════════════════════════
        "SimplexNoise2D" => {
            let freq = field_f64(&fields, "Frequency", 0.01);
            let amp = field_f64(&fields, "Amplitude", 1.0);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let octaves = field_i32(&fields, "Octaves", 1).max(1) as u32;
            let lacunarity = field_f64(&fields, "Lacunarity", 2.0);
            let gain = field_f64(&fields, "Gain", 0.5);
            let perm = ctx.get_perm_table(seed);
            noise::fbm_2d(&perm, x, z, freq, octaves, lacunarity, gain) * amp
        }

        "SimplexNoise3D" => {
            let freq = field_f64(&fields, "Frequency", 0.01);
            let amp = field_f64(&fields, "Amplitude", 1.0);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let octaves = field_i32(&fields, "Octaves", 1).max(1) as u32;
            let lacunarity = field_f64(&fields, "Lacunarity", 2.0);
            let gain = field_f64(&fields, "Gain", 0.5);
            let perm = ctx.get_perm_table(seed);
            noise::fbm_3d(&perm, x, y, z, freq, octaves, lacunarity, gain) * amp
        }

        "SimplexRidgeNoise2D" => {
            let freq = field_f64(&fields, "Frequency", 0.01);
            let amp = field_f64(&fields, "Amplitude", 1.0);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let octaves = field_i32(&fields, "Octaves", 1).max(1) as u32;
            let perm = ctx.get_perm_table(seed);
            noise::ridge_fbm_2d(&perm, x, z, freq, octaves) * amp
        }

        "SimplexRidgeNoise3D" => {
            let freq = field_f64(&fields, "Frequency", 0.01);
            let amp = field_f64(&fields, "Amplitude", 1.0);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let octaves = field_i32(&fields, "Octaves", 1).max(1) as u32;
            let perm = ctx.get_perm_table(seed);
            noise::ridge_fbm_3d(&perm, x, y, z, freq, octaves) * amp
        }

        "FractalNoise2D" => {
            let freq = field_f64(&fields, "Frequency", 0.01);
            let octaves = field_i32(&fields, "Octaves", 4).max(1) as u32;
            let lacunarity = field_f64(&fields, "Lacunarity", 2.0);
            let gain = field_f64(&fields, "Gain", 0.5);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let perm = ctx.get_perm_table(seed);
            noise::fbm_2d(&perm, x, z, freq, octaves, lacunarity, gain)
        }

        "FractalNoise3D" => {
            let freq = field_f64(&fields, "Frequency", 0.01);
            let octaves = field_i32(&fields, "Octaves", 4).max(1) as u32;
            let lacunarity = field_f64(&fields, "Lacunarity", 2.0);
            let gain = field_f64(&fields, "Gain", 0.5);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let perm = ctx.get_perm_table(seed);
            noise::fbm_3d(&perm, x, y, z, freq, octaves, lacunarity, gain)
        }

        "VoronoiNoise2D" => {
            let freq = field_f64(&fields, "Frequency", 0.01);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let cell_type = field_str(&fields, "CellType", "Euclidean");
            let jitter = field_f64(&fields, "Jitter", 1.0);
            let octaves = field_i32(&fields, "Octaves", 1).max(1) as u32;
            let lacunarity = field_f64(&fields, "Lacunarity", 2.0);
            let gain = field_f64(&fields, "Gain", 0.5);
            let return_type = field_str(&fields, "ReturnType", "Distance");

            let mut raw = if octaves > 1 {
                noise::voronoi_fbm_2d(
                    x, z, freq, octaves, lacunarity, gain, seed, cell_type, jitter,
                )
            } else {
                noise::voronoi_2d(x * freq, z * freq, seed, cell_type, jitter)
            };

            if return_type == "Curve" {
                raw = apply_curve_from_handle(ctx, node_id, "ReturnCurve", raw);
            } else if return_type == "Density" {
                raw = get_input(ctx, node_id, "ReturnDensity", x, y, z) * raw;
            }

            raw
        }

        "VoronoiNoise3D" => {
            let freq = field_f64(&fields, "Frequency", 0.01);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let cell_type = field_str(&fields, "CellType", "Euclidean");
            let jitter = field_f64(&fields, "Jitter", 1.0);
            let octaves = field_i32(&fields, "Octaves", 1).max(1) as u32;
            let lacunarity = field_f64(&fields, "Lacunarity", 2.0);
            let gain = field_f64(&fields, "Gain", 0.5);
            let return_type = field_str(&fields, "ReturnType", "Distance");

            let mut raw = if octaves > 1 {
                noise::voronoi_fbm_3d(
                    x, y, z, freq, octaves, lacunarity, gain, seed, cell_type, jitter,
                )
            } else {
                noise::voronoi_3d(x * freq, y * freq, z * freq, seed, cell_type, jitter)
            };

            if return_type == "Curve" {
                raw = apply_curve_from_handle(ctx, node_id, "ReturnCurve", raw);
            } else if return_type == "Density" {
                raw = get_input(ctx, node_id, "ReturnDensity", x, y, z) * raw;
            }

            raw
        }

        // ══════════════════════════════════════════════════════════════
        // Position & Distance
        // ══════════════════════════════════════════════════════════════
        "DistanceFromOrigin" => (x * x + y * y + z * z).sqrt(),

        "DistanceFromAxis" => {
            let axis = field_str(&fields, "Axis", "Y");
            match axis {
                "X" => (y * y + z * z).sqrt(),
                "Z" => (x * x + y * y).sqrt(),
                _ => (x * x + z * z).sqrt(), // Y axis
            }
        }

        "DistanceFromPoint" => {
            let (px, py, pz) = field_vec3(&fields, "Point", 0.0, 0.0, 0.0);
            ((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz)).sqrt()
        }

        "AngleFromOrigin" => z.atan2(x),

        "AngleFromPoint" => {
            let (px, _py, pz) = field_vec3(&fields, "Point", 0.0, 0.0, 0.0);
            (z - pz).atan2(x - px)
        }

        "Angle" => {
            // Get reference vector
            let (ref_x, ref_y, ref_z) = {
                let vec_prov_id = get_input_source(ctx, node_id, "VectorProvider");
                let vec_id = get_input_source(ctx, node_id, "Vector");
                if let Some(vp_id) = vec_prov_id {
                    let v = vectors::evaluate_vector(
                        ctx.graph,
                        &mut ctx.memo,
                        &mut ctx.visiting,
                        &mut ctx.perm_cache,
                        &ctx.content_fields,
                        &vp_id,
                        x,
                        y,
                        z,
                    );
                    (v.x, v.y, v.z)
                } else if let Some(v_id) = vec_id {
                    let v = vectors::evaluate_vector(
                        ctx.graph,
                        &mut ctx.memo,
                        &mut ctx.visiting,
                        &mut ctx.perm_cache,
                        &ctx.content_fields,
                        &v_id,
                        x,
                        y,
                        z,
                    );
                    (v.x, v.y, v.z)
                } else {
                    field_vec3(&fields, "Vector", 0.0, 1.0, 0.0)
                }
            };

            let pos_len = (x * x + y * y + z * z).sqrt();
            let ref_len = (ref_x * ref_x + ref_y * ref_y + ref_z * ref_z).sqrt();

            if pos_len < 1e-10 || ref_len < 1e-10 {
                0.0
            } else {
                let dot_p = (x * ref_x + y * ref_y + z * ref_z) / (pos_len * ref_len);
                let mut angle_deg =
                    dot_p.max(-1.0).min(1.0).acos() * (180.0 / std::f64::consts::PI);
                if field_bool(&fields, "IsAxis", false) && angle_deg > 90.0 {
                    angle_deg = 180.0 - angle_deg;
                }
                angle_deg
            }
        }

        "YGradient" | "GradientDensity" | "Gradient" => {
            let from_y = field_f64(&fields, "FromY", 0.0);
            let to_y = field_f64(&fields, "ToY", DEFAULT_WORLD_HEIGHT);
            let range = to_y - from_y;
            if range == 0.0 {
                0.0
            } else {
                (y - from_y) / range
            }
        }

        "BaseHeight" => {
            let name = field_str(&fields, "BaseHeightName", "Base");
            let base_y = ctx.content_fields.get(name).copied().unwrap_or(100.0);
            let distance = field_bool(&fields, "Distance", false);
            if distance {
                y - base_y
            } else {
                base_y
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Position transforms
        // ══════════════════════════════════════════════════════════════
        "TranslatedPosition" => {
            let (dx, dy, dz) = field_vec3(&fields, "Translation", 0.0, 0.0, 0.0);
            get_input(ctx, node_id, "Input", x - dx, y - dy, z - dz)
        }

        "ScaledPosition" => {
            let (sx, sy, sz) = field_vec3(&fields, "Scale", 1.0, 1.0, 1.0);
            let sx = if sx == 0.0 { 1.0 } else { sx };
            let sy = if sy == 0.0 { 1.0 } else { sy };
            let sz = if sz == 0.0 { 1.0 } else { sz };
            get_input(ctx, node_id, "Input", x / sx, y / sy, z / sz)
        }

        "RotatedPosition" => {
            let angle_deg = field_f64(&fields, "AngleDegrees", 0.0);
            let rad = angle_deg * std::f64::consts::PI / 180.0;
            let cos_a = rad.cos();
            let sin_a = rad.sin();
            let rx = x * cos_a + z * sin_a;
            let rz = -x * sin_a + z * cos_a;
            get_input(ctx, node_id, "Input", rx, y, rz)
        }

        "MirroredPosition" => {
            let axis = field_str(&fields, "Axis", "X");
            let mx = if axis == "X" { x.abs() } else { x };
            let my = if axis == "Y" { y.abs() } else { y };
            let mz = if axis == "Z" { z.abs() } else { z };
            get_input(ctx, node_id, "Input", mx, my, mz)
        }

        "QuantizedPosition" => {
            let step = {
                let s = field_f64(&fields, "StepSize", 1.0);
                if s == 0.0 {
                    1.0
                } else {
                    s
                }
            };
            get_input(
                ctx,
                node_id,
                "Input",
                (x / step).floor() * step,
                (y / step).floor() * step,
                (z / step).floor() * step,
            )
        }

        // ══════════════════════════════════════════════════════════════
        // Position overrides
        // ══════════════════════════════════════════════════════════════
        "YOverride" => {
            let override_y = field_f64(&fields, "OverrideY", field_f64(&fields, "Y", 0.0));
            get_input(ctx, node_id, "Input", x, override_y, z)
        }

        "XOverride" => {
            let override_x = field_f64(&fields, "OverrideX", 0.0);
            get_input(ctx, node_id, "Input", override_x, y, z)
        }

        "ZOverride" => {
            let override_z = field_f64(&fields, "OverrideZ", 0.0);
            get_input(ctx, node_id, "Input", x, y, override_z)
        }

        "YSampled" => {
            let target_y = if has_input(ctx, node_id, "YProvider") {
                get_input(ctx, node_id, "YProvider", x, y, z)
            } else {
                let sample_dist = field_f64(&fields, "SampleDistance", 4.0);
                let sample_offset = field_f64(&fields, "SampleOffset", 0.0);
                if sample_dist > 0.0 {
                    ((y - sample_offset) / sample_dist).round() * sample_dist + sample_offset
                } else {
                    y
                }
            };
            get_input(ctx, node_id, "Input", x, target_y, z)
        }

        // ══════════════════════════════════════════════════════════════
        // Anchor
        // ══════════════════════════════════════════════════════════════
        "Anchor" => {
            let is_reversed = field_bool(&fields, "Reversed", false);
            if ctx.anchor_set {
                if is_reversed {
                    get_input(
                        ctx,
                        node_id,
                        "Input",
                        x + ctx.anchor[0],
                        y + ctx.anchor[1],
                        z + ctx.anchor[2],
                    )
                } else {
                    get_input(
                        ctx,
                        node_id,
                        "Input",
                        x - ctx.anchor[0],
                        y - ctx.anchor[1],
                        z - ctx.anchor[2],
                    )
                }
            } else {
                get_input(ctx, node_id, "Input", 0.0, 0.0, 0.0)
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Conditionals & Blending
        // ══════════════════════════════════════════════════════════════
        "Conditional" => {
            let cond = get_input(ctx, node_id, "Condition", x, y, z);
            let threshold = field_f64(&fields, "Threshold", 0.0);
            if cond >= threshold {
                get_input(ctx, node_id, "TrueInput", x, y, z)
            } else {
                get_input(ctx, node_id, "FalseInput", x, y, z)
            }
        }

        "Blend" => {
            let a = get_input(ctx, node_id, "InputA", x, y, z);
            let b = get_input(ctx, node_id, "InputB", x, y, z);
            let has_factor = has_input(ctx, node_id, "Factor");
            let f = if has_factor {
                get_input(ctx, node_id, "Factor", x, y, z)
            } else {
                0.5
            };
            a + (b - a) * f
        }

        "BlendCurve" => {
            let a = get_input(ctx, node_id, "InputA", x, y, z);
            let b = get_input(ctx, node_id, "InputB", x, y, z);
            let raw_factor = get_input(ctx, node_id, "Factor", x, y, z);
            let curved_factor = apply_curve_from_handle(ctx, node_id, "Curve", raw_factor);
            a + (b - a) * curved_factor
        }

        "Switch" => {
            let switch_states: Option<Vec<Value>> = fields
                .get("SwitchStates")
                .and_then(|v| v.as_array().cloned());

            if let Some(states) = switch_states {
                if !states.is_empty() {
                    let mut matched = false;
                    let mut result = 0.0;
                    for (i, state_val) in states.iter().enumerate() {
                        let state_seed = noise::seed_to_int(state_val);
                        if state_seed == ctx.switch_state {
                            let handle = format!("Inputs[{}]", i);
                            let src = get_input_source(ctx, node_id, &handle);
                            result = match src {
                                Some(src_id) => evaluate(ctx, &src_id, x, y, z),
                                None => 0.0,
                            };
                            matched = true;
                            break;
                        }
                    }
                    if !matched {
                        0.0
                    } else {
                        result
                    }
                } else {
                    let selector = field_f64(&fields, "Selector", 0.0).max(0.0).floor() as usize;
                    let handle = format!("Inputs[{}]", selector);
                    let src = get_input_source(ctx, node_id, &handle);
                    match src {
                        Some(src_id) => evaluate(ctx, &src_id, x, y, z),
                        None => 0.0,
                    }
                }
            } else {
                let selector = field_f64(&fields, "Selector", 0.0).max(0.0).floor() as usize;
                let handle = format!("Inputs[{}]", selector);
                let src = get_input_source(ctx, node_id, &handle);
                match src {
                    Some(src_id) => evaluate(ctx, &src_id, x, y, z),
                    None => 0.0,
                }
            }
        }

        "SwitchState" => {
            if !has_input(ctx, node_id, "Input") {
                field_f64(&fields, "State", 0.0)
            } else {
                let prev_state = ctx.switch_state;
                let state_val = fields.get("State").unwrap_or(&Value::Null);
                ctx.switch_state = noise::seed_to_int(state_val);
                let result = get_input(ctx, node_id, "Input", x, y, z);
                ctx.switch_state = prev_state;
                result
            }
        }

        "MultiMix" => {
            let keys: Vec<f64> = match fields.get("Keys") {
                Some(Value::Array(arr)) => arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect(),
                _ => vec![],
            };
            if keys.is_empty() {
                0.0
            } else {
                let selector = get_input(ctx, node_id, "Selector", x, y, z);
                let mut lo = 0usize;
                for i in 1..keys.len() {
                    if keys[i] <= selector {
                        lo = i;
                    }
                }
                let hi = (lo + 1).min(keys.len() - 1);
                if lo == hi {
                    let handle = format!("Densities[{}]", lo);
                    get_input(ctx, node_id, &handle, x, y, z)
                } else {
                    let range = keys[hi] - keys[lo];
                    let t = if range == 0.0 {
                        0.0
                    } else {
                        ((selector - keys[lo]) / range).max(0.0).min(1.0)
                    };
                    let handle_a = format!("Densities[{}]", lo);
                    let handle_b = format!("Densities[{}]", hi);
                    let a = get_input(ctx, node_id, &handle_a, x, y, z);
                    let b = get_input(ctx, node_id, &handle_b, x, y, z);
                    a + (b - a) * t
                }
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Curves & Splines
        // ══════════════════════════════════════════════════════════════
        "CurveFunction" => {
            let input_val = get_input(ctx, node_id, "Input", x, y, z);
            apply_curve_from_handle(ctx, node_id, "Curve", input_val)
        }

        "SplineFunction" => {
            let input_val = get_input(ctx, node_id, "Input", x, y, z);
            curves::apply_spline(&fields, input_val)
        }

        // ══════════════════════════════════════════════════════════════
        // Domain warp
        // ══════════════════════════════════════════════════════════════
        "DomainWarp2D" => {
            let amp = field_f64(&fields, "Amplitude", 1.0);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let freq = field_f64(&fields, "Frequency", 0.01);
            let perm_x = ctx.get_perm_table(seed);
            let perm_z = ctx.get_perm_table(seed + 1);
            let warp_x = noise::simplex_2d(&perm_x, x * freq, z * freq) * amp;
            let warp_z = noise::simplex_2d(&perm_z, x * freq, z * freq) * amp;
            get_input(ctx, node_id, "Input", x + warp_x, y, z + warp_z)
        }

        "DomainWarp3D" => {
            let amp = field_f64(&fields, "Amplitude", 1.0);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let freq = field_f64(&fields, "Frequency", 0.01);
            let perm_x = ctx.get_perm_table(seed);
            let perm_y = ctx.get_perm_table(seed + 1);
            let perm_z = ctx.get_perm_table(seed + 2);
            let warp_x = noise::simplex_3d(&perm_x, x * freq, y * freq, z * freq) * amp;
            let warp_y = noise::simplex_3d(&perm_y, x * freq, y * freq, z * freq) * amp;
            let warp_z = noise::simplex_3d(&perm_z, x * freq, y * freq, z * freq) * amp;
            get_input(ctx, node_id, "Input", x + warp_x, y + warp_y, z + warp_z)
        }

        "GradientWarp" => {
            let warp_factor =
                field_f64(&fields, "WarpFactor", field_f64(&fields, "WarpScale", 1.0));
            let eps = field_f64(&fields, "SampleRange", 1.0);
            let is_2d = field_bool(&fields, "Is2D", false);
            let y_for_2d = field_f64(&fields, "YFor2D", 0.0);
            let inv2e = 1.0 / (2.0 * eps);
            let sample_y = if is_2d { y_for_2d } else { y };

            let dfdx = (get_input(ctx, node_id, "WarpSource", x + eps, sample_y, z)
                - get_input(ctx, node_id, "WarpSource", x - eps, sample_y, z))
                * inv2e;
            let dfdz = (get_input(ctx, node_id, "WarpSource", x, sample_y, z + eps)
                - get_input(ctx, node_id, "WarpSource", x, sample_y, z - eps))
                * inv2e;

            let wx = x + warp_factor * dfdx;
            let mut wy = y;
            let wz = z + warp_factor * dfdz;

            if !is_2d {
                let dfdy = (get_input(ctx, node_id, "WarpSource", x, sample_y + eps, z)
                    - get_input(ctx, node_id, "WarpSource", x, sample_y - eps, z))
                    * inv2e;
                wy = y + warp_factor * dfdy;
            }

            get_input(ctx, node_id, "Input", wx, wy, wz)
        }

        "FastGradientWarp" => {
            let warp_factor = field_f64(&fields, "WarpFactor", 1.0);
            let warp_seed_val = fields.get("WarpSeed").unwrap_or(&Value::Null);
            let warp_seed = noise::seed_to_int(warp_seed_val);
            let warp_scale = field_f64(&fields, "WarpScale", 0.01);
            let warp_octaves = field_i32(&fields, "WarpOctaves", 3).max(1) as u32;
            let warp_lacunarity = field_f64(&fields, "WarpLacunarity", 2.0);
            let warp_persistence = field_f64(&fields, "WarpPersistence", 0.5);
            let is_2d = field_bool(&fields, "Is2D", false);

            if is_2d {
                let mut gx = 0.0;
                let mut gz = 0.0;
                let mut amp = 1.0;
                let mut freq = warp_scale;
                for i in 0..warp_octaves {
                    let perm = ctx.get_perm_table(warp_seed + i as i32);
                    let r = noise::simplex_2d_with_gradient(&perm, x * freq, z * freq);
                    gx += amp * r.dx * freq;
                    gz += amp * r.dy * freq; // dy in 2D maps to z
                    amp *= warp_persistence;
                    freq *= warp_lacunarity;
                }
                get_input(
                    ctx,
                    node_id,
                    "Input",
                    x + warp_factor * gx,
                    y,
                    z + warp_factor * gz,
                )
            } else {
                let mut gx = 0.0;
                let mut gy = 0.0;
                let mut gz = 0.0;
                let mut amp = 1.0;
                let mut freq = warp_scale;
                for i in 0..warp_octaves {
                    let perm = ctx.get_perm_table(warp_seed + i as i32);
                    let r = noise::simplex_3d_with_gradient(&perm, x * freq, y * freq, z * freq);
                    gx += amp * r.dx * freq;
                    gy += amp * r.dy * freq;
                    gz += amp * r.dz * freq;
                    amp *= warp_persistence;
                    freq *= warp_lacunarity;
                }
                get_input(
                    ctx,
                    node_id,
                    "Input",
                    x + warp_factor * gx,
                    y + warp_factor * gy,
                    z + warp_factor * gz,
                )
            }
        }

        "VectorWarp" => {
            let warp_factor = field_f64(&fields, "WarpFactor", 1.0);
            let dir_node_id = get_input_source(ctx, node_id, "Direction")
                .or_else(|| get_input_source(ctx, node_id, "WarpVector"));

            let dir = match dir_node_id {
                Some(ref id) => vectors::evaluate_vector(
                    ctx.graph,
                    &mut ctx.memo,
                    &mut ctx.visiting,
                    &mut ctx.perm_cache,
                    &ctx.content_fields,
                    id,
                    x,
                    y,
                    z,
                ),
                None => vectors::ZERO_VEC3,
            };

            let dir_len = vectors::vec3_length(dir);
            if dir_len < 1e-10 {
                get_input(ctx, node_id, "Input", x, y, z)
            } else {
                let dir_norm = vectors::vec3_normalize(dir);
                let magnitude = get_input(ctx, node_id, "Magnitude", x, y, z);
                let displacement = magnitude * warp_factor;
                get_input(
                    ctx,
                    node_id,
                    "Input",
                    x + dir_norm.x * displacement,
                    y + dir_norm.y * displacement,
                    z + dir_norm.z * displacement,
                )
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Shape SDFs
        // ══════════════════════════════════════════════════════════════
        "Ellipsoid" => {
            let scale = fields.get("Scale").or_else(|| fields.get("Radius"));
            let (sx, sy, sz) = match scale {
                Some(Value::Object(obj)) => {
                    let x = obj.get("x").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    let y = obj.get("y").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    let z = obj.get("z").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    (
                        if x == 0.0 { 1.0 } else { x },
                        if y == 0.0 { 1.0 } else { y },
                        if z == 0.0 { 1.0 } else { z },
                    )
                }
                _ => (1.0, 1.0, 1.0),
            };
            let rot =
                build_rotation_matrix(fields.get("NewYAxis"), field_f64(&fields, "SpinAngle", 0.0));
            let (erx, ery, erz) = mat3_apply(&rot, x, y, z);
            let esx = erx / sx;
            let esy = ery / sy;
            let esz = erz / sz;
            (esx * esx + esy * esy + esz * esz).sqrt() - 1.0
        }

        "Cuboid" => {
            let scale = fields.get("Scale").or_else(|| fields.get("Size"));
            let (sx, sy, sz) = match scale {
                Some(Value::Object(obj)) => {
                    let x = obj.get("x").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    let y = obj.get("y").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    let z = obj.get("z").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    (
                        if x == 0.0 { 1.0 } else { x },
                        if y == 0.0 { 1.0 } else { y },
                        if z == 0.0 { 1.0 } else { z },
                    )
                }
                _ => (1.0, 1.0, 1.0),
            };
            let rot =
                build_rotation_matrix(fields.get("NewYAxis"), field_f64(&fields, "SpinAngle", 0.0));
            let (crx, cry, crz) = mat3_apply(&rot, x, y, z);
            let csx = crx / sx;
            let csy = cry / sy;
            let csz = crz / sz;
            let qx = csx.abs() - 1.0;
            let qy = csy.abs() - 1.0;
            let qz = csz.abs() - 1.0;
            let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2) + qz.max(0.0).powi(2)).sqrt();
            let inside = qx.max(qy).max(qz).min(0.0);
            outside + inside
        }

        "Cube" => {
            let q = x.abs().max(y.abs()).max(z.abs()) - 1.0;
            apply_curve_from_handle(ctx, node_id, "Curve", q)
        }

        "Cylinder" => {
            let rot =
                build_rotation_matrix(fields.get("NewYAxis"), field_f64(&fields, "SpinAngle", 0.0));
            let (cylrx, cylry, cylrz) = mat3_apply(&rot, x, y, z);
            let radius = {
                let r = field_f64(&fields, "Radius", 1.0);
                if r == 0.0 {
                    1.0
                } else {
                    r
                }
            };
            let height = field_f64(&fields, "Height", 2.0);
            let half_h = height / 2.0;
            let d_radial = (cylrx * cylrx + cylrz * cylrz).sqrt() - radius;
            let d_vertical = cylry.abs() - half_h;
            let outside_r = d_radial.max(0.0);
            let outside_v = d_vertical.max(0.0);
            (outside_r.powi(2) + outside_v.powi(2)).sqrt() + d_radial.max(d_vertical).min(0.0)
        }

        "Plane" => {
            let (mut nx, mut ny, mut nz) = field_vec3(&fields, "Normal", 0.0, 1.0, 0.0);
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let len = if len == 0.0 { 1.0 } else { len };
            nx /= len;
            ny /= len;
            nz /= len;
            let d = field_f64(&fields, "Distance", 0.0);
            nx * x + ny * y + nz * z - d
        }

        "Shell" => {
            // SDF branch when no curves connected and InnerRadius/OuterRadius present
            let has_dist_curve = has_input(ctx, node_id, "DistanceCurve");
            let has_angle_curve = has_input(ctx, node_id, "AngleCurve");

            if !has_dist_curve
                && !has_angle_curve
                && (fields.contains_key("InnerRadius") || fields.contains_key("OuterRadius"))
            {
                let inner = field_f64(&fields, "InnerRadius", 0.0);
                let outer = field_f64(&fields, "OuterRadius", 0.0);
                let mid_radius = (inner + outer) / 2.0;
                let half_thickness = (outer - inner) / 2.0;
                let dist = (x * x + y * y + z * z).sqrt();
                (dist - mid_radius).abs() - half_thickness
            } else {
                // Curve-based Shell
                let (mut sax, mut say, mut saz) = field_vec3(&fields, "Axis", 0.0, 1.0, 0.0);
                let sa_len = (sax * sax + say * say + saz * saz).sqrt();
                if sa_len > 1e-9 {
                    sax /= sa_len;
                    say /= sa_len;
                    saz /= sa_len;
                }
                let shell_mirror = field_bool(&fields, "Mirror", false);

                let shell_dist = (x * x + y * y + z * z).sqrt();
                let dist_amplitude =
                    apply_curve_from_handle(ctx, node_id, "DistanceCurve", shell_dist);

                if dist_amplitude.abs() < 1e-12 {
                    0.0
                } else if shell_dist <= 1e-9 {
                    dist_amplitude
                } else {
                    let dot_p = (x * sax + y * say + z * saz) / shell_dist;
                    let mut angle_deg =
                        dot_p.max(-1.0).min(1.0).acos() * (180.0 / std::f64::consts::PI);
                    if shell_mirror && angle_deg > 90.0 {
                        angle_deg = 180.0 - angle_deg;
                    }
                    let angle_val = apply_curve_from_handle(ctx, node_id, "AngleCurve", angle_deg);
                    dist_amplitude * angle_val
                }
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Voronoi / Positions
        // ══════════════════════════════════════════════════════════════
        "PositionsCellNoise" => {
            let max_dist = field_f64(&fields, "MaxDistance", 0.0);
            let freq = if max_dist > 0.0 {
                1.0 / max_dist
            } else {
                field_f64(&fields, "Frequency", 0.01)
            };
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            let dist_fn = field_str(&fields, "DistanceFunction", "Euclidean");
            let raw = noise::voronoi_2d(x * freq, z * freq, seed, dist_fn, 1.0);

            // Set cell wall distance in context
            ctx.cell_wall_dist = (0.5 - raw.abs()).max(0.0);

            raw
        }

        "CellWallDistance" => {
            if ctx.cell_wall_dist < f64::INFINITY {
                ctx.cell_wall_dist
            } else {
                0.0
            }
        }

        "Positions3D" => {
            let freq = field_f64(&fields, "Frequency", 0.01);
            let seed_val = fields.get("Seed").unwrap_or(&Value::Null);
            let seed = noise::seed_to_int(seed_val);
            noise::voronoi_3d(x * freq, y * freq, z * freq, seed, "Euclidean", 1.0)
        }

        "PositionsPinch" => {
            let strength = field_f64(&fields, "Strength", 1.0);
            let dist = (x * x + z * z).sqrt();
            let pinch_factor = if dist > 0.0 {
                dist.powf(strength) / dist
            } else {
                1.0
            };
            get_input(ctx, node_id, "Input", x * pinch_factor, y, z * pinch_factor)
        }

        "PositionsTwist" => {
            let angle = field_f64(&fields, "Angle", 0.0);
            let rad = (angle * std::f64::consts::PI / 180.0) * y;
            let cos_a = rad.cos();
            let sin_a = rad.sin();
            get_input(
                ctx,
                node_id,
                "Input",
                x * cos_a - z * sin_a,
                y,
                x * sin_a + z * cos_a,
            )
        }

        // ══════════════════════════════════════════════════════════════
        // Distance with curve
        // ══════════════════════════════════════════════════════════════
        "Distance" => {
            let dist = (x * x + y * y + z * z).sqrt();
            apply_curve_from_handle(ctx, node_id, "Curve", dist)
        }

        // ══════════════════════════════════════════════════════════════
        // Passthrough / tagging / caching
        // ══════════════════════════════════════════════════════════════
        "Exported" | "ImportedValue" => {
            if has_input(ctx, node_id, "Input") {
                get_input(ctx, node_id, "Input", x, y, z)
            } else {
                // ImportedValue with no input: read from content_fields
                let name = field_str(&fields, "Name", "");
                ctx.content_fields.get(name).copied().unwrap_or(0.0)
            }
        }

        "CacheOnce" => {
            let key = format!("{}:{}:{}:{}", node_id, x, y, z);
            if let Some(&cached) = ctx.memo.get(&key) {
                cached
            } else {
                let result = get_input(ctx, node_id, "Input", x, y, z);
                ctx.memo.insert(key, result);
                result
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Unsupported (context-dependent, return 0)
        // ══════════════════════════════════════════════════════════════
        "HeightAboveSurface"
        | "SurfaceDensity"
        | "TerrainBoolean"
        | "TerrainMask"
        | "BeardDensity"
        | "ColumnDensity"
        | "CaveDensity"
        | "Terrain"
        | "DistanceToBiomeEdge"
        | "Pipeline" => 0.0,

        // ══════════════════════════════════════════════════════════════
        // Unknown — try to follow Input handle
        // ══════════════════════════════════════════════════════════════
        _ => {
            if has_input(ctx, node_id, "Input") {
                get_input(ctx, node_id, "Input", x, y, z)
            } else if has_input(ctx, node_id, "Inputs[0]") {
                get_input(ctx, node_id, "Inputs[0]", x, y, z)
            } else {
                0.0
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::graph::{EvalGraph, GraphEdge, GraphNode, NodeData};
    use serde_json::json;

    // ── Fixture loading ──────────────────────────────────────────────

    #[derive(serde::Deserialize)]
    struct FixtureCase {
        name: String,
        nodes: Vec<serde_json::Value>,
        edges: Vec<serde_json::Value>,
        sample_points: Vec<SamplePoint>,
    }

    #[derive(serde::Deserialize)]
    struct SamplePoint {
        x: f64,
        y: f64,
        z: f64,
        expected: f64,
    }

    fn load_fixtures() -> Vec<FixtureCase> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tests/fixtures/eval_cases.json");
        let data = std::fs::read_to_string(&path).expect("Failed to read eval_cases.json");
        serde_json::from_str(&data).expect("Failed to parse eval_cases.json")
    }

    fn run_fixture(case: &FixtureCase) {
        let graph_nodes: Vec<GraphNode> = case
            .nodes
            .iter()
            .map(|v| serde_json::from_value(v.clone()).unwrap())
            .collect();
        let graph_edges: Vec<GraphEdge> = case
            .edges
            .iter()
            .map(|v| serde_json::from_value(v.clone()).unwrap())
            .collect();

        let graph = EvalGraph::from_raw(graph_nodes, graph_edges, None)
            .unwrap_or_else(|e| panic!("Failed to build graph for '{}': {}", case.name, e));

        let mut ctx = EvalContext::new(&graph, HashMap::new());

        for (i, point) in case.sample_points.iter().enumerate() {
            ctx.clear_memo();
            let result = evaluate(&mut ctx, &graph.root_id, point.x, point.y, point.z);
            let diff = (result - point.expected).abs();
            assert!(
                diff < 1e-6,
                "Fixture '{}' point {}: expected {}, got {} (diff={})\n  at ({}, {}, {})",
                case.name,
                i,
                point.expected,
                result,
                diff,
                point.x,
                point.y,
                point.z,
            );
        }
    }

    #[test]
    fn all_fixture_cases() {
        let cases = load_fixtures();
        assert!(
            cases.len() >= 5,
            "Expected at least 5 fixture cases, got {}",
            cases.len()
        );
        for case in &cases {
            run_fixture(case);
        }
    }

    // ── Unit tests for individual node types ─────────────────────────

    fn make_node(id: &str, density_type: &str, fields: serde_json::Value) -> GraphNode {
        let fields_map: HashMap<String, Value> = match fields {
            Value::Object(m) => m.into_iter().collect(),
            _ => HashMap::new(),
        };
        GraphNode {
            id: id.to_string(),
            data: NodeData {
                density_type: Some(density_type.to_string()),
                fields: fields_map,
                is_output: false,
                biome_field: None,
            },
            node_type: None,
        }
    }

    fn make_output_node(id: &str, density_type: &str, fields: serde_json::Value) -> GraphNode {
        let fields_map: HashMap<String, Value> = match fields {
            Value::Object(m) => m.into_iter().collect(),
            _ => HashMap::new(),
        };
        GraphNode {
            id: id.to_string(),
            data: NodeData {
                density_type: Some(density_type.to_string()),
                fields: fields_map,
                is_output: true,
                biome_field: None,
            },
            node_type: None,
        }
    }

    fn make_edge(source: &str, target: &str, handle: &str) -> GraphEdge {
        GraphEdge {
            source: source.to_string(),
            target: target.to_string(),
            target_handle: Some(handle.to_string()),
        }
    }

    fn eval_single(nodes: Vec<GraphNode>, edges: Vec<GraphEdge>, x: f64, y: f64, z: f64) -> f64 {
        let graph = EvalGraph::from_raw(nodes, edges, None).unwrap();
        let mut ctx = EvalContext::new(&graph, HashMap::new());
        evaluate(&mut ctx, &graph.root_id, x, y, z)
    }

    // ── Constants ────────────────────────────────────────────────────

    #[test]
    fn constant_value() {
        let nodes = vec![make_node("c", "Constant", json!({"Value": 42.0}))];
        assert_eq!(eval_single(nodes, vec![], 0.0, 0.0, 0.0), 42.0);
    }

    #[test]
    fn constant_default() {
        let nodes = vec![make_node("c", "Constant", json!({}))];
        assert_eq!(eval_single(nodes, vec![], 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn zero_node() {
        let nodes = vec![make_node("z", "Zero", json!({}))];
        assert_eq!(eval_single(nodes, vec![], 5.0, 10.0, 15.0), 0.0);
    }

    #[test]
    fn one_node() {
        let nodes = vec![make_node("o", "One", json!({}))];
        assert_eq!(eval_single(nodes, vec![], 5.0, 10.0, 15.0), 1.0);
    }

    // ── Coordinates ──────────────────────────────────────────────────

    #[test]
    fn coordinate_x() {
        let nodes = vec![make_node("cx", "CoordinateX", json!({}))];
        assert_eq!(eval_single(nodes, vec![], 42.0, 0.0, 0.0), 42.0);
    }

    #[test]
    fn coordinate_y() {
        let nodes = vec![make_node("cy", "CoordinateY", json!({}))];
        assert_eq!(eval_single(nodes, vec![], 0.0, 64.0, 0.0), 64.0);
    }

    #[test]
    fn coordinate_z() {
        let nodes = vec![make_node("cz", "CoordinateZ", json!({}))];
        assert_eq!(eval_single(nodes, vec![], 0.0, 0.0, 99.0), 99.0);
    }

    // ── Arithmetic (single input) ────────────────────────────────────

    #[test]
    fn negate() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 7.0})),
            make_output_node("n", "Negate", json!({})),
        ];
        let edges = vec![make_edge("c", "n", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), -7.0);
    }

    #[test]
    fn abs_negative() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": -13.0})),
            make_output_node("a", "Abs", json!({})),
        ];
        let edges = vec![make_edge("c", "a", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 13.0);
    }

    #[test]
    fn abs_positive() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 13.0})),
            make_output_node("a", "Abs", json!({})),
        ];
        let edges = vec![make_edge("c", "a", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 13.0);
    }

    #[test]
    fn square_root() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 16.0})),
            make_output_node("s", "SquareRoot", json!({})),
        ];
        let edges = vec![make_edge("c", "s", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn square_root_negative() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": -9.0})),
            make_output_node("s", "SquareRoot", json!({})),
        ];
        let edges = vec![make_edge("c", "s", "Input")];
        // sqrt(|-9|) = 3
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn cube_root() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 27.0})),
            make_output_node("cr", "CubeRoot", json!({})),
        ];
        let edges = vec![make_edge("c", "cr", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn square() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 5.0})),
            make_output_node("sq", "Square", json!({})),
        ];
        let edges = vec![make_edge("c", "sq", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn cube_math() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 3.0})),
            make_output_node("cm", "CubeMath", json!({})),
        ];
        let edges = vec![make_edge("c", "cm", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 27.0).abs() < 1e-10);
    }

    #[test]
    fn inverse() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 4.0})),
            make_output_node("inv", "Inverse", json!({})),
        ];
        let edges = vec![make_edge("c", "inv", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn inverse_zero() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 0.0})),
            make_output_node("inv", "Inverse", json!({})),
        ];
        let edges = vec![make_edge("c", "inv", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn sum_self() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 5.0})),
            make_output_node("ss", "SumSelf", json!({"Count": 3.0})),
        ];
        let edges = vec![make_edge("c", "ss", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn modulo() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 7.0})),
            make_output_node("m", "Modulo", json!({"Divisor": 3.0})),
        ];
        let edges = vec![make_edge("c", "m", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn modulo_zero_divisor() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 7.0})),
            make_output_node("m", "Modulo", json!({"Divisor": 0.0})),
        ];
        let edges = vec![make_edge("c", "m", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn amplitude_constant() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 3.0})),
            make_output_node("ac", "AmplitudeConstant", json!({"Value": 5.0})),
        ];
        let edges = vec![make_edge("c", "ac", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn pow() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": -3.0})),
            make_output_node("p", "Pow", json!({"Exponent": 2.0})),
        ];
        let edges = vec![make_edge("c", "p", "Input")];
        // |−3|^2 * sign(−3) = 9 * (−1) = −9
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - (-9.0)).abs() < 1e-10);
    }

    #[test]
    fn linear_transform() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 5.0})),
            make_output_node(
                "lt",
                "LinearTransform",
                json!({"Scale": 3.0, "Offset": 10.0}),
            ),
        ];
        let edges = vec![make_edge("c", "lt", "Input")];
        // 5 * 3 + 10 = 25
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 25.0);
    }

    #[test]
    fn floor_node() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 3.7})),
            make_output_node("f", "Floor", json!({})),
        ];
        let edges = vec![make_edge("c", "f", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 3.0);
    }

    #[test]
    fn ceiling_node() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 3.2})),
            make_output_node("ce", "Ceiling", json!({})),
        ];
        let edges = vec![make_edge("c", "ce", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 4.0);
    }

    // ── Arithmetic (array input) ─────────────────────────────────────

    #[test]
    fn sum_of_two() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 10.0})),
            make_node("b", "Constant", json!({"Value": 20.0})),
            make_output_node("s", "Sum", json!({})),
        ];
        let edges = vec![
            make_edge("a", "s", "Inputs[0]"),
            make_edge("b", "s", "Inputs[1]"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 30.0);
    }

    #[test]
    fn sum_of_three() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 1.0})),
            make_node("b", "Constant", json!({"Value": 2.0})),
            make_node("c", "Constant", json!({"Value": 3.0})),
            make_output_node("s", "Sum", json!({})),
        ];
        let edges = vec![
            make_edge("a", "s", "Inputs[0]"),
            make_edge("b", "s", "Inputs[1]"),
            make_edge("c", "s", "Inputs[2]"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 6.0);
    }

    #[test]
    fn product_of_two() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 3.0})),
            make_node("b", "Constant", json!({"Value": 7.0})),
            make_output_node("p", "Product", json!({})),
        ];
        let edges = vec![
            make_edge("a", "p", "Inputs[0]"),
            make_edge("b", "p", "Inputs[1]"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 21.0);
    }

    #[test]
    fn product_with_zero() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 0.0})),
            make_node("b", "Constant", json!({"Value": 999.0})),
            make_output_node("p", "Product", json!({})),
        ];
        let edges = vec![
            make_edge("a", "p", "Inputs[0]"),
            make_edge("b", "p", "Inputs[1]"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn min_function() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 5.0})),
            make_node("b", "Constant", json!({"Value": 3.0})),
            make_output_node("m", "MinFunction", json!({})),
        ];
        let edges = vec![
            make_edge("a", "m", "Inputs[0]"),
            make_edge("b", "m", "Inputs[1]"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 3.0);
    }

    #[test]
    fn max_function() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 5.0})),
            make_node("b", "Constant", json!({"Value": 3.0})),
            make_output_node("m", "MaxFunction", json!({})),
        ];
        let edges = vec![
            make_edge("a", "m", "Inputs[0]"),
            make_edge("b", "m", "Inputs[1]"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 5.0);
    }

    #[test]
    fn average_function() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 4.0})),
            make_node("b", "Constant", json!({"Value": 8.0})),
            make_output_node("avg", "AverageFunction", json!({})),
        ];
        let edges = vec![
            make_edge("a", "avg", "Inputs[0]"),
            make_edge("b", "avg", "Inputs[1]"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 6.0);
    }

    #[test]
    fn interpolate() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 0.0})),
            make_node("b", "Constant", json!({"Value": 10.0})),
            make_node("f", "Constant", json!({"Value": 0.3})),
            make_output_node("interp", "Interpolate", json!({})),
        ];
        let edges = vec![
            make_edge("a", "interp", "InputA"),
            make_edge("b", "interp", "InputB"),
            make_edge("f", "interp", "Factor"),
        ];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 3.0).abs() < 1e-10);
    }

    // ── Clamping & range ─────────────────────────────────────────────

    #[test]
    fn clamp_above_max() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 150.0})),
            make_output_node("cl", "Clamp", json!({"Min": 0.0, "Max": 100.0})),
        ];
        let edges = vec![make_edge("c", "cl", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 100.0);
    }

    #[test]
    fn clamp_below_min() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": -20.0})),
            make_output_node("cl", "Clamp", json!({"Min": 0.0, "Max": 100.0})),
        ];
        let edges = vec![make_edge("c", "cl", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn clamp_within_range() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 50.0})),
            make_output_node("cl", "Clamp", json!({"Min": 0.0, "Max": 100.0})),
        ];
        let edges = vec![make_edge("c", "cl", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 50.0);
    }

    #[test]
    fn clamp_to_index() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 3.7})),
            make_output_node("ci", "ClampToIndex", json!({"Min": 0.0, "Max": 255.0})),
        ];
        let edges = vec![make_edge("c", "ci", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 3.0);
    }

    #[test]
    fn normalizer_remap() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 0.5})),
            make_output_node(
                "n",
                "Normalizer",
                json!({
                    "SourceRange": {"Min": 0.0, "Max": 1.0},
                    "TargetRange": {"Min": -100.0, "Max": 100.0}
                }),
            ),
        ];
        let edges = vec![make_edge("c", "n", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn normalizer_remap_edge() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": -1.0})),
            make_output_node(
                "n",
                "Normalizer",
                json!({
                    "SourceRange": {"Min": -1.0, "Max": 1.0},
                    "TargetRange": {"Min": 0.0, "Max": 100.0}
                }),
            ),
        ];
        let edges = vec![make_edge("c", "n", "Input")];
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn double_normalizer() {
        // Negative value: map from [-1,0] to [0,0.5]
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": -0.5})),
            make_output_node(
                "dn",
                "DoubleNormalizer",
                json!({
                    "SourceRangeA": {"Min": -1.0, "Max": 0.0},
                    "TargetRangeA": {"Min": 0.0, "Max": 0.5},
                    "SourceRangeB": {"Min": 0.0, "Max": 1.0},
                    "TargetRangeB": {"Min": 0.5, "Max": 1.0}
                }),
            ),
        ];
        let edges = vec![make_edge("c", "dn", "Input")];
        // -0.5 in [-1,0] → t=0.5 → 0.0 + 0.5*0.5 = 0.25
        assert!((eval_single(nodes, edges, 0.0, 0.0, 0.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn range_choice() {
        let nodes = vec![
            make_node("cond", "Constant", json!({"Value": 0.8})),
            make_node("t", "Constant", json!({"Value": 10.0})),
            make_node("f", "Constant", json!({"Value": 20.0})),
            make_output_node("rc", "RangeChoice", json!({"Threshold": 0.5})),
        ];
        let edges = vec![
            make_edge("cond", "rc", "Condition"),
            make_edge("t", "rc", "TrueInput"),
            make_edge("f", "rc", "FalseInput"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 10.0);
    }

    // ── Smooth operations ────────────────────────────────────────────

    #[test]
    fn smooth_min_basic() {
        // Direct function test
        assert!(smooth_min(3.0, 5.0, 0.0) == 3.0);
        let result = smooth_min(3.0, 5.0, 1.0);
        assert!(result <= 3.0, "SmoothMin should be <= min: {}", result);
    }

    #[test]
    fn smooth_max_basic() {
        assert!(smooth_max(3.0, 5.0, 0.0) == 5.0);
        let result = smooth_max(3.0, 5.0, 1.0);
        assert!(result >= 5.0, "SmoothMax should be >= max: {}", result);
    }

    // ── Position & Distance ──────────────────────────────────────────

    #[test]
    fn distance_from_origin() {
        let nodes = vec![make_node("d", "DistanceFromOrigin", json!({}))];
        let result = eval_single(nodes, vec![], 3.0, 4.0, 0.0);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn distance_from_axis_y() {
        let nodes = vec![make_node("d", "DistanceFromAxis", json!({"Axis": "Y"}))];
        let result = eval_single(nodes, vec![], 3.0, 100.0, 4.0);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn distance_from_point() {
        let nodes = vec![make_node(
            "d",
            "DistanceFromPoint",
            json!({"Point": {"x": 1.0, "y": 0.0, "z": 0.0}}),
        )];
        let result = eval_single(nodes, vec![], 4.0, 0.0, 0.0);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn y_gradient() {
        let nodes = vec![make_node(
            "yg",
            "YGradient",
            json!({"FromY": 0.0, "ToY": 100.0}),
        )];
        let result = eval_single(nodes, vec![], 0.0, 50.0, 0.0);
        assert!((result - 0.5).abs() < 1e-10);
    }

    // ── Position transforms ──────────────────────────────────────────

    #[test]
    fn translated_position() {
        // Constant(1) at translated position: CoordinateX shifted by 10
        let nodes = vec![
            make_node("cx", "CoordinateX", json!({})),
            make_output_node(
                "tp",
                "TranslatedPosition",
                json!({"Translation": {"x": 10.0, "y": 0.0, "z": 0.0}}),
            ),
        ];
        let edges = vec![make_edge("cx", "tp", "Input")];
        // Input at x=5 → child sees x=5-10=-5
        assert_eq!(eval_single(nodes, edges, 5.0, 0.0, 0.0), -5.0);
    }

    #[test]
    fn scaled_position() {
        let nodes = vec![
            make_node("cx", "CoordinateX", json!({})),
            make_output_node(
                "sp",
                "ScaledPosition",
                json!({"Scale": {"x": 2.0, "y": 1.0, "z": 1.0}}),
            ),
        ];
        let edges = vec![make_edge("cx", "sp", "Input")];
        // Input at x=10 → child sees x=10/2=5
        assert_eq!(eval_single(nodes, edges, 10.0, 0.0, 0.0), 5.0);
    }

    #[test]
    fn mirrored_position() {
        let nodes = vec![
            make_node("cx", "CoordinateX", json!({})),
            make_output_node("mp", "MirroredPosition", json!({"Axis": "X"})),
        ];
        let edges = vec![make_edge("cx", "mp", "Input")];
        // Input at x=-5 → child sees x=5
        assert_eq!(eval_single(nodes, edges, -5.0, 0.0, 0.0), 5.0);
    }

    #[test]
    fn quantized_position() {
        let nodes = vec![
            make_node("cx", "CoordinateX", json!({})),
            make_output_node("qp", "QuantizedPosition", json!({"StepSize": 10.0})),
        ];
        let edges = vec![make_edge("cx", "qp", "Input")];
        // Input at x=15 → floor(15/10)*10 = 10
        assert_eq!(eval_single(nodes, edges, 15.0, 0.0, 0.0), 10.0);
    }

    // ── Conditionals ─────────────────────────────────────────────────

    #[test]
    fn conditional_true() {
        let nodes = vec![
            make_node("cond", "Constant", json!({"Value": 1.0})),
            make_node("t", "Constant", json!({"Value": 42.0})),
            make_node("f", "Constant", json!({"Value": 0.0})),
            make_output_node("c", "Conditional", json!({"Threshold": 0.5})),
        ];
        let edges = vec![
            make_edge("cond", "c", "Condition"),
            make_edge("t", "c", "TrueInput"),
            make_edge("f", "c", "FalseInput"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 42.0);
    }

    #[test]
    fn conditional_false() {
        let nodes = vec![
            make_node("cond", "Constant", json!({"Value": 0.0})),
            make_node("t", "Constant", json!({"Value": 42.0})),
            make_node("f", "Constant", json!({"Value": 99.0})),
            make_output_node("c", "Conditional", json!({"Threshold": 0.5})),
        ];
        let edges = vec![
            make_edge("cond", "c", "Condition"),
            make_edge("t", "c", "TrueInput"),
            make_edge("f", "c", "FalseInput"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 99.0);
    }

    #[test]
    fn blend_with_factor() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 0.0})),
            make_node("b", "Constant", json!({"Value": 10.0})),
            make_node("f", "Constant", json!({"Value": 0.5})),
            make_output_node("bl", "Blend", json!({})),
        ];
        let edges = vec![
            make_edge("a", "bl", "InputA"),
            make_edge("b", "bl", "InputB"),
            make_edge("f", "bl", "Factor"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 5.0);
    }

    #[test]
    fn blend_default_factor() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 0.0})),
            make_node("b", "Constant", json!({"Value": 10.0})),
            make_output_node("bl", "Blend", json!({})),
        ];
        let edges = vec![
            make_edge("a", "bl", "InputA"),
            make_edge("b", "bl", "InputB"),
        ];
        // Default factor is 0.5
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 5.0);
    }

    // ── Shape SDFs ───────────────────────────────────────────────────

    #[test]
    fn ellipsoid_at_origin() {
        let nodes = vec![make_node(
            "e",
            "Ellipsoid",
            json!({"Scale": {"x": 1.0, "y": 1.0, "z": 1.0}}),
        )];
        // At origin: sqrt(0) - 1 = -1 (inside)
        assert!((eval_single(nodes, vec![], 0.0, 0.0, 0.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn ellipsoid_on_surface() {
        let nodes = vec![make_node(
            "e",
            "Ellipsoid",
            json!({"Scale": {"x": 1.0, "y": 1.0, "z": 1.0}}),
        )];
        // At (1,0,0): sqrt(1) - 1 = 0 (on surface)
        assert!((eval_single(nodes, vec![], 1.0, 0.0, 0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn plane_sdf() {
        let nodes = vec![make_node(
            "p",
            "Plane",
            json!({"Normal": {"x": 0.0, "y": 1.0, "z": 0.0}, "Distance": 10.0}),
        )];
        // At y=15: 0*x + 1*15 + 0*z - 10 = 5
        assert!((eval_single(nodes, vec![], 0.0, 15.0, 0.0) - 5.0).abs() < 1e-10);
    }

    // ── Noise ────────────────────────────────────────────────────────

    #[test]
    fn simplex_2d_produces_nonzero() {
        let nodes = vec![make_node(
            "n",
            "SimplexNoise2D",
            json!({
                "Frequency": 0.1,
                "Amplitude": 1.0,
                "Seed": 42,
                "Octaves": 4,
                "Lacunarity": 2.0,
                "Gain": 0.5
            }),
        )];
        let result = eval_single(nodes, vec![], 10.0, 64.0, 20.0);
        assert!(result.is_finite());
    }

    #[test]
    fn simplex_3d_produces_nonzero() {
        let nodes = vec![make_node(
            "n",
            "SimplexNoise3D",
            json!({
                "Frequency": 0.1,
                "Amplitude": 1.0,
                "Seed": 42,
                "Octaves": 4,
                "Lacunarity": 2.0,
                "Gain": 0.5
            }),
        )];
        let result = eval_single(nodes, vec![], 10.0, 64.0, 20.0);
        assert!(result.is_finite());
    }

    #[test]
    fn simplex_2d_deterministic() {
        let nodes1 = vec![make_node(
            "n",
            "SimplexNoise2D",
            json!({"Frequency": 0.05, "Amplitude": 2.0, "Seed": "hello"}),
        )];
        let nodes2 = vec![make_node(
            "n",
            "SimplexNoise2D",
            json!({"Frequency": 0.05, "Amplitude": 2.0, "Seed": "hello"}),
        )];
        let a = eval_single(nodes1, vec![], 50.0, 64.0, -30.0);
        let b = eval_single(nodes2, vec![], 50.0, 64.0, -30.0);
        assert_eq!(a, b);
    }

    #[test]
    fn simplex_noise_different_seeds_differ() {
        let nodes1 = vec![make_node(
            "n",
            "SimplexNoise2D",
            json!({"Frequency": 0.1, "Amplitude": 1.0, "Seed": 1}),
        )];
        let nodes2 = vec![make_node(
            "n",
            "SimplexNoise2D",
            json!({"Frequency": 0.1, "Amplitude": 1.0, "Seed": 2}),
        )];
        let a = eval_single(nodes1, vec![], 10.0, 0.0, 20.0);
        let b = eval_single(nodes2, vec![], 10.0, 0.0, 20.0);
        assert_ne!(a, b);
    }

    #[test]
    fn ridge_noise_2d() {
        let nodes = vec![make_node(
            "n",
            "SimplexRidgeNoise2D",
            json!({"Frequency": 0.1, "Amplitude": 1.0, "Seed": 42, "Octaves": 3}),
        )];
        let result = eval_single(nodes, vec![], 10.0, 64.0, 20.0);
        assert!(result.is_finite());
    }

    #[test]
    fn voronoi_noise_2d() {
        let nodes = vec![make_node(
            "n",
            "VoronoiNoise2D",
            json!({"Frequency": 0.1, "Seed": 42}),
        )];
        let result = eval_single(nodes, vec![], 10.0, 64.0, 20.0);
        assert!(result.is_finite());
    }

    #[test]
    fn fractal_noise_2d() {
        let nodes = vec![make_node(
            "n",
            "FractalNoise2D",
            json!({"Frequency": 0.05, "Seed": 123, "Octaves": 4}),
        )];
        let result = eval_single(nodes, vec![], 10.0, 64.0, 20.0);
        assert!(result.is_finite());
    }

    // ── Passthrough / caching ────────────────────────────────────────

    #[test]
    fn passthrough() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 7.0})),
            make_output_node("p", "Passthrough", json!({})),
        ];
        let edges = vec![make_edge("c", "p", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 7.0);
    }

    #[test]
    fn flat_cache() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 7.0})),
            make_output_node("fc", "FlatCache", json!({})),
        ];
        let edges = vec![make_edge("c", "fc", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 7.0);
    }

    #[test]
    fn exported_passthrough() {
        let nodes = vec![
            make_node("c", "Constant", json!({"Value": 42.0})),
            make_output_node("e", "Exported", json!({})),
        ];
        let edges = vec![make_edge("c", "e", "Input")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 42.0);
    }

    // ── Chained/complex ──────────────────────────────────────────────

    #[test]
    fn chained_sum_negate_abs() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 3.0})),
            make_node("b", "Constant", json!({"Value": 5.0})),
            make_node("s", "Sum", json!({})),
            make_node("n", "Negate", json!({})),
            make_output_node("ab", "Abs", json!({})),
        ];
        let edges = vec![
            make_edge("a", "s", "Inputs[0]"),
            make_edge("b", "s", "Inputs[1]"),
            make_edge("s", "n", "Input"),
            make_edge("n", "ab", "Input"),
        ];
        // sum=8, negate=-8, abs=8
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 8.0);
    }

    #[test]
    fn output_node_root_selection() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 1.0})),
            make_output_node("b", "Constant", json!({"Value": 99.0})),
        ];
        assert_eq!(eval_single(nodes, vec![], 0.0, 0.0, 0.0), 99.0);
    }

    #[test]
    fn cycle_detection() {
        let nodes = vec![
            make_node("a", "Sum", json!({})),
            make_output_node("b", "Sum", json!({})),
        ];
        let edges = vec![
            make_edge("a", "b", "Inputs[0]"),
            make_edge("b", "a", "Inputs[0]"),
        ];
        let result = eval_single(nodes, edges, 0.0, 0.0, 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn missing_input_returns_zero() {
        let nodes = vec![make_node("n", "Negate", json!({}))];
        assert_eq!(eval_single(nodes, vec![], 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn memoization_works() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 5.0})),
            make_output_node("s", "Sum", json!({})),
        ];
        let edges = vec![
            make_edge("a", "s", "Inputs[0]"),
            make_edge("a", "s", "Inputs[1]"),
        ];
        let result = eval_single(nodes, edges, 0.0, 0.0, 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn unknown_type_returns_zero() {
        let _nodes = vec![make_node("x", "UnknownMagicType", json!({}))];
        let nodes = vec![
            make_node("x", "UnknownMagicType", json!({})),
            make_output_node("s", "Sum", json!({})),
        ];
        let edges = vec![make_edge("x", "s", "Inputs[0]")];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn clear_memo_between_samples() {
        let nodes = vec![make_node("cx", "CoordinateX", json!({}))];
        let graph = EvalGraph::from_raw(nodes, vec![], None).unwrap();
        let mut ctx = EvalContext::new(&graph, HashMap::new());

        let r1 = evaluate(&mut ctx, &graph.root_id, 10.0, 0.0, 0.0);
        assert_eq!(r1, 10.0);

        ctx.clear_memo();
        let r2 = evaluate(&mut ctx, &graph.root_id, 20.0, 0.0, 0.0);
        assert_eq!(r2, 20.0);
    }

    // ── Unsupported types ────────────────────────────────────────────

    #[test]
    fn unsupported_types_return_zero() {
        for t in &[
            "HeightAboveSurface",
            "SurfaceDensity",
            "TerrainBoolean",
            "TerrainMask",
            "Pipeline",
        ] {
            let nodes = vec![
                make_node("u", t, json!({})),
                make_output_node("s", "Sum", json!({})),
            ];
            let edges = vec![make_edge("u", "s", "Inputs[0]")];
            assert_eq!(
                eval_single(nodes, edges, 0.0, 0.0, 0.0),
                0.0,
                "Failed for type {}",
                t
            );
        }
    }

    // ── BaseHeight ───────────────────────────────────────────────────

    #[test]
    fn base_height_default() {
        let nodes = vec![make_node("bh", "BaseHeight", json!({}))];
        let graph = EvalGraph::from_raw(nodes, vec![], None).unwrap();
        let mut ctx = EvalContext::new(&graph, HashMap::new());
        let result = evaluate(&mut ctx, &graph.root_id, 0.0, 64.0, 0.0);
        // Default: Base=100, Distance=false → returns 100
        assert_eq!(result, 100.0);
    }

    #[test]
    fn base_height_distance() {
        let nodes = vec![make_node("bh", "BaseHeight", json!({"Distance": true}))];
        let graph = EvalGraph::from_raw(nodes, vec![], None).unwrap();
        let mut content = HashMap::new();
        content.insert("Base".to_string(), 100.0);
        let mut ctx = EvalContext::new(&graph, content);
        let result = evaluate(&mut ctx, &graph.root_id, 0.0, 64.0, 0.0);
        // Distance mode: y - baseY = 64 - 100 = -36
        assert_eq!(result, -36.0);
    }

    // ── Offset ───────────────────────────────────────────────────────

    #[test]
    fn offset_node() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 3.0})),
            make_node("b", "Constant", json!({"Value": 7.0})),
            make_output_node("off", "Offset", json!({})),
        ];
        let edges = vec![
            make_edge("a", "off", "Input"),
            make_edge("b", "off", "Offset"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 10.0);
    }

    // ── Amplitude ────────────────────────────────────────────────────

    #[test]
    fn amplitude_node() {
        let nodes = vec![
            make_node("a", "Constant", json!({"Value": 3.0})),
            make_node("b", "Constant", json!({"Value": 5.0})),
            make_output_node("amp", "Amplitude", json!({})),
        ];
        let edges = vec![
            make_edge("a", "amp", "Input"),
            make_edge("b", "amp", "Amplitude"),
        ];
        assert_eq!(eval_single(nodes, edges, 0.0, 0.0, 0.0), 15.0);
    }
}
