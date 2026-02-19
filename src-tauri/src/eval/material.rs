// eval/material.rs — Material graph evaluator
//
// Walks the material node graph per-voxel to determine material assignments,
// mirroring the TS `materialEvaluator.ts` architecture. Falls back to
// depth-based assignment when no material graph exists.
//
// Layout: densities[y * n * n + z * n + x]

use crate::eval::graph::{EvalGraph, GraphEdge, GraphNode, NodeData};
use crate::eval::nodes::{evaluate, EvalContext};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

// ── Constants ───────────────────────────────────────────────────────

/// Voxels with density >= SOLID_THRESHOLD are considered solid.
const SOLID_THRESHOLD: f32 = 0.0;

/// Deterministic hash primes (mirrors TS constants).
const HASH_PRIME_A: i64 = 374761393;
const HASH_PRIME_B: i64 = 668265263;
const HASH_PRIME_C: i64 = 1103515245;

// ── Result types ────────────────────────────────────────────────────

/// Result of material graph evaluation.
#[derive(Debug, Serialize, Clone)]
pub struct MaterialResult {
    /// Per-voxel material index (indexes into palette).
    /// Length = resolution × resolution × y_slices.
    pub material_ids: Vec<u8>,
    /// Material palette.
    pub palette: Vec<MaterialEntry>,
}

/// A single material in the palette.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MaterialEntry {
    pub name: String,
    pub color: String,
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    #[serde(default)]
    pub metalness: f32,
    #[serde(default = "default_emissive")]
    pub emissive: String,
    #[serde(default)]
    pub emissive_intensity: f32,
}

fn default_roughness() -> f32 {
    0.8
}
fn default_emissive() -> String {
    "#000000".to_string()
}

// ── Voxel context ───────────────────────────────────────────────────

/// Per-voxel context passed to material node evaluation.
#[allow(dead_code)]
struct MaterialVoxelContext {
    /// World coordinates.
    x: f64,
    y: f64,
    z: f64,
    /// Grid indices.
    xi: usize,
    yi: usize,
    zi: usize,
    /// Whether this voxel is solid.
    is_solid: bool,
    /// Raw density value.
    density: f32,
    /// Number of solid voxels from surface downward to this voxel.
    downward_depth: i32,
    /// Number of solid voxels from bottom upward to this voxel.
    upward_depth: i32,
    /// Number of air voxels above the surface in this column.
    space_above: i32,
    /// Number of air voxels below the bottom solid in this column.
    space_below: i32,
    /// World-space Y of the surface in this column.
    surface_y: f64,
}

/// Pre-computed column context (one per XZ column).
#[derive(Debug, Clone, Copy)]
struct ColumnContext {
    surface_yi: i32,
    bottom_yi: i32,
    space_above: i32,
    space_below: i32,
}

// ── Material color/property tables ──────────────────────────────────

fn material_color(name: &str) -> String {
    let lower = name.to_lowercase();
    // Exact matches first
    let exact = match name {
        "Stone" | "Rock_Stone" => "#909090",
        "Rock_Granite" => "#9e8b7e",
        "Rock_Slate" => "#5c5c6e",
        "Rock_Limestone" => "#c4b99a",
        "Rock_Basalt" => "#3d3d3d",
        "Rock_Sandstone" => "#d2b48c",
        "Rock_Magma_Cooled" => "#4a2020",
        "Marble" => "#e0ddd5",
        "Quartzite" => "#d6cec0",
        "Cobblestone" => "#7a7a7a",
        "Dirt" | "Soil_Dirt" => "#a0724a",
        "Dirt_Dark" => "#6e4e30",
        "Soil_Mud" => "#6b4423",
        "Soil_Clay" => "#b87333",
        "Soil_Sand" => "#c2b280",
        "Soil_Gravel" | "Gravel" => "#a0a0a0",
        "Soil_Moss" => "#4a7a4a",
        "Tilled_Soil" => "#8a6035",
        "Sand" => "#d4c590",
        "Sand_White" => "#e8dfc8",
        "Grass" | "Soil_Grass" => "#5cb85c",
        "GrassDeep" => "#3e8a3e",
        "GrassDeepSunny" => "#6ec86e",
        "Snow" => "#e8e8f0",
        "Ice" => "#b0e0e6",
        "Wood" => "#8b6b4a",
        "Lightwoods" => "#c4a870",
        "Softwoods" => "#a68b5b",
        "Bedrock" => "#2a2a2a",
        "Water" => "#4169e1",
        "Fluid_Slime_Red" => "#c0392b",
        "Lava" | "Lava_Source" => "#ff4500",
        _ => "",
    };
    if !exact.is_empty() {
        return exact.to_string();
    }

    // Keyword matching on lowercase
    if lower.contains("lava") {
        return "#ff4500".to_string();
    }
    if lower.contains("stone") || lower.contains("rock") {
        return "#909090".to_string();
    }
    if lower.contains("grass") {
        return "#5cb85c".to_string();
    }
    if lower.contains("dirt") || lower.contains("soil") {
        return "#a0724a".to_string();
    }
    if lower.contains("sand") {
        return "#d4c590".to_string();
    }
    if lower.contains("snow") {
        return "#e8e8f0".to_string();
    }
    if lower.contains("ice") {
        return "#b0e0e6".to_string();
    }
    if lower.contains("clay") {
        return "#b87333".to_string();
    }
    if lower.contains("moss") {
        return "#4a7a4a".to_string();
    }
    if lower.contains("mud") {
        return "#6b4423".to_string();
    }
    if lower.contains("gravel") {
        return "#a0a0a0".to_string();
    }
    if lower.contains("water") {
        return "#4169e1".to_string();
    }
    if lower.contains("wood") {
        return "#8b6b4a".to_string();
    }

    "#808080".to_string()
}

struct MaterialPBR {
    roughness: f32,
    metalness: f32,
    emissive: String,
    emissive_intensity: f32,
}

fn material_pbr(name: &str) -> MaterialPBR {
    let lower = name.to_lowercase();
    let (r, m, e, ei) = match name {
        "Stone" | "Rock_Stone" | "Rock_Limestone" | "Rock_Basalt" => (0.9, 0.0, "", 0.0),
        "Rock_Granite" | "Rock_Slate" | "Rock_Sandstone" => (0.85, 0.0, "", 0.0),
        "Rock_Magma_Cooled" => (0.8, 0.0, "", 0.0),
        "Marble" => (0.4, 0.05, "", 0.0),
        "Quartzite" => (0.5, 0.0, "", 0.0),
        "Cobblestone" => (0.9, 0.0, "", 0.0),
        "Dirt" | "Dirt_Dark" | "Soil_Dirt" | "Soil_Gravel" | "Soil_Moss" | "Tilled_Soil" => {
            (0.9, 0.0, "", 0.0)
        }
        "Soil_Mud" => (0.95, 0.0, "", 0.0),
        "Soil_Clay" => (0.85, 0.0, "", 0.0),
        "Soil_Sand" => (0.8, 0.0, "", 0.0),
        "Sand" => (0.8, 0.0, "", 0.0),
        "Sand_White" => (0.75, 0.0, "", 0.0),
        "Grass" | "Soil_Grass" | "GrassDeep" | "GrassDeepSunny" => (0.7, 0.0, "", 0.0),
        "Snow" => (0.6, 0.0, "", 0.0),
        "Ice" => (0.2, 0.1, "", 0.0),
        "Wood" | "Softwoods" => (0.7, 0.0, "", 0.0),
        "Lightwoods" => (0.65, 0.0, "", 0.0),
        "Bedrock" => (0.95, 0.0, "", 0.0),
        "Water" => (0.1, 0.0, "", 0.0),
        "Lava" | "Lava_Source" => (0.3, 0.0, "#ff4500", 2.0),
        "Fluid_Slime_Red" => (0.4, 0.0, "#c0392b", 0.8),
        _ => {
            // Keyword fallback
            if lower.contains("lava") {
                (0.3, 0.0, "#ff4500", 2.0)
            } else if lower.contains("ice") {
                (0.2, 0.1, "", 0.0)
            } else if lower.contains("stone") || lower.contains("rock") {
                (0.9, 0.0, "", 0.0)
            } else if lower.contains("grass") {
                (0.7, 0.0, "", 0.0)
            } else if lower.contains("sand") {
                (0.8, 0.0, "", 0.0)
            } else if lower.contains("snow") {
                (0.6, 0.0, "", 0.0)
            } else if lower.contains("dirt") || lower.contains("soil") {
                (0.9, 0.0, "", 0.0)
            } else {
                (0.8, 0.0, "", 0.0)
            }
        }
    };
    MaterialPBR {
        roughness: r,
        metalness: m,
        emissive: if e.is_empty() {
            "#000000".to_string()
        } else {
            e.to_string()
        },
        emissive_intensity: ei,
    }
}

// ── Material type set ───────────────────────────────────────────────

fn is_material_type(ty: &str) -> bool {
    matches!(
        ty,
        "Material:Constant"
            | "Material:Solid"
            | "Material:Empty"
            | "Material:Imported"
            | "Material:Queue"
            | "Material:FieldFunction"
            | "Material:Striped"
            | "Material:DownwardDepth"
            | "Material:UpwardDepth"
            | "Material:DownwardSpace"
            | "Material:UpwardSpace"
            | "Material:Surface"
            | "Material:Cave"
            | "Material:Cluster"
            | "Material:Exported"
            | "Material:Solidity"
            | "Material:TerrainDensity"
            | "Material:SpaceAndDepth"
            | "Material:WeightedRandom"
            | "Material:HeightGradient"
            | "Material:NoiseSelector"
            | "Material:NoiseSelectorMaterial"
            | "Material:Conditional"
            | "Material:Blend"
            | "Material:ConstantThickness"
            | "Material:NoiseThickness"
            | "Material:RangeThickness"
            | "Material:WeightedThickness"
    )
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Get the density type from a node's data.
fn get_node_type(node: &MaterialNode) -> &str {
    &node.node_type
}

/// Hash a seed value (string or number) into an integer.
fn hash_seed(seed: &Value) -> i64 {
    match seed {
        Value::Number(n) => n.as_i64().unwrap_or(0),
        Value::String(s) => {
            let mut h: i32 = 0;
            for ch in s.chars() {
                h = h.wrapping_mul(31).wrapping_add(ch as i32);
            }
            h as i64
        }
        _ => 0,
    }
}

/// Deterministic position hash → [0, 1).
fn position_hash(x: f64, y: f64, z: f64, seed: i64) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let iz = z.floor() as i64;
    let h = ((ix
        .wrapping_mul(HASH_PRIME_A)
        .wrapping_add(iy.wrapping_mul(HASH_PRIME_B))
        .wrapping_add(iz.wrapping_mul(HASH_PRIME_C)))
        ^ seed) as u32;
    h as f64 / 4294967296.0
}

/// Parse a material name from a field value.
fn parse_material_name(value: &Value) -> Option<String> {
    match value {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Object(obj) => {
            if let Some(Value::String(s)) = obj.get("Solid") {
                if !s.is_empty() {
                    return Some(s.clone());
                }
            }
            if let Some(Value::String(s)) = obj.get("Material") {
                if !s.is_empty() {
                    return Some(s.clone());
                }
            }
            None
        }
        _ => None,
    }
}

/// Get a field value as f64.
fn field_f64(fields: &HashMap<String, Value>, key: &str, default: f64) -> f64 {
    fields.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Get a field value as i64.
#[allow(dead_code)]
fn field_i64(fields: &HashMap<String, Value>, key: &str, default: i64) -> i64 {
    fields.get(key).and_then(|v| v.as_i64()).unwrap_or(default)
}

// ── Simplified node representation ──────────────────────────────────

/// Simplified material node for evaluation.
struct MaterialNode {
    id: String,
    node_type: String,
    fields: HashMap<String, Value>,
}

impl MaterialNode {
    fn from_raw(id: &str, raw_type: &str, data: &Value) -> Self {
        let node_type = if raw_type.starts_with("Material:") {
            // React Flow type already has the Material: prefix — use directly
            raw_type.to_string()
        } else if !raw_type.is_empty() {
            // Non-empty raw_type that doesn't start with "Material:" — this is
            // a density node (e.g. "Constant", "Sum"). Keep it as-is; don't
            // reinterpret it as a material type even if the name collides.
            raw_type.to_string()
        } else {
            // raw_type is empty — fall back to data.type
            let dt = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if dt.starts_with("Material:") {
                dt.to_string()
            } else if !dt.is_empty() && is_material_type(&format!("Material:{}", dt)) {
                format!("Material:{}", dt)
            } else {
                dt.to_string()
            }
        };

        let fields = data
            .get("fields")
            .and_then(|v| v.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        MaterialNode {
            id: id.to_string(),
            node_type,
            fields,
        }
    }
}

// ── Column context pre-computation ──────────────────────────────────

fn compute_column_contexts(densities: &[f32], n: usize, ys: usize) -> Vec<Vec<ColumnContext>> {
    let mut columns = Vec::with_capacity(n);
    for z in 0..n {
        let mut row = Vec::with_capacity(n);
        for x in 0..n {
            let mut surface_yi: i32 = -1;
            let mut bottom_yi: i32 = -1;

            // Scan top-down for surface (first solid with air above)
            for y in (0..ys).rev() {
                let idx = y * n * n + z * n + x;
                if densities[idx] >= SOLID_THRESHOLD {
                    if y == ys - 1 || densities[(y + 1) * n * n + z * n + x] < SOLID_THRESHOLD {
                        surface_yi = y as i32;
                        break;
                    }
                }
            }

            // Scan bottom-up for lowest contiguous solid from surface
            if surface_yi >= 0 {
                bottom_yi = surface_yi;
                for y in (0..surface_yi as usize).rev() {
                    let idx = y * n * n + z * n + x;
                    if densities[idx] >= SOLID_THRESHOLD {
                        bottom_yi = y as i32;
                    } else {
                        break;
                    }
                }
            }

            // Count air above surface
            let space_above = if surface_yi >= 0 {
                let mut count = 0i32;
                for y in (surface_yi as usize + 1)..ys {
                    let idx = y * n * n + z * n + x;
                    if densities[idx] < SOLID_THRESHOLD {
                        count += 1;
                    } else {
                        break;
                    }
                }
                count
            } else {
                ys as i32
            };

            // Count air below bottom solid
            let space_below = if bottom_yi > 0 {
                let mut count = 0i32;
                for y in (0..bottom_yi as usize).rev() {
                    let idx = y * n * n + z * n + x;
                    if densities[idx] < SOLID_THRESHOLD {
                        count += 1;
                    } else {
                        break;
                    }
                }
                count
            } else {
                0
            };

            row.push(ColumnContext {
                surface_yi,
                bottom_yi,
                space_above,
                space_below,
            });
        }
        columns.push(row);
    }
    columns
}

// ── Root finding ────────────────────────────────────────────────────

fn find_material_root<'a>(
    nodes: &'a [MaterialNode],
    edge_sources_with_material_target: &HashSet<String>,
) -> Option<&'a MaterialNode> {
    let material_nodes: Vec<&MaterialNode> = nodes
        .iter()
        .filter(|n| is_material_type(&n.node_type))
        .collect();

    if material_nodes.is_empty() {
        return None;
    }

    // Terminal: material node that is not a source feeding into another material node
    let terminals: Vec<&&MaterialNode> = material_nodes
        .iter()
        .filter(|n| !edge_sources_with_material_target.contains(&n.id))
        .collect();

    // Prefer non-layer, non-exported terminal nodes
    let preferred: Vec<&&&MaterialNode> = terminals
        .iter()
        .filter(|n| !n.node_type.contains("Thickness") && n.node_type != "Material:Exported")
        .collect();

    if let Some(n) = preferred.first() {
        return Some(n);
    }
    if let Some(n) = terminals.first() {
        return Some(n);
    }
    material_nodes.first().copied()
}

// ── Condition evaluation ────────────────────────────────────────────

fn get_context_value(param: &str, ctx: &MaterialVoxelContext) -> f64 {
    match param {
        "SPACE_ABOVE_FLOOR" => ctx.space_above as f64,
        "SPACE_BELOW_CEILING" => ctx.space_below as f64,
        _ => 0.0,
    }
}

fn evaluate_condition(condition: &Value, ctx: &MaterialVoxelContext) -> bool {
    let obj = match condition.as_object() {
        Some(o) => o,
        None => return true,
    };

    let cond_type = obj.get("Type").and_then(|v| v.as_str()).unwrap_or("");

    match cond_type {
        "AlwaysTrueCondition" => true,

        "SmallerThanCondition" => {
            let param = obj
                .get("ContextToCheck")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let threshold = obj.get("Threshold").and_then(|v| v.as_f64()).unwrap_or(0.0);
            get_context_value(param, ctx) < threshold
        }

        "GreaterThanCondition" => {
            let param = obj
                .get("ContextToCheck")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let threshold = obj.get("Threshold").and_then(|v| v.as_f64()).unwrap_or(0.0);
            get_context_value(param, ctx) > threshold
        }

        "EqualsCondition" => {
            let param = obj
                .get("ContextToCheck")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let value = obj.get("Value").and_then(|v| v.as_f64()).unwrap_or(0.0);
            (get_context_value(param, ctx) - value).abs() < f64::EPSILON
        }

        "AndCondition" => {
            if let Some(Value::Array(conditions)) = obj.get("Conditions") {
                conditions.iter().all(|sub| evaluate_condition(sub, ctx))
            } else {
                true
            }
        }

        "OrCondition" => {
            if let Some(Value::Array(conditions)) = obj.get("Conditions") {
                conditions.iter().any(|sub| evaluate_condition(sub, ctx))
            } else {
                false
            }
        }

        "NotCondition" => {
            if let Some(inner) = obj.get("Condition") {
                !evaluate_condition(inner, ctx)
            } else {
                true
            }
        }

        _ => true,
    }
}

// ── Material graph evaluator ────────────────────────────────────────

/// Evaluate the material node graph for a 3D density volume.
///
/// Returns `None` if no material graph is found.
///
/// # Arguments
/// * `raw_nodes` - React Flow nodes (JSON values).
/// * `raw_edges` - React Flow edges (JSON values).
/// * `densities` - Y-major density volume: `densities[y * n * n + z * n + x]`.
/// * `resolution` - XZ grid resolution (n).
/// * `y_slices` - Number of Y slices.
/// * `range_min` / `range_max` - World-space XZ range.
/// * `y_min` / `y_max` - World-space Y range.
/// * `density_graph` - Optional density graph for nodes that need density evaluation
///   (e.g., FieldFunction, Conditional, Blend).
/// * `content_fields` - Content fields for density evaluation.
pub fn evaluate_material_graph(
    raw_nodes: &[Value],
    raw_edges: &[Value],
    densities: &[f32],
    resolution: u32,
    y_slices: u32,
    range_min: f64,
    range_max: f64,
    y_min: f64,
    y_max: f64,
    density_graph: Option<&EvalGraph>,
    content_fields: &HashMap<String, f64>,
) -> Option<MaterialResult> {
    let n = resolution as usize;
    let ys = y_slices as usize;
    let total_size = n * n * ys;

    // Parse nodes and edges
    let mut mat_nodes: Vec<MaterialNode> = Vec::new();
    for raw in raw_nodes {
        let id = raw.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let raw_type = raw.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let data = raw
            .get("data")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        if !id.is_empty() {
            mat_nodes.push(MaterialNode::from_raw(id, raw_type, &data));
        }
    }

    // Build edge lookup: target_id → Map<handle, source_node_id>
    let mut input_edges: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut sources_with_material_target: HashSet<String> = HashSet::new();

    for raw in raw_edges {
        let source = raw
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let target = raw
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let handle = raw
            .get("targetHandle")
            .and_then(|v| v.as_str())
            .unwrap_or("Input")
            .to_string();

        if !source.is_empty() && !target.is_empty() {
            // Track which sources feed into material nodes
            if mat_nodes
                .iter()
                .any(|n| n.id == target && is_material_type(&n.node_type))
            {
                sources_with_material_target.insert(source.clone());
            }

            input_edges
                .entry(target)
                .or_default()
                .insert(handle, source);
        }
    }

    // Find root material node
    let root = find_material_root(&mat_nodes, &sources_with_material_target)?;
    let root_id = root.id.clone();

    // Build node-by-id lookup
    let node_by_id: HashMap<String, &MaterialNode> =
        mat_nodes.iter().map(|n| (n.id.clone(), n)).collect();

    // Pre-compute column contexts
    let columns = compute_column_contexts(densities, n, ys);

    // World coordinate mapping
    let step_xz = (range_max - range_min) / n as f64;
    let step_y = if ys > 1 {
        (y_max - y_min) / (ys as f64 - 1.0)
    } else {
        0.0
    };

    // Build palette dynamically
    let mut palette: Vec<MaterialEntry> = Vec::new();
    let mut name_to_index: HashMap<String, u8> = HashMap::new();

    // Ensure at least one fallback material
    add_material("Stone", &mut palette, &mut name_to_index);

    let mut material_ids = vec![0u8; total_size];

    // Evaluate each solid voxel
    for z in 0..n {
        for x in 0..n {
            let col = &columns[z][x];
            if col.surface_yi < 0 {
                continue; // all-air column
            }

            for y in 0..ys {
                let idx = y * n * n + z * n + x;
                let density = densities[idx];
                if density < SOLID_THRESHOLD {
                    continue; // air
                }

                let wx = range_min + (x as f64 + 0.5) * step_xz;
                let wy = if ys > 1 {
                    y_min + y as f64 * step_y
                } else {
                    y_min
                };
                let wz = range_min + (z as f64 + 0.5) * step_xz;

                let voxel_ctx = MaterialVoxelContext {
                    x: wx,
                    y: wy,
                    z: wz,
                    xi: x,
                    yi: y,
                    zi: z,
                    is_solid: true,
                    density,
                    downward_depth: (col.surface_yi - y as i32).max(0),
                    upward_depth: (y as i32 - col.bottom_yi).max(0),
                    space_above: col.space_above,
                    space_below: col.space_below,
                    surface_y: y_min + col.surface_yi as f64 * step_y,
                };

                let mut visiting = HashSet::new();
                let mat_name = evaluate_material_node(
                    &root_id,
                    &voxel_ctx,
                    &node_by_id,
                    &input_edges,
                    &mut visiting,
                    density_graph,
                    content_fields,
                    y_min,
                    y_max,
                );

                if let Some(name) = mat_name {
                    let mat_idx = add_material(&name, &mut palette, &mut name_to_index);
                    material_ids[idx] = mat_idx;
                }
                // else stays 0 (Stone fallback)
            }
        }
    }

    Some(MaterialResult {
        material_ids,
        palette,
    })
}

/// Add a material to the palette (or return its existing index).
fn add_material(
    name: &str,
    palette: &mut Vec<MaterialEntry>,
    name_to_index: &mut HashMap<String, u8>,
) -> u8 {
    if let Some(&idx) = name_to_index.get(name) {
        return idx;
    }
    let idx = palette.len() as u8;
    let pbr = material_pbr(name);
    palette.push(MaterialEntry {
        name: name.to_string(),
        color: material_color(name),
        roughness: pbr.roughness,
        metalness: pbr.metalness,
        emissive: pbr.emissive,
        emissive_intensity: pbr.emissive_intensity,
    });
    name_to_index.insert(name.to_string(), idx);
    idx
}

/// Get material input ID by trying multiple handle names.
fn get_material_input<'a>(
    inputs: &'a HashMap<String, String>,
    handles: &[&str],
) -> Option<&'a String> {
    for handle in handles {
        if let Some(src) = inputs.get(*handle) {
            return Some(src);
        }
    }
    None
}

/// Recursively evaluate a material node.
fn evaluate_material_node(
    node_id: &str,
    ctx: &MaterialVoxelContext,
    node_by_id: &HashMap<String, &MaterialNode>,
    input_edges: &HashMap<String, HashMap<String, String>>,
    visiting: &mut HashSet<String>,
    density_graph: Option<&EvalGraph>,
    content_fields: &HashMap<String, f64>,
    y_min: f64,
    y_max: f64,
) -> Option<String> {
    // Cycle detection
    if visiting.contains(node_id) {
        return None;
    }
    visiting.insert(node_id.to_string());

    let node = match node_by_id.get(node_id) {
        Some(n) => n,
        None => {
            visiting.remove(node_id);
            return None;
        }
    };

    let node_type = get_node_type(node);
    let fields = &node.fields;
    let empty_map = HashMap::new();
    let inputs = input_edges.get(node_id).unwrap_or(&empty_map);

    let result = match node_type {
        // ── Leaf nodes ──────────────────────────────────────────────
        "Material:Constant" | "Material:Solid" => {
            let name = fields
                .get("Material")
                .and_then(|v| parse_material_name(v))
                .unwrap_or_else(|| "Stone".to_string());
            Some(name)
        }

        "Material:Empty" => None,

        "Material:Imported" => fields.get("Material").and_then(|v| parse_material_name(v)),

        "Material:Solidity" | "Material:TerrainDensity" => None,

        // ── Filter nodes (conditional pass-through) ─────────────────
        "Material:DownwardDepth" => {
            let max_depth = field_f64(fields, "MaxDepth", field_f64(fields, "Depth", 1.0));
            if (ctx.downward_depth as f64) <= max_depth {
                let input_id = get_material_input(inputs, &["Input"]);
                input_id.and_then(|id| {
                    evaluate_material_node(
                        id,
                        ctx,
                        node_by_id,
                        input_edges,
                        visiting,
                        density_graph,
                        content_fields,
                        y_min,
                        y_max,
                    )
                })
            } else {
                None
            }
        }

        "Material:UpwardDepth" => {
            let max_depth = field_f64(fields, "MaxDepth", field_f64(fields, "Depth", 1.0));
            if (ctx.upward_depth as f64) <= max_depth {
                let input_id = get_material_input(inputs, &["Input"]);
                input_id.and_then(|id| {
                    evaluate_material_node(
                        id,
                        ctx,
                        node_by_id,
                        input_edges,
                        visiting,
                        density_graph,
                        content_fields,
                        y_min,
                        y_max,
                    )
                })
            } else {
                None
            }
        }

        "Material:DownwardSpace" => {
            let max_space = field_f64(fields, "MaxSpace", field_f64(fields, "Space", 1.0));
            if (ctx.space_below as f64) <= max_space {
                let input_id = get_material_input(inputs, &["Input"]);
                input_id.and_then(|id| {
                    evaluate_material_node(
                        id,
                        ctx,
                        node_by_id,
                        input_edges,
                        visiting,
                        density_graph,
                        content_fields,
                        y_min,
                        y_max,
                    )
                })
            } else {
                None
            }
        }

        "Material:UpwardSpace" => {
            let max_space = field_f64(fields, "MaxSpace", field_f64(fields, "Space", 1.0));
            if (ctx.space_above as f64) <= max_space {
                let input_id = get_material_input(inputs, &["Input"]);
                input_id.and_then(|id| {
                    evaluate_material_node(
                        id,
                        ctx,
                        node_by_id,
                        input_edges,
                        visiting,
                        density_graph,
                        content_fields,
                        y_min,
                        y_max,
                    )
                })
            } else {
                None
            }
        }

        "Material:Surface" | "Material:Cave" | "Material:Cluster" | "Material:Exported" => {
            let input_id = get_material_input(inputs, &["Input"]);
            input_id.and_then(|id| {
                evaluate_material_node(
                    id,
                    ctx,
                    node_by_id,
                    input_edges,
                    visiting,
                    density_graph,
                    content_fields,
                    y_min,
                    y_max,
                )
            })
        }

        // ── Combinator nodes ────────────────────────────────────────
        "Material:Queue" => {
            let mut result = None;
            for i in 0..16 {
                let handles = [format!("Queue[{}]", i), format!("Entries[{}]", i)];
                let handle_refs: Vec<&str> = handles.iter().map(|s| s.as_str()).collect();
                let entry_id = get_material_input(inputs, &handle_refs);
                match entry_id {
                    None => break,
                    Some(id) => {
                        let mat = evaluate_material_node(
                            id,
                            ctx,
                            node_by_id,
                            input_edges,
                            visiting,
                            density_graph,
                            content_fields,
                            y_min,
                            y_max,
                        );
                        if mat.is_some() {
                            result = mat;
                            break;
                        }
                    }
                }
            }
            result
        }

        "Material:FieldFunction" => {
            let density_val = if let Some(dg) = density_graph {
                if let Some(density_input_id) = inputs.get("FieldFunction") {
                    let mut eval_ctx = EvalContext::new(dg, content_fields.clone());
                    eval_ctx.clear_memo();
                    evaluate(&mut eval_ctx, density_input_id, ctx.x, ctx.y, ctx.z)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let mat_idx = if density_val >= 0.5 { 1 } else { 0 };
            let handle = format!("Materials[{}]", mat_idx);
            let mat_id = get_material_input(inputs, &[handle.as_str()]);
            mat_id.and_then(|id| {
                evaluate_material_node(
                    id,
                    ctx,
                    node_by_id,
                    input_edges,
                    visiting,
                    density_graph,
                    content_fields,
                    y_min,
                    y_max,
                )
            })
        }

        "Material:Striped" => {
            let seed = fields.get("Seed").map(|v| hash_seed(v)).unwrap_or(0);
            let thickness = field_f64(fields, "Thickness", 1.0).max(1.0) as i64;
            let stripe_index = ((ctx.y as i64 + seed) / thickness) % 2;
            let idx = if stripe_index < 0 {
                stripe_index + 2
            } else {
                stripe_index
            };
            let handle = format!("Materials[{}]", idx);
            let mat_id = get_material_input(inputs, &[handle.as_str()]);
            mat_id.and_then(|id| {
                evaluate_material_node(
                    id,
                    ctx,
                    node_by_id,
                    input_edges,
                    visiting,
                    density_graph,
                    content_fields,
                    y_min,
                    y_max,
                )
            })
        }

        "Material:Conditional" => {
            let cond_val = if let Some(dg) = density_graph {
                if let Some(cond_id) = inputs.get("Condition") {
                    let mut eval_ctx = EvalContext::new(dg, content_fields.clone());
                    eval_ctx.clear_memo();
                    evaluate(&mut eval_ctx, cond_id, ctx.x, ctx.y, ctx.z)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let threshold = field_f64(fields, "Threshold", 0.5);
            let selected_handle = if cond_val >= threshold {
                "TrueInput"
            } else {
                "FalseInput"
            };
            let selected_id = get_material_input(inputs, &[selected_handle]);
            selected_id.and_then(|id| {
                evaluate_material_node(
                    id,
                    ctx,
                    node_by_id,
                    input_edges,
                    visiting,
                    density_graph,
                    content_fields,
                    y_min,
                    y_max,
                )
            })
        }

        "Material:Blend" => {
            let factor_val = if let Some(dg) = density_graph {
                if let Some(factor_id) = inputs.get("Factor") {
                    let mut eval_ctx = EvalContext::new(dg, content_fields.clone());
                    eval_ctx.clear_memo();
                    evaluate(&mut eval_ctx, factor_id, ctx.x, ctx.y, ctx.z)
                } else {
                    0.5
                }
            } else {
                0.5
            };

            let selected_handle = if factor_val >= 0.5 {
                "InputB"
            } else {
                "InputA"
            };
            let selected_id = get_material_input(inputs, &[selected_handle]);
            selected_id.and_then(|id| {
                evaluate_material_node(
                    id,
                    ctx,
                    node_by_id,
                    input_edges,
                    visiting,
                    density_graph,
                    content_fields,
                    y_min,
                    y_max,
                )
            })
        }

        "Material:HeightGradient" => {
            let (range_min_y, range_max_y) = if let Some(range) = fields.get("Range") {
                let r_min = range.get("Min").and_then(|v| v.as_f64()).unwrap_or(y_min);
                let r_max = range.get("Max").and_then(|v| v.as_f64()).unwrap_or(y_max);
                (r_min, r_max)
            } else {
                (y_min, y_max)
            };

            let mid = (range_min_y + range_max_y) / 2.0;
            let selected_handle = if ctx.y >= mid { "High" } else { "Low" };
            let selected_id = get_material_input(inputs, &[selected_handle]);
            selected_id.and_then(|id| {
                evaluate_material_node(
                    id,
                    ctx,
                    node_by_id,
                    input_edges,
                    visiting,
                    density_graph,
                    content_fields,
                    y_min,
                    y_max,
                )
            })
        }

        "Material:WeightedRandom" => {
            let h = position_hash(ctx.x, ctx.y, ctx.z, 0);
            let mut count = 0usize;
            for i in 0..16 {
                let handle = format!("Entries[{}]", i);
                if inputs.contains_key(handle.as_str()) {
                    count += 1;
                } else {
                    break;
                }
            }
            if count > 0 {
                let selected = (h * count as f64).floor() as usize;
                let handle = format!("Entries[{}]", selected);
                let entry_id = get_material_input(inputs, &[handle.as_str()]);
                entry_id.and_then(|id| {
                    evaluate_material_node(
                        id,
                        ctx,
                        node_by_id,
                        input_edges,
                        visiting,
                        density_graph,
                        content_fields,
                        y_min,
                        y_max,
                    )
                })
            } else {
                None
            }
        }

        "Material:NoiseSelector" | "Material:NoiseSelectorMaterial" => {
            let mut count = 0usize;
            for i in 0..16 {
                let handle = format!("Inputs[{}]", i);
                if inputs.contains_key(handle.as_str()) {
                    count += 1;
                } else {
                    break;
                }
            }
            if count > 0 {
                let h = position_hash(ctx.x, ctx.y, ctx.z, 42);
                let selected = (h * count as f64).floor() as usize;
                let handle = format!("Inputs[{}]", selected);
                let selected_id = get_material_input(inputs, &[handle.as_str()]);
                selected_id.and_then(|id| {
                    evaluate_material_node(
                        id,
                        ctx,
                        node_by_id,
                        input_edges,
                        visiting,
                        density_graph,
                        content_fields,
                        y_min,
                        y_max,
                    )
                })
            } else {
                None
            }
        }

        // ── SpaceAndDepth V2 (layer accumulation) ───────────────────
        "Material:SpaceAndDepth" => {
            // Check condition
            if let Some(condition) = fields.get("Condition") {
                if !evaluate_condition(condition, ctx) {
                    visiting.remove(node_id);
                    return None;
                }
            }

            let layer_context = fields
                .get("LayerContext")
                .and_then(|v| v.as_str())
                .unwrap_or("DEPTH_INTO_FLOOR");
            let is_floor = layer_context == "DEPTH_INTO_FLOOR";
            let depth = if is_floor {
                ctx.downward_depth
            } else {
                ctx.upward_depth
            };

            let mut accumulated = 0i32;
            let mut result = None;

            for i in 0..16 {
                let handles = [format!("Layers[{}]", i)];
                let handle_refs: Vec<&str> = handles.iter().map(|s| s.as_str()).collect();
                let layer_id = match get_material_input(inputs, &handle_refs) {
                    Some(id) => id.clone(),
                    None => break,
                };

                let layer_node = match node_by_id.get(&layer_id) {
                    Some(n) => n,
                    None => continue,
                };

                let layer_type = get_node_type(layer_node);
                let layer_fields = &layer_node.fields;
                let layer_inputs = input_edges.get(&layer_id);
                let empty_inputs = HashMap::new();
                let layer_inp = layer_inputs.unwrap_or(&empty_inputs);

                let thickness = match layer_type {
                    "Material:ConstantThickness" => {
                        field_f64(layer_fields, "Thickness", 1.0).max(1.0) as i32
                    }

                    "Material:RangeThickness" => {
                        let r_min = field_f64(layer_fields, "RangeMin", 1.0);
                        let r_max = field_f64(layer_fields, "RangeMax", 3.0);
                        let seed = layer_fields.get("Seed").map(|v| hash_seed(v)).unwrap_or(0);
                        let h = position_hash(ctx.x, 0.0, ctx.z, seed);
                        (r_min + h * (r_max - r_min)).round() as i32
                    }

                    "Material:WeightedThickness" => {
                        if let Some(Value::Array(entries)) = layer_fields.get("PossibleThicknesses")
                        {
                            let seed = layer_fields.get("Seed").map(|v| hash_seed(v)).unwrap_or(0);
                            let h = position_hash(ctx.x, 0.0, ctx.z, seed);

                            let mut total_w = 0.0f64;
                            for entry in entries {
                                total_w +=
                                    entry.get("Weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            }

                            let t = h * total_w;
                            let mut cum = 0.0f64;
                            let mut chosen = 1i32;
                            for entry in entries {
                                cum += entry.get("Weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                                if t < cum {
                                    chosen = entry
                                        .get("Thickness")
                                        .and_then(|v| v.as_i64())
                                        .unwrap_or(1)
                                        as i32;
                                    break;
                                }
                            }
                            chosen
                        } else {
                            1
                        }
                    }

                    "Material:NoiseThickness" => {
                        if let Some(dg) = density_graph {
                            if let Some(noise_id) = layer_inp.get("ThicknessFunctionXZ") {
                                let mut eval_ctx = EvalContext::new(dg, content_fields.clone());
                                eval_ctx.clear_memo();
                                let noise_val =
                                    evaluate(&mut eval_ctx, noise_id, ctx.x, 0.0, ctx.z);
                                noise_val.abs().round().max(1.0) as i32
                            } else {
                                1
                            }
                        } else {
                            1
                        }
                    }

                    _ => 1,
                };

                let prev_accumulated = accumulated;
                accumulated += thickness;

                // Check if this voxel's depth falls within this layer
                if depth >= prev_accumulated && depth < accumulated {
                    let mat_id = get_material_input(layer_inp, &["Material"]);
                    if let Some(id) = mat_id {
                        result = evaluate_material_node(
                            id,
                            ctx,
                            node_by_id,
                            input_edges,
                            visiting,
                            density_graph,
                            content_fields,
                            y_min,
                            y_max,
                        );
                    }
                    break;
                }
            }
            result
        }

        // ── Layer sub-types (when evaluated directly) ───────────────
        "Material:ConstantThickness"
        | "Material:RangeThickness"
        | "Material:WeightedThickness"
        | "Material:NoiseThickness" => {
            let mat_id = get_material_input(inputs, &["Material"]);
            mat_id.and_then(|id| {
                evaluate_material_node(
                    id,
                    ctx,
                    node_by_id,
                    input_edges,
                    visiting,
                    density_graph,
                    content_fields,
                    y_min,
                    y_max,
                )
            })
        }

        // ── Unknown type — try Input handle ─────────────────────────
        _ => {
            let fallback_id = get_material_input(inputs, &["Input", "Inputs[0]"]);
            fallback_id.and_then(|id| {
                evaluate_material_node(
                    id,
                    ctx,
                    node_by_id,
                    input_edges,
                    visiting,
                    density_graph,
                    content_fields,
                    y_min,
                    y_max,
                )
            })
        }
    };

    visiting.remove(node_id);
    result
}

// ── Tauri command request/response types ─────────────────────────────

/// Combined voxel preview request (density + material evaluation).
#[derive(Deserialize)]
pub struct VoxelPreviewRequest {
    pub nodes: Vec<Value>,
    pub edges: Vec<Value>,
    pub resolution: u32,
    pub range_min: f64,
    pub range_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub y_slices: u32,
    pub root_node_id: Option<String>,
    pub content_fields: Option<HashMap<String, f64>>,
}

/// Combined voxel preview response (density + optional material).
#[derive(Serialize)]
pub struct VoxelPreviewResult {
    pub densities: Vec<f32>,
    pub resolution: u32,
    pub y_slices: u32,
    pub min_value: f32,
    pub max_value: f32,
    /// Per-voxel material IDs (if material nodes present), otherwise None.
    pub material_ids: Option<Vec<u8>>,
    /// Material palette (if material nodes present), otherwise None.
    pub palette: Option<Vec<MaterialEntry>>,
}

/// Evaluate both density volume and material graph in one call.
///
/// 1. Evaluates the density volume (reuses `eval::volume`).
/// 2. If material nodes are present, evaluates the material graph.
/// 3. Returns the combined result.
pub fn evaluate_voxel_preview(
    request: &VoxelPreviewRequest,
    density_graph: &EvalGraph,
    content_fields: &HashMap<String, f64>,
) -> VoxelPreviewResult {
    // 1. Evaluate density volume
    let volume = crate::eval::volume::evaluate_volume(
        density_graph,
        request.resolution,
        request.range_min,
        request.range_max,
        request.y_min,
        request.y_max,
        request.y_slices,
        content_fields,
    );

    // 2. Check if any material nodes exist
    let has_material_nodes = request.nodes.iter().any(|n| {
        n.get("type")
            .and_then(|v| v.as_str())
            .map(|t| t.starts_with("Material:"))
            .unwrap_or(false)
    });

    let (material_ids, palette) = if has_material_nodes {
        match evaluate_material_graph(
            &request.nodes,
            &request.edges,
            &volume.densities,
            request.resolution,
            request.y_slices,
            request.range_min,
            request.range_max,
            request.y_min,
            request.y_max,
            Some(density_graph),
            content_fields,
        ) {
            Some(mat_result) => (Some(mat_result.material_ids), Some(mat_result.palette)),
            None => (None, None),
        }
    } else {
        (None, None)
    };

    VoxelPreviewResult {
        densities: volume.densities,
        resolution: volume.resolution,
        y_slices: volume.y_slices,
        min_value: volume.min_value,
        max_value: volume.max_value,
        material_ids,
        palette,
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_mat_node(id: &str, mat_type: &str, fields: Value) -> Value {
        json!({
            "id": id,
            "type": mat_type,
            "data": {
                "type": mat_type.strip_prefix("Material:").unwrap_or(mat_type),
                "fields": fields
            }
        })
    }

    fn make_edge(source: &str, target: &str, handle: &str) -> Value {
        json!({
            "source": source,
            "target": target,
            "targetHandle": handle
        })
    }

    fn solid_volume(n: usize, ys: usize) -> Vec<f32> {
        // All solid (density = 1.0)
        vec![1.0f32; n * n * ys]
    }

    fn half_solid_volume(n: usize, ys: usize) -> Vec<f32> {
        // Bottom half solid, top half air (air must be negative)
        let mut densities = vec![-1.0f32; n * n * ys];
        let half_y = ys / 2;
        for y in 0..half_y {
            for z in 0..n {
                for x in 0..n {
                    densities[y * n * n + z * n + x] = 1.0;
                }
            }
        }
        densities
    }

    #[test]
    fn no_material_nodes_returns_none() {
        let nodes = vec![json!({
            "id": "const1",
            "type": "Constant",
            "data": { "type": "Constant", "fields": { "Value": 1.0 } }
        })];
        let edges: Vec<Value> = vec![];
        let densities = solid_volume(4, 4);

        let result = evaluate_material_graph(
            &nodes,
            &edges,
            &densities,
            4,
            4,
            -10.0,
            10.0,
            0.0,
            100.0,
            None,
            &HashMap::new(),
        );

        assert!(result.is_none());
    }

    #[test]
    fn constant_material_assigns_to_all_solid() {
        let nodes = vec![make_mat_node(
            "m1",
            "Material:Constant",
            json!({ "Material": "Grass" }),
        )];
        let edges: Vec<Value> = vec![];
        let densities = solid_volume(2, 2);

        let result = evaluate_material_graph(
            &nodes,
            &edges,
            &densities,
            2,
            2,
            -10.0,
            10.0,
            0.0,
            10.0,
            None,
            &HashMap::new(),
        );

        let mat = result.unwrap();
        // Should have Stone (fallback at 0) and Grass (at 1)
        assert!(mat.palette.len() >= 2);
        assert_eq!(mat.palette[0].name, "Stone");
        assert_eq!(mat.palette[1].name, "Grass");

        // All solid voxels should be Grass (index 1)
        for &id in &mat.material_ids {
            assert_eq!(id, 1, "Expected Grass (1), got {}", id);
        }
    }

    #[test]
    fn empty_material_keeps_stone_fallback() {
        let nodes = vec![make_mat_node("m1", "Material:Empty", json!({}))];
        let edges: Vec<Value> = vec![];
        let densities = solid_volume(2, 2);

        let result = evaluate_material_graph(
            &nodes,
            &edges,
            &densities,
            2,
            2,
            -10.0,
            10.0,
            0.0,
            10.0,
            None,
            &HashMap::new(),
        );

        let mat = result.unwrap();
        // All should be Stone fallback (index 0)
        for &id in &mat.material_ids {
            assert_eq!(id, 0);
        }
    }

    #[test]
    fn queue_first_non_null_wins() {
        let nodes = vec![
            make_mat_node("q", "Material:Queue", json!({})),
            make_mat_node("empty", "Material:Empty", json!({})),
            make_mat_node("grass", "Material:Constant", json!({ "Material": "Grass" })),
            make_mat_node("dirt", "Material:Constant", json!({ "Material": "Dirt" })),
        ];
        let edges = vec![
            make_edge("empty", "q", "Queue[0]"),
            make_edge("grass", "q", "Queue[1]"),
            make_edge("dirt", "q", "Queue[2]"),
        ];
        let densities = solid_volume(2, 2);

        let result = evaluate_material_graph(
            &nodes,
            &edges,
            &densities,
            2,
            2,
            -10.0,
            10.0,
            0.0,
            10.0,
            None,
            &HashMap::new(),
        );

        let mat = result.unwrap();
        // Empty returns None, so Queue should pick Grass (second entry)
        let grass_idx = mat.palette.iter().position(|p| p.name == "Grass").unwrap();
        for &id in &mat.material_ids {
            assert_eq!(id, grass_idx as u8);
        }
    }

    #[test]
    fn downward_depth_filter() {
        let nodes = vec![
            make_mat_node("q", "Material:Queue", json!({})),
            make_mat_node(
                "depth_filter",
                "Material:DownwardDepth",
                json!({ "MaxDepth": 1.0 }),
            ),
            make_mat_node("grass", "Material:Constant", json!({ "Material": "Grass" })),
            make_mat_node(
                "stone",
                "Material:Constant",
                json!({ "Material": "Rock_Stone" }),
            ),
        ];
        let edges = vec![
            make_edge("depth_filter", "q", "Queue[0]"),
            make_edge("stone", "q", "Queue[1]"),
            make_edge("grass", "depth_filter", "Input"),
        ];

        // 2x2 XZ, 4 Y slices, all solid
        let densities = solid_volume(2, 4);

        let result = evaluate_material_graph(
            &nodes,
            &edges,
            &densities,
            2,
            4,
            -10.0,
            10.0,
            0.0,
            30.0,
            None,
            &HashMap::new(),
        );

        let mat = result.unwrap();
        let grass_idx = mat.palette.iter().position(|p| p.name == "Grass").unwrap() as u8;
        let stone_idx = mat
            .palette
            .iter()
            .position(|p| p.name == "Rock_Stone")
            .unwrap() as u8;

        // Surface is y=3 (top). downward_depth = surface_yi - y
        // y=3: depth=0 → Grass
        // y=2: depth=1 → Grass (MaxDepth is 1, so depth<=1 passes)
        // y=1: depth=2 → Stone (depth > MaxDepth, filter fails, fallback to Queue[1])
        // y=0: depth=3 → Stone
        let n = 2;
        for z in 0..n {
            for x in 0..n {
                assert_eq!(
                    mat.material_ids[3 * n * n + z * n + x],
                    grass_idx,
                    "y=3 should be Grass"
                );
                assert_eq!(
                    mat.material_ids[2 * n * n + z * n + x],
                    grass_idx,
                    "y=2 should be Grass"
                );
                assert_eq!(
                    mat.material_ids[1 * n * n + z * n + x],
                    stone_idx,
                    "y=1 should be Stone"
                );
                assert_eq!(
                    mat.material_ids[0 * n * n + z * n + x],
                    stone_idx,
                    "y=0 should be Stone"
                );
            }
        }
    }

    #[test]
    fn air_voxels_stay_at_zero() {
        let nodes = vec![make_mat_node(
            "m1",
            "Material:Constant",
            json!({ "Material": "Grass" }),
        )];
        let edges: Vec<Value> = vec![];
        let densities = half_solid_volume(2, 4);

        let result = evaluate_material_graph(
            &nodes,
            &edges,
            &densities,
            2,
            4,
            -10.0,
            10.0,
            0.0,
            30.0,
            None,
            &HashMap::new(),
        );

        let mat = result.unwrap();
        let grass_idx = mat.palette.iter().position(|p| p.name == "Grass").unwrap() as u8;
        let n = 2;
        let ys = 4;
        let half_y = ys / 2;

        for y in 0..ys {
            for z in 0..n {
                for x in 0..n {
                    let idx = y * n * n + z * n + x;
                    if y < half_y {
                        // Solid → should be Grass
                        assert_eq!(mat.material_ids[idx], grass_idx);
                    } else {
                        // Air → should be Stone fallback (0)
                        assert_eq!(mat.material_ids[idx], 0);
                    }
                }
            }
        }
    }

    #[test]
    fn height_gradient_splits() {
        let nodes = vec![
            make_mat_node(
                "hg",
                "Material:HeightGradient",
                json!({ "Range": { "Min": 0.0, "Max": 100.0 } }),
            ),
            make_mat_node("low", "Material:Constant", json!({ "Material": "Stone" })),
            make_mat_node("high", "Material:Constant", json!({ "Material": "Snow" })),
        ];
        let edges = vec![
            make_edge("low", "hg", "Low"),
            make_edge("high", "hg", "High"),
        ];
        // 1x1 XZ, 4 Y slices from 0..100, all solid
        let densities = solid_volume(1, 4);

        let result = evaluate_material_graph(
            &nodes,
            &edges,
            &densities,
            1,
            4,
            -10.0,
            10.0,
            0.0,
            100.0,
            None,
            &HashMap::new(),
        );

        let mat = result.unwrap();
        let stone_idx = mat.palette.iter().position(|p| p.name == "Stone").unwrap() as u8;
        let snow_idx = mat.palette.iter().position(|p| p.name == "Snow").unwrap() as u8;

        // Mid = 50. Y at slices: 0, 33.3, 66.7, 100
        // y=0 (0.0) < 50 → Stone
        // y=1 (33.3) < 50 → Stone
        // y=2 (66.7) >= 50 → Snow
        // y=3 (100) >= 50 → Snow
        assert_eq!(mat.material_ids[0], stone_idx);
        assert_eq!(mat.material_ids[1], stone_idx);
        assert_eq!(mat.material_ids[2], snow_idx);
        assert_eq!(mat.material_ids[3], snow_idx);
    }

    #[test]
    fn column_context_computation() {
        let n = 2;
        let ys = 5;
        // Column (0,0): air, solid, solid, solid, air = surface at y=3, bottom at y=1
        // Column (1,0): all solid = surface at y=4, bottom at y=0
        let mut densities = vec![-1.0f32; n * n * ys];
        // Column (0,0)
        densities[1 * n * n + 0 * n + 0] = 1.0; // y=1
        densities[2 * n * n + 0 * n + 0] = 1.0; // y=2
        densities[3 * n * n + 0 * n + 0] = 1.0; // y=3
                                                // Column (1,0) all solid
        for y in 0..ys {
            densities[y * n * n + 0 * n + 1] = 1.0;
        }

        let cols = compute_column_contexts(&densities, n, ys);

        let c00 = &cols[0][0];
        assert_eq!(c00.surface_yi, 3);
        assert_eq!(c00.bottom_yi, 1);
        assert_eq!(c00.space_above, 1); // y=4 is air
        assert_eq!(c00.space_below, 1); // y=0 is air

        let c10 = &cols[0][1];
        assert_eq!(c10.surface_yi, 4);
        assert_eq!(c10.bottom_yi, 0);
        assert_eq!(c10.space_above, 0);
        assert_eq!(c10.space_below, 0);
    }

    #[test]
    fn material_color_lookup() {
        assert_eq!(material_color("Grass"), "#5cb85c");
        assert_eq!(material_color("Stone"), "#909090");
        assert_eq!(material_color("Lava"), "#ff4500");
        assert_eq!(material_color("UnknownStoneType"), "#909090"); // keyword match
        assert_eq!(material_color("TotallyUnknown"), "#808080"); // fallback
    }

    #[test]
    fn position_hash_deterministic() {
        let h1 = position_hash(10.0, 20.0, 30.0, 42);
        let h2 = position_hash(10.0, 20.0, 30.0, 42);
        assert_eq!(h1, h2);
        assert!(h1 >= 0.0 && h1 < 1.0);

        // Different positions should give different hashes (very likely)
        let h3 = position_hash(11.0, 20.0, 30.0, 42);
        assert_ne!(h1, h3);
    }

    #[test]
    fn condition_evaluation() {
        let ctx = MaterialVoxelContext {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            xi: 0,
            yi: 0,
            zi: 0,
            is_solid: true,
            density: 1.0,
            downward_depth: 0,
            upward_depth: 0,
            space_above: 5,
            space_below: 3,
            surface_y: 0.0,
        };

        assert!(evaluate_condition(
            &json!({ "Type": "AlwaysTrueCondition" }),
            &ctx
        ));

        assert!(evaluate_condition(
            &json!({
                "Type": "GreaterThanCondition",
                "ContextToCheck": "SPACE_ABOVE_FLOOR",
                "Threshold": 3
            }),
            &ctx
        ));

        assert!(!evaluate_condition(
            &json!({
                "Type": "SmallerThanCondition",
                "ContextToCheck": "SPACE_ABOVE_FLOOR",
                "Threshold": 3
            }),
            &ctx
        ));

        assert!(evaluate_condition(
            &json!({
                "Type": "AndCondition",
                "Conditions": [
                    { "Type": "AlwaysTrueCondition" },
                    { "Type": "GreaterThanCondition", "ContextToCheck": "SPACE_BELOW_CEILING", "Threshold": 2 }
                ]
            }),
            &ctx
        ));

        assert!(evaluate_condition(
            &json!({
                "Type": "NotCondition",
                "Condition": { "Type": "SmallerThanCondition", "ContextToCheck": "SPACE_ABOVE_FLOOR", "Threshold": 3 }
            }),
            &ctx
        ));
    }
}
