// eval/vectors.rs — Vector provider evaluation
//
// Translates `vectorEvaluator.ts` into Rust. Vector providers produce
// 3D direction vectors used by VectorWarp, Angle, and other nodes.

use crate::eval::graph::EvalGraph;
use crate::eval::nodes::evaluate;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

// ── Vec3 type ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub const ZERO_VEC3: Vec3 = Vec3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

pub const UP_VEC3: Vec3 = Vec3 {
    x: 0.0,
    y: 1.0,
    z: 0.0,
};

// ── Vector math helpers ─────────────────────────────────────────────

pub fn vec3_length(v: Vec3) -> f64 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

pub fn vec3_normalize(v: Vec3) -> Vec3 {
    let len = vec3_length(v);
    if len < 1e-10 {
        return ZERO_VEC3;
    }
    Vec3 {
        x: v.x / len,
        y: v.y / len,
        z: v.z / len,
    }
}

// ── Vector Provider Evaluation ──────────────────────────────────────

/// Evaluate a vector provider node, returning a 3D direction vector.
///
/// Vector providers produce Vec3 outputs used by VectorWarp, Angle, and
/// other nodes that need directional information.
///
/// Types supported:
/// - Constant: reads fields.Value.{x,y,z}
/// - DensityGradient: finite differences of connected density function
/// - Cache / Exported / Imported: follow input edge (passthrough)
///
/// Matches `evaluateVectorProvider` in `vectorEvaluator.ts`.
pub fn evaluate_vector(
    graph: &EvalGraph,
    memo: &mut HashMap<String, f64>,
    visiting: &mut HashSet<String>,
    perm_cache: &mut FxHashMap<i32, [u8; 512]>,
    content_fields: &HashMap<String, f64>,
    node_id: &str,
    x: f64,
    y: f64,
    z: f64,
) -> Vec3 {
    let node = match graph.nodes.get(node_id) {
        Some(n) => n.clone(),
        None => return ZERO_VEC3,
    };

    let data = &node.data;
    let raw_type = data.density_type.as_deref().unwrap_or("");
    // Vector nodes may be prefixed "Vector:" in their type field
    let vec_type = raw_type.strip_prefix("Vector:").unwrap_or(raw_type);
    let fields = &data.fields;

    match vec_type {
        "Constant" => {
            // Read fields.Value.{x,y,z}
            let val = fields.get("Value");
            match val {
                Some(Value::Object(obj)) => Vec3 {
                    x: obj.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    y: obj.get("y").and_then(|v| v.as_f64()).unwrap_or(1.0),
                    z: obj.get("z").and_then(|v| v.as_f64()).unwrap_or(0.0),
                },
                _ => UP_VEC3,
            }
        }

        "DensityGradient" => {
            // Compute the gradient of a connected density function via finite differences
            let density_node_id = graph
                .inputs
                .get(node_id)
                .and_then(|m| m.get("DensityFunction").or_else(|| m.get("Input")))
                .cloned();

            let density_id = match density_node_id {
                Some(id) => id,
                None => return UP_VEC3,
            };

            let eps = 0.5;
            let inv2e = 1.0 / (2.0 * eps);

            // We need to evaluate the density function at offset positions.
            // We use the evaluate function from nodes.rs, creating temporary
            // EvalContext-like state inline.

            let mut eval = |ex: f64, ey: f64, ez: f64| -> f64 {
                // We can't easily borrow &mut EvalContext here, so we use the
                // lower-level approach: call evaluate directly with the graph.
                // Note: This is a simplified approach — the caller is responsible
                // for managing context state.
                evaluate_density_at(
                    graph,
                    memo,
                    visiting,
                    perm_cache,
                    content_fields,
                    &density_id,
                    ex,
                    ey,
                    ez,
                )
            };

            let dfdx = (eval(x + eps, y, z) - eval(x - eps, y, z)) * inv2e;
            let dfdy = (eval(x, y + eps, z) - eval(x, y - eps, z)) * inv2e;
            let dfdz = (eval(x, y, z + eps) - eval(x, y, z - eps)) * inv2e;

            Vec3 {
                x: dfdx,
                y: dfdy,
                z: dfdz,
            }
        }

        "Cache" | "Exported" | "Imported" => {
            // Passthrough: follow the connected vector input
            let src_id = graph
                .inputs
                .get(node_id)
                .and_then(|m| m.get("VectorProvider").or_else(|| m.get("Input")))
                .cloned();

            match src_id {
                Some(id) => evaluate_vector(
                    graph,
                    memo,
                    visiting,
                    perm_cache,
                    content_fields,
                    &id,
                    x,
                    y,
                    z,
                ),
                None => ZERO_VEC3,
            }
        }

        _ => ZERO_VEC3,
    }
}

/// Helper: evaluate a density node at a specific position.
///
/// This wraps the `evaluate` function from nodes.rs, temporarily clearing
/// and restoring the memo to evaluate at offset positions (needed for
/// gradient computation).
fn evaluate_density_at(
    graph: &EvalGraph,
    _memo: &mut HashMap<String, f64>,
    visiting: &mut HashSet<String>,
    perm_cache: &mut FxHashMap<i32, [u8; 512]>,
    content_fields: &HashMap<String, f64>,
    node_id: &str,
    x: f64,
    y: f64,
    z: f64,
) -> f64 {
    // For gradient computation we need fresh evaluation at each offset point.
    // We create a temporary mini-context to avoid polluting the main memo.
    let mut temp_ctx = crate::eval::nodes::EvalContext {
        graph,
        memo: HashMap::new(),
        visiting: visiting.clone(),
        perm_cache: perm_cache.clone(),
        content_fields: content_fields.clone(),
        anchor: [0.0, 0.0, 0.0],
        anchor_set: false,
        switch_state: 0,
        cell_wall_dist: f64::INFINITY,
    };

    let result = evaluate(&mut temp_ctx, node_id, x, y, z);

    // Merge any new perm_cache entries back
    for (k, v) in temp_ctx.perm_cache {
        perm_cache.entry(k).or_insert(v);
    }

    result
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::graph::{EvalGraph, GraphEdge, GraphNode, NodeData};
    use serde_json::json;

    fn make_vec_node(id: &str, vec_type: &str, fields: HashMap<String, Value>) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            data: NodeData {
                density_type: Some(format!("Vector:{}", vec_type)),
                fields,
                is_output: false,
                biome_field: None,
            },
            node_type: None,
        }
    }

    fn make_density_node(
        id: &str,
        density_type: &str,
        fields: HashMap<String, Value>,
    ) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            data: NodeData {
                density_type: Some(density_type.to_string()),
                fields,
                is_output: false,
                biome_field: None,
            },
            node_type: None,
        }
    }

    fn make_edge(source: &str, target: &str, handle: Option<&str>) -> GraphEdge {
        GraphEdge {
            source: source.to_string(),
            target: target.to_string(),
            target_handle: handle.map(|h| h.to_string()),
        }
    }

    #[test]
    fn constant_vector() {
        let mut fields = HashMap::new();
        fields.insert("Value".to_string(), json!({"x": 1.0, "y": 0.0, "z": 0.0}));
        let nodes = vec![make_vec_node("v1", "Constant", fields)];
        // Need a dummy density node to make the graph valid
        let nodes_with_root = vec![
            nodes[0].clone(),
            make_density_node("root", "Constant", {
                let mut f = HashMap::new();
                f.insert("Value".to_string(), json!(1.0));
                f
            }),
        ];
        let graph = EvalGraph::from_raw(nodes_with_root, vec![], Some("root")).unwrap();

        let mut memo = HashMap::new();
        let mut visiting = HashSet::new();
        let mut perm_cache = FxHashMap::default();
        let content_fields = HashMap::new();

        let result = evaluate_vector(
            &graph,
            &mut memo,
            &mut visiting,
            &mut perm_cache,
            &content_fields,
            "v1",
            0.0,
            0.0,
            0.0,
        );

        assert!((result.x - 1.0).abs() < 1e-10);
        assert!((result.y - 0.0).abs() < 1e-10);
        assert!((result.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn constant_vector_defaults() {
        let fields = HashMap::new();
        let nodes = vec![
            make_vec_node("v1", "Constant", fields),
            make_density_node("root", "Constant", {
                let mut f = HashMap::new();
                f.insert("Value".to_string(), json!(1.0));
                f
            }),
        ];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("root")).unwrap();

        let mut memo = HashMap::new();
        let mut visiting = HashSet::new();
        let mut perm_cache = FxHashMap::default();
        let content_fields = HashMap::new();

        let result = evaluate_vector(
            &graph,
            &mut memo,
            &mut visiting,
            &mut perm_cache,
            &content_fields,
            "v1",
            0.0,
            0.0,
            0.0,
        );

        // Default is UP (0, 1, 0)
        assert!((result.x - 0.0).abs() < 1e-10);
        assert!((result.y - 1.0).abs() < 1e-10);
        assert!((result.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn unknown_vector_type_returns_zero() {
        let fields = HashMap::new();
        let nodes = vec![
            make_vec_node("v1", "UnknownVecType", fields),
            make_density_node("root", "Constant", {
                let mut f = HashMap::new();
                f.insert("Value".to_string(), json!(1.0));
                f
            }),
        ];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("root")).unwrap();

        let mut memo = HashMap::new();
        let mut visiting = HashSet::new();
        let mut perm_cache = FxHashMap::default();
        let content_fields = HashMap::new();

        let result = evaluate_vector(
            &graph,
            &mut memo,
            &mut visiting,
            &mut perm_cache,
            &content_fields,
            "v1",
            0.0,
            0.0,
            0.0,
        );

        assert!((result.x).abs() < 1e-10);
        assert!((result.y).abs() < 1e-10);
        assert!((result.z).abs() < 1e-10);
    }

    #[test]
    fn passthrough_vector() {
        let mut fields_const = HashMap::new();
        fields_const.insert("Value".to_string(), json!({"x": 3.0, "y": 4.0, "z": 5.0}));

        let nodes = vec![
            make_vec_node("v_const", "Constant", fields_const),
            make_vec_node("v_exported", "Exported", HashMap::new()),
            make_density_node("root", "Constant", {
                let mut f = HashMap::new();
                f.insert("Value".to_string(), json!(1.0));
                f
            }),
        ];
        let edges = vec![make_edge("v_const", "v_exported", Some("VectorProvider"))];
        let graph = EvalGraph::from_raw(nodes, edges, Some("root")).unwrap();

        let mut memo = HashMap::new();
        let mut visiting = HashSet::new();
        let mut perm_cache = FxHashMap::default();
        let content_fields = HashMap::new();

        let result = evaluate_vector(
            &graph,
            &mut memo,
            &mut visiting,
            &mut perm_cache,
            &content_fields,
            "v_exported",
            0.0,
            0.0,
            0.0,
        );

        assert!((result.x - 3.0).abs() < 1e-10);
        assert!((result.y - 4.0).abs() < 1e-10);
        assert!((result.z - 5.0).abs() < 1e-10);
    }

    #[test]
    fn density_gradient_vector() {
        // Create a graph with a CoordinateX density node (returns x).
        // The gradient should be approximately (1, 0, 0).
        let nodes = vec![
            make_density_node("coord_x", "CoordinateX", HashMap::new()),
            make_vec_node("grad", "DensityGradient", HashMap::new()),
            make_density_node("root", "Constant", {
                let mut f = HashMap::new();
                f.insert("Value".to_string(), json!(1.0));
                f
            }),
        ];
        let edges = vec![make_edge("coord_x", "grad", Some("DensityFunction"))];
        let graph = EvalGraph::from_raw(nodes, edges, Some("root")).unwrap();

        let mut memo = HashMap::new();
        let mut visiting = HashSet::new();
        let mut perm_cache = FxHashMap::default();
        let content_fields = HashMap::new();

        let result = evaluate_vector(
            &graph,
            &mut memo,
            &mut visiting,
            &mut perm_cache,
            &content_fields,
            "grad",
            5.0,
            3.0,
            2.0,
        );

        // Gradient of f(x,y,z)=x is (1, 0, 0)
        assert!(
            (result.x - 1.0).abs() < 1e-6,
            "Expected dx ~1.0, got {}",
            result.x
        );
        assert!(
            (result.y).abs() < 1e-6,
            "Expected dy ~0.0, got {}",
            result.y
        );
        assert!(
            (result.z).abs() < 1e-6,
            "Expected dz ~0.0, got {}",
            result.z
        );
    }
}
