// eval/grid.rs — 2D grid evaluation with rayon parallelism
//
// Evaluates a density graph over an NxN grid of world-space positions.
// Each row is evaluated in parallel using rayon::par_iter.
//
// Uses the fast indexed evaluator (`evaluate_fast` + `EvalState`) which
// eliminates all String hashing, HashMap cloning, and node cloning from
// the hot evaluation loop.

use crate::eval::graph::EvalGraph;
use crate::eval::nodes::{evaluate_compiled, EvalState};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;

/// Result of evaluating a density graph over a 2D grid.
#[derive(Debug, Clone, Serialize)]
pub struct GridResult {
    /// Row-major f32 values (matches TS Float32Array layout).
    /// Length = resolution × resolution.
    pub values: Vec<f32>,
    /// Grid resolution (NxN).
    pub resolution: u32,
    /// Minimum value in the result (for normalization).
    pub min_value: f32,
    /// Maximum value in the result (for normalization).
    pub max_value: f32,
}

/// Evaluate a density graph over an NxN grid.
///
/// The grid covers world-space coordinates [range_min, range_max] in both X
/// and Z. Each sample is taken at the cell center (offset by half a step).
/// The Y coordinate is fixed at `y_level`.
///
/// Rows (Z-axis) are evaluated in parallel using rayon. Each thread gets its
/// own `EvalContext` so there is no contention on memoization or perm caches.
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

    let root_idx = graph.root_idx;
    let node_count = graph.node_count();

    // Evaluate rows in parallel — each row is one Z coordinate
    let row_results: Vec<(Vec<f32>, f32, f32)> = (0..n)
        .into_par_iter()
        .map(|z_idx| {
            // Each thread gets its own EvalState (perm caches are per-thread).
            // Uses flat Vec-based memo with generation counter — O(1) clear.
            let mut state = EvalState::new(node_count, content_fields.clone());
            let z = range_min + (z_idx as f64 + 0.5) * step;

            let mut row = Vec::with_capacity(n);
            let mut row_min = f32::MAX;
            let mut row_max = f32::MIN;

            for x_idx in 0..n {
                let x = range_min + (x_idx as f64 + 0.5) * step;
                state.clear_memo(); // O(1) — just bumps generation counter
                let val = evaluate_compiled(graph, &mut state, root_idx, x, y_level, z) as f32;
                row_min = row_min.min(val);
                row_max = row_max.max(val);
                row.push(val);
            }

            (row, row_min, row_max)
        })
        .collect();

    // Flatten rows into a single Vec and compute global min/max
    let mut values = Vec::with_capacity(n * n);
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for (row, r_min, r_max) in row_results {
        min_val = min_val.min(r_min);
        max_val = max_val.max(r_max);
        values.extend(row);
    }

    GridResult {
        values,
        resolution,
        min_value: min_val,
        max_value: max_val,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::graph::{GraphEdge, GraphNode, NodeData};
    use serde_json::json;

    fn make_node(
        id: &str,
        density_type: &str,
        fields: HashMap<String, serde_json::Value>,
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

    #[test]
    fn constant_grid() {
        let mut fields = HashMap::new();
        fields.insert("Value".to_string(), json!(42.0));
        let nodes = vec![make_node("c", "Constant", fields)];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();

        let result = evaluate_grid(&graph, 4, -10.0, 10.0, 64.0, &HashMap::new());

        assert_eq!(result.values.len(), 16);
        assert_eq!(result.resolution, 4);
        for &v in &result.values {
            assert!((v - 42.0).abs() < 1e-6);
        }
        assert!((result.min_value - 42.0).abs() < 1e-6);
        assert!((result.max_value - 42.0).abs() < 1e-6);
    }

    #[test]
    fn coordinate_x_grid() {
        let nodes = vec![make_node("cx", "CoordinateX", HashMap::new())];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("cx")).unwrap();

        // 2x2 grid over [-10, 10], step=10, centers at -5 and 5
        let result = evaluate_grid(&graph, 2, -10.0, 10.0, 0.0, &HashMap::new());

        assert_eq!(result.values.len(), 4);
        // Row 0 (z=-5): x=-5, x=5
        // Row 1 (z=5):  x=-5, x=5
        assert!(
            (result.values[0] - (-5.0)).abs() < 1e-6,
            "got {}",
            result.values[0]
        );
        assert!(
            (result.values[1] - 5.0).abs() < 1e-6,
            "got {}",
            result.values[1]
        );
        assert!(
            (result.values[2] - (-5.0)).abs() < 1e-6,
            "got {}",
            result.values[2]
        );
        assert!(
            (result.values[3] - 5.0).abs() < 1e-6,
            "got {}",
            result.values[3]
        );
        assert!((result.min_value - (-5.0)).abs() < 1e-6);
        assert!((result.max_value - 5.0).abs() < 1e-6);
    }

    #[test]
    fn sum_grid() {
        // Sum of two constants: 10 + 20 = 30
        let mut fields_a = HashMap::new();
        fields_a.insert("Value".to_string(), json!(10.0));
        let mut fields_b = HashMap::new();
        fields_b.insert("Value".to_string(), json!(20.0));

        let nodes = vec![
            make_node("a", "Constant", fields_a),
            make_node("b", "Constant", fields_b),
            make_node("s", "Sum", HashMap::new()),
        ];
        let edges = vec![
            GraphEdge {
                source: "a".to_string(),
                target: "s".to_string(),
                target_handle: Some("Inputs[0]".to_string()),
            },
            GraphEdge {
                source: "b".to_string(),
                target: "s".to_string(),
                target_handle: Some("Inputs[1]".to_string()),
            },
        ];
        let graph = EvalGraph::from_raw(nodes, edges, Some("s")).unwrap();

        let result = evaluate_grid(&graph, 8, -64.0, 64.0, 64.0, &HashMap::new());

        assert_eq!(result.values.len(), 64);
        for &v in &result.values {
            assert!((v - 30.0).abs() < 1e-6);
        }
    }

    #[test]
    fn grid_resolution_1() {
        let mut fields = HashMap::new();
        fields.insert("Value".to_string(), json!(7.0));
        let nodes = vec![make_node("c", "Constant", fields)];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();

        let result = evaluate_grid(&graph, 1, 0.0, 10.0, 0.0, &HashMap::new());

        assert_eq!(result.values.len(), 1);
        assert!((result.values[0] - 7.0).abs() < 1e-6);
    }
}
