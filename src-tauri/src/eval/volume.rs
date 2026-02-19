// eval/volume.rs — 3D volume evaluation with rayon parallelism
//
// Evaluates a density graph over a 3D volume (NxN XZ grid × Y slices).
// Y-slices are evaluated in parallel using rayon::par_iter.

use crate::eval::graph::EvalGraph;
use crate::eval::nodes::{evaluate, EvalContext};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;

/// Result of evaluating a density graph over a 3D volume.
#[derive(Debug, Clone, Serialize)]
pub struct VolumeResult {
    /// Y-major layout: densities[y * n * n + z * n + x].
    /// Length = resolution × resolution × y_slices.
    pub densities: Vec<f32>,
    /// XZ grid resolution (NxN per slice).
    pub resolution: u32,
    /// Number of Y slices.
    pub y_slices: u32,
    /// Minimum value in the result (for normalization).
    pub min_value: f32,
    /// Maximum value in the result (for normalization).
    pub max_value: f32,
}

/// Evaluate a density graph over a 3D volume.
///
/// The volume covers world-space coordinates [range_min, range_max] in X and Z,
/// and [y_min, y_max] in Y. Each XZ sample is taken at the cell center.
///
/// Y-slices are evaluated in parallel using rayon. Each thread gets its own
/// `EvalContext` so there is no contention on memoization or perm caches.
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
    let step_y = if ys > 1 {
        (y_max - y_min) / (ys as f64 - 1.0)
    } else {
        0.0
    };

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
                let wz = range_min + (zi as f64 + 0.5) * step_xz;
                for xi in 0..n {
                    let wx = range_min + (xi as f64 + 0.5) * step_xz;
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

    // Assemble slices into Y-major layout and compute global min/max
    let total = n * n * ys;
    let mut densities = Vec::with_capacity(total);
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for (slice, s_min, s_max) in slice_results {
        min_val = min_val.min(s_min);
        max_val = max_val.max(s_max);
        densities.extend(slice);
    }

    VolumeResult {
        densities,
        resolution,
        y_slices,
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
    fn constant_volume() {
        let mut fields = HashMap::new();
        fields.insert("Value".to_string(), json!(7.0));
        let nodes = vec![make_node("c", "Constant", fields)];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();

        let result = evaluate_volume(
            &graph,
            4, // 4x4 XZ grid
            -10.0,
            10.0,
            0.0,
            100.0,
            3, // 3 Y slices
            &HashMap::new(),
        );

        assert_eq!(result.densities.len(), 4 * 4 * 3); // 48
        assert_eq!(result.resolution, 4);
        assert_eq!(result.y_slices, 3);
        for &v in &result.densities {
            assert!((v - 7.0).abs() < 1e-6);
        }
        assert!((result.min_value - 7.0).abs() < 1e-6);
        assert!((result.max_value - 7.0).abs() < 1e-6);
    }

    #[test]
    fn coordinate_y_volume() {
        // CoordinateY should return the Y value at each slice
        let nodes = vec![make_node("cy", "CoordinateY", HashMap::new())];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("cy")).unwrap();

        // 1x1 XZ grid, 5 Y slices from 0 to 100
        let result = evaluate_volume(
            &graph,
            1,
            0.0,
            10.0,
            0.0,
            100.0,
            5, // Y slices at 0, 25, 50, 75, 100
            &HashMap::new(),
        );

        assert_eq!(result.densities.len(), 5);
        let expected_ys = [0.0, 25.0, 50.0, 75.0, 100.0];
        for (i, &expected_y) in expected_ys.iter().enumerate() {
            assert!(
                (result.densities[i] - expected_y as f32).abs() < 1e-4,
                "slice {}: expected {}, got {}",
                i,
                expected_y,
                result.densities[i]
            );
        }
    }

    #[test]
    fn single_y_slice_volume() {
        let mut fields = HashMap::new();
        fields.insert("Value".to_string(), json!(3.14));
        let nodes = vec![make_node("c", "Constant", fields)];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();

        let result = evaluate_volume(
            &graph,
            2,
            -5.0,
            5.0,
            64.0,
            64.0,
            1, // Single Y slice
            &HashMap::new(),
        );

        assert_eq!(result.densities.len(), 4); // 2x2 * 1 slice
        assert_eq!(result.y_slices, 1);
        for &v in &result.densities {
            assert!((v - 3.14).abs() < 1e-4);
        }
    }

    #[test]
    fn volume_min_max_tracking() {
        // Use CoordinateX so values vary across the grid
        let nodes = vec![make_node("cx", "CoordinateX", HashMap::new())];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("cx")).unwrap();

        // 4x4 grid over [-20, 20], cell centers at -15, -5, 5, 15
        let result = evaluate_volume(&graph, 4, -20.0, 20.0, 0.0, 0.0, 1, &HashMap::new());

        assert!((result.min_value - (-15.0)).abs() < 1e-4);
        assert!((result.max_value - 15.0).abs() < 1e-4);
    }

    #[test]
    fn sum_volume() {
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

        let result = evaluate_volume(&graph, 2, -10.0, 10.0, 0.0, 64.0, 2, &HashMap::new());

        assert_eq!(result.densities.len(), 2 * 2 * 2); // 8
        for &v in &result.densities {
            assert!((v - 30.0).abs() < 1e-6);
        }
    }
}
