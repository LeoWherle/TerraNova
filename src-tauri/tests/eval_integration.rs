//! Integration tests for the Rust evaluation pipeline.
//!
//! These tests verify:
//!   1. Grid evaluation produces correct values for various graph types
//!   2. Volume evaluation produces correct values
//!   3. Grid vs point-by-point evaluation parity (every grid cell matches
//!      a single-point evaluation at the same coordinates)
//!   4. JSON round-trip: parse JSON → build graph → evaluate → serialize
//!      (simulates the full IPC path a Tauri command takes)
//!   5. Complex multi-node graphs produce finite, reasonable results
//!   6. Edge cases: empty inputs, single-cell grids, extreme ranges
//!   7. Determinism: same graph + same inputs = same outputs

use serde_json::{json, Value};
use std::collections::HashMap;
use terranova_lib::eval::graph::{EvalGraph, GraphEdge, GraphNode, NodeData};
use terranova_lib::eval::grid::{evaluate_grid, GridResult};
use terranova_lib::eval::nodes::{evaluate, EvalContext};
use terranova_lib::eval::volume::{evaluate_volume, VolumeResult};

// ── Helpers ────────────────────────────────────────────────────────

fn make_node(id: &str, density_type: &str, fields: HashMap<String, Value>) -> GraphNode {
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

/// Evaluate a single point using the EvalContext directly.
fn eval_point(graph: &EvalGraph, x: f64, y: f64, z: f64) -> f64 {
    let mut ctx = EvalContext::new(graph, HashMap::new());
    ctx.clear_memo();
    evaluate(&mut ctx, &graph.root_id, x, y, z)
}

/// Parse a JSON request (simulating Tauri IPC) into an EvalGraph and run grid evaluation.
fn ipc_evaluate_grid(json_str: &str) -> GridResult {
    let request: Value = serde_json::from_str(json_str).unwrap();
    let nodes: Vec<GraphNode> = serde_json::from_value(request["nodes"].clone()).unwrap();
    let edges: Vec<GraphEdge> = serde_json::from_value(request["edges"].clone()).unwrap();
    let root_node_id = request["root_node_id"].as_str();
    let resolution = request["resolution"].as_u64().unwrap() as u32;
    let range_min = request["range_min"].as_f64().unwrap();
    let range_max = request["range_max"].as_f64().unwrap();
    let y_level = request["y_level"].as_f64().unwrap();
    let content_fields: HashMap<String, f64> = request
        .get("content_fields")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let graph = EvalGraph::from_raw(nodes, edges, root_node_id).unwrap();
    evaluate_grid(
        &graph,
        resolution,
        range_min,
        range_max,
        y_level,
        &content_fields,
    )
}

/// Parse a JSON request (simulating Tauri IPC) into an EvalGraph and run volume evaluation.
fn ipc_evaluate_volume(json_str: &str) -> VolumeResult {
    let request: Value = serde_json::from_str(json_str).unwrap();
    let nodes: Vec<GraphNode> = serde_json::from_value(request["nodes"].clone()).unwrap();
    let edges: Vec<GraphEdge> = serde_json::from_value(request["edges"].clone()).unwrap();
    let root_node_id = request["root_node_id"].as_str();
    let resolution = request["resolution"].as_u64().unwrap() as u32;
    let range_min = request["range_min"].as_f64().unwrap();
    let range_max = request["range_max"].as_f64().unwrap();
    let y_min = request["y_min"].as_f64().unwrap();
    let y_max = request["y_max"].as_f64().unwrap();
    let y_slices = request["y_slices"].as_u64().unwrap() as u32;
    let content_fields: HashMap<String, f64> = request
        .get("content_fields")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let graph = EvalGraph::from_raw(nodes, edges, root_node_id).unwrap();
    evaluate_volume(
        &graph,
        resolution,
        range_min,
        range_max,
        y_min,
        y_max,
        y_slices,
        &content_fields,
    )
}

// ── 1. Grid evaluation correctness ────────────────────────────────

#[test]
fn grid_constant_all_values_equal() {
    let mut fields = HashMap::new();
    fields.insert("Value".to_string(), json!(17.5));
    let nodes = vec![make_node("c", "Constant", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();

    let result = evaluate_grid(&graph, 16, -100.0, 100.0, 64.0, &HashMap::new());

    assert_eq!(result.values.len(), 256);
    for (i, &v) in result.values.iter().enumerate() {
        assert!(
            (v - 17.5).abs() < 1e-5,
            "cell {} expected 17.5, got {}",
            i,
            v
        );
    }
    assert!((result.min_value - 17.5).abs() < 1e-5);
    assert!((result.max_value - 17.5).abs() < 1e-5);
}

#[test]
fn grid_coordinate_x_varies_across_columns() {
    let nodes = vec![make_node("cx", "CoordinateX", HashMap::new())];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("cx")).unwrap();

    let result = evaluate_grid(&graph, 4, -20.0, 20.0, 0.0, &HashMap::new());

    // Step = 10, cell centers at x = -15, -5, 5, 15
    assert_eq!(result.values.len(), 16);
    // All rows should have the same X pattern
    for row in 0..4 {
        let expected = [-15.0_f32, -5.0, 5.0, 15.0];
        for col in 0..4 {
            let idx = row * 4 + col;
            assert!(
                (result.values[idx] - expected[col]).abs() < 1e-4,
                "row {} col {}: expected {}, got {}",
                row,
                col,
                expected[col],
                result.values[idx]
            );
        }
    }
}

#[test]
fn grid_coordinate_z_varies_across_rows() {
    let nodes = vec![make_node("cz", "CoordinateZ", HashMap::new())];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("cz")).unwrap();

    let result = evaluate_grid(&graph, 4, -20.0, 20.0, 0.0, &HashMap::new());

    // Step = 10, cell centers at z = -15, -5, 5, 15
    for row in 0..4 {
        let expected_z = -15.0 + row as f32 * 10.0;
        for col in 0..4 {
            let idx = row * 4 + col;
            assert!(
                (result.values[idx] - expected_z).abs() < 1e-4,
                "row {} col {}: expected z={}, got {}",
                row,
                col,
                expected_z,
                result.values[idx]
            );
        }
    }
}

#[test]
fn grid_coordinate_y_returns_y_level() {
    let nodes = vec![make_node("cy", "CoordinateY", HashMap::new())];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("cy")).unwrap();

    let result = evaluate_grid(&graph, 8, -50.0, 50.0, 128.0, &HashMap::new());

    for (i, &v) in result.values.iter().enumerate() {
        assert!(
            (v - 128.0).abs() < 1e-4,
            "cell {}: expected Y=128, got {}",
            i,
            v
        );
    }
}

#[test]
fn grid_negate_constant() {
    let mut fields = HashMap::new();
    fields.insert("Value".to_string(), json!(7.0));
    let nodes = vec![
        make_node("c", "Constant", fields),
        make_node("neg", "Negate", HashMap::new()),
    ];
    let edges = vec![make_edge("c", "neg", Some("Input"))];
    let graph = EvalGraph::from_raw(nodes, edges, Some("neg")).unwrap();

    let result = evaluate_grid(&graph, 4, -10.0, 10.0, 0.0, &HashMap::new());

    for &v in &result.values {
        assert!((v - (-7.0)).abs() < 1e-5);
    }
}

#[test]
fn grid_sum_three_constants() {
    let mut fa = HashMap::new();
    fa.insert("Value".to_string(), json!(10.0));
    let mut fb = HashMap::new();
    fb.insert("Value".to_string(), json!(20.0));
    let mut fc = HashMap::new();
    fc.insert("Value".to_string(), json!(30.0));

    let nodes = vec![
        make_node("a", "Constant", fa),
        make_node("b", "Constant", fb),
        make_node("c", "Constant", fc),
        make_node("s", "Sum", HashMap::new()),
    ];
    let edges = vec![
        make_edge("a", "s", Some("Inputs[0]")),
        make_edge("b", "s", Some("Inputs[1]")),
        make_edge("c", "s", Some("Inputs[2]")),
    ];
    let graph = EvalGraph::from_raw(nodes, edges, Some("s")).unwrap();

    let result = evaluate_grid(&graph, 4, -10.0, 10.0, 0.0, &HashMap::new());

    for &v in &result.values {
        assert!((v - 60.0).abs() < 1e-5);
    }
}

#[test]
fn grid_clamp_coordinate_x() {
    let mut clamp_fields = HashMap::new();
    clamp_fields.insert("Min".to_string(), json!(-5.0));
    clamp_fields.insert("Max".to_string(), json!(5.0));

    let nodes = vec![
        make_node("cx", "CoordinateX", HashMap::new()),
        make_node("cl", "Clamp", clamp_fields),
    ];
    let edges = vec![make_edge("cx", "cl", Some("Input"))];
    let graph = EvalGraph::from_raw(nodes, edges, Some("cl")).unwrap();

    // Grid over [-20, 20], so X ranges from -17.5 to 17.5 (at centers)
    let result = evaluate_grid(&graph, 8, -20.0, 20.0, 0.0, &HashMap::new());

    for &v in &result.values {
        assert!(v >= -5.0 - 1e-5, "value {} below clamp min", v);
        assert!(v <= 5.0 + 1e-5, "value {} above clamp max", v);
    }
}

// ── 2. Volume evaluation correctness ──────────────────────────────

#[test]
fn volume_constant_all_slices() {
    let mut fields = HashMap::new();
    fields.insert("Value".to_string(), json!(3.0));
    let nodes = vec![make_node("c", "Constant", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();

    let result = evaluate_volume(&graph, 4, -10.0, 10.0, 0.0, 100.0, 5, &HashMap::new());

    assert_eq!(result.densities.len(), 4 * 4 * 5);
    for &v in &result.densities {
        assert!((v - 3.0).abs() < 1e-5);
    }
}

#[test]
fn volume_coordinate_y_varies_across_slices() {
    let nodes = vec![make_node("cy", "CoordinateY", HashMap::new())];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("cy")).unwrap();

    // 1x1 XZ, 5 Y slices from 0 to 200 → Y = 0, 50, 100, 150, 200
    let result = evaluate_volume(&graph, 1, 0.0, 10.0, 0.0, 200.0, 5, &HashMap::new());

    assert_eq!(result.densities.len(), 5);
    let expected = [0.0f32, 50.0, 100.0, 150.0, 200.0];
    for (i, &v) in result.densities.iter().enumerate() {
        assert!(
            (v - expected[i]).abs() < 1e-3,
            "slice {}: expected {}, got {}",
            i,
            expected[i],
            v
        );
    }
}

#[test]
fn volume_y_gradient() {
    // YGradient returns (y - fromY) / (toY - fromY), a 0→1 normalized gradient
    let mut yg_fields = HashMap::new();
    yg_fields.insert("FromY".to_string(), json!(0.0));
    yg_fields.insert("ToY".to_string(), json!(100.0));

    let nodes = vec![make_node("yg", "YGradient", yg_fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("yg")).unwrap();

    // 1x1 XZ, 3 Y slices at Y = 0, 50, 100
    let result = evaluate_volume(&graph, 1, 0.0, 10.0, 0.0, 100.0, 3, &HashMap::new());

    assert_eq!(result.densities.len(), 3);
    // Y=0 → 0.0, Y=50 → 0.5, Y=100 → 1.0
    assert!(
        (result.densities[0] - 0.0).abs() < 1e-4,
        "Y=0: {}",
        result.densities[0]
    );
    assert!(
        (result.densities[1] - 0.5).abs() < 1e-4,
        "Y=50: {}",
        result.densities[1]
    );
    assert!(
        (result.densities[2] - 1.0).abs() < 1e-4,
        "Y=100: {}",
        result.densities[2]
    );
}

#[test]
fn volume_min_max_tracking() {
    let nodes = vec![make_node("cx", "CoordinateX", HashMap::new())];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("cx")).unwrap();

    // 8x8 over [-40, 40], Y doesn't matter for CoordinateX
    let result = evaluate_volume(&graph, 8, -40.0, 40.0, 0.0, 10.0, 2, &HashMap::new());

    // Cell centers at x = -35, -25, -15, -5, 5, 15, 25, 35
    assert!((result.min_value - (-35.0)).abs() < 1e-3);
    assert!((result.max_value - 35.0).abs() < 1e-3);
}

// ── 3. Grid vs point-by-point parity ──────────────────────────────

/// Verify that every cell in a grid evaluation matches a single-point
/// evaluation at the same world coordinates.
fn assert_grid_point_parity(
    graph: &EvalGraph,
    resolution: u32,
    range_min: f64,
    range_max: f64,
    y_level: f64,
) {
    let n = resolution as usize;
    let step = (range_max - range_min) / n as f64;
    let cf = HashMap::new();

    let grid_result = evaluate_grid(graph, resolution, range_min, range_max, y_level, &cf);

    for z_idx in 0..n {
        let z = range_min + (z_idx as f64 + 0.5) * step;
        for x_idx in 0..n {
            let x = range_min + (x_idx as f64 + 0.5) * step;
            let grid_val = grid_result.values[z_idx * n + x_idx];
            let point_val = eval_point(graph, x, y_level, z) as f32;

            assert!(
                (grid_val - point_val).abs() < 1e-5,
                "Mismatch at ({}, {}, {}): grid={}, point={}",
                x,
                y_level,
                z,
                grid_val,
                point_val
            );
        }
    }
}

#[test]
fn parity_constant() {
    let mut fields = HashMap::new();
    fields.insert("Value".to_string(), json!(99.0));
    let nodes = vec![make_node("c", "Constant", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();
    assert_grid_point_parity(&graph, 8, -50.0, 50.0, 64.0);
}

#[test]
fn parity_coordinate_x() {
    let nodes = vec![make_node("cx", "CoordinateX", HashMap::new())];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("cx")).unwrap();
    assert_grid_point_parity(&graph, 16, -100.0, 100.0, 0.0);
}

#[test]
fn parity_simplex_noise_2d() {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.01));
    fields.insert("Seed".to_string(), json!(12345));
    let nodes = vec![make_node("n", "SimplexNoise2D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("n")).unwrap();
    assert_grid_point_parity(&graph, 16, -64.0, 64.0, 64.0);
}

#[test]
fn parity_fractal_noise_2d() {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.005));
    fields.insert("Octaves".to_string(), json!(4));
    fields.insert("Lacunarity".to_string(), json!(2.0));
    fields.insert("Gain".to_string(), json!(0.5));
    fields.insert("Seed".to_string(), json!(777));
    let nodes = vec![make_node("fn", "FractalNoise2D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("fn")).unwrap();
    assert_grid_point_parity(&graph, 16, -64.0, 64.0, 64.0);
}

#[test]
fn parity_sum_and_negate() {
    let mut fa = HashMap::new();
    fa.insert("Value".to_string(), json!(10.0));

    let nodes = vec![
        make_node("cx", "CoordinateX", HashMap::new()),
        make_node("c", "Constant", fa),
        make_node("s", "Sum", HashMap::new()),
        make_node("neg", "Negate", HashMap::new()),
    ];
    let edges = vec![
        make_edge("cx", "s", Some("Inputs[0]")),
        make_edge("c", "s", Some("Inputs[1]")),
        make_edge("s", "neg", Some("Input")),
    ];
    let graph = EvalGraph::from_raw(nodes, edges, Some("neg")).unwrap();
    assert_grid_point_parity(&graph, 8, -30.0, 30.0, 0.0);
}

#[test]
fn parity_deep_chain() {
    let mut fields = HashMap::new();
    fields.insert("Value".to_string(), json!(5.0));
    let nodes = vec![
        make_node("c", "Constant", fields),
        make_node("n1", "Negate", HashMap::new()),
        make_node("a1", "Abs", HashMap::new()),
        make_node("sq", "Square", HashMap::new()),
        make_node("sr", "SquareRoot", HashMap::new()),
    ];
    let edges = vec![
        make_edge("c", "n1", Some("Input")),
        make_edge("n1", "a1", Some("Input")),
        make_edge("a1", "sq", Some("Input")),
        make_edge("sq", "sr", Some("Input")),
    ];
    let graph = EvalGraph::from_raw(nodes, edges, Some("sr")).unwrap();
    assert_grid_point_parity(&graph, 4, -10.0, 10.0, 0.0);
}

/// Verify volume vs point-by-point parity.
fn assert_volume_point_parity(
    graph: &EvalGraph,
    resolution: u32,
    range_min: f64,
    range_max: f64,
    y_min: f64,
    y_max: f64,
    y_slices: u32,
) {
    let n = resolution as usize;
    let ys = y_slices as usize;
    let step_xz = (range_max - range_min) / n as f64;
    let step_y = if ys > 1 {
        (y_max - y_min) / (ys as f64 - 1.0)
    } else {
        0.0
    };
    let cf = HashMap::new();

    let vol_result = evaluate_volume(
        graph, resolution, range_min, range_max, y_min, y_max, y_slices, &cf,
    );

    for yi in 0..ys {
        let wy = y_min + yi as f64 * step_y;
        for zi in 0..n {
            let wz = range_min + (zi as f64 + 0.5) * step_xz;
            for xi in 0..n {
                let wx = range_min + (xi as f64 + 0.5) * step_xz;
                let vol_val = vol_result.densities[yi * n * n + zi * n + xi];
                let point_val = eval_point(graph, wx, wy, wz) as f32;

                assert!(
                    (vol_val - point_val).abs() < 1e-5,
                    "Mismatch at ({}, {}, {}): volume={}, point={}",
                    wx,
                    wy,
                    wz,
                    vol_val,
                    point_val
                );
            }
        }
    }
}

#[test]
fn parity_volume_noise_3d() {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.02));
    fields.insert("Seed".to_string(), json!(54321));
    let nodes = vec![make_node("n", "SimplexNoise3D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("n")).unwrap();
    assert_volume_point_parity(&graph, 4, -32.0, 32.0, 0.0, 128.0, 4);
}

#[test]
fn parity_volume_y_gradient() {
    let mut yg = HashMap::new();
    yg.insert("FromY".to_string(), json!(0.0));
    yg.insert("ToY".to_string(), json!(256.0));
    yg.insert("FromValue".to_string(), json!(1.0));
    yg.insert("ToValue".to_string(), json!(-1.0));
    let nodes = vec![make_node("yg", "YGradient", yg)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("yg")).unwrap();
    assert_volume_point_parity(&graph, 4, -10.0, 10.0, 0.0, 256.0, 8);
}

// ── 4. JSON round-trip (IPC simulation) ───────────────────────────

#[test]
fn ipc_grid_constant() {
    let request = json!({
        "nodes": [
            {"id": "c", "data": {"type": "Constant", "fields": {"Value": 42.0}}}
        ],
        "edges": [],
        "resolution": 8,
        "range_min": -50.0,
        "range_max": 50.0,
        "y_level": 64.0,
        "root_node_id": "c",
        "content_fields": {},
    });

    let result = ipc_evaluate_grid(&serde_json::to_string(&request).unwrap());

    assert_eq!(result.values.len(), 64);
    for &v in &result.values {
        assert!((v - 42.0).abs() < 1e-5);
    }
}

#[test]
fn ipc_grid_sum_of_noise_and_constant() {
    let request = json!({
        "nodes": [
            {"id": "n", "data": {"type": "SimplexNoise2D", "fields": {"Frequency": 0.01, "Seed": 42}}},
            {"id": "c", "data": {"type": "Constant", "fields": {"Value": 100.0}}},
            {"id": "s", "data": {"type": "Sum", "fields": {}}}
        ],
        "edges": [
            {"source": "n", "target": "s", "targetHandle": "Inputs[0]"},
            {"source": "c", "target": "s", "targetHandle": "Inputs[1]"}
        ],
        "resolution": 16,
        "range_min": -64.0,
        "range_max": 64.0,
        "y_level": 64.0,
        "root_node_id": "s",
        "content_fields": {},
    });

    let result = ipc_evaluate_grid(&serde_json::to_string(&request).unwrap());

    assert_eq!(result.values.len(), 256);
    // All values should be around 100 ± noise amplitude (simplex is in [-1, 1])
    for &v in &result.values {
        assert!(v > 98.0 && v < 102.0, "unexpected value: {}", v);
    }
}

#[test]
fn ipc_grid_response_serializes_correctly() {
    let request = json!({
        "nodes": [
            {"id": "cx", "data": {"type": "CoordinateX", "fields": {}}}
        ],
        "edges": [],
        "resolution": 4,
        "range_min": -20.0,
        "range_max": 20.0,
        "y_level": 0.0,
        "root_node_id": "cx",
        "content_fields": {},
    });

    let result = ipc_evaluate_grid(&serde_json::to_string(&request).unwrap());

    // Verify the result can be serialized back to JSON (as Tauri would do)
    let json_response = serde_json::to_string(&result).unwrap();
    let parsed: Value = serde_json::from_str(&json_response).unwrap();

    assert_eq!(parsed["resolution"].as_u64().unwrap(), 4);
    let values = parsed["values"].as_array().unwrap();
    assert_eq!(values.len(), 16);
    // First row, first cell should be X center = -15
    assert!((values[0].as_f64().unwrap() - (-15.0)).abs() < 1e-3);
}

#[test]
fn ipc_volume_fractal_noise() {
    let request = json!({
        "nodes": [
            {"id": "fn", "data": {"type": "FractalNoise2D", "fields": {
                "Frequency": 0.01, "Octaves": 3, "Lacunarity": 2.0, "Gain": 0.5, "Seed": 99
            }}}
        ],
        "edges": [],
        "resolution": 8,
        "range_min": -32.0,
        "range_max": 32.0,
        "y_min": 0.0,
        "y_max": 100.0,
        "y_slices": 4,
        "root_node_id": "fn",
        "content_fields": {},
    });

    let result = ipc_evaluate_volume(&serde_json::to_string(&request).unwrap());

    assert_eq!(result.densities.len(), 8 * 8 * 4);
    assert_eq!(result.resolution, 8);
    assert_eq!(result.y_slices, 4);

    // All values should be finite
    for &v in &result.densities {
        assert!(v.is_finite(), "non-finite value in volume: {}", v);
    }
}

#[test]
fn ipc_volume_response_serializes_correctly() {
    let request = json!({
        "nodes": [
            {"id": "c", "data": {"type": "Constant", "fields": {"Value": 1.0}}}
        ],
        "edges": [],
        "resolution": 2,
        "range_min": -10.0,
        "range_max": 10.0,
        "y_min": 0.0,
        "y_max": 50.0,
        "y_slices": 3,
        "root_node_id": "c",
        "content_fields": {},
    });

    let result = ipc_evaluate_volume(&serde_json::to_string(&request).unwrap());

    let json_response = serde_json::to_string(&result).unwrap();
    let parsed: Value = serde_json::from_str(&json_response).unwrap();

    assert_eq!(parsed["resolution"].as_u64().unwrap(), 2);
    assert_eq!(parsed["y_slices"].as_u64().unwrap(), 3);
    let densities = parsed["densities"].as_array().unwrap();
    assert_eq!(densities.len(), 12); // 2*2*3
}

// ── 5. Complex multi-node graph tests ─────────────────────────────

#[test]
fn complex_terrain_pipeline_produces_finite_values() {
    // YGradient (0→1 normalized) + fractal noise + amplitude + sum + clamp
    let mut yg_fields = HashMap::new();
    yg_fields.insert("FromY".to_string(), json!(0.0));
    yg_fields.insert("ToY".to_string(), json!(256.0));

    let mut noise_fields = HashMap::new();
    noise_fields.insert("Frequency".to_string(), json!(0.01));
    noise_fields.insert("Octaves".to_string(), json!(4));
    noise_fields.insert("Lacunarity".to_string(), json!(2.0));
    noise_fields.insert("Gain".to_string(), json!(0.5));
    noise_fields.insert("Seed".to_string(), json!(42));

    let mut amp_fields = HashMap::new();
    amp_fields.insert("Value".to_string(), json!(30.0));

    let mut clamp_fields = HashMap::new();
    clamp_fields.insert("Min".to_string(), json!(-1.0));
    clamp_fields.insert("Max".to_string(), json!(1.0));

    // LinearTransform uses Scale/Offset: v * scale + offset
    let mut lt_fields = HashMap::new();
    lt_fields.insert("Scale".to_string(), json!(0.5));
    lt_fields.insert("Offset".to_string(), json!(0.5));

    let nodes = vec![
        make_node("yg", "YGradient", yg_fields),
        make_node("fn", "FractalNoise2D", noise_fields),
        make_node("amp", "AmplitudeConstant", amp_fields),
        make_node("sum", "Sum", HashMap::new()),
        make_node("clamp", "Clamp", clamp_fields),
        make_node("lt", "LinearTransform", lt_fields),
    ];
    let edges = vec![
        make_edge("fn", "amp", Some("Input")),
        make_edge("yg", "sum", Some("Inputs[0]")),
        make_edge("amp", "sum", Some("Inputs[1]")),
        make_edge("sum", "clamp", Some("Input")),
        make_edge("clamp", "lt", Some("Input")),
    ];
    let graph = EvalGraph::from_raw(nodes, edges, Some("lt")).unwrap();

    // Grid evaluation
    let grid_result = evaluate_grid(&graph, 32, -64.0, 64.0, 64.0, &HashMap::new());
    for &v in &grid_result.values {
        assert!(v.is_finite(), "non-finite in grid: {}", v);
        // LinearTransform(clamp[-1,1]): v * 0.5 + 0.5 → output in [0, 1]
        assert!(v >= -0.001 && v <= 1.001, "out of range: {}", v);
    }

    // Volume evaluation
    let vol_result = evaluate_volume(&graph, 16, -64.0, 64.0, 0.0, 256.0, 16, &HashMap::new());
    for &v in &vol_result.densities {
        assert!(v.is_finite(), "non-finite in volume: {}", v);
        assert!(v >= -0.001 && v <= 1.001, "out of range: {}", v);
    }
}

#[test]
fn multi_noise_blend_produces_finite_values() {
    // Two noise sources blended together
    let mut n1f = HashMap::new();
    n1f.insert("Frequency".to_string(), json!(0.005));
    n1f.insert("Octaves".to_string(), json!(5));
    n1f.insert("Lacunarity".to_string(), json!(2.0));
    n1f.insert("Gain".to_string(), json!(0.5));
    n1f.insert("Seed".to_string(), json!(100));

    let mut n2f = HashMap::new();
    n2f.insert("Frequency".to_string(), json!(0.03));
    n2f.insert("Seed".to_string(), json!(200));

    let mut blend_factor = HashMap::new();
    blend_factor.insert("Value".to_string(), json!(0.5));

    let nodes = vec![
        make_node("n1", "FractalNoise2D", n1f),
        make_node("n2", "SimplexNoise2D", n2f),
        make_node("bf", "Constant", blend_factor),
        make_node("blend", "Blend", HashMap::new()),
    ];
    let edges = vec![
        make_edge("n1", "blend", Some("InputA")),
        make_edge("n2", "blend", Some("InputB")),
        make_edge("bf", "blend", Some("Factor")),
    ];
    let graph = EvalGraph::from_raw(nodes, edges, Some("blend")).unwrap();

    let result = evaluate_grid(&graph, 32, -100.0, 100.0, 64.0, &HashMap::new());

    let mut has_variation = false;
    let first = result.values[0];
    for &v in &result.values {
        assert!(v.is_finite(), "non-finite: {}", v);
        if (v - first).abs() > 0.001 {
            has_variation = true;
        }
    }
    assert!(has_variation, "noise blend should produce variation");
}

#[test]
fn product_of_coordinates() {
    // Product(CoordinateX, CoordinateZ) — value should equal x * z
    let nodes = vec![
        make_node("cx", "CoordinateX", HashMap::new()),
        make_node("cz", "CoordinateZ", HashMap::new()),
        make_node("p", "Product", HashMap::new()),
    ];
    let edges = vec![
        make_edge("cx", "p", Some("Inputs[0]")),
        make_edge("cz", "p", Some("Inputs[1]")),
    ];
    let graph = EvalGraph::from_raw(nodes, edges, Some("p")).unwrap();

    let result = evaluate_grid(&graph, 4, -20.0, 20.0, 0.0, &HashMap::new());

    let step = 10.0;
    for z_idx in 0..4 {
        let z = -15.0 + z_idx as f64 * step;
        for x_idx in 0..4 {
            let x = -15.0 + x_idx as f64 * step;
            let expected = (x * z) as f32;
            let actual = result.values[z_idx * 4 + x_idx];
            assert!(
                (actual - expected).abs() < 1e-3,
                "({}, {}): expected {}, got {}",
                x,
                z,
                expected,
                actual
            );
        }
    }
}

// ── 6. Edge cases ─────────────────────────────────────────────────

#[test]
fn grid_resolution_1() {
    let mut fields = HashMap::new();
    fields.insert("Value".to_string(), json!(7.0));
    let nodes = vec![make_node("c", "Constant", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();

    let result = evaluate_grid(&graph, 1, -10.0, 10.0, 0.0, &HashMap::new());

    assert_eq!(result.values.len(), 1);
    assert!((result.values[0] - 7.0).abs() < 1e-5);
}

#[test]
fn volume_single_slice() {
    let mut fields = HashMap::new();
    fields.insert("Value".to_string(), json!(3.14));
    let nodes = vec![make_node("c", "Constant", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();

    let result = evaluate_volume(&graph, 2, -5.0, 5.0, 64.0, 64.0, 1, &HashMap::new());

    assert_eq!(result.densities.len(), 4); // 2x2 * 1
    assert_eq!(result.y_slices, 1);
}

#[test]
fn grid_large_range() {
    let nodes = vec![make_node("cx", "CoordinateX", HashMap::new())];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("cx")).unwrap();

    let result = evaluate_grid(&graph, 4, -10000.0, 10000.0, 0.0, &HashMap::new());

    assert_eq!(result.values.len(), 16);
    for &v in &result.values {
        assert!(v.is_finite());
    }
    // Cell centers at x = -7500, -2500, 2500, 7500
    assert!((result.min_value - (-7500.0)).abs() < 1.0);
    assert!((result.max_value - 7500.0).abs() < 1.0);
}

#[test]
fn grid_zero_width_range() {
    // range_min == range_max → step = 0, all points at same position
    let nodes = vec![make_node("cx", "CoordinateX", HashMap::new())];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("cx")).unwrap();

    let result = evaluate_grid(&graph, 4, 50.0, 50.0, 0.0, &HashMap::new());

    // With zero range the step is 0, all cell centers are at range_min
    // All values should be the same (or NaN from 0/0 division)
    // Just verify no panics
    assert_eq!(result.values.len(), 16);
}

#[test]
fn content_fields_used_by_imported_value() {
    let mut fields = HashMap::new();
    fields.insert("Name".to_string(), json!("myField"));

    let nodes = vec![make_node("iv", "ImportedValue", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("iv")).unwrap();

    let mut cf = HashMap::new();
    cf.insert("myField".to_string(), 77.0);

    let result = evaluate_grid(&graph, 2, -10.0, 10.0, 0.0, &cf);

    for &v in &result.values {
        assert!(
            (v - 77.0).abs() < 1e-5,
            "ImportedValue should return 77.0, got {}",
            v
        );
    }
}

// ── 7. Determinism ────────────────────────────────────────────────

#[test]
fn grid_deterministic_across_runs() {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.01));
    fields.insert("Seed".to_string(), json!(12345));
    let nodes = vec![make_node("n", "SimplexNoise2D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("n")).unwrap();

    let r1 = evaluate_grid(&graph, 32, -64.0, 64.0, 64.0, &HashMap::new());
    let r2 = evaluate_grid(&graph, 32, -64.0, 64.0, 64.0, &HashMap::new());

    assert_eq!(r1.values.len(), r2.values.len());
    for (i, (&a, &b)) in r1.values.iter().zip(r2.values.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5_f32,
            "cell {} differs between runs: {} vs {}",
            i,
            a,
            b
        );
    }
    assert!((r1.min_value - r2.min_value).abs() < 1e-5_f32);
    assert!((r1.max_value - r2.max_value).abs() < 1e-5_f32);
}

#[test]
fn volume_deterministic_across_runs() {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.02));
    fields.insert("Seed".to_string(), json!(54321));
    let nodes = vec![make_node("n", "SimplexNoise3D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("n")).unwrap();

    let r1 = evaluate_volume(&graph, 8, -32.0, 32.0, 0.0, 128.0, 8, &HashMap::new());
    let r2 = evaluate_volume(&graph, 8, -32.0, 32.0, 0.0, 128.0, 8, &HashMap::new());

    assert_eq!(r1.densities.len(), r2.densities.len());
    for (i, (&a, &b)) in r1.densities.iter().zip(r2.densities.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5_f32,
            "voxel {} differs: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn different_seeds_produce_different_noise() {
    let mut fields_a = HashMap::new();
    fields_a.insert("Frequency".to_string(), json!(0.01));
    fields_a.insert("Seed".to_string(), json!(111));
    let graph_a = EvalGraph::from_raw(
        vec![make_node("n", "SimplexNoise2D", fields_a)],
        vec![],
        Some("n"),
    )
    .unwrap();

    let mut fields_b = HashMap::new();
    fields_b.insert("Frequency".to_string(), json!(0.01));
    fields_b.insert("Seed".to_string(), json!(222));
    let graph_b = EvalGraph::from_raw(
        vec![make_node("n", "SimplexNoise2D", fields_b)],
        vec![],
        Some("n"),
    )
    .unwrap();

    let r_a = evaluate_grid(&graph_a, 16, -64.0, 64.0, 64.0, &HashMap::new());
    let r_b = evaluate_grid(&graph_b, 16, -64.0, 64.0, 64.0, &HashMap::new());

    let mut differ = 0;
    for (&a, &b) in r_a.values.iter().zip(r_b.values.iter()) {
        if (a - b).abs() > 1e-6_f32 {
            differ += 1;
        }
    }
    // Most cells should differ between different seeds
    assert!(
        differ > r_a.values.len() / 2,
        "only {} of {} cells differ between seeds",
        differ,
        r_a.values.len()
    );
}

// ── 8. Noise variation sanity checks ──────────────────────────────

#[test]
fn noise_output_has_meaningful_range() {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.01));
    fields.insert("Seed".to_string(), json!(42));
    let nodes = vec![make_node("n", "SimplexNoise2D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("n")).unwrap();

    let result = evaluate_grid(&graph, 64, -100.0, 100.0, 64.0, &HashMap::new());

    // Noise should span a reasonable range (not all zeros)
    let range = result.max_value - result.min_value;
    assert!(
        range > 0.1,
        "noise range too small: {} (min={}, max={})",
        range,
        result.min_value,
        result.max_value
    );
}

#[test]
fn fractal_noise_respects_octave_count() {
    // More octaves → more detail → potentially wider range and more variation
    let mut fields_1oct = HashMap::new();
    fields_1oct.insert("Frequency".to_string(), json!(0.01));
    fields_1oct.insert("Octaves".to_string(), json!(1));
    fields_1oct.insert("Seed".to_string(), json!(42));

    let mut fields_6oct = HashMap::new();
    fields_6oct.insert("Frequency".to_string(), json!(0.01));
    fields_6oct.insert("Octaves".to_string(), json!(6));
    fields_6oct.insert("Seed".to_string(), json!(42));

    let g1 = EvalGraph::from_raw(
        vec![make_node("fn", "FractalNoise2D", fields_1oct)],
        vec![],
        Some("fn"),
    )
    .unwrap();

    let g6 = EvalGraph::from_raw(
        vec![make_node("fn", "FractalNoise2D", fields_6oct)],
        vec![],
        Some("fn"),
    )
    .unwrap();

    let r1 = evaluate_grid(&g1, 32, -100.0, 100.0, 0.0, &HashMap::new());
    let r6 = evaluate_grid(&g6, 32, -100.0, 100.0, 0.0, &HashMap::new());

    // 6-octave should differ from 1-octave
    let mut differ = 0;
    for (&a, &b) in r1.values.iter().zip(r6.values.iter()) {
        if (a - b).abs() > 1e-6_f32 {
            differ += 1;
        }
    }
    assert!(
        differ > r1.values.len() / 2,
        "6-octave should differ from 1-octave in most cells"
    );
}

// ── 9. Regression: grid and volume consistency ────────────────────

#[test]
fn grid_matches_volume_y_slice() {
    // A 2D grid at y=64 should produce the same values as a volume
    // with a single Y slice at y=64.
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.01));
    fields.insert("Seed".to_string(), json!(42));
    let nodes = vec![make_node("n", "SimplexNoise2D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("n")).unwrap();

    let grid = evaluate_grid(&graph, 16, -64.0, 64.0, 64.0, &HashMap::new());
    let vol = evaluate_volume(&graph, 16, -64.0, 64.0, 64.0, 64.0, 1, &HashMap::new());

    assert_eq!(grid.values.len(), vol.densities.len());
    for (i, (&gv, &vv)) in grid.values.iter().zip(vol.densities.iter()).enumerate() {
        assert!(
            (gv - vv).abs() < 1e-5_f32,
            "cell {}: grid={}, volume={}",
            i,
            gv,
            vv
        );
    }
}

#[test]
fn grid_matches_volume_middle_slice() {
    // For a 3D-invariant function (SimplexNoise2D depends only on x,z),
    // every Y slice in a volume should match the grid result.
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.01));
    fields.insert("Seed".to_string(), json!(42));
    let nodes = vec![make_node("n", "SimplexNoise2D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("n")).unwrap();

    let vol = evaluate_volume(&graph, 8, -32.0, 32.0, 0.0, 100.0, 5, &HashMap::new());
    let n = 8usize;

    // All 5 slices should be identical (since SimplexNoise2D ignores Y)
    let first_slice = &vol.densities[0..n * n];
    for yi in 1..5 {
        let slice = &vol.densities[yi * n * n..(yi + 1) * n * n];
        for (i, (&a, &b)) in first_slice.iter().zip(slice.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5_f32,
                "slice {} cell {} differs from slice 0: {} vs {}",
                yi,
                i,
                b,
                a
            );
        }
    }
}

// ── 10. Timing sanity (not a strict benchmark, just "doesn't hang") ──

#[test]
fn grid_128_completes_in_reasonable_time() {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.01));
    fields.insert("Octaves".to_string(), json!(4));
    fields.insert("Lacunarity".to_string(), json!(2.0));
    fields.insert("Gain".to_string(), json!(0.5));
    fields.insert("Seed".to_string(), json!(42));
    let nodes = vec![make_node("fn", "FractalNoise2D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("fn")).unwrap();

    let start = std::time::Instant::now();
    let result = evaluate_grid(&graph, 128, -64.0, 64.0, 64.0, &HashMap::new());
    let elapsed = start.elapsed();

    assert_eq!(result.values.len(), 128 * 128);
    // Should complete well under 5 seconds even on slow CI
    assert!(
        elapsed.as_secs() < 5,
        "grid 128x128 took {:?}, too slow",
        elapsed
    );
    eprintln!("grid 128x128 fractal_2d_4oct: {:?}", elapsed);
}

#[test]
fn volume_32x32x32_completes_in_reasonable_time() {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.01));
    fields.insert("Octaves".to_string(), json!(4));
    fields.insert("Lacunarity".to_string(), json!(2.0));
    fields.insert("Gain".to_string(), json!(0.5));
    fields.insert("Seed".to_string(), json!(42));
    let nodes = vec![make_node("fn", "FractalNoise2D", fields)];
    let graph = EvalGraph::from_raw(nodes, vec![], Some("fn")).unwrap();

    let start = std::time::Instant::now();
    let result = evaluate_volume(&graph, 32, -64.0, 64.0, 0.0, 256.0, 32, &HashMap::new());
    let elapsed = start.elapsed();

    assert_eq!(result.densities.len(), 32 * 32 * 32);
    assert!(
        elapsed.as_secs() < 10,
        "volume 32x32x32 took {:?}, too slow",
        elapsed
    );
    eprintln!("volume 32x32x32 fractal_2d_4oct: {:?}", elapsed);
}

#[test]
fn ipc_roundtrip_128_completes_in_reasonable_time() {
    // Full IPC simulation: JSON string → parse → build graph → evaluate → serialize
    let request = json!({
        "nodes": [
            {"id": "fn", "data": {"type": "FractalNoise2D", "fields": {
                "Frequency": 0.01, "Octaves": 4, "Lacunarity": 2.0, "Gain": 0.5, "Seed": 42
            }}}
        ],
        "edges": [],
        "resolution": 128,
        "range_min": -64.0,
        "range_max": 64.0,
        "y_level": 64.0,
        "root_node_id": "fn",
        "content_fields": {},
    });
    let json_str = serde_json::to_string(&request).unwrap();

    let start = std::time::Instant::now();
    let result = ipc_evaluate_grid(&json_str);
    let _response_json = serde_json::to_string(&result).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(result.values.len(), 128 * 128);
    assert!(elapsed.as_secs() < 5, "IPC roundtrip took {:?}", elapsed);
    eprintln!("IPC roundtrip grid 128x128 fractal: {:?}", elapsed);
}
