//! Benchmarks for the Rust evaluation pipeline.
//!
//! Measures:
//!   1. Pure evaluation speed (grid & volume) at various resolutions
//!   2. Graph complexity scaling (simple → deep chains → noise-heavy)
//!   3. IPC-simulating overhead: JSON parse → graph build → evaluate → JSON serialize
//!
//! Run with:
//!   cargo bench --manifest-path src-tauri/Cargo.toml --bench eval_bench
//!
//! Results are written to `target/criterion/` with HTML reports.

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use terranova_lib::eval::graph::{EvalGraph, GraphEdge, GraphNode, NodeData};
use terranova_lib::eval::grid::evaluate_grid;
use terranova_lib::eval::mesh::build_voxel_meshes;
use terranova_lib::eval::volume::evaluate_volume;
use terranova_lib::eval::voxel::extract_surface_voxels;

// ── Graph builder helpers ──────────────────────────────────────────

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

// ── Graph factories ────────────────────────────────────────────────

/// Trivial graph: single Constant node.
fn graph_constant() -> EvalGraph {
    let mut fields = HashMap::new();
    fields.insert("Value".to_string(), json!(42.0));
    let nodes = vec![make_node("c", "Constant", fields)];
    EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap()
}

/// Simple arithmetic: Sum of two constants.
fn graph_sum_two() -> EvalGraph {
    let mut fa = HashMap::new();
    fa.insert("Value".to_string(), json!(10.0));
    let mut fb = HashMap::new();
    fb.insert("Value".to_string(), json!(20.0));
    let nodes = vec![
        make_node("a", "Constant", fa),
        make_node("b", "Constant", fb),
        make_node("s", "Sum", HashMap::new()),
    ];
    let edges = vec![
        make_edge("a", "s", Some("Inputs[0]")),
        make_edge("b", "s", Some("Inputs[1]")),
    ];
    EvalGraph::from_raw(nodes, edges, Some("s")).unwrap()
}

/// 2D simplex noise with typical worldgen parameters.
fn graph_noise_2d() -> EvalGraph {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.01));
    fields.insert("Seed".to_string(), json!(12345));
    let nodes = vec![make_node("n", "SimplexNoise2D", fields)];
    EvalGraph::from_raw(nodes, vec![], Some("n")).unwrap()
}

/// 3D simplex noise.
fn graph_noise_3d() -> EvalGraph {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.02));
    fields.insert("Seed".to_string(), json!(54321));
    let nodes = vec![make_node("n", "SimplexNoise3D", fields)];
    EvalGraph::from_raw(nodes, vec![], Some("n")).unwrap()
}

/// Fractal noise 2D — multi-octave, more expensive.
fn graph_fractal_noise_2d() -> EvalGraph {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.005));
    fields.insert("Octaves".to_string(), json!(6));
    fields.insert("Lacunarity".to_string(), json!(2.0));
    fields.insert("Gain".to_string(), json!(0.5));
    fields.insert("Seed".to_string(), json!(999));
    let nodes = vec![make_node("fn2", "FractalNoise2D", fields)];
    EvalGraph::from_raw(nodes, vec![], Some("fn2")).unwrap()
}

/// Fractal noise 3D — most expensive single-node evaluation.
fn graph_fractal_noise_3d() -> EvalGraph {
    let mut fields = HashMap::new();
    fields.insert("Frequency".to_string(), json!(0.005));
    fields.insert("Octaves".to_string(), json!(6));
    fields.insert("Lacunarity".to_string(), json!(2.0));
    fields.insert("Gain".to_string(), json!(0.5));
    fields.insert("Seed".to_string(), json!(777));
    let nodes = vec![make_node("fn3", "FractalNoise3D", fields)];
    EvalGraph::from_raw(nodes, vec![], Some("fn3")).unwrap()
}

/// Deep chain: Constant → Negate → Abs → Square → SquareRoot → Negate → Abs (7 levels)
fn graph_deep_chain() -> EvalGraph {
    let mut fields = HashMap::new();
    fields.insert("Value".to_string(), json!(5.0));
    let nodes = vec![
        make_node("c", "Constant", fields),
        make_node("n1", "Negate", HashMap::new()),
        make_node("a1", "Abs", HashMap::new()),
        make_node("sq", "Square", HashMap::new()),
        make_node("sr", "SquareRoot", HashMap::new()),
        make_node("n2", "Negate", HashMap::new()),
        make_node("a2", "Abs", HashMap::new()),
    ];
    let edges = vec![
        make_edge("c", "n1", Some("Input")),
        make_edge("n1", "a1", Some("Input")),
        make_edge("a1", "sq", Some("Input")),
        make_edge("sq", "sr", Some("Input")),
        make_edge("sr", "n2", Some("Input")),
        make_edge("n2", "a2", Some("Input")),
    ];
    EvalGraph::from_raw(nodes, edges, Some("a2")).unwrap()
}

/// Realistic terrain graph: Y-gradient + noise + clamp + linear transform.
/// Simulates a typical worldgen density pipeline.
fn graph_realistic_terrain() -> EvalGraph {
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
    EvalGraph::from_raw(nodes, edges, Some("lt")).unwrap()
}

/// Complex graph: multiple noise sources blended together (15+ nodes).
/// Simulates an advanced terrain with caves/overhangs.
fn graph_complex_multi_noise() -> EvalGraph {
    // Noise source 1: terrain shape (fractal 2D)
    let mut n1f = HashMap::new();
    n1f.insert("Frequency".to_string(), json!(0.005));
    n1f.insert("Octaves".to_string(), json!(5));
    n1f.insert("Lacunarity".to_string(), json!(2.0));
    n1f.insert("Gain".to_string(), json!(0.5));
    n1f.insert("Seed".to_string(), json!(100));

    // Noise source 2: detail (simplex 2D, high freq)
    let mut n2f = HashMap::new();
    n2f.insert("Frequency".to_string(), json!(0.05));
    n2f.insert("Seed".to_string(), json!(200));

    // Noise source 3: cave carving (3D)
    let mut n3f = HashMap::new();
    n3f.insert("Frequency".to_string(), json!(0.03));
    n3f.insert("Seed".to_string(), json!(300));

    // Y gradient for base terrain shape
    let mut yg = HashMap::new();
    yg.insert("FromY".to_string(), json!(0.0));
    yg.insert("ToY".to_string(), json!(256.0));

    // Amplitude nodes
    let mut amp1f = HashMap::new();
    amp1f.insert("Value".to_string(), json!(40.0));
    let mut amp2f = HashMap::new();
    amp2f.insert("Value".to_string(), json!(8.0));
    let mut amp3f = HashMap::new();
    amp3f.insert("Value".to_string(), json!(0.5));

    // Clamp for cave
    let mut cave_clamp = HashMap::new();
    cave_clamp.insert("Min".to_string(), json!(-0.3));
    cave_clamp.insert("Max".to_string(), json!(0.3));

    // Blend factor
    let mut blend_f = HashMap::new();
    blend_f.insert("Value".to_string(), json!(0.3));

    // Final clamp
    let mut final_clamp = HashMap::new();
    final_clamp.insert("Min".to_string(), json!(-1.0));
    final_clamp.insert("Max".to_string(), json!(1.0));

    let nodes = vec![
        make_node("n1", "FractalNoise2D", n1f),
        make_node("n2", "SimplexNoise2D", n2f),
        make_node("n3", "SimplexNoise3D", n3f),
        make_node("yg", "YGradient", yg),
        make_node("amp1", "AmplitudeConstant", amp1f),
        make_node("amp2", "AmplitudeConstant", amp2f),
        make_node("amp3", "AmplitudeConstant", amp3f),
        make_node("sum_terrain", "Sum", HashMap::new()),
        make_node("sum_detail", "Sum", HashMap::new()),
        make_node("cave_clamp", "Clamp", cave_clamp),
        make_node("blend_factor", "Constant", blend_f),
        make_node("blend", "Blend", HashMap::new()),
        make_node("product_cave", "Product", HashMap::new()),
        make_node("sum_final", "Sum", HashMap::new()),
        make_node("final_clamp", "Clamp", final_clamp),
    ];
    let edges = vec![
        // Terrain: fractal noise → amplitude → sum with Y gradient
        make_edge("n1", "amp1", Some("Input")),
        make_edge("yg", "sum_terrain", Some("Inputs[0]")),
        make_edge("amp1", "sum_terrain", Some("Inputs[1]")),
        // Detail: simplex noise → amplitude → sum with terrain
        make_edge("n2", "amp2", Some("Input")),
        make_edge("sum_terrain", "sum_detail", Some("Inputs[0]")),
        make_edge("amp2", "sum_detail", Some("Inputs[1]")),
        // Cave: 3D noise → amplitude → clamp
        make_edge("n3", "amp3", Some("Input")),
        make_edge("amp3", "cave_clamp", Some("Input")),
        // Blend terrain+detail with cave
        make_edge("sum_detail", "blend", Some("InputA")),
        make_edge("cave_clamp", "blend", Some("InputB")),
        make_edge("blend_factor", "blend", Some("Factor")),
        // Product for cave influence
        make_edge("blend", "product_cave", Some("Inputs[0]")),
        // Final sum and clamp
        make_edge("product_cave", "sum_final", Some("Inputs[0]")),
        make_edge("sum_final", "final_clamp", Some("Input")),
    ];
    EvalGraph::from_raw(nodes, edges, Some("final_clamp")).unwrap()
}

// ── JSON round-trip helpers (simulating IPC) ───────────────────────

/// Serialize a graph to JSON (simulating what the frontend sends over IPC).
fn graph_to_json(
    nodes: &[GraphNode],
    edges: &[GraphEdge],
    root_node_id: Option<&str>,
    resolution: u32,
    range_min: f64,
    range_max: f64,
    y_level: f64,
    content_fields: &HashMap<String, f64>,
) -> Value {
    let json_nodes: Vec<Value> = nodes
        .iter()
        .map(|n| {
            let mut obj = serde_json::Map::new();
            obj.insert("id".into(), json!(n.id));
            let mut data = serde_json::Map::new();
            if let Some(ref dt) = n.data.density_type {
                data.insert("type".into(), json!(dt));
            }
            data.insert("fields".into(), json!(n.data.fields));
            if n.data.is_output {
                data.insert("_outputNode".into(), json!(true));
            }
            if let Some(ref bf) = n.data.biome_field {
                data.insert("_biomeField".into(), json!(bf));
            }
            obj.insert("data".into(), Value::Object(data));
            Value::Object(obj)
        })
        .collect();

    let json_edges: Vec<Value> = edges
        .iter()
        .map(|e| {
            let mut obj = serde_json::Map::new();
            obj.insert("source".into(), json!(e.source));
            obj.insert("target".into(), json!(e.target));
            if let Some(ref th) = e.target_handle {
                obj.insert("targetHandle".into(), json!(th));
            }
            Value::Object(obj)
        })
        .collect();

    json!({
        "nodes": json_nodes,
        "edges": json_edges,
        "resolution": resolution,
        "range_min": range_min,
        "range_max": range_max,
        "y_level": y_level,
        "root_node_id": root_node_id,
        "content_fields": content_fields,
    })
}

/// Parse JSON back into graph components (simulating IPC deserialization).
fn json_to_graph(request: &Value) -> (EvalGraph, u32, f64, f64, f64, HashMap<String, f64>) {
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
    (
        graph,
        resolution,
        range_min,
        range_max,
        y_level,
        content_fields,
    )
}

/// Serialize a GridResult to JSON (simulating IPC response serialization).
fn serialize_grid_result(result: &terranova_lib::eval::grid::GridResult) -> String {
    serde_json::to_string(result).unwrap()
}

/// Serialize a VolumeResult to JSON (simulating IPC response serialization).
fn serialize_volume_result(result: &terranova_lib::eval::volume::VolumeResult) -> String {
    serde_json::to_string(result).unwrap()
}

// ── Benchmark groups ───────────────────────────────────────────────

/// Pure grid evaluation at increasing resolutions.
fn bench_grid_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_resolution");
    let resolutions: &[u32] = &[32, 64, 128, 256];

    for graph_label in &["constant", "noise_2d", "fractal_2d", "realistic_terrain"] {
        let graph = match *graph_label {
            "constant" => graph_constant(),
            "noise_2d" => graph_noise_2d(),
            "fractal_2d" => graph_fractal_noise_2d(),
            "realistic_terrain" => graph_realistic_terrain(),
            _ => unreachable!(),
        };

        for &res in resolutions {
            let param = format!("{}/{}", graph_label, res);
            group.throughput(Throughput::Elements((res as u64) * (res as u64)));
            group.bench_with_input(BenchmarkId::new("pure_eval", &param), &res, |b, &res| {
                b.iter(|| {
                    black_box(evaluate_grid(
                        &graph,
                        res,
                        -64.0,
                        64.0,
                        64.0,
                        &HashMap::new(),
                    ))
                });
            });
        }
    }

    group.finish();
}

/// Pure volume evaluation at increasing resolutions.
fn bench_volume_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("volume_resolution");
    let configs: &[(u32, u32)] = &[(16, 16), (32, 32), (64, 32), (64, 64)];

    for graph_label in &["constant", "noise_3d", "fractal_3d", "realistic_terrain"] {
        let graph = match *graph_label {
            "constant" => graph_constant(),
            "noise_3d" => graph_noise_3d(),
            "fractal_3d" => graph_fractal_noise_3d(),
            "realistic_terrain" => graph_realistic_terrain(),
            _ => unreachable!(),
        };

        for &(res, y_slices) in configs {
            let param = format!("{}/{}x{}x{}", graph_label, res, y_slices, res);
            group.throughput(Throughput::Elements(
                (res as u64) * (res as u64) * (y_slices as u64),
            ));
            group.bench_with_input(
                BenchmarkId::new("pure_eval", &param),
                &(res, y_slices),
                |b, &(res, ys)| {
                    b.iter(|| {
                        black_box(evaluate_volume(
                            &graph,
                            res,
                            -64.0,
                            64.0,
                            0.0,
                            256.0,
                            ys,
                            &HashMap::new(),
                        ))
                    });
                },
            );
        }
    }

    group.finish();
}

/// Graph complexity scaling: same resolution, increasing graph depth/complexity.
fn bench_graph_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_complexity");
    let res: u32 = 128;

    let graphs: Vec<(&str, EvalGraph)> = vec![
        ("constant", graph_constant()),
        ("sum_two", graph_sum_two()),
        ("deep_chain_7", graph_deep_chain()),
        ("noise_2d", graph_noise_2d()),
        ("fractal_2d_6oct", graph_fractal_noise_2d()),
        ("realistic_terrain", graph_realistic_terrain()),
        ("complex_multi_noise", graph_complex_multi_noise()),
    ];

    for (label, graph) in &graphs {
        group.throughput(Throughput::Elements((res as u64) * (res as u64)));
        group.bench_function(BenchmarkId::new("grid_128", label), |b| {
            b.iter(|| {
                black_box(evaluate_grid(
                    graph,
                    res,
                    -64.0,
                    64.0,
                    64.0,
                    &HashMap::new(),
                ))
            });
        });
    }

    group.finish();
}

/// IPC overhead simulation: measure JSON deser → graph build → eval → JSON ser.
/// This mirrors exactly what happens in a Tauri command call.
fn bench_ipc_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_roundtrip");

    // Build several representative request payloads
    struct BenchCase {
        label: &'static str,
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        root_id: Option<&'static str>,
        resolution: u32,
    }

    let cases = vec![
        {
            let mut f = HashMap::new();
            f.insert("Value".to_string(), json!(42.0));
            BenchCase {
                label: "constant_128",
                nodes: vec![make_node("c", "Constant", f)],
                edges: vec![],
                root_id: Some("c"),
                resolution: 128,
            }
        },
        {
            let mut f = HashMap::new();
            f.insert("Frequency".to_string(), json!(0.01));
            f.insert("Octaves".to_string(), json!(4));
            f.insert("Lacunarity".to_string(), json!(2.0));
            f.insert("Gain".to_string(), json!(0.5));
            f.insert("Seed".to_string(), json!(42));
            BenchCase {
                label: "fractal_2d_128",
                nodes: vec![make_node("fn", "FractalNoise2D", f)],
                edges: vec![],
                root_id: Some("fn"),
                resolution: 128,
            }
        },
    ];

    for case in &cases {
        let request_json = graph_to_json(
            &case.nodes,
            &case.edges,
            case.root_id,
            case.resolution,
            -64.0,
            64.0,
            64.0,
            &HashMap::new(),
        );
        let json_string = serde_json::to_string(&request_json).unwrap();

        // 1) Measure full IPC round-trip: deserialize → build graph → evaluate → serialize
        group.throughput(Throughput::Elements(
            (case.resolution as u64) * (case.resolution as u64),
        ));
        group.bench_function(BenchmarkId::new("full_roundtrip", case.label), |b| {
            b.iter(|| {
                // Deserialize (simulates Tauri JSON parsing)
                let parsed: Value = serde_json::from_str(&json_string).unwrap();
                let (graph, res, rmin, rmax, ylev, cf) = json_to_graph(&parsed);
                // Evaluate
                let result = evaluate_grid(&graph, res, rmin, rmax, ylev, &cf);
                // Serialize response (simulates Tauri JSON response)
                let _response = black_box(serialize_grid_result(&result));
            });
        });

        // 2) Measure just the deserialization step
        group.bench_function(BenchmarkId::new("deser_only", case.label), |b| {
            b.iter(|| {
                let parsed: Value = serde_json::from_str(black_box(&json_string)).unwrap();
                let (_graph, _res, _rmin, _rmax, _ylev, _cf) = black_box(json_to_graph(&parsed));
            });
        });

        // 3) Measure just the serialization step
        let pre_graph =
            EvalGraph::from_raw(case.nodes.clone(), case.edges.clone(), case.root_id).unwrap();
        let pre_result = evaluate_grid(
            &pre_graph,
            case.resolution,
            -64.0,
            64.0,
            64.0,
            &HashMap::new(),
        );
        group.bench_function(BenchmarkId::new("ser_only", case.label), |b| {
            b.iter(|| {
                black_box(serialize_grid_result(&pre_result));
            });
        });

        // 4) Measure pure evaluation only (no serde), for comparison
        group.bench_function(BenchmarkId::new("eval_only", case.label), |b| {
            b.iter(|| {
                black_box(evaluate_grid(
                    &pre_graph,
                    case.resolution,
                    -64.0,
                    64.0,
                    64.0,
                    &HashMap::new(),
                ));
            });
        });
    }

    group.finish();
}

/// IPC overhead for volume evaluation — same structure as grid but 3D.
fn bench_ipc_roundtrip_volume(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_roundtrip_volume");

    let mut noise_fields = HashMap::new();
    noise_fields.insert("Frequency".to_string(), json!(0.01));
    noise_fields.insert("Octaves".to_string(), json!(4));
    noise_fields.insert("Lacunarity".to_string(), json!(2.0));
    noise_fields.insert("Gain".to_string(), json!(0.5));
    noise_fields.insert("Seed".to_string(), json!(42));

    let nodes = vec![make_node("fn", "FractalNoise2D", noise_fields)];
    let edges: Vec<GraphEdge> = vec![];
    let resolution: u32 = 32;
    let y_slices: u32 = 32;

    // Build JSON request (same shape as VolumeRequest)
    let json_nodes: Vec<Value> = nodes
        .iter()
        .map(|n| {
            json!({
                "id": n.id,
                "data": {
                    "type": n.data.density_type,
                    "fields": n.data.fields,
                }
            })
        })
        .collect();

    let request = json!({
        "nodes": json_nodes,
        "edges": [],
        "resolution": resolution,
        "range_min": -64.0,
        "range_max": 64.0,
        "y_min": 0.0,
        "y_max": 256.0,
        "y_slices": y_slices,
        "root_node_id": "fn",
        "content_fields": {},
    });
    let json_string = serde_json::to_string(&request).unwrap();

    let total_samples = (resolution as u64) * (resolution as u64) * (y_slices as u64);
    group.throughput(Throughput::Elements(total_samples));

    // Full round-trip
    group.bench_function("full_roundtrip/fractal_32x32x32", |b| {
        b.iter(|| {
            let parsed: Value = serde_json::from_str(&json_string).unwrap();
            let nodes: Vec<GraphNode> = serde_json::from_value(parsed["nodes"].clone()).unwrap();
            let edges: Vec<GraphEdge> = serde_json::from_value(parsed["edges"].clone()).unwrap();
            let root = parsed["root_node_id"].as_str();
            let graph = EvalGraph::from_raw(nodes, edges, root).unwrap();
            let result = evaluate_volume(
                &graph,
                parsed["resolution"].as_u64().unwrap() as u32,
                parsed["range_min"].as_f64().unwrap(),
                parsed["range_max"].as_f64().unwrap(),
                parsed["y_min"].as_f64().unwrap(),
                parsed["y_max"].as_f64().unwrap(),
                parsed["y_slices"].as_u64().unwrap() as u32,
                &HashMap::new(),
            );
            black_box(serialize_volume_result(&result));
        });
    });

    // Pure evaluation only
    let graph = EvalGraph::from_raw(nodes.clone(), edges.clone(), Some("fn")).unwrap();
    group.bench_function("eval_only/fractal_32x32x32", |b| {
        b.iter(|| {
            black_box(evaluate_volume(
                &graph,
                resolution,
                -64.0,
                64.0,
                0.0,
                256.0,
                y_slices,
                &HashMap::new(),
            ));
        });
    });

    group.finish();
}

/// Response payload size benchmark — measures how large the serialized
/// responses are, which directly impacts IPC transfer time.
fn bench_response_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("response_payload_size");

    let graph = graph_realistic_terrain();

    for &res in &[32u32, 64, 128, 256] {
        let result = evaluate_grid(&graph, res, -64.0, 64.0, 64.0, &HashMap::new());
        let json_size = serialize_grid_result(&result).len();

        // Use bytes as throughput so criterion reports MB/s
        group.throughput(Throughput::Bytes(json_size as u64));
        group.bench_function(
            BenchmarkId::new("grid_serialize", format!("{}x{}_{}B", res, res, json_size)),
            |b| {
                b.iter(|| black_box(serialize_grid_result(&result)));
            },
        );
    }

    // Volume
    for &(res, ys) in &[(16u32, 16u32), (32, 32), (64, 32)] {
        let result = evaluate_volume(&graph, res, -64.0, 64.0, 0.0, 256.0, ys, &HashMap::new());
        let json_size = serialize_volume_result(&result).len();

        group.throughput(Throughput::Bytes(json_size as u64));
        group.bench_function(
            BenchmarkId::new(
                "volume_serialize",
                format!("{}x{}x{}_{}B", res, ys, res, json_size),
            ),
            |b| {
                b.iter(|| black_box(serialize_volume_result(&result)));
            },
        );
    }

    group.finish();
}

// ── Voxel extraction benchmarks ────────────────────────────────────

/// Benchmarks surface voxel extraction at various volume resolutions.
fn bench_voxel_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("voxel_extraction");
    group.sample_size(20);

    let graph = graph_realistic_terrain();

    for &(res, ys) in &[(16u32, 16u32), (32, 32), (64, 32), (64, 64)] {
        let volume = evaluate_volume(&graph, res, -64.0, 64.0, 0.0, 256.0, ys, &HashMap::new());
        let total_voxels = (res * res * ys) as usize;

        group.throughput(Throughput::Elements(total_voxels as u64));
        group.bench_function(
            BenchmarkId::new("extract_surface", format!("{}x{}x{}", res, ys, res)),
            |b| {
                b.iter(|| {
                    black_box(extract_surface_voxels(
                        &volume.densities,
                        volume.resolution,
                        volume.y_slices,
                        None,
                        None,
                        None,
                    ))
                });
            },
        );
    }

    group.finish();
}

// ── Mesh building benchmarks ───────────────────────────────────────

/// Benchmarks greedy mesh building at various volume resolutions.
fn bench_mesh_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh_building");
    group.sample_size(20);

    let graph = graph_realistic_terrain();

    for &(res, ys) in &[(16u32, 16u32), (32, 32), (64, 32)] {
        let volume = evaluate_volume(&graph, res, -64.0, 64.0, 0.0, 256.0, ys, &HashMap::new());
        let voxels = extract_surface_voxels(
            &volume.densities,
            volume.resolution,
            volume.y_slices,
            None,
            None,
            None,
        );
        let surface_count = voxels.count;

        group.throughput(Throughput::Elements(surface_count as u64));
        group.bench_function(
            BenchmarkId::new(
                "build_meshes",
                format!("{}x{}x{}_{}_surf", res, ys, res, surface_count),
            ),
            |b| {
                b.iter(|| {
                    black_box(build_voxel_meshes(
                        &voxels,
                        &volume.densities,
                        volume.resolution,
                        volume.y_slices,
                        (1.0, 1.0, 1.0),
                        (0.0, 0.0, 0.0),
                    ))
                });
            },
        );
    }

    group.finish();
}

// ── Full pipeline benchmarks ───────────────────────────────────────

/// Benchmarks the entire pipeline: volume eval → voxel extraction → mesh building.
/// This is the workload that a single `evaluate_voxel_mesh` IPC call performs.
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(10);

    let graphs: Vec<(&str, EvalGraph)> = vec![
        ("constant", graph_constant()),
        ("fractal2d", graph_fractal_noise_2d()),
        ("terrain", graph_realistic_terrain()),
    ];

    for (label, graph) in &graphs {
        for &(res, ys) in &[(16u32, 16u32), (32, 32), (64, 32)] {
            let total = (res * res * ys) as u64;
            group.throughput(Throughput::Elements(total));
            group.bench_function(
                BenchmarkId::new(*label, format!("{}x{}x{}", res, ys, res)),
                |b| {
                    b.iter(|| {
                        let volume = evaluate_volume(
                            graph,
                            res,
                            -64.0,
                            64.0,
                            0.0,
                            256.0,
                            ys,
                            &HashMap::new(),
                        );
                        let voxels = extract_surface_voxels(
                            &volume.densities,
                            volume.resolution,
                            volume.y_slices,
                            None,
                            None,
                            None,
                        );
                        let meshes = build_voxel_meshes(
                            &voxels,
                            &volume.densities,
                            volume.resolution,
                            volume.y_slices,
                            (1.0, 1.0, 1.0),
                            (0.0, 0.0, 0.0),
                        );
                        black_box(meshes)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmarks the full pipeline including JSON serialization of the mesh result,
/// simulating the complete IPC path for `evaluate_voxel_mesh`.
fn bench_full_pipeline_with_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline_ipc");
    group.sample_size(10);

    let graph = graph_realistic_terrain();

    for &(res, ys) in &[(16u32, 16u32), (32, 32), (64, 32)] {
        group.bench_function(
            BenchmarkId::new("terrain_ipc", format!("{}x{}x{}", res, ys, res)),
            |b| {
                b.iter(|| {
                    let volume =
                        evaluate_volume(&graph, res, -64.0, 64.0, 0.0, 256.0, ys, &HashMap::new());
                    let voxels = extract_surface_voxels(
                        &volume.densities,
                        volume.resolution,
                        volume.y_slices,
                        None,
                        None,
                        None,
                    );
                    let meshes = build_voxel_meshes(
                        &voxels,
                        &volume.densities,
                        volume.resolution,
                        volume.y_slices,
                        (1.0, 1.0, 1.0),
                        (0.0, 0.0, 0.0),
                    );
                    // Simulate IPC serialization of the full response
                    let json = serde_json::to_string(&meshes).unwrap();
                    black_box(json)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_grid_resolution,
    bench_volume_resolution,
    bench_graph_complexity,
    bench_ipc_roundtrip,
    bench_ipc_roundtrip_volume,
    bench_response_sizes,
    bench_voxel_extraction,
    bench_mesh_building,
    bench_full_pipeline,
    bench_full_pipeline_with_serialization,
);
criterion_main!(benches);
