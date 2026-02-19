use crate::eval::graph::{EvalGraph, GraphEdge, GraphNode};
use crate::eval::grid::GridResult;
use crate::eval::nodes::{evaluate, EvalContext};
use crate::eval::volume::VolumeResult;
use crate::noise::evaluator::DensityEvaluator;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct EvaluateRequest {
    /// The density graph as V2 JSON
    pub graph: Value,
    /// Grid resolution (e.g., 128 for 128x128)
    pub resolution: u32,
    /// World coordinate range
    pub range_min: f64,
    pub range_max: f64,
    /// Y level for 2D evaluation
    pub y_level: f64,
}

#[derive(Serialize)]
pub struct EvaluateResponse {
    /// Flattened NxN density values (row-major)
    pub values: Vec<f32>,
    /// Grid resolution
    pub resolution: u32,
    /// Min/max values in the result (for normalization)
    pub min_value: f32,
    pub max_value: f32,
}

/// Evaluate a density function graph at an NxN grid of positions.
#[tauri::command]
pub fn evaluate_density(request: EvaluateRequest) -> Result<EvaluateResponse, String> {
    let evaluator =
        DensityEvaluator::from_json(&request.graph).map_err(|e| format!("Parse error: {}", e))?;

    let n = request.resolution as usize;
    let mut values = Vec::with_capacity(n * n);
    let step = (request.range_max - request.range_min) / n as f64;

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for z_idx in 0..n {
        let z = request.range_min + (z_idx as f64 + 0.5) * step;
        for x_idx in 0..n {
            let x = request.range_min + (x_idx as f64 + 0.5) * step;
            let val = evaluator.evaluate(x, request.y_level, z) as f32;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            values.push(val);
        }
    }

    Ok(EvaluateResponse {
        values,
        resolution: request.resolution,
        min_value: min_val,
        max_value: max_val,
    })
}

/// Evaluate a React Flow graph at specific sample points using the Rust evaluator.
/// Used for parity comparison between JS and Rust evaluation pipelines.
#[tauri::command]
pub fn evaluate_points(
    nodes: Vec<Value>,
    edges: Vec<Value>,
    points: Vec<[f64; 3]>,
    root_node_id: Option<String>,
    content_fields: Option<HashMap<String, f64>>,
) -> Result<Vec<f64>, String> {
    let (graph_nodes, graph_edges) = parse_raw_graph(nodes, edges)?;
    let graph = EvalGraph::from_raw(graph_nodes, graph_edges, root_node_id.as_deref())?;

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

// ── Grid evaluation (Phase 3) ──────────────────────────────────────

#[derive(Deserialize)]
pub struct GridRequest {
    pub nodes: Vec<Value>,
    pub edges: Vec<Value>,
    pub resolution: u32,
    pub range_min: f64,
    pub range_max: f64,
    pub y_level: f64,
    pub root_node_id: Option<String>,
    pub content_fields: Option<HashMap<String, f64>>,
}

/// Evaluate a React Flow graph over an NxN 2D grid using the Rust evaluator.
/// Uses rayon for multi-core parallelism across rows.
#[tauri::command]
pub fn evaluate_grid(request: GridRequest) -> Result<GridResult, String> {
    let (graph_nodes, graph_edges) = parse_raw_graph(request.nodes, request.edges)?;
    let graph = EvalGraph::from_raw(graph_nodes, graph_edges, request.root_node_id.as_deref())?;

    Ok(crate::eval::grid::evaluate_grid(
        &graph,
        request.resolution,
        request.range_min,
        request.range_max,
        request.y_level,
        &request.content_fields.unwrap_or_default(),
    ))
}

// ── Volume evaluation (Phase 3) ────────────────────────────────────

#[derive(Deserialize)]
pub struct VolumeRequest {
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

/// Evaluate a React Flow graph over a 3D volume using the Rust evaluator.
/// Uses rayon for multi-core parallelism across Y-slices.
#[tauri::command]
pub fn evaluate_volume(request: VolumeRequest) -> Result<VolumeResult, String> {
    let (graph_nodes, graph_edges) = parse_raw_graph(request.nodes, request.edges)?;
    let graph = EvalGraph::from_raw(graph_nodes, graph_edges, request.root_node_id.as_deref())?;

    Ok(crate::eval::volume::evaluate_volume(
        &graph,
        request.resolution,
        request.range_min,
        request.range_max,
        request.y_min,
        request.y_max,
        request.y_slices,
        &request.content_fields.unwrap_or_default(),
    ))
}

// ── Shared helpers ─────────────────────────────────────────────────

/// Parse raw JSON node/edge arrays into typed graph structures.
fn parse_raw_graph(
    nodes: Vec<Value>,
    edges: Vec<Value>,
) -> Result<(Vec<GraphNode>, Vec<GraphEdge>), String> {
    let graph_nodes: Vec<GraphNode> = nodes
        .into_iter()
        .map(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    let graph_edges: Vec<GraphEdge> = edges
        .into_iter()
        .map(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((graph_nodes, graph_edges))
}
