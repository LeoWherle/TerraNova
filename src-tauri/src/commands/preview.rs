use crate::eval::graph::{EvalGraph, GraphEdge, GraphNode};
use crate::eval::grid::GridResult;
use crate::eval::material::{VoxelPreviewRequest, VoxelPreviewResult};
use crate::eval::mesh::{self, VoxelMeshData};
use crate::eval::nodes::{evaluate, EvalContext};
use crate::eval::volume::VolumeResult;
use crate::eval::voxel::{self, FluidConfig, VoxelMaterial};
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

// ── Combined voxel preview (Phase 5) ───────────────────────────────

/// Evaluate both density volume and material graph in one IPC call.
/// Runs density evaluation first, then (if material nodes are present)
/// evaluates the material graph over the resulting volume.
#[tauri::command]
pub fn evaluate_voxel_preview(request: VoxelPreviewRequest) -> Result<VoxelPreviewResult, String> {
    let (graph_nodes, graph_edges) = parse_raw_graph(request.nodes.clone(), request.edges.clone())?;
    let density_graph =
        EvalGraph::from_raw(graph_nodes, graph_edges, request.root_node_id.as_deref())?;
    let content_fields = request.content_fields.clone().unwrap_or_default();

    Ok(crate::eval::material::evaluate_voxel_preview(
        &request,
        &density_graph,
        &content_fields,
    ))
}

// ── Full voxel mesh pipeline (Phase 6) ─────────────────────────────

/// Request for the full voxel mesh pipeline: density → materials → extraction → meshing.
#[derive(Deserialize)]
pub struct VoxelMeshRequest {
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
    /// Scene-space scale factors (scale_x, scale_y, scale_z).
    pub scale: [f32; 3],
    /// Scene-space offsets (offset_x, offset_y, offset_z).
    pub offset: [f32; 3],
    /// Whether to evaluate and apply material colors.
    pub show_material_colors: bool,
    /// Optional fluid configuration (water/lava).
    pub fluid_config: Option<FluidConfig>,
    /// Optional fluid material entry to add to the palette if fluid_config is set.
    pub fluid_material: Option<VoxelMaterial>,
}

/// Response from the full voxel mesh pipeline.
#[derive(Serialize)]
pub struct VoxelMeshResponse {
    /// Per-material mesh data, ready to upload to GPU buffers.
    pub meshes: Vec<VoxelMeshData>,
    /// Volume densities (kept for downstream use like auto-fit Y).
    pub densities: Vec<f32>,
    /// Volume resolution.
    pub resolution: u32,
    /// Number of Y slices.
    pub y_slices: u32,
    /// Min density value.
    pub min_value: f32,
    /// Max density value.
    pub max_value: f32,
    /// Number of surface voxels extracted.
    pub surface_voxel_count: u32,
    /// Material IDs per surface voxel (for store state).
    pub surface_material_ids: Vec<u8>,
    /// Material palette used.
    pub surface_materials: Vec<VoxelMaterial>,
}

/// Full voxel mesh pipeline in a single IPC call:
/// 1. Evaluate density volume (rayon-parallel).
/// 2. Evaluate material graph (if present).
/// 3. Extract surface voxels.
/// 4. Build greedy-meshed geometry with AO + face shading.
///
/// Returns ready-to-render mesh buffers, eliminating all JS computation.
#[tauri::command]
pub fn evaluate_voxel_mesh(request: VoxelMeshRequest) -> Result<VoxelMeshResponse, String> {
    let (graph_nodes, graph_edges) = parse_raw_graph(request.nodes.clone(), request.edges.clone())?;
    let density_graph =
        EvalGraph::from_raw(graph_nodes, graph_edges, request.root_node_id.as_deref())?;
    let content_fields = request.content_fields.clone().unwrap_or_default();

    // 1. Evaluate density volume
    let volume = crate::eval::volume::evaluate_volume(
        &density_graph,
        request.resolution,
        request.range_min,
        request.range_max,
        request.y_min,
        request.y_max,
        request.y_slices,
        &content_fields,
    );

    // 2. Evaluate material graph (if material nodes exist and colors are wanted)
    let mut material_ids: Option<Vec<u8>> = None;
    let mut palette: Option<Vec<VoxelMaterial>> = None;

    if request.show_material_colors {
        let has_material_nodes = request.nodes.iter().any(|n| {
            n.get("type")
                .and_then(|v| v.as_str())
                .map(|t| t.starts_with("Material:"))
                .unwrap_or(false)
        });

        if has_material_nodes {
            let mat_result = crate::eval::material::evaluate_material_graph(
                &request.nodes,
                &request.edges,
                &volume.densities,
                request.resolution,
                request.y_slices,
                request.range_min,
                request.range_max,
                request.y_min,
                request.y_max,
                Some(&density_graph),
                &content_fields,
            );
            if let Some(mr) = mat_result {
                material_ids = Some(mr.material_ids);
                palette = Some(
                    mr.palette
                        .into_iter()
                        .map(|m| VoxelMaterial {
                            name: m.name,
                            color: m.color,
                            roughness: m.roughness,
                            metalness: m.metalness,
                            emissive: m.emissive,
                            emissive_intensity: m.emissive_intensity,
                        })
                        .collect(),
                );
            }
        }
    }

    // Add fluid material to palette if configured
    let mut final_palette = palette.clone();
    let mut fluid_cfg = request.fluid_config.clone();
    if let (Some(fc), Some(fluid_mat)) = (&fluid_cfg, &request.fluid_material) {
        let p = final_palette.get_or_insert_with(Vec::new);
        // Check if the fluid material already exists
        let existing_idx = p.iter().position(|m| m.name == fluid_mat.name);
        let fluid_idx = match existing_idx {
            Some(idx) => idx as u8,
            None => {
                let idx = p.len() as u8;
                p.push(fluid_mat.clone());
                idx
            }
        };
        // Update fluid config with the correct palette index
        fluid_cfg = Some(FluidConfig {
            fluid_level: fc.fluid_level,
            fluid_material_index: fluid_idx,
        });
    }

    // 3. Extract surface voxels
    let voxels = voxel::extract_surface_voxels(
        &volume.densities,
        volume.resolution,
        volume.y_slices,
        material_ids.as_deref(),
        final_palette.as_deref(),
        fluid_cfg.as_ref(),
    );

    // 4. Build greedy-meshed geometry
    let meshes = mesh::build_voxel_meshes(
        &voxels,
        &volume.densities,
        volume.resolution,
        volume.y_slices,
        (request.scale[0], request.scale[1], request.scale[2]),
        (request.offset[0], request.offset[1], request.offset[2]),
    );

    Ok(VoxelMeshResponse {
        meshes,
        densities: volume.densities,
        resolution: volume.resolution,
        y_slices: volume.y_slices,
        min_value: volume.min_value,
        max_value: volume.max_value,
        surface_voxel_count: voxels.count,
        surface_material_ids: voxels.material_ids,
        surface_materials: voxels.materials,
    })
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
