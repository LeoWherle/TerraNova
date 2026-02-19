use crate::eval::cache::{hash_grid_request, hash_volume_request, EvalCache};
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
use std::sync::Arc;
use tauri::{Emitter, State};

// NOTE: Arc<T> implements Serialize when T: Serialize, so we can return
// Arc<GridResult> / Arc<VolumeResult> from commands. This avoids cloning
// potentially multi-MB Vec<f32> buffers on cache hits.

// ── Legacy density evaluator (synchronous, lightweight) ────────────

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
pub async fn evaluate_density(request: EvaluateRequest) -> Result<EvaluateResponse, String> {
    tokio::task::spawn_blocking(move || {
        let evaluator = DensityEvaluator::from_json(&request.graph)
            .map_err(|e| format!("Parse error: {}", e))?;

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
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Evaluate a React Flow graph at specific sample points using the Rust evaluator.
/// Used for parity comparison between JS and Rust evaluation pipelines.
#[tauri::command]
pub async fn evaluate_points(
    nodes: Vec<Value>,
    edges: Vec<Value>,
    points: Vec<[f64; 3]>,
    root_node_id: Option<String>,
    content_fields: Option<HashMap<String, f64>>,
) -> Result<Vec<f64>, String> {
    tokio::task::spawn_blocking(move || {
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
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
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
/// Results are cached via LRU so repeated identical requests return instantly.
///
/// Runs on a blocking thread pool so the UI stays responsive.
#[tauri::command]
pub async fn evaluate_grid(
    request: GridRequest,
    cache: State<'_, Arc<EvalCache>>,
) -> Result<Arc<GridResult>, String> {
    let cache = cache.inner().clone();

    tokio::task::spawn_blocking(move || {
        let (graph_nodes, graph_edges) = parse_raw_graph(request.nodes, request.edges)?;
        let content_fields = request.content_fields.unwrap_or_default();

        let hash = hash_grid_request(
            &graph_nodes,
            &graph_edges,
            request.resolution,
            request.range_min,
            request.range_max,
            request.y_level,
            request.root_node_id.as_deref(),
            &content_fields,
        );

        // Cache hit — return Arc (cheap ref-count bump, no Vec clone)
        if let Some(cached) = cache.get_grid(hash) {
            if import_meta_dev() {
                eprintln!("[cache] grid hit hash={:#018x}", hash);
            }
            return Ok(cached);
        }

        let graph = EvalGraph::from_raw(graph_nodes, graph_edges, request.root_node_id.as_deref())?;

        let result = crate::eval::grid::evaluate_grid(
            &graph,
            request.resolution,
            request.range_min,
            request.range_max,
            request.y_level,
            &content_fields,
        );

        let arc = Arc::new(result);
        cache.put_grid_arc(hash, arc.clone());
        Ok(arc)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
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
/// Results are cached via LRU.
///
/// Runs on a blocking thread pool so the UI stays responsive.
#[tauri::command]
pub async fn evaluate_volume(
    request: VolumeRequest,
    cache: State<'_, Arc<EvalCache>>,
) -> Result<Arc<VolumeResult>, String> {
    let cache = cache.inner().clone();

    tokio::task::spawn_blocking(move || {
        let (graph_nodes, graph_edges) = parse_raw_graph(request.nodes, request.edges)?;
        let content_fields = request.content_fields.unwrap_or_default();

        let hash = hash_volume_request(
            &graph_nodes,
            &graph_edges,
            request.resolution,
            request.range_min,
            request.range_max,
            request.y_min,
            request.y_max,
            request.y_slices,
            request.root_node_id.as_deref(),
            &content_fields,
        );

        // Cache hit — return Arc (cheap ref-count bump, no Vec clone)
        if let Some(cached) = cache.get_volume(hash) {
            if import_meta_dev() {
                eprintln!("[cache] volume hit hash={:#018x}", hash);
            }
            return Ok(cached);
        }

        let graph = EvalGraph::from_raw(graph_nodes, graph_edges, request.root_node_id.as_deref())?;

        let result = crate::eval::volume::evaluate_volume(
            &graph,
            request.resolution,
            request.range_min,
            request.range_max,
            request.y_min,
            request.y_max,
            request.y_slices,
            &content_fields,
        );

        let arc = Arc::new(result);
        cache.put_volume_arc(hash, arc.clone());
        Ok(arc)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

// ── Combined voxel preview (Phase 5) ───────────────────────────────

/// Evaluate both density volume and material graph in one IPC call.
/// Runs density evaluation first, then (if material nodes are present)
/// evaluates the material graph over the resulting volume.
///
/// Runs on a blocking thread pool so the UI stays responsive.
#[tauri::command]
pub async fn evaluate_voxel_preview(
    request: VoxelPreviewRequest,
) -> Result<VoxelPreviewResult, String> {
    tokio::task::spawn_blocking(move || {
        let (graph_nodes, graph_edges) =
            parse_raw_graph(request.nodes.clone(), request.edges.clone())?;
        let density_graph =
            EvalGraph::from_raw(graph_nodes, graph_edges, request.root_node_id.as_deref())?;
        let content_fields = request.content_fields.clone().unwrap_or_default();

        Ok(crate::eval::material::evaluate_voxel_preview(
            &request,
            &density_graph,
            &content_fields,
        ))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
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
/// Returns ready-to-render mesh buffers as JSON.
///
/// NOTE: The primary frontend path now uses `evaluate_volume` (smaller IPC
/// payload) and builds meshes in JS. This command is kept as a convenience
/// for callers that want the full pipeline in one round-trip.
///
/// Runs on a blocking thread pool so the UI stays responsive.
#[tauri::command]
pub async fn evaluate_voxel_mesh(
    request: VoxelMeshRequest,
    cache: State<'_, Arc<EvalCache>>,
) -> Result<VoxelMeshResponse, String> {
    let cache = cache.inner().clone();

    tokio::task::spawn_blocking(move || {
        let (graph_nodes, graph_edges) =
            parse_raw_graph(request.nodes.clone(), request.edges.clone())?;
        let density_graph = EvalGraph::from_raw(
            graph_nodes.clone(),
            graph_edges.clone(),
            request.root_node_id.as_deref(),
        )?;
        let content_fields = request.content_fields.clone().unwrap_or_default();

        // 1. Evaluate density volume (check cache first)
        let volume_hash = hash_volume_request(
            &graph_nodes,
            &graph_edges,
            request.resolution,
            request.range_min,
            request.range_max,
            request.y_min,
            request.y_max,
            request.y_slices,
            request.root_node_id.as_deref(),
            &content_fields,
        );

        // Volume is wrapped in Arc — cache hit is a cheap ref-count bump
        let volume: Arc<VolumeResult> = if let Some(cached) = cache.get_volume(volume_hash) {
            if import_meta_dev() {
                eprintln!("[cache] voxel-mesh volume hit hash={:#018x}", volume_hash);
            }
            cached
        } else {
            let v = crate::eval::volume::evaluate_volume(
                &density_graph,
                request.resolution,
                request.range_min,
                request.range_max,
                request.y_min,
                request.y_max,
                request.y_slices,
                &content_fields,
            );
            let arc = Arc::new(v);
            cache.put_volume_arc(volume_hash, arc.clone());
            arc
        };

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
            let existing_idx = p.iter().position(|m| m.name == fluid_mat.name);
            let fluid_idx = match existing_idx {
                Some(idx) => idx as u8,
                None => {
                    let idx = p.len() as u8;
                    p.push(fluid_mat.clone());
                    idx
                }
            };
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
            densities: volume.densities.clone(),
            resolution: volume.resolution,
            y_slices: volume.y_slices,
            min_value: volume.min_value,
            max_value: volume.max_value,
            surface_voxel_count: voxels.count,
            surface_material_ids: voxels.material_ids,
            surface_materials: voxels.materials,
        })
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

// ── Progressive grid streaming (Phase 7) ───────────────────────────

/// Progressive grid evaluation request.
/// Same fields as GridRequest, plus the target resolution steps are computed
/// automatically: 16 → 32 → 64 → target.
#[derive(Deserialize)]
pub struct ProgressiveGridRequest {
    pub nodes: Vec<Value>,
    pub edges: Vec<Value>,
    pub resolution: u32,
    pub range_min: f64,
    pub range_max: f64,
    pub y_level: f64,
    pub root_node_id: Option<String>,
    pub content_fields: Option<HashMap<String, f64>>,
}

/// Evaluate a grid at progressively increasing resolutions, emitting each
/// intermediate result as a Tauri event (`eval_progressive_grid`).
///
/// The frontend listens for `eval_progressive_grid` events and updates the
/// preview as each step arrives, giving instant visual feedback even for
/// expensive graphs.
///
/// Resolutions: 16 → 32 → 64 → target (duplicates and steps > target are filtered).
/// Each step checks the LRU cache before evaluating.
///
/// Runs on a blocking thread pool so the UI stays responsive.
#[tauri::command]
pub async fn evaluate_grid_progressive(
    app: tauri::AppHandle,
    request: ProgressiveGridRequest,
    cache: State<'_, Arc<EvalCache>>,
) -> Result<(), String> {
    let cache = cache.inner().clone();

    tokio::task::spawn_blocking(move || {
        let (graph_nodes, graph_edges) = parse_raw_graph(request.nodes, request.edges)?;
        let content_fields = request.content_fields.unwrap_or_default();

        // Build progressive resolution steps: 16 → 32 → 64 → target
        let mut steps: Vec<u32> = vec![16, 32, 64, request.resolution]
            .into_iter()
            .filter(|&r| r <= request.resolution && r > 0)
            .collect();
        steps.sort();
        steps.dedup();

        let graph = EvalGraph::from_raw(
            graph_nodes.clone(),
            graph_edges.clone(),
            request.root_node_id.as_deref(),
        )?;

        for resolution in steps {
            let hash = hash_grid_request(
                &graph_nodes,
                &graph_edges,
                resolution,
                request.range_min,
                request.range_max,
                request.y_level,
                request.root_node_id.as_deref(),
                &content_fields,
            );

            // Arc-based: cache hit is a cheap ref-count bump
            let result: Arc<GridResult> = if let Some(cached) = cache.get_grid(hash) {
                if import_meta_dev() {
                    eprintln!(
                        "[progressive] cache hit res={} hash={:#018x}",
                        resolution, hash
                    );
                }
                cached
            } else {
                let r = crate::eval::grid::evaluate_grid(
                    &graph,
                    resolution,
                    request.range_min,
                    request.range_max,
                    request.y_level,
                    &content_fields,
                );
                let arc = Arc::new(r);
                cache.put_grid_arc(hash, arc.clone());
                arc
            };

            // Arc<T: Serialize> serializes as T
            app.emit("eval_progressive_grid", &*result)
                .map_err(|e| e.to_string())?;
        }

        Ok(())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

// ── Cache management (Phase 7) ─────────────────────────────────────

/// Clear all evaluation caches. Useful when the user explicitly wants to
/// force re-evaluation (e.g. after toggling a global setting).
#[tauri::command]
pub fn clear_eval_cache(cache: State<'_, Arc<EvalCache>>) -> Result<(), String> {
    cache.clear();
    Ok(())
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

/// Returns true in debug/dev builds for verbose cache logging.
/// In release builds this always returns false and the logging is
/// compiled away.
#[inline]
fn import_meta_dev() -> bool {
    cfg!(debug_assertions)
}
