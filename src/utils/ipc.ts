import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export interface AssetPackData {
  path: string;
  assets: Record<string, unknown>;
}

export interface DirectoryEntryData {
  name: string;
  path: string;
  is_dir: boolean;
  children?: DirectoryEntryData[];
}

export interface EvaluateRequest {
  graph: unknown;
  resolution: number;
  range_min: number;
  range_max: number;
  y_level: number;
}

export interface EvaluateResponse {
  values: number[];
  resolution: number;
  min_value: number;
  max_value: number;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  files_checked: number;
}

export interface ValidationError {
  file: string;
  field: string;
  message: string;
  severity: "Error" | "Warning" | "Info";
}

export async function openAssetPack(path: string): Promise<AssetPackData> {
  return invoke<AssetPackData>("open_asset_pack", { path });
}

export async function saveAssetPack(pack: AssetPackData): Promise<void> {
  return invoke("save_asset_pack", { pack });
}

export async function readAssetFile(path: string): Promise<unknown> {
  return invoke("read_asset_file", { path });
}

export async function writeAssetFile(path: string, content: unknown): Promise<void> {
  return invoke("write_asset_file", { path, content });
}

export async function exportAssetFile(path: string, content: unknown): Promise<void> {
  return invoke("export_asset_file", { path, content });
}

export async function writeTextFile(path: string, content: string): Promise<void> {
  return invoke("write_text_file", { path, content });
}

export async function copyFile(source: string, destination: string): Promise<void> {
  return invoke("copy_file", { source, destination });
}

export async function listDirectory(path: string): Promise<DirectoryEntryData[]> {
  return invoke<DirectoryEntryData[]>("list_directory", { path });
}

export async function createFromTemplate(
  templateName: string,
  targetPath: string,
): Promise<void> {
  return invoke("create_from_template", {
    templateName,
    targetPath,
  });
}

export async function createBlankProject(targetPath: string): Promise<void> {
  return invoke("create_blank_project", { targetPath });
}

export async function evaluateDensity(request: EvaluateRequest): Promise<EvaluateResponse> {
  return invoke<EvaluateResponse>("evaluate_density", { request });
}

// ── Rust graph evaluator (Phase 3) ──

export interface GridRequest {
  nodes: unknown[];
  edges: unknown[];
  resolution: number;
  range_min: number;
  range_max: number;
  y_level: number;
  root_node_id?: string;
  content_fields?: Record<string, number>;
}

export interface GridResponse {
  values: number[];
  resolution: number;
  min_value: number;
  max_value: number;
}

export async function evaluateGrid(request: GridRequest): Promise<GridResponse> {
  return invoke<GridResponse>("evaluate_grid", { request });
}

// ── Progressive grid evaluation (Phase 7) ──

/**
 * Evaluate a grid at progressively increasing resolutions (16 → 32 → 64 → target).
 * Each intermediate result is emitted as a Tauri event (`eval_progressive_grid`).
 *
 * @param request  Same shape as GridRequest
 * @param onStep   Callback invoked for each progressive result
 * @returns A promise that resolves when all steps are complete
 */
export async function evaluateGridProgressive(
  request: GridRequest,
  onStep: (result: GridResponse) => void,
): Promise<void> {
  const unlisten: UnlistenFn = await listen<GridResponse>(
    "eval_progressive_grid",
    (event) => {
      onStep(event.payload);
    },
  );

  try {
    await invoke("evaluate_grid_progressive", { request });
  } finally {
    unlisten();
  }
}

// ── Cache management (Phase 7) ──

/** Clear all Rust-side evaluation caches (grid + volume). */
export async function clearEvalCache(): Promise<void> {
  return invoke("clear_eval_cache");
}

export interface VolumeRequest {
  nodes: unknown[];
  edges: unknown[];
  resolution: number;
  range_min: number;
  range_max: number;
  y_min: number;
  y_max: number;
  y_slices: number;
  root_node_id?: string;
  content_fields?: Record<string, number>;
}

export interface VolumeResponse {
  densities: number[];
  resolution: number;
  y_slices: number;
  min_value: number;
  max_value: number;
}

export async function evaluateVolume(request: VolumeRequest): Promise<VolumeResponse> {
  return invoke<VolumeResponse>("evaluate_volume", { request });
}

// ── Combined voxel preview (Phase 5) ──

export interface MaterialEntry {
  name: string;
  color: string;
  roughness: number;
  metalness: number;
  emissive: string;
  emissive_intensity: number;
}

export interface VoxelPreviewRequest {
  nodes: unknown[];
  edges: unknown[];
  resolution: number;
  range_min: number;
  range_max: number;
  y_min: number;
  y_max: number;
  y_slices: number;
  root_node_id?: string;
  content_fields?: Record<string, number>;
}

export interface VoxelPreviewResponse {
  densities: number[];
  resolution: number;
  y_slices: number;
  min_value: number;
  max_value: number;
  material_ids: number[] | null;
  palette: MaterialEntry[] | null;
}

export async function evaluateVoxelPreview(request: VoxelPreviewRequest): Promise<VoxelPreviewResponse> {
  return invoke<VoxelPreviewResponse>("evaluate_voxel_preview", { request });
}

// ── Full voxel mesh pipeline (Phase 6) ──

export interface FluidConfig {
  fluid_level: number;
  fluid_material_index: number;
}

export interface VoxelMeshRequest {
  nodes: unknown[];
  edges: unknown[];
  resolution: number;
  range_min: number;
  range_max: number;
  y_min: number;
  y_max: number;
  y_slices: number;
  root_node_id?: string;
  content_fields?: Record<string, number>;
  scale: [number, number, number];
  offset: [number, number, number];
  show_material_colors: boolean;
  fluid_config?: FluidConfig;
  fluid_material?: MaterialEntry;
}

export interface VoxelMeshMaterialProperties {
  roughness: number;
  metalness: number;
  emissive: string;
  emissive_intensity: number;
}

export interface VoxelMeshDataEntry {
  material_index: number;
  color: string;
  positions: number[];
  normals: number[];
  colors: number[];
  indices: number[];
  material_properties: VoxelMeshMaterialProperties;
}

export interface VoxelMeshResponse {
  meshes: VoxelMeshDataEntry[];
  densities: number[];
  resolution: number;
  y_slices: number;
  min_value: number;
  max_value: number;
  surface_voxel_count: number;
  surface_material_ids: number[];
  surface_materials: MaterialEntry[];
}

export async function evaluateVoxelMesh(request: VoxelMeshRequest): Promise<VoxelMeshResponse> {
  return invoke<VoxelMeshResponse>("evaluate_voxel_mesh", { request });
}

export async function validateAssetPack(path: string): Promise<ValidationResult> {
  return invoke<ValidationResult>("validate_asset_pack", { path });
}

// ── Bridge types ──

export interface ServerStatus {
  status: string;
  bridge_version: string;
  player_count: number;
  port: number;
  singleplayer?: boolean;
}

export interface BridgeResponse {
  success: boolean;
  message: string;
}

export interface PlayerInfo {
  name: string;
  uuid: string;
  x?: number;
  y?: number;
  z?: number;
  world?: string;
}

// ── Bridge IPC wrappers ──

export async function bridgeConnect(host: string, port: number, authToken: string): Promise<ServerStatus> {
  return invoke<ServerStatus>("bridge_connect", { host, port, authToken });
}

export async function bridgeDisconnect(): Promise<void> {
  return invoke("bridge_disconnect");
}

export async function bridgeStatus(): Promise<ServerStatus> {
  return invoke<ServerStatus>("bridge_status");
}

export async function bridgeReloadWorldgen(): Promise<BridgeResponse> {
  return invoke<BridgeResponse>("bridge_reload_worldgen");
}

export async function bridgeRegenerateChunks(x: number, z: number, radius: number): Promise<BridgeResponse> {
  return invoke<BridgeResponse>("bridge_regenerate_chunks", { x, z, radius });
}

export async function bridgeTeleport(playerName: string, x: number, y: number, z: number): Promise<BridgeResponse> {
  return invoke<BridgeResponse>("bridge_teleport", { playerName, x, y, z });
}

export async function bridgePlayerInfo(): Promise<PlayerInfo> {
  return invoke<PlayerInfo>("bridge_player_info");
}

export async function bridgeSyncFile(sourcePath: string, serverModPath: string, relativePath: string): Promise<BridgeResponse> {
  return invoke<BridgeResponse>("bridge_sync_file", { sourcePath, serverModPath, relativePath });
}

// ── World preview types ──

export interface ChunkDataResponse {
  chunkX: number;
  chunkZ: number;
  yMin: number;
  yMax: number;
  sizeX: number;
  sizeZ: number;
  blocks: number[];
  heightmap: number[];
}

export interface BlockPaletteResponse {
  palette: Record<string, string>;
}

// ── World preview IPC wrappers ──

export async function bridgeFetchPalette(): Promise<BlockPaletteResponse> {
  return invoke<BlockPaletteResponse>("bridge_fetch_palette");
}

export async function bridgeFetchChunk(chunkX: number, chunkZ: number, yMin: number, yMax: number, forceLoad: boolean = false): Promise<ChunkDataResponse> {
  return invoke<ChunkDataResponse>("bridge_fetch_chunk", { chunkX, chunkZ, yMin, yMax, forceLoad });
}
