// eval/mesh.rs — Greedy voxel mesh builder
//
// Ports `buildVoxelMeshes` from `src/utils/voxelMeshBuilder.ts`.
// Produces per-material mesh data (positions, normals, colors, indices)
// with ambient occlusion, directional face shading, greedy quad merging,
// and AO-aware triangle winding.
//
// Layout: densities[y * n * n + z * n + x]

use crate::eval::voxel::{is_solid, VoxelData};
use serde::Serialize;

// ── Constants ───────────────────────────────────────────────────────

/// Deterministic hash primes (mirrors TS constants in `src/constants.ts`).
const HASH_PRIME_A: i32 = 374761393;
const HASH_PRIME_B: i32 = 668265263;
const HASH_PRIME_C: i32 = 1103515245;
const HASH_PRIME_D: i32 = 1274126177;

/// Per-face brightness multipliers: +Y, -Y, +X, -X, +Z, -Z
const FACE_BRIGHTNESS: [f32; 6] = [1.0, 0.6, 0.80, 0.72, 0.85, 0.68];

/// AO curve: maps AO level (0–3) to brightness multiplier.
const AO_CURVE: [f32; 4] = [1.0, 0.78, 0.60, 0.45];

// ── Face definitions ────────────────────────────────────────────────

/// Face direction (neighbor to check for occlusion).
type Dir = [i32; 3];
/// 4 vertex positions (offsets from block origin).
type Verts = [[i32; 3]; 4];
/// For each of 4 vertices: [edge1, edge2, corner] neighbor offsets for AO.
type AoOffsets = [[[i32; 3]; 3]; 4];

struct FaceDef {
    dir: Dir,
    vertices: Verts,
    ao: AoOffsets,
}

static FACES: [FaceDef; 6] = [
    // +Y (top)
    FaceDef {
        dir: [0, 1, 0],
        vertices: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
        ao: [
            [[-1, 1, 0], [0, 1, -1], [-1, 1, -1]],
            [[-1, 1, 0], [0, 1, 1], [-1, 1, 1]],
            [[1, 1, 0], [0, 1, 1], [1, 1, 1]],
            [[1, 1, 0], [0, 1, -1], [1, 1, -1]],
        ],
    },
    // -Y (bottom)
    FaceDef {
        dir: [0, -1, 0],
        vertices: [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]],
        ao: [
            [[-1, -1, 0], [0, -1, 1], [-1, -1, 1]],
            [[-1, -1, 0], [0, -1, -1], [-1, -1, -1]],
            [[1, -1, 0], [0, -1, -1], [1, -1, -1]],
            [[1, -1, 0], [0, -1, 1], [1, -1, 1]],
        ],
    },
    // +X
    FaceDef {
        dir: [1, 0, 0],
        vertices: [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
        ao: [
            [[1, -1, 0], [1, 0, -1], [1, -1, -1]],
            [[1, 1, 0], [1, 0, -1], [1, 1, -1]],
            [[1, 1, 0], [1, 0, 1], [1, 1, 1]],
            [[1, -1, 0], [1, 0, 1], [1, -1, 1]],
        ],
    },
    // -X
    FaceDef {
        dir: [-1, 0, 0],
        vertices: [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]],
        ao: [
            [[-1, -1, 0], [-1, 0, 1], [-1, -1, 1]],
            [[-1, 1, 0], [-1, 0, 1], [-1, 1, 1]],
            [[-1, 1, 0], [-1, 0, -1], [-1, 1, -1]],
            [[-1, -1, 0], [-1, 0, -1], [-1, -1, -1]],
        ],
    },
    // +Z
    FaceDef {
        dir: [0, 0, 1],
        vertices: [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]],
        ao: [
            [[1, 0, 1], [0, -1, 1], [1, -1, 1]],
            [[1, 0, 1], [0, 1, 1], [1, 1, 1]],
            [[-1, 0, 1], [0, 1, 1], [-1, 1, 1]],
            [[-1, 0, 1], [0, -1, 1], [-1, -1, 1]],
        ],
    },
    // -Z
    FaceDef {
        dir: [0, 0, -1],
        vertices: [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ao: [
            [[-1, 0, -1], [0, -1, -1], [-1, -1, -1]],
            [[-1, 0, -1], [0, 1, -1], [-1, 1, -1]],
            [[1, 0, -1], [0, 1, -1], [1, 1, -1]],
            [[1, 0, -1], [0, -1, -1], [1, -1, -1]],
        ],
    },
];

// ── Axis mapping for greedy meshing ─────────────────────────────────

/// Maps (slice, u, v) coordinates into block-space (bx, by, bz) and defines
/// how to iterate over the 2D face plane for each of the 6 face directions.
struct FaceAxisMap {
    /// Which vertex component (0=x, 1=y, 2=z) aligns with the U axis.
    u_vert_component: usize,
    /// Which vertex component aligns with the V axis.
    v_vert_component: usize,
}

/// Convert (slice, u, v) to (bx, by, bz) for a given face index.
#[inline]
fn to_bxyz(fi: usize, slice: i32, u: i32, v: i32) -> (i32, i32, i32) {
    match fi {
        0 | 1 => (u, slice, v), // ±Y: slice=Y, u=X, v=Z
        2 | 3 => (slice, v, u), // ±X: slice=X, u=Z, v=Y
        4 | 5 => (u, v, slice), // ±Z: slice=Z, u=X, v=Y
        _ => unreachable!(),
    }
}

/// Compute the neighbor position in the face direction.
#[inline]
fn neighbor_offset(fi: usize, slice: i32, u: i32, v: i32) -> (i32, i32, i32) {
    match fi {
        0 => (u, slice + 1, v), // +Y
        1 => (u, slice - 1, v), // -Y
        2 => (slice + 1, v, u), // +X
        3 => (slice - 1, v, u), // -X
        4 => (u, v, slice + 1), // +Z
        5 => (u, v, slice - 1), // -Z
        _ => unreachable!(),
    }
}

/// Returns (slice_count, u_count, v_count) for each face direction.
#[inline]
fn axis_counts(fi: usize, n: i32, ys: i32) -> (i32, i32, i32) {
    match fi {
        0 | 1 => (ys, n, n), // ±Y: slice=Y, u=X, v=Z
        2 | 3 => (n, n, ys), // ±X: slice=X, u=Z, v=Y
        4 | 5 => (n, n, ys), // ±Z: slice=Z, u=X, v=Y
        _ => unreachable!(),
    }
}

static FACE_AXIS_MAPS: [FaceAxisMap; 6] = [
    FaceAxisMap {
        u_vert_component: 0,
        v_vert_component: 2,
    }, // +Y: u=X, v=Z
    FaceAxisMap {
        u_vert_component: 0,
        v_vert_component: 2,
    }, // -Y: u=X, v=Z
    FaceAxisMap {
        u_vert_component: 2,
        v_vert_component: 1,
    }, // +X: u=Z, v=Y
    FaceAxisMap {
        u_vert_component: 2,
        v_vert_component: 1,
    }, // -X: u=Z, v=Y
    FaceAxisMap {
        u_vert_component: 0,
        v_vert_component: 1,
    }, // +Z: u=X, v=Y
    FaceAxisMap {
        u_vert_component: 0,
        v_vert_component: 1,
    }, // -Z: u=X, v=Y
];

// ── Output types ────────────────────────────────────────────────────

/// Material-specific PBR properties for rendering.
#[derive(Debug, Clone, Serialize)]
pub struct MaterialProperties {
    pub roughness: f32,
    pub metalness: f32,
    pub emissive: String,
    pub emissive_intensity: f32,
}

/// Mesh data for a single material, ready to upload to a GPU buffer.
#[derive(Debug, Serialize)]
pub struct VoxelMeshData {
    pub material_index: u32,
    pub color: String,
    /// Vertex positions (3 floats per vertex).
    pub positions: Vec<f32>,
    /// Vertex normals (3 floats per vertex).
    pub normals: Vec<f32>,
    /// Vertex colors (3 floats per vertex, pre-multiplied with AO + face brightness).
    pub colors: Vec<f32>,
    /// Triangle indices (3 per triangle, 6 per quad).
    pub indices: Vec<u32>,
    /// PBR material properties.
    pub material_properties: MaterialProperties,
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Deterministic block hash for per-block color jitter.
#[inline]
fn block_hash(x: i32, y: i32, z: i32) -> i32 {
    let mut h = (x.wrapping_mul(HASH_PRIME_A))
        .wrapping_add(y.wrapping_mul(HASH_PRIME_B))
        .wrapping_add(z.wrapping_mul(HASH_PRIME_D));
    h = (h ^ (h >> 13)).wrapping_mul(HASH_PRIME_C);
    (h ^ (h >> 16)) & 0x7fffffff
}

/// Returns jitter in [-0.08, +0.08] based on block position.
#[inline]
fn block_jitter(x: i32, y: i32, z: i32) -> f32 {
    ((block_hash(x, y, z) % 1000) as f32 / 1000.0 - 0.5) * 0.16
}

/// Parse "#RRGGBB" hex color to (r, g, b) in [0..1].
fn hex_to_rgb(hex: &str) -> [f32; 3] {
    let h = hex.trim_start_matches('#');
    if h.len() < 6 {
        return [0.5, 0.5, 0.5]; // fallback gray
    }
    let r = u8::from_str_radix(&h[0..2], 16).unwrap_or(128) as f32 / 255.0;
    let g = u8::from_str_radix(&h[2..4], 16).unwrap_or(128) as f32 / 255.0;
    let b = u8::from_str_radix(&h[4..6], 16).unwrap_or(128) as f32 / 255.0;
    [r, g, b]
}

/// Compute vertex ambient occlusion level (0–3).
/// Uses the standard voxel AO algorithm: check two edge neighbors and one corner.
#[inline]
fn compute_vertex_ao(
    densities: &[f32],
    bx: i32,
    by: i32,
    bz: i32,
    edge1: &[i32; 3],
    edge2: &[i32; 3],
    corner: &[i32; 3],
    n: i32,
    ys: i32,
) -> u8 {
    let s1 = is_solid(
        densities,
        bx + edge1[0],
        by + edge1[1],
        bz + edge1[2],
        n,
        ys,
    ) as u8;
    let s2 = is_solid(
        densities,
        bx + edge2[0],
        by + edge2[1],
        bz + edge2[2],
        n,
        ys,
    ) as u8;
    let sc = is_solid(
        densities,
        bx + corner[0],
        by + corner[1],
        bz + corner[2],
        n,
        ys,
    ) as u8;

    if s1 == 1 && s2 == 1 {
        3 // corner is occluded by both edges
    } else {
        s1 + s2 + sc
    }
}

// ── Mask entry for greedy meshing ───────────────────────────────────

#[derive(Clone)]
struct MaskEntry {
    mat_id: u8,
    ao: [u8; 4],
    jitter: f32,
}

/// Two mask entries can merge if they have the same material and AO values.
#[inline]
fn can_merge(a: &MaskEntry, b: &MaskEntry) -> bool {
    a.mat_id == b.mat_id && a.ao == b.ao
}

// ── Quad representation ─────────────────────────────────────────────

struct Quad {
    /// 4 corner positions in block space.
    corners: [[f32; 3]; 4],
    /// Face index (0–5) for normal and brightness lookup.
    fi: usize,
    /// AO values for the 4 vertices.
    ao: [u8; 4],
    /// Per-block color jitter.
    jitter: f32,
}

// ── Compute merged quad corners ─────────────────────────────────────

/// Given a face definition and a merged w×h rectangle starting at (u, v) in the
/// slice, compute the 4 corner positions in block coordinates.
fn compute_merged_corners(fi: usize, slice: i32, u: i32, v: i32, w: i32, h: i32) -> [[f32; 3]; 4] {
    let face = &FACES[fi];
    let axis_map = &FACE_AXIS_MAPS[fi];

    let (ox, oy, oz) = to_bxyz(fi, slice, u, v);

    // U/V direction vectors in block-coordinate space
    let (u1x, u1y, u1z) = to_bxyz(fi, slice, 1, 0);
    let (u0x, u0y, u0z) = to_bxyz(fi, slice, 0, 0);
    let u_dir = [u1x - u0x, u1y - u0y, u1z - u0z];

    let (v1x, v1y, v1z) = to_bxyz(fi, slice, 0, 1);
    let v_dir = [v1x - u0x, v1y - u0y, v1z - u0z];

    let mut corners = [[0.0_f32; 3]; 4];

    for i in 0..4 {
        let vert = &face.vertices[i];
        // Determine if this vertex sits at the "high" end of U or V
        let is_high_u = if vert[axis_map.u_vert_component] > 0 {
            1
        } else {
            0
        };
        let is_high_v = if vert[axis_map.v_vert_component] > 0 {
            1
        } else {
            0
        };
        let u_ext = is_high_u * (w - 1);
        let v_ext = is_high_v * (h - 1);

        corners[i] = [
            (ox + vert[0] + u_ext * u_dir[0] + v_ext * v_dir[0]) as f32,
            (oy + vert[1] + u_ext * u_dir[1] + v_ext * v_dir[1]) as f32,
            (oz + vert[2] + u_ext * u_dir[2] + v_ext * v_dir[2]) as f32,
        ];
    }

    corners
}

// ── Main mesh builder ───────────────────────────────────────────────

/// Build greedy-meshed VoxelMeshData for each material from the extracted surface
/// voxels and the raw density volume.
///
/// This is the Rust port of `buildVoxelMeshes` from `voxelMeshBuilder.ts`.
///
/// # Arguments
/// * `voxel_data` — Extracted surface voxels (from `extract_surface_voxels`).
/// * `densities` — Raw 3D density volume (for solid/AO lookups).
/// * `resolution` — XZ grid resolution (N).
/// * `y_slices` — Number of Y slices.
/// * `scale` — (scale_x, scale_y, scale_z) applied to vertex positions.
/// * `offset` — (offset_x, offset_y, offset_z) added to vertex positions.
pub fn build_voxel_meshes(
    voxel_data: &VoxelData,
    densities: &[f32],
    resolution: u32,
    y_slices: u32,
    scale: (f32, f32, f32),
    offset: (f32, f32, f32),
) -> Vec<VoxelMeshData> {
    let n = resolution as i32;
    let ys = y_slices as i32;
    let (scale_x, scale_y, scale_z) = scale;
    let (offset_x, offset_y, offset_z) = offset;

    // Build material grid: (x, y, z) → material ID (-1 for air)
    let grid_size = (n * n * ys) as usize;
    let mut mat_grid = vec![-1_i8; grid_size];

    for i in 0..voxel_data.count as usize {
        let bx = voxel_data.positions[i * 3] as i32;
        let by = voxel_data.positions[i * 3 + 1] as i32;
        let bz = voxel_data.positions[i * 3 + 2] as i32;
        if bx < 0 || bx >= n || by < 0 || by >= ys || bz < 0 || bz >= n {
            continue;
        }
        let idx = (by * n * n + bz * n + bx) as usize;
        mat_grid[idx] = voxel_data.material_ids[i] as i8;
    }

    // Collect quads per material: mat_id → Vec<Quad>
    let mut material_quads: Vec<(u8, Vec<Quad>)> = Vec::new();

    // Helper to find or create the quad list for a material
    fn get_or_create_quads<'a>(
        material_quads: &'a mut Vec<(u8, Vec<Quad>)>,
        mat_id: u8,
    ) -> &'a mut Vec<Quad> {
        let pos = material_quads.iter().position(|(m, _)| *m == mat_id);
        match pos {
            Some(idx) => &mut material_quads[idx].1,
            None => {
                material_quads.push((mat_id, Vec::new()));
                let last = material_quads.len() - 1;
                &mut material_quads[last].1
            }
        }
    }

    // Process each of the 6 face directions
    for fi in 0..6_usize {
        let face = &FACES[fi];
        let (slice_count, u_count, v_count) = axis_counts(fi, n, ys);

        // Process each slice
        for slice in 0..slice_count {
            let mask_size = (u_count * v_count) as usize;

            // Build 2D mask for this slice
            let mut mask: Vec<Option<MaskEntry>> = vec![None; mask_size];

            for v in 0..v_count {
                for u in 0..u_count {
                    let (bx, by, bz) = to_bxyz(fi, slice, u, v);

                    // Check if this voxel has a surface voxel (is in matGrid)
                    let grid_idx = (by * n * n + bz * n + bx) as usize;
                    let mat_id_raw = mat_grid[grid_idx];
                    if mat_id_raw < 0 {
                        continue; // not a surface voxel
                    }

                    // Check if face is exposed (neighbor in face direction is not solid)
                    let (nx, ny, nz) = neighbor_offset(fi, slice, u, v);
                    if is_solid(densities, nx, ny, nz, n, ys) {
                        continue;
                    }

                    // Compute AO for 4 vertices
                    let mut ao = [0_u8; 4];
                    for vi in 0..4 {
                        let ao_offsets = &face.ao[vi];
                        ao[vi] = compute_vertex_ao(
                            densities,
                            bx,
                            by,
                            bz,
                            &ao_offsets[0],
                            &ao_offsets[1],
                            &ao_offsets[2],
                            n,
                            ys,
                        );
                    }

                    let jitter = block_jitter(bx, by, bz);
                    let mask_idx = (v * u_count + u) as usize;
                    mask[mask_idx] = Some(MaskEntry {
                        mat_id: mat_id_raw as u8,
                        ao,
                        jitter,
                    });
                }
            }

            // Greedy merge the mask
            let mut visited = vec![false; mask_size];

            for v in 0..v_count {
                for u in 0..u_count {
                    let mask_idx = (v * u_count + u) as usize;
                    if visited[mask_idx] {
                        continue;
                    }
                    let entry = match &mask[mask_idx] {
                        Some(e) => e.clone(),
                        None => continue,
                    };

                    // Extend width
                    let mut w = 1_i32;
                    while u + w < u_count {
                        let next_idx = (v * u_count + (u + w)) as usize;
                        if visited[next_idx] {
                            break;
                        }
                        match &mask[next_idx] {
                            Some(next) if can_merge(&entry, next) => w += 1,
                            _ => break,
                        }
                    }

                    // Extend height
                    let mut h = 1_i32;
                    'outer: while v + h < v_count {
                        for du in 0..w {
                            let check_idx = ((v + h) * u_count + (u + du)) as usize;
                            if visited[check_idx] {
                                break 'outer;
                            }
                            match &mask[check_idx] {
                                Some(check) if can_merge(&entry, check) => {}
                                _ => break 'outer,
                            }
                        }
                        h += 1;
                    }

                    // Mark as visited
                    for dv in 0..h {
                        for du in 0..w {
                            visited[((v + dv) * u_count + (u + du)) as usize] = true;
                        }
                    }

                    // Emit merged quad
                    let corners = compute_merged_corners(fi, slice, u, v, w, h);
                    let quads = get_or_create_quads(&mut material_quads, entry.mat_id);
                    quads.push(Quad {
                        corners,
                        fi,
                        ao: entry.ao,
                        jitter: entry.jitter,
                    });
                }
            }
        }
    }

    // Build VoxelMeshData per material
    let mut results: Vec<VoxelMeshData> = Vec::with_capacity(material_quads.len());

    for (mat_id, quads) in &material_quads {
        let material = voxel_data
            .materials
            .get(*mat_id as usize)
            .or_else(|| voxel_data.materials.first());

        let base_color_hex = material.map(|m| m.color.as_str()).unwrap_or("#808080");
        let base_rgb = hex_to_rgb(base_color_hex);
        let face_count = quads.len();

        // Each quad = 4 vertices, 6 indices
        let mut positions = Vec::with_capacity(face_count * 4 * 3);
        let mut normals = Vec::with_capacity(face_count * 4 * 3);
        let mut colors = Vec::with_capacity(face_count * 4 * 3);
        let mut indices = Vec::with_capacity(face_count * 6);

        let mut vert_idx: u32 = 0;

        for quad in quads {
            let face = &FACES[quad.fi];
            let face_brightness = FACE_BRIGHTNESS[quad.fi];
            let base_vert = vert_idx;

            // Emit 4 vertices
            for vi in 0..4 {
                let corner = &quad.corners[vi];
                positions.push(corner[0] * scale_x + offset_x);
                positions.push(corner[1] * scale_y + offset_y);
                positions.push(corner[2] * scale_z + offset_z);

                normals.push(face.dir[0] as f32);
                normals.push(face.dir[1] as f32);
                normals.push(face.dir[2] as f32);

                let ao_brightness = AO_CURVE[quad.ao[vi] as usize];
                let brightness = face_brightness * ao_brightness * (1.0 + quad.jitter);

                colors.push((base_rgb[0] * brightness).clamp(0.0, 1.0));
                colors.push((base_rgb[1] * brightness).clamp(0.0, 1.0));
                colors.push((base_rgb[2] * brightness).clamp(0.0, 1.0));

                vert_idx += 1;
            }

            // AO-aware triangle winding
            if quad.ao[0] as u16 + quad.ao[2] as u16 > quad.ao[1] as u16 + quad.ao[3] as u16 {
                indices.push(base_vert + 1);
                indices.push(base_vert + 2);
                indices.push(base_vert + 3);
                indices.push(base_vert + 1);
                indices.push(base_vert + 3);
                indices.push(base_vert + 0);
            } else {
                indices.push(base_vert + 0);
                indices.push(base_vert + 1);
                indices.push(base_vert + 2);
                indices.push(base_vert + 0);
                indices.push(base_vert + 2);
                indices.push(base_vert + 3);
            }
        }

        results.push(VoxelMeshData {
            material_index: *mat_id as u32,
            color: base_color_hex.to_string(),
            positions,
            normals,
            colors,
            indices,
            material_properties: MaterialProperties {
                roughness: material.map(|m| m.roughness).unwrap_or(0.8),
                metalness: material.map(|m| m.metalness).unwrap_or(0.0),
                emissive: material
                    .map(|m| m.emissive.clone())
                    .unwrap_or_else(|| "#000000".to_string()),
                emissive_intensity: material.map(|m| m.emissive_intensity).unwrap_or(0.0),
            },
        });
    }

    results
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::voxel::{extract_surface_voxels, VoxelMaterial};

    #[test]
    fn hex_to_rgb_basic() {
        let rgb = hex_to_rgb("#808080");
        assert!((rgb[0] - 0.502).abs() < 0.01);
        assert!((rgb[1] - 0.502).abs() < 0.01);
        assert!((rgb[2] - 0.502).abs() < 0.01);
    }

    #[test]
    fn hex_to_rgb_pure_colors() {
        let r = hex_to_rgb("#FF0000");
        assert!((r[0] - 1.0).abs() < 0.01);
        assert!((r[1]).abs() < 0.01);
        assert!((r[2]).abs() < 0.01);

        let g = hex_to_rgb("#00FF00");
        assert!((g[0]).abs() < 0.01);
        assert!((g[1] - 1.0).abs() < 0.01);
        assert!((g[2]).abs() < 0.01);
    }

    #[test]
    fn hex_to_rgb_no_hash() {
        let rgb = hex_to_rgb("808080");
        assert!((rgb[0] - 0.502).abs() < 0.01);
    }

    #[test]
    fn block_hash_deterministic() {
        let h1 = block_hash(10, 20, 30);
        let h2 = block_hash(10, 20, 30);
        assert_eq!(h1, h2);
        assert!(h1 >= 0);
    }

    #[test]
    fn block_hash_varies() {
        let h1 = block_hash(0, 0, 0);
        let h2 = block_hash(1, 0, 0);
        let h3 = block_hash(0, 1, 0);
        // Different inputs should (almost certainly) produce different hashes
        assert!(h1 != h2 || h1 != h3);
    }

    #[test]
    fn block_jitter_in_range() {
        for x in 0..10 {
            for y in 0..5 {
                for z in 0..5 {
                    let j = block_jitter(x, y, z);
                    assert!(
                        j >= -0.09 && j <= 0.09,
                        "jitter({},{},{}) = {} out of range",
                        x,
                        y,
                        z,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn compute_vertex_ao_no_neighbors() {
        // 3x3x3 volume with only center solid
        let mut densities = vec![-1.0_f32; 27];
        densities[1 * 9 + 1 * 3 + 1] = 1.0; // center
        let ao = compute_vertex_ao(
            &densities,
            1,
            1,
            1,
            &[-1, 1, 0],
            &[0, 1, -1],
            &[-1, 1, -1],
            3,
            3,
        );
        assert_eq!(ao, 0); // no solid neighbors above
    }

    #[test]
    fn compute_vertex_ao_both_edges_gives_3() {
        // Dense 3x3x3 volume
        let densities = vec![1.0_f32; 27];
        let ao = compute_vertex_ao(
            &densities,
            1,
            0,
            1,
            &[-1, 1, 0],
            &[0, 1, -1],
            &[-1, 1, -1],
            3,
            3,
        );
        assert_eq!(ao, 3); // both edges solid → always 3
    }

    #[test]
    fn single_voxel_mesh() {
        // Single solid voxel in a 1x1x1 volume
        let densities = vec![1.0_f32; 1];
        let voxels = extract_surface_voxels(&densities, 1, 1, None, None, None);
        assert_eq!(voxels.count, 1);

        let meshes =
            build_voxel_meshes(&voxels, &densities, 1, 1, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0));

        // Should produce exactly 1 material group
        assert_eq!(meshes.len(), 1);
        let mesh = &meshes[0];

        // Single voxel, 6 exposed faces → 6 quads
        // Each quad: 4 vertices, 6 indices
        assert_eq!(mesh.positions.len(), 6 * 4 * 3); // 72
        assert_eq!(mesh.normals.len(), 6 * 4 * 3);
        assert_eq!(mesh.colors.len(), 6 * 4 * 3);
        assert_eq!(mesh.indices.len(), 6 * 6); // 36
    }

    #[test]
    fn two_material_mesh() {
        // 2x1x1 volume with two different materials
        let n = 2_u32;
        let ys = 1_u32;
        let densities = vec![1.0_f32; (n * n * ys) as usize]; // 2x2x1 all solid
        let mat_ids = vec![0_u8, 1, 0, 1]; // alternating materials

        let palette = vec![
            VoxelMaterial {
                name: "Stone".into(),
                color: "#808080".into(),
                ..Default::default()
            },
            VoxelMaterial {
                name: "Dirt".into(),
                color: "#8B4513".into(),
                ..Default::default()
            },
        ];

        let voxels =
            extract_surface_voxels(&densities, n, ys, Some(&mat_ids), Some(&palette), None);

        let meshes =
            build_voxel_meshes(&voxels, &densities, n, ys, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0));

        // Should have 2 material groups
        assert_eq!(meshes.len(), 2);

        // Check materials are assigned correctly
        let mat_indices: Vec<u32> = meshes.iter().map(|m| m.material_index).collect();
        assert!(mat_indices.contains(&0));
        assert!(mat_indices.contains(&1));
    }

    #[test]
    fn mesh_scale_and_offset() {
        // Single voxel with scale and offset
        let densities = vec![1.0_f32; 1];
        let voxels = extract_surface_voxels(&densities, 1, 1, None, None, None);

        let meshes = build_voxel_meshes(
            &voxels,
            &densities,
            1,
            1,
            (2.0, 3.0, 4.0),
            (10.0, 20.0, 30.0),
        );

        assert_eq!(meshes.len(), 1);
        let positions = &meshes[0].positions;

        // All positions should be offset by at least (10, 20, 30) since the
        // voxel is at origin and vertices are at 0 or 1 in each axis.
        for i in (0..positions.len()).step_by(3) {
            assert!(positions[i] >= 10.0, "x={} should be >= 10.0", positions[i]);
            assert!(
                positions[i + 1] >= 20.0,
                "y={} should be >= 20.0",
                positions[i + 1]
            );
            assert!(
                positions[i + 2] >= 30.0,
                "z={} should be >= 30.0",
                positions[i + 2]
            );
        }
    }

    #[test]
    fn greedy_merging_reduces_quad_count() {
        // A flat 4x1x4 solid layer — top and bottom faces should merge into
        // fewer quads than 16 per face since all have the same material and
        // (likely) the same AO values.
        let n = 4_u32;
        let ys = 1_u32;
        let densities = vec![1.0_f32; (n * n * ys) as usize];
        let voxels = extract_surface_voxels(&densities, n, ys, None, None, None);

        let meshes =
            build_voxel_meshes(&voxels, &densities, n, ys, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0));

        assert_eq!(meshes.len(), 1);
        let mesh = &meshes[0];

        // Without greedy merging, 16 voxels × 6 faces = 96 quads.
        // But interior faces between adjacent voxels are culled (not exposed),
        // and exposed faces should be merged. The total quad count should be
        // much less than 96.
        // Each quad has 4 vertices, so total_quads = positions.len() / 12
        let total_quads = mesh.positions.len() / 12;
        assert!(
            total_quads < 96,
            "Expected greedy merging to reduce quads from ~96, got {}",
            total_quads
        );

        // For a 4x4x1 flat layer, the top and bottom should each be 1 quad
        // (fully merged), and the 4 sides should each be 1 quad = 6 total.
        // (Side faces: 4 edges × 1 quad each, but each side has 4 voxels that
        // merge into 1 quad = 4 side quads × 1 per edge direction?)
        // Actually: +Y: 1 quad, -Y: 1 quad, +X edge: 4 voxels → 1 merged,
        // -X edge: 1 merged, +Z edge: 1 merged, -Z edge: 1 merged = 6 total
        assert_eq!(
            total_quads, 6,
            "A 4x4x1 flat layer should produce exactly 6 merged quads"
        );
    }

    #[test]
    fn ao_aware_winding() {
        // Check that AO-aware winding produces valid indices
        let n = 3_u32;
        let ys = 3_u32;
        let mut densities = vec![-1.0_f32; (n * n * ys) as usize];

        // Create an L-shaped solid at y=0 to get varying AO
        for z in 0..3 {
            for x in 0..3 {
                densities[(0 * 9 + z * 3 + x) as usize] = 1.0;
            }
        }
        // Add a column at (0, 1, 0) to create AO asymmetry
        densities[(1 * 9 + 0 * 3 + 0) as usize] = 1.0;

        let voxels = extract_surface_voxels(&densities, n, ys, None, None, None);
        let meshes =
            build_voxel_meshes(&voxels, &densities, n, ys, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0));

        for mesh in &meshes {
            let num_verts = mesh.positions.len() / 3;
            // All indices should be valid
            for &idx in &mesh.indices {
                assert!(
                    (idx as usize) < num_verts,
                    "Index {} out of bounds (num_verts={})",
                    idx,
                    num_verts
                );
            }
            // Index count should be a multiple of 3 (triangles)
            assert_eq!(mesh.indices.len() % 3, 0);
            // Vertex count should be a multiple of 4 (quads)
            assert_eq!(num_verts % 4, 0);
        }
    }

    #[test]
    fn colors_are_in_valid_range() {
        let densities = vec![1.0_f32; 8]; // 2x2x2
        let voxels = extract_surface_voxels(&densities, 2, 2, None, None, None);
        let meshes =
            build_voxel_meshes(&voxels, &densities, 2, 2, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0));

        for mesh in &meshes {
            for &c in &mesh.colors {
                assert!(c >= 0.0 && c <= 1.0, "Color component {} out of [0, 1]", c);
            }
        }
    }

    #[test]
    fn normals_are_unit_length() {
        let densities = vec![1.0_f32; 1];
        let voxels = extract_surface_voxels(&densities, 1, 1, None, None, None);
        let meshes =
            build_voxel_meshes(&voxels, &densities, 1, 1, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0));

        for mesh in &meshes {
            for i in (0..mesh.normals.len()).step_by(3) {
                let nx = mesh.normals[i];
                let ny = mesh.normals[i + 1];
                let nz = mesh.normals[i + 2];
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                assert!(
                    (len - 1.0).abs() < 0.001,
                    "Normal ({}, {}, {}) has length {}",
                    nx,
                    ny,
                    nz,
                    len
                );
            }
        }
    }

    #[test]
    fn empty_voxels_produce_no_meshes() {
        let densities = vec![-1.0_f32; 27]; // all air
        let voxels = extract_surface_voxels(&densities, 3, 3, None, None, None);
        assert_eq!(voxels.count, 0);

        let meshes =
            build_voxel_meshes(&voxels, &densities, 3, 3, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0));

        assert!(meshes.is_empty());
    }

    #[test]
    fn material_properties_propagated() {
        let densities = vec![1.0_f32; 1];
        let mat_ids = vec![0_u8];
        let palette = vec![VoxelMaterial {
            name: "Metal".into(),
            color: "#C0C0C0".into(),
            roughness: 0.2,
            metalness: 0.9,
            emissive: "#FF0000".into(),
            emissive_intensity: 0.5,
        }];

        let voxels = extract_surface_voxels(&densities, 1, 1, Some(&mat_ids), Some(&palette), None);
        let meshes =
            build_voxel_meshes(&voxels, &densities, 1, 1, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0));

        assert_eq!(meshes.len(), 1);
        let props = &meshes[0].material_properties;
        assert!((props.roughness - 0.2).abs() < 0.01);
        assert!((props.metalness - 0.9).abs() < 0.01);
        assert_eq!(props.emissive, "#FF0000");
        assert!((props.emissive_intensity - 0.5).abs() < 0.01);
    }

    #[test]
    fn can_merge_same() {
        let a = MaskEntry {
            mat_id: 0,
            ao: [0, 0, 0, 0],
            jitter: 0.01,
        };
        let b = MaskEntry {
            mat_id: 0,
            ao: [0, 0, 0, 0],
            jitter: 0.05,
        };
        assert!(can_merge(&a, &b)); // jitter doesn't affect merge
    }

    #[test]
    fn can_merge_different_material() {
        let a = MaskEntry {
            mat_id: 0,
            ao: [0, 0, 0, 0],
            jitter: 0.0,
        };
        let b = MaskEntry {
            mat_id: 1,
            ao: [0, 0, 0, 0],
            jitter: 0.0,
        };
        assert!(!can_merge(&a, &b));
    }

    #[test]
    fn can_merge_different_ao() {
        let a = MaskEntry {
            mat_id: 0,
            ao: [0, 0, 0, 0],
            jitter: 0.0,
        };
        let b = MaskEntry {
            mat_id: 0,
            ao: [1, 0, 0, 0],
            jitter: 0.0,
        };
        assert!(!can_merge(&a, &b));
    }
}
