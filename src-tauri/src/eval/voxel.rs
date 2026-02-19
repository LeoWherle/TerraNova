// eval/voxel.rs — Surface voxel extraction
//
// Ports `extractSurfaceVoxels` from `src/utils/voxelExtractor.ts`.
// Extracts surface voxels from a 3D density volume: a surface voxel is solid
// (density >= SOLID_THRESHOLD) with at least one air neighbor.
//
// Layout: densities[y * n * n + z * n + x]

use serde::{Deserialize, Serialize};

// ── Constants ───────────────────────────────────────────────────────

/// Voxels with density >= SOLID_THRESHOLD are considered solid.
/// Matches Hytale convention: density >= 0 = solid, density < 0 = air.
pub const SOLID_THRESHOLD: f32 = 0.0;

// ── Types ───────────────────────────────────────────────────────────

/// Material definition for a voxel type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelMaterial {
    pub name: String,
    pub color: String,
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    #[serde(default)]
    pub metalness: f32,
    #[serde(default = "default_emissive")]
    pub emissive: String,
    #[serde(default)]
    pub emissive_intensity: f32,
}

fn default_roughness() -> f32 {
    0.8
}
fn default_emissive() -> String {
    "#000000".to_string()
}

impl Default for VoxelMaterial {
    fn default() -> Self {
        VoxelMaterial {
            name: "Stone".to_string(),
            color: "#808080".to_string(),
            roughness: 0.8,
            metalness: 0.0,
            emissive: "#000000".to_string(),
            emissive_intensity: 0.0,
        }
    }
}

/// Fluid configuration for rendering fluid bodies (water, lava).
#[derive(Debug, Clone, Deserialize)]
pub struct FluidConfig {
    /// Y level (in voxel coordinates) at or below which air becomes fluid.
    pub fluid_level: i32,
    /// Material palette index for fluid voxels.
    pub fluid_material_index: u8,
}

/// Extracted surface voxel data.
#[derive(Debug, Serialize)]
pub struct VoxelData {
    /// Packed x,y,z positions of surface voxels (3 values per voxel).
    pub positions: Vec<f32>,
    /// Material ID per voxel (indexes into materials array).
    pub material_ids: Vec<u8>,
    /// Material palette.
    pub materials: Vec<VoxelMaterial>,
    /// Number of surface voxels.
    pub count: u32,
}

// ── Default palette ─────────────────────────────────────────────────

fn default_palette() -> Vec<VoxelMaterial> {
    vec![VoxelMaterial::default()]
}

// ── Neighbor directions ─────────────────────────────────────────────

const DIRS: [[i32; 3]; 6] = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
];

// ── Solid check ─────────────────────────────────────────────────────

/// Check if a voxel at (x, y, z) is solid. Out-of-bounds = air (not solid).
#[inline]
pub fn is_solid(densities: &[f32], x: i32, y: i32, z: i32, n: i32, ys: i32) -> bool {
    if x < 0 || x >= n || y < 0 || y >= ys || z < 0 || z >= n {
        return false;
    }
    densities[(y * n * n + z * n + x) as usize] >= SOLID_THRESHOLD
}

// ── Surface detection ───────────────────────────────────────────────

/// A solid voxel is a surface voxel if at least one of its 6 neighbors is air
/// (not solid or out of bounds).
fn is_surface(densities: &[f32], x: i32, y: i32, z: i32, n: i32, ys: i32) -> bool {
    for [dx, dy, dz] in &DIRS {
        let nx = x + dx;
        let ny = y + dy;
        let nz = z + dz;

        // Out of bounds = air = exposed surface
        if nx < 0 || nx >= n || ny < 0 || ny >= ys || nz < 0 || nz >= n {
            return true;
        }

        let n_idx = (ny * n * n + nz * n + nx) as usize;
        if densities[n_idx] < SOLID_THRESHOLD {
            return true;
        }
    }

    false
}

/// Check if an air voxel below the fluid level is a fluid surface voxel.
/// A fluid surface voxel is air, at or below the fluid level, and has at least one
/// neighbor that is either above the fluid level (and air) or is solid.
fn is_fluid_surface(
    densities: &[f32],
    x: i32,
    y: i32,
    z: i32,
    n: i32,
    ys: i32,
    fluid_level: i32,
) -> bool {
    for [dx, dy, dz] in &DIRS {
        let nx = x + dx;
        let ny = y + dy;
        let nz = z + dz;

        // Out of bounds = exposed
        if nx < 0 || nx >= n || ny < 0 || ny >= ys || nz < 0 || nz >= n {
            return true;
        }

        let n_idx = (ny * n * n + nz * n + nx) as usize;

        // Neighbor above fluid level and also air = exposed surface
        if ny > fluid_level && densities[n_idx] < SOLID_THRESHOLD {
            return true;
        }

        // Neighbor is solid = fluid touches terrain
        if densities[n_idx] >= SOLID_THRESHOLD {
            return true;
        }
    }

    false
}

// ── Main extraction function ────────────────────────────────────────

/// Extract surface voxels from a 3D density volume.
///
/// A surface voxel is solid (density >= SOLID_THRESHOLD) with at least one air neighbor.
/// When `fluid_config` is provided, air voxels at or below the fluid level that are
/// exposed become fluid surface voxels.
///
/// Layout: densities[y * n * n + z * n + x]
pub fn extract_surface_voxels(
    densities: &[f32],
    resolution: u32,
    y_slices: u32,
    material_ids: Option<&[u8]>,
    palette: Option<&[VoxelMaterial]>,
    fluid_config: Option<&FluidConfig>,
) -> VoxelData {
    let n = resolution as i32;
    let ys = y_slices as i32;
    let total = (n * n * ys) as usize;
    let materials: Vec<VoxelMaterial> = palette.map(|p| p.to_vec()).unwrap_or_else(default_palette);

    // Estimate: surface voxels are typically a small fraction of the volume.
    // A reasonable heuristic is ~6× the cross-section area (the six faces of
    // a cube). For a 64³ volume that's ~24 576 out of 262 144 — about 9%.
    // Pre-allocating to 10% avoids most re-allocs without wasting memory.
    let estimate = (total / 10).max(64);
    let mut positions = Vec::with_capacity(estimate * 3);
    let mut out_material_ids = Vec::with_capacity(estimate);

    // Single pass — each voxel is checked exactly once.
    // The old two-pass approach (count then fill) doubled the work by running
    // is_surface / is_fluid_surface twice per voxel.
    let mut count: u32 = 0;

    for y in 0..ys {
        let y_off = y * n * n;
        for z in 0..n {
            let yz_off = y_off + z * n;
            for x in 0..n {
                let idx = (yz_off + x) as usize;
                let d = densities[idx];
                if d >= SOLID_THRESHOLD {
                    if is_surface(densities, x, y, z, n, ys) {
                        positions.push(x as f32);
                        positions.push(y as f32);
                        positions.push(z as f32);
                        out_material_ids.push(material_ids.map(|m| m[idx]).unwrap_or(0));
                        count += 1;
                    }
                } else if let Some(fc) = fluid_config {
                    if y <= fc.fluid_level
                        && is_fluid_surface(densities, x, y, z, n, ys, fc.fluid_level)
                    {
                        positions.push(x as f32);
                        positions.push(y as f32);
                        positions.push(z as f32);
                        out_material_ids.push(fc.fluid_material_index);
                        count += 1;
                    }
                }
            }
        }
    }

    VoxelData {
        positions,
        material_ids: out_material_ids,
        materials,
        count,
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn stone_palette() -> Vec<VoxelMaterial> {
        vec![VoxelMaterial {
            name: "Stone".into(),
            color: "#808080".into(),
            ..Default::default()
        }]
    }

    #[test]
    fn empty_volume_gives_no_voxels() {
        // All air
        let densities = vec![-1.0_f32; 4 * 4 * 4];
        let result = extract_surface_voxels(&densities, 4, 4, None, None, None);
        assert_eq!(result.count, 0);
        assert!(result.positions.is_empty());
        assert!(result.material_ids.is_empty());
    }

    #[test]
    fn fully_solid_volume_has_only_surface() {
        // 4x4x4 all solid — interior voxels are NOT surface
        let n = 4_u32;
        let ys = 4_u32;
        let densities = vec![1.0_f32; (n * n * ys) as usize];
        let result = extract_surface_voxels(&densities, n, ys, None, None, None);

        // All voxels are surface because the volume is small enough that every
        // voxel has at least one out-of-bounds neighbor (since 4x4x4 means
        // only the 2x2x2 interior could possibly be fully surrounded).
        // Actually in a 4x4x4, interior voxels at (1,1,1), (1,1,2), (1,2,1),
        // (1,2,2), (2,1,1), (2,1,2), (2,2,1), (2,2,2) have all 6 neighbors in bounds.
        // Total = 64, interior = 8, surface = 56.
        assert_eq!(result.count, 56);
        assert_eq!(result.positions.len(), 56 * 3);
    }

    #[test]
    fn single_solid_voxel_is_surface() {
        // 3x3x3 volume with only center voxel solid
        let n = 3_u32;
        let ys = 3_u32;
        let mut densities = vec![-1.0_f32; (n * n * ys) as usize];
        // Center = (1, 1, 1)
        let idx = 1 * 9 + 1 * 3 + 1;
        densities[idx] = 1.0;

        let result = extract_surface_voxels(&densities, n, ys, None, None, None);
        assert_eq!(result.count, 1);
        assert_eq!(result.positions.len(), 3);
        assert_eq!(result.positions[0], 1.0); // x
        assert_eq!(result.positions[1], 1.0); // y
        assert_eq!(result.positions[2], 1.0); // z
    }

    #[test]
    fn material_ids_are_preserved() {
        let n = 2_u32;
        let ys = 1_u32;
        let densities = vec![1.0_f32; (n * n * ys) as usize]; // all solid
        let mat_ids: Vec<u8> = vec![0, 1, 2, 3];
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
            VoxelMaterial {
                name: "Grass".into(),
                color: "#228B22".into(),
                ..Default::default()
            },
            VoxelMaterial {
                name: "Sand".into(),
                color: "#C2B280".into(),
                ..Default::default()
            },
        ];

        let result =
            extract_surface_voxels(&densities, n, ys, Some(&mat_ids), Some(&palette), None);

        // All 4 voxels in a 2x2x1 grid are surface (all have y+1 and y-1 out of bounds)
        assert_eq!(result.count, 4);
        assert_eq!(result.materials.len(), 4);

        // Material IDs should match input order (z-major, then x)
        // y=0: z=0,x=0 -> idx 0 (mat 0), z=0,x=1 -> idx 1 (mat 1),
        //       z=1,x=0 -> idx 2 (mat 2), z=1,x=1 -> idx 3 (mat 3)
        assert_eq!(result.material_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn default_palette_is_stone() {
        let densities = vec![1.0_f32; 1];
        let result = extract_surface_voxels(&densities, 1, 1, None, None, None);
        assert_eq!(result.count, 1);
        assert_eq!(result.materials.len(), 1);
        assert_eq!(result.materials[0].name, "Stone");
        assert_eq!(result.materials[0].color, "#808080");
    }

    #[test]
    fn is_solid_out_of_bounds_is_false() {
        let densities = vec![1.0_f32; 8]; // 2x2x2
        assert!(!is_solid(&densities, -1, 0, 0, 2, 2));
        assert!(!is_solid(&densities, 2, 0, 0, 2, 2));
        assert!(!is_solid(&densities, 0, -1, 0, 2, 2));
        assert!(!is_solid(&densities, 0, 2, 0, 2, 2));
        assert!(!is_solid(&densities, 0, 0, -1, 2, 2));
        assert!(!is_solid(&densities, 0, 0, 2, 2, 2));
    }

    #[test]
    fn is_solid_for_air_is_false() {
        let densities = vec![-0.5_f32; 8];
        assert!(!is_solid(&densities, 0, 0, 0, 2, 2));
    }

    #[test]
    fn is_solid_at_threshold_is_true() {
        // density == 0.0 is solid (>= threshold)
        let densities = vec![0.0_f32; 8];
        assert!(is_solid(&densities, 0, 0, 0, 2, 2));
    }

    #[test]
    fn fluid_surface_extraction() {
        // 4x4x4 volume — bottom 2 layers solid, top 2 air
        let n = 4_u32;
        let ys = 4_u32;
        let total = (n * n * ys) as usize;
        let mut densities = vec![-1.0_f32; total];

        // Make y=0 and y=1 solid
        for y in 0..2_i32 {
            for z in 0..4_i32 {
                for x in 0..4_i32 {
                    densities[(y * 16 + z * 4 + x) as usize] = 1.0;
                }
            }
        }

        let fluid_config = FluidConfig {
            fluid_level: 2, // fluid fills y <= 2
            fluid_material_index: 1,
        };

        let palette = vec![
            VoxelMaterial {
                name: "Stone".into(),
                color: "#808080".into(),
                ..Default::default()
            },
            VoxelMaterial {
                name: "Water".into(),
                color: "#4488CC".into(),
                ..Default::default()
            },
        ];

        let result =
            extract_surface_voxels(&densities, n, ys, None, Some(&palette), Some(&fluid_config));

        // Should have both solid surface voxels and fluid surface voxels
        assert!(result.count > 0);

        // Check that fluid voxels get the fluid material index
        let fluid_voxels: Vec<usize> = result
            .material_ids
            .iter()
            .enumerate()
            .filter(|(_, &m)| m == 1)
            .map(|(i, _)| i)
            .collect();
        assert!(!fluid_voxels.is_empty(), "Should have fluid surface voxels");

        // Fluid voxels should be at y=2 (the air layer at or below fluid_level
        // that neighbors solid terrain below or air above)
        for &vi in &fluid_voxels {
            let vy = result.positions[vi * 3 + 1];
            assert!(
                vy <= fluid_config.fluid_level as f32,
                "Fluid voxel y={} should be <= fluid_level={}",
                vy,
                fluid_config.fluid_level
            );
        }
    }

    #[test]
    fn flat_layer_surface_count() {
        // Single solid layer at y=0 in a 4x4x2 volume
        // All 16 voxels in y=0 are solid, y=1 is air
        let n = 4_u32;
        let ys = 2_u32;
        let total = (n * n * ys) as usize;
        let mut densities = vec![-1.0_f32; total];

        for z in 0..4_i32 {
            for x in 0..4_i32 {
                densities[(0 * 16 + z * 4 + x) as usize] = 1.0;
            }
        }

        let result = extract_surface_voxels(&densities, n, ys, None, None, None);

        // All 16 solid voxels are surface (they have air above at y=1 or
        // out of bounds below at y=-1)
        assert_eq!(result.count, 16);
    }

    #[test]
    fn interior_voxels_not_extracted() {
        // 6x6x6 volume all solid — interior 4x4x4 block should not be surface
        let n = 6_u32;
        let ys = 6_u32;
        let densities = vec![1.0_f32; (n * n * ys) as usize];
        let result = extract_surface_voxels(&densities, n, ys, None, None, None);

        // Total = 216, interior = 4*4*4 = 64, surface = 152
        assert_eq!(result.count, 152);

        // Verify no interior positions are in the result
        for i in 0..result.count as usize {
            let x = result.positions[i * 3] as i32;
            let y = result.positions[i * 3 + 1] as i32;
            let z = result.positions[i * 3 + 2] as i32;
            let is_boundary = x == 0 || x == 5 || y == 0 || y == 5 || z == 0 || z == 5;
            assert!(
                is_boundary,
                "Voxel ({}, {}, {}) is interior but was extracted",
                x, y, z
            );
        }
    }
}
