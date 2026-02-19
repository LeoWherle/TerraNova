// eval/compiled.rs — Pre-resolved, indexed input handles for zero-overhead evaluation
//
// At graph construction time, all input handle names (strings like "Input",
// "InputA", "Factor", "Inputs[0]", etc.) are resolved into small integer IDs
// and stored in a compact per-node structure. This eliminates ALL string
// hashing and HashMap lookups from the evaluation hot path.
//
// Named handles → fixed-position array slots (O(1) lookup by constant index)
// Array handles (Inputs[N], Densities[N]) → pre-sorted SmallVec (no alloc, no sort at eval time)

use smallvec::SmallVec;

// ── Handle IDs ──────────────────────────────────────────────────────
// Each known named handle is assigned a unique u8 ID.
// These are used as indices into `ResolvedInputs::named`.

pub const H_INPUT: u8 = 0;
pub const H_INPUT_A: u8 = 1;
pub const H_INPUT_B: u8 = 2;
pub const H_FACTOR: u8 = 3;
pub const H_CONDITION: u8 = 4;
pub const H_TRUE_INPUT: u8 = 5;
pub const H_FALSE_INPUT: u8 = 6;
pub const H_OFFSET: u8 = 7;
pub const H_AMPLITUDE: u8 = 8;
pub const H_SELECTOR: u8 = 9;
pub const H_Y_PROVIDER: u8 = 10;
pub const H_MAGNITUDE: u8 = 11;
pub const H_WARP_SOURCE: u8 = 12;
pub const H_DIRECTION: u8 = 13;
pub const H_WARP_VECTOR: u8 = 14;
pub const H_VECTOR_PROVIDER: u8 = 15;
pub const H_VECTOR: u8 = 16;
pub const H_RETURN_DENSITY: u8 = 17;
pub const H_RETURN_CURVE: u8 = 18;
pub const H_DISTANCE_CURVE: u8 = 19;
pub const H_ANGLE_CURVE: u8 = 20;
pub const H_CURVE: u8 = 21;

/// Total number of known named handle slots.
pub const NAMED_HANDLE_COUNT: usize = 22;

// ── None sentinel ───────────────────────────────────────────────────
// Using u32::MAX as "no connection" sentinel instead of Option<usize>
// saves 8 bytes per slot (Option<usize> is 16 bytes, u32 is 4 bytes).
// This limits graph size to ~4 billion nodes, which is more than enough.

pub const NO_INPUT: u32 = u32::MAX;

// ── ResolvedInputs ─────────────────────────────────────────────────

/// Pre-resolved inputs for a single node, built once at graph construction time.
///
/// Named inputs are stored in a fixed-size array indexed by handle ID constants
/// (e.g., `named[H_INPUT]`). Array inputs are pre-sorted `SmallVec`s.
///
/// At evaluation time, looking up an input is a single array index operation
/// instead of a HashMap string-hash lookup.
#[derive(Clone)]
pub struct ResolvedInputs {
    /// Named input slots: `named[H_INPUT]` = source node index, or `NO_INPUT`.
    /// Indexed by handle ID constants (H_INPUT, H_INPUT_A, etc.).
    pub named: [u32; NAMED_HANDLE_COUNT],

    /// Pre-sorted array inputs for "Inputs[N]" handles.
    /// Sorted by array index at construction time — no sorting needed at eval time.
    /// Each element is the source node index.
    pub array_inputs: SmallVec<[u32; 8]>,

    /// Pre-sorted array inputs for "Densities[N]" handles.
    /// Used by MultiMix and similar nodes.
    pub array_densities: SmallVec<[u32; 4]>,
}

impl ResolvedInputs {
    /// Create empty resolved inputs (no connections).
    pub fn new() -> Self {
        ResolvedInputs {
            named: [NO_INPUT; NAMED_HANDLE_COUNT],
            array_inputs: SmallVec::new(),
            array_densities: SmallVec::new(),
        }
    }

    /// Check if a named handle is connected.
    #[inline(always)]
    pub fn has(&self, handle_id: u8) -> bool {
        self.named[handle_id as usize] != NO_INPUT
    }

    /// Get the source node index for a named handle, or None.
    #[inline(always)]
    pub fn get(&self, handle_id: u8) -> Option<usize> {
        let v = self.named[handle_id as usize];
        if v == NO_INPUT {
            None
        } else {
            Some(v as usize)
        }
    }

    /// Get the source node index for a named handle (unchecked — caller must verify `has()`).
    #[inline(always)]
    pub fn get_unchecked(&self, handle_id: u8) -> usize {
        self.named[handle_id as usize] as usize
    }
}

impl Default for ResolvedInputs {
    fn default() -> Self {
        Self::new()
    }
}

// ── Handle name → ID mapping ───────────────────────────────────────

/// Convert a handle name string to a named handle ID, if it's a known name.
/// Returns `None` for array handles like "Inputs[0]" or unknown handles.
pub fn handle_name_to_id(name: &str) -> Option<u8> {
    // Use a match for compile-time optimization (LLVM turns this into a
    // perfect hash or jump table).
    match name {
        "Input" => Some(H_INPUT),
        "InputA" => Some(H_INPUT_A),
        "InputB" => Some(H_INPUT_B),
        "Factor" => Some(H_FACTOR),
        "Condition" => Some(H_CONDITION),
        "TrueInput" => Some(H_TRUE_INPUT),
        "FalseInput" => Some(H_FALSE_INPUT),
        "Offset" => Some(H_OFFSET),
        "Amplitude" => Some(H_AMPLITUDE),
        "Selector" => Some(H_SELECTOR),
        "YProvider" => Some(H_Y_PROVIDER),
        "Magnitude" => Some(H_MAGNITUDE),
        "WarpSource" => Some(H_WARP_SOURCE),
        "Direction" => Some(H_DIRECTION),
        "WarpVector" => Some(H_WARP_VECTOR),
        "VectorProvider" => Some(H_VECTOR_PROVIDER),
        "Vector" => Some(H_VECTOR),
        "ReturnDensity" => Some(H_RETURN_DENSITY),
        "ReturnCurve" => Some(H_RETURN_CURVE),
        "DistanceCurve" => Some(H_DISTANCE_CURVE),
        "AngleCurve" => Some(H_ANGLE_CURVE),
        "Curve" => Some(H_CURVE),
        _ => None,
    }
}

/// Classify an edge handle string as either:
/// - A known named handle (returns `HandleClass::Named(id)`)
/// - An array input "Inputs[N]" (returns `HandleClass::ArrayInput(index)`)
/// - An array density "Densities[N]" (returns `HandleClass::ArrayDensity(index)`)
/// - Unknown (returns `HandleClass::Unknown`)
pub enum HandleClass {
    Named(u8),
    ArrayInput(usize),
    ArrayDensity(usize),
    Unknown,
}

/// Parse a handle string into its classification.
pub fn classify_handle(handle: &str) -> HandleClass {
    // Try named handles first (most common)
    if let Some(id) = handle_name_to_id(handle) {
        return HandleClass::Named(id);
    }

    // Try "Inputs[N]"
    if let Some(rest) = handle.strip_prefix("Inputs[") {
        if let Some(idx_str) = rest.strip_suffix(']') {
            if let Ok(idx) = idx_str.parse::<usize>() {
                return HandleClass::ArrayInput(idx);
            }
        }
    }

    // Try "Densities[N]"
    if let Some(rest) = handle.strip_prefix("Densities[") {
        if let Some(idx_str) = rest.strip_suffix(']') {
            if let Ok(idx) = idx_str.parse::<usize>() {
                return HandleClass::ArrayDensity(idx);
            }
        }
    }

    // Also accept bare "Inputs" as Inputs[0]
    if handle == "Inputs" {
        return HandleClass::ArrayInput(0);
    }
    if handle == "Densities" {
        return HandleClass::ArrayDensity(0);
    }

    HandleClass::Unknown
}

/// Build resolved inputs for a single node from its string-keyed input map.
///
/// Called once per node during graph construction. All subsequent evaluations
/// use the resolved representation with zero string operations.
pub fn resolve_inputs(string_inputs: &std::collections::HashMap<String, usize>) -> ResolvedInputs {
    let mut resolved = ResolvedInputs::new();

    // Temporary storage for array inputs that need sorting
    let mut indexed_inputs: SmallVec<[(usize, u32); 8]> = SmallVec::new();
    let mut indexed_densities: SmallVec<[(usize, u32); 4]> = SmallVec::new();

    for (handle, &source_idx) in string_inputs {
        let src = source_idx as u32;
        match classify_handle(handle) {
            HandleClass::Named(id) => {
                resolved.named[id as usize] = src;
            }
            HandleClass::ArrayInput(idx) => {
                indexed_inputs.push((idx, src));
            }
            HandleClass::ArrayDensity(idx) => {
                indexed_densities.push((idx, src));
            }
            HandleClass::Unknown => {
                // Unknown handle — ignore (or could log in debug mode)
            }
        }
    }

    // Sort array inputs by index and extract source node indices
    indexed_inputs.sort_unstable_by_key(|(idx, _)| *idx);
    resolved.array_inputs = indexed_inputs.into_iter().map(|(_, src)| src).collect();

    indexed_densities.sort_unstable_by_key(|(idx, _)| *idx);
    resolved.array_densities = indexed_densities.into_iter().map(|(_, src)| src).collect();

    resolved
}

// ── Density type tag ────────────────────────────────────────────────
// Pre-map density type strings to u16 tags at construction time.
// The match in evaluate_inner then dispatches on u16 instead of &str.
// (LLVM can optimize string matches well, but this is still measurably
// faster for 100+ variants because it eliminates all string comparison.)

// Constants for each density type — grouped logically.
// Using u16 to allow for future expansion.

// Constants
pub const DT_CONSTANT: u16 = 0;
pub const DT_ZERO: u16 = 1;
pub const DT_ONE: u16 = 2;

// Coordinates
pub const DT_COORDINATE_X: u16 = 10;
pub const DT_COORDINATE_Y: u16 = 11;
pub const DT_COORDINATE_Z: u16 = 12;

// Arithmetic — single input
pub const DT_NEGATE: u16 = 20;
pub const DT_ABS: u16 = 21;
pub const DT_SQUARE_ROOT: u16 = 22;
pub const DT_CUBE_ROOT: u16 = 23;
pub const DT_SQUARE: u16 = 24;
pub const DT_CUBE_MATH: u16 = 25;
pub const DT_INVERSE: u16 = 26;
pub const DT_SUM_SELF: u16 = 27;
pub const DT_MODULO: u16 = 28;
pub const DT_AMPLITUDE_CONSTANT: u16 = 29;
pub const DT_POW: u16 = 30;
pub const DT_LINEAR_TRANSFORM: u16 = 31;
pub const DT_FLOOR: u16 = 32;
pub const DT_CEILING: u16 = 33;

// Arithmetic — array input
pub const DT_SUM: u16 = 40;
pub const DT_PRODUCT: u16 = 41;
pub const DT_WEIGHTED_SUM: u16 = 42;
pub const DT_MIN_FUNCTION: u16 = 43;
pub const DT_MAX_FUNCTION: u16 = 44;
pub const DT_AVERAGE_FUNCTION: u16 = 45;

// Two inputs / interpolation
pub const DT_INTERPOLATE: u16 = 50;
pub const DT_OFFSET: u16 = 51;
pub const DT_AMPLITUDE: u16 = 52;

// Smooth
pub const DT_SMOOTH_MIN: u16 = 60;
pub const DT_SMOOTH_MAX: u16 = 61;
pub const DT_SMOOTH_CLAMP: u16 = 62;
pub const DT_SMOOTH_FLOOR: u16 = 63;
pub const DT_SMOOTH_CEILING: u16 = 64;

// Clamping & Range
pub const DT_CLAMP: u16 = 70;
pub const DT_CLAMP_TO_INDEX: u16 = 71;
pub const DT_NORMALIZER: u16 = 72;
pub const DT_DOUBLE_NORMALIZER: u16 = 73;
pub const DT_RANGE_CHOICE: u16 = 74;

// Passthrough
pub const DT_WRAP: u16 = 80;
pub const DT_PASSTHROUGH: u16 = 81;
pub const DT_DEBUG: u16 = 82;
pub const DT_FLAT_CACHE: u16 = 83;
pub const DT_CACHE_ONCE: u16 = 84;
pub const DT_EXPORTED: u16 = 85;
pub const DT_IMPORTED_VALUE: u16 = 86;

// Noise
pub const DT_SIMPLEX_NOISE_2D: u16 = 100;
pub const DT_SIMPLEX_NOISE_3D: u16 = 101;
pub const DT_SIMPLEX_RIDGE_NOISE_2D: u16 = 102;
pub const DT_SIMPLEX_RIDGE_NOISE_3D: u16 = 103;
pub const DT_FRACTAL_NOISE_2D: u16 = 104;
pub const DT_FRACTAL_NOISE_3D: u16 = 105;
pub const DT_VORONOI_NOISE_2D: u16 = 106;
pub const DT_VORONOI_NOISE_3D: u16 = 107;

// Position & Distance
pub const DT_DISTANCE_FROM_ORIGIN: u16 = 120;
pub const DT_DISTANCE_FROM_AXIS: u16 = 121;
pub const DT_DISTANCE_FROM_POINT: u16 = 122;
pub const DT_ANGLE_FROM_ORIGIN: u16 = 123;
pub const DT_ANGLE_FROM_POINT: u16 = 124;
pub const DT_ANGLE: u16 = 125;
pub const DT_Y_GRADIENT: u16 = 126;
pub const DT_BASE_HEIGHT: u16 = 127;
pub const DT_DISTANCE: u16 = 128;

// Position transforms
pub const DT_TRANSLATED_POSITION: u16 = 140;
pub const DT_SCALED_POSITION: u16 = 141;
pub const DT_ROTATED_POSITION: u16 = 142;
pub const DT_MIRRORED_POSITION: u16 = 143;
pub const DT_QUANTIZED_POSITION: u16 = 144;
pub const DT_Y_OVERRIDE: u16 = 145;
pub const DT_X_OVERRIDE: u16 = 146;
pub const DT_Z_OVERRIDE: u16 = 147;
pub const DT_Y_SAMPLED: u16 = 148;

// Anchor
pub const DT_ANCHOR: u16 = 150;

// Conditionals & Blending
pub const DT_CONDITIONAL: u16 = 160;
pub const DT_BLEND: u16 = 161;
pub const DT_BLEND_CURVE: u16 = 162;
pub const DT_SWITCH: u16 = 163;
pub const DT_SWITCH_STATE: u16 = 164;
pub const DT_MULTI_MIX: u16 = 165;

// Curves & Splines
pub const DT_CURVE_FUNCTION: u16 = 170;
pub const DT_SPLINE_FUNCTION: u16 = 171;

// Domain warp
pub const DT_DOMAIN_WARP_2D: u16 = 180;
pub const DT_DOMAIN_WARP_3D: u16 = 181;
pub const DT_GRADIENT_WARP: u16 = 182;
pub const DT_FAST_GRADIENT_WARP: u16 = 183;
pub const DT_VECTOR_WARP: u16 = 184;

// Shape SDFs
pub const DT_ELLIPSOID: u16 = 200;
pub const DT_CUBOID: u16 = 201;
pub const DT_CUBE: u16 = 202;
pub const DT_CYLINDER: u16 = 203;
pub const DT_PLANE: u16 = 204;
pub const DT_SHELL: u16 = 205;

// Voronoi / Positions
pub const DT_POSITIONS_CELL_NOISE: u16 = 220;
pub const DT_CELL_WALL_DISTANCE: u16 = 221;
pub const DT_POSITIONS_3D: u16 = 222;
pub const DT_POSITIONS_PINCH: u16 = 223;
pub const DT_POSITIONS_TWIST: u16 = 224;

// Unsupported (context-dependent)
pub const DT_UNSUPPORTED: u16 = 300;

// Unknown (try to follow Input handle)
pub const DT_UNKNOWN: u16 = u16::MAX;

/// Map a density type string to its u16 tag.
/// Called once per node during graph construction.
pub fn density_type_to_tag(dt: &str) -> u16 {
    match dt {
        "Constant" => DT_CONSTANT,
        "Zero" => DT_ZERO,
        "One" => DT_ONE,

        "CoordinateX" => DT_COORDINATE_X,
        "CoordinateY" => DT_COORDINATE_Y,
        "CoordinateZ" => DT_COORDINATE_Z,

        "Negate" => DT_NEGATE,
        "Abs" => DT_ABS,
        "SquareRoot" => DT_SQUARE_ROOT,
        "CubeRoot" => DT_CUBE_ROOT,
        "Square" => DT_SQUARE,
        "CubeMath" => DT_CUBE_MATH,
        "Inverse" => DT_INVERSE,
        "SumSelf" => DT_SUM_SELF,
        "Modulo" => DT_MODULO,
        "AmplitudeConstant" => DT_AMPLITUDE_CONSTANT,
        "Pow" => DT_POW,
        "LinearTransform" => DT_LINEAR_TRANSFORM,
        "Floor" => DT_FLOOR,
        "Ceiling" => DT_CEILING,

        "Sum" => DT_SUM,
        "Product" => DT_PRODUCT,
        "WeightedSum" => DT_WEIGHTED_SUM,
        "MinFunction" => DT_MIN_FUNCTION,
        "MaxFunction" => DT_MAX_FUNCTION,
        "AverageFunction" => DT_AVERAGE_FUNCTION,

        "Interpolate" => DT_INTERPOLATE,
        "Offset" => DT_OFFSET,
        "Amplitude" => DT_AMPLITUDE,

        "SmoothMin" => DT_SMOOTH_MIN,
        "SmoothMax" => DT_SMOOTH_MAX,
        "SmoothClamp" => DT_SMOOTH_CLAMP,
        "SmoothFloor" => DT_SMOOTH_FLOOR,
        "SmoothCeiling" => DT_SMOOTH_CEILING,

        "Clamp" => DT_CLAMP,
        "ClampToIndex" => DT_CLAMP_TO_INDEX,
        "Normalizer" => DT_NORMALIZER,
        "DoubleNormalizer" => DT_DOUBLE_NORMALIZER,
        "RangeChoice" => DT_RANGE_CHOICE,

        "Wrap" => DT_WRAP,
        "Passthrough" => DT_PASSTHROUGH,
        "Debug" => DT_DEBUG,
        "FlatCache" => DT_FLAT_CACHE,
        "CacheOnce" => DT_CACHE_ONCE,
        "Exported" => DT_EXPORTED,
        "ImportedValue" => DT_IMPORTED_VALUE,

        "SimplexNoise2D" => DT_SIMPLEX_NOISE_2D,
        "SimplexNoise3D" => DT_SIMPLEX_NOISE_3D,
        "SimplexRidgeNoise2D" => DT_SIMPLEX_RIDGE_NOISE_2D,
        "SimplexRidgeNoise3D" => DT_SIMPLEX_RIDGE_NOISE_3D,
        "FractalNoise2D" => DT_FRACTAL_NOISE_2D,
        "FractalNoise3D" => DT_FRACTAL_NOISE_3D,
        "VoronoiNoise2D" => DT_VORONOI_NOISE_2D,
        "VoronoiNoise3D" => DT_VORONOI_NOISE_3D,

        "DistanceFromOrigin" => DT_DISTANCE_FROM_ORIGIN,
        "DistanceFromAxis" => DT_DISTANCE_FROM_AXIS,
        "DistanceFromPoint" => DT_DISTANCE_FROM_POINT,
        "AngleFromOrigin" => DT_ANGLE_FROM_ORIGIN,
        "AngleFromPoint" => DT_ANGLE_FROM_POINT,
        "Angle" => DT_ANGLE,
        "YGradient" | "GradientDensity" | "Gradient" => DT_Y_GRADIENT,
        "BaseHeight" => DT_BASE_HEIGHT,
        "Distance" => DT_DISTANCE,

        "TranslatedPosition" => DT_TRANSLATED_POSITION,
        "ScaledPosition" => DT_SCALED_POSITION,
        "RotatedPosition" => DT_ROTATED_POSITION,
        "MirroredPosition" => DT_MIRRORED_POSITION,
        "QuantizedPosition" => DT_QUANTIZED_POSITION,
        "YOverride" => DT_Y_OVERRIDE,
        "XOverride" => DT_X_OVERRIDE,
        "ZOverride" => DT_Z_OVERRIDE,
        "YSampled" => DT_Y_SAMPLED,

        "Anchor" => DT_ANCHOR,

        "Conditional" => DT_CONDITIONAL,
        "Blend" => DT_BLEND,
        "BlendCurve" => DT_BLEND_CURVE,
        "Switch" => DT_SWITCH,
        "SwitchState" => DT_SWITCH_STATE,
        "MultiMix" => DT_MULTI_MIX,

        "CurveFunction" => DT_CURVE_FUNCTION,
        "SplineFunction" => DT_SPLINE_FUNCTION,

        "DomainWarp2D" => DT_DOMAIN_WARP_2D,
        "DomainWarp3D" => DT_DOMAIN_WARP_3D,
        "GradientWarp" => DT_GRADIENT_WARP,
        "FastGradientWarp" => DT_FAST_GRADIENT_WARP,
        "VectorWarp" => DT_VECTOR_WARP,

        "Ellipsoid" => DT_ELLIPSOID,
        "Cuboid" => DT_CUBOID,
        "Cube" => DT_CUBE,
        "Cylinder" => DT_CYLINDER,
        "Plane" => DT_PLANE,
        "Shell" => DT_SHELL,

        "PositionsCellNoise" => DT_POSITIONS_CELL_NOISE,
        "CellWallDistance" => DT_CELL_WALL_DISTANCE,
        "Positions3D" => DT_POSITIONS_3D,
        "PositionsPinch" => DT_POSITIONS_PINCH,
        "PositionsTwist" => DT_POSITIONS_TWIST,

        "HeightAboveSurface"
        | "SurfaceDensity"
        | "TerrainBoolean"
        | "TerrainMask"
        | "BeardDensity"
        | "ColumnDensity"
        | "CaveDensity"
        | "Terrain"
        | "DistanceToBiomeEdge"
        | "Pipeline" => DT_UNSUPPORTED,

        _ => DT_UNKNOWN,
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn resolve_empty_inputs() {
        let map: HashMap<String, usize> = HashMap::new();
        let ri = resolve_inputs(&map);
        assert!(!ri.has(H_INPUT));
        assert!(ri.get(H_INPUT).is_none());
        assert!(ri.array_inputs.is_empty());
        assert!(ri.array_densities.is_empty());
    }

    #[test]
    fn resolve_named_inputs() {
        let mut map = HashMap::new();
        map.insert("Input".to_string(), 5);
        map.insert("InputA".to_string(), 3);
        map.insert("Factor".to_string(), 7);

        let ri = resolve_inputs(&map);
        assert!(ri.has(H_INPUT));
        assert_eq!(ri.get(H_INPUT), Some(5));
        assert_eq!(ri.get(H_INPUT_A), Some(3));
        assert_eq!(ri.get(H_FACTOR), Some(7));
        assert!(!ri.has(H_INPUT_B));
        assert!(ri.get(H_INPUT_B).is_none());
    }

    #[test]
    fn resolve_array_inputs_sorted() {
        let mut map = HashMap::new();
        map.insert("Inputs[2]".to_string(), 10);
        map.insert("Inputs[0]".to_string(), 20);
        map.insert("Inputs[1]".to_string(), 30);

        let ri = resolve_inputs(&map);
        assert_eq!(ri.array_inputs.len(), 3);
        // Should be sorted by index: [0]→20, [1]→30, [2]→10
        assert_eq!(ri.array_inputs[0], 20);
        assert_eq!(ri.array_inputs[1], 30);
        assert_eq!(ri.array_inputs[2], 10);
    }

    #[test]
    fn resolve_array_densities_sorted() {
        let mut map = HashMap::new();
        map.insert("Densities[1]".to_string(), 5);
        map.insert("Densities[0]".to_string(), 3);

        let ri = resolve_inputs(&map);
        assert_eq!(ri.array_densities.len(), 2);
        assert_eq!(ri.array_densities[0], 3);
        assert_eq!(ri.array_densities[1], 5);
    }

    #[test]
    fn resolve_mixed_inputs() {
        let mut map = HashMap::new();
        map.insert("Input".to_string(), 1);
        map.insert("Inputs[0]".to_string(), 2);
        map.insert("Inputs[1]".to_string(), 3);
        map.insert("Curve".to_string(), 4);

        let ri = resolve_inputs(&map);
        assert_eq!(ri.get(H_INPUT), Some(1));
        assert_eq!(ri.get(H_CURVE), Some(4));
        assert_eq!(ri.array_inputs.len(), 2);
        assert_eq!(ri.array_inputs[0], 2);
        assert_eq!(ri.array_inputs[1], 3);
    }

    #[test]
    fn bare_inputs_handle_becomes_index_zero() {
        let mut map = HashMap::new();
        map.insert("Inputs".to_string(), 42);

        let ri = resolve_inputs(&map);
        assert_eq!(ri.array_inputs.len(), 1);
        assert_eq!(ri.array_inputs[0], 42);
    }

    #[test]
    fn density_type_known_types() {
        assert_eq!(density_type_to_tag("Constant"), DT_CONSTANT);
        assert_eq!(density_type_to_tag("Sum"), DT_SUM);
        assert_eq!(density_type_to_tag("SimplexNoise2D"), DT_SIMPLEX_NOISE_2D);
        assert_eq!(density_type_to_tag("Ellipsoid"), DT_ELLIPSOID);
    }

    #[test]
    fn density_type_aliases() {
        // YGradient, GradientDensity, Gradient all map to the same tag
        assert_eq!(density_type_to_tag("YGradient"), DT_Y_GRADIENT);
        assert_eq!(density_type_to_tag("GradientDensity"), DT_Y_GRADIENT);
        assert_eq!(density_type_to_tag("Gradient"), DT_Y_GRADIENT);
    }

    #[test]
    fn density_type_unsupported() {
        assert_eq!(density_type_to_tag("HeightAboveSurface"), DT_UNSUPPORTED);
        assert_eq!(density_type_to_tag("Terrain"), DT_UNSUPPORTED);
    }

    #[test]
    fn density_type_unknown() {
        assert_eq!(density_type_to_tag("NotARealType"), DT_UNKNOWN);
    }

    #[test]
    fn classify_named_handle() {
        match classify_handle("Input") {
            HandleClass::Named(id) => assert_eq!(id, H_INPUT),
            _ => panic!("Expected Named"),
        }
        match classify_handle("Factor") {
            HandleClass::Named(id) => assert_eq!(id, H_FACTOR),
            _ => panic!("Expected Named"),
        }
    }

    #[test]
    fn classify_array_input() {
        match classify_handle("Inputs[3]") {
            HandleClass::ArrayInput(idx) => assert_eq!(idx, 3),
            _ => panic!("Expected ArrayInput"),
        }
    }

    #[test]
    fn classify_array_density() {
        match classify_handle("Densities[1]") {
            HandleClass::ArrayDensity(idx) => assert_eq!(idx, 1),
            _ => panic!("Expected ArrayDensity"),
        }
    }

    #[test]
    fn classify_unknown() {
        match classify_handle("SomethingElse[0]") {
            HandleClass::Unknown => {}
            _ => panic!("Expected Unknown"),
        }
    }

    #[test]
    fn get_unchecked_works() {
        let mut map = HashMap::new();
        map.insert("Input".to_string(), 42);
        let ri = resolve_inputs(&map);
        assert!(ri.has(H_INPUT));
        assert_eq!(ri.get_unchecked(H_INPUT), 42);
    }

    #[test]
    fn no_input_sentinel_value() {
        let ri = ResolvedInputs::new();
        // All named slots should be NO_INPUT
        for i in 0..NAMED_HANDLE_COUNT {
            assert_eq!(ri.named[i], NO_INPUT);
        }
    }
}
