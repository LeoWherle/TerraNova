// eval/noise.rs — Hytale-parity noise functions
//
// This module provides simplex noise that produces **identical** output to
// the TypeScript implementation in `hytaleNoise.ts`. Every constant, every
// bit-trick, every rounding decision is matched exactly.

use serde_json::Value;

// ── Gradient vectors ────────────────────────────────────────────────

// 2D: 8 gradient directions (cardinal + diagonal, unnormalized)
const GRAD2: [[f64; 2]; 8] = [
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
];

// 3D: 12 gradient directions (edges of a cube)
const GRAD3: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

// ── Java-compatible string hash ─────────────────────────────────────

/// Java's `String.hashCode()` algorithm.
/// Produces the same integer hash as Java for ASCII strings.
///
/// Must match `javaStringHashCode` in `hytaleNoise.ts`.
pub fn java_string_hash_code(s: &str) -> i32 {
    let mut hash: i32 = 0;
    for ch in s.chars() {
        // Java uses UTF-16 code units; for BMP characters charCodeAt === char as u32
        hash = hash.wrapping_mul(31).wrapping_add(ch as u32 as i32);
    }
    hash
}

/// Convert a seed value (number or string) to an integer.
/// Matches `seedToInt` in `hytaleNoise.ts`.
pub fn seed_to_int(seed: &Value) -> i32 {
    match seed {
        Value::Number(n) => {
            // JS `seed | 0` — truncate to i32
            n.as_i64().unwrap_or(0) as i32
        }
        Value::String(s) => java_string_hash_code(s),
        _ => 0,
    }
}

// ── Mulberry32 PRNG ─────────────────────────────────────────────────

/// Mulberry32 PRNG — must produce the same sequence as the TS version.
///
/// The TS version uses `Math.imul` (32-bit integer multiply) and
/// unsigned right shift (`>>>`). We replicate with Rust wrapping ops.
pub struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    pub fn new(seed: i32) -> Self {
        Self { state: seed as u32 }
    }

    /// Generate the next f64 in [0, 1).
    ///
    /// Must match the TS closure returned by `mulberry32(seed)`.
    pub fn next_f64(&mut self) -> f64 {
        // s = (s + 0x6d2b79f5) | 0
        self.state = self.state.wrapping_add(0x6d2b79f5);

        // let t = Math.imul(s ^ (s >>> 15), 1 | s);
        let mut t: u32 = (self.state ^ (self.state >> 15)).wrapping_mul(1 | self.state);

        // t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;

        // return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

// ── Permutation table ───────────────────────────────────────────────

/// Build a 512-entry permutation table using Fisher-Yates shuffle.
/// First 256 entries are a shuffled identity [0..255],
/// second 256 entries are a copy (to avoid modular arithmetic).
///
/// Must match `buildPermutationTable` in `hytaleNoise.ts`.
pub fn build_perm_table(seed: i32) -> [u8; 512] {
    let mut rng = Mulberry32::new(seed);
    let mut perm = [0u8; 512];

    // Initialize identity
    for i in 0..256u16 {
        perm[i as usize] = i as u8;
    }

    // Fisher-Yates shuffle (descending, matching TS: i = 255 down to 1)
    for i in (1..=255usize).rev() {
        let j = (rng.next_f64() * (i as f64 + 1.0)).floor() as usize;
        perm.swap(i, j);
    }

    // Double for wrap-around
    for i in 0..256 {
        perm[i + 256] = perm[i];
    }

    perm
}

// ── 2D Simplex Noise ────────────────────────────────────────────────

// Skew/unskew constants — 2D
// F2 = (sqrt(3) - 1) / 2
// G2 = (3 - sqrt(3)) / 6
const F2: f64 = 0.36602540378443864676; // (sqrt(3) - 1) / 2
const G2: f64 = 0.21132486540518711775; // (3 - sqrt(3)) / 6

/// 2D simplex noise.
/// Returns a value in approximately [-1, 1].
///
/// Must produce identical output to `createHytaleNoise2D` in `hytaleNoise.ts`.
pub fn simplex_2d(perm: &[u8; 512], x: f64, y: f64) -> f64 {
    // Step 1: Skew input to simplex cell coordinates
    let s = (x + y) * F2;
    let i = (x + s).floor() as i64;
    let j = (y + s).floor() as i64;

    // Step 2: Unskew to find cell origin in input space
    let t = (i + j) as f64 * G2;
    let x0 = x - (i as f64 - t);
    let y0 = y - (j as f64 - t);

    // Step 3: Determine which simplex (triangle)
    let (i1, j1) = if x0 > y0 {
        (1i64, 0i64) // Upper triangle
    } else {
        (0i64, 1i64) // Lower triangle
    };

    // Step 4: Offsets for 2nd and 3rd corners
    let x1 = x0 - i1 as f64 + G2;
    let y1 = y0 - j1 as f64 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    // Step 5: Hash gradient indices
    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;
    let gi0 = (perm[ii + perm[jj] as usize] % 8) as usize;
    let gi1 = (perm[ii + i1 as usize + perm[jj + j1 as usize] as usize] % 8) as usize;
    let gi2 = (perm[ii + 1 + perm[jj + 1] as usize] % 8) as usize;

    // Step 6: Corner contributions
    let mut n0 = 0.0;
    let mut n1 = 0.0;
    let mut n2 = 0.0;

    let mut t0 = 0.5 - x0 * x0 - y0 * y0;
    if t0 >= 0.0 {
        t0 *= t0;
        n0 = t0 * t0 * (GRAD2[gi0][0] * x0 + GRAD2[gi0][1] * y0);
    }

    let mut t1 = 0.5 - x1 * x1 - y1 * y1;
    if t1 >= 0.0 {
        t1 *= t1;
        n1 = t1 * t1 * (GRAD2[gi1][0] * x1 + GRAD2[gi1][1] * y1);
    }

    let mut t2 = 0.5 - x2 * x2 - y2 * y2;
    if t2 >= 0.0 {
        t2 *= t2;
        n2 = t2 * t2 * (GRAD2[gi2][0] * x2 + GRAD2[gi2][1] * y2);
    }

    // Step 7: Scale to approximately [-1, 1]
    70.0 * (n0 + n1 + n2)
}

// ── 3D Simplex Noise ────────────────────────────────────────────────

// Skew/unskew constants — 3D
const F3: f64 = 1.0 / 3.0;
const G3: f64 = 1.0 / 6.0;

/// 3D simplex noise.
/// Returns a value in approximately [-1, 1].
///
/// Must produce identical output to `createHytaleNoise3D` in `hytaleNoise.ts`.
pub fn simplex_3d(perm: &[u8; 512], x: f64, y: f64, z: f64) -> f64 {
    // Step 1: Skew input to simplex cell coordinates
    let s = (x + y + z) * F3;
    let i = (x + s).floor() as i64;
    let j = (y + s).floor() as i64;
    let k = (z + s).floor() as i64;

    // Step 2: Unskew to find cell origin
    let t = (i + j + k) as f64 * G3;
    let x0 = x - (i as f64 - t);
    let y0 = y - (j as f64 - t);
    let z0 = z - (k as f64 - t);

    // Step 3: Determine which simplex (tetrahedron) by comparing x0, y0, z0
    let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
        if y0 >= z0 {
            (1, 0, 0, 1, 1, 0) // XYZ
        } else if x0 >= z0 {
            (1, 0, 0, 1, 0, 1) // XZY
        } else {
            (0, 0, 1, 1, 0, 1) // ZXY
        }
    } else {
        if y0 < z0 {
            (0, 0, 1, 0, 1, 1) // ZYX
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1) // YZX
        } else {
            (0, 1, 0, 1, 1, 0) // YXZ
        }
    };

    // Step 4: Offsets for remaining corners
    let x1 = x0 - i1 as f64 + G3;
    let y1 = y0 - j1 as f64 + G3;
    let z1 = z0 - k1 as f64 + G3;
    let x2 = x0 - i2 as f64 + 2.0 * G3;
    let y2 = y0 - j2 as f64 + 2.0 * G3;
    let z2 = z0 - k2 as f64 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    // Step 5: Hash gradient indices
    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;
    let kk = (k & 255) as usize;
    let gi0 = (perm[ii + perm[jj + perm[kk] as usize] as usize] % 12) as usize;
    let gi1 = (perm[ii + i1 + perm[jj + j1 + perm[kk + k1] as usize] as usize] % 12) as usize;
    let gi2 = (perm[ii + i2 + perm[jj + j2 + perm[kk + k2] as usize] as usize] % 12) as usize;
    let gi3 = (perm[ii + 1 + perm[jj + 1 + perm[kk + 1] as usize] as usize] % 12) as usize;

    // Step 6: Corner contributions (kernel radius = 0.6 for 3D)
    let mut n0 = 0.0;
    let mut n1 = 0.0;
    let mut n2 = 0.0;
    let mut n3 = 0.0;

    let mut t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if t0 >= 0.0 {
        t0 *= t0;
        n0 = t0 * t0 * (GRAD3[gi0][0] * x0 + GRAD3[gi0][1] * y0 + GRAD3[gi0][2] * z0);
    }

    let mut t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if t1 >= 0.0 {
        t1 *= t1;
        n1 = t1 * t1 * (GRAD3[gi1][0] * x1 + GRAD3[gi1][1] * y1 + GRAD3[gi1][2] * z1);
    }

    let mut t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if t2 >= 0.0 {
        t2 *= t2;
        n2 = t2 * t2 * (GRAD3[gi2][0] * x2 + GRAD3[gi2][1] * y2 + GRAD3[gi2][2] * z2);
    }

    let mut t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if t3 >= 0.0 {
        t3 *= t3;
        n3 = t3 * t3 * (GRAD3[gi3][0] * x3 + GRAD3[gi3][1] * y3 + GRAD3[gi3][2] * z3);
    }

    // Step 7: Scale to approximately [-1, 1]
    32.0 * (n0 + n1 + n2 + n3)
}

// ── 2D Simplex Noise with analytic gradient ─────────────────────────

/// Result of a noise evaluation that also returns the analytic gradient.
pub struct NoiseWithGradient2D {
    pub value: f64,
    pub dx: f64,
    pub dy: f64,
}

/// 2D simplex noise with analytic gradient.
/// Used by FastGradientWarp for efficient gradient-based warping.
///
/// Must match `createHytaleNoise2DWithGradient` in `hytaleNoise.ts`.
pub fn simplex_2d_with_gradient(perm: &[u8; 512], x: f64, y: f64) -> NoiseWithGradient2D {
    let s = (x + y) * F2;
    let i = (x + s).floor() as i64;
    let j = (y + s).floor() as i64;
    let t = (i + j) as f64 * G2;
    let x0 = x - (i as f64 - t);
    let y0 = y - (j as f64 - t);

    let (i1, j1) = if x0 > y0 { (1i64, 0i64) } else { (0i64, 1i64) };

    let x1 = x0 - i1 as f64 + G2;
    let y1 = y0 - j1 as f64 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;
    let gi0 = (perm[ii + perm[jj] as usize] % 8) as usize;
    let gi1 = (perm[ii + i1 as usize + perm[jj + j1 as usize] as usize] % 8) as usize;
    let gi2 = (perm[ii + 1 + perm[jj + 1] as usize] % 8) as usize;

    let mut value = 0.0;
    let mut gdx = 0.0;
    let mut gdy = 0.0;

    // Corner 0
    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if t0 >= 0.0 {
        let g0x = GRAD2[gi0][0];
        let g0y = GRAD2[gi0][1];
        let dot0 = g0x * x0 + g0y * y0;
        let t02 = t0 * t0;
        let t04 = t02 * t02;
        value += t04 * dot0;
        let dt0dx = -2.0 * x0;
        let dt0dy = -2.0 * y0;
        gdx += t04 * g0x + 4.0 * t02 * t0 * dt0dx * dot0;
        gdy += t04 * g0y + 4.0 * t02 * t0 * dt0dy * dot0;
    }

    // Corner 1
    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if t1 >= 0.0 {
        let g1x = GRAD2[gi1][0];
        let g1y = GRAD2[gi1][1];
        let dot1 = g1x * x1 + g1y * y1;
        let t12 = t1 * t1;
        let t14 = t12 * t12;
        value += t14 * dot1;
        let dt1dx = -2.0 * x1;
        let dt1dy = -2.0 * y1;
        gdx += t14 * g1x + 4.0 * t12 * t1 * dt1dx * dot1;
        gdy += t14 * g1y + 4.0 * t12 * t1 * dt1dy * dot1;
    }

    // Corner 2
    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if t2 >= 0.0 {
        let g2x = GRAD2[gi2][0];
        let g2y = GRAD2[gi2][1];
        let dot2 = g2x * x2 + g2y * y2;
        let t22 = t2 * t2;
        let t24 = t22 * t22;
        value += t24 * dot2;
        let dt2dx = -2.0 * x2;
        let dt2dy = -2.0 * y2;
        gdx += t24 * g2x + 4.0 * t22 * t2 * dt2dx * dot2;
        gdy += t24 * g2y + 4.0 * t22 * t2 * dt2dy * dot2;
    }

    NoiseWithGradient2D {
        value: 70.0 * value,
        dx: 70.0 * gdx,
        dy: 70.0 * gdy,
    }
}

// ── 3D Simplex Noise with analytic gradient ─────────────────────────

/// Result of a 3D noise evaluation that also returns the analytic gradient.
pub struct NoiseWithGradient3D {
    pub value: f64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

/// 3D simplex noise with analytic gradient.
/// Must match `createHytaleNoise3DWithGradient` in `hytaleNoise.ts`.
pub fn simplex_3d_with_gradient(perm: &[u8; 512], x: f64, y: f64, z: f64) -> NoiseWithGradient3D {
    let s = (x + y + z) * F3;
    let i = (x + s).floor() as i64;
    let j = (y + s).floor() as i64;
    let k = (z + s).floor() as i64;
    let t = (i + j + k) as f64 * G3;
    let x0 = x - (i as f64 - t);
    let y0 = y - (j as f64 - t);
    let z0 = z - (k as f64 - t);

    let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
        if y0 >= z0 {
            (1, 0, 0, 1, 1, 0)
        } else if x0 >= z0 {
            (1, 0, 0, 1, 0, 1)
        } else {
            (0, 0, 1, 1, 0, 1)
        }
    } else {
        if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        }
    };

    let x1 = x0 - i1 as f64 + G3;
    let y1 = y0 - j1 as f64 + G3;
    let z1 = z0 - k1 as f64 + G3;
    let x2 = x0 - i2 as f64 + 2.0 * G3;
    let y2 = y0 - j2 as f64 + 2.0 * G3;
    let z2 = z0 - k2 as f64 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;
    let kk = (k & 255) as usize;
    let gi0 = (perm[ii + perm[jj + perm[kk] as usize] as usize] % 12) as usize;
    let gi1 = (perm[ii + i1 + perm[jj + j1 + perm[kk + k1] as usize] as usize] % 12) as usize;
    let gi2 = (perm[ii + i2 + perm[jj + j2 + perm[kk + k2] as usize] as usize] % 12) as usize;
    let gi3 = (perm[ii + 1 + perm[jj + 1 + perm[kk + 1] as usize] as usize] % 12) as usize;

    let mut value = 0.0;
    let mut gdx = 0.0;
    let mut gdy = 0.0;
    let mut gdz = 0.0;

    // Helper closure for each corner
    let mut corner = |gx: f64, gy: f64, gz: f64, ox: f64, oy: f64, oz: f64| {
        let tc = 0.6 - ox * ox - oy * oy - oz * oz;
        if tc >= 0.0 {
            let dot = gx * ox + gy * oy + gz * oz;
            let tc2 = tc * tc;
            let tc4 = tc2 * tc2;
            value += tc4 * dot;
            let dtdx = -2.0 * ox;
            let dtdy = -2.0 * oy;
            let dtdz = -2.0 * oz;
            gdx += tc4 * gx + 4.0 * tc2 * tc * dtdx * dot;
            gdy += tc4 * gy + 4.0 * tc2 * tc * dtdy * dot;
            gdz += tc4 * gz + 4.0 * tc2 * tc * dtdz * dot;
        }
    };

    corner(GRAD3[gi0][0], GRAD3[gi0][1], GRAD3[gi0][2], x0, y0, z0);
    corner(GRAD3[gi1][0], GRAD3[gi1][1], GRAD3[gi1][2], x1, y1, z1);
    corner(GRAD3[gi2][0], GRAD3[gi2][1], GRAD3[gi2][2], x2, y2, z2);
    corner(GRAD3[gi3][0], GRAD3[gi3][1], GRAD3[gi3][2], x3, y3, z3);

    NoiseWithGradient3D {
        value: 32.0 * value,
        dx: 32.0 * gdx,
        dy: 32.0 * gdy,
        dz: 32.0 * gdz,
    }
}

// ── FBM helpers ─────────────────────────────────────────────────────

/// Fractal Brownian Motion — 2D.
/// Matches `fbm2D` in `densityEvaluator.ts`.
pub fn fbm_2d(
    perm: &[u8; 512],
    x: f64,
    z: f64,
    freq: f64,
    octaves: u32,
    lacunarity: f64,
    gain: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut f = freq;
    for _ in 0..octaves {
        sum += simplex_2d(perm, x * f, z * f) * amp;
        f *= lacunarity;
        amp *= gain;
    }
    sum
}

/// Fractal Brownian Motion — 3D.
/// Matches `fbm3D` in `densityEvaluator.ts`.
pub fn fbm_3d(
    perm: &[u8; 512],
    x: f64,
    y: f64,
    z: f64,
    freq: f64,
    octaves: u32,
    lacunarity: f64,
    gain: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut f = freq;
    for _ in 0..octaves {
        sum += simplex_3d(perm, x * f, y * f, z * f) * amp;
        f *= lacunarity;
        amp *= gain;
    }
    sum
}

/// Ridge FBM — 2D.
/// Matches `ridgeFbm2D` in `densityEvaluator.ts`.
pub fn ridge_fbm_2d(perm: &[u8; 512], x: f64, z: f64, freq: f64, octaves: u32) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut f = freq;
    for _ in 0..octaves {
        let n = 1.0 - simplex_2d(perm, x * f, z * f).abs();
        sum += n * n * amp;
        f *= 2.0;
        amp *= 0.5;
    }
    sum * 2.0 - 1.0
}

/// Ridge FBM — 3D.
/// Matches `ridgeFbm3D` in `densityEvaluator.ts`.
pub fn ridge_fbm_3d(perm: &[u8; 512], x: f64, y: f64, z: f64, freq: f64, octaves: u32) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut f = freq;
    for _ in 0..octaves {
        let n = 1.0 - simplex_3d(perm, x * f, y * f, z * f).abs();
        sum += n * n * amp;
        f *= 2.0;
        amp *= 0.5;
    }
    sum * 2.0 - 1.0
}

// ── Voronoi Noise ───────────────────────────────────────────────────

/// Hash primes matching the TypeScript constants in `constants.ts`.
const HASH_PRIME_A: i64 = 374761393;
const HASH_PRIME_B: i64 = 668265263;
const HASH_PRIME_C: i64 = 1103515245;

/// 2D Voronoi (cellular) noise factory.
///
/// Returns a closure that computes voronoi noise at (x, y).
/// `cell_type` controls the distance metric:
///   - "Euclidean" / "Distance" → F1 * 2 - 1
///   - "Distance2Div" → (F1 / F2) * 2 - 1
///   - "Distance2Sub" → (F2 - F1) * 2 - 1
///
/// Matches `voronoiNoise2D` in `densityEvaluator.ts`.
pub fn voronoi_2d(x: f64, y: f64, _seed: i32, cell_type: &str, jitter: f64) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let mut d1 = f64::INFINITY;
    let mut d2 = f64::INFINITY;

    for dx in -1..=1i64 {
        for dy in -1..=1i64 {
            let cx = ix + dx;
            let cy = iy + dy;
            let s = hash_seed_int((cx * HASH_PRIME_A + cy * HASH_PRIME_B) as i32);
            let mut rng = Mulberry32::new(s);
            let px = cx as f64 + rng.next_f64() * jitter;
            let py = cy as f64 + rng.next_f64() * jitter;
            let dist = ((x - px) * (x - px) + (y - py) * (y - py)).sqrt();
            if dist < d1 {
                d2 = d1;
                d1 = dist;
            } else if dist < d2 {
                d2 = dist;
            }
        }
    }

    match cell_type {
        "Distance2Div" => {
            if d2 > 0.0 {
                (d1 / d2) * 2.0 - 1.0
            } else {
                0.0
            }
        }
        "Distance2Sub" => (d2 - d1) * 2.0 - 1.0,
        _ => d1 * 2.0 - 1.0, // Euclidean / Distance default
    }
}

/// 3D Voronoi (cellular) noise.
///
/// Matches `voronoiNoise3D` in `densityEvaluator.ts`.
pub fn voronoi_3d(x: f64, y: f64, z: f64, _seed: i32, cell_type: &str, jitter: f64) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let iz = z.floor() as i64;
    let mut d1 = f64::INFINITY;
    let mut d2 = f64::INFINITY;

    for dx in -1..=1i64 {
        for dy in -1..=1i64 {
            for dz in -1..=1i64 {
                let cx = ix + dx;
                let cy = iy + dy;
                let cz = iz + dz;
                let s = hash_seed_int(
                    (cx * HASH_PRIME_A + cy * HASH_PRIME_B + cz * HASH_PRIME_C) as i32,
                );
                let mut rng = Mulberry32::new(s);
                let px = cx as f64 + rng.next_f64() * jitter;
                let py = cy as f64 + rng.next_f64() * jitter;
                let pz = cz as f64 + rng.next_f64() * jitter;
                let dist = ((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz)).sqrt();
                if dist < d1 {
                    d2 = d1;
                    d1 = dist;
                } else if dist < d2 {
                    d2 = dist;
                }
            }
        }
    }

    match cell_type {
        "Distance2Div" => {
            if d2 > 0.0 {
                (d1 / d2) * 2.0 - 1.0
            } else {
                0.0
            }
        }
        "Distance2Sub" => (d2 - d1) * 2.0 - 1.0,
        _ => d1 * 2.0 - 1.0,
    }
}

/// Internal helper: hash an integer seed the same way TS `hashSeed(n)` → `seedToInt(n)` does.
/// For numeric seeds, TS does `seed | 0` (truncate to i32).
fn hash_seed_int(n: i32) -> i32 {
    n
}

/// Voronoi FBM — 2D.
/// Matches `voronoiFbm2D` in `densityEvaluator.ts`.
pub fn voronoi_fbm_2d(
    x: f64,
    z: f64,
    freq: f64,
    octaves: u32,
    lacunarity: f64,
    gain: f64,
    seed: i32,
    cell_type: &str,
    jitter: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut f = freq;
    for _ in 0..octaves {
        sum += voronoi_2d(x * f, z * f, seed, cell_type, jitter) * amp;
        f *= lacunarity;
        amp *= gain;
    }
    sum
}

/// Voronoi FBM — 3D.
/// Matches `voronoiFbm3D` in `densityEvaluator.ts`.
pub fn voronoi_fbm_3d(
    x: f64,
    y: f64,
    z: f64,
    freq: f64,
    octaves: u32,
    lacunarity: f64,
    gain: f64,
    seed: i32,
    cell_type: &str,
    jitter: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut f = freq;
    for _ in 0..octaves {
        sum += voronoi_3d(x * f, y * f, z * f, seed, cell_type, jitter) * amp;
        f *= lacunarity;
        amp *= gain;
    }
    sum
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_hash_code_empty() {
        assert_eq!(java_string_hash_code(""), 0);
    }

    #[test]
    fn java_hash_code_known_values() {
        // These values can be verified against Java's String.hashCode()
        assert_eq!(java_string_hash_code("hello"), 99162322);
        assert_eq!(java_string_hash_code("world"), 113318802);
        assert_eq!(java_string_hash_code("test"), 3556498);
        assert_eq!(java_string_hash_code("a"), 97);
        assert_eq!(java_string_hash_code("ab"), 3105);
    }

    #[test]
    fn seed_to_int_number() {
        let v = serde_json::json!(42);
        assert_eq!(seed_to_int(&v), 42);
    }

    #[test]
    fn seed_to_int_string() {
        let v = serde_json::json!("test");
        assert_eq!(seed_to_int(&v), java_string_hash_code("test"));
    }

    #[test]
    fn seed_to_int_null() {
        assert_eq!(seed_to_int(&Value::Null), 0);
    }

    #[test]
    fn mulberry32_sequence() {
        // Verify first few values of the PRNG with seed 0
        let mut rng = Mulberry32::new(0);
        let v0 = rng.next_f64();
        let v1 = rng.next_f64();
        let v2 = rng.next_f64();
        // Values should be in [0, 1)
        assert!(v0 >= 0.0 && v0 < 1.0);
        assert!(v1 >= 0.0 && v1 < 1.0);
        assert!(v2 >= 0.0 && v2 < 1.0);
        // Should be deterministic
        let mut rng2 = Mulberry32::new(0);
        assert_eq!(rng2.next_f64(), v0);
        assert_eq!(rng2.next_f64(), v1);
        assert_eq!(rng2.next_f64(), v2);
    }

    #[test]
    fn mulberry32_different_seeds() {
        let mut rng1 = Mulberry32::new(1);
        let mut rng2 = Mulberry32::new(2);
        // Different seeds should produce different sequences
        assert_ne!(rng1.next_f64(), rng2.next_f64());
    }

    #[test]
    fn perm_table_has_valid_entries() {
        let perm = build_perm_table(42);
        // Second half should mirror first half
        for i in 0..256 {
            assert_eq!(perm[i], perm[i + 256]);
        }
        // First 256 should be a permutation of 0..255
        let mut counts = [0u32; 256];
        for i in 0..256 {
            counts[perm[i] as usize] += 1;
        }
        for c in &counts {
            assert_eq!(*c, 1, "Each value 0..255 should appear exactly once");
        }
    }

    #[test]
    fn perm_table_deterministic() {
        let perm1 = build_perm_table(12345);
        let perm2 = build_perm_table(12345);
        assert_eq!(perm1, perm2);
    }

    #[test]
    fn simplex_2d_deterministic() {
        let perm = build_perm_table(42);
        let a = simplex_2d(&perm, 1.5, 2.5);
        let b = simplex_2d(&perm, 1.5, 2.5);
        assert_eq!(a, b);
    }

    #[test]
    fn simplex_2d_range() {
        let perm = build_perm_table(0);
        for i in 0..100 {
            let x = (i as f64) * 0.73 - 30.0;
            let y = (i as f64) * 1.17 - 50.0;
            let v = simplex_2d(&perm, x, y);
            assert!(
                v >= -1.5 && v <= 1.5,
                "simplex_2d({}, {}) = {} out of expected range",
                x,
                y,
                v
            );
        }
    }

    #[test]
    fn simplex_3d_deterministic() {
        let perm = build_perm_table(42);
        let a = simplex_3d(&perm, 1.5, 2.5, 3.5);
        let b = simplex_3d(&perm, 1.5, 2.5, 3.5);
        assert_eq!(a, b);
    }

    #[test]
    fn simplex_3d_range() {
        let perm = build_perm_table(0);
        for i in 0..100 {
            let x = (i as f64) * 0.73 - 30.0;
            let y = (i as f64) * 0.41 + 10.0;
            let z = (i as f64) * 1.17 - 50.0;
            let v = simplex_3d(&perm, x, y, z);
            assert!(
                v >= -1.5 && v <= 1.5,
                "simplex_3d({}, {}, {}) = {} out of expected range",
                x,
                y,
                z,
                v
            );
        }
    }

    #[test]
    fn fbm_2d_deterministic() {
        let perm = build_perm_table(42);
        let a = fbm_2d(&perm, 10.0, 20.0, 0.01, 4, 2.0, 0.5);
        let b = fbm_2d(&perm, 10.0, 20.0, 0.01, 4, 2.0, 0.5);
        assert_eq!(a, b);
    }

    #[test]
    fn fbm_3d_deterministic() {
        let perm = build_perm_table(42);
        let a = fbm_3d(&perm, 10.0, 64.0, 20.0, 0.01, 4, 2.0, 0.5);
        let b = fbm_3d(&perm, 10.0, 64.0, 20.0, 0.01, 4, 2.0, 0.5);
        assert_eq!(a, b);
    }

    #[test]
    fn ridge_fbm_2d_deterministic() {
        let perm = build_perm_table(42);
        let a = ridge_fbm_2d(&perm, 10.0, 20.0, 0.01, 4);
        let b = ridge_fbm_2d(&perm, 10.0, 20.0, 0.01, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn ridge_fbm_3d_deterministic() {
        let perm = build_perm_table(42);
        let a = ridge_fbm_3d(&perm, 10.0, 64.0, 20.0, 0.01, 4);
        let b = ridge_fbm_3d(&perm, 10.0, 64.0, 20.0, 0.01, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn gradient_2d_value_matches_plain() {
        let perm = build_perm_table(42);
        for i in 0..50 {
            let x = (i as f64) * 0.73 - 10.0;
            let y = (i as f64) * 1.17 - 20.0;
            let plain = simplex_2d(&perm, x, y);
            let grad = simplex_2d_with_gradient(&perm, x, y);
            assert!(
                (plain - grad.value).abs() < 1e-12,
                "Mismatch at ({}, {}): plain={}, grad.value={}",
                x,
                y,
                plain,
                grad.value
            );
        }
    }

    #[test]
    fn gradient_3d_value_matches_plain() {
        let perm = build_perm_table(42);
        for i in 0..50 {
            let x = (i as f64) * 0.73 - 10.0;
            let y = (i as f64) * 0.41 + 10.0;
            let z = (i as f64) * 1.17 - 20.0;
            let plain = simplex_3d(&perm, x, y, z);
            let grad = simplex_3d_with_gradient(&perm, x, y, z);
            assert!(
                (plain - grad.value).abs() < 1e-12,
                "Mismatch at ({}, {}, {}): plain={}, grad.value={}",
                x,
                y,
                z,
                plain,
                grad.value
            );
        }
    }
}
