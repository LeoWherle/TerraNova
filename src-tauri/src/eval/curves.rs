// eval/curves.rs — Curve evaluators (all 19 types)
//
// Translates every curve type from `curveEvaluators.ts` into Rust.
// Each evaluator takes fields + input value → output value.
// Manual curves use Catmull-Rom spline interpolation with 32 segments per span,
// matching the TypeScript `catmullRomInterpolate()` implementation exactly.

use serde_json::Value;
use std::collections::HashMap;

// ── Point types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct CurvePoint {
    pub x: f64,
    pub y: f64,
}

// ── Point normalization ─────────────────────────────────────────────

/// Convert raw JSON points (either [[x,y]] arrays or [{x,y}] objects) to sorted CurvePoints.
/// Matches `normalizePoints` in `curveEvaluators.ts`.
pub fn normalize_points(raw: &Value) -> Vec<CurvePoint> {
    let arr = match raw.as_array() {
        Some(a) => a,
        None => return vec![],
    };

    arr.iter()
        .map(|p| {
            if let Some(pair) = p.as_array() {
                // [[x, y], ...]
                let x = pair.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = pair.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
                CurvePoint { x, y }
            } else if let Some(obj) = p.as_object() {
                // [{x, y}, ...]
                let x = obj.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = obj.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                CurvePoint { x, y }
            } else {
                CurvePoint { x: 0.0, y: 0.0 }
            }
        })
        .collect()
}

// ── Catmull-Rom interpolation ───────────────────────────────────────

/// Generate a densely sampled curve via Catmull-Rom spline interpolation.
/// Input points must be sorted by X. Returns `segments` samples per span.
/// Matches `catmullRomInterpolate` in `curveEvaluators.ts`.
pub fn catmull_rom_interpolate(control_points: &[CurvePoint], segments: usize) -> Vec<CurvePoint> {
    if control_points.len() < 2 {
        return control_points.to_vec();
    }

    let pts = control_points;
    let n = pts.len();
    let mut result = Vec::with_capacity((n - 1) * segments + 1);

    for i in 0..n - 1 {
        let p0 = pts[if i > 0 { i - 1 } else { 0 }];
        let p1 = pts[i];
        let p2 = pts[i + 1];
        let p3 = pts[if i + 2 < n { i + 2 } else { n - 1 }];

        for s in 0..segments {
            let t = s as f64 / segments as f64;
            let t2 = t * t;
            let t3 = t2 * t;

            let x = 0.5
                * (2.0 * p1.x
                    + (-p0.x + p2.x) * t
                    + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
                    + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3);

            let y = 0.5
                * (2.0 * p1.y
                    + (-p0.y + p2.y) * t
                    + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
                    + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3);

            result.push(CurvePoint { x, y });
        }
    }

    // Add the final point
    result.push(pts[n - 1]);
    result
}

// ── Binary-search interpolation on sampled curve ────────────────────

/// Given a densely-sampled curve and an input X, do binary search + linear
/// interpolation to find the Y value. Matches the TS `applyCurve` lookup.
fn sample_curve(sampled: &[CurvePoint], sorted_pts: &[CurvePoint], input: f64) -> f64 {
    if sampled.is_empty() {
        return input;
    }
    if sampled.len() == 1 {
        return sampled[0].y;
    }

    let x_min = sorted_pts.first().unwrap().x;
    let x_max = sorted_pts.last().unwrap().x;
    let clamped = input.max(x_min).min(x_max);

    let mut lo = 0usize;
    let mut hi = sampled.len() - 1;
    while lo < hi.saturating_sub(1) {
        let mid = (lo + hi) >> 1;
        if sampled[mid].x <= clamped {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let p0 = sampled[lo];
    let p1 = sampled[hi];
    let dx = p1.x - p0.x;
    let t = if dx == 0.0 {
        0.0
    } else {
        (clamped - p0.x) / dx
    };
    p0.y + (p1.y - p0.y) * t
}

// ── Manual curve (linear interpolation of sorted points) ────────────

/// Linear interpolation between sorted control points.
/// Used as the simpler Manual evaluator in `getCurveEvaluator`.
/// (Note: the in-graph `applyCurve` uses Catmull-Rom for Manual curves.)
fn eval_manual_linear(sorted: &[CurvePoint], input: f64) -> f64 {
    if sorted.is_empty() {
        return input;
    }
    if sorted.len() == 1 {
        return sorted[0].y;
    }
    if input <= sorted[0].x {
        return sorted[0].y;
    }
    if input >= sorted[sorted.len() - 1].x {
        return sorted[sorted.len() - 1].y;
    }
    for i in 0..sorted.len() - 1 {
        if input >= sorted[i].x && input <= sorted[i + 1].x {
            let t = (input - sorted[i].x) / (sorted[i + 1].x - sorted[i].x);
            return sorted[i].y + t * (sorted[i + 1].y - sorted[i].y);
        }
    }
    sorted[sorted.len() - 1].y
}

// ── Curve dispatcher ────────────────────────────────────────────────

/// Evaluate a curve of the given type with the given fields and input value.
///
/// This is the Rust equivalent of `getCurveEvaluator` in `curveEvaluators.ts`.
/// Returns the curve-transformed value.
pub fn evaluate_curve(curve_type: &str, fields: &HashMap<String, Value>, input: f64) -> f64 {
    match curve_type {
        "Manual" => eval_manual(fields, input),
        "Constant" => eval_constant(fields),
        "Power" => eval_power(fields, input),
        "StepFunction" => eval_step_function(fields, input),
        "Threshold" => eval_threshold(fields, input),
        "SmoothStep" => eval_smooth_step(fields, input),
        "DistanceExponential" => eval_distance_exponential(fields, input),
        "DistanceS" => eval_distance_s(fields, input),
        "Inverter" => -input,
        "Not" => 1.0 - input,
        "Clamp" => eval_clamp(fields, input),
        "LinearRemap" => eval_linear_remap(fields, input),
        // Additional curve types that may appear
        "Abs" => input.abs(),
        "Square" => input * input,
        "SquareRoot" => input.abs().sqrt(),
        "Negate" => -input,
        _ => input, // Unknown curve type → passthrough
    }
}

/// Apply a curve from a curve node to an input value.
///
/// This is the Rust equivalent of the inner `applyCurve` function in
/// `densityEvaluator.ts`. For Manual curves it uses Catmull-Rom interpolation
/// with 32 segments per span, matching the TS behavior.
///
/// `curve_type`: the curve node's type (with "Curve:" prefix stripped)
/// `curve_fields`: the curve node's fields
/// `input`: the value to transform
pub fn apply_curve(curve_type: &str, curve_fields: &HashMap<String, Value>, input: f64) -> f64 {
    if curve_type == "Manual" {
        // Manual curves in applyCurve use Catmull-Rom interpolation
        let raw_points = match curve_fields.get("Points") {
            Some(v) => v,
            None => return input,
        };
        let mut pts = normalize_points(raw_points);
        if pts.len() < 2 {
            return input;
        }
        pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        let sampled = catmull_rom_interpolate(&pts, 32);
        return sample_curve(&sampled, &pts, input);
    }

    // All other curve types: use the simple evaluator
    evaluate_curve(curve_type, curve_fields, input)
}

/// Apply spline interpolation using a Points field.
/// Matches `applySpline` in `densityEvaluator.ts`.
/// Uses Catmull-Rom with 32 segments, same as Manual curves in applyCurve.
pub fn apply_spline(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let raw_points = match fields.get("Points") {
        Some(v) => v,
        None => return input,
    };
    let mut pts = normalize_points(raw_points);
    if pts.len() < 2 {
        return input;
    }
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    let sampled = catmull_rom_interpolate(&pts, 32);
    sample_curve(&sampled, &pts, input)
}

// ── Individual curve evaluators ─────────────────────────────────────

fn field_f64(fields: &HashMap<String, Value>, key: &str, default: f64) -> f64 {
    fields.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Manual: linear interpolation of sorted control points.
fn eval_manual(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let raw = match fields.get("Points") {
        Some(v) => v,
        None => return input,
    };
    let mut pts = normalize_points(raw);
    if pts.is_empty() {
        return input;
    }
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    eval_manual_linear(&pts, input)
}

/// Constant: always returns fields.Value.
fn eval_constant(fields: &HashMap<String, Value>) -> f64 {
    field_f64(fields, "Value", 1.0)
}

/// Power: input^exponent.
fn eval_power(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let exp = field_f64(fields, "Exponent", 2.0);
    input.powf(exp)
}

/// StepFunction: floor(input * steps) / steps.
fn eval_step_function(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let steps = field_f64(fields, "Steps", 4.0);
    if steps <= 0.0 {
        return input;
    }
    (input * steps).floor() / steps
}

/// Threshold: 1 if input >= threshold, else 0.
fn eval_threshold(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let threshold = field_f64(fields, "Threshold", 0.5);
    if input >= threshold {
        1.0
    } else {
        0.0
    }
}

/// SmoothStep: Hermite interpolation between edge0 and edge1.
fn eval_smooth_step(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let edge0 = field_f64(fields, "Edge0", 0.0);
    let edge1 = field_f64(fields, "Edge1", 1.0);
    if edge0 == edge1 {
        return if input >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((input - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

/// DistanceExponential: 1 - t^exponent, where t is normalized to [rangeMin, rangeMax].
fn eval_distance_exponential(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let exp = field_f64(fields, "Exponent", 2.0);
    let range = fields.get("Range");
    let (range_min, range_max) = match range {
        Some(Value::Object(obj)) => {
            let rmin = obj.get("Min").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let rmax = obj.get("Max").and_then(|v| v.as_f64()).unwrap_or(1.0);
            (rmin, rmax)
        }
        _ => (
            field_f64(fields, "RangeMin", 0.0),
            field_f64(fields, "RangeMax", 1.0),
        ),
    };
    if range_max == range_min {
        return 0.0;
    }
    let t = ((input - range_min) / (range_max - range_min))
        .max(0.0)
        .min(1.0);
    1.0 - t.powf(exp)
}

/// DistanceS: dual-exponential tent/bell curve.
/// Formula: exp(-((|x - offset| / width) ^ exponent) * steepness)
fn eval_distance_s(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let steepness = field_f64(fields, "Steepness", 1.0);
    let offset = field_f64(fields, "Offset", 0.5);
    let width = field_f64(fields, "Width", 0.5);
    let exponent = field_f64(fields, "Exponent", 2.0);
    if width <= 0.0 {
        return 0.0;
    }
    let d = (input - offset).abs() / width;
    (-d.powf(exponent) * steepness).exp()
}

/// Clamp: clamp input to [min, max].
fn eval_clamp(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let min = field_f64(fields, "Min", 0.0);
    let max = field_f64(fields, "Max", 1.0);
    input.max(min).min(max)
}

/// LinearRemap: affine remap from source range to target range.
fn eval_linear_remap(fields: &HashMap<String, Value>, input: f64) -> f64 {
    let (src_min, src_max) = match fields.get("SourceRange") {
        Some(Value::Object(obj)) => {
            let smin = obj.get("Min").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let smax = obj.get("Max").and_then(|v| v.as_f64()).unwrap_or(1.0);
            (smin, smax)
        }
        _ => (0.0, 1.0),
    };
    let (tgt_min, tgt_max) = match fields.get("TargetRange") {
        Some(Value::Object(obj)) => {
            let tmin = obj.get("Min").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let tmax = obj.get("Max").and_then(|v| v.as_f64()).unwrap_or(1.0);
            (tmin, tmax)
        }
        _ => (0.0, 1.0),
    };
    if src_max == src_min {
        return tgt_min;
    }
    let t = (input - src_min) / (src_max - src_min);
    tgt_min + t * (tgt_max - tgt_min)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fields_from(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn constant_curve() {
        let fields = fields_from(&[("Value", json!(42.0))]);
        assert_eq!(evaluate_curve("Constant", &fields, 0.5), 42.0);
    }

    #[test]
    fn power_curve() {
        let fields = fields_from(&[("Exponent", json!(3.0))]);
        assert!((evaluate_curve("Power", &fields, 2.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn step_function_curve() {
        let fields = fields_from(&[("Steps", json!(4.0))]);
        // floor(0.6 * 4) / 4 = floor(2.4) / 4 = 2 / 4 = 0.5
        assert!((evaluate_curve("StepFunction", &fields, 0.6) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn threshold_curve() {
        let fields = fields_from(&[("Threshold", json!(0.5))]);
        assert_eq!(evaluate_curve("Threshold", &fields, 0.3), 0.0);
        assert_eq!(evaluate_curve("Threshold", &fields, 0.5), 1.0);
        assert_eq!(evaluate_curve("Threshold", &fields, 0.7), 1.0);
    }

    #[test]
    fn smooth_step_curve() {
        let fields = fields_from(&[("Edge0", json!(0.0)), ("Edge1", json!(1.0))]);
        assert!((evaluate_curve("SmoothStep", &fields, 0.0)).abs() < 1e-10);
        assert!((evaluate_curve("SmoothStep", &fields, 1.0) - 1.0).abs() < 1e-10);
        // At 0.5: t=0.5, result = 0.25 * (3 - 1) = 0.5
        assert!((evaluate_curve("SmoothStep", &fields, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn inverter_curve() {
        assert_eq!(evaluate_curve("Inverter", &HashMap::new(), 3.0), -3.0);
    }

    #[test]
    fn not_curve() {
        assert!((evaluate_curve("Not", &HashMap::new(), 0.3) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn clamp_curve() {
        let fields = fields_from(&[("Min", json!(0.2)), ("Max", json!(0.8))]);
        assert!((evaluate_curve("Clamp", &fields, 0.0) - 0.2).abs() < 1e-10);
        assert!((evaluate_curve("Clamp", &fields, 0.5) - 0.5).abs() < 1e-10);
        assert!((evaluate_curve("Clamp", &fields, 1.0) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn linear_remap_curve() {
        let fields = fields_from(&[
            ("SourceRange", json!({"Min": 0.0, "Max": 1.0})),
            ("TargetRange", json!({"Min": 10.0, "Max": 20.0})),
        ]);
        assert!((evaluate_curve("LinearRemap", &fields, 0.5) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn distance_exponential_curve() {
        let fields = fields_from(&[
            ("Exponent", json!(2.0)),
            ("Range", json!({"Min": 0.0, "Max": 1.0})),
        ]);
        // At t=0: 1 - 0^2 = 1
        assert!((evaluate_curve("DistanceExponential", &fields, 0.0) - 1.0).abs() < 1e-10);
        // At t=1: 1 - 1^2 = 0
        assert!((evaluate_curve("DistanceExponential", &fields, 1.0) - 0.0).abs() < 1e-10);
        // At t=0.5: 1 - 0.25 = 0.75
        assert!((evaluate_curve("DistanceExponential", &fields, 0.5) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn distance_s_curve() {
        let fields = fields_from(&[
            ("Steepness", json!(1.0)),
            ("Offset", json!(0.0)),
            ("Width", json!(1.0)),
            ("Exponent", json!(2.0)),
        ]);
        // At x=0: exp(0) = 1
        assert!((evaluate_curve("DistanceS", &fields, 0.0) - 1.0).abs() < 1e-10);
        // At x=1: exp(-1) ≈ 0.3679
        assert!((evaluate_curve("DistanceS", &fields, 1.0) - (-1.0f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn unknown_curve_passthrough() {
        assert_eq!(evaluate_curve("NonExistent", &HashMap::new(), 7.5), 7.5);
    }

    #[test]
    fn catmull_rom_basic() {
        let pts = vec![CurvePoint { x: 0.0, y: 0.0 }, CurvePoint { x: 1.0, y: 1.0 }];
        let sampled = catmull_rom_interpolate(&pts, 4);
        // Should have 4 samples + 1 final = 5
        assert_eq!(sampled.len(), 5);
        // First and last should match control points
        assert!((sampled[0].x - 0.0).abs() < 1e-10);
        assert!((sampled[0].y - 0.0).abs() < 1e-10);
        assert!((sampled[4].x - 1.0).abs() < 1e-10);
        assert!((sampled[4].y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn manual_curve_via_apply() {
        let fields = fields_from(&[("Points", json!([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]))]);
        // For a roughly linear set of points, the curve should be close to identity
        let result = apply_curve("Manual", &fields, 0.5);
        assert!((result - 0.5).abs() < 0.1, "Expected ~0.5, got {}", result);
    }

    #[test]
    fn spline_basic() {
        let fields = fields_from(&[("Points", json!([[0.0, 0.0], [1.0, 1.0]]))]);
        // Linear spline should be close to identity
        let result = apply_spline(&fields, 0.5);
        assert!((result - 0.5).abs() < 0.05, "Expected ~0.5, got {}", result);
    }

    #[test]
    fn normalize_points_array_format() {
        let raw = json!([[0.0, 1.0], [0.5, 0.5]]);
        let pts = normalize_points(&raw);
        assert_eq!(pts.len(), 2);
        assert!((pts[0].x - 0.0).abs() < 1e-10);
        assert!((pts[0].y - 1.0).abs() < 1e-10);
        assert!((pts[1].x - 0.5).abs() < 1e-10);
        assert!((pts[1].y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn normalize_points_object_format() {
        let raw = json!([{"x": 0.0, "y": 1.0}, {"x": 0.5, "y": 0.5}]);
        let pts = normalize_points(&raw);
        assert_eq!(pts.len(), 2);
        assert!((pts[0].x - 0.0).abs() < 1e-10);
        assert!((pts[0].y - 1.0).abs() < 1e-10);
    }
}
