// eval/graph.rs — React Flow graph → evaluable tree
//
// Parses the flat { nodes, edges } format that React Flow uses into an
// indexed graph structure ready for recursive evaluation.

use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Known density node types (mirrors the TS `DENSITY_TYPES` set).
const DENSITY_TYPES: &[&str] = &[
    "SimplexNoise2D",
    "SimplexNoise3D",
    "SimplexRidgeNoise2D",
    "SimplexRidgeNoise3D",
    "VoronoiNoise2D",
    "VoronoiNoise3D",
    "FractalNoise2D",
    "FractalNoise3D",
    "DomainWarp2D",
    "DomainWarp3D",
    "Sum",
    "SumSelf",
    "WeightedSum",
    "Product",
    "Negate",
    "Abs",
    "SquareRoot",
    "CubeRoot",
    "Square",
    "CubeMath",
    "Inverse",
    "Modulo",
    "Constant",
    "ImportedValue",
    "Zero",
    "One",
    "Clamp",
    "ClampToIndex",
    "Normalizer",
    "DoubleNormalizer",
    "RangeChoice",
    "LinearTransform",
    "Interpolate",
    "CoordinateX",
    "CoordinateY",
    "CoordinateZ",
    "DistanceFromOrigin",
    "DistanceFromAxis",
    "DistanceFromPoint",
    "AngleFromOrigin",
    "AngleFromPoint",
    "HeightAboveSurface",
    "CurveFunction",
    "SplineFunction",
    "FlatCache",
    "Conditional",
    "Switch",
    "Blend",
    "BlendCurve",
    "MinFunction",
    "MaxFunction",
    "AverageFunction",
    "CacheOnce",
    "Wrap",
    "TranslatedPosition",
    "ScaledPosition",
    "RotatedPosition",
    "MirroredPosition",
    "QuantizedPosition",
    "SurfaceDensity",
    "TerrainBoolean",
    "TerrainMask",
    "GradientDensity",
    "BeardDensity",
    "ColumnDensity",
    "CaveDensity",
    "Debug",
    "YGradient",
    "Passthrough",
    "BaseHeight",
    "Floor",
    "Ceiling",
    "SmoothClamp",
    "SmoothFloor",
    "SmoothMin",
    "SmoothMax",
    "YOverride",
    "Anchor",
    "Exported",
    "Offset",
    "Distance",
    "PositionsCellNoise",
    "AmplitudeConstant",
    "Pow",
    "XOverride",
    "ZOverride",
    "SmoothCeiling",
    "Gradient",
    "Amplitude",
    "YSampled",
    "SwitchState",
    "MultiMix",
    "Positions3D",
    "PositionsPinch",
    "PositionsTwist",
    "GradientWarp",
    "FastGradientWarp",
    "VectorWarp",
    "Terrain",
    "CellWallDistance",
    "DistanceToBiomeEdge",
    "Pipeline",
    "Ellipsoid",
    "Cuboid",
    "Cylinder",
    "Plane",
    "Shell",
    "Angle",
];

/// A node from the React Flow graph, as sent over IPC.
/// We only deserialize the fields we need — id and data.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub data: NodeData,
    #[serde(rename = "type")]
    pub node_type: Option<String>,
}

/// The `data` payload of a React Flow node.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeData {
    /// Density function type, e.g. "SimplexNoise2D", "Constant", "Sum"
    #[serde(rename = "type", default)]
    pub density_type: Option<String>,

    /// Field values (Frequency, Amplitude, Seed, Value, Min, Max, …)
    #[serde(default)]
    pub fields: HashMap<String, Value>,

    /// User-designated output node flag
    #[serde(rename = "_outputNode", default)]
    pub is_output: bool,

    /// Biome field tag (e.g. "Terrain")
    #[serde(rename = "_biomeField", default)]
    pub biome_field: Option<String>,
}

/// An edge from the React Flow graph.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    #[serde(rename = "targetHandle")]
    pub target_handle: Option<String>,
}

/// Parsed, indexed graph ready for evaluation.
pub struct EvalGraph {
    /// All nodes keyed by id.
    pub nodes: HashMap<String, GraphNode>,
    /// Input adjacency: target_id → { handle_name → source_id }
    pub inputs: HashMap<String, HashMap<String, String>>,
    /// The root node to start evaluation from.
    pub root_id: String,
}

impl EvalGraph {
    /// Build an `EvalGraph` from raw React Flow nodes and edges.
    pub fn from_raw(
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        root_node_id: Option<&str>,
    ) -> Result<Self, String> {
        // Build node map
        let node_map: HashMap<String, GraphNode> =
            nodes.into_iter().map(|n| (n.id.clone(), n)).collect();

        // Build input adjacency: target → { handle → source }
        let mut inputs: HashMap<String, HashMap<String, String>> = HashMap::new();
        for edge in &edges {
            let handle = edge.target_handle.clone().unwrap_or_else(|| "Input".into());
            inputs
                .entry(edge.target.clone())
                .or_default()
                .insert(handle, edge.source.clone());
        }

        // Find root (same strategy as TS findDensityRoot())
        let root_id = Self::find_root(&node_map, &edges, root_node_id)?;

        Ok(EvalGraph {
            nodes: node_map,
            inputs,
            root_id,
        })
    }

    /// Determine which node is the evaluation root.
    ///
    /// Strategy (matches TypeScript `findDensityRoot`):
    ///   0. Explicit `root_node_id` parameter
    ///   1. `_outputNode === true`
    ///   2. `_biomeField === "Terrain"`
    ///   3. Terminal density nodes (no outgoing edges)
    ///   4. Any density node
    fn find_root(
        nodes: &HashMap<String, GraphNode>,
        edges: &[GraphEdge],
        explicit_id: Option<&str>,
    ) -> Result<String, String> {
        let density_set: HashSet<&str> = DENSITY_TYPES.iter().copied().collect();

        let is_density = |node: &GraphNode| -> bool {
            node.data
                .density_type
                .as_deref()
                .map(|t| density_set.contains(t))
                .unwrap_or(false)
        };

        // Strategy 0: explicit root_node_id
        if let Some(id) = explicit_id {
            if nodes.contains_key(id) {
                return Ok(id.to_string());
            }
        }

        // Strategy 1: _outputNode === true
        for node in nodes.values() {
            if node.data.is_output {
                return Ok(node.id.clone());
            }
        }

        // Strategy 2: _biomeField === "Terrain"
        for node in nodes.values() {
            if node.data.biome_field.as_deref() == Some("Terrain") {
                return Ok(node.id.clone());
            }
        }

        // Strategy 3: terminal nodes (no outgoing edges) that are density types
        let sources_with_outgoing: HashSet<&str> =
            edges.iter().map(|e| e.source.as_str()).collect();

        let terminal_density = nodes
            .values()
            .find(|n| !sources_with_outgoing.contains(n.id.as_str()) && is_density(n));

        if let Some(node) = terminal_density {
            return Ok(node.id.clone());
        }

        // Strategy 4: any density node
        let any_density = nodes.values().find(|n| is_density(n));
        if let Some(node) = any_density {
            return Ok(node.id.clone());
        }

        Err("No evaluable root node found in graph".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, density_type: &str) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            data: NodeData {
                density_type: Some(density_type.to_string()),
                fields: HashMap::new(),
                is_output: false,
                biome_field: None,
            },
            node_type: None,
        }
    }

    fn make_output_node(id: &str, density_type: &str) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            data: NodeData {
                density_type: Some(density_type.to_string()),
                fields: HashMap::new(),
                is_output: true,
                biome_field: None,
            },
            node_type: None,
        }
    }

    fn make_terrain_node(id: &str, density_type: &str) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            data: NodeData {
                density_type: Some(density_type.to_string()),
                fields: HashMap::new(),
                is_output: false,
                biome_field: Some("Terrain".to_string()),
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

    #[test]
    fn explicit_root_node_id() {
        let nodes = vec![make_node("a", "Constant"), make_node("b", "Constant")];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("b")).unwrap();
        assert_eq!(graph.root_id, "b");
    }

    #[test]
    fn output_node_takes_priority() {
        let nodes = vec![
            make_node("a", "Constant"),
            make_output_node("b", "Constant"),
        ];
        let graph = EvalGraph::from_raw(nodes, vec![], None).unwrap();
        assert_eq!(graph.root_id, "b");
    }

    #[test]
    fn terrain_biome_field() {
        let nodes = vec![make_node("a", "Constant"), make_terrain_node("b", "Sum")];
        let graph = EvalGraph::from_raw(nodes, vec![], None).unwrap();
        assert_eq!(graph.root_id, "b");
    }

    #[test]
    fn terminal_density_node() {
        let nodes = vec![make_node("a", "Constant"), make_node("b", "Sum")];
        let edges = vec![make_edge("a", "b", Some("Inputs[0]"))];
        let graph = EvalGraph::from_raw(nodes, edges, None).unwrap();
        // "b" has no outgoing edges → terminal
        assert_eq!(graph.root_id, "b");
    }

    #[test]
    fn input_adjacency_built_correctly() {
        let nodes = vec![
            make_node("a", "Constant"),
            make_node("b", "Constant"),
            make_node("s", "Sum"),
        ];
        let edges = vec![
            make_edge("a", "s", Some("Inputs[0]")),
            make_edge("b", "s", Some("Inputs[1]")),
        ];
        let graph = EvalGraph::from_raw(nodes, edges, None).unwrap();
        let s_inputs = graph.inputs.get("s").unwrap();
        assert_eq!(s_inputs.get("Inputs[0]").unwrap(), "a");
        assert_eq!(s_inputs.get("Inputs[1]").unwrap(), "b");
    }

    #[test]
    fn default_handle_when_missing() {
        let nodes = vec![make_node("a", "Constant"), make_node("b", "Negate")];
        let edges = vec![make_edge("a", "b", None)];
        let graph = EvalGraph::from_raw(nodes, edges, None).unwrap();
        let b_inputs = graph.inputs.get("b").unwrap();
        assert_eq!(b_inputs.get("Input").unwrap(), "a");
    }

    #[test]
    fn empty_graph_returns_error() {
        let result = EvalGraph::from_raw(vec![], vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn non_density_only_graph_returns_error() {
        let nodes = vec![GraphNode {
            id: "x".to_string(),
            data: NodeData {
                density_type: Some("UnknownType".to_string()),
                fields: HashMap::new(),
                is_output: false,
                biome_field: None,
            },
            node_type: None,
        }];
        let result = EvalGraph::from_raw(nodes, vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn deserialize_from_json() {
        let json = r#"[
            {"id": "c", "data": {"type": "Constant", "fields": {"Value": 42}}}
        ]"#;
        let nodes: Vec<GraphNode> = serde_json::from_str(json).unwrap();
        assert_eq!(nodes[0].id, "c");
        assert_eq!(nodes[0].data.density_type.as_deref(), Some("Constant"));
        assert_eq!(
            nodes[0].data.fields.get("Value").and_then(|v| v.as_f64()),
            Some(42.0)
        );
    }
}
