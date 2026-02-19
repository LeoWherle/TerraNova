// eval/graph.rs — React Flow graph → evaluable tree
//
// Parses the flat { nodes, edges } format that React Flow uses into an
// indexed graph structure ready for recursive evaluation.
//
// Performance: nodes are stored both by-name (for backward compatibility)
// and by-index (for the fast evaluator). The indexed path eliminates all
// String hashing from the hot evaluation loop.
//
// The compiled layer (`ResolvedInputs` + density type tags) further
// eliminates per-evaluation string operations by pre-resolving all input
// handle names into fixed-position array slots and mapping density type
// strings to u16 tags at construction time.

use crate::eval::compiled::{self, ResolvedInputs};
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
///
/// Contains both string-keyed maps (for backward compatibility and IPC
/// commands) and flat indexed vectors (for the fast evaluation path).
/// The indexed path eliminates all String hashing from the hot loop.
///
/// The compiled layer adds:
/// - `resolved`: per-node `ResolvedInputs` with fixed-position array slots
///   for input handles (zero string hashing at eval time)
/// - `type_tags`: per-node u16 density type tag (eliminates string matching)
pub struct EvalGraph {
    /// All nodes keyed by id (legacy — used by commands and tests).
    pub nodes: HashMap<String, GraphNode>,
    /// Input adjacency: target_id → { handle_name → source_id } (legacy).
    pub inputs: HashMap<String, HashMap<String, String>>,
    /// The root node to start evaluation from (string id).
    pub root_id: String,

    // ── Indexed fields (fast path) ──────────────────────────────────
    /// Flat node list indexed by `usize`. Contiguous memory for cache
    /// locality. Built once at construction time.
    pub node_list: Vec<GraphNode>,
    /// Map from node string id → dense index into `node_list`.
    pub id_to_idx: HashMap<String, usize>,
    /// Root node index (indexes into `node_list`).
    pub root_idx: usize,
    /// Per-node input adjacency indexed by node index.
    /// `inputs_by_idx[target_idx]` maps handle_name → source_node_idx.
    pub inputs_by_idx: Vec<HashMap<String, usize>>,

    // ── Compiled fields (zero-overhead path) ────────────────────────
    /// Per-node pre-resolved inputs. Named handles are fixed-position array
    /// slots; array handles are pre-sorted SmallVecs. No string operations
    /// at evaluation time.
    pub resolved: Vec<ResolvedInputs>,
    /// Per-node density type tag (u16). Eliminates string matching in the
    /// evaluation dispatch loop.
    pub type_tags: Vec<u16>,
}

impl EvalGraph {
    /// Number of nodes in the graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_list.len()
    }

    /// Look up the dense index for a node id. Returns `None` if not found.
    #[inline]
    pub fn idx_of(&self, id: &str) -> Option<usize> {
        self.id_to_idx.get(id).copied()
    }
}

impl EvalGraph {
    /// Build an `EvalGraph` from raw React Flow nodes and edges.
    ///
    /// Constructs both the legacy string-keyed maps, the fast indexed
    /// vectors, and the compiled zero-overhead representation in a
    /// single pass.
    pub fn from_raw(
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        root_node_id: Option<&str>,
    ) -> Result<Self, String> {
        // ── 1. Build legacy string-keyed maps ──

        let node_map: HashMap<String, GraphNode> =
            nodes.into_iter().map(|n| (n.id.clone(), n)).collect();

        let mut inputs: HashMap<String, HashMap<String, String>> = HashMap::new();
        for edge in &edges {
            let handle = edge.target_handle.clone().unwrap_or_else(|| "Input".into());
            inputs
                .entry(edge.target.clone())
                .or_default()
                .insert(handle, edge.source.clone());
        }

        let root_id = Self::find_root(&node_map, &edges, root_node_id)?;

        // ── 2. Build dense index mapping ──

        // Deterministic ordering: sort node ids so indices are stable.
        let mut sorted_ids: Vec<&String> = node_map.keys().collect();
        sorted_ids.sort();

        let id_to_idx: HashMap<String, usize> = sorted_ids
            .iter()
            .enumerate()
            .map(|(i, id)| ((*id).clone(), i))
            .collect();

        let node_list: Vec<GraphNode> = sorted_ids.iter().map(|id| node_map[*id].clone()).collect();

        let root_idx = *id_to_idx
            .get(&root_id)
            .ok_or_else(|| "Root node id not in index map".to_string())?;

        // ── 3. Build per-node indexed input adjacency ──

        let node_count = node_list.len();
        let mut inputs_by_idx: Vec<HashMap<String, usize>> =
            (0..node_count).map(|_| HashMap::new()).collect();

        for (target_id, handle_map) in &inputs {
            if let Some(&target_idx) = id_to_idx.get(target_id) {
                for (handle, source_id) in handle_map {
                    if let Some(&source_idx) = id_to_idx.get(source_id) {
                        inputs_by_idx[target_idx].insert(handle.clone(), source_idx);
                    }
                }
            }
        }

        // ── 4. Build compiled representation ──

        // Resolve inputs: convert string-keyed HashMap per node into
        // fixed-position array slots + pre-sorted SmallVecs.
        let resolved: Vec<ResolvedInputs> = inputs_by_idx
            .iter()
            .map(|handle_map| compiled::resolve_inputs(handle_map))
            .collect();

        // Map density type strings to u16 tags.
        let type_tags: Vec<u16> = node_list
            .iter()
            .map(|node| {
                node.data
                    .density_type
                    .as_deref()
                    .map(compiled::density_type_to_tag)
                    .unwrap_or(compiled::DT_UNKNOWN)
            })
            .collect();

        Ok(EvalGraph {
            nodes: node_map,
            inputs,
            root_id,
            node_list,
            id_to_idx,
            root_idx,
            inputs_by_idx,
            resolved,
            type_tags,
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

    // ── Compiled layer tests ────────────────────────────────────────

    #[test]
    fn resolved_inputs_for_sum_node() {
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

        let s_idx = graph.idx_of("s").unwrap();
        let a_idx = graph.idx_of("a").unwrap();
        let b_idx = graph.idx_of("b").unwrap();

        let ri = &graph.resolved[s_idx];
        assert_eq!(ri.array_inputs.len(), 2);
        // Pre-sorted by array index: [0]→a, [1]→b
        assert_eq!(ri.array_inputs[0] as usize, a_idx);
        assert_eq!(ri.array_inputs[1] as usize, b_idx);
    }

    #[test]
    fn resolved_inputs_for_negate_node() {
        use crate::eval::compiled::H_INPUT;

        let nodes = vec![make_node("a", "Constant"), make_node("n", "Negate")];
        let edges = vec![make_edge("a", "n", None)]; // defaults to "Input"
        let graph = EvalGraph::from_raw(nodes, edges, None).unwrap();

        let n_idx = graph.idx_of("n").unwrap();
        let a_idx = graph.idx_of("a").unwrap();

        let ri = &graph.resolved[n_idx];
        assert!(ri.has(H_INPUT));
        assert_eq!(ri.get(H_INPUT), Some(a_idx));
    }

    #[test]
    fn type_tags_assigned_correctly() {
        use crate::eval::compiled::{DT_CONSTANT, DT_SUM};

        let nodes = vec![make_node("a", "Constant"), make_node("s", "Sum")];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("s")).unwrap();

        let a_idx = graph.idx_of("a").unwrap();
        let s_idx = graph.idx_of("s").unwrap();

        assert_eq!(graph.type_tags[a_idx], DT_CONSTANT);
        assert_eq!(graph.type_tags[s_idx], DT_SUM);
    }

    #[test]
    fn resolved_inputs_for_blend_node() {
        use crate::eval::compiled::{H_FACTOR, H_INPUT_A, H_INPUT_B};

        let nodes = vec![
            make_node("a", "Constant"),
            make_node("b", "Constant"),
            make_node("f", "Constant"),
            make_node("bl", "Blend"),
        ];
        let edges = vec![
            make_edge("a", "bl", Some("InputA")),
            make_edge("b", "bl", Some("InputB")),
            make_edge("f", "bl", Some("Factor")),
        ];
        let graph = EvalGraph::from_raw(nodes, edges, None).unwrap();

        let bl_idx = graph.idx_of("bl").unwrap();
        let a_idx = graph.idx_of("a").unwrap();
        let b_idx = graph.idx_of("b").unwrap();
        let f_idx = graph.idx_of("f").unwrap();

        let ri = &graph.resolved[bl_idx];
        assert_eq!(ri.get(H_INPUT_A), Some(a_idx));
        assert_eq!(ri.get(H_INPUT_B), Some(b_idx));
        assert_eq!(ri.get(H_FACTOR), Some(f_idx));
    }

    #[test]
    fn resolved_inputs_no_connections() {
        use crate::eval::compiled::H_INPUT;

        let nodes = vec![make_node("c", "Constant")];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("c")).unwrap();

        let c_idx = graph.idx_of("c").unwrap();
        let ri = &graph.resolved[c_idx];
        assert!(!ri.has(H_INPUT));
        assert!(ri.array_inputs.is_empty());
    }

    #[test]
    fn node_count_matches() {
        let nodes = vec![
            make_node("a", "Constant"),
            make_node("b", "Sum"),
            make_node("c", "Negate"),
        ];
        let graph = EvalGraph::from_raw(nodes, vec![], Some("a")).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.resolved.len(), 3);
        assert_eq!(graph.type_tags.len(), 3);
    }
}
