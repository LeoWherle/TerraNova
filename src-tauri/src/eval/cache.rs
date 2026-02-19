// eval/cache.rs — LRU evaluation cache with deterministic graph hashing
//
// Caches grid and volume evaluation results keyed by a hash of the graph
// structure + evaluation parameters. Thread-safe via Mutex.
//
// Results are stored behind `Arc` so cache hits return a cheap reference
// count bump instead of cloning potentially multi-MB `Vec<f32>` buffers.

use crate::eval::graph::{GraphEdge, GraphNode};
use crate::eval::grid::GridResult;
use crate::eval::volume::VolumeResult;
use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

/// Content-addressable cache for evaluation results.
/// Key = hash(graph_structure + params), Value = Arc<result>.
///
/// Using `Arc` means cache hits only bump a reference count instead of
/// cloning the entire result buffer (which can be 64³ = 262 144 floats
/// for volume results — over 1 MB).
pub struct EvalCache {
    grid_cache: Mutex<LruCache<u64, Arc<GridResult>>>,
    volume_cache: Mutex<LruCache<u64, Arc<VolumeResult>>>,
}

impl EvalCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            grid_cache: Mutex::new(LruCache::new(cap)),
            volume_cache: Mutex::new(LruCache::new(cap)),
        }
    }

    // ── Grid cache ──

    /// Get a cached grid result. Returns a cheap `Arc` clone (ref-count bump).
    pub fn get_grid(&self, hash: u64) -> Option<Arc<GridResult>> {
        self.grid_cache.lock().unwrap().get(&hash).cloned()
    }

    /// Insert a grid result into the cache, wrapping it in an `Arc`.
    pub fn put_grid(&self, hash: u64, result: GridResult) {
        self.grid_cache.lock().unwrap().put(hash, Arc::new(result));
    }

    /// Insert a pre-wrapped `Arc<GridResult>` into the cache.
    pub fn put_grid_arc(&self, hash: u64, result: Arc<GridResult>) {
        self.grid_cache.lock().unwrap().put(hash, result);
    }

    // ── Volume cache ──

    /// Get a cached volume result. Returns a cheap `Arc` clone (ref-count bump).
    pub fn get_volume(&self, hash: u64) -> Option<Arc<VolumeResult>> {
        self.volume_cache.lock().unwrap().get(&hash).cloned()
    }

    /// Insert a volume result into the cache, wrapping it in an `Arc`.
    pub fn put_volume(&self, hash: u64, result: VolumeResult) {
        self.volume_cache
            .lock()
            .unwrap()
            .put(hash, Arc::new(result));
    }

    /// Insert a pre-wrapped `Arc<VolumeResult>` into the cache.
    pub fn put_volume_arc(&self, hash: u64, result: Arc<VolumeResult>) {
        self.volume_cache.lock().unwrap().put(hash, result);
    }

    // ── Maintenance ──

    /// Clear all cached entries (useful when the user explicitly invalidates).
    pub fn clear(&self) {
        self.grid_cache.lock().unwrap().clear();
        self.volume_cache.lock().unwrap().clear();
    }

    /// Number of entries currently cached (grid + volume).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        let g = self.grid_cache.lock().unwrap().len();
        let v = self.volume_cache.lock().unwrap().len();
        g + v
    }
}

// ── Graph hashing ──────────────────────────────────────────────────

/// Hash the graph structure + evaluation parameters deterministically.
/// Same graph + params = same hash.
///
/// We hash:
///   - Number of nodes, and for each node: id, density_type, sorted fields
///   - Number of edges, and for each edge: source, target, target_handle
///   - All numeric parameters (resolution, ranges, etc.)
pub fn hash_grid_request(
    nodes: &[GraphNode],
    edges: &[GraphEdge],
    resolution: u32,
    range_min: f64,
    range_max: f64,
    y_level: f64,
    root_node_id: Option<&str>,
    content_fields: &std::collections::HashMap<String, f64>,
) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Discriminant so grid and volume hashes never collide
    "grid".hash(&mut hasher);

    hash_graph(&mut hasher, nodes, edges);

    resolution.hash(&mut hasher);
    range_min.to_bits().hash(&mut hasher);
    range_max.to_bits().hash(&mut hasher);
    y_level.to_bits().hash(&mut hasher);

    if let Some(id) = root_node_id {
        true.hash(&mut hasher);
        id.hash(&mut hasher);
    } else {
        false.hash(&mut hasher);
    }

    hash_content_fields(&mut hasher, content_fields);

    hasher.finish()
}

pub fn hash_volume_request(
    nodes: &[GraphNode],
    edges: &[GraphEdge],
    resolution: u32,
    range_min: f64,
    range_max: f64,
    y_min: f64,
    y_max: f64,
    y_slices: u32,
    root_node_id: Option<&str>,
    content_fields: &std::collections::HashMap<String, f64>,
) -> u64 {
    let mut hasher = DefaultHasher::new();

    "volume".hash(&mut hasher);

    hash_graph(&mut hasher, nodes, edges);

    resolution.hash(&mut hasher);
    range_min.to_bits().hash(&mut hasher);
    range_max.to_bits().hash(&mut hasher);
    y_min.to_bits().hash(&mut hasher);
    y_max.to_bits().hash(&mut hasher);
    y_slices.hash(&mut hasher);

    if let Some(id) = root_node_id {
        true.hash(&mut hasher);
        id.hash(&mut hasher);
    } else {
        false.hash(&mut hasher);
    }

    hash_content_fields(&mut hasher, content_fields);

    hasher.finish()
}

/// Hash the graph structure (nodes + edges) deterministically.
///
/// Nodes are sorted by id so that insertion order doesn't matter.
/// Fields within each node are sorted by key name.
fn hash_graph(hasher: &mut DefaultHasher, nodes: &[GraphNode], edges: &[GraphEdge]) {
    // Sort nodes by id for deterministic ordering
    let mut sorted_node_ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    sorted_node_ids.sort();

    sorted_node_ids.len().hash(hasher);
    for id in &sorted_node_ids {
        let node = nodes.iter().find(|n| n.id.as_str() == *id).unwrap();
        node.id.hash(hasher);

        // Hash density type
        if let Some(ref dt) = node.data.density_type {
            true.hash(hasher);
            dt.hash(hasher);
        } else {
            false.hash(hasher);
        }

        // Hash fields in sorted order for determinism
        let mut field_keys: Vec<&String> = node.data.fields.keys().collect();
        field_keys.sort();
        field_keys.len().hash(hasher);
        for key in field_keys {
            key.hash(hasher);
            // Hash the JSON value as a canonical string
            let val = &node.data.fields[key];
            let canonical = serde_json::to_string(val).unwrap_or_default();
            canonical.hash(hasher);
        }

        // Hash output flag and biome field
        node.data.is_output.hash(hasher);
        if let Some(ref bf) = node.data.biome_field {
            true.hash(hasher);
            bf.hash(hasher);
        } else {
            false.hash(hasher);
        }
    }

    // Sort edges by (source, target, handle) for determinism
    let mut sorted_edges: Vec<(&str, &str, Option<&str>)> = edges
        .iter()
        .map(|e| {
            (
                e.source.as_str(),
                e.target.as_str(),
                e.target_handle.as_deref(),
            )
        })
        .collect();
    sorted_edges.sort();

    sorted_edges.len().hash(hasher);
    for (src, tgt, handle) in sorted_edges {
        src.hash(hasher);
        tgt.hash(hasher);
        if let Some(h) = handle {
            true.hash(hasher);
            h.hash(hasher);
        } else {
            false.hash(hasher);
        }
    }
}

/// Hash content fields deterministically (sorted by key).
fn hash_content_fields(
    hasher: &mut DefaultHasher,
    fields: &std::collections::HashMap<String, f64>,
) {
    let mut keys: Vec<&String> = fields.keys().collect();
    keys.sort();
    keys.len().hash(hasher);
    for key in keys {
        key.hash(hasher);
        fields[key].to_bits().hash(hasher);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::graph::{GraphEdge, GraphNode, NodeData};
    use serde_json::json;
    use std::collections::HashMap;

    fn make_node(
        id: &str,
        density_type: &str,
        fields: HashMap<String, serde_json::Value>,
    ) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            data: NodeData {
                density_type: Some(density_type.to_string()),
                fields,
                is_output: false,
                biome_field: None,
            },
            node_type: None,
        }
    }

    fn make_edge(source: &str, target: &str, handle: Option<&str>) -> GraphEdge {
        GraphEdge {
            source: source.to_string(),
            target: target.to_string(),
            target_handle: handle.map(|s| s.to_string()),
        }
    }

    #[test]
    fn same_graph_same_hash() {
        let mut f = HashMap::new();
        f.insert("Value".to_string(), json!(42.0));
        let nodes = vec![make_node("c", "Constant", f.clone())];
        let edges = vec![];
        let cf = HashMap::new();

        let h1 = hash_grid_request(&nodes, &edges, 128, -64.0, 64.0, 64.0, Some("c"), &cf);

        let nodes2 = vec![make_node("c", "Constant", f)];
        let h2 = hash_grid_request(&nodes2, &edges, 128, -64.0, 64.0, 64.0, Some("c"), &cf);

        assert_eq!(h1, h2);
    }

    #[test]
    fn different_field_value_different_hash() {
        let mut f1 = HashMap::new();
        f1.insert("Value".to_string(), json!(42.0));
        let mut f2 = HashMap::new();
        f2.insert("Value".to_string(), json!(43.0));

        let nodes1 = vec![make_node("c", "Constant", f1)];
        let nodes2 = vec![make_node("c", "Constant", f2)];
        let cf = HashMap::new();

        let h1 = hash_grid_request(&nodes1, &[], 128, -64.0, 64.0, 64.0, Some("c"), &cf);
        let h2 = hash_grid_request(&nodes2, &[], 128, -64.0, 64.0, 64.0, Some("c"), &cf);

        assert_ne!(h1, h2);
    }

    #[test]
    fn different_resolution_different_hash() {
        let mut f = HashMap::new();
        f.insert("Value".to_string(), json!(42.0));
        let nodes = vec![make_node("c", "Constant", f)];
        let cf = HashMap::new();

        let h1 = hash_grid_request(&nodes, &[], 64, -64.0, 64.0, 64.0, Some("c"), &cf);
        let h2 = hash_grid_request(&nodes, &[], 128, -64.0, 64.0, 64.0, Some("c"), &cf);

        assert_ne!(h1, h2);
    }

    #[test]
    fn grid_vs_volume_different_hash() {
        let mut f = HashMap::new();
        f.insert("Value".to_string(), json!(42.0));
        let nodes = vec![make_node("c", "Constant", f)];
        let cf = HashMap::new();

        let h1 = hash_grid_request(&nodes, &[], 64, -64.0, 64.0, 64.0, Some("c"), &cf);
        let h2 = hash_volume_request(&nodes, &[], 64, -64.0, 64.0, 0.0, 128.0, 32, Some("c"), &cf);

        assert_ne!(h1, h2);
    }

    #[test]
    fn node_order_independent() {
        let mut fa = HashMap::new();
        fa.insert("Value".to_string(), json!(10.0));
        let mut fb = HashMap::new();
        fb.insert("Value".to_string(), json!(20.0));

        let nodes1 = vec![
            make_node("a", "Constant", fa.clone()),
            make_node("b", "Constant", fb.clone()),
            make_node("s", "Sum", HashMap::new()),
        ];
        // Reversed node order
        let nodes2 = vec![
            make_node("s", "Sum", HashMap::new()),
            make_node("b", "Constant", fb),
            make_node("a", "Constant", fa),
        ];
        let edges = vec![
            make_edge("a", "s", Some("Inputs[0]")),
            make_edge("b", "s", Some("Inputs[1]")),
        ];
        let cf = HashMap::new();

        let h1 = hash_grid_request(&nodes1, &edges, 64, -64.0, 64.0, 64.0, Some("s"), &cf);
        let h2 = hash_grid_request(&nodes2, &edges, 64, -64.0, 64.0, 64.0, Some("s"), &cf);

        assert_eq!(h1, h2);
    }

    #[test]
    fn edge_order_independent() {
        let nodes = vec![
            make_node("a", "Constant", HashMap::new()),
            make_node("b", "Constant", HashMap::new()),
            make_node("s", "Sum", HashMap::new()),
        ];
        let edges1 = vec![
            make_edge("a", "s", Some("Inputs[0]")),
            make_edge("b", "s", Some("Inputs[1]")),
        ];
        let edges2 = vec![
            make_edge("b", "s", Some("Inputs[1]")),
            make_edge("a", "s", Some("Inputs[0]")),
        ];
        let cf = HashMap::new();

        let h1 = hash_grid_request(&nodes, &edges1, 64, -64.0, 64.0, 64.0, Some("s"), &cf);
        let h2 = hash_grid_request(&nodes, &edges2, 64, -64.0, 64.0, 64.0, Some("s"), &cf);

        assert_eq!(h1, h2);
    }

    #[test]
    fn content_fields_affect_hash() {
        let mut f = HashMap::new();
        f.insert("Value".to_string(), json!(1.0));
        let nodes = vec![make_node("c", "Constant", f)];

        let mut cf1 = HashMap::new();
        cf1.insert("Terrain".to_string(), 1.0);

        let mut cf2 = HashMap::new();
        cf2.insert("Terrain".to_string(), 2.0);

        let h1 = hash_grid_request(&nodes, &[], 64, -64.0, 64.0, 64.0, Some("c"), &cf1);
        let h2 = hash_grid_request(&nodes, &[], 64, -64.0, 64.0, 64.0, Some("c"), &cf2);

        assert_ne!(h1, h2);
    }

    #[test]
    fn cache_put_get_grid() {
        let cache = EvalCache::new(4);
        let result = GridResult {
            values: vec![1.0, 2.0, 3.0, 4.0],
            resolution: 2,
            min_value: 1.0,
            max_value: 4.0,
        };

        assert!(cache.get_grid(42).is_none());
        cache.put_grid(42, result.clone());
        let cached = cache.get_grid(42).unwrap();
        assert_eq!(cached.values, result.values);
        assert_eq!(cached.resolution, result.resolution);
    }

    #[test]
    fn cache_put_get_volume() {
        let cache = EvalCache::new(4);
        let result = VolumeResult {
            densities: vec![1.0; 8],
            resolution: 2,
            y_slices: 2,
            min_value: 1.0,
            max_value: 1.0,
        };

        assert!(cache.get_volume(99).is_none());
        cache.put_volume(99, result.clone());
        let cached = cache.get_volume(99).unwrap();
        assert_eq!(cached.densities.len(), 8);
    }

    #[test]
    fn cache_lru_eviction() {
        let cache = EvalCache::new(2);
        let r1 = GridResult {
            values: vec![1.0],
            resolution: 1,
            min_value: 1.0,
            max_value: 1.0,
        };
        let r2 = GridResult {
            values: vec![2.0],
            resolution: 1,
            min_value: 2.0,
            max_value: 2.0,
        };
        let r3 = GridResult {
            values: vec![3.0],
            resolution: 1,
            min_value: 3.0,
            max_value: 3.0,
        };

        cache.put_grid(1, r1);
        cache.put_grid(2, r2);
        // This should evict key=1
        cache.put_grid(3, r3);

        assert!(cache.get_grid(1).is_none());
        assert!(cache.get_grid(2).is_some());
        assert!(cache.get_grid(3).is_some());
    }

    #[test]
    fn cache_clear() {
        let cache = EvalCache::new(4);
        let r = GridResult {
            values: vec![1.0],
            resolution: 1,
            min_value: 1.0,
            max_value: 1.0,
        };
        cache.put_grid(1, r);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get_grid(1).is_none());
    }

    #[test]
    fn cache_arc_returns_same_data() {
        // Verify that Arc-based cache returns the same data without full clone
        let cache = EvalCache::new(4);
        let result = VolumeResult {
            densities: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            resolution: 2,
            y_slices: 2,
            min_value: 1.0,
            max_value: 8.0,
        };
        cache.put_volume(100, result);

        let arc1 = cache.get_volume(100).unwrap();
        let arc2 = cache.get_volume(100).unwrap();

        // Both Arcs point to the same allocation
        assert!(Arc::ptr_eq(&arc1, &arc2));
        assert_eq!(arc1.densities.len(), 8);
        assert_eq!(arc1.min_value, 1.0);
        assert_eq!(arc1.max_value, 8.0);
    }

    #[test]
    fn cache_put_arc_directly() {
        let cache = EvalCache::new(4);
        let result = Arc::new(GridResult {
            values: vec![42.0],
            resolution: 1,
            min_value: 42.0,
            max_value: 42.0,
        });

        cache.put_grid_arc(55, result.clone());
        let cached = cache.get_grid(55).unwrap();
        assert!(Arc::ptr_eq(&result, &cached));
    }
}
