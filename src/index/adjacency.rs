//! CSR (Compressed Sparse Row) adjacency index for graph traversals.
//!
//! This module implements a cache-friendly adjacency list representation using
//! the Compressed Sparse Row format. This enables fast sequential access during
//! graph traversals with minimal cache misses.
//!
//! # CSR Format
//!
//! The CSR format stores the graph as two arrays:
//! - `offsets` where `offsets[i]` is the starting position in `edges` for node i's adjacency list
//! - `edges`: Flat array of (target_node, edge_id, edge_label) tuples
//!
//! This layout is cache-friendly because traversing from a node requires
//! sequential access to a contiguous region of memory.

use crate::core::hasher::IdentityHasher;
use crate::core::id::{EdgeId, NodeId};
use crate::core::interning::InternedString;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::hash::BuildHasherDefault;

/// A single entry in the adjacency list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdjacencyEntry {
    /// Target node ID.
    pub target: NodeId,
    /// Edge ID connecting source to target.
    pub edge_id: EdgeId,
    /// Edge label (interned for memory efficiency).
    pub label: InternedString,
}

impl AdjacencyEntry {
    /// Create a new adjacency entry.
    #[inline]
    pub const fn new(target: NodeId, edge_id: EdgeId, label: InternedString) -> Self {
        AdjacencyEntry {
            target,
            edge_id,
            label,
        }
    }
}

/// Compressed Sparse Row adjacency index.
///
/// This structure provides O(log n) access to a node's adjacency list with
/// excellent cache locality and memory efficiency for sparse node IDs.
///
/// The index uses a sparse representation where only nodes with outgoing edges
/// are stored, making it efficient even with large gaps in node IDs (e.g., after deletions).
#[derive(Debug, Clone)]
pub struct AdjacencyIndex {
    /// Sorted list of node IDs that have outgoing edges.
    /// Used for binary search to map node_id -> index in offsets array.
    node_ids: Vec<NodeId>,
    /// Offsets into the edges array for each node in node_ids.
    /// `offsets[i]` = start index in edges array for `node_ids[i]`
    /// `offsets[i + 1]` = end index (exclusive)
    offsets: Vec<usize>,
    /// Flat array of adjacency entries, sorted by source node.
    edges: Vec<AdjacencyEntry>,
    /// Maximum node ID (for bounds checking).
    max_node_id: u64,
}

impl AdjacencyIndex {
    /// Export CSR data for persistence.
    ///
    /// The CSR structure is decomposed into three raw arrays suitable for fast
    /// binary serialization to disk. This is heavily utilized by the persistence
    /// engine to avoid serializing rust-specific enum wrappers or iterating over
    /// complex graphs.
    ///
    /// Returns a tuple `(node_ids, offsets, edge_ids)` where:
    /// - `node_ids`: Sorted array of `NodeId`s that have outgoing edges.
    /// - `offsets`: The CSR offset array defining edge boundaries per node.
    /// - `edge_ids`: Flat array of all outgoing `EdgeId`s in traversal order.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// let edges = vec![
    ///     (NodeId::new(1).unwrap(), NodeId::new(2).unwrap(), EdgeId::new(100).unwrap(), label)
    /// ];
    ///
    /// let index = AdjacencyIndex::build(edges);
    /// let (nodes, offsets, edges_out) = index.export_csr();
    ///
    /// assert_eq!(nodes, vec![1]);
    /// assert_eq!(offsets, vec![0, 1]);
    /// assert_eq!(edges_out, vec![100]);
    /// ```
    pub fn export_csr(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let node_ids = self.node_ids.iter().map(|n| n.as_u64()).collect();
        let offsets = self.offsets.iter().map(|&x| x as u64).collect();
        let edge_ids = self.edges.iter().map(|e| e.edge_id.as_u64()).collect();
        (node_ids, offsets, edge_ids)
    }

    /// Import CSR data from persistence, reconstructing adjacency entries from edges.
    ///
    /// Re-hydrates a CSR structure from its raw binary components. This reconstructs
    /// the full `AdjacencyEntry` data by looking up the edge metadata (target, label)
    /// in the provided `edges_map`.
    ///
    /// This method is highly optimized and performs zero-copy vector transmutations
    /// where possible (such as on 64-bit systems converting `u64` to `usize`).
    ///
    /// # Arguments
    /// * `node_ids` - Sorted array of node IDs that have outgoing edges.
    /// * `offsets` - CSR offset array defining edge boundaries per node.
    /// * `edge_ids` - Flat array of edge IDs corresponding to the offsets.
    /// * `edges_map` - Map from `EdgeId` to `(target, label)` for full reconstruction.
    ///
    /// ## Panics
    ///
    /// Panics if the provided CSR invariants are violated (e.g., offsets array length mismatch,
    /// non-monotonic sequences, or invalid bounds) to prevent corrupted database state.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    /// use aletheiadb::core::hasher::IdentityHasher;
    /// use std::collections::HashMap;
    /// use std::hash::BuildHasherDefault;
    ///
    /// let nodes = vec![1];
    /// let offsets = vec![0, 1];
    /// let edge_ids = vec![100];
    ///
    /// let mut edge_map = HashMap::with_hasher(BuildHasherDefault::<IdentityHasher>::default());
    /// let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// edge_map.insert(EdgeId::new(100).unwrap(), (NodeId::new(2).unwrap(), label));
    ///
    /// let index = AdjacencyIndex::import_csr(nodes, offsets, edge_ids, &edge_map);
    /// assert_eq!(index.edge_count(), 1);
    /// ```
    pub fn import_csr(
        node_ids: Vec<u64>,
        offsets: Vec<u64>,
        edge_ids: Vec<u64>,
        edges_map: &std::collections::HashMap<
            EdgeId,
            (NodeId, InternedString),
            BuildHasherDefault<IdentityHasher>,
        >,
    ) -> Self {
        if offsets.is_empty() || edge_ids.is_empty() {
            return Self::new();
        }

        // Validate CSR invariants
        Self::validate_csr_invariants(&node_ids, &offsets, &edge_ids).unwrap();

        let max_node_id = node_ids.iter().max().copied().unwrap_or(0);

        // Zero-copy conversion: NodeId(u64) has same layout as u64
        let node_ids_typed: Vec<NodeId> = bytemuck::cast_vec(node_ids);

        // Convert offsets (zero-copy on 64-bit, allocating on 32-bit)
        let offsets_usize = Self::convert_offsets(offsets);

        let mut adjacency_entries = Vec::with_capacity(edge_ids.len());
        // Offsets are rebuilt as we go, because an edge id present in the CSR
        // but absent from `edges_map` is DROPPED rather than materialized as a
        // placeholder (Issue #3810).
        //
        // The persisted edge list and the persisted CSR are two snapshots of a
        // live database taken at slightly different instants, so the CSR can
        // legitimately name an edge the edge list no longer has (deleted, but
        // its tombstone not yet compacted away). A placeholder entry for it was
        // a phantom edge -- adjacency to node 0 under label 0 -- silently
        // materialized on restore, and it also broke the per-node
        // `(target, edge_id)` ordering the frozen runs rely on. Dropping it is
        // sound: the edge is either genuinely gone, or still in the edge list
        // and therefore re-added to the delta by the caller's reconstruction
        // pass.
        let mut rebuilt_offsets: Vec<usize> = Vec::with_capacity(offsets_usize.len());
        let mut dropped = 0usize;

        for window in offsets_usize.windows(2) {
            rebuilt_offsets.push(adjacency_entries.len());
            for &edge_id_u64 in &edge_ids[window[0]..window[1]] {
                let edge_id = EdgeId::new_unchecked(edge_id_u64);
                match edges_map.get(&edge_id) {
                    Some((target, label)) => {
                        adjacency_entries.push(AdjacencyEntry::new(*target, edge_id, *label))
                    }
                    None => dropped += 1,
                }
            }
        }
        rebuilt_offsets.push(adjacency_entries.len());

        if dropped > 0 {
            eprintln!(
                "Warning: {} persisted adjacency entries referenced edges that are not in the \
                 persisted edge list and were dropped on import",
                dropped
            );
        }

        Self {
            node_ids: node_ids_typed,
            offsets: rebuilt_offsets,
            edges: adjacency_entries,
            max_node_id,
        }
    }
}

impl AdjacencyIndex {
    /// Create a new empty adjacency index.
    ///
    /// Initializes an empty CSR structure that allocates no heap memory until
    /// edges are explicitly added via building or importing.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::index::AdjacencyIndex;
    ///
    /// let index = AdjacencyIndex::new();
    /// assert_eq!(index.node_count(), 0);
    /// assert_eq!(index.edge_count(), 0);
    /// ```
    pub fn new() -> Self {
        AdjacencyIndex {
            node_ids: Vec::new(),
            offsets: vec![0],
            edges: Vec::new(),
            max_node_id: 0,
        }
    }

    /// Build an adjacency index from a list of edges.
    ///
    /// Accepts a flat list of edges and dynamically constructs the sparse CSR representation.
    /// The input is automatically sorted in parallel by `(source, target, edge_id)` to ensure
    /// deterministic adjacency lists and correct offset calculation.
    ///
    /// Edges should be provided as `(source, target, edge_id, label)` tuples.
    ///
    /// This uses a sparse representation: only nodes with outgoing edges are stored. This makes
    /// it extremely memory efficient even with large gaps in node IDs (O(num_nodes) instead of O(max_node_id)).
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// let edges = vec![
    ///     (NodeId::new(1).unwrap(), NodeId::new(2).unwrap(), EdgeId::new(100).unwrap(), label),
    ///     (NodeId::new(1).unwrap(), NodeId::new(3).unwrap(), EdgeId::new(101).unwrap(), label),
    /// ];
    ///
    /// let index = AdjacencyIndex::build(edges);
    /// assert_eq!(index.degree(NodeId::new(1).unwrap()), 2);
    /// ```
    pub fn build(mut edges: Vec<(NodeId, NodeId, EdgeId, InternedString)>) -> Self {
        if edges.is_empty() {
            return Self::new();
        }

        let edge_count = edges.len();

        // Sort by source node, then target node for deterministic ordering.
        // We use parallel sort for performance on large graphs (serial on wasm,
        // which has no rayon); the total key yields identical ordering either way.
        // We include edge_id for canonical deterministic ordering.
        #[cfg(not(target_arch = "wasm32"))]
        edges.par_sort_unstable_by_key(|(src, target, edge_id, _)| (*src, *target, *edge_id));
        #[cfg(target_arch = "wasm32")]
        edges.sort_unstable_by_key(|(src, target, edge_id, _)| (*src, *target, *edge_id));

        // Pre-allocate assuming some average degree > 1 to avoid resizing
        let estimated_nodes = (edge_count / 4).max(16);
        let mut node_ids = Vec::with_capacity(estimated_nodes);
        let mut offsets = Vec::with_capacity(estimated_nodes + 1);
        let mut flat_edges = Vec::with_capacity(edge_count);
        let mut max_node_id = 0;

        offsets.push(0);

        if !edges.is_empty() {
            let mut current_source = edges[0].0;
            node_ids.push(current_source);

            for (source, target, edge_id, label) in edges {
                let src_val = source.as_u64();
                let tgt_val = target.as_u64();
                if src_val > max_node_id {
                    max_node_id = src_val;
                }
                if tgt_val > max_node_id {
                    max_node_id = tgt_val;
                }

                if source != current_source {
                    offsets.push(flat_edges.len());
                    current_source = source;
                    node_ids.push(current_source);
                }
                flat_edges.push(AdjacencyEntry::new(target, edge_id, label));
            }
            offsets.push(flat_edges.len());
        }

        // Optimize memory usage by releasing unused capacity
        node_ids.shrink_to_fit();
        offsets.shrink_to_fit();

        AdjacencyIndex {
            node_ids,
            offsets,
            edges: flat_edges,
            max_node_id,
        }
    }

    /// Get the adjacency list for a node.
    ///
    /// Returns a sequential, cache-friendly slice of all outgoing edges originating
    /// from the specified node. Because the CSR edges are stored contiguously, iterating
    /// through this slice ensures near-zero cache misses during graph traversal.
    ///
    /// Returns an empty slice if the node has no outgoing edges.
    ///
    /// This uses a fast binary search over the sparse `node_ids` array to locate the node,
    /// providing O(log n) lookup time where `n` is the number of nodes with outgoing edges.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// let edges = vec![
    ///     (NodeId::new(1).unwrap(), NodeId::new(2).unwrap(), EdgeId::new(100).unwrap(), label)
    /// ];
    ///
    /// let index = AdjacencyIndex::build(edges);
    /// let adj = index.get_adjacency(NodeId::new(1).unwrap());
    ///
    /// assert_eq!(adj.len(), 1);
    /// assert_eq!(adj[0].target, NodeId::new(2).unwrap());
    /// ```
    #[inline]
    pub fn get_adjacency(&self, node: NodeId) -> &[AdjacencyEntry] {
        // Binary search to find the node's index in node_ids
        match self.node_ids.binary_search(&node) {
            Ok(idx) => {
                let start = self.offsets[idx];
                let end = self.offsets[idx + 1];
                &self.edges[start..end]
            }
            Err(_) => {
                // Node not found (no outgoing edges)
                &[]
            }
        }
    }

    /// Resolve the flat-array bounds of `node`'s adjacency run with a single
    /// O(log N) binary search (Issue #3813).
    ///
    /// Returns `(start, end)` such that
    /// `&self.all_entries()[start..end]` is exactly what
    /// [`get_adjacency`](Self::get_adjacency) would return; `(0, 0)` for
    /// nodes with no edges.
    #[inline]
    pub(crate) fn adjacency_range(&self, node: NodeId) -> (usize, usize) {
        match self.node_ids.binary_search(&node) {
            Ok(idx) => (self.offsets[idx], self.offsets[idx + 1]),
            Err(_) => (0, 0),
        }
    }

    /// The entire flat adjacency array (O(1), no binary search).
    ///
    /// Only index this with bounds from
    /// [`adjacency_range`](Self::adjacency_range). Callers obtain the array
    /// through a `Guard<Arc<AdjacencyIndex>>` that pins the CSR, so resolved
    /// bounds stay valid for the guard's lifetime.
    #[inline]
    pub(crate) fn all_entries(&self) -> &[AdjacencyEntry] {
        &self.edges
    }

    /// Get outgoing edges for a node with a specific label.
    ///
    /// Performs an `O(log N) + O(E)` traversal where `N` is the number of nodes with
    /// outgoing edges, and `E` is the degree of the given node. It yields an iterator
    /// over only the adjacency entries that possess the specified `InternedString` label.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// let follows = GLOBAL_INTERNER.intern("FOLLOWS").unwrap();
    /// let edges = vec![
    ///     (NodeId::new(1).unwrap(), NodeId::new(2).unwrap(), EdgeId::new(100).unwrap(), knows),
    ///     (NodeId::new(1).unwrap(), NodeId::new(3).unwrap(), EdgeId::new(101).unwrap(), follows),
    /// ];
    ///
    /// let index = AdjacencyIndex::build(edges);
    /// let knows_edges: Vec<_> = index.get_adjacency_with_label(NodeId::new(1).unwrap(), knows).collect();
    ///
    /// assert_eq!(knows_edges.len(), 1);
    /// assert_eq!(knows_edges[0].target, NodeId::new(2).unwrap());
    /// ```
    pub fn get_adjacency_with_label(
        &self,
        node: NodeId,
        label: InternedString,
    ) -> impl Iterator<Item = &AdjacencyEntry> {
        self.get_adjacency(node)
            .iter()
            .filter(move |entry| entry.label == label)
    }

    /// Get the number of outgoing edges for a node (out-degree).
    ///
    /// Determines how many edges originate from this node. This relies on the
    /// same O(log N) binary search as `get_adjacency` to calculate the slice length.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// let edges = vec![
    ///     (NodeId::new(1).unwrap(), NodeId::new(2).unwrap(), EdgeId::new(100).unwrap(), label)
    /// ];
    ///
    /// let index = AdjacencyIndex::build(edges);
    /// assert_eq!(index.degree(NodeId::new(1).unwrap()), 1);
    /// assert_eq!(index.degree(NodeId::new(99).unwrap()), 0);
    /// ```
    #[inline]
    pub fn degree(&self, node: NodeId) -> usize {
        self.get_adjacency(node).len()
    }

    /// Check if a node has any outgoing edges.
    ///
    /// Fast boolean check to see if traversing out of this node is possible.
    /// This is equivalent to `degree(node) > 0`.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// let edges = vec![
    ///     (NodeId::new(1).unwrap(), NodeId::new(2).unwrap(), EdgeId::new(100).unwrap(), label)
    /// ];
    ///
    /// let index = AdjacencyIndex::build(edges);
    /// assert!(index.has_edges(NodeId::new(1).unwrap()));
    /// assert!(!index.has_edges(NodeId::new(99).unwrap()));
    /// ```
    #[inline]
    pub fn has_edges(&self, node: NodeId) -> bool {
        self.degree(node) > 0
    }

    /// Get total number of edges in the index.
    ///
    /// Returns the exact size of the underlying flat CSR edges array.
    /// Because the array is flat, this is an O(1) operation.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::index::AdjacencyIndex;
    ///
    /// let index = AdjacencyIndex::new();
    /// assert_eq!(index.edge_count(), 0);
    /// ```
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get the maximum node ID in this index.
    ///
    /// Returns the highest numerical `NodeId` encountered across both edge
    /// sources and targets during construction. This is an O(1) operation
    /// heavily utilized by the execution engine for query bounds checking.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// let edges = vec![
    ///     (NodeId::new(1).unwrap(), NodeId::new(500).unwrap(), EdgeId::new(100).unwrap(), label)
    /// ];
    ///
    /// let index = AdjacencyIndex::build(edges);
    /// assert_eq!(index.max_node_id(), 500);
    /// ```
    #[inline]
    pub fn max_node_id(&self) -> u64 {
        self.max_node_id
    }

    /// Iterate over all nodes that have outgoing edges.
    ///
    /// Yields a stream of `NodeId`s representing every unique source node in the graph.
    ///
    /// This is extremely efficient for sparse graphs as it only yields nodes
    /// that actually have outgoing edges, completely bypassing "gaps" or deleted nodes
    /// that would otherwise be traversed if checking `0..max_node_id`.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// let edges = vec![
    ///     (NodeId::new(10).unwrap(), NodeId::new(20).unwrap(), EdgeId::new(100).unwrap(), label),
    ///     (NodeId::new(99).unwrap(), NodeId::new(20).unwrap(), EdgeId::new(101).unwrap(), label),
    /// ];
    ///
    /// let index = AdjacencyIndex::build(edges);
    /// let nodes: Vec<_> = index.iter_nodes().collect();
    ///
    /// // Only nodes with outgoing edges are yielded!
    /// assert_eq!(nodes, vec![NodeId::new(10).unwrap(), NodeId::new(99).unwrap()]);
    /// ```
    #[inline]
    pub fn iter_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.node_ids.iter().copied()
    }

    /// Get the number of nodes with outgoing edges.
    ///
    /// Returns the exact size of the underlying CSR `node_ids` array.
    /// Because the array only stores source nodes, this reflects the number
    /// of unique nodes with an out-degree > 0. This is an O(1) operation.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::{NodeId, EdgeId};
    /// use aletheiadb::index::AdjacencyIndex;
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    /// let edges = vec![
    ///     (NodeId::new(1).unwrap(), NodeId::new(2).unwrap(), EdgeId::new(100).unwrap(), label),
    ///     (NodeId::new(1).unwrap(), NodeId::new(3).unwrap(), EdgeId::new(101).unwrap(), label),
    /// ];
    ///
    /// let index = AdjacencyIndex::build(edges);
    /// // Even though there are 2 edges and 3 total unique nodes involved,
    /// // only 1 node is a source node!
    /// assert_eq!(index.node_count(), 1);
    /// ```
    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_ids.len()
    }

    /// Validate CSR invariants.
    fn validate_csr_invariants(
        node_ids: &[u64],
        offsets: &[u64],
        edge_ids: &[u64],
    ) -> Result<(), String> {
        if offsets.len() != node_ids.len() + 1 {
            return Err(format!(
                "CSR offsets length mismatch: expected {}, got {}",
                node_ids.len() + 1,
                offsets.len()
            ));
        }

        #[allow(clippy::collapsible_if)]
        if let Some(&first_offset) = offsets.first() {
            if first_offset != 0 {
                return Err(format!(
                    "CSR first offset mismatch: expected 0, got {}",
                    first_offset
                ));
            }
        }

        for window in offsets.windows(2) {
            if window[0] > window[1] {
                return Err(format!(
                    "CSR offsets are not monotonically increasing: {} > {}",
                    window[0], window[1]
                ));
            }
        }

        for window in node_ids.windows(2) {
            if window[0] >= window[1] {
                return Err(format!(
                    "CSR node_ids are not strictly monotonically increasing: {} >= {}",
                    window[0], window[1]
                ));
            }
        }

        #[allow(clippy::collapsible_if)]
        if let Some(&last_offset) = offsets.last() {
            if last_offset != edge_ids.len() as u64 {
                return Err(format!(
                    "CSR last offset mismatch: expected {}, got {}",
                    edge_ids.len(),
                    last_offset
                ));
            }
        }

        Ok(())
    }

    /// Convert offsets to usize vector.
    ///
    /// On 64-bit systems, this is a zero-copy operation because usize == u64.
    /// On 32-bit systems, this allocates a new vector because usize == u32 != u64.
    fn convert_offsets(offsets: Vec<u64>) -> Vec<usize> {
        #[cfg(target_pointer_width = "64")]
        {
            bytemuck::cast_vec(offsets)
        }

        #[cfg(not(target_pointer_width = "64"))]
        {
            offsets.iter().map(|&x| x as usize).collect()
        }
    }
}

impl Default for AdjacencyIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::interning::GLOBAL_INTERNER;

    #[test]
    fn test_empty_index() {
        let index = AdjacencyIndex::new();
        assert_eq!(index.edge_count(), 0);
        assert_eq!(index.degree(NodeId::new(0).unwrap()), 0);
        assert_eq!(index.get_adjacency(NodeId::new(0).unwrap()).len(), 0);
    }

    #[test]
    fn test_build_simple_graph() {
        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();

        let edges = vec![
            (
                NodeId::new(0).unwrap(),
                NodeId::new(1).unwrap(),
                EdgeId::new(0).unwrap(),
                knows,
            ),
            (
                NodeId::new(0).unwrap(),
                NodeId::new(2).unwrap(),
                EdgeId::new(1).unwrap(),
                knows,
            ),
            (
                NodeId::new(1).unwrap(),
                NodeId::new(2).unwrap(),
                EdgeId::new(2).unwrap(),
                knows,
            ),
        ];

        let index = AdjacencyIndex::build(edges);

        // Node 0 has 2 outgoing edges
        assert_eq!(index.degree(NodeId::new(0).unwrap()), 2);
        let adj0 = index.get_adjacency(NodeId::new(0).unwrap());
        assert_eq!(adj0.len(), 2);
        assert_eq!(adj0[0].target, NodeId::new(1).unwrap());
        assert_eq!(adj0[1].target, NodeId::new(2).unwrap());

        // Node 1 has 1 outgoing edge
        assert_eq!(index.degree(NodeId::new(1).unwrap()), 1);
        let adj1 = index.get_adjacency(NodeId::new(1).unwrap());
        assert_eq!(adj1.len(), 1);
        assert_eq!(adj1[0].target, NodeId::new(2).unwrap());

        // Node 2 has no outgoing edges
        assert_eq!(index.degree(NodeId::new(2).unwrap()), 0);

        // Total edges
        assert_eq!(index.edge_count(), 3);
    }

    #[test]
    fn test_multiple_edge_labels() {
        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let follows = GLOBAL_INTERNER.intern("FOLLOWS").unwrap();

        let edges = vec![
            (
                NodeId::new(0).unwrap(),
                NodeId::new(1).unwrap(),
                EdgeId::new(0).unwrap(),
                knows,
            ),
            (
                NodeId::new(0).unwrap(),
                NodeId::new(2).unwrap(),
                EdgeId::new(1).unwrap(),
                follows,
            ),
            (
                NodeId::new(0).unwrap(),
                NodeId::new(3).unwrap(),
                EdgeId::new(2).unwrap(),
                knows,
            ),
        ];

        let index = AdjacencyIndex::build(edges);

        // Get all edges from node 0
        assert_eq!(index.degree(NodeId::new(0).unwrap()), 3);

        // Get only KNOWS edges from node 0
        let knows_edges: Vec<_> = index
            .get_adjacency_with_label(NodeId::new(0).unwrap(), knows)
            .collect();
        assert_eq!(knows_edges.len(), 2);

        // Get only FOLLOWS edges from node 0
        let follows_edges: Vec<_> = index
            .get_adjacency_with_label(NodeId::new(0).unwrap(), follows)
            .collect();
        assert_eq!(follows_edges.len(), 1);
        assert_eq!(follows_edges[0].target, NodeId::new(2).unwrap());
    }

    #[test]
    fn test_node_without_edges() {
        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();

        let edges = vec![(
            NodeId::new(0).unwrap(),
            NodeId::new(1).unwrap(),
            EdgeId::new(0).unwrap(),
            knows,
        )];

        let index = AdjacencyIndex::build(edges);

        // Node 5 doesn't exist
        assert_eq!(index.degree(NodeId::new(5).unwrap()), 0);
        assert!(!index.has_edges(NodeId::new(5).unwrap()));
        assert_eq!(index.get_adjacency(NodeId::new(5).unwrap()).len(), 0);
    }

    #[test]
    fn test_adjacency_entry() {
        let label = GLOBAL_INTERNER.intern("TEST").unwrap();
        let entry = AdjacencyEntry::new(NodeId::new(1).unwrap(), EdgeId::new(100).unwrap(), label);

        assert_eq!(entry.target, NodeId::new(1).unwrap());
        assert_eq!(entry.edge_id, EdgeId::new(100).unwrap());
        assert_eq!(entry.label, label);
    }

    #[test]
    fn test_sorted_adjacency() {
        // Edges deliberately out of order
        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let edges = vec![
            (
                NodeId::new(0).unwrap(),
                NodeId::new(3).unwrap(),
                EdgeId::new(2).unwrap(),
                knows,
            ),
            (
                NodeId::new(0).unwrap(),
                NodeId::new(1).unwrap(),
                EdgeId::new(0).unwrap(),
                knows,
            ),
            (
                NodeId::new(0).unwrap(),
                NodeId::new(2).unwrap(),
                EdgeId::new(1).unwrap(),
                knows,
            ),
        ];

        let index = AdjacencyIndex::build(edges);
        let adj = index.get_adjacency(NodeId::new(0).unwrap());

        // Should be sorted by target
        assert_eq!(adj[0].target, NodeId::new(1).unwrap());
        assert_eq!(adj[1].target, NodeId::new(2).unwrap());
        assert_eq!(adj[2].target, NodeId::new(3).unwrap());
    }

    #[test]
    fn test_sparse_node_ids() {
        // Simulate scenario after deletions: only nodes 10, 1000, and 1_000_000 exist
        // This tests that we handle sparse IDs efficiently
        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let edges = vec![
            (
                NodeId::new(10).unwrap(),
                NodeId::new(20).unwrap(),
                EdgeId::new(0).unwrap(),
                knows,
            ),
            (
                NodeId::new(1000).unwrap(),
                NodeId::new(2000).unwrap(),
                EdgeId::new(1).unwrap(),
                knows,
            ),
            (
                NodeId::new(1_000_000).unwrap(),
                NodeId::new(2_000_000).unwrap(),
                EdgeId::new(2).unwrap(),
                knows,
            ),
        ];

        let index = AdjacencyIndex::build(edges);

        // Verify correctness for sparse nodes
        assert_eq!(index.degree(NodeId::new(10).unwrap()), 1);
        assert_eq!(index.degree(NodeId::new(1000).unwrap()), 1);
        assert_eq!(index.degree(NodeId::new(1_000_000).unwrap()), 1);

        // Verify intermediate non-existent nodes return empty adjacency
        assert_eq!(index.degree(NodeId::new(0).unwrap()), 0);
        assert_eq!(index.degree(NodeId::new(100).unwrap()), 0);
        assert_eq!(index.degree(NodeId::new(50000).unwrap()), 0);

        // Verify adjacency list content
        let adj10 = index.get_adjacency(NodeId::new(10).unwrap());
        assert_eq!(adj10.len(), 1);
        assert_eq!(adj10[0].target, NodeId::new(20).unwrap());

        let adj1000 = index.get_adjacency(NodeId::new(1000).unwrap());
        assert_eq!(adj1000.len(), 1);
        assert_eq!(adj1000[0].target, NodeId::new(2000).unwrap());

        let adj1m = index.get_adjacency(NodeId::new(1_000_000).unwrap());
        assert_eq!(adj1m.len(), 1);
        assert_eq!(adj1m[0].target, NodeId::new(2_000_000).unwrap());

        // Total edges should still be 3
        assert_eq!(index.edge_count(), 3);
    }

    #[test]
    fn test_sparse_ids_memory_efficiency() {
        // Test that sparse IDs don't cause excessive memory allocation
        // With old implementation: offsets would be Vec with 1_000_001 elements
        // With new implementation: offsets should only have entries for actual nodes
        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let edges = vec![
            (
                NodeId::new(0).unwrap(),
                NodeId::new(1).unwrap(),
                EdgeId::new(0).unwrap(),
                knows,
            ),
            (
                NodeId::new(1_000_000).unwrap(),
                NodeId::new(1_000_001).unwrap(),
                EdgeId::new(1).unwrap(),
                knows,
            ),
        ];

        let index = AdjacencyIndex::build(edges);

        // After optimization, offsets should be proportional to number of nodes, not max_node_id
        // With 2 source nodes, we should have at most a few entries, not 1_000_001
        // Allow some overhead for implementation details
        assert!(
            index.offsets.len() < 100,
            "Offsets array should be compact for sparse IDs, got {} entries",
            index.offsets.len()
        );

        // Verify correctness
        assert_eq!(index.degree(NodeId::new(0).unwrap()), 1);
        assert_eq!(index.degree(NodeId::new(1_000_000).unwrap()), 1);
        assert_eq!(index.edge_count(), 2);
    }

    #[test]
    fn test_sparse_ids_with_multiple_edges_per_node() {
        // Test sparse IDs where some nodes have multiple outgoing edges
        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let follows = GLOBAL_INTERNER.intern("FOLLOWS").unwrap();
        let edges = vec![
            (
                NodeId::new(100).unwrap(),
                NodeId::new(101).unwrap(),
                EdgeId::new(0).unwrap(),
                knows,
            ),
            (
                NodeId::new(100).unwrap(),
                NodeId::new(102).unwrap(),
                EdgeId::new(1).unwrap(),
                follows,
            ),
            (
                NodeId::new(100).unwrap(),
                NodeId::new(103).unwrap(),
                EdgeId::new(2).unwrap(),
                knows,
            ),
            (
                NodeId::new(500_000).unwrap(),
                NodeId::new(500_001).unwrap(),
                EdgeId::new(3).unwrap(),
                knows,
            ),
            (
                NodeId::new(500_000).unwrap(),
                NodeId::new(500_002).unwrap(),
                EdgeId::new(4).unwrap(),
                follows,
            ),
        ];

        let index = AdjacencyIndex::build(edges);

        // Verify node 100 has 3 edges
        assert_eq!(index.degree(NodeId::new(100).unwrap()), 3);
        let adj100 = index.get_adjacency(NodeId::new(100).unwrap());
        assert_eq!(adj100.len(), 3);
        // Should be sorted by target
        assert_eq!(adj100[0].target, NodeId::new(101).unwrap());
        assert_eq!(adj100[1].target, NodeId::new(102).unwrap());
        assert_eq!(adj100[2].target, NodeId::new(103).unwrap());

        // Verify node 500_000 has 2 edges
        assert_eq!(index.degree(NodeId::new(500_000).unwrap()), 2);
        let adj500k = index.get_adjacency(NodeId::new(500_000).unwrap());
        assert_eq!(adj500k.len(), 2);
        assert_eq!(adj500k[0].target, NodeId::new(500_001).unwrap());
        assert_eq!(adj500k[1].target, NodeId::new(500_002).unwrap());

        // Verify intermediate nodes have no edges
        assert_eq!(index.degree(NodeId::new(200_000).unwrap()), 0);
        assert_eq!(index.degree(NodeId::new(300_000).unwrap()), 0);

        // Total edges
        assert_eq!(index.edge_count(), 5);
    }

    #[test]
    fn test_build_with_many_edges_preallocation() {
        // Test that building with many edges works correctly.
        // This test verifies the scenario mentioned in issue #193 where
        // pre-allocating the flat_edges Vec avoids ~14 reallocations for 10,000 edges.
        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();

        // Create 10,000 edges across 1,000 nodes
        let edge_count = 10_000;
        let node_count = 1_000;

        let mut edges = Vec::with_capacity(edge_count);
        for i in 0..edge_count {
            let source = NodeId::new((i % node_count) as u64).unwrap();
            let target = NodeId::new(((i + 1) % node_count) as u64).unwrap();
            let edge_id = EdgeId::new(i as u64).unwrap();
            edges.push((source, target, edge_id, knows));
        }

        // Build the index (should pre-allocate to avoid reallocations)
        let index = AdjacencyIndex::build(edges);

        // Verify correctness
        assert_eq!(index.edge_count(), edge_count);

        // Verify that each node has the correct number of outgoing edges.
        // In this test setup, each node is a source for `edge_count / node_count` edges.
        let expected_degree = edge_count / node_count;
        for i in 0..node_count {
            let node = NodeId::new(i as u64).unwrap();
            let adj = index.get_adjacency(node);
            assert_eq!(
                adj.len(),
                expected_degree,
                "Node {} has an unexpected degree",
                i
            );
            // All adjacency entries should be valid
            for entry in adj {
                assert!(entry.edge_id.as_u64() < edge_count as u64);
                assert!(entry.target.as_u64() < node_count as u64);
            }
        }
    }

    #[test]
    fn test_max_node_id_from_target() {
        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let edges = vec![
            (
                NodeId::new(1).unwrap(),
                NodeId::new(1000).unwrap(),
                EdgeId::new(0).unwrap(),
                knows,
            ),
            (
                NodeId::new(2).unwrap(),
                NodeId::new(500).unwrap(),
                EdgeId::new(1).unwrap(),
                knows,
            ),
        ];
        let index = AdjacencyIndex::build(edges);
        assert_eq!(
            index.max_node_id(),
            1000,
            "max_node_id should consider target nodes"
        );
    }

    #[test]
    fn test_transmute_vec_correctness() {
        let original = vec![1u64, 2, 3];
        let ptr = original.as_ptr();
        let cap = original.capacity();

        // Use NodeId which is transparent wrapper around u64
        let transmuted: Vec<NodeId> = bytemuck::cast_vec(original);

        assert_eq!(transmuted.len(), 3);
        assert_eq!(transmuted.capacity(), cap);
        assert_eq!(transmuted[0], NodeId::new(1).unwrap());
        assert_eq!(transmuted[1], NodeId::new(2).unwrap());
        assert_eq!(transmuted[2], NodeId::new(3).unwrap());

        // Verify no copy happened (best effort check, pointers should match)
        assert_eq!(transmuted.as_ptr() as *const u64, ptr);
    }

    #[test]
    fn test_import_csr_integration() {
        // This test ensures import_csr works in the standard test module scope
        let node_ids = vec![1, 2];
        let offsets = vec![0, 1, 2];
        let edge_ids = vec![10, 20];
        let mut edges_map =
            std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::<
                crate::core::hasher::IdentityHasher,
            >::default());

        let label = crate::core::interning::GLOBAL_INTERNER
            .intern("TEST")
            .unwrap();
        edges_map.insert(EdgeId::new(10).unwrap(), (NodeId::new(2).unwrap(), label));
        edges_map.insert(EdgeId::new(20).unwrap(), (NodeId::new(1).unwrap(), label));

        let index = AdjacencyIndex::import_csr(node_ids, offsets, edge_ids, &edges_map);
        assert_eq!(index.node_count(), 2);
        assert_eq!(index.edge_count(), 2);
    }
}

#[cfg(test)]
mod sentry_tests {
    use super::*;
    use std::collections::HashMap;
    use std::hash::BuildHasherDefault;

    #[test]
    fn test_validate_csr_invariants_logic() {
        // 1. Valid case
        let node_ids = vec![10, 20];
        let offsets = vec![0, 1, 2];
        let edge_ids = vec![100, 101];
        assert!(AdjacencyIndex::validate_csr_invariants(&node_ids, &offsets, &edge_ids).is_ok());

        // 2. Invalid offsets length
        let invalid_offsets_len = vec![0, 1]; // too short
        let err_len =
            AdjacencyIndex::validate_csr_invariants(&node_ids, &invalid_offsets_len, &edge_ids)
                .unwrap_err();
        assert!(err_len.contains("CSR offsets length mismatch"));

        // 3. Invalid last offset
        let invalid_offsets_val = vec![0, 1, 5]; // last is 5, but edges len is 2
        let err_val =
            AdjacencyIndex::validate_csr_invariants(&node_ids, &invalid_offsets_val, &edge_ids)
                .unwrap_err();
        assert!(err_val.contains("CSR last offset mismatch"));

        // 4. Invalid first offset
        let invalid_first_offset = vec![1, 1, 2];
        let err_first =
            AdjacencyIndex::validate_csr_invariants(&node_ids, &invalid_first_offset, &edge_ids)
                .unwrap_err();
        assert!(err_first.contains("CSR first offset mismatch"));

        // 5. Non-monotonic offsets
        let non_monotonic_offsets = vec![0, 2, 1]; // 2 > 1
        // We need 3 edge ids to match the last offset 1, or wait, last offset is 1, so edge len = 1
        let err_monotonic =
            AdjacencyIndex::validate_csr_invariants(&node_ids, &non_monotonic_offsets, &[100])
                .unwrap_err();
        assert!(err_monotonic.contains("CSR offsets are not monotonically increasing"));

        // 6. Unsorted node ids
        let unsorted_node_ids = vec![20, 10]; // unsorted
        let err_unsorted =
            AdjacencyIndex::validate_csr_invariants(&unsorted_node_ids, &offsets, &edge_ids)
                .unwrap_err();
        assert!(err_unsorted.contains("CSR node_ids are not strictly monotonically increasing"));

        // 7. Duplicate node ids
        let duplicate_node_ids = vec![10, 10]; // duplicate
        let err_duplicate =
            AdjacencyIndex::validate_csr_invariants(&duplicate_node_ids, &offsets, &edge_ids)
                .unwrap_err();
        assert!(err_duplicate.contains("CSR node_ids are not strictly monotonically increasing"));
    }

    #[test]
    #[should_panic(expected = "CSR offsets length mismatch")]
    fn test_import_csr_panics_on_invalid() {
        // Integration check: ensure import_csr actually calls validate and panics
        let node_ids = vec![10];
        let offsets = vec![0]; // invalid len (should be 2)
        let edge_ids = vec![100]; // Non-empty to bypass early return
        let edges_map = HashMap::with_hasher(BuildHasherDefault::<IdentityHasher>::default());
        AdjacencyIndex::import_csr(node_ids, offsets, edge_ids, &edges_map);
    }

    #[test]
    fn test_import_csr_success() {
        // Multi-node case to fully exercise loop and max_node_id logic
        let node_ids: Vec<u64> = vec![10, 20];
        // Offsets: node 10 has 1 edge (0..1), node 20 has 1 edge (1..2)
        let offsets: Vec<u64> = vec![0, 1, 2];
        let edge_ids: Vec<u64> = vec![100, 101];

        let mut edges_map = HashMap::with_hasher(BuildHasherDefault::<IdentityHasher>::default());
        let target = NodeId::new(99).unwrap();
        let label = crate::core::interning::InternedString::from_raw(1);

        edges_map.insert(EdgeId::new(100).unwrap(), (target, label));
        edges_map.insert(EdgeId::new(101).unwrap(), (target, label));

        // Should not panic
        let index = AdjacencyIndex::import_csr(node_ids, offsets, edge_ids, &edges_map);

        assert_eq!(index.edge_count(), 2);
        assert_eq!(index.node_count(), 2);
        assert_eq!(index.max_node_id(), 20);
    }

    /// A persisted CSR may name an edge the persisted edge list no longer has:
    /// the two are snapshots of a live database taken at slightly different
    /// instants, so an edge deleted in between is in one and not the other.
    ///
    /// Such an entry must be DROPPED, not materialized (Issue #3810). Before
    /// the fix it became `AdjacencyEntry::new(NodeId(0), edge_id, from_raw(0))`
    /// -- a phantom adjacency to node 0 under label 0, conjured on restore --
    /// and because the placeholder kept its slot, every following node's
    /// offsets still pointed at it.
    #[test]
    fn import_csr_drops_entries_whose_edge_is_absent_and_rebuilds_offsets() {
        // Node 10 owns edges 100, 101; node 20 owns edge 102.
        let node_ids: Vec<u64> = vec![10, 20];
        let offsets: Vec<u64> = vec![0, 2, 3];
        let edge_ids: Vec<u64> = vec![100, 101, 102];

        // Edge 101 is missing from the edge list: deleted after the CSR
        // snapshot was taken, its tombstone not yet compacted away.
        let mut edges_map = HashMap::with_hasher(BuildHasherDefault::<IdentityHasher>::default());
        let target = NodeId::new(99).unwrap();
        let label = crate::core::interning::InternedString::from_raw(1);
        edges_map.insert(EdgeId::new(100).unwrap(), (target, label));
        edges_map.insert(EdgeId::new(102).unwrap(), (target, label));

        let index = AdjacencyIndex::import_csr(node_ids, offsets, edge_ids, &edges_map);

        // The absent edge is gone, not replaced by a placeholder.
        assert_eq!(index.edge_count(), 2, "the absent edge must be dropped");

        // No phantom: nothing points at node 0 under label 0, and edge 101
        // appears nowhere.
        let node_10 = index.get_adjacency(NodeId::new(10).unwrap());
        let node_20 = index.get_adjacency(NodeId::new(20).unwrap());
        for entry in node_10.iter().chain(node_20.iter()) {
            assert_ne!(
                entry.target,
                NodeId::new_unchecked(0),
                "phantom adjacency to node 0 materialized on import"
            );
            assert_ne!(
                entry.edge_id,
                EdgeId::new(101).unwrap(),
                "the absent edge leaked into the index"
            );
        }

        // Offsets were rebuilt around the hole rather than left pointing past
        // it: node 10 keeps only edge 100, and node 20 still resolves to 102
        // (with stale offsets it would have read the dropped slot instead).
        assert_eq!(node_10.len(), 1, "node 10 keeps only its surviving edge");
        assert_eq!(node_10[0].edge_id, EdgeId::new(100).unwrap());
        assert_eq!(node_20.len(), 1, "node 20's edge must not be shifted away");
        assert_eq!(node_20[0].edge_id, EdgeId::new(102).unwrap());
    }

    /// The all-present path must not drop anything -- guards the `dropped`
    /// bookkeeping in the opposite direction from the test above.
    #[test]
    fn import_csr_keeps_every_entry_when_no_edge_is_absent() {
        let node_ids: Vec<u64> = vec![10, 20];
        let offsets: Vec<u64> = vec![0, 2, 3];
        let edge_ids: Vec<u64> = vec![100, 101, 102];

        let mut edges_map = HashMap::with_hasher(BuildHasherDefault::<IdentityHasher>::default());
        let target = NodeId::new(99).unwrap();
        let label = crate::core::interning::InternedString::from_raw(1);
        for id in [100u64, 101, 102] {
            edges_map.insert(EdgeId::new(id).unwrap(), (target, label));
        }

        let index = AdjacencyIndex::import_csr(node_ids, offsets, edge_ids, &edges_map);

        assert_eq!(index.edge_count(), 3);
        assert_eq!(index.get_adjacency(NodeId::new(10).unwrap()).len(), 2);
        assert_eq!(index.get_adjacency(NodeId::new(20).unwrap()).len(), 1);
    }
}
