//! Current-state indexes using concurrent data structures.
//!
//! This module provides indexes for the current state of the graph using
//! DashMap for lock-free concurrent access. These are the "hot path" indexes
//! that must be extremely fast.

use crate::core::graph::{Edge, Node, NodeHeader};
use crate::core::hasher::IdHashBuilder;
use crate::core::id::{EdgeId, NodeId};
use crate::core::interning::InternedString;
use crate::core::namespace::{Namespace, NamespaceId, intern_namespace, resolve_namespace_id};
use crate::core::property::PropertyMap;
use crate::index::adjacency::AdjacencyEntry;
use crate::index::adjacency_maintenance::{self, AdjacencyMaintenanceConfig};
use crate::index::incremental_adjacency::{
    AdjacencyLayerStats, CompactionScheduler, IncrementalAdjacencyIndex,
};
use crate::index::property_index::{ValueKey, value_key};
use dashmap::{DashMap, DashSet};
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

/// One direction's exported CSR arrays: `(node_ids, offsets, edge_ids)`.
///
/// Matches [`IncrementalAdjacencyIndex::export_frozen_csr`](crate::index::IncrementalAdjacencyIndex::export_frozen_csr).
pub type ExportedCsr = (Vec<u64>, Vec<u64>, Vec<u64>);

/// Both directions' exported CSR arrays: `(outgoing, incoming)`.
pub type ExportedCsrPair = (ExportedCsr, ExportedCsr);

/// Layer occupancy of a database's two adjacency indexes (Issue #3810).
///
/// Returned by [`CurrentIndexes::adjacency_stats`],
/// [`CurrentStorage::adjacency_stats`](crate::storage::current::CurrentStorage::adjacency_stats)
/// and [`AletheiaDB::adjacency_stats`](crate::AletheiaDB::adjacency_stats).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdjacencyIndexStats {
    /// Outgoing (source -> targets) adjacency index.
    pub outgoing: AdjacencyLayerStats,
    /// Incoming (target -> sources) adjacency index.
    pub incoming: AdjacencyLayerStats,
}

impl AdjacencyIndexStats {
    /// Whether **both** directions are compacted, i.e. every adjacency read
    /// currently takes the frozen-CSR fast path.
    #[inline]
    pub fn is_fully_compacted(&self) -> bool {
        self.outgoing.is_compacted() && self.incoming.is_compacted()
    }
}

// Note: AdjacencyGuard was removed in favor of MergedAdjacencyGuard from incremental_adjacency.
// The new guard supports merging frozen CSR + delta buffer on-the-fly.

/// Concurrent indexes for current-state graph queries.
///
/// These indexes provide O(1) lookups for nodes and edges, plus efficient
/// graph traversal through incremental CSR adjacency indexes.
///
/// # Incremental Adjacency (Issue #259)
///
/// Uses LSM-tree inspired incremental adjacency indexes:
/// - **O(1) edge inserts**: No rebuild cliff, edges go to delta buffer
/// - **O(1) edge deletes**: Tombstone marking, no rebuild needed
/// - **Lock-free reads**: Merge frozen + delta on-the-fly
/// - **Background compaction**: Automatic delta → frozen merging
///
/// # Concurrency Model
///
/// - **Node/Edge maps**: DashMap for lock-free concurrent access
/// - **Adjacency indexes**: IncrementalAdjacencyIndex with lock-free reads
/// - **No rebuild lock needed**: Incremental inserts/deletes are thread-safe
///
/// # Performance
///
/// - **Insert edge**: ~100ns (was ~10ms+ due to rebuild)
/// - **Read adjacency**: ~20-30ns (merge overhead vs pure CSR)
/// - **Compaction**: Automatic in background, doesn't block operations
pub struct CurrentIndexes {
    /// Node ID → Node (O(1) lookup, cold path with full PropertyMap)
    ///
    /// Keyed by `NodeId`, an already-unique internal u64, so identity-hashed
    /// via [`IdHashBuilder`] to avoid SipHash overhead on the point-lookup
    /// hot path (`get_node`, `get_node_with`, ...).
    nodes: DashMap<NodeId, Node, IdHashBuilder>,
    /// Node ID → NodeHeader (hot path for label filtering, 16 bytes per entry)
    ///
    /// Kept in sync with `nodes`: every `insert_node` writes a header and every
    /// `remove_node` removes it. Scanning this map for label matches touches far
    /// less memory than scanning the full `nodes` map. Identity-hashed for the
    /// same reason as `nodes`.
    node_headers: DashMap<NodeId, NodeHeader, IdHashBuilder>,
    /// Edge ID → Edge (O(1) lookup)
    ///
    /// Identity-hashed for the same reason as `nodes` — `EdgeId` is already a
    /// unique internal u64.
    edges: DashMap<EdgeId, Edge, IdHashBuilder>,
    /// Outgoing edges: source node → adjacency list (incremental with O(1) inserts)
    outgoing: Arc<IncrementalAdjacencyIndex>,
    /// Incoming edges: target node → adjacency list (incremental with O(1) inserts)
    incoming: Arc<IncrementalAdjacencyIndex>,
    /// Optional background compaction scheduler for outgoing index
    outgoing_compaction: Option<(CompactionScheduler, JoinHandle<()>)>,
    /// Optional background compaction scheduler for incoming index
    incoming_compaction: Option<(CompactionScheduler, JoinHandle<()>)>,
    /// Monotonic high-water-mark: `max(node id ever inserted) + 1`, i.e. an
    /// exclusive upper bound on the node id space.
    ///
    /// Maintained by [`insert_node`](Self::insert_node) via a relaxed
    /// `fetch_max` and never decremented on delete, so it bounds the full range
    /// of ids that could hold a live node. The query executor's full-scan
    /// iterator uses it to sweep `[0, bound)` lazily without materializing the
    /// id set. Reading the index directly (rather than an id generator) is
    /// essential: in the transactional write path node ids are allocated by a
    /// database-level generator and inserted here via `insert_node_direct`, so
    /// the storage's own id generator is never advanced and cannot bound a
    /// scan.
    max_node_id: AtomicU64,
    /// Secondary namespace → member-node membership index (Issue #3349, PR2).
    ///
    /// A derived, **non-persisted** acceleration structure mapping each
    /// namespace to the set of node ids that currently live in it, so a
    /// namespace-scoped read can enumerate members without a full property
    /// scan. The durable ground truth is the reserved ride-along property on
    /// each node ([`crate::core::namespace::NAMESPACE_KEY`]); this index is
    /// rebuilt for free at load / WAL-replay because every node insertion
    /// funnels through [`insert_node`](Self::insert_node).
    ///
    /// # Lock discipline
    ///
    /// This is a **leaf** in the lock-acquisition order: a lock-free `DashMap`
    /// of lock-free `DashSet`s, mutated only inside
    /// `insert_node`/`remove_node`, and it never calls back into `historical`,
    /// `wal`, or `current_timestamp`.
    ///
    /// Keyed by the interned [`NamespaceId`] (Issue #3349, PR2 performance) so
    /// each per-edge scoped-read probe is an integer (identity) hash rather than
    /// a heap-string hash. The inner member set is likewise identity-hashed (its
    /// keys are `NodeId`s — already unique u64s), so the per-edge `contains`
    /// probe is a plain integer hash, not SipHash.
    ns_nodes: DashMap<NamespaceId, DashSet<NodeId, IdHashBuilder>, IdHashBuilder>,
    /// Secondary namespace → member-edge membership index (Issue #3349, PR2).
    /// The edge counterpart of [`ns_nodes`](Self::ns_nodes); see its docs.
    ns_edges: DashMap<NamespaceId, DashSet<EdgeId, IdHashBuilder>, IdHashBuilder>,
    /// Opt-in registry of `(label_id, prop_key_id)` pairs that have a secondary
    /// equality index enabled. Empty by default — the property-index maintenance
    /// hooks in `insert_node` / `remove_node` short-circuit on `is_empty()`, so a
    /// database with no property index pays only that check on the write path.
    prop_index_enabled: DashMap<(InternedString, InternedString), ()>,
    /// Secondary equality index: `(label_id, prop_key_id, ValueKey)` → the set of
    /// node ids currently holding that value.
    ///
    /// A **derived, non-persisted** acceleration structure — the durable ground
    /// truth is each node's `PropertyMap`. Rebuilt for free at load / WAL-replay
    /// because every node insertion funnels through
    /// [`insert_node`](Self::insert_node) (the opt-in *decision* is in-memory
    /// only; re-enable after restart to backfill). See
    /// [`crate::index::property_index`] for the full design, the `Float`
    /// exclusion rationale, and the namespace fail-closed contract.
    ///
    /// # Lock discipline
    ///
    /// A **leaf** in the lock-acquisition order (like `ns_nodes`): lock-free
    /// `DashMap` of lock-free `DashSet`s, mutated only inside
    /// `insert_node`/`remove_node`, never calling back into `historical`, `wal`,
    /// or `current_timestamp`. The inner set is `NodeId`-keyed so it is
    /// identity-hashed, not SipHashed.
    prop_index: DashMap<(InternedString, InternedString, ValueKey), DashSet<NodeId, IdHashBuilder>>,
}

impl CurrentIndexes {
    /// Create new empty indexes with incremental adjacency.
    ///
    /// The adjacency indexes are registered with the shared background
    /// maintenance worker (Issue #3810), which compacts them once writes go
    /// quiet so reads reach the frozen-CSR fast path. Registration is a `Weak`
    /// reference: dropping these indexes deregisters them, with no shutdown
    /// call and no `Drop` impl required.
    ///
    /// Use [`with_maintenance_config`](Self::with_maintenance_config) with
    /// [`AdjacencyMaintenanceConfig::disabled`] for the pre-#3810 behavior
    /// (compaction only when `compact_adjacency()` is called explicitly).
    pub fn new() -> Self {
        Self::with_maintenance_config(AdjacencyMaintenanceConfig::default())
    }

    /// Create new empty indexes with an explicit background-maintenance policy.
    pub fn with_maintenance_config(maintenance: AdjacencyMaintenanceConfig) -> Self {
        let outgoing = Arc::new(IncrementalAdjacencyIndex::new());
        let incoming = Arc::new(IncrementalAdjacencyIndex::new());
        adjacency_maintenance::register(&outgoing, maintenance.clone());
        adjacency_maintenance::register(&incoming, maintenance);

        CurrentIndexes {
            nodes: DashMap::with_hasher(BuildHasherDefault::default()),
            node_headers: DashMap::with_hasher(BuildHasherDefault::default()),
            edges: DashMap::with_hasher(BuildHasherDefault::default()),
            outgoing,
            incoming,
            outgoing_compaction: None,
            incoming_compaction: None,
            max_node_id: AtomicU64::new(0),
            ns_nodes: DashMap::with_hasher(BuildHasherDefault::default()),
            ns_edges: DashMap::with_hasher(BuildHasherDefault::default()),
            prop_index_enabled: DashMap::new(),
            prop_index: DashMap::new(),
        }
    }

    /// Create new indexes with a **dedicated** per-index compaction thread.
    ///
    /// Two threads (outgoing + incoming) are spawned for these indexes alone and
    /// must be stopped with [`shutdown_background_compaction`](Self::shutdown_background_compaction)
    /// before dropping.
    ///
    /// Prefer plain [`new`](Self::new): since Issue #3810 it enrolls the indexes
    /// in the shared, process-wide maintenance worker instead -- one thread for
    /// the whole process, no shutdown obligation. These indexes deliberately do
    /// **not** also enroll there, so exactly one compactor owns them.
    pub fn new_with_background_compaction() -> Self {
        let outgoing = Arc::new(IncrementalAdjacencyIndex::new());
        let incoming = Arc::new(IncrementalAdjacencyIndex::new());

        // Start background compaction for both outgoing and incoming indexes
        let outgoing_scheduler = CompactionScheduler::new(Arc::clone(&outgoing));
        let outgoing_handle = outgoing_scheduler.start();

        let incoming_scheduler = CompactionScheduler::new(Arc::clone(&incoming));
        let incoming_handle = incoming_scheduler.start();

        CurrentIndexes {
            nodes: DashMap::with_hasher(BuildHasherDefault::default()),
            node_headers: DashMap::with_hasher(BuildHasherDefault::default()),
            edges: DashMap::with_hasher(BuildHasherDefault::default()),
            outgoing,
            incoming,
            outgoing_compaction: Some((outgoing_scheduler, outgoing_handle)),
            incoming_compaction: Some((incoming_scheduler, incoming_handle)),
            max_node_id: AtomicU64::new(0),
            ns_nodes: DashMap::with_hasher(BuildHasherDefault::default()),
            ns_edges: DashMap::with_hasher(BuildHasherDefault::default()),
            prop_index_enabled: DashMap::new(),
            prop_index: DashMap::new(),
        }
    }

    /// Shutdown background compaction if enabled.
    ///
    /// Waits for background thread to finish. Safe to call even if
    /// background compaction is not enabled (no-op).
    ///
    /// # Returns
    ///
    /// - `Ok(())` if shutdown was successful or no scheduler was running
    /// - `Err(String)` if the background thread panicked during shutdown
    pub fn shutdown_background_compaction(&mut self) -> Result<(), String> {
        // Shutdown outgoing compaction
        if let Some((scheduler, handle)) = self.outgoing_compaction.take() {
            scheduler.shutdown();
            handle
                .join()
                .map_err(|e| format!("Outgoing compaction thread panicked: {:?}", e))?;
        }
        // Shutdown incoming compaction
        if let Some((scheduler, handle)) = self.incoming_compaction.take() {
            scheduler.shutdown();
            handle
                .join()
                .map_err(|e| format!("Incoming compaction thread panicked: {:?}", e))?;
        }
        Ok(())
    }

    /// Snapshot of both adjacency indexes' layer occupancy (Issue #3810).
    ///
    /// `delta_edges == 0 && tombstones == 0` is exactly the condition under
    /// which reads take the frozen-CSR fast path, so this is the direct
    /// observable for whether background maintenance has caught up.
    pub fn adjacency_stats(&self) -> AdjacencyIndexStats {
        AdjacencyIndexStats {
            outgoing: self.outgoing.layer_stats(),
            incoming: self.incoming.layer_stats(),
        }
    }

    /// Get frozen edge count for outgoing adjacency (for testing).
    #[doc(hidden)]
    pub fn frozen_edge_count(&self) -> usize {
        self.outgoing.frozen_edge_count()
    }

    /// Get the number of edges in the delta buffer.
    ///
    /// This represents edges that have been inserted but not yet compacted into frozen.
    pub fn delta_edge_count(&self) -> usize {
        self.outgoing.delta_edge_count()
    }

    /// Insert a node into the indexes.
    ///
    /// Also inserts a [`NodeHeader`] into the hot-path header map so label
    /// filtering via [`filter_nodes_by_label`](Self::filter_nodes_by_label)
    /// can scan 16-byte headers instead of full nodes.
    pub fn insert_node(&self, node: Node) {
        let header = NodeHeader::from(&node);
        // Advance the exclusive high-water-mark before moving the node. Relaxed
        // ordering suffices: the scan bound is an approximate upper bound, and
        // node visibility itself is governed by the DashMap insert below.
        self.max_node_id
            .fetch_max(node.id.as_u64().wrapping_add(1), Ordering::Relaxed);
        // Maintain the namespace membership index (Issue #3349). Read the
        // namespace off the node before moving it into the map. The insert is
        // idempotent, so re-inserting on an in-place update is a no-op (the
        // namespace is immutable for the life of the entity).
        let node_id = node.id;
        // Derive the interned namespace id from the node's durable ride-along
        // string (Issue #3349, PR2). This is the single derivation site for the
        // node membership index, so every hydration path (write, WAL replay,
        // index-persistence load, cold rehydration) that funnels through
        // `insert_node` produces the identical id for the same namespace.
        let ns_id = intern_namespace(&node.namespace());
        // Maintain the secondary property (equality) index. Only pay the cost
        // when at least one (label, property) is enabled; otherwise this is a
        // single `is_empty` check on the create/update hot path. On an in-place
        // update the prior node is still present in `self.nodes`, so we capture
        // its LABEL and properties before overwriting: a create sees no prior
        // node (pure add), a same-label update diffs old vs new to de-index the
        // stale value, and a LABEL-CHANGING replace (`replace_node`, Issue
        // #3549 — full overwrite reaches `insert_node` with a new label)
        // de-indexes under the old label and re-indexes under the new one. This
        // is the single choke point every hydration path funnels through
        // (create, update, replace, WAL replay, index-persistence load), so the
        // index rebuilds for free at load.
        //
        // ORDERING: the node must be published in `self.nodes` BEFORE it is
        // published in the property index, never the other way round. The index
        // is a pointer *to* the node, so indexing first opens a window in which
        // `find_nodes_by_property` returns an id that `get_node` cannot resolve
        // -- a lookup racing an insert gets `NodeNotFound` for a node the index
        // just told it about. The window is a couple of instructions wide, and
        // it went unnoticed because group commit could only apply ~93 writes a
        // second; once commits actually batch it reproduces in seconds.
        let reindex = if self.prop_index_enabled.is_empty() {
            None
        } else {
            let old = self.nodes.get(&node_id).map(|e| {
                let prior = e.value();
                (prior.label, prior.properties.clone())
            });
            // Cloning the new label and property map is cheap -- `PropertyMap`
            // is `Arc`-backed, so this is a refcount bump, not a deep copy --
            // and it is what lets the node move into the map first.
            Some((old, node.label, node.properties.clone()))
        };

        self.nodes.insert(node.id, node);

        if let Some((old, new_label, new_props)) = reindex {
            let (old_label, old_props) = match &old {
                Some((label, props)) => (Some(*label), Some(props)),
                None => (None, None),
            };
            self.reindex_node_property(node_id, old_label, old_props, new_label, &new_props);
        }

        self.node_headers.insert(header.id, header);
        self.ns_nodes.entry(ns_id).or_default().insert(node_id);
    }

    /// Reconcile a node's property-index membership across a create, update, or
    /// label-changing replace. Called from [`insert_node`](Self::insert_node); a
    /// pure create passes `old_label = None` / `old_props = None`.
    ///
    /// Old and new labels are handled **independently** so a label change
    /// (`old_label != new_label`, e.g. `replace_node`) both de-indexes the
    /// node's old value under the *old* label and indexes its new value under
    /// the *new* label — otherwise a lone new-label filter would leave a stale
    /// entry under the old label (false positive) and, because the old and new
    /// values compare equal, skip adding under the new label (false negative).
    /// The `old_vk == new_vk` no-op fast path applies only to a same-label
    /// update, where a bucket move is genuinely unnecessary.
    fn reindex_node_property(
        &self,
        id: NodeId,
        old_label: Option<InternedString>,
        old_props: Option<&PropertyMap>,
        new_label: InternedString,
        new_props: &PropertyMap,
    ) {
        for entry in self.prop_index_enabled.iter() {
            let (lbl, key) = *entry.key();
            let is_old = Some(lbl) == old_label;
            let is_new = lbl == new_label;
            if !is_old && !is_new {
                continue;
            }
            if is_old && is_new {
                // Same-label create/update: move the value bucket only if it
                // actually changed.
                let old_vk = old_props
                    .and_then(|p| p.get_by_interned_key(&key))
                    .and_then(value_key);
                let new_vk = new_props.get_by_interned_key(&key).and_then(value_key);
                if old_vk == new_vk {
                    continue;
                }
                if let Some(ov) = old_vk {
                    self.prop_index_remove((lbl, key, ov), id);
                }
                if let Some(nv) = new_vk {
                    self.prop_index
                        .entry((lbl, key, nv))
                        .or_default()
                        .insert(id);
                }
            } else if is_old {
                // Label changed away from `lbl`: drop the node's old value.
                if let Some(ov) = old_props
                    .and_then(|p| p.get_by_interned_key(&key))
                    .and_then(value_key)
                {
                    self.prop_index_remove((lbl, key, ov), id);
                }
            } else {
                // Label changed to `lbl` (or a fresh create under `lbl`): add the
                // node's new value.
                if let Some(nv) = new_props.get_by_interned_key(&key).and_then(value_key) {
                    self.prop_index
                        .entry((lbl, key, nv))
                        .or_default()
                        .insert(id);
                }
            }
        }
    }

    /// Remove `id` from a property-index value bucket, reclaiming the bucket if
    /// it drains empty. Mirrors the `ns_nodes` reclaim: the read guard is dropped
    /// before `remove_if`, which re-checks emptiness under the shard write lock,
    /// so a concurrent insert racing the reclaim never loses its set.
    fn prop_index_remove(&self, key: (InternedString, InternedString, ValueKey), id: NodeId) {
        let now_empty = match self.prop_index.get(&key) {
            Some(set) => {
                set.remove(&id);
                set.is_empty()
            }
            None => false,
        };
        if now_empty {
            self.prop_index.remove_if(&key, |_, set| set.is_empty());
        }
    }

    /// Exclusive upper bound on the node id space: `max(id ever inserted) + 1`.
    ///
    /// Returns `0` for an empty index. This is a relaxed, monotonically
    /// non-decreasing read intended for bounding a lazy full scan over
    /// `[0, bound)`; it is not a live node count and does not shrink when nodes
    /// are deleted.
    #[inline]
    pub fn max_node_id_exclusive(&self) -> u64 {
        self.max_node_id.load(Ordering::Relaxed)
    }

    /// Insert an edge into the indexes.
    ///
    /// Updates both the edge map and adjacency indexes incrementally.
    /// - **O(1) amortized**: Edge goes to delta buffer, no rebuild
    /// - **Thread-safe**: Lock-free concurrent inserts
    /// - **Atomic visibility**: Edge immediately visible in adjacency queries
    pub fn insert_edge(&self, edge: Edge) {
        // Insert into edge map
        let edge_id = edge.id;
        let source = edge.source;
        let target = edge.target;
        let label = edge.label;

        // Use entry API to atomically check+insert, avoiding TOCTOU race condition
        // For updates, the adjacency doesn't change (source/target stay the same)
        // so we skip adjacency insertion to avoid duplicates
        use dashmap::mapref::entry::Entry;

        match self.edges.entry(edge_id) {
            Entry::Occupied(mut e) => {
                // Update: just replace edge data, don't modify adjacency
                e.insert(edge);
            }
            Entry::Vacant(e) => {
                // New edge: insert into both edge map and adjacency
                // Single derivation site for the edge membership index id
                // (Issue #3349, PR2); see `insert_node`.
                let ns_id = intern_namespace(&edge.namespace());
                e.insert(edge);
                self.outgoing
                    .insert(source, AdjacencyEntry::new(target, edge_id, label));
                self.incoming
                    .insert(target, AdjacencyEntry::new(source, edge_id, label));
                // Maintain the namespace membership index (Issue #3349).
                self.ns_edges.entry(ns_id).or_default().insert(edge_id);
            }
        }
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.nodes.get(&id).map(|entry| entry.value().clone())
    }

    /// Get an edge by ID.
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.edges.get(&id).map(|entry| entry.value().clone())
    }

    // ========================================================================
    // Zero-copy access methods (Issue #190)
    //
    // These methods provide efficient access to node/edge data without cloning
    // the entire structure. Use these for hot paths where you only need to
    // read specific fields.
    // ========================================================================

    /// Access a node without cloning, executing a closure on the node data.
    ///
    /// This method provides zero-copy read access to node data for hot paths
    /// where only specific fields are needed. The closure receives a reference
    /// to the node and can return any computed value.
    ///
    /// # Performance
    ///
    /// - **No allocation**: Does not clone the Node (~56 bytes saved)
    /// - **No Arc increment**: Does not increment PropertyMap reference count
    /// - **Lock duration**: Holds DashMap read lock only during closure execution
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Check if node has a specific label (zero-copy)
    /// let is_person = indexes.with_node(node_id, |n| n.label == person_label)
    ///     .unwrap_or(false);
    ///
    /// // Extract multiple fields in one call
    /// let (id, label) = indexes.with_node(node_id, |n| (n.id, n.label))?;
    /// ```
    #[inline]
    pub fn with_node<F, R>(&self, id: NodeId, f: F) -> Option<R>
    where
        F: FnOnce(&Node) -> R,
    {
        self.nodes.get(&id).map(|entry| f(entry.value()))
    }

    /// Mutably access a node without replacing it, executing a closure on mutable node data.
    ///
    /// After the closure runs, the [`NodeHeader`] is refreshed so that
    /// `get_node_header` and `filter_nodes_by_label` reflect any label change
    /// made inside the closure.
    pub fn with_node_mut<F>(&self, id: NodeId, f: F)
    where
        F: FnOnce(&mut Node),
    {
        if let Some(mut entry) = self.nodes.get_mut(&id) {
            f(entry.value_mut());
            // Refresh the header while still holding the write lock so that
            // label changes are immediately visible to filter_nodes_by_label.
            let header = NodeHeader::from(entry.value());
            self.node_headers.insert(id, header);
        }
    }

    /// Mutably access an edge without replacing it.
    pub fn with_edge_mut<F>(&self, id: EdgeId, f: F)
    where
        F: FnOnce(&mut Edge),
    {
        if let Some(mut entry) = self.edges.get_mut(&id) {
            f(entry.value_mut());
        }
    }

    /// Access an edge without cloning, executing a closure on the edge data.
    ///
    /// This method provides zero-copy read access to edge data for hot paths
    /// where only specific fields are needed. The closure receives a reference
    /// to the edge and can return any computed value.
    ///
    /// # Performance
    ///
    /// - **No allocation**: Does not clone the Edge (~72 bytes saved)
    /// - **No Arc increment**: Does not increment PropertyMap reference count
    /// - **Lock duration**: Holds DashMap read lock only during closure execution
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Get edge endpoints (zero-copy)
    /// let (source, target) = indexes.with_edge(edge_id, |e| (e.source, e.target))?;
    ///
    /// // Check if edge connects specific nodes
    /// let connects = indexes.with_edge(edge_id, |e| e.connects(src, tgt))
    ///     .unwrap_or(false);
    /// ```
    #[inline]
    pub fn with_edge<F, R>(&self, id: EdgeId, f: F) -> Option<R>
    where
        F: FnOnce(&Edge) -> R,
    {
        self.edges.get(&id).map(|entry| f(entry.value()))
    }

    /// Get the label of a node without cloning the entire node.
    ///
    /// This is a convenience method for the common case of checking a node's
    /// label. It's equivalent to `with_node(id, |n| n.label)` but more concise.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns the label (8 bytes)
    /// - **No allocation**: Does not clone Node or PropertyMap
    ///
    /// # Example
    ///
    /// ```ignore
    /// if indexes.get_node_label(node_id) == Some(person_label) {
    ///     // Node is a Person
    /// }
    /// ```
    #[inline]
    pub fn get_node_label(&self, id: NodeId) -> Option<InternedString> {
        self.nodes.get(&id).map(|entry| entry.value().label)
    }

    /// Check whether a node's label equals the given interned string, without cloning.
    ///
    /// This is the preferred hot-path API for label-filtered vector search (Issue #339).
    /// It avoids the `Option`-wrapping overhead of `get_node_label` when only a boolean
    /// answer is needed, saving a branch and keeping the closure passed to the HNSW
    /// filter as minimal as possible.
    ///
    /// Returns `false` if the node does not exist.
    #[inline]
    pub fn check_node_label(&self, id: NodeId, expected: InternedString) -> bool {
        self.nodes
            .get(&id)
            .is_some_and(|entry| entry.value().label == expected)
    }

    /// Get the endpoints (source, target) of an edge without cloning.
    ///
    /// This is a convenience method for the common case of getting an edge's
    /// source and target nodes. It's equivalent to
    /// `with_edge(id, |e| (e.source, e.target))` but more concise.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns two NodeIds (16 bytes)
    /// - **No allocation**: Does not clone Edge or PropertyMap
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some((source, target)) = indexes.get_edge_endpoints(edge_id) {
    ///     // Process source and target
    /// }
    /// ```
    #[inline]
    pub fn get_edge_endpoints(&self, id: EdgeId) -> Option<(NodeId, NodeId)> {
        self.edges
            .get(&id)
            .map(|entry| (entry.value().source, entry.value().target))
    }

    /// Get the label of an edge without cloning the entire edge.
    ///
    /// This is a convenience method for the common case of checking an edge's
    /// label. It's equivalent to `with_edge(id, |e| e.label)` but more concise.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns the label (8 bytes)
    /// - **No allocation**: Does not clone Edge or PropertyMap
    ///
    /// # Example
    ///
    /// ```ignore
    /// if indexes.get_edge_label(edge_id) == Some(knows_label) {
    ///     // Edge is a KNOWS relationship
    /// }
    /// ```
    #[inline]
    pub fn get_edge_label(&self, id: EdgeId) -> Option<InternedString> {
        self.edges.get(&id).map(|entry| entry.value().label)
    }

    /// Get the source node of an edge without cloning the entire edge.
    ///
    /// This is a convenience method for getting just the source node ID.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns the source NodeId (8 bytes)
    /// - **No allocation**: Does not clone Edge or PropertyMap
    #[inline]
    pub fn get_edge_source(&self, id: EdgeId) -> Option<NodeId> {
        self.edges.get(&id).map(|entry| entry.value().source)
    }

    /// Get the target node of an edge without cloning the entire edge.
    ///
    /// This is a convenience method for getting just the target node ID.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns the target NodeId (8 bytes)
    /// - **No allocation**: Does not clone Edge or PropertyMap
    #[inline]
    pub fn get_edge_target(&self, id: EdgeId) -> Option<NodeId> {
        self.edges.get(&id).map(|entry| entry.value().target)
    }

    /// Get a property value from a node without cloning the entire node.
    ///
    /// This is useful when you only need to read a single property from a node.
    /// The property value is cloned (Arc types are cheap to clone), but the
    /// rest of the node structure is not.
    ///
    /// # Performance
    ///
    /// - **Partial clone**: Only clones the PropertyValue (cheap for Arc types)
    /// - **No Node clone**: Does not clone the entire Node structure
    /// - **Interning cost**: The key is interned on each call; for hot paths
    ///   with repeated lookups, use [`get_node_property_by_key`](Self::get_node_property_by_key)
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(name) = indexes.get_node_property(node_id, "name") {
    ///     println!("Node name: {:?}", name);
    /// }
    /// ```
    #[inline]
    pub fn get_node_property(
        &self,
        id: NodeId,
        key: &str,
    ) -> Option<crate::core::property::PropertyValue> {
        self.nodes
            .get(&id)
            .and_then(|entry| entry.value().properties.get(key).cloned())
    }

    /// Get a property value from an edge without cloning the entire edge.
    ///
    /// This is useful when you only need to read a single property from an edge.
    /// The property value is cloned (Arc types are cheap to clone), but the
    /// rest of the edge structure is not.
    ///
    /// # Performance
    ///
    /// - **Partial clone**: Only clones the PropertyValue (cheap for Arc types)
    /// - **No Edge clone**: Does not clone the entire Edge structure
    /// - **Interning cost**: The key is interned on each call; for hot paths
    ///   with repeated lookups, use [`get_edge_property_by_key`](Self::get_edge_property_by_key)
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(weight) = indexes.get_edge_property(edge_id, "weight") {
    ///     println!("Edge weight: {:?}", weight);
    /// }
    /// ```
    #[inline]
    pub fn get_edge_property(
        &self,
        id: EdgeId,
        key: &str,
    ) -> Option<crate::core::property::PropertyValue> {
        self.edges
            .get(&id)
            .and_then(|entry| entry.value().properties.get(key).cloned())
    }

    /// Get a property value from a node using a pre-interned key.
    ///
    /// This is the high-performance version of [`get_node_property`](Self::get_node_property)
    /// for hot paths where the same property key is accessed repeatedly.
    ///
    /// # Performance
    ///
    /// - **No interning**: Avoids HashMap lookup in the global interner
    /// - **Partial clone**: Only clones the PropertyValue (cheap for Arc types)
    /// - **Ideal for loops**: Pre-intern key once, use this method in the loop
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// // Pre-intern the key once outside the loop
    /// let name_key = GLOBAL_INTERNER.intern("name").unwrap();
    ///
    /// // Use the interned key in the hot loop
    /// for node_id in node_ids {
    ///     if let Some(name) = indexes.get_node_property_by_key(node_id, &name_key) {
    ///         // Process name
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn get_node_property_by_key(
        &self,
        id: NodeId,
        key: &crate::core::property::PropertyKey,
    ) -> Option<crate::core::property::PropertyValue> {
        self.nodes
            .get(&id)
            .and_then(|entry| entry.value().properties.get_by_interned_key(key).cloned())
    }

    /// Get a property value from an edge using a pre-interned key.
    ///
    /// This is the high-performance version of [`get_edge_property`](Self::get_edge_property)
    /// for hot paths where the same property key is accessed repeatedly.
    ///
    /// # Performance
    ///
    /// - **No interning**: Avoids HashMap lookup in the global interner
    /// - **Partial clone**: Only clones the PropertyValue (cheap for Arc types)
    /// - **Ideal for loops**: Pre-intern key once, use this method in the loop
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aletheiadb::core::interning::GLOBAL_INTERNER;
    ///
    /// // Pre-intern the key once outside the loop
    /// let weight_key = GLOBAL_INTERNER.intern("weight").unwrap();
    ///
    /// // Use the interned key in the hot loop
    /// for edge_id in edge_ids {
    ///     if let Some(weight) = indexes.get_edge_property_by_key(edge_id, &weight_key) {
    ///         // Process weight
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn get_edge_property_by_key(
        &self,
        id: EdgeId,
        key: &crate::core::property::PropertyKey,
    ) -> Option<crate::core::property::PropertyValue> {
        self.edges
            .get(&id)
            .and_then(|entry| entry.value().properties.get_by_interned_key(key).cloned())
    }

    /// Remove a node from the indexes.
    ///
    /// Removes from the primary node map first, then removes the header only
    /// if the node was actually present. This avoids a brief window where the
    /// header is absent but the node still exists, which could cause a
    /// label-filter miss under concurrent reads.
    pub fn remove_node(&self, id: NodeId) -> Option<Node> {
        self.nodes.remove(&id).map(|(_, node)| {
            self.node_headers.remove(&id);
            // Drop the node from its namespace membership set (Issue #3349).
            let ns_id = intern_namespace(&node.namespace());
            let now_empty = match self.ns_nodes.get(&ns_id) {
                Some(members) => {
                    members.remove(&id);
                    members.is_empty()
                }
                None => false,
            };
            // Reclaim a fully-drained namespace set so the outer DashMap does not
            // retain empty entries (Issue #3349 A6). The read guard above is
            // dropped before `remove_if`, which re-checks emptiness under the
            // shard write lock — so a concurrent insert racing the reclaim never
            // removes a set that has just become non-empty.
            if now_empty {
                self.ns_nodes.remove_if(&ns_id, |_, set| set.is_empty());
            }
            // Drop the node from every enabled property-index value bucket. This
            // single choke point covers plain delete, cascade delete, and
            // retraction (they all land on `remove_node`).
            if !self.prop_index_enabled.is_empty() {
                for entry in self.prop_index_enabled.iter() {
                    let (lbl, key) = *entry.key();
                    if lbl != node.label {
                        continue;
                    }
                    if let Some(vk) = node
                        .properties
                        .get_by_interned_key(&key)
                        .and_then(value_key)
                    {
                        self.prop_index_remove((lbl, key, vk), id);
                    }
                }
            }
            node
        })
    }

    /// Get the hot-path header for a node by ID.
    ///
    /// Returns the 16-byte [`NodeHeader`] (id + label) without touching the
    /// full node struct or its `PropertyMap`. Useful when the caller only
    /// needs the label, e.g. for quick existence or type checks.
    ///
    /// Returns `None` if the node does not exist.
    #[inline]
    pub fn get_node_header(&self, id: NodeId) -> Option<NodeHeader> {
        self.node_headers.get(&id).map(|entry| *entry.value())
    }

    /// Return an iterator over the IDs of all nodes whose label matches `label`.
    ///
    /// Scans the compact header map (16 bytes per entry) instead of the full
    /// node map, reducing memory bandwidth by ~10-30x for large property maps.
    /// No allocation occurs until the caller collects the iterator.
    ///
    /// To limit results, chain `.take(n)` on the returned iterator.
    ///
    /// # Performance
    ///
    /// - Reads 16 bytes per candidate (header) vs 100-500+ bytes (full node)
    /// - Zero allocation during scan; caller controls when/whether to collect
    pub fn filter_nodes_by_label(
        &self,
        label: InternedString,
    ) -> impl Iterator<Item = NodeId> + '_ {
        self.node_headers
            .iter()
            .filter(move |entry| entry.value().label == label)
            .map(|entry| *entry.key())
    }

    /// Remove an edge from the indexes.
    ///
    /// Uses tombstone marking for O(1) deletion.
    /// - **O(1)**: Marks edge as deleted in both adjacency indexes
    /// - **Thread-safe**: Lock-free concurrent deletes
    /// - **Immediate**: Edge filtered from adjacency queries immediately
    /// - **Temporal metadata**: Tombstone includes deletion timestamps
    pub fn remove_edge(&self, id: EdgeId) -> Option<Edge> {
        self.edges.remove(&id).map(|(_, edge)| {
            // Mark as deleted in both adjacency indexes (O(1))
            self.outgoing.delete(id);
            self.incoming.delete(id);
            // Drop the edge from its namespace membership set (Issue #3349).
            let ns_id = intern_namespace(&edge.namespace());
            let now_empty = match self.ns_edges.get(&ns_id) {
                Some(members) => {
                    members.remove(&id);
                    members.is_empty()
                }
                None => false,
            };
            // Reclaim a fully-drained namespace set (Issue #3349 A6); the read
            // guard is dropped before `remove_if`, which re-checks under the shard
            // write lock so a concurrent insert never loses its set.
            if now_empty {
                self.ns_edges.remove_if(&ns_id, |_, set| set.is_empty());
            }
            edge
        })
    }

    /// Get the number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ========================================================================
    // Secondary property (equality) index
    // See `crate::index::property_index` for the design and scope boundaries.
    // ========================================================================

    /// Enable the equality index for `(label, prop_key)` and backfill it from
    /// current state. Returns `false` if it was already enabled (no-op).
    ///
    /// # Serialization requirement
    ///
    /// The caller **must** hold the storage `snapshot_lock.write()` for the
    /// duration of this call (see
    /// [`CurrentStorage::enable_property_index`](crate::storage::current::CurrentStorage::enable_property_index)).
    /// Every write path (`create_node`, `insert_node_direct`,
    /// `update_node_direct`, `delete_node_direct`) holds `snapshot_lock.read()`,
    /// so the write lock makes enable **mutually exclusive** with all concurrent
    /// mutators. Without it, two interleavings corrupt the index: (a) a create
    /// whose `insert_node` observed `prop_index_enabled` still empty (so it
    /// skipped maintenance) but installs its node *after* the backfill scan
    /// passed it — leaving the node in no bucket; and (b) a mid-update node read
    /// by the backfill under its old value while the update installs the new
    /// value — leaving it in two buckets.
    ///
    /// The enabled flag is published **last** (after the backfill completes), so
    /// a lock-free reader — which takes no lock — never observes the index as
    /// "enabled" while it is only half-populated: it keeps scanning until the
    /// flag flips, and the flag's release on the `prop_index_enabled` shard
    /// happens-after all bucket writes.
    pub fn enable_property_index(&self, label: InternedString, prop_key: InternedString) -> bool {
        if self.prop_index_enabled.contains_key(&(label, prop_key)) {
            return false;
        }
        // Defensively clear any stale buckets a prior disable may have left for
        // this pair if it raced a concurrent writer (belt-and-suspenders for the
        // #S3 race; the caller's snapshot_lock.write() already excludes writers).
        self.prop_index
            .retain(|k, _| !(k.0 == label && k.1 == prop_key));
        // Backfill BEFORE publishing the flag: index every current node of this
        // label whose property value is indexable.
        for entry in self.nodes.iter() {
            let node = entry.value();
            if node.label != label {
                continue;
            }
            if let Some(vk) = node
                .properties
                .get_by_interned_key(&prop_key)
                .and_then(value_key)
            {
                self.prop_index
                    .entry((label, prop_key, vk))
                    .or_default()
                    .insert(node.id);
            }
        }
        // Publish the flag last (see the release-ordering note above). Enable
        // calls are serialized by the caller's write lock, so the
        // check-then-insert cannot race another enable.
        self.prop_index_enabled.insert((label, prop_key), ());
        true
    }

    /// Disable the equality index for `(label, prop_key)` and purge its buckets.
    /// Returns `false` if no index was enabled for the pair.
    pub fn disable_property_index(&self, label: InternedString, prop_key: InternedString) -> bool {
        if self.prop_index_enabled.remove(&(label, prop_key)).is_none() {
            return false;
        }
        self.prop_index
            .retain(|k, _| !(k.0 == label && k.1 == prop_key));
        true
    }

    /// Whether an equality index is enabled for `(label, prop_key)`.
    #[inline]
    pub fn has_property_index(&self, label: InternedString, prop_key: InternedString) -> bool {
        self.prop_index_enabled.contains_key(&(label, prop_key))
    }

    /// The `(label_id, prop_key_id)` pairs with an enabled equality index.
    pub fn property_index_pairs(&self) -> Vec<(InternedString, InternedString)> {
        self.prop_index_enabled.iter().map(|e| *e.key()).collect()
    }

    /// Probe the equality index for the node ids currently holding `value` under
    /// `(label, prop_key)`. Order is unspecified (`DashSet` iteration); callers
    /// that need determinism sort. Caller must have checked
    /// [`has_property_index`](Self::has_property_index).
    pub fn find_nodes_by_property_indexed(
        &self,
        label: InternedString,
        prop_key: InternedString,
        value: &ValueKey,
    ) -> Vec<NodeId> {
        self.prop_index
            .get(&(label, prop_key, value.clone()))
            .map(|set| {
                // Pre-size from the set's own length: `DashSet::iter()`'s
                // `size_hint()` doesn't reflect the true element count, so an
                // unsized `collect()` under-guesses and `Vec` grows by
                // repeated reallocation as it fills (mirrors the
                // `capacity_hint()` idiom already used by the adjacency
                // merged-path readers in `storage/current/mod.rs`).
                let mut result = Vec::with_capacity(set.len());
                result.extend(set.iter().map(|e| *e.key()));
                result
            })
            .unwrap_or_default()
    }

    // ========================================================================
    // Namespace membership index (Issue #3349, PR2)
    // ========================================================================

    /// Node ids currently in `namespace`, from the secondary membership index.
    ///
    /// O(members) with no property scan. Returns an empty vec for a namespace
    /// with no members. Order is unspecified (DashSet iteration); callers that
    /// need determinism should sort.
    pub fn namespace_node_ids(&self, namespace: &Namespace) -> Vec<NodeId> {
        self.ns_nodes
            .get(&intern_namespace(namespace))
            .map(|members| members.iter().map(|e| *e.key()).collect())
            .unwrap_or_default()
    }

    /// Edge ids currently in `namespace`, from the secondary membership index.
    pub fn namespace_edge_ids(&self, namespace: &Namespace) -> Vec<EdgeId> {
        self.ns_edges
            .get(&intern_namespace(namespace))
            .map(|members| members.iter().map(|e| *e.key()).collect())
            .unwrap_or_default()
    }

    /// Whether `node_id` currently lives in `namespace` (Issue #3349, PR2).
    ///
    /// Convenience wrapper that interns `namespace` and delegates to
    /// [`node_in_namespace_id`](Self::node_in_namespace_id). Hot loops that probe
    /// the same scope repeatedly should intern once (via
    /// [`NamespaceScope::resolved_ids`](crate::core::namespace::NamespaceScope::resolved_ids))
    /// and call the `_id` form to avoid re-interning per candidate.
    #[inline]
    pub fn node_in_namespace(&self, node_id: NodeId, namespace: &Namespace) -> bool {
        self.node_in_namespace_id(node_id, intern_namespace(namespace))
    }

    /// Whether `node_id` currently lives in the interned namespace `ns_id`
    /// (Issue #3349, PR2). O(1): an identity-hashed outer probe plus a set
    /// membership test — no property load, no `Namespace` allocation, no string
    /// hashing — so a scoped traversal's per-edge boundary check is two integer
    /// hash probes.
    #[inline]
    pub fn node_in_namespace_id(&self, node_id: NodeId, ns_id: NamespaceId) -> bool {
        self.ns_nodes
            .get(&ns_id)
            .is_some_and(|members| members.contains(&node_id))
    }

    /// Whether `edge_id` currently lives in `namespace` (Issue #3349, PR2). The
    /// edge counterpart of [`node_in_namespace`](Self::node_in_namespace).
    #[inline]
    pub fn edge_in_namespace(&self, edge_id: EdgeId, namespace: &Namespace) -> bool {
        self.edge_in_namespace_id(edge_id, intern_namespace(namespace))
    }

    /// Whether `edge_id` currently lives in the interned namespace `ns_id`
    /// (Issue #3349, PR2). The edge counterpart of
    /// [`node_in_namespace_id`](Self::node_in_namespace_id).
    #[inline]
    pub fn edge_in_namespace_id(&self, edge_id: EdgeId, ns_id: NamespaceId) -> bool {
        self.ns_edges
            .get(&ns_id)
            .is_some_and(|members| members.contains(&edge_id))
    }

    /// Number of nodes currently in `namespace` (O(1) set-length read).
    #[inline]
    pub fn namespace_node_count(&self, namespace: &Namespace) -> usize {
        self.ns_nodes
            .get(&intern_namespace(namespace))
            .map_or(0, |members| members.len())
    }

    /// Number of edges currently in `namespace` (O(1) set-length read).
    #[inline]
    pub fn namespace_edge_count(&self, namespace: &Namespace) -> usize {
        self.ns_edges
            .get(&intern_namespace(namespace))
            .map_or(0, |members| members.len())
    }

    /// Every namespace that currently holds at least one node or edge, from
    /// the membership index. Used to enumerate populated namespaces for
    /// per-namespace stats/schema without a full entity scan.
    pub fn populated_namespaces(&self) -> std::collections::BTreeSet<Namespace> {
        let mut out = std::collections::BTreeSet::new();
        for entry in self.ns_nodes.iter() {
            if !entry.value().is_empty()
                && let Some(ns) = resolve_namespace_id(*entry.key())
            {
                out.insert(ns);
            }
        }
        for entry in self.ns_edges.iter() {
            if !entry.value().is_empty()
                && let Some(ns) = resolve_namespace_id(*entry.key())
            {
                out.insert(ns);
            }
        }
        out
    }

    /// Get the number of edges.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if a node exists.
    #[inline]
    pub fn contains_node(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Check if an edge exists.
    #[inline]
    pub fn contains_edge(&self, id: EdgeId) -> bool {
        self.edges.contains_key(&id)
    }

    /// Get outgoing edges for a node.
    ///
    /// Returns a guard that provides zero-copy iterator access to adjacency.
    /// Merges frozen CSR + delta buffer + tombstone filtering on-the-fly.
    ///
    /// **Lock-free**: No locks, just atomic loads
    /// **O(1) access**: Returns immediately, no rebuild needed
    /// **Incremental**: Sees edges in both frozen and delta layers
    ///
    /// # Performance
    ///
    /// - Fast path (no delta/tombstones): Direct slice access
    /// - Merge path (with delta): Iterator over frozen + delta (~20-30ns overhead)
    #[inline]
    pub fn get_outgoing(
        &self,
        source: NodeId,
    ) -> crate::index::incremental_adjacency::MergedAdjacencyGuard<'_> {
        self.outgoing.get_adjacency(source)
    }

    /// Get incoming edges for a node.
    ///
    /// Returns a guard that provides zero-copy iterator access to adjacency.
    /// Merges frozen CSR + delta buffer + tombstone filtering on-the-fly.
    ///
    /// **Lock-free**: No locks, just atomic loads
    /// **O(1) access**: Returns immediately, no rebuild needed
    /// **Incremental**: Sees edges in both frozen and delta layers
    #[inline]
    pub fn get_incoming(
        &self,
        target: NodeId,
    ) -> crate::index::incremental_adjacency::MergedAdjacencyGuard<'_> {
        self.incoming.get_adjacency(target)
    }

    /// Get outgoing edges with a specific label.
    ///
    /// **Lock-free**: No locks, just atomic loads
    /// **Incremental**: Filters merged frozen + delta results by label
    pub fn get_outgoing_with_label(
        &self,
        source: NodeId,
        label: InternedString,
    ) -> Vec<AdjacencyEntry> {
        let guard = self.outgoing.get_adjacency(source);
        let mut result = Vec::with_capacity(guard.capacity_hint());
        result.extend(guard.iter().filter(|entry| entry.label == label).copied());
        result
    }

    /// Get incoming edges with a specific label.
    ///
    /// **Lock-free**: No locks, just atomic loads
    /// **Incremental**: Filters merged frozen + delta results by label
    pub fn get_incoming_with_label(
        &self,
        target: NodeId,
        label: InternedString,
    ) -> Vec<AdjacencyEntry> {
        let guard = self.incoming.get_adjacency(target);
        let mut result = Vec::with_capacity(guard.capacity_hint());
        result.extend(guard.iter().filter(|entry| entry.label == label).copied());
        result
    }

    /// Get a frozen view for outgoing adjacency (read transaction hot path).
    ///
    /// Returns `Some(FrozenAdjacencyView)` if the index is in a clean state
    /// (no delta edges or tombstones), allowing direct slice access at ~8-14ns.
    /// Returns `None` if there are pending delta operations, in which case
    /// callers should fall back to `get_outgoing()` for merged access.
    ///
    /// # Performance
    ///
    /// - FrozenView access: ~8-14ns (direct slice)
    /// - MergedGuard access: ~16-17ns (iterator with fast path)
    ///
    /// # Use Case
    ///
    /// Read transactions that don't modify the graph can use this for
    /// maximum performance. The view remains valid for the lifetime of
    /// the borrow, and provides `&[AdjacencyEntry]` directly.
    #[inline]
    pub fn frozen_outgoing_view(
        &self,
    ) -> Option<crate::index::incremental_adjacency::FrozenAdjacencyView> {
        self.outgoing.frozen_view()
    }

    /// Get a frozen view for incoming adjacency (read transaction hot path).
    ///
    /// Same as `frozen_outgoing_view()` but for incoming edges.
    /// Returns `None` if there are pending delta operations.
    #[inline]
    pub fn frozen_incoming_view(
        &self,
    ) -> Option<crate::index::incremental_adjacency::FrozenAdjacencyView> {
        self.incoming.frozen_view()
    }

    /// Get the out-degree of a node (number of outgoing edges).
    ///
    /// **Lock-free**: Uses incremental adjacency with merge-on-read.
    #[inline]
    pub fn out_degree(&self, node: NodeId) -> usize {
        self.get_outgoing(node).len()
    }

    /// Get the in-degree of a node (number of incoming edges).
    ///
    /// **Lock-free**: Uses incremental adjacency with merge-on-read.
    #[inline]
    pub fn in_degree(&self, node: NodeId) -> usize {
        self.get_incoming(node).len()
    }

    /// Compact adjacency indexes to merge delta → frozen.
    ///
    /// With incremental adjacency, this is optional - indexes are always correct.
    /// Compaction improves read performance by moving delta edges to frozen CSR.
    ///
    /// # When to Call
    ///
    /// - After bulk inserts (to move many edges from delta to frozen)
    /// - To reduce delta size before persistence
    /// - Rarely needed since Issue #3810: background maintenance compacts on
    ///   its own once writes go quiet (unless it is disabled by config)
    ///
    /// # Performance
    ///
    /// - Complexity: O(E log E) where E is total edges
    /// - **Lock-free**: Doesn't block concurrent inserts/reads
    /// - **Atomic**: New frozen CSR is swapped in atomically
    /// - Typically <10ms for 10K edges
    pub fn compact_adjacency(&self) {
        self.outgoing.compact();
        self.incoming.compact();
    }

    /// Iterate over all nodes.
    ///
    /// Returns an iterator over DashMap guard references to avoid cloning.
    /// If you need owned `Node` values, call `.map(|n| n.clone())` on the iterator.
    ///
    /// # Performance
    ///
    /// - **Zero allocation**: Returns guards/references to nodes without cloning (~56 bytes + Arc overhead)
    /// - **Efficient**: Avoids implicit cloning in hot loops
    pub fn iter_nodes(
        &self,
    ) -> impl Iterator<Item = impl std::ops::Deref<Target = Node> + '_> + '_ {
        self.nodes.iter()
    }

    /// Iterate over all node IDs.
    ///
    /// This is more efficient than `iter_nodes().map(|n| n.id)` when only
    /// IDs are needed, as it avoids cloning the full Node objects.
    pub fn iter_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().map(|entry| *entry.key())
    }

    /// Iterate over all edges.
    ///
    /// Returns an iterator over DashMap guard references to avoid cloning.
    /// If you need owned `Edge` values, call `.map(|e| e.clone())` on the iterator.
    ///
    /// # Performance
    ///
    /// - **Zero allocation**: Returns guards/references to edges without cloning (~72 bytes + Arc overhead)
    /// - **Efficient**: Avoids implicit cloning in hot loops (benchmark: ~370µs -> ~200µs for 10k edges)
    pub fn iter_edges(
        &self,
    ) -> impl Iterator<Item = impl std::ops::Deref<Target = Edge> + '_> + '_ {
        self.edges.iter()
    }

    /// Iterate over all edge IDs.
    ///
    /// This is more efficient than `iter_edges().map(|e| e.id)` when only
    /// IDs are needed, as it avoids cloning the full Edge objects.
    pub fn iter_edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.edges.iter().map(|entry| *entry.key())
    }

    /// Export outgoing CSR data for persistence.
    ///
    /// Note: This exports only the frozen CSR, not the delta buffer.
    /// Call `compact_adjacency()` first to include recent changes.
    pub fn export_outgoing_csr(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        self.outgoing.export_frozen_csr()
    }

    /// Export incoming CSR data for persistence.
    ///
    /// Note: This exports only the frozen CSR, not the delta buffer.
    /// Call `compact_adjacency()` first to include recent changes.
    pub fn export_incoming_csr(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        self.incoming.export_frozen_csr()
    }

    /// Export both directions' frozen CSRs as one compaction-consistent pair.
    ///
    /// The two indexes are compacted independently by the background
    /// maintenance worker (Issue #3810), so exporting them with two separate
    /// calls can capture an edge in one direction's CSR and not the other's --
    /// a skew the restore path then has to guess about. Holding both compaction
    /// locks across the two exports rules that out: no edge can move from a
    /// delta buffer into a frozen CSR in between.
    ///
    /// Locks are taken outgoing-then-incoming, matching the adjacency lock
    /// order in CLAUDE.md.
    pub fn export_csr_pair(&self) -> ExportedCsrPair {
        let _outgoing = self.outgoing.lock_compaction();
        let _incoming = self.incoming.lock_compaction();
        (
            self.outgoing.export_frozen_csr(),
            self.incoming.export_frozen_csr(),
        )
    }

    /// Import CSR data for both outgoing and incoming adjacency.
    ///
    /// This is used when loading persisted indexes to avoid rebuilding CSR from scratch.
    ///
    /// **Phase 7: Delta Reconstruction**
    ///
    /// After importing the frozen CSR, this method reconstructs the delta buffer by
    /// comparing edges in the DashMap against edges in the frozen CSR. Any edges not
    /// in frozen are inserted into delta, preserving the incremental state without
    /// requiring explicit delta persistence.
    ///
    /// This implements the "implicit delta reconstruction" strategy from ADR-0026.
    ///
    /// # Panics
    ///
    /// Panics if there are uncommitted tombstones (pending edge deletions).
    /// Tombstones cannot be reconstructed from the CSR import, so importing
    /// would cause deleted edges to reappear. This is a correctness invariant.
    ///
    /// In normal startup flow, tombstones should be empty. If you hit this panic,
    /// either compact first to apply tombstones, or ensure import is only called
    /// on a fresh index during startup.
    pub fn import_csr(
        &self,
        outgoing_node_ids: Vec<u64>,
        outgoing_offsets: Vec<u64>,
        outgoing_edge_ids: Vec<u64>,
        incoming_node_ids: Vec<u64>,
        incoming_offsets: Vec<u64>,
        incoming_edge_ids: Vec<u64>,
    ) {
        use crate::core::hasher::IdentityHasher;
        use std::collections::{HashMap, HashSet};
        use std::hash::BuildHasherDefault;

        // Safety check: tombstones cannot be reconstructed from CSR import.
        // If we have tombstones, importing would cause deleted edges to reappear.
        let outgoing_tombstones = self.outgoing.tombstone_count();
        let incoming_tombstones = self.incoming.tombstone_count();
        assert!(
            outgoing_tombstones == 0 && incoming_tombstones == 0,
            "Cannot import CSR with uncommitted tombstones: {} outgoing, {} incoming. \
             Deleted edges would reappear! Call compact() first or ensure import is \
             only called on a fresh index during startup.",
            outgoing_tombstones,
            incoming_tombstones
        );

        // Build edges map for CSR reconstruction
        let mut edges_map = HashMap::with_hasher(BuildHasherDefault::<IdentityHasher>::default());
        for entry in self.edges.iter() {
            let edge = entry.value();
            edges_map.insert(edge.id, (edge.target, edge.label));
        }

        // Import outgoing adjacency frozen CSR
        let outgoing_csr = crate::index::adjacency::AdjacencyIndex::import_csr(
            outgoing_node_ids,
            outgoing_offsets.clone(),
            outgoing_edge_ids.clone(),
            &edges_map,
        );
        self.outgoing.import_frozen_csr(Arc::new(outgoing_csr));

        // Rebuild edges map for incoming (maps edge_id to source, not target)
        edges_map.clear();
        for entry in self.edges.iter() {
            let edge = entry.value();
            edges_map.insert(edge.id, (edge.source, edge.label));
        }

        // Import incoming adjacency frozen CSR
        let incoming_csr = crate::index::adjacency::AdjacencyIndex::import_csr(
            incoming_node_ids,
            incoming_offsets,
            incoming_edge_ids.clone(),
            &edges_map,
        );
        self.incoming.import_frozen_csr(Arc::new(incoming_csr));

        // ===== Phase 7: Reconstruct Delta Buffer =====
        // Build the set of edge IDs each direction's frozen CSR carries.
        //
        // The two sets are derived SEPARATELY (Issue #3810). They used to be
        // assumed identical, which held only while nothing ever compacted: the
        // two indexes are now compacted independently by the background
        // maintenance worker, so a persisted snapshot can legitimately carry an
        // edge in one direction's CSR and not the other's. Deciding both deltas
        // from the outgoing set alone would then either duplicate that edge in
        // the incoming adjacency (frozen + delta) or drop it from the incoming
        // adjacency entirely.
        let frozen_outgoing_ids: HashSet<EdgeId> = outgoing_edge_ids
            .iter()
            .filter_map(|&id| EdgeId::new(id).ok())
            .collect();
        let frozen_incoming_ids: HashSet<EdgeId> = incoming_edge_ids
            .iter()
            .filter_map(|&id| EdgeId::new(id).ok())
            .collect();

        // Iterate through all edges in DashMap
        // For edges NOT in frozen, insert into delta -- per direction.
        for entry in self.edges.iter() {
            let edge = entry.value();

            if !frozen_outgoing_ids.contains(&edge.id) {
                self.outgoing.insert(
                    edge.source,
                    crate::index::adjacency::AdjacencyEntry::new(edge.target, edge.id, edge.label),
                );
            }

            if !frozen_incoming_ids.contains(&edge.id) {
                self.incoming.insert(
                    edge.target,
                    crate::index::adjacency::AdjacencyEntry::new(edge.source, edge.id, edge.label),
                );
            }
        }
    }
}

impl Default for CurrentIndexes {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CurrentIndexes {
    fn drop(&mut self) {
        // Ensure background compaction threads are shut down gracefully
        // to prevent thread leaks when CurrentIndexes is dropped.
        if let Some((scheduler, handle)) = self.outgoing_compaction.take() {
            scheduler.shutdown();
            // Give thread a moment to finish, but don't block indefinitely
            // on drop since that could cause deadlocks.
            let _ = handle.join();
        }
        if let Some((scheduler, handle)) = self.incoming_compaction.take() {
            scheduler.shutdown();
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::id::VersionId;
    use crate::core::interning::GLOBAL_INTERNER;
    use crate::core::property::PropertyMapBuilder;

    pub(super) fn create_test_node(id: u64, label: &str) -> Node {
        Node::new(
            NodeId::new(id).unwrap(),
            GLOBAL_INTERNER.intern(label).unwrap(),
            PropertyMapBuilder::new().build(),
            VersionId::new(1).unwrap(),
        )
    }

    pub(super) fn create_test_edge(id: u64, source: u64, target: u64, label: &str) -> Edge {
        Edge::new(
            EdgeId::new(id).unwrap(),
            GLOBAL_INTERNER.intern(label).unwrap(),
            NodeId::new(source).unwrap(),
            NodeId::new(target).unwrap(),
            PropertyMapBuilder::new().build(),
            VersionId::new(1).unwrap(),
        )
    }

    #[test]
    fn test_node_operations() {
        let indexes = CurrentIndexes::new();

        // Initially empty
        assert_eq!(indexes.node_count(), 0);
        assert!(!indexes.contains_node(NodeId::new(1).unwrap()));

        // Insert node
        let node = create_test_node(1, "Person");
        indexes.insert_node(node.clone());

        assert_eq!(indexes.node_count(), 1);
        assert!(indexes.contains_node(NodeId::new(1).unwrap()));

        // Get node
        let retrieved = indexes.get_node(NodeId::new(1).unwrap()).unwrap();
        assert_eq!(retrieved.id, node.id);
        assert_eq!(retrieved.label, node.label);

        // Remove node
        let removed = indexes.remove_node(NodeId::new(1).unwrap()).unwrap();
        assert_eq!(removed.id, node.id);
        assert_eq!(indexes.node_count(), 0);
    }

    #[test]
    fn test_edge_operations() {
        let indexes = CurrentIndexes::new();

        // Insert edge
        let edge = create_test_edge(1, 0, 1, "KNOWS");
        indexes.insert_edge(edge.clone());

        assert_eq!(indexes.edge_count(), 1);
        assert!(indexes.contains_edge(EdgeId::new(1).unwrap()));

        // Get edge
        let retrieved = indexes.get_edge(EdgeId::new(1).unwrap()).unwrap();
        assert_eq!(retrieved.id, edge.id);
        assert_eq!(retrieved.source, edge.source);
        assert_eq!(retrieved.target, edge.target);

        // Remove edge
        let removed = indexes.remove_edge(EdgeId::new(1).unwrap()).unwrap();
        assert_eq!(removed.id, edge.id);
        assert_eq!(indexes.edge_count(), 0);
    }

    #[test]
    fn test_adjacency_rebuild() {
        let indexes = CurrentIndexes::new();

        // Add nodes
        indexes.insert_node(create_test_node(0, "Person"));
        indexes.insert_node(create_test_node(1, "Person"));
        indexes.insert_node(create_test_node(2, "Person"));

        // Add edges
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 0, 2, "KNOWS"));
        indexes.insert_edge(create_test_edge(2, 1, 2, "KNOWS"));

        // Rebuild adjacency indexes
        indexes.compact_adjacency();

        // Test outgoing edges
        assert_eq!(indexes.out_degree(NodeId::new(0).unwrap()), 2);
        assert_eq!(indexes.out_degree(NodeId::new(1).unwrap()), 1);
        assert_eq!(indexes.out_degree(NodeId::new(2).unwrap()), 0);

        let outgoing = indexes.get_outgoing(NodeId::new(0).unwrap());
        assert_eq!(outgoing.len(), 2);

        // Test incoming edges
        assert_eq!(indexes.in_degree(NodeId::new(0).unwrap()), 0);
        assert_eq!(indexes.in_degree(NodeId::new(1).unwrap()), 1);
        assert_eq!(indexes.in_degree(NodeId::new(2).unwrap()), 2);
    }

    #[test]
    fn test_labeled_traversal() {
        let indexes = CurrentIndexes::new();

        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let follows = GLOBAL_INTERNER.intern("FOLLOWS").unwrap();

        // Add edges with different labels
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 0, 2, "FOLLOWS"));
        indexes.insert_edge(create_test_edge(2, 0, 3, "KNOWS"));

        indexes.compact_adjacency();

        // Get only KNOWS edges
        let knows_edges = indexes.get_outgoing_with_label(NodeId::new(0).unwrap(), knows);
        assert_eq!(knows_edges.len(), 2);

        // Get only FOLLOWS edges
        let follows_edges = indexes.get_outgoing_with_label(NodeId::new(0).unwrap(), follows);
        assert_eq!(follows_edges.len(), 1);
    }

    #[test]
    fn test_iteration() {
        let indexes = CurrentIndexes::new();

        // Add some nodes and edges
        indexes.insert_node(create_test_node(0, "Person"));
        indexes.insert_node(create_test_node(1, "Person"));
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));

        // Test iteration
        let nodes: Vec<_> = indexes.iter_nodes().collect();
        assert_eq!(nodes.len(), 2);

        let edges: Vec<_> = indexes.iter_edges().collect();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_rebuild_idempotent() {
        let indexes = CurrentIndexes::new();

        // Add edges
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 0, 2, "KNOWS"));

        // Rebuild once
        indexes.compact_adjacency();
        let first_out = indexes.get_outgoing(NodeId::new(0).unwrap());
        let first_in = indexes.get_incoming(NodeId::new(1).unwrap());

        // Rebuild again
        indexes.compact_adjacency();
        let second_out = indexes.get_outgoing(NodeId::new(0).unwrap());
        let second_in = indexes.get_incoming(NodeId::new(1).unwrap());

        // Results should be identical
        assert_eq!(first_out.len(), second_out.len());
        assert_eq!(first_in.len(), second_in.len());
        assert_eq!(first_out, second_out);
        assert_eq!(first_in, second_in);
    }

    #[test]
    fn test_lazy_rebuild_on_access() {
        let indexes = CurrentIndexes::new();

        // Add edges WITHOUT calling rebuild_adjacency()
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 0, 2, "KNOWS"));
        indexes.insert_edge(create_test_edge(2, 1, 2, "KNOWS"));

        // Adjacency is immediately visible via incremental index
        let outgoing = indexes.get_outgoing(NodeId::new(0).unwrap());
        assert_eq!(
            outgoing.len(),
            2,
            "Lazy rebuild should make edges accessible"
        );

        // Verify all adjacency data is correct
        assert_eq!(indexes.out_degree(NodeId::new(0).unwrap()), 2);
        assert_eq!(indexes.out_degree(NodeId::new(1).unwrap()), 1);
        assert_eq!(indexes.in_degree(NodeId::new(2).unwrap()), 2);
    }

    /// The two adjacency indexes compact independently, so exporting them with
    /// two separate calls could capture an edge in one direction's frozen CSR
    /// and not the other's -- and the restore path then reconstructed *both*
    /// deltas from the outgoing set alone (Issue #3810).
    ///
    /// `export_csr_pair` holds both compaction locks across the two exports, so
    /// the halves always describe the same edge set.
    #[test]
    fn export_csr_pair_captures_both_directions_at_the_same_point() {
        let indexes = CurrentIndexes::new();
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 0, 2, "KNOWS"));
        indexes.insert_edge(create_test_edge(2, 1, 2, "KNOWS"));
        indexes.compact_adjacency();

        let ((_, _, out_edge_ids), (_, _, in_edge_ids)) = indexes.export_csr_pair();

        let mut out_sorted = out_edge_ids.clone();
        out_sorted.sort_unstable();
        let mut in_sorted = in_edge_ids.clone();
        in_sorted.sort_unstable();

        assert_eq!(
            out_sorted, in_sorted,
            "the exported outgoing and incoming CSRs must describe the same edge \
             set; a difference is the skew export_csr_pair exists to prevent"
        );
        assert_eq!(
            out_sorted,
            vec![0, 1, 2],
            "all three edges must be captured"
        );
    }

    #[test]
    fn test_lazy_rebuild_after_delete() {
        let indexes = CurrentIndexes::new();

        // Add edges and access to trigger initial rebuild
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 1, 2, "KNOWS"));
        let _ = indexes.get_outgoing(NodeId::new(0).unwrap());

        // Remove edge WITHOUT calling rebuild_adjacency()
        indexes.remove_edge(EdgeId::new(1).unwrap());

        // Adjacency should be rebuilt lazily on next access
        assert_eq!(indexes.out_degree(NodeId::new(1).unwrap()), 0);
        assert_eq!(indexes.in_degree(NodeId::new(2).unwrap()), 0);
    }

    #[test]
    fn test_no_unnecessary_rebuilds() {
        let indexes = CurrentIndexes::new();

        // Add edges and trigger rebuild
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        let _ = indexes.get_outgoing(NodeId::new(0).unwrap());

        // Multiple accesses should not trigger additional rebuilds
        // (We can't directly observe this, but it's important for performance)
        for _ in 0..10 {
            let outgoing = indexes.get_outgoing(NodeId::new(0).unwrap());
            assert_eq!(outgoing.len(), 1);
        }

        // After accessing, if no modifications, adjacency should stay current
        assert_eq!(indexes.in_degree(NodeId::new(1).unwrap()), 1);
    }

    #[test]
    fn test_lazy_rebuild_with_labeled_traversal() {
        let indexes = CurrentIndexes::new();

        let knows = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let follows = GLOBAL_INTERNER.intern("FOLLOWS").unwrap();

        // Add edges with different labels WITHOUT explicit rebuild
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 0, 2, "FOLLOWS"));
        indexes.insert_edge(create_test_edge(2, 0, 3, "KNOWS"));

        // Lazy rebuild should happen on labeled access
        let knows_edges = indexes.get_outgoing_with_label(NodeId::new(0).unwrap(), knows);
        assert_eq!(knows_edges.len(), 2);

        let follows_edges = indexes.get_outgoing_with_label(NodeId::new(0).unwrap(), follows);
        assert_eq!(follows_edges.len(), 1);
    }

    /// Test that AdjacencyGuard works correctly and derefs to slice.
    #[test]
    fn test_adjacency_guard_deref() {
        let indexes = CurrentIndexes::new();

        // Add edges
        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 0, 2, "KNOWS"));
        indexes.insert_edge(create_test_edge(2, 1, 2, "KNOWS"));
        indexes.compact_adjacency();

        // Get guard
        let guard = indexes.get_outgoing(NodeId::new(0).unwrap());

        // Should deref to slice
        assert_eq!(guard.len(), 2);
        assert_eq!(guard[0].target, NodeId::new(1).unwrap());
        assert_eq!(guard[1].target, NodeId::new(2).unwrap());

        // Should work with slice methods
        let targets: Vec<_> = guard.iter().map(|e| e.target).collect();
        assert_eq!(targets.len(), 2);
    }

    /// Test that AdjacencyGuard can be used in iterators.
    #[test]
    fn test_adjacency_guard_iteration() {
        let indexes = CurrentIndexes::new();

        // Add edges
        for i in 0..10 {
            indexes.insert_edge(create_test_edge(i, 0, i + 1, "LINK"));
        }
        indexes.compact_adjacency();

        // Get guard and iterate
        let guard = indexes.get_outgoing(NodeId::new(0).unwrap());
        let mut count = 0;
        for entry in guard.iter() {
            assert_eq!(entry.target.as_u64(), count + 1);
            count += 1;
        }
        assert_eq!(count, 10);
    }

    /// Test that AdjacencyGuard works with empty adjacency lists.
    #[test]
    fn test_adjacency_guard_empty() {
        let indexes = CurrentIndexes::new();
        indexes.compact_adjacency();

        // Get guard for node with no edges
        let guard = indexes.get_outgoing(NodeId::new(0).unwrap());
        assert_eq!(guard.len(), 0);
        assert!(guard.is_empty());
    }

    /// Test that AdjacencyGuard can be cloned (by cloning Arc).
    #[test]
    fn test_adjacency_guard_usage_patterns() {
        let indexes = CurrentIndexes::new();

        indexes.insert_edge(create_test_edge(0, 0, 1, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 0, 2, "KNOWS"));
        indexes.compact_adjacency();

        // Get guard
        let guard = indexes.get_outgoing(NodeId::new(0).unwrap());

        // Can use with functional operations
        let edge_ids: Vec<_> = guard.iter().map(|e| e.edge_id).collect();
        assert_eq!(edge_ids.len(), 2);

        // Can use with for loops
        for (i, entry) in guard.iter().enumerate() {
            assert_eq!(entry.edge_id, EdgeId::new(i as u64).unwrap());
        }

        // Can get length
        assert_eq!(guard.len(), 2);

        // Can index
        assert_eq!(guard[0].edge_id, EdgeId::new(0).unwrap());
        assert_eq!(guard[1].edge_id, EdgeId::new(1).unwrap());
    }

    /// Test that incoming guard works the same way.
    #[test]
    fn test_incoming_guard() {
        let indexes = CurrentIndexes::new();

        indexes.insert_edge(create_test_edge(0, 0, 2, "KNOWS"));
        indexes.insert_edge(create_test_edge(1, 1, 2, "KNOWS"));
        indexes.compact_adjacency();

        // Get incoming guard for node 2
        let guard = indexes.get_incoming(NodeId::new(2).unwrap());
        assert_eq!(guard.len(), 2);

        // AdjacencyIndex guarantees sorted order by target node
        let sources: Vec<_> = guard.iter().map(|e| e.target).collect();
        assert_eq!(
            sources,
            vec![NodeId::new(0).unwrap(), NodeId::new(1).unwrap()]
        );
    }

    /// Test AdjacencyGuard Debug implementation for coverage.
    #[test]
    fn test_adjacency_guard_debug() {
        let indexes = CurrentIndexes::new();

        indexes.insert_edge(create_test_edge(0, 5, 10, "KNOWS"));
        indexes.compact_adjacency();

        // Get guard and format with Debug
        let guard = indexes.get_outgoing(NodeId::new(5).unwrap());
        let debug_str = format!("{:?}", guard);

        // Should contain node ID and show it's an AdjacencyGuard
        assert!(debug_str.contains("AdjacencyGuard"));
        assert!(debug_str.contains("node"));
        assert!(debug_str.contains("entry_count"));
    }

    /// Test AdjacencyGuard with empty list for Debug coverage.
    #[test]
    fn test_adjacency_guard_debug_empty() {
        let indexes = CurrentIndexes::new();
        indexes.compact_adjacency();

        // Get guard for non-existent node (empty adjacency list)
        let guard = indexes.get_outgoing(NodeId::new(99).unwrap());
        let debug_str = format!("{:?}", guard);

        // Should format successfully even with empty entries
        assert!(debug_str.contains("AdjacencyGuard"));
        assert!(debug_str.contains("entry_count"));
    }

    /// Test rebuild_adjacency correctly handles many edges.
    ///
    /// This test exercises the pre-allocated vector optimization in
    /// rebuild_adjacency_internal() by creating a graph with many edges
    /// and verifying all adjacencies are correctly computed.
    #[test]
    fn test_rebuild_adjacency_many_edges() {
        let indexes = CurrentIndexes::new();

        // Create a star graph: node 0 connects to nodes 1..=500
        // and nodes 501..=1000 connect to node 0
        // This creates 1000 total edges
        const NUM_EDGES: u64 = 1000;
        const HALF_EDGES: u64 = NUM_EDGES / 2;

        // Add outgoing edges from node 0
        for i in 0..HALF_EDGES {
            indexes.insert_edge(create_test_edge(i, 0, i + 1, "OUTGOING"));
        }

        // Add incoming edges to node 0
        for i in HALF_EDGES..NUM_EDGES {
            indexes.insert_edge(create_test_edge(i, i + 1, 0, "INCOMING"));
        }

        // Rebuild adjacency (this exercises the pre-allocation optimization)
        indexes.compact_adjacency();

        // Verify edge count
        assert_eq!(indexes.edge_count(), NUM_EDGES as usize);

        // Verify outgoing from node 0
        assert_eq!(
            indexes.out_degree(NodeId::new(0).unwrap()),
            HALF_EDGES as usize
        );

        // Verify incoming to node 0
        assert_eq!(
            indexes.in_degree(NodeId::new(0).unwrap()),
            HALF_EDGES as usize
        );

        // Verify all outgoing edges are accessible
        let outgoing = indexes.get_outgoing(NodeId::new(0).unwrap());
        assert_eq!(outgoing.len(), HALF_EDGES as usize);

        // Verify all targets are in expected range (1..=500)
        for entry in outgoing.iter() {
            let target_id = entry.target.as_u64();
            assert!(
                (1..=HALF_EDGES).contains(&target_id),
                "Unexpected outgoing target: {}",
                target_id
            );
        }

        // Verify all incoming edges are accessible
        let incoming = indexes.get_incoming(NodeId::new(0).unwrap());
        assert_eq!(incoming.len(), HALF_EDGES as usize);

        // Verify all sources are in expected range (501..=1000)
        for entry in incoming.iter() {
            let source_id = entry.target.as_u64(); // In incoming, target is the source node
            assert!(
                (HALF_EDGES + 1..=NUM_EDGES).contains(&source_id),
                "Unexpected incoming source: {}",
                source_id
            );
        }
    }
}

// Concurrency tests for rebuild race condition fix
#[cfg(test)]
mod concurrency_tests {
    use super::tests::{create_test_edge, create_test_node};
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    /// Test that rebuilds complete successfully without deadlock.
    #[test]
    fn test_no_deadlock_insert_rebuild() {
        use std::time::Instant;

        let indexes = Arc::new(CurrentIndexes::new());

        for i in 0..5 {
            indexes.insert_node(create_test_node(i, "Node"));
        }

        let start = Instant::now();
        let timeout = Duration::from_secs(5);

        let indexes_clone = Arc::clone(&indexes);
        let insert_handle = thread::spawn(move || {
            for i in 0..500 {
                indexes_clone.insert_edge(create_test_edge(i, i % 5, (i + 1) % 5, "EDGE"));
            }
        });

        let indexes_clone = Arc::clone(&indexes);
        let rebuild_handle = thread::spawn(move || {
            for _ in 0..50 {
                indexes_clone.compact_adjacency();
            }
        });

        // Both should complete within timeout (no deadlock)
        insert_handle.join().unwrap();
        rebuild_handle.join().unwrap();

        assert!(
            start.elapsed() < timeout,
            "Operations should complete without deadlock"
        );

        // Verify final state
        indexes.compact_adjacency();
        assert_eq!(indexes.edge_count(), 500);
    }

    /// Test concurrent lazy rebuild without explicit rebuild_adjacency() calls.
    ///
    /// This test verifies that multiple threads can safely insert edges and
    /// access adjacency concurrently, relying on lazy rebuilding rather than
    /// explicit rebuild calls.
    #[test]
    fn test_concurrent_lazy_rebuild() {
        use std::sync::Arc;
        use std::thread;

        let indexes = Arc::new(CurrentIndexes::new());

        // Add initial nodes
        for i in 0..20 {
            indexes.insert_node(create_test_node(i, "Node"));
        }

        let mut handles = Vec::new();

        // Spawn 3 threads that insert edges WITHOUT calling rebuild_adjacency()
        for thread_id in 0..3 {
            let indexes_clone = Arc::clone(&indexes);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let edge_id = thread_id * 100 + i;
                    indexes_clone.insert_edge(create_test_edge(
                        edge_id,
                        edge_id % 20,
                        (edge_id + 1) % 20,
                        "LINK",
                    ));
                }
            }));
        }

        // Spawn 2 threads that read adjacency (triggering lazy rebuilds)
        for _ in 0..2 {
            let indexes_clone = Arc::clone(&indexes);
            handles.push(thread::spawn(move || {
                for node_id in 0..20 {
                    let node = NodeId::new(node_id).unwrap();
                    // This should trigger lazy rebuild if needed
                    let _outgoing = indexes_clone.get_outgoing(node);
                    let _degree = indexes_clone.out_degree(node);
                }
            }));
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state: all edges should be accessible via adjacency
        assert_eq!(
            indexes.edge_count(),
            150,
            "Should have 150 edges (3 threads × 50 edges)"
        );

        // Verify adjacency is correct (may trigger final lazy rebuild)
        let mut total_out_degree = 0;
        let mut total_in_degree = 0;
        for i in 0..20 {
            total_out_degree += indexes.out_degree(NodeId::new(i).unwrap());
            total_in_degree += indexes.in_degree(NodeId::new(i).unwrap());
        }

        assert_eq!(
            total_out_degree, 150,
            "Lazy rebuild should make all edges accessible via outgoing adjacency"
        );
        assert_eq!(
            total_in_degree, 150,
            "Lazy rebuild should make all edges accessible via incoming adjacency"
        );
    }
}

/// Tests for zero-copy node/edge access methods (Issue #190).
///
/// These tests verify the performance-optimized access patterns that avoid
/// cloning entire Node/Edge structures when only partial data is needed.
#[cfg(test)]
mod zero_copy_access_tests {
    use super::tests::{create_test_edge, create_test_node};
    use super::*;
    use crate::core::interning::GLOBAL_INTERNER;

    // ========================================================================
    // Tests for with_node callback method
    // ========================================================================

    /// Test that with_node provides zero-copy access to node data.
    #[test]
    fn test_with_node_returns_value() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(1, "Person");
        indexes.insert_node(node);

        // Use with_node to extract label without cloning the entire node
        let label = indexes.with_node(NodeId::new(1).unwrap(), |n| n.label);
        assert!(label.is_some());

        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();
        assert_eq!(label.unwrap(), person_label);
    }

    /// Test that with_node returns None for non-existent node.
    #[test]
    fn test_with_node_nonexistent() {
        let indexes = CurrentIndexes::new();

        let result = indexes.with_node(NodeId::new(999).unwrap(), |n| n.label);
        assert!(result.is_none());
    }

    /// Test that with_node can extract multiple fields in a single call.
    #[test]
    fn test_with_node_extract_multiple_fields() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(42, "Company");
        indexes.insert_node(node);

        // Extract multiple fields in one closure
        let (id, label) = indexes
            .with_node(NodeId::new(42).unwrap(), |n| (n.id, n.label))
            .unwrap();

        assert_eq!(id, NodeId::new(42).unwrap());
        let company_label = GLOBAL_INTERNER.intern("Company").unwrap();
        assert_eq!(label, company_label);
    }

    /// Test that with_node can perform computations on node data.
    #[test]
    fn test_with_node_computation() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(5, "Person");
        indexes.insert_node(node);

        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();

        // Check if node has a specific label
        let is_person = indexes
            .with_node(NodeId::new(5).unwrap(), |n| n.label == person_label)
            .unwrap_or(false);

        assert!(is_person);
    }

    // ========================================================================
    // Tests for with_edge callback method
    // ========================================================================

    /// Test that with_edge provides zero-copy access to edge data.
    #[test]
    fn test_with_edge_returns_value() {
        let indexes = CurrentIndexes::new();
        let edge = create_test_edge(1, 10, 20, "KNOWS");
        indexes.insert_edge(edge);

        // Use with_edge to extract endpoints without cloning
        let endpoints = indexes.with_edge(EdgeId::new(1).unwrap(), |e| (e.source, e.target));
        assert!(endpoints.is_some());

        let (source, target) = endpoints.unwrap();
        assert_eq!(source, NodeId::new(10).unwrap());
        assert_eq!(target, NodeId::new(20).unwrap());
    }

    /// Test that with_edge returns None for non-existent edge.
    #[test]
    fn test_with_edge_nonexistent() {
        let indexes = CurrentIndexes::new();

        let result = indexes.with_edge(EdgeId::new(999).unwrap(), |e| e.label);
        assert!(result.is_none());
    }

    /// Test that with_edge can extract label.
    #[test]
    fn test_with_edge_extract_label() {
        let indexes = CurrentIndexes::new();
        let edge = create_test_edge(7, 1, 2, "FOLLOWS");
        indexes.insert_edge(edge);

        let label = indexes
            .with_edge(EdgeId::new(7).unwrap(), |e| e.label)
            .unwrap();

        let follows_label = GLOBAL_INTERNER.intern("FOLLOWS").unwrap();
        assert_eq!(label, follows_label);
    }

    // ========================================================================
    // Tests for get_node_label field accessor
    // ========================================================================

    /// Test that get_node_label returns the label without cloning.
    #[test]
    fn test_get_node_label() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(1, "Person");
        indexes.insert_node(node);

        let label = indexes.get_node_label(NodeId::new(1).unwrap());
        assert!(label.is_some());

        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();
        assert_eq!(label.unwrap(), person_label);
    }

    /// Test that get_node_label returns None for non-existent node.
    #[test]
    fn test_get_node_label_nonexistent() {
        let indexes = CurrentIndexes::new();

        let label = indexes.get_node_label(NodeId::new(999).unwrap());
        assert!(label.is_none());
    }

    // ========================================================================
    // Tests for get_edge_endpoints field accessor
    // ========================================================================

    /// Test that get_edge_endpoints returns source and target.
    #[test]
    fn test_get_edge_endpoints() {
        let indexes = CurrentIndexes::new();
        let edge = create_test_edge(1, 10, 20, "KNOWS");
        indexes.insert_edge(edge);

        let endpoints = indexes.get_edge_endpoints(EdgeId::new(1).unwrap());
        assert!(endpoints.is_some());

        let (source, target) = endpoints.unwrap();
        assert_eq!(source, NodeId::new(10).unwrap());
        assert_eq!(target, NodeId::new(20).unwrap());
    }

    /// Test that get_edge_endpoints returns None for non-existent edge.
    #[test]
    fn test_get_edge_endpoints_nonexistent() {
        let indexes = CurrentIndexes::new();

        let endpoints = indexes.get_edge_endpoints(EdgeId::new(999).unwrap());
        assert!(endpoints.is_none());
    }

    // ========================================================================
    // Tests for get_edge_label field accessor
    // ========================================================================

    /// Test that get_edge_label returns the label without cloning.
    #[test]
    fn test_get_edge_label() {
        let indexes = CurrentIndexes::new();
        let edge = create_test_edge(1, 10, 20, "KNOWS");
        indexes.insert_edge(edge);

        let label = indexes.get_edge_label(EdgeId::new(1).unwrap());
        assert!(label.is_some());

        let knows_label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        assert_eq!(label.unwrap(), knows_label);
    }

    /// Test that get_edge_label returns None for non-existent edge.
    #[test]
    fn test_get_edge_label_nonexistent() {
        let indexes = CurrentIndexes::new();

        let label = indexes.get_edge_label(EdgeId::new(999).unwrap());
        assert!(label.is_none());
    }

    // ========================================================================
    // Tests for get_edge_source and get_edge_target field accessors
    // ========================================================================

    /// Test that get_edge_source returns the source node.
    #[test]
    fn test_get_edge_source() {
        let indexes = CurrentIndexes::new();
        let edge = create_test_edge(1, 10, 20, "KNOWS");
        indexes.insert_edge(edge);

        let source = indexes.get_edge_source(EdgeId::new(1).unwrap());
        assert!(source.is_some());
        assert_eq!(source.unwrap(), NodeId::new(10).unwrap());
    }

    /// Test that get_edge_source returns None for non-existent edge.
    #[test]
    fn test_get_edge_source_nonexistent() {
        let indexes = CurrentIndexes::new();

        let source = indexes.get_edge_source(EdgeId::new(999).unwrap());
        assert!(source.is_none());
    }

    /// Test that get_edge_target returns the target node.
    #[test]
    fn test_get_edge_target() {
        let indexes = CurrentIndexes::new();
        let edge = create_test_edge(1, 10, 20, "KNOWS");
        indexes.insert_edge(edge);

        let target = indexes.get_edge_target(EdgeId::new(1).unwrap());
        assert!(target.is_some());
        assert_eq!(target.unwrap(), NodeId::new(20).unwrap());
    }

    /// Test that get_edge_target returns None for non-existent edge.
    #[test]
    fn test_get_edge_target_nonexistent() {
        let indexes = CurrentIndexes::new();

        let target = indexes.get_edge_target(EdgeId::new(999).unwrap());
        assert!(target.is_none());
    }

    #[test]
    fn test_iter_nodes_returns_references() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(1, "Person");
        indexes.insert_node(node.clone());

        // Should be able to deref without clone
        let collected: Vec<_> = indexes.iter_nodes().map(|n| n.id).collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0], node.id);

        // Should be able to clone when needed
        let cloned: Vec<Node> = indexes.iter_nodes().map(|n| n.clone()).collect();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned[0].id, node.id);
    }

    #[test]
    fn test_iter_edges_returns_references() {
        let indexes = CurrentIndexes::new();
        let edge = create_test_edge(1, 0, 1, "KNOWS");
        indexes.insert_edge(edge.clone());

        // Should be able to deref without clone
        let collected: Vec<_> = indexes.iter_edges().map(|e| e.id).collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0], edge.id);

        // Should be able to clone when needed
        let cloned: Vec<Edge> = indexes.iter_edges().map(|e| e.clone()).collect();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned[0].id, edge.id);
    }

    // ========================================================================
    // Tests for property accessors
    // ========================================================================

    /// Test that get_node_property returns a property value.
    #[test]
    fn test_get_node_property() {
        let indexes = CurrentIndexes::new();

        // Create a node with properties
        use crate::core::id::VersionId;
        use crate::core::property::PropertyMapBuilder;
        let node = Node::new(
            NodeId::new(1).unwrap(),
            GLOBAL_INTERNER.intern("Person").unwrap(),
            PropertyMapBuilder::new()
                .insert("name", "Alice")
                .insert("age", 30i64)
                .build(),
            VersionId::new(1).unwrap(),
        );
        indexes.insert_node(node);

        // Get specific properties
        let name = indexes.get_node_property(NodeId::new(1).unwrap(), "name");
        assert!(name.is_some());
        assert_eq!(name.unwrap().as_str(), Some("Alice"));

        let age = indexes.get_node_property(NodeId::new(1).unwrap(), "age");
        assert!(age.is_some());
        assert_eq!(age.unwrap().as_int(), Some(30));
    }

    /// Test that get_node_property returns None for non-existent property.
    #[test]
    fn test_get_node_property_missing_key() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(1, "Person");
        indexes.insert_node(node);

        let prop = indexes.get_node_property(NodeId::new(1).unwrap(), "nonexistent");
        assert!(prop.is_none());
    }

    /// Test that get_node_property returns None for non-existent node.
    #[test]
    fn test_get_node_property_nonexistent_node() {
        let indexes = CurrentIndexes::new();

        let prop = indexes.get_node_property(NodeId::new(999).unwrap(), "name");
        assert!(prop.is_none());
    }

    /// Test that get_edge_property returns a property value.
    #[test]
    fn test_get_edge_property() {
        let indexes = CurrentIndexes::new();

        // Create an edge with properties
        use crate::core::id::VersionId;
        use crate::core::property::PropertyMapBuilder;
        let edge = Edge::new(
            EdgeId::new(1).unwrap(),
            GLOBAL_INTERNER.intern("KNOWS").unwrap(),
            NodeId::new(10).unwrap(),
            NodeId::new(20).unwrap(),
            PropertyMapBuilder::new()
                .insert("since", 2020i64)
                .insert("strength", 0.9f64)
                .build(),
            VersionId::new(1).unwrap(),
        );
        indexes.insert_edge(edge);

        // Get specific properties
        let since = indexes.get_edge_property(EdgeId::new(1).unwrap(), "since");
        assert!(since.is_some());
        assert_eq!(since.unwrap().as_int(), Some(2020));

        let strength = indexes.get_edge_property(EdgeId::new(1).unwrap(), "strength");
        assert!(strength.is_some());
        assert_eq!(strength.unwrap().as_float(), Some(0.9));
    }

    /// Test that get_edge_property returns None for non-existent property.
    #[test]
    fn test_get_edge_property_missing_key() {
        let indexes = CurrentIndexes::new();
        let edge = create_test_edge(1, 10, 20, "KNOWS");
        indexes.insert_edge(edge);

        let prop = indexes.get_edge_property(EdgeId::new(1).unwrap(), "nonexistent");
        assert!(prop.is_none());
    }

    /// Test that get_edge_property returns None for non-existent edge.
    #[test]
    fn test_get_edge_property_nonexistent_edge() {
        let indexes = CurrentIndexes::new();

        let prop = indexes.get_edge_property(EdgeId::new(999).unwrap(), "weight");
        assert!(prop.is_none());
    }

    // ========================================================================
    // Tests for interned-key property accessors (hot path optimization)
    // ========================================================================

    /// Test that get_node_property_by_key works with pre-interned keys.
    #[test]
    fn test_get_node_property_by_key() {
        let indexes = CurrentIndexes::new();

        // Create a node with properties
        use crate::core::id::VersionId;
        use crate::core::property::PropertyMapBuilder;
        let node = Node::new(
            NodeId::new(1).unwrap(),
            GLOBAL_INTERNER.intern("Person").unwrap(),
            PropertyMapBuilder::new()
                .insert("name", "Alice")
                .insert("age", 30i64)
                .build(),
            VersionId::new(1).unwrap(),
        );
        indexes.insert_node(node);

        // Pre-intern the key
        let name_key = GLOBAL_INTERNER.intern("name").unwrap();

        // Get property using interned key
        let name = indexes.get_node_property_by_key(NodeId::new(1).unwrap(), &name_key);
        assert!(name.is_some());
        assert_eq!(name.unwrap().as_str(), Some("Alice"));
    }

    /// Test that get_node_property_by_key returns None for missing key.
    #[test]
    fn test_get_node_property_by_key_missing() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(1, "Person");
        indexes.insert_node(node);

        let nonexistent_key = GLOBAL_INTERNER.intern("nonexistent_key_test").unwrap();
        let prop = indexes.get_node_property_by_key(NodeId::new(1).unwrap(), &nonexistent_key);
        assert!(prop.is_none());
    }

    /// Test that get_edge_property_by_key works with pre-interned keys.
    #[test]
    fn test_get_edge_property_by_key() {
        let indexes = CurrentIndexes::new();

        // Create an edge with properties
        use crate::core::id::VersionId;
        use crate::core::property::PropertyMapBuilder;
        let edge = Edge::new(
            EdgeId::new(1).unwrap(),
            GLOBAL_INTERNER.intern("KNOWS").unwrap(),
            NodeId::new(10).unwrap(),
            NodeId::new(20).unwrap(),
            PropertyMapBuilder::new().insert("since", 2020i64).build(),
            VersionId::new(1).unwrap(),
        );
        indexes.insert_edge(edge);

        // Pre-intern the key
        let since_key = GLOBAL_INTERNER.intern("since").unwrap();

        // Get property using interned key
        let since = indexes.get_edge_property_by_key(EdgeId::new(1).unwrap(), &since_key);
        assert!(since.is_some());
        assert_eq!(since.unwrap().as_int(), Some(2020));
    }

    /// Test that get_edge_property_by_key returns None for missing key.
    #[test]
    fn test_get_edge_property_by_key_missing() {
        let indexes = CurrentIndexes::new();
        let edge = create_test_edge(1, 10, 20, "KNOWS");
        indexes.insert_edge(edge);

        let nonexistent_key = GLOBAL_INTERNER.intern("nonexistent_edge_key_test").unwrap();
        let prop = indexes.get_edge_property_by_key(EdgeId::new(1).unwrap(), &nonexistent_key);
        assert!(prop.is_none());
    }

    // ========================================================================
    // Integration tests - verify zero-copy methods work correctly with
    // concurrent operations
    // ========================================================================

    /// Test that zero-copy methods work correctly under concurrent access.
    #[test]
    fn test_zero_copy_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let indexes = Arc::new(CurrentIndexes::new());

        // Insert initial data
        for i in 0..100 {
            indexes.insert_node(create_test_node(i, "Node"));
            if i > 0 {
                indexes.insert_edge(create_test_edge(i, i - 1, i, "LINK"));
            }
        }

        let mut handles = Vec::new();

        // Spawn readers using zero-copy methods
        for _ in 0..4 {
            let indexes_clone = Arc::clone(&indexes);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let node_id = NodeId::new(i).unwrap();
                    let _label = indexes_clone.get_node_label(node_id);

                    if i > 0 {
                        let edge_id = EdgeId::new(i).unwrap();
                        let _endpoints = indexes_clone.get_edge_endpoints(edge_id);
                        let _edge_label = indexes_clone.get_edge_label(edge_id);
                    }
                }
            }));
        }

        // All threads should complete without issues
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
    }

    #[test]
    fn test_iter_node_ids_optimization() {
        let indexes = CurrentIndexes::new();

        // Create nodes with properties to verify we're not cloning them
        let n1 = create_test_node(1, "Person");
        let n2 = create_test_node(2, "Person");
        indexes.insert_node(n1);
        indexes.insert_node(n2);

        // Collect IDs using new optimized method
        let ids: Vec<NodeId> = indexes.iter_node_ids().collect();

        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&NodeId::new(1).unwrap()));
        assert!(ids.contains(&NodeId::new(2).unwrap()));
    }

    #[test]
    fn test_iter_edge_ids_optimization() {
        let indexes = CurrentIndexes::new();

        let e1 = create_test_edge(1, 0, 1, "KNOWS");
        let e2 = create_test_edge(2, 1, 2, "FOLLOWS");
        indexes.insert_edge(e1);
        indexes.insert_edge(e2);

        let ids: Vec<EdgeId> = indexes.iter_edge_ids().collect();

        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&EdgeId::new(1).unwrap()));
        assert!(ids.contains(&EdgeId::new(2).unwrap()));
    }

    // ========================================================================
    // Tests for check_node_label (Issue #339)
    // ========================================================================

    /// check_node_label returns true when the node exists and its label matches.
    #[test]
    fn test_check_node_label_matches() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(1, "Person");
        indexes.insert_node(node);

        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();
        assert!(indexes.check_node_label(NodeId::new(1).unwrap(), person_label));
    }

    /// check_node_label returns false when the node exists but its label differs.
    #[test]
    fn test_check_node_label_mismatch() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(1, "Person");
        indexes.insert_node(node);

        let company_label = GLOBAL_INTERNER.intern("Company").unwrap();
        assert!(!indexes.check_node_label(NodeId::new(1).unwrap(), company_label));
    }

    /// check_node_label returns false for a node that does not exist.
    #[test]
    fn test_check_node_label_nonexistent() {
        let indexes = CurrentIndexes::new();

        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();
        assert!(!indexes.check_node_label(NodeId::new(999).unwrap(), person_label));
    }

    /// check_node_label is consistent with get_node_label for the same node.
    #[test]
    fn test_check_node_label_consistent_with_get_node_label() {
        let indexes = CurrentIndexes::new();
        let node = create_test_node(5, "Document");
        indexes.insert_node(node);

        let doc_label = GLOBAL_INTERNER.intern("Document").unwrap();
        let other_label = GLOBAL_INTERNER.intern("Article").unwrap();

        // check_node_label should agree with manual get_node_label comparison
        let node_id = NodeId::new(5).unwrap();
        let via_get = indexes
            .get_node_label(node_id)
            .map(|l| l == doc_label)
            .unwrap_or(false);
        assert_eq!(indexes.check_node_label(node_id, doc_label), via_get);

        let via_get_other = indexes
            .get_node_label(node_id)
            .map(|l| l == other_label)
            .unwrap_or(false);
        assert_eq!(
            indexes.check_node_label(node_id, other_label),
            via_get_other
        );
    }
}

/// Tests for hot/cold path split via NodeHeader (Issue #340).
///
/// NodeHeader stores only id+label (16 bytes) so label-filtering scans
/// touch far less memory than iterating full Node structs.
#[cfg(test)]
mod node_header_index_tests {
    use super::tests::create_test_node;
    use super::*;
    use crate::core::interning::GLOBAL_INTERNER;

    #[test]
    fn test_insert_node_creates_header() {
        let indexes = CurrentIndexes::new();
        indexes.insert_node(create_test_node(1, "Person"));

        let header = indexes.get_node_header(NodeId::new(1).unwrap());
        assert!(
            header.is_some(),
            "inserting a node should create its header"
        );

        let h = header.unwrap();
        assert_eq!(h.id, NodeId::new(1).unwrap());
        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();
        assert_eq!(h.label, person_label);
    }

    #[test]
    fn test_remove_node_removes_header() {
        let indexes = CurrentIndexes::new();
        indexes.insert_node(create_test_node(2, "Person"));
        assert!(indexes.get_node_header(NodeId::new(2).unwrap()).is_some());

        indexes.remove_node(NodeId::new(2).unwrap());
        assert!(
            indexes.get_node_header(NodeId::new(2).unwrap()).is_none(),
            "removing a node must also remove its header"
        );
    }

    #[test]
    fn test_get_node_header_nonexistent() {
        let indexes = CurrentIndexes::new();
        assert!(indexes.get_node_header(NodeId::new(999).unwrap()).is_none());
    }

    #[test]
    fn test_filter_nodes_by_label_matches() {
        let indexes = CurrentIndexes::new();
        indexes.insert_node(create_test_node(1, "Person"));
        indexes.insert_node(create_test_node(2, "Person"));
        indexes.insert_node(create_test_node(3, "Company"));

        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();
        let mut ids: Vec<_> = indexes.filter_nodes_by_label(person_label).collect();
        ids.sort();

        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], NodeId::new(1).unwrap());
        assert_eq!(ids[1], NodeId::new(2).unwrap());
    }

    #[test]
    fn test_filter_nodes_by_label_no_match() {
        let indexes = CurrentIndexes::new();
        indexes.insert_node(create_test_node(1, "Person"));

        let company_label = GLOBAL_INTERNER.intern("Company").unwrap();
        assert_eq!(indexes.filter_nodes_by_label(company_label).count(), 0);
    }

    #[test]
    fn test_filter_nodes_by_label_all_match() {
        let indexes = CurrentIndexes::new();
        for i in 1u64..=5 {
            indexes.insert_node(create_test_node(i, "Event"));
        }

        let event_label = GLOBAL_INTERNER.intern("Event").unwrap();
        assert_eq!(indexes.filter_nodes_by_label(event_label).count(), 5);
    }

    #[test]
    fn test_filter_nodes_by_label_empty_index() {
        let indexes = CurrentIndexes::new();
        let label = GLOBAL_INTERNER.intern("Anything").unwrap();
        assert_eq!(indexes.filter_nodes_by_label(label).count(), 0);
    }

    #[test]
    fn test_header_consistent_with_node() {
        let indexes = CurrentIndexes::new();
        indexes.insert_node(create_test_node(10, "User"));

        let node = indexes.get_node(NodeId::new(10).unwrap()).unwrap();
        let header = indexes.get_node_header(NodeId::new(10).unwrap()).unwrap();

        assert_eq!(header.id, node.id);
        assert_eq!(header.label, node.label);
    }

    #[test]
    fn test_header_updated_after_label_mutation_via_with_node_mut() {
        let indexes = CurrentIndexes::new();
        indexes.insert_node(create_test_node(20, "Person"));

        let user_label = GLOBAL_INTERNER.intern("User").unwrap();
        indexes.with_node_mut(NodeId::new(20).unwrap(), |n| {
            n.label = user_label;
        });

        // Header must reflect the new label immediately
        let header = indexes.get_node_header(NodeId::new(20).unwrap()).unwrap();
        assert_eq!(
            header.label, user_label,
            "header must reflect mutated label"
        );

        // filter_nodes_by_label must use updated header
        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();
        assert_eq!(
            indexes.filter_nodes_by_label(person_label).count(),
            0,
            "old label should no longer match"
        );
        assert_eq!(
            indexes.filter_nodes_by_label(user_label).count(),
            1,
            "new label should match"
        );
    }
}

#[cfg(test)]
mod property_index_publish_order_tests {
    use super::*;
    use crate::core::graph::Node;
    use crate::core::id::{NodeId, VersionId};
    use crate::core::interning::GLOBAL_INTERNER;
    use crate::core::property::PropertyMapBuilder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};

    /// A secondary index entry must never be visible before the node it points
    /// at.
    ///
    /// `insert_node` used to publish the property index first and the node
    /// second, so a lookup racing an insert could be handed an id and then fail
    /// to resolve it -- `find_nodes_by_property` returns `NodeId(n)`,
    /// `get_node(n)` says `NodeNotFound`. The window is two instructions wide,
    /// which is why it survived: at the commit rates group commit could reach
    /// (~93/s, before the flush wait moved out of the commit lock) the
    /// integration suite never hit it in 120 runs. Driving `insert_node`
    /// directly removes the commit path from the picture and reproduces it in
    /// well under a second.
    #[test]
    fn a_property_index_hit_always_resolves_to_a_node() {
        let indexes = Arc::new(CurrentIndexes::new());
        let label = GLOBAL_INTERNER.intern("Item").unwrap();
        let key = GLOBAL_INTERNER.intern("batch").unwrap();
        assert!(indexes.enable_property_index(label, key));

        let stop = Arc::new(AtomicBool::new(false));
        let dangling = Arc::new(AtomicU64::new(0));
        let probes = Arc::new(AtomicU64::new(0));

        let writer = {
            let indexes = Arc::clone(&indexes);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut id = 1u64;
                while !stop.load(AtomicOrdering::Relaxed) && id < 200_000 {
                    indexes.insert_node(Node::new(
                        NodeId::new(id).unwrap(),
                        label,
                        PropertyMapBuilder::new().insert("batch", "v").build(),
                        VersionId::new(1).unwrap(),
                    ));
                    id += 1;
                }
                id
            })
        };

        let readers: Vec<_> = (0..3)
            .map(|_| {
                let indexes = Arc::clone(&indexes);
                let stop = Arc::clone(&stop);
                let dangling = Arc::clone(&dangling);
                let probes = Arc::clone(&probes);
                std::thread::spawn(move || {
                    let vk = crate::index::property_index::value_key(
                        &crate::core::property::PropertyValue::from("v"),
                    )
                    .expect("string values are indexable");
                    while !stop.load(AtomicOrdering::Relaxed) {
                        for id in indexes.find_nodes_by_property_indexed(label, key, &vk) {
                            probes.fetch_add(1, AtomicOrdering::Relaxed);
                            if indexes.get_node(id).is_none() {
                                dangling.fetch_add(1, AtomicOrdering::Relaxed);
                            }
                        }
                    }
                })
            })
            .collect();

        std::thread::sleep(std::time::Duration::from_millis(400));
        stop.store(true, AtomicOrdering::Relaxed);

        let written = writer.join().expect("writer");
        for r in readers {
            r.join().expect("reader");
        }

        assert!(written > 1, "writer made no progress");
        assert!(
            probes.load(AtomicOrdering::Relaxed) > 0,
            "readers probed nothing, so the test proved nothing"
        );
        assert_eq!(
            dangling.load(AtomicOrdering::Relaxed),
            0,
            "the property index handed out ids that did not resolve to a node"
        );
    }
}
