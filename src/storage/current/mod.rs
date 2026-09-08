//! Current-state storage engine.
//!
//! This module implements the "hot path" storage for the current state of the
//! graph. It provides O(1) lookups and cache-friendly traversals optimized for
//! non-temporal queries.

use crate::core::error::{Result, StorageError};
use crate::core::graph::{Edge, Node};
use crate::core::id::{EdgeId, IdGenerator, NodeId, VersionId};
use crate::core::interning::{GLOBAL_INTERNER, InternedString};
use crate::core::property::{PropertyMap, PropertyValue};
use crate::core::temporal::Timestamp;
use crate::index::adjacency_maintenance::AdjacencyMaintenanceConfig;
use crate::index::current::{AdjacencyIndexStats, CurrentIndexes, ExportedCsrPair};
use crate::index::vector::hnsw::{HnswConfig, HnswIndex};
use crate::index::vector::temporal::{TemporalVectorConfig, TemporalVectorIndex};
use crate::index::vector::{TemporalSearchResults, VectorIndex};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use parking_lot::RwLock;
use std::sync::Arc;

mod apply_gate;
mod iterators;
mod stats;
mod vector;

use apply_gate::ApplyGate;
pub(crate) use apply_gate::PublishGuard;
pub use iterators::*;
pub use stats::CurrentStats;
use stats::FilterStats;
pub use vector::VectorIndexInfo;
use vector::{TemporalVectorIndexEntry, VectorIndexEntry};

/// Maximum number of vector-indexed properties allowed per database.
///
/// This limit prevents resource exhaustion from enabling too many vector indexes,
/// as each index consumes significant memory (1.5-2x the raw vector data size
/// due to HNSW graph overhead).
///
/// Override via configuration if your use case requires more properties.
pub const DEFAULT_MAX_VECTOR_PROPERTIES: usize = 10;

/// Current-state storage engine.
///
/// This storage engine maintains the current version of all nodes and edges,
/// optimized for fast queries without temporal overhead. This is the "fast path"
/// that should achieve <1µs single-hop traversals.
pub struct CurrentStorage {
    /// Indexes for nodes and edges
    indexes: CurrentIndexes,
    /// ID generator for nodes
    node_id_gen: IdGenerator,
    /// ID generator for edges
    edge_id_gen: IdGenerator,
    /// ID generator for versions
    version_id_gen: IdGenerator,
    /// Multi-property vector indexes (Issue #389)
    /// Maps property name -> VectorIndexEntry
    vector_indexes: DashMap<String, VectorIndexEntry>,
    /// Multi-property temporal vector indexes (Issue #389 fix)
    /// Maps property name -> TemporalVectorIndexEntry
    temporal_vector_indexes: DashMap<String, TemporalVectorIndexEntry>,
    /// Adaptive over-fetch statistics per label (Issue #334)
    /// Maps label -> FilterStats for tracking label-specific filter pass rates
    ///
    /// **Memory Considerations**: This map grows unbounded as new labels are encountered.
    /// For workloads with bounded label sets (typical case), memory usage is negligible
    /// (~100 bytes per unique label). For unbounded labels (e.g., using UUIDs as labels),
    /// memory usage scales linearly with unique label count (~50MB per 1M labels).
    ///
    /// **Recommendation**: Use a bounded set of labels for optimal performance.
    /// Future enhancement could add LRU eviction for unbounded scenarios.
    filter_stats: DashMap<String, Arc<FilterStats>>,
    /// Lock to synchronize snapshot creation with concurrent writes
    pub(crate) snapshot_lock: RwLock<()>,
    /// Atomic-publish gate for replica batch application (Issue #3788).
    ///
    /// Disarmed (and free) on a primary; armed when the owning database enters
    /// replica mode, at which point point lookups become atomic with respect to
    /// the applier's batch replay. See [`apply_gate`].
    apply_gate: ApplyGate,
}

impl CurrentStorage {
    /// Create a new empty current storage.
    ///
    /// Its adjacency indexes are enrolled in the shared background maintenance
    /// worker (Issue #3810) with the default policy, so reads reach the
    /// frozen-CSR fast path once writes go quiet. Use
    /// [`with_adjacency_maintenance`](Self::with_adjacency_maintenance) to
    /// choose a different policy (or none).
    pub fn new() -> Self {
        Self::with_adjacency_maintenance(AdjacencyMaintenanceConfig::default())
    }

    /// Create a new empty current storage with an explicit background
    /// adjacency-maintenance policy (Issue #3810).
    pub fn with_adjacency_maintenance(maintenance: AdjacencyMaintenanceConfig) -> Self {
        CurrentStorage {
            indexes: CurrentIndexes::with_maintenance_config(maintenance),
            node_id_gen: IdGenerator::new(),
            edge_id_gen: IdGenerator::new(),
            version_id_gen: IdGenerator::new(),
            vector_indexes: DashMap::new(),
            temporal_vector_indexes: DashMap::new(),
            filter_stats: DashMap::new(),
            snapshot_lock: RwLock::new(()),
            apply_gate: ApplyGate::new(),
        }
    }

    /// Arm the replica apply gate so current-state point lookups become atomic
    /// with respect to [`Self::begin_apply_publish`] (Issue #3788).
    ///
    /// Called when the owning database enters replica mode, before the applier
    /// thread is spawned. Idempotent.
    pub(crate) fn arm_apply_gate(&self) {
        self.apply_gate.arm();
    }

    /// Disarm the replica apply gate, restoring the ungated read fast path.
    ///
    /// Only valid once no applier can still publish -- `promote_to_primary`
    /// stops and joins the applier thread first.
    pub(crate) fn disarm_apply_gate(&self) {
        self.apply_gate.disarm();
    }

    /// Whether current-state reads are currently gated (i.e. this storage backs
    /// a replica). Exposed for tests and diagnostics.
    #[cfg(test)]
    pub(crate) fn apply_gate_armed(&self) -> bool {
        self.apply_gate.is_armed()
    }

    /// Open an atomic publish window over current state.
    ///
    /// Every mutation applied until the returned guard is dropped becomes
    /// visible to gated point lookups all at once, on drop. The replica applier
    /// wraps a complete-frame batch replay in one of these so a concurrent read
    /// can never observe a half-applied transaction (Issue #3788).
    ///
    /// Re-entrant for the publishing thread: the replay engine's idempotency
    /// guards read current state from inside the window.
    #[must_use = "the publish window closes when the guard is dropped"]
    pub(crate) fn begin_apply_publish(&self) -> PublishGuard<'_> {
        self.apply_gate.begin_publish()
    }

    /// Initialize the node ID generator with a specific starting value.
    ///
    /// This is used during recovery to ensure the ID generator continues from
    /// the maximum ID found in the WAL, preventing ID conflicts.
    ///
    /// # Arguments
    ///
    /// * `start` - The next ID to generate (typically max_id + 1)
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn init_node_id_generator(&self, start: u64) {
        self.node_id_gen.reset_to(start);
    }

    /// Initialize the edge ID generator with a specific starting value.
    ///
    /// This is used during recovery to ensure the ID generator continues from
    /// the maximum ID found in the WAL, preventing ID conflicts.
    ///
    /// # Arguments
    ///
    /// * `start` - The next ID to generate (typically max_id + 1)
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn init_edge_id_generator(&self, start: u64) {
        self.edge_id_gen.reset_to(start);
    }

    /// Initialize the version ID generator with a specific starting value.
    ///
    /// This is used during recovery to ensure the ID generator continues from
    /// the maximum ID found in the WAL, preventing ID conflicts.
    ///
    /// # Arguments
    ///
    /// * `start` - The next ID to generate (typically max_id + 1)
    #[inline]
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn init_version_id_generator(&self, start: u64) {
        self.version_id_gen.reset_to(start);
    }

    /// Ensure the version ID generator's next value is at least the specified minimum.
    ///
    /// This is used during recovery when we need to account for version IDs from multiple
    /// sources (e.g., current storage and historical storage) without overwriting
    /// a higher value that was already set.
    ///
    /// # Arguments
    ///
    /// * `min_value` - The minimum next version ID to generate
    #[inline]
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn ensure_version_id_generator_at_least(&self, min_value: u64) {
        self.version_id_gen.ensure_at_least(min_value);
    }

    /// Get the current value of the version ID generator.
    ///
    /// This is used during recovery to determine the starting version ID for
    /// WAL replay.
    #[inline]
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn get_version_id_generator_current(&self) -> u64 {
        self.version_id_gen.current()
    }

    /// Enable vector indexing for a specific property.
    ///
    /// Multiple properties can be indexed simultaneously with different configurations.
    /// Each property can have its own dimensions and distance metric.
    ///
    /// # Arguments
    ///
    /// * `property_name` - Name of the property containing vectors
    /// * `config` - HNSW index configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - This specific property already has a vector index enabled
    /// - The maximum number of indexed properties ([`DEFAULT_MAX_VECTOR_PROPERTIES`]) is reached
    ///
    /// # Memory Usage
    ///
    /// Each vector index consumes approximately **1.5-2x** the raw vector data size due to
    /// HNSW graph overhead (neighbor lists, layer structure). For example:
    ///
    /// | Vectors | Dimensions | Raw Size | Index Size |
    /// |---------|------------|----------|------------|
    /// | 100K    | 384        | ~150 MB  | ~225-300 MB |
    /// | 1M      | 768        | ~3 GB    | ~4.5-6 GB   |
    /// | 1M      | 1536       | ~6 GB    | ~9-12 GB    |
    ///
    /// Plan capacity accordingly when enabling multiple property indexes.
    ///
    /// # Persistence Limitation
    ///
    /// **Warning**: Currently, the checkpoint format only supports persisting a single
    /// vector index configuration (the "default" one, which is alphabetically first).
    /// If you enable multiple vector indexes, **only one will be persisted** in checkpoints.
    /// The others will be lost upon recovery from a checkpoint. This limitation will be
    /// addressed in a future update to the checkpoint format.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Index title embeddings (384 dimensions)
    /// storage.enable_vector_index("title_embedding", config_384)?;
    /// // Index body embeddings (1536 dimensions) - different property, OK!
    /// storage.enable_vector_index("body_embedding", config_1536)?;
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn enable_vector_index(&self, property_name: &str, config: HnswConfig) -> Result<()> {
        // Check property limit before attempting to add
        if self.vector_indexes.len() >= DEFAULT_MAX_VECTOR_PROPERTIES {
            return Err(crate::core::error::Error::Vector(
                crate::core::error::VectorError::IndexError(format!(
                    "Maximum number of vector-indexed properties ({}) reached. \
                     Cannot enable index for property '{}'",
                    DEFAULT_MAX_VECTOR_PROPERTIES, property_name
                )),
            ));
        }

        // Use atomic entry() API to avoid TOCTOU race condition
        match self.vector_indexes.entry(property_name.to_string()) {
            Entry::Occupied(_) => {
                return Err(crate::core::error::Error::Vector(
                    crate::core::error::VectorError::IndexError(format!(
                        "Vector index already enabled for property '{}'",
                        property_name
                    )),
                ));
            }
            Entry::Vacant(vacant) => {
                // Create the HNSW index
                let index = HnswIndex::new(config.clone())?;
                let entry = VectorIndexEntry {
                    index: Arc::new(index),
                    config: config.clone(),
                };
                vacant.insert(entry);
            }
        }

        Ok(())
    }

    /// Check if any vector indexing is enabled.
    ///
    /// Returns `true` if at least one property has a vector index.
    pub fn is_vector_index_enabled(&self) -> bool {
        !self.vector_indexes.is_empty()
    }

    /// Check if vector indexing is enabled for a specific property.
    ///
    /// Returns `true` if the property has a vector index configured.
    pub fn is_vector_index_enabled_for(&self, property_name: &str) -> bool {
        self.vector_indexes.contains_key(property_name)
    }

    /// Get the first/default property name that is currently indexed.
    ///
    /// Equivalent to `get_default_vector_property_name` (internal).
    /// For multi-property setups, use [`list_vector_indexes`](Self::list_vector_indexes) instead.
    pub fn get_indexed_property_name(&self) -> Option<String> {
        self.get_default_vector_property_name()
    }

    /// List all configured vector indexes.
    ///
    /// Returns information about each enabled vector index including
    /// property name, dimensions, and distance metric.
    pub fn list_vector_indexes(&self) -> Vec<VectorIndexInfo> {
        self.vector_indexes
            .iter()
            .map(|entry| VectorIndexInfo {
                property_name: entry.key().clone(),
                dimensions: entry.value().config.dimensions,
                distance_metric: entry.value().config.metric,
            })
            .collect()
    }

    /// Check if a vector index is enabled for a specific property.
    ///
    /// Equivalent to [`is_vector_index_enabled_for`](Self::is_vector_index_enabled_for).
    pub fn has_vector_index(&self, property_name: &str) -> bool {
        self.is_vector_index_enabled_for(property_name)
    }

    /// Get the HNSW configuration for a specific property's vector index.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property name to get configuration for
    ///
    /// # Returns
    ///
    /// `Some(HnswConfig)` if a vector index exists for this property, `None` otherwise.
    pub fn get_hnsw_config_for(&self, property_name: &str) -> Option<HnswConfig> {
        self.vector_indexes
            .get(property_name)
            .map(|entry| entry.config.clone())
    }

    /// Register a vector index (used during index loading from disk).
    ///
    /// This directly inserts the index without any initialization logic.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn register_vector_index(
        &self,
        property_name: &str,
        index: crate::index::vector::HnswIndex,
        config: crate::index::vector::HnswConfig,
    ) {
        let index_arc = Arc::new(index);

        // Insert into multi-property map
        self.vector_indexes.insert(
            property_name.to_string(),
            VectorIndexEntry {
                index: Arc::clone(&index_arc),
                config: config.clone(),
            },
        );
    }

    /// Rebuild a vector index from the vector properties of current nodes.
    ///
    /// This is the recovery path for a vector index that was skipped at
    /// startup because its persisted files were corrupted or unreadable
    /// (Issue #451): it enables the index with `config` if it is not
    /// registered, then scans every current node and (re-)indexes the
    /// node's `property_name` vector.
    ///
    /// If an index is already registered for `property_name`, `config` is
    /// ignored and the backfill runs against the existing index (existing
    /// entries for a node are updated in place, so the call is idempotent).
    /// The rebuild covers **current state only** — temporal vector indexes
    /// are not touched.
    ///
    /// Returns the number of node vectors indexed by the backfill.
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be created (e.g. the maximum
    /// number of vector-indexed properties is reached) or if indexing a
    /// node's vector fails (e.g. dimension mismatch with `config`).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn rebuild_vector_index(&self, property_name: &str, config: HnswConfig) -> Result<usize> {
        // Ensure an index is registered for the property, creating one from
        // `config` if absent. A concurrent enable between the check and the
        // call is benign: the index exists afterwards either way, and the
        // backfill proceeds against whichever index won.
        if !self.is_vector_index_enabled_for(property_name) {
            match self.enable_vector_index(property_name, config) {
                Ok(()) => {}
                Err(_) if self.is_vector_index_enabled_for(property_name) => {}
                Err(e) => return Err(e),
            }
        }

        // Clone the Arc so no DashMap shard guard is held while scanning
        // nodes (only the vector index map and node storage reads are
        // involved; no other synchronization primitive is acquired).
        let index = self
            .vector_indexes
            .get(property_name)
            .map(|entry| Arc::clone(&entry.value().index))
            .ok_or_else(|| {
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                    format!(
                        "Vector index for property '{}' disappeared during rebuild",
                        property_name
                    ),
                ))
            })?;

        // Backfill: registering the index BEFORE scanning means nodes created
        // concurrently are indexed by the normal write path; `add` updates
        // an existing entry in place, so double-indexing is harmless.
        let mut indexed = 0usize;
        for node in self.all_nodes() {
            if let Some(vector) = node
                .properties
                .get(property_name)
                .and_then(|v| v.as_vector())
            {
                index.add(node.id, vector)?;
                indexed += 1;
            }
        }
        Ok(indexed)
    }

    /// Get a reference to the HNSW index and its config for a specific property.
    ///
    /// Used for persistence operations. Returns (index, config, vector_count, mappings).
    #[allow(clippy::type_complexity)]
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn get_vector_index_for_persistence(
        &self,
        property_name: &str,
    ) -> Option<(
        Arc<crate::index::vector::HnswIndex>,
        crate::index::vector::HnswConfig,
        usize,
        Vec<(u64, u64)>,
    )> {
        use crate::index::vector::VectorIndex;
        self.vector_indexes.get(property_name).map(|entry| {
            let index = entry.value().index.clone();
            let config = entry.value().config.clone();
            let count = index.len();

            // Extract ID mappings from the index
            let mappings = index.get_id_mappings();

            (index, config, count, mappings)
        })
    }

    /// Try to add a node's vectors to all enabled indexes.
    ///
    /// For each enabled vector index, checks if the node has that property
    /// and adds it to the index if so.
    ///
    /// Returns Ok(true) if any vector was indexed, Ok(false) if none applicable,
    /// Err on failure (will have already done partial work).
    fn try_index_vector(&self, node_id: NodeId, properties: &PropertyMap) -> Result<bool> {
        // DashMap::iter() allocates a guard per shard even when the map is
        // empty; every node write pays that cost otherwise, so short-circuit
        // the overwhelmingly common case of zero vector indexes registered.
        if self.vector_indexes.is_empty() {
            return Ok(false);
        }

        let mut indexed_any = false;

        // Index in all multi-property indexes
        for entry in self.vector_indexes.iter() {
            let prop_name = entry.key();
            if let Some(vector) = properties.get(prop_name).and_then(|v| v.as_vector()) {
                let index = Arc::clone(&entry.value().index);
                index.add(node_id, vector)?;
                indexed_any = true;
            }
        }

        Ok(indexed_any)
    }

    /// Try to remove a node from all vector indexes.
    ///
    /// Returns Ok(true) if removed from any index, Ok(false) if not applicable.
    fn try_remove_from_index(&self, node_id: NodeId) -> Result<bool> {
        if self.vector_indexes.is_empty() {
            return Ok(false);
        }

        let mut removed_any = false;

        // Remove from all multi-property indexes
        for entry in self.vector_indexes.iter() {
            let index = Arc::clone(&entry.value().index);
            // Remove may fail if node wasn't in this index, which is OK
            if index.remove(node_id).is_ok() {
                removed_any = true;
            }
        }

        Ok(removed_any)
    }

    /// Update the vector indexes when node properties change.
    ///
    /// For each enabled vector index, handles add/remove/update based on
    /// whether the property changed.
    fn update_vector_index(
        &self,
        node_id: NodeId,
        new_props: &PropertyMap,
        old_props: &PropertyMap,
    ) -> Result<()> {
        if self.vector_indexes.is_empty() {
            return Ok(());
        }

        // Update all multi-property indexes
        for entry in self.vector_indexes.iter() {
            let prop_name = entry.key();
            let index = Arc::clone(&entry.value().index);

            let old_vec = old_props.get(prop_name).and_then(|v| v.as_vector());
            let new_vec = new_props.get(prop_name).and_then(|v| v.as_vector());

            match (old_vec, new_vec) {
                (None, None) => {} // No change for this property
                (None, Some(v)) => {
                    index.add(node_id, v)?;
                }
                (Some(_), None) => {
                    let _ = index.remove(node_id); // May not exist, ignore error
                }
                (Some(o), Some(n)) => {
                    if o != n {
                        // Note: HnswIndex::add() is an upsert operation (remove + add internally)
                        index.add(node_id, n)?;
                    }
                }
            }
        }

        Ok(())
    }
    /// Create a node with the given label and properties.
    ///
    /// Returns the ID of the newly created node.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn create_node(&self, label: &str, properties: PropertyMap) -> Result<NodeId> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        let node_id = NodeId::new_unchecked(self.node_id_gen.next()?);
        let version_id = VersionId::new_unchecked(self.version_id_gen.next()?);
        let label_interned = GLOBAL_INTERNER.intern(label)?;

        // CRITICAL: Index vector BEFORE inserting node. If vector indexing fails,
        // we have not modified any graph state, so we can safely return error without rollback.
        // This also avoids unnecessary clones of Node and PropertyMap.
        self.try_index_vector(node_id, &properties)?;

        let node = Node::new(node_id, label_interned, properties, version_id);
        self.indexes.insert_node(node);

        Ok(node_id)
    }

    /// Create an edge between two nodes.
    ///
    /// Returns the ID of the newly created edge.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn create_edge(
        &self,
        source: NodeId,
        target: NodeId,
        label: &str,
        properties: PropertyMap,
    ) -> Result<EdgeId> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        // Verify nodes exist
        if !self.indexes.contains_node(source) {
            return Err(StorageError::NodeNotFound(source).into());
        }
        if !self.indexes.contains_node(target) {
            return Err(StorageError::NodeNotFound(target).into());
        }

        let edge_id = EdgeId::new_unchecked(self.edge_id_gen.next()?);
        let version_id = VersionId::new_unchecked(self.version_id_gen.next()?);
        let label_interned = GLOBAL_INTERNER.intern(label)?;

        let edge = Edge::new(
            edge_id,
            label_interned,
            source,
            target,
            properties,
            version_id,
        );
        self.indexes.insert_edge(edge);

        // Adjacency indexes are now rebuilt lazily on first access (see CurrentIndexes)
        // This eliminates O(n² log n) performance regression for batch operations

        Ok(edge_id)
    }

    /// Get a node by ID.
    ///
    /// On a replica this read is atomic with respect to the applier's batch
    /// replay: it never observes a half-applied replicated transaction
    /// (Issue #3788). On a primary the gate is disarmed and costs a branch.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn get_node(&self, id: NodeId) -> Result<Node> {
        self.apply_gate.read(|| {
            self.indexes
                .get_node(id)
                .ok_or_else(|| StorageError::NodeNotFound(id).into())
        })
    }

    /// Access a node without cloning, executing a closure on the node data.
    ///
    /// This method provides zero-copy read access to node data for hot paths
    /// where only specific fields are needed.
    ///
    /// # Performance
    ///
    /// - **No allocation**: Does not clone the Node
    /// - **No Arc increment**: Does not increment PropertyMap reference count (unless cloned in closure)
    /// - **Lock duration**: Holds DashMap read lock only during closure execution
    ///
    /// # Safety & Deadlocks
    ///
    /// **WARNING**: The closure is executed while holding a read lock on the node shard.
    /// Do NOT attempt to modify the graph or perform operations that might acquire a
    /// write lock on the same shard (e.g., `update_node`, `delete_node`) within the closure.
    /// Doing so will cause a deadlock (lock re-entrancy hazard).
    #[inline]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn with_node<F, R>(&self, id: NodeId, f: F) -> Result<R>
    where
        F: FnOnce(&Node) -> R,
    {
        // `FnOnce` cannot be re-run, so on a replica this takes the apply
        // gate's blocking fallback rather than its seqlock fast path
        // (Issue #3788). Disarmed (primary) it is a branch, as before.
        self.apply_gate.read_once(|| {
            self.indexes
                .with_node(id, f)
                .ok_or_else(|| StorageError::NodeNotFound(id).into())
        })
    }

    /// Get an edge by ID.
    ///
    /// Atomic with respect to a replica apply -- see [`Self::get_node`].
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn get_edge(&self, id: EdgeId) -> Result<Edge> {
        self.apply_gate.read(|| {
            self.indexes
                .get_edge(id)
                .ok_or_else(|| StorageError::EdgeNotFound(id).into())
        })
    }

    // ========================================================================
    // Zero-copy access methods (Issue #190)
    //
    // These methods provide efficient access to node/edge data without cloning
    // the entire structure. Use these for hot paths where you only need to
    // read specific fields.
    // ========================================================================

    /// Get the target node of an edge without cloning the entire edge.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns the target NodeId (8 bytes)
    /// - **No allocation**: Does not clone Edge or PropertyMap
    #[inline]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn get_edge_target(&self, id: EdgeId) -> Result<NodeId> {
        self.apply_gate.read(|| {
            self.indexes
                .get_edge_target(id)
                .ok_or_else(|| StorageError::EdgeNotFound(id).into())
        })
    }

    /// Get the source node of an edge without cloning the entire edge.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns the source NodeId (8 bytes)
    /// - **No allocation**: Does not clone Edge or PropertyMap
    #[inline]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn get_edge_source(&self, id: EdgeId) -> Result<NodeId> {
        self.apply_gate.read(|| {
            self.indexes
                .get_edge_source(id)
                .ok_or_else(|| StorageError::EdgeNotFound(id).into())
        })
    }

    /// Get the endpoints (source, target) of an edge without cloning.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns two NodeIds (16 bytes)
    /// - **No allocation**: Does not clone Edge or PropertyMap
    #[inline]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn get_edge_endpoints(&self, id: EdgeId) -> Result<(NodeId, NodeId)> {
        self.apply_gate.read(|| {
            self.indexes
                .get_edge_endpoints(id)
                .ok_or_else(|| StorageError::EdgeNotFound(id).into())
        })
    }

    /// Get the label of an edge without cloning the entire edge.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns the label (8 bytes)
    /// - **No allocation**: Does not clone Edge or PropertyMap
    #[inline]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn get_edge_label(&self, id: EdgeId) -> Result<InternedString> {
        self.apply_gate.read(|| {
            self.indexes
                .get_edge_label(id)
                .ok_or_else(|| StorageError::EdgeNotFound(id).into())
        })
    }

    /// Get the label of a node without cloning the entire node.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Only reads and returns the label (8 bytes)
    /// - **No allocation**: Does not clone Node or PropertyMap
    #[inline]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn get_node_label(&self, id: NodeId) -> Result<InternedString> {
        self.apply_gate.read(|| {
            self.indexes
                .get_node_label(id)
                .ok_or_else(|| StorageError::NodeNotFound(id).into())
        })
    }

    /// Delete a node.
    ///
    /// # Important
    ///
    /// This method does NOT delete edges connected to the node. This may leave
    /// orphaned edges in the graph. For most use cases, prefer using
    /// [`crate::api::transaction::WriteOps::delete_node_cascade`] which automatically removes
    /// all connected edges to maintain referential integrity.
    ///
    /// Only use this method if you explicitly need to preserve edges for some
    /// specialized use case (e.g., maintaining edge history for audit purposes).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn delete_node(&self, id: NodeId) -> Result<Node> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        let node = self
            .indexes
            .remove_node(id)
            .ok_or(StorageError::NodeNotFound(id))?;

        // Best-effort vector index removal (ignore errors) - Issue #323
        // This prevents memory leaks and ensures deleted nodes don't appear in similarity searches
        // Note: Temporal vector index cleanup is only available in delete_node_direct()
        // which has timestamp context from the transaction. Non-transactional deletes
        // via this method don't have temporal semantics.
        let _ = self.try_remove_from_index(id);

        Ok(node)
    }

    /// Delete an edge.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn delete_edge(&self, id: EdgeId) -> Result<Edge> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        let edge = self
            .indexes
            .remove_edge(id)
            .ok_or(StorageError::EdgeNotFound(id))?;

        // Adjacency indexes are now rebuilt lazily on first access (see CurrentIndexes)
        // This eliminates O(n² log n) performance regression for batch operations

        Ok(edge)
    }

    // Direct insert/update/delete methods for transaction commit
    // These methods are used by WriteTransaction to apply buffered changes

    /// Insert a node directly (used by WriteTransaction).
    /// Does not generate IDs - caller must provide them.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn insert_node_direct(&self, node: Node, timestamp: Timestamp) -> Result<()> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        // CRITICAL: Index vectors BEFORE inserting the node so an indexing
        // failure never leaves a node visible in the graph but missing from an
        // index. This prevents the VS-030 bug where transaction-created nodes
        // bypassed indexing, causing them to be missing from the HNSW index and
        // invisible to find_similar queries.
        // NOTE: with multiple vector/temporal indexes enabled, a mid-fan-out
        // failure returns Err with indexes visited earlier already updated
        // (partial work); the node itself is never inserted into the main
        // indexes.
        self.try_index_vector(node.id, &node.properties)?;

        // Index in every enabled temporal vector index
        self.try_index_temporal_vector(node.id, &node.properties, timestamp)?;

        // Vector indexing succeeded, now insert the node into the main indexes.
        self.indexes.insert_node(node);

        Ok(())
    }

    /// Insert an edge directly (used by WriteTransaction).
    /// Does not generate IDs or rebuild adjacency - caller must handle.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn insert_edge_direct(&self, edge: Edge) -> Result<()> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        self.indexes.insert_edge(edge);
        Ok(())
    }

    /// Update a node directly (used by WriteTransaction).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn update_node_direct(&self, node: Node, timestamp: Timestamp) -> Result<()> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        // Save old node for vector index update
        let old_props = self.indexes.with_node(node.id, |n| n.properties.clone());

        // Update vector indexes BEFORE updating the node in the main indexes,
        // so the node's visible state never gets ahead of the indexes.
        // NOTE: with multiple vector/temporal indexes enabled, a mid-fan-out
        // failure returns Err with indexes visited earlier already updated
        // (partial work); the node update itself is not applied.
        // Update every enabled current-state AND temporal vector index using the
        // old properties, so a removed/overwritten embedding is de-indexed (not
        // just re-added) on both. Without the old props the node is not yet in
        // current storage — e.g. WAL replay applying an `UpdateNode` against a
        // lagging index snapshot that never saw the node's create (recovery.rs
        // applies these idempotently to converge). In that case there is no
        // prior vector to remove, so we ADD to both the current-state HNSW and
        // the temporal index, exactly mirroring `insert_node_direct`; indexing
        // only the temporal side would leave the vector invisible to
        // current-state `find_similar`.
        if let Some(old_p) = old_props {
            self.update_vector_index(node.id, &node.properties, &old_p)?;
            self.update_temporal_vector_index(node.id, &node.properties, &old_p, timestamp)?;
        } else {
            self.try_index_vector(node.id, &node.properties)?;
            self.try_index_temporal_vector(node.id, &node.properties, timestamp)?;
        }

        // Finally, insert node into main indexes. This avoids node.clone().
        self.indexes.insert_node(node);

        Ok(())
    }

    /// Finalize the commit timestamp on a stored node after successful `apply_changes`.
    ///
    /// During `apply_changes`, nodes are written with `commit_timestamp: None` (pending).
    /// This method is called after the full apply succeeds to atomically mark the node
    /// as committed, making it visible to future snapshot readers.
    pub fn set_node_commit_timestamp(&self, node_id: NodeId, ts: Timestamp) {
        self.indexes
            .with_node_mut(node_id, |n| n.metadata.commit_timestamp = Some(ts));
    }

    /// Finalize the commit timestamp on a stored edge after successful `apply_changes`.
    pub fn set_edge_commit_timestamp(&self, edge_id: EdgeId, ts: Timestamp) {
        self.indexes
            .with_edge_mut(edge_id, |e| e.metadata.commit_timestamp = Some(ts));
    }

    /// Update an edge directly (used by WriteTransaction).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn update_edge_direct(&self, edge: Edge) -> Result<()> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        // Remove old version and insert new
        self.indexes.insert_edge(edge);
        Ok(())
    }

    /// Delete a node directly (used by WriteTransaction).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn delete_node_direct(&self, id: NodeId, timestamp: Timestamp) -> Result<()> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        self.indexes
            .remove_node(id)
            .ok_or(StorageError::NodeNotFound(id))?;

        // Best-effort vector index removal (ignore errors)
        let _ = self.try_remove_from_index(id);

        // Remove from temporal vector index if enabled
        let _ = self.try_remove_temporal_vector(id, timestamp);

        Ok(())
    }

    /// Delete an edge directly (used by WriteTransaction).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn delete_edge_direct(&self, id: EdgeId) -> Result<()> {
        // Synchronize with snapshot creation
        let _lock = self.snapshot_lock.read();
        self.indexes
            .remove_edge(id)
            .ok_or(StorageError::EdgeNotFound(id))?;
        Ok(())
    }

    /// Rebuild adjacency indexes from current edges.
    ///
    /// This should be called after batch edge operations to update the
    /// adjacency indexes for efficient graph traversal.
    ///
    /// # Concurrency Safety
    ///
    /// This method is safe to call concurrently with read operations:
    /// - Uses `RwLock` on adjacency indexes for safe concurrent access
    /// - Readers can continue traversing old indexes while rebuild occurs
    /// - New index is swapped in atomically when rebuild completes
    /// - No stale reads: readers either see old (consistent) or new (consistent) state
    ///
    /// However, concurrent writes should be serialized at a higher level
    /// (e.g., through transaction isolation) to prevent race conditions.
    ///
    /// Compact adjacency indexes to merge delta buffer into frozen CSR.
    ///
    /// With incremental adjacency, edges are immediately visible without compaction.
    /// This method improves read performance by moving delta edges to frozen CSR.
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
    /// Complexity: O(D log D) where D is the number of delta edges.
    /// Typically much faster than the old O(E log E) rebuild where E is total edges.
    pub fn compact_adjacency(&self) {
        self.indexes.compact_adjacency();
    }

    /// Layer occupancy of both adjacency indexes (Issue #3810).
    ///
    /// `is_fully_compacted()` is true exactly when adjacency reads take the
    /// frozen-CSR fast path.
    pub fn adjacency_stats(&self) -> AdjacencyIndexStats {
        self.indexes.adjacency_stats()
    }

    /// Get the current number of delta edges (test-only).
    ///
    /// Delta edges are insertions not yet compacted into frozen CSR.
    #[cfg(test)]
    pub(crate) fn delta_edge_count(&self) -> usize {
        self.indexes.delta_edge_count()
    }

    /// Rebuild adjacency indexes (deprecated - use `compact_adjacency` instead).
    ///
    /// This method is kept for backward compatibility and simply calls `compact_adjacency`.
    /// Edges are always immediately visible without calling this method.
    #[deprecated(since = "0.1.0", note = "Use compact_adjacency() instead")]
    pub fn rebuild_adjacency(&self) {
        self.compact_adjacency();
    }

    /// Get a frozen view for outgoing adjacency (read transaction hot path).
    ///
    /// Returns `Some(FrozenAdjacencyView)` if the index is in a clean state
    /// (no pending delta edges or tombstones), allowing direct slice access.
    /// Returns `None` if there are pending delta operations.
    ///
    /// # Performance
    ///
    /// When available, frozen view provides ~8-14ns access vs ~16-17ns for
    /// the merged path. Use this for read-only workloads.
    #[inline]
    pub fn frozen_outgoing_view(
        &self,
    ) -> Option<crate::index::incremental_adjacency::FrozenAdjacencyView> {
        self.indexes.frozen_outgoing_view()
    }

    /// Get a frozen view for incoming adjacency (read transaction hot path).
    ///
    /// Same as `frozen_outgoing_view()` but for incoming edges.
    #[inline]
    pub fn frozen_incoming_view(
        &self,
    ) -> Option<crate::index::incremental_adjacency::FrozenAdjacencyView> {
        self.indexes.frozen_incoming_view()
    }

    /// Get all outgoing edges from a node.
    ///
    /// This is the critical "hot path" operation that must be fast.
    /// Uses frozen view (~8-14ns) when available, falls back to merged guard (~16-17ns).
    #[inline]
    pub fn get_outgoing_edges(&self, source: NodeId) -> Vec<EdgeId> {
        self.apply_gate.read(|| {
            // HOT PATH: Use frozen view for direct slice access when no delta/tombstones
            if let Some(frozen) = self.indexes.frozen_outgoing_view() {
                return frozen
                    .get_adjacency(source)
                    .iter()
                    .map(|entry| entry.edge_id)
                    .collect();
            }

            // SLOW PATH: Use merged guard when delta/tombstones exist.
            // Pre-allocate using capacity hint to avoid multiple reallocations.
            let guard = self.indexes.get_outgoing(source);
            let mut result = Vec::with_capacity(guard.capacity_hint());
            result.extend(guard.iter().map(|entry| entry.edge_id));
            result
        })
    }

    /// Get all incoming edges to a node.
    ///
    /// Uses frozen view (~8-14ns) when available, falls back to merged guard (~16-17ns).
    #[inline]
    pub fn get_incoming_edges(&self, target: NodeId) -> Vec<EdgeId> {
        self.apply_gate.read(|| {
            // HOT PATH: Use frozen view for direct slice access when no delta/tombstones
            if let Some(frozen) = self.indexes.frozen_incoming_view() {
                return frozen
                    .get_adjacency(target)
                    .iter()
                    .map(|entry| entry.edge_id)
                    .collect();
            }

            // SLOW PATH: Use merged guard when delta/tombstones exist.
            // Pre-allocate using capacity hint to avoid multiple reallocations.
            let guard = self.indexes.get_incoming(target);
            let mut result = Vec::with_capacity(guard.capacity_hint());
            result.extend(guard.iter().map(|entry| entry.edge_id));
            result
        })
    }

    /// Get outgoing edges with a specific label.
    #[inline]
    pub fn get_outgoing_edges_with_label(&self, source: NodeId, label: &str) -> Vec<EdgeId> {
        let label_id = match GLOBAL_INTERNER.get_id(label) {
            Some(id) => id,
            None => return Vec::new(), // Label doesn't exist
        };

        // Optimized to avoid intermediate Vec allocation and pre-allocate result
        self.apply_gate.read(|| {
            let guard = self.indexes.get_outgoing(source);
            let mut result = Vec::with_capacity(guard.capacity_hint());
            result.extend(
                guard
                    .iter()
                    .filter(|entry| entry.label == label_id)
                    .map(|entry| entry.edge_id),
            );
            result
        })
    }

    /// Get incoming edges with a specific label.
    #[inline]
    pub fn get_incoming_edges_with_label(&self, target: NodeId, label: &str) -> Vec<EdgeId> {
        let label_id = match GLOBAL_INTERNER.get_id(label) {
            Some(id) => id,
            None => return Vec::new(),
        };

        // Optimized to avoid intermediate Vec allocation and pre-allocate result
        self.apply_gate.read(|| {
            let guard = self.indexes.get_incoming(target);
            let mut result = Vec::with_capacity(guard.capacity_hint());
            result.extend(
                guard
                    .iter()
                    .filter(|entry| entry.label == label_id)
                    .map(|entry| entry.edge_id),
            );
            result
        })
    }

    /// Get all outgoing edges from a node as an iterator.
    ///
    /// This is a zero-allocation alternative to [`Self::get_outgoing_edges`] that returns
    /// an iterator instead of collecting into a `Vec`. Use this for performance-critical
    /// traversals where you don't need to store all edges.
    ///
    /// # Performance
    ///
    /// - No `Vec` allocation (saves 100-500ns per call)
    /// - Lazy evaluation - only computes what you consume
    /// - Can be short-circuited with `.take()`, `.find()`, etc.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Get first outgoing edge without allocating a Vec
    /// let first_edge = storage.get_outgoing_edges_iter(node).next();
    ///
    /// // Count edges without allocation
    /// let count = storage.get_outgoing_edges_iter(node).count();
    /// ```
    #[inline]
    pub fn get_outgoing_edges_iter(&self, source: NodeId) -> OutgoingEdgesIter<'_> {
        OutgoingEdgesIter::new(self.indexes.get_outgoing(source))
    }

    /// Get all incoming edges to a node as an iterator.
    ///
    /// This is a zero-allocation alternative to [`Self::get_incoming_edges`] that returns
    /// an iterator instead of collecting into a `Vec`.
    ///
    /// See [`Self::get_outgoing_edges_iter`] for performance details and usage examples.
    #[inline]
    pub fn get_incoming_edges_iter(&self, target: NodeId) -> IncomingEdgesIter<'_> {
        IncomingEdgesIter::new(self.indexes.get_incoming(target))
    }

    /// Get outgoing edges with a specific label as an iterator.
    ///
    /// This is a zero-allocation alternative to [`Self::get_outgoing_edges_with_label`].
    ///
    /// Returns an empty iterator if the label doesn't exist.
    #[inline]
    pub fn get_outgoing_edges_with_label_iter(
        &self,
        source: NodeId,
        label: &str,
    ) -> OutgoingEdgesWithLabelIter<'_> {
        let label_id = GLOBAL_INTERNER.get_id(label);
        OutgoingEdgesWithLabelIter::new(self.indexes.get_outgoing(source), label_id)
    }

    /// Get incoming edges with a specific label as an iterator.
    ///
    /// This is a zero-allocation alternative to [`Self::get_incoming_edges_with_label`].
    ///
    /// Returns an empty iterator if the label doesn't exist.
    #[inline]
    pub fn get_incoming_edges_with_label_iter(
        &self,
        target: NodeId,
        label: &str,
    ) -> IncomingEdgesWithLabelIter<'_> {
        let label_id = GLOBAL_INTERNER.get_id(label);
        IncomingEdgesWithLabelIter::new(self.indexes.get_incoming(target), label_id)
    }

    /// Export outgoing CSR adjacency data for persistence.
    pub fn export_outgoing_csr(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        self.indexes.export_outgoing_csr()
    }

    /// Export incoming CSR adjacency data for persistence.
    pub fn export_incoming_csr(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        self.indexes.export_incoming_csr()
    }

    /// Export both directions' frozen CSRs as one compaction-consistent pair
    /// (Issue #3810). See
    /// [`CurrentIndexes::export_csr_pair`](crate::index::CurrentIndexes::export_csr_pair).
    pub fn export_csr_pair(&self) -> ExportedCsrPair {
        self.indexes.export_csr_pair()
    }

    /// Import CSR adjacency data from persistence.
    ///
    /// This bypasses the need to rebuild adjacency structures from scratch.
    pub fn import_csr(
        &self,
        outgoing_node_ids: Vec<u64>,
        outgoing_offsets: Vec<u64>,
        outgoing_edge_ids: Vec<u64>,
        incoming_node_ids: Vec<u64>,
        incoming_offsets: Vec<u64>,
        incoming_edge_ids: Vec<u64>,
    ) {
        self.indexes.import_csr(
            outgoing_node_ids,
            outgoing_offsets,
            outgoing_edge_ids,
            incoming_node_ids,
            incoming_offsets,
            incoming_edge_ids,
        );
    }

    /// Get the number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.indexes.node_count()
    }

    /// Node ids currently in `namespace`, from the secondary membership index
    /// (Issue #3349). O(members), no property scan.
    #[inline]
    pub fn namespace_node_ids(&self, namespace: &crate::core::namespace::Namespace) -> Vec<NodeId> {
        self.indexes.namespace_node_ids(namespace)
    }

    /// Edge ids currently in `namespace`, from the secondary membership index.
    #[inline]
    pub fn namespace_edge_ids(&self, namespace: &crate::core::namespace::Namespace) -> Vec<EdgeId> {
        self.indexes.namespace_edge_ids(namespace)
    }

    /// Whether `node_id` currently lives in `namespace` (Issue #3349, PR2). O(1)
    /// membership probe — no property load — for scoped-traversal boundary checks.
    #[inline]
    pub fn node_in_namespace(
        &self,
        node_id: NodeId,
        namespace: &crate::core::namespace::Namespace,
    ) -> bool {
        self.indexes.node_in_namespace(node_id, namespace)
    }

    /// Whether `node_id` currently lives in the interned namespace `ns_id`
    /// (Issue #3349, PR2 performance). The pre-interned hot-path form of
    /// [`node_in_namespace`](Self::node_in_namespace) — no per-call string
    /// hashing.
    #[inline]
    pub fn node_in_namespace_id(
        &self,
        node_id: NodeId,
        ns_id: crate::core::namespace::NamespaceId,
    ) -> bool {
        self.indexes.node_in_namespace_id(node_id, ns_id)
    }

    /// Whether `edge_id` currently lives in `namespace` (Issue #3349, PR2). O(1)
    /// membership probe — no property load. See
    /// [`node_in_namespace`](Self::node_in_namespace).
    #[inline]
    pub fn edge_in_namespace(
        &self,
        edge_id: EdgeId,
        namespace: &crate::core::namespace::Namespace,
    ) -> bool {
        self.indexes.edge_in_namespace(edge_id, namespace)
    }

    /// Whether `edge_id` currently lives in the interned namespace `ns_id`
    /// (Issue #3349, PR2 performance). The pre-interned hot-path form of
    /// [`edge_in_namespace`](Self::edge_in_namespace).
    #[inline]
    pub fn edge_in_namespace_id(
        &self,
        edge_id: EdgeId,
        ns_id: crate::core::namespace::NamespaceId,
    ) -> bool {
        self.indexes.edge_in_namespace_id(edge_id, ns_id)
    }

    /// Number of nodes currently in `namespace` (O(1) membership-set read).
    #[inline]
    pub fn namespace_node_count(&self, namespace: &crate::core::namespace::Namespace) -> usize {
        self.indexes.namespace_node_count(namespace)
    }

    /// Number of edges currently in `namespace` (O(1) membership-set read).
    #[inline]
    pub fn namespace_edge_count(&self, namespace: &crate::core::namespace::Namespace) -> usize {
        self.indexes.namespace_edge_count(namespace)
    }

    /// Every namespace currently holding at least one node or edge.
    #[inline]
    pub fn populated_namespaces(
        &self,
    ) -> std::collections::BTreeSet<crate::core::namespace::Namespace> {
        self.indexes.populated_namespaces()
    }

    /// Get an exclusive upper bound on node ids present in current storage.
    ///
    /// Used by the query executor's full-scan iterator to bound a lazy id sweep
    /// over `[0, max_id)` without materializing the id set.
    ///
    /// This reads the index's insert-maintained high-water-mark
    /// ([`CurrentIndexes::max_node_id_exclusive`]), **not** this storage's own
    /// `node_id_gen`. In the transactional write path node ids are allocated by
    /// a database-level generator and applied here via
    /// [`Self::insert_node_direct`], so the storage-local `node_id_gen` is never
    /// advanced and would report `0` -- collapsing every full scan to zero rows.
    /// The index high-water-mark is advanced by every insert regardless of where
    /// the id originated, so it is the correct scan bound. It is a relaxed,
    /// monotonically non-decreasing read and must not be used for
    /// snapshot-isolation decisions.
    #[inline]
    pub fn get_max_node_id(&self) -> u64 {
        self.indexes.max_node_id_exclusive()
    }

    /// Fast-path label check using the compact node header.
    ///
    /// Returns `true` iff a live node with `node_id` exists and its interned
    /// label equals `label_id`, reading only the 16-byte header instead of
    /// loading and cloning the full node. Returns `false` for a missing/invalid
    /// id, so callers can use it to skip gaps in a sparse id space.
    #[inline]
    pub fn node_has_label(
        &self,
        node_id: u64,
        label_id: crate::core::interning::InternedString,
    ) -> bool {
        match NodeId::new(node_id) {
            Ok(id) => self.apply_gate.read(|| {
                self.indexes
                    .get_node_header(id)
                    .is_some_and(|header| header.label == label_id)
            }),
            Err(_) => false,
        }
    }

    /// Fast-path existence check using the compact node header.
    ///
    /// Returns `true` iff a live node with `node_id` exists, reading only the
    /// 16-byte header instead of loading and cloning the full node. Returns
    /// `false` for a missing/invalid id, so callers can use it to cheaply skip
    /// gaps in a sparse id space. This mirrors [`Self::node_has_label`] but
    /// omits the label comparison. Acquires only a brief DashMap shard lock,
    /// which is released before returning.
    #[inline]
    pub fn contains_node(&self, node_id: u64) -> bool {
        match NodeId::new(node_id) {
            Ok(id) => self
                .apply_gate
                .read(|| self.indexes.get_node_header(id).is_some()),
            Err(_) => false,
        }
    }

    /// Get the number of edges.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.indexes.edge_count()
    }

    /// Get the out-degree of a node.
    #[inline]
    pub fn out_degree(&self, node: NodeId) -> usize {
        self.apply_gate.read(|| self.indexes.out_degree(node))
    }

    /// Get the in-degree of a node.
    #[inline]
    pub fn in_degree(&self, node: NodeId) -> usize {
        self.apply_gate.read(|| self.indexes.in_degree(node))
    }

    /// Get the default property name for vector indexing.
    ///
    /// This is used to maintain backward compatibility for legacy APIs that
    /// don't specify a property name. It deterministically selects the
    /// alphabetically first property name from the enabled vector indexes.
    ///
    /// Returns `None` if no vector indexes are enabled.
    fn get_default_vector_property_name(&self) -> Option<String> {
        if self.vector_indexes.is_empty() {
            return None;
        }

        // Optimization: if len == 1, just take it (no sorting needed)
        if self.vector_indexes.len() == 1 {
            return self.vector_indexes.iter().next().map(|r| r.key().clone());
        }

        // Find min key alphabetically
        // Note: DashMap iteration order is not guaranteed, so we must scan all keys
        // We use `.fold` instead of `.min_by` to prevent deadlocks. `.min_by` holds
        // multiple `RefMulti` read guards across iteration steps, which can block
        // concurrent writers indefinitely. `.fold` immediately drops the guard for
        // each element after extracting the necessary data.
        self.vector_indexes
            .iter()
            .fold(None, |min: Option<String>, current| {
                let key = current.key();
                match min {
                    Some(m) if m.as_str() <= key.as_str() => Some(m),
                    _ => Some(key.clone()),
                }
            })
    }

    /// Get or create filter statistics for a label (Issue #334).
    ///
    /// Returns an Arc to the FilterStats for the given label, creating it if needed.
    /// This enables adaptive over-fetch multiplier calculation based on historical
    /// filter pass rates.
    fn get_or_create_filter_stats(&self, label: &str) -> Arc<FilterStats> {
        self.filter_stats
            .entry(label.to_string())
            .or_insert_with(|| Arc::new(FilterStats::new()))
            .value()
            .clone()
    }

    /// Calculate adaptive over-fetch candidates for filtered search (Issue #334).
    ///
    /// Returns the number of candidates to fetch and the FilterStats for recording results.
    /// This method centralizes the adaptive over-fetch logic used by all filtered search methods.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of results requested by the user
    /// * `label` - Label to filter by
    ///
    /// # Returns
    ///
    /// Tuple of (candidates_to_fetch, stats) where:
    /// - `candidates_to_fetch` is the adaptive number of candidates to retrieve from HNSW
    /// - `stats` is the FilterStats for recording search results
    ///
    /// # Algorithm
    ///
    /// - Uses adaptive multiplier based on historical pass rate (5x to 50x)
    /// - Guarantees minimum of k + 20 candidates
    /// - Caps maximum at k + 1000 to prevent excessive memory usage
    fn calculate_adaptive_candidates(&self, k: usize, label: &str) -> (usize, Arc<FilterStats>) {
        const MIN_ABSOLUTE_OVERFETCH: usize = 20;
        const MAX_ABSOLUTE_OVERFETCH: usize = 1000;

        let stats = self.get_or_create_filter_stats(label);
        let multiplier = stats.get_adaptive_multiplier();
        let candidates = ((k as f64 * multiplier) as usize)
            .max(k + MIN_ABSOLUTE_OVERFETCH)
            .min(k + MAX_ABSOLUTE_OVERFETCH);

        (candidates, stats)
    }

    /// Get filter statistics for a label (test-only helper).
    ///
    /// Returns the current statistics (search_count, total_candidates, total_results)
    /// for the given label, or None if no searches have been performed yet.
    ///
    /// This is used for testing to verify that adaptive learning is working correctly.
    pub(crate) fn get_filter_stats(&self, label: &str) -> Option<(u64, u64, u64)> {
        use std::sync::atomic::Ordering;

        self.filter_stats.get(label).map(|entry| {
            let stats = entry.value();
            (
                stats.search_count.load(Ordering::Relaxed),
                stats.total_candidates.load(Ordering::Relaxed),
                stats.total_results.load(Ordering::Relaxed),
            )
        })
    }

    /// Find k most similar nodes to the query node based on vector similarity.
    ///
    /// Returns a list of (NodeId, score) pairs sorted by similarity (highest first).
    /// The query node itself is excluded from results.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vector index is not enabled
    /// - Query node is not found
    /// - Query node does not have the indexed vector property
    /// - The property is not a vector
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar(&self, query_node_id: NodeId, k: usize) -> Result<Vec<(NodeId, f32)>> {
        let (prop_name, index, _) = self.get_vector_index_internal(None)?;
        let query_vector = self.get_node_vector(query_node_id, &prop_name)?;
        self.run_vector_search(&index, &query_vector, k, None, Some(query_node_id))
    }

    /// Find k most similar nodes with a specific label.
    ///
    /// Returns a list of (NodeId, score) pairs sorted by similarity (highest first).
    /// Only nodes with the specified label are returned. The query node is excluded.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_with_label(
        &self,
        query_node_id: NodeId,
        label: &str,
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        let (prop_name, index, _) = self.get_vector_index_internal(None)?;
        let query_vector = self.get_node_vector(query_node_id, &prop_name)?;
        self.run_vector_search(&index, &query_vector, k, Some(label), Some(query_node_id))
    }

    /// Find k most similar nodes to a raw embedding vector.
    ///
    /// This is useful when searching with embeddings that don't correspond to any
    /// existing node in the graph (e.g., query embeddings from external sources).
    ///
    /// # Arguments
    ///
    /// * `embedding` - The query embedding vector
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A list of (NodeId, similarity_score) pairs sorted by similarity (highest first).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vector index is not enabled
    /// - Embedding dimensions don't match the indexed property
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_by_embedding(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        let (_, index, config) = self.get_vector_index_internal(None)?;
        Self::validate_embedding_dimensions(embedding, &config)?;
        self.run_vector_search(&index, embedding, k, None, None)
    }

    /// Find k most similar nodes with a specific label to a raw embedding vector.
    ///
    /// Like `find_similar_by_embedding()`, but filters results to only include
    /// nodes with the specified label.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The query embedding vector
    /// * `label` - Only return nodes with this label
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A list of (NodeId, similarity_score) pairs sorted by similarity (highest first).
    /// All results have the specified label.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vector index is not enabled
    /// - Embedding dimensions don't match the indexed property
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_by_embedding_with_label(
        &self,
        embedding: &[f32],
        label: &str,
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        let (_, index, config) = self.get_vector_index_internal(None)?;
        Self::validate_embedding_dimensions(embedding, &config)?;
        self.run_vector_search(&index, embedding, k, Some(label), None)
    }

    /// Find k most similar nodes to a raw embedding in a specific property's index.
    ///
    /// This is the property-specific version of `find_similar_by_embedding()`.
    /// Use this when you have multiple vector indexes and need to search a specific one.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property with the vector index to search
    /// * `embedding` - The query embedding vector
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A list of (NodeId, similarity_score) pairs sorted by similarity (highest first).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No vector index is enabled for the specified property
    /// - Embedding dimensions don't match the indexed property
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_by_embedding_in(
        &self,
        property_name: &str,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        let (_, index, config) = self.get_vector_index_internal(Some(property_name))?;
        Self::validate_embedding_dimensions(embedding, &config)?;
        self.run_vector_search(&index, embedding, k, None, None)
    }

    /// Find k most similar nodes with a label to a raw embedding in a specific property's index.
    ///
    /// This is the property-specific version of `find_similar_by_embedding_with_label()`.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property with the vector index to search
    /// * `embedding` - The query embedding vector
    /// * `label` - Only return nodes with this label
    /// * `k` - Maximum number of results to return
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_by_embedding_in_with_label(
        &self,
        property_name: &str,
        embedding: &[f32],
        label: &str,
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        let (_, index, config) = self.get_vector_index_internal(Some(property_name))?;
        Self::validate_embedding_dimensions(embedding, &config)?;
        self.run_vector_search(&index, embedding, k, Some(label), None)
    }

    /// Find k most similar nodes with a custom predicate.
    ///
    /// This allows filtering candidates based on arbitrary criteria (e.g., property values).
    /// The predicate is called for each candidate node. If it returns true, the node is included.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property with the vector index to search
    /// * `query` - The query embedding vector
    /// * `k` - Maximum number of results to return
    /// * `predicate` - A closure that takes a `NodeId` and returns `true` if the node should be included
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_with_predicate<F>(
        &self,
        property_name: &str,
        query: &[f32],
        k: usize,
        predicate: F,
    ) -> Result<Vec<(NodeId, f32)>>
    where
        F: Fn(&NodeId) -> bool + Send + Sync,
    {
        let (_, index, config) = self.get_vector_index_internal(Some(property_name))?;
        Self::validate_embedding_dimensions(query, &config)?;
        index.search_with_filter(query, k, predicate)
    }

    /// Filter-complete k-NN from an existing node's embedding on the **default**
    /// vector index (Issue #3349, PR2 / test T6).
    ///
    /// Resolves the default vector property and the query node's embedding, then
    /// delegates to the index's [`search_with_filter`](crate::index::VectorIndex::search_with_filter),
    /// which **over-fetches** candidates until it has `k` results satisfying
    /// `predicate` (or the index is exhausted) — so a namespace-scoped caller
    /// never receives fewer than `k` in-scope results when that many exist. The
    /// query node itself is always excluded (folded into the predicate), so
    /// callers pass only their scope predicate.
    ///
    /// The predicate is deliberately **ignorant of why a vector is absent** from
    /// the index: it is called only for candidates the index actually returns, so
    /// a vector excluded from the index for any reason (e.g. a crypto-shredded
    /// embedding) simply never reaches the predicate — this method makes no
    /// assumption about vector presence.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_by_node_with_predicate<F>(
        &self,
        query_node_id: NodeId,
        k: usize,
        predicate: F,
    ) -> Result<Vec<(NodeId, f32)>>
    where
        F: Fn(&NodeId) -> bool + Send + Sync,
    {
        let (prop_name, index, _) = self.get_vector_index_internal(None)?;
        let query_vector = self.get_node_vector(query_node_id, &prop_name)?;
        index.search_with_filter(&query_vector, k, move |id| {
            *id != query_node_id && predicate(id)
        })
    }

    /// Filter-complete k-NN from a raw embedding on the **default** vector index
    /// (Issue #3349, PR2 / test T6). Like
    /// [`find_similar_by_node_with_predicate`](Self::find_similar_by_node_with_predicate)
    /// but the query is an externally-supplied embedding (no node exclusion). See
    /// that method for the filter-completeness and absent-vector guarantees.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_by_embedding_with_predicate<F>(
        &self,
        embedding: &[f32],
        k: usize,
        predicate: F,
    ) -> Result<Vec<(NodeId, f32)>>
    where
        F: Fn(&NodeId) -> bool + Send + Sync,
    {
        let (_, index, config) = self.get_vector_index_internal(None)?;
        Self::validate_embedding_dimensions(embedding, &config)?;
        index.search_with_filter(embedding, k, predicate)
    }

    // ========================================================================
    // Multi-Property Vector Search Methods (Issue #389)
    // ========================================================================

    /// Find k most similar nodes in a specific property's vector index.
    ///
    /// Use this method when you have multiple vector indexes and need to search
    /// a specific one. The property must have a vector index enabled.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The indexed property to search
    /// * `query_node_id` - The node to find similar nodes for
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A list of (NodeId, similarity_score) pairs sorted by similarity (highest first).
    /// The query node itself is excluded from results.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No vector index is enabled for the specified property
    /// - Query node is not found
    /// - Query node does not have the specified property
    /// - The property value is not a vector
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Search title embeddings for similar nodes
    /// let similar = storage.find_similar_in("title_embedding", node_id, 10)?;
    ///
    /// // Search body embeddings (different property, potentially different results)
    /// let similar_body = storage.find_similar_in("body_embedding", node_id, 10)?;
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_in(
        &self,
        property_name: &str,
        query_node_id: NodeId,
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        let (_, index, _) = self.get_vector_index_internal(Some(property_name))?;
        let query_vector = self.get_node_vector(query_node_id, property_name)?;
        self.run_vector_search(&index, &query_vector, k, None, Some(query_node_id))
    }

    /// Search a specific property's vector index with a raw embedding.
    /// Search a specific property's vector index with a raw embedding.
    ///
    /// Equivalent to [`find_similar_by_embedding_in`](Self::find_similar_by_embedding_in).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn search_vectors_in(
        &self,
        property_name: &str,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        self.find_similar_by_embedding_in(property_name, embedding, k)
    }

    // ========================================================================
    // Temporal Vector Indexing (Phase 3)
    // ========================================================================

    /// Enable temporal vector indexing for a specific property.
    ///
    /// Once enabled, vector changes will be tracked over time using snapshot-based
    /// indexing, enabling point-in-time vector queries and semantic drift tracking.
    ///
    /// # Arguments
    ///
    /// * `property_name` - Name of the property containing vectors
    /// * `config` - Temporal vector index configuration
    ///
    /// # Errors
    ///
    /// Returns an error if temporal vector indexing is already enabled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aletheiadb::index::vector::temporal::{TemporalVectorConfig, SnapshotStrategy};
    /// use aletheiadb::index::vector::HnswConfig;
    ///
    /// let hnsw_config = HnswConfig::new(384, DistanceMetric::Cosine);
    /// let temporal_config = TemporalVectorConfig::default_with_hnsw(hnsw_config);
    /// storage.enable_temporal_vector_index("embedding", temporal_config)?;
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn enable_temporal_vector_index(
        &self,
        property_name: &str,
        config: TemporalVectorConfig,
    ) -> Result<()> {
        // Check if already enabled for this specific property
        if self.temporal_vector_indexes.contains_key(property_name) {
            return Err(crate::core::error::Error::Vector(
                crate::core::error::VectorError::IndexError(format!(
                    "Temporal vector index is already enabled for property '{}'",
                    property_name
                )),
            ));
        }

        // Create temporal vector index wrapped in Arc for sharing
        let index = Arc::new(TemporalVectorIndex::new(config.clone())?);

        // Insert into multi-property DashMap
        self.temporal_vector_indexes.insert(
            property_name.to_string(),
            TemporalVectorIndexEntry { index, config },
        );

        Ok(())
    }

    /// Check if temporal vector indexing is enabled for any property.
    pub fn is_temporal_vector_index_enabled(&self) -> bool {
        !self.temporal_vector_indexes.is_empty()
    }

    /// Check if temporal vector indexing is enabled for a specific property.
    pub fn is_temporal_vector_index_enabled_for(&self, property_name: &str) -> bool {
        self.temporal_vector_indexes.contains_key(property_name)
    }

    /// List all property names that have temporal vector indexes enabled.
    ///
    /// Returns a vector of property names that have temporal vector indexing configured.
    pub fn list_temporal_vector_indexes(&self) -> Vec<String> {
        self.temporal_vector_indexes
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get a reference to the temporal vector index for a specific property.
    ///
    /// Returns `None` if temporal vector indexing is not enabled for that property.
    pub(crate) fn get_temporal_vector_index_for(
        &self,
        property_name: &str,
    ) -> Option<Arc<TemporalVectorIndex>> {
        self.temporal_vector_indexes
            .get(property_name)
            .map(|entry| Arc::clone(&entry.index))
    }

    /// Get the default property name for temporal vector indexing.
    ///
    /// This backs the property-less temporal query APIs (`find_similar_as_of`,
    /// `find_similar_in_range`). It deterministically selects the alphabetically
    /// first property name from the enabled temporal vector indexes, mirroring
    /// [`get_default_vector_property_name`](Self::get_default_vector_property_name).
    ///
    /// Returns `None` if no temporal vector indexes are enabled.
    fn get_default_temporal_vector_property_name(&self) -> Option<String> {
        if self.temporal_vector_indexes.is_empty() {
            return None;
        }

        // Optimization: if len == 1, just take it (no sorting needed)
        if self.temporal_vector_indexes.len() == 1 {
            return self
                .temporal_vector_indexes
                .iter()
                .next()
                .map(|r| r.key().clone());
        }

        // Find min key alphabetically. Use `.fold` instead of `.min_by` so each
        // DashMap guard is dropped immediately (see get_default_vector_property_name).
        self.temporal_vector_indexes
            .iter()
            .fold(None, |min: Option<String>, current| {
                let key = current.key();
                match min {
                    Some(m) if m.as_str() <= key.as_str() => Some(m),
                    _ => Some(key.clone()),
                }
            })
    }

    /// Find k most similar nodes at a specific point in time.
    ///
    /// Returns nodes similar to the query embedding as they existed at the given timestamp.
    /// Queries the default temporal vector index (the alphabetically first indexed
    /// property when multiple temporal indexes are enabled). Use
    /// [`find_similar_as_of_in`](Self::find_similar_as_of_in) to target a specific property.
    ///
    /// # Arguments
    ///
    /// * `embedding` - Query vector
    /// * `k` - Number of results
    /// * `timestamp` - Point in time to query
    ///
    /// # Returns
    ///
    /// Vector of (NodeId, similarity) pairs sorted by similarity (descending).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Temporal vector index is not enabled
    /// - No snapshot exists at or before the timestamp
    /// - Embedding dimensions don't match the indexed property
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Find similar documents as they existed in the past
    /// let query_embedding = vec![0.1; 384];
    /// let timestamp = 1234567890000000; // microseconds since epoch
    /// let results = storage.find_similar_as_of(&query_embedding, 10, timestamp)?;
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_as_of(
        &self,
        embedding: &[f32],
        k: usize,
        timestamp: Timestamp,
    ) -> Result<Vec<(NodeId, f32)>> {
        let property_name = self
            .get_default_temporal_vector_property_name()
            .ok_or_else(|| {
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                "Temporal vector index is not enabled. Call enable_temporal_vector_index() first."
                    .to_string(),
            ))
            })?;

        self.find_similar_as_of_in(&property_name, embedding, k, timestamp)
    }

    /// Find k most similar nodes at a specific point in time for a specific property.
    ///
    /// This is the property-specific version of [`find_similar_as_of`](Self::find_similar_as_of).
    /// It validates that the requested property matches the property for which
    /// the temporal vector index was enabled.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property containing the vector embeddings
    /// * `embedding` - Query vector to find similar vectors to
    /// * `k` - Number of results to return
    /// * `timestamp` - The point in time to query
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Temporal vector index is not enabled
    /// - The property name doesn't match the indexed property
    /// - Query embedding dimensions don't match
    /// - No snapshot exists at the given timestamp
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Find similar documents as they existed in the past for a specific property
    /// let query_embedding = vec![0.1; 384];
    /// let timestamp = 1234567890000000;
    /// let results = storage.find_similar_as_of_in("content_embedding", &query_embedding, 10, timestamp)?;
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_as_of_in(
        &self,
        property_name: &str,
        embedding: &[f32],
        k: usize,
        timestamp: crate::core::temporal::Timestamp,
    ) -> Result<Vec<(crate::core::NodeId, f32)>> {
        // Look up the temporal index for this specific property
        let index = self
            .get_temporal_vector_index_for(property_name)
            .ok_or_else(|| {
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                    format!(
                        "No temporal vector index enabled for property '{}'. \
                     Call db.vector_index(\"{}\").temporal(...).enable() first.",
                        property_name, property_name
                    ),
                ))
            })?;

        index.find_similar_as_of(embedding, k, timestamp)
    }

    /// Track semantic drift for a node over time in a specific property's temporal index.
    ///
    /// This method tracks how a node's embedding has changed relative to a reference
    /// embedding over time. It validates that the requested property matches the
    /// property for which the temporal vector index was enabled.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property containing the vector embeddings
    /// * `node_id` - The node to track drift for
    /// * `reference_embedding` - Reference vector to measure drift against
    /// * `time_range` - The time range to search for drift
    ///
    /// # Returns
    ///
    /// A vector of (timestamp, drift_score) pairs showing how the node's embedding
    /// drifted from the reference at each snapshot time.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Temporal vector index is not enabled
    /// - The property name doesn't match the indexed property
    /// - Reference embedding dimensions don't match
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn track_drift_in(
        &self,
        property_name: &str,
        node_id: crate::core::NodeId,
        reference_embedding: &[f32],
        time_range: crate::core::temporal::TimeRange,
    ) -> Result<Vec<(crate::core::temporal::Timestamp, f32)>> {
        // Look up the temporal index for this specific property (multi-property support)
        let index = self
            .get_temporal_vector_index_for(property_name)
            .ok_or_else(|| {
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                    format!(
                        "No temporal vector index enabled for property '{}'. \
                     Call db.vector_index(\"{}\").temporal(...).enable() first.",
                        property_name, property_name
                    ),
                ))
            })?;

        index.track_semantic_drift(node_id, reference_embedding, time_range)
    }

    /// Get the semantic evolution of a node's embedding over time in a specific property.
    ///
    /// Returns the actual embedding vectors at each snapshot timestamp.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn semantic_evolution_in(
        &self,
        property_name: &str,
        node_id: crate::core::NodeId,
        time_range: crate::core::temporal::TimeRange,
    ) -> Result<Vec<(crate::core::temporal::Timestamp, std::sync::Arc<[f32]>)>> {
        // Look up the temporal index for this specific property (multi-property support)
        let index = self
            .get_temporal_vector_index_for(property_name)
            .ok_or_else(|| {
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                    format!(
                        "No temporal vector index enabled for property '{}'. \
                     Call db.vector_index(\"{}\").temporal(...).enable() first.",
                        property_name, property_name
                    ),
                ))
            })?;

        index.semantic_evolution(node_id, time_range)
    }

    /// Find all nodes with semantic drift above a threshold in a specific property.
    ///
    /// Scans all nodes and identifies those whose embeddings have changed
    /// by more than the specified threshold over the time range.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_drift_in(
        &self,
        property_name: &str,
        threshold: f32,
        time_range: crate::core::temporal::TimeRange,
        metric: crate::index::vector::temporal::DriftMetric,
    ) -> Result<Vec<(crate::core::NodeId, f32)>> {
        // Look up the temporal index for this specific property (multi-property support)
        let index = self
            .get_temporal_vector_index_for(property_name)
            .ok_or_else(|| {
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                    format!(
                        "No temporal vector index enabled for property '{}'. \
                     Call db.vector_index(\"{}\").temporal(...).enable() first.",
                        property_name, property_name
                    ),
                ))
            })?;

        index.find_semantic_drift(threshold, time_range, metric)
    }

    /// Find k most similar nodes across a time range.
    ///
    /// Returns results for each snapshot within the time range, showing how
    /// semantic similarity evolved over time.
    ///
    /// # Arguments
    ///
    /// * `embedding` - Query vector
    /// * `k` - Number of results per snapshot
    /// * `time_range` - Time range to query
    ///
    /// # Returns
    ///
    /// Vector of (timestamp, results) pairs where results are Vec<(NodeId, similarity)>.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aletheiadb::core::temporal::TimeRange;
    ///
    /// // Track how similar documents changed over time
    /// let query = vec![0.1; 384];
    /// let time_range = TimeRange::between(start_ts, end_ts);
    /// let results = storage.find_similar_in_range(&query, 10, time_range)?;
    /// for (timestamp, similar_nodes) in results {
    ///     println!("At timestamp {}: {} similar nodes", timestamp, similar_nodes.len());
    /// }
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_in_range(
        &self,
        embedding: &[f32],
        k: usize,
        time_range: crate::core::temporal::TimeRange,
    ) -> Result<TemporalSearchResults> {
        let property_name = self
            .get_default_temporal_vector_property_name()
            .ok_or_else(|| {
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                "Temporal vector index is not enabled. Call enable_temporal_vector_index() first."
                    .to_string(),
            ))
            })?;

        let index = self
            .get_temporal_vector_index_for(&property_name)
            .ok_or_else(|| {
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                "Temporal vector index is not enabled. Call enable_temporal_vector_index() first."
                    .to_string(),
            ))
            })?;

        index.find_similar_in_range(embedding, k, time_range)
    }

    /// Notify all temporal vector indexes of a transaction.
    ///
    /// This should be called after committing a transaction to trigger snapshot
    /// creation based on each index's configured strategy.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn on_temporal_vector_transaction(&self) -> Result<()> {
        for (_, index) in self.collect_temporal_vector_indexes() {
            index.on_transaction()?;
        }
        Ok(())
    }

    /// Collect `(property_name, index)` pairs for all enabled temporal vector indexes.
    ///
    /// Snapshots the DashMap into a `Vec` so callers never invoke index operations
    /// while holding DashMap shard guards.
    fn collect_temporal_vector_indexes(&self) -> Vec<(String, Arc<TemporalVectorIndex>)> {
        if self.temporal_vector_indexes.is_empty() {
            return Vec::new();
        }
        self.temporal_vector_indexes
            .iter()
            .map(|entry| (entry.key().clone(), Arc::clone(&entry.value().index)))
            .collect()
    }

    /// Helper to index a node's vectors in every enabled temporal vector index.
    ///
    /// Returns `Err` on the first index that fails; indexes visited earlier in
    /// the fan-out will have already been updated (partial work), matching the
    /// behavior of [`try_index_vector`](Self::try_index_vector). Callers must
    /// not assume a failure left every temporal index untouched.
    fn try_index_temporal_vector(
        &self,
        node_id: NodeId,
        properties: &PropertyMap,
        timestamp: Timestamp,
    ) -> Result<()> {
        for (property_name, index) in self.collect_temporal_vector_indexes() {
            if let Some(vector) = properties.get(&property_name).and_then(|v| v.as_vector()) {
                index.add(node_id, vector, timestamp)?;
            }
        }
        Ok(())
    }

    /// Update every enabled temporal vector index for a node whose properties
    /// changed, mirroring [`update_vector_index`](Self::update_vector_index) for
    /// the current-state HNSW.
    ///
    /// For each indexed property this diffs the old vs new vector and applies the
    /// matching temporal operation:
    ///
    /// - `(None, None)` — property absent before and after: no-op.
    /// - `(None, Some)` — vector added: [`TemporalVectorIndex::add`].
    /// - `(Some, None)` — vector removed: [`TemporalVectorIndex::remove`], which
    ///   drops the embedding from the live HNSW **and** from `current_state.vectors`
    ///   so it no longer leaks into the next snapshot.
    /// - `(Some, Some)` — vector overwritten: only re-`add` (an upsert) when the
    ///   value actually changed, avoiding needless HNSW churn.
    ///
    /// # Bi-temporal tombstone semantics
    ///
    /// `remove`/overwrite CLOSE the vector's interval at `timestamp` — the change
    /// is reflected in any snapshot taken **after** it — but they never erase
    /// snapshots taken **before** it. History is therefore preserved: a
    /// `find_similar_as_of` anchored before the change still returns the old
    /// embedding (Issue #3621). The de-index is also applied to the live
    /// current-state HNSW **immediately** (via `current_state.vectors`), so a
    /// current-state `find_similar` stops returning the stale phantom right away.
    /// The as-of-now temporal side, by contrast, is only as fresh as the last
    /// snapshot: between this de-index and the next snapshot a
    /// `find_similar_as_of(now)` still reads the prior snapshot and returns the
    /// old embedding — inherent snapshot cadence, symmetric with how newly added
    /// vectors only appear in an as-of-now query once the next snapshot is taken.
    /// Once a snapshot is taken after the change, the as-of-now query no longer
    /// returns the phantom. Because this is the exact method the WAL-replay
    /// `update_node_direct` invokes, crash recovery reconstructs the same
    /// corrected point-in-time state.
    ///
    /// Fan-out matches [`try_index_temporal_vector`](Self::try_index_temporal_vector):
    /// on the first failing index this returns `Err` with earlier indexes already
    /// updated (partial work); the node update itself is not applied by the caller.
    fn update_temporal_vector_index(
        &self,
        node_id: NodeId,
        new_props: &PropertyMap,
        old_props: &PropertyMap,
        timestamp: Timestamp,
    ) -> Result<()> {
        for (property_name, index) in self.collect_temporal_vector_indexes() {
            let old_vec = old_props.get(&property_name).and_then(|v| v.as_vector());
            let new_vec = new_props.get(&property_name).and_then(|v| v.as_vector());

            match (old_vec, new_vec) {
                (None, None) => {}
                (None, Some(v)) => {
                    index.add(node_id, v, timestamp)?;
                }
                (Some(_), None) => {
                    // Propagate the error via `?` on purpose: unlike the
                    // current-state `update_vector_index` (which swallows removal
                    // errors with `let _ =`), a fault from the temporal index is a
                    // real bi-temporal-bookkeeping failure the caller must see so
                    // the node update aborts rather than committing a stale phantom.
                    // Safe because `HnswIndex::remove` returns `Ok(())` for a
                    // missing id — enabling an index after the node already exists
                    // (so the id was never added) is a no-op, not a spurious abort.
                    index.remove(node_id, timestamp)?;
                }
                (Some(o), Some(n)) => {
                    if o != n {
                        // add() is an upsert (remove + add) on the temporal index.
                        index.add(node_id, n, timestamp)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Helper to remove a node's vectors from every enabled temporal vector index.
    ///
    /// Mirrors [`try_remove_from_index`](Self::try_remove_from_index): removal is
    /// attempted on ALL indexes even when one fails, so a per-index error never
    /// leaves stale entries behind in the remaining indexes. The first error
    /// encountered is returned after every index has been attempted. A missing
    /// id is **not** an error — `HnswIndex::remove` returns `Ok(())` when the
    /// node was never indexed — so an index that never contained the node does
    /// not contribute a spurious failure here. Any error returned therefore
    /// reflects a genuine index fault; callers treating removal as best-effort
    /// (e.g. `delete_node_direct`) may still ignore it.
    fn try_remove_temporal_vector(&self, node_id: NodeId, timestamp: Timestamp) -> Result<()> {
        let mut first_err = None;
        for (_, index) in self.collect_temporal_vector_indexes() {
            if let Err(e) = index.remove(node_id, timestamp) {
                first_err.get_or_insert(e);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Get statistics about the current storage
    pub fn stats(&self) -> CurrentStats {
        CurrentStats {
            node_count: self.node_count(),
            edge_count: self.edge_count(),
        }
    }

    // ========================================================================
    // Query Executor Support Methods (VS-060)
    // ========================================================================

    /// Get all node IDs in the current storage.
    ///
    /// This is used by the query executor for full node scans.
    /// For large graphs, prefer using label-filtered scans instead.
    pub fn get_all_node_ids(&self) -> Vec<NodeId> {
        self.indexes.iter_node_ids().collect()
    }

    /// Get all edge IDs in the current storage.
    ///
    /// This is used by recovery tests and query executor for full edge scans.
    /// For large graphs, prefer using filtered scans instead.
    pub fn get_all_edge_ids(&self) -> Vec<EdgeId> {
        self.indexes.iter_edge_ids().collect()
    }

    /// Get all nodes in the current storage.
    ///
    /// Used by recovery property tests to verify invariants.
    pub fn get_all_nodes(&self) -> Vec<crate::Node> {
        let mut nodes = Vec::with_capacity(self.indexes.node_count());
        nodes.extend(self.indexes.iter_nodes().map(|n| n.clone()));
        nodes
    }

    /// Get all edges in the current storage.
    ///
    /// Used by recovery property tests to verify invariants.
    pub fn get_all_edges(&self) -> Vec<crate::Edge> {
        let mut edges = Vec::with_capacity(self.indexes.edge_count());
        edges.extend(self.indexes.iter_edges().map(|e| e.clone()));
        edges
    }

    /// Visit every node in the current storage by reference, without cloning.
    ///
    /// Prefer this over `get_all_nodes()` when only a read-only pass over each
    /// node is needed (e.g. aggregation), since it avoids allocating an owned
    /// `Vec<Node>` and cloning each node.
    pub fn visit_nodes<F: FnMut(&crate::Node)>(&self, mut f: F) {
        for node in self.indexes.iter_nodes() {
            f(&node);
        }
    }

    /// Visit every edge in the current storage by reference, without cloning.
    ///
    /// See [`Self::visit_nodes`] for why this is preferable to `get_all_edges()`
    /// for read-only aggregation passes.
    pub fn visit_edges<F: FnMut(&crate::Edge)>(&self, mut f: F) {
        for edge in self.indexes.iter_edges() {
            f(&edge);
        }
    }

    /// Get node IDs by label.
    ///
    /// Returns the IDs of all nodes with the given label.
    /// This is more memory-efficient than `get_nodes_by_label` when only IDs are needed,
    /// as it avoids cloning the full node data.
    pub fn get_node_ids_by_label(&self, label: &str) -> Vec<NodeId> {
        let label_id = match crate::core::interning::GLOBAL_INTERNER.get_id(label) {
            Some(id) => id,
            None => return Vec::new(),
        };
        self.indexes
            .iter_nodes()
            .filter(|n| n.label == label_id)
            .map(|n| n.id)
            .collect()
    }

    /// Collect one bounded page of live node ids for a chunked full scan
    /// (Issue #3422).
    ///
    /// Returns the (at most) `limit` **smallest** live node ids strictly
    /// greater than `after`, in ascending order. Passing `after = None` starts
    /// from the beginning; passing the last id of the previous page as `after`
    /// yields the next page. This ascending-id keyset makes paging
    /// duplicate-free and gap-free across calls.
    ///
    /// When `label` is `Some`, only ids of live nodes carrying that interned
    /// label are considered (via the hot 16-byte header index); when `None`,
    /// every live node id is considered.
    ///
    /// # Why this recovers O(live) scan time
    ///
    /// The candidates come from the live `node_headers`/`nodes` index, so ids
    /// with no live node (deletion tombstones, reserved-but-unused ids,
    /// post-compaction gaps) are **never visited** -- unlike a dense
    /// `[0, max_id)` sweep whose cost tracks the id high-water mark. A full scan
    /// built on this therefore costs O(live), not O(max_id).
    ///
    /// # Memory
    ///
    /// Selection uses a bounded max-heap of at most `limit` ids, so resident
    /// memory is O(`limit`) regardless of how many live nodes exist -- the
    /// caller never materializes the whole id set (the OOM vector PR #3418
    /// removed). `limit == 0` returns an empty page.
    ///
    /// # Concurrency
    ///
    /// Enumeration briefly holds each `node_headers` DashMap shard lock only
    /// while copying ids into the heap, releasing every lock before returning;
    /// no lock is held across the returned page. The snapshot is relaxed (not
    /// isolated): a node inserted or deleted concurrently with enumeration may
    /// or may not appear, matching the existing unsynchronized full-scan
    /// semantics.
    pub fn collect_node_id_page(
        &self,
        after: Option<u64>,
        limit: usize,
        label: Option<crate::core::interning::InternedString>,
    ) -> Vec<NodeId> {
        self.collect_node_id_page_counted(after, limit, label).0
    }

    /// Like [`Self::collect_node_id_page`], but also returns the number of live
    /// candidate ids **examined** while building the page.
    ///
    /// The examined count is the full per-page enumeration cost -- every id
    /// yielded by `iter_node_ids` / `filter_nodes_by_label` that the selector
    /// inspected, not just the (at most) `limit` ids retained. The chunked full
    /// scan (Issue #3422) uses this as its honest work proxy: because each page
    /// re-enumerates the whole live key set to find the next `limit` smallest
    /// ids, the real cost of a full paged scan is `live * pages`, not `live`.
    /// Counting only the retained ids would understate paged cost by a factor of
    /// `pages` and make a multi-page scan look artificially cheap.
    pub(crate) fn collect_node_id_page_counted(
        &self,
        after: Option<u64>,
        limit: usize,
        label: Option<crate::core::interning::InternedString>,
    ) -> (Vec<NodeId>, u64) {
        if limit == 0 {
            return (Vec::new(), 0);
        }

        // Bounded max-heap keeping the `limit` smallest ids seen so far that are
        // strictly greater than `after`. The heap's max is the current page
        // boundary; a smaller candidate evicts it. The heap never exceeds
        // `limit` (it pops before pushing once full), so `with_capacity(limit)`
        // is exact and avoids the `limit + 1` overflow risk when `limit` is at
        // the top of `usize`'s range.
        let mut examined: u64 = 0;
        let mut heap: std::collections::BinaryHeap<u64> =
            std::collections::BinaryHeap::with_capacity(limit);
        let mut consider = |id: NodeId| {
            // Count every candidate the enumeration inspected: this is the
            // per-page scan cost the paged strategy pays.
            examined += 1;
            let value = id.as_u64();
            if let Some(a) = after
                && value <= a
            {
                return;
            }
            if heap.len() < limit {
                heap.push(value);
            } else if let Some(&current_max) = heap.peek()
                && value < current_max
            {
                heap.pop();
                heap.push(value);
            }
        };

        match label {
            Some(label_id) => {
                for id in self.indexes.filter_nodes_by_label(label_id) {
                    consider(id);
                }
            }
            None => {
                for id in self.indexes.iter_node_ids() {
                    consider(id);
                }
            }
        }

        // `into_sorted_vec` on a max-heap yields ascending order.
        let page = heap
            .into_sorted_vec()
            .into_iter()
            .filter_map(|value| NodeId::new(value).ok())
            .collect();
        (page, examined)
    }

    /// Get nodes by label.
    ///
    /// Returns an iterator over all nodes with the given label.
    /// This is more efficient than scanning all nodes and filtering.
    pub fn get_nodes_by_label(&self, label: &str) -> Vec<Node> {
        let label_id = match crate::core::interning::GLOBAL_INTERNER.get_id(label) {
            Some(id) => id,
            None => return Vec::new(), // Label doesn't exist
        };
        self.indexes
            .iter_nodes()
            .filter(|n| n.label == label_id)
            .map(|n| n.clone())
            .collect()
    }

    /// Find nodes by label and property value.
    ///
    /// Returns the IDs of all nodes with the given label whose specified property
    /// equals the given value. Returns an empty `Vec` if no matches are found,
    /// or if the label or property key has never been interned.
    ///
    /// This is more efficient than `get_nodes_by_label` followed by manual filtering
    /// because it avoids cloning non-matching nodes.
    ///
    /// # Performance
    ///
    /// - **Time**: O(N) where N = nodes with the given label
    /// - **Space**: O(M) where M = number of matching nodes
    pub fn find_nodes_by_property(
        &self,
        label: &str,
        property_key: &str,
        property_value: &PropertyValue,
    ) -> Vec<NodeId> {
        let label_id = match crate::core::interning::GLOBAL_INTERNER.get_id(label) {
            Some(id) => id,
            None => return Vec::new(),
        };
        let key_id = match crate::core::interning::GLOBAL_INTERNER.get_id(property_key) {
            Some(id) => id,
            None => return Vec::new(),
        };
        // Fast path: an equality index covering this (label, property) with an
        // indexable value type (String/Int/Bool) — an O(matches) probe instead
        // of the O(nodes-per-label) scan. Non-indexable value types (notably
        // Float) and uncovered pairs fall through to the scan, returning
        // byte-identical results. See `crate::index::property_index`.
        let covered_value = crate::index::property_index::value_key(property_value)
            .filter(|_| self.indexes.has_property_index(label_id, key_id));
        if let Some(vk) = covered_value {
            return self
                .indexes
                .find_nodes_by_property_indexed(label_id, key_id, &vk);
        }
        self.indexes
            .iter_nodes()
            .filter(|n| n.label == label_id)
            .filter(|n| {
                n.properties
                    .get_by_interned_key(&key_id)
                    .is_some_and(|v| v == property_value)
            })
            .map(|n| n.id)
            .collect()
    }

    // ========================================================================
    // Secondary property (equality) index — opt-in, node-only, current-state.
    // See `crate::index::property_index` for the design and scope boundaries.
    // ========================================================================

    /// Enable a current-state equality index on `(label, property)` and backfill
    /// it from current nodes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FailedPrecondition`](crate::core::error::Error::FailedPrecondition)
    /// if an index is already enabled for this `(label, property)` pair.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn enable_property_index(&self, label: &str, property: &str) -> Result<()> {
        let label_id = GLOBAL_INTERNER.intern(label)?;
        let key_id = GLOBAL_INTERNER.intern(property)?;
        // Exclude ALL concurrent writers (each holds `snapshot_lock.read()`) for
        // the whole set-flag + backfill, so no create/update/delete can
        // interleave and leave a node in the wrong bucket or no bucket. See
        // `CurrentIndexes::enable_property_index` for the two racy interleavings
        // this closes.
        let _lock = self.snapshot_lock.write();
        if !self.indexes.enable_property_index(label_id, key_id) {
            return Err(crate::core::error::Error::FailedPrecondition(format!(
                "property index already enabled for ({label}, {property})"
            )));
        }
        Ok(())
    }

    /// Whether an equality index is enabled for `(label, property)`.
    pub fn has_property_index(&self, label: &str, property: &str) -> bool {
        let (Some(label_id), Some(key_id)) = (
            GLOBAL_INTERNER.get_id(label),
            GLOBAL_INTERNER.get_id(property),
        ) else {
            return false;
        };
        self.indexes.has_property_index(label_id, key_id)
    }

    /// List all enabled property indexes.
    pub fn list_property_indexes(&self) -> Vec<crate::index::property_index::PropertyIndexInfo> {
        self.indexes
            .property_index_pairs()
            .into_iter()
            .filter_map(|(label_id, key_id)| {
                let label = GLOBAL_INTERNER.resolve_with(label_id, |s| s.to_string())?;
                let property = GLOBAL_INTERNER.resolve_with(key_id, |s| s.to_string())?;
                Some(crate::index::property_index::PropertyIndexInfo { label, property })
            })
            .collect()
    }

    /// Drop the equality index on `(label, property)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FailedPrecondition`](crate::core::error::Error::FailedPrecondition)
    /// if no index is enabled for this `(label, property)` pair.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn drop_property_index(&self, label: &str, property: &str) -> Result<()> {
        let not_enabled = || {
            crate::core::error::Error::FailedPrecondition(format!(
                "no property index enabled for ({label}, {property})"
            ))
        };
        let (Some(label_id), Some(key_id)) = (
            GLOBAL_INTERNER.get_id(label),
            GLOBAL_INTERNER.get_id(property),
        ) else {
            return Err(not_enabled());
        };
        // Same exclusion as enable: hold the write lock so no concurrent writer's
        // reindex can add a bucket after the purge (which a later re-enable would
        // otherwise merge into fresh results — the #S3 race).
        let _lock = self.snapshot_lock.write();
        if !self.indexes.disable_property_index(label_id, key_id) {
            return Err(not_enabled());
        }
        Ok(())
    }

    /// Get the name of the property used for vector indexing.
    ///
    /// Equivalent to `get_default_vector_property_name` (internal).
    /// Used by the query executor for vector reranking operations.
    pub fn get_vector_property_name(&self) -> Option<String> {
        self.get_default_vector_property_name()
    }

    /// Get node counts grouped by label.
    ///
    /// Returns an iterator of (interned_label, count) pairs.
    /// Used by the query planner for cost estimation and cardinality estimation.
    pub fn label_counts(&self) -> Vec<(crate::core::interning::InternedString, usize)> {
        use crate::core::hasher::IdentityHasher;
        use std::collections::HashMap;
        use std::hash::BuildHasherDefault;

        // ⚡ Bolt: Use `IdentityHasher` instead of default SipHash.
        // `InternedString` is already a high-quality unique integer ID (u32),
        // so computing SipHash per node (potentially millions) is pure overhead.
        // Bypassing hashing reduces cardinality estimation time significantly.
        let mut counts: HashMap<
            crate::core::interning::InternedString,
            usize,
            BuildHasherDefault<IdentityHasher>,
        > = HashMap::default();
        for node in self.indexes.iter_nodes() {
            *counts.entry(node.label).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    /// Get average out-degree across all nodes.
    ///
    /// Used by the query planner for cost estimation.
    pub fn avg_out_degree(&self) -> f64 {
        let node_count = self.node_count();
        if node_count == 0 {
            return 0.0;
        }
        self.edge_count() as f64 / node_count as f64
    }

    /// Get target node IDs from outgoing edges (used for traversal iterators).
    ///
    /// Returns the target node IDs of all outgoing edges from the source node.
    pub fn get_outgoing_targets(&self, source: NodeId) -> Vec<NodeId> {
        self.apply_gate.read(|| {
            let guard = self.indexes.get_outgoing(source);
            let mut result = Vec::with_capacity(guard.capacity_hint());
            result.extend(guard.iter().map(|entry| entry.target));
            result
        })
    }

    /// Get target node IDs from outgoing edges with a specific label.
    pub fn get_outgoing_targets_with_label(&self, source: NodeId, label: &str) -> Vec<NodeId> {
        let label_id = match GLOBAL_INTERNER.get_id(label) {
            Some(id) => id,
            None => return Vec::new(),
        };
        // Optimized to avoid intermediate Vec allocation and pre-allocate result
        self.apply_gate.read(|| {
            let guard = self.indexes.get_outgoing(source);
            let mut result = Vec::with_capacity(guard.capacity_hint());
            result.extend(
                guard
                    .iter()
                    .filter(|entry| entry.label == label_id)
                    .map(|entry| entry.target),
            );
            result
        })
    }

    /// Get source node IDs from incoming edges (used for traversal iterators).
    ///
    /// Returns the source node IDs of all incoming edges to the target node.
    /// Note: For incoming edges, the "target" field in AdjacencyEntry represents
    /// the source node (the node the edge is coming from).
    pub fn get_incoming_sources(&self, target: NodeId) -> Vec<NodeId> {
        self.apply_gate.read(|| {
            let guard = self.indexes.get_incoming(target);
            let mut result = Vec::with_capacity(guard.capacity_hint());
            result.extend(guard.iter().map(|entry| entry.target));
            result
        })
    }

    /// Get source node IDs from incoming edges with a specific label.
    pub fn get_incoming_sources_with_label(&self, target: NodeId, label: &str) -> Vec<NodeId> {
        let label_id = match GLOBAL_INTERNER.get_id(label) {
            Some(id) => id,
            None => return Vec::new(),
        };
        // Optimized to avoid intermediate Vec allocation and pre-allocate result
        self.apply_gate.read(|| {
            let guard = self.indexes.get_incoming(target);
            let mut result = Vec::with_capacity(guard.capacity_hint());
            result.extend(
                guard
                    .iter()
                    .filter(|entry| entry.label == label_id)
                    .map(|entry| entry.target),
            );
            result
        })
    }

    /// Get the number of vectors in the HNSW index.
    ///
    /// Used by the query planner for statistics.
    pub fn vector_count(&self) -> usize {
        self.get_default_vector_property_name()
            .and_then(|prop_name| self.vector_indexes.get(&prop_name))
            .map(|entry| entry.value().index.len())
            .unwrap_or(0)
    }

    /// Iterate over all nodes (for persistence).
    ///
    /// This is a helper method for index persistence that provides
    /// access to all nodes in the current storage.
    ///
    /// Returns an iterator to avoid allocating a Vec for large graphs,
    /// improving memory efficiency during persistence operations.
    pub(crate) fn all_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        self.indexes.iter_nodes().map(|n| n.clone())
    }

    /// Iterate over all edges (for persistence).
    ///
    /// This is a helper method for index persistence that provides
    /// access to all edges in the current storage.
    ///
    /// Returns an iterator to avoid allocating a Vec for large graphs,
    /// improving memory efficiency during persistence operations.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn all_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.indexes.iter_edges().map(|e| e.clone())
    }

    /// Iterate over all nodes by borrow (no per-element clone).
    ///
    /// Yields DashMap guard references (`Deref<Target = Node>`) delegated to
    /// [`CurrentIndexes::iter_nodes`], for read-only scans (e.g. the HLC
    /// startup seed) that only need to read fields off each node. Prefer this
    /// over [`Self::all_nodes`], which clones every `Node`, when owned values
    /// are not required.
    pub(crate) fn iter_nodes(
        &self,
    ) -> impl Iterator<Item = impl std::ops::Deref<Target = Node> + '_> + '_ {
        self.indexes.iter_nodes()
    }

    /// Iterate over all edges by borrow (no per-element clone).
    ///
    /// Yields DashMap guard references (`Deref<Target = Edge>`) delegated to
    /// [`CurrentIndexes::iter_edges`], for read-only scans (e.g. the HLC
    /// startup seed) that only need to read fields off each edge. Prefer this
    /// over [`Self::all_edges`], which clones every `Edge`, when owned values
    /// are not required.
    pub(crate) fn iter_edges(
        &self,
    ) -> impl Iterator<Item = impl std::ops::Deref<Target = Edge> + '_> + '_ {
        self.indexes.iter_edges()
    }

    /// Scan nodes by label, returning an iterator over matching node IDs.
    ///
    /// This method efficiently filters the node collection to find all nodes
    /// with the specified label.
    ///
    /// # Arguments
    ///
    /// * `label` - The label/type to filter by (e.g., "Person", "Product")
    ///
    /// # Returns
    ///
    /// An iterator yielding `NodeId` for all nodes matching the label.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(n) where n is the total number of nodes
    /// - **Space Complexity**: O(1) - iterator, no allocation
    /// - **Filtering**: Uses interned string comparison (pointer equality)
    ///
    /// Note: This operation scans all nodes. For workloads with frequent label scans,
    /// consider using the query engine's optimized label index lookups.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Scan all Person nodes
    /// for node_id in storage.scan_nodes_by_label("Person") {
    ///     println!("Found Person: {}", node_id);
    /// }
    /// ```
    pub fn scan_nodes_by_label(&self, label: &str) -> impl Iterator<Item = NodeId> + '_ {
        // Look up the label in the interner without creating a new entry
        // If the label was never interned, no nodes with that label exist
        let interned_label = GLOBAL_INTERNER.get_id(label);

        self.indexes
            .iter_nodes()
            .filter(move |node_ref| {
                interned_label.is_some_and(|label_id| node_ref.label == label_id)
            })
            .map(|node_ref| node_ref.id)
    }

    /// Create an MVCC snapshot at the specified LSN.
    ///
    /// This provides snapshot isolation for checkpoint operations, preventing
    /// fuzzy checkpointing (mixed state from different LSNs) that can lead to
    /// data corruption.
    ///
    /// # Snapshot Isolation
    ///
    /// The snapshot captures Arc references to all nodes and edges at the time
    /// of snapshot creation. Concurrent modifications after snapshot creation
    /// do NOT affect the snapshot's iteration.
    ///
    /// # Memory Overhead
    ///
    /// - Does ONE iteration over DashMap to collect Arc references
    /// - Memory: ~8 bytes per entity (just Arc pointers, not full clones)
    /// - For 10M nodes: ~80MB overhead
    ///
    /// # Performance
    ///
    /// - Snapshot creation: ~100ms for 10M nodes (DashMap iteration)
    /// - Concurrent writes: Unaffected, continue during snapshot iteration
    ///
    /// # Arguments
    ///
    /// * `lsn` - LSN at which snapshot is taken (for tracking)
    ///
    /// # Returns
    ///
    /// A snapshot that provides isolated iteration over nodes and edges.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let snapshot = current.create_snapshot(LSN(100));
    ///
    /// // Concurrent writes after this point don't affect snapshot
    /// for node in snapshot.iter_nodes() {
    ///     // Stream to disk without loading entire DB
    ///     persist_node(&node)?;
    /// }
    /// ```
    pub fn create_snapshot(
        &self,
        lsn: crate::storage::wal::LSN,
    ) -> crate::storage::snapshot::CurrentStorageSnapshot {
        use crate::storage::snapshot::CurrentStorageSnapshot;
        use std::sync::Arc;

        let mut n = Vec::with_capacity(self.indexes.node_count());
        n.extend(self.indexes.iter_nodes().map(|n| n.clone()));
        let nodes: Arc<Vec<Node>> = Arc::new(n);

        let mut e = Vec::with_capacity(self.indexes.edge_count());
        e.extend(self.indexes.iter_edges().map(|e| e.clone()));
        let edges: Arc<Vec<Edge>> = Arc::new(e);

        CurrentStorageSnapshot::new(lsn, nodes, edges)
    }

    /// Internal helper to get vector index and configuration.
    ///
    /// If `property_name` is Some, looks up that specific property.
    /// If None, looks up the default property.
    ///
    /// Returns (property_name, index, config).
    fn get_node_vector(&self, node_id: NodeId, prop_name: &str) -> Result<Arc<[f32]>> {
        self.indexes
            .with_node(node_id, |node| {
                node.properties
                    .get(prop_name)
                    .ok_or_else(|| {
                        crate::core::error::Error::Storage(StorageError::PropertyNotFound(
                            prop_name.to_string(),
                        ))
                    })?
                    .as_arc_vector()
                    .ok_or_else(|| {
                        crate::core::error::Error::Vector(
                            crate::core::error::VectorError::InvalidVector {
                                reason: "Property is not a vector".to_string(),
                            },
                        )
                    })
            })
            .transpose()?
            .ok_or(crate::core::error::Error::Storage(
                StorageError::NodeNotFound(node_id),
            ))
    }

    /// Validate that an embedding's dimensions match the index configuration.
    fn validate_embedding_dimensions(embedding: &[f32], config: &HnswConfig) -> Result<()> {
        if embedding.len() != config.dimensions {
            return Err(crate::core::error::Error::Vector(
                crate::core::error::VectorError::DimensionMismatch {
                    expected: config.dimensions,
                    actual: embedding.len(),
                },
            ));
        }
        Ok(())
    }

    fn get_vector_index_internal(
        &self,
        property_name: Option<&str>,
    ) -> Result<(String, Arc<HnswIndex>, HnswConfig)> {
        let prop_name = if let Some(name) = property_name {
            name.to_string()
        } else {
            self.get_default_vector_property_name().ok_or_else(|| {
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                    "Vector index is not enabled. Call db.vector_index(\"...\").hnsw(...).enable() first.".to_string(),
                ))
            })?
        };

        let entry = self.vector_indexes.get(&prop_name).ok_or_else(|| {
            crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(format!(
                "Vector index not found for property '{}'",
                prop_name
            )))
        })?;

        Ok((
            prop_name,
            entry.value().index.clone(),
            entry.value().config.clone(),
        ))
    }

    /// Internal helper to execute vector search with common logic.
    ///
    /// Handles:
    /// - Adaptive over-fetch candidate calculation
    /// - Filtering by label
    /// - Excluding specific nodes
    /// - Recording statistics
    fn run_vector_search(
        &self,
        index: &Arc<HnswIndex>,
        query: &[f32],
        k: usize,
        label_filter: Option<&str>,
        exclude_node: Option<NodeId>,
    ) -> Result<Vec<(NodeId, f32)>> {
        let mut results = if let Some(label) = label_filter {
            let label_id = GLOBAL_INTERNER.intern(label)?;
            let (candidates, stats) = self.calculate_adaptive_candidates(k, label);

            // Important: apply adaptive over-fetch exactly once.
            //
            // `search_with_filter` already performs its own iterative over-fetch expansion to satisfy
            // the requested `k`. Passing our already-overfetched candidate count into that API causes
            // redundant expansion under contention. Instead, fetch `candidates` once and filter locally.
            let candidate_results = index.search(query, candidates)?;
            let candidate_count = candidate_results.len();
            let mut results = Vec::with_capacity(candidate_count);

            for (node_id, similarity) in candidate_results {
                // HOT PATH: check_node_label avoids cloning Node and Option overhead (Issue #339)
                if self.indexes.check_node_label(node_id, label_id) {
                    results.push((node_id, similarity));
                }
            }

            if let Some(exclude_id) = exclude_node {
                results.retain(|(id, _)| *id != exclude_id);
            }

            // Track observed pass rate from actual candidates examined.
            stats.record_search(candidate_count, results.len());
            results
        } else {
            // If we need to exclude a node, we might need one more result
            let search_k = if exclude_node.is_some() { k + 1 } else { k };
            let mut results = index.search(query, search_k)?;

            if let Some(exclude_id) = exclude_node {
                results.retain(|(id, _)| *id != exclude_id);
            }
            results
        };

        results.truncate(k);
        Ok(results)
    }
}

impl Default for CurrentStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;

/// Issue #3788: the replica apply gate, exercised through `CurrentStorage`'s
/// own surface rather than the gate primitive in isolation.
#[cfg(test)]
mod apply_gate_tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn node(id: u64, label: &str) -> Node {
        Node::new(
            NodeId::new(id).unwrap(),
            GLOBAL_INTERNER.intern(label).unwrap(),
            PropertyMapBuilder::new().insert("n", id as i64).build(),
            VersionId::new(id).unwrap(),
        )
    }

    #[test]
    fn gate_starts_disarmed_and_arms_on_demand() {
        let storage = CurrentStorage::new();
        assert!(
            !storage.apply_gate_armed(),
            "a primary's current storage must not pay the gate"
        );
        storage.arm_apply_gate();
        assert!(storage.apply_gate_armed());
        storage.disarm_apply_gate();
        assert!(!storage.apply_gate_armed());
    }

    #[test]
    fn reads_still_work_inside_a_publish_window_on_the_publishing_thread() {
        // The replay engine's #3419 idempotency guards call `get_node` from
        // inside the window; without re-entrancy this would deadlock.
        let storage = CurrentStorage::new();
        storage.arm_apply_gate();

        let _publish = storage.begin_apply_publish();
        let ts = crate::core::temporal::time::now();
        storage.insert_node_direct(node(1, "Person"), ts).unwrap();
        assert!(storage.get_node(NodeId::new(1).unwrap()).is_ok());
        assert!(storage.contains_node(1));
    }

    /// The failure this issue reports: a two-node "transaction" applied one
    /// operation at a time must never be observed half-present.
    ///
    /// Writes are append-only (each round publishes a fresh pair), so presence
    /// is monotone and the pair's two reads can be taken independently -- the
    /// same shape the `torn_frame_safety_*` integration tests observe. Seeing
    /// the *later* node of a pair without its *earlier* one means the reader
    /// caught the publish mid-flight.
    #[test]
    fn concurrent_reads_never_observe_half_of_a_published_batch() {
        const ROUNDS: u64 = 400;

        let storage = Arc::new(CurrentStorage::new());
        storage.arm_apply_gate();
        let stop = Arc::new(AtomicBool::new(false));

        let publisher = {
            let (storage, stop) = (Arc::clone(&storage), Arc::clone(&stop));
            std::thread::spawn(move || {
                let ts = crate::core::temporal::time::now();
                for k in 0..ROUNDS {
                    let _publish = storage.begin_apply_publish();
                    storage
                        .insert_node_direct(node(2 * k + 1, "Person"), ts)
                        .unwrap();
                    std::thread::yield_now();
                    storage
                        .insert_node_direct(node(2 * k + 2, "Person"), ts)
                        .unwrap();
                }
                stop.store(true, Ordering::SeqCst);
            })
        };

        let readers: Vec<_> = (0..3)
            .map(|_| {
                let (storage, stop) = (Arc::clone(&storage), Arc::clone(&stop));
                std::thread::spawn(move || {
                    while !stop.load(Ordering::SeqCst) {
                        for k in 0..ROUNDS {
                            // Read in PUBLISH order. Ungated, a reader landing
                            // between the two inserts sees (true, false). Gated,
                            // `first` can only be true once the whole window has
                            // closed -- at which point `second`, read strictly
                            // later, must be true as well.
                            let first = storage.contains_node(2 * k + 1);
                            let second = storage.contains_node(2 * k + 2);
                            assert!(
                                !first || second,
                                "a gated read observed half of an atomically published batch \
                                 (round {k}: first present, second missing)"
                            );
                        }
                    }
                })
            })
            .collect();

        publisher.join().expect("publisher");
        for reader in readers {
            reader.join().expect("reader");
        }

        for k in 0..ROUNDS {
            assert!(storage.contains_node(2 * k + 1));
            assert!(storage.contains_node(2 * k + 2));
        }
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;

    #[test]
    fn test_vector_search_index_not_found() {
        let storage = CurrentStorage::new();
        // Don't enable vector index

        let node_id = NodeId::new(1).unwrap();
        // Try search
        let result = storage.find_similar(node_id, 5);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("Vector index is not enabled"));

        // Try search with specific property
        let result = storage.find_similar_in("embedding", node_id, 5);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("Vector index not found"));
    }
}

/// Issue #3621: WAL replay reruns `update_node_direct`, so the corrected
/// temporal de-indexing must reconstruct the right point-in-time state after a
/// crash-recovery replay. These tests drive `insert_node_direct` /
/// `update_node_direct` — the exact entry points recovery calls at
/// `storage/recovery.rs` (`current.insert_node_direct` / `current.update_node_direct`)
/// — with a temporal vector index enabled, then trigger snapshots via
/// `on_temporal_vector_transaction` (the same hook the commit path fires). A
/// true open→reopen test cannot cover this: the temporal vector index is an
/// in-memory index whose config is not persisted, so replay only repopulates it
/// when it is re-enabled beforehand — which is precisely what exercising the
/// replay entry points here simulates.
#[cfg(test)]
mod deindex_replay_tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;
    use crate::core::temporal::time;
    use crate::index::vector::DistanceMetric;

    fn temporal_config() -> TemporalVectorConfig {
        use crate::index::vector::temporal::{RetentionPolicy, SnapshotStrategy};
        TemporalVectorConfig {
            snapshot_strategy: SnapshotStrategy::TransactionInterval(1),
            retention_policy: RetentionPolicy::KeepN(100),
            max_snapshots: 100,
            full_snapshot_interval: 5,
            hnsw_config: Some(HnswConfig::new(4, DistanceMetric::Cosine)),
        }
    }

    fn node(id: u64, version: u64, props: PropertyMap) -> Node {
        let label = GLOBAL_INTERNER.intern("Doc").unwrap();
        Node::new(
            NodeId::new(id).unwrap(),
            label,
            props,
            VersionId::new(version).unwrap(),
        )
    }

    fn advance() {
        std::thread::sleep(std::time::Duration::from_millis(25));
    }

    /// Replaying a create followed by a vector-property removal through the
    /// direct (WAL-replay) methods leaves the temporal index de-indexed at the
    /// current time while an earlier snapshot still finds the vector.
    #[test]
    fn replay_remove_vector_deindexes_current_but_preserves_history() {
        let storage = CurrentStorage::new();
        storage
            .enable_temporal_vector_index("embedding", temporal_config())
            .unwrap();

        // Replay: create a node with an embedding (recovery.rs -> insert_node_direct).
        let target_id = NodeId::new(1).unwrap();
        let create_props = PropertyMapBuilder::new()
            .insert("name", "target")
            .insert_vector("embedding", &[1.0f32, 0.0, 0.0, 0.0])
            .build();
        let t1 = time::now();
        storage
            .insert_node_direct(node(1, 1, create_props), t1)
            .unwrap();
        storage.on_temporal_vector_transaction().unwrap();

        advance();
        let t_before_remove = time::now();
        advance();

        // Replay: update the node, dropping the embedding (recovery.rs -> update_node_direct).
        let removed_props = PropertyMapBuilder::new().insert("name", "target").build();
        let t2 = time::now();
        storage
            .update_node_direct(node(1, 2, removed_props), t2)
            .unwrap();
        storage.on_temporal_vector_transaction().unwrap();

        advance();
        let t_after_remove = time::now();

        let query = [1.0f32, 0.0, 0.0, 0.0];

        // History preserved: the pre-removal snapshot still finds the target.
        let historical = storage
            .find_similar_as_of(&query, 10, t_before_remove)
            .unwrap();
        assert!(
            historical.iter().any(|(id, _)| *id == target_id),
            "replay must preserve the pre-removal snapshot, got {historical:?}"
        );

        // De-indexed now: the post-removal snapshot must NOT return the phantom.
        let current = storage
            .find_similar_as_of(&query, 10, t_after_remove)
            .unwrap();
        assert!(
            !current.iter().any(|(id, _)| *id == target_id),
            "replay through update_node_direct must de-index the removed vector, got {current:?}"
        );
    }

    /// Replaying a create then an embedding overwrite reconstructs the new
    /// vector at the current time and the old vector as-of the earlier snapshot.
    ///
    /// Regression lock, not a fix discriminator: this `Some -> Some` overwrite
    /// already reconstructs correctly on trunk (the add-only temporal helper
    /// upserted on overwrite). The fix-discriminating replay guard is
    /// `replay_remove_vector_deindexes_current_but_preserves_history` above (the
    /// `Some -> None` removal that trunk's add-only helper leaked).
    #[test]
    fn replay_overwrite_vector_reconstructs_corrected_state() {
        let storage = CurrentStorage::new();
        storage
            .enable_temporal_vector_index("embedding", temporal_config())
            .unwrap();

        let target_id = NodeId::new(1).unwrap();
        let create_props = PropertyMapBuilder::new()
            .insert_vector("embedding", &[1.0f32, 0.0, 0.0, 0.0])
            .build();
        storage
            .insert_node_direct(node(1, 1, create_props), time::now())
            .unwrap();
        storage.on_temporal_vector_transaction().unwrap();

        advance();
        let t_before = time::now();
        advance();

        let overwrite_props = PropertyMapBuilder::new()
            .insert_vector("embedding", &[0.0f32, 1.0, 0.0, 0.0])
            .build();
        storage
            .update_node_direct(node(1, 2, overwrite_props), time::now())
            .unwrap();
        storage.on_temporal_vector_transaction().unwrap();

        advance();
        let t_after = time::now();

        // As-of before: old embedding ~identical to old query.
        let before = storage
            .find_similar_as_of(&[1.0f32, 0.0, 0.0, 0.0], 10, t_before)
            .unwrap();
        assert!(
            before
                .iter()
                .find(|(id, _)| *id == target_id)
                .is_some_and(|(_, s)| *s > 0.99),
            "replay must reconstruct the old embedding as-of before, got {before:?}"
        );

        // As-of after: new embedding ~identical to new query; old region no longer matches.
        let after_new = storage
            .find_similar_as_of(&[0.0f32, 1.0, 0.0, 0.0], 10, t_after)
            .unwrap();
        assert!(
            after_new
                .iter()
                .find(|(id, _)| *id == target_id)
                .is_some_and(|(_, s)| *s > 0.99),
            "replay must reconstruct the new embedding as-of after, got {after_new:?}"
        );
        let after_old = storage
            .find_similar_as_of(&[1.0f32, 0.0, 0.0, 0.0], 10, t_after)
            .unwrap();
        assert!(
            after_old
                .iter()
                .find(|(id, _)| *id == target_id)
                .is_none_or(|(_, s)| *s < 0.1),
            "replay must drop the old embedding region after overwrite, got {after_old:?}"
        );
    }

    /// Regression for the Gemini review on PR #3632: an `UpdateNode` replayed
    /// against a node ABSENT from the (lagging) current-state index — the
    /// `old_props == None` create-through-update branch of `update_node_direct`
    /// — must index the vector into the current-state HNSW too, not only the
    /// temporal index. Otherwise the vector is invisible to current-state
    /// `find_similar` after recovery. Mirrors `insert_node_direct`, which
    /// indexes both.
    #[test]
    fn update_create_path_indexes_current_state_hnsw() {
        let storage = CurrentStorage::new();
        // Current-state (non-temporal) vector index enabled; temporal one too,
        // to exercise the exact fan-out the create-through-update branch runs.
        storage
            .enable_vector_index("embedding", HnswConfig::new(4, DistanceMetric::Cosine))
            .unwrap();
        storage
            .enable_temporal_vector_index("embedding", temporal_config())
            .unwrap();

        // Drive update_node_direct on a node that was NEVER inserted, so
        // old_props resolves to None (the create-through-update / lagging-replay
        // path). This is exactly `current.update_node_direct(...)` at
        // recovery.rs when the create was already truncated past the snapshot.
        let target_id = NodeId::new(1).unwrap();
        let props = PropertyMapBuilder::new()
            .insert("name", "target")
            .insert_vector("embedding", &[1.0f32, 0.0, 0.0, 0.0])
            .build();
        storage
            .update_node_direct(node(1, 1, props), time::now())
            .unwrap();

        // The node must now be findable via the CURRENT-STATE HNSW.
        let results = storage
            .find_similar_by_embedding(&[1.0f32, 0.0, 0.0, 0.0], 10)
            .unwrap();
        assert!(
            results.iter().any(|(id, _)| *id == target_id),
            "create-through-update must index the current-state HNSW, got {results:?}"
        );
    }
}
