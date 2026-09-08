//! Historical storage engine with anchor+delta compression.
//!
//! This module implements temporal versioning using version chains. Each node
//! and edge can have multiple versions over time, linked together in a chain
//! ordered by transaction time.
//!
//! The anchor+delta strategy minimizes storage overhead:
//! - Anchors are created every N versions (configurable)
//! - Deltas store only changed properties
//! - Reconstruction walks backward to nearest anchor and applies deltas forward
//! - TinyLFU cache reduces redundant delta chain traversals for concurrent reads

use crate::core::changefeed::{
    BoundedChanges, ChangeCursor, EntityKind, RawChange, build_raw_change, consider_version,
};
use crate::core::error::{Result, StorageError, TemporalError};
use crate::core::graph::{Edge, Node};
use crate::core::history::{EntityHistory, VersionDiff, VersionInfo};
use crate::core::id::{EdgeId, NodeId, VersionId};
use crate::core::interning::{GLOBAL_INTERNER, InternedString};
use crate::core::namespace::{
    NamespaceId, intern_namespace, namespace_of, unresolved_namespace_id,
};
use crate::core::observer::{Observer, StorageEvent, notify_observers};
use crate::core::property::PropertyMap;
use crate::core::provenance::Provenance;
use crate::core::temporal::{BiTemporalInterval, TIMESTAMP_MAX, TimeRange, Timestamp};
use crate::core::version::{
    AnchorConfig, EdgeVersion, EntityVersion, FastHashMap, NodeVersion, TemporalVersion,
    VersionData,
};
use quick_cache::sync::Cache;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[cfg(feature = "observability")]
use tracing;

mod hooks;
mod snapshot_policy;

use hooks::{AnchorHookContext, HookMetrics};
pub use hooks::{HookMetricsSnapshot, PreAnchorHook};
pub use snapshot_policy::SnapshotPolicy;
use snapshot_policy::SnapshotPolicyRegistry;

/// Default maximum number of versions per entity (DoS protection)
pub const DEFAULT_MAX_VERSIONS_PER_ENTITY: usize = 1_000;

/// Default maximum age for versions in milliseconds (365 days)
pub const DEFAULT_MAX_VERSION_AGE_MS: i64 = 365 * 24 * 60 * 60 * 1000;

/// Maximum recursion depth for version reconstruction (DoS protection).
///
/// This limit prevents stack overflow from corrupted version chains or cycles.
/// Increased from 100 to 1,000 to support business scenarios:
/// - High-update entities: Stock prices, sensor data, real-time feeds
/// - Long-running systems without compaction
/// - 1,000 deltas enables longer operational periods before compaction
///
///   Still provides infinite loop protection while enabling practical use cases.
pub const MAX_RECONSTRUCTION_DEPTH: usize = 1_000;

/// Default safety cap on the number of ever-versioned entities
/// `AletheiaDB::schema_as_of` will reconstruct in a single call, per entity
/// kind (nodes/edges). See [`crate::config::HistoricalConfigBuilder::max_schema_as_of_entities`]
/// to override this.
pub const DEFAULT_MAX_SCHEMA_AS_OF_ENTITIES: usize = 50_000;

/// Retention policy for version history (DoS protection).
///
/// Controls how many versions are kept and for how long to prevent
/// unbounded memory growth from malicious or buggy clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetentionPolicy {
    /// Maximum number of versions to keep per entity
    pub max_versions_per_entity: usize,
    /// Maximum age of versions in milliseconds (older versions are pruned)
    pub max_age_ms: i64,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        RetentionPolicy {
            max_versions_per_entity: DEFAULT_MAX_VERSIONS_PER_ENTITY,
            max_age_ms: DEFAULT_MAX_VERSION_AGE_MS,
        }
    }
}

impl RetentionPolicy {
    /// Create a new retention policy with custom limits
    pub fn new(max_versions_per_entity: usize, max_age_ms: i64) -> Self {
        RetentionPolicy {
            max_versions_per_entity,
            max_age_ms,
        }
    }

    /// Create a policy with no retention limits (unbounded)
    pub fn unbounded() -> Self {
        RetentionPolicy {
            max_versions_per_entity: usize::MAX,
            max_age_ms: i64::MAX,
        }
    }
}

/// Default cache size for reconstructed properties (10,000 entries)
const DEFAULT_RECONSTRUCTION_CACHE_SIZE: usize = 10_000;

/// Anchor cache size ratio relative to main cache (Improvement #1: Issue #338).
///
/// Typically 10-20% of versions become anchors depending on `anchor_interval`.
/// With default interval of 10, we get ~10% anchors. Setting to 1/5 (20%)
/// provides headroom for configurations with smaller intervals.
const ANCHOR_CACHE_SIZE_RATIO: usize = 5; // 20% of main cache

/// Minimum anchor cache size to ensure reasonable performance (Improvement #1: Issue #338).
///
/// Even with very small main caches, we want enough anchor cache to hold
/// at least a few anchors to avoid immediate evictions.
const MIN_ANCHOR_CACHE_SIZE: usize = 100;

/// Default average delta chain length estimate used when historical storage
/// is empty (Issue #366).
///
/// Assumes the default `anchor_interval` of 10 (one anchor followed by up to
/// nine deltas), matching the query planner's fallback in
/// [`Statistics::average_delta_chain_length`](crate::query::planner::Statistics::average_delta_chain_length).
pub const DEFAULT_AVG_DELTA_CHAIN: f64 = 5.0;

/// Historical storage for versioned nodes and edges.
///
/// This storage engine maintains version chains for all temporal data,
/// using anchor+delta compression to minimize storage overhead.
/// A TinyLFU cache reduces redundant delta chain traversals for concurrent reads.
pub struct HistoricalStorage {
    /// Configuration for anchor creation strategy
    config: AnchorConfig,
    /// Retention policy for version pruning (DoS protection)
    retention_policy: RetentionPolicy,
    /// Maximum depth for version reconstruction (DoS protection)
    max_reconstruction_depth: usize,
    /// Safety cap (per entity kind) on the number of ever-versioned entities
    /// `AletheiaDB::schema_as_of` will reconstruct in a single call.
    max_schema_as_of_entities: usize,
    // All FastHashMap fields below use IdentityHasher to avoid SipHash overhead
    // on integer keys (NodeId, EdgeId, VersionId are all newtypes over u64).
    /// All node versions, indexed by version ID.
    node_versions: FastHashMap<VersionId, NodeVersion>,
    /// All edge versions, indexed by version ID.
    edge_versions: FastHashMap<VersionId, EdgeVersion>,
    /// Head version ID for each node (most recent).
    node_version_heads: FastHashMap<NodeId, VersionId>,
    /// Head version ID for each edge (most recent).
    edge_version_heads: FastHashMap<EdgeId, VersionId>,
    /// Cached version counts per node (for O(1) capacity checks).
    node_version_counts: FastHashMap<NodeId, usize>,
    /// Cached version counts per edge (for O(1) capacity checks).
    edge_version_counts: FastHashMap<EdgeId, usize>,
    /// Versions since last anchor per node (for O(1) anchor interval checks).
    /// Avoids walking the version chain on every add operation.
    node_versions_since_anchor: FastHashMap<NodeId, usize>,
    /// Versions since last anchor per edge (for O(1) anchor interval checks).
    /// Avoids walking the version chain on every add operation.
    edge_versions_since_anchor: FastHashMap<EdgeId, usize>,
    /// Cached anchor/delta counts for O(1) stats() retrieval.
    /// Maintained incrementally as versions are added/migrated.
    cached_node_anchor_count: usize,
    cached_node_delta_count: usize,
    cached_edge_anchor_count: usize,
    cached_edge_delta_count: usize,
    /// TinyLFU cache for reconstructed node properties (reduces lock contention)
    node_property_cache: Arc<Cache<VersionId, Arc<PropertyMap>>>,
    /// TinyLFU cache for reconstructed edge properties
    edge_property_cache: Arc<Cache<VersionId, Arc<PropertyMap>>>,
    /// Improvement #1: Dedicated cache for node anchor properties.
    ///
    /// This separate cache ensures anchors are never evicted by delta cache pressure,
    /// providing guaranteed O(1) access to anchors even under heavy load. Anchors are
    /// frequently reused as base points for delta reconstruction, so keeping them cached
    /// reduces average reconstruction cost from O(N) to O(M) where M << N.
    node_anchor_cache: Arc<Cache<VersionId, Arc<PropertyMap>>>,
    /// Improvement #1: Dedicated cache for edge anchor properties.
    ///
    /// This separate cache ensures anchors are never evicted by delta cache pressure,
    /// providing guaranteed O(1) access to anchors even under heavy load.
    edge_anchor_cache: Arc<Cache<VersionId, Arc<PropertyMap>>>,
    /// Improvement #3: Primary cache hit counter for adaptive sizing.
    ///
    /// Tracks successful lookups in the primary property cache (fast path).
    /// This is the most common case for recently accessed properties.
    primary_cache_hits: Arc<AtomicU64>,
    /// Improvement #3: Anchor cache hit counter for adaptive sizing.
    ///
    /// Tracks successful lookups in the dedicated anchor cache (fallback path).
    /// High values indicate anchors are being evicted from primary cache under
    /// delta pressure, suggesting the primary cache may need to be larger.
    anchor_cache_hits: Arc<AtomicU64>,
    /// Improvement #3: Full reconstruction counter for adaptive sizing.
    ///
    /// Tracks cache misses requiring full property reconstruction from deltas.
    /// High values indicate insufficient cache capacity overall.
    full_reconstructions: Arc<AtomicU64>,
    /// Observers subscribed to storage events
    ///
    /// Multiple components can observe storage events (anchors, deletes, etc.)
    /// for indexing, metrics, logging, or coordination. Observers are notified
    /// asynchronously and errors don't block storage operations.
    observers: Vec<Observer>,
    /// Ordered pre-anchor hooks for node anchors (called before storage).
    ///
    /// These hooks are called **before** storing a node anchor, in registration
    /// order, to create synchronized snapshots. The returned snapshot ID is
    /// stored atomically with the anchor, enabling strong consistency for
    /// provenance tracking. A failure, panic, or timeout of one hook is isolated
    /// and does not prevent the others from running (Issue #3525).
    pre_node_anchor_hooks: Vec<PreAnchorHook>,
    /// Ordered pre-anchor hooks for edge anchors (called before storage).
    ///
    /// These hooks are called **before** storing an edge anchor, in registration
    /// order, to create synchronized snapshots. The returned snapshot ID is
    /// stored atomically with the anchor, enabling strong consistency for
    /// provenance tracking. A failure, panic, or timeout of one hook is isolated
    /// and does not prevent the others from running (Issue #3525).
    pre_edge_anchor_hooks: Vec<PreAnchorHook>,
    /// Optional bound on how long the write path waits for a single pre-anchor
    /// hook before degrading gracefully (Issue #3525).
    ///
    /// `None` (default) runs hooks inline under the `historical` lock exactly as
    /// before (panics still isolated). `Some(d)` runs each hook on a detached
    /// worker thread and waits at most `d`; if the deadline elapses the write
    /// path stops waiting, records a timeout, and creates the anchor without a
    /// snapshot id from that hook. See [`hooks`] for the locking rationale.
    hook_timeout: Option<Duration>,
    /// Observability counters for pre-anchor hook execution (Issue #3525).
    hook_metrics: HookMetrics,
    /// Per-node temporal vector snapshot policy (Issue #383).
    ///
    /// Consulted at node-anchor creation to decide whether that entity's anchor
    /// should trigger the pre-anchor snapshot hooks. Entities resolving to
    /// [`SnapshotPolicy::Skip`] form graph anchors normally but do **not** run
    /// the snapshot hooks, decoupling graph anchors from vector snapshots. The
    /// default policy is [`SnapshotPolicy::Snapshot`], reproducing the pre-#383
    /// global behavior when no per-entity policy is configured.
    node_snapshot_policies: SnapshotPolicyRegistry<NodeId>,
    /// Per-edge temporal vector snapshot policy (Issue #383).
    ///
    /// The edge counterpart of [`node_snapshot_policies`](Self::node_snapshot_policies);
    /// resolved independently at edge-anchor creation.
    edge_snapshot_policies: SnapshotPolicyRegistry<EdgeId>,
    /// Optional tiered storage for cold data access.
    ///
    /// When configured, versions not found in hot storage will be looked up
    /// from cold storage via the tiered storage layer.
    ///
    /// The cold/tiered subsystem (redb-backed) is unavailable on `wasm32`, so
    /// this field is elided there — the wasm profile is hot-only.
    #[cfg(not(target_arch = "wasm32"))]
    tiered_storage: Option<Arc<super::tiered_storage::TieredStorage>>,
    /// Temporal indexes for O(log n) version lookups (Issue #209).
    ///
    /// When available, `find_node_version_at_time` and `find_edge_version_at_time`
    /// use these indexes for efficient binary search instead of O(n) linear scans
    /// through version chains. This is particularly important for entities with
    /// long version histories (100s-1000s of versions).
    ///
    /// The temporal indexes are maintained externally by the database and shared
    /// with HistoricalStorage for query optimization.
    temporal_indexes: Option<Arc<crate::index::temporal::TemporalIndexes>>,
    /// Temporal adjacency index for fast temporal graph traversal.
    ///
    /// When available, pathfinding queries can efficiently find edges that existed
    /// at specific points in time, including edges that have been deleted.
    temporal_adjacency_index: Option<Arc<crate::index::temporal_adjacency::TemporalAdjacencyIndex>>,
}

impl HistoricalStorage {
    /// Create a new historical storage with default configuration.
    pub fn new() -> Self {
        Self::with_config(AnchorConfig::default())
    }

    /// Create a new historical storage with custom configuration.
    pub fn with_config(config: AnchorConfig) -> Self {
        Self::with_config_and_retention(config, RetentionPolicy::default())
    }

    /// Create a new historical storage with custom configuration and retention policy.
    pub fn with_config_and_retention(
        config: AnchorConfig,
        retention_policy: RetentionPolicy,
    ) -> Self {
        Self::with_config_retention_and_cache_size(
            config,
            retention_policy,
            DEFAULT_RECONSTRUCTION_CACHE_SIZE,
        )
    }

    /// Create a new historical storage from unified configuration.
    ///
    /// This constructor accepts `crate::config::HistoricalConfig` which consolidates
    /// all historical storage settings (max_versions, max_reconstruction_depth, cache_size).
    ///
    /// # Example
    /// ```ignore
    /// use aletheiadb::config::{HistoricalConfig, HistoricalConfigBuilder};
    /// use aletheiadb::storage::historical::HistoricalStorage;
    ///
    /// let config = HistoricalConfigBuilder::new()
    ///     .max_versions_per_entity(5000).unwrap()
    ///     .max_reconstruction_depth(200).unwrap()
    ///     .reconstruction_cache_size(20000).unwrap()
    ///     .build();
    ///
    /// let storage = HistoricalStorage::from_unified_config(config);
    /// ```
    pub fn from_unified_config(config: crate::config::HistoricalConfig) -> Self {
        let retention_policy = RetentionPolicy::new(
            config.max_versions_per_entity,
            DEFAULT_MAX_VERSION_AGE_MS, // Keep existing default for age
        );

        let anchor_config = AnchorConfig {
            anchor_interval: config.anchor_interval,
            max_delta_chain: config.max_delta_chain,
        };

        let mut storage = Self::with_config_retention_and_cache_size(
            anchor_config,
            retention_policy,
            config.reconstruction_cache_size,
        );

        // Override max_reconstruction_depth from config
        storage.max_reconstruction_depth = config.max_reconstruction_depth;
        storage.max_schema_as_of_entities = config.max_schema_as_of_entities;

        storage
    }

    /// Create a new historical storage with full customization including cache size.
    ///
    /// # Arguments
    /// * `config` - Anchor creation configuration
    /// * `retention_policy` - Version retention limits (DoS protection)
    /// * `cache_size` - Maximum number of cached property reconstructions per type (node/edge)
    ///
    /// # Cache Sizing
    /// Consider your workload when sizing the cache:
    /// - Small properties (no vectors): 1,000-10,000 entries
    /// - Large properties (1536-dim vectors ~6KB): 100-1,000 entries
    /// - Default: 10,000 entries
    pub fn with_config_retention_and_cache_size(
        config: AnchorConfig,
        retention_policy: RetentionPolicy,
        cache_size: usize,
    ) -> Self {
        // Calculate anchor cache size: typically 10-20% of entities become anchors
        // depending on anchor_interval (Improvement #1: Issue #338)
        let anchor_cache_size = (cache_size / ANCHOR_CACHE_SIZE_RATIO).max(MIN_ANCHOR_CACHE_SIZE);

        HistoricalStorage {
            config,
            retention_policy,
            max_reconstruction_depth: MAX_RECONSTRUCTION_DEPTH,
            max_schema_as_of_entities: DEFAULT_MAX_SCHEMA_AS_OF_ENTITIES,
            node_versions: FastHashMap::default(),
            edge_versions: FastHashMap::default(),
            node_version_heads: FastHashMap::default(),
            edge_version_heads: FastHashMap::default(),
            node_version_counts: FastHashMap::default(),
            edge_version_counts: FastHashMap::default(),
            node_versions_since_anchor: FastHashMap::default(),
            edge_versions_since_anchor: FastHashMap::default(),
            cached_node_anchor_count: 0,
            cached_node_delta_count: 0,
            cached_edge_anchor_count: 0,
            cached_edge_delta_count: 0,
            node_property_cache: Arc::new(Cache::new(cache_size)),
            edge_property_cache: Arc::new(Cache::new(cache_size)),
            node_anchor_cache: Arc::new(Cache::new(anchor_cache_size)),
            edge_anchor_cache: Arc::new(Cache::new(anchor_cache_size)),
            primary_cache_hits: Arc::new(AtomicU64::new(0)),
            anchor_cache_hits: Arc::new(AtomicU64::new(0)),
            full_reconstructions: Arc::new(AtomicU64::new(0)),
            observers: Vec::new(),
            pre_node_anchor_hooks: Vec::new(),
            pre_edge_anchor_hooks: Vec::new(),
            hook_timeout: None,
            hook_metrics: HookMetrics::default(),
            node_snapshot_policies: SnapshotPolicyRegistry::default(),
            edge_snapshot_policies: SnapshotPolicyRegistry::default(),
            #[cfg(not(target_arch = "wasm32"))]
            tiered_storage: None,
            temporal_indexes: None,
            temporal_adjacency_index: None,
        }
    }

    /// Add an observer to receive storage events.
    ///
    /// Observers are notified of storage events (anchors, deletes, etc.) for indexing,
    /// metrics, logging, or coordination. Multiple observers can be registered, and all
    /// will be notified of events they're interested in.
    ///
    /// # Arguments
    /// * `observer` - Component implementing the StorageObserver trait
    ///
    /// # Example
    /// ```no_run
    /// # use aletheiadb::storage::historical::HistoricalStorage;
    /// # use aletheiadb::core::observer::{StorageObserver, StorageEvent};
    /// # use std::sync::Arc;
    /// struct VectorIndexObserver;
    ///
    /// impl StorageObserver for VectorIndexObserver {
    ///     fn on_event(&self, event: &StorageEvent) -> aletheiadb::core::error::Result<()> {
    ///         match event {
    ///             StorageEvent::NodeAnchorCreated { version_id, timestamp, .. } => {
    ///                 println!("Anchor {} created at {}", version_id, timestamp);
    ///                 Ok(())
    ///             }
    ///             _ => Ok(())
    ///         }
    ///     }
    /// }
    ///
    /// let mut storage = HistoricalStorage::new();
    /// storage.add_observer(Arc::new(VectorIndexObserver));
    /// ```
    pub fn add_observer(&mut self, observer: Observer) {
        self.observers.push(observer);
    }

    /// Register a pre-anchor hook for nodes.
    ///
    /// This hook is called **before** storing a node anchor, allowing the hook
    /// to create synchronized snapshots and return a snapshot ID to be stored
    /// atomically with the anchor. This enables strong consistency for provenance
    /// tracking between graph versioning and vector indexing.
    ///
    /// # Arguments
    /// * `hook` - Function that creates snapshots and returns snapshot IDs
    ///
    /// # Hook Signature
    /// The hook receives:
    /// * `entity_type` - Always "node" for this hook
    /// * `entity_id` - Node ID as u64
    /// * `timestamp` - Transaction timestamp
    /// * `properties` - Property map being stored in the anchor
    ///
    /// The hook returns:
    /// * `Ok(Some(snapshot_id))` - Snapshot created, link to anchor
    /// * `Ok(None)` - No snapshot needed (e.g., empty index)
    /// * `Err(e)` - Error (anchor still created, graceful degradation)
    ///
    /// # Example
    /// ```ignore
    /// use aletheiadb::storage::historical::PreAnchorHook;
    /// use std::sync::Arc;
    ///
    /// let hook: PreAnchorHook = Arc::new(|_entity_type, _entity_id, timestamp, _properties| {
    ///     // Create vector snapshot before anchor storage
    ///     let snapshot_id = temporal_vector_index.create_snapshot_for_anchor(timestamp)?;
    ///     Ok(snapshot_id)
    /// });
    ///
    /// let mut storage = HistoricalStorage::new();
    /// storage.register_pre_node_anchor_hook(hook);
    /// ```
    ///
    /// # Backward compatibility (Issue #3525)
    ///
    /// This setter **replaces** all currently registered node pre-anchor hooks
    /// with the single `hook`, preserving the original single-hook "set THE
    /// hook" semantics. To register multiple ordered hooks without discarding
    /// existing ones, use [`add_pre_node_anchor_hook`](Self::add_pre_node_anchor_hook).
    pub fn register_pre_node_anchor_hook(&mut self, hook: PreAnchorHook) {
        self.pre_node_anchor_hooks = vec![hook];
    }

    /// Append a pre-anchor hook for nodes, preserving registration order (Issue #3525).
    ///
    /// Unlike [`register_pre_node_anchor_hook`](Self::register_pre_node_anchor_hook)
    /// (which replaces), this **appends** `hook` after any previously registered
    /// node hooks. When an anchor is created, all registered node hooks run in
    /// registration order. Each invocation is isolated: a failure, panic, or
    /// timeout of one hook is recorded and logged but does not prevent later
    /// hooks from running. Because an anchor has a single snapshot-id slot, the
    /// **last** hook that returns `Ok(Some(id))` wins.
    pub fn add_pre_node_anchor_hook(&mut self, hook: PreAnchorHook) {
        self.pre_node_anchor_hooks.push(hook);
    }

    /// Register a pre-anchor hook for edges.
    ///
    /// This hook is called **before** storing an edge anchor, allowing the hook
    /// to create synchronized snapshots and return a snapshot ID to be stored
    /// atomically with the anchor. This enables strong consistency for provenance
    /// tracking between graph versioning and vector indexing.
    ///
    /// # Arguments
    /// * `hook` - Function that creates snapshots and returns snapshot IDs
    ///
    /// # Hook Signature
    /// The hook receives:
    /// * `entity_type` - Always "edge" for this hook
    /// * `entity_id` - Edge ID as u64
    /// * `timestamp` - Transaction timestamp
    /// * `properties` - Property map being stored in the anchor
    ///
    /// The hook returns:
    /// * `Ok(Some(snapshot_id))` - Snapshot created, link to anchor
    /// * `Ok(None)` - No snapshot needed (e.g., empty index)
    /// * `Err(e)` - Error (anchor still created, graceful degradation)
    ///
    /// # Example
    /// ```ignore
    /// use aletheiadb::storage::historical::PreAnchorHook;
    /// use std::sync::Arc;
    ///
    /// let hook: PreAnchorHook = Arc::new(|_entity_type, _entity_id, timestamp, _properties| {
    ///     // Create vector snapshot before anchor storage
    ///     let snapshot_id = temporal_vector_index.create_snapshot_for_anchor(timestamp)?;
    ///     Ok(snapshot_id)
    /// });
    ///
    /// let mut storage = HistoricalStorage::new();
    /// storage.register_pre_edge_anchor_hook(hook);
    /// ```
    ///
    /// # Backward compatibility (Issue #3525)
    ///
    /// This setter **replaces** all currently registered edge pre-anchor hooks
    /// with the single `hook`, preserving the original single-hook "set THE
    /// hook" semantics. To register multiple ordered hooks without discarding
    /// existing ones, use [`add_pre_edge_anchor_hook`](Self::add_pre_edge_anchor_hook).
    pub fn register_pre_edge_anchor_hook(&mut self, hook: PreAnchorHook) {
        self.pre_edge_anchor_hooks = vec![hook];
    }

    /// Append a pre-anchor hook for edges, preserving registration order (Issue #3525).
    ///
    /// Unlike [`register_pre_edge_anchor_hook`](Self::register_pre_edge_anchor_hook)
    /// (which replaces), this **appends** `hook` after any previously registered
    /// edge hooks. See [`add_pre_node_anchor_hook`](Self::add_pre_node_anchor_hook)
    /// for the multi-hook ordering and partial-failure semantics.
    pub fn add_pre_edge_anchor_hook(&mut self, hook: PreAnchorHook) {
        self.pre_edge_anchor_hooks.push(hook);
    }

    /// Set the bound on how long the write path waits for a single pre-anchor
    /// hook before degrading gracefully (Issue #3525).
    ///
    /// * `None` (default) runs hooks **inline** under the `historical` lock,
    ///   exactly as before — identical lock-hold profile, with panics isolated.
    /// * `Some(timeout)` runs each hook on a detached worker thread and waits at
    ///   most `timeout`. If the deadline elapses the write path stops waiting,
    ///   records a timeout in [`hook_metrics`](Self::hook_metrics), and creates
    ///   the anchor without a snapshot id from that hook.
    ///
    /// # Locking / cancellation
    ///
    /// A configured timeout **bounds** the time the `historical` lock is held
    /// across a hung hook (previously unbounded), so it never holds the lock
    /// longer than before. Rust cannot safely cancel a running closure, so a
    /// timed-out hook thread is detached and may run to completion in the
    /// background; "timeout" means the write path stops waiting, not that the
    /// hook is killed.
    pub fn set_pre_anchor_hook_timeout(&mut self, timeout: Option<Duration>) {
        self.hook_timeout = timeout;
    }

    /// Return the currently configured pre-anchor hook timeout, if any (Issue #3525).
    pub fn pre_anchor_hook_timeout(&self) -> Option<Duration> {
        self.hook_timeout
    }

    /// Return a point-in-time snapshot of pre-anchor hook execution metrics
    /// (invocations, successes, failures, panics, timeouts) — the observability
    /// surface for graceful degradation (Issue #3525).
    pub fn hook_metrics(&self) -> HookMetricsSnapshot {
        self.hook_metrics.snapshot()
    }

    // --- Per-entity temporal vector snapshot policy (Issue #383) ---------------
    //
    // These control, per node/edge, whether that entity's anchor triggers the
    // pre-anchor snapshot hooks (registered via the #3525 multi-hook API).
    // Replacing the single global anchor hook, they let production deployments
    // snapshot only the vectors that actually change instead of re-snapshotting
    // the whole index on every anchor. The default policy is
    // [`SnapshotPolicy::Snapshot`], so a database that configures nothing behaves
    // exactly as before Issue #383.

    /// Set the temporal vector snapshot policy for a specific node (Issue #383).
    ///
    /// A [`SnapshotPolicy::Skip`] node still forms graph anchors normally but no
    /// longer runs the pre-anchor snapshot hooks, so no vector snapshot is
    /// captured for its anchors. Overrides the default policy for this node
    /// until [`clear_node_snapshot_policy`](Self::clear_node_snapshot_policy) is
    /// called.
    pub fn set_node_snapshot_policy(&mut self, node_id: NodeId, policy: SnapshotPolicy) {
        self.node_snapshot_policies.set(node_id, policy);
    }

    /// Set the temporal vector snapshot policy for a specific edge (Issue #383).
    ///
    /// The edge counterpart of
    /// [`set_node_snapshot_policy`](Self::set_node_snapshot_policy).
    pub fn set_edge_snapshot_policy(&mut self, edge_id: EdgeId, policy: SnapshotPolicy) {
        self.edge_snapshot_policies.set(edge_id, policy);
    }

    /// Remove any per-node snapshot policy override, reverting the node to the
    /// current default node policy (Issue #383). Returns the removed override,
    /// if any.
    pub fn clear_node_snapshot_policy(&mut self, node_id: NodeId) -> Option<SnapshotPolicy> {
        self.node_snapshot_policies.clear(node_id)
    }

    /// Remove any per-edge snapshot policy override, reverting the edge to the
    /// current default edge policy (Issue #383). Returns the removed override,
    /// if any.
    pub fn clear_edge_snapshot_policy(&mut self, edge_id: EdgeId) -> Option<SnapshotPolicy> {
        self.edge_snapshot_policies.clear(edge_id)
    }

    /// Set the fall-through default snapshot policy for **nodes** without an
    /// explicit override (Issue #383).
    ///
    /// Flipping this to [`SnapshotPolicy::Skip`] gives an opt-in model: no node
    /// is snapshotted unless individually set to [`SnapshotPolicy::Snapshot`].
    pub fn set_default_node_snapshot_policy(&mut self, policy: SnapshotPolicy) {
        self.node_snapshot_policies.set_default(policy);
    }

    /// Set the fall-through default snapshot policy for **edges** without an
    /// explicit override (Issue #383).
    pub fn set_default_edge_snapshot_policy(&mut self, policy: SnapshotPolicy) {
        self.edge_snapshot_policies.set_default(policy);
    }

    /// Return the current default node snapshot policy (Issue #383).
    pub fn default_node_snapshot_policy(&self) -> SnapshotPolicy {
        self.node_snapshot_policies.default_policy()
    }

    /// Return the current default edge snapshot policy (Issue #383).
    pub fn default_edge_snapshot_policy(&self) -> SnapshotPolicy {
        self.edge_snapshot_policies.default_policy()
    }

    /// Resolve the effective snapshot policy for a node: its explicit override
    /// if set, otherwise the default node policy (Issue #383).
    pub fn node_snapshot_policy(&self, node_id: NodeId) -> SnapshotPolicy {
        self.node_snapshot_policies.resolve(node_id)
    }

    /// Resolve the effective snapshot policy for an edge: its explicit override
    /// if set, otherwise the default edge policy (Issue #383).
    pub fn edge_snapshot_policy(&self, edge_id: EdgeId) -> SnapshotPolicy {
        self.edge_snapshot_policies.resolve(edge_id)
    }

    /// Whether a node's anchor should trigger a temporal vector snapshot
    /// (Issue #383).
    ///
    /// This is the single per-entity decision consulted by BOTH anchor-driven
    /// snapshot triggers so they can never diverge:
    /// 1. the **pre-anchor hook** run in `add_node_version*` (returns a snapshot
    ///    id stored on the anchor), and
    /// 2. the **`NodeAnchorCreated` observer event** delivered right after the
    ///    anchor is stored, which the [`VectorIndexObserver`] reacts to by
    ///    calling `create_snapshot_for_anchor`.
    ///
    /// A `Skip` node still forms its graph anchor normally; only these two
    /// vector-snapshot triggers are suppressed. The default policy is
    /// `Snapshot`, so absent any per-entity configuration this is always `true`
    /// — byte-identical to the pre-#383 behavior.
    ///
    /// [`VectorIndexObserver`]: crate::index::vector::temporal::VectorIndexObserver
    #[inline]
    fn node_anchor_triggers_vector_snapshot(&self, node_id: NodeId) -> bool {
        self.node_snapshot_policies
            .resolve(node_id)
            .should_snapshot()
    }

    /// Edge counterpart of
    /// [`node_anchor_triggers_vector_snapshot`](Self::node_anchor_triggers_vector_snapshot)
    /// (Issue #383). Gates both the edge pre-anchor hook and the
    /// `EdgeAnchorCreated` observer event on the shared per-edge policy.
    #[inline]
    fn edge_anchor_triggers_vector_snapshot(&self, edge_id: EdgeId) -> bool {
        self.edge_snapshot_policies
            .resolve(edge_id)
            .should_snapshot()
    }

    /// Add a new version of a node.
    ///
    /// This will automatically determine whether to create an anchor or delta
    /// based on the version chain length.
    /// Returns an error if the version limit for this entity is exceeded (DoS protection).
    ///
    /// Equivalent to [`add_node_version_with_provenance`](Self::add_node_version_with_provenance)
    /// with `provenance: None`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_node_version(
        &mut self,
        node_id: NodeId,
        version_id: VersionId,
        valid_from: Timestamp,
        tx_time: Timestamp,
        label: InternedString,
        properties: PropertyMap,
        is_tombstone: bool,
    ) -> Result<()> {
        self.add_node_version_with_provenance(
            node_id,
            version_id,
            valid_from,
            tx_time,
            label,
            properties,
            is_tombstone,
            None,
        )
    }

    /// Add a new version of a node, optionally attaching a write-time
    /// [`Provenance`] bundle (Issue #3224).
    ///
    /// Behaves identically to [`add_node_version`](Self::add_node_version) other
    /// than persisting `provenance` on the created version (anchor or delta).
    #[allow(clippy::too_many_arguments)]
    pub fn add_node_version_with_provenance(
        &mut self,
        node_id: NodeId,
        version_id: VersionId,
        valid_from: Timestamp,
        tx_time: Timestamp,
        label: InternedString,
        properties: PropertyMap,
        is_tombstone: bool,
        provenance: Option<Arc<Provenance>>,
    ) -> Result<()> {
        // Construct bi-temporal interval from separate dimensions
        let mut temporal = BiTemporalInterval::with_valid_time(valid_from, tx_time);

        // For tombstones, close the valid_time at valid_from to create an empty interval [valid_from, valid_from)
        // This represents "entity is no longer valid starting from this point"
        if is_tombstone {
            temporal = temporal.close_valid_time(valid_from)?;
        }

        self.add_node_version_with_interval(
            node_id, version_id, temporal, label, properties, provenance,
        )
    }

    /// Add a *retraction* version of a node (Issue #3230): a real (non-tombstone)
    /// version whose valid-time interval is closed at `valid_to` —
    /// `[valid_from, valid_to)` — recorded at transaction time `tx_time`.
    ///
    /// Unlike a delete tombstone (whose valid interval is empty,
    /// `[valid_from, valid_from)`), a retraction preserves the fact's
    /// pre-`valid_to` validity: `AS OF VALID_TIME` queries strictly before
    /// `valid_to` continue to see the entity.
    ///
    /// `valid_to == valid_from` is allowed and yields an empty interval.
    #[allow(clippy::too_many_arguments)]
    pub fn add_retracted_node_version(
        &mut self,
        node_id: NodeId,
        version_id: VersionId,
        valid_from: Timestamp,
        valid_to: Timestamp,
        tx_time: Timestamp,
        label: InternedString,
        properties: PropertyMap,
    ) -> Result<()> {
        self.add_retracted_node_version_with_provenance(
            node_id, version_id, valid_from, valid_to, tx_time, label, properties, None,
        )
    }

    /// Add a *retraction* version of a node (Issue #3230), optionally attaching a
    /// write-time [`Provenance`] bundle
    /// recording the acting principal (Issue #3427).
    ///
    /// Behaves identically to
    /// [`add_retracted_node_version`](Self::add_retracted_node_version) other
    /// than persisting `provenance` on the created retraction version.
    #[allow(clippy::too_many_arguments)]
    pub fn add_retracted_node_version_with_provenance(
        &mut self,
        node_id: NodeId,
        version_id: VersionId,
        valid_from: Timestamp,
        valid_to: Timestamp,
        tx_time: Timestamp,
        label: InternedString,
        properties: PropertyMap,
        provenance: Option<Arc<Provenance>>,
    ) -> Result<()> {
        let temporal =
            BiTemporalInterval::with_valid_time(valid_from, tx_time).close_valid_time(valid_to)?;
        self.add_node_version_with_interval(
            node_id, version_id, temporal, label, properties, provenance,
        )
    }

    /// Shared implementation for appending a node version with a fully
    /// constructed bi-temporal interval. See
    /// [`add_node_version_with_provenance`](Self::add_node_version_with_provenance)
    /// for the standard open/tombstone intervals and
    /// [`add_retracted_node_version`](Self::add_retracted_node_version) for
    /// retraction intervals (Issue #3230).
    fn add_node_version_with_interval(
        &mut self,
        node_id: NodeId,
        version_id: VersionId,
        temporal: BiTemporalInterval,
        label: InternedString,
        properties: PropertyMap,
        provenance: Option<Arc<Provenance>>,
    ) -> Result<()> {
        // Check capacity limit using cached count (O(1) operation, DoS protection)
        let version_count = self.node_version_counts.get(&node_id).copied().unwrap_or(0);
        if version_count >= self.retention_policy.max_versions_per_entity {
            return Err(StorageError::CapacityExceeded {
                resource: format!("node {} versions", node_id),
                current: version_count,
                limit: self.retention_policy.max_versions_per_entity,
            }
            .into());
        }

        // Check if this node already has versions
        let prev_version_id = self.node_version_heads.get(&node_id).copied();

        // Create version (anchor or delta based on chain length)
        let mut version = if let Some(prev_id) = prev_version_id {
            // Verify previous version exists (properties reconstructed later via reconstruct_node_properties)
            if !self.node_versions.contains_key(&prev_id) {
                return Err(StorageError::VersionNotFound(prev_id).into());
            }

            // Get cached counter and increment (O(1) instead of O(anchor_interval))
            // Issue #208: Use cached counter to avoid walking version chain
            let current_count = self
                .node_versions_since_anchor
                .get(&node_id)
                .copied()
                .unwrap_or(0);
            let versions_since_anchor = current_count + 1;

            if versions_since_anchor >= self.config.anchor_interval as usize {
                // Create anchor with link to previous version
                // Use properties.clone() here as we need original for caching later
                let mut anchor = NodeVersion::new_anchor(
                    version_id,
                    node_id,
                    temporal,
                    label,
                    properties.clone(),
                );
                anchor.prev_version = Some(prev_id);
                // Reset counter to 0 after creating anchor
                self.node_versions_since_anchor.insert(node_id, 0);
                anchor
            } else {
                // Create delta from previous version
                let old_properties = self.reconstruct_node_properties(prev_id)?;
                // Update counter for next iteration
                self.node_versions_since_anchor
                    .insert(node_id, versions_since_anchor);
                NodeVersion::new_delta(
                    version_id,
                    node_id,
                    temporal,
                    label,
                    &old_properties,
                    &properties,
                    prev_id,
                )
            }
        } else {
            // First version is always an anchor
            // Initialize counter to 0
            self.node_versions_since_anchor.insert(node_id, 0);
            NodeVersion::new_anchor(version_id, node_id, temporal, label, properties.clone())
        };
        version.provenance = provenance;

        // Handle pre-anchor hooks (BEFORE storing).
        //
        // Issue #383: the per-entity snapshot policy gates whether this anchor
        // triggers the (potentially whole-index) snapshot hooks. A `Skip` entity
        // still forms the anchor above but runs no snapshot hooks, so no vector
        // snapshot is captured for it — decoupling graph anchors from vector
        // snapshots. The default policy is `Snapshot`, so absent any per-entity
        // configuration this is identical to the pre-#383 behavior.
        if version.is_anchor() && self.node_anchor_triggers_vector_snapshot(node_id) {
            self.run_pre_anchor_hooks_into(
                &self.pre_node_anchor_hooks,
                AnchorHookContext {
                    entity_type: "node",
                    entity_id: node_id.as_u64(),
                    timestamp: temporal.transaction_time().start(),
                    properties: &properties,
                },
                &mut version.data,
            );
        }

        // Link previous version to this one and close its temporal intervals
        if let Some(prev_id) = prev_version_id
            && let Some(prev) = self.node_versions.get_mut(&prev_id)
        {
            // Capture the intervals before modification for temporal index update
            let old_temporal = *prev.temporal();

            Self::close_previous_version_intervals(prev, version_id, &temporal)?;

            // Update temporal indexes to reflect the closed intervals (Issue #209)
            if let Some(ref indexes) = self.temporal_indexes {
                let new_temporal = *prev.temporal();

                // Update valid time end if it was closed
                if old_temporal.valid_time().end() != new_temporal.valid_time().end() {
                    indexes.update_node_valid_time_end(
                        node_id,
                        prev_id,
                        new_temporal.valid_time().end(),
                    );
                }

                // Update transaction time end if it was closed
                if old_temporal.transaction_time().end() != new_temporal.transaction_time().end() {
                    indexes.update_node_transaction_time_end(
                        node_id,
                        prev_id,
                        new_temporal.transaction_time().end(),
                    );
                }
            }
        }

        // Check if anchor before storing (for notifications and caching)
        let is_anchor = version.is_anchor();

        // Store the version and update indexes
        self.node_versions.insert(version_id, version);
        self.node_version_heads.insert(node_id, version_id);
        *self.node_version_counts.entry(node_id).or_insert(0) += 1;

        // Issue #212: Update cached stats counters for O(1) stats() retrieval
        if is_anchor {
            self.cached_node_anchor_count += 1;
        } else {
            self.cached_node_delta_count += 1;
        }

        // Issue #210: Cache properties for ALL versions (anchors and deltas) to avoid
        // reconstructing properties we just added when creating the next delta.
        //
        // BEFORE: Only anchors were cached, causing delta creation to reconstruct
        //         the previous delta's properties even though we just added them.
        // AFTER:  All versions are cached in the main property cache, eliminating
        //         unnecessary reconstructions during consecutive writes.
        let props_arc = Arc::new(properties);
        self.node_property_cache
            .insert(version_id, props_arc.clone());

        // Anchors are also cached in the dedicated anchor cache for fallback
        if is_anchor {
            self.node_anchor_cache.insert(version_id, props_arc);
        }

        // Notify observers
        let timestamp = temporal.transaction_time().start();
        notify_observers(
            &self.observers,
            &StorageEvent::NodeVersionCreated {
                version_id,
                node_id,
                timestamp,
                is_anchor,
            },
        );
        // Issue #383: the `NodeAnchorCreated` event is the second temporal
        // vector snapshot trigger (the `VectorIndexObserver` reacts to it by
        // calling `create_snapshot_for_anchor`). Gate it on the SAME per-entity
        // policy as the pre-anchor hook above so a `Skip` node captures no
        // snapshot via EITHER path. The general `NodeVersionCreated` event
        // (emitted unconditionally above) stays available for metrics/audit
        // observers, so this never over-suppresses non-vector observers.
        if is_anchor && self.node_anchor_triggers_vector_snapshot(node_id) {
            notify_observers(
                &self.observers,
                &StorageEvent::NodeAnchorCreated {
                    version_id,
                    node_id,
                    timestamp,
                },
            );
        }

        Ok(())
    }

    /// Add a new version of an edge.
    /// Returns an error if the version limit for this entity is exceeded (DoS protection).
    ///
    /// Equivalent to [`add_edge_version_with_provenance`](Self::add_edge_version_with_provenance)
    /// with `provenance: None`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_edge_version(
        &mut self,
        edge_id: EdgeId,
        version_id: VersionId,
        valid_from: Timestamp,
        tx_time: Timestamp,
        label: InternedString,
        source: NodeId,
        target: NodeId,
        properties: PropertyMap,
        is_tombstone: bool,
    ) -> Result<()> {
        self.add_edge_version_with_provenance(
            edge_id,
            version_id,
            valid_from,
            tx_time,
            label,
            source,
            target,
            properties,
            is_tombstone,
            None,
        )
    }

    /// Add a new version of an edge, optionally attaching a write-time
    /// [`Provenance`] bundle (Issue #3224).
    ///
    /// Behaves identically to [`add_edge_version`](Self::add_edge_version) other
    /// than persisting `provenance` on the created version (anchor or delta).
    #[allow(clippy::too_many_arguments)]
    pub fn add_edge_version_with_provenance(
        &mut self,
        edge_id: EdgeId,
        version_id: VersionId,
        valid_from: Timestamp,
        tx_time: Timestamp,
        label: InternedString,
        source: NodeId,
        target: NodeId,
        properties: PropertyMap,
        is_tombstone: bool,
        provenance: Option<Arc<Provenance>>,
    ) -> Result<()> {
        // Construct bi-temporal interval from separate dimensions
        let mut temporal = BiTemporalInterval::with_valid_time(valid_from, tx_time);

        // For tombstones, close the valid_time at valid_from to create an empty interval [valid_from, valid_from)
        // This represents "entity is no longer valid starting from this point"
        if is_tombstone {
            temporal = temporal.close_valid_time(valid_from)?;
        }

        self.add_edge_version_with_interval(
            edge_id,
            version_id,
            temporal,
            label,
            source,
            target,
            properties,
            is_tombstone,
            provenance,
        )
    }

    /// Add a *retraction* version of an edge (Issue #3230). See
    /// [`add_retracted_node_version`](Self::add_retracted_node_version) for
    /// the retraction semantics; the edge variant additionally keeps the
    /// closed interval visible to the temporal adjacency index so AS OF
    /// traversals strictly before `valid_to` still follow the edge.
    #[allow(clippy::too_many_arguments)]
    pub fn add_retracted_edge_version(
        &mut self,
        edge_id: EdgeId,
        version_id: VersionId,
        valid_from: Timestamp,
        valid_to: Timestamp,
        tx_time: Timestamp,
        label: InternedString,
        source: NodeId,
        target: NodeId,
        properties: PropertyMap,
    ) -> Result<()> {
        self.add_retracted_edge_version_with_provenance(
            edge_id, version_id, valid_from, valid_to, tx_time, label, source, target, properties,
            None,
        )
    }

    /// Add a *retraction* version of an edge (Issue #3230), optionally attaching a
    /// write-time [`Provenance`] bundle
    /// recording the acting principal (Issue #3427). See
    /// [`add_retracted_node_version_with_provenance`](Self::add_retracted_node_version_with_provenance).
    #[allow(clippy::too_many_arguments)]
    pub fn add_retracted_edge_version_with_provenance(
        &mut self,
        edge_id: EdgeId,
        version_id: VersionId,
        valid_from: Timestamp,
        valid_to: Timestamp,
        tx_time: Timestamp,
        label: InternedString,
        source: NodeId,
        target: NodeId,
        properties: PropertyMap,
        provenance: Option<Arc<Provenance>>,
    ) -> Result<()> {
        let temporal =
            BiTemporalInterval::with_valid_time(valid_from, tx_time).close_valid_time(valid_to)?;
        self.add_edge_version_with_interval(
            edge_id, version_id, temporal, label, source, target, properties,
            false, // not a tombstone: the closed interval must stay traversable pre-valid_to
            provenance,
        )
    }

    /// Shared implementation for appending an edge version with a fully
    /// constructed bi-temporal interval.
    ///
    /// `is_tombstone` only controls whether the version is skipped when
    /// inserting into the temporal adjacency index (tombstones represent
    /// deletions and must not appear in traversal queries).
    #[allow(clippy::too_many_arguments)]
    fn add_edge_version_with_interval(
        &mut self,
        edge_id: EdgeId,
        version_id: VersionId,
        temporal: BiTemporalInterval,
        label: InternedString,
        source: NodeId,
        target: NodeId,
        properties: PropertyMap,
        is_tombstone: bool,
        provenance: Option<Arc<Provenance>>,
    ) -> Result<()> {
        // Check capacity limit using cached count (O(1) operation, DoS protection)
        let version_count = self.edge_version_counts.get(&edge_id).copied().unwrap_or(0);
        if version_count >= self.retention_policy.max_versions_per_entity {
            return Err(StorageError::CapacityExceeded {
                resource: format!("edge {} versions", edge_id),
                current: version_count,
                limit: self.retention_policy.max_versions_per_entity,
            }
            .into());
        }

        // Check if this edge already has versions
        let prev_version_id = self.edge_version_heads.get(&edge_id).copied();

        // Create version (anchor or delta based on chain length)
        let mut version = if let Some(prev_id) = prev_version_id {
            // Verify previous version exists (properties reconstructed later via reconstruct_edge_properties)
            if !self.edge_versions.contains_key(&prev_id) {
                return Err(StorageError::VersionNotFound(prev_id).into());
            }

            // Get cached counter and increment (O(1) instead of O(anchor_interval))
            // Issue #208: Use cached counter to avoid walking version chain
            let current_count = self
                .edge_versions_since_anchor
                .get(&edge_id)
                .copied()
                .unwrap_or(0);
            let versions_since_anchor = current_count + 1;

            if versions_since_anchor >= self.config.anchor_interval as usize {
                // Create anchor with link to previous version
                // Use properties.clone() here as we need original for caching later
                let mut anchor = EdgeVersion::new_anchor(
                    version_id,
                    edge_id,
                    temporal,
                    label,
                    source,
                    target,
                    properties.clone(),
                );
                anchor.prev_version = Some(prev_id);
                // Reset counter to 0 after creating anchor
                self.edge_versions_since_anchor.insert(edge_id, 0);
                anchor
            } else {
                // Create delta from previous version
                let old_properties = self.reconstruct_edge_properties(prev_id)?;
                // Update counter for next iteration
                self.edge_versions_since_anchor
                    .insert(edge_id, versions_since_anchor);
                EdgeVersion::new_delta(
                    version_id,
                    edge_id,
                    temporal,
                    label,
                    source,
                    target,
                    &old_properties,
                    &properties,
                    prev_id,
                )
            }
        } else {
            // First version is always an anchor
            // Initialize counter to 0
            self.edge_versions_since_anchor.insert(edge_id, 0);
            EdgeVersion::new_anchor(
                version_id,
                edge_id,
                temporal,
                label,
                source,
                target,
                properties.clone(),
            )
        };
        version.provenance = provenance;

        // Handle pre-anchor hooks (BEFORE storing).
        //
        // Issue #383: gated by the per-edge snapshot policy, symmetric with the
        // node path above. See `snapshot_policy` for the rationale and the
        // backward-compatible default (`Snapshot`).
        if version.is_anchor() && self.edge_anchor_triggers_vector_snapshot(edge_id) {
            self.run_pre_anchor_hooks_into(
                &self.pre_edge_anchor_hooks,
                AnchorHookContext {
                    entity_type: "edge",
                    entity_id: edge_id.as_u64(),
                    timestamp: temporal.transaction_time().start(),
                    properties: &properties,
                },
                &mut version.data,
            );
        }

        // Link previous version to this one and close its temporal intervals
        if let Some(prev_id) = prev_version_id
            && let Some(prev) = self.edge_versions.get_mut(&prev_id)
        {
            // Capture the intervals before modification for temporal index update
            let old_temporal = *prev.temporal();

            Self::close_previous_version_intervals(prev, version_id, &temporal)?;

            // Update temporal indexes to reflect the closed intervals (Issue #209)
            if let Some(ref indexes) = self.temporal_indexes {
                let new_temporal = *prev.temporal();

                // Update valid time end if it was closed
                if old_temporal.valid_time().end() != new_temporal.valid_time().end() {
                    indexes.update_edge_valid_time_end(
                        edge_id,
                        prev_id,
                        new_temporal.valid_time().end(),
                    );
                }

                // Update transaction time end if it was closed
                if old_temporal.transaction_time().end() != new_temporal.transaction_time().end() {
                    indexes.update_edge_transaction_time_end(
                        edge_id,
                        prev_id,
                        new_temporal.transaction_time().end(),
                    );
                }
            }

            // #3504: the superseded version's OWN valid interval now stays open
            // (append-only) -- close_previous_version_intervals no longer closes
            // it. Mirror the version chain's masking in the denormalized temporal
            // adjacency index by closing the prior entry's TRANSACTION time (the
            // tx-close that still happens on supersession and that hides the
            // superseded version from current-state reads) instead of its valid
            // time. This keeps a deleted/updated edge from reappearing in
            // traversals while preserving snapshot isolation for reads anchored
            // before the supersession (an earlier-tx query still sees the entry).
            if let Some(ref adj_index) = self.temporal_adjacency_index {
                let new_temporal = *prev.temporal();
                if old_temporal.transaction_time().end() != new_temporal.transaction_time().end() {
                    adj_index.close_edge_transaction_time(
                        edge_id,
                        source,
                        target,
                        new_temporal.transaction_time().end(),
                    );
                }
            }
        }

        // Check if anchor before storing (for notifications and caching)
        let is_anchor = version.is_anchor();

        // Store the version and update indexes
        self.edge_versions.insert(version_id, version);
        self.edge_version_heads.insert(edge_id, version_id);
        *self.edge_version_counts.entry(edge_id).or_insert(0) += 1;

        // Issue #212: Update cached stats counters for O(1) stats() retrieval
        if is_anchor {
            self.cached_edge_anchor_count += 1;
        } else {
            self.cached_edge_delta_count += 1;
        }

        // Issue #210: Cache properties for ALL versions (anchors and deltas) to avoid
        // reconstructing properties we just added when creating the next delta.
        let props_arc = Arc::new(properties);
        self.edge_property_cache
            .insert(version_id, props_arc.clone());

        // Anchors are also cached in the dedicated anchor cache for fallback
        if is_anchor {
            self.edge_anchor_cache.insert(version_id, props_arc);
        }

        // Notify observers
        let timestamp = temporal.transaction_time().start();
        notify_observers(
            &self.observers,
            &StorageEvent::EdgeVersionCreated {
                version_id,
                edge_id,
                timestamp,
                is_anchor,
            },
        );
        // Issue #383: symmetric with the node path — gate the second snapshot
        // trigger (`EdgeAnchorCreated`, consumed by `VectorIndexObserver`) on
        // the per-edge policy. `EdgeVersionCreated` above remains ungated.
        if is_anchor && self.edge_anchor_triggers_vector_snapshot(edge_id) {
            notify_observers(
                &self.observers,
                &StorageEvent::EdgeAnchorCreated {
                    version_id,
                    edge_id,
                    timestamp,
                },
            );
        }

        // Update temporal adjacency index if configured
        // Insert after all operations complete so temporal intervals are finalized
        // Skip tombstones - they represent deletions and shouldn't appear in traversal queries
        if !is_tombstone
            && let Some(ref adj_index) = self.temporal_adjacency_index
            && let Err(_e) = adj_index.insert_edge(
                edge_id,
                source,
                target,
                label,
                temporal.valid_time().start(),
                temporal.valid_time().end(),
                temporal.transaction_time().start(),
                temporal.transaction_time().end(),
            )
        {
            #[cfg(feature = "observability")]
            tracing::warn!(
                edge_id = %edge_id,
                source = %source,
                target = %target,
                error = %_e,
                "Failed to insert edge into temporal adjacency index"
            );
        }

        Ok(())
    }

    /// Reconstruct the properties of a node version.
    ///
    /// This walks backward to find the nearest anchor, then applies all deltas
    /// forward to reconstruct the full property state.
    ///
    /// **Cache Behavior**: Properties are cached by VersionId. Since properties are
    /// immutable per version and temporal visibility is checked separately in
    /// `find_node_version_at_time()`, cached properties are always valid and don't
    /// require invalidation when temporal intervals are modified.
    ///
    /// **Depth Limit**: Returns `TemporalError::MaxDepthExceeded` if the delta
    /// chain exceeds `MAX_RECONSTRUCTION_DEPTH` (100). This protects against
    /// stack overflow from corrupted version chains or cycles.
    pub fn reconstruct_node_properties(&self, version_id: VersionId) -> Result<PropertyMap> {
        self.reconstruct_node_properties_with_depth(version_id)
    }

    /// Get a node version from hot or cold storage.
    ///
    /// This is a helper for reconstruction that checks hot storage first (fast path),
    /// then falls back to tiered storage for cold data access.
    ///
    /// Returns `Err(VersionNotFound)` if the version doesn't exist in any tier.
    #[inline]
    fn get_node_version_any_tier(&self, version_id: VersionId) -> Result<Arc<NodeVersion>> {
        if let Some(v) = self.node_versions.get(&version_id) {
            // Fast path: version in hot storage
            Ok(Arc::new(v.clone()))
        } else {
            // Slow path: check cold storage via tiered layer
            self.get_node_version_tiered(version_id)?
                .ok_or(StorageError::VersionNotFound(version_id).into())
        }
    }

    /// Get an edge version from hot or cold storage.
    ///
    /// This is a helper for reconstruction that checks hot storage first (fast path),
    /// then falls back to tiered storage for cold data access.
    ///
    /// Returns `Err(VersionNotFound)` if the version doesn't exist in any tier.
    #[inline]
    fn get_edge_version_any_tier(&self, version_id: VersionId) -> Result<Arc<EdgeVersion>> {
        if let Some(v) = self.edge_versions.get(&version_id) {
            // Fast path: version in hot storage
            Ok(Arc::new(v.clone()))
        } else {
            // Slow path: check cold storage via tiered layer
            self.get_edge_version_tiered(version_id)?
                .ok_or(StorageError::VersionNotFound(version_id).into())
        }
    }

    /// Iterative property reconstruction helper for nodes (Issue #211).
    ///
    /// This function implements the core iterative reconstruction algorithm.
    /// It eliminates intermediate PropertyMap allocations and stack overflow risks.
    ///
    /// # Algorithm
    /// 1. Collect version IDs backwards from target to anchor (O(anchor_interval) IDs)
    /// 2. Extract anchor properties as base state
    /// 3. Apply deltas in forward order (O(anchor_interval) delta applications)
    ///
    /// # Arguments
    /// * `version_id` - The version to reconstruct properties for
    ///
    /// # Returns
    /// * `Ok(PropertyMap)` - Reconstructed properties
    /// * `Err(TemporalError::MaxDepthExceeded)` - Delta chain too deep (DoS protection)
    /// * `Err(StorageError::VersionNotFound)` - Requested version does not exist
    /// * `Err(TemporalError::MissingAnchor)` - An ancestor in the chain was removed
    /// * `Err(TemporalError::CorruptedVersionChain)` - Invalid chain structure
    fn reconstruct_node_properties_iterative(&self, version_id: VersionId) -> Result<PropertyMap> {
        // Collect version IDs backwards from target to anchor
        // Pre-allocate with anchor_interval capacity to avoid reallocations
        let mut version_ids: Vec<VersionId> =
            Vec::with_capacity(self.config.anchor_interval as usize);
        let mut current_id = version_id;
        let mut chain_length = 0;

        // Walk backwards until we find an anchor or hit depth limit
        loop {
            // Check depth limit for DoS protection
            if chain_length >= self.max_reconstruction_depth {
                let entity_id = self
                    .node_versions
                    .get(&version_id)
                    .map(|v| v.node_id.to_string())
                    .unwrap_or_else(|| format!("version {}", version_id));
                return Err(TemporalError::MaxDepthExceeded {
                    max_depth: MAX_RECONSTRUCTION_DEPTH,
                    entity_id,
                }
                .into());
            }

            // After the first version is retrieved, a VersionNotFound on a subsequent
            // lookup means the backward chain is broken: the ancestor (ultimately the
            // anchor) has been removed. Return MissingAnchor rather than the generic
            // VersionNotFound so callers can distinguish "version never existed" from
            // "chain integrity was broken after creation".
            let version = match self.get_node_version_any_tier(current_id) {
                Ok(v) => v,
                Err(crate::core::error::Error::Storage(StorageError::VersionNotFound(_)))
                    if !version_ids.is_empty() =>
                {
                    let entity_id = version_ids
                        .first()
                        .and_then(|&vid| self.get_node_version_any_tier(vid).ok())
                        .map(|v| v.node_id.to_string())
                        .unwrap_or_else(|| format!("version {}", version_id));
                    return Err(TemporalError::MissingAnchor { entity_id }.into());
                }
                Err(e) => return Err(e),
            };

            let is_anchor = version.is_anchor();
            let prev_id = version.prev_version;

            // Store version ID (we'll process these in reverse)
            version_ids.push(current_id);

            // If we found an anchor, we're done collecting
            if is_anchor {
                break;
            }

            // Get previous version for delta chain traversal
            current_id = prev_id.ok_or_else(|| TemporalError::CorruptedVersionChain {
                entity_id: version.node_id.to_string(),
                reason: "Delta version has no previous version".to_string(),
            })?;

            chain_length += 1;
        }

        // Now reconstruct properties by applying deltas in forward order
        // The last element in version_ids is the anchor (base state)
        let anchor_id =
            version_ids
                .last()
                .copied()
                .ok_or_else(|| TemporalError::CorruptedVersionChain {
                    entity_id: format!("version {}", version_id),
                    reason: "Empty version chain during reconstruction".to_string(),
                })?;

        let anchor_version = self.get_node_version_any_tier(anchor_id)?;

        let mut properties = match &anchor_version.data {
            VersionData::Anchor { properties, .. } => properties.clone(),
            VersionData::Delta { .. } => {
                // This should never happen due to the is_anchor() check above
                return Err(TemporalError::CorruptedVersionChain {
                    entity_id: anchor_version.node_id.to_string(),
                    reason: "Expected anchor at base of version chain".to_string(),
                }
                .into());
            }
        };

        // Apply deltas in forward order (reverse of collection order)
        // Skip the last element (anchor) since we already have its properties
        for &vid in version_ids.iter().rev().skip(1) {
            let version = self.get_node_version_any_tier(vid)?;

            match &version.data {
                VersionData::Delta { delta } => {
                    properties = delta.apply(&properties);
                }
                VersionData::Anchor { .. } => {
                    // This should never happen - only the last element should be an anchor
                    return Err(TemporalError::CorruptedVersionChain {
                        entity_id: version.node_id.to_string(),
                        reason: "Found anchor in middle of delta chain".to_string(),
                    }
                    .into());
                }
            }
        }

        Ok(properties)
    }

    /// Iterative property reconstruction helper for edges (Issue #211).
    ///
    /// Mirrors the node reconstruction algorithm for consistency. See
    /// `reconstruct_node_properties_iterative` for algorithm details.
    fn reconstruct_edge_properties_iterative(&self, version_id: VersionId) -> Result<PropertyMap> {
        // Collect version IDs backwards from target to anchor
        // Pre-allocate with anchor_interval capacity to avoid reallocations
        let mut version_ids: Vec<VersionId> =
            Vec::with_capacity(self.config.anchor_interval as usize);
        let mut current_id = version_id;
        let mut chain_length = 0;

        // Walk backwards until we find an anchor or hit depth limit
        loop {
            // Check depth limit for DoS protection
            if chain_length >= self.max_reconstruction_depth {
                let entity_id = self
                    .edge_versions
                    .get(&version_id)
                    .map(|v| v.edge_id.to_string())
                    .unwrap_or_else(|| format!("version {}", version_id));
                return Err(TemporalError::MaxDepthExceeded {
                    max_depth: MAX_RECONSTRUCTION_DEPTH,
                    entity_id,
                }
                .into());
            }

            // After the first version is retrieved, a VersionNotFound on a subsequent
            // lookup means the backward chain is broken — the ancestor anchor has been
            // removed. Return MissingAnchor rather than VersionNotFound.
            let version = match self.get_edge_version_any_tier(current_id) {
                Ok(v) => v,
                Err(crate::core::error::Error::Storage(StorageError::VersionNotFound(_)))
                    if !version_ids.is_empty() =>
                {
                    let entity_id = version_ids
                        .first()
                        .and_then(|&vid| self.get_edge_version_any_tier(vid).ok())
                        .map(|v| v.edge_id.to_string())
                        .unwrap_or_else(|| format!("version {}", version_id));
                    return Err(TemporalError::MissingAnchor { entity_id }.into());
                }
                Err(e) => return Err(e),
            };

            let is_anchor = version.is_anchor();
            let prev_id = version.prev_version;

            // Store version ID (we'll process these in reverse)
            version_ids.push(current_id);

            // If we found an anchor, we're done collecting
            if is_anchor {
                break;
            }

            // Get previous version for delta chain traversal
            current_id = prev_id.ok_or_else(|| TemporalError::CorruptedVersionChain {
                entity_id: version.edge_id.to_string(),
                reason: "Delta version has no previous version".to_string(),
            })?;

            chain_length += 1;
        }

        // Now reconstruct properties by applying deltas in forward order
        // The last element in version_ids is the anchor (base state)
        let anchor_id =
            version_ids
                .last()
                .copied()
                .ok_or_else(|| TemporalError::CorruptedVersionChain {
                    entity_id: format!("version {}", version_id),
                    reason: "Empty version chain during reconstruction".to_string(),
                })?;

        let anchor_version = self.get_edge_version_any_tier(anchor_id)?;

        let mut properties = match &anchor_version.data {
            VersionData::Anchor { properties, .. } => properties.clone(),
            VersionData::Delta { .. } => {
                // This should never happen due to the is_anchor() check above
                return Err(TemporalError::CorruptedVersionChain {
                    entity_id: anchor_version.edge_id.to_string(),
                    reason: "Expected anchor at base of version chain".to_string(),
                }
                .into());
            }
        };

        // Apply deltas in forward order (reverse of collection order)
        // Skip the last element (anchor) since we already have its properties
        for &vid in version_ids.iter().rev().skip(1) {
            let version = self.get_edge_version_any_tier(vid)?;

            match &version.data {
                VersionData::Delta { delta } => {
                    properties = delta.apply(&properties);
                }
                VersionData::Anchor { .. } => {
                    // This should never happen - only the last element should be an anchor
                    return Err(TemporalError::CorruptedVersionChain {
                        entity_id: version.edge_id.to_string(),
                        reason: "Found anchor in middle of delta chain".to_string(),
                    }
                    .into());
                }
            }
        }

        Ok(properties)
    }

    /// Internal node property reconstruction with dual-cache lookup.
    ///
    /// Checks primary cache, then anchor cache fallback, then reconstructs
    /// iteratively from the delta chain.
    fn reconstruct_node_properties_with_depth(&self, version_id: VersionId) -> Result<PropertyMap> {
        if let Some(cached) = self.node_property_cache.get(&version_id) {
            self.primary_cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(cached.as_ref().clone());
        }

        // Anchor cache fallback: survives primary cache eviction under delta pressure
        if let Some(cached) = self.node_anchor_cache.get(&version_id) {
            self.anchor_cache_hits.fetch_add(1, Ordering::Relaxed);
            self.node_property_cache.insert(version_id, cached.clone());
            return Ok(cached.as_ref().clone());
        }

        self.full_reconstructions.fetch_add(1, Ordering::Relaxed);

        let properties = self.reconstruct_node_properties_iterative(version_id)?;

        // Populate cache for future reads
        self.node_property_cache
            .insert(version_id, Arc::new(properties.clone()));

        Ok(properties)
    }

    /// Reconstruct the properties of an edge version.
    ///
    /// **Cache Behavior**: Same as `reconstruct_node_properties()` - properties are
    /// immutable per VersionId, so caching doesn't require invalidation.
    ///
    /// **Depth Limit**: Returns `TemporalError::MaxDepthExceeded` if the delta
    /// chain exceeds `MAX_RECONSTRUCTION_DEPTH` (100). This protects against
    /// stack overflow from corrupted version chains or cycles.
    pub fn reconstruct_edge_properties(&self, version_id: VersionId) -> Result<PropertyMap> {
        self.reconstruct_edge_properties_with_depth(version_id)
    }

    /// Internal edge property reconstruction with dual-cache lookup.
    ///
    /// Checks primary cache, then anchor cache fallback, then reconstructs
    /// iteratively from the delta chain.
    fn reconstruct_edge_properties_with_depth(&self, version_id: VersionId) -> Result<PropertyMap> {
        if let Some(cached) = self.edge_property_cache.get(&version_id) {
            self.primary_cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(cached.as_ref().clone());
        }

        // Anchor cache fallback: survives primary cache eviction under delta pressure
        if let Some(cached) = self.edge_anchor_cache.get(&version_id) {
            self.anchor_cache_hits.fetch_add(1, Ordering::Relaxed);
            self.edge_property_cache.insert(version_id, cached.clone());
            return Ok(cached.as_ref().clone());
        }

        self.full_reconstructions.fetch_add(1, Ordering::Relaxed);

        let properties = self.reconstruct_edge_properties_iterative(version_id)?;

        // Populate cache for future reads
        self.edge_property_cache
            .insert(version_id, Arc::new(properties.clone()));

        Ok(properties)
    }

    /// Get a node version by ID.
    pub fn get_node_version(&self, version_id: VersionId) -> Option<&NodeVersion> {
        self.node_versions.get(&version_id)
    }

    /// Get an edge version by ID.
    pub fn get_edge_version(&self, version_id: VersionId) -> Option<&EdgeVersion> {
        self.edge_versions.get(&version_id)
    }

    /// Get the current version ID for a node.
    pub fn get_current_node_version(&self, node_id: NodeId) -> Option<VersionId> {
        self.node_version_heads.get(&node_id).copied()
    }

    /// Get the current version ID for an edge.
    pub fn get_current_edge_version(&self, edge_id: EdgeId) -> Option<VersionId> {
        self.edge_version_heads.get(&edge_id).copied()
    }

    /// Get the provenance bundle attached to a node's *current* version, if any.
    ///
    /// Returns `Ok(None)` (not an error) if the node exists but its current
    /// version carries no provenance -- this is the common case for writes
    /// that never supplied a bundle (Issue #3224). Falls back to cold/tiered
    /// storage if the current version has been migrated out of hot storage.
    ///
    /// This re-resolves "whichever version is current right now", which is a
    /// separate lookup from any node snapshot the caller may already hold --
    /// prefer [`get_node_version_provenance`](Self::get_node_version_provenance)
    /// with an already-fetched `Node`'s `current_version` when consistency
    /// with that snapshot matters (e.g. a concurrent write must not be able
    /// to return one version's properties paired with a different version's
    /// provenance).
    pub fn get_current_node_provenance(&self, node_id: NodeId) -> Result<Option<Provenance>> {
        let Some(version_id) = self.get_current_node_version(node_id) else {
            return Ok(None);
        };
        self.get_node_version_provenance(version_id)
    }

    /// Get the provenance bundle attached to a specific node version, if any.
    ///
    /// Unlike [`get_current_node_provenance`](Self::get_current_node_provenance),
    /// this looks up an exact, caller-supplied version rather than
    /// re-resolving "whichever version is current right now" -- callers that
    /// already hold a `Node` snapshot (e.g. from `get_node`) should pass that
    /// snapshot's `current_version` here for a consistent, race-free read.
    pub fn get_node_version_provenance(&self, version_id: VersionId) -> Result<Option<Provenance>> {
        let provenance = if let Some(v) = self.node_versions.get(&version_id) {
            v.provenance.as_deref().cloned()
        } else {
            self.get_node_version_any_tier(version_id)?
                .provenance
                .as_deref()
                .cloned()
        };
        Ok(provenance)
    }

    /// Get the provenance bundle attached to an edge's *current* version, if any.
    ///
    /// See [`get_current_node_provenance`](Self::get_current_node_provenance)
    /// for semantics, including the race-condition note about
    /// [`get_edge_version_provenance`](Self::get_edge_version_provenance).
    pub fn get_current_edge_provenance(&self, edge_id: EdgeId) -> Result<Option<Provenance>> {
        let Some(version_id) = self.get_current_edge_version(edge_id) else {
            return Ok(None);
        };
        self.get_edge_version_provenance(version_id)
    }

    /// Get the provenance bundle attached to a specific edge version, if any.
    ///
    /// See [`get_node_version_provenance`](Self::get_node_version_provenance)
    /// for why callers holding an `Edge` snapshot should prefer this over
    /// [`get_current_edge_provenance`](Self::get_current_edge_provenance).
    pub fn get_edge_version_provenance(&self, version_id: VersionId) -> Result<Option<Provenance>> {
        let provenance = if let Some(v) = self.edge_versions.get(&version_id) {
            v.provenance.as_deref().cloned()
        } else {
            self.get_edge_version_any_tier(version_id)?
                .provenance
                .as_deref()
                .cloned()
        };
        Ok(provenance)
    }

    /// Get the provenance bundle and bi-temporal interval of a specific node
    /// version in a single lookup (Issue #3232).
    ///
    /// Like [`get_node_version_provenance`](Self::get_node_version_provenance),
    /// this resolves an exact, caller-supplied version (typically a `Node`
    /// snapshot's `current_version`) so the returned metadata is consistent
    /// with the properties already read from that snapshot -- including for
    /// point-in-time reads, whose lookup paths set `current_version` to the
    /// matched historical version. Fetching both fields together keeps read
    /// responses at one historical lookup per entity.
    ///
    /// Returns `Ok(None)` when the version cannot be found in any tier.
    pub fn get_node_version_read_metadata(
        &self,
        version_id: VersionId,
    ) -> Result<Option<(Option<Provenance>, BiTemporalInterval)>> {
        if let Some(v) = self.node_versions.get(&version_id) {
            // Fast path: read from hot storage without cloning version data.
            return Ok(Some((v.provenance.as_deref().cloned(), v.temporal)));
        }
        Ok(self
            .get_node_version_tiered(version_id)?
            .map(|v| (v.provenance.as_deref().cloned(), v.temporal)))
    }

    /// Edge counterpart of
    /// [`get_node_version_read_metadata`](Self::get_node_version_read_metadata).
    pub fn get_edge_version_read_metadata(
        &self,
        version_id: VersionId,
    ) -> Result<Option<(Option<Provenance>, BiTemporalInterval)>> {
        if let Some(v) = self.edge_versions.get(&version_id) {
            // Fast path: read from hot storage without cloning version data.
            return Ok(Some((v.provenance.as_deref().cloned(), v.temporal)));
        }
        Ok(self
            .get_edge_version_tiered(version_id)?
            .map(|v| (v.provenance.as_deref().cloned(), v.temporal)))
    }

    /// Get the node's true creation time: the `valid_from` of its first-ever version.
    ///
    /// Walks `prev_version` links from the current head back to the terminal (oldest)
    /// version. Unlike `get_current_node_version`, which returns the *latest* version,
    /// this is the correct floor for "was this valid_from before the entity existed"
    /// checks — the latest version's `valid_from` can be later than the entity's true
    /// creation time when a prior write already backdated it.
    pub(crate) fn node_creation_time(&self, node_id: NodeId) -> Option<Timestamp> {
        let mut version_id = self.get_current_node_version(node_id)?;
        loop {
            let version = self.get_node_version(version_id)?;
            match version.prev_version {
                Some(prev) => version_id = prev,
                None => return Some(version.temporal.valid_time().start()),
            }
        }
    }

    /// Get the edge's true creation time: the `valid_from` of its first-ever version.
    ///
    /// See [`Self::node_creation_time`] for why this must walk to the terminal version
    /// rather than reading the current (latest) one.
    pub(crate) fn edge_creation_time(&self, edge_id: EdgeId) -> Option<Timestamp> {
        let mut version_id = self.get_current_edge_version(edge_id)?;
        loop {
            let version = self.get_edge_version(version_id)?;
            match version.prev_version {
                Some(prev) => version_id = prev,
                None => return Some(version.temporal.valid_time().start()),
            }
        }
    }

    /// Get the IDs of every node that has ever had at least one version recorded.
    ///
    /// Used for bi-temporal schema discovery (Issue #3214): the caller reconstructs
    /// each ID at a given instant via [`AletheiaDB::get_nodes_at_time`](crate::db::AletheiaDB::get_nodes_at_time)
    /// to determine which were visible.
    pub fn versioned_node_ids(&self) -> Vec<NodeId> {
        self.node_version_heads.keys().copied().collect()
    }

    /// Get the IDs of every edge that has ever had at least one version recorded.
    ///
    /// Used for bi-temporal schema discovery (Issue #3214): the caller reconstructs
    /// each ID at a given instant via [`AletheiaDB::get_edges_at_time`](crate::db::AletheiaDB::get_edges_at_time)
    /// to determine which were visible.
    pub fn versioned_edge_ids(&self) -> Vec<EdgeId> {
        self.edge_version_heads.keys().copied().collect()
    }

    /// Get the configured safety cap (per entity kind) on the number of
    /// ever-versioned entities `AletheiaDB::schema_as_of` will reconstruct
    /// in a single call. See
    /// [`crate::config::HistoricalConfigBuilder::max_schema_as_of_entities`].
    pub fn max_schema_as_of_entities(&self) -> usize {
        self.max_schema_as_of_entities
    }

    /// Get all node versions for all nodes.
    ///
    /// Returns a map of NodeId -> `Vec<NodeVersion>` for recovery property tests.
    /// This walks through all node versions and groups them by entity ID.
    pub fn get_all_node_versions(&self) -> FastHashMap<NodeId, Vec<&NodeVersion>> {
        let mut result: FastHashMap<NodeId, Vec<&NodeVersion>> = FastHashMap::default();

        for version in self.node_versions.values() {
            result.entry(version.node_id).or_default().push(version);
        }

        result
    }

    /// Get all edge versions for all edges.
    ///
    /// Returns a map of EdgeId -> `Vec<EdgeVersion>` for recovery property tests.
    /// This walks through all edge versions and groups them by entity ID.
    pub fn get_all_edge_versions(&self) -> FastHashMap<EdgeId, Vec<&EdgeVersion>> {
        let mut result: FastHashMap<EdgeId, Vec<&EdgeVersion>> = FastHashMap::default();

        for version in self.edge_versions.values() {
            result.entry(version.edge_id).or_default().push(version);
        }

        result
    }

    /// Visit every node version retained in hot historical storage
    /// (read-only, unordered).
    ///
    /// Includes expired/superseded versions and delete tombstones — hot
    /// storage never prunes versions (the retention policy only rejects
    /// over-limit writes). Versions whose payload has been migrated to cold
    /// storage are not visited. Used by `AletheiaDB::temporal_extent_by_label`
    /// (Issue #3238) to fold per-label temporal bounds without allocating an
    /// intermediate map.
    pub fn visit_node_versions(&self, mut f: impl FnMut(&NodeVersion)) {
        for version in self.node_versions.values() {
            f(version);
        }
    }

    /// Visit every edge version retained in hot historical storage
    /// (read-only, unordered).
    ///
    /// Edge counterpart of [`visit_node_versions`](Self::visit_node_versions);
    /// the same coverage notes apply.
    pub fn visit_edge_versions(&self, mut f: impl FnMut(&EdgeVersion)) {
        for version in self.edge_versions.values() {
            f(version);
        }
    }

    /// Collect changefeed records for the committed versions that fall within the given
    /// transaction-time window, optionally constrained by a valid-time window, a node-label /
    /// edge-type filter, a resume cursor, and a `bound` on how many smallest-cursor rows to
    /// retain (Issue #3216; filter + limit pushdown, PR 2).
    ///
    /// This is a read-only scan over the in-memory version maps. Only committed versions are
    /// ever present in these maps (versions are inserted on the commit-apply path), so the
    /// changefeed never surfaces uncommitted or rolled-back data. The result is unordered; the
    /// caller is responsible for deterministic ordering and final page selection.
    ///
    /// The `resume_after` cursor (strict `> cursor`) and the `bound` are applied **during** the
    /// scan via [`BoundedChanges`], so the working set held in memory is `O(bound)` rather than
    /// `O(matches)` — no post-collection materialize-then-refilter. Pass `usize::MAX` as `bound`
    /// (and `None` as `resume_after`) to recover an unbounded, unresumed collection.
    ///
    /// # Performance
    ///
    /// The candidate enumeration is still an O(V) walk of the hot maps (a future
    /// `(commit_timestamp, kind, id)` index could make this O(log V + page)), but only the
    /// `bound`-smallest survivors are retained. To keep the historical lock hold short, this
    /// produces lightweight [`RawChange`]s (no owned label `String`); label resolution is
    /// deferred to the query layer for surviving rows only. Cold-tier versions are scanned
    /// separately by the caller after the lock is released (see `tiered_storage_arc`).
    pub(crate) fn collect_changes(
        &self,
        tx_window: &TimeRange,
        valid_window: Option<&TimeRange>,
        label_filter: Option<&str>,
        resume_after: Option<ChangeCursor>,
        bound: usize,
    ) -> Vec<RawChange> {
        let mut acc = BoundedChanges::new(bound);

        for v in self.node_versions.values() {
            consider_version(
                &mut acc,
                resume_after,
                v.id.as_u64(),
                v.node_id.as_u64(),
                EntityKind::Node,
                &v.temporal,
                v.label,
                v.prev_version.is_none(),
                tx_window,
                valid_window,
                label_filter,
                // Lazy (Issue #3349, PR3c): derived only for a candidate that
                // passed the cheap tx/valid/label filters.
                || self.node_version_namespace_id(v),
            );
        }

        for v in self.edge_versions.values() {
            consider_version(
                &mut acc,
                resume_after,
                v.id.as_u64(),
                v.edge_id.as_u64(),
                EntityKind::Edge,
                &v.temporal,
                v.label,
                v.prev_version.is_none(),
                tx_window,
                valid_window,
                label_filter,
                || self.edge_version_namespace_id(v),
            );
        }

        acc.into_vec()
    }

    /// Derive the interned [`NamespaceId`] of a node version (Issue #3349, PR3c).
    ///
    /// The namespace is immutable and rides along the property map under
    /// [`crate::core::namespace::NAMESPACE_KEY`], stamped on every anchor. An
    /// anchor carries it directly (cheap, no walk); a delta does not (the
    /// immutable key never diffs), so it is recovered by reconstructing the
    /// version's properties (cached, and correct across the anchor chain / cold
    /// tier). A legacy / `default` entity has no key and resolves to the default
    /// namespace.
    ///
    /// # Fail-closed on reconstruction failure (Issue #3349, PR3c security fix)
    ///
    /// If reconstructing a delta's properties hard-fails (a delta chain deeper
    /// than `max_reconstruction_depth` → `MaxDepthExceeded`, or a `MissingAnchor`),
    /// the namespace **cannot** be derived. It is stamped with the reserved
    /// [`crate::core::namespace::UNRESOLVED_NAMESPACE`] sentinel — **never**
    /// [`Namespace::default`](crate::core::namespace::Namespace::default): failing
    /// open to `default` (a real, subscribable namespace) would leak a non-default
    /// entity's change to `default`-scoped subscribers and hide it from its own
    /// namespace. The sentinel matches no user scope, so the change is withheld
    /// from every user-scoped read while still surfacing under an `All` / unset
    /// scope.
    fn node_version_namespace_id(&self, v: &NodeVersion) -> NamespaceId {
        match &v.data {
            VersionData::Anchor { properties, .. } => intern_namespace(&namespace_of(properties)),
            VersionData::Delta { .. } => match self.reconstruct_node_properties(v.id) {
                Ok(p) => intern_namespace(&namespace_of(&p)),
                Err(e) => {
                    eprintln!(
                        "Warning: changefeed namespace derivation failed for node version {} \
                         ({e}); record scoped to the reserved '{}' namespace (fail-closed)",
                        v.id,
                        crate::core::namespace::UNRESOLVED_NAMESPACE
                    );
                    unresolved_namespace_id()
                }
            },
        }
    }

    /// Edge counterpart of [`node_version_namespace_id`](Self::node_version_namespace_id).
    /// Fail-closes to the [`crate::core::namespace::UNRESOLVED_NAMESPACE`] sentinel
    /// on a reconstruction hard-failure (see that method).
    fn edge_version_namespace_id(&self, v: &EdgeVersion) -> NamespaceId {
        match &v.data {
            VersionData::Anchor { properties, .. } => intern_namespace(&namespace_of(properties)),
            VersionData::Delta { .. } => match self.reconstruct_edge_properties(v.id) {
                Ok(p) => intern_namespace(&namespace_of(&p)),
                Err(e) => {
                    eprintln!(
                        "Warning: changefeed namespace derivation failed for edge version {} \
                         ({e}); record scoped to the reserved '{}' namespace (fail-closed)",
                        v.id,
                        crate::core::namespace::UNRESOLVED_NAMESPACE
                    );
                    unresolved_namespace_id()
                }
            },
        }
    }

    /// Re-derive the namespace of any changefeed [`RawChange`]s left **unresolved**
    /// (stamped with the [`crate::core::namespace::UNRESOLVED_NAMESPACE`] sentinel)
    /// by a fast-path derivation, using tier-aware reconstruction (Issue #3349,
    /// PR3c security fix).
    ///
    /// The cold-tier scan derives a delta's namespace from a running anchor map it
    /// builds **within the cold scan** (see
    /// [`crate::storage::redb_cold_storage::RedbColdStorage::collect_changes_filtered`]).
    /// That map necessarily misses a delta whose covering anchor is **not in the
    /// cold tier** — the LRU anchor-split case: under `MigrationPolicy::aggressive()`
    /// / `enable_lru`, a frequently-accessed anchor stays HOT while an older delta
    /// migrates COLD. The cold scan cannot resolve such a delta (it cannot see the
    /// hot tier), so it marks it with the sentinel and defers to this method, which
    /// runs at the [`HistoricalStorage`] layer where **both** tiers are visible via
    /// [`reconstruct_node_properties`](Self::reconstruct_node_properties) /
    /// [`reconstruct_edge_properties`](Self::reconstruct_edge_properties).
    ///
    /// A record whose namespace is genuinely unresolvable even with both tiers
    /// (reconstruction hard-fails) keeps the sentinel — fail-closed, never
    /// `default`. Records not carrying the sentinel are left untouched, so this is a
    /// cheap no-op on the overwhelmingly common fully-resolved page.
    pub(crate) fn resolve_unresolved_namespaces(&self, changes: &mut [RawChange]) {
        let sentinel = unresolved_namespace_id();
        for rec in changes.iter_mut() {
            if rec.namespace_id != sentinel {
                continue;
            }
            let Ok(version_id) = VersionId::new(rec.cursor.version_id) else {
                continue;
            };
            let derived = match EntityKind::from_ord(rec.cursor.kind_ord) {
                EntityKind::Node => self.reconstruct_node_properties(version_id),
                EntityKind::Edge => self.reconstruct_edge_properties(version_id),
            };
            match derived {
                Ok(p) => rec.namespace_id = intern_namespace(&namespace_of(&p)),
                Err(e) => {
                    // Genuinely unresolvable across both tiers: keep the fail-closed
                    // sentinel (never `default`).
                    eprintln!(
                        "Warning: changefeed namespace derivation failed for version {} \
                         ({e}); record scoped to the reserved '{}' namespace (fail-closed)",
                        rec.cursor.version_id,
                        crate::core::namespace::UNRESOLVED_NAMESPACE
                    );
                }
            }
        }
    }

    /// Build changefeed [`RawChange`]s for a specific, known set of just-committed version
    /// ids — the push-changefeed counterpart to the window scan in
    /// [`collect_changes`](Self::collect_changes) (Issue #3375).
    ///
    /// Where `collect_changes` scans *all* hot versions and filters by a transaction-time
    /// window, this looks up only the `node_version_ids` / `edge_version_ids` produced by a
    /// single transaction — an O(txn size) targeted read rather than an O(total) scan, so
    /// the post-commit broadcast never pays for a full history rescan. Each record is built
    /// through the **same** [`build_raw_change`] helper `collect_changes` uses (with the
    /// version's own transaction-time interval as the window, which always contains its own
    /// start), so the produced records are byte-identical to what `list_changes` would
    /// return for that version — including tombstone empty valid ranges and Created vs
    /// Modified classification. Unknown ids (e.g. a version already migrated to the cold
    /// tier) are silently skipped.
    ///
    /// The result is unordered; the caller sorts by [`ChangeCursor`](crate::core::changefeed)
    /// to obtain the #3216 deterministic total order.
    pub(crate) fn collect_committed_changes(
        &self,
        node_version_ids: &[VersionId],
        edge_version_ids: &[VersionId],
    ) -> Vec<RawChange> {
        let mut out = Vec::with_capacity(node_version_ids.len() + edge_version_ids.len());

        for vid in node_version_ids {
            if let Some(v) = self.node_versions.get(vid) {
                let tx_window = v.temporal.transaction_time();
                if let Some(rec) = build_raw_change(
                    v.id.as_u64(),
                    v.node_id.as_u64(),
                    EntityKind::Node,
                    &v.temporal,
                    v.label,
                    v.prev_version.is_none(),
                    &tx_window,
                    None,
                    None,
                    || self.node_version_namespace_id(v),
                ) {
                    out.push(rec);
                }
            }
        }

        for vid in edge_version_ids {
            if let Some(v) = self.edge_versions.get(vid) {
                let tx_window = v.temporal.transaction_time();
                if let Some(rec) = build_raw_change(
                    v.id.as_u64(),
                    v.edge_id.as_u64(),
                    EntityKind::Edge,
                    &v.temporal,
                    v.label,
                    v.prev_version.is_none(),
                    &tx_window,
                    None,
                    None,
                    || self.edge_version_namespace_id(v),
                ) {
                    out.push(rec);
                }
            }
        }

        out
    }

    /// Clone the configured tiered-storage handle, if any.
    ///
    /// Lets a caller release the historical lock and then scan the cold tier (disk I/O) without
    /// holding the lock across that I/O.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn tiered_storage_arc(&self) -> Option<Arc<super::tiered_storage::TieredStorage>> {
        self.tiered_storage.clone()
    }

    // ========================================================================
    // Tiered Storage Integration
    // ========================================================================

    /// Configure tiered storage for this historical storage.
    ///
    /// When tiered storage is configured, versions not found in hot storage
    /// will be looked up from cold storage via the tiered storage layer.
    ///
    /// # Arguments
    ///
    /// * `tiered` - The tiered storage instance to use
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aletheiadb::storage::historical::HistoricalStorage;
    /// use aletheiadb::storage::tiered_storage::TieredStorage;
    /// use aletheiadb::storage::redb_cold_storage::RedbColdStorage;
    ///
    /// let mut historical = HistoricalStorage::new();
    /// let cold = RedbColdStorage::with_default_config("data/cold.redb")?;
    /// let tiered = TieredStorage::with_default_config(Box::new(cold));
    /// historical.set_tiered_storage(Arc::new(tiered));
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_tiered_storage(&mut self, tiered: Arc<super::tiered_storage::TieredStorage>) {
        self.tiered_storage = Some(tiered);
    }

    /// Get the tiered storage instance, if configured.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn tiered_storage(&self) -> Option<&super::tiered_storage::TieredStorage> {
        self.tiered_storage.as_deref()
    }

    /// Check if tiered storage is enabled.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn has_tiered_storage(&self) -> bool {
        self.tiered_storage.is_some()
    }

    /// Check if tiered storage is enabled.
    ///
    /// The cold/tiered subsystem is unavailable on `wasm32`, so this is always
    /// `false` there (the wasm profile is hot-only).
    #[cfg(target_arch = "wasm32")]
    pub fn has_tiered_storage(&self) -> bool {
        false
    }

    /// Set the temporal indexes for optimized version lookups (Issue #209).
    ///
    /// When temporal indexes are configured, `find_node_version_at_time` and
    /// `find_edge_version_at_time` will use O(log n) binary search instead of
    /// O(n) linear scans through version chains.
    ///
    /// This is typically called during database initialization to share the
    /// temporal indexes between the database and historical storage.
    pub fn set_temporal_indexes(&mut self, indexes: Arc<crate::index::temporal::TemporalIndexes>) {
        self.temporal_indexes = Some(indexes);
    }

    /// Temporarily detach the wired temporal indexes, returning whatever was
    /// set (if anything), so a caller can run a multi-version replay pass
    /// with the intra-replay "close previous version" hooks reduced to
    /// no-ops, then reattach (`set_temporal_indexes`) and follow up with
    /// [`rebuild_temporal_index_from_versions`](Self::rebuild_temporal_index_from_versions).
    ///
    /// # Why this exists (Issue #3355, Slice D)
    ///
    /// `add_node_version_with_interval`/[`close_node_version_transaction_time`](Self::close_node_version_transaction_time)
    /// (and their edge equivalents) only ever *update* an existing temporal-index
    /// entry (closing the previous version's interval) -- they never *insert* the
    /// entry for the version just being created; only a full
    /// [`rebuild_temporal_index_from_versions`](Self::rebuild_temporal_index_from_versions)
    /// pass populates new entries. Startup replay (`db/config.rs`) and PITR
    /// (`db/pitr.rs::finish_pitr_replay`) are both safe by construction: they
    /// replay into a `HistoricalStorage` whose temporal indexes are not yet
    /// wired (`None`) at all, so every intra-replay close call is already a
    /// no-op, and the single rebuild call at the end is what populates the
    /// index. A **live, already-wired** `HistoricalStorage` -- the replica
    /// applier's case (`storage::replication::apply::apply_replica_batch`),
    /// which replays into a database that has been serving temporal reads
    /// since it was constructed -- has no such luxury: without detaching
    /// first, a replayed batch containing two-or-more versions of the SAME
    /// entity (e.g. a create immediately followed by an update, batched
    /// together by a single fetch) hits a "version not found in metadata"
    /// inconsistency, because the second version's *predecessor* was created
    /// moments earlier in this same still-in-progress replay pass and has not
    /// been indexed yet (indexing only happens at the end, via the rebuild).
    /// Detaching first reproduces the startup/PITR no-op behavior on a live
    /// store, deferring all indexing to the rebuild that already follows.
    pub(crate) fn take_temporal_indexes(
        &mut self,
    ) -> Option<Arc<crate::index::temporal::TemporalIndexes>> {
        self.temporal_indexes.take()
    }

    /// Set the temporal adjacency index for this storage.
    ///
    /// When the temporal adjacency index is set, it will be automatically updated
    /// when edges are added or modified, enabling efficient temporal pathfinding
    /// queries that can find paths through deleted edges.
    ///
    /// This is typically called during database initialization.
    pub fn set_temporal_adjacency_index(
        &mut self,
        index: Arc<crate::index::temporal_adjacency::TemporalAdjacencyIndex>,
    ) {
        self.temporal_adjacency_index = Some(index);
    }

    /// Get a reference to the temporal adjacency index if configured.
    ///
    /// Used by persistence layer to save the index to disk.
    pub fn get_temporal_adjacency_index(
        &self,
    ) -> Option<&Arc<crate::index::temporal_adjacency::TemporalAdjacencyIndex>> {
        self.temporal_adjacency_index.as_ref()
    }

    /// Get outgoing edges from a node at a specific point in time.
    ///
    /// This method uses the temporal adjacency index to efficiently find all
    /// edges that were valid at the specified time, including edges that have
    /// been deleted in current storage.
    ///
    /// # Arguments
    ///
    /// * `source` - The source node ID
    /// * `valid_time` - The valid time to query
    /// * `tx_time` - The transaction time to query
    ///
    /// # Returns
    ///
    /// A vector of edge IDs that were valid at the specified time. Returns an
    /// empty vector if no temporal adjacency index is configured.
    pub fn get_outgoing_edges_at_time(
        &self,
        source: NodeId,
        valid_time: Timestamp,
        tx_time: Timestamp,
    ) -> Vec<EdgeId> {
        if let Some(ref index) = self.temporal_adjacency_index {
            index.get_outgoing_at_time(source, valid_time, tx_time)
        } else {
            Vec::new()
        }
    }

    /// Get incoming edges to a node at a specific point in time.
    ///
    /// This method uses the temporal adjacency index to efficiently find all
    /// edges that were valid at the specified time, including edges that have
    /// been deleted in current storage.
    ///
    /// # Arguments
    ///
    /// * `target` - The target node ID
    /// * `valid_time` - The valid time to query
    /// * `tx_time` - The transaction time to query
    ///
    /// # Returns
    ///
    /// A vector of edge IDs that were valid at the specified time. Returns an
    /// empty vector if no temporal adjacency index is configured.
    pub fn get_incoming_edges_at_time(
        &self,
        target: NodeId,
        valid_time: Timestamp,
        tx_time: Timestamp,
    ) -> Vec<EdgeId> {
        if let Some(ref index) = self.temporal_adjacency_index {
            index.get_incoming_at_time(target, valid_time, tx_time)
        } else {
            Vec::new()
        }
    }

    /// Get a node version from any tier (hot or cold).
    ///
    /// This method first checks hot storage, then falls back to cold storage
    /// via the tiered storage layer (if configured).
    ///
    /// # Arguments
    ///
    /// * `version_id` - The version ID to retrieve
    ///
    /// # Returns
    ///
    /// Returns `Ok(Some(version))` if found in either tier, `Ok(None)` if not found,
    /// or an error if cold storage access fails.
    pub fn get_node_version_tiered(
        &self,
        version_id: VersionId,
    ) -> Result<Option<Arc<NodeVersion>>> {
        // Check hot storage first
        if let Some(version) = self.node_versions.get(&version_id) {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(ref tiered) = self.tiered_storage {
                tiered.record_hot_hit();
            }
            return Ok(Some(Arc::new(version.clone())));
        }

        // Fall back to cold storage if tiered storage is configured
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(ref tiered) = self.tiered_storage {
            return tiered.get_node_version_cold(version_id);
        }

        Ok(None)
    }

    /// Get an edge version from any tier (hot or cold).
    ///
    /// This method first checks hot storage, then falls back to cold storage
    /// via the tiered storage layer (if configured).
    pub fn get_edge_version_tiered(
        &self,
        version_id: VersionId,
    ) -> Result<Option<Arc<EdgeVersion>>> {
        // Check hot storage first
        if let Some(version) = self.edge_versions.get(&version_id) {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(ref tiered) = self.tiered_storage {
                tiered.record_hot_hit();
            }
            return Ok(Some(Arc::new(version.clone())));
        }

        // Fall back to cold storage if tiered storage is configured
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(ref tiered) = self.tiered_storage {
            return tiered.get_edge_version_cold(version_id);
        }

        Ok(None)
    }

    /// Migrate old versions from hot storage to cold storage.
    ///
    /// This method identifies versions that meet the migration policy criteria
    /// and moves them to cold storage. The migration service handles the actual
    /// transfer, and this method removes migrated versions from hot storage.
    ///
    /// # Arguments
    ///
    /// * `migration_service` - The migration service to use for transferring versions
    ///
    /// # Returns
    ///
    /// Returns the number of versions migrated, or an error if migration fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn migrate_to_cold(
        &mut self,
        migration_service: &super::migration::MigrationService,
    ) -> Result<usize> {
        use std::time::Instant;

        if self.tiered_storage.is_none() {
            return Ok(0);
        }

        let mut total_migrated = 0;

        // Identify node version candidates
        let node_candidates = migration_service.identify_node_candidates(
            &self.node_versions,
            &self.node_version_heads,
            &self.node_version_counts,
            Instant::now(),
        );

        // Collect versions to migrate
        let node_versions_to_migrate: Vec<NodeVersion> = node_candidates
            .iter()
            .filter_map(|c| self.node_versions.get(&c.version_id).cloned())
            .collect();

        // Migrate to cold storage
        if !node_versions_to_migrate.is_empty() {
            let migrated = migration_service.migrate_node_versions(&node_versions_to_migrate)?;
            total_migrated += migrated;

            // Issue #3677: record each migrated version's ChangeCursor in the cold-change
            // directory AFTER it is durably in cold and BEFORE it is removed from the hot maps, so
            // it is never absent from both hot and directory during the migration window.
            if let Some(tiered) = self.tiered_storage.as_ref() {
                tiered.record_cold_cursors(node_versions_to_migrate[..migrated].iter().map(|v| {
                    ChangeCursor::for_version(
                        v.temporal.transaction_time().start(),
                        EntityKind::Node,
                        v.node_id.as_u64(),
                        v.id.as_u64(),
                    )
                }));
            }

            // Remove migrated versions from hot storage
            for candidate in &node_candidates[..migrated] {
                if let Some(version) = self.node_versions.remove(&candidate.version_id) {
                    // Update version count
                    if let Some(count) = self.node_version_counts.get_mut(&version.node_id) {
                        *count = count.saturating_sub(1);
                    }
                    // Issue #212: Update cached stats counters when migrating to cold storage
                    if version.is_anchor() {
                        self.cached_node_anchor_count =
                            self.cached_node_anchor_count.saturating_sub(1);
                    } else {
                        self.cached_node_delta_count =
                            self.cached_node_delta_count.saturating_sub(1);
                    }
                }
            }
        }

        // Identify edge version candidates
        let edge_candidates = migration_service.identify_edge_candidates(
            &self.edge_versions,
            &self.edge_version_heads,
            &self.edge_version_counts,
            Instant::now(),
        );

        // Collect versions to migrate
        let edge_versions_to_migrate: Vec<EdgeVersion> = edge_candidates
            .iter()
            .filter_map(|c| self.edge_versions.get(&c.version_id).cloned())
            .collect();

        // Migrate to cold storage
        if !edge_versions_to_migrate.is_empty() {
            let migrated = migration_service.migrate_edge_versions(&edge_versions_to_migrate)?;
            total_migrated += migrated;

            // Issue #3677: record migrated edge cursors before removing them from the hot maps
            // (see the node path above for the ordering invariant).
            if let Some(tiered) = self.tiered_storage.as_ref() {
                tiered.record_cold_cursors(edge_versions_to_migrate[..migrated].iter().map(|v| {
                    ChangeCursor::for_version(
                        v.temporal.transaction_time().start(),
                        EntityKind::Edge,
                        v.edge_id.as_u64(),
                        v.id.as_u64(),
                    )
                }));
            }

            // Remove migrated versions from hot storage
            for candidate in &edge_candidates[..migrated] {
                if let Some(version) = self.edge_versions.remove(&candidate.version_id)
                    && let Some(count) = self.edge_version_counts.get_mut(&version.edge_id)
                {
                    *count = count.saturating_sub(1);
                    // Issue #212: Update cached stats counters when migrating to cold storage
                    if version.is_anchor() {
                        self.cached_edge_anchor_count =
                            self.cached_edge_anchor_count.saturating_sub(1);
                    } else {
                        self.cached_edge_delta_count =
                            self.cached_edge_delta_count.saturating_sub(1);
                    }
                }
            }
        }

        Ok(total_migrated)
    }

    /// Get the total number of versions in hot storage.
    pub fn hot_version_count(&self) -> usize {
        self.node_versions.len() + self.edge_versions.len()
    }

    /// Get the estimated memory usage of hot storage in bytes.
    pub fn hot_memory_usage(&self) -> usize {
        let node_size = self.node_versions.len() * std::mem::size_of::<NodeVersion>();
        let edge_size = self.edge_versions.len() * std::mem::size_of::<EdgeVersion>();
        node_size + edge_size
    }

    /// Close the transaction time of a node version.
    ///
    /// This marks the version as no longer being the "current knowledge" after
    /// the specified timestamp. Used when a node is deleted or superseded.
    ///
    /// # Arguments
    /// * `version_id` - The version to close
    /// * `end_timestamp` - The timestamp at which this version is no longer valid
    ///
    /// # Returns
    /// `Ok(())` if successful, `Err` if version not found
    pub fn close_node_version_transaction_time(
        &mut self,
        version_id: VersionId,
        end_timestamp: Timestamp,
    ) -> Result<()> {
        let version = self
            .node_versions
            .get_mut(&version_id)
            .ok_or(StorageError::VersionNotFound(version_id))?;

        // Get the node ID before closing (needed for temporal index update)
        let node_id = version.node_id;

        // Use TemporalVersion trait method
        version.close_transaction_time(end_timestamp)?;

        // Update temporal index to reflect the closed interval (Issue #209)
        if let Some(ref indexes) = self.temporal_indexes {
            indexes.update_node_transaction_time_end(node_id, version_id, end_timestamp);
        }

        Ok(())
    }

    /// Close the transaction time of an edge version.
    ///
    /// This marks the version as no longer being the "current knowledge" after
    /// the specified timestamp. Used when an edge is deleted or superseded.
    pub fn close_edge_version_transaction_time(
        &mut self,
        version_id: VersionId,
        end_timestamp: Timestamp,
    ) -> Result<()> {
        let version = self
            .edge_versions
            .get_mut(&version_id)
            .ok_or(StorageError::VersionNotFound(version_id))?;

        // Get the edge ID and node IDs before closing (needed for index updates)
        let edge_id = version.edge_id;
        let source = version.source;
        let target = version.target;

        // Use TemporalVersion trait method
        version.close_transaction_time(end_timestamp)?;

        // Update temporal index to reflect the closed interval (Issue #209)
        if let Some(ref indexes) = self.temporal_indexes {
            indexes.update_edge_transaction_time_end(edge_id, version_id, end_timestamp);
        }

        // Update temporal adjacency index to reflect the closed transaction time
        if let Some(ref adj_index) = self.temporal_adjacency_index {
            adj_index.close_edge_transaction_time(edge_id, source, target, end_timestamp);
        }

        Ok(())
    }

    /// Find a node version valid at a specific point in time.
    ///
    /// **Performance (Issue #209)**:
    /// - **With temporal indexes**: O(log N) binary search where N = version count
    /// - **Without temporal indexes**: O(N) linear scan through version chain
    ///
    /// When temporal indexes are configured via `set_temporal_indexes()`, this
    /// method uses efficient binary search. Otherwise, it falls back to walking
    /// the version chain. For entities with 100s-1000s of versions, the temporal
    /// index provides significant performance improvements (10-100x faster).
    ///
    /// # Arguments
    /// * `node_id` - The node to query
    /// * `valid_time` - When the fact was true in reality
    /// * `transaction_time` - When the fact was recorded in the database
    ///
    /// # Returns
    /// The version ID visible at the given bi-temporal point, or None if no
    /// version exists at that time.
    pub fn find_node_version_at_time(
        &self,
        node_id: NodeId,
        valid_time: Timestamp,
        transaction_time: Timestamp,
    ) -> Option<VersionId> {
        // Fast path: Use temporal index if available (O(log n)) - Issue #209
        // The temporal indexes are now properly updated when intervals are closed
        if let Some(ref indexes) = self.temporal_indexes {
            return indexes
                .find_node_version_at_point_iter(node_id, valid_time, transaction_time)
                .find(|&version_id| {
                    // Robustness check: verify visibility against actual version data
                    self.node_versions
                        .get(&version_id)
                        .map(|v| v.temporal.is_visible_at(valid_time, transaction_time))
                        .unwrap_or(false)
                });
        }

        // Fallback: Linear scan through version chain (O(n))
        // This is only used when temporal indexes are not configured
        let mut current_id = self.node_version_heads.get(&node_id).copied()?;
        // Safety guard: prevent infinite loops from cyclic version chains (data corruption)
        let max_iterations = self.node_versions.len() + 1;
        let mut iterations = 0;

        loop {
            let version = self.node_versions.get(&current_id)?;

            // Check if this version's temporal interval contains the query time
            if version.temporal.is_visible_at(valid_time, transaction_time) {
                return Some(current_id);
            }

            // Move to previous version
            current_id = version.prev_version?;

            iterations += 1;
            if iterations > max_iterations {
                #[cfg(feature = "observability")]
                tracing::error!(
                    node_id = %node_id,
                    max_iterations = %max_iterations,
                    "Infinite loop detected in node version chain"
                );
                return None;
            }
        }
    }

    /// Find an edge version valid at a specific point in time.
    ///
    /// **Performance (Issue #209)**:
    /// - **With temporal indexes**: O(log N) binary search where N = version count
    /// - **Without temporal indexes**: O(N) linear scan through version chain
    ///
    /// When temporal indexes are configured via `set_temporal_indexes()`, this
    /// method uses efficient binary search. Otherwise, it falls back to walking
    /// the version chain. For entities with 100s-1000s of versions, the temporal
    /// index provides significant performance improvements (10-100x faster).
    ///
    /// # Arguments
    /// * `edge_id` - The edge to query
    /// * `valid_time` - When the fact was true in reality
    /// * `transaction_time` - When the fact was recorded in the database
    ///
    /// # Returns
    /// The version ID visible at the given bi-temporal point, or None if no
    /// version exists at that time.
    pub fn find_edge_version_at_time(
        &self,
        edge_id: EdgeId,
        valid_time: Timestamp,
        transaction_time: Timestamp,
    ) -> Option<VersionId> {
        // Fast path: Use temporal index if available (O(log n)) - Issue #209
        // The temporal indexes are now properly updated when intervals are closed
        if let Some(ref indexes) = self.temporal_indexes {
            return indexes
                .find_edge_version_at_point_iter(edge_id, valid_time, transaction_time)
                .find(|&version_id| {
                    // Robustness check: verify visibility against actual version data
                    self.edge_versions
                        .get(&version_id)
                        .map(|v| v.temporal.is_visible_at(valid_time, transaction_time))
                        .unwrap_or(false)
                });
        }

        // Fallback: Linear scan through version chain (O(n))
        // This is only used when temporal indexes are not configured
        let mut current_id = self.edge_version_heads.get(&edge_id).copied()?;
        // Safety guard: prevent infinite loops from cyclic version chains (data corruption)
        let max_iterations = self.edge_versions.len() + 1;
        let mut iterations = 0;

        loop {
            let version = self.edge_versions.get(&current_id)?;

            if version.temporal.is_visible_at(valid_time, transaction_time) {
                return Some(current_id);
            }

            current_id = version.prev_version?;

            iterations += 1;
            if iterations > max_iterations {
                #[cfg(feature = "observability")]
                tracing::error!(
                    edge_id = %edge_id,
                    max_iterations = %max_iterations,
                    "Infinite loop detected in edge version chain"
                );
                return None;
            }
        }
    }

    /// Get a node as it existed at a specific point in bi-temporal space.
    ///
    /// Uses the temporal index for O(log n) candidate lookup, then verifies
    /// visibility (handles closed intervals from deletions).
    pub fn get_node_at_time(
        &self,
        node_id: NodeId,
        valid_time: Timestamp,
        transaction_time: Timestamp,
    ) -> Result<Node> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("get_node_at_time").entered();

        let version_id = self
            .find_node_version_at_time(node_id, valid_time, transaction_time)
            .ok_or(StorageError::NodeNotFound(node_id))?;

        // Note: find_node_version_at_time already checked visibility
        let version = self
            .get_node_version(version_id)
            .ok_or(StorageError::VersionNotFound(version_id))?;

        let properties = self.reconstruct_node_properties(version_id)?;

        Ok(Node::new(
            version.node_id,
            version.label,
            properties,
            version.id,
        ))
    }

    /// Get an edge as it existed at a specific point in bi-temporal space.
    ///
    /// Uses the temporal index for O(log n) candidate lookup, then verifies
    /// visibility (handles closed intervals from deletions).
    pub fn get_edge_at_time(
        &self,
        edge_id: EdgeId,
        valid_time: Timestamp,
        transaction_time: Timestamp,
    ) -> Result<Edge> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("get_edge_at_time").entered();

        let version_id = self
            .find_edge_version_at_time(edge_id, valid_time, transaction_time)
            .ok_or(StorageError::EdgeNotFound(edge_id))?;

        // Note: find_edge_version_at_time already checked visibility
        let version = self
            .get_edge_version(version_id)
            .ok_or(StorageError::VersionNotFound(version_id))?;

        let properties = self.reconstruct_edge_properties(version_id)?;

        Ok(Edge::new(
            version.edge_id,
            version.label,
            version.source,
            version.target,
            properties,
            version.id,
        ))
    }

    /// Get multiple nodes as they existed at a specific point in bi-temporal space.
    ///
    /// This retrieves nodes in batch to minimize overhead.
    /// If a node is not found or not visible at the time, the Option will be None.
    pub fn get_nodes_at_time(
        &self,
        node_ids: &[NodeId],
        valid_time: Timestamp,
        transaction_time: Timestamp,
    ) -> Result<Vec<(NodeId, Option<Node>)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("get_nodes_at_time").entered();

        let mut results = Vec::with_capacity(node_ids.len());

        for &node_id in node_ids {
            let node = if let Some(version_id) =
                self.find_node_version_at_time(node_id, valid_time, transaction_time)
            {
                // We found a visible version. Reconstruct it.
                match self.reconstruct_node_properties(version_id) {
                    Ok(properties) => {
                        let version = self
                            .node_versions
                            .get(&version_id)
                            .ok_or(StorageError::VersionNotFound(version_id))?;
                        Some(Node::new(
                            version.node_id,
                            version.label,
                            properties,
                            version.id,
                        ))
                    }
                    Err(e) => {
                        // Per this method's documented contract, a reconstruction failure
                        // is a systemic failure (corruption, not "didn't exist") and must
                        // propagate as `Err`, not be silently downgraded to `None` --
                        // otherwise callers can't distinguish "not visible at this instant"
                        // from "we couldn't tell".
                        #[cfg(feature = "observability")]
                        tracing::error!(
                            version_id = %version_id,
                            node_id = %node_id,
                            error = %e,
                            "Property reconstruction failed in batch query"
                        );
                        return Err(e);
                    }
                }
            } else {
                None
            };
            results.push((node_id, node));
        }

        Ok(results)
    }

    /// Batch point-in-time node lookup restricted to a single label
    /// (Issue #3236).
    ///
    /// For each candidate id, resolves the version visible at
    /// `(valid_time, transaction_time)` and checks the version's recorded
    /// `label` **before** reconstructing properties, so an off-label
    /// candidate costs only the version-at-time lookup, never a property
    /// chain replay. Candidates that are not visible at the coordinate, or
    /// whose at-time label differs, are simply skipped (no `None`
    /// placeholders), so the output length may be shorter than the input.
    /// Output order follows input order.
    ///
    /// Like [`get_nodes_at_time`](Self::get_nodes_at_time), a property
    /// reconstruction failure is a systemic error and propagates as `Err`.
    pub(crate) fn get_nodes_at_time_with_label(
        &self,
        node_ids: &[NodeId],
        label: InternedString,
        valid_time: Timestamp,
        transaction_time: Timestamp,
    ) -> Result<Vec<Node>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("get_nodes_at_time_with_label")
                .entered();

        let mut results = Vec::new();

        for &node_id in node_ids {
            let Some(version_id) =
                self.find_node_version_at_time(node_id, valid_time, transaction_time)
            else {
                continue;
            };
            let version = self
                .node_versions
                .get(&version_id)
                .ok_or(StorageError::VersionNotFound(version_id))?;
            if version.label != label {
                continue;
            }
            let (matched_node_id, matched_label) = (version.node_id, version.label);
            let properties = self.reconstruct_node_properties(version_id)?;
            results.push(Node::new(
                matched_node_id,
                matched_label,
                properties,
                version_id,
            ));
        }

        Ok(results)
    }

    /// Get multiple edges as they existed at a specific point in bi-temporal space.
    ///
    /// This retrieves edges in batch to minimize overhead.
    /// If an edge is not found or not visible at the time, the Option will be None.
    pub fn get_edges_at_time(
        &self,
        edge_ids: &[EdgeId],
        valid_time: Timestamp,
        transaction_time: Timestamp,
    ) -> Result<Vec<(EdgeId, Option<Edge>)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("get_edges_at_time").entered();

        let mut results = Vec::with_capacity(edge_ids.len());

        for &edge_id in edge_ids {
            let edge = if let Some(version_id) =
                self.find_edge_version_at_time(edge_id, valid_time, transaction_time)
            {
                // We found a visible version. Reconstruct it.
                match self.reconstruct_edge_properties(version_id) {
                    Ok(properties) => {
                        let version = self
                            .edge_versions
                            .get(&version_id)
                            .ok_or(StorageError::VersionNotFound(version_id))?;
                        Some(Edge::new(
                            version.edge_id,
                            version.label,
                            version.source,
                            version.target,
                            properties,
                            version.id,
                        ))
                    }
                    Err(e) => {
                        // See the matching comment in get_nodes_at_time: reconstruction
                        // failures are systemic and must propagate, not become `None`.
                        #[cfg(feature = "observability")]
                        tracing::error!(
                            version_id = %version_id,
                            edge_id = %edge_id,
                            error = %e,
                            "Property reconstruction failed in batch query"
                        );
                        return Err(e);
                    }
                }
            } else {
                None
            };
            results.push((edge_id, edge));
        }

        Ok(results)
    }

    /// Get the complete version history of a node.
    ///
    /// Returns all versions in chronological order (oldest first).
    pub fn get_node_history(&self, node_id: NodeId) -> Result<EntityHistory> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("get_node_history").entered();

        // Get the current version ID
        let current_version_id = self
            .get_current_node_version(node_id)
            .ok_or(StorageError::NodeNotFound(node_id))?;

        // Traverse the version chain backwards to get all versions in order
        let mut version_ids = Vec::new();
        let mut current_id = Some(current_version_id);

        while let Some(vid) = current_id {
            version_ids.push(vid);
            current_id = self.get_node_version(vid).and_then(|v| v.prev_version);
        }

        // Reverse to get oldest-first order
        version_ids.reverse();

        // Build VersionInfo for each version
        let mut versions = Vec::with_capacity(version_ids.len());
        for (version_number, version_id) in version_ids.iter().enumerate() {
            if let Some(version) = self.get_node_version(*version_id) {
                let properties = self.reconstruct_node_properties(*version_id)?;

                versions.push(VersionInfo {
                    version_number: (version_number + 1) as u64, // 1-indexed
                    version_id: *version_id,
                    temporal: version.temporal,
                    properties,
                    label: GLOBAL_INTERNER
                        .resolve_with(version.label, |s| s.to_string())
                        .unwrap_or_else(|| version.label.to_string()),
                    provenance: version.provenance.as_deref().cloned(),
                });
            }
        }

        Ok(EntityHistory { versions })
    }

    /// Get a node at a specific logical version number.
    ///
    /// Version numbers are 1-indexed (1 = first version, 2 = second version, etc.).
    pub fn get_node_at_version(&self, node_id: NodeId, version_number: u64) -> Result<Node> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("get_node_at_version").entered();

        // Get the current version ID
        let current_version_id = self
            .get_current_node_version(node_id)
            .ok_or(StorageError::NodeNotFound(node_id))?;

        // Traverse the version chain backwards to collect all versions
        let mut version_ids = Vec::new();
        let mut current_id = Some(current_version_id);

        while let Some(vid) = current_id {
            version_ids.push(vid);
            current_id = self.get_node_version(vid).and_then(|v| v.prev_version);
        }

        // Reverse to get oldest-first order
        version_ids.reverse();

        // Convert 1-indexed version number to 0-indexed array index
        let index = version_number
            .checked_sub(1)
            .ok_or(StorageError::NodeNotFound(node_id))? as usize;

        // Get the version ID at that index
        let version_id = version_ids
            .get(index)
            .ok_or(StorageError::NodeNotFound(node_id))?;

        // Reconstruct the node from that version
        let version = self
            .get_node_version(*version_id)
            .ok_or(StorageError::VersionNotFound(*version_id))?;

        let properties = self.reconstruct_node_properties(*version_id)?;

        Ok(Node::new(
            version.node_id,
            version.label,
            properties,
            version.id,
        ))
    }

    /// Compute the difference between two versions of a node.
    ///
    /// Shows which properties were added, removed, or modified.
    pub fn diff_node_versions(
        &self,
        node_id: NodeId,
        from_version: VersionId,
        to_version: VersionId,
    ) -> Result<VersionDiff> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("diff_node_versions").entered();

        // Validate that both versions belong to the requested node
        let from_ver = self
            .get_node_version(from_version)
            .ok_or(StorageError::VersionNotFound(from_version))?;
        let to_ver = self
            .get_node_version(to_version)
            .ok_or(StorageError::VersionNotFound(to_version))?;

        if from_ver.node_id != node_id {
            return Err(StorageError::InconsistentState {
                reason: format!(
                    "Version {} belongs to node {}, not node {}",
                    from_version, from_ver.node_id, node_id
                ),
            }
            .into());
        }
        if to_ver.node_id != node_id {
            return Err(StorageError::InconsistentState {
                reason: format!(
                    "Version {} belongs to node {}, not node {}",
                    to_version, to_ver.node_id, node_id
                ),
            }
            .into());
        }

        // Reconstruct both versions
        let from_props = self.reconstruct_node_properties(from_version)?;
        let to_props = self.reconstruct_node_properties(to_version)?;

        // Compute diff
        Ok(VersionDiff::compute(
            &from_props,
            &to_props,
            from_version,
            to_version,
        ))
    }

    /// Get the complete version history of an edge.
    ///
    /// Returns all versions in chronological order (oldest first).
    pub fn get_edge_history(&self, edge_id: EdgeId) -> Result<EntityHistory> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("get_edge_history").entered();

        // Get the current version ID
        let current_version_id = self
            .get_current_edge_version(edge_id)
            .ok_or(StorageError::EdgeNotFound(edge_id))?;

        // Traverse the version chain backwards to get all versions
        let mut version_ids = Vec::new();
        let mut current_id = Some(current_version_id);

        while let Some(vid) = current_id {
            version_ids.push(vid);
            current_id = self.get_edge_version(vid).and_then(|v| v.prev_version);
        }

        // Reverse to get oldest-first order
        version_ids.reverse();

        // Build VersionInfo for each version
        let mut versions = Vec::with_capacity(version_ids.len());
        for (version_number, version_id) in version_ids.iter().enumerate() {
            if let Some(version) = self.get_edge_version(*version_id) {
                let properties = self.reconstruct_edge_properties(*version_id)?;

                versions.push(VersionInfo {
                    version_number: (version_number + 1) as u64, // 1-indexed
                    version_id: *version_id,
                    temporal: version.temporal,
                    properties,
                    label: GLOBAL_INTERNER
                        .resolve_with(version.label, |s| s.to_string())
                        .unwrap_or_else(|| version.label.to_string()),
                    provenance: version.provenance.as_deref().cloned(),
                });
            }
        }

        Ok(EntityHistory { versions })
    }

    /// Compute the difference between two versions of an edge.
    ///
    /// Shows which properties were added, removed, or modified.
    pub fn diff_edge_versions(
        &self,
        edge_id: EdgeId,
        from_version: VersionId,
        to_version: VersionId,
    ) -> Result<VersionDiff> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::historical_storage_query_span("diff_edge_versions").entered();

        // Validate that both versions belong to the requested edge
        let from_ver = self
            .get_edge_version(from_version)
            .ok_or(StorageError::VersionNotFound(from_version))?;
        let to_ver = self
            .get_edge_version(to_version)
            .ok_or(StorageError::VersionNotFound(to_version))?;

        if from_ver.edge_id != edge_id {
            return Err(StorageError::InconsistentState {
                reason: format!(
                    "Version {} belongs to edge {}, not edge {}",
                    from_version, from_ver.edge_id, edge_id
                ),
            }
            .into());
        }
        if to_ver.edge_id != edge_id {
            return Err(StorageError::InconsistentState {
                reason: format!(
                    "Version {} belongs to edge {}, not edge {}",
                    to_version, to_ver.edge_id, edge_id
                ),
            }
            .into());
        }

        // Reconstruct both versions
        let from_props = self.reconstruct_edge_properties(from_version)?;
        let to_props = self.reconstruct_edge_properties(to_version)?;

        // Compute diff
        Ok(VersionDiff::compute(
            &from_props,
            &to_props,
            from_version,
            to_version,
        ))
    }

    /// Count versions since the last anchor using a generic version lookup function.
    ///
    /// This is a generic helper that works for both nodes and edges, reducing code duplication.
    /// The `get_version` closure provides type-specific version lookup.
    ///
    /// # Note (Issue #208)
    /// This method is no longer used in production code (replaced by cached counters for O(1) performance),
    /// but is retained for testing purposes to verify cache correctness.
    #[cfg(test)]
    fn count_versions_since_anchor<'a, V: EntityVersion + 'a>(
        &'a self,
        version_id: VersionId,
        get_version: impl Fn(VersionId) -> Option<&'a V>,
    ) -> usize {
        let mut count = 0;
        let mut current_id = version_id;

        loop {
            if let Some(version) = get_version(current_id) {
                if version.is_anchor() {
                    return count;
                }
                count += 1;

                if let Some(prev_id) = version.prev_version() {
                    current_id = prev_id;
                } else {
                    return count;
                }
            } else {
                return count;
            }
        }
    }

    /// Count how many versions exist since the last anchor (for a node).
    ///
    /// Note: The closure overhead is negligible and typically optimized away by LLVM.
    /// If profiling shows this is a hotspot, consider monomorphizing.
    ///
    /// # Note (Issue #208)
    /// This method is no longer used in production code, retained for testing only.
    #[cfg(test)]
    fn count_versions_since_anchor_node(&self, version_id: VersionId) -> usize {
        self.count_versions_since_anchor(version_id, |vid| self.node_versions.get(&vid))
    }

    /// Count how many versions exist since the last anchor (for an edge).
    ///
    /// Note: The closure overhead is negligible and typically optimized away by LLVM.
    /// If profiling shows this is a hotspot, consider monomorphizing.
    ///
    /// # Note (Issue #208)
    /// This method is no longer used in production code, retained for testing only.
    #[cfg(test)]
    #[allow(dead_code)]
    fn count_versions_since_anchor_edge(&self, version_id: VersionId) -> usize {
        self.count_versions_since_anchor(version_id, |vid| self.edge_versions.get(&vid))
    }

    /// Run the given ordered pre-anchor hooks and, if any resolves a snapshot
    /// id, store it on the anchor's [`VersionData`] (Issue #3525).
    ///
    /// This is a thin, allocation-free-when-empty wrapper over
    /// [`hooks::run_pre_anchor_hooks`]. Hooks run in registration order under
    /// the configured timeout policy; panics and errors are isolated (see the
    /// [`hooks`] module for the locking and partial-failure semantics). Because
    /// an anchor has a single snapshot-id slot, the last hook returning
    /// `Ok(Some(id))` wins.
    fn run_pre_anchor_hooks_into(
        &self,
        hooks: &[PreAnchorHook],
        context: AnchorHookContext<'_>,
        version_data: &mut VersionData,
    ) {
        if hooks.is_empty() {
            return;
        }
        if let Some(snapshot_id) =
            hooks::run_pre_anchor_hooks(hooks, &context, self.hook_timeout, &self.hook_metrics)
        {
            version_data.set_vector_snapshot_id(snapshot_id);
        }
    }

    /// Close the temporal intervals of a previous version when a new version is created.
    ///
    /// This helper handles the common logic of linking versions together and closing
    /// the temporal intervals of the previous version at the new version's start time.
    fn close_previous_version_intervals<V: EntityVersion>(
        prev_version: &mut V,
        new_version_id: VersionId,
        new_temporal: &BiTemporalInterval,
    ) -> Result<()> {
        prev_version.set_next_version(Some(new_version_id));

        // Work on a local copy, apply modifications, then write back
        let mut prev_temporal = *prev_version.temporal();

        // #3504: The superseded version's valid-time interval must stay
        // append-only -- we deliberately do NOT close it here. The version
        // chain is a transaction-time partition: every write/replay path
        // unconditionally tx-closes the prior head (below), so at any tx
        // coordinate at most one version is visible and current-state reads
        // already skip the superseded version. Closing its valid_to in place at
        // the successor's valid_from would retroactively shrink an interval that
        // earlier-tx-time read snapshots still observe (the prior version's
        // transaction-time window remains open to them), making a node that was
        // alive at snapshot time disappear -- a snapshot-isolation violation on
        // the valid dimension (residual of #3435; #3437 fixed the tx-time half).
        // The tx-close alone is both necessary and sufficient.

        if prev_temporal.is_currently_recorded()
            && new_temporal.transaction_time().start() > prev_temporal.transaction_time().start()
        {
            prev_temporal =
                prev_temporal.close_transaction_time(new_temporal.transaction_time().start())?;
        }

        *prev_version.temporal_mut() = prev_temporal;
        Ok(())
    }

    /// Extract version metadata and data for copy-out reconstruction (nodes).
    ///
    /// This method copies the necessary version chain data while holding the lock,
    /// allowing reconstruction to proceed outside the lock.
    ///
    /// Returns (version metadata, label, data) that can be used for lock-free reconstruction.
    pub fn extract_node_version_data(
        &self,
        version_id: VersionId,
    ) -> Result<(VersionId, NodeId, InternedString, VersionData)> {
        let version = self
            .node_versions
            .get(&version_id)
            .ok_or(StorageError::VersionNotFound(version_id))?;

        // Clone the version data - these are cheap copies (Arc-based)
        Ok((
            version.id,
            version.node_id,
            version.label,
            version.data.clone(),
        ))
    }

    /// Extract version metadata and data for copy-out reconstruction (edges).
    pub fn extract_edge_version_data(
        &self,
        version_id: VersionId,
    ) -> Result<(
        VersionId,
        EdgeId,
        InternedString,
        NodeId,
        NodeId,
        VersionData,
    )> {
        let version = self
            .edge_versions
            .get(&version_id)
            .ok_or(StorageError::VersionNotFound(version_id))?;

        Ok((
            version.id,
            version.edge_id,
            version.label,
            version.source,
            version.target,
            version.data.clone(),
        ))
    }

    /// Get statistics about the storage.
    ///
    /// Issue #212: This method now returns cached counters in O(1) time instead of
    /// iterating through all versions. The counters are maintained incrementally as
    /// versions are added, making stats retrieval constant-time regardless of the
    /// number of versions stored.
    pub fn stats(&self) -> HistoricalStats {
        // Debug assertions to verify counter invariants (zero cost in release builds)
        debug_assert_eq!(
            self.cached_node_anchor_count + self.cached_node_delta_count,
            self.node_versions.len(),
            "Node counter invariant violated: anchors({}) + deltas({}) != total({})",
            self.cached_node_anchor_count,
            self.cached_node_delta_count,
            self.node_versions.len()
        );
        debug_assert_eq!(
            self.cached_edge_anchor_count + self.cached_edge_delta_count,
            self.edge_versions.len(),
            "Edge counter invariant violated: anchors({}) + deltas({}) != total({})",
            self.cached_edge_anchor_count,
            self.cached_edge_delta_count,
            self.edge_versions.len()
        );

        HistoricalStats {
            total_node_versions: self.node_versions.len(),
            total_edge_versions: self.edge_versions.len(),
            // Issue #212: Use cached counters instead of iterating (O(1) vs O(versions))
            node_anchor_count: self.cached_node_anchor_count,
            node_delta_count: self.cached_node_delta_count,
            edge_anchor_count: self.cached_edge_anchor_count,
            edge_delta_count: self.cached_edge_delta_count,
            unique_nodes: self.node_version_heads.len(),
            unique_edges: self.edge_version_heads.len(),
            // Separate regular and anchor cache entries for better visibility (Issue #338)
            node_cache_entries: self.node_property_cache.len(),
            edge_cache_entries: self.edge_property_cache.len(),
            node_anchor_cache_entries: self.node_anchor_cache.len(),
            edge_anchor_cache_entries: self.edge_anchor_cache.len(),
        }
    }

    /// Calculate the average delta chain length across all version chains (Issue #366).
    ///
    /// A delta chain is the run of delta versions that follows an anchor in the
    /// anchor+delta compression scheme. Every chain starts at exactly one anchor,
    /// so the chains partition the delta versions among the anchors and the exact
    /// average chain length is `total deltas / total anchors`, computed over both
    /// node and edge versions.
    ///
    /// The result feeds the query planner's temporal-lookup cost model (via
    /// [`AletheiaDB::refresh_statistics`](crate::db::AletheiaDB::refresh_statistics)):
    /// it estimates how many delta versions a point-in-time reconstruction must
    /// walk back through before reaching an anchor.
    ///
    /// # Returns
    ///
    /// - The actual average delta chain length when historical data exists.
    ///   A storage containing only anchors returns `0.0` (every lookup hits an
    ///   anchor directly).
    /// - [`DEFAULT_AVG_DELTA_CHAIN`] (5.0) when the storage is empty (no anchors),
    ///   matching the query planner's default assumption of `anchor_interval = 10`.
    ///
    /// # Performance
    ///
    /// O(1): reads the incrementally-maintained anchor/delta counters (Issue #212);
    /// no version scan is performed, so this is safe to call on every statistics
    /// refresh.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use aletheiadb::storage::historical::HistoricalStorage;
    /// let storage = HistoricalStorage::new();
    /// // Empty storage falls back to the default estimate
    /// assert_eq!(storage.calculate_avg_delta_chain(), 5.0);
    /// ```
    #[must_use]
    pub fn calculate_avg_delta_chain(&self) -> f64 {
        let total_anchors = self.cached_node_anchor_count + self.cached_edge_anchor_count;
        if total_anchors == 0 {
            // No historical data: fall back to the default estimate
            // (assumes anchor_interval=10, so avg chain length ~5)
            return DEFAULT_AVG_DELTA_CHAIN;
        }

        let total_deltas = self.cached_node_delta_count + self.cached_edge_delta_count;
        total_deltas as f64 / total_anchors as f64
    }

    /// Get cache performance metrics (Improvement #3: Adaptive Cache Sizing).
    ///
    /// Returns granular cache performance metrics that show:
    /// - Primary cache hits (fast path)
    /// - Anchor cache hits (fallback path)
    /// - Full reconstructions (slow path)
    ///
    /// This provides actionable insights for cache tuning:
    /// - High anchor_cache_hits → increase primary cache size
    /// - High full_reconstructions → increase overall cache capacity
    ///
    /// # Example
    /// ```no_run
    /// # use aletheiadb::storage::historical::HistoricalStorage;
    /// let storage = HistoricalStorage::new();
    /// // ... perform some operations ...
    /// let metrics = storage.cache_metrics();
    ///
    /// if let Some(hit_rate) = metrics.hit_rate() {
    ///     println!("Overall cache hit rate: {:.2}%", hit_rate * 100.0);
    /// }
    ///
    /// if let Some(fallback_rate) = metrics.anchor_fallback_rate() {
    ///     if fallback_rate > 0.2 {
    ///         println!("Warning: High anchor cache fallback rate ({:.2}%), \
    ///                   consider increasing primary cache size", fallback_rate * 100.0);
    ///     }
    /// }
    ///
    /// if let Some(recon_rate) = metrics.reconstruction_rate() {
    ///     if recon_rate > 0.2 {
    ///         println!("Warning: High reconstruction rate ({:.2}%), \
    ///                   increase overall cache size", recon_rate * 100.0);
    ///     }
    /// }
    /// ```
    pub fn cache_metrics(&self) -> CacheMetrics {
        CacheMetrics {
            primary_cache_hits: self.primary_cache_hits.load(Ordering::Relaxed),
            anchor_cache_hits: self.anchor_cache_hits.load(Ordering::Relaxed),
            full_reconstructions: self.full_reconstructions.load(Ordering::Relaxed),
        }
    }

    /// Calculate the cache hit rate as a percentage (Improvement #3).
    ///
    /// Returns the cache hit rate as a value between 0.0 and 1.0, or None if
    /// no cache operations have been performed yet.
    ///
    /// # Example
    /// ```no_run
    /// # use aletheiadb::storage::historical::HistoricalStorage;
    /// let storage = HistoricalStorage::new();
    /// // ... perform some operations ...
    /// if let Some(hit_rate) = storage.cache_hit_rate() {
    ///     println!("Cache hit rate: {:.2}%", hit_rate * 100.0);
    /// }
    /// ```
    pub fn cache_hit_rate(&self) -> Option<f64> {
        self.cache_metrics().hit_rate()
    }

    /// Check if the cache should be resized based on hit rate (Improvement #3).
    ///
    /// Returns true if the cache hit rate is below the threshold (default 80%)
    /// and there have been enough operations to make a meaningful assessment
    /// (at least 100 operations).
    ///
    /// # Arguments
    /// * `threshold` - Minimum acceptable hit rate (0.0 to 1.0). Defaults to 0.8 (80%).
    /// * `min_operations` - Minimum number of cache operations before assessment. Defaults to 100.
    ///
    /// # Returns
    /// * `Some(current_hit_rate)` - If resizing is recommended, returns current hit rate
    /// * `None` - If cache performance is acceptable or insufficient data
    ///
    /// # Example
    /// ```no_run
    /// # use aletheiadb::storage::historical::HistoricalStorage;
    /// let storage = HistoricalStorage::new();
    /// // ... perform some operations ...
    ///
    /// if let Some(hit_rate) = storage.should_resize_cache(0.8, 100) {
    ///     println!("Cache hit rate ({:.2}%) is low, consider doubling cache size", hit_rate * 100.0);
    ///     // Create new storage with larger cache:
    ///     // let new_storage = HistoricalStorage::with_config_retention_and_cache_size(
    ///     //     config, retention, current_size * 2
    ///     // );
    /// }
    /// ```
    pub fn should_resize_cache(&self, threshold: f64, min_operations: u64) -> Option<f64> {
        let metrics = self.cache_metrics();
        let total = metrics.total_operations();

        // Need enough operations to make meaningful assessment
        if total < min_operations {
            return None;
        }

        let hit_rate = metrics.hit_rate().unwrap_or(0.0);

        if hit_rate < threshold {
            Some(hit_rate)
        } else {
            None
        }
    }

    /// Get an iterator over all node versions (test-only helper).
    ///
    /// This method provides access to the node versions for integration test
    /// verification purposes. It is public to allow access from integration tests
    /// but is hidden from documentation and marked with `__test_` prefix to
    /// discourage production use.
    ///
    /// **Warning**: This method exposes internal implementation details and
    /// should only be used in tests.
    #[doc(hidden)]
    pub fn __test_get_node_versions_iterator(&self) -> impl Iterator<Item = &NodeVersion> {
        self.node_versions.values()
    }

    /// Get all node versions for persistence.
    ///
    /// This is a crate-internal method used by the index persistence layer.
    pub(crate) fn get_node_versions(&self) -> &FastHashMap<VersionId, NodeVersion> {
        &self.node_versions
    }

    /// Get all edge versions for persistence.
    ///
    /// This is a crate-internal method used by the index persistence layer.
    pub(crate) fn get_edge_versions(&self) -> &FastHashMap<VersionId, EdgeVersion> {
        &self.edge_versions
    }

    /// Reserve capacity for batch restoration from persistence.
    ///
    /// Pre-allocating capacity improves restoration performance by reducing
    /// reallocations during bulk insertion. Call this before restoring
    /// persisted versions.
    ///
    /// # Arguments
    ///
    /// * `node_versions` - Expected number of node versions to restore
    /// * `edge_versions` - Expected number of edge versions to restore
    pub(crate) fn reserve_restoration_capacity(
        &mut self,
        node_versions: usize,
        edge_versions: usize,
    ) {
        self.node_versions.reserve(node_versions);
        self.edge_versions.reserve(edge_versions);
        // Conservatively estimate unique entities as half of versions
        // (typical case: each entity has ~2 versions on average)
        self.node_version_heads.reserve(node_versions / 2);
        self.edge_version_heads.reserve(edge_versions / 2);
        self.node_version_counts.reserve(node_versions / 2);
        self.edge_version_counts.reserve(edge_versions / 2);
    }

    /// Insert a restored node version directly into storage.
    ///
    /// This is used during index loading to restore persisted versions.
    /// Unlike normal version insertion, this bypasses transaction processing
    /// since the data comes from a trusted source (our own persistence layer).
    ///
    /// # Errors
    ///
    /// Returns an error if the version ID or node ID is invalid.
    pub(crate) fn insert_restored_node_version(&mut self, version: NodeVersion) -> Result<()> {
        let version_id = version.id;
        let node_id = version.node_id;
        let is_anchor = version.is_anchor();

        // Store the version
        self.node_versions.insert(version_id, version);

        // Update version head
        self.node_version_heads.insert(node_id, version_id);

        // Update version count
        *self.node_version_counts.entry(node_id).or_insert(0) += 1;

        // Issue #212: Update cached stats counters during persistence restore
        if is_anchor {
            self.cached_node_anchor_count += 1;
        } else {
            self.cached_node_delta_count += 1;
        }

        Ok(())
    }

    /// Insert a restored edge version directly into storage.
    ///
    /// This is used during index loading to restore persisted versions.
    /// Unlike normal version insertion, this bypasses transaction processing
    /// since the data comes from a trusted source (our own persistence layer).
    ///
    /// # Errors
    ///
    /// Returns an error if the version ID or edge ID is invalid.
    pub(crate) fn insert_restored_edge_version(&mut self, version: EdgeVersion) -> Result<()> {
        let version_id = version.id;
        let edge_id = version.edge_id;
        let is_anchor = version.is_anchor();

        // Store the version
        self.edge_versions.insert(version_id, version);

        // Update version head
        self.edge_version_heads.insert(edge_id, version_id);

        // Update version count
        *self.edge_version_counts.entry(edge_id).or_insert(0) += 1;

        // Issue #212: Update cached stats counters during persistence restore
        if is_anchor {
            self.cached_edge_anchor_count += 1;
        } else {
            self.cached_edge_delta_count += 1;
        }

        Ok(())
    }

    /// Rebuild version chains after restoration from persistence.
    ///
    /// This method reconstructs the `prev_version` and `next_version` links for all
    /// versions, and ensures version heads point to the correct (latest tx_time) version.
    /// Must be called after all versions have been inserted via `insert_restored_node_version`
    /// and `insert_restored_edge_version`.
    ///
    /// # Version Chain Semantics
    ///
    /// - Versions are ordered by transaction time (tx_time start)
    /// - `prev_version` points to the temporally previous version (earlier tx_time)
    /// - `next_version` points to the temporally next version (later tx_time)
    /// - Version heads point to the version with the latest tx_time
    ///
    /// # Persisted state is authoritative (Issue #3387)
    ///
    /// Current-format checkpoints persist chain links and tx-time closures
    /// exactly; only *missing* links are filled and only *open* tx intervals
    /// are heuristically closed here, so faithful restored state is never
    /// overwritten. Legacy (pre-#3387) files restore with all links `None`
    /// and all tx intervals open, and get the full heuristic rebuild exactly
    /// as before. Preserving restored links also keeps links that point at
    /// cold-migrated versions absent from the hot map (the heuristic would
    /// rewire around them).
    pub(crate) fn rebuild_version_chains(&mut self) {
        // === Rebuild node version chains ===

        // Group versions by node ID
        let mut node_versions_by_id: FastHashMap<NodeId, Vec<VersionId>> = FastHashMap::default();
        for (vid, version) in &self.node_versions {
            node_versions_by_id
                .entry(version.node_id)
                .or_default()
                .push(*vid);
        }

        // For each node, sort versions by tx_time and link them
        for (node_id, mut version_ids) in node_versions_by_id {
            // Sort by transaction time start (ascending order), tie-breaking on
            // the unique `version_id`. The `version_id` tie-break makes the chain
            // order deterministic when two versions share a `tx_start` (the input
            // `version_ids` come from HashMap iteration, whose order is
            // non-deterministic); without it the head and every reconstructed read
            // would depend on iteration order (AC6 for counterfactual replay).
            // Phase 2: Use TIMESTAMP_MAX instead of i64::MAX
            use crate::core::temporal::TIMESTAMP_MAX;
            version_ids.sort_by_key(|vid| {
                (
                    self.node_versions
                        .get(vid)
                        .map(|v| v.temporal.transaction_time().start())
                        .unwrap_or(TIMESTAMP_MAX),
                    *vid,
                )
            });

            // Link prev/next
            for i in 0..version_ids.len() {
                let vid = version_ids[i];

                // Link to previous version (earlier in time)
                let prev = if i > 0 {
                    Some(version_ids[i - 1])
                } else {
                    None
                };

                // Link to next version (later in time)
                let next = if i < version_ids.len() - 1 {
                    Some(version_ids[i + 1])
                } else {
                    None
                };

                if let Some(version) = self.node_versions.get_mut(&vid) {
                    // Fill only missing links: restored links are exact
                    // (Issue #3387) and must not be overwritten.
                    if version.prev_version.is_none() {
                        version.prev_version = prev;
                    }
                    if version.next_version.is_none() {
                        version.next_version = next;
                    }
                }
            }

            // Fix transaction-time end for non-latest versions.
            // Persistence only stores tx_start; after loading every version has
            // tx_end = TIMESTAMP_MAX.  Reconstruct: version[i].tx_end = version[i+1].tx_start.
            // Guard: skip if next_tx_start <= this version's tx_start to avoid zero-width [T,T) intervals.
            for i in 0..version_ids.len().saturating_sub(1) {
                let next_tx_start = self
                    .node_versions
                    .get(&version_ids[i + 1])
                    .map(|v| v.temporal.transaction_time().start())
                    .unwrap_or(TIMESTAMP_MAX);

                if let Some(version) = self.node_versions.get_mut(&version_ids[i])
                    && version.temporal.transaction_time().is_current()
                    && next_tx_start > version.temporal.transaction_time().start()
                    && let Ok(new_temporal) = version.temporal.close_transaction_time(next_tx_start)
                {
                    version.temporal = new_temporal;
                }
            }

            // Set head to the latest version (last in sorted order)
            if let Some(&latest_vid) = version_ids.last() {
                self.node_version_heads.insert(node_id, latest_vid);

                // Issue #208: Rebuild counter cache for anchor interval checks
                // Count versions since last anchor by walking backwards from head
                let mut count = 0;
                let mut current_id = latest_vid;

                while let Some(version) = self.node_versions.get(&current_id) {
                    if version.is_anchor() {
                        // Found anchor, counter is 0
                        break;
                    }
                    // Delta version, increment counter and continue
                    count += 1;
                    if let Some(prev_id) = version.prev_version {
                        current_id = prev_id;
                    } else {
                        // No more versions (shouldn't happen, first is always anchor)
                        break;
                    }
                }

                self.node_versions_since_anchor.insert(node_id, count);
            }
        }

        // === Rebuild edge version chains ===

        // Group versions by edge ID
        let mut edge_versions_by_id: FastHashMap<EdgeId, Vec<VersionId>> = FastHashMap::default();
        for (vid, version) in &self.edge_versions {
            edge_versions_by_id
                .entry(version.edge_id)
                .or_default()
                .push(*vid);
        }

        // For each edge, sort versions by tx_time and link them
        for (edge_id, mut version_ids) in edge_versions_by_id {
            // Sort by transaction time start (ascending order), tie-breaking on
            // the unique `version_id` for deterministic chain order under equal
            // `tx_start` (see the node-chain comment above).
            // Phase 2: Use TIMESTAMP_MAX (already imported above)
            version_ids.sort_by_key(|vid| {
                (
                    self.edge_versions
                        .get(vid)
                        .map(|v| v.temporal.transaction_time().start())
                        .unwrap_or(TIMESTAMP_MAX),
                    *vid,
                )
            });

            // Link prev/next
            for i in 0..version_ids.len() {
                let vid = version_ids[i];

                // Link to previous version (earlier in time)
                let prev = if i > 0 {
                    Some(version_ids[i - 1])
                } else {
                    None
                };

                // Link to next version (later in time)
                let next = if i < version_ids.len() - 1 {
                    Some(version_ids[i + 1])
                } else {
                    None
                };

                if let Some(version) = self.edge_versions.get_mut(&vid) {
                    // Fill only missing links: restored links are exact
                    // (Issue #3387) and must not be overwritten.
                    if version.prev_version.is_none() {
                        version.prev_version = prev;
                    }
                    if version.next_version.is_none() {
                        version.next_version = next;
                    }
                }
            }

            // Fix transaction-time end for non-latest edge versions (mirror node logic).
            // Guard: skip if next_tx_start <= this version's tx_start to avoid zero-width [T,T) intervals.
            for i in 0..version_ids.len().saturating_sub(1) {
                let next_tx_start = self
                    .edge_versions
                    .get(&version_ids[i + 1])
                    .map(|v| v.temporal.transaction_time().start())
                    .unwrap_or(TIMESTAMP_MAX);

                if let Some(version) = self.edge_versions.get_mut(&version_ids[i])
                    && version.temporal.transaction_time().is_current()
                    && next_tx_start > version.temporal.transaction_time().start()
                    && let Ok(new_temporal) = version.temporal.close_transaction_time(next_tx_start)
                {
                    version.temporal = new_temporal;
                }
            }

            // Set head to the latest version (last in sorted order)
            if let Some(&latest_vid) = version_ids.last() {
                self.edge_version_heads.insert(edge_id, latest_vid);

                // Issue #208: Rebuild counter cache for anchor interval checks
                // Count versions since last anchor by walking backwards from head
                let mut count = 0;
                let mut current_id = latest_vid;

                while let Some(version) = self.edge_versions.get(&current_id) {
                    if version.is_anchor() {
                        // Found anchor, counter is 0
                        break;
                    }
                    // Delta version, increment counter and continue
                    count += 1;
                    if let Some(prev_id) = version.prev_version {
                        current_id = prev_id;
                    } else {
                        // No more versions (shouldn't happen, first is always anchor)
                        break;
                    }
                }

                self.edge_versions_since_anchor.insert(edge_id, count);
            }
        }
    }

    /// Repopulate temporal indexes from existing version data.
    ///
    /// This must be called after both `rebuild_version_chains` (which sets correct
    /// tx_ends) and `set_temporal_indexes` (which wires the index struct in).
    /// It is idempotent: a version already in the index is a no-op duplicate insert.
    pub(crate) fn rebuild_temporal_index_from_versions(&self) {
        if self.node_versions.is_empty() && self.edge_versions.is_empty() {
            return;
        }
        let Some(ref indexes) = self.temporal_indexes else {
            return;
        };
        for (vid, version) in &self.node_versions {
            let _ = indexes.insert_node_version(version.node_id, *vid, version.temporal);
        }
        for (vid, version) in &self.edge_versions {
            let _ = indexes.insert_edge_version(version.edge_id, *vid, version.temporal);
        }
    }

    /// Create an MVCC snapshot of historical storage at the specified LSN.
    ///
    /// This provides snapshot isolation for checkpoint operations, capturing
    /// all node and edge versions at a consistent point in time.
    ///
    /// # Snapshot Isolation
    ///
    /// The snapshot captures Arc references to all versions. Concurrent
    /// modifications after snapshot creation do NOT affect the snapshot's
    /// iteration.
    ///
    /// # Memory Overhead
    ///
    /// - Iterates once over version HashMaps to collect Arc references
    /// - Memory: ~8 bytes per version (just Arc pointers)
    /// - For 10M versions: ~80MB overhead
    ///
    /// # Arguments
    ///
    /// * `lsn` - LSN at which snapshot is taken (for tracking)
    ///
    /// # Returns
    ///
    /// A snapshot that provides isolated iteration over versions.
    pub fn create_snapshot(
        &self,
        lsn: crate::storage::wal::LSN,
    ) -> crate::storage::snapshot::HistoricalStorageSnapshot {
        use crate::storage::snapshot::HistoricalStorageSnapshot;
        use std::sync::Arc;

        let mut node_versions = Vec::with_capacity(self.node_versions.len());
        node_versions.extend(
            self.node_versions
                .values()
                .map(|version| Arc::new(version.clone())),
        );

        let mut edge_versions = Vec::with_capacity(self.edge_versions.len());
        edge_versions.extend(
            self.edge_versions
                .values()
                .map(|version| Arc::new(version.clone())),
        );

        HistoricalStorageSnapshot::new(lsn, node_versions, edge_versions)
    }

    /// **Test-only helper**: Remove a node version from hot storage.
    ///
    /// This is used in tests to simulate version migration to cold storage.
    /// In production, versions are migrated by the `MigrationService` which
    /// atomically moves versions from hot to cold storage.
    ///
    /// # Safety
    /// This method directly modifies internal state and should only be used
    /// in tests. It does not update caches or notify observers.
    #[doc(hidden)]
    pub fn __test_remove_node_version(&mut self, version_id: VersionId) {
        self.node_versions.remove(&version_id);
    }

    /// **Test-only helper**: Clear the property reconstruction cache.
    ///
    /// This is used in tests to force actual property reconstruction instead
    /// of returning cached values. This is essential for testing that reconstruction
    /// works correctly when versions are in cold storage.
    ///
    /// # Safety
    /// This method clears caches and should only be used in tests where you
    /// want to verify reconstruction behavior without cache interference.
    #[doc(hidden)]
    pub fn __test_clear_property_cache(&self) {
        self.node_property_cache.clear();
        self.node_anchor_cache.clear();
    }

    /// **Test-only helper**: Remove an edge version from hot storage.
    ///
    /// Used in tests to simulate version migration or corruption scenarios.
    ///
    /// # Safety
    /// This method directly modifies internal state and should only be used
    /// in tests. It does not update caches or notify observers.
    #[doc(hidden)]
    pub fn __test_remove_edge_version(&mut self, version_id: VersionId) {
        self.edge_versions.remove(&version_id);
    }

    /// **Test-only helper**: Clear the edge property reconstruction cache.
    ///
    /// Used in tests to force actual property reconstruction for edge versions
    /// instead of returning cached values.
    ///
    /// # Safety
    /// This method clears caches and should only be used in tests.
    #[doc(hidden)]
    pub fn __test_clear_edge_property_cache(&self) {
        self.edge_property_cache.clear();
        self.edge_anchor_cache.clear();
    }
}

impl Default for HistoricalStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache performance metrics (Issue #338: Improvement #3).
///
/// Provides granular insight into cache behavior:
/// - `primary_cache_hits`: Fast path hits (most common)
/// - `anchor_cache_hits`: Fallback hits (indicates primary cache pressure)
/// - `full_reconstructions`: Slow path (indicates insufficient cache capacity)
///
/// # Interpretation
/// - High `anchor_cache_hits` + low `primary_cache_hits` → increase primary cache size
/// - High `full_reconstructions` → increase overall cache capacity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheMetrics {
    /// Number of successful lookups in primary property cache (fast path)
    pub primary_cache_hits: u64,
    /// Number of successful lookups in anchor cache fallback
    pub anchor_cache_hits: u64,
    /// Number of full property reconstructions from deltas
    pub full_reconstructions: u64,
}

impl CacheMetrics {
    /// Calculate total cache operations (hits + reconstructions).
    pub fn total_operations(&self) -> u64 {
        self.primary_cache_hits + self.anchor_cache_hits + self.full_reconstructions
    }

    /// Calculate overall cache hit rate (0.0 to 1.0).
    ///
    /// Returns None if no operations have been performed yet.
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.total_operations();
        if total == 0 {
            None
        } else {
            Some((self.primary_cache_hits + self.anchor_cache_hits) as f64 / total as f64)
        }
    }

    /// Calculate primary cache hit rate (0.0 to 1.0).
    ///
    /// This shows how often the primary cache is sufficient without fallback.
    /// Returns None if no operations have been performed yet.
    pub fn primary_hit_rate(&self) -> Option<f64> {
        let total = self.total_operations();
        if total == 0 {
            None
        } else {
            Some(self.primary_cache_hits as f64 / total as f64)
        }
    }

    /// Calculate anchor cache fallback rate (0.0 to 1.0).
    ///
    /// This shows how often we need to fall back to the anchor cache.
    /// High values indicate the primary cache is under pressure.
    pub fn anchor_fallback_rate(&self) -> Option<f64> {
        let total = self.total_operations();
        if total == 0 {
            None
        } else {
            Some(self.anchor_cache_hits as f64 / total as f64)
        }
    }

    /// Calculate reconstruction rate (0.0 to 1.0).
    ///
    /// This shows how often we need to perform full reconstruction.
    /// High values indicate insufficient overall cache capacity.
    pub fn reconstruction_rate(&self) -> Option<f64> {
        let total = self.total_operations();
        if total == 0 {
            None
        } else {
            Some(self.full_reconstructions as f64 / total as f64)
        }
    }
}

/// Statistics about the historical storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalStats {
    /// Total number of node versions stored
    pub total_node_versions: usize,
    /// Total number of edge versions stored
    pub total_edge_versions: usize,
    /// Number of anchor node versions
    pub node_anchor_count: usize,
    /// Number of delta node versions
    pub node_delta_count: usize,
    /// Number of anchor edge versions
    pub edge_anchor_count: usize,
    /// Number of delta edge versions
    pub edge_delta_count: usize,
    /// Number of unique nodes with version history
    pub unique_nodes: usize,
    /// Number of unique edges with version history
    pub unique_edges: usize,
    /// Number of cached node property reconstructions (regular cache)
    pub node_cache_entries: usize,
    /// Number of cached edge property reconstructions (regular cache)
    pub edge_cache_entries: usize,
    /// Number of cached node anchor properties (dedicated anchor cache, Issue #338)
    pub node_anchor_cache_entries: usize,
    /// Number of cached edge anchor properties (dedicated anchor cache, Issue #338)
    pub edge_anchor_cache_entries: usize,
}

impl HistoricalStats {
    /// Calculate the compression ratio (anchors vs total versions).
    pub fn compression_ratio(&self) -> f64 {
        let total_versions = self.total_node_versions + self.total_edge_versions;
        let total_anchors = self.node_anchor_count + self.edge_anchor_count;

        if total_versions == 0 {
            return 1.0;
        }

        total_anchors as f64 / total_versions as f64
    }

    /// Estimate total cache memory usage in bytes (Issue #338: Memory Accounting).
    ///
    /// Provides rough estimate of memory consumed by all caches. Actual memory
    /// usage may vary based on property sizes, Arc overhead, and allocator behavior.
    ///
    /// # Formula
    /// Per entry overhead:
    /// - VersionId: 8 bytes
    /// - Arc pointer: 8 bytes
    /// - PropertyMap overhead: ~16 bytes
    /// - Average property data: ~100 bytes (varies by use case)
    /// - Total: ~132 bytes per entry (rounded to 150 for safety margin)
    ///
    /// # Returns
    /// Estimated bytes used by all caches (primary + anchor)
    ///
    /// # Example
    /// ```no_run
    /// # use aletheiadb::storage::historical::HistoricalStorage;
    /// let storage = HistoricalStorage::new();
    /// // ... perform operations ...
    /// let stats = storage.stats();
    /// let bytes = stats.estimated_cache_memory_bytes();
    /// println!("Cache using ~{:.2} MB", bytes as f64 / 1_048_576.0);
    /// ```
    pub fn estimated_cache_memory_bytes(&self) -> usize {
        // Rough estimate per cache entry:
        // - VersionId (u64): 8 bytes
        // - Arc<PropertyMap> pointer: 8 bytes
        // - PropertyMap struct overhead: ~16 bytes
        // - Average property data: ~100 bytes (varies widely)
        // Total: ~132 bytes, rounded to 150 for safety margin
        const BYTES_PER_ENTRY: usize = 150;

        let total_entries = self.node_cache_entries
            + self.edge_cache_entries
            + self.node_anchor_cache_entries
            + self.edge_anchor_cache_entries;

        total_entries * BYTES_PER_ENTRY
    }
}

#[cfg(test)]
mod tests;
