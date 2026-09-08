//! Result Iterators
//!
//! Pull-based iterators for query execution. Each physical operator
//! has a corresponding iterator that lazily produces results.

use parking_lot::RwLock;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::sync::Arc;

#[cfg(feature = "observability")]
use tracing;

use crate::core::error::Result;
use crate::core::graph::Node;
use crate::core::interning::GLOBAL_INTERNER;
use crate::core::namespace::{NamespaceScope, ResolvedScope};
use crate::core::property::{PropertyMap, PropertyValue};
use crate::core::provenance::Provenance;
use crate::core::vector::DistanceMetric as VectorMetric;
use crate::core::{EdgeId, NodeId, Timestamp};
use crate::query::ir::{
    AggregateArg, AggregateFunc, AggregateGroupKey, AggregateSpec, Direction, Predicate,
    PredicateValue, ProvenanceField, ProvenancePredicate, ProvenanceProjection, ScoreThreshold,
    SortKey, TemporalWindowSpec, WindowAggArg, WindowAggFunc,
};
use crate::query::ir::{AlignNodeSource, TemporalAlignSpec};
use crate::storage::current::CurrentStorage;
use crate::storage::historical::HistoricalStorage;
use std::cell::RefCell;

use super::results::{EntityId, EntityResult, QueryRow};

/// Trait for result iteration (pull-based).
///
/// Query execution uses a pull-based iterator model, where each physical
/// operator is implemented as an iterator. Calling `next()` pulls results
/// sequentially through the pipeline.
pub trait ResultIterator: Send {
    /// Get the next result row
    fn next(&mut self) -> Option<Result<QueryRow>>;

    /// Estimate the remaining results
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

/// Empty iterator that produces no results.
///
/// Used for query plans that evaluate to empty at planning time.
pub struct EmptyIterator;

impl ResultIterator for EmptyIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

/// Direct node lookup iterator.
///
/// Yields nodes corresponding to a specific list of IDs. O(1) per node.
///
/// # Examples
///
/// ```ignore
/// use aletheiadb::query::executor::NodeLookupIterator;
/// use aletheiadb::storage::current::CurrentStorage;
/// use aletheiadb::core::id::NodeId;
/// use std::sync::Arc;
///
/// // Assuming `current` is a valid Arc<CurrentStorage>
/// let node_ids = vec![NodeId::new(1).unwrap(), NodeId::new(2).unwrap()];
/// let iter = NodeLookupIterator::new(node_ids, current);
/// ```
pub struct NodeLookupIterator {
    node_ids: std::vec::IntoIter<NodeId>,
    current: Arc<CurrentStorage>,
}

impl NodeLookupIterator {
    /// Initialize the iterator with a predefined list of node identifiers.
    ///
    /// # Why?
    /// This is used for `NodeLookup` physical operations where the query
    /// planner has already resolved exact node IDs (e.g., from an index or literal).
    pub fn new(node_ids: Vec<NodeId>, current: Arc<CurrentStorage>) -> Self {
        NodeLookupIterator {
            node_ids: node_ids.into_iter(),
            current,
        }
    }
}

impl ResultIterator for NodeLookupIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        self.node_ids.next().map(|id| {
            self.current
                .get_node(id)
                .map(|node| QueryRow::from_entity(EntityResult::Node(node)))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.node_ids.size_hint()
    }
}

/// Label filter state for a [`NodeScanIterator`], resolved once at construction.
///
/// Resolving the requested label to its interned id up front lets the scan use
/// the fast 16-byte `node_headers` path (see [`CurrentStorage::node_has_label`])
/// instead of loading and cloning a full `Node` just to inspect its label.
enum ScanFilter {
    /// No label filter: every existing node is yielded.
    All,
    /// Yield only nodes whose interned label equals this id.
    Label(crate::core::interning::InternedString),
    /// A label was requested but has never been interned, so no node can match.
    /// The scan yields nothing (as opposed to [`ScanFilter::All`]).
    None,
}

/// Sequential node scan iterator, optionally applying a label filter.
///
/// # Memory Behavior
///
/// This iterator streams node ids lazily over the half-open range
/// `[0, max_id)`, where `max_id` is current storage's insert-maintained node-id
/// high-water-mark (see [`CurrentStorage::get_max_node_id`]), captured once at
/// construction. It never materializes the full id set, so each `next()` call
/// allocates O(1) regardless of graph size. This makes a full scan immune to
/// the out-of-memory failure that eager `Vec<NodeId>` materialization caused on
/// very large graphs.
///
/// Ids with no live node (gaps from deletion, or ids reserved but never used)
/// are skipped cheaply: both label and unfiltered scans reject them via the
/// compact 16-byte header fast path before loading any full node.
///
/// # Performance
///
/// The iterator picks one of two strategies at construction (Issue #3422),
/// based on O(1) reads of the live node count and the id high-water mark:
///
/// - **Sweep** (dense / huge-live / tiny id spaces): streams ids over
///   `[0, max_id)` exactly as PR #3418 did. Resident memory is O(1), time is
///   O(max_id). Ids with no live node are skipped cheaply via the 16-byte
///   header fast path. This is optimal when the id space is densely populated,
///   since `max_id` is close to the live count.
/// - **Paged** (sparse id spaces): pages the live keys of `node_headers` in
///   ascending-id chunks of `page_size` (K), yielding each page's ids before
///   fetching the next. Dead ids (deletion tombstones, reserved-but-unused
///   ids, post-compaction gaps) are **never visited**, so time is O(live)
///   rather than O(max_id), while resident memory stays O(K) -- it never
///   materializes the whole id set (the OOM vector PR #3418 removed).
///
/// The sparse/paged path is chosen only when it does **comfortably** less work
/// than the sweep -- at least 2x less (`2 * live * ceil(live/K) < max_id`) --
/// and the live count is bounded, so dense *and near-dense* graphs never regress
/// (a graph with only a few deletion gaps stays on the sweep rather than paying
/// a heap-sort to dodge them), while a once-huge-now-tiny graph (e.g. 1B ids
/// ever allocated, 1K now live) recovers O(live) full-scan time instead of
/// sweeping ~1B dead ids. A fully bulk-deleted graph (0 live, huge `max_id`) is
/// special-cased to an O(1) exhausted scan rather than an O(max_id) sweep of all
/// dead ids. Node ids are monotonic and never reused, so `max_id` is the
/// high-water mark of ids ever allocated, not the live count -- which is
/// exactly why the naive sweep degraded.
///
/// # Concurrency
///
/// Neither strategy holds a DashMap shard lock across a yield. The sweep
/// acquires a shard lock only for the duration of a single `node_has_label` /
/// `contains_node` / `get_node` call. The paged path collects each page under
/// brief per-shard locks that are all released before any row is yielded (see
/// [`CurrentStorage::collect_node_id_page`]); between yields no lock is held.
/// So the iterator cannot deadlock against concurrent writers and does not
/// violate the crate's lock acquisition order. Both the `max_id` sweep bound
/// and the paged key snapshot are relaxed (non-isolated): nodes created or
/// deleted after a page is captured may or may not be observed, which is the
/// expected semantics for an unsynchronized full scan.
///
/// # Examples
///
/// ```ignore
/// use aletheiadb::query::executor::NodeScanIterator;
/// use aletheiadb::storage::current::CurrentStorage;
/// use std::sync::Arc;
///
/// // Assuming `current` is a valid Arc<CurrentStorage>
/// let iter = NodeScanIterator::new(Some("Person".to_string()), current);
/// ```
pub struct NodeScanIterator {
    filter: ScanFilter,
    current: Arc<CurrentStorage>,
    /// Which scanning strategy is in flight, plus its live cursor state.
    mode: ScanMode,
    /// Page size (K) for the paged strategy; ignored by the sweep.
    page_size: usize,
    /// Instrumentation: number of candidate ids examined so far. In the sweep
    /// path this counts every id in `[0, max_id)` visited (including dead ids
    /// skipped by the header fast paths). In the paged path it counts every live
    /// candidate the per-page enumeration inspected across all pages -- i.e. the
    /// real `live * pages` re-scan cost, **not** just the live ids materialized
    /// (a single page examines `live` candidates to retain up to `K`, so
    /// counting only retained ids would understate a multi-page scan by a factor
    /// of `pages`). It is the test/bench proxy for scan work, exposed via
    /// [`Self::work_units`], and lets a test assert that a full scan's cost
    /// tracks the live node count rather than `max_id`.
    work_units: u64,
}

/// Default page size (K) for the chunked `node_headers` scan (Issue #3422).
///
/// Bounds resident memory of the paged strategy to O(K) node ids while keeping
/// per-page overhead amortized. Tunable in tests/benches via
/// [`NodeScanIterator::with_strategy`].
const DEFAULT_NODE_SCAN_PAGE_SIZE: usize = 4096;

/// Above this live-node count the paged strategy's per-page re-scan cost stops
/// paying for itself, so the scan stays on the memory-flat sweep even for a
/// sparse id space. See [`NodeScanIterator::prefer_paged`].
const MAX_PAGED_LIVE_NODES: u64 = 2_000_000;

/// Full-scan strategy selector (Issue #3422). `Auto` picks sweep vs paged from
/// the live count and id high-water mark; the `Force*` variants are test/bench
/// hooks to exercise a specific path deterministically.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanStrategy {
    /// Choose sweep or paged automatically from live count vs `max_id`.
    Auto,
    /// Always use the dense `[0, max_id)` sweep (PR #3418 behavior).
    ForceSweep,
    /// Always use the chunked `node_headers` keyset paging (Issue #3422).
    ForcePaged,
}

/// Internal scan strategy state.
enum ScanMode {
    /// Dense streaming sweep of `[0, max_id)` (PR #3418).
    Sweep { current_id: u64, max_id: u64 },
    /// Chunked keyset paging over live `node_headers` keys (Issue #3422).
    ///
    /// `buffer` drains the current ascending-id page; `cursor` is the last id
    /// yielded (the keyset boundary for the next page); `exhausted` is set once
    /// a short/empty page proves the live key space after the cursor is drained.
    Paged {
        buffer: std::vec::IntoIter<NodeId>,
        cursor: Option<u64>,
        exhausted: bool,
    },
}

impl NodeScanIterator {
    /// Create a new NodeScanIterator, auto-selecting the scan strategy.
    ///
    /// See the type-level `# Performance` docs for how the sweep vs paged
    /// strategy is chosen; behavior and results are identical to the prior
    /// sweep-only iterator, only the traversal cost differs.
    pub fn new(label: Option<String>, current: Arc<CurrentStorage>) -> Self {
        Self::with_strategy(
            label,
            current,
            ScanStrategy::Auto,
            DEFAULT_NODE_SCAN_PAGE_SIZE,
        )
    }

    /// Construct with an explicit [`ScanStrategy`] and page size.
    ///
    /// Public but hidden: `new` (auto strategy, default page size) is the
    /// supported entry point. This exists so tests/benches can pin a specific
    /// strategy and page size to measure before/after scan cost deterministically.
    #[doc(hidden)]
    pub fn with_strategy(
        label: Option<String>,
        current: Arc<CurrentStorage>,
        strategy: ScanStrategy,
        page_size: usize,
    ) -> Self {
        // Resolve the label to an interned id exactly once. A requested-but-
        // unknown label short-circuits to `None` so the scan yields nothing,
        // rather than degrading to an unfiltered scan.
        let filter = match label {
            Option::None => ScanFilter::All,
            Some(ref l) => match GLOBAL_INTERNER.get_id(l) {
                Some(id) => ScanFilter::Label(id),
                Option::None => ScanFilter::None,
            },
        };

        let page_size = page_size.max(1);

        let mode = if matches!(filter, ScanFilter::None) {
            // No node can ever match; an empty sweep drains immediately without
            // touching storage.
            ScanMode::Sweep {
                current_id: 0,
                max_id: 0,
            }
        } else {
            // Snapshot the id upper bound once. `get_max_node_id` returns the
            // index's insert-maintained high-water-mark (`max id ever inserted
            // + 1`) rather than any id generator, so a sweep of `[0, max_id)`
            // covers every node present regardless of which generator allocated
            // its id.
            let max_id = current.get_max_node_id();
            match strategy {
                ScanStrategy::ForceSweep => ScanMode::Sweep {
                    current_id: 0,
                    max_id,
                },
                ScanStrategy::ForcePaged => Self::new_paged_mode(),
                ScanStrategy::Auto => {
                    Self::choose_mode(current.node_count() as u64, max_id, page_size)
                }
            }
        };

        NodeScanIterator {
            filter,
            current,
            mode,
            page_size,
            work_units: 0,
        }
    }

    /// Fresh, empty paged-mode state (cursor at the start, nothing buffered).
    fn new_paged_mode() -> ScanMode {
        ScanMode::Paged {
            buffer: Vec::new().into_iter(),
            cursor: None,
            exhausted: false,
        }
    }

    /// Pick sweep vs paged from O(1) counters (Issue #3422).
    fn choose_mode(live: u64, max_id: u64, page_size: usize) -> ScanMode {
        // Zero live nodes with a non-trivial id high-water mark is the exact
        // pathology this PR targets: e.g. a fully bulk-deleted graph where
        // `node_count() == 0` but `max_id` is still the (possibly huge)
        // high-water mark of ids ever allocated. A `[0, max_id)` sweep would
        // examine every dead id just to yield nothing -- O(max_id). Start in an
        // already-exhausted paged mode so the scan does O(1) work instead.
        if live == 0 {
            return ScanMode::Paged {
                buffer: Vec::new().into_iter(),
                cursor: None,
                exhausted: true,
            };
        }
        if Self::prefer_paged(live, max_id, page_size) {
            Self::new_paged_mode()
        } else {
            ScanMode::Sweep {
                current_id: 0,
                max_id,
            }
        }
    }

    /// Whether the chunked paged strategy is expected to beat the dense sweep.
    ///
    /// The sweep probes `max_id` ids. The paged strategy re-scans the live
    /// header set once per page, so it costs about `live * ceil(live / K)`
    /// header reads across `ceil(live / K)` pages. Prefer paging only when that
    /// paged cost is comfortably cheaper than the sweep -- specifically at least
    /// **2x** cheaper (`2 * paged_cost < max_id`) and the live count is bounded.
    ///
    /// The 2x margin is deliberate: without it, a *near-dense* graph (say
    /// `live == max_id - few`, a handful of deletion gaps) would page purely to
    /// dodge a few cheap gap-skips, paying a heap-sort and per-page re-scan for
    /// no real win -- contradicting the "strictly less work" intent. Requiring a
    /// clear margin keeps dense and near-dense graphs on the memory-flat sweep
    /// and reserves paging for genuinely sparse id spaces (a once-huge-now-tiny
    /// graph), where it recovers O(live) time. The live bound additionally keeps
    /// a huge-live graph -- where the sweep is already ~O(live) and memory-flat
    /// -- off the quadratic re-scan. (`live == 0` is handled earlier in
    /// [`Self::choose_mode`] as an O(1) exhausted scan, so it never reaches
    /// here.)
    fn prefer_paged(live: u64, max_id: u64, page_size: usize) -> bool {
        if live == 0 || live > MAX_PAGED_LIVE_NODES {
            return false;
        }
        let k = page_size.max(1) as u128;
        let pages = (live as u128).div_ceil(k);
        let paged_cost = (live as u128).saturating_mul(pages);
        // Require paged to be at least 2x cheaper than the sweep (see above).
        paged_cost.saturating_mul(2) < max_id as u128
    }

    /// Total candidate ids examined so far (see the [`work_units`] field docs).
    ///
    /// Test/bench instrumentation, not part of the query contract.
    ///
    /// [`work_units`]: Self::work_units
    #[doc(hidden)]
    pub fn work_units(&self) -> u64 {
        self.work_units
    }

    /// Collect one page of live node ids honoring the active label filter.
    ///
    /// Returns the page plus the number of live candidates the enumeration
    /// **examined** to build it (the per-page re-scan cost, `~live`), which the
    /// caller folds into `work_units` so the paged work proxy reflects the true
    /// `live * pages` cost rather than only the materialized ids.
    ///
    /// Static (no `self`) so the caller can borrow `current`/`filter`
    /// immutably alongside a mutable borrow of `self.mode` (disjoint fields).
    fn collect_page(
        current: &CurrentStorage,
        filter: &ScanFilter,
        after: Option<u64>,
        k: usize,
    ) -> (Vec<NodeId>, u64) {
        match filter {
            ScanFilter::All => current.collect_node_id_page_counted(after, k, None),
            ScanFilter::Label(label_id) => {
                current.collect_node_id_page_counted(after, k, Some(*label_id))
            }
            ScanFilter::None => (Vec::new(), 0),
        }
    }

    /// Refill the paged buffer with the next chunk of live node ids.
    ///
    /// Returns `false` when the scan is exhausted (no more live ids after the
    /// cursor); otherwise the buffer holds at least one id.
    fn refill_page(&mut self) -> bool {
        let (after, already_exhausted) = match &self.mode {
            ScanMode::Paged {
                cursor, exhausted, ..
            } => (*cursor, *exhausted),
            ScanMode::Sweep { .. } => return false,
        };
        if already_exhausted {
            return false;
        }

        let k = self.page_size;
        // Disjoint borrows: `current`/`filter` immutable, `mode` untouched here.
        let (page, examined) = Self::collect_page(&self.current, &self.filter, after, k);
        // The per-page enumeration examined `examined` live candidates; that is
        // the real cost this page paid, so fold it into the work proxy here (at
        // page granularity) rather than per yielded id.
        self.work_units += examined;

        match &mut self.mode {
            ScanMode::Paged {
                buffer,
                cursor,
                exhausted,
            } => {
                if page.is_empty() {
                    *exhausted = true;
                    return false;
                }
                // A short page means the live key space after the cursor is
                // drained; mark exhausted so the next drain stops without a
                // fruitless re-scan.
                if page.len() < k {
                    *exhausted = true;
                }
                *cursor = page.last().map(|id| id.as_u64());
                *buffer = page.into_iter();
                true
            }
            ScanMode::Sweep { .. } => false,
        }
    }

    /// Advance the dense `[0, max_id)` sweep (PR #3418 behavior).
    fn next_sweep(&mut self) -> Option<Result<QueryRow>> {
        loop {
            let id_val = match &mut self.mode {
                ScanMode::Sweep { current_id, max_id } => {
                    if *current_id >= *max_id {
                        return None;
                    }
                    let v = *current_id;
                    *current_id += 1;
                    v
                }
                ScanMode::Paged { .. } => return None,
            };
            self.work_units += 1;

            // Fast path: reject non-matching labels via the compact node header
            // before ever loading the full node.
            if let ScanFilter::Label(label_id) = self.filter
                && !self.current.node_has_label(id_val, label_id)
            {
                continue;
            }

            // Fast path: for an unfiltered scan, skip gaps in the sparse id
            // space with a cheap O(1) header existence check.
            if matches!(self.filter, ScanFilter::All) && !self.current.contains_node(id_val) {
                continue;
            }

            let id = match NodeId::new(id_val) {
                Ok(id) => id,
                Err(_) => continue,
            };

            match self.current.get_node(id) {
                Ok(node) => return Some(Ok(QueryRow::from_entity(EntityResult::Node(node)))),
                // Sparse id space: skip ids with no live node.
                Err(crate::core::error::Error::Storage(
                    crate::core::error::StorageError::NodeNotFound(_),
                )) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }

    /// Advance the chunked `node_headers` keyset paging (Issue #3422).
    fn next_paged(&mut self) -> Option<Result<QueryRow>> {
        loop {
            // Pull the next id from the current page, refilling as it drains.
            let id_val = loop {
                let next = match &mut self.mode {
                    ScanMode::Paged { buffer, .. } => buffer.next(),
                    ScanMode::Sweep { .. } => return None,
                };
                match next {
                    Some(id) => break id.as_u64(),
                    None => {
                        if !self.refill_page() {
                            return None;
                        }
                    }
                }
            };

            // Note: work is accounted per-page in `refill_page` (the enumeration
            // cost), not per yielded id, so the paged work proxy reflects the
            // true `live * pages` re-scan cost. Do not increment here.

            let id = match NodeId::new(id_val) {
                Ok(id) => id,
                Err(_) => continue,
            };

            match self.current.get_node(id) {
                Ok(node) => return Some(Ok(QueryRow::from_entity(EntityResult::Node(node)))),
                // The id was live when the page was captured but was deleted
                // before this load; skip it (relaxed snapshot semantics).
                Err(crate::core::error::Error::Storage(
                    crate::core::error::StorageError::NodeNotFound(_),
                )) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

impl ResultIterator for NodeScanIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        // A label that was never interned can match no node.
        if matches!(self.filter, ScanFilter::None) {
            return None;
        }

        match self.mode {
            ScanMode::Sweep { .. } => self.next_sweep(),
            ScanMode::Paged { .. } => self.next_paged(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if matches!(self.filter, ScanFilter::None) {
            return (0, Some(0));
        }
        match &self.mode {
            // Upper bound only: the range may contain gaps that are skipped.
            ScanMode::Sweep { current_id, max_id } => {
                (0, Some(max_id.saturating_sub(*current_id) as usize))
            }
            // Paged: the remaining live count is not cheaply known here.
            ScanMode::Paged { .. } => (0, None),
        }
    }
}

/// Edge-type filter state for an [`EdgeScanIterator`], resolved once at
/// construction.
///
/// Mirrors [`ScanFilter`] for the node scan: a requested edge type is resolved
/// to its interned id up front so each candidate edge is accepted/rejected by a
/// cheap `InternedString` comparison rather than a string compare.
enum EdgeScanFilter {
    /// No edge-type filter: every existing edge is yielded.
    All,
    /// Yield only edges whose interned label equals this id.
    Type(crate::core::interning::InternedString),
    /// An edge type was requested but has never been interned, so no edge can
    /// match. The scan yields nothing (as opposed to [`EdgeScanFilter::All`]).
    None,
}

/// Sequential edge scan iterator, optionally applying an edge-type filter.
///
/// This is the edge counterpart to [`NodeScanIterator`] and backs the SQL
/// `SELECT * FROM edges` path (`PhysicalOp::EdgeScan`). Edges have no id
/// high-water mark analogous to [`CurrentStorage::get_max_node_id`], so the
/// dense `[0, max_id)` sweep the node scan uses does not apply. Instead the
/// iterator snapshots the live edge id set once at construction
/// (via [`CurrentStorage::get_all_edge_ids`] -- an id-only snapshot, which does
/// not clone the edges) and then materializes each [`Edge`] lazily with
/// [`CurrentStorage::get_edge`] as `next()` is pulled. Only 8-byte ids are held
/// resident; edge payloads are never all materialized at once.
///
/// # Filtering
///
/// An edge type is resolved to its interned id once. A requested-but-never-
/// interned type short-circuits to an empty scan (mirroring the node scan's
/// unknown-label semantics: unknown type yields nothing, **not** an unfiltered
/// scan). SQL always emits `edge_type: None` (a bare `FROM edges`), so the
/// `Type` path is exercised at the iterator/IR level; a `WHERE` predicate over
/// edge properties is applied by the `FilterIterator` layered above this scan.
///
/// # Concurrency
///
/// No storage lock is held across a `next()` yield: the id snapshot is taken
/// once, and each per-edge `get_edge` acquires and releases its shard lock
/// internally. An edge deleted after the snapshot but before its `get_edge`
/// surfaces as `EdgeNotFound` and is skipped (relaxed, non-isolated snapshot
/// semantics, exactly like the node scan tolerating deleted ids).
pub struct EdgeScanIterator {
    filter: EdgeScanFilter,
    edge_ids: std::vec::IntoIter<EdgeId>,
    current: Arc<CurrentStorage>,
}

impl EdgeScanIterator {
    /// Create a new edge scan, optionally filtered to a single edge type.
    ///
    /// Passing `None` scans every edge; passing `Some(type)` yields only edges
    /// of that type (or nothing, if the type was never interned).
    pub fn new(edge_type: Option<String>, current: Arc<CurrentStorage>) -> Self {
        // Resolve the requested type to an interned id exactly once. An unknown
        // type short-circuits to `None` so the scan yields nothing rather than
        // degrading to an unfiltered scan.
        let filter = match edge_type {
            Option::None => EdgeScanFilter::All,
            Some(ref t) => match GLOBAL_INTERNER.get_id(t) {
                Some(id) => EdgeScanFilter::Type(id),
                Option::None => EdgeScanFilter::None,
            },
        };

        // Snapshot the live edge id set once. For an unknown type the snapshot
        // is skipped entirely (no edge can match), so the scan drains
        // immediately without touching storage.
        let edge_ids = if matches!(filter, EdgeScanFilter::None) {
            Vec::new()
        } else {
            current.get_all_edge_ids()
        };

        EdgeScanIterator {
            filter,
            edge_ids: edge_ids.into_iter(),
            current,
        }
    }
}

impl ResultIterator for EdgeScanIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        // An edge type that was never interned can match no edge.
        if matches!(self.filter, EdgeScanFilter::None) {
            return None;
        }

        loop {
            let id = self.edge_ids.next()?;

            match self.current.get_edge(id) {
                Ok(edge) => {
                    // Apply the edge-type filter on the fully-loaded edge.
                    if let EdgeScanFilter::Type(type_id) = self.filter
                        && edge.label != type_id
                    {
                        continue;
                    }
                    return Some(Ok(QueryRow::from_entity(EntityResult::Edge(edge))));
                }
                // The id was live when the snapshot was taken but was deleted
                // before this load; skip it (relaxed snapshot semantics).
                Err(crate::core::error::Error::Storage(
                    crate::core::error::StorageError::EdgeNotFound(_),
                )) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if matches!(self.filter, EdgeScanFilter::None) {
            return (0, Some(0));
        }
        // Upper bound only: a type filter or a concurrent deletion may drop ids.
        (0, Some(self.edge_ids.len()))
    }
}

/// Iterator for vector search results.
///
/// # Context
/// Transforms raw `(NodeId, score)` pairs from a vector search operation
/// (like `HnswSearch`) into fully populated `QueryRow` results containing
/// the actual `Node` entities.
///
/// # Details
/// Performs lazy, row-by-row lookups against `CurrentStorage`. This ensures that
/// node properties are only materialized in memory when `next()` is explicitly called.
///
/// # Panics
/// Does not panic. If a node ID from the search results is no longer found in storage
/// (e.g., due to a concurrent deletion), it returns an `Err` which is yielded to the caller.
///
/// # Examples
///
/// ```rust
/// # use std::sync::Arc;
/// # use aletheiadb::storage::current::CurrentStorage;
/// # use aletheiadb::core::id::NodeId;
/// # use aletheiadb::query::executor::VectorResultIterator;
/// # use aletheiadb::query::executor::ResultIterator;
/// #
/// # let current = Arc::new(CurrentStorage::new());
/// # let node_id = current.create_node("Doc", aletheiadb::core::PropertyMapBuilder::new().build()).unwrap();
/// // Raw results from an HNSW index search
/// let raw_results = vec![(node_id, 0.95), (node_id, 0.85)];
///
/// let mut iter = VectorResultIterator::new(raw_results, current);
///
/// while let Some(result) = iter.next() {
///     let row = result.expect("Node should exist");
///     println!("Found node {:?} with similarity score: {}", row.entity, row.score.unwrap());
/// }
/// ```
pub struct VectorResultIterator {
    results: std::vec::IntoIter<(NodeId, f32)>,
    current: Arc<CurrentStorage>,
}

impl VectorResultIterator {
    /// Create a new VectorResultIterator.
    pub fn new(results: Vec<(NodeId, f32)>, current: Arc<CurrentStorage>) -> Self {
        VectorResultIterator {
            results: results.into_iter(),
            current,
        }
    }
}

impl ResultIterator for VectorResultIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        self.results.next().map(|(node_id, score)| {
            self.current
                .get_node(node_id)
                .map(|node| QueryRow::with_score(EntityResult::Node(node), score))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.results.size_hint()
    }
}

/// Reconstruct a node as it existed at `(valid_time, transaction_time)` from an
/// already-locked historical storage (Issue #356).
///
/// This is the single, flat implementation of the temporal reconstruction path
/// shared by [`TemporalNodeIterator`], [`BatchTemporalNodeIterator`], and
/// [`TemporalNodeScanIterator`], replacing the previously duplicated (and
/// originally deeply nested) per-iterator logic:
///
/// 1. Find the version valid at the requested bi-temporal point.
/// 2. Retrieve the version metadata (validates that
///    `find_node_version_at_time` only returns existing version IDs).
/// 3. Reconstruct properties from the anchor+delta compression.
/// 4. Build a `Node` with the historical label and properties.
///
/// The caller decides the locking strategy (per-node vs batch) and passes the
/// already-locked storage, so this helper never affects lock semantics.
///
/// # Errors
///
/// - [`TemporalError::NodeNotFoundAtTime`](crate::core::error::TemporalError::NodeNotFoundAtTime)
///   if no version exists at the requested time point
/// - [`TemporalError::VersionNotFound`](crate::core::error::TemporalError::VersionNotFound)
///   if version metadata is missing (data inconsistency)
/// - Any error from property reconstruction
fn reconstruct_node_at(
    historical: &HistoricalStorage,
    node_id: NodeId,
    valid_time: Timestamp,
    transaction_time: Timestamp,
) -> Result<Node> {
    // Step 1: Find the version valid at the requested time
    let version_id = historical
        .find_node_version_at_time(node_id, valid_time, transaction_time)
        .ok_or(crate::core::error::TemporalError::NodeNotFoundAtTime {
            node_id,
            valid_time,
            transaction_time,
        })?;

    // Step 2: Get the version metadata (also validates the invariant that
    // find_node_version_at_time only returns existing version IDs)
    let version = historical.get_node_version(version_id).ok_or(
        crate::core::error::TemporalError::VersionNotFound(version_id),
    )?;

    // Step 3: Reconstruct the properties from the version
    let properties = historical.reconstruct_node_properties(version_id)?;

    // Step 4: Build and return the node with the historical data
    Ok(Node::new(node_id, version.label, properties, version_id))
}

/// Iterator for temporal node lookups.
///
/// # Context
/// Reconstructs nodes at a specific point in bi-temporal time by querying
/// historical storage. It transforms a sequence of `NodeId`s into fully
/// populated `Node`s representing their exact state at `(valid_time, transaction_time)`.
///
/// # Details
/// The reconstruction process:
/// 1. Finds the version valid at the requested bi-temporal point.
/// 2. Reconstructs properties from the version using the anchor+delta compression strategy.
/// 3. Returns a `Node` with the historical label and properties.
///
/// This iterator acquires a brief, per-node read lock on `HistoricalStorage`.
/// For bulk queries where lock overhead is a concern, use [`BatchTemporalNodeIterator`] instead.
///
/// # Panics
/// Does not panic. If a node or version is not found, or if property reconstruction fails,
/// it returns an `Err(TemporalError)`.
///
/// # Examples
///
/// ```rust
/// # use std::sync::Arc;
/// # use parking_lot::RwLock;
/// # use aletheiadb::storage::historical::HistoricalStorage;
/// # use aletheiadb::core::id::NodeId;
/// # use aletheiadb::core::temporal::time;
/// # use aletheiadb::query::executor::TemporalNodeIterator;
/// # use aletheiadb::query::executor::ResultIterator;
/// #
/// # let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
/// # let node_id = NodeId::new(1).unwrap();
/// let now = time::now();
/// let node_ids = vec![node_id];
///
/// let mut iter = TemporalNodeIterator::new(
///     node_ids,
///     now, // valid_time
///     now, // transaction_time
///     historical
/// );
///
/// // Iterate over the historical states
/// while let Some(result) = iter.next() {
///     // Handle potential TemporalError if node didn't exist at `now`
///     if let Ok(row) = result {
///         println!("Historical node state: {:?}", row.entity);
///     }
/// }
/// ```
pub struct TemporalNodeIterator {
    node_ids: std::vec::IntoIter<NodeId>,
    valid_time: Timestamp,
    transaction_time: Timestamp,
    historical: Arc<RwLock<HistoricalStorage>>,
}

impl TemporalNodeIterator {
    /// Create a new TemporalNodeIterator.
    pub fn new(
        node_ids: Vec<NodeId>,
        valid_time: Timestamp,
        transaction_time: Timestamp,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Self {
        TemporalNodeIterator {
            node_ids: node_ids.into_iter(),
            valid_time,
            transaction_time,
            historical,
        }
    }
}

impl ResultIterator for TemporalNodeIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        self.node_ids.next().map(|id| {
            // Acquire read lock on historical storage (per-node)
            // For bulk queries, use BatchTemporalNodeIterator instead
            let historical = self.historical.read();

            // Reconstruct the node at the requested bi-temporal point (Issue #356)
            let node =
                reconstruct_node_at(&historical, id, self.valid_time, self.transaction_time)?;

            Ok(QueryRow::from_entity(EntityResult::Node(node)).at_time(self.valid_time))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.node_ids.size_hint()
    }
}

/// Batch temporal node iterator for bulk queries.
///
/// # Context
/// An optimized alternative to [`TemporalNodeIterator`] for reconstructing
/// many historical nodes simultaneously. It minimizes lock contention by acquiring
/// a single read lock on `HistoricalStorage`, processing all nodes, and releasing it immediately.
///
/// # Details
/// **Performance**: Use this for bulk queries (>100 nodes) where lock acquisition
/// overhead is significant.
///
/// **Trade-off**: Collects all results eagerly into memory during construction.
/// This requires more upfront memory allocation (O(n)) but avoids per-node lock
/// overhead and allows writer threads to proceed without waiting for the entire
/// iteration to complete.
///
/// # Panics
/// Does not panic. Returns an error during construction if the `HistoricalStorage` lock is poisoned.
///
/// # Examples
///
/// ```rust
/// # use std::sync::Arc;
/// # use parking_lot::RwLock;
/// # use aletheiadb::storage::historical::HistoricalStorage;
/// # use aletheiadb::core::id::NodeId;
/// # use aletheiadb::core::temporal::time;
/// # use aletheiadb::query::executor::BatchTemporalNodeIterator;
/// # use aletheiadb::query::executor::ResultIterator;
/// #
/// # let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
/// # let node_ids = vec![NodeId::new(1).unwrap(), NodeId::new(2).unwrap()];
/// let point_in_time = time::now();
///
/// // Lock is acquired, nodes are processed, and lock is released during `new()`
/// let mut batch_iter = BatchTemporalNodeIterator::new(
///     node_ids,
///     point_in_time,
///     point_in_time,
///     historical
/// ).expect("Lock should not be poisoned");
///
/// // Iteration is lock-free and pulls from memory
/// while let Some(Ok(row)) = batch_iter.next() {
///     println!("Bulk historical data: {:?}", row.entity);
/// }
/// ```
pub struct BatchTemporalNodeIterator {
    results: std::vec::IntoIter<Result<QueryRow>>,
}

impl BatchTemporalNodeIterator {
    /// Create a new batch temporal node iterator.
    ///
    /// Initialize an iterator that resolves multiple historical nodes simultaneously.
    ///
    /// # Why?
    /// Unlike the standard `TemporalNodeIterator`, this optimizes lock acquisition
    /// by grabbing the read lock once for the entire batch. This prevents lock contention
    /// on highly active graphs during deep historical traversals.
    ///
    /// Acquires the historical storage lock once, reconstructs all nodes,
    /// then releases the lock and returns the iterator over results.
    ///
    /// # Errors
    /// Returns an error if the historical storage lock is poisoned.
    pub fn new(
        node_ids: Vec<NodeId>,
        valid_time: Timestamp,
        transaction_time: Timestamp,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Result<Self> {
        // Acquire lock once for all nodes
        let guard = historical.read();

        // Reconstruct all nodes while holding the lock
        let results: Vec<Result<QueryRow>> = node_ids
            .into_iter()
            .map(|id| {
                // Reconstruct the node at the requested bi-temporal point (Issue #356)
                let node = reconstruct_node_at(&guard, id, valid_time, transaction_time)?;

                Ok(QueryRow::from_entity(EntityResult::Node(node)).at_time(valid_time))
            })
            .collect();

        // Lock is automatically released here when `guard` goes out of scope

        Ok(BatchTemporalNodeIterator {
            results: results.into_iter(),
        })
    }
}

impl ResultIterator for BatchTemporalNodeIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        self.results.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.results.size_hint()
    }
}

/// Iterator for temporal node lookups with optional label filtering.
///
/// This iterator addresses the deep nesting issue (#356) by extracting the
/// filtering logic into well-defined helper methods:
///
/// - `get_temporal_version()` - Retrieves a node at a specific point in bi-temporal time
/// - `apply_label_filter()` - Checks if a node matches the optional label filter
/// - `filter_node()` - Orchestrates the filtering logic with maximum 2-3 levels of nesting
///
/// ## Design Rationale
///
/// Instead of deeply nested conditionals (8+ levels), this design:
/// 1. Separates concerns into small, focused methods
/// 2. Keeps each method at 2-3 levels of nesting maximum
/// 3. Makes each component independently testable
/// 4. Improves readability and maintainability
///
/// ## Lock Duration Trade-off
///
/// The `next()` method holds the historical read lock for the entire iteration
/// loop until a matching node is found. This is intentional:
/// - **Advantage**: Avoids lock thrashing (acquiring/releasing on every node)
/// - **Trade-off**: For large result sets with many filtered-out nodes, the lock
///   may be held longer, potentially increasing writer latency
///
/// For bulk queries where this is a concern, consider using `BatchTemporalNodeIterator`
/// which processes all nodes upfront and releases the lock immediately.
///
/// ## Example
///
/// ```ignore
/// let iter = TemporalNodeScanIterator::new(
///     node_ids,
///     valid_time,
///     transaction_time,
///     historical,
///     Some("Person".to_string()), // Optional label filter
/// );
///
/// for result in iter {
///     // Only Person nodes at the specified time point
/// }
/// ```
pub struct TemporalNodeScanIterator {
    node_ids: std::vec::IntoIter<NodeId>,
    valid_time: Timestamp,
    transaction_time: Timestamp,
    historical: Arc<RwLock<HistoricalStorage>>,
    /// Optional label filter - if Some, only nodes with matching label are returned
    label_filter: Option<String>,
    /// Pre-computed interned ID of the label filter for efficient comparison.
    /// Avoids repeated hashmap lookups in apply_label_filter().
    interned_label_filter: Option<crate::core::interning::InternedString>,
    /// When `true`, a candidate that has no version at the requested
    /// bi-temporal point (or whose version metadata is missing) is silently
    /// skipped instead of surfacing an error. This is the correct semantics
    /// for a point-in-time label *scan* (Issues #550/#551): the candidate set
    /// is "every node ever versioned", most of which need not exist at the
    /// queried instant. The per-node lookup path (default `false`) keeps
    /// propagating the error, since there the caller explicitly named the id.
    skip_missing: bool,
}

impl TemporalNodeScanIterator {
    /// Initialize a scanning iterator that searches historical storage.
    ///
    /// # Why?
    /// This provides a fallback mechanism to find nodes in the past when temporal
    /// indexes are either unavailable or explicitly bypassed.
    ///
    /// # Arguments
    ///
    /// * `node_ids` - The node IDs to iterate over
    /// * `valid_time` - The valid time for temporal reconstruction
    /// * `transaction_time` - The transaction time for temporal reconstruction
    /// * `historical` - Reference to historical storage
    /// * `label_filter` - Optional label to filter nodes by
    pub fn new(
        node_ids: Vec<NodeId>,
        valid_time: Timestamp,
        transaction_time: Timestamp,
        historical: Arc<RwLock<HistoricalStorage>>,
        label_filter: Option<String>,
    ) -> Self {
        // Pre-compute the interned label ID once during construction
        // to avoid repeated hashmap lookups during iteration
        let interned_label_filter = label_filter
            .as_ref()
            .and_then(|label| GLOBAL_INTERNER.get_id(label));

        TemporalNodeScanIterator {
            node_ids: node_ids.into_iter(),
            valid_time,
            transaction_time,
            historical,
            label_filter,
            interned_label_filter,
            skip_missing: false,
        }
    }

    /// Enable point-in-time *scan* semantics: candidates absent at the queried
    /// bi-temporal point are skipped rather than raising
    /// [`TemporalError::NodeNotFoundAtTime`](crate::core::error::TemporalError::NodeNotFoundAtTime).
    ///
    /// Used by the Cypher/AQL label-scan `AS OF` path (Issues #550/#551), where
    /// the candidate set is every ever-versioned node and most of them simply
    /// did not exist at the instant being asked about.
    #[must_use]
    pub fn skipping_missing(mut self) -> Self {
        self.skip_missing = true;
        self
    }

    /// Retrieve the temporal version of a node at the configured time point.
    ///
    /// Thin wrapper over the shared [`reconstruct_node_at`] helper (Issue #356),
    /// binding this iterator's configured `(valid_time, transaction_time)`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No version exists at the specified time point
    /// - Version metadata is missing (data inconsistency)
    /// - Property reconstruction fails
    pub(crate) fn get_temporal_version(
        &self,
        node_id: NodeId,
        guard: &parking_lot::RwLockReadGuard<'_, HistoricalStorage>,
    ) -> Result<Node> {
        reconstruct_node_at(guard, node_id, self.valid_time, self.transaction_time)
    }

    /// Check if a node passes the label filter.
    ///
    /// Returns `true` if:
    /// - No label filter is configured (all nodes pass)
    /// - The node's label matches the filter
    ///
    /// Returns `false` if:
    /// - The node's label doesn't match the filter
    /// - The filter label doesn't exist in the interner (no nodes can match)
    ///
    /// Uses the pre-computed interned label ID for O(1) comparison.
    #[inline]
    pub(crate) fn apply_label_filter(&self, node: &Node) -> bool {
        match (&self.label_filter, self.interned_label_filter) {
            (None, _) => true,        // No filter, all nodes pass
            (Some(_), None) => false, // Filter label doesn't exist, no nodes match
            (Some(_), Some(filter_id)) => filter_id == node.label,
        }
    }

    /// Orchestrate the filtering logic for a single node.
    ///
    /// This method combines temporal reconstruction with label filtering
    /// while maintaining flat control flow (2-3 levels of nesting max).
    ///
    /// # Returns
    ///
    /// - `Some(Ok(QueryRow))` - Node exists at time point and passes label filter
    /// - `Some(Err(...))` - Node lookup failed (error should be propagated)
    /// - `None` - Node exists but doesn't pass label filter (skip to next)
    pub(crate) fn filter_node(
        &self,
        node_id: NodeId,
        guard: &parking_lot::RwLockReadGuard<'_, HistoricalStorage>,
    ) -> Option<Result<QueryRow>> {
        // Step 1: Get the temporal version
        let node = match self.get_temporal_version(node_id, guard) {
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        };

        // Step 2: Apply label filter
        if !self.apply_label_filter(&node) {
            return None; // Skip this node
        }

        // Step 3: Build and return the query row
        Some(Ok(
            QueryRow::from_entity(EntityResult::Node(node)).at_time(self.valid_time)
        ))
    }
}

impl ResultIterator for TemporalNodeScanIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        // Acquire read lock once for the duration of finding the next valid node
        let guard = self.historical.read();

        loop {
            let node_id = self.node_ids.next()?;

            match self.filter_node(node_id, &guard) {
                // In scan mode, a candidate absent at the queried instant is not
                // an error -- it simply isn't in the result set. Skip it and
                // keep scanning instead of aborting the whole query.
                Some(Err(e)) if self.skip_missing && is_missing_at_time(&e) => continue,
                Some(result) => return Some(result), // Found valid node or error
                None => continue,                    // Label filter didn't match, try next
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_lower, upper) = self.node_ids.size_hint();
        // When a label filter is active, this iterator may skip node IDs,
        // so we cannot safely use the underlying lower bound.
        // Upper bound remains valid as we can't return more than remaining IDs.
        if self.label_filter.is_some() {
            (0, upper)
        } else {
            // No label filtering: all remaining node_ids will be yielded
            // (assuming they exist in storage at the requested time point).
            self.node_ids.size_hint()
        }
    }
}

/// Returns `true` for the "this entity has no version at the queried
/// bi-temporal point" family of errors, which a point-in-time *scan*
/// (Issues #550/#551) treats as "not in the result set" rather than a hard
/// failure. A per-node lookup, where the caller explicitly named the id, keeps
/// propagating these as errors.
fn is_missing_at_time(err: &crate::core::error::Error) -> bool {
    use crate::core::error::{Error, TemporalError};
    matches!(
        err,
        Error::Temporal(TemporalError::NodeNotFoundAtTime { .. })
            | Error::Temporal(TemporalError::VersionNotFound(_))
    )
}

/// Whether an error from `get_node_at_time` means "this node is not visible at
/// the coordinate" (so a temporal traversal should skip it, Issue #3622 F3)
/// rather than a systemic failure. Covers the storage `NodeNotFound` /
/// `VersionNotFound` that `get_node_at_time` raises plus the temporal variants.
fn node_missing_at_time(err: &crate::core::error::Error) -> bool {
    use crate::core::error::{Error, StorageError};
    is_missing_at_time(err)
        || matches!(
            err,
            Error::Storage(StorageError::NodeNotFound(_) | StorageError::VersionNotFound(_))
        )
}

/// Iterator for valid-time range label scans (`BETWEEN ... AND ...`, Issue #552).
///
/// # Semantics
///
/// Given a candidate set of node ids and a valid-time range `[valid_from,
/// valid_to)`, this yields every version of each candidate that is **believed
/// at** the observed `transaction_time` (its transaction interval contains TT
/// -- the same predicate a point-in-time `AS OF` uses) **and** whose valid-time
/// interval *overlaps* the range, optionally filtered by `label`.
///
/// Semantically this is an **as-of-TT snapshot across a valid-time range**: it
/// equals the union, over every valid instant `v` in `[valid_from, valid_to)`,
/// of `AS OF (v, TT)`, deduplicated by version. Earlier valid segments that are
/// no longer believed at TT (superseded by a later transaction-time write, or
/// closed by a retraction) are **excluded**, consistent with `AS OF`, so no
/// stale beliefs and no duplicate rows appear.
///
/// In the current storage model each forward write closes the prior version's
/// transaction interval (transaction-time supersession), so at a fixed TT at
/// most one version per node is believed; a node therefore contributes at most
/// one row -- its believed-at-TT state -- and only when that state's valid
/// interval overlaps the range. (Were the store to retain multiple co-current
/// valid segments, this would naturally emit one row per in-range version.)
/// Because a single node can have several versions overlapping the range, this
/// iterator may emit **multiple rows per node** -- one per overlapping version
/// -- which is the openCypher-ish reading of a `BETWEEN` range query.
///
/// # Eager reconstruction
///
/// Like [`BatchTemporalNodeIterator`], all rows are reconstructed eagerly under
/// a single historical read lock during construction. A range scan is inherently
/// heavier than a point lookup, and eager collection keeps the lock held for a
/// bounded, predictable window instead of across the entire (lazy) drain.
///
/// # Ordering
///
/// At a fixed `transaction_time` at most one version per node is believed (see
/// the type-level note above), so a node contributes at most one row; multiple
/// rows arise only across DISTINCT nodes. Nodes are emitted in the order of the
/// supplied candidate list. The per-node selection is still sorted
/// oldest-`valid_from`-first (ties broken by version id) so that, were the store
/// ever to retain multiple co-current versions, the output would remain
/// deterministic.
pub struct TemporalNodeRangeScanIterator {
    results: std::vec::IntoIter<Result<QueryRow>>,
}

impl TemporalNodeRangeScanIterator {
    /// Build a range-scan iterator, reconstructing all overlapping versions up
    /// front under a single read lock.
    ///
    /// # Arguments
    ///
    /// * `node_ids` - Candidate node ids (typically every ever-versioned node)
    /// * `valid_from` - Inclusive start of the valid-time range
    /// * `valid_to` - Exclusive end of the valid-time range
    /// * `transaction_time` - Transaction time the range is observed at
    /// * `historical` - Historical storage
    /// * `label_filter` - Optional label the versions must match
    pub fn new(
        node_ids: Vec<NodeId>,
        valid_from: Timestamp,
        valid_to: Timestamp,
        transaction_time: Timestamp,
        historical: Arc<RwLock<HistoricalStorage>>,
        label_filter: Option<String>,
    ) -> Self {
        // An invalid range (end < start) yields nothing rather than erroring:
        // the converter already validated it, but defend anyway.
        let Ok(range) = crate::core::temporal::TimeRange::new(valid_from, valid_to) else {
            return TemporalNodeRangeScanIterator {
                results: Vec::new().into_iter(),
            };
        };

        // Resolve the label filter to its interned id once. If the label was
        // never interned, nothing can match -- yield an empty result.
        let interned_label = label_filter
            .as_ref()
            .and_then(|label| GLOBAL_INTERNER.get_id(label));
        if label_filter.is_some() && interned_label.is_none() {
            return TemporalNodeRangeScanIterator {
                results: Vec::new().into_iter(),
            };
        }

        let guard = historical.read();
        let mut results: Vec<Result<QueryRow>> = Vec::new();

        for node_id in node_ids {
            // Walk the version chain from the head backwards.
            let Some(head) = guard.get_current_node_version(node_id) else {
                continue;
            };
            let mut selected = Vec::new();
            let mut cursor = Some(head);
            while let Some(vid) = cursor {
                let Some(version) = guard.get_node_version(vid) else {
                    break;
                };
                cursor = version.prev_version;

                // Label filter (label can change across versions).
                if let Some(lbl) = interned_label
                    && version.label != lbl
                {
                    continue;
                }
                // Keep only versions BELIEVED at the observation transaction time
                // -- the exact same tx-visibility predicate the point-in-time
                // `AS OF` selector (`find_node_version_at_time`) uses. This makes
                // the range scan an as-of-TT snapshot ACROSS the valid range:
                // versions whose transaction interval was closed by a later
                // correction or retraction (beliefs no longer held at TT) are
                // excluded, exactly as a point `AS OF` would exclude them. Because
                // at a fixed TT there is at most one version per valid instant,
                // this also yields no duplicate rows.
                if !version
                    .temporal
                    .transaction_time()
                    .contains(transaction_time)
                {
                    continue;
                }
                // Keep versions whose valid interval overlaps the range.
                // (`overlaps` already excludes empty tombstone intervals.)
                if !version.temporal.valid_time().overlaps(&range) {
                    continue;
                }
                selected.push((
                    version.temporal.valid_time().start(),
                    version.node_id,
                    version.label,
                    vid,
                ));
            }

            // Deterministic per-node ordering: oldest valid_from first.
            selected.sort_by(|a, b| a.0.cmp(&b.0).then(a.3.cmp(&b.3)));

            for (valid_from_ts, matched_node_id, matched_label, vid) in selected {
                match guard.reconstruct_node_properties(vid) {
                    Ok(properties) => {
                        let node = Node::new(matched_node_id, matched_label, properties, vid);
                        results
                            .push(Ok(QueryRow::from_entity(EntityResult::Node(node))
                                .at_time(valid_from_ts)));
                    }
                    Err(e) => results.push(Err(e)),
                }
            }
        }

        drop(guard);

        TemporalNodeRangeScanIterator {
            results: results.into_iter(),
        }
    }
}

impl ResultIterator for TemporalNodeRangeScanIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        self.results.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.results.size_hint()
    }
}

/// Iterator for graph traversal using BFS.
///
/// # Variable-length depth-range semantics (`*min..max`)
///
/// For a variable-length pattern this iterator binds each **distinct** reachable
/// target node **once**, at its **shortest** hop-distance from the anchor, and
/// emits it iff `min <= shortestDepth <= max`. Because the per-input `visited`
/// set is populated when a node is first enqueued (its shortest BFS depth), a
/// node whose shortest path is shorter than `min` is marked visited early and is
/// **not** re-emitted at a longer, in-range depth; likewise an anchor reachable
/// again only via an in-range cycle is not re-bound.
///
/// This is **node-distinct / shortest-path reachability**, a deliberate v1
/// simplification of openCypher's trail (path-enumeration) semantics. Under full
/// trail semantics `MATCH (a)-[*2..2]->(b)` over `a->x, a->y, x->y` would also
/// bind `y` via `a->x->y`; here `y`'s shortest depth is 1, so it is excluded.
/// Full trail semantics is a tracked follow-up (it requires per-path state and
/// carries cross-lane perf/regression risk in this shared engine).
///
/// # Deduplication Semantics
///
/// The `visited` set is cleared for each new input node. This means:
/// - Each input node gets independent traversal results
/// - If multiple input nodes can reach the same target, it appears multiple times
/// - This is intentional for path-based semantics (e.g., "all friends of each person")
///
/// For global deduplication across all inputs, wrap the output in a `DistinctIterator`.
///
/// # Example
///
/// ```text
/// Input: [A, B]
/// Graph: A → C, B → C
///
/// Output: [C (from A), C (from B)]  // C appears twice
/// ```
pub struct TraversalIterator {
    input: Box<dyn ResultIterator>,
    direction: Direction,
    label: Option<String>,
    /// Minimum depth (inclusive) at which a reached node is emitted. A node is
    /// bound iff `min_depth <= shortestDepth <= depth` (node-distinct /
    /// shortest-path reachability; see the struct-level docs).
    min_depth: usize,
    /// Maximum depth (inclusive); BFS expansion stops beyond this depth.
    depth: usize,
    current: Arc<CurrentStorage>,
    historical: Arc<RwLock<HistoricalStorage>>,
    /// Optional temporal context (valid_time, transaction_time) for edge filtering.
    /// When present, only edges that existed at the specified point in time are traversed.
    temporal_context: Option<(Timestamp, Timestamp)>,
    /// When `true`, each emitted row carries the immediately-traversed edge
    /// (reconstructed at `temporal_context`, or current-state) on its
    /// [`QueryRow::edge`] side channel, for edge-property `WHERE` / `ORDER BY`
    /// (Issue #3622). Set by the executor only when the physical plan references
    /// an edge variable (a `Predicate::EdgeScoped` / `SortKey::EdgeProperty`),
    /// so the overwhelming common case pays zero extra edge fetches.
    bind_edge: bool,
    /// Optional namespace scope (Issue #3349, PR2). When set, the traversal never
    /// crosses an edge whose namespace ∉ scope, nor enqueues a target node whose
    /// namespace ∉ scope — so an out-of-scope node is never reached, emitted, or
    /// used as a bridge. `None` ⇒ prior namespace-agnostic behavior.
    scope: Option<Arc<NamespaceScope>>,
    /// The attached scope resolved to interned ids **once** at execute start and
    /// threaded in (Issue #3349, PR2), so the current-state boundary check is a
    /// per-edge integer-hash membership probe (a single integer compare for a
    /// single-namespace scope) rather than a per-edge string hash.
    /// [`ResolvedScope::All`] (or no attached scope) is a no-op boundary.
    resolved_scope: ResolvedScope,
    // BFS state - reset for each input node (see doc comment above)
    frontier: VecDeque<(NodeId, Vec<EntityId>, usize)>,
    visited: HashSet<NodeId>,
    input_exhausted: bool,
}

impl TraversalIterator {
    /// Initialize a Breadth-First Search (BFS) graph traversal iterator.
    ///
    /// # Why?
    /// This is the core engine for `MATCH (a)-[*]->(b)` operations. It manages
    /// a frontier of visited nodes to prevent infinite loops in cyclic graphs,
    /// and conditionally queries the historical storage if a temporal context is present.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Box<dyn ResultIterator>,
        direction: Direction,
        label: Option<String>,
        min_depth: usize,
        depth: usize,
        current: Arc<CurrentStorage>,
        historical: Arc<RwLock<HistoricalStorage>>,
        temporal_context: Option<(Timestamp, Timestamp)>,
    ) -> Self {
        TraversalIterator {
            input,
            direction,
            label,
            min_depth,
            depth,
            current,
            historical,
            temporal_context,
            bind_edge: false,
            scope: None,
            resolved_scope: ResolvedScope::All,
            frontier: VecDeque::new(),
            visited: HashSet::new(),
            input_exhausted: false,
        }
    }

    /// Attach a namespace scope (Issue #3349, PR2) so the traversal honors the
    /// scope boundary: an edge whose namespace ∉ scope is never crossed, and a
    /// target node whose namespace ∉ scope is never enqueued (so it is neither
    /// emitted nor used to bridge deeper). [`NamespaceScope::All`] is a no-op.
    #[must_use]
    pub fn with_namespace_scope(
        mut self,
        scope: Arc<NamespaceScope>,
        resolved: ResolvedScope,
    ) -> Self {
        // The scope was already resolved to interned ids once at execute start
        // (Issue #3349, PR2 hoist); we only store the pre-resolved handle here so
        // the current-state boundary check avoids per-edge string hashing (and,
        // for a single-namespace scope, is a single integer compare).
        self.resolved_scope = resolved;
        self.scope = Some(scope);
        self
    }

    /// Whether crossing to `target` over `edge_id` is permitted by the namespace
    /// scope boundary (Issue #3349, PR2). Returns `true` when no restricting
    /// scope is attached (fetching nothing). When a scope is attached, the edge
    /// and target are reconstructed at the query coordinate and both their
    /// (immutable) namespaces must be in scope.
    fn boundary_allows(&self, edge_id: EdgeId, target: NodeId) -> bool {
        let Some(scope) = &self.scope else {
            return true;
        };
        if matches!(scope.as_ref(), NamespaceScope::All) {
            return true;
        }
        // Fast path (current-state traversal): namespace is immutable, so the
        // O(1) membership index is authoritative — no full edge/node
        // reconstruction. Two interned-id hash probes per edge (the scope was
        // resolved to ids once in `with_namespace_scope`).
        if self.temporal_context.is_none() {
            return scope_contains_current_edge(&self.current, edge_id, &self.resolved_scope)
                && scope_contains_current_node(&self.current, target, &self.resolved_scope);
        }
        // Temporal traversal: reconstruct the edge and target AS-OF the
        // coordinate so a since-deleted-but-then-valid edge is judged by its
        // historical state (the membership index only reflects current state).
        match self.reconstruct_bound_edge(edge_id) {
            Ok(edge) if scope.contains(&edge.namespace()) => {}
            _ => return false,
        }
        matches!(self.fetch_target_node(target), Ok(Some(node)) if scope.contains(&node.namespace()))
    }

    /// Enable attaching the immediately-traversed edge (reconstructed at the
    /// query's bi-temporal coordinate) to each emitted row's
    /// [`QueryRow::edge`] channel, for edge-property `WHERE` / `ORDER BY`
    /// (Issue #3622). Defaults off, so ordinary traversals do no extra edge
    /// work. Set by the executor only when the plan references an edge variable.
    #[must_use]
    pub fn bind_edges(mut self, enabled: bool) -> Self {
        self.bind_edge = enabled;
        self
    }

    /// Reconstruct the full edge `edge_id` as it exists at the traversal's
    /// bi-temporal coordinate (or current state when none), for the
    /// [`QueryRow::edge`] side channel (Issue #3622).
    fn reconstruct_bound_edge(
        &self,
        edge_id: crate::core::EdgeId,
    ) -> Result<crate::core::graph::Edge> {
        match self.temporal_context {
            Some((valid_time, tx_time)) => self
                .historical
                .read()
                .get_edge_at_time(edge_id, valid_time, tx_time),
            None => self.current.get_edge(edge_id),
        }
    }

    /// Fetch the traversal target node, reconstructed at the query's bi-temporal
    /// coordinate under a temporal context (Issue #3622 F3), or current state
    /// otherwise. `Ok(None)` means the node is not valid/visible at the
    /// coordinate and the row must be skipped (openCypher #3225); other errors
    /// propagate.
    fn fetch_target_node(&self, node_id: NodeId) -> Result<Option<Node>> {
        match self.temporal_context {
            Some((valid_time, tx_time)) => {
                match self
                    .historical
                    .read()
                    .get_node_at_time(node_id, valid_time, tx_time)
                {
                    Ok(node) => Ok(Some(node)),
                    Err(e) if node_missing_at_time(&e) => Ok(None),
                    Err(e) => Err(e),
                }
            }
            None => self.current.get_node(node_id).map(Some),
        }
    }

    /// Check if an edge existed at the specified temporal context using a pre-acquired lock guard.
    /// Returns true if no temporal context is set (current state query).
    #[inline]
    fn edge_visible_at_time(
        &self,
        edge_id: crate::core::EdgeId,
        historical_guard: &Option<parking_lot::RwLockReadGuard<'_, HistoricalStorage>>,
    ) -> bool {
        match self.temporal_context {
            Some((valid_time, tx_time)) => {
                // Use the pre-acquired guard to avoid per-edge lock acquisition
                historical_guard
                    .as_ref()
                    .expect("historical_guard must be Some when temporal_context is Some")
                    .find_edge_version_at_time(edge_id, valid_time, tx_time)
                    .is_some()
            }
            None => true, // No temporal context, use current state
        }
    }

    fn get_neighbors(&self, node_id: NodeId) -> Vec<(NodeId, crate::core::EdgeId)> {
        // Acquire historical lock ONCE for all edge checks in this call.
        // This avoids the performance regression of acquiring per-edge locks.
        let historical_guard = self.temporal_context.map(|_| self.historical.read());

        match self.direction {
            Direction::Outgoing => {
                // Use iterator methods to avoid intermediate Vec allocation (Issue #187)
                if let Some(ref label) = self.label {
                    self.current
                        .get_outgoing_edges_with_label_iter(node_id, label)
                        .filter_map(|edge_id| {
                            if !self.edge_visible_at_time(edge_id, &historical_guard) {
                                return None;
                            }
                            // Zero-copy: only get target NodeId, not full Edge (Issue #190)
                            self.current
                                .get_edge_target(edge_id)
                                .ok()
                                .map(|target| (target, edge_id))
                        })
                        .collect()
                } else {
                    self.current
                        .get_outgoing_edges_iter(node_id)
                        .filter_map(|edge_id| {
                            if !self.edge_visible_at_time(edge_id, &historical_guard) {
                                return None;
                            }
                            // Zero-copy: only get target NodeId, not full Edge (Issue #190)
                            self.current
                                .get_edge_target(edge_id)
                                .ok()
                                .map(|target| (target, edge_id))
                        })
                        .collect()
                }
            }
            Direction::Incoming => {
                // Use iterator methods to avoid intermediate Vec allocation (Issue #187)
                if let Some(ref label) = self.label {
                    self.current
                        .get_incoming_edges_with_label_iter(node_id, label)
                        .filter_map(|edge_id| {
                            if !self.edge_visible_at_time(edge_id, &historical_guard) {
                                return None;
                            }
                            // Zero-copy: only get source NodeId, not full Edge (Issue #190)
                            self.current
                                .get_edge_source(edge_id)
                                .ok()
                                .map(|source| (source, edge_id))
                        })
                        .collect()
                } else {
                    self.current
                        .get_incoming_edges_iter(node_id)
                        .filter_map(|edge_id| {
                            if !self.edge_visible_at_time(edge_id, &historical_guard) {
                                return None;
                            }
                            // Zero-copy: only get source NodeId, not full Edge (Issue #190)
                            self.current
                                .get_edge_source(edge_id)
                                .ok()
                                .map(|source| (source, edge_id))
                        })
                        .collect()
                }
            }
            Direction::Both => {
                // Use iterator methods to avoid intermediate Vec allocation (Issue #187)
                // Helper closure to process edges and add to neighbors
                // Zero-copy: only get target NodeId, not full Edge (Issue #190)
                let process_outgoing =
                    |edge_id, neighbors: &mut Vec<(NodeId, crate::core::EdgeId)>| {
                        if !self.edge_visible_at_time(edge_id, &historical_guard) {
                            return;
                        }
                        if let Ok(target) = self.current.get_edge_target(edge_id) {
                            neighbors.push((target, edge_id));
                        }
                    };

                // Zero-copy: only get source NodeId, not full Edge (Issue #190)
                let process_incoming =
                    |edge_id, neighbors: &mut Vec<(NodeId, crate::core::EdgeId)>| {
                        if !self.edge_visible_at_time(edge_id, &historical_guard) {
                            return;
                        }
                        if let Ok(source) = self.current.get_edge_source(edge_id) {
                            neighbors.push((source, edge_id));
                        }
                    };

                if let Some(ref label) = self.label {
                    // ⚡ Bolt Optimization: Instantiate iterators once to avoid duplicate lookups,
                    // calculate required capacity, and pre-allocate to prevent heap reallocations.
                    let out_iter = self
                        .current
                        .get_outgoing_edges_with_label_iter(node_id, label);
                    let in_iter = self
                        .current
                        .get_incoming_edges_with_label_iter(node_id, label);
                    let capacity = out_iter.size_hint().0 + in_iter.size_hint().0;

                    let mut neighbors = Vec::with_capacity(capacity);
                    for edge_id in out_iter {
                        process_outgoing(edge_id, &mut neighbors);
                    }
                    for edge_id in in_iter {
                        process_incoming(edge_id, &mut neighbors);
                    }
                    neighbors
                } else {
                    // ⚡ Bolt Optimization: Instantiate iterators once to avoid duplicate lookups,
                    // calculate required capacity, and pre-allocate to prevent heap reallocations.
                    let out_iter = self.current.get_outgoing_edges_iter(node_id);
                    let in_iter = self.current.get_incoming_edges_iter(node_id);
                    let capacity = out_iter.size_hint().0 + in_iter.size_hint().0;

                    let mut neighbors = Vec::with_capacity(capacity);
                    for edge_id in out_iter {
                        process_outgoing(edge_id, &mut neighbors);
                    }
                    for edge_id in in_iter {
                        process_incoming(edge_id, &mut neighbors);
                    }
                    neighbors
                }
            }
        }
    }
}

impl ResultIterator for TraversalIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        loop {
            // Process current frontier
            if let Some((node_id, path, current_depth)) = self.frontier.pop_front() {
                // Expansion and yielding are INDEPENDENT so a range like `*1..3`
                // both emits intermediate-depth nodes AND keeps exploring to the
                // maximum depth. First expand (if we have not reached the max
                // depth), enqueuing unvisited neighbors one hop deeper.
                if current_depth < self.depth {
                    let neighbors = self.get_neighbors(node_id);
                    for (target, edge_id) in neighbors {
                        // Namespace scope boundary (Issue #3349, PR2): skip this
                        // edge entirely when the edge's or the target's namespace
                        // is out of scope, so an out-of-scope node is never
                        // enqueued (never emitted, never a bridge). Skipping
                        // *before* `visited.insert` means the same target can
                        // still be reached over a different, in-scope edge.
                        if !self.boundary_allows(edge_id, target) {
                            continue;
                        }
                        // Node-distinct / shortest-path reachability: a node is
                        // enqueued (and thus later emitted) once, at its shortest
                        // BFS depth. This also makes cyclic graphs terminate. It
                        // is a v1 simplification of openCypher trail semantics --
                        // a target whose shortest path is below `min_depth` is
                        // marked visited here and never re-emitted deeper (see the
                        // struct-level docs).
                        //
                        // Relationship-distinct exception (Issue #3622): when an
                        // edge variable is bound (`bind_edge`), emit ONE row per
                        // traversed edge -- parallel edges to the same target (and
                        // a self-loop) each surface with their own edge bound,
                        // matching openCypher/Cypher per-edge semantics. This is
                        // safe because `bind_edge` is only ever set on a single
                        // fixed-length hop (the AQL converter rejects edge-var
                        // refs on multi-hop / variable-length patterns), so the
                        // depth-1 frontier never re-expands and cannot loop.
                        let enqueue = self.visited.insert(target) || self.bind_edge;
                        if enqueue {
                            // ⚡ Bolt Optimization: Pre-allocate capacity for new path to avoid reallocations.
                            // We are adding exactly 2 elements (edge and node) to the current path length.
                            let mut new_path = Vec::with_capacity(path.len() + 2);
                            new_path.extend_from_slice(&path);
                            new_path.push(EntityId::Edge(edge_id));
                            new_path.push(EntityId::Node(target));
                            self.frontier
                                .push_back((target, new_path, current_depth + 1));
                        }
                    }
                }

                // Then, if this node's depth falls within [min_depth, depth],
                // yield it. The depth-0 start node is never emitted.
                if current_depth >= 1
                    && current_depth >= self.min_depth
                    && current_depth <= self.depth
                {
                    // Reconstruct the target node at the query's bi-temporal
                    // coordinate (Issue #3622 F3): under a temporal context the
                    // node is read AS-OF the coordinate, symmetric with the
                    // reconstructed edge -- so an edge predicate and a node
                    // predicate in one WHERE see mutually consistent state. A
                    // node not valid at the coordinate is excluded (openCypher
                    // #3225: a node no longer valid there does not continue the
                    // traversal). Current-state traversal is unchanged.
                    let node = match self.fetch_target_node(node_id) {
                        Ok(Some(node)) => node,
                        Ok(None) => continue,
                        Err(e) => return Some(Err(e)),
                    };
                    let mut row = QueryRow::with_path(EntityResult::Node(node), path);
                    // Edge-property WHERE / ORDER BY (Issue #3622): attach the
                    // immediately-traversed edge (the last edge in the path),
                    // reconstructed at the query coordinate. Only when the plan
                    // referenced an edge variable.
                    if self.bind_edge
                        && let Some(edge_id) = last_edge_in_path(&row.path)
                    {
                        match self.reconstruct_bound_edge(edge_id) {
                            Ok(edge) => row = row.with_edge_binding(edge),
                            Err(e) => return Some(Err(e)),
                        }
                    }
                    return Some(Ok(row));
                }
                continue;
            }

            // Frontier exhausted, get next from input
            if self.input_exhausted {
                return None;
            }

            match self.input.next() {
                Some(Ok(row)) => {
                    if let Some(node_id) = row.entity.node_id() {
                        self.visited.clear();
                        self.visited.insert(node_id);
                        self.frontier
                            .push_back((node_id, vec![EntityId::Node(node_id)], 0));
                    }
                }
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    self.input_exhausted = true;
                    // Process any remaining frontier
                    if self.frontier.is_empty() {
                        return None;
                    }
                }
            }
        }
    }
}

/// Whether the current edge `edge_id` is inside the **pre-resolved** scope
/// `resolved` (Issue #3349, PR2), via O(1) interned-id membership probes.
/// [`ResolvedScope::All`] matches everything. The scope is resolved to ids once
/// at execute start (per query) and threaded down, so this per-edge check does
/// no string hashing — and, for a single-namespace scope, is a single integer
/// compare with no set scan.
fn scope_contains_current_edge(
    current: &CurrentStorage,
    edge_id: EdgeId,
    resolved: &ResolvedScope,
) -> bool {
    resolved.contains_by(|id| current.edge_in_namespace_id(edge_id, id))
}

/// Whether the current node `node_id` is inside the pre-resolved scope
/// `resolved` (Issue #3349, PR2). See [`scope_contains_current_edge`].
fn scope_contains_current_node(
    current: &CurrentStorage,
    node_id: NodeId,
    resolved: &ResolvedScope,
) -> bool {
    resolved.contains_by(|id| current.node_in_namespace_id(node_id, id))
}

/// Namespace-scope entity filter (Issue #3349, PR2).
///
/// Drops rows whose primary entity's (immutable) namespace ∉ scope, so a scoped
/// [`QueryBuilder`](crate::query::QueryBuilder) query returns only in-scope
/// entities. It wraps the **source leaves** that produce entities (scans, id
/// lookups, property/vector sources), so every downstream operator — LIMIT/SKIP,
/// ORDER BY, DISTINCT, aggregation, COUNT — already sees only in-scope rows. The
/// traversal *boundary* (never bridging through an out-of-scope node) is enforced
/// separately in [`TraversalIterator`], so this filter only has to reject
/// out-of-scope terminal entities.
///
/// Rows that carry no single scoped entity — a null binding, or an aggregation /
/// multi-variable computed row — pass through unchanged (namespace scoping of
/// aggregates is out of PR2 scope). An id-only entity is resolved against
/// current storage to read its namespace; if it can no longer be loaded it is
/// dropped (indistinguishable, to a scoped caller, from out-of-scope).
pub struct ScopeFilterIterator {
    input: Box<dyn ResultIterator>,
    scope: Arc<NamespaceScope>,
    current: Arc<CurrentStorage>,
    /// The scope resolved to interned ids **once** at execute start and threaded
    /// in (Issue #3349, PR2 hoist), so the id-only branches probe by integer id
    /// without per-row string hashing — a single integer compare for a
    /// single-namespace scope. [`ResolvedScope::All`] is a no-op filter.
    resolved: ResolvedScope,
}

impl ScopeFilterIterator {
    /// Wrap `input`, keeping only rows whose entity is in `scope`. `resolved` is
    /// the scope pre-resolved to interned ids once at execute start (Issue #3349,
    /// PR2 hoist) and shared into every source-leaf filter.
    #[must_use]
    pub fn new(
        input: Box<dyn ResultIterator>,
        scope: Arc<NamespaceScope>,
        resolved: ResolvedScope,
        current: Arc<CurrentStorage>,
    ) -> Self {
        ScopeFilterIterator {
            input,
            scope,
            current,
            resolved,
        }
    }

    /// Whether `row`'s entity is visible under the scope.
    ///
    /// Full `Node`/`Edge` rows carry their own (immutable) namespace, read
    /// directly — this is temporally correct because every mainline source
    /// (current *and* point-in-time: `TemporalNode*`/`VectorResult` iterators)
    /// emits a fully reconstructed entity, and a reconstructed entity carries
    /// exactly the namespace it held at that coordinate.
    ///
    /// The id-only (`NodeId`/`EdgeId`) branches resolve membership via the O(1)
    /// current-state membership index (no whole-entity clone; Issue #3349 B2).
    /// This is a **current-state** probe: it would wrongly drop a since-deleted
    /// id-only row under an `AS OF` read. That is safe because no mainline source
    /// emits an id-only row in a temporal context — every temporal source emits a
    /// full reconstructed `Node`/`Edge`, which is handled by the branches above
    /// (Issue #3349 A5). If a future id-only temporal source is added, resolve it
    /// via the historical path here instead.
    fn row_in_scope(&self, row: &QueryRow) -> bool {
        match &row.entity {
            EntityResult::Node(node) => self.scope.contains(&node.namespace()),
            EntityResult::Edge(edge) => self.scope.contains(&edge.namespace()),
            EntityResult::NodeId(id) => {
                scope_contains_current_node(&self.current, *id, &self.resolved)
            }
            EntityResult::EdgeId(id) => {
                scope_contains_current_edge(&self.current, *id, &self.resolved)
            }
            // A null binding or a computed (aggregate / multi-variable) row has
            // no single scoped entity — pass it through unchanged.
            EntityResult::Null => true,
        }
    }
}

impl ResultIterator for ScopeFilterIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        loop {
            match self.input.next()? {
                Ok(row) => {
                    if self.row_in_scope(&row) {
                        return Some(Ok(row));
                    }
                    // otherwise skip and pull the next row
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Iterator for filtering results.
///
/// # Example
///
/// ```rust
/// use aletheiadb::query::executor::{FilterIterator, NodeScanIterator};
/// use aletheiadb::query::ir::Predicate;
/// use std::sync::Arc;
///
/// let current = Arc::new(aletheiadb::storage::CurrentStorage::new());
/// let input = Box::new(NodeScanIterator::new(Some("Person".to_string()), current));
/// let predicate = Predicate::eq("name", "Alice");
/// let filter_iter = FilterIterator::new(input, predicate);
///
/// // Iterate results
/// // for row in filter_iter { ... }
/// ```
///
/// Iterator that applies a predicate filter.
///
/// Pulls rows from the input and yields only those matching the predicate.
pub struct FilterIterator {
    input: Box<dyn ResultIterator>,
    predicate: Predicate,
    /// Historical storage, needed only to resolve a row entity's write-time
    /// provenance for [`Predicate::Provenance`] leaves (Issue #3354a). `None`
    /// when the predicate carries no provenance leaf (the overwhelming common
    /// case), so a plain property filter never touches historical storage.
    historical: Option<Arc<RwLock<HistoricalStorage>>>,
    /// Cached: does the predicate tree contain any provenance leaf? Computed
    /// once so `next()` avoids resolving provenance for property-only filters.
    needs_provenance: bool,
    /// When `true`, property leaves are evaluated against an EDGE row's own
    /// properties (real edge-property `WHERE`, Issue #3622) instead of the
    /// single-entity pass-through. Set only by the executor when the filter's
    /// input subtree is rooted at an `EdgeScan` (the SQL `FROM edges` lane),
    /// where every bare property leaf unambiguously refers to the edge. Defaults
    /// to `false`, preserving the AQL/Cypher pass-through semantics pinned by
    /// `edge_non_provenance_leaf_is_pass_through`.
    edge_mode: bool,
}

impl FilterIterator {
    /// Create a new FilterIterator that filters results based on the predicate.
    ///
    /// This constructor carries no historical-storage handle, so a
    /// [`Predicate::Provenance`] leaf would have no bundle to resolve and be
    /// evaluated as absent (excluded). Use
    /// [`with_historical`](Self::with_historical) when the predicate may
    /// reference provenance.
    pub fn new(input: Box<dyn ResultIterator>, predicate: Predicate) -> Self {
        let needs_provenance = Self::predicate_needs_provenance(&predicate);
        FilterIterator {
            input,
            predicate,
            historical: None,
            needs_provenance,
            edge_mode: false,
        }
    }

    /// Create a FilterIterator with access to historical storage so
    /// [`Predicate::Provenance`] leaves (Issue #3354a) can resolve each row
    /// entity's write-time provenance bundle at the version the row represents.
    pub fn with_historical(
        input: Box<dyn ResultIterator>,
        predicate: Predicate,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Self {
        let needs_provenance = Self::predicate_needs_provenance(&predicate);
        FilterIterator {
            input,
            predicate,
            historical: Some(historical),
            needs_provenance,
            edge_mode: false,
        }
    }

    /// Enable real edge-property evaluation (Issue #3622): when the row is an
    /// edge, property leaves are matched against the edge's own properties
    /// instead of the AQL/Cypher single-entity pass-through.
    ///
    /// The executor sets this only for a filter whose input stream is rooted at
    /// an `EdgeScan` (the SQL `FROM edges` lane), where the stream is pure edges
    /// and every bare property leaf unambiguously refers to the edge. Node and
    /// null row handling are unchanged.
    #[must_use]
    pub fn evaluate_edge_properties(mut self, enabled: bool) -> Self {
        self.edge_mode = enabled;
        self
    }

    /// Returns `true` if the predicate tree contains any provenance leaf.
    fn predicate_needs_provenance(predicate: &Predicate) -> bool {
        match predicate {
            Predicate::Provenance(_) => true,
            Predicate::And(preds) | Predicate::Or(preds) => {
                preds.iter().any(Self::predicate_needs_provenance)
            }
            Predicate::Not(pred) => Self::predicate_needs_provenance(pred),
            // An edge-scoped wrapper (Issue #3622) carries only edge
            // property/structural leaves (never provenance) in this lane, so it
            // contributes no node-provenance requirement. Explicit rather than
            // folded into the wildcard so a future edge-provenance leaf is a
            // deliberate decision here.
            Predicate::EdgeScoped(_) => false,
            _ => false,
        }
    }

    /// Resolve the write-time provenance bundle recorded on the version a node
    /// row represents (Issue #3354a).
    ///
    /// Uses the node's `current_version` — which, for a point-in-time (`AS OF`)
    /// reconstructed node, is the historical version at that coordinate — so
    /// provenance is evaluated as recorded at the query's bi-temporal
    /// coordinate (AC3). A missing handle or a lookup error resolves to `None`
    /// (treated as unattributed, i.e. excluded by any provenance comparison).
    fn resolve_node_provenance(&self, node: &Node) -> Option<Provenance> {
        let historical = self.historical.as_ref()?;
        historical
            .read()
            .get_node_version_provenance(node.current_version)
            .ok()
            .flatten()
    }

    /// Resolve the write-time provenance bundle recorded on the version an edge
    /// row represents (Issue #3354a). See [`resolve_node_provenance`](Self::resolve_node_provenance).
    fn resolve_edge_provenance(&self, edge: &crate::core::graph::Edge) -> Option<Provenance> {
        let historical = self.historical.as_ref()?;
        historical
            .read()
            .get_edge_version_provenance(edge.current_version)
            .ok()
            .flatten()
    }

    /// Evaluate a provenance leaf lazily: resolve the row entity's provenance
    /// bundle **on demand** the first time a [`Predicate::Provenance`] leaf is
    /// reached, memoizing it in `cache` for the rest of this row's evaluation
    /// (Issue #3354a, DoS perf follow-up). Because resolution happens at the
    /// leaf rather than eagerly per scanned row, an AND whose cheaper conjunct
    /// already rejected the row never pays the `historical.read()` lock +
    /// version lookup (short-circuit skips this call entirely). At most one
    /// resolution per row is performed; the result is identical to the prior
    /// eager scheme.
    fn eval_provenance_leaf(
        p: &ProvenancePredicate,
        cache: &RefCell<Option<Option<Provenance>>>,
        resolve: impl FnOnce() -> Option<Provenance>,
    ) -> bool {
        if cache.borrow().is_none() {
            let resolved = resolve();
            *cache.borrow_mut() = Some(resolved);
        }
        let guard = cache.borrow();
        // Outer `Option` = "resolved yet?"; inner `Option<Provenance>` = the
        // bundle (or `None` for an unattributed version).
        p.evaluate(guard.as_ref().and_then(|inner| inner.as_ref()))
    }

    /// Evaluate the full predicate tree against a node with a caller-supplied
    /// provenance bundle. The bundle is seeded into the memo cache so
    /// provenance leaves evaluate against it directly without a historical
    /// lookup; the streaming [`next`](ResultIterator::next) path resolves
    /// provenance lazily instead. Test-facing convenience: the production path
    /// resolves lazily in [`evaluate_predicate`](Self::evaluate_predicate).
    #[cfg(test)]
    fn evaluate(&self, node: &Node, prov: Option<&Provenance>) -> bool {
        let prov_cache: RefCell<Option<Option<Provenance>>> = RefCell::new(Some(prov.cloned()));
        self.evaluate_predicate(&self.predicate, node, None, &prov_cache)
    }

    fn evaluate_predicate(
        &self,
        predicate: &Predicate,
        node: &Node,
        edge: Option<&crate::core::graph::Edge>,
        prov_cache: &RefCell<Option<Option<Provenance>>>,
    ) -> bool {
        match predicate {
            Predicate::True => true,
            Predicate::False => false,
            Predicate::Eq { key, value } => self.evaluate_eq(&node.properties, key, value),
            Predicate::Ne { key, value } => self.evaluate_ne(&node.properties, key, value),
            Predicate::Gt { key, value } => self.evaluate_gt(&node.properties, key, value),
            Predicate::Lt { key, value } => self.evaluate_lt(&node.properties, key, value),
            Predicate::Gte { key, value } => self.evaluate_gte(&node.properties, key, value),
            Predicate::Lte { key, value } => self.evaluate_lte(&node.properties, key, value),
            Predicate::Exists(key) => node.properties.get(key).is_some(),
            Predicate::NotExists(key) => node.properties.get(key).is_none(),
            Predicate::Contains { key, substring } => {
                self.evaluate_contains(&node.properties, key, substring)
            }
            Predicate::StartsWith { key, prefix } => {
                self.evaluate_starts_with(&node.properties, key, prefix)
            }
            Predicate::EndsWith { key, suffix } => {
                self.evaluate_ends_with(&node.properties, key, suffix)
            }
            Predicate::In { key, values } => self.evaluate_in(&node.properties, key, values),
            // Edge-scoped leaf (Issue #3622): evaluate the wrapped sub-tree
            // against the row's traversed edge with the shared openCypher edge
            // semantics. A row whose edge is not valid at the query coordinate
            // (no edge channel) fails the leaf.
            Predicate::EdgeScoped(inner) => match edge {
                Some(e) => self.evaluate_edge_full(inner, e, &RefCell::new(None)),
                None => false,
            },
            // Resolve provenance lazily so a cheaper conjunct that already
            // rejected the row short-circuits before any historical read.
            Predicate::Provenance(p) => {
                Self::eval_provenance_leaf(p, prov_cache, || self.resolve_node_provenance(node))
            }
            Predicate::And(preds) => preds
                .iter()
                .all(|p| self.evaluate_predicate(p, node, edge, prov_cache)),
            Predicate::Or(preds) => preds
                .iter()
                .any(|p| self.evaluate_predicate(p, node, edge, prov_cache)),
            Predicate::Not(pred) => !self.evaluate_predicate(pred, node, edge, prov_cache),
        }
    }

    /// Evaluate a predicate against an edge row for the provenance leaves only.
    ///
    /// The AQL single-entity pipeline historically passes edge rows through the
    /// filter untouched; this preserves that for non-provenance leaves (which
    /// evaluate to `true`, i.e. pass-through) while honoring
    /// [`Predicate::Provenance`] leaves against the edge's own provenance
    /// bundle (Issue #3354a). Only invoked when the predicate contains a
    /// provenance leaf. Provenance is resolved lazily (memoized in `prov_cache`)
    /// so an AND short-circuit skips the historical read for a row already
    /// rejected by a cheaper leaf.
    fn evaluate_edge_predicate(
        &self,
        predicate: &Predicate,
        edge: Option<&crate::core::graph::Edge>,
        prov_cache: &RefCell<Option<Option<Provenance>>>,
    ) -> bool {
        match predicate {
            Predicate::Provenance(p) => Self::eval_provenance_leaf(p, prov_cache, || {
                edge.and_then(|e| self.resolve_edge_provenance(e))
            }),
            Predicate::And(preds) => preds
                .iter()
                .all(|p| self.evaluate_edge_predicate(p, edge, prov_cache)),
            Predicate::Or(preds) => preds
                .iter()
                .any(|p| self.evaluate_edge_predicate(p, edge, prov_cache)),
            Predicate::Not(pred) => !self.evaluate_edge_predicate(pred, edge, prov_cache),
            Predicate::False => false,
            // An edge-scoped wrapper (Issue #3622) does not occur on this
            // edge-ROW pass-through path (the AQL converter only emits it over
            // node rows carrying an edge side channel). Handle it correctly for
            // robustness: evaluate the inner tree against the edge row itself,
            // matching the deliberately-exhaustive style of `evaluate_edge_full`.
            Predicate::EdgeScoped(inner) => match edge {
                Some(e) => self.evaluate_edge_full(inner, e, prov_cache),
                None => true,
            },
            // Non-provenance leaves retain the pre-existing edge pass-through.
            _ => true,
        }
    }

    fn evaluate_eq(&self, props: &PropertyMap, key: &str, value: &PredicateValue) -> bool {
        let Some(prop) = props.get(key) else {
            return false;
        };
        self.compare_eq(prop, value)
    }

    fn evaluate_ne(&self, props: &PropertyMap, key: &str, value: &PredicateValue) -> bool {
        let Some(prop) = props.get(key) else {
            return true; // Non-existent != anything
        };
        !self.compare_eq(prop, value)
    }

    fn evaluate_gt(&self, props: &PropertyMap, key: &str, value: &PredicateValue) -> bool {
        let Some(prop) = props.get(key) else {
            return false;
        };
        self.compare_gt(prop, value)
    }

    fn evaluate_lt(&self, props: &PropertyMap, key: &str, value: &PredicateValue) -> bool {
        let Some(prop) = props.get(key) else {
            return false;
        };
        self.compare_lt(prop, value)
    }

    fn evaluate_gte(&self, props: &PropertyMap, key: &str, value: &PredicateValue) -> bool {
        let Some(prop) = props.get(key) else {
            return false;
        };
        self.compare_gte(prop, value)
    }

    fn evaluate_lte(&self, props: &PropertyMap, key: &str, value: &PredicateValue) -> bool {
        let Some(prop) = props.get(key) else {
            return false;
        };
        self.compare_lte(prop, value)
    }

    fn evaluate_contains(&self, props: &PropertyMap, key: &str, substring: &str) -> bool {
        let Some(PropertyValue::String(s)) = props.get(key) else {
            return false;
        };
        s.contains(substring)
    }

    fn evaluate_starts_with(&self, props: &PropertyMap, key: &str, prefix: &str) -> bool {
        let Some(PropertyValue::String(s)) = props.get(key) else {
            return false;
        };
        s.starts_with(prefix)
    }

    fn evaluate_ends_with(&self, props: &PropertyMap, key: &str, suffix: &str) -> bool {
        let Some(PropertyValue::String(s)) = props.get(key) else {
            return false;
        };
        s.ends_with(suffix)
    }

    fn evaluate_in(&self, props: &PropertyMap, key: &str, values: &[PredicateValue]) -> bool {
        let Some(prop) = props.get(key) else {
            return false;
        };
        values.iter().any(|v| self.compare_eq(prop, v))
    }

    /// Evaluate the property/structural leaves of a predicate against a bare
    /// [`PropertyMap`], shared by the node and edge (Issue #3622) paths. Returns
    /// `Some(bool)` for a property/structural leaf, `None` for a leaf that needs
    /// entity context (provenance) or a logical combinator, which the caller
    /// handles with the appropriate entity-aware resolver.
    fn evaluate_property_leaf(&self, predicate: &Predicate, props: &PropertyMap) -> Option<bool> {
        match predicate {
            Predicate::True => Some(true),
            Predicate::False => Some(false),
            Predicate::Eq { key, value } => Some(self.evaluate_eq(props, key, value)),
            Predicate::Ne { key, value } => Some(self.evaluate_ne(props, key, value)),
            Predicate::Gt { key, value } => Some(self.evaluate_gt(props, key, value)),
            Predicate::Lt { key, value } => Some(self.evaluate_lt(props, key, value)),
            Predicate::Gte { key, value } => Some(self.evaluate_gte(props, key, value)),
            Predicate::Lte { key, value } => Some(self.evaluate_lte(props, key, value)),
            Predicate::Exists(key) => Some(props.get(key).is_some()),
            Predicate::NotExists(key) => Some(props.get(key).is_none()),
            Predicate::Contains { key, substring } => {
                Some(self.evaluate_contains(props, key, substring))
            }
            Predicate::StartsWith { key, prefix } => {
                Some(self.evaluate_starts_with(props, key, prefix))
            }
            Predicate::EndsWith { key, suffix } => {
                Some(self.evaluate_ends_with(props, key, suffix))
            }
            Predicate::In { key, values } => Some(self.evaluate_in(props, key, values)),
            // Provenance / logical combinators / edge-scoped wrappers need
            // entity context (resolved by the caller).
            Predicate::Provenance(_)
            | Predicate::And(_)
            | Predicate::Or(_)
            | Predicate::Not(_)
            | Predicate::EdgeScoped(_) => None,
        }
    }

    /// Evaluate the full predicate tree against an EDGE row, matching reserved
    /// structural fields against the edge struct (Issue #3622 review fix),
    /// property leaves against the edge's own properties (Issue #3622), and
    /// provenance leaves against the edge's provenance bundle. Used only in
    /// `edge_mode` (the SQL `FROM edges` lane); the AQL/Cypher pass-through path
    /// stays in [`evaluate_edge_predicate`](Self::evaluate_edge_predicate).
    ///
    /// Edge WHERE follows **openCypher / node semantics**, NOT SQL three-valued
    /// (UNKNOWN) logic: an `Eq` on an absent property is `false` (excludes the
    /// row) while a `Ne` on an absent property is `true` (keeps the row), exactly
    /// as [`evaluate_predicate`](Self::evaluate_predicate) treats node rows.
    fn evaluate_edge_full(
        &self,
        predicate: &Predicate,
        edge: &crate::core::graph::Edge,
        prov_cache: &RefCell<Option<Option<Provenance>>>,
    ) -> bool {
        // Reserved structural fields (`type`/`label`/`source`/`target`/`id`) live
        // on the edge STRUCT, not in `properties`, so resolve them there first
        // (Issue #3622 review fix); otherwise `WHERE type = 'KNOWS'` etc. would
        // silently match nothing.
        if let Some(result) = self.evaluate_edge_structural_leaf(predicate, edge) {
            return result;
        }
        if let Some(result) = self.evaluate_property_leaf(predicate, &edge.properties) {
            return result;
        }
        match predicate {
            Predicate::Provenance(p) => {
                Self::eval_provenance_leaf(p, prov_cache, || self.resolve_edge_provenance(edge))
            }
            Predicate::And(preds) => preds
                .iter()
                .all(|p| self.evaluate_edge_full(p, edge, prov_cache)),
            Predicate::Or(preds) => preds
                .iter()
                .any(|p| self.evaluate_edge_full(p, edge, prov_cache)),
            Predicate::Not(pred) => !self.evaluate_edge_full(pred, edge, prov_cache),
            // A nested edge-scoped wrapper (should not normally occur -- the AQL
            // converter wraps leaves, not sub-trees) simply re-scopes to the
            // same edge; unwrap and evaluate the inner tree.
            Predicate::EdgeScoped(inner) => self.evaluate_edge_full(inner, edge, prov_cache),
            // Every property/structural leaf is resolved above; only provenance
            // leaves and logical combinators reach this match, and all four are
            // handled explicitly. A future `Predicate` variant that is neither
            // must be routed deliberately rather than silently passing through.
            Predicate::True
            | Predicate::False
            | Predicate::Eq { .. }
            | Predicate::Ne { .. }
            | Predicate::Gt { .. }
            | Predicate::Lt { .. }
            | Predicate::Gte { .. }
            | Predicate::Lte { .. }
            | Predicate::Exists(_)
            | Predicate::NotExists(_)
            | Predicate::Contains { .. }
            | Predicate::StartsWith { .. }
            | Predicate::EndsWith { .. }
            | Predicate::In { .. } => {
                unreachable!("property/structural leaves are resolved before this match")
            }
        }
    }

    /// Evaluate a single predicate leaf that references a reserved *structural*
    /// edge field (Issue #3622 review fix). Returns `Some(bool)` when `predicate`
    /// is a keyed leaf whose key is a reserved structural field (resolving it
    /// against the edge's struct via [`edge_structural_value`]), and `None`
    /// otherwise (not a keyed leaf, or a genuine user-property key -- both of
    /// which the caller resolves against `edge.properties`).
    ///
    /// Structural fields always exist, so the openCypher three-valued
    /// asymmetries around missing keys never fire here; the operator semantics
    /// (`Eq`/`Ne`/comparisons/`In`/string ops/`Exists`) mirror the property path.
    fn evaluate_edge_structural_leaf(
        &self,
        predicate: &Predicate,
        edge: &crate::core::graph::Edge,
    ) -> Option<bool> {
        let key = match predicate {
            Predicate::Eq { key, .. }
            | Predicate::Ne { key, .. }
            | Predicate::Gt { key, .. }
            | Predicate::Lt { key, .. }
            | Predicate::Gte { key, .. }
            | Predicate::Lte { key, .. }
            | Predicate::Exists(key)
            | Predicate::NotExists(key)
            | Predicate::Contains { key, .. }
            | Predicate::StartsWith { key, .. }
            | Predicate::EndsWith { key, .. }
            | Predicate::In { key, .. } => key.as_str(),
            _ => return None,
        };
        let value = edge_structural_value(edge, key)?;
        Some(match predicate {
            Predicate::Eq { value: v, .. } => self.compare_eq(&value, v),
            Predicate::Ne { value: v, .. } => !self.compare_eq(&value, v),
            Predicate::Gt { value: v, .. } => self.compare_gt(&value, v),
            Predicate::Lt { value: v, .. } => self.compare_lt(&value, v),
            Predicate::Gte { value: v, .. } => self.compare_gte(&value, v),
            Predicate::Lte { value: v, .. } => self.compare_lte(&value, v),
            // A structural field is always present.
            Predicate::Exists(_) => true,
            Predicate::NotExists(_) => false,
            Predicate::Contains { substring, .. } => match &value {
                PropertyValue::String(s) => s.contains(substring.as_str()),
                _ => false,
            },
            Predicate::StartsWith { prefix, .. } => match &value {
                PropertyValue::String(s) => s.starts_with(prefix.as_str()),
                _ => false,
            },
            Predicate::EndsWith { suffix, .. } => match &value {
                PropertyValue::String(s) => s.ends_with(suffix.as_str()),
                _ => false,
            },
            Predicate::In { values, .. } => values.iter().any(|v| self.compare_eq(&value, v)),
            // `key` was bound above only for the keyed-leaf arms.
            _ => unreachable!("key resolved above implies a keyed leaf predicate"),
        })
    }

    fn compare_eq(&self, prop: &PropertyValue, value: &PredicateValue) -> bool {
        match (prop, value) {
            (PropertyValue::Bool(a), PredicateValue::Bool(b)) => a == b,
            (PropertyValue::Int(a), PredicateValue::Int(b)) => a == b,
            (PropertyValue::Float(a), PredicateValue::Float(b)) => (a - b).abs() < f64::EPSILON,
            (PropertyValue::String(a), PredicateValue::String(b)) => a.as_ref() == b.as_str(),
            (PropertyValue::Null, PredicateValue::Null) => true,
            _ => false,
        }
    }

    fn compare_gt(&self, prop: &PropertyValue, value: &PredicateValue) -> bool {
        match (prop, value) {
            (PropertyValue::Int(a), PredicateValue::Int(b)) => a > b,
            (PropertyValue::Float(a), PredicateValue::Float(b)) => a > b,
            _ => false,
        }
    }

    fn compare_lt(&self, prop: &PropertyValue, value: &PredicateValue) -> bool {
        match (prop, value) {
            (PropertyValue::Int(a), PredicateValue::Int(b)) => a < b,
            (PropertyValue::Float(a), PredicateValue::Float(b)) => a < b,
            _ => false,
        }
    }

    fn compare_gte(&self, prop: &PropertyValue, value: &PredicateValue) -> bool {
        match (prop, value) {
            (PropertyValue::Int(a), PredicateValue::Int(b)) => a >= b,
            (PropertyValue::Float(a), PredicateValue::Float(b)) => a >= b,
            _ => false,
        }
    }

    fn compare_lte(&self, prop: &PropertyValue, value: &PredicateValue) -> bool {
        match (prop, value) {
            (PropertyValue::Int(a), PredicateValue::Int(b)) => a <= b,
            (PropertyValue::Float(a), PredicateValue::Float(b)) => a <= b,
            _ => false,
        }
    }
}

impl FilterIterator {
    /// Evaluate a predicate against a null binding (an unmatched
    /// `OPTIONAL MATCH` row).
    ///
    /// Approximates openCypher three-valued logic at the "not true" level:
    /// any comparison, string predicate, or membership test involving the
    /// null binding is not-true (row filtered). A node/edge-level `x IS NULL`
    /// (bare variable) is lowered by the Cypher converter to
    /// `Eq { value: Null }` (true here) and `x IS NOT NULL` to
    /// `Ne { value: Null }` (false here). A property-level `x.p IS NULL` is
    /// instead lowered to `Or(NotExists, Eq { value: Null })` (true for a null
    /// binding via its `NotExists` arm) and `x.p IS NOT NULL` to
    /// `And(Exists, Ne { value: Null })` (false via its `Exists` arm); the bare
    /// `Eq`/`Ne { value: Null }` arms below also serve that composed form.
    ///
    /// Known deviation: `NOT` uses two-valued negation (`NOT (null.p = 1)`
    /// keeps the row where openCypher's `NOT null` would drop it). This
    /// mirrors the engine's existing missing-property semantics.
    fn evaluate_null(&self, predicate: &Predicate) -> bool {
        match predicate {
            Predicate::True => true,
            Predicate::False => false,
            // `x IS NULL` is converted to Eq { value: Null }: true for a null row.
            Predicate::Eq {
                value: PredicateValue::Null,
                ..
            } => true,
            // `x IS NOT NULL` is converted to Ne { value: Null }: false for a null row.
            Predicate::Ne {
                value: PredicateValue::Null,
                ..
            } => false,
            // A null binding has no properties.
            Predicate::NotExists(_) => true,
            Predicate::Exists(_) => false,
            // Any comparison/membership/string predicate against null is not-true.
            Predicate::Eq { .. }
            | Predicate::Ne { .. }
            | Predicate::Gt { .. }
            | Predicate::Gte { .. }
            | Predicate::Lt { .. }
            | Predicate::Lte { .. }
            | Predicate::In { .. }
            | Predicate::Contains { .. }
            | Predicate::StartsWith { .. }
            | Predicate::EndsWith { .. } => false,
            // A null OPTIONAL MATCH binding carries no traversed edge either, so
            // an edge-scoped leaf is not-true (Issue #3622).
            Predicate::EdgeScoped(_) => false,
            // A null OPTIONAL MATCH binding carries no entity and therefore no
            // provenance bundle: `provenance(x) IS NULL` is true, every other
            // provenance comparison is not-true (Issue #3354a).
            Predicate::Provenance(p) => p.evaluate(None),
            Predicate::And(preds) => preds.iter().all(|p| self.evaluate_null(p)),
            Predicate::Or(preds) => preds.iter().any(|p| self.evaluate_null(p)),
            Predicate::Not(pred) => !self.evaluate_null(pred),
        }
    }
}

impl ResultIterator for FilterIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        loop {
            match self.input.next() {
                Some(Ok(row)) => {
                    if let Some(node) = row.entity.as_node() {
                        // Provenance is resolved lazily at the provenance leaf
                        // and memoized per row, so an AND/OR of a property
                        // filter and a provenance filter shares one lookup and a
                        // cheaper conjunct's rejection skips the historical read
                        // entirely (Issue #3354a). A `Predicate::EdgeScoped` leaf
                        // (Issue #3622) evaluates against the row's traversed
                        // edge side channel.
                        let prov_cache: RefCell<Option<Option<Provenance>>> = RefCell::new(None);
                        if self.evaluate_predicate(
                            &self.predicate,
                            node,
                            row.edge.as_ref(),
                            &prov_cache,
                        ) {
                            return Some(Ok(row));
                        }
                        // Filter didn't pass, continue to next
                    } else if row.entity.is_null() {
                        // Null bindings from unmatched OPTIONAL MATCH rows are
                        // evaluated with null semantics (comparisons are
                        // not-true, IS NULL is true).
                        if self.evaluate_null(&self.predicate) {
                            return Some(Ok(row));
                        }
                        // Filter didn't pass, continue to next
                    } else if self.edge_mode && row.entity.as_edge().is_some() {
                        // Real edge-property evaluation (Issue #3622, SQL `FROM
                        // edges` lane): the stream is pure edges, so every
                        // property leaf refers to the edge's own properties.
                        // Provenance leaves resolve against the edge's bundle.
                        let edge = row
                            .entity
                            .as_edge()
                            .expect("as_edge checked in the branch guard");
                        let prov_cache: RefCell<Option<Option<Provenance>>> = RefCell::new(None);
                        if self.evaluate_edge_full(&self.predicate, edge, &prov_cache) {
                            return Some(Ok(row));
                        }
                        // Filter didn't pass, continue to next
                    } else if self.needs_provenance {
                        // Edge rows historically pass through the filter. When
                        // the predicate carries a provenance leaf, evaluate that
                        // leaf against the edge's own provenance (resolved lazily
                        // and memoized); non-provenance leaves keep the
                        // pass-through behavior. A non-edge entity resolves to no
                        // provenance bundle, preserving the prior semantics
                        // (Issue #3354a).
                        let edge = row.entity.as_edge();
                        let prov_cache: RefCell<Option<Option<Provenance>>> = RefCell::new(None);
                        if self.evaluate_edge_predicate(&self.predicate, edge, &prov_cache) {
                            return Some(Ok(row));
                        }
                        // Filter didn't pass, continue to next
                    } else {
                        // Non-node entities pass through (no provenance leaf).
                        return Some(Ok(row));
                    }
                }
                Some(Err(e)) => return Some(Err(e)),
                None => return None,
            }
        }
    }
}

/// Iterator that yields a single, pre-built row.
///
/// Used by [`OptionalApplyIterator`] to seed the per-row optional
/// sub-pipeline with the current input row.
struct SeedRowIterator {
    row: Option<QueryRow>,
}

impl SeedRowIterator {
    fn new(row: QueryRow) -> Self {
        SeedRowIterator { row: Some(row) }
    }
}

impl ResultIterator for SeedRowIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        self.row.take().map(Ok)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = usize::from(self.row.is_some());
        (n, Some(n))
    }
}

/// Iterator implementing left-outer (`OPTIONAL MATCH`) semantics.
///
/// For each input row, the configured sub-pipeline (traversal hops and
/// filters) is executed seeded from that row:
///
/// - If it produces at least one row, those rows are yielded (matched case).
/// - If it produces nothing, a single row with [`EntityResult::Null`] is
///   yielded instead, preserving the input row (unmatched case).
///
/// Because the sub-pipeline includes the optional pattern's inline property
/// filters and its `WHERE` clause, filtering happens *before* the
/// matched/unmatched decision, per openCypher semantics.
///
/// When the first step is a `Scan` the iterator is *standalone* (a leading
/// `OPTIONAL MATCH` with no prior rows): the sub-pipeline runs exactly once
/// from its own node scan, and an empty result yields one null row.
///
/// A null seed row (from a preceding unmatched optional) traverses to nothing,
/// so a chained `OPTIONAL MATCH` over it yields another null row -- null
/// propagates without dropping the row.
pub struct OptionalApplyIterator {
    input: Box<dyn ResultIterator>,
    steps: Vec<crate::query::planner::physical::OptionalPhysicalStep>,
    current: Arc<CurrentStorage>,
    historical: Arc<RwLock<HistoricalStorage>>,
    /// Optional namespace scope (Issue #3349, PR2). Threaded from the executor so
    /// the sub-pipeline built per seed row (its `Scan` source leaf and every
    /// `Traverse` hop) honors the same scope boundary as the outer plan — else a
    /// scoped `OPTIONAL MATCH` would leak cross-namespace bindings. `None` ⇒ prior
    /// namespace-agnostic behavior. Unreachable until PR3 wires Cypher/AQL scope
    /// to `OPTIONAL MATCH`, but wired here (defense in depth) so it can never be
    /// silently wrong.
    scope: Option<Arc<NamespaceScope>>,
    /// The attached `scope` resolved to interned ids once (Issue #3349, PR2), so
    /// the sub-pipeline's per-edge/per-row checks avoid re-resolving. `All` when
    /// no restricting scope is attached.
    resolved_scope: ResolvedScope,
    /// True when the first step is a Scan (leading OPTIONAL MATCH form).
    standalone: bool,
    /// The sub-pipeline currently being drained (one per seed row).
    inner: Option<Box<dyn ResultIterator>>,
    /// Whether the current sub-pipeline has produced at least one row.
    inner_matched: bool,
    /// Set when the input (or the single standalone run) is exhausted.
    done: bool,
    /// The seed row currently being processed, kept so an unmatched fallback
    /// preserves its metadata (score, path, timestamp). `None` in the
    /// standalone form, which has no seed row.
    current_seed: Option<QueryRow>,
}

impl OptionalApplyIterator {
    /// Create a new OptionalApplyIterator.
    pub fn new(
        input: Box<dyn ResultIterator>,
        steps: Vec<crate::query::planner::physical::OptionalPhysicalStep>,
        current: Arc<CurrentStorage>,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Self {
        Self::with_namespace_scope(input, steps, current, historical, None, ResolvedScope::All)
    }

    /// Create an OptionalApplyIterator whose sub-pipeline honors a namespace
    /// scope (Issue #3349, PR2). `scope` is the outer plan's scope and `resolved`
    /// its once-resolved interned-id form; both are threaded into the per-seed
    /// sub-pipeline so its source leaf and traversal hops are scope-bounded. A
    /// `None` / [`ResolvedScope::All`] scope reproduces the namespace-agnostic
    /// behavior of [`new`](Self::new).
    #[must_use]
    pub fn with_namespace_scope(
        input: Box<dyn ResultIterator>,
        steps: Vec<crate::query::planner::physical::OptionalPhysicalStep>,
        current: Arc<CurrentStorage>,
        historical: Arc<RwLock<HistoricalStorage>>,
        scope: Option<Arc<NamespaceScope>>,
        resolved: ResolvedScope,
    ) -> Self {
        use crate::query::planner::physical::OptionalPhysicalStep;
        let standalone = matches!(steps.first(), Some(OptionalPhysicalStep::Scan { .. }));
        Self {
            input,
            steps,
            current,
            historical,
            scope,
            resolved_scope: resolved,
            standalone,
            inner: None,
            inner_matched: false,
            done: false,
            current_seed: None,
        }
    }

    /// The attached scope only if it actually restricts (present and not `All`).
    fn restricting_scope(&self) -> Option<&Arc<NamespaceScope>> {
        self.scope
            .as_ref()
            .filter(|s| !matches!(s.as_ref(), NamespaceScope::All))
    }

    /// Build the optional sub-pipeline for one seed row (or, for the
    /// standalone form, from the leading scan step).
    fn build_pipeline(&self, seed: Option<QueryRow>) -> Box<dyn ResultIterator> {
        use crate::query::planner::physical::OptionalPhysicalStep;

        let mut iter: Box<dyn ResultIterator> = match seed {
            Some(row) => Box::new(SeedRowIterator::new(row)),
            None => Box::new(EmptyIterator),
        };

        for step in &self.steps {
            iter = match step {
                OptionalPhysicalStep::Scan { label } => {
                    // Source step (standalone form): replaces the seed input. When
                    // scoped (Issue #3349, PR2), wrap it in the source-leaf
                    // namespace filter exactly as the main pipeline does.
                    let mut scan: Box<dyn ResultIterator> = Box::new(NodeScanIterator::new(
                        label.clone(),
                        Arc::clone(&self.current),
                    ));
                    if let Some(scope) = self.restricting_scope() {
                        scan = Box::new(ScopeFilterIterator::new(
                            scan,
                            Arc::clone(scope),
                            self.resolved_scope.clone(),
                            Arc::clone(&self.current),
                        ));
                    }
                    scan
                }
                OptionalPhysicalStep::Traverse {
                    direction,
                    label,
                    min_depth,
                    depth,
                    temporal_context,
                } => {
                    let mut traversal = TraversalIterator::new(
                        iter,
                        *direction,
                        label.clone(),
                        *min_depth,
                        *depth,
                        Arc::clone(&self.current),
                        Arc::clone(&self.historical),
                        *temporal_context,
                    );
                    // Namespace boundary (Issue #3349, PR2): a scoped optional
                    // traversal never crosses an out-of-scope edge/node.
                    if let Some(scope) = self.restricting_scope() {
                        traversal = traversal
                            .with_namespace_scope(Arc::clone(scope), self.resolved_scope.clone());
                    }
                    Box::new(traversal)
                }
                OptionalPhysicalStep::Filter(predicate) => {
                    Box::new(FilterIterator::with_historical(
                        iter,
                        predicate.clone(),
                        Arc::clone(&self.historical),
                    ))
                }
            };
        }

        iter
    }
}

impl ResultIterator for OptionalApplyIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        loop {
            // Drain the current sub-pipeline, if any.
            if let Some(inner) = self.inner.as_mut() {
                match inner.next() {
                    Some(Ok(row)) => {
                        self.inner_matched = true;
                        return Some(Ok(row));
                    }
                    Some(Err(e)) => {
                        // Abandon the errored seed entirely: a consumer that
                        // iterates past the error must not receive a
                        // fabricated null row for it.
                        self.inner = None;
                        self.inner_matched = false;
                        self.current_seed = None;
                        return Some(Err(e));
                    }
                    None => {
                        let unmatched = !self.inner_matched;
                        self.inner = None;
                        let seed = self.current_seed.take();
                        if unmatched {
                            // Left-outer semantics: preserve the input row --
                            // including its metadata (score, path, timestamp)
                            // -- with a null binding. The standalone form has
                            // no seed row, so it falls back to a bare null row.
                            let mut row =
                                seed.unwrap_or_else(|| QueryRow::from_entity(EntityResult::Null));
                            row.entity = EntityResult::Null;
                            return Some(Ok(row));
                        }
                        continue;
                    }
                }
            }

            if self.done {
                return None;
            }

            if self.standalone {
                // Leading OPTIONAL MATCH: run the scan pipeline exactly once.
                self.inner = Some(self.build_pipeline(None));
                self.inner_matched = false;
                self.done = true;
                continue;
            }

            // Pull the next seed row from the input.
            match self.input.next() {
                Some(Ok(seed)) => {
                    self.current_seed = Some(seed.clone());
                    self.inner = Some(self.build_pipeline(Some(seed)));
                    self.inner_matched = false;
                }
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    self.done = true;
                    return None;
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // At least one row per remaining input row (matched rows may add
        // more, so no useful upper bound).
        let (lower, _) = self.input.size_hint();
        (lower, None)
    }
}

/// Helper struct for maintaining query rows with similarity scores in a heap.
/// Ordered by score (higher is better) via Ord implementation.
#[derive(Clone)]
struct ScoredRow {
    row: QueryRow,
    score: f32,
}

impl PartialEq for ScoredRow {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
    }
}
impl Eq for ScoredRow {}

impl PartialOrd for ScoredRow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredRow {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Invariant: compute_similarity() filters out non-finite values,
        // so all scores in the heap are finite.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Iterator for vector reranking.
pub struct VectorRerankIterator {
    sorted: Option<std::vec::IntoIter<Reverse<ScoredRow>>>,
    input: Option<Box<dyn ResultIterator>>,
    embedding: Arc<[f32]>,
    k: usize,
    _current: Arc<CurrentStorage>,
    /// Vector property name, or None if no vector index is configured
    vector_property: Option<String>,
    /// Metric used to score each candidate row (Cosine reproduces the historical default).
    metric: VectorMetric,
    /// Optional similarity-score threshold; rows failing it are dropped.
    threshold: Option<ScoreThreshold>,
    /// When set, the computed score is also materialized as an output column
    /// under this alias (in addition to [`QueryRow::score`]).
    score_alias: Option<String>,
}

impl VectorRerankIterator {
    /// Create a new VectorRerankIterator.
    ///
    /// # Arguments
    /// * `input` - The input iterator to rerank
    /// * `embedding` - The target embedding for similarity comparison
    /// * `k` - Maximum number of results to keep
    /// * `current` - Reference to current storage
    /// * `property_key` - Optional property to use for reranking. If None, uses default.
    pub fn new(
        input: Box<dyn ResultIterator>,
        embedding: Arc<[f32]>,
        k: usize,
        current: Arc<CurrentStorage>,
        property_key: Option<String>,
    ) -> Self {
        Self::with_options(
            input,
            embedding,
            k,
            current,
            property_key,
            VectorMetric::Cosine,
            None,
            None,
        )
    }

    /// Create a new VectorRerankIterator with explicit metric / threshold /
    /// score-alias options (Cypher `vector.cosine`/`vector.euclidean`/
    /// `vector.dot_product`, `WHERE vector.similarity(...) > t`, and
    /// `vector.cosine(...) AS score` respectively).
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        input: Box<dyn ResultIterator>,
        embedding: Arc<[f32]>,
        k: usize,
        current: Arc<CurrentStorage>,
        property_key: Option<String>,
        metric: VectorMetric,
        threshold: Option<ScoreThreshold>,
        score_alias: Option<String>,
    ) -> Self {
        // Use explicit property if provided, otherwise get default from storage
        let vector_property = property_key.or_else(|| current.get_vector_property_name());

        VectorRerankIterator {
            sorted: None,
            input: Some(input),
            embedding,
            k,
            _current: current,
            vector_property,
            metric,
            threshold,
            score_alias,
        }
    }

    /// Compute the similarity score for a query row if it has a vector property.
    ///
    /// Returns:
    /// - `Ok(Some(score))` when a finite score passing the threshold is computed,
    /// - `Ok(None)` when the row has no vector property, an invalid (NaN/Inf)
    ///   score, or a score that fails the threshold (all skip the row),
    /// - `Err(_)` when the metric computation fails -- notably a **dimension
    ///   mismatch** between the query embedding and the stored vector, which is
    ///   surfaced as a clear error rather than silently dropping the row.
    fn compute_similarity(&self, row: &QueryRow, vector_property: &str) -> Result<Option<f32>> {
        let Some(node) = row.entity.as_node() else {
            return Ok(None);
        };
        let Some(PropertyValue::Vector(vec)) = node.properties.get(vector_property) else {
            return Ok(None);
        };
        // A length mismatch is a genuine dimension error (clear message from the
        // underlying vector op); propagate it instead of swallowing it.
        let similarity = self.metric.compute_similarity(&self.embedding, vec)?;
        // Reject NaN/Inf values - these indicate invalid input (e.g., zero-length vectors)
        if !similarity.is_finite() {
            #[cfg(feature = "observability")]
            tracing::debug!(
                "Skipping node {:?} with non-finite similarity score: {}",
                node.id,
                similarity
            );
            return Ok(None);
        }
        // Apply an optional score threshold (Cypher WHERE vector.similarity > t).
        if let Some(threshold) = self.threshold
            && !threshold.passes(similarity)
        {
            return Ok(None);
        }
        Ok(Some(similarity))
    }
}

impl ResultIterator for VectorRerankIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        // Lazy initialization: collect and sort on first call
        if self.sorted.is_none() && self.input.is_some() {
            // Check if vector index is configured
            let vector_property = match &self.vector_property {
                Some(prop) => prop.as_str(),
                None => {
                    return Some(Err(crate::core::error::Error::Vector(
                        crate::core::error::VectorError::IndexError(
                            "VectorRerank requires a vector index to be enabled. \
                             Call db.vector_index(\"...\").hnsw(...).enable() first."
                                .to_string(),
                        ),
                    )));
                }
            };

            let mut input = self.input.take()?;
            // Use a min-heap to keep the top-k results. `k` may be unbounded
            // (usize::MAX) for a pure threshold filter (keep every passing row),
            // so cap the *pre-allocation* while leaving the logical bound intact.
            let mut heap = BinaryHeap::with_capacity(self.k.min(1024));

            while let Some(result) = input.next() {
                match result {
                    Ok(row) => {
                        // Get vector from node and compute similarity. A
                        // dimension mismatch propagates as a clear error.
                        let similarity = match self.compute_similarity(&row, vector_property) {
                            Ok(Some(s)) => s,
                            Ok(None) => continue,
                            Err(e) => return Some(Err(e)),
                        };
                        debug_assert!(similarity.is_finite(), "Non-finite similarity score");
                        if heap.len() < self.k {
                            heap.push(Reverse(ScoredRow {
                                row,
                                score: similarity,
                            }));
                        } else {
                            #[allow(clippy::collapsible_if)]
                            if let Some(Reverse(min_row)) = heap.peek() {
                                if similarity > min_row.score {
                                    heap.pop();
                                    heap.push(Reverse(ScoredRow {
                                        row,
                                        score: similarity,
                                    }));
                                }
                            }
                        }
                    }
                    Err(e) => return Some(Err(e)),
                }
            }

            // Convert heap to sorted vector (descending score)
            // BinaryHeap::into_sorted_vec() returns elements in ascending order of T.
            // Since T is Reverse<ScoredRow>, the order is:
            // [Smallest Reverse<ScoredRow>, ..., Largest Reverse<ScoredRow>]
            // Smallest Reverse<ScoredRow> corresponds to Largest ScoredRow (highest score).
            // So the result is [Highest Score, ..., Lowest Score], which is exactly what we want.
            self.sorted = Some(heap.into_sorted_vec().into_iter());
        }

        let score_alias = self.score_alias.clone();
        self.sorted.as_mut()?.next().map(|Reverse(item)| {
            let mut row = item.row;
            row.score = Some(item.score);
            // Materialize the score as a named output column when the query
            // aliased the vector function (Cypher `vector.cosine(...) AS score`),
            // preserving the row's entity (unlike aggregation's Null-entity rows).
            // NOTE (cross-lane follow-up): the MCP serializer (`query_row_to_json`
            // in src/mcp/server.rs) currently ignores `QueryRow.columns`, so this
            // aliased score is NOT yet surfaced through the MCP `query` tool --
            // same gap as aggregation columns (#558). Lane-4 follow-up.
            if let Some(alias) = score_alias {
                let col = (alias, PropertyValue::Float(f64::from(item.score)));
                match row.columns {
                    Some(ref mut cols) => cols.push(col),
                    None => row.columns = Some(vec![col]),
                }
            }
            Ok(row)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(ref sorted) = self.sorted {
            sorted.size_hint()
        } else {
            // `k` may be unbounded (usize::MAX) for a pure threshold filter; the
            // real upper bound is then the input size, which we do not know here.
            let upper = (self.k != usize::MAX).then_some(self.k);
            (0, upper)
        }
    }
}

/// Iterator for limiting results.
///
/// # Example
///
/// ```rust
/// use aletheiadb::query::executor::{LimitIterator, NodeScanIterator};
/// use std::sync::Arc;
///
/// let current = Arc::new(aletheiadb::storage::CurrentStorage::new());
/// let input = Box::new(NodeScanIterator::new(Some("Person".to_string()), current));
///
/// // Skip 5, take 10
/// let limit_iter = LimitIterator::new(input, 5, 10);
/// ```
pub struct LimitIterator {
    input: Box<dyn ResultIterator>,
    offset: usize,
    count: usize,
    skipped: usize,
    returned: usize,
}

impl LimitIterator {
    /// Create a new LimitIterator that applies offset and limit to the input.
    pub fn new(input: Box<dyn ResultIterator>, offset: usize, count: usize) -> Self {
        LimitIterator {
            input,
            offset,
            count,
            skipped: 0,
            returned: 0,
        }
    }
}

impl ResultIterator for LimitIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        // Skip offset
        while self.skipped < self.offset {
            match self.input.next() {
                Some(Ok(_)) => self.skipped += 1,
                Some(Err(e)) => return Some(Err(e)),
                None => return None,
            }
        }

        // Check count limit
        if self.returned >= self.count {
            return None;
        }

        match self.input.next() {
            Some(result) => {
                self.returned += 1;
                Some(result)
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count.saturating_sub(self.returned);
        let (lower, upper) = self.input.size_hint();
        (lower.min(remaining), upper.map(|u| u.min(remaining)))
    }
}

/// Wrapper iterator that strips provenance metadata when include_provenance is false.
///
/// This iterator conditionally removes timestamp and path information from QueryRow
/// results based on the query hint. When include_provenance is false, these fields
/// are set to None for better performance and reduced memory usage.
pub struct ProvenanceFilterIterator {
    inner: Box<dyn ResultIterator>,
    include_provenance: bool,
}

impl ProvenanceFilterIterator {
    /// Create a new ProvenanceFilterIterator that conditionally strips metadata.
    pub fn new(inner: Box<dyn ResultIterator>, include_provenance: bool) -> Self {
        ProvenanceFilterIterator {
            inner,
            include_provenance,
        }
    }
}

impl ResultIterator for ProvenanceFilterIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        self.inner.next().map(|result| {
            result.map(|mut row| {
                if !self.include_provenance {
                    // Strip provenance metadata
                    row.path = None;
                    row.timestamp = None;
                }
                row
            })
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Row count below which the wall-clock deadline is polled on *every* row, and
/// at/above which it is polled once per [`DEADLINE_CHECK_STRIDE`] rows.
///
/// This hybrid exists to reconcile two requirements that a single fixed stride
/// cannot:
///
/// - **Correctness for slow, low-row-count queries** (e.g. a handful of deep
///   temporal reconstructions, each taking many milliseconds). A pure stride
///   would only re-poll the clock every `STRIDE` rows, so a query that produces
///   *fewer than* `STRIDE` rows would be checked only at row 0 and then run to
///   completion no matter how long it took — the deadline would never fire.
///   Polling every row for the first `PRECISE_DEADLINE_ROWS` rows guarantees
///   such a query is cut at the first row boundary past its deadline.
/// - **Cheapness for large benign scans** (the AC6 "fast queries are not taxed"
///   concern). `Instant::now()` is a ~15-25ns vDSO call; paying it on every one
///   of a million rows would be a measurable throughput tax. Beyond
///   `PRECISE_DEADLINE_ROWS` the query is, by definition, a *many-row* scan, so
///   the clock read is amortized to `1/STRIDE`.
const PRECISE_DEADLINE_ROWS: usize = 4096;

/// Stride (in rows) at which the wall-clock deadline is polled once a scan has
/// produced more than [`PRECISE_DEADLINE_ROWS`] rows. See that constant for the
/// precision/overhead rationale.
///
/// Row index 0 always polls (so an already-past deadline short-circuits *before*
/// the first pull). The residual precision tradeoff — cancellation is a
/// row-boundary event, so a *many-fast-rows* scan can overshoot its deadline by
/// up to the time to produce `STRIDE` more rows (tens of microseconds; far
/// inside the ≤10% grace bound even for a 100ms timeout) — applies only in the
/// amortized regime. A query whose *individual rows* each exceed the grace
/// budget can overshoot by one such row regardless of stride: the inherent floor
/// of cooperative, row-granular cancellation.
const DEADLINE_CHECK_STRIDE: usize = 64;

/// Cooperative-cancellation wrapper enforcing per-query resource limits
/// (Issue #3368 engine lane; see [`crate::query::limits`]).
///
/// Installed by [`super::QueryExecutor::execute`] only when
/// [`QueryResourceLimits::is_unlimited`] is `false` -- the common, unlimited
/// case never allocates this wrapper and pays no per-row overhead (see the
/// module-level docs on `crate::query::limits`).
///
/// Each dimension is checked in a fixed order on every [`next`](Self::next)
/// call:
///
/// 1. **Wall-clock timeout**, checked *before* pulling from the inner
///    iterator, so a query that already blew its budget performs no further
///    work at all (provably: the test suite wraps a mock inner iterator that
///    panics if `next()` is ever called and confirms a past deadline never
///    reaches it).
/// 2. **Result rows**, checked after a row is pulled.
/// 3. **Memory bytes** (via [`estimate_row_bytes`]), checked after a row is
///    pulled and has passed the row-count check.
///
/// Once any dimension is breached the guard is "fused": every subsequent
/// `next()` call returns `None` without touching the inner iterator again,
/// mirroring the standard library's `Fuse` behavior for an iterator that has
/// just yielded an error it cannot meaningfully recover from mid-stream.
pub struct ResourceGuardIterator {
    inner: Box<dyn ResultIterator>,
    deadline: Option<std::time::Instant>,
    max_rows: Option<usize>,
    max_memory_bytes: Option<usize>,
    counters: Option<Arc<crate::query::limits::LimitCounters>>,
    /// When the guard was constructed. Used to report `consumed` (elapsed
    /// milliseconds) on a wall-clock timeout error.
    started: std::time::Instant,
    /// The wall-clock budget in milliseconds, captured at construction as the
    /// remaining time-to-deadline (an honest proxy for the originally
    /// configured timeout, since the guard is installed immediately after the
    /// deadline is computed). `0` when there is no deadline.
    configured_timeout_ms: u64,
    rows_emitted: usize,
    bytes_accumulated: usize,
    /// Set once any dimension has been breached; short-circuits all further
    /// `next()` calls to `None`.
    terminated: bool,
}

impl ResourceGuardIterator {
    /// Wrap `inner` with the given resolved `limits`, optionally recording
    /// terminations into `counters`.
    pub fn new(
        inner: Box<dyn ResultIterator>,
        limits: crate::query::limits::QueryResourceLimits,
        counters: Option<Arc<crate::query::limits::LimitCounters>>,
    ) -> Self {
        let started = std::time::Instant::now();
        let configured_timeout_ms = limits
            .deadline
            .map(|d| d.saturating_duration_since(started).as_millis() as u64)
            .unwrap_or(0);
        ResourceGuardIterator {
            inner,
            deadline: limits.deadline,
            max_rows: limits.max_rows,
            max_memory_bytes: limits.max_memory_bytes,
            counters,
            started,
            configured_timeout_ms,
            rows_emitted: 0,
            bytes_accumulated: 0,
            terminated: false,
        }
    }

    fn record(&self, dimension: crate::query::limits::LimitDimension) {
        if let Some(counters) = &self.counters {
            counters.record_termination(dimension);
        }
    }
}

impl ResultIterator for ResourceGuardIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        use crate::core::error::{Error, QueryError};
        use crate::query::limits::LimitDimension;

        if self.terminated {
            return None;
        }

        // 1. Wall-clock timeout: checked BEFORE pulling from `inner`, so a
        // query that has already exhausted its budget never does another unit of
        // work. The clock is polled on every row for the first
        // `PRECISE_DEADLINE_ROWS` rows (so a slow, low-row-count query is cut
        // precisely) and once per `DEADLINE_CHECK_STRIDE` rows thereafter (so a
        // large benign scan pays no per-row clock syscall). Row index 0 always
        // polls. See the two constants' docs for the full rationale.
        if let Some(deadline) = self.deadline
            && (self.rows_emitted < PRECISE_DEADLINE_ROWS
                || self.rows_emitted.is_multiple_of(DEADLINE_CHECK_STRIDE))
            && std::time::Instant::now() >= deadline
        {
            self.terminated = true;
            self.record(LimitDimension::WallClockTimeout);
            let consumed = self.started.elapsed().as_millis() as u64;
            return Some(Err(Error::Query(QueryError::ResourceExhausted {
                dimension: LimitDimension::WallClockTimeout.as_str(),
                limit: self.configured_timeout_ms,
                consumed,
                retriable: true,
            })));
        }

        let row = match self.inner.next()? {
            Ok(row) => row,
            Err(e) => return Some(Err(e)),
        };

        // 2. Result rows.
        self.rows_emitted += 1;
        if let Some(max_rows) = self.max_rows
            && self.rows_emitted > max_rows
        {
            self.terminated = true;
            self.record(LimitDimension::ResultRows);
            return Some(Err(Error::Query(QueryError::ResourceExhausted {
                dimension: LimitDimension::ResultRows.as_str(),
                limit: max_rows as u64,
                consumed: self.rows_emitted as u64,
                retriable: false,
            })));
        }

        // 3. Memory bytes. Only pay the row-byte-estimation cost when the
        // dimension is actually enforced.
        if let Some(budget) = self.max_memory_bytes {
            self.bytes_accumulated += crate::query::limits::estimate_row_bytes(&row);
            if self.bytes_accumulated > budget {
                self.terminated = true;
                self.record(LimitDimension::MemoryBytes);
                return Some(Err(Error::Query(QueryError::ResourceExhausted {
                    dimension: LimitDimension::MemoryBytes.as_str(),
                    limit: budget as u64,
                    consumed: self.bytes_accumulated as u64,
                    retriable: false,
                })));
            }
        }

        Some(Ok(row))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Iterator for projecting specific properties from query results.
pub struct ProjectIterator {
    input: Box<dyn ResultIterator>,
    properties: Vec<String>,
}

impl ProjectIterator {
    /// Create a new ProjectIterator that projects specific properties from the results.
    pub fn new(input: Box<dyn ResultIterator>, mut properties: Vec<String>) -> Self {
        // Deduplicate properties to prevent errors when projecting same property multiple times
        properties.sort();
        properties.dedup();
        ProjectIterator { input, properties }
    }
}

impl ResultIterator for ProjectIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        match self.input.next() {
            Some(Ok(mut row)) => {
                if let Some(node) = row.entity.as_node() {
                    let mut new_props = crate::core::PropertyMapBuilder::new();
                    for prop in &self.properties {
                        if let Some(val) = node.properties.get(prop) {
                            new_props = match new_props.try_insert(prop, val.clone()) {
                                Ok(p) => p,
                                Err(e) => return Some(Err(e)),
                            };
                        }
                    }
                    let new_node = crate::core::graph::Node::new(
                        node.id,
                        node.label,
                        new_props.build(),
                        node.current_version,
                    );
                    row.entity = EntityResult::Node(new_node);
                }
                Some(Ok(row))
            }
            other => other,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

/// Convert a `PredicateValue` to a `PropertyValue` for storage-level lookups.
fn predicate_to_property_value(pv: &PredicateValue) -> PropertyValue {
    match pv {
        PredicateValue::Null => PropertyValue::Null,
        PredicateValue::Bool(b) => PropertyValue::Bool(*b),
        PredicateValue::Int(i) => PropertyValue::Int(*i),
        PredicateValue::Float(f) => PropertyValue::Float(*f),
        PredicateValue::String(s) => PropertyValue::String(Arc::from(s.as_str())),
    }
}

/// Iterator for property-based node scans.
///
/// Calls `CurrentStorage::find_nodes_by_property` to get matching node IDs,
/// then resolves each to a full `Node` for the query result.
pub struct PropertyScanIterator {
    current: Arc<CurrentStorage>,
    initialized: bool,
    node_ids: Option<std::vec::IntoIter<NodeId>>,
    label: String,
    property_value: PropertyValue,
    property_key: String,
}

impl PropertyScanIterator {
    /// Initialize a full-scan iterator that evaluates a property predicate against all nodes.
    ///
    /// # Why?
    /// Use this as a fallback when no index is available for the requested `label` and `key`.
    /// It eagerly loads all matching nodes into memory, so it is best used on small datasets.
    pub fn new(
        label: String,
        key: String,
        value: &PredicateValue,
        current: Arc<CurrentStorage>,
    ) -> Self {
        PropertyScanIterator {
            current,
            initialized: false,
            node_ids: None,
            label,
            property_value: predicate_to_property_value(value),
            property_key: key,
        }
    }

    fn initialize(&mut self) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        let ids = self.current.find_nodes_by_property(
            &self.label,
            &self.property_key,
            &self.property_value,
        );
        self.node_ids = Some(ids.into_iter());
    }
}

impl ResultIterator for PropertyScanIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        self.initialize();

        match self.node_ids.as_mut()?.next() {
            Some(id) => match self.current.get_node(id) {
                Ok(node) => Some(Ok(QueryRow::from_entity(EntityResult::Node(node)))),
                Err(e) => Some(Err(e)),
            },
            None => None,
        }
    }
}

// ============================================================================
// Aggregation, DISTINCT, and ORDER BY iterators
// ============================================================================

/// Read a node property value from a row, for grouping / aggregation / sorting.
///
/// Returns `None` when the row has no backing node (e.g. a null binding or a
/// computed aggregate row) or the property is absent.
fn row_property<'a>(row: &'a QueryRow, key: &str) -> Option<&'a PropertyValue> {
    row.entity.as_node().and_then(|n| n.properties.get(key))
}

/// Numeric view of a scalar property (`Int`/`Float`), used for sum/avg and
/// numeric ordering. `None` for non-numeric values.
fn property_as_f64(value: &PropertyValue) -> Option<f64> {
    match value {
        PropertyValue::Int(i) => Some(*i as f64),
        PropertyValue::Float(f) => Some(*f),
        _ => None,
    }
}

/// Total order over comparable scalar property values (numbers compared
/// numerically, strings lexicographically, bools by value). Non-comparable or
/// mixed pairs compare `Equal` so they retain input order.
fn compare_property(a: &PropertyValue, b: &PropertyValue) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (PropertyValue::Int(x), PropertyValue::Int(y)) => x.cmp(y),
        (
            PropertyValue::Int(_) | PropertyValue::Float(_),
            PropertyValue::Int(_) | PropertyValue::Float(_),
        ) => match (property_as_f64(a), property_as_f64(b)) {
            (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
            _ => Ordering::Equal,
        },
        (PropertyValue::String(x), PropertyValue::String(y)) => x.as_ref().cmp(y.as_ref()),
        (PropertyValue::Bool(x), PropertyValue::Bool(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

/// A hashable, `Eq` projection of a scalar property value used both for
/// grouping keys and for `DISTINCT` deduplication of aggregate arguments.
/// Floats are keyed by their bit pattern (total order); non-scalar values fall
/// back to their debug representation.
#[derive(Clone, PartialEq, Eq, Hash)]
enum GroupKeyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(u64),
    Str(String),
    Other(String),
}

impl GroupKeyValue {
    fn from_property(value: Option<&PropertyValue>) -> Self {
        match value {
            None | Some(PropertyValue::Null) => GroupKeyValue::Null,
            Some(PropertyValue::Bool(b)) => GroupKeyValue::Bool(*b),
            Some(PropertyValue::Int(i)) => GroupKeyValue::Int(*i),
            Some(PropertyValue::Float(f)) => Self::float_key(*f),
            Some(PropertyValue::String(s)) => GroupKeyValue::Str(s.to_string()),
            Some(other) => GroupKeyValue::Other(format!("{other:?}")),
        }
    }

    /// Map a float to a grouping/DISTINCT key, canonicalizing `-0.0 -> 0.0`
    /// (signed zeros group together) and unifying an integral float with the
    /// equal integer (so `1` and `1.0` land in the same group / distinct set).
    fn float_key(f: f64) -> Self {
        let f = if f == 0.0 { 0.0 } else { f };
        if f.is_finite() && f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
            GroupKeyValue::Int(f as i64)
        } else {
            GroupKeyValue::Float(f.to_bits())
        }
    }
}

/// The resolved per-row input to one aggregate, computed once per row in
/// [`AggregateIterator::drain`].
enum AggInput<'a> {
    /// `count(*)` -- count the row unconditionally.
    Star,
    /// `count(n)` -- count the row iff its bound entity is non-null.
    Entity { present: bool },
    /// `func(n.prop)` -- the row's value for the property (`None`/`Null` =
    /// absent, skipped by every value aggregate).
    Value(Option<&'a PropertyValue>),
}

/// Per-group, per-aggregate accumulator.
struct AggAccumulator {
    func: AggregateFunc,
    distinct: bool,
    /// Values already counted, for the `DISTINCT` quantifier.
    seen: HashSet<GroupKeyValue>,
    count: i64,
    /// Integer running sum (i128 headroom against overflow) for all-integer sums.
    sum_i: i128,
    /// Floating running sum, always maintained (used for avg and float sums).
    sum_f: f64,
    saw_float: bool,
    numeric_count: i64,
    min: Option<PropertyValue>,
    max: Option<PropertyValue>,
    collected: Vec<PropertyValue>,
}

impl AggAccumulator {
    fn new(spec: &AggregateSpec) -> Self {
        AggAccumulator {
            func: spec.func,
            distinct: spec.distinct,
            seen: HashSet::new(),
            count: 0,
            sum_i: 0,
            sum_f: 0.0,
            saw_float: false,
            numeric_count: 0,
            min: None,
            max: None,
            collected: Vec::new(),
        }
    }

    /// Fold one input row into the accumulator.
    ///
    /// `count(*)` ([`AggInput::Star`]) counts the row unconditionally and
    /// `count(n)` ([`AggInput::Entity`]) counts non-null bindings, both
    /// ignoring `DISTINCT`. Otherwise the value ([`AggInput::Value`]) feeds the
    /// function; `None`/`Null` values are skipped (openCypher semantics).
    fn update(&mut self, input: AggInput<'_>) {
        let v = match input {
            AggInput::Star => {
                // Only Count uses Star (enforced by the converter).
                self.count += 1;
                return;
            }
            AggInput::Entity { present } => {
                // Only Count uses Entity; count non-null bindings.
                if present {
                    self.count += 1;
                }
                return;
            }
            AggInput::Value(Some(v)) if !matches!(v, PropertyValue::Null) => v,
            AggInput::Value(_) => return,
        };

        if self.distinct {
            let key = GroupKeyValue::from_property(Some(v));
            if !self.seen.insert(key) {
                return;
            }
        }

        match self.func {
            AggregateFunc::Count => self.count += 1,
            AggregateFunc::Sum | AggregateFunc::Avg => {
                if let Some(f) = property_as_f64(v) {
                    self.sum_f += f;
                    self.numeric_count += 1;
                    match v {
                        PropertyValue::Int(i) => self.sum_i += i128::from(*i),
                        PropertyValue::Float(_) => self.saw_float = true,
                        _ => {}
                    }
                }
            }
            AggregateFunc::Min => {
                let replace = match &self.min {
                    None => true,
                    Some(m) => compare_property(v, m) == std::cmp::Ordering::Less,
                };
                if replace {
                    self.min = Some(v.clone());
                }
            }
            AggregateFunc::Max => {
                let replace = match &self.max {
                    None => true,
                    Some(m) => compare_property(v, m) == std::cmp::Ordering::Greater,
                };
                if replace {
                    self.max = Some(v.clone());
                }
            }
            AggregateFunc::Collect => self.collected.push(v.clone()),
        }
    }

    /// Produce the final aggregate value for this group.
    fn finalize(self) -> PropertyValue {
        match self.func {
            AggregateFunc::Count => PropertyValue::Int(self.count),
            AggregateFunc::Sum => {
                if self.numeric_count == 0 {
                    // openCypher: sum over no values is 0 (integer).
                    PropertyValue::Int(0)
                } else if self.saw_float {
                    PropertyValue::Float(self.sum_f)
                } else {
                    // Checked cast: an all-integer sum that overflows i64 falls
                    // back to Float rather than silently wrapping.
                    match i64::try_from(self.sum_i) {
                        Ok(v) => PropertyValue::Int(v),
                        Err(_) => PropertyValue::Float(self.sum_i as f64),
                    }
                }
            }
            AggregateFunc::Avg => {
                if self.numeric_count == 0 {
                    PropertyValue::Null
                } else {
                    PropertyValue::Float(self.sum_f / self.numeric_count as f64)
                }
            }
            AggregateFunc::Min => self.min.unwrap_or(PropertyValue::Null),
            AggregateFunc::Max => self.max.unwrap_or(PropertyValue::Null),
            AggregateFunc::Collect => PropertyValue::Array(Arc::new(self.collected)),
        }
    }
}

/// A single group's state during aggregation.
struct AggGroup {
    /// The group-key values (in `group_keys` order), captured from the first
    /// row of the group for output.
    key_values: Vec<PropertyValue>,
    accumulators: Vec<AggAccumulator>,
}

/// Grouped aggregation iterator (openCypher implicit grouping).
///
/// Eagerly drains its input on the first `next()`, hash-groups rows by the
/// group-key tuple, folds each aggregate per group, and then emits exactly one
/// computed-column [`QueryRow`] per group (via [`QueryRow::from_columns`]) in
/// group-discovery order. A global aggregation (no group keys) always emits one
/// row even over empty input; a grouped aggregation over empty input emits no
/// rows.
pub struct AggregateIterator {
    input: Option<Box<dyn ResultIterator>>,
    group_keys: Vec<AggregateGroupKey>,
    aggregates: Vec<AggregateSpec>,
    output: std::vec::IntoIter<QueryRow>,
    drained: bool,
}

impl AggregateIterator {
    /// Create a new aggregation iterator.
    pub fn new(
        input: Box<dyn ResultIterator>,
        group_keys: Vec<AggregateGroupKey>,
        aggregates: Vec<AggregateSpec>,
    ) -> Self {
        AggregateIterator {
            input: Some(input),
            group_keys,
            aggregates,
            output: Vec::new().into_iter(),
            drained: false,
        }
    }

    fn drain(&mut self) -> Result<()> {
        let mut input = match self.input.take() {
            Some(i) => i,
            None => return Ok(()),
        };

        let mut order: Vec<Vec<GroupKeyValue>> = Vec::new();
        let mut groups: std::collections::HashMap<Vec<GroupKeyValue>, AggGroup> =
            std::collections::HashMap::new();

        // A global aggregation always yields exactly one row, even over empty
        // input: seed the single (empty-key) group up front.
        if self.group_keys.is_empty() {
            let key: Vec<GroupKeyValue> = Vec::new();
            order.push(key.clone());
            groups.insert(
                key,
                AggGroup {
                    key_values: Vec::new(),
                    accumulators: self.aggregates.iter().map(AggAccumulator::new).collect(),
                },
            );
        }

        while let Some(row) = input.next() {
            let row = row?;
            let key: Vec<GroupKeyValue> = self
                .group_keys
                .iter()
                .map(|gk| GroupKeyValue::from_property(row_property(&row, &gk.property_key)))
                .collect();

            if !groups.contains_key(&key) {
                order.push(key.clone());
                let key_values = self
                    .group_keys
                    .iter()
                    .map(|gk| {
                        row_property(&row, &gk.property_key)
                            .cloned()
                            .unwrap_or(PropertyValue::Null)
                    })
                    .collect();
                let accumulators = self.aggregates.iter().map(AggAccumulator::new).collect();
                groups.insert(
                    key.clone(),
                    AggGroup {
                        key_values,
                        accumulators,
                    },
                );
            }

            let group = groups
                .get_mut(&key)
                .expect("group inserted above must be present");
            for (spec, acc) in self.aggregates.iter().zip(group.accumulators.iter_mut()) {
                let input = match &spec.arg {
                    AggregateArg::Star => AggInput::Star,
                    AggregateArg::Entity => AggInput::Entity {
                        present: !row.entity.is_null(),
                    },
                    AggregateArg::Property(k) => AggInput::Value(row_property(&row, k)),
                };
                acc.update(input);
            }
        }

        let mut rows = Vec::with_capacity(order.len());
        for key in order {
            let group = groups.remove(&key).expect("group present in order");
            let mut columns: Vec<(String, PropertyValue)> =
                Vec::with_capacity(self.group_keys.len() + self.aggregates.len());
            for (gk, val) in self.group_keys.iter().zip(group.key_values) {
                columns.push((gk.alias.clone(), val));
            }
            for (spec, acc) in self.aggregates.iter().zip(group.accumulators) {
                columns.push((spec.alias.clone(), acc.finalize()));
            }
            rows.push(QueryRow::from_columns(columns));
        }
        self.output = rows.into_iter();
        Ok(())
    }
}

impl ResultIterator for AggregateIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        if !self.drained {
            self.drained = true;
            if let Err(e) = self.drain() {
                return Some(Err(e));
            }
        }
        self.output.next().map(Ok)
    }
}

/// One change-point in an entity's reconstructed valid-time timeline, as
/// believed at the query's transaction-time coordinate (Issue #3363). The
/// version's value holds from `valid_from` until the next change-point (or
/// forever, for the last one).
struct BelievedVersion {
    /// Valid-time interval start, microseconds since epoch.
    valid_from: i64,
    /// The version's reconstructed properties.
    properties: crate::core::property::PropertyMap,
}

/// Per-window, per-aggregate accumulator for temporal window aggregation.
struct WindowAcc {
    func: WindowAggFunc,
    /// COUNT: entities/samples present; CHANGES: version-start count.
    count: i64,
    /// Count of numeric samples folded (value aggregates).
    numeric_count: i64,
    sum_i: i128,
    sum_f: f64,
    saw_float: bool,
    min: Option<PropertyValue>,
    max: Option<PropertyValue>,
}

impl WindowAcc {
    fn new(func: WindowAggFunc) -> Self {
        WindowAcc {
            func,
            count: 0,
            numeric_count: 0,
            sum_i: 0,
            sum_f: 0.0,
            saw_float: false,
            min: None,
            max: None,
        }
    }

    /// Fold one numeric value sample (SUM/AVG/MIN/MAX/COUNT-of-property).
    fn fold_value(&mut self, v: &PropertyValue) {
        if let Some(f) = property_as_f64(v) {
            self.numeric_count += 1;
            self.sum_f += f;
            match v {
                PropertyValue::Int(i) => self.sum_i += i128::from(*i),
                PropertyValue::Float(_) => self.saw_float = true,
                _ => {}
            }
            match &self.min {
                Some(m) if compare_property(v, m) != std::cmp::Ordering::Less => {}
                _ => self.min = Some(v.clone()),
            }
            match &self.max {
                Some(m) if compare_property(v, m) != std::cmp::Ordering::Greater => {}
                _ => self.max = Some(v.clone()),
            }
        }
    }

    /// Produce the final [`PropertyValue`] for this window/aggregate. Value
    /// aggregates over an empty/absent sample set yield `Null` (an explicit
    /// "no data" marker per the #3363 contract); COUNT/CHANGES yield `0`.
    fn finalize(self) -> PropertyValue {
        match self.func {
            WindowAggFunc::Count | WindowAggFunc::Changes => PropertyValue::Int(self.count),
            WindowAggFunc::Sum => {
                if self.numeric_count == 0 {
                    PropertyValue::Null
                } else if self.saw_float {
                    PropertyValue::Float(self.sum_f)
                } else {
                    // All-integer sum; promote to float only if it overflows i64.
                    i64::try_from(self.sum_i)
                        .map(PropertyValue::Int)
                        .unwrap_or(PropertyValue::Float(self.sum_f))
                }
            }
            WindowAggFunc::Avg => {
                if self.numeric_count == 0 {
                    PropertyValue::Null
                } else {
                    PropertyValue::Float(self.sum_f / self.numeric_count as f64)
                }
            }
            WindowAggFunc::Min => self.min.unwrap_or(PropertyValue::Null),
            WindowAggFunc::Max => self.max.unwrap_or(PropertyValue::Null),
        }
    }
}

/// Temporal aggregation window iterator (Issue #3363).
///
/// On the first `next()` it drains the upstream matched-entity stream to a set
/// of node ids, reconstructs each entity's valid-time history as believed at the
/// spec's transaction-time coordinate, buckets that history into the spec's
/// tumbling windows, and emits one computed-column [`QueryRow`] per window
/// carrying `window_start`/`window_end` (RFC 3339) followed by each aggregate.
/// See [`crate::query::temporal_window`] for the boundary contract and the
/// crate guide for the sampling rules.
pub struct TemporalWindowAggregateIterator {
    input: Option<Box<dyn ResultIterator>>,
    spec: TemporalWindowSpec,
    historical: Arc<RwLock<HistoricalStorage>>,
    output: std::vec::IntoIter<QueryRow>,
    drained: bool,
}

impl TemporalWindowAggregateIterator {
    /// Create a new temporal window aggregation iterator.
    pub fn new(
        input: Box<dyn ResultIterator>,
        spec: TemporalWindowSpec,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Self {
        TemporalWindowAggregateIterator {
            input: Some(input),
            spec,
            historical,
            output: Vec::new().into_iter(),
            drained: false,
        }
    }

    fn drain(&mut self) -> Result<()> {
        let mut input = match self.input.take() {
            Some(i) => i,
            None => return Ok(()),
        };

        // Collect distinct matched node ids, preserving discovery order. v1
        // windows nodes only (the converter rejects edge/traversal windows).
        let mut node_ids: Vec<NodeId> = Vec::new();
        let mut seen: HashSet<NodeId> = HashSet::new();
        while let Some(row) = input.next() {
            let row = row?;
            if let Some(EntityId::Node(id)) = row.entity.id()
                && seen.insert(id)
            {
                node_ids.push(id);
            }
        }

        // Generate the tumbling windows (validated at convert time; re-run here
        // to obtain the boundaries). A structured error would already have been
        // raised during conversion, so this is effectively infallible.
        let windows = crate::query::temporal_window::generate_windows(
            self.spec.range_start_micros,
            self.spec.range_end_micros,
            self.spec.granularity,
        )
        .map_err(|e| {
            crate::core::error::Error::Query(crate::core::error::QueryError::InvalidParameter {
                parameter: "temporal window".to_string(),
                reason: format!("{e:?}"),
            })
        })?;

        let as_of = self.spec.as_of_system_time;
        let naggs = self.spec.aggregates.len();
        // accs[window_index][aggregate_index]
        let mut accs: Vec<Vec<WindowAcc>> = windows
            .iter()
            .map(|_| {
                self.spec
                    .aggregates
                    .iter()
                    .map(|a| WindowAcc::new(a.func))
                    .collect()
            })
            .collect();

        {
            let hist = self.historical.read();
            for node_id in &node_ids {
                // Reconstruct the entity's valid-time timeline as believed at
                // the transaction-time coordinate (#3363 AC: later corrections
                // must not rewrite past analytics). Every version *recorded by*
                // `as_of` (`transaction_from <= as_of`) contributes a
                // change-point at its `valid_from`; versions recorded later are
                // invisible. Multiple versions sharing a `valid_from` are a
                // same-instant correction, so we keep the one with the latest
                // belief (greatest `transaction_from <= as_of`). Both value
                // sampling and CHANGES read from this single, consistent source.
                // Entities absent from current state (e.g. deleted) cannot be
                // walked in v1 and are skipped (documented).
                let believed: Vec<BelievedVersion> = match hist.get_node_history(*node_id) {
                    Ok(h) => {
                        let mut by_vf: std::collections::HashMap<
                            i64,
                            (Timestamp, crate::core::property::PropertyMap),
                        > = std::collections::HashMap::new();
                        for ver in h.versions {
                            let tx_from = ver.temporal.transaction_time().start();
                            if tx_from > as_of {
                                continue;
                            }
                            let vf = ver.temporal.valid_time().start().wallclock();
                            match by_vf.get(&vf) {
                                Some((existing_tx, _)) if *existing_tx >= tx_from => {}
                                _ => {
                                    by_vf.insert(vf, (tx_from, ver.properties));
                                }
                            }
                        }
                        let mut v: Vec<BelievedVersion> = by_vf
                            .into_iter()
                            .map(|(valid_from, (_, properties))| BelievedVersion {
                                valid_from,
                                properties,
                            })
                            .collect();
                        v.sort_by_key(|b| b.valid_from);
                        v
                    }
                    Err(_) => continue,
                };

                for (w_idx, window) in windows.iter().enumerate() {
                    // Value sample: the value in effect at the window start --
                    // the change-point with the greatest `valid_from <= start`.
                    // `None` => the entity did not yet exist at the window start.
                    let sample = believed
                        .iter()
                        .rev()
                        .find(|b| b.valid_from <= window.start_micros);

                    for (a_idx, agg) in self.spec.aggregates.iter().enumerate() {
                        let acc = &mut accs[w_idx][a_idx];
                        match (agg.func, &agg.arg) {
                            (WindowAggFunc::Changes, arg) => {
                                acc.count += changes_in_window(&believed, window, arg);
                            }
                            (WindowAggFunc::Count, WindowAggArg::Star) => {
                                if sample.is_some() {
                                    acc.count += 1;
                                }
                            }
                            (WindowAggFunc::Count, WindowAggArg::Property(key)) => {
                                if let Some(v) = sample.and_then(|b| b.properties.get(key.as_str()))
                                    && property_as_f64(v).is_some()
                                {
                                    acc.count += 1;
                                }
                            }
                            // Value aggregates (SUM/AVG/MIN/MAX) over a property.
                            (_, WindowAggArg::Property(key)) => {
                                if let Some(v) = sample.and_then(|b| b.properties.get(key.as_str()))
                                {
                                    acc.fold_value(v);
                                }
                            }
                            // SUM/AVG/MIN/MAX with a `*` argument is rejected at
                            // convert time; nothing to fold if it slips through.
                            (_, WindowAggArg::Star) => {}
                        }
                    }
                }
            }
        }

        // Emit one row per window: window_start, window_end, then aggregates.
        let mut rows = Vec::with_capacity(windows.len());
        for (w_idx, window) in windows.iter().enumerate() {
            let mut columns: Vec<(String, PropertyValue)> = Vec::with_capacity(2 + naggs);
            columns.push((
                "window_start".to_string(),
                PropertyValue::String(Arc::from(
                    crate::query::temporal_window::format_rfc3339(window.start_micros).as_str(),
                )),
            ));
            columns.push((
                "window_end".to_string(),
                PropertyValue::String(Arc::from(
                    crate::query::temporal_window::format_rfc3339(window.end_micros).as_str(),
                )),
            ));
            let window_accs = std::mem::take(&mut accs[w_idx]);
            for (agg, acc) in self.spec.aggregates.iter().zip(window_accs) {
                columns.push((agg.output_name.clone(), acc.finalize()));
            }
            rows.push(QueryRow::from_columns(columns));
        }
        self.output = rows.into_iter();
        Ok(())
    }
}

/// Count the versions whose valid interval *starts* within `window`
/// (`[start, end)`), among the believed versions, for a `CHANGES` aggregate.
///
/// - `CHANGES(*)` / `CHANGES(v)` counts every entity version starting in the
///   window.
/// - `CHANGES(v.prop)` counts only versions whose value for `prop` differs from
///   the immediately-preceding believed version's value (a genuine property
///   change; the first believed version counts as a change from "absent").
fn changes_in_window(
    believed: &[BelievedVersion],
    window: &crate::query::temporal_window::Window,
    arg: &WindowAggArg,
) -> i64 {
    let in_window = |vf: i64| vf >= window.start_micros && vf < window.end_micros;
    match arg {
        WindowAggArg::Star => believed.iter().filter(|b| in_window(b.valid_from)).count() as i64,
        WindowAggArg::Property(key) => {
            let mut count = 0i64;
            for (i, b) in believed.iter().enumerate() {
                if !in_window(b.valid_from) {
                    continue;
                }
                let cur = b.properties.get(key.as_str());
                let prev = if i == 0 {
                    None
                } else {
                    believed[i - 1].properties.get(key.as_str())
                };
                if !property_values_equal(cur, prev) {
                    count += 1;
                }
            }
            count
        }
    }
}

/// Equality of two optional property values for property-change detection.
fn property_values_equal(a: Option<&PropertyValue>, b: Option<&PropertyValue>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

impl ResultIterator for TemporalWindowAggregateIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        if !self.drained {
            self.drained = true;
            if let Err(e) = self.drain() {
                return Some(Err(e));
            }
        }
        self.output.next().map(Ok)
    }
}

/// Temporal join / align iterator (Issue #3379).
///
/// On the first `next()` it drains the upstream matched-participant stream. For
/// each matched row it extracts each participant's node id (and any gating edge
/// id) per the spec, reconstructs each participant's believed valid-time
/// timeline (and gating-edge presence intervals) as of the spec's
/// transaction-time coordinate, and runs the storage-free
/// [`crate::query::temporal_join`] alignment for the requested mode. Each
/// resulting alignment coordinate becomes one computed-column [`QueryRow`]
/// carrying the coordinate (RFC 3339) followed by the resolved `RETURN` items.
pub struct TemporalJoinIterator {
    input: Option<Box<dyn ResultIterator>>,
    spec: TemporalAlignSpec,
    historical: Arc<RwLock<HistoricalStorage>>,
    output: std::vec::IntoIter<QueryRow>,
    drained: bool,
}

impl TemporalJoinIterator {
    /// Create a new temporal join iterator.
    pub fn new(
        input: Box<dyn ResultIterator>,
        spec: TemporalAlignSpec,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Self {
        TemporalJoinIterator {
            input: Some(input),
            spec,
            historical,
            output: Vec::new().into_iter(),
            drained: false,
        }
    }

    /// Extract the node id for a participant from a matched row, per its
    /// configured [`AlignNodeSource`].
    fn participant_node_id(row: &QueryRow, source: AlignNodeSource) -> Option<NodeId> {
        match source {
            AlignNodeSource::Entity => row.entity.node_id(),
            AlignNodeSource::PathIndex(i) => match row.path.as_ref()?.get(i)? {
                EntityId::Node(id) => Some(*id),
                EntityId::Edge(_) => None,
            },
        }
    }

    /// Extract the gating edge id for a participant from a matched row.
    fn gate_edge_id(row: &QueryRow, path_index: usize) -> Option<EdgeId> {
        match row.path.as_ref()?.get(path_index)? {
            EntityId::Edge(id) => Some(*id),
            EntityId::Node(_) => None,
        }
    }

    /// Reconstruct a node's believed valid-time timeline as of `as_of`: every
    /// version recorded by `as_of` (`transaction_from <= as_of`) contributes a
    /// change-point over its half-open valid interval `[valid_from, valid_to)`;
    /// same-`valid_from` corrections keep the latest belief (mirrors the #3363
    /// believed-version reconstruction). A retraction (#3230) / delete closes
    /// the covering version's `valid_to`, so the entity becomes **absent** at/
    /// after it rather than presumed present forever (Issue #3379 correctness,
    /// symmetric with the edge-gate path).
    ///
    /// A genuinely missing node (`NodeNotFound`) yields an **empty** timeline
    /// (a legitimate "absent" — the participant is simply null everywhere); any
    /// **other** storage error is a real failure and is propagated so it can
    /// surface as a structured `QueryError` rather than being silently read as
    /// "not found" (Issue #3379 review finding).
    fn node_timeline(
        hist: &HistoricalStorage,
        node_id: NodeId,
        as_of: Timestamp,
    ) -> Result<crate::query::temporal_join::Timeline> {
        use crate::core::error::{Error, StorageError};
        use crate::query::temporal_join::{ChangePoint, Timeline};
        let h = match hist.get_node_history(node_id) {
            Ok(h) => h,
            Err(Error::Storage(StorageError::NodeNotFound(_))) => {
                return Ok(Timeline::default());
            }
            Err(e) => return Err(e),
        };
        // valid_from -> (latest believed tx_from, properties, valid_to).
        let mut by_vf: std::collections::HashMap<
            i64,
            (Timestamp, crate::core::property::PropertyMap, i64),
        > = std::collections::HashMap::new();
        for ver in h.versions {
            let tx_from = ver.temporal.transaction_time().start();
            if tx_from > as_of {
                continue;
            }
            let vf = ver.temporal.valid_time().start().wallclock();
            let vt = ver.temporal.valid_time().end().wallclock();
            match by_vf.get(&vf) {
                Some((existing_tx, _, _)) if *existing_tx >= tx_from => {}
                _ => {
                    by_vf.insert(vf, (tx_from, ver.properties, vt));
                }
            }
        }
        let changes = by_vf
            .into_iter()
            .map(|(valid_from, (_, properties, valid_to))| ChangePoint {
                valid_from,
                valid_to,
                properties,
            })
            .collect();
        Ok(Timeline::from_changes(changes))
    }

    /// Reconstruct an edge's believed presence intervals as of `as_of`: each
    /// believed version contributes its half-open valid interval
    /// `[valid_from, valid_to)` (an open interval ends at `i64::MAX`).
    ///
    /// As with [`node_timeline`](Self::node_timeline), a missing edge
    /// (`EdgeNotFound`) yields **empty** presence intervals (a gate that is
    /// never open); any other storage error is propagated (Issue #3379 review
    /// finding — do not conflate a genuine error with not-found).
    fn edge_presence(
        hist: &HistoricalStorage,
        edge_id: EdgeId,
        as_of: Timestamp,
    ) -> Result<crate::query::temporal_join::PresenceIntervals> {
        use crate::core::error::{Error, StorageError};
        use crate::query::temporal_join::PresenceIntervals;
        let h = match hist.get_edge_history(edge_id) {
            Ok(h) => h,
            Err(Error::Storage(StorageError::EdgeNotFound(_))) => {
                return Ok(PresenceIntervals::default());
            }
            Err(e) => return Err(e),
        };
        let mut by_vf: std::collections::HashMap<i64, (Timestamp, i64)> =
            std::collections::HashMap::new();
        for ver in h.versions {
            let tx_from = ver.temporal.transaction_time().start();
            if tx_from > as_of {
                continue;
            }
            let vf = ver.temporal.valid_time().start().wallclock();
            let vt = ver.temporal.valid_time().end().wallclock();
            match by_vf.get(&vf) {
                Some((existing_tx, _)) if *existing_tx >= tx_from => {}
                _ => {
                    by_vf.insert(vf, (tx_from, vt));
                }
            }
        }
        let intervals = by_vf.into_iter().map(|(vf, (_, vt))| (vf, vt)).collect();
        Ok(PresenceIntervals { intervals })
    }

    fn drain(&mut self) -> Result<()> {
        use crate::query::temporal_join::{
            self, AlignCoordinate, AlignMode, Participant, PresenceIntervals,
        };

        let input = match self.input.take() {
            Some(i) => i,
            None => return Ok(()),
        };

        let as_of = self.spec.as_of_system_time;
        let from = self.spec.range_start_micros;
        let to = self.spec.range_end_micros;

        // Phase 1: fully drain the upstream matched-participant stream into a
        // Vec of per-row resolved (node id + gating edge id) tuples, THEN drop
        // the input iterator, all BEFORE acquiring the `historical` read guard.
        // Holding a parking_lot read guard across `input.next()` risks a
        // recursive-read deadlock (an upstream iterator may itself take the
        // same lock) and starves writers; mirror the #3363 sibling which drains
        // first (Issue #3379 review finding — BLOCKER). A row whose participant
        // node id cannot be located is skipped whole (defensive against a
        // null/optional binding; the pattern shape normally guarantees it).
        struct ResolvedParticipant {
            node_id: NodeId,
            gate_edge_id: Option<EdgeId>,
        }
        let mut resolved_rows: Vec<Vec<ResolvedParticipant>> = Vec::new();
        {
            let mut input = input;
            while let Some(row) = input.next() {
                let row = row?;
                let mut resolved: Vec<ResolvedParticipant> =
                    Vec::with_capacity(self.spec.participants.len());
                let mut incomplete = false;
                for p in &self.spec.participants {
                    let Some(nid) = Self::participant_node_id(&row, p.node_source) else {
                        incomplete = true;
                        break;
                    };
                    let gate_edge_id = p
                        .edge_gate_path_index
                        .and_then(|idx| Self::gate_edge_id(&row, idx));
                    resolved.push(ResolvedParticipant {
                        node_id: nid,
                        gate_edge_id,
                    });
                }
                if !incomplete {
                    resolved_rows.push(resolved);
                }
            }
        }

        // Phase 2: acquire the `historical` read guard in a scoped block and
        // reconstruct each participant's believed timeline / gating-edge
        // presence, run the storage-free alignment, and materialize rows. No
        // `input.next()` runs while this guard is held.
        let mut rows: Vec<QueryRow> = Vec::new();
        let hist = self.historical.read();
        for resolved in &resolved_rows {
            let mut node_ids: Vec<NodeId> = Vec::with_capacity(resolved.len());
            let mut participants: Vec<Participant> = Vec::with_capacity(resolved.len());
            for (p, r) in self.spec.participants.iter().zip(resolved) {
                node_ids.push(r.node_id);
                let timeline = Self::node_timeline(&hist, r.node_id, as_of)?;
                let gate: Option<PresenceIntervals> = match (p.edge_gate_path_index, r.gate_edge_id)
                {
                    (Some(_), Some(eid)) => Some(Self::edge_presence(&hist, eid, as_of)?),
                    _ => None,
                };
                participants.push(Participant { timeline, gate });
            }

            // Run the storage-free alignment for the requested mode.
            let aligned = match self.spec.mode {
                AlignMode::Events => {
                    temporal_join::align_events(&participants, self.spec.driver_index, from, to)
                }
                AlignMode::Overlap => temporal_join::align_overlap(&participants, from, to),
            }
            .map_err(temporal_join::align_error_to_query_error)?;

            // Materialize each aligned coordinate into a computed-column row.
            for arow in aligned {
                let mut columns: Vec<(String, PropertyValue)> = Vec::new();
                match arow.coordinate {
                    AlignCoordinate::Instant(t) => columns.push((
                        "align_valid_time".to_string(),
                        PropertyValue::String(Arc::from(
                            crate::query::temporal_window::format_rfc3339(t).as_str(),
                        )),
                    )),
                    AlignCoordinate::Interval { from: s, to: e } => {
                        columns.push((
                            "overlap_from".to_string(),
                            PropertyValue::String(Arc::from(
                                crate::query::temporal_window::format_rfc3339(s).as_str(),
                            )),
                        ));
                        columns.push((
                            "overlap_to".to_string(),
                            PropertyValue::String(Arc::from(
                                crate::query::temporal_window::format_rfc3339(e).as_str(),
                            )),
                        ));
                    }
                }

                for item in &self.spec.output_items {
                    let pidx = item.participant_index;
                    let value = match arow.sample_index.get(pidx).copied().flatten() {
                        None => PropertyValue::Null,
                        Some(sample_idx) => match &item.key {
                            None => {
                                // `RETURN v` (no key) -> the participant's node id.
                                PropertyValue::Int(node_ids[pidx].as_u64() as i64)
                            }
                            Some(key) => participants[pidx].timeline.changes[sample_idx]
                                .properties
                                .get(key.as_str())
                                .cloned()
                                .unwrap_or(PropertyValue::Null),
                        },
                    };
                    columns.push((item.output_name.clone(), value));
                }

                rows.push(QueryRow::from_columns(columns));
            }
        }
        drop(hist);

        self.output = rows.into_iter();
        Ok(())
    }
}

impl ResultIterator for TemporalJoinIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        if !self.drained {
            self.drained = true;
            if let Err(e) = self.drain() {
                return Some(Err(e));
            }
        }
        self.output.next().map(Ok)
    }
}

/// A whole-row deduplication key for `DISTINCT`.
#[derive(PartialEq, Eq, Hash)]
enum DistinctRowKey {
    /// Keyed by the row's entity identity (node/edge id, or `None` for a null
    /// binding).
    Entity(Option<EntityId>),
    /// A computed aggregate row, keyed by its rendered columns.
    Columns(String),
}

/// `RETURN DISTINCT` iterator: yields each distinct row once, preserving first
/// occurrence order.
pub struct DistinctIterator {
    input: Box<dyn ResultIterator>,
    seen: HashSet<DistinctRowKey>,
}

impl DistinctIterator {
    /// Create a new DISTINCT iterator over `input`.
    pub fn new(input: Box<dyn ResultIterator>) -> Self {
        DistinctIterator {
            input,
            seen: HashSet::new(),
        }
    }

    fn key(row: &QueryRow) -> DistinctRowKey {
        if let Some(cols) = &row.columns {
            DistinctRowKey::Columns(format!("{cols:?}"))
        } else {
            DistinctRowKey::Entity(row.entity.id())
        }
    }
}

impl ResultIterator for DistinctIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        loop {
            match self.input.next()? {
                Ok(row) => {
                    if self.seen.insert(Self::key(&row)) {
                        return Some(Ok(row));
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Resolve the write-time provenance bundle recorded on the version a row's
/// entity represents (node or edge), at the query's bi-temporal coordinate
/// (Issue #3354).
///
/// The entity's `current_version` already reflects an `AS OF` reconstruction
/// (mirroring [`FilterIterator::resolve_node_provenance`]), so provenance is
/// read as recorded at that coordinate. Returns `None` for an unattributed
/// version, a non-entity row, or a lookup error.
fn resolve_entity_provenance(
    entity: &EntityResult,
    historical: &Arc<RwLock<HistoricalStorage>>,
) -> Option<Provenance> {
    match entity {
        EntityResult::Node(n) => historical
            .read()
            .get_node_version_provenance(n.current_version)
            .ok()
            .flatten(),
        EntityResult::Edge(e) => historical
            .read()
            .get_edge_version_provenance(e.current_version)
            .ok()
            .flatten(),
        _ => None,
    }
}

/// Map a provenance field to its projected [`PropertyValue`] (Issue #3354).
///
/// An absent bundle, or a bundle missing the requested field, projects
/// [`PropertyValue::Null`] -- distinguishing "no value" from any real value,
/// exactly as the `WHERE` accessor semantics treat a missing field.
fn provenance_field_value(prov: Option<&Provenance>, field: ProvenanceField) -> PropertyValue {
    match field {
        ProvenanceField::Source => prov
            .and_then(Provenance::source)
            .map_or(PropertyValue::Null, |s| PropertyValue::String(s.into())),
        ProvenanceField::Confidence => prov
            .and_then(Provenance::confidence)
            .map_or(PropertyValue::Null, PropertyValue::Float),
        ProvenanceField::Reason => prov
            .and_then(Provenance::note)
            .map_or(PropertyValue::Null, |s| PropertyValue::String(s.into())),
    }
}

/// Provenance field value as an `Option`, collapsing [`PropertyValue::Null`]
/// (absent bundle or missing field) to `None` so a sort treats it as a null
/// (Issue #3354). Used by the decorate-sort-undecorate provenance ordering path.
fn provenance_field_value_opt(
    prov: Option<&Provenance>,
    field: ProvenanceField,
) -> Option<PropertyValue> {
    match provenance_field_value(prov, field) {
        PropertyValue::Null => None,
        v => Some(v),
    }
}

/// Look up a value for ordering by column/property name: first a node property
/// (entity rows), then a computed aggregate column (`row.columns`) so ORDER BY
/// can sort grouped/aggregate rows by a group key or aggregate alias.
fn row_value<'a>(row: &'a QueryRow, key: &str) -> Option<&'a PropertyValue> {
    if let Some(v) = row_property(row, key) {
        return Some(v);
    }
    row.columns
        .as_ref()
        .and_then(|cols| cols.iter().find(|(k, _)| k == key).map(|(_, v)| v))
}

/// Resolve a reserved *structural* edge field to a synthetic [`PropertyValue`]
/// for edge-property WHERE / ORDER BY evaluation (Issue #3622).
///
/// Returns `None` for any key that is not a reserved structural field, so a
/// genuine user property falls through to the edge's `properties` map. Resolving
/// these reserved names against `properties` (where they never live) would
/// silently yield null and mis-evaluate every predicate/sort key on them -- the
/// exact silent-wrong-result the pre-#3622 rejection guarded against.
///
/// - `type` / `label` -> the edge's interned type string (compared / sorted as a
///   `String`).
/// - `source` / `target` -> the endpoint `NodeId` as its integer id.
/// - `id` -> the edge's own `EdgeId` as its integer id.
pub(crate) fn edge_structural_value(
    edge: &crate::core::graph::Edge,
    key: &str,
) -> Option<PropertyValue> {
    match key {
        "type" | "label" => GLOBAL_INTERNER.resolve_with(edge.label, |s| PropertyValue::from(s)),
        "source" => Some(PropertyValue::Int(edge.source.as_u64() as i64)),
        "target" => Some(PropertyValue::Int(edge.target.as_u64() as i64)),
        "id" => Some(PropertyValue::Int(edge.id.as_u64() as i64)),
        _ => None,
    }
}

/// The id of the last edge in a traversal path (the edge that reached the
/// path's final node), for attaching the single-hop traversed edge to a row
/// (Issue #3622). `None` when the path is absent or contains no edge.
fn last_edge_in_path(path: &Option<Vec<EntityId>>) -> Option<crate::core::EdgeId> {
    path.as_ref()?.iter().rev().find_map(|e| match e {
        EntityId::Edge(id) => Some(*id),
        EntityId::Node(_) => None,
    })
}

/// Resolve a [`SortKey::EdgeProperty`] key from a row's traversed edge side
/// channel (Issue #3622): reserved structural fields (via
/// [`edge_structural_value`], shadowing user props) first, then the edge's own
/// properties. `None` when the row has no edge channel or the key is absent, so
/// the row sorts as a null value. Returns a [`Cow`] so a user property is
/// borrowed (no clone per comparison); only a synthesized structural value is
/// owned.
fn edge_sort_value<'a>(
    row: &'a QueryRow,
    key: &str,
) -> Option<std::borrow::Cow<'a, PropertyValue>> {
    use std::borrow::Cow;
    let edge = row.edge.as_ref()?;
    if let Some(v) = edge_structural_value(edge, key) {
        return Some(Cow::Owned(v));
    }
    edge.properties.get(key).map(Cow::Borrowed)
}

/// `ORDER BY` iterator: buffers the entire input and stably sorts it by one or
/// more keys in precedence order (first key primary), then streams the result.
///
/// Each key carries its own ascending/descending flag. Null placement follows
/// openCypher: a missing/null sort value orders **last** for an ascending key
/// and **first** for a descending key. Property keys fall back to computed
/// aggregate columns via [`row_value`], so grouped results can be ordered by a
/// group key or aggregate alias.
pub struct SortIterator {
    input: Option<Box<dyn ResultIterator>>,
    keys: Vec<(SortKey, bool)>,
    output: std::vec::IntoIter<QueryRow>,
    drained: bool,
    /// Historical storage, needed only to resolve a [`SortKey::Provenance`] key
    /// (Issue #3354) from each row entity's write-time provenance. `None` when
    /// no provenance sort key is present (the common case), so an ordinary
    /// property/score sort never touches historical storage.
    historical: Option<Arc<RwLock<HistoricalStorage>>>,
    /// When `true`, a [`SortKey::Property`] on an EDGE row reads the edge's own
    /// properties (real edge-property `ORDER BY`, Issue #3622) instead of
    /// resolving to null. Set only by the executor when the sort's input subtree
    /// is rooted at an `EdgeScan` (the SQL `FROM edges` lane). Defaults to
    /// `false`, preserving the node-only sort-key extraction for AQL/Cypher.
    edge_mode: bool,
}

impl SortIterator {
    /// Create a new ORDER BY iterator sorting by `keys` (first = primary).
    ///
    /// Carries no historical handle, so a [`SortKey::Provenance`] key would
    /// resolve to null for every row. Use [`with_historical`](Self::with_historical)
    /// when an `ORDER BY` may reference a provenance accessor.
    pub fn new(input: Box<dyn ResultIterator>, keys: Vec<(SortKey, bool)>) -> Self {
        SortIterator {
            input: Some(input),
            keys,
            output: Vec::new().into_iter(),
            drained: false,
            historical: None,
            edge_mode: false,
        }
    }

    /// Enable real edge-property ordering (Issue #3622): a [`SortKey::Property`]
    /// on an edge row reads the edge's own properties instead of resolving to
    /// null. The executor sets this only for a sort whose input stream is rooted
    /// at an `EdgeScan` (the SQL `FROM edges` lane). Node-row and aggregate
    /// column ordering are unchanged.
    #[must_use]
    pub fn order_by_edge_properties(mut self, enabled: bool) -> Self {
        self.edge_mode = enabled;
        self
    }

    /// Resolve the ordering value for a property key on a single row, honoring
    /// edge-property mode (Issue #3622): a node row (or aggregate column) uses
    /// the shared [`row_value`] lookup; an edge row in `edge_mode` reads its own
    /// properties, with the reserved structural fields
    /// (`type`/`label`/`source`/`target`/`id`) resolved against the edge struct
    /// (review fix) -- otherwise `ORDER BY type` would sort by a null key on
    /// every row. Structural values are synthesized owned, hence the [`Cow`].
    fn property_sort_value<'a>(
        &self,
        row: &'a QueryRow,
        prop: &str,
    ) -> Option<std::borrow::Cow<'a, PropertyValue>> {
        use std::borrow::Cow;
        if let Some(v) = row_value(row, prop) {
            return Some(Cow::Borrowed(v));
        }
        if self.edge_mode
            && let Some(edge) = row.entity.as_edge()
        {
            if let Some(v) = edge_structural_value(edge, prop) {
                return Some(Cow::Owned(v));
            }
            return edge.properties.get(prop).map(Cow::Borrowed);
        }
        None
    }

    /// Create a sort iterator with access to historical storage so a
    /// [`SortKey::Provenance`] key (Issue #3354) resolves each row entity's
    /// write-time provenance at the version the row represents.
    pub fn with_historical(
        input: Box<dyn ResultIterator>,
        keys: Vec<(SortKey, bool)>,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Self {
        SortIterator {
            input: Some(input),
            keys,
            output: Vec::new().into_iter(),
            drained: false,
            historical: Some(historical),
            edge_mode: false,
        }
    }

    fn drain(&mut self) -> Result<()> {
        let mut input = match self.input.take() {
            Some(i) => i,
            None => return Ok(()),
        };
        let (lower, _) = input.size_hint();
        // ⚡ Bolt Optimization: Pre-allocate vector based on iterator's lower size bound
        // to reduce heap allocations during collection of sorted results.
        let mut rows: Vec<QueryRow> = Vec::with_capacity(lower);
        while let Some(row) = input.next() {
            rows.push(row?);
        }

        // Decorate-sort-undecorate for provenance sort keys (Issue #3354): a
        // `SortKey::Provenance` resolution is a historical read under the shared
        // `historical` lock. Resolving it inside the comparator would cost
        // O(n log n) locked lookups; instead resolve each row's provenance
        // bundle EXACTLY ONCE (n lookups, a single lock span), sort against the
        // precomputed bundles, then strip. This mirrors the WHERE path's
        // per-row memoization. Null placement and stable ordering are identical
        // to the on-the-fly path. When no provenance key is present (the common
        // case) the ordinary comparator runs and historical is never touched.
        if self
            .keys
            .iter()
            .any(|(k, _)| matches!(k, SortKey::Provenance(_)))
        {
            let mut decorated: Vec<(Option<Provenance>, QueryRow)> = rows
                .into_iter()
                .map(|row| {
                    let prov = self
                        .historical
                        .as_ref()
                        .and_then(|h| resolve_entity_provenance(&row.entity, h));
                    (prov, row)
                })
                .collect();
            decorated.sort_by(|(a_prov, a), (b_prov, b)| {
                self.cmp_decorated(a_prov.as_ref(), a, b_prov.as_ref(), b)
            });
            self.output = decorated
                .into_iter()
                .map(|(_, row)| row)
                .collect::<Vec<_>>()
                .into_iter();
        } else {
            rows.sort_by(|a, b| self.cmp_rows(a, b));
            self.output = rows.into_iter();
        }
        Ok(())
    }

    fn cmp_rows(&self, a: &QueryRow, b: &QueryRow) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        for (key, descending) in &self.keys {
            let ord = self.cmp_by_key(a, b, key, *descending);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    }

    /// Compare two rows whose provenance bundles have already been resolved once
    /// (decorate-sort-undecorate, Issue #3354). A `SortKey::Provenance` key reads
    /// the precomputed bundle instead of re-resolving; every other key falls
    /// through to [`cmp_by_key`](Self::cmp_by_key). Ordering (including
    /// openCypher null placement) is identical to the on-the-fly comparator.
    fn cmp_decorated(
        &self,
        a_prov: Option<&Provenance>,
        a: &QueryRow,
        b_prov: Option<&Provenance>,
        b: &QueryRow,
    ) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        for (key, descending) in &self.keys {
            let ord = match key {
                SortKey::Provenance(field) => {
                    let av = provenance_field_value_opt(a_prov, *field);
                    let bv = provenance_field_value_opt(b_prov, *field);
                    Self::cmp_optional(av.as_ref(), bv.as_ref(), *descending)
                }
                other => self.cmp_by_key(a, b, other, *descending),
            };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    }

    /// Compare two nullable values with openCypher null placement (nulls last
    /// for ASC, first for DESC), applying `descending` to present-present pairs.
    fn cmp_optional(
        x: Option<&PropertyValue>,
        y: Option<&PropertyValue>,
        descending: bool,
    ) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (x, y) {
            (Some(x), Some(y)) => {
                let ord = compare_property(x, y);
                if descending { ord.reverse() } else { ord }
            }
            (Some(_), None) => {
                if descending {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            (None, Some(_)) => {
                if descending {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (None, None) => Ordering::Equal,
        }
    }

    /// Compare two rows by a single key, applying openCypher null placement
    /// (nulls last for ASC, first for DESC).
    fn cmp_by_key(
        &self,
        a: &QueryRow,
        b: &QueryRow,
        key: &SortKey,
        descending: bool,
    ) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match key {
            SortKey::Property(prop) => {
                let av = self.property_sort_value(a, prop);
                let bv = self.property_sort_value(b, prop);
                Self::cmp_optional(av.as_deref(), bv.as_deref(), descending)
            }
            // Edge-property ORDER BY (Issue #3622): resolve the key against the
            // row's traversed edge side channel -- reserved structural fields
            // (`type`/`label`/`source`/`target`/`id`) via `edge_structural_value`
            // (shadowing user props), otherwise `edge.properties`. A row with no
            // edge channel, or an absent key, sorts as null (openCypher null
            // placement).
            SortKey::EdgeProperty(prop) => {
                let av = edge_sort_value(a, prop);
                let bv = edge_sort_value(b, prop);
                Self::cmp_optional(av.as_deref(), bv.as_deref(), descending)
            }
            // Fallback on-the-fly resolution (Issue #3354). The hot path routes
            // provenance keys through `cmp_decorated` (resolve once per row), so
            // this arm is only a defensive fallback; it matches the WHERE-clause
            // resolution: an unattributed row (or a bundle missing the field)
            // sorts as null.
            SortKey::Provenance(field) => {
                let resolve = |row: &QueryRow| -> Option<PropertyValue> {
                    let historical = self.historical.as_ref()?;
                    let prov = resolve_entity_provenance(&row.entity, historical);
                    provenance_field_value_opt(prov.as_ref(), *field)
                };
                Self::cmp_optional(resolve(a).as_ref(), resolve(b).as_ref(), descending)
            }
            SortKey::Score => {
                let ord = a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal);
                if descending { ord.reverse() } else { ord }
            }
            SortKey::Timestamp => {
                let av = a.timestamp.map(|t| t.wallclock());
                let bv = b.timestamp.map(|t| t.wallclock());
                let ord = av.cmp(&bv);
                if descending { ord.reverse() } else { ord }
            }
        }
    }
}

impl ResultIterator for SortIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        if !self.drained {
            self.drained = true;
            if let Err(e) = self.drain() {
                return Some(Err(e));
            }
        }
        self.output.next().map(Ok)
    }
}

/// Provenance-projection iterator (Issue #3354): for each input row, resolves
/// the row entity's write-time provenance and attaches one output column per
/// projected accessor (`RETURN source(n)`, `RETURN n, confidence(n) AS conf`).
///
/// When the projection names a bare entity variable (`entity_binding`), the row
/// is emitted through the **bindings + columns** shape so the entity stays
/// observable alongside the projected columns (mirroring the multi-variable /
/// aggregation row shape, and keeping the entity from being dropped by a
/// columns-only consumer); otherwise a pure columns row is produced. The score,
/// path, and timestamp of the input row are preserved.
///
/// Runs last in the pipeline (after Sort/Skip/Limit), so it is 1:1 and never
/// perturbs ordering or pagination.
pub struct ProvenanceProjectIterator {
    input: Box<dyn ResultIterator>,
    projection: ProvenanceProjection,
    historical: Arc<RwLock<HistoricalStorage>>,
}

impl ProvenanceProjectIterator {
    /// Create a provenance-projection iterator resolving against `historical`.
    pub fn new(
        input: Box<dyn ResultIterator>,
        projection: ProvenanceProjection,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Self {
        ProvenanceProjectIterator {
            input,
            projection,
            historical,
        }
    }

    /// Build the projected columns for one row's entity, consuming the input
    /// row (it is 1:1 replaced by the projected row).
    fn project_row(&self, row: QueryRow) -> QueryRow {
        let prov = resolve_entity_provenance(&row.entity, &self.historical);
        let projected: Vec<(String, PropertyValue)> = self
            .projection
            .items
            .iter()
            .map(|item| {
                (
                    item.output_name.clone(),
                    provenance_field_value(prov.as_ref(), item.field),
                )
            })
            .collect();

        // Preserve any bare entity via the bindings channel so it survives
        // columns-only consumers (e.g. the MCP serializer); otherwise emit a
        // pure columns row like aggregation output.
        let bindings = self
            .projection
            .entity_binding
            .as_ref()
            .map(|var| vec![(var.clone(), row.entity.clone())]);

        // Defense-in-depth (Issue #3354 findings #5/#7): MERGE onto any
        // pre-existing columns rather than overwriting them. No upstream op
        // currently populates `columns` before this iterator, but merging keeps
        // the projection strictly additive if one ever does.
        let mut columns = row.columns.unwrap_or_default();
        columns.extend(projected);

        QueryRow {
            entity: EntityResult::Null,
            score: row.score,
            path: row.path,
            timestamp: row.timestamp,
            columns: Some(columns),
            bindings,
            edge: row.edge,
        }
    }
}

impl ResultIterator for ProvenanceProjectIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        match self.input.next()? {
            Ok(row) => Some(Ok(self.project_row(row))),
            Err(e) => Some(Err(e)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

/// `Count` aggregate iterator: drains the input and yields a single computed
/// row with the total row count under the column name `count`.
pub struct CountIterator {
    input: Option<Box<dyn ResultIterator>>,
    done: bool,
}

impl CountIterator {
    /// Create a new count iterator over `input`.
    pub fn new(input: Box<dyn ResultIterator>) -> Self {
        CountIterator {
            input: Some(input),
            done: false,
        }
    }
}

impl ResultIterator for CountIterator {
    fn next(&mut self) -> Option<Result<QueryRow>> {
        if self.done {
            return None;
        }
        self.done = true;
        let mut input = self.input.take()?;
        let mut count: i64 = 0;
        while let Some(row) = input.next() {
            if let Err(e) = row {
                return Some(Err(e));
            }
            count += 1;
        }
        Some(Ok(QueryRow::from_columns(vec![(
            "count".to_string(),
            PropertyValue::Int(count),
        )])))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::from(!self.done), Some(usize::from(!self.done)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::id::VersionId;
    use crate::core::interning::InternedString;
    use crate::core::property::PropertyMapBuilder;

    fn test_node(id: u64, name: &str) -> Node {
        let props = PropertyMapBuilder::new().insert("name", name).build();
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        Node::new(
            NodeId::new(id).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        )
    }

    fn test_node_with_age(id: u64, name: &str, age: i64) -> Node {
        let props = PropertyMapBuilder::new()
            .insert("name", name)
            .insert("age", age)
            .build();
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        Node::new(
            NodeId::new(id).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        )
    }

    fn test_node_with_vector(id: u64, name: &str, embedding: Vec<f32>) -> Node {
        let props = PropertyMapBuilder::new()
            .insert("name", name)
            .insert_vector("embedding", &embedding)
            .build();
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        Node::new(
            NodeId::new(id).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        )
    }

    /// Mock iterator for testing
    struct MockIterator {
        items: std::vec::IntoIter<Result<QueryRow>>,
    }

    impl MockIterator {
        fn from_nodes(nodes: Vec<Node>) -> Self {
            let items: Vec<Result<QueryRow>> = nodes
                .into_iter()
                .map(|n| Ok(QueryRow::from_entity(EntityResult::Node(n))))
                .collect();
            MockIterator {
                items: items.into_iter(),
            }
        }

        fn from_results(results: Vec<Result<QueryRow>>) -> Self {
            MockIterator {
                items: results.into_iter(),
            }
        }
    }

    impl ResultIterator for MockIterator {
        fn next(&mut self) -> Option<Result<QueryRow>> {
            self.items.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.items.size_hint()
        }
    }

    // ==================== ResourceGuardIterator (Issue #3368 engine lane) ==

    /// An inner iterator that panics if `next()` is ever called. Used to
    /// prove the wall-clock check runs BEFORE any pull from the wrapped
    /// iterator.
    struct PanicIfPulledIterator;

    impl ResultIterator for PanicIfPulledIterator {
        fn next(&mut self) -> Option<Result<QueryRow>> {
            panic!("inner iterator must not be pulled once the deadline has passed");
        }
    }

    fn resource_guard_test_rows(n: usize) -> Vec<Result<QueryRow>> {
        (0..n)
            .map(|i| {
                Ok(QueryRow::from_entity(EntityResult::NodeId(
                    NodeId::new(i as u64 + 1).unwrap(),
                )))
            })
            .collect()
    }

    #[test]
    fn resource_guard_unlimited_passes_all_rows_through_unchanged() {
        let inner = Box::new(MockIterator::from_results(resource_guard_test_rows(5)));
        let mut guard = ResourceGuardIterator::new(
            inner,
            crate::query::limits::QueryResourceLimits::unlimited(),
            None,
        );

        let mut count = 0;
        while let Some(row) = guard.next() {
            row.expect("unlimited guard must never error");
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn resource_guard_past_deadline_short_circuits_without_pulling_inner() {
        let inner = Box::new(PanicIfPulledIterator);
        let limits = crate::query::limits::QueryResourceLimits {
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_millis(10)),
            ..crate::query::limits::QueryResourceLimits::unlimited()
        };
        let mut guard = ResourceGuardIterator::new(inner, limits, None);

        let err = guard
            .next()
            .expect("must yield a result")
            .expect_err("deadline already passed");
        match err {
            crate::core::error::Error::Query(
                crate::core::error::QueryError::ResourceExhausted {
                    dimension,
                    retriable,
                    ..
                },
            ) => {
                assert_eq!(dimension, "wall_clock_timeout");
                assert!(retriable, "a timeout is safe to retry");
            }
            other => panic!("expected ResourceExhausted, got {other:?}"),
        }
        // Fused: once terminated, subsequent calls return None (and would
        // panic if they touched the inner PanicIfPulledIterator).
        assert!(guard.next().is_none());
    }

    #[test]
    fn resource_guard_row_cap_emits_exactly_the_cap_then_errors() {
        let inner = Box::new(MockIterator::from_results(resource_guard_test_rows(5)));
        let limits = crate::query::limits::QueryResourceLimits {
            max_rows: Some(2),
            ..crate::query::limits::QueryResourceLimits::unlimited()
        };
        let mut guard = ResourceGuardIterator::new(inner, limits, None);

        assert!(guard.next().expect("row 1").is_ok());
        assert!(guard.next().expect("row 2").is_ok());

        let err = guard
            .next()
            .expect("must yield a result")
            .expect_err("row 3 exceeds the cap of 2");
        match err {
            crate::core::error::Error::Query(
                crate::core::error::QueryError::ResourceExhausted {
                    dimension,
                    limit,
                    consumed,
                    retriable,
                },
            ) => {
                assert_eq!(dimension, "result_rows");
                assert_eq!(limit, 2);
                assert_eq!(consumed, 3);
                assert!(!retriable, "a row-cap breach is not retriable");
            }
            other => panic!("expected ResourceExhausted, got {other:?}"),
        }
        // Fused.
        assert!(guard.next().is_none());
    }

    #[test]
    fn resource_guard_memory_cap_terminates_once_budget_exceeded() {
        let inner = Box::new(MockIterator::from_results(resource_guard_test_rows(50)));
        // A tiny budget: the very first row's estimated size already exceeds
        // it (a bare NodeId row is small but non-zero), so the guard must
        // still emit at least the rows that fit before terminating.
        let first_row_bytes = crate::query::limits::estimate_row_bytes(&QueryRow::from_entity(
            EntityResult::NodeId(NodeId::new(1).unwrap()),
        ));
        let limits = crate::query::limits::QueryResourceLimits {
            max_memory_bytes: Some(first_row_bytes * 3),
            ..crate::query::limits::QueryResourceLimits::unlimited()
        };
        let mut guard = ResourceGuardIterator::new(inner, limits, None);

        let mut ok_count = 0;
        loop {
            match guard.next() {
                Some(Ok(_)) => ok_count += 1,
                Some(Err(crate::core::error::Error::Query(
                    crate::core::error::QueryError::ResourceExhausted {
                        dimension,
                        retriable,
                        ..
                    },
                ))) => {
                    assert_eq!(dimension, "memory_bytes");
                    assert!(!retriable, "a memory-cap breach is not retriable");
                    break;
                }
                Some(Err(other)) => panic!("unexpected error: {other:?}"),
                None => panic!("guard must terminate with an error before exhausting input"),
            }
        }
        // Terminated strictly before all 50 rows were consumed.
        assert!(ok_count < 50);
        // Fused.
        assert!(guard.next().is_none());
    }

    #[test]
    fn resource_guard_records_counters_on_termination() {
        let counters = Arc::new(crate::query::limits::LimitCounters::default());

        let inner = Box::new(MockIterator::from_results(resource_guard_test_rows(3)));
        let limits = crate::query::limits::QueryResourceLimits {
            max_rows: Some(1),
            ..crate::query::limits::QueryResourceLimits::unlimited()
        };
        let mut guard = ResourceGuardIterator::new(inner, limits, Some(counters.clone()));
        assert!(guard.next().expect("row 1").is_ok());
        assert!(guard.next().expect("row 2").is_err());

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.result_rows, 1);
        assert_eq!(snapshot.wall_clock_timeout, 0);
        assert_eq!(snapshot.memory_bytes, 0);
    }

    // ==================== Edge provenance-accessor tests (Issue #3354a) ====

    /// Build an edge whose `current_version` is `version`, plus register that
    /// version in `historical` with the given optional provenance bundle.
    /// Returns the edge for feeding into a [`FilterIterator`].
    fn edge_with_provenance(
        historical: &Arc<RwLock<HistoricalStorage>>,
        edge_id: u64,
        version: u64,
        provenance: Option<Provenance>,
    ) -> crate::core::graph::Edge {
        use crate::core::temporal::time;
        let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let src = NodeId::new(1).unwrap();
        let tgt = NodeId::new(2).unwrap();
        let props = PropertyMapBuilder::new().build();
        let version_id = VersionId::new(version).unwrap();
        let now = time::now();
        historical
            .write()
            .add_edge_version_with_provenance(
                crate::core::id::EdgeId::new(edge_id).unwrap(),
                version_id,
                now,
                now,
                label,
                src,
                tgt,
                props.clone(),
                false,
                provenance.map(std::sync::Arc::new),
            )
            .unwrap();
        crate::core::graph::Edge::new(
            crate::core::id::EdgeId::new(edge_id).unwrap(),
            label,
            src,
            tgt,
            props,
            version_id,
        )
    }

    /// Collect the ids of edge rows that survive `predicate` through a
    /// `FilterIterator` fed only edge rows (the pass-through / provenance-leaf
    /// path exercised by `evaluate_edge_predicate`).
    fn filter_edge_ids(
        historical: &Arc<RwLock<HistoricalStorage>>,
        edges: Vec<crate::core::graph::Edge>,
        predicate: Predicate,
    ) -> Vec<u64> {
        let rows: Vec<Result<QueryRow>> = edges
            .into_iter()
            .map(|e| Ok(QueryRow::from_entity(EntityResult::Edge(e))))
            .collect();
        let input = Box::new(MockIterator::from_results(rows));
        let mut filter = FilterIterator::with_historical(input, predicate, Arc::clone(historical));
        let mut ids = Vec::new();
        while let Some(row) = filter.next() {
            if let EntityResult::Edge(e) = row.unwrap().entity {
                ids.push(e.id.as_u64());
            }
        }
        ids.sort_unstable();
        ids
    }

    #[test]
    fn edge_confidence_accessor_filters_edge_rows() {
        // FIX 5 (Issue #3354a): a provenance accessor on EDGE rows filters by the
        // edge's own write-time provenance. `RETURN e` in AQL projects the
        // traversal node rather than the edge, so this path is covered here at
        // the FilterIterator level where edge rows exist.
        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let attributed = edge_with_provenance(
            &historical,
            1,
            10,
            Some(
                Provenance::builder()
                    .source("hr-system")
                    .confidence(0.95)
                    .build()
                    .unwrap(),
            ),
        );
        let unattributed = edge_with_provenance(&historical, 2, 11, None);

        let predicate = Predicate::Provenance(ProvenancePredicate::Accessor {
            field: crate::query::ir::ProvenanceField::Confidence,
            op: crate::query::ir::ProvenanceCmp::Ge,
            operand: crate::query::ir::ProvenanceOperand::Num(0.9),
        });
        let kept = filter_edge_ids(&historical, vec![attributed, unattributed], predicate);
        assert_eq!(kept, vec![1], "only the >=0.9-confidence edge survives");
    }

    #[test]
    fn edge_provenance_is_null_selects_unattributed_edges() {
        // FIX 5 (Issue #3354a): `provenance(e) IS NULL` selects the edge rows
        // with no recorded provenance bundle (mirroring the node semantics).
        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let attributed = edge_with_provenance(
            &historical,
            1,
            10,
            Some(Provenance::builder().source("hr-system").build().unwrap()),
        );
        let unattributed = edge_with_provenance(&historical, 2, 11, None);

        let predicate = Predicate::Provenance(ProvenancePredicate::IsNull { negated: false });
        let kept = filter_edge_ids(&historical, vec![attributed, unattributed], predicate);
        assert_eq!(kept, vec![2], "only the unattributed edge is IS NULL");
    }

    #[test]
    fn edge_non_provenance_leaf_is_pass_through() {
        // FIX 5 (Issue #3354a): in the single-entity pipeline a non-provenance
        // property leaf on an EDGE row is pass-through (evaluates true), so a
        // mixed `AND` filters an edge only on its provenance clause. Here the
        // property leaf `foo = 'x'` (never present on the edge) passes through,
        // and the confidence leaf decides the result.
        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let attributed = edge_with_provenance(
            &historical,
            1,
            10,
            Some(Provenance::builder().confidence(0.95).build().unwrap()),
        );
        let unattributed = edge_with_provenance(&historical, 2, 11, None);

        let predicate = Predicate::And(vec![
            Predicate::eq("foo", "x"),
            Predicate::Provenance(ProvenancePredicate::Accessor {
                field: crate::query::ir::ProvenanceField::Confidence,
                op: crate::query::ir::ProvenanceCmp::Ge,
                operand: crate::query::ir::ProvenanceOperand::Num(0.9),
            }),
        ]);
        let kept = filter_edge_ids(&historical, vec![attributed, unattributed], predicate);
        assert_eq!(
            kept,
            vec![1],
            "non-provenance leaf passes through; only the provenance clause filters edges"
        );
    }

    // ============ Edge-property WHERE / ORDER BY (Issue #3622) =============

    /// Build a bare current-state edge carrying a single integer `since`
    /// property. No historical version is registered (edge-property evaluation
    /// does not read historical storage), so this is cheap.
    fn edge_with_since(edge_id: u64, since: i64) -> crate::core::graph::Edge {
        let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let props = PropertyMapBuilder::new().insert("since", since).build();
        crate::core::graph::Edge::new(
            crate::core::id::EdgeId::new(edge_id).unwrap(),
            label,
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            props,
            VersionId::new(edge_id).unwrap(),
        )
    }

    /// Feed edge rows through a `FilterIterator` in edge-property mode
    /// (Issue #3622) and return the surviving edge ids (input order preserved).
    fn filter_edge_ids_edge_mode(
        edges: Vec<crate::core::graph::Edge>,
        predicate: Predicate,
    ) -> Vec<u64> {
        let rows: Vec<Result<QueryRow>> = edges
            .into_iter()
            .map(|e| Ok(QueryRow::from_entity(EntityResult::Edge(e))))
            .collect();
        let input = Box::new(MockIterator::from_results(rows));
        let mut filter = FilterIterator::new(input, predicate).evaluate_edge_properties(true);
        let mut ids = Vec::new();
        while let Some(row) = filter.next() {
            if let EntityResult::Edge(e) = row.unwrap().entity {
                ids.push(e.id.as_u64());
            }
        }
        ids
    }

    /// Sort edge rows by `keys` in edge-property mode (Issue #3622) and return
    /// the resulting edge ids in output order.
    fn sort_edge_ids_edge_mode(
        edges: Vec<crate::core::graph::Edge>,
        keys: Vec<(SortKey, bool)>,
    ) -> Vec<u64> {
        let rows: Vec<Result<QueryRow>> = edges
            .into_iter()
            .map(|e| Ok(QueryRow::from_entity(EntityResult::Edge(e))))
            .collect();
        let input = Box::new(MockIterator::from_results(rows));
        let mut sort = SortIterator::new(input, keys).order_by_edge_properties(true);
        let mut ids = Vec::new();
        while let Some(row) = sort.next() {
            if let EntityResult::Edge(e) = row.unwrap().entity {
                ids.push(e.id.as_u64());
            }
        }
        ids
    }

    #[test]
    fn edge_mode_filter_matches_edge_property_eq() {
        // In edge-property mode a bare `Eq` leaf is evaluated against the edge's
        // own properties: `since = 2021` keeps only edge 2.
        let edges = vec![edge_with_since(1, 2020), edge_with_since(2, 2021)];
        let predicate = Predicate::Eq {
            key: "since".to_string(),
            value: PredicateValue::Int(2021),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![2]);
    }

    #[test]
    fn edge_mode_filter_matches_edge_property_range() {
        // A comparison leaf also evaluates: `since > 2020` keeps only edge 2.
        let edges = vec![edge_with_since(1, 2020), edge_with_since(2, 2021)];
        let predicate = Predicate::Gt {
            key: "since".to_string(),
            value: PredicateValue::Int(2020),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![2]);
    }

    #[test]
    fn edge_mode_filter_absent_property_excludes_all() {
        // `Eq` on a property no edge carries excludes every edge (absent != any).
        let edges = vec![edge_with_since(1, 2020), edge_with_since(2, 2021)];
        let predicate = Predicate::Eq {
            key: "missing".to_string(),
            value: PredicateValue::Int(1),
        };
        assert_eq!(
            filter_edge_ids_edge_mode(edges, predicate),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn default_mode_filter_passes_edge_property_leaf_through() {
        // Regression guard for the AQL/Cypher contract: WITHOUT edge mode a bare
        // property leaf on an edge row still passes through (mirrors the pinned
        // `edge_non_provenance_leaf_is_pass_through`), so both edges survive.
        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let predicate = Predicate::Eq {
            key: "since".to_string(),
            value: PredicateValue::Int(2021),
        };
        let kept = filter_edge_ids(
            &historical,
            vec![edge_with_since(1, 2020), edge_with_since(2, 2021)],
            predicate,
        );
        assert_eq!(
            kept,
            vec![1, 2],
            "default (non-edge) mode passes property leaves through on edge rows"
        );
    }

    #[test]
    fn edge_mode_sort_orders_by_edge_property_desc() {
        // Edge rows are sorted by their own `since` property, descending.
        let edges = vec![
            edge_with_since(1, 2020),
            edge_with_since(2, 2022),
            edge_with_since(3, 2021),
        ];
        let keys = vec![(SortKey::Property("since".to_string()), true)];
        // 2022 (id 2), 2021 (id 3), 2020 (id 1).
        assert_eq!(sort_edge_ids_edge_mode(edges, keys), vec![2, 3, 1]);
    }

    #[test]
    fn edge_mode_sort_orders_by_edge_property_asc() {
        let edges = vec![
            edge_with_since(1, 2020),
            edge_with_since(2, 2022),
            edge_with_since(3, 2021),
        ];
        let keys = vec![(SortKey::Property("since".to_string()), false)];
        // 2020 (id 1), 2021 (id 3), 2022 (id 2).
        assert_eq!(sort_edge_ids_edge_mode(edges, keys), vec![1, 3, 2]);
    }

    #[test]
    fn edge_mode_sort_absent_property_places_nulls_last_asc() {
        // An edge missing the sort key sorts as null: last for ascending order.
        let edges = vec![
            edge_with_since(1, 2021),
            edge_with_no_props(2),
            edge_with_since(3, 2020),
        ];
        let keys = vec![(SortKey::Property("since".to_string()), false)];
        // 2020 (id 3), 2021 (id 1), then the null-key edge (id 2) last.
        assert_eq!(sort_edge_ids_edge_mode(edges, keys), vec![3, 1, 2]);
    }

    #[test]
    fn default_mode_sort_leaves_edge_property_unsorted() {
        // Regression guard: WITHOUT edge mode a property sort key on edge rows
        // resolves to null for every row, so input order is preserved (stable
        // sort) -- the pre-#3622 node-only behavior.
        let edges = vec![
            edge_with_since(3, 2020),
            edge_with_since(1, 2022),
            edge_with_since(2, 2021),
        ];
        let rows: Vec<Result<QueryRow>> = edges
            .into_iter()
            .map(|e| Ok(QueryRow::from_entity(EntityResult::Edge(e))))
            .collect();
        let input = Box::new(MockIterator::from_results(rows));
        let keys = vec![(SortKey::Property("since".to_string()), true)];
        let mut sort = SortIterator::new(input, keys); // edge_mode defaults false
        let mut ids = Vec::new();
        while let Some(row) = sort.next() {
            if let EntityResult::Edge(e) = row.unwrap().entity {
                ids.push(e.id.as_u64());
            }
        }
        assert_eq!(
            ids,
            vec![3, 1, 2],
            "default mode leaves edge rows in input order (property key is null)"
        );
    }

    /// A bare current-state edge with no properties (for null-placement tests).
    fn edge_with_no_props(edge_id: u64) -> crate::core::graph::Edge {
        let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        crate::core::graph::Edge::new(
            crate::core::id::EdgeId::new(edge_id).unwrap(),
            label,
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            PropertyMapBuilder::new().build(),
            VersionId::new(edge_id).unwrap(),
        )
    }

    /// Build a bare current-state edge with an explicit label, source, and
    /// target and no user properties -- for structural-field (`type`/`source`/
    /// `target`/`id`) WHERE / ORDER BY tests (Issue #3622 review fix).
    fn edge_typed(edge_id: u64, label: &str, source: u64, target: u64) -> crate::core::graph::Edge {
        let label = GLOBAL_INTERNER.intern(label).unwrap();
        crate::core::graph::Edge::new(
            crate::core::id::EdgeId::new(edge_id).unwrap(),
            label,
            NodeId::new(source).unwrap(),
            NodeId::new(target).unwrap(),
            PropertyMapBuilder::new().build(),
            VersionId::new(edge_id).unwrap(),
        )
    }

    /// Build a bare current-state edge carrying a single STRING property.
    fn edge_with_str(edge_id: u64, key: &str, value: &str) -> crate::core::graph::Edge {
        let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
        let props = PropertyMapBuilder::new().insert(key, value).build();
        crate::core::graph::Edge::new(
            crate::core::id::EdgeId::new(edge_id).unwrap(),
            label,
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            props,
            VersionId::new(edge_id).unwrap(),
        )
    }

    // ---- Reserved STRUCTURAL fields (type/label/source/target/id) -----------
    // These live on the Edge struct, NOT in `properties`; edge_mode must resolve
    // them there (Issue #3622 review fix) or every predicate/sort key on them is
    // silently wrong.

    #[test]
    fn edge_mode_filter_matches_edge_type() {
        // `WHERE type = 'KNOWS'` resolves the edge's LABEL, not a `type` property
        // (which no edge has). Only the KNOWS edge (id 1) survives.
        let edges = vec![
            edge_typed(1, "KNOWS", 10, 20),
            edge_typed(2, "FOLLOWS", 30, 40),
        ];
        let predicate = Predicate::Eq {
            key: "type".to_string(),
            value: PredicateValue::String("KNOWS".to_string()),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![1]);
    }

    #[test]
    fn edge_mode_filter_matches_edge_label_alias() {
        // `label` is an accepted alias for the edge type.
        let edges = vec![
            edge_typed(1, "KNOWS", 10, 20),
            edge_typed(2, "FOLLOWS", 30, 40),
        ];
        let predicate = Predicate::Eq {
            key: "label".to_string(),
            value: PredicateValue::String("FOLLOWS".to_string()),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![2]);
    }

    #[test]
    fn edge_mode_filter_matches_edge_source_and_target() {
        // `source`/`target` resolve the endpoint NodeIds as integers.
        let edges = vec![
            edge_typed(1, "KNOWS", 10, 20),
            edge_typed(2, "KNOWS", 11, 20),
            edge_typed(3, "KNOWS", 10, 21),
        ];
        let by_source = Predicate::Eq {
            key: "source".to_string(),
            value: PredicateValue::Int(10),
        };
        assert_eq!(
            filter_edge_ids_edge_mode(edges.clone(), by_source),
            vec![1, 3],
            "source = 10 keeps the two edges out of node 10"
        );
        let by_target = Predicate::Eq {
            key: "target".to_string(),
            value: PredicateValue::Int(20),
        };
        assert_eq!(
            filter_edge_ids_edge_mode(edges, by_target),
            vec![1, 2],
            "target = 20 keeps the two edges into node 20"
        );
    }

    #[test]
    fn edge_mode_filter_matches_edge_id() {
        // `id` resolves the edge's own EdgeId.
        let edges = vec![edge_typed(7, "KNOWS", 1, 2), edge_typed(9, "KNOWS", 1, 2)];
        let predicate = Predicate::Eq {
            key: "id".to_string(),
            value: PredicateValue::Int(9),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![9]);
    }

    #[test]
    fn edge_mode_filter_source_range() {
        // Comparison operators work on structural integer fields too.
        let edges = vec![
            edge_typed(1, "KNOWS", 5, 2),
            edge_typed(2, "KNOWS", 15, 2),
            edge_typed(3, "KNOWS", 25, 2),
        ];
        let predicate = Predicate::Gt {
            key: "source".to_string(),
            value: PredicateValue::Int(10),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![2, 3]);
    }

    #[test]
    fn edge_mode_sort_by_type_source_target() {
        // ORDER BY a structural field sorts by the resolved struct value, not a
        // null key. `type` sorts lexicographically; `source`/`target` numerically.
        let edges = vec![
            edge_typed(1, "KNOWS", 30, 5),
            edge_typed(2, "FOLLOWS", 10, 5),
            edge_typed(3, "LIKES", 20, 5),
        ];
        let by_type = vec![(SortKey::Property("type".to_string()), false)];
        // FOLLOWS < KNOWS < LIKES.
        assert_eq!(
            sort_edge_ids_edge_mode(edges.clone(), by_type),
            vec![2, 1, 3]
        );
        let by_source = vec![(SortKey::Property("source".to_string()), false)];
        // 10 (id 2) < 20 (id 3) < 30 (id 1).
        assert_eq!(sort_edge_ids_edge_mode(edges, by_source), vec![2, 3, 1]);
    }

    // ---- Recursive combinators (AND / OR / NOT) in edge_mode ----------------

    #[test]
    fn edge_mode_filter_and_combinator() {
        // `since > 2019 AND since < 2022` keeps only the middle edge (2021).
        let edges = vec![
            edge_with_since(1, 2018),
            edge_with_since(2, 2021),
            edge_with_since(3, 2023),
        ];
        let predicate = Predicate::And(vec![
            Predicate::Gt {
                key: "since".to_string(),
                value: PredicateValue::Int(2019),
            },
            Predicate::Lt {
                key: "since".to_string(),
                value: PredicateValue::Int(2022),
            },
        ]);
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![2]);
    }

    #[test]
    fn edge_mode_filter_or_combinator() {
        // `since = 2018 OR since = 2023` keeps the two extremes.
        let edges = vec![
            edge_with_since(1, 2018),
            edge_with_since(2, 2021),
            edge_with_since(3, 2023),
        ];
        let predicate = Predicate::Or(vec![
            Predicate::Eq {
                key: "since".to_string(),
                value: PredicateValue::Int(2018),
            },
            Predicate::Eq {
                key: "since".to_string(),
                value: PredicateValue::Int(2023),
            },
        ]);
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![1, 3]);
    }

    #[test]
    fn edge_mode_filter_not_combinator() {
        // `NOT (since = 2021)` drops only the 2021 edge.
        let edges = vec![
            edge_with_since(1, 2018),
            edge_with_since(2, 2021),
            edge_with_since(3, 2023),
        ];
        let predicate = Predicate::Not(Box::new(Predicate::Eq {
            key: "since".to_string(),
            value: PredicateValue::Int(2021),
        }));
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![1, 3]);
    }

    #[test]
    fn edge_mode_filter_mixes_structural_and_property() {
        // A combinator mixing a structural field and a user property.
        let mut e1 = edge_typed(1, "KNOWS", 10, 2);
        e1.properties = PropertyMapBuilder::new().insert("since", 2020).build();
        let mut e2 = edge_typed(2, "FOLLOWS", 10, 2);
        e2.properties = PropertyMapBuilder::new().insert("since", 2020).build();
        let predicate = Predicate::And(vec![
            Predicate::Eq {
                key: "type".to_string(),
                value: PredicateValue::String("KNOWS".to_string()),
            },
            Predicate::Eq {
                key: "since".to_string(),
                value: PredicateValue::Int(2020),
            },
        ]);
        assert_eq!(filter_edge_ids_edge_mode(vec![e1, e2], predicate), vec![1]);
    }

    // ---- String edge properties: equality, LIKE family, ordering ------------

    #[test]
    fn edge_mode_filter_string_equality() {
        let edges = vec![
            edge_with_str(1, "role", "admin"),
            edge_with_str(2, "role", "member"),
        ];
        let predicate = Predicate::Eq {
            key: "role".to_string(),
            value: PredicateValue::String("member".to_string()),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges, predicate), vec![2]);
    }

    #[test]
    fn edge_mode_filter_string_like_family() {
        let edges = vec![
            edge_with_str(1, "role", "administrator"),
            edge_with_str(2, "role", "moderator"),
            edge_with_str(3, "role", "member"),
        ];
        let contains = Predicate::Contains {
            key: "role".to_string(),
            substring: "era".to_string(),
        };
        assert_eq!(
            filter_edge_ids_edge_mode(edges.clone(), contains),
            vec![2],
            "only 'moderator' contains 'era'"
        );
        let starts = Predicate::StartsWith {
            key: "role".to_string(),
            prefix: "m".to_string(),
        };
        assert_eq!(
            filter_edge_ids_edge_mode(edges.clone(), starts),
            vec![2, 3],
            "'moderator' and 'member' start with 'm'"
        );
        let ends = Predicate::EndsWith {
            key: "role".to_string(),
            suffix: "ator".to_string(),
        };
        assert_eq!(
            filter_edge_ids_edge_mode(edges, ends),
            vec![1, 2],
            "'administrator' and 'moderator' end with 'ator'"
        );
    }

    #[test]
    fn edge_mode_sort_string_lexicographic() {
        let edges = vec![
            edge_with_str(1, "role", "member"),
            edge_with_str(2, "role", "admin"),
            edge_with_str(3, "role", "owner"),
        ];
        let keys = vec![(SortKey::Property("role".to_string()), false)];
        // admin (2) < member (1) < owner (3).
        assert_eq!(sort_edge_ids_edge_mode(edges, keys), vec![2, 1, 3]);
    }

    // ---- Null placement (DESC) ---------------------------------------------

    #[test]
    fn edge_mode_sort_absent_property_places_nulls_first_desc() {
        // DESC: a missing sort key (null) orders FIRST (complements the ASC
        // nulls-last case).
        let edges = vec![
            edge_with_since(1, 2021),
            edge_with_no_props(2),
            edge_with_since(3, 2020),
        ];
        let keys = vec![(SortKey::Property("since".to_string()), true)];
        // null edge (id 2) first, then 2021 (id 1), then 2020 (id 3).
        assert_eq!(sort_edge_ids_edge_mode(edges, keys), vec![2, 1, 3]);
    }

    // ---- Multi-key ORDER BY + tie stability ---------------------------------

    #[test]
    fn edge_mode_sort_multi_key_with_tie_stability() {
        // Order by `since` ASC, then `id`... but `id` here is a genuine property
        // to keep the test independent of structural resolution; ties on `since`
        // retain input order (stable sort).
        let mk = |edge_id: u64, since: i64| {
            let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
            let props = PropertyMapBuilder::new()
                .insert("since", since)
                .insert("rank", (edge_id as i64) % 2) // 1,0,1,0 -> secondary key
                .build();
            crate::core::graph::Edge::new(
                crate::core::id::EdgeId::new(edge_id).unwrap(),
                label,
                NodeId::new(1).unwrap(),
                NodeId::new(2).unwrap(),
                props,
                VersionId::new(edge_id).unwrap(),
            )
        };
        // Two edges share since=2020; primary sorts them together, secondary
        // `rank` ASC orders rank 0 before rank 1.
        let edges = vec![mk(1, 2020), mk(2, 2020), mk(3, 2019)];
        let keys = vec![
            (SortKey::Property("since".to_string()), false),
            (SortKey::Property("rank".to_string()), false),
        ];
        // since: 2019 (id3) first; then the 2020 pair ordered by rank: id2(rank0),id1(rank1).
        assert_eq!(sort_edge_ids_edge_mode(edges, keys), vec![3, 2, 1]);
    }

    #[test]
    fn edge_mode_sort_ties_retain_input_order() {
        // All equal keys -> stable sort preserves input order.
        let edges = vec![
            edge_with_since(5, 2020),
            edge_with_since(3, 2020),
            edge_with_since(8, 2020),
        ];
        let keys = vec![(SortKey::Property("since".to_string()), false)];
        assert_eq!(sort_edge_ids_edge_mode(edges, keys), vec![5, 3, 8]);
    }

    // ---- Operator coverage: Ne / Gte / Lte / Lt / Exists --------------------

    #[test]
    fn edge_mode_filter_ne_gte_lte_lt() {
        let edges = vec![
            edge_with_since(1, 2019),
            edge_with_since(2, 2020),
            edge_with_since(3, 2021),
        ];
        let ne = Predicate::Ne {
            key: "since".to_string(),
            value: PredicateValue::Int(2020),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges.clone(), ne), vec![1, 3]);
        let gte = Predicate::Gte {
            key: "since".to_string(),
            value: PredicateValue::Int(2020),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges.clone(), gte), vec![2, 3]);
        let lte = Predicate::Lte {
            key: "since".to_string(),
            value: PredicateValue::Int(2020),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges.clone(), lte), vec![1, 2]);
        let lt = Predicate::Lt {
            key: "since".to_string(),
            value: PredicateValue::Int(2020),
        };
        assert_eq!(filter_edge_ids_edge_mode(edges, lt), vec![1]);
    }

    #[test]
    fn edge_mode_filter_exists() {
        let edges = vec![edge_with_since(1, 2020), edge_with_no_props(2)];
        let predicate = Predicate::Exists("since".to_string());
        assert_eq!(
            filter_edge_ids_edge_mode(edges, predicate),
            vec![1],
            "only the edge carrying `since` exists"
        );
    }

    #[test]
    fn edge_mode_filter_ne_on_missing_property_returns_row() {
        // openCypher three-valued asymmetry: `Ne` on an ABSENT edge property
        // returns the row (unlike `Eq`, which excludes it). Edge WHERE follows
        // openCypher/node semantics here, NOT SQL three-valued UNKNOWN.
        let edges = vec![edge_with_since(1, 2020), edge_with_no_props(2)];
        let predicate = Predicate::Ne {
            key: "missing".to_string(),
            value: PredicateValue::Int(1),
        };
        assert_eq!(
            filter_edge_ids_edge_mode(edges, predicate),
            vec![1, 2],
            "Ne on a property no edge carries returns every edge (present != absent)"
        );
    }

    // ---- Empty edge stream through edge_mode Filter and Sort ----------------

    #[test]
    fn edge_mode_filter_empty_stream_is_empty() {
        let predicate = Predicate::Eq {
            key: "since".to_string(),
            value: PredicateValue::Int(2020),
        };
        assert_eq!(
            filter_edge_ids_edge_mode(Vec::new(), predicate),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn edge_mode_sort_empty_stream_is_empty() {
        let keys = vec![(SortKey::Property("since".to_string()), false)];
        assert_eq!(sort_edge_ids_edge_mode(Vec::new(), keys), Vec::<u64>::new());
    }

    // ==================== EmptyIterator Tests ====================

    #[test]
    fn test_empty_iterator() {
        let mut iter = EmptyIterator;
        assert!(iter.next().is_none());
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_empty_iterator_multiple_calls() {
        let mut iter = EmptyIterator;
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    // ==================== FilterIterator Predicate Tests ====================

    #[test]
    fn test_filter_predicate_eq() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::eq("name", "Alice");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_eq_false() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::eq("name", "Bob");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_eq_missing_property() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::eq("missing", "value");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_ne() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::ne("name", "Bob");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_ne_same_value() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::ne("name", "Alice");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_ne_missing_property() {
        let node = test_node(1, "Alice");
        // Missing property != anything is true
        let predicate = Predicate::ne("missing", "value");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_gt() {
        let node = test_node_with_age(1, "Alice", 30);
        let predicate = Predicate::gt("age", 18i64);

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_gt_equal_value() {
        let node = test_node_with_age(1, "Alice", 18);
        let predicate = Predicate::gt("age", 18i64);

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_gt_less_value() {
        let node = test_node_with_age(1, "Alice", 15);
        let predicate = Predicate::gt("age", 18i64);

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_lt() {
        let node = test_node_with_age(1, "Alice", 15);
        let predicate = Predicate::lt("age", 18i64);

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_lt_equal_value() {
        let node = test_node_with_age(1, "Alice", 18);
        let predicate = Predicate::lt("age", 18i64);

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_gte() {
        let node = test_node_with_age(1, "Alice", 18);
        let predicate = Predicate::Gte {
            key: "age".to_string(),
            value: PredicateValue::Int(18),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_gte_greater() {
        let node = test_node_with_age(1, "Alice", 20);
        let predicate = Predicate::Gte {
            key: "age".to_string(),
            value: PredicateValue::Int(18),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_gte_less() {
        let node = test_node_with_age(1, "Alice", 15);
        let predicate = Predicate::Gte {
            key: "age".to_string(),
            value: PredicateValue::Int(18),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_lte() {
        let node = test_node_with_age(1, "Alice", 18);
        let predicate = Predicate::Lte {
            key: "age".to_string(),
            value: PredicateValue::Int(18),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_lte_less() {
        let node = test_node_with_age(1, "Alice", 15);
        let predicate = Predicate::Lte {
            key: "age".to_string(),
            value: PredicateValue::Int(18),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_lte_greater() {
        let node = test_node_with_age(1, "Alice", 20);
        let predicate = Predicate::Lte {
            key: "age".to_string(),
            value: PredicateValue::Int(18),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_exists() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::exists("name");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_exists_missing() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::exists("missing");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_not_exists() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::NotExists("missing".to_string());

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_not_exists_present() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::NotExists("name".to_string());

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_contains() {
        let node = test_node(1, "Alice Johnson");
        let predicate = Predicate::contains("name", "John");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_contains_not_found() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::contains("name", "Bob");

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_starts_with() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::StartsWith {
            key: "name".to_string(),
            prefix: "Ali".to_string(),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_starts_with_not_match() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::StartsWith {
            key: "name".to_string(),
            prefix: "Bob".to_string(),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_ends_with() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::EndsWith {
            key: "name".to_string(),
            suffix: "ice".to_string(),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_ends_with_not_match() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::EndsWith {
            key: "name".to_string(),
            suffix: "Bob".to_string(),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_in() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::In {
            key: "name".to_string(),
            values: vec![
                PredicateValue::String("Alice".to_string()),
                PredicateValue::String("Bob".to_string()),
                PredicateValue::String("Charlie".to_string()),
            ],
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_in_not_found() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::In {
            key: "name".to_string(),
            values: vec![
                PredicateValue::String("Bob".to_string()),
                PredicateValue::String("Charlie".to_string()),
            ],
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_and() {
        let props = PropertyMapBuilder::new()
            .insert("name", "Alice")
            .insert("age", 30i64)
            .build();

        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        let node = Node::new(
            NodeId::new(1).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        );

        let predicate = Predicate::eq("name", "Alice").and(Predicate::gt("age", 18i64));

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_and_one_false() {
        let node = test_node_with_age(1, "Alice", 15);
        let predicate = Predicate::eq("name", "Alice").and(Predicate::gt("age", 18i64));

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_or() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::eq("name", "Alice").or(Predicate::eq("name", "Bob"));

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_or_second_true() {
        let node = test_node(1, "Bob");
        let predicate = Predicate::eq("name", "Alice").or(Predicate::eq("name", "Bob"));

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_or_both_false() {
        let node = test_node(1, "Charlie");
        let predicate = Predicate::eq("name", "Alice").or(Predicate::eq("name", "Bob"));

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_not() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::Not(Box::new(Predicate::eq("name", "Bob")));

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_not_negates_true() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::Not(Box::new(Predicate::eq("name", "Alice")));

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_true() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::True;

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_false() {
        let node = test_node(1, "Alice");
        let predicate = Predicate::False;

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_float_comparison() {
        let props = PropertyMapBuilder::new().insert("score", 3.5f64).build();
        let label = GLOBAL_INTERNER.intern("Score").unwrap();
        let node = Node::new(
            NodeId::new(1).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        );

        let predicate = Predicate::gt("score", 3.0f64);
        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));

        let predicate = Predicate::lt("score", 4.0f64);
        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_predicate_bool_comparison() {
        let props = PropertyMapBuilder::new().insert("active", true).build();
        let label = GLOBAL_INTERNER.intern("Status").unwrap();
        let node = Node::new(
            NodeId::new(1).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        );

        let predicate = Predicate::eq("active", true);
        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));

        let predicate = Predicate::eq("active", false);
        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    // ==================== FilterIterator Integration Tests ====================

    #[test]
    fn test_filter_iterator_passes_matching_nodes() {
        let nodes = vec![
            test_node_with_age(1, "Alice", 30),
            test_node_with_age(2, "Bob", 25),
            test_node_with_age(3, "Charlie", 35),
        ];

        let input = MockIterator::from_nodes(nodes);
        let predicate = Predicate::gt("age", 28i64);
        let mut filter = FilterIterator::new(Box::new(input), predicate);

        let mut results = Vec::new();
        while let Some(Ok(row)) = filter.next() {
            results.push(row);
        }

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entity.node_id(), Some(NodeId::new(1).unwrap())); // Alice (30)
        assert_eq!(results[1].entity.node_id(), Some(NodeId::new(3).unwrap())); // Charlie (35)
    }

    #[test]
    fn test_filter_iterator_no_matches() {
        let nodes = vec![
            test_node_with_age(1, "Alice", 20),
            test_node_with_age(2, "Bob", 25),
        ];

        let input = MockIterator::from_nodes(nodes);
        let predicate = Predicate::gt("age", 100i64);
        let mut filter = FilterIterator::new(Box::new(input), predicate);

        assert!(filter.next().is_none());
    }

    #[test]
    fn test_filter_iterator_propagates_errors() {
        let results = vec![
            Ok(QueryRow::from_entity(EntityResult::Node(test_node(
                1, "Alice",
            )))),
            Err(crate::core::error::Error::other("test error")),
        ];

        let input = MockIterator::from_results(results);
        let predicate = Predicate::True;
        let mut filter = FilterIterator::new(Box::new(input), predicate);

        // First result succeeds
        assert!(filter.next().unwrap().is_ok());
        // Second result is error
        assert!(filter.next().unwrap().is_err());
    }

    // ==================== LimitIterator Tests ====================

    #[test]
    fn test_limit_iterator() {
        let test_label = GLOBAL_INTERNER.intern("Test").unwrap();

        struct CountingIterator {
            count: usize,
            max: usize,
            label: InternedString,
        }

        impl ResultIterator for CountingIterator {
            fn next(&mut self) -> Option<Result<QueryRow>> {
                if self.count < self.max {
                    self.count += 1;
                    let node = Node::new(
                        NodeId::new(self.count as u64).unwrap(),
                        self.label,
                        PropertyMapBuilder::new().build(),
                        VersionId::new(1).unwrap(),
                    );
                    Some(Ok(QueryRow::from_entity(EntityResult::Node(node))))
                } else {
                    None
                }
            }
        }

        let input = Box::new(CountingIterator {
            count: 0,
            max: 10,
            label: test_label,
        });
        let mut limit = LimitIterator::new(input, 2, 3);

        // Should skip 2, return 3
        let mut results = Vec::new();
        while let Some(Ok(row)) = limit.next() {
            results.push(row);
        }

        assert_eq!(results.len(), 3);
        // First result should be node 3 (after skipping 2)
        assert_eq!(results[0].entity.node_id(), Some(NodeId::new(3).unwrap()));
    }

    #[test]
    fn test_limit_iterator_no_offset() {
        let nodes = vec![
            test_node(1, "Alice"),
            test_node(2, "Bob"),
            test_node(3, "Charlie"),
            test_node(4, "Dave"),
        ];

        let input = MockIterator::from_nodes(nodes);
        let mut limit = LimitIterator::new(Box::new(input), 0, 2);

        let mut results = Vec::new();
        while let Some(Ok(row)) = limit.next() {
            results.push(row);
        }

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entity.node_id(), Some(NodeId::new(1).unwrap()));
        assert_eq!(results[1].entity.node_id(), Some(NodeId::new(2).unwrap()));
    }

    #[test]
    fn test_limit_iterator_offset_exceeds_input() {
        let nodes = vec![test_node(1, "Alice"), test_node(2, "Bob")];

        let input = MockIterator::from_nodes(nodes);
        let mut limit = LimitIterator::new(Box::new(input), 5, 10);

        // Offset exceeds input, should return nothing
        assert!(limit.next().is_none());
    }

    #[test]
    fn test_limit_iterator_count_zero() {
        let nodes = vec![test_node(1, "Alice"), test_node(2, "Bob")];

        let input = MockIterator::from_nodes(nodes);
        let mut limit = LimitIterator::new(Box::new(input), 0, 0);

        // Count is 0, should return nothing
        assert!(limit.next().is_none());
    }

    #[test]
    fn test_limit_iterator_count_exceeds_remaining() {
        let nodes = vec![test_node(1, "Alice"), test_node(2, "Bob")];

        let input = MockIterator::from_nodes(nodes);
        let mut limit = LimitIterator::new(Box::new(input), 1, 10);

        let mut results = Vec::new();
        while let Some(Ok(row)) = limit.next() {
            results.push(row);
        }

        // Skipped 1, only 1 remaining
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity.node_id(), Some(NodeId::new(2).unwrap()));
    }

    #[test]
    fn test_limit_iterator_propagates_errors_during_skip() {
        let results = vec![
            Err(crate::core::error::Error::other("test error")),
            Ok(QueryRow::from_entity(EntityResult::Node(test_node(
                1, "Alice",
            )))),
        ];

        let input = MockIterator::from_results(results);
        let mut limit = LimitIterator::new(Box::new(input), 1, 5);

        // Should get error during skip phase
        let result = limit.next();
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn test_limit_iterator_size_hint() {
        let nodes = vec![
            test_node(1, "Alice"),
            test_node(2, "Bob"),
            test_node(3, "Charlie"),
        ];

        let input = MockIterator::from_nodes(nodes);
        let limit = LimitIterator::new(Box::new(input), 0, 2);

        // Size hint should respect the limit
        let (lower, upper) = limit.size_hint();
        assert!(lower <= 2);
        assert!(upper.map(|u| u <= 2).unwrap_or(true));
    }

    // ==================== VectorRerankIterator Tests ====================

    #[test]
    fn test_vector_rerank_no_vector_index_error() {
        let nodes = vec![test_node_with_vector(1, "Alice", vec![1.0, 0.0, 0.0, 0.0])];

        // Create CurrentStorage without vector index
        let current = Arc::new(CurrentStorage::new());

        let input = MockIterator::from_nodes(nodes);
        let query = Arc::from(vec![1.0f32, 0.0, 0.0, 0.0]);

        let mut rerank = VectorRerankIterator::new(Box::new(input), query, 10, current, None);

        // Should return error because no vector index is configured
        let result = rerank.next();
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn test_vector_rerank_size_hint_before_init() {
        let nodes = vec![test_node_with_vector(1, "Alice", vec![1.0, 0.0, 0.0, 0.0])];

        let current = Arc::new(CurrentStorage::new());
        let input = MockIterator::from_nodes(nodes);
        let query = Arc::from(vec![1.0f32, 0.0, 0.0, 0.0]);

        let rerank = VectorRerankIterator::new(Box::new(input), query, 5, current, None);

        // Before initialization, size_hint upper bound is k
        let (lower, upper) = rerank.size_hint();
        assert_eq!(lower, 0);
        assert_eq!(upper, Some(5));
    }

    // ==================== ProjectIterator Tests ====================

    #[test]
    fn test_project_iterator_filters_properties() {
        let props = PropertyMapBuilder::new()
            .insert("name", "Alice")
            .insert("age", 30)
            .insert("city", "Paris")
            .build();
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        let node = Node::new(
            NodeId::new(1).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        );

        let input = MockIterator::from_nodes(vec![node]);
        let mut project = ProjectIterator::new(
            Box::new(input),
            vec!["name".to_string(), "city".to_string()],
        );

        let row = project.next().unwrap().unwrap();
        let projected_node = row.entity.as_node().unwrap();

        assert_eq!(
            projected_node
                .properties
                .get("name")
                .unwrap()
                .as_str()
                .unwrap(),
            "Alice"
        );
        assert_eq!(
            projected_node
                .properties
                .get("city")
                .unwrap()
                .as_str()
                .unwrap(),
            "Paris"
        );
        assert!(projected_node.properties.get("age").is_none());
    }

    #[test]
    fn test_project_iterator_missing_property() {
        let node = test_node(1, "Alice"); // Only has "name"
        let input = MockIterator::from_nodes(vec![node]);
        let mut project =
            ProjectIterator::new(Box::new(input), vec!["name".to_string(), "age".to_string()]);

        let row = project.next().unwrap().unwrap();
        let projected_node = row.entity.as_node().unwrap();

        assert_eq!(
            projected_node
                .properties
                .get("name")
                .unwrap()
                .as_str()
                .unwrap(),
            "Alice"
        );
        assert!(projected_node.properties.get("age").is_none());
    }

    #[test]
    fn test_project_iterator_non_node_pass_through() {
        // Projecting on non-node entities (like EdgeId) should be a no-op currently
        // as the implementation only checks for Node
        let row = QueryRow::from_entity(EntityResult::NodeId(NodeId::new(1).unwrap()));
        let input = MockIterator::from_results(vec![Ok(row)]);

        let mut project = ProjectIterator::new(Box::new(input), vec!["name".to_string()]);

        let result = project.next().unwrap().unwrap();
        assert!(matches!(result.entity, EntityResult::NodeId(_)));
    }

    // ==================== MockIterator Tests ====================

    #[test]
    fn test_project_iterator_error_passthrough() {
        // ProjectIterator should pass through errors from the underlying iterator
        let err_row = Err(crate::core::error::Error::Storage(
            crate::core::error::StorageError::CorruptedData("test".to_string()),
        ));
        let mock_iter = MockIterator::from_results(vec![err_row]);

        let mut project_iter = ProjectIterator::new(Box::new(mock_iter), vec!["deep".to_string()]);

        let res = project_iter.next().unwrap();
        assert!(res.is_err());
    }

    #[test]
    fn test_project_iterator_handles_recursion_error_gracefully() {
        // Create a property value that fails serialized_size()
        let mut deep_val = PropertyValue::Int(1);
        for _ in 0..101 {
            deep_val = PropertyValue::Array(std::sync::Arc::new(vec![deep_val.clone()]));
        }

        // We can create a Node by bypassing try_insert. Since PropertyMap uses Arc<HashMap...>, let's just make one.
        let mut map = std::collections::HashMap::default();
        let key = crate::core::interning::GLOBAL_INTERNER
            .intern("deep")
            .unwrap();
        map.insert(key, deep_val);

        let props = crate::core::PropertyMap {
            inner: std::sync::Arc::new(map),
            cached_size: 100, // Lie about size to avoid computing it
        };

        let node = Node::new(
            NodeId::new(1).unwrap(),
            crate::core::interning::GLOBAL_INTERNER
                .intern("Test")
                .unwrap(),
            props,
            crate::core::id::VersionId::new(1).unwrap(),
        );

        let row = QueryRow::from_entity(EntityResult::Node(node));
        let mock_iter = MockIterator::from_results(vec![Ok(row)]);
        let mut project_iter = ProjectIterator::new(Box::new(mock_iter), vec!["deep".to_string()]);

        let res = project_iter.next().unwrap();
        assert!(
            res.is_err(),
            "ProjectIterator should gracefully handle property insertion errors"
        );
    }

    #[test]
    fn test_mock_iterator_from_nodes() {
        let nodes = vec![test_node(1, "Alice"), test_node(2, "Bob")];

        let mut iter = MockIterator::from_nodes(nodes);

        let row1 = iter.next().unwrap().unwrap();
        assert_eq!(row1.entity.node_id(), Some(NodeId::new(1).unwrap()));

        let row2 = iter.next().unwrap().unwrap();
        assert_eq!(row2.entity.node_id(), Some(NodeId::new(2).unwrap()));

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_mock_iterator_size_hint() {
        let nodes = vec![test_node(1, "Alice"), test_node(2, "Bob")];

        let iter = MockIterator::from_nodes(nodes);

        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 2);
        assert_eq!(upper, Some(2));
    }

    // ==================== Type comparison edge cases ====================

    #[test]
    fn test_filter_type_mismatch_returns_false() {
        // String property compared to Int predicate
        let node = test_node(1, "Alice"); // name is String
        let predicate = Predicate::gt("name", 10i64); // Comparing String to Int

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None)); // Type mismatch returns false
    }

    #[test]
    fn test_filter_contains_on_non_string_returns_false() {
        let node = test_node_with_age(1, "Alice", 30);
        let predicate = Predicate::contains("age", "30"); // age is Int, not String

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_starts_with_on_non_string_returns_false() {
        let node = test_node_with_age(1, "Alice", 30);
        let predicate = Predicate::StartsWith {
            key: "age".to_string(),
            prefix: "3".to_string(),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_ends_with_on_non_string_returns_false() {
        let node = test_node_with_age(1, "Alice", 30);
        let predicate = Predicate::EndsWith {
            key: "age".to_string(),
            suffix: "0".to_string(),
        };

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    // ==================== Null handling ====================

    #[test]
    fn test_filter_null_equality() {
        let props = PropertyMapBuilder::new()
            .insert("name", "Alice")
            .insert("optional", PropertyValue::Null)
            .build();
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        let node = Node::new(
            NodeId::new(1).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        );

        // Null == Null should be true
        let predicate = Predicate::Eq {
            key: "optional".to_string(),
            value: PredicateValue::Null,
        };
        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    // ==================== Complex nested predicates ====================

    #[test]
    fn test_filter_deeply_nested_predicate() {
        let node = test_node_with_age(1, "Alice", 30);

        // (name == "Alice" AND age > 20) OR (name == "Bob")
        let predicate = Predicate::Or(vec![
            Predicate::And(vec![
                Predicate::eq("name", "Alice"),
                Predicate::gt("age", 20i64),
            ]),
            Predicate::eq("name", "Bob"),
        ]);

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_empty_and_is_true() {
        let node = test_node(1, "Alice");
        // Empty AND is vacuously true
        let predicate = Predicate::And(vec![]);

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(filter.evaluate(&node, None));
    }

    #[test]
    fn test_filter_empty_or_is_false() {
        let node = test_node(1, "Alice");
        // Empty OR is vacuously false
        let predicate = Predicate::Or(vec![]);

        let filter = FilterIterator::new(Box::new(EmptyIterator), predicate);
        assert!(!filter.evaluate(&node, None));
    }

    // ==================== NodeLookupIterator Tests ====================

    #[test]
    fn test_node_lookup_iterator_success() {
        let current = Arc::new(CurrentStorage::new());

        // Create test nodes
        let node1 = current
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
            )
            .unwrap();
        let node2 = current
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Bob").build(),
            )
            .unwrap();

        let node_ids = vec![node1, node2];
        let mut iter = NodeLookupIterator::new(node_ids, current);

        // Should get both nodes
        let row1 = iter.next().unwrap().unwrap();
        assert_eq!(row1.entity.node_id(), Some(node1));

        let row2 = iter.next().unwrap().unwrap();
        assert_eq!(row2.entity.node_id(), Some(node2));

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_node_lookup_iterator_missing_node() {
        let current = Arc::new(CurrentStorage::new());

        // Don't add the node
        let node_ids = vec![NodeId::new(999).unwrap()];
        let mut iter = NodeLookupIterator::new(node_ids, current);

        // Should return error for missing node
        let result = iter.next().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_node_lookup_iterator_size_hint() {
        let current = Arc::new(CurrentStorage::new());
        let node_ids = vec![NodeId::new(1).unwrap(), NodeId::new(2).unwrap()];
        let iter = NodeLookupIterator::new(node_ids, current);

        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 2);
        assert_eq!(upper, Some(2));
    }

    // ==================== NodeScanIterator Tests ====================

    #[test]
    fn test_node_scan_iterator_all_nodes() {
        let current = Arc::new(CurrentStorage::new());

        current
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
            )
            .unwrap();
        current
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Bob").build(),
            )
            .unwrap();

        let mut iter = NodeScanIterator::new(None, current);

        let mut results = Vec::new();
        while let Some(Ok(row)) = iter.next() {
            results.push(row);
        }

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_node_scan_iterator_with_label_filter() {
        let current = Arc::new(CurrentStorage::new());

        let person = current
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
            )
            .unwrap();
        current
            .create_node(
                "Company",
                PropertyMapBuilder::new().insert("name", "Acme").build(),
            )
            .unwrap();

        let mut iter = NodeScanIterator::new(Some("Person".to_string()), current);

        let mut results = Vec::new();
        while let Some(Ok(row)) = iter.next() {
            results.push(row);
        }

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity.node_id(), Some(person));
    }

    #[test]
    fn test_node_scan_iterator_empty_storage() {
        let current = Arc::new(CurrentStorage::new());
        let mut iter = NodeScanIterator::new(None, current);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_node_scan_iterator_skips_deleted_ids() {
        // The lazy id-range scan must tolerate gaps: deleting a node in the
        // middle of the id space leaves a `NodeNotFound` hole that the scan
        // skips rather than erroring on.
        let current = Arc::new(CurrentStorage::new());
        let a = current
            .create_node("Person", PropertyMapBuilder::new().insert("n", "a").build())
            .unwrap();
        let b = current
            .create_node("Person", PropertyMapBuilder::new().insert("n", "b").build())
            .unwrap();
        let c = current
            .create_node("Person", PropertyMapBuilder::new().insert("n", "c").build())
            .unwrap();

        current.delete_node(b).unwrap();

        let mut iter = NodeScanIterator::new(None, Arc::clone(&current));
        let mut ids = Vec::new();
        while let Some(Ok(row)) = iter.next() {
            ids.push(row.entity.node_id().unwrap());
        }

        assert_eq!(ids, vec![a, c], "scan should skip the deleted id {b:?}");
    }

    #[test]
    fn test_node_scan_iterator_bounds_by_index_not_storage_id_gen() {
        // Regression (PR #3418): the scan bound must come from the index's
        // insert-maintained high-water-mark, NOT CurrentStorage's own
        // `node_id_gen`. The transactional write path allocates node ids from a
        // database-level generator and applies them via `insert_node_direct`,
        // leaving the storage-local generator at 0. If the scan bounded itself
        // by that generator it would see `max_id == 0` and yield zero rows for
        // every node in the database. Here we reproduce that path directly:
        // insert nodes whose ids did NOT come from `current.node_id_gen`.
        let current = Arc::new(CurrentStorage::new());
        let ts = crate::core::temporal::time::now();
        let label = GLOBAL_INTERNER.intern("Person").unwrap();

        let expected: Vec<NodeId> = (0u64..3)
            .map(|i| {
                let id = NodeId::new(i).unwrap();
                let node = Node::new(
                    id,
                    label,
                    PropertyMapBuilder::new()
                        .insert("n", format!("{i}"))
                        .build(),
                    VersionId::new(i + 1).unwrap(),
                );
                current.insert_node_direct(node, ts).unwrap();
                id
            })
            .collect();

        // The scan bound must reflect the ids inserted (index high-water-mark),
        // not the storage's own id generator (which insert_node_direct never
        // advances, so it is still 0). Under the pre-fix implementation this
        // returned 0 and the scan below yielded nothing.
        assert_eq!(
            current.get_max_node_id(),
            3,
            "scan bound must come from the index high-water-mark, not the storage id generator"
        );

        let mut iter = NodeScanIterator::new(None, Arc::clone(&current));
        let mut ids = Vec::new();
        while let Some(Ok(row)) = iter.next() {
            ids.push(row.entity.node_id().unwrap());
        }

        assert_eq!(
            ids, expected,
            "full scan must find nodes applied via insert_node_direct, \
             proving the bound comes from the index not the storage id generator"
        );
    }

    #[test]
    fn test_node_scan_iterator_unknown_label_yields_nothing() {
        // A label filter whose label was never interned must yield zero rows,
        // NOT degrade into an unfiltered full scan.
        let current = Arc::new(CurrentStorage::new());
        current
            .create_node("Person", PropertyMapBuilder::new().insert("n", "a").build())
            .unwrap();

        let mut iter =
            NodeScanIterator::new(Some("NeverInternedLabel".to_string()), Arc::clone(&current));
        assert!(
            iter.next().is_none(),
            "unknown label must match no nodes, not all of them"
        );
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_node_scan_iterator_size_hint_is_bounded_without_materializing() {
        // size_hint reports a finite upper bound derived from the id range,
        // proving the iterator knows its bound without collecting all ids.
        let current = Arc::new(CurrentStorage::new());
        for i in 0..4 {
            current
                .create_node(
                    "Person",
                    PropertyMapBuilder::new()
                        .insert("n", format!("{i}"))
                        .build(),
                )
                .unwrap();
        }

        let mut iter = NodeScanIterator::new(None, Arc::clone(&current));
        // Upper bound equals the number of ids allocated so far.
        assert_eq!(iter.size_hint(), (0, Some(4)));

        iter.next();
        // After consuming one id the remaining upper bound shrinks.
        assert_eq!(iter.size_hint(), (0, Some(3)));
    }

    // ==================== EdgeScanIterator Tests ====================

    /// Build a small graph (2 nodes, 2 KNOWS edges + 1 FOLLOWS edge) for edge
    /// scan tests. Returns the storage and the count of KNOWS edges (2).
    fn edge_scan_fixture() -> Arc<CurrentStorage> {
        let current = Arc::new(CurrentStorage::new());
        let a = current
            .create_node("Person", PropertyMapBuilder::new().insert("n", "a").build())
            .unwrap();
        let b = current
            .create_node("Person", PropertyMapBuilder::new().insert("n", "b").build())
            .unwrap();
        current
            .create_edge(a, b, "KNOWS", PropertyMapBuilder::new().build())
            .unwrap();
        current
            .create_edge(b, a, "KNOWS", PropertyMapBuilder::new().build())
            .unwrap();
        current
            .create_edge(a, b, "FOLLOWS", PropertyMapBuilder::new().build())
            .unwrap();
        current
    }

    fn drain(mut iter: EdgeScanIterator) -> Vec<crate::core::graph::Edge> {
        let mut edges = Vec::new();
        while let Some(item) = iter.next() {
            edges.push(item.unwrap().entity.as_edge().unwrap().clone());
        }
        edges
    }

    #[test]
    fn test_edge_scan_iterator_all_edges() {
        // `edge_type: None` scans every edge regardless of type.
        let current = edge_scan_fixture();
        let edges = drain(EdgeScanIterator::new(None, current));
        assert_eq!(edges.len(), 3, "unfiltered scan yields all 3 edges");
    }

    #[test]
    fn test_edge_scan_iterator_type_filter_matches_only_that_type() {
        // `edge_type: Some("KNOWS")` yields only the two KNOWS edges.
        let current = edge_scan_fixture();
        let edges = drain(EdgeScanIterator::new(Some("KNOWS".to_string()), current));
        assert_eq!(edges.len(), 2, "type filter must yield only KNOWS edges");
        assert!(
            edges.iter().all(|e| e.has_label_str("KNOWS")),
            "every yielded edge must be a KNOWS edge"
        );
    }

    #[test]
    fn test_edge_scan_iterator_unknown_type_yields_nothing() {
        // A type filter whose type was never interned must yield zero rows,
        // NOT degrade into an unfiltered full scan (mirrors NodeScan).
        let current = edge_scan_fixture();
        let mut iter = EdgeScanIterator::new(Some("NONEXISTENT".to_string()), current);
        assert!(
            iter.next().is_none(),
            "unknown edge type must match no edges, not all of them"
        );
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_edge_scan_iterator_empty_storage() {
        let current = Arc::new(CurrentStorage::new());
        let mut iter = EdgeScanIterator::new(None, current);
        assert!(iter.next().is_none(), "no edges -> no rows");
    }

    #[test]
    fn test_edge_scan_iterator_yields_edge_entities() {
        // Every yielded row must be an Edge entity (not a Node).
        let current = edge_scan_fixture();
        let mut iter = EdgeScanIterator::new(None, current);
        while let Some(item) = iter.next() {
            let row = item.unwrap();
            assert!(
                row.entity.as_edge().is_some(),
                "edge scan must yield Edge entities"
            );
        }
    }

    #[test]
    fn test_edge_scan_iterator_skips_edge_deleted_after_snapshot() {
        // Exercises the `EdgeNotFound => continue` skip arm: the id snapshot is
        // taken at construction, then one edge is deleted from storage before
        // the scan is drained. The deleted id must be skipped (no panic, no
        // error) and only the surviving edges yielded.
        let current = edge_scan_fixture(); // 3 edges
        let all_ids = current.get_all_edge_ids();
        assert_eq!(all_ids.len(), 3, "fixture must start with 3 edges");

        // Construct the iterator first so it snapshots all 3 ids...
        let iter = EdgeScanIterator::new(None, Arc::clone(&current));
        // ...then delete one edge out from under the snapshot.
        current
            .delete_edge(all_ids[0])
            .expect("delete_edge should succeed");

        let edges = drain(iter);
        assert_eq!(
            edges.len(),
            2,
            "the edge deleted after the snapshot must be skipped, leaving 2"
        );
    }

    #[test]
    fn test_edge_scan_iterator_size_hint_reports_edge_count() {
        // A full (unfiltered) scan's size_hint upper bound equals the number of
        // snapshotted edge ids.
        let current = edge_scan_fixture(); // 3 edges
        let iter = EdgeScanIterator::new(None, current);
        assert_eq!(iter.size_hint(), (0, Some(3)));
    }

    // ==================== VectorResultIterator Tests ====================

    #[test]
    fn test_vector_result_iterator_with_scores() {
        let current = Arc::new(CurrentStorage::new());

        let node1 = current
            .create_node(
                "Person",
                PropertyMapBuilder::new()
                    .insert("name", "Alice")
                    .insert_vector("embedding", &[1.0f32, 0.0, 0.0, 0.0])
                    .build(),
            )
            .unwrap();
        let node2 = current
            .create_node(
                "Person",
                PropertyMapBuilder::new()
                    .insert("name", "Bob")
                    .insert_vector("embedding", &[0.0f32, 1.0, 0.0, 0.0])
                    .build(),
            )
            .unwrap();

        let results = vec![(node1, 0.95), (node2, 0.85)];

        let mut iter = VectorResultIterator::new(results, current);

        let row1 = iter.next().unwrap().unwrap();
        assert_eq!(row1.entity.node_id(), Some(node1));
        assert_eq!(row1.score, Some(0.95));

        let row2 = iter.next().unwrap().unwrap();
        assert_eq!(row2.entity.node_id(), Some(node2));
        assert_eq!(row2.score, Some(0.85));

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_vector_result_iterator_missing_node() {
        let current = Arc::new(CurrentStorage::new());

        // Node doesn't exist
        let results = vec![(NodeId::new(999).unwrap(), 0.95)];
        let mut iter = VectorResultIterator::new(results, current);

        let result = iter.next().unwrap();
        assert!(result.is_err());
    }

    // ==================== TemporalNodeIterator Tests ====================

    #[test]
    fn test_temporal_node_iterator_returns_current_state() {
        use crate::core::version::AnchorConfig;
        use crate::storage::historical::HistoricalStorage;

        let current = Arc::new(CurrentStorage::new());
        let historical = Arc::new(RwLock::new(HistoricalStorage::with_config(
            AnchorConfig::default(),
        )));

        let props = PropertyMapBuilder::new().insert("name", "Alice").build();
        let node = current.create_node("Person", props.clone()).unwrap();

        // Add version to historical storage
        use crate::core::temporal::time;
        let now = time::now();
        let label = crate::core::interning::GLOBAL_INTERNER
            .intern("Person")
            .unwrap();
        {
            let mut hist = historical.write();
            hist.add_node_version(
                node,
                crate::core::id::VersionId::new(1).unwrap(),
                now,
                now,
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();
        }

        let node_ids = vec![node];

        let mut iter = TemporalNodeIterator::new(node_ids, now, now, historical);

        let row = iter.next().unwrap().unwrap();
        assert_eq!(row.entity.node_id(), Some(node));
        assert_eq!(row.timestamp, Some(now));
    }

    #[test]
    fn test_temporal_node_iterator_empty() {
        use crate::core::version::AnchorConfig;
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::with_config(
            AnchorConfig::default(),
        )));

        let node_ids = vec![];
        let now = crate::core::temporal::time::now();

        let mut iter = TemporalNodeIterator::new(node_ids, now, now, historical);

        assert!(iter.next().is_none());
    }

    // Characterization tests (Issue #356): capture the exact behavior of the
    // temporal reconstruction path before/after flattening it into a shared
    // helper. Behavior must not change.

    /// Helper: extract the "name" string property from a QueryRow's node.
    fn row_name(row: &QueryRow) -> String {
        let node = row.entity.as_node().expect("row should contain a node");
        match node.properties.get("name") {
            Some(PropertyValue::String(s)) => s.to_string(),
            other => panic!("unexpected name property: {:?}", other),
        }
    }

    /// Helper: historical storage with two versions of node 1:
    /// name=Alice at t=1000, name=Alicia at t=2000 (valid == tx time).
    fn historical_with_two_versions() -> Arc<RwLock<HistoricalStorage>> {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        {
            let mut hist = historical.write();
            hist.add_node_version(
                NodeId::new(1).unwrap(),
                VersionId::new(100).unwrap(),
                1000.into(),
                1000.into(),
                label,
                PropertyMapBuilder::new().insert("name", "Alice").build(),
                false, // not a tombstone
            )
            .unwrap();
            hist.add_node_version(
                NodeId::new(1).unwrap(),
                VersionId::new(101).unwrap(),
                2000.into(),
                2000.into(),
                label,
                PropertyMapBuilder::new().insert("name", "Alicia").build(),
                false, // not a tombstone
            )
            .unwrap();
        }
        historical
    }

    #[test]
    fn test_temporal_node_iterator_not_found_before_first_version() {
        let historical = historical_with_two_versions();
        let node_ids = vec![NodeId::new(1).unwrap()];

        // Query before the node's first version: not found at that time
        let mut iter = TemporalNodeIterator::new(node_ids, 500.into(), 500.into(), historical);

        let result = iter.next().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_temporal_node_iterator_multiple_versions_selects_point_in_time() {
        let historical = historical_with_two_versions();

        // Between the two versions: the older version's properties are returned
        let mut iter = TemporalNodeIterator::new(
            vec![NodeId::new(1).unwrap()],
            1500.into(),
            1500.into(),
            historical.clone(),
        );
        let row = iter.next().unwrap().unwrap();
        assert_eq!(row_name(&row), "Alice");
        assert_eq!(row.timestamp, Some(1500.into()));

        // At/after the second version: the newer version's properties are returned
        let mut iter = TemporalNodeIterator::new(
            vec![NodeId::new(1).unwrap()],
            2500.into(),
            2500.into(),
            historical,
        );
        let row = iter.next().unwrap().unwrap();
        assert_eq!(row_name(&row), "Alicia");
    }

    #[test]
    fn test_temporal_node_iterator_boundary_timestamps() {
        let historical = historical_with_two_versions();

        // Exactly at the first version's start: that version is visible
        let mut iter = TemporalNodeIterator::new(
            vec![NodeId::new(1).unwrap()],
            1000.into(),
            1000.into(),
            historical.clone(),
        );
        let row = iter.next().unwrap().unwrap();
        assert_eq!(row_name(&row), "Alice");

        // Exactly at the second version's start: the new version wins
        // (intervals are half-open: the old version ends at 2000, the new begins)
        let mut iter = TemporalNodeIterator::new(
            vec![NodeId::new(1).unwrap()],
            2000.into(),
            2000.into(),
            historical,
        );
        let row = iter.next().unwrap().unwrap();
        assert_eq!(row_name(&row), "Alicia");
    }

    #[test]
    fn test_temporal_node_iterator_tombstone_semantics() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        {
            let mut hist = historical.write();
            hist.add_node_version(
                NodeId::new(1).unwrap(),
                VersionId::new(100).unwrap(),
                1000.into(),
                1000.into(),
                label,
                PropertyMapBuilder::new().insert("name", "Alice").build(),
                false, // not a tombstone
            )
            .unwrap();
            // Tombstone (deletion) at t=2000
            hist.add_node_version(
                NodeId::new(1).unwrap(),
                VersionId::new(101).unwrap(),
                2000.into(),
                2000.into(),
                label,
                PropertyMapBuilder::new().build(),
                true, // tombstone
            )
            .unwrap();
        }

        // Before the deletion: node is visible with its original properties
        let mut iter = TemporalNodeIterator::new(
            vec![NodeId::new(1).unwrap()],
            1500.into(),
            1500.into(),
            historical.clone(),
        );
        let row = iter.next().unwrap().unwrap();
        assert_eq!(row_name(&row), "Alice");

        // After the deletion: node is not found at that time
        let mut iter = TemporalNodeIterator::new(
            vec![NodeId::new(1).unwrap()],
            2500.into(),
            2500.into(),
            historical,
        );
        let result = iter.next().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_temporal_node_iterator_multiple_versions_selects_point_in_time() {
        // The batch iterator must reconstruct the same point-in-time state as
        // the per-node iterator (shared logic, Issue #356).
        let historical = historical_with_two_versions();

        let mut iter = BatchTemporalNodeIterator::new(
            vec![NodeId::new(1).unwrap()],
            1500.into(),
            1500.into(),
            historical,
        )
        .unwrap();

        let row = iter.next().unwrap().unwrap();
        assert_eq!(row_name(&row), "Alice");
        assert_eq!(row.timestamp, Some(1500.into()));
    }

    // ==================== BatchTemporalNodeIterator Tests ====================

    #[test]
    fn test_batch_temporal_node_iterator_success() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let mut hist = historical.write();

        // Add 3 nodes
        for i in 1..=3 {
            let node_id = NodeId::new(i).unwrap();
            let version_id = VersionId::new(i * 100).unwrap();
            let label = GLOBAL_INTERNER.intern("Person").unwrap();
            let timestamp = ((i * 1000) as i64).into();

            let props = PropertyMapBuilder::new()
                .insert("name", format!("Person{}", i).as_str())
                .build();

            hist.add_node_version(
                node_id, version_id, timestamp, timestamp, label, props, false,
            )
            .unwrap();
        }
        drop(hist);

        // Create batch iterator
        let node_ids = vec![
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            NodeId::new(3).unwrap(),
        ];
        let mut iter =
            BatchTemporalNodeIterator::new(node_ids, 5000.into(), 5000.into(), historical).unwrap();

        // Verify all nodes retrieved
        let mut count = 0;
        while let Some(Ok(_)) = iter.next() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_batch_temporal_node_iterator_node_not_found() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));

        let node_ids = vec![NodeId::new(999).unwrap()];
        let mut iter =
            BatchTemporalNodeIterator::new(node_ids, 1000.into(), 1000.into(), historical).unwrap();

        let result = iter.next().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_temporal_node_iterator_empty() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let node_ids = vec![];
        let mut iter =
            BatchTemporalNodeIterator::new(node_ids, 1000.into(), 1000.into(), historical).unwrap();

        assert!(iter.next().is_none());
    }

    // ==================== TemporalNodeScanIterator Tests (Issue #356) ====================
    //
    // These tests verify the refactored iterator with helper methods:
    // - get_temporal_version(): Handles timestamp-based node retrieval
    // - apply_label_filter(): Manages label-based filtering
    // - filter_node(): Orchestrates filtering logic

    #[test]
    fn test_temporal_node_scan_iterator_get_temporal_version_success() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let node_id = NodeId::new(1).unwrap();
        let version_id = VersionId::new(100).unwrap();
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        let timestamp: Timestamp = 1000.into();

        let props = PropertyMapBuilder::new().insert("name", "Alice").build();

        {
            let mut hist = historical.write();
            hist.add_node_version(
                node_id, version_id, timestamp, timestamp, label, props, false,
            )
            .unwrap();
        }

        // Test the get_temporal_version helper method directly
        let iter = TemporalNodeScanIterator::new(
            vec![node_id],
            timestamp,
            timestamp,
            historical.clone(),
            None, // No label filter
        );

        let guard = historical.read();
        let result = iter.get_temporal_version(node_id, &guard);
        assert!(result.is_ok());

        let node = result.unwrap();
        assert_eq!(node.id, node_id);
    }

    #[test]
    fn test_temporal_node_scan_iterator_get_temporal_version_not_found() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let node_id = NodeId::new(999).unwrap();
        let timestamp: Timestamp = 1000.into();

        let iter = TemporalNodeScanIterator::new(
            vec![node_id],
            timestamp,
            timestamp,
            historical.clone(),
            None,
        );

        let guard = historical.read();
        let result = iter.get_temporal_version(node_id, &guard);
        assert!(result.is_err());
    }

    #[test]
    fn test_temporal_node_scan_iterator_apply_label_filter_matches() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let timestamp: Timestamp = 1000.into();

        // Intern label BEFORE creating iterator (simulates real-world usage
        // where labels are interned when nodes are created in storage)
        let label = GLOBAL_INTERNER.intern("Person").unwrap();

        // Create iterator with "Person" label filter
        let iter = TemporalNodeScanIterator::new(
            vec![],
            timestamp,
            timestamp,
            historical,
            Some("Person".to_string()),
        );

        let props = PropertyMapBuilder::new().insert("name", "Alice").build();
        let node = Node::new(
            NodeId::new(1).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        );

        // Label matches, should return true
        assert!(iter.apply_label_filter(&node));
    }

    #[test]
    fn test_temporal_node_scan_iterator_apply_label_filter_no_match() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let timestamp: Timestamp = 1000.into();

        // Intern both labels BEFORE creating iterator
        let _company_label = GLOBAL_INTERNER.intern("Company").unwrap();
        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();

        // Create iterator with "Company" label filter
        let iter = TemporalNodeScanIterator::new(
            vec![],
            timestamp,
            timestamp,
            historical,
            Some("Company".to_string()),
        );

        let props = PropertyMapBuilder::new().insert("name", "Alice").build();
        let node = Node::new(
            NodeId::new(1).unwrap(),
            person_label,
            props,
            VersionId::new(1).unwrap(),
        );

        // Label doesn't match (Company != Person), should return false
        assert!(!iter.apply_label_filter(&node));
    }

    #[test]
    fn test_temporal_node_scan_iterator_apply_label_filter_no_filter() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let timestamp: Timestamp = 1000.into();

        // Create iterator with no label filter
        let iter = TemporalNodeScanIterator::new(vec![], timestamp, timestamp, historical, None);

        let label = GLOBAL_INTERNER.intern("AnyLabel").unwrap();
        let props = PropertyMapBuilder::new().build();
        let node = Node::new(
            NodeId::new(1).unwrap(),
            label,
            props,
            VersionId::new(1).unwrap(),
        );

        // No filter, should always return true
        assert!(iter.apply_label_filter(&node));
    }

    #[test]
    fn test_temporal_node_scan_iterator_filter_node_success() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let node_id = NodeId::new(1).unwrap();
        let version_id = VersionId::new(100).unwrap();
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        let timestamp: Timestamp = 1000.into();

        let props = PropertyMapBuilder::new().insert("name", "Alice").build();

        {
            let mut hist = historical.write();
            hist.add_node_version(
                node_id, version_id, timestamp, timestamp, label, props, false,
            )
            .unwrap();
        }

        // Test filter_node orchestrator with matching label
        let iter = TemporalNodeScanIterator::new(
            vec![node_id],
            timestamp,
            timestamp,
            historical.clone(),
            Some("Person".to_string()),
        );

        let guard = historical.read();
        let result = iter.filter_node(node_id, &guard);

        // Should return Some(Ok(QueryRow)) for matching node
        assert!(result.is_some());
        let query_row = result.unwrap();
        assert!(query_row.is_ok());
        assert_eq!(query_row.unwrap().entity.node_id(), Some(node_id));
    }

    #[test]
    fn test_temporal_node_scan_iterator_filter_node_label_mismatch() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let node_id = NodeId::new(1).unwrap();
        let version_id = VersionId::new(100).unwrap();
        // Intern both labels before use
        let _company_label = GLOBAL_INTERNER.intern("Company").unwrap();
        let person_label = GLOBAL_INTERNER.intern("Person").unwrap();
        let timestamp: Timestamp = 1000.into();

        let props = PropertyMapBuilder::new().insert("name", "Alice").build();

        {
            let mut hist = historical.write();
            hist.add_node_version(
                node_id,
                version_id,
                timestamp,
                timestamp,
                person_label,
                props,
                false, // not a tombstone
            )
            .unwrap();
        }

        // Test filter_node with non-matching label
        let iter = TemporalNodeScanIterator::new(
            vec![node_id],
            timestamp,
            timestamp,
            historical.clone(),
            Some("Company".to_string()), // Different label
        );

        let guard = historical.read();
        let result = iter.filter_node(node_id, &guard);

        // Should return None when label doesn't match
        assert!(result.is_none());
    }

    #[test]
    fn test_temporal_node_scan_iterator_filter_node_not_found() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let node_id = NodeId::new(999).unwrap();
        let timestamp: Timestamp = 1000.into();

        let iter = TemporalNodeScanIterator::new(
            vec![node_id],
            timestamp,
            timestamp,
            historical.clone(),
            None,
        );

        let guard = historical.read();
        let result = iter.filter_node(node_id, &guard);

        // Should return Some(Err(...)) when node not found
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn test_temporal_node_scan_iterator_full_iteration() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let timestamp: Timestamp = 5000.into();

        // Add 3 Person nodes and 1 Company node
        {
            let mut hist = historical.write();
            for i in 1..=3 {
                let node_id = NodeId::new(i).unwrap();
                let version_id = VersionId::new(i * 100).unwrap();
                let label = GLOBAL_INTERNER.intern("Person").unwrap();

                let props = PropertyMapBuilder::new()
                    .insert("name", format!("Person{}", i).as_str())
                    .build();

                hist.add_node_version(
                    node_id, version_id, timestamp, timestamp, label, props, false,
                )
                .unwrap();
            }

            // Add Company node
            let company_label = GLOBAL_INTERNER.intern("Company").unwrap();
            hist.add_node_version(
                NodeId::new(4).unwrap(),
                VersionId::new(400).unwrap(),
                timestamp,
                timestamp,
                company_label,
                PropertyMapBuilder::new().insert("name", "Acme").build(),
                false, // not a tombstone
            )
            .unwrap();
        }

        // Iterate with "Person" filter - should get 3 results
        let node_ids = vec![
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            NodeId::new(3).unwrap(),
            NodeId::new(4).unwrap(),
        ];

        let mut iter = TemporalNodeScanIterator::new(
            node_ids,
            timestamp,
            timestamp,
            historical.clone(),
            Some("Person".to_string()),
        );

        let mut count = 0;
        while let Some(result) = iter.next() {
            assert!(result.is_ok());
            count += 1;
        }

        assert_eq!(count, 3); // Only Person nodes, not Company
    }

    #[test]
    fn test_temporal_node_scan_iterator_no_label_filter() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let timestamp: Timestamp = 5000.into();

        // Add 2 nodes with different labels
        {
            let mut hist = historical.write();

            let person_label = GLOBAL_INTERNER.intern("Person").unwrap();
            hist.add_node_version(
                NodeId::new(1).unwrap(),
                VersionId::new(100).unwrap(),
                timestamp,
                timestamp,
                person_label,
                PropertyMapBuilder::new().insert("name", "Alice").build(),
                false, // not a tombstone
            )
            .unwrap();

            let company_label = GLOBAL_INTERNER.intern("Company").unwrap();
            hist.add_node_version(
                NodeId::new(2).unwrap(),
                VersionId::new(200).unwrap(),
                timestamp,
                timestamp,
                company_label,
                PropertyMapBuilder::new().insert("name", "Acme").build(),
                false, // not a tombstone
            )
            .unwrap();
        }

        // Iterate without label filter - should get all nodes
        let node_ids = vec![NodeId::new(1).unwrap(), NodeId::new(2).unwrap()];

        let mut iter =
            TemporalNodeScanIterator::new(node_ids, timestamp, timestamp, historical, None);

        let mut count = 0;
        while let Some(result) = iter.next() {
            assert!(result.is_ok());
            count += 1;
        }

        assert_eq!(count, 2); // Both nodes returned
    }

    #[test]
    fn test_temporal_node_scan_iterator_size_hint() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let timestamp: Timestamp = 1000.into();

        let node_ids = vec![
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            NodeId::new(3).unwrap(),
        ];

        let iter = TemporalNodeScanIterator::new(node_ids, timestamp, timestamp, historical, None);

        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 3);
        assert_eq!(upper, Some(3));
    }

    #[test]
    fn test_temporal_node_scan_iterator_empty() {
        use crate::storage::historical::HistoricalStorage;

        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let timestamp: Timestamp = 1000.into();

        let mut iter =
            TemporalNodeScanIterator::new(vec![], timestamp, timestamp, historical, None);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_vector_rerank_heap_logic() {
        use crate::core::property::PropertyMapBuilder;
        use crate::index::vector::{DistanceMetric, HnswConfig};

        // This test verifies that the heap logic correctly maintains the top-k items
        // and orders them correctly (descending score).

        let current = Arc::new(CurrentStorage::new());
        // Enable vector index
        current
            .enable_vector_index("embedding", HnswConfig::new(4, DistanceMetric::Cosine))
            .unwrap();

        // Create 5 nodes with predictable embeddings/scores relative to query [1,0,0,0]
        // Node 1: [1,0,0,0] -> score 1.0 (Best)
        // Node 2: [0,1,0,0] -> score 0.0
        // Node 3: [0.5, 0.866, 0, 0] -> score 0.5
        // Node 4: [0.8, 0.6, 0, 0] -> score 0.8
        // Node 5: [-1, 0, 0, 0] -> score -1.0 (Worst)

        let create_node = |name: &str, vec: Vec<f32>| {
            let props = PropertyMapBuilder::new()
                .insert("name", name)
                .insert_vector("embedding", &vec)
                .build();
            current.create_node("Person", props).unwrap()
        };

        let n1 = create_node("N1", vec![1.0, 0.0, 0.0, 0.0]);
        let n2 = create_node("N2", vec![0.0, 1.0, 0.0, 0.0]);
        let n3 = create_node("N3", vec![0.5, 0.866, 0.0, 0.0]);
        let n4 = create_node("N4", vec![0.8, 0.6, 0.0, 0.0]);
        let n5 = create_node("N5", vec![-1.0, 0.0, 0.0, 0.0]);

        // Case 1: k=3. Expect top 3: N1 (1.0), N4 (0.8), N3 (0.5)
        let nodes = vec![n1, n2, n3, n4, n5];
        let input = Box::new(NodeLookupIterator::new(nodes.clone(), current.clone()));
        let query_embedding: Arc<[f32]> = vec![1.0, 0.0, 0.0, 0.0].into();

        let mut rerank =
            VectorRerankIterator::new(input, query_embedding.clone(), 3, current.clone(), None);

        let mut results = Vec::new();
        while let Some(Ok(row)) = rerank.next() {
            results.push(row);
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].entity.node_id(), Some(n1)); // 1.0
        assert_eq!(results[1].entity.node_id(), Some(n4)); // 0.8
        assert_eq!(results[2].entity.node_id(), Some(n3)); // 0.5

        // Case 2: k=1. Expect top 1: N1
        let input = Box::new(NodeLookupIterator::new(nodes.clone(), current.clone()));
        let mut rerank =
            VectorRerankIterator::new(input, query_embedding.clone(), 1, current.clone(), None);
        let mut results = Vec::new();
        while let Some(Ok(row)) = rerank.next() {
            results.push(row);
        }
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity.node_id(), Some(n1));

        // Case 3: k=10 (more than available). Expect all 5 sorted.
        let input = Box::new(NodeLookupIterator::new(nodes.clone(), current.clone()));
        let mut rerank =
            VectorRerankIterator::new(input, query_embedding.clone(), 10, current.clone(), None);
        let mut results = Vec::new();
        while let Some(Ok(row)) = rerank.next() {
            results.push(row);
        }
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].entity.node_id(), Some(n1));
        assert_eq!(results[4].entity.node_id(), Some(n5));

        // Case 4: k=0. Expect 0 results.
        let input = Box::new(NodeLookupIterator::new(nodes.clone(), current.clone()));
        let mut rerank =
            VectorRerankIterator::new(input, query_embedding.clone(), 0, current.clone(), None);
        let mut results = Vec::new();
        while let Some(Ok(row)) = rerank.next() {
            results.push(row);
        }
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_vector_rerank_iterator_safely_handles_empty_input() {
        // 🛡️ Sentry: Ensure `take().unwrap()` was removed and gracefully handles missing input
        let current = Arc::new(CurrentStorage::new());
        let embedding: Arc<[f32]> = Arc::new([1.0, 0.0]);
        let input = Box::new(EmptyIterator);

        let mut iter =
            VectorRerankIterator::new(input, embedding, 10, current, Some("embedding".to_string()));

        // First call will take the input and exhaust it, and build self.sorted = Some(empty)
        assert!(iter.next().is_none());

        // A second call will see self.sorted.is_some(), but it's empty
        assert!(iter.next().is_none());

        // Let's explicitly trigger the path where input is missing but we're trying to init
        iter.sorted = None; // Force re-initialization

        // This used to unwrap and panic because input was taken in the first call
        assert!(iter.next().is_none());
    }

    // ==================== evaluate_null Tests ====================

    /// Exhaustively covers every arm of [`FilterIterator::evaluate_null`]:
    /// comparisons/membership/string predicates over a null binding are
    /// not-true, `IS NULL` (encoded `Eq { value: Null }`) is true,
    /// `IS NOT NULL` (encoded `Ne { value: Null }`) is false, a null binding
    /// has no properties (`Exists` false / `NotExists` true), and the
    /// boolean connectives compose (with `Not` documented as two-valued).
    #[test]
    fn test_evaluate_null_all_predicate_arms() {
        let filter = FilterIterator::new(Box::new(EmptyIterator), Predicate::True);
        let eval = |p: &Predicate| filter.evaluate_null(p);

        // Constants.
        assert!(eval(&Predicate::True));
        assert!(!eval(&Predicate::False));

        // IS NULL / IS NOT NULL encodings.
        assert!(eval(&Predicate::Eq {
            key: "x".into(),
            value: PredicateValue::Null,
        }));
        assert!(!eval(&Predicate::Ne {
            key: "x".into(),
            value: PredicateValue::Null,
        }));

        // Eq/Ne against non-null values are not-true.
        assert!(!eval(&Predicate::eq("x", 1i64)));
        assert!(!eval(&Predicate::ne("x", 1i64)));

        // Ordering comparisons are not-true.
        assert!(!eval(&Predicate::Gt {
            key: "x".into(),
            value: PredicateValue::Int(1),
        }));
        assert!(!eval(&Predicate::Lt {
            key: "x".into(),
            value: PredicateValue::Int(1),
        }));
        assert!(!eval(&Predicate::Gte {
            key: "x".into(),
            value: PredicateValue::Int(1),
        }));
        assert!(!eval(&Predicate::Lte {
            key: "x".into(),
            value: PredicateValue::Int(1),
        }));

        // Membership and string predicates are not-true.
        assert!(!eval(&Predicate::In {
            key: "x".into(),
            values: vec![PredicateValue::Int(1), PredicateValue::Null],
        }));
        assert!(!eval(&Predicate::Contains {
            key: "x".into(),
            substring: "a".into(),
        }));
        assert!(!eval(&Predicate::StartsWith {
            key: "x".into(),
            prefix: "a".into(),
        }));
        assert!(!eval(&Predicate::EndsWith {
            key: "x".into(),
            suffix: "a".into(),
        }));

        // A null binding has no properties.
        assert!(!eval(&Predicate::Exists("x".into())));
        assert!(eval(&Predicate::NotExists("x".into())));

        // Connectives compose.
        assert!(eval(&Predicate::And(vec![
            Predicate::True,
            Predicate::NotExists("x".into()),
        ])));
        assert!(!eval(&Predicate::And(vec![
            Predicate::True,
            Predicate::eq("x", 1i64),
        ])));
        assert!(eval(&Predicate::Or(vec![
            Predicate::eq("x", 1i64),
            Predicate::Eq {
                key: "x".into(),
                value: PredicateValue::Null,
            },
        ])));
        assert!(!eval(&Predicate::Or(vec![
            Predicate::False,
            Predicate::eq("x", 1i64),
        ])));

        // Not is two-valued (documented deviation from openCypher 3VL):
        // NOT (not-true) is true.
        assert!(eval(&Predicate::Not(Box::new(Predicate::eq("x", 1i64)))));
        assert!(!eval(&Predicate::Not(Box::new(Predicate::True))));
    }

    // ==================== OptionalApplyIterator Tests ====================

    fn optional_test_storages() -> (
        Arc<CurrentStorage>,
        Arc<RwLock<crate::storage::historical::HistoricalStorage>>,
    ) {
        let current = Arc::new(CurrentStorage::new());
        let historical = Arc::new(RwLock::new(
            crate::storage::historical::HistoricalStorage::with_config(
                crate::core::version::AnchorConfig::default(),
            ),
        ));
        (current, historical)
    }

    /// The unmatched fallback row must preserve the seed row's metadata
    /// (score, path, timestamp), with only the entity replaced by Null.
    #[test]
    fn test_optional_apply_unmatched_preserves_seed_metadata() {
        use crate::query::planner::physical::OptionalPhysicalStep;

        let (current, historical) = optional_test_storages();

        let seed = QueryRow::with_score(EntityResult::Node(test_node(1, "Alice")), 0.75)
            .at_time(Timestamp::from(12345i64));
        let input = Box::new(MockIterator::from_results(vec![Ok(seed)]));

        // A Filter(False) step can never match: the seed row falls back to
        // the null form.
        let mut iter = OptionalApplyIterator::new(
            input,
            vec![OptionalPhysicalStep::Filter(Predicate::False)],
            current,
            historical,
        );

        let row = iter.next().expect("one row expected").expect("no error");
        assert!(row.entity.is_null(), "unmatched seed must bind null");
        assert_eq!(row.score, Some(0.75), "seed score must be preserved");
        assert_eq!(
            row.timestamp,
            Some(Timestamp::from(12345i64)),
            "seed timestamp must be preserved"
        );
        assert!(iter.next().is_none());
    }

    /// The standalone (leading OPTIONAL MATCH) form has no seed row: the
    /// unmatched fallback is a bare null row.
    #[test]
    fn test_optional_apply_standalone_unmatched_bare_null_row() {
        use crate::query::planner::physical::OptionalPhysicalStep;

        let (current, historical) = optional_test_storages();

        let mut iter = OptionalApplyIterator::new(
            Box::new(EmptyIterator),
            vec![OptionalPhysicalStep::Scan {
                label: Some("Person".to_string()),
            }],
            current,
            historical,
        );

        let row = iter.next().expect("one row expected").expect("no error");
        assert!(row.entity.is_null());
        assert!(row.score.is_none());
        assert!(row.path.is_none());
        assert!(row.timestamp.is_none());
        assert!(iter.next().is_none());
    }

    /// The matched case: a seed row whose optional traversal produces rows
    /// yields those rows (not a null fallback), then moves to the next seed.
    #[test]
    fn test_optional_apply_matched_rows_pass_through() {
        use crate::query::planner::physical::OptionalPhysicalStep;

        let (current, historical) = optional_test_storages();
        let alice = current
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
            )
            .unwrap();
        let bob = current
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Bob").build(),
            )
            .unwrap();
        current
            .create_edge(alice, bob, "KNOWS", PropertyMapBuilder::new().build())
            .unwrap();

        let seed = QueryRow::from_entity(EntityResult::Node(current.get_node(alice).unwrap()));
        let input = Box::new(MockIterator::from_results(vec![Ok(seed)]));

        let mut iter = OptionalApplyIterator::new(
            input,
            vec![OptionalPhysicalStep::Traverse {
                direction: Direction::Outgoing,
                label: Some("KNOWS".to_string()),
                min_depth: 1,
                depth: 1,
                temporal_context: None,
            }],
            current,
            historical,
        );

        let row = iter.next().expect("one row expected").expect("no error");
        assert_eq!(
            row.entity.node_id(),
            Some(bob),
            "matched optional must yield the traversal target, not null"
        );
        // The matched seed must NOT be followed by a fabricated null row.
        assert!(iter.next().is_none());
    }

    /// An error from the input (seed) iterator is propagated as-is.
    #[test]
    fn test_optional_apply_input_error_propagates() {
        use crate::query::planner::physical::OptionalPhysicalStep;

        let (current, historical) = optional_test_storages();
        let input = Box::new(MockIterator::from_results(vec![Err(
            crate::core::error::Error::other("test error"),
        )]));

        let mut iter = OptionalApplyIterator::new(
            input,
            vec![OptionalPhysicalStep::Filter(Predicate::True)],
            current,
            historical,
        );

        assert!(iter.next().expect("error row expected").is_err());
        assert!(iter.next().is_none());
    }

    /// An error inside the optional sub-pipeline abandons the seed: the
    /// error is surfaced and NO fabricated null row follows for that seed.
    #[test]
    fn test_optional_apply_inner_error_abandons_seed() {
        use crate::query::planner::physical::OptionalPhysicalStep;

        let (current, historical) = optional_test_storages();
        let alice = current
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
            )
            .unwrap();
        let bob = current
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Bob").build(),
            )
            .unwrap();
        current
            .create_edge(alice, bob, "KNOWS", PropertyMapBuilder::new().build())
            .unwrap();
        let seed = QueryRow::from_entity(EntityResult::Node(current.get_node(alice).unwrap()));
        // Delete Bob but keep the edge (low-level delete_node preserves
        // edges): the traversal reaches a missing endpoint and errors.
        current.delete_node(bob).unwrap();

        let input = Box::new(MockIterator::from_results(vec![Ok(seed)]));
        let mut iter = OptionalApplyIterator::new(
            input,
            vec![OptionalPhysicalStep::Traverse {
                direction: Direction::Outgoing,
                label: Some("KNOWS".to_string()),
                min_depth: 1,
                depth: 1,
                temporal_context: None,
            }],
            current,
            historical,
        );

        assert!(iter.next().expect("error row expected").is_err());
        // The errored seed must not fall back to a null row.
        assert!(iter.next().is_none());
    }

    /// size_hint: at least one row per remaining input row, no upper bound.
    #[test]
    fn test_optional_apply_size_hint() {
        use crate::query::planner::physical::OptionalPhysicalStep;

        let (current, historical) = optional_test_storages();
        let input = Box::new(MockIterator::from_nodes(vec![
            test_node(1, "Alice"),
            test_node(2, "Bob"),
        ]));

        let iter = OptionalApplyIterator::new(
            input,
            vec![OptionalPhysicalStep::Filter(Predicate::True)],
            current,
            historical,
        );
        assert_eq!(iter.size_hint(), (2, None));
    }

    /// SeedRowIterator yields its single row exactly once, with exact hints.
    #[test]
    fn test_seed_row_iterator() {
        let row = QueryRow::from_entity(EntityResult::Node(test_node(1, "Alice")));
        let mut iter = SeedRowIterator::new(row);

        assert_eq!(iter.size_hint(), (1, Some(1)));
        assert!(iter.next().expect("one row").is_ok());
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());
    }

    /// FilterIterator applies null semantics to null-binding rows from an
    /// unmatched OPTIONAL MATCH: IS NULL keeps them, comparisons drop them.
    #[test]
    fn test_filter_iterator_null_row_semantics() {
        // `x IS NULL` (encoded Eq { value: Null }) keeps the null row.
        let input = Box::new(MockIterator::from_results(vec![Ok(QueryRow::from_entity(
            EntityResult::Null,
        ))]));
        let mut keep = FilterIterator::new(
            input,
            Predicate::Eq {
                key: "x".into(),
                value: PredicateValue::Null,
            },
        );
        let row = keep.next().expect("one row expected").expect("no error");
        assert!(row.entity.is_null());
        assert!(keep.next().is_none());

        // A comparison against the null binding is not-true: row dropped.
        let input = Box::new(MockIterator::from_results(vec![Ok(QueryRow::from_entity(
            EntityResult::Null,
        ))]));
        let mut dropped = FilterIterator::new(input, Predicate::eq("x", 1i64));
        assert!(dropped.next().is_none());
    }
}
