//! Write transactions with ACID guarantees.
//!
//! Write transactions provide full ACID properties:
//! - **Atomicity**: All-or-nothing commit via write buffering
//! - **Consistency**: Referential integrity validation before commit
//! - **Isolation**: Snapshot Isolation with write-write conflict detection
//! - **Durability**: WAL with fsync guarantees
//!
//! Write transactions buffer all changes in memory until commit.
//! On commit, changes are validated and applied atomically.
//!
//! # The commit clock amplifies WAL back-pressure (Issue #3804)
//!
//! `commit()` holds the commit-timestamp lock across the WAL append, so
//! anything that makes the append slow does not merely slow *this*
//! transaction — it serializes every other committer behind it. The WAL's
//! append bound is deliberately a **stall** detector rather than a cap on
//! total call time (see `docs/WAL.md`), so a batch that keeps inching forward
//! against a degraded-but-alive drainer can legitimately run for many
//! multiples of that bound, and the commit clock is held for all of it. The
//! WAL narrates such a call once per elapsed stall window; it does not
//! shorten it. Narrowing the commit-clock scope so a blocking append no longer
//! holds it is the actual cure and is tracked as **Issue #3804**.
//!
//! # Frames are appended before the epoch is registered
//!
//! `log_operations_to_wal` runs before `ConcurrentWalSystem::commit`, so a
//! GroupCommit flush cycle can drain this transaction's frames into an epoch
//! it never registers into. On the success path that is harmless; a *failing*
//! flush inside that window is currently misattributed. The full statement of
//! the gap, and why the flush coordinator's LSN watermark cannot be used to
//! detect it, lives on `ConcurrentWalSystem::commit`.

use super::{
    ReadOps, TransactionSnapshot, TxId, TxMetadata, TxState, TxVisibilityManager, WriteBuffer,
    WriteOps, WriteRequestOptions,
};
use crate::core::commit_clock::CommitClock;
use crate::core::error::{Result, ResultExt, StorageError, TransactionError};
use crate::core::graph::{Edge, Node};
use crate::core::hlc::{
    SendWithSelfHealError, evaluate_clock_skew, is_clock_skew_self_heal_enabled,
    send_with_overflow_self_heal,
};
use crate::core::id::{EdgeId, IdGenerator, NodeId, VersionId};
use crate::core::interning::GLOBAL_INTERNER;
use crate::core::namespace;
use crate::core::property::{PropertyMap, PropertyMapBuilder, PropertyValue};
use crate::core::provenance::Provenance;
use crate::core::temporal::{Timestamp, time};
use crate::core::version::VersionMetadata;
use crate::index::temporal::TemporalIndexes;
use crate::storage::current::CurrentStorage;
use crate::storage::historical::HistoricalStorage;
use crate::storage::wal::DurabilityMode;
use crate::storage::wal::concurrent_system::ConcurrentWalSystem;
use parking_lot::RwLock;
use std::sync::atomic::AtomicU8;
use std::sync::{Arc, Mutex};
use std::time::Instant;

mod apply;
pub(crate) mod cas;
mod conflict;
pub(crate) mod constraint;
mod in_flight;
mod validation;
mod wal;

pub(crate) use in_flight::InFlightLsns;

#[cfg(test)]
pub(crate) const MAX_BACKWARD_DRIFT_US: i64 = crate::core::hlc::MAX_BACKWARD_DRIFT_US;
pub(crate) const MAX_FORWARD_JUMP_US: i64 = crate::core::hlc::MAX_FORWARD_JUMP_US;

/// Maximum offset for valid_from timestamps in the future.
///
/// Users can backdate facts (valid_from < transaction_time) for historical corrections,
/// but we limit how far into the future they can set valid_from to prevent abuse and
/// maintain query semantics.
pub(crate) const MAX_VALID_TIME_FUTURE_OFFSET_US: i64 = 365 * 24 * 60 * 60 * 1_000_000; // 1 year

/// Write transaction with full ACID guarantees.
///
/// Write transactions buffer all operations in memory and apply them
/// atomically on commit. This ensures consistency and enables rollback.
///
/// # Example
///
/// ```rust,no_run
/// # use aletheiadb::{AletheiaDB, properties, api::transaction::WriteOps};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let db = AletheiaDB::new()?;
/// # let other = aletheiadb::core::NodeId::new(1)?;
/// # let props = properties! { "name" => "Alice" };
/// # let edge_props = properties! { "since" => 2024 };
/// let mut tx = db.write_transaction()?;
/// let node_id = tx.create_node("Person", props)?;
/// tx.create_edge(node_id, other, "KNOWS", edge_props)?;
/// tx.commit()?;  // or tx.rollback()
/// # Ok(())
/// # }
/// ```
pub struct WriteTransaction {
    pub(crate) tx_id: TxId,
    pub(crate) start_timestamp: Timestamp,
    pub(crate) state: TxState,

    // Snapshot for Snapshot Isolation
    pub(crate) snapshot: TransactionSnapshot,

    // Write buffer for uncommitted changes
    pub(crate) buffer: WriteBuffer,

    // Shared references to storage (Arc for zero-copy sharing)
    pub(crate) current: Arc<CurrentStorage>,
    pub(crate) historical: Arc<RwLock<HistoricalStorage>>,
    pub(crate) temporal_indexes: Arc<TemporalIndexes>,
    pub(crate) wal: Arc<ConcurrentWalSystem>,
    pub(crate) current_timestamp: Arc<CommitClock>,
    pub(crate) commit_clock_observed_at: Arc<Mutex<Instant>>,
    pub(crate) visibility_manager: Arc<TxVisibilityManager>,

    // ID generators (needed for creating new entities)
    // IdGenerator uses AtomicU64 internally, so no external Mutex is needed.
    pub(crate) node_id_gen: Arc<IdGenerator>,
    pub(crate) edge_id_gen: Arc<IdGenerator>,
    pub(crate) version_id_gen: Arc<IdGenerator>,

    /// Durability mode for this transaction's commit
    pub(crate) durability_mode: DurabilityMode,

    /// Uniqueness constraint registry shared with the parent database.
    /// `None` for transactions created without a registry (legacy / tests).
    pub(crate) constraint_registry: Option<Arc<crate::core::constraint::ConstraintRegistry>>,

    /// In-flight (durable-but-not-yet-applied) LSN tracker shared with the
    /// parent database (lost-write persist race fix). `None` for transactions
    /// created without a tracker (legacy / tests): such commits skip watermark
    /// registration and behave exactly as before.
    pub(crate) in_flight: Option<Arc<InFlightLsns>>,

    /// The parent database's node-role cell (Issue #3355, Slice A), shared
    /// (not owned) so a demotion racing this transaction's commit is caught.
    /// `None` for transactions created without it (legacy / low-level
    /// tests): such commits skip the recheck, matching pre-#3355 behavior.
    /// The construction-time seam
    /// (`AletheiaDB::write_transaction`/`write_transaction_with_options`)
    /// already rejected before this transaction was built; this is the
    /// defensive recheck at commit time, closing the gap where
    /// `enter_replica_mode` runs after construction but before commit.
    pub(crate) role: Option<Arc<AtomicU8>>,

    /// Compare-and-set preconditions buffered by `compare_and_set_*` /
    /// `claim_with_lease` (Issue #3577). Each carries the condition (expected
    /// version, optional lease OR-branch) for its buffered full-replace
    /// `UpdateNode`/`UpdateEdge`; they are re-checked at commit under the
    /// `historical.write()` guard by [`cas::detect_cas_precondition_violations`]
    /// and excluded from the pre-lock snapshot-isolation conflict check. Empty
    /// for transactions that use no conditional writes (zero overhead).
    pub(crate) cas_preconditions: Vec<cas::CasPrecondition>,

    /// Push-changefeed broadcaster (Issue #3375). `None` for transactions created
    /// without the DB-layer wiring (legacy / low-level tests): such commits emit no
    /// changefeed events. When present, the committed change set is broadcast to
    /// matching subscribers AFTER the commit is durable, applied, and visible — outside
    /// every write-path lock (the broadcaster's lock is a leaf).
    pub(crate) changefeed: Option<Arc<crate::core::changefeed_subscription::ChangefeedBroadcaster>>,

    /// GDPR crypto-shred sealing context (Issue #3359, PR-1b). `None` unless the
    /// parent database has at least one **active** erasure designation, so a
    /// database that never used crypto-shred pays nothing. When present, the
    /// apply hook (`apply_node_write`/`apply_edge_write`) replaces designated
    /// property values with sealed `SUBJ` envelopes before they reach either the
    /// current or the historical tier.
    #[cfg(feature = "audit-export")]
    pub(crate) sealing: Option<crate::db::crypto_shred::SealingContext>,
}

impl WriteTransaction {
    /// Create a new write transaction with default durability mode (Synchronous).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        tx_id: TxId,
        snapshot: TransactionSnapshot,
        current: Arc<CurrentStorage>,
        historical: Arc<RwLock<HistoricalStorage>>,
        temporal_indexes: Arc<TemporalIndexes>,
        wal: Arc<ConcurrentWalSystem>,
        current_timestamp: Arc<CommitClock>,
        visibility_manager: Arc<TxVisibilityManager>,
        node_id_gen: Arc<IdGenerator>,
        edge_id_gen: Arc<IdGenerator>,
        version_id_gen: Arc<IdGenerator>,
    ) -> Self {
        Self::new_with_durability(
            tx_id,
            snapshot,
            current,
            historical,
            temporal_indexes,
            wal,
            current_timestamp,
            visibility_manager,
            node_id_gen,
            edge_id_gen,
            version_id_gen,
            DurabilityMode::Synchronous,
        )
    }

    /// Create a new write transaction with explicit commit clock observation state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_clock_observed_at(
        tx_id: TxId,
        snapshot: TransactionSnapshot,
        current: Arc<CurrentStorage>,
        historical: Arc<RwLock<HistoricalStorage>>,
        temporal_indexes: Arc<TemporalIndexes>,
        wal: Arc<ConcurrentWalSystem>,
        current_timestamp: Arc<CommitClock>,
        commit_clock_observed_at: Arc<Mutex<Instant>>,
        visibility_manager: Arc<TxVisibilityManager>,
        node_id_gen: Arc<IdGenerator>,
        edge_id_gen: Arc<IdGenerator>,
        version_id_gen: Arc<IdGenerator>,
    ) -> Self {
        Self::new_with_durability_and_clock_observed_at(
            tx_id,
            snapshot,
            current,
            historical,
            temporal_indexes,
            wal,
            current_timestamp,
            commit_clock_observed_at,
            visibility_manager,
            node_id_gen,
            edge_id_gen,
            version_id_gen,
            DurabilityMode::Synchronous,
        )
    }

    /// Create a new write transaction with a specific durability mode.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_durability(
        tx_id: TxId,
        snapshot: TransactionSnapshot,
        current: Arc<CurrentStorage>,
        historical: Arc<RwLock<HistoricalStorage>>,
        temporal_indexes: Arc<TemporalIndexes>,
        wal: Arc<ConcurrentWalSystem>,
        current_timestamp: Arc<CommitClock>,
        visibility_manager: Arc<TxVisibilityManager>,
        node_id_gen: Arc<IdGenerator>,
        edge_id_gen: Arc<IdGenerator>,
        version_id_gen: Arc<IdGenerator>,
        durability_mode: DurabilityMode,
    ) -> Self {
        Self::new_with_durability_and_clock_observed_at(
            tx_id,
            snapshot,
            current,
            historical,
            temporal_indexes,
            wal,
            current_timestamp,
            Arc::new(Mutex::new(Instant::now())),
            visibility_manager,
            node_id_gen,
            edge_id_gen,
            version_id_gen,
            durability_mode,
        )
    }

    /// Create a new write transaction with specific durability and clock observation state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_durability_and_clock_observed_at(
        tx_id: TxId,
        snapshot: TransactionSnapshot,
        current: Arc<CurrentStorage>,
        historical: Arc<RwLock<HistoricalStorage>>,
        temporal_indexes: Arc<TemporalIndexes>,
        wal: Arc<ConcurrentWalSystem>,
        current_timestamp: Arc<CommitClock>,
        commit_clock_observed_at: Arc<Mutex<Instant>>,
        visibility_manager: Arc<TxVisibilityManager>,
        node_id_gen: Arc<IdGenerator>,
        edge_id_gen: Arc<IdGenerator>,
        version_id_gen: Arc<IdGenerator>,
        durability_mode: DurabilityMode,
    ) -> Self {
        WriteTransaction {
            tx_id,
            start_timestamp: time::now(),
            state: TxState::Active,
            snapshot,
            buffer: WriteBuffer::new(),
            current,
            historical,
            temporal_indexes,
            wal,
            current_timestamp,
            commit_clock_observed_at,
            visibility_manager,
            node_id_gen,
            edge_id_gen,
            version_id_gen,
            durability_mode,
            constraint_registry: None,
            in_flight: None,
            role: None,
            cas_preconditions: Vec::new(),
            changefeed: None,
            #[cfg(feature = "audit-export")]
            sealing: None,
        }
    }

    /// Attach a constraint registry to this transaction (called by the DB layer).
    pub(crate) fn with_constraint_registry(
        mut self,
        registry: Arc<crate::core::constraint::ConstraintRegistry>,
    ) -> Self {
        self.constraint_registry = Some(registry);
        self
    }

    /// Attach the parent database's in-flight LSN tracker (called by the DB
    /// layer). Enables the durable-but-unapplied watermark used by index
    /// persistence to avoid lost writes across a crash.
    pub(crate) fn with_in_flight_tracker(mut self, tracker: Arc<InFlightLsns>) -> Self {
        self.in_flight = Some(tracker);
        self
    }

    /// Attach the parent database's node-role cell (Issue #3355, Slice A,
    /// called by the DB layer). Enables the commit-time promotion-race
    /// recheck in `commit_with_timestamp_inner`.
    pub(crate) fn with_role_cell(mut self, role: Arc<AtomicU8>) -> Self {
        self.role = Some(role);
        self
    }

    /// Attach the parent database's push-changefeed broadcaster (Issue #3375, called by
    /// the DB layer). When set, a successful commit broadcasts its committed change set to
    /// matching subscribers after the write is durable, applied, and visible.
    pub(crate) fn with_changefeed(
        mut self,
        broadcaster: Arc<crate::core::changefeed_subscription::ChangefeedBroadcaster>,
    ) -> Self {
        self.changefeed = Some(broadcaster);
        self
    }

    /// Attach the GDPR crypto-shred sealing context (Issue #3359, PR-1b, called
    /// by the DB layer when the database has active designations). When set, the
    /// apply hook seals designated property values before persisting them.
    #[cfg(feature = "audit-export")]
    pub(crate) fn with_sealing_context(
        mut self,
        sealing: crate::db::crypto_shred::SealingContext,
    ) -> Self {
        self.sealing = Some(sealing);
        self
    }

    /// Seal designated property values in the write buffer (Issue #3359, PR-1b).
    ///
    /// Called from `commit_with_timestamp_inner` AFTER validation + constraint
    /// checks (which must see plaintext values) but BEFORE WAL logging and apply,
    /// so the WAL segment, the current tier, and the historical tier all receive
    /// **byte-identical** sealed ciphertext (each envelope is sealed exactly once;
    /// its random per-encryption nonce is shared across every tier). No-op when no
    /// sealing context is attached (no active designation) or when no buffered
    /// entity carries a designated key.
    ///
    /// # Errors
    /// Propagates the seal error (fail-closed: a seal failure aborts the commit
    /// rather than persisting plaintext of erasable data).
    #[cfg(feature = "audit-export")]
    pub(crate) fn seal_buffer(&mut self) -> Result<()> {
        use super::BufferedWrite as BW;
        // Cheap clone (two Arc + a Copy enum) so we can borrow the buffer mutably.
        let Some(ctx) = self.sealing.clone() else {
            return Ok(());
        };
        for op in self.buffer.operations_mut() {
            let (is_node, entity_id) = match op {
                BW::CreateNode { node_id, .. } | BW::UpdateNode { node_id, .. } => {
                    (true, node_id.as_u64())
                }
                BW::CreateEdge { edge_id, .. } | BW::UpdateEdge { edge_id, .. } => {
                    (false, edge_id.as_u64())
                }
                // Deletes/retracts re-persist already-sealed bytes read back from
                // storage — no re-seal needed.
                _ => continue,
            };
            if let Some(props_ref) = op.properties_mut() {
                let props = std::mem::take(props_ref);
                *props_ref = ctx.seal_map(is_node, entity_id, props)?;
            }
        }
        Ok(())
    }

    /// Get transaction metadata.
    pub fn metadata(&self) -> TxMetadata {
        TxMetadata {
            tx_id: self.tx_id,
            start_timestamp: self.start_timestamp,
            commit_timestamp: None,
            state: self.state,
            is_read_only: false,
        }
    }

    /// Get transaction ID.
    pub fn tx_id(&self) -> TxId {
        self.tx_id
    }

    fn lock_adaptive_forward_jump_limit(
        &self,
        observed_at: Instant,
    ) -> Result<(std::sync::MutexGuard<'_, Instant>, i64)> {
        let previous_observed_at =
            self.commit_clock_observed_at
                .lock()
                .map_err(|_| TransactionError::LockPoisoned {
                    resource: "commit_clock_observed_at".to_string(),
                })?;
        let elapsed = observed_at.duration_since(*previous_observed_at);
        let elapsed_us = i64::try_from(elapsed.as_micros()).unwrap_or(i64::MAX);
        Ok((
            previous_observed_at,
            MAX_FORWARD_JUMP_US.saturating_add(elapsed_us),
        ))
    }

    /// Commit the transaction.
    ///
    /// This validates all buffered writes and applies them atomically
    /// to the storage. If validation fails or any operation fails,
    /// the transaction is rolled back.
    ///
    /// The durability behavior depends on the transaction's durability mode:
    /// - **Synchronous**: Waits for fsync (maximum durability, default)
    /// - **Async**: Returns after flush to OS cache (background thread syncs)
    /// - **GroupCommit**: Waits for batch fsync (ACID + high throughput)
    /// - **AsyncBatched**: Returns after flush to OS cache, batched fsync in background (<100µs latency)
    ///
    /// ## Examples
    ///
    /// ```rust,no_run
    /// # use aletheiadb::{AletheiaDB, PropertyMapBuilder, properties};
    /// # use aletheiadb::api::transaction::WriteOps;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = AletheiaDB::new()?;
    /// let mut tx = db.write_transaction()?;
    ///
    /// tx.create_node("Person", properties! { "name" => "Alice" })?;
    ///
    /// // Commit the changes to the database
    /// tx.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn commit(self) -> Result<()> {
        self.commit_with_timestamp().map(|_| ())
    }

    /// Commit the transaction and return the commit timestamp.
    ///
    /// This is useful for benchmarks and tests that need to query the database
    /// at the exact commit timestamp to verify temporal semantics.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use aletheiadb::{AletheiaDB, properties, api::transaction::WriteOps};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = AletheiaDB::new()?;
    /// # let properties = properties! { "name" => "Alice" };
    /// let mut tx = db.write_transaction()?;
    /// let node_id = tx.create_node("Person", properties)?;
    /// let commit_ts = tx.commit_with_timestamp()?;
    ///
    /// // Query at exact commit timestamp
    /// let node = db.get_node_at_time(node_id, commit_ts, commit_ts)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Durability Modes
    ///
    /// - **Synchronous**: WAL fsynced to disk before returning (ACID, ~1.5ms)
    /// - **Async**: Returns after flush to OS cache (background thread syncs)
    /// - **GroupCommit**: Waits for batch fsync (ACID + high throughput)
    /// - **AsyncBatched**: Returns after flush to OS cache, batched fsync in background (<100µs latency)
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn commit_with_timestamp(mut self) -> Result<Timestamp> {
        self.commit_with_timestamp_inner().record_error_metric()
    }

    fn commit_with_timestamp_inner(&mut self) -> Result<Timestamp> {
        #[cfg(feature = "observability")]
        let tx_id = self.tx_id.to_string();

        #[cfg(feature = "observability")]
        let durability_mode = format!("{:?}", self.durability_mode);

        #[cfg(feature = "observability")]
        let _span =
            crate::observability::transaction_commit_span(&tx_id, &durability_mode).entered();

        #[cfg(feature = "observability")]
        let commit_start = std::time::Instant::now();

        #[cfg(feature = "observability")]
        let operations_count = self.buffer.operations().len();

        // Check transaction state
        if self.state != TxState::Active {
            return Err(TransactionError::InvalidState {
                current: format!("{:?}", self.state),
                expected: "Active".to_string(),
            }
            .into());
        }

        // Read-only replica promotion/demotion-race guard (Issue #3355, Slice
        // A). `write_transaction`/`write_transaction_with_options` already
        // rejected before this transaction was constructed; this closes the
        // gap where `enter_replica_mode` runs after construction but before
        // commit. `None` (a transaction built without the role cell attached
        // -- legacy/low-level construction) skips the recheck, matching
        // pre-#3355 behavior.
        if let Some(role) = &self.role {
            crate::db::replication_role::reject_if_replica(role)?;
        }

        // Transition to preparing state
        self.state = TxState::Preparing;

        // Validate all buffered writes
        self.validate()?;

        // Detect write-write conflicts (Snapshot Isolation)
        self.detect_conflicts()?;

        // Enforce uniqueness constraints (reservation step).
        // Must sit before the timestamp lock so we never hold a DashMap shard
        // guard across `current_timestamp` (preserves lock-ordering contract).
        // The guard is intentionally kept alive (not prefixed with `_`) so its
        // Drop impl releases reserved keys on any subsequent error path.
        let constraint_guard = if let Some(ref registry) = self.constraint_registry {
            Some(
                constraint::check_constraints(self, registry)
                    .map_err(crate::core::error::Error::Constraint)?,
            )
        } else {
            None
        };

        // GDPR crypto-shred (Issue #3359, PR-1b): seal designated property values
        // in the buffer AFTER validation + constraint checks (which must see
        // plaintext) but BEFORE WAL logging + apply, so every tier (WAL segment,
        // current, historical) records byte-identical sealed ciphertext. Runs
        // outside the `current_timestamp` critical section, keeping the keyring
        // lock off the timestamp-held path. Fail-closed. No-op without an active
        // designation.
        #[cfg(feature = "audit-export")]
        self.seal_buffer()?;

        // Issue #3406: pre-generate the closing (delete tombstone / retraction)
        // version ids BEFORE WAL logging, so the SAME ids are both (a) recorded
        // in the WAL delete/retract payloads and (b) applied to historical
        // storage. This makes crash recovery reproduce the live version chain
        // bit-for-bit instead of synthesizing ids that can diverge from — or
        // collide with — later-logged version ids. Generated here (id
        // generators are lock-free atomics, order class 5) before the
        // `current_timestamp` lock is taken, preserving the lock-order contract.
        let closing_version_ids = self.pregenerate_closing_version_ids()?;

        // In-flight watermark guard (lost-write persist race fix). Declared in
        // the OUTER function scope so it lives past the `current_timestamp`
        // critical section and across `apply_changes` + finalization: the LSN is
        // registered AFTER the WAL append allocates it but BEFORE the durability
        // fsync (so a durable write is always registered), and this guard is only
        // dropped at end-of-function (after apply + finalize on the success path,
        // or on any early-return / panic path), so a *deregistered* LSN is
        // guaranteed present in any current+historical snapshot.
        let mut in_flight_guard: Option<in_flight::InFlightGuard> = None;

        // Issue #3375 (ordered emit, HIGH review fix): the push-changefeed emit ticket.
        // Reserved UNDER the `current_timestamp` lock below (right after the commit timestamp
        // is assigned) so its sequence totally orders commits by commit timestamp; consumed on
        // the success path by `EmitTicket::submit`. Declared in the OUTER scope so that if the
        // commit aborts/`?`-returns after reservation but before submit (a WAL failure, or the
        // Issue #3416/#3577 commit-time precondition guards — now run before the WAL append,
        // Issue #3413 — or a storage error during apply), dropping this `Option` releases an
        // empty slot for the reserved seq — the sequencer never stalls on a
        // reserved-but-never-submitted sequence.
        let mut emit_ticket: Option<crate::core::changefeed_subscription::EmitTicket> = None;

        // Acquire commit timestamp and perform mode-aware WAL flush.
        //
        // CRITICAL: We must hold the timestamp lock until WAL logging is complete
        // to prevent a race condition where transactions commit out-of-order.
        //
        // Timestamp management for Bi-Temporal Database:
        //
        // For temporal queries to work correctly, we MUST use wallclock timestamps
        // for transaction_time, not a logical clock. This allows querying historical
        // state at specific points in time (e.g., "what was the state at 2PM yesterday?").
        //
        // Monotonicity: Wallclock time is monotonically increasing (assuming NTP is working),
        // which satisfies the ordering requirements for Snapshot Isolation:
        // - commit_ts > snapshot_ts for transactions that started before this commit
        // - future snapshots will have timestamp >= this commit
        //
        // DURABILITY MODES (handled by ConcurrentWalSystem):
        // - Synchronous: Appends drain and flush immediately with fsync
        // - Async: Appends go to ring buffers, background thread syncs
        // - GroupCommit: Appends go to ring buffers, wait for epoch completion
        // Issue #3413 (WAL abort framing): hold `current_timestamp` across the
        // ENTIRE commit — the precondition guards (below, BEFORE the WAL append),
        // the WAL append + durability, apply, and finalize — released explicitly
        // after finalize. The guards read only current storage, so this single
        // held lock serializes commits end-to-end: a transaction the guards reject
        // returns with NO WAL frame, so crash recovery can never reapply a rejected
        // write. The documented lock order (`current_timestamp` → `wal` →
        // `historical`) is preserved — `wal` and `historical` are each acquired
        // only while holding `current_timestamp`, and `wal` is never appended under
        // `historical`.
        //
        // Test-only interleaving seam (Issue #3416 / #3413): fires here — after
        // `validate`/`detect_conflicts`/`check_constraints`, BEFORE the
        // `current_timestamp` lock — so a test can commit a concurrent
        // delete/create in this window (the victim holds no commit-clock lock yet)
        // and then observe the victim abort at its own commit-time write-skew
        // re-check under the lock. Production builds compile this away.
        #[cfg(test)]
        commit_test_hooks::run_pre_commit_clock_hook();

        #[cfg(feature = "observability")]
        let ts_lock_start = std::time::Instant::now();

        let mut ts = self
            .current_timestamp
            .lock()
            .map_err(|_| TransactionError::LockPoisoned {
                resource: "current_timestamp".to_string(),
            })?;

        #[cfg(feature = "observability")]
        let ts_lock_acquired = std::time::Instant::now();

        let commit_timestamp = {
            let observed_at = Instant::now();
            let (mut previous_observed_at, adaptive_forward_limit_us) =
                self.lock_adaptive_forward_jump_limit(observed_at)?;

            let self_heal_clock_skew = is_clock_skew_self_heal_enabled();

            // Assign the commit stamp and publish it to the allocation frontier
            // as one atomic step, marking the clock in-flight so lock-free
            // readers park on this stamp instead of reserving past it (see
            // `core::commit_clock`). The closure derives the stamp from
            // whatever frontier it is handed rather than from a value captured
            // earlier: a reader can reserve between this guard being taken and
            // the exchange landing, and the retry has to build on that
            // reservation instead of clobbering it.
            let commit = ts.advance_with(|frontier| {
                // Phase 2: Use HLC for distributed temporal consistency
                // Get current physical wallclock
                let current_wallclock = crate::core::temporal::time::now();

                let skew_decision = evaluate_clock_skew(
                    current_wallclock.wallclock(),
                    frontier.wallclock(),
                    Some(adaptive_forward_limit_us),
                    self_heal_clock_skew,
                )
                .map_err(|violation| TransactionError::ClockSkew {
                    wallclock: current_wallclock.wallclock(),
                    previous: frontier.wallclock(),
                    drift_us: violation.drift_us,
                    max_allowed: violation.max_allowed,
                })?;

                if self_heal_clock_skew && let Some(_direction) = skew_decision.healed_direction {
                    #[cfg(feature = "observability")]
                    tracing::warn!(
                        wallclock_ts = %current_wallclock,
                        prev_ts = %frontier,
                        drift_us = skew_decision.drift_us,
                        reason = _direction.as_str(),
                        "Self-healing clock skew by clamping to local HLC frontier"
                    );
                }

                // Phase 2: Use HLC .send() method for monotonic timestamp generation
                // This ensures: if wallclock advances, reset logical; otherwise increment logical
                let commit = send_with_overflow_self_heal(
                    &frontier,
                    skew_decision.effective_wallclock,
                    self_heal_clock_skew,
                    |error| match error {
                        SendWithSelfHealError::InitialSend(error) => {
                            TransactionError::CommitFailed {
                                reason: format!("HLC timestamp generation failed: {}", error),
                            }
                        }
                        SendWithSelfHealError::FallbackWallclockOverflow {
                            wallclock,
                            current_logical,
                        } => TransactionError::CommitFailed {
                            reason: format!(
                                "HLC logical counter overflow at wallclock={}: {}",
                                wallclock, current_logical
                            ),
                        },
                        SendWithSelfHealError::FallbackSend(fallback_error) => {
                            TransactionError::CommitFailed {
                                reason: format!(
                                    "HLC timestamp generation failed while self-healing: {}",
                                    fallback_error
                                ),
                            }
                        }
                    },
                )?;

                // Observability: Warn about clock skew issues
                #[cfg(feature = "observability")]
                {
                    // Clock went backwards: wallclock < previous wallclock
                    if current_wallclock.wallclock() < frontier.wallclock() {
                        tracing::warn!(
                            wallclock_ts = %current_wallclock,
                            prev_ts = %frontier,
                            skew_us = frontier.wallclock() - current_wallclock.wallclock(),
                            logical_counter = commit.logical(),
                            "Clock skew detected: wallclock went backwards (NTP adjustment?)"
                        );
                    } else if commit.wallclock() > frontier.wallclock() + 60_000_000 {
                        // Large forward jump (>60 seconds)
                        tracing::warn!(
                            wallclock_ts = %current_wallclock,
                            prev_ts = %frontier,
                            jump_us = commit.wallclock() - frontier.wallclock(),
                            "Large clock jump detected: timestamps will be lumpy"
                        );
                    }
                }

                Ok(commit)
            })?;

            // Persist observation only after we successfully advanced the frontier.
            *previous_observed_at = observed_at;
            drop(previous_observed_at);

            // Issue #3375 (ordered emit): reserve the push-changefeed emit ticket HERE, while
            // the `current_timestamp` lock is still held and monotonic ordering is guaranteed,
            // so the ticket's sequence == this commit's timestamp order == ChangeCursor order.
            // Gated on `has_subscribers()` so the zero-subscriber fast path stays a single
            // atomic load with no sequencer work (no reservation, no ticket). The reservation
            // itself is a lock-free `fetch_add`, so taking it under `current_timestamp` does
            // not violate the write-path lock order.
            if let Some(broadcaster) = &self.changefeed
                && broadcaster.has_subscribers()
            {
                emit_ticket = Some(broadcaster.reserve_emit_ticket());
            }

            commit
        };

        // Issue #3413 (WAL abort framing): run the commit-time precondition guards
        // HERE — under `current_timestamp`, BEFORE the WAL append — instead of
        // inside `apply_changes` after the frame is already durable. Each guard
        // reads only current storage (never `historical`), and `current_timestamp`
        // is held across `apply_changes` below, so a guard that passes now stays
        // valid through apply: no other committer can apply in between. A rejection
        // returns here with NO WAL frame appended, so a transaction refused at
        // runtime can never be reapplied by crash recovery. These are the three
        // previously-post-WAL rejection paths: the #3416 delete-orphan /
        // dangling-endpoint write-skew re-checks and the #3577/#3755
        // CAS/lease/fence precondition re-check.
        apply::detect_delete_orphan_write_skew(self)?;
        apply::detect_create_edge_dangling_endpoint(self)?;
        cas::detect_cas_precondition_violations(self, commit_timestamp)?;

        #[cfg(feature = "observability")]
        let wal_start = std::time::Instant::now();

        // Log operations to WAL (lock-free striped append!). Runs AFTER the guards
        // above (so a rejected transaction leaves no durable frame) and BEFORE
        // applying changes (for durability). Returns the base (lowest) LSN
        // allocated for this commit, if any.
        let base_lsn = wal::log_operations_to_wal(self, commit_timestamp, &closing_version_ids)?;

        // Register the commit's base LSN as in-flight BEFORE the durability fsync,
        // so a write that becomes durable is always registered. The guard survives
        // to end-of-function (see its declaration above).
        if let (Some(tracker), Some(base_lsn)) = (self.in_flight.as_ref(), base_lsn) {
            in_flight_guard = Some(tracker.register(base_lsn.0));
        }

        #[cfg(feature = "observability")]
        let wal_logged = std::time::Instant::now();

        // Commit with configured durability mode.
        // For Sync: drains and flushes immediately.
        // For Async: returns immediately.
        // For GroupCommit: registers and returns epoch.
        let wait_epoch = self.wal.commit()?;

        #[cfg(feature = "observability")]
        let wal_commit_completed = std::time::Instant::now();

        // NOTE: the GroupCommit flush wait used to happen HERE, inside the held
        // `current_timestamp` lock. That is what made group commit unable to
        // group: the first committer parked on its own fsync while still holding
        // the lock every other committer needs to reach the WAL at all, so no two
        // transactions could ever be registered-and-unflushed at the same time
        // and the batch size was structurally pinned at one. Measured, throughput
        // tracked `1 / max_delay_ms` exactly (92/s at 10ms, 176/s at 5ms, 664/s at
        // 1ms) and plain `Synchronous` beat it 45x. The wait now happens after
        // apply and outside the lock -- see below.

        #[cfg(feature = "observability")]
        {
            // Record detailed breakdown for tracing and metrics.
            let ts_lock_wait_us = ts_lock_acquired.duration_since(ts_lock_start).as_micros() as u64;
            let wal_log_us = wal_logged.duration_since(wal_start).as_micros() as u64;
            let wal_commit_us = wal_commit_completed.duration_since(wal_logged).as_micros() as u64;
            let total_us = wal_commit_completed
                .duration_since(ts_lock_start)
                .as_micros() as u64;

            let total_commit_us = commit_start.elapsed().as_micros() as u64;
            tracing::info!(
                ts_lock_wait_us,
                wal_log_us,
                wal_commit_us,
                total_us,
                total_commit_us,
                operations_count,
                commit_ts = %commit_timestamp,
                durability_mode = ?self.durability_mode,
                "Transaction commit breakdown (concurrent WAL)"
            );
        }

        // Apply all changes atomically, still holding `current_timestamp` (Issue
        // #3413): the guards above ran under this same held lock, so no other
        // committer can have applied in between and their results remain valid.
        // Nodes/edges are written with commit_timestamp: None during this phase.
        //
        // Issue #3425: `apply_changes` returns the `historical.write()` guard
        // instead of dropping it internally. We hold it across finalization below
        // so the current-side writes and their `commit_timestamp` finalization are
        // atomic w.r.t. any snapshot (checkpoint / backup) holding `historical.read()`
        // — closing the *in-memory* window in which a committed node/edge could be
        // captured with `commit_timestamp: None`. (The persisted index format does
        // not serialize `commit_timestamp` and reconstructs it as committed on
        // restore, so this fix hardens the in-memory snapshot invariant rather than
        // fixing any on-disk corruption.) This respects the documented lock order
        // (`historical` precedes `snapshot_lock` and adjacency), so no lock-order
        // inversion is introduced.
        //
        // The pre-generated closing version ids (Issue #3406) are consumed here
        // in the same buffer order they were logged to the WAL.

        // Always-compiled one-shot interleaving seam for the lost-write persist
        // race regression tests (which live in `tests/`). This fires at the
        // logged-but-not-yet-applied point — WAL appended and the in-flight LSN
        // registered, `current_timestamp` STILL held (Issue #3413). (Before the
        // flush wait moved out of this lock, this was also post-fsync; the
        // invariant the tests pin is about persist_indexes not observing a
        // logged-but-unapplied write, which is unchanged, and the window it
        // guards is now strictly narrower.) A racing
        // `persist_indexes()` briefly needs `current_timestamp` too (`db/admin.rs`,
        // to read the frontier + in-flight set consistently), so it SERIALIZES
        // behind a commit parked here rather than observing a durable-but-unapplied
        // write — the invariant the reworked tests pin. The tests therefore park
        // the commit on a SEPARATE thread and release it via a channel, so persist
        // blocks then proceeds (no self-deadlock); a persist driven on the SAME
        // thread as a parked commit WOULD hang, by design. In production the seam is
        // a single relaxed atomic load (unarmed), so it is effectively zero-cost.
        race_seam::run_pre_apply_once();

        let historical_guard = apply::apply_changes(self, commit_timestamp, &closing_version_ids)?;

        // Test-only interleaving hook (Issue #3425): fires at the exact point
        // between the historical writes and the commit-timestamp finalization,
        // while `historical_guard` is held. Production builds compile this away.
        #[cfg(test)]
        commit_test_hooks::run_pre_finalize_hook();

        // Finalize commit timestamps in current storage.
        // Only reached on the success path — sets commit_timestamp: Some(T) on all
        // written nodes/edges, making them visible to future snapshot readers.
        apply::finalize_current_commit_timestamps(self, commit_timestamp);

        // The write is now present in current AND historical storage, so it is
        // safe for a lock-free reader to be handed a snapshot strictly after it.
        // Publishing any earlier would let a reader consider this commit visible
        // (MVCC is `commit_ts < snapshot_ts`) and then fail to find it. Until
        // this call, readers park on `commit_timestamp` itself and treat it as
        // unseen -- see `core::commit_clock`.
        ts.publish_applied(commit_timestamp);

        // Finalization complete: release the historical write guard. Snapshots
        // blocked on `historical.read()` now proceed and observe resolved timestamps.
        //
        // Issue #3425: only `finalize_current_commit_timestamps` must run under the
        // guard to close the snapshot finalize window. The temporal-vector notify
        // below is deliberately kept OUTSIDE the guard: it touches only the vector
        // index's own locks and is independent of current-side commit_timestamp
        // finalization, so serializing its (potentially O(N log N)) HNSW snapshot
        // build under the global historical write lock would only add latency
        // without any correctness benefit.
        drop(historical_guard);

        // Issue #3413 (WAL abort framing): release `current_timestamp` now that the
        // precondition guards, WAL append + durability, apply, and finalize are all
        // complete. Holding it across the whole commit is what makes a guard that
        // passed pre-WAL still valid at apply time, so a rejected transaction never
        // leaves a durable frame for crash recovery to reapply. Everything below
        // (vector-index notify, changefeed broadcast, visibility registration) is
        // independent of commit serialization and runs off the held lock.
        drop(ts);

        // Durability, now that the commit lock is released (GroupCommit only --
        // `Synchronous` already fsynced inside `self.wal.commit()` above, and
        // `AsyncBatched` returns an epoch it deliberately does not wait on).
        //
        // Waiting here rather than under the lock is what lets group commit
        // actually group: N committers reach this point concurrently and collapse
        // into one fsync, instead of each parking on its own flush while holding
        // the lock the next one needs.
        //
        // Ordering this after apply is safe, and is what Postgres and friends do
        // (log the record, apply to memory, flush the log, only then acknowledge):
        //
        // - `commit()` still does not return until the write is durable, so the
        //   caller's ACID promise is unchanged.
        // - A crash between apply and flush loses the in-memory state too, and
        //   recovery correctly omits a frame that never reached disk. No caller
        //   was ever told the commit succeeded.
        // - A transaction that reads this write and commits durably cannot
        //   "overtake" it: the WAL is flushed in LSN order, so flushing through
        //   the reader's frame necessarily flushes this one first.
        //
        // What DOES change is the failure path. If the flush errors, this
        // transaction returns `Err` with its writes already applied and visible,
        // where previously the error arrived before apply and the writes stayed
        // invisible. That is the standard consequence of applying before flushing
        // (a WAL flush failure means the in-memory state is ahead of the log and
        // the database can no longer honour its own durability contract). It
        // affects GroupCommit only.
        if let Some(epoch) = wait_epoch
            && let Some(gc) = self.wal.group_commit_coordinator()
            && self.durability_mode.waits_for_durability()
        {
            gc.wait_for_flush(epoch)?;
        }

        // Durable AND applied: deregister the in-flight LSN so index persistence
        // no longer needs to hold the manifest watermark below it. Deregistering
        // strictly after BOTH finalize and the flush is what lets the watermark
        // guarantee "every deregistered LSN is durable and in the snapshot".
        drop(in_flight_guard.take());

        // Notify temporal vector index of transaction completion (for snapshot creation).
        // Only call this if the transaction modified vector properties to avoid unnecessary overhead.
        // No ordering dependency on finalize: the vectors were already added to the
        // index during `apply_changes`; this only triggers snapshot creation.
        if self.buffer.has_vector_operations() {
            self.current.on_temporal_vector_transaction()?;
        }

        // Commit the constraint guard before registering with the visibility manager.
        // Committing first ensures that freed old-value slots are released before
        // other transactions observe this commit as visible — otherwise a concurrent
        // delete+create of the same key could see the old reservation still present
        // and raise a spurious UniqueViolation.
        if let Some(guard) = constraint_guard {
            guard.commit();
        }

        // Register commit with visibility manager
        self.visibility_manager.register_commit(self.tx_id);

        // Mark as committed
        self.state = TxState::Committed;

        // Issue #3375: broadcast this transaction's committed change set to push-changefeed
        // subscribers. This runs on the committed path only (never for an aborted/rolled-back
        // transaction, which returns early before reaching here), and strictly AFTER the write
        // is durable (WAL fsynced), applied (current + historical), and visible
        // (`register_commit`). No write-path lock is held: the records are built via a fresh,
        // short `historical.read()` (a leaf acquisition — nothing later in the lock order is
        // held) whose guard is dropped before submitting.
        //
        // HIGH review fix (ordered emit): we consume the ticket reserved under the
        // `current_timestamp` lock. `submit` releases records to subscriber buffers in strict
        // reserved-sequence (== commit-timestamp == ChangeCursor) order, so per-subscriber
        // delivery is cursor-ascending even when concurrent commits finish their record-build
        // out of order — the precondition that makes `last_delivered` a zero-loss resume
        // high-water-mark. A slow subscriber cannot back-pressure this writer. We submit even
        // an empty record set so the reserved sequence is released and never stalls the
        // sequencer.
        if let Some(ticket) = emit_ticket.take() {
            let (node_vids, edge_vids) =
                self.committed_changefeed_version_ids(&closing_version_ids);
            let mut records: Vec<crate::core::changefeed::ChangeRecord> =
                if node_vids.is_empty() && edge_vids.is_empty() {
                    Vec::new()
                } else {
                    let hist = self.historical.read();
                    hist.collect_committed_changes(&node_vids, &edge_vids)
                        .into_iter()
                        .map(|raw| raw.into_record())
                        .collect()
                };
            // Deterministic #3216 total order (tx-time, kind, entity, version).
            records.sort_by_key(|r| r.cursor());
            ticket.submit(records);
        }

        #[cfg(feature = "observability")]
        {
            let total_commit_us = commit_start.elapsed().as_micros() as u64;
            crate::observability::record_transaction_commit(
                commit_start.elapsed().as_secs_f64(),
                operations_count as u64,
                &durability_mode,
                "committed",
            );
            tracing::debug!(
                total_commit_us,
                tx_id = %self.tx_id,
                "Transaction committed"
            );
        }

        Ok(commit_timestamp)
    }

    /// Pre-generate the closing version ids for every buffered delete/retract
    /// operation, in buffer order (Issue #3406).
    ///
    /// Delete tombstones and valid-time retractions each append exactly one
    /// closing version; both draw from the shared `version_id_gen`. Generating
    /// them here — before the WAL log phase — lets the exact same id be written
    /// into the WAL payload AND applied to historical storage, so crash
    /// recovery reproduces the live version chain instead of synthesizing a
    /// (potentially colliding) id.
    ///
    /// The returned vector has one id per delete/retract op in the buffer,
    /// ordered to match the iteration order used by both
    /// [`wal::log_operations_to_wal`] and [`apply::apply_changes`].
    fn pregenerate_closing_version_ids(&self) -> Result<Vec<VersionId>> {
        let num_closing = self
            .buffer
            .operations()
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    super::BufferedWrite::DeleteNode { .. }
                        | super::BufferedWrite::DeleteEdge { .. }
                        | super::BufferedWrite::RetractNode { .. }
                        | super::BufferedWrite::RetractEdge { .. }
                )
            })
            .count();

        let mut ids = Vec::with_capacity(num_closing);
        for _ in 0..num_closing {
            ids.push(VersionId::new_unchecked(self.version_id_gen.next()?));
        }
        Ok(ids)
    }

    /// Collect the node and edge version ids this transaction just committed, split by
    /// kind, for the push-changefeed broadcast (Issue #3375).
    ///
    /// Create/update ops carry their own `version_id`; delete/retract ops draw their
    /// closing (tombstone) `version_id` from `closing_version_ids` in buffer order — the
    /// exact order [`pregenerate_closing_version_ids`](Self::pregenerate_closing_version_ids)
    /// generated them and [`apply::apply_changes`] consumed them. Iterating the buffer in
    /// order and advancing a cursor into `closing_version_ids` for each delete/retract keeps
    /// the pairing correct.
    fn committed_changefeed_version_ids(
        &self,
        closing_version_ids: &[VersionId],
    ) -> (Vec<VersionId>, Vec<VersionId>) {
        use super::BufferedWrite as BW;
        let mut node_vids = Vec::new();
        let mut edge_vids = Vec::new();
        let mut closing_cursor = 0usize;
        let mut next_closing = || {
            let id = closing_version_ids.get(closing_cursor).copied();
            closing_cursor += 1;
            id
        };
        for op in self.buffer.operations() {
            match op {
                BW::CreateNode { version_id, .. } | BW::UpdateNode { version_id, .. } => {
                    node_vids.push(*version_id);
                }
                BW::CreateEdge { version_id, .. } | BW::UpdateEdge { version_id, .. } => {
                    edge_vids.push(*version_id);
                }
                BW::DeleteNode { .. } | BW::RetractNode { .. } => {
                    if let Some(id) = next_closing() {
                        node_vids.push(id);
                    }
                }
                BW::DeleteEdge { .. } | BW::RetractEdge { .. } => {
                    if let Some(id) = next_closing() {
                        edge_vids.push(id);
                    }
                }
            }
        }
        (node_vids, edge_vids)
    }

    /// Rollback the transaction.
    ///
    /// Discards all buffered writes. This is automatically called
    /// if the transaction is dropped without committing.
    ///
    /// ## Examples
    ///
    /// ```rust,no_run
    /// # use aletheiadb::{AletheiaDB, PropertyMapBuilder, properties};
    /// # use aletheiadb::api::transaction::WriteOps;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = AletheiaDB::new()?;
    /// let mut tx = db.write_transaction()?;
    ///
    /// tx.create_node("Person", properties! { "name" => "Alice" })?;
    ///
    /// // Discard the changes
    /// tx.rollback()?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn rollback(mut self) -> Result<()> {
        if self.state == TxState::Committed {
            return Err(TransactionError::AlreadyCommitted {
                tx_id: self.tx_id.as_u64(),
            }
            .into());
        }

        // Clear the write buffer
        self.buffer.clear();

        // Register abort with visibility manager
        self.visibility_manager.register_abort(self.tx_id);

        self.state = TxState::Aborted;

        Ok(())
    }

    /// Check if this transaction has any node writes (create, update, delete).
    pub(crate) fn has_node_writes(&self) -> bool {
        self.buffer
            .operations()
            .iter()
            .any(|op| op.is_node_operation())
    }

    /// Check if this transaction has any edge writes (create, update, delete).
    pub(crate) fn has_edge_writes(&self) -> bool {
        self.buffer
            .operations()
            .iter()
            .any(|op| op.is_edge_operation())
    }

    /// Check if this transaction has any vector property writes.
    pub(crate) fn has_vector_writes(&self) -> bool {
        self.buffer.has_vector_operations()
    }

    /// Validate all buffered writes.
    ///
    /// Checks:
    /// - Referential integrity (edges reference valid nodes)
    /// - No constraint violations
    fn validate(&self) -> Result<()> {
        validation::validate(self)
    }

    /// Detect write-write conflicts for Snapshot Isolation.
    ///
    /// Checks if any entity modified by this transaction has been committed
    /// by another transaction after our snapshot was taken. This implements
    /// the First-Committer-Wins rule of Snapshot Isolation.
    ///
    /// # Errors
    ///
    /// Returns `SerializationFailure` if a write-write conflict is detected.
    fn detect_conflicts(&self) -> Result<()> {
        conflict::detect_conflicts(self)
    }
}

impl WriteTransaction {
    /// Resolve a node purely from THIS transaction's write buffer, if it has a
    /// buffered write for it (read-your-own-writes, Issue #3417).
    ///
    /// Returns:
    /// - `Some(Ok(node))` for a buffered `CreateNode`/`UpdateNode` (the node's
    ///   in-flight state),
    /// - `Some(Err(NodeNotFound))` for a buffered `DeleteNode`/`RetractNode`
    ///   (removed in this tx),
    /// - `None` when the buffer holds no write for this id, so the caller
    ///   falls back to committed storage.
    ///
    /// This is the single source of buffer-resolution shared by the
    /// snapshot-isolated [`ReadOps::get_node`] and the write-path
    /// [`read_own_node`](Self::read_own_node); the two differ only in how they
    /// treat the committed fallback.
    fn buffered_node(&self, id: NodeId) -> Option<Result<Node>> {
        self.buffer
            .get_node_write(id)
            .and_then(|buffered| match buffered {
                super::BufferedWrite::CreateNode {
                    node_id,
                    label,
                    properties,
                    version_id,
                    ..
                }
                | super::BufferedWrite::UpdateNode {
                    node_id,
                    label,
                    properties,
                    version_id,
                    ..
                } => Some(Ok(Node::with_metadata(
                    *node_id,
                    *label,
                    properties.clone(),
                    *version_id,
                    VersionMetadata {
                        created_by_tx: self.tx_id,
                        commit_timestamp: None, // Not yet committed
                    },
                ))),
                super::BufferedWrite::DeleteNode { .. }
                | super::BufferedWrite::RetractNode { .. } => {
                    Some(Err(StorageError::NodeNotFound(id).into()))
                }
                _ => None, // Not a node operation
            })
    }

    /// Resolve an edge purely from THIS transaction's write buffer; see
    /// [`buffered_node`](Self::buffered_node).
    fn buffered_edge(&self, id: EdgeId) -> Option<Result<Edge>> {
        self.buffer
            .get_edge_write(id)
            .and_then(|buffered| match buffered {
                super::BufferedWrite::CreateEdge {
                    edge_id,
                    source,
                    target,
                    label,
                    properties,
                    version_id,
                    ..
                }
                | super::BufferedWrite::UpdateEdge {
                    edge_id,
                    source,
                    target,
                    label,
                    properties,
                    version_id,
                    ..
                } => Some(Ok(Edge::with_metadata(
                    *edge_id,
                    *label,
                    *source,
                    *target,
                    properties.clone(),
                    *version_id,
                    VersionMetadata {
                        created_by_tx: self.tx_id,
                        commit_timestamp: None,
                    },
                ))),
                super::BufferedWrite::DeleteEdge { .. }
                | super::BufferedWrite::RetractEdge { .. } => {
                    Some(Err(StorageError::EdgeNotFound(id).into()))
                }
                _ => None, // Not an edge operation
            })
    }

    /// Read-your-own-writes resolution for the mutating WRITE path (Issue
    /// #3417): consult this tx's buffer first, then fall back to the latest
    /// committed version via a **direct** current-storage read.
    ///
    /// Unlike [`ReadOps::get_node`], the committed fallback here does NOT apply
    /// snapshot-visibility filtering. The write path must operate on the latest
    /// committed version and relies on [`conflict::detect_conflicts`] for
    /// snapshot-isolation correctness: a version committed at or after this
    /// tx's snapshot must still be found so the update/delete/retract proceeds
    /// and the write-write conflict is caught at commit — filtering it out
    /// would spuriously 404 a live node. The buffer branch makes an entity
    /// created/updated earlier in THIS transaction visible to a later write.
    fn read_own_node(&self, id: NodeId) -> Result<Node> {
        if let Some(buffered) = self.buffered_node(id) {
            return buffered;
        }
        self.current.get_node(id)
    }

    /// Edge counterpart of [`read_own_node`](Self::read_own_node).
    fn read_own_edge(&self, id: EdgeId) -> Result<Edge> {
        if let Some(buffered) = self.buffered_edge(id) {
            return buffered;
        }
        self.current.get_edge(id)
    }

    /// Shared body for node replace (Issue #3549): buffer a full-overwrite
    /// `UpdateNode` carrying `label_interned` + the *exact* `properties` map
    /// (no PATCH merge). Callers pass an already-interned label so
    /// [`remove_node_property`](WriteOps::remove_node_property) can reuse the
    /// node's existing label without a (deprecated) interner `resolve`.
    ///
    /// Does not record the error metric — the public `WriteOps` entry points
    /// wrap the call and do so.
    fn buffer_node_replace(
        &mut self,
        node_id: NodeId,
        label_interned: crate::core::interning::InternedString,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<()> {
        // Check transaction state
        if self.state != TxState::Active {
            return Err(TransactionError::InvalidState {
                current: format!("{:?}", self.state),
                expected: "Active".to_string(),
            }
            .into());
        }

        // Verify the node exists (read-your-own-writes, Issue #3417) — a
        // replace targets an existing entity, never conjures one. Unlike PATCH
        // `update_node`, the existing map is NOT seeded into the new version:
        // `properties` becomes the node's map *exactly*, so any prior key it
        // omits is removed from current state (removal is encoded natively by
        // the anchor/delta diff and survives WAL replay). The label is
        // overwritten too.
        let existing = self.read_own_node(node_id)?;
        let version_id = VersionId::new_unchecked(self.version_id_gen.next()?);

        // Re-stamp the immutable namespace from the existing node (Issue #3349)
        // so a full overwrite (replace / remove-property) can never drop it.
        let properties = namespace::restamp_namespace(properties, &existing.properties);

        // A replace may add, change, OR remove vector properties; mark the
        // buffer so the temporal vector index is notified on commit if either
        // the old or the new state carries a vector.
        if !self.buffer.has_vector_operations()
            && (existing.properties.contains_vector() || properties.contains_vector())
        {
            self.buffer.mark_has_vector_operations();
        }

        // Get timestamp: use provided valid_from or default to transaction start time
        let timestamp = self.start_timestamp;
        let valid_from = options.valid_from.unwrap_or(timestamp);

        // Validate valid_from is not too far in future
        validation::validate_valid_from_future(valid_from)?;

        // Validate valid_from is not before entity creation.
        let creation_time = {
            let historical = self.historical.read();
            historical.node_creation_time(node_id)
        };
        if let Some(creation_time) = creation_time {
            validation::validate_valid_from_not_before_creation(
                &format!("node:{}", node_id.as_u64()),
                creation_time,
                valid_from,
            )?;
        }

        // Normalize an all-absent provenance bundle to `None` (Issue #3224).
        let provenance = options
            .provenance
            .filter(|p| !p.is_empty())
            .map(std::sync::Arc::new);

        // Buffer the FULL overwrite. Reuses the UpdateNode op (no new WAL
        // variant / format bump, Issue #3549): the exact target map plus a
        // possibly-new label is the payload.
        self.buffer.add(super::BufferedWrite::UpdateNode {
            node_id,
            version_id,
            label: label_interned,
            properties,
            valid_from,
            provenance,
        })?;

        Ok(())
    }

    /// Shared body for edge replace (Issue #3549): buffer a full-overwrite
    /// `UpdateEdge` carrying the *exact* `properties` map (no PATCH merge),
    /// preserving the already-read `edge`'s immutable source/target/type.
    /// Callers pass an already-read `edge` so
    /// [`remove_edge_property`](WriteOps::remove_edge_property) need not read it
    /// a second time (mirrors [`buffer_node_replace`](Self::buffer_node_replace)).
    ///
    /// Does not record the error metric — the public `WriteOps` entry points
    /// wrap the call and do so.
    fn buffer_edge_replace(
        &mut self,
        edge: Edge,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<()> {
        // Check transaction state
        if self.state != TxState::Active {
            return Err(TransactionError::InvalidState {
                current: format!("{:?}", self.state),
                expected: "Active".to_string(),
            }
            .into());
        }

        let edge_id = edge.id;
        let version_id = VersionId::new_unchecked(self.version_id_gen.next()?);

        // Re-stamp the immutable namespace from the existing edge (Issue #3349)
        // so a full overwrite (replace / remove-property) can never drop it.
        let properties = namespace::restamp_namespace(properties, &edge.properties);

        // A replace may add/change/remove vector properties.
        if !self.buffer.has_vector_operations()
            && (edge.properties.contains_vector() || properties.contains_vector())
        {
            self.buffer.mark_has_vector_operations();
        }

        // Get timestamp: use provided valid_from or default to transaction start time
        let timestamp = self.start_timestamp;
        let valid_from = options.valid_from.unwrap_or(timestamp);

        // Validate valid_from is not too far in future
        validation::validate_valid_from_future(valid_from)?;

        // Validate valid_from is not before the edge's own creation time
        let creation_time = {
            let historical = self.historical.read();
            historical.edge_creation_time(edge_id)
        };
        if let Some(creation_time) = creation_time {
            validation::validate_valid_from_not_before_creation(
                &format!("edge:{}", edge_id.as_u64()),
                creation_time,
                valid_from,
            )?;
        }

        // Normalize an all-absent provenance bundle to `None` (Issue #3224).
        let provenance = options
            .provenance
            .filter(|p| !p.is_empty())
            .map(std::sync::Arc::new);

        // Buffer the FULL overwrite via the existing UpdateEdge op (no WAL
        // format change, Issue #3549), preserving endpoints and type.
        self.buffer.add(super::BufferedWrite::UpdateEdge {
            edge_id,
            version_id,
            source: edge.source,
            target: edge.target,
            label: edge.label,
            properties,
            valid_from,
            provenance,
        })?;

        Ok(())
    }

    /// Enumerate edges CREATED earlier in THIS transaction that reference
    /// `node_id` as source or target and are still live in the buffer (Issue
    /// #3416 Pt4).
    ///
    /// `delete_node_cascade` unions this with committed adjacency
    /// (`self.current.get_outgoing_edges` / `get_incoming_edges`) so a same-tx
    /// `create_node → create_edge → cascade` reaches the same-tx-created edge,
    /// which committed adjacency cannot yet see. Only `CreateEdge` ops matter:
    /// an already-committed edge (including one this tx `UpdateEdge`s) is
    /// already returned by the current-adjacency read. An edge created then
    /// deleted/retracted within this tx is filtered out (its latest buffered
    /// state resolves to Not-Found via [`buffered_edge`](Self::buffered_edge)),
    /// so the cascade never double-closes it.
    fn buffered_connected_edges(&self, node_id: NodeId) -> Vec<EdgeId> {
        let mut edges = Vec::new();
        for op in self.buffer.operations() {
            if let super::BufferedWrite::CreateEdge {
                edge_id,
                source,
                target,
                ..
            } = op
                && (*source == node_id || *target == node_id)
                && matches!(self.buffered_edge(*edge_id), Some(Ok(_)))
            {
                edges.push(*edge_id);
            }
        }
        edges
    }
}

impl ReadOps for WriteTransaction {
    fn get_node(&self, id: NodeId) -> Result<Node> {
        let buffered_result = self.buffered_node(id);

        let result = if let Some(result) = buffered_result {
            result
        } else {
            // Fall back to snapshot-isolated read from storage
            match self.current.get_node(id) {
                Ok(node) => {
                    // Use embedded commit_timestamp for visibility check (Issue #238).
                    if !self.visibility_manager.is_visible_with_embedded_ts(
                        &self.snapshot,
                        node.metadata.created_by_tx,
                        node.metadata.commit_timestamp,
                    ) {
                        Err(StorageError::NodeNotFound(id).into())
                    } else {
                        Ok(node)
                    }
                }
                Err(err) => Err(err),
            }
        };

        result.record_error_metric()
    }

    fn get_edge(&self, id: EdgeId) -> Result<Edge> {
        let buffered_result = self.buffered_edge(id);

        let result = if let Some(result) = buffered_result {
            result
        } else {
            // Fall back to snapshot-isolated read from storage
            match self.current.get_edge(id) {
                Ok(edge) => {
                    // Use embedded commit_timestamp for visibility check (Issue #238).
                    if !self.visibility_manager.is_visible_with_embedded_ts(
                        &self.snapshot,
                        edge.metadata.created_by_tx,
                        edge.metadata.commit_timestamp,
                    ) {
                        Err(StorageError::EdgeNotFound(id).into())
                    } else {
                        Ok(edge)
                    }
                }
                Err(err) => Err(err),
            }
        };

        result.record_error_metric()
    }

    fn get_outgoing_edges(&self, node_id: NodeId) -> Result<Vec<EdgeId>> {
        // Existence check (Issue #359): consults the write buffer first (a node
        // created in this tx exists; a node deleted in this tx does not), then
        // falls back to a snapshot-isolated storage read.
        self.get_node(node_id)?;
        Ok(self.current.get_outgoing_edges(node_id))
    }

    fn get_incoming_edges(&self, node_id: NodeId) -> Result<Vec<EdgeId>> {
        // Existence check (Issue #359): see get_outgoing_edges.
        self.get_node(node_id)?;
        Ok(self.current.get_incoming_edges(node_id))
    }

    fn get_outgoing_edges_with_label(&self, node_id: NodeId, label: &str) -> Result<Vec<EdgeId>> {
        // Existence check (Issue #359): see get_outgoing_edges. An existing node
        // with no edges matching `label` is Ok(empty).
        self.get_node(node_id)?;
        Ok(self.current.get_outgoing_edges_with_label(node_id, label))
    }

    fn node_count(&self) -> usize {
        self.current.node_count()
    }

    fn edge_count(&self) -> usize {
        self.current.edge_count()
    }

    fn find_nodes_by_property(
        &self,
        label: &str,
        property_key: &str,
        property_value: &crate::core::property::PropertyValue,
    ) -> Vec<NodeId> {
        use crate::core::interning::GLOBAL_INTERNER;

        // 1. Get committed matches, excluding nodes modified in the buffer
        // ⚡ Bolt Optimization: Uses `.retain()` to filter in-place and avoid allocating a new `Vec`.
        let mut results = self
            .current
            .find_nodes_by_property(label, property_key, property_value);

        results.retain(|node_id| {
            !self.buffer.has_modified_node(*node_id) && {
                self.current
                    .get_node(*node_id)
                    .map(|node| {
                        // Use embedded commit_timestamp for visibility check (Issue #238).
                        self.visibility_manager.is_visible_with_embedded_ts(
                            &self.snapshot,
                            node.metadata.created_by_tx,
                            node.metadata.commit_timestamp,
                        )
                    })
                    .unwrap_or(false)
            }
        });

        // 2. Scan buffered writes for matching CreateNode/UpdateNode
        let label_id = GLOBAL_INTERNER.get_id(label);
        let key_id = GLOBAL_INTERNER.get_id(property_key);

        if let (Some(label_id), Some(key_id)) = (label_id, key_id) {
            for op in self.buffer.operations() {
                match op {
                    super::BufferedWrite::CreateNode {
                        node_id,
                        label: node_label,
                        properties,
                        ..
                    }
                    | super::BufferedWrite::UpdateNode {
                        node_id,
                        label: node_label,
                        properties,
                        ..
                    } => {
                        if *node_label == label_id
                            && let Some(val) = properties.get_by_interned_key(&key_id)
                            && val == property_value
                        {
                            results.push(*node_id);
                        }
                    }
                    // DeleteNode is already excluded by has_modified_node filter
                    _ => {}
                }
            }
        }

        results
    }
}

/// Buffered-version introspection (Issue #3231).
///
/// The `apply_batch` MCP surface reports a per-operation `version_id` for
/// creates and updates. Those version ids are allocated eagerly at op call
/// time and stored in the corresponding [`BufferedWrite`](super::BufferedWrite)
/// variant; these accessors surface them without touching storage or any
/// synchronization primitive — they are plain reads of the transaction's own
/// write buffer (no lock-order implications).
impl WriteTransaction {
    /// Version ID of the latest buffered create/update for `node_id`, if any.
    ///
    /// Returns `None` when the node has no buffered write in this transaction,
    /// or when its latest buffered write is a delete/retract (tombstone
    /// versions are allocated at apply time and are not surfaced here).
    pub fn buffered_node_version(&self, node_id: NodeId) -> Option<VersionId> {
        match self.buffer.get_node_write(node_id) {
            Some(
                super::BufferedWrite::CreateNode { version_id, .. }
                | super::BufferedWrite::UpdateNode { version_id, .. },
            ) => Some(*version_id),
            _ => None,
        }
    }

    /// Version ID of the latest buffered create/update for `edge_id`, if any.
    ///
    /// Returns `None` when the edge has no buffered write in this transaction,
    /// or when its latest buffered write is a delete/retract.
    pub fn buffered_edge_version(&self, edge_id: EdgeId) -> Option<VersionId> {
        match self.buffer.get_edge_write(edge_id) {
            Some(
                super::BufferedWrite::CreateEdge { version_id, .. }
                | super::BufferedWrite::UpdateEdge { version_id, .. },
            ) => Some(*version_id),
            _ => None,
        }
    }
}

impl WriteOps for WriteTransaction {
    fn create_node_with_options(
        &mut self,
        label: &str,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<NodeId> {
        let result = (|| {
            // Check transaction state
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // Reject any engine-reserved property key on the user-supplied map
            // BEFORE stamping the namespace (Issue #3349): a caller may not
            // forge/overwrite a `__aletheia_*` / `__shred_*` ride-along key.
            namespace::reject_reserved_keys(&properties)?;

            // Generate IDs
            let node_id = NodeId::new_unchecked(self.node_id_gen.next()?);
            let version_id = VersionId::new_unchecked(self.version_id_gen.next()?);
            let label_interned = GLOBAL_INTERNER.intern(label)?;

            // Stamp the namespace ride-along (Issue #3349). `None` ⇒ default,
            // which is deliberately NOT stamped (byte-identical to legacy data).
            let ns = options.namespace.clone().unwrap_or_default();
            let properties = namespace::stamp_namespace(properties, &ns);

            // Get timestamp: use provided valid_from or default to transaction start time
            let timestamp = self.start_timestamp;
            let valid_from = options.valid_from.unwrap_or(timestamp);

            // Validate valid_from is not too far in future
            validation::validate_valid_from_future(valid_from)?;

            // Normalize an all-absent provenance bundle to `None` -- never
            // persist a fabricated empty object (Issue #3224).
            let provenance = options
                .provenance
                .filter(|p| !p.is_empty())
                .map(std::sync::Arc::new);

            // Buffer the write
            self.buffer.add(super::BufferedWrite::CreateNode {
                node_id,
                version_id,
                label: label_interned,
                properties,
                valid_from,
                provenance,
            })?;

            Ok(node_id)
        })();

        result.record_error_metric()
    }

    fn create_edge_with_options(
        &mut self,
        source: NodeId,
        target: NodeId,
        label: &str,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<EdgeId> {
        let result = (|| {
            // Check transaction state
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // Reject engine-reserved keys before stamping the namespace (#3349).
            namespace::reject_reserved_keys(&properties)?;

            // Generate IDs
            let edge_id = EdgeId::new_unchecked(self.edge_id_gen.next()?);
            let version_id = VersionId::new_unchecked(self.version_id_gen.next()?);
            let label_interned = GLOBAL_INTERNER.intern(label)?;

            // Stamp the namespace ride-along (Issue #3349); default not stamped.
            let ns = options.namespace.clone().unwrap_or_default();
            let properties = namespace::stamp_namespace(properties, &ns);

            // Get timestamp: use provided valid_from or default to transaction start time
            let timestamp = self.start_timestamp;
            let valid_from = options.valid_from.unwrap_or(timestamp);

            // Validate valid_from is not too far in future
            validation::validate_valid_from_future(valid_from)?;

            // Normalize an all-absent provenance bundle to `None` -- never
            // persist a fabricated empty object (Issue #3224).
            let provenance = options
                .provenance
                .filter(|p| !p.is_empty())
                .map(std::sync::Arc::new);

            // Buffer the write
            self.buffer.add(super::BufferedWrite::CreateEdge {
                edge_id,
                version_id,
                source,
                target,
                label: label_interned,
                properties,
                valid_from,
                provenance,
            })?;

            Ok(edge_id)
        })();

        result.record_error_metric()
    }

    fn compare_and_set_node_with_options(
        &mut self,
        node_id: NodeId,
        expected_version: VersionId,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<VersionId> {
        self.cas_node_impl(node_id, expected_version, properties, None, None, options)
            .record_error_metric()
    }

    fn compare_and_set_edge_with_options(
        &mut self,
        edge_id: EdgeId,
        expected_version: VersionId,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<VersionId> {
        self.cas_edge_impl(edge_id, expected_version, properties, options)
            .record_error_metric()
    }

    #[allow(clippy::too_many_arguments)]
    fn claim_with_lease_with_options(
        &mut self,
        node_id: NodeId,
        expected_version: VersionId,
        lease_owner_key: &str,
        lease_until_key: &str,
        owner: PropertyValue,
        lease_until: Timestamp,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<VersionId> {
        self.claim_with_lease_impl(
            node_id,
            expected_version,
            lease_owner_key,
            lease_until_key,
            owner,
            lease_until,
            properties,
            options,
        )
        .record_error_metric()
    }

    #[allow(clippy::too_many_arguments)]
    fn claim_with_lease_fenced_with_options(
        &mut self,
        node_id: NodeId,
        expected_version: VersionId,
        lease_owner_key: &str,
        lease_until_key: &str,
        fence_key: &str,
        owner: PropertyValue,
        lease_ttl: std::time::Duration,
        new_fence: i64,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<VersionId> {
        self.claim_with_lease_fenced_impl(
            node_id,
            expected_version,
            lease_owner_key,
            lease_until_key,
            fence_key,
            owner,
            lease_ttl,
            new_fence,
            properties,
            options,
        )
        .record_error_metric()
    }

    fn update_node_with_options(
        &mut self,
        node_id: NodeId,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<()> {
        let result = (|| {
            // Check transaction state
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // A namespace is immutable after creation: reject an explicit
            // namespace on this update rather than silently ignoring it (#3349).
            namespace::reject_namespace_on_update(options.namespace.as_ref())?;

            // Reject engine-reserved keys on the incoming PATCH (Issue #3349):
            // a user may not set/overwrite a namespace/shred ride-along key.
            namespace::reject_reserved_keys(&properties)?;

            // Get current node to preserve label and existing properties.
            // Buffer-aware read (Issue #3417): read-your-own-writes so an
            // update of a node created/updated earlier in THIS transaction
            // merges onto the buffered state, not committed-only storage
            // (which would 404 a same-tx-created node).
            let node = self.read_own_node(node_id)?;
            let version_id = VersionId::new_unchecked(self.version_id_gen.next()?);

            // PATCH semantics: Merge new properties with existing ones
            // Start with existing properties
            let mut builder = PropertyMapBuilder::from_map(node.properties.clone());

            // Update/add properties from the incoming map
            for (key, value) in properties.iter() {
                builder = builder.insert_by_key(*key, value.clone());
            }

            // Build the final merged property map, then re-stamp the immutable
            // namespace from the existing node (Issue #3349). A PATCH inherently
            // preserves the ride-along key; re-stamping makes that a hard
            // guarantee that no update path can ever drop or change it.
            let merged_properties = namespace::restamp_namespace(builder.build(), &node.properties);

            // A PATCH may drop or downgrade a vector (e.g. overwrite the
            // embedding key with a scalar, or remove it): mark the buffer so
            // the temporal vector index snapshots on commit if EITHER the
            // existing node or the merged map carries a vector. `WriteBuffer::add`
            // only inspects the merged map, so without this a vector-removing
            // PATCH leaves `has_vector_operations` false and no snapshot fires,
            // leaking the phantom as-of-now (Issue #3621). Mirrors
            // `buffer_node_replace`.
            if !self.buffer.has_vector_operations()
                && (node.properties.contains_vector() || merged_properties.contains_vector())
            {
                self.buffer.mark_has_vector_operations();
            }

            // Get timestamp: use provided valid_from or default to transaction start time
            let timestamp = self.start_timestamp;
            let valid_from = options.valid_from.unwrap_or(timestamp);

            // Validate valid_from is not too far in future
            validation::validate_valid_from_future(valid_from)?;

            // Validate valid_from is not before entity creation.
            // A node created earlier in THIS transaction is not yet in
            // historical storage (that happens at commit), so
            // `node_creation_time` returns None and this check is skipped —
            // acceptable for v1 read-your-own-writes (Issue #3417).
            let creation_time = {
                let historical = self.historical.read();
                historical.node_creation_time(node_id)
            };
            if let Some(creation_time) = creation_time {
                validation::validate_valid_from_not_before_creation(
                    &format!("node:{}", node_id.as_u64()),
                    creation_time,
                    valid_from,
                )?;
            }

            // Normalize an all-absent provenance bundle to `None` -- never
            // persist a fabricated empty object (Issue #3224). This update's
            // provenance is independent of whatever the prior version carried.
            let provenance = options
                .provenance
                .filter(|p| !p.is_empty())
                .map(std::sync::Arc::new);

            // Buffer the write with merged properties
            self.buffer.add(super::BufferedWrite::UpdateNode {
                node_id,
                version_id,
                label: node.label,
                properties: merged_properties,
                valid_from,
                provenance,
            })?;

            Ok(())
        })();

        result.record_error_metric()
    }

    fn update_edge_with_options(
        &mut self,
        edge_id: EdgeId,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<()> {
        let result = (|| {
            // Check transaction state
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // A namespace is immutable after creation: reject an explicit
            // namespace on this update rather than silently ignoring it (#3349).
            namespace::reject_namespace_on_update(options.namespace.as_ref())?;

            // Reject engine-reserved keys on the incoming PATCH (Issue #3349).
            namespace::reject_reserved_keys(&properties)?;

            // Get current edge to preserve source, target, label and existing
            // properties. Buffer-aware read (Issue #3417): read-your-own-writes
            // so an update of a same-tx-created/updated edge merges onto the
            // buffered state instead of 404-ing against committed storage.
            let edge = self.read_own_edge(edge_id)?;
            let version_id = VersionId::new_unchecked(self.version_id_gen.next()?);

            // PATCH semantics: Merge new properties with existing ones
            // Start with existing properties
            let mut builder = PropertyMapBuilder::from_map(edge.properties.clone());

            // Update/add properties from the incoming map
            for (key, value) in properties.iter() {
                builder = builder.insert_by_key(*key, value.clone());
            }

            // Build the merged map, then re-stamp the immutable namespace from
            // the existing edge (Issue #3349).
            let merged_properties = namespace::restamp_namespace(builder.build(), &edge.properties);

            // Get timestamp: use provided valid_from or default to transaction start time
            let timestamp = self.start_timestamp;
            let valid_from = options.valid_from.unwrap_or(timestamp);

            // Validate valid_from is not too far in future
            validation::validate_valid_from_future(valid_from)?;

            // Validate valid_from is not before the edge's own creation time
            let creation_time = {
                let historical = self.historical.read();
                historical.edge_creation_time(edge_id)
            };
            if let Some(creation_time) = creation_time {
                validation::validate_valid_from_not_before_creation(
                    &format!("edge:{}", edge_id.as_u64()),
                    creation_time,
                    valid_from,
                )?;
            }

            // Normalize an all-absent provenance bundle to `None` -- never
            // persist a fabricated empty object (Issue #3224). This update's
            // provenance is independent of whatever the prior version carried.
            let provenance = options
                .provenance
                .filter(|p| !p.is_empty())
                .map(std::sync::Arc::new);

            // Buffer the write with merged properties
            self.buffer.add(super::BufferedWrite::UpdateEdge {
                edge_id,
                version_id,
                source: edge.source,
                target: edge.target,
                label: edge.label,
                properties: merged_properties,
                valid_from,
                provenance,
            })?;

            Ok(())
        })();

        result.record_error_metric()
    }

    fn replace_node_with_options(
        &mut self,
        node_id: NodeId,
        label: &str,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<()> {
        let result = (|| {
            // Check transaction state BEFORE interning so a replace on an
            // inactive tx never interns the label (mirrors
            // `update_node_with_options`; `buffer_node_replace` re-checks state
            // for the `remove_node_property` caller).
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // A namespace is immutable after creation: reject an explicit
            // namespace on this replace rather than silently ignoring it (#3349).
            namespace::reject_namespace_on_update(options.namespace.as_ref())?;

            // Reject engine-reserved keys on the user-supplied overwrite map
            // (Issue #3349); the immutable namespace is re-stamped from the
            // existing node inside `buffer_node_replace`.
            namespace::reject_reserved_keys(&properties)?;

            let label_interned = GLOBAL_INTERNER.intern(label)?;
            self.buffer_node_replace(node_id, label_interned, properties, options)
        })();

        result.record_error_metric()
    }

    fn replace_edge_with_options(
        &mut self,
        edge_id: EdgeId,
        properties: PropertyMap,
        options: WriteRequestOptions,
    ) -> Result<()> {
        let result = (|| {
            // Check transaction state
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // A namespace is immutable after creation: reject an explicit
            // namespace on this replace rather than silently ignoring it (#3349).
            namespace::reject_namespace_on_update(options.namespace.as_ref())?;

            // Reject engine-reserved keys on the user-supplied overwrite map
            // (Issue #3349); the immutable namespace is re-stamped from the
            // existing edge inside `buffer_edge_replace`.
            namespace::reject_reserved_keys(&properties)?;

            // Verify the edge exists and capture its immutable endpoints/type
            // (read-your-own-writes, Issue #3417). Full overwrite of properties
            // only: source/target/label are preserved from the existing edge.
            let edge = self.read_own_edge(edge_id)?;
            self.buffer_edge_replace(edge, properties, options)
        })();

        result.record_error_metric()
    }

    fn remove_node_property(&mut self, node_id: NodeId, key: &str) -> Result<()> {
        // An engine-reserved key (the namespace ride-along, #3349) is not a
        // user property and may not be targeted for removal.
        if namespace::is_reserved_property_key(key) {
            return Err(namespace::NamespaceError::ReservedPropertyKey {
                key: key.to_string(),
            }
            .into())
            .record_error_metric();
        }
        // Check transaction state up front so an absent-key no-op on an
        // inactive transaction still reports the state error.
        if self.state != TxState::Active {
            return Err(TransactionError::InvalidState {
                current: format!("{:?}", self.state),
                expected: "Active".to_string(),
            }
            .into())
            .record_error_metric();
        }

        // Read-your-own-writes current map.
        let node = match self.read_own_node(node_id) {
            Ok(node) => node,
            Err(e) => return Err(e).record_error_metric(),
        };

        // Absent key: no-op success, record NO new version (Issue #3549).
        if !node.properties.contains_key(key) {
            return Ok(());
        }

        // Drop the key and replace with the reduced map, preserving the node's
        // existing (already-interned) label — no interner `resolve` needed.
        let new_properties = PropertyMapBuilder::from_map(node.properties.clone())
            .remove(key)
            .build();

        self.buffer_node_replace(
            node_id,
            node.label,
            new_properties,
            WriteRequestOptions::default(),
        )
        .record_error_metric()
    }

    fn remove_edge_property(&mut self, edge_id: EdgeId, key: &str) -> Result<()> {
        // An engine-reserved key (#3349) is not a user property.
        if namespace::is_reserved_property_key(key) {
            return Err(namespace::NamespaceError::ReservedPropertyKey {
                key: key.to_string(),
            }
            .into())
            .record_error_metric();
        }
        if self.state != TxState::Active {
            return Err(TransactionError::InvalidState {
                current: format!("{:?}", self.state),
                expected: "Active".to_string(),
            }
            .into())
            .record_error_metric();
        }

        let edge = match self.read_own_edge(edge_id) {
            Ok(edge) => edge,
            Err(e) => return Err(e).record_error_metric(),
        };

        // Absent key: no-op success, record NO new version.
        if !edge.properties.contains_key(key) {
            return Ok(());
        }

        let new_properties = PropertyMapBuilder::from_map(edge.properties.clone())
            .remove(key)
            .build();

        // Reuse the already-read edge (NIT: avoid a second `read_own_edge`).
        // `buffer_edge_replace` preserves endpoints/type; record the error
        // metric here since the helper does not.
        self.buffer_edge_replace(edge, new_properties, WriteRequestOptions::default())
            .record_error_metric()
    }

    fn delete_node_with_options(
        &mut self,
        node_id: NodeId,
        options: WriteRequestOptions,
    ) -> Result<()> {
        let result = (|| {
            // Check transaction state
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // Verify node exists and check for vector properties.
            // Buffer-aware read (Issue #3417): a node created earlier in THIS
            // transaction is deletable (read-your-own-writes).
            let node = self.read_own_node(node_id)?;

            // If the node being deleted contains vector properties, mark the buffer
            // to ensure the temporal vector index is notified on commit
            if !self.buffer.has_vector_operations() && node.properties.contains_vector() {
                self.buffer.mark_has_vector_operations();
            }

            // Get timestamp: use provided valid_from or default to transaction start time
            let timestamp = self.start_timestamp;
            let valid_from = options.valid_from.unwrap_or(timestamp);

            // Validate valid_from is not too far in future
            validation::validate_valid_from_future(valid_from)?;

            // Validate valid_from is not before entity creation
            let creation_time = {
                let historical = self.historical.read();
                historical.node_creation_time(node_id)
            };
            if let Some(creation_time) = creation_time {
                validation::validate_valid_from_not_before_creation(
                    &format!("node:{}", node_id.as_u64()),
                    creation_time,
                    valid_from,
                )?;
            }

            // Normalize an all-absent provenance bundle to `None` -- never
            // persist a fabricated empty object (Issue #3224/#3427).
            let provenance = options
                .provenance
                .filter(|p| !p.is_empty())
                .map(std::sync::Arc::new);

            // Buffer the write
            self.buffer.add(super::BufferedWrite::DeleteNode {
                node_id,
                valid_from,
                provenance,
            })?;

            Ok(())
        })();

        result.record_error_metric()
    }

    fn delete_node_cascade_with_options(
        &mut self,
        node_id: NodeId,
        options: WriteRequestOptions,
    ) -> Result<()> {
        // Check transaction state
        if self.state != TxState::Active {
            return Err(TransactionError::InvalidState {
                current: format!("{:?}", self.state),
                expected: "Active".to_string(),
            }
            .into());
        }

        // Verify node exists before attempting deletion.
        // Buffer-aware read (Issue #3417): a node created earlier in THIS
        // transaction can be cascade-deleted (read-your-own-writes).
        let _node = self.read_own_node(node_id)?;

        // Collect all edges connected to this node (both outgoing and incoming).
        // We do this before any deletions to avoid borrowing issues.
        //
        // Buffer-aware cascade adjacency (Issue #3416 Pt4, folded from the
        // #3417 review): the committed adjacency (get_outgoing_edges /
        // get_incoming_edges) is unioned with edges CREATED earlier in THIS
        // transaction (buffered CreateEdge touching this node), minus edges
        // already deleted/retracted in this tx. Without the buffer union a
        // same-tx create_node → create_edge → cascade would leave the
        // same-tx-created edge orphaned (its endpoint gets a tombstone but the
        // edge does not).
        let mut edge_ids = self.get_outgoing_edges(node_id)?;
        edge_ids.extend(self.get_incoming_edges(node_id)?);
        edge_ids.extend(self.buffered_connected_edges(node_id));

        // Dedup by EdgeId (Issue #3416 Pt3): a self-loop (node -> node) appears
        // in BOTH the outgoing and incoming lists; without dedup delete_edge
        // would be called twice for the same edge, buffering two DeleteEdge ops
        // (double tombstone / double-close at apply time). Sort + dedup means
        // each distinct connected edge is deleted exactly once. This mirrors
        // retract_node_detach and apply_batch's manual cascade.
        edge_ids.sort_unstable();
        edge_ids.dedup();

        // Delete all connected edges first to maintain referential integrity.
        // This prevents orphaned edges that reference a deleted node.
        // Performance: O(degree) where degree is the number of connected edges.
        //
        // Issue #3427: stamp the SAME options (backdated valid_from and/or the
        // acting principal's provenance) onto every co-deleted edge's tombstone,
        // not just the node — a cascade delete attributes every tombstone it
        // creates. `options` is cloned per edge; the node takes the final move.
        for edge_id in edge_ids {
            self.delete_edge_with_options(edge_id, options.clone())?;
        }

        // Finally, delete the node itself
        // This is safe now because all edges referencing this node have been removed
        self.delete_node_with_options(node_id, options)?;

        Ok(())
    }

    fn delete_edge_with_options(
        &mut self,
        edge_id: EdgeId,
        options: WriteRequestOptions,
    ) -> Result<()> {
        let result = (|| {
            // Check transaction state
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // Verify edge exists and check for vector properties.
            // Buffer-aware read (Issue #3417): an edge created earlier in THIS
            // transaction is deletable (read-your-own-writes).
            let edge = self.read_own_edge(edge_id)?;

            // If the edge being deleted contains vector properties, mark the buffer
            // to ensure the temporal vector index is notified on commit
            if !self.buffer.has_vector_operations() && edge.properties.contains_vector() {
                self.buffer.mark_has_vector_operations();
            }

            // Get timestamp: use provided valid_from or default to transaction start time
            let timestamp = self.start_timestamp;
            let valid_from = options.valid_from.unwrap_or(timestamp);

            // Validate valid_from is not too far in future
            validation::validate_valid_from_future(valid_from)?;

            // Validate valid_from is not before the edge's own creation time
            let creation_time = {
                let historical = self.historical.read();
                historical.edge_creation_time(edge_id)
            };
            if let Some(creation_time) = creation_time {
                validation::validate_valid_from_not_before_creation(
                    &format!("edge:{}", edge_id.as_u64()),
                    creation_time,
                    valid_from,
                )?;
            }

            // Normalize an all-absent provenance bundle to `None` -- never
            // persist a fabricated empty object (Issue #3224/#3427).
            let provenance = options
                .provenance
                .filter(|p| !p.is_empty())
                .map(std::sync::Arc::new);

            // Buffer the write
            self.buffer.add(super::BufferedWrite::DeleteEdge {
                edge_id,
                valid_from,
                provenance,
            })?;

            Ok(())
        })();

        result.record_error_metric()
    }
}

/// Valid-time retraction (Issue #3230).
///
/// Retraction closes an entity's valid-time interval at `valid_to` without
/// deleting its history. It is a first-class bi-temporal operation, distinct
/// from deletion:
///
/// - The pre-retraction head version's **transaction time** is closed at the
///   retraction's commit time while its valid interval stays open-ended, so
///   `AS OF SYSTEM_TIME` queries positioned before the retraction still show
///   the fact as currently valid (append-only — the past record is never
///   rewritten).
/// - A **new version** is appended with the same properties and the closed
///   valid interval `[valid_from, valid_to)`, so `AS OF VALID_TIME` queries
///   strictly before `valid_to` keep returning the entity, while queries at
///   or after `valid_to` do not.
/// - The entity is removed from **current storage**: like a delete, a
///   retracted entity is absent from current-state queries — but its full
///   history remains queryable.
impl WriteTransaction {
    /// Buffer a valid-time retraction of a node.
    ///
    /// Idempotency: retracting an already-retracted (or deleted) node is a
    /// no-op that returns the *existing* `valid_to` (with
    /// `already_retracted: true`) — no version is appended and no WAL entry
    /// is written, regardless of the `valid_to` passed here. This holds for
    /// writes already buffered in THIS transaction too: a second
    /// `retract_node` of the same node returns the first call's buffered
    /// `valid_to`, and retracting a node this transaction has already
    /// `delete_node`-ed mirrors the committed-delete case (a degenerate
    /// `[d, d)` interval where `d` is the buffered deletion's valid time)
    /// rather than double-buffering a write.
    ///
    /// # Errors
    ///
    /// - [`TemporalError::ValidTimeBeforeEntityCreation`](crate::core::error::TemporalError::ValidTimeBeforeEntityCreation)
    ///   if `valid_to` precedes the node's current `valid_from`
    ///   (`valid_to == valid_from` is allowed and yields an empty interval).
    /// - [`TemporalError::ValidTimeTooFarInFuture`](crate::core::error::TemporalError::ValidTimeTooFarInFuture)
    ///   if `valid_to` is more than one year in the future.
    /// - [`StorageError::NodeNotFound`]
    ///   if the node never existed.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn retract_node(
        &mut self,
        node_id: NodeId,
        valid_to: Timestamp,
    ) -> Result<super::RetractionResult> {
        self.retract_node_with_provenance(node_id, valid_to, None)
    }

    /// Buffer a valid-time retraction of a node, stamping a write-time
    /// [`Provenance`] bundle recording the
    /// acting principal onto the retraction version (Issue #3427).
    ///
    /// Behaves identically to [`retract_node`](Self::retract_node) other than
    /// persisting `provenance` on the appended retraction version. Passing
    /// `None` is exactly [`retract_node`](Self::retract_node). An all-absent
    /// bundle is normalized to no provenance (never a fabricated empty object).
    /// On the idempotent no-op path (already retracted/deleted) no version is
    /// appended, so the provenance is not recorded — matching the "no new
    /// version, no WAL entry" contract.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn retract_node_with_provenance(
        &mut self,
        node_id: NodeId,
        valid_to: Timestamp,
        provenance: Option<Provenance>,
    ) -> Result<super::RetractionResult> {
        let result = (|| {
            // Check transaction state
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // A retraction or deletion already buffered in THIS transaction
            // is a no-op (mirrors the committed already-retracted/deleted
            // cases below): without this guard a double retract — or a
            // delete followed by a retract — would buffer two writes for
            // the same node, WAL-appending twice and risking a half-applied
            // state if the commit fails between them.
            match self.buffer.get_node_write(node_id) {
                Some(super::BufferedWrite::RetractNode {
                    valid_to: buffered_valid_to,
                    ..
                }) => {
                    let buffered_valid_to = *buffered_valid_to;
                    let valid_from = self.head_node_valid_from(node_id);
                    return Ok(super::RetractionResult {
                        valid_from,
                        valid_to: buffered_valid_to,
                        already_retracted: true,
                        edges_retracted: 0,
                    });
                }
                Some(super::BufferedWrite::DeleteNode { valid_from, .. }) => {
                    // Delete-parity: a committed delete leaves a tombstone
                    // with the degenerate interval [d, d); report the same
                    // shape for a deletion still sitting in this buffer.
                    let deleted_at = *valid_from;
                    return Ok(super::RetractionResult {
                        valid_from: deleted_at,
                        valid_to: deleted_at,
                        already_retracted: true,
                        edges_retracted: 0,
                    });
                }
                _ => {}
            }

            // Buffer-aware read (Issue #3417): a node created earlier in THIS
            // transaction is retractable (read-your-own-writes). The buffered
            // Retract/Delete cases are already handled by the guard above; a
            // buffered Create/Update falls through to here.
            match self.read_own_node(node_id) {
                Ok(node) => {
                    // If the node being retracted contains vector properties, mark
                    // the buffer so the temporal vector index is notified on commit.
                    if !self.buffer.has_vector_operations() && node.properties.contains_vector() {
                        self.buffer.mark_has_vector_operations();
                    }

                    // The closed interval starts at the head version's
                    // valid_from; `valid_to` must not precede it.
                    let valid_from = self.head_node_valid_from(node_id);

                    validation::validate_valid_from_future(valid_to)?;
                    validation::validate_valid_from_not_before_creation(
                        &format!("node:{}", node_id.as_u64()),
                        valid_from,
                        valid_to,
                    )?;

                    // Normalize an all-absent provenance bundle to `None` --
                    // never persist a fabricated empty object (Issue #3224/#3427).
                    let provenance = provenance
                        .filter(|p| !p.is_empty())
                        .map(std::sync::Arc::new);

                    self.buffer.add(super::BufferedWrite::RetractNode {
                        node_id,
                        valid_to,
                        provenance,
                    })?;

                    Ok(super::RetractionResult {
                        valid_from,
                        valid_to,
                        already_retracted: false,
                        edges_retracted: 0,
                    })
                }
                Err(err) => {
                    // Absent from current state: either already retracted or
                    // deleted (idempotent no-op) or it never existed.
                    if let Some(interval) = self.closed_node_valid_interval(node_id) {
                        return Ok(super::RetractionResult {
                            valid_from: interval.start(),
                            valid_to: interval.end(),
                            already_retracted: true,
                            edges_retracted: 0,
                        });
                    }
                    Err(err)
                }
            }
        })();

        result.record_error_metric()
    }

    /// Buffer a valid-time retraction of an edge.
    ///
    /// See [`retract_node`](Self::retract_node) for the bi-temporal
    /// semantics, idempotency contract, and errors (with
    /// [`StorageError::EdgeNotFound`]
    /// for a never-existing edge).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn retract_edge(
        &mut self,
        edge_id: EdgeId,
        valid_to: Timestamp,
    ) -> Result<super::RetractionResult> {
        self.retract_edge_with_provenance(edge_id, valid_to, None)
    }

    /// Buffer a valid-time retraction of an edge, stamping a write-time
    /// [`Provenance`] bundle recording the
    /// acting principal onto the retraction version (Issue #3427).
    ///
    /// See [`retract_node_with_provenance`](Self::retract_node_with_provenance)
    /// for the provenance-normalization and idempotency contract, and
    /// [`retract_edge`](Self::retract_edge) for the bi-temporal semantics.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn retract_edge_with_provenance(
        &mut self,
        edge_id: EdgeId,
        valid_to: Timestamp,
        provenance: Option<Provenance>,
    ) -> Result<super::RetractionResult> {
        let result = (|| {
            // Check transaction state
            if self.state != TxState::Active {
                return Err(TransactionError::InvalidState {
                    current: format!("{:?}", self.state),
                    expected: "Active".to_string(),
                }
                .into());
            }

            // A retraction already buffered in THIS transaction (e.g. the
            // detach form visiting a self-loop from both adjacency
            // directions) is a no-op returning the buffered end; a deletion
            // already buffered in this transaction mirrors the
            // committed-delete case (degenerate [d, d) tombstone interval)
            // — see the identical guard in retract_node.
            match self.buffer.get_edge_write(edge_id) {
                Some(super::BufferedWrite::RetractEdge {
                    valid_to: buffered_valid_to,
                    ..
                }) => {
                    let buffered_valid_to = *buffered_valid_to;
                    let valid_from = self.head_edge_valid_from(edge_id);
                    return Ok(super::RetractionResult {
                        valid_from,
                        valid_to: buffered_valid_to,
                        already_retracted: true,
                        edges_retracted: 0,
                    });
                }
                Some(super::BufferedWrite::DeleteEdge { valid_from, .. }) => {
                    let deleted_at = *valid_from;
                    return Ok(super::RetractionResult {
                        valid_from: deleted_at,
                        valid_to: deleted_at,
                        already_retracted: true,
                        edges_retracted: 0,
                    });
                }
                _ => {}
            }

            // Buffer-aware read (Issue #3417): an edge created earlier in THIS
            // transaction is retractable (read-your-own-writes).
            match self.read_own_edge(edge_id) {
                Ok(edge) => {
                    if !self.buffer.has_vector_operations() && edge.properties.contains_vector() {
                        self.buffer.mark_has_vector_operations();
                    }

                    let valid_from = self.head_edge_valid_from(edge_id);

                    validation::validate_valid_from_future(valid_to)?;
                    validation::validate_valid_from_not_before_creation(
                        &format!("edge:{}", edge_id.as_u64()),
                        valid_from,
                        valid_to,
                    )?;

                    // Normalize an all-absent provenance bundle to `None` --
                    // never persist a fabricated empty object (Issue #3224/#3427).
                    let provenance = provenance
                        .filter(|p| !p.is_empty())
                        .map(std::sync::Arc::new);

                    self.buffer.add(super::BufferedWrite::RetractEdge {
                        edge_id,
                        valid_to,
                        provenance,
                    })?;

                    Ok(super::RetractionResult {
                        valid_from,
                        valid_to,
                        already_retracted: false,
                        edges_retracted: 0,
                    })
                }
                Err(err) => {
                    if let Some(interval) = self.closed_edge_valid_interval(edge_id) {
                        return Ok(super::RetractionResult {
                            valid_from: interval.start(),
                            valid_to: interval.end(),
                            already_retracted: true,
                            edges_retracted: 0,
                        });
                    }
                    Err(err)
                }
            }
        })();

        result.record_error_metric()
    }

    /// The head version's `valid_from` for a node, falling back to the
    /// transaction start time when the node has no recorded versions (e.g.
    /// storage assembled without historical bookkeeping in tests).
    fn head_node_valid_from(&self, node_id: NodeId) -> Timestamp {
        let historical = self.historical.read();
        historical
            .get_current_node_version(node_id)
            .and_then(|vid| historical.get_node_version(vid))
            .map(|v| v.temporal.valid_time().start())
            .unwrap_or(self.start_timestamp)
    }

    /// The head version's `valid_from` for an edge; see
    /// [`head_node_valid_from`](Self::head_node_valid_from).
    fn head_edge_valid_from(&self, edge_id: EdgeId) -> Timestamp {
        let historical = self.historical.read();
        historical
            .get_current_edge_version(edge_id)
            .and_then(|vid| historical.get_edge_version(vid))
            .map(|v| v.temporal.valid_time().start())
            .unwrap_or(self.start_timestamp)
    }

    /// If the node's head version carries a CLOSED valid interval (it was
    /// retracted, or deleted via a tombstone), return that interval.
    fn closed_node_valid_interval(
        &self,
        node_id: NodeId,
    ) -> Option<crate::core::temporal::TimeRange> {
        let historical = self.historical.read();
        historical
            .get_current_node_version(node_id)
            .and_then(|vid| historical.get_node_version(vid))
            .map(|v| v.temporal.valid_time())
            .filter(|valid| !valid.is_current())
    }

    /// Edge counterpart of
    /// [`closed_node_valid_interval`](Self::closed_node_valid_interval).
    fn closed_edge_valid_interval(
        &self,
        edge_id: EdgeId,
    ) -> Option<crate::core::temporal::TimeRange> {
        let historical = self.historical.read();
        historical
            .get_current_edge_version(edge_id)
            .and_then(|vid| historical.get_edge_version(vid))
            .map(|v| v.temporal.valid_time())
            .filter(|valid| !valid.is_current())
    }
}

impl Drop for WriteTransaction {
    fn drop(&mut self) {
        // Auto-rollback if the transaction never reached a terminal state.
        //
        // `Preparing` must be released here too (Issue #3415): a commit that
        // fails at any pre-`register_commit` step (validation, conflict,
        // constraint, WAL, or apply) leaves the consumed transaction in
        // `Preparing`. Without covering that state, its `TxId` would never be
        // removed from `TxVisibilityManager::active`, leaking one entry per
        // failed commit and pinning the snapshot horizon forever. The success
        // path transitions `Preparing -> Committed` before drop, so a committed
        // transaction never matches and is never double-aborted; an explicit
        // `rollback()`/`abort()` sets `Aborted` and likewise does not match.
        if matches!(self.state, TxState::Active | TxState::Preparing) {
            self.buffer.clear();
            // Register abort with visibility manager
            self.visibility_manager.register_abort(self.tx_id);
            self.state = TxState::Aborted;
        }
    }
}

/// Always-compiled one-shot commit-path interleaving seam for the lost-write
/// persist-race regression tests.
///
/// Unlike [`commit_test_hooks`] (which is `#[cfg(test)]` and thus invisible to
/// the crate's `tests/` integration tests), this seam is always compiled and
/// `#[doc(hidden)] pub`, so an integration test can arm it. It lets a test park
/// exactly one commit at the **durable-but-not-yet-applied** point of the commit
/// path (WAL fsynced + acknowledged, in-flight LSN registered, `apply_changes`
/// not yet called) while it drives a racing `persist_indexes()`.
///
/// # Cost
///
/// The fire path ([`run_pre_apply_once`]) is a single acquire atomic load when
/// unarmed (the production state), so it adds no measurable cost to the commit
/// hot path — the "an `Option` that's `None`" contract, implemented as an
/// `AtomicBool` that is `false` in production.
///
/// # Arming surface is feature-gated
///
/// The arming API (`arm_pre_apply_once`/`clear`, the `HOOK` static, and the
/// armed branch that locks `HOOK`) is compiled ONLY under
/// `#[cfg(any(test, feature = "testing"))]`. In a default build there is no way
/// to set `ARMED = true`, so [`run_pre_apply_once`] reduces to its inert fast
/// path (`if !ARMED { return; }`) and can never park the commit path — it is
/// inert by construction, not merely by convention. A downstream crate that
/// links the library without the `testing` feature cannot reach the parking
/// surface.
///
/// # One-shot
///
/// `arm_pre_apply_once` installs a hook fired by the **next** commit only; the
/// fire path atomically disarms before running the hook, so under concurrency
/// exactly one commit fires it. `clear` disarms without firing (test teardown).
#[doc(hidden)]
pub mod race_seam {
    #[cfg(any(test, feature = "testing"))]
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[cfg(any(test, feature = "testing"))]
    type Hook = Box<dyn FnOnce() + Send>;

    /// Fast-path guard: `true` only while a hook is armed. Kept separate from
    /// the `Mutex` so the unarmed fire path never touches the lock. In a default
    /// (non-`testing`) build nothing can ever set this to `true`.
    static ARMED: AtomicBool = AtomicBool::new(false);
    #[cfg(any(test, feature = "testing"))]
    static HOOK: Mutex<Option<Hook>> = Mutex::new(None);

    /// Arm a one-shot hook fired by the next commit at the durable-but-unapplied
    /// point. Intended for regression tests only; only compiled under
    /// `#[cfg(any(test, feature = "testing"))]`.
    #[cfg(any(test, feature = "testing"))]
    pub fn arm_pre_apply_once(hook: Box<dyn FnOnce() + Send>) {
        if let Ok(mut slot) = HOOK.lock() {
            *slot = Some(hook);
            ARMED.store(true, Ordering::Release);
        }
    }

    /// Disarm any pending hook without firing it (test teardown). Only compiled
    /// under `#[cfg(any(test, feature = "testing"))]`.
    #[cfg(any(test, feature = "testing"))]
    pub fn clear() {
        ARMED.store(false, Ordering::Release);
        if let Ok(mut slot) = HOOK.lock() {
            *slot = None;
        }
    }

    /// Fire and consume the armed hook exactly once. Called on the commit hot
    /// path just before `apply_changes`. Always compiled, but in a default build
    /// `ARMED` can never be `true` (the arming surface is gated out), so this is
    /// just the inert fast-path load.
    #[inline]
    pub(crate) fn run_pre_apply_once() {
        // Fast path: one acquire atomic load; `false` in production (the arming
        // surface is gated out of default builds, so `ARMED` is always `false`
        // and this whole body is inert by construction).
        if ARMED.load(Ordering::Acquire) {
            // Disarm atomically so concurrent commits never double-fire. This
            // branch is unreachable without the arming surface (test/testing only).
            #[cfg(any(test, feature = "testing"))]
            if ARMED.swap(false, Ordering::AcqRel) {
                let hook = HOOK.lock().ok().and_then(|mut slot| slot.take());
                if let Some(hook) = hook {
                    hook();
                }
            }
        }
    }
}

/// Test-only interleaving hooks for the commit path (Issue #3425).
///
/// These let a test park a committing writer at a precise point — between the
/// historical writes and the commit-timestamp finalization — so a concurrent
/// snapshot (checkpoint / backup) can be driven into the exact window the
/// hardening closes. The entire module is `#[cfg(test)]`, so it is compiled
/// out of production builds and adds zero cost to the commit hot path.
#[cfg(test)]
pub(crate) mod commit_test_hooks {
    use std::sync::{Arc, Mutex};

    type Hook = Arc<dyn Fn() + Send + Sync>;

    static PRE_FINALIZE_HOOK: Mutex<Option<Hook>> = Mutex::new(None);

    /// Install a hook fired just before `finalize_current_commit_timestamps`.
    pub(crate) fn set_pre_finalize_hook(hook: Hook) {
        *PRE_FINALIZE_HOOK
            .lock()
            .expect("pre-finalize hook mutex poisoned") = Some(hook);
    }

    /// Remove any installed pre-finalize hook.
    pub(crate) fn clear_pre_finalize_hook() {
        *PRE_FINALIZE_HOOK
            .lock()
            .expect("pre-finalize hook mutex poisoned") = None;
    }

    /// Invoke the installed pre-finalize hook, if any.
    ///
    /// The `Arc` is cloned out from under the mutex so the hook body never runs
    /// while holding the registry lock (the hook may block for a long time).
    pub(crate) fn run_pre_finalize_hook() {
        let hook = PRE_FINALIZE_HOOK
            .lock()
            .expect("pre-finalize hook mutex poisoned")
            .clone();
        if let Some(hook) = hook {
            hook();
        }
    }

    static PRE_COMMIT_CLOCK_HOOK: Mutex<Option<Hook>> = Mutex::new(None);

    /// Install a hook fired just BEFORE the committing transaction acquires the
    /// `current_timestamp` lock — i.e. after `validate`/`detect_conflicts`/
    /// `check_constraints` but before the WAL abort-framing critical section
    /// (Issue #3413). This is the seam that drives the #3416 write-skew re-checks
    /// deterministically now that `current_timestamp` is held across apply: a test
    /// parks the victim here (holding no commit-path lock yet) while a concurrent
    /// delete/create commits, so the victim then aborts at its own commit-time
    /// re-check under `current_timestamp`.
    ///
    /// Firing HERE (rather than after the WAL append, as the removed pre-apply
    /// hook did) is what avoids a self-deadlock: `current_timestamp` is NOT yet
    /// held, so the concurrent committer can acquire it and commit.
    pub(crate) fn set_pre_commit_clock_hook(hook: Hook) {
        *PRE_COMMIT_CLOCK_HOOK
            .lock()
            .expect("pre-commit-clock hook mutex poisoned") = Some(hook);
    }

    /// Remove any installed pre-commit-clock hook.
    pub(crate) fn clear_pre_commit_clock_hook() {
        *PRE_COMMIT_CLOCK_HOOK
            .lock()
            .expect("pre-commit-clock hook mutex poisoned") = None;
    }

    /// Invoke the installed pre-commit-clock hook, if any (Arc cloned out from
    /// under the mutex, as with `run_pre_finalize_hook`).
    pub(crate) fn run_pre_commit_clock_hook() {
        let hook = PRE_COMMIT_CLOCK_HOOK
            .lock()
            .expect("pre-commit-clock hook mutex poisoned")
            .clone();
        if let Some(hook) = hook {
            hook();
        }
    }
}

#[cfg(test)]
mod tests;
