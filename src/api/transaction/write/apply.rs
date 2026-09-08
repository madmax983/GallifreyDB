//! Apply buffered writes to storage.
//!
//! This module handles the "Apply" phase of the commit protocol:
//! 1.  Update current state (fast path)
//! 2.  Archive previous state to historical storage (temporal path)
//! 3.  Update temporal indexes (time travel path)
//!
//! # Atomicity
//!
//! Operations in this module are executed after validation and conflict detection
//! have passed. They must be applied atomically (all or nothing). Since we are
//! in the commit phase, we assume success unless a catastrophic storage failure
//! occurs.
//!
//! # Temporal Semantics
//!
//! - **Valid Time**: Controlled by the user (or defaults to commit time).
//! - **Transaction Time**: Controlled by the system (always commit time).
//! - **Tombstones**: Deletions create "tombstone" versions that close the valid time interval.
//!
//! # Lock Acquisition Order
//!
//! Write commits acquire synchronization in this order:
//! 1. `current_timestamp`
//! 2. `wal`
//! 3. `historical`
//! 4. `temporal_indexes`
//! 5. `id generators`
//! 6. `outgoing`
//! 7. `incoming`
//!
//! `commit()` handles `current_timestamp` and `wal` before entering this module.
//! `apply_changes()` then acquires `historical` once for the whole batch. The
//! current `temporal_indexes`, `id generators`, `outgoing`, and `incoming`
//! adjacency storage use atomic or fine-grained internal synchronization, but the
//! ordering is still the contract for future explicit locks: do not acquire an
//! earlier entry while holding a later one.

use super::WriteTransaction;
use crate::core::error::{Result, StorageError};
use crate::core::graph::{Edge, Node};
use crate::core::id::{EdgeId, NodeId, VersionId};
use crate::core::interning::InternedString;
use crate::core::property::PropertyMap;
use crate::core::provenance::Provenance;
use crate::core::temporal::{BiTemporalInterval, Timestamp};
use crate::core::version::VersionMetadata;
use crate::storage::historical::HistoricalStorage;
use std::sync::Arc;

/// Helper function to create a bi-temporal interval with proper closing logic.
///
/// For standard versions, the interval is open-ended `[start, infinity)`.
/// For tombstones (deletions), the valid-time interval is closed immediately `[start, start)`,
/// representing that the entity is no longer valid from that point onward.
#[inline]
pub(crate) fn create_temporal_interval(
    valid_from: Timestamp,
    tx_time: Timestamp,
    is_tombstone: bool,
) -> Result<BiTemporalInterval> {
    let mut temporal = BiTemporalInterval::with_valid_time(valid_from, tx_time);
    if is_tombstone {
        temporal = temporal.close_valid_time(valid_from)?;
    }
    Ok(temporal)
}

/// Apply a node creation or update to storage.
///
/// This performs three actions:
/// 1.  Updates `CurrentStorage` with the new node state.
/// 2.  Adds a new version to `HistoricalStorage`.
/// 3.  Updates `TemporalIndexes` for time-travel queries.
///
/// If this is an update, it also closes the transaction time of the previous version
/// in historical storage to maintain history continuity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_node_write(
    tx: &WriteTransaction,
    is_create: bool,
    node_id: NodeId,
    version_id: VersionId,
    label: InternedString,
    properties: PropertyMap,
    valid_from: Timestamp,
    commit_timestamp: Timestamp,
    historical: &mut HistoricalStorage,
    provenance: Option<Arc<Provenance>>,
) -> Result<()> {
    // GDPR crypto-shred (Issue #3359, PR-1b): `properties` here already carry any
    // sealed `SUBJ` envelopes — sealing happens once on the write buffer in
    // `commit_with_timestamp_inner` (post-validation, pre-WAL), so the WAL
    // segment, the current tier below, and the historical tier below all receive
    // byte-identical ciphertext. Do NOT seal again here (it would double-seal).

    // Create node with pending metadata (commit_timestamp finalized after full apply_changes).
    // Using uncommitted here prevents phantom visibility if apply_changes fails partway through.
    let metadata = VersionMetadata::uncommitted(tx.tx_id);
    let node = Node::with_metadata(node_id, label, properties.clone(), version_id, metadata);

    // Insert or update in current storage
    if is_create {
        tx.current.insert_node_direct(node, commit_timestamp)?;
    } else {
        tx.current.update_node_direct(node, commit_timestamp)?;

        if let Some(current_version_id) = historical.get_current_node_version(node_id) {
            historical.close_node_version_transaction_time(current_version_id, commit_timestamp)?;
        }
    }

    // Store in historical storage (consume properties, avoiding second clone)
    historical.add_node_version_with_provenance(
        node_id,
        version_id,
        valid_from,
        commit_timestamp,
        label,
        properties,
        false, // not a tombstone
        provenance,
    )?;

    // Index in temporal indexes with bi-temporal interval
    let temporal = create_temporal_interval(valid_from, commit_timestamp, false)?;
    tx.temporal_indexes
        .insert_node_version(node_id, version_id, temporal)?;

    Ok(())
}

/// Apply an edge creation or update to storage.
///
/// Similar to `apply_node_write`, this updates current storage, historical storage,
/// and temporal indexes.
///
/// # Referential Integrity
///
/// This function assumes that `source` and `target` nodes exist. Referential integrity
/// checks should be performed during the validation phase (see `validation::validate`),
/// not during application.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_edge_write(
    tx: &WriteTransaction,
    is_create: bool,
    edge_id: EdgeId,
    version_id: VersionId,
    source: NodeId,
    target: NodeId,
    label: InternedString,
    properties: PropertyMap,
    valid_from: Timestamp,
    commit_timestamp: Timestamp,
    historical: &mut HistoricalStorage,
    provenance: Option<Arc<Provenance>>,
) -> Result<()> {
    // GDPR crypto-shred (Issue #3359, PR-1b): `properties` already carry any
    // sealed envelopes (sealed on the write buffer in `commit_with_timestamp_inner`
    // before WAL logging). Do NOT seal again here.

    // Create edge with pending metadata (commit_timestamp finalized after full apply_changes).
    let metadata = VersionMetadata::uncommitted(tx.tx_id);
    let edge = Edge::with_metadata(
        edge_id,
        label,
        source,
        target,
        properties.clone(),
        version_id,
        metadata,
    );

    // Insert or update in current storage
    if is_create {
        tx.current.insert_edge_direct(edge)?;
    } else {
        tx.current.update_edge_direct(edge)?;

        if let Some(current_version_id) = historical.get_current_edge_version(edge_id) {
            historical.close_edge_version_transaction_time(current_version_id, commit_timestamp)?;
        }
    }

    // Store in historical storage (consume properties, avoiding second clone)
    historical.add_edge_version_with_provenance(
        edge_id,
        version_id,
        valid_from,
        commit_timestamp,
        label,
        source,
        target,
        properties,
        false, // not a tombstone
        provenance,
    )?;

    // Index in temporal indexes with bi-temporal interval
    let temporal = create_temporal_interval(valid_from, commit_timestamp, false)?;
    tx.temporal_indexes
        .insert_edge_version(edge_id, version_id, temporal)?;

    Ok(())
}

/// Apply a node deletion.
///
/// Deletion involves:
/// 1.  Closing the current version in historical storage.
/// 2.  Creating a "tombstone" version in historical storage to mark the end of validity.
/// 3.  Removing the node from current storage.
///
/// # Tombstones
///
/// A tombstone is a version with `is_tombstone = true`. Its valid-time interval
/// is effectively empty `[valid_from, valid_from)`, indicating that the entity
/// ceases to exist at that point.
pub(crate) fn apply_node_delete(
    tx: &WriteTransaction,
    node_id: NodeId,
    valid_from: Timestamp,
    commit_timestamp: Timestamp,
    tombstone_id: VersionId,
    historical: &mut HistoricalStorage,
    provenance: Option<Arc<Provenance>>,
) -> Result<()> {
    // Get the node before deleting
    let node = tx.current.get_node(node_id)?;

    // Close the current version's transaction_time in historical storage
    if let Some(current_version_id) = historical.get_current_node_version(node_id) {
        historical.close_node_version_transaction_time(current_version_id, commit_timestamp)?;
    }

    // Create tombstone interval using centralized logic
    let tombstone_temporal = create_temporal_interval(valid_from, commit_timestamp, true)?;

    // Add tombstone version to historical storage, stamping the acting
    // principal's provenance onto the tombstone (Issue #3427).
    historical.add_node_version_with_provenance(
        node_id,
        tombstone_id,
        valid_from,
        commit_timestamp,
        node.label,
        node.properties.clone(),
        true, // is_tombstone
        provenance,
    )?;

    // Index the tombstone version with the same interval
    tx.temporal_indexes
        .insert_node_version(node_id, tombstone_id, tombstone_temporal)?;

    // Delete from current storage
    tx.current.delete_node_direct(node_id, commit_timestamp)?;

    Ok(())
}

/// Apply an edge deletion.
///
/// Similar to `apply_node_delete`, creates a tombstone and removes from current storage.
pub(crate) fn apply_edge_delete(
    tx: &WriteTransaction,
    edge_id: EdgeId,
    valid_from: Timestamp,
    commit_timestamp: Timestamp,
    tombstone_id: VersionId,
    historical: &mut HistoricalStorage,
    provenance: Option<Arc<Provenance>>,
) -> Result<()> {
    // Get the edge before deleting
    let edge = tx.current.get_edge(edge_id)?;

    // Close the current version's transaction_time in historical storage
    if let Some(current_version_id) = historical.get_current_edge_version(edge_id) {
        historical.close_edge_version_transaction_time(current_version_id, commit_timestamp)?;
    }

    // Create tombstone interval using centralized logic
    let tombstone_temporal = create_temporal_interval(valid_from, commit_timestamp, true)?;

    // Add tombstone version to historical storage, stamping the acting
    // principal's provenance onto the tombstone (Issue #3427).
    historical.add_edge_version_with_provenance(
        edge_id,
        tombstone_id,
        valid_from,
        commit_timestamp,
        edge.label,
        edge.source,
        edge.target,
        edge.properties.clone(),
        true, // is_tombstone
        provenance,
    )?;

    // Index the tombstone version with the same interval
    tx.temporal_indexes
        .insert_edge_version(edge_id, tombstone_id, tombstone_temporal)?;

    // Delete from current storage
    tx.current.delete_edge_direct(edge_id)?;

    Ok(())
}

/// Apply a node retraction (Issue #3230).
///
/// Retraction closes the node's valid-time interval at `valid_to` without
/// deleting its history:
/// 1.  Closes the head version's TRANSACTION time at the commit timestamp,
///     leaving its valid interval untouched (still open-ended) — this is
///     what makes `AS OF SYSTEM_TIME` before the retraction show the fact
///     open-ended (append-only, never rewrite).
/// 2.  Appends a NEW version with the same properties and the closed valid
///     interval `[valid_from, valid_to)` (tx time `[commit, infinity)`).
/// 3.  Removes the node from current storage and updates temporal indexes.
///
/// After retraction the node is absent from current-state queries, like a
/// delete, while the full history remains queryable.
pub(crate) fn apply_node_retract(
    tx: &WriteTransaction,
    node_id: NodeId,
    valid_to: Timestamp,
    commit_timestamp: Timestamp,
    retraction_version_id: VersionId,
    historical: &mut HistoricalStorage,
    provenance: Option<Arc<Provenance>>,
) -> Result<()> {
    // Get the node before removing it from current storage
    let node = tx.current.get_node(node_id)?;

    // Capture the head's valid_from BEFORE mutating, so the retraction
    // version reproduces the same interval start.
    let head_version_id = historical.get_current_node_version(node_id);
    let valid_from = head_version_id
        .and_then(|vid| historical.get_node_version(vid))
        .map(|v| v.temporal.valid_time().start())
        .unwrap_or(commit_timestamp);

    // (1) Close the head version's transaction time only.
    if let Some(current_version_id) = head_version_id {
        historical.close_node_version_transaction_time(current_version_id, commit_timestamp)?;
    }

    // (2) Append the retraction version with the closed valid interval,
    // stamping the acting principal's provenance onto it (Issue #3427).
    historical.add_retracted_node_version_with_provenance(
        node_id,
        retraction_version_id,
        valid_from,
        valid_to,
        commit_timestamp,
        node.label,
        node.properties.clone(),
        provenance,
    )?;

    // Index the retraction version with the same closed interval.
    let temporal = BiTemporalInterval::with_valid_time(valid_from, commit_timestamp)
        .close_valid_time(valid_to)?;
    tx.temporal_indexes
        .insert_node_version(node_id, retraction_version_id, temporal)?;

    // (3) Remove from current storage.
    tx.current.delete_node_direct(node_id, commit_timestamp)?;

    Ok(())
}

/// Apply an edge retraction (Issue #3230).
///
/// See [`apply_node_retract`] for the bi-temporal contract.
pub(crate) fn apply_edge_retract(
    tx: &WriteTransaction,
    edge_id: EdgeId,
    valid_to: Timestamp,
    commit_timestamp: Timestamp,
    retraction_version_id: VersionId,
    historical: &mut HistoricalStorage,
    provenance: Option<Arc<Provenance>>,
) -> Result<()> {
    // Get the edge before removing it from current storage
    let edge = tx.current.get_edge(edge_id)?;

    let head_version_id = historical.get_current_edge_version(edge_id);
    let valid_from = head_version_id
        .and_then(|vid| historical.get_edge_version(vid))
        .map(|v| v.temporal.valid_time().start())
        .unwrap_or(commit_timestamp);

    // (1) Close the head version's transaction time only.
    if let Some(current_version_id) = head_version_id {
        historical.close_edge_version_transaction_time(current_version_id, commit_timestamp)?;
    }

    // (2) Append the retraction version with the closed valid interval,
    // stamping the acting principal's provenance onto it (Issue #3427).
    historical.add_retracted_edge_version_with_provenance(
        edge_id,
        retraction_version_id,
        valid_from,
        valid_to,
        commit_timestamp,
        edge.label,
        edge.source,
        edge.target,
        edge.properties.clone(),
        provenance,
    )?;

    let temporal = BiTemporalInterval::with_valid_time(valid_from, commit_timestamp)
        .close_valid_time(valid_to)?;
    tx.temporal_indexes
        .insert_edge_version(edge_id, retraction_version_id, temporal)?;

    // (3) Remove from current storage.
    tx.current.delete_edge_direct(edge_id)?;

    Ok(())
}

/// Apply a single buffered write operation.
///
/// Dispatches to the appropriate `apply_*` function based on the operation type.
///
/// # Arguments
///
/// * `tombstone_ids` - Iterator of pre-generated Version IDs for tombstones. This optimization
///   avoids acquiring the ID generator lock for every delete operation.
pub(crate) fn apply_single_write(
    tx: &WriteTransaction,
    write: &crate::api::transaction::BufferedWrite,
    commit_timestamp: Timestamp,
    historical: &mut HistoricalStorage,
    tombstone_ids: &mut impl Iterator<Item = u64>,
    num_deletes: usize,
) -> Result<()> {
    match write {
        crate::api::transaction::BufferedWrite::CreateNode {
            node_id,
            version_id,
            label,
            properties,
            valid_from,
            provenance,
        }
        | crate::api::transaction::BufferedWrite::UpdateNode {
            node_id,
            version_id,
            label,
            properties,
            valid_from,
            provenance,
        } => {
            let is_create = matches!(
                write,
                crate::api::transaction::BufferedWrite::CreateNode { .. }
            );
            apply_node_write(
                tx,
                is_create,
                *node_id,
                *version_id,
                *label,
                properties.clone(),
                *valid_from,
                commit_timestamp,
                historical,
                provenance.clone(),
            )?;
        }
        crate::api::transaction::BufferedWrite::CreateEdge {
            edge_id,
            version_id,
            source,
            target,
            label,
            properties,
            valid_from,
            provenance,
        }
        | crate::api::transaction::BufferedWrite::UpdateEdge {
            edge_id,
            version_id,
            source,
            target,
            label,
            properties,
            valid_from,
            provenance,
        } => {
            let is_create = matches!(
                write,
                crate::api::transaction::BufferedWrite::CreateEdge { .. }
            );
            apply_edge_write(
                tx,
                is_create,
                *edge_id,
                *version_id,
                *source,
                *target,
                *label,
                properties.clone(),
                *valid_from,
                commit_timestamp,
                historical,
                provenance.clone(),
            )?;
        }
        crate::api::transaction::BufferedWrite::DeleteNode {
            node_id,
            valid_from,
            provenance,
        } => {
            // Use pre-generated tombstone version ID (no lock needed)
            let tombstone_version_id = VersionId::new_unchecked(tombstone_ids.next().ok_or_else(|| {
                StorageError::InconsistentState {
                    reason: format!(
                        "Tombstone ID exhaustion for DeleteNode: expected {} deletes, iterator depleted at node_id {:?}",
                        num_deletes, node_id
                    ),
                }
            })?);

            apply_node_delete(
                tx,
                *node_id,
                *valid_from,
                commit_timestamp,
                tombstone_version_id,
                historical,
                provenance.clone(),
            )?;
        }
        crate::api::transaction::BufferedWrite::DeleteEdge {
            edge_id,
            valid_from,
            provenance,
        } => {
            // Use pre-generated tombstone version ID (no lock needed)
            let tombstone_version_id = VersionId::new_unchecked(tombstone_ids.next().ok_or_else(|| {
                StorageError::InconsistentState {
                    reason: format!(
                        "Tombstone ID exhaustion for DeleteEdge: expected {} deletes, iterator depleted at edge_id {:?}",
                        num_deletes, edge_id
                    ),
                }
            })?);

            apply_edge_delete(
                tx,
                *edge_id,
                *valid_from,
                commit_timestamp,
                tombstone_version_id,
                historical,
                provenance.clone(),
            )?;
        }
        crate::api::transaction::BufferedWrite::RetractNode {
            node_id,
            valid_to,
            provenance,
        } => {
            // Retractions draw from the same pre-generated version ID pool
            // as deletes (both append exactly one closing version).
            let retraction_version_id = VersionId::new_unchecked(tombstone_ids.next().ok_or_else(|| {
                StorageError::InconsistentState {
                    reason: format!(
                        "Version ID exhaustion for RetractNode: expected {} closing versions, iterator depleted at node_id {:?}",
                        num_deletes, node_id
                    ),
                }
            })?);

            apply_node_retract(
                tx,
                *node_id,
                *valid_to,
                commit_timestamp,
                retraction_version_id,
                historical,
                provenance.clone(),
            )?;
        }
        crate::api::transaction::BufferedWrite::RetractEdge {
            edge_id,
            valid_to,
            provenance,
        } => {
            let retraction_version_id = VersionId::new_unchecked(tombstone_ids.next().ok_or_else(|| {
                StorageError::InconsistentState {
                    reason: format!(
                        "Version ID exhaustion for RetractEdge: expected {} closing versions, iterator depleted at edge_id {:?}",
                        num_deletes, edge_id
                    ),
                }
            })?);

            apply_edge_retract(
                tx,
                *edge_id,
                *valid_to,
                commit_timestamp,
                retraction_version_id,
                historical,
                provenance.clone(),
            )?;
        }
    }
    Ok(())
}

/// Apply all buffered changes in the transaction.
///
/// This is the main entry point for the "Apply" phase. It iterates through the
/// write buffer and applies each operation to the storage engine.
///
/// # Locking Strategy
///
/// This function acquires the write lock on `HistoricalStorage` *once* and holds it
/// for the duration of all operations. This avoids lock thrashing (acquiring/releasing
/// for every single write) and ensures atomicity of the historical updates.
///
/// The acquired `historical.write()` guard is **returned to the caller** rather than
/// dropped internally (Issue #3425). The caller must keep it alive across
/// [`finalize_current_commit_timestamps`] so that the current-side writes and their
/// `commit_timestamp` finalization are atomic with respect to any snapshot
/// (checkpoint / backup) that holds `historical.read()`. Otherwise a snapshot landing
/// between the guard drop and finalize could capture a committed node/edge whose
/// `commit_timestamp` is still `None`. Respecting the documented lock order
/// (`historical` precedes `snapshot_lock`, adjacency, and the temporal-vector index
/// lock), holding this guard across finalize introduces no lock-order inversion.
///
/// # Performance
///
/// - **Locking**: Single lock acquisition for historical storage, held across finalize.
/// - **ID Generation**: Tombstone IDs are pre-generated in a batch to minimize lock contention.
/// - **Adjacency**: Adjacency indexes are compacted *once* after all edge operations are complete.
pub(crate) fn apply_changes<'a>(
    tx: &'a WriteTransaction,
    commit_timestamp: Timestamp,
    closing_version_ids: &[VersionId],
) -> Result<parking_lot::RwLockWriteGuard<'a, HistoricalStorage>> {
    // Create temporal interval for all operations in this transaction.
    let _temporal = BiTemporalInterval::current(commit_timestamp);

    // Acquire lock on historical storage once before processing all operations.
    let mut historical = tx.historical.write();

    // Issue #3413 (WAL abort framing): the commit-time precondition guards — the
    // #3416 delete-orphan / dangling-endpoint write-skew re-checks and the
    // #3577/#3755 CAS/lease/fence re-check — USED to run HERE, under this
    // `historical.write()` guard, AFTER the WAL frame was already durable, so a
    // rejection left a phantom `[BeginTx, ..ops.., CommitTx]` frame that crash
    // recovery reapplied. They now run in `commit_with_timestamp_inner` BEFORE
    // the WAL append, under the `current_timestamp` lock held across this apply.
    // `current_timestamp` serializes commits end-to-end, so a guard that passed
    // there is still valid here (no other committer can have applied in between),
    // and a rejected transaction returns with NO WAL frame — nothing to replay.
    // See [`detect_delete_orphan_write_skew`], [`detect_create_edge_dangling_endpoint`]
    // (this module) and [`super::cas::detect_cas_precondition_violations`].

    // Issue #3406: the closing-version IDs (delete tombstones + retraction
    // versions) are pre-generated once per commit — BEFORE the WAL log phase —
    // so the identical ids are recorded in the WAL and applied here. We simply
    // replay them in the same buffer order they were generated and logged.
    let num_deletes = closing_version_ids.len();
    let mut tombstone_ids = closing_version_ids.iter().map(|v| v.as_u64());

    for write in tx.buffer.operations() {
        apply_single_write(
            tx,
            write,
            commit_timestamp,
            &mut historical,
            &mut tombstone_ids,
            num_deletes,
        )?;
    }

    // Safety check
    debug_assert!(
        tombstone_ids.next().is_none(),
        "Tombstone ID surplus: expected {} deletes, but iterator has remaining IDs",
        num_deletes
    );

    // With incremental adjacency indexes, edge updates are immediately visible
    // via merged reads. Background compaction handles delta->frozen promotion.

    // Return the guard (Issue #3425): the caller holds it across
    // `finalize_current_commit_timestamps` so a concurrent `historical.read()`
    // snapshot cannot observe a committed node/edge with `commit_timestamp: None`.
    Ok(historical)
}

/// Commit-time re-check that a buffered node delete/retract does not orphan an
/// edge that a concurrent transaction committed after this tx's snapshot
/// (Issue #3416 Pt1 -- snapshot-isolation write-skew).
///
/// For each buffered `DeleteNode`/`RetractNode`, we enumerate the victim's
/// committed connected edges (distinct). An edge trips the abort only when it
/// is **both**: (1) committed **after** this tx's snapshot (`commit_timestamp >
/// snapshot_timestamp`) -- i.e. it is a concurrently-appeared edge, the SI
/// write-skew -- and (2) **not** also being closed (deleted/retracted) by this
/// same transaction.
///
/// The `commit_timestamp > snapshot_timestamp` gate is deliberate: an edge that
/// was already visible at this tx's snapshot is the documented low-level
/// orphan-preserving `delete_node` behavior (see CLAUDE.md "Orphaned Edges on
/// Node Deletion") and must NOT abort. Only a NEW concurrent edge violates the
/// #3209 "never orphan" guarantee.
///
/// On a violation we abort with `TransactionError::ValidationFailed`, which the
/// MCP layer classifies as `FAILED_PRECONDITION` (non-retriable): the correct
/// caller response is to re-evaluate (the edge now exists, so a fresh delete
/// would be refused by #3209 or require detach), not to blindly retry.
///
/// # Locking
///
/// Called while only `current_timestamp` (order class 1) is held, before the WAL
/// append (Issue #3413). It reads the adjacency indexes (`outgoing`/`incoming`,
/// classes 6/7) and current-storage edge map -- all LATER than `current_timestamp`
/// in the documented lock order -- and never calls back into
/// `historical`/`wal`/`current_timestamp`, so no lock-order inversion is
/// introduced.
///
/// # Cost
///
/// O(degree) committed-adjacency reads per buffered node delete/retract, at
/// commit only. Zero cost for transactions that delete/retract no nodes.
///
/// Issue #3413: invoked from `commit_with_timestamp_inner` under the
/// `current_timestamp` lock BEFORE the WAL append (not from `apply_changes`), so
/// a rejection leaves no durable frame. Reads only current storage / adjacency.
pub(super) fn detect_delete_orphan_write_skew(tx: &WriteTransaction) -> Result<()> {
    use crate::api::transaction::BufferedWrite;

    // Short-circuit (NIT): most transactions delete/retract no nodes; skip
    // building the closed-edges set and rescanning when there is nothing to
    // check.
    let has_node_close = tx.buffer.operations().iter().any(|w| {
        matches!(
            w,
            BufferedWrite::DeleteNode { .. } | BufferedWrite::RetractNode { .. }
        )
    });
    if !has_node_close {
        return Ok(());
    }

    // Edges this tx itself closes (delete or retract) never orphan the victim.
    let mut closed_edges: std::collections::HashSet<EdgeId> = std::collections::HashSet::new();
    for write in tx.buffer.operations() {
        match write {
            BufferedWrite::DeleteEdge { edge_id, .. }
            | BufferedWrite::RetractEdge { edge_id, .. } => {
                closed_edges.insert(*edge_id);
            }
            _ => {}
        }
    }

    let snapshot_ts = tx.snapshot.snapshot_timestamp;

    for write in tx.buffer.operations() {
        let node_id = match write {
            BufferedWrite::DeleteNode { node_id, .. }
            | BufferedWrite::RetractNode { node_id, .. } => *node_id,
            _ => continue,
        };

        // Distinct committed connected edges (a self-loop appears in both
        // directions but is one edge -- mirrors Issue #3416 Pt2/Pt3 counting).
        let mut edge_ids = tx.current.get_outgoing_edges(node_id);
        edge_ids.extend(tx.current.get_incoming_edges(node_id));
        edge_ids.sort_unstable();
        edge_ids.dedup();

        for edge_id in edge_ids {
            if closed_edges.contains(&edge_id) {
                continue; // this tx closes the edge too -- no orphan
            }
            // Only a concurrently-committed edge (commit_ts > our snapshot) is a
            // write-skew; a pre-snapshot edge is the documented orphan-preserving
            // low-level delete and is left untouched.
            if let Ok(edge) = tx.current.get_edge(edge_id)
                && let Some(commit_ts) = edge.metadata.commit_timestamp
                && commit_ts > snapshot_ts
            {
                return Err(crate::core::error::TransactionError::ValidationFailed {
                    reason: format!(
                        "Write-skew detected: node {} was deleted/retracted, but edge {} \
                         referencing it was created by a concurrent transaction (committed at \
                         {} after this transaction's snapshot at {}). Committing would orphan \
                         the edge. Re-evaluate the delete (the node now has a connected edge) \
                         or use a detach/cascade delete.",
                        node_id.as_u64(),
                        edge_id.as_u64(),
                        commit_ts,
                        snapshot_ts
                    ),
                }
                .into());
            }
        }
    }

    Ok(())
}

/// Commit-time re-check that a buffered `CreateEdge` does not commit with an
/// endpoint that a concurrent transaction deleted/retracted after this tx's
/// snapshot (Issue #3416 Pt1 -- the MIRROR of the delete-side write-skew).
///
/// This closes the second ordering: the edge creator's endpoints are validated
/// by [`validation::validate`](super::validation::validate) *before* the
/// timestamp/WAL/apply phase, and [`apply_edge_write`] does NO endpoint check,
/// so a concurrent node delete that commits between `validate()` and this point
/// would leave a dangling edge. Run under the same `historical.write()` guard as
/// the delete-side check, the two orderings become symmetric: whichever tx
/// applies second aborts.
///
/// An endpoint is considered present iff it is resolvable at apply time --
/// **either** live in committed current storage **or** created/updated earlier
/// in THIS transaction's buffer (a same-tx-created endpoint is valid; a same-tx
/// *delete* of an endpoint under a still-live edge is already rejected by
/// `validate()`, mirroring that function's `node_exists` logic exactly). A
/// `CreateEdge` whose edge is itself deleted/retracted later in this same tx is
/// skipped -- it commits no live edge, so a concurrently-deleted endpoint
/// orphans nothing (this also keeps a same-tx `create_edge` + cascade / Pt4 from
/// self-aborting). Because `validate()` already passed, an endpoint absent here
/// is definitionally a concurrent deletion, so no snapshot-timestamp gate is
/// needed (unlike the delete side, where a pre-snapshot edge is the documented
/// orphan-preserving case).
///
/// Aborts with `TransactionError::ValidationFailed` (-> MCP `FAILED_PRECONDITION`,
/// non-retriable).
///
/// # Locking
///
/// Called while only `current_timestamp` (order class 1) is held, before the WAL
/// append (Issue #3413); reads only the current-storage node map (a leaf) and
/// this tx's buffer, never calling back into `historical`/`wal`/`current_timestamp`.
/// No lock-order inversion.
///
/// # Cost
///
/// O(1) buffer/current lookups per buffered `CreateEdge`, at commit only. Zero
/// cost for transactions that create no edges.
///
/// Issue #3413: invoked from `commit_with_timestamp_inner` under the
/// `current_timestamp` lock BEFORE the WAL append (not from `apply_changes`), so
/// a rejection leaves no durable frame. Reads only this tx's buffer + current
/// storage.
pub(super) fn detect_create_edge_dangling_endpoint(tx: &WriteTransaction) -> Result<()> {
    use crate::api::transaction::BufferedWrite;

    // Resolve an endpoint exactly as `validation::validate` does: a same-tx
    // buffered node counts as present iff its latest op is a create/update; a
    // committed node counts as present iff it is still in current storage.
    let endpoint_present = |node_id: NodeId| -> bool {
        if tx.buffer.has_modified_node(node_id) {
            matches!(
                tx.buffer.get_node_write(node_id),
                Some(BufferedWrite::CreateNode { .. } | BufferedWrite::UpdateNode { .. })
            )
        } else {
            tx.current.get_node(node_id).is_ok()
        }
    };

    for write in tx.buffer.operations() {
        let (edge_id, source, target) = match write {
            BufferedWrite::CreateEdge {
                edge_id,
                source,
                target,
                ..
            } => (*edge_id, *source, *target),
            _ => continue,
        };

        // Skip an edge created and then deleted/retracted within this same tx:
        // it commits no live edge (mirrors the Pt4 skip in `validation::validate`).
        if matches!(
            tx.buffer.get_edge_write(edge_id),
            Some(BufferedWrite::DeleteEdge { .. } | BufferedWrite::RetractEdge { .. })
        ) {
            continue;
        }

        for endpoint in [source, target] {
            if !endpoint_present(endpoint) {
                return Err(crate::core::error::TransactionError::ValidationFailed {
                    reason: format!(
                        "Write-skew detected: edge {} references node {}, which was \
                         deleted or retracted by a concurrent transaction after this \
                         transaction's snapshot. Committing would create a dangling edge. \
                         Re-check the endpoint's existence and retry.",
                        edge_id.as_u64(),
                        endpoint.as_u64(),
                    ),
                }
                .into());
            }
        }
    }

    Ok(())
}

/// Finalize commit timestamps in current storage after a successful `apply_changes`.
///
/// During `apply_changes`, nodes and edges are written to current storage with
/// `commit_timestamp: None` (via `VersionMetadata::uncommitted`).  This prevents
/// phantom visibility if `apply_changes` fails partway through: an aborted
/// transaction's partial writes stay invisible because `is_visible_with_embedded_ts`
/// treats `None` as uncommitted.
///
/// This function is called only on the success path, just before `register_commit`.
/// It sets `commit_timestamp: Some(T)` on every node/edge written in this
/// transaction, making them visible to snapshots taken after the commit.
pub(crate) fn finalize_current_commit_timestamps(
    tx: &WriteTransaction,
    commit_timestamp: Timestamp,
) {
    for write in tx.buffer.operations() {
        match write {
            crate::api::transaction::BufferedWrite::CreateNode { node_id, .. }
            | crate::api::transaction::BufferedWrite::UpdateNode { node_id, .. } => {
                tx.current
                    .set_node_commit_timestamp(*node_id, commit_timestamp);
            }
            crate::api::transaction::BufferedWrite::CreateEdge { edge_id, .. }
            | crate::api::transaction::BufferedWrite::UpdateEdge { edge_id, .. } => {
                tx.current
                    .set_edge_commit_timestamp(*edge_id, commit_timestamp);
            }
            // Deletes remove the node/edge from current storage; no finalization needed.
            _ => {}
        }
    }
}
