//! Read-only transactions
//!
//! Read-only transactions are lightweight:
//! - No write buffer
//! - No WAL logging
//! - Snapshot-based reads for consistency
//! - No commit overhead

use super::{ReadOps, TransactionSnapshot, TxId, TxMetadata, TxState, TxVisibilityManager};
use crate::core::error::{Result, ResultExt, StorageError};
use crate::core::graph::{Edge, Node};
use crate::core::id::{EdgeId, NodeId};
use crate::core::property::PropertyValue;

use crate::core::version::VersionMetadata;
use crate::storage::current::CurrentStorage;
use crate::storage::historical::HistoricalStorage;
use parking_lot::RwLock;
use std::sync::Arc;

/// Read-only transaction
///
/// Read-only transactions are lightweight:
/// - No write buffer
/// - No WAL logging
/// - Snapshot-based reads for consistency
/// - No commit overhead
///
/// # Example
///
/// ```rust,no_run
/// # use aletheiadb::{AletheiaDB, core::NodeId, api::transaction::ReadOps};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let db = AletheiaDB::new()?;
/// # let node_id = NodeId::new(1)?;
/// let tx = db.read_transaction()?;
/// let node = tx.get_node(node_id)?;
/// // No commit needed - transaction is read-only
/// # Ok(())
/// # }
/// ```
pub struct ReadTransaction {
    tx_id: TxId,
    start_timestamp: crate::core::temporal::Timestamp,
    snapshot: TransactionSnapshot,
    handles: Arc<ReadHandles>,
}

/// The storage handles a read transaction needs, bundled behind one `Arc`.
///
/// Opening a read transaction used to clone three separate `Arc`s (and drop
/// them again), which is three atomic increments and three decrements on three
/// cache lines every thread shares. That refcount traffic -- not any lock --
/// was the dominant cost of `read_transaction()` once the visibility manager's
/// lock was gone: a control measuring four contended `Arc` clone+drop pairs and
/// nothing else reproduced most of the cost, and the same collapse in scaling.
///
/// Bundling makes it one clone and one drop. The handles are immutable for the
/// lifetime of the database, so the bundle is built once and shared.
pub struct ReadHandles {
    pub(crate) current: Arc<CurrentStorage>,
    pub(crate) visibility_manager: Arc<TxVisibilityManager>,
    pub(crate) historical: Arc<RwLock<HistoricalStorage>>,
}

impl std::fmt::Debug for ReadHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The handles themselves are not Debug; identity is all that is useful.
        f.write_str("ReadHandles { .. }")
    }
}

impl ReadHandles {
    /// Bundle three handles for sharing across read transactions.
    pub fn new(
        current: Arc<CurrentStorage>,
        visibility_manager: Arc<TxVisibilityManager>,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Self {
        Self {
            current,
            visibility_manager,
            historical,
        }
    }
}

impl ReadTransaction {
    /// Create a new read-only transaction
    ///
    /// # Arguments
    ///
    /// * `tx_id` - The unique identifier assigned to this transaction
    /// * `snapshot` - The isolated view of the database state captured at the start of this transaction
    /// * `current` - Reference to the current (latest) graph state storage
    /// * `visibility_manager` - Manages which versions of entities are visible to this transaction
    /// * `historical` - Reference to the historical graph state storage for resolving older versions
    pub(crate) fn new(
        tx_id: TxId,
        snapshot: TransactionSnapshot,
        handles: Arc<ReadHandles>,
    ) -> Self {
        ReadTransaction {
            tx_id,
            start_timestamp: snapshot.snapshot_timestamp,
            snapshot,
            handles,
        }
    }

    /// Construct from loose handles, bundling them.
    ///
    /// For tests and callers that hold the three `Arc`s separately. The hot
    /// path uses [`ReadTransaction::new`] with a bundle the database built once,
    /// so it pays one refcount increment instead of three.
    #[cfg(test)]
    pub(crate) fn from_parts(
        tx_id: TxId,
        snapshot: TransactionSnapshot,
        current: Arc<CurrentStorage>,
        visibility_manager: Arc<TxVisibilityManager>,
        historical: Arc<RwLock<HistoricalStorage>>,
    ) -> Self {
        Self::new(
            tx_id,
            snapshot,
            Arc::new(ReadHandles::new(current, visibility_manager, historical)),
        )
    }

    /// Get transaction metadata.
    ///
    /// This metadata provides information about the transaction's lifecycle,
    /// such as its unique ID, start timestamp, and current state. Since this
    /// is a read-only transaction, the state will always be `Active` until dropped,
    /// and the commit timestamp will be `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use aletheiadb::AletheiaDB;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = AletheiaDB::new()?;
    /// let tx = db.read_transaction()?;
    /// let meta = tx.metadata();
    ///
    /// assert!(meta.is_read_only);
    /// assert_eq!(meta.tx_id, tx.tx_id());
    /// # Ok(())
    /// # }
    /// ```
    pub fn metadata(&self) -> TxMetadata {
        TxMetadata {
            tx_id: self.tx_id,
            start_timestamp: self.start_timestamp,
            commit_timestamp: None,
            state: TxState::Active,
            is_read_only: true,
        }
    }

    /// Get transaction ID.
    ///
    /// Returns the unique identifier assigned to this transaction when it was created.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use aletheiadb::AletheiaDB;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = AletheiaDB::new()?;
    /// let tx = db.read_transaction()?;
    /// let id = tx.tx_id();
    ///
    /// println!("Running in transaction context: {:?}", id);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tx_id(&self) -> TxId {
        self.tx_id
    }

    /// Query historical storage for a node version visible at snapshot time.
    ///
    /// This is the slow path used when the current version is not visible
    /// or when the node has been deleted from current storage.
    fn get_node_from_historical(&self, id: NodeId) -> Result<Node> {
        let historical = self.handles.historical.read();

        // Find version visible at our snapshot timestamp
        let version_id = historical.find_node_version_at_time(
            id,
            self.snapshot.snapshot_timestamp, // valid_time
            self.snapshot.snapshot_timestamp, // transaction_time
        );

        match version_id {
            Some(vid) => {
                // Found a visible version - reconstruct it
                let version = historical
                    .get_node_version(vid)
                    .ok_or(StorageError::VersionNotFound(vid))?;

                // Reconstruct properties from anchor+delta
                let properties = historical.reconstruct_node_properties(vid)?;

                // Extract metadata from the temporal interval
                // NOTE: Historical versions don't track created_by_tx, so we use TxId(0)
                // The commit_timestamp is extracted from transaction_time.start
                let metadata = VersionMetadata::new(
                    super::TxId::new(0), // Historical versions don't track creating tx
                    version.temporal.transaction_time().start(), // Extract commit timestamp
                );

                // Build Node from historical version
                Ok(Node::with_metadata(
                    id,
                    version.label,
                    properties,
                    vid,
                    metadata,
                ))
            }
            None => {
                // No version visible at snapshot time
                Err(StorageError::NodeNotFound(id).into())
            }
        }
    }

    /// Filter a list of edge IDs to only include those visible in our snapshot.
    ///
    /// This prevents phantom reads by checking each edge's visibility.
    /// Edges that don't exist or aren't visible are filtered out.
    ///
    /// ⚡ Bolt Optimization: Uses `.retain()` instead of `.into_iter().filter(...).collect()`
    /// to filter in-place and avoid allocating a new `Vec`.
    fn filter_visible_edges(&self, mut edge_ids: Vec<EdgeId>) -> Vec<EdgeId> {
        edge_ids.retain(|&edge_id| {
            // Use embedded commit_timestamp for visibility check (Issue #238).
            if let Ok(edge) = self.handles.current.get_edge(edge_id) {
                self.handles.visibility_manager.is_visible_with_embedded_ts(
                    &self.snapshot,
                    edge.metadata.created_by_tx,
                    edge.metadata.commit_timestamp,
                )
            } else {
                // Edge doesn't exist or was deleted - not visible
                false
            }
        });
        edge_ids
    }

    /// Query historical storage for an edge version visible at snapshot time.
    ///
    /// This is the slow path used when the current version is not visible
    /// or when the edge has been deleted from current storage.
    fn get_edge_from_historical(&self, id: EdgeId) -> Result<Edge> {
        let historical = self.handles.historical.read();

        // Find version visible at our snapshot timestamp
        let version_id = historical.find_edge_version_at_time(
            id,
            self.snapshot.snapshot_timestamp, // valid_time
            self.snapshot.snapshot_timestamp, // transaction_time
        );

        match version_id {
            Some(vid) => {
                // Found a visible version - reconstruct it
                let version = historical
                    .get_edge_version(vid)
                    .ok_or(StorageError::VersionNotFound(vid))?;

                // Reconstruct properties from anchor+delta
                let properties = historical.reconstruct_edge_properties(vid)?;

                // Extract metadata from the temporal interval
                // NOTE: Historical versions don't track created_by_tx, so we use TxId(0)
                // The commit_timestamp is extracted from transaction_time.start
                let metadata = VersionMetadata::new(
                    super::TxId::new(0), // Historical versions don't track creating tx
                    version.temporal.transaction_time().start(), // Extract commit timestamp
                );

                // Build Edge from historical version
                Ok(Edge::with_metadata(
                    id,
                    version.label,
                    version.source,
                    version.target,
                    properties,
                    vid,
                    metadata,
                ))
            }
            None => {
                // No version visible at snapshot time
                Err(StorageError::EdgeNotFound(id).into())
            }
        }
    }
}

impl ReadOps for ReadTransaction {
    fn get_node(&self, id: NodeId) -> Result<Node> {
        let result = if let Ok(current_node) = self.handles.current.get_node(id) {
            // Use embedded commit_timestamp for visibility check (HyPer/TiDB pattern, Issue #238).
            // Bypasses the TxVisibilityManager::committed map lock for the common fast path.
            if self.handles.visibility_manager.is_visible_with_embedded_ts(
                &self.snapshot,
                current_node.metadata.created_by_tx,
                current_node.metadata.commit_timestamp,
            ) {
                Ok(current_node)
            } else {
                // If not visible, fall through to the slow path.
                self.get_node_from_historical(id)
            }
        } else {
            // If the node is not in current storage (e.g., it was deleted),
            // we must still check historical storage.
            self.get_node_from_historical(id)
        };

        result.record_error_metric()
    }

    fn get_edge(&self, id: EdgeId) -> Result<Edge> {
        let result = if let Ok(current_edge) = self.handles.current.get_edge(id) {
            // Use embedded commit_timestamp for visibility check (HyPer/TiDB pattern, Issue #238).
            if self.handles.visibility_manager.is_visible_with_embedded_ts(
                &self.snapshot,
                current_edge.metadata.created_by_tx,
                current_edge.metadata.commit_timestamp,
            ) {
                Ok(current_edge)
            } else {
                // If not visible, fall through to the slow path.
                self.get_edge_from_historical(id)
            }
        } else {
            // If the edge is not in current storage (e.g., it was deleted),
            // we must still check historical storage.
            self.get_edge_from_historical(id)
        };

        result.record_error_metric()
    }

    fn get_outgoing_edges(&self, node_id: NodeId) -> Result<Vec<EdgeId>> {
        // Existence check (Issue #359): a missing node is an error, so callers can
        // distinguish "node has no edges" (Ok(empty)) from "node doesn't exist".
        self.get_node(node_id)?;
        // Filter edges to only return those visible in our snapshot
        // This prevents phantom reads where we see edges created after our snapshot
        // Note: CurrentStorage::get_outgoing_edges() uses frozen view when available
        let edge_ids = self.handles.current.get_outgoing_edges(node_id);
        Ok(self.filter_visible_edges(edge_ids))
    }

    fn get_incoming_edges(&self, node_id: NodeId) -> Result<Vec<EdgeId>> {
        // Existence check (Issue #359): a missing node is an error.
        self.get_node(node_id)?;
        // Filter edges to only return those visible in our snapshot
        // Note: CurrentStorage::get_incoming_edges() uses frozen view when available
        let edge_ids = self.handles.current.get_incoming_edges(node_id);
        Ok(self.filter_visible_edges(edge_ids))
    }

    fn get_outgoing_edges_with_label(&self, node_id: NodeId, label: &str) -> Result<Vec<EdgeId>> {
        // Existence check (Issue #359): a missing node is an error. An existing
        // node with no edges matching `label` is Ok(empty) - the label is a
        // filter, not an existence check.
        self.get_node(node_id)?;
        // Filter edges to only return those visible in our snapshot
        let edge_ids = self
            .handles
            .current
            .get_outgoing_edges_with_label(node_id, label);
        Ok(self.filter_visible_edges(edge_ids))
    }

    fn node_count(&self) -> usize {
        self.handles.current.node_count()
    }

    fn edge_count(&self) -> usize {
        self.handles.current.edge_count()
    }

    fn find_nodes_by_property(
        &self,
        label: &str,
        property_key: &str,
        property_value: &PropertyValue,
    ) -> Vec<NodeId> {
        // ⚡ Bolt Optimization: Uses `.retain()` to filter in-place and avoid allocating a new `Vec`.
        let mut node_ids =
            self.handles
                .current
                .find_nodes_by_property(label, property_key, property_value);

        node_ids.retain(|node_id| {
            self.handles
                .current
                .get_node(*node_id)
                .map(|node| {
                    // Use embedded commit_timestamp for visibility check (Issue #238).
                    self.handles.visibility_manager.is_visible_with_embedded_ts(
                        &self.snapshot,
                        node.metadata.created_by_tx,
                        node.metadata.commit_timestamp,
                    )
                })
                .unwrap_or(false)
        });

        node_ids
    }
}

// No `Drop` impl on purpose.
//
// This used to call `register_abort` to take the transaction back out of the
// visibility manager's active set. Read transactions no longer go *into* that
// set (see `TxVisibilityManager`), so there is nothing to remove -- and removing
// it was a lock acquisition plus a copy-on-write clone of the entire set on
// every read transaction, on the drop path where it is easiest to miss.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;
    use crate::core::temporal::time;

    use parking_lot::RwLock;
    use std::sync::Arc;

    // Helper to create a test ReadTransaction with snapshot
    fn create_test_read_tx(tx_id: TxId, current: Arc<CurrentStorage>) -> ReadTransaction {
        let visibility_manager = Arc::new(TxVisibilityManager::new());
        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));
        let snapshot = TransactionSnapshot {
            snapshot_timestamp: time::now(),
            active_transactions: None,
        };
        ReadTransaction::from_parts(tx_id, snapshot, current, visibility_manager, historical)
    }

    #[test]
    fn test_read_transaction_creation() {
        let current = Arc::new(CurrentStorage::new());
        let tx = create_test_read_tx(TxId::new(1), current);

        assert_eq!(tx.tx_id(), TxId::new(1));
        let metadata = tx.metadata();
        assert_eq!(metadata.tx_id, TxId::new(1));
        assert!(metadata.is_read_only);
        assert_eq!(metadata.state, TxState::Active);
        assert_eq!(metadata.commit_timestamp, None);
    }

    #[test]
    fn test_read_transaction_get_node() {
        let current = Arc::new(CurrentStorage::new());

        // Create a node in the storage
        let props = PropertyMapBuilder::new()
            .insert("name", "Alice")
            .insert("age", 30i64)
            .build();
        let node_id = current.create_node("Person", props.clone()).unwrap();

        // Read through transaction
        let tx = create_test_read_tx(TxId::new(1), Arc::clone(&current));
        let node = tx.get_node(node_id).unwrap();

        assert_eq!(node.id, node_id);
        assert_eq!(
            node.get_property("name").and_then(|v| v.as_str()),
            Some("Alice")
        );
        assert_eq!(node.get_property("age").and_then(|v| v.as_int()), Some(30));
    }


    #[test]
    fn test_read_transaction_get_node_from_historical_not_found() {
        // 🛡️ Sentry Test: Verify `get_node_from_historical` branch coverage.
        let current = Arc::new(CurrentStorage::new());
        let historical = Arc::new(RwLock::new(HistoricalStorage::new()));

        let tx_id = TxId::new(1);
        let visibility_manager = Arc::new(TxVisibilityManager::new());

        let valid_time = crate::core::temporal::time::now();

        let snapshot = TransactionSnapshot {
            snapshot_timestamp: valid_time,
            active_transactions: Arc::new(HashSet::new()),
        };

        let tx = ReadTransaction::new(tx_id, snapshot, current, visibility_manager, historical);

        // This exercises the `None` branch of `get_node_from_historical` since history is empty
        let result = tx.get_node_from_historical(NodeId::new(123).unwrap());

        assert!(result.is_err());
    }

    #[test]
    fn test_read_transaction_get_node_not_found() {
        let current = Arc::new(CurrentStorage::new());
        let tx = create_test_read_tx(TxId::new(1), current);

        let result = tx.get_node(NodeId::new(999).unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_read_transaction_node_count() {
        let current = Arc::new(CurrentStorage::new());

        // Create some nodes
        let props = PropertyMapBuilder::new().build();
        current.create_node("Person", props.clone()).unwrap();
        current.create_node("Person", props.clone()).unwrap();
        current.create_node("Person", props).unwrap();

        let tx = create_test_read_tx(TxId::new(1), current);
        assert_eq!(tx.node_count(), 3);
    }

    #[test]
    fn test_read_transaction_get_edges() {
        let current = Arc::new(CurrentStorage::new());

        // Create nodes and edge
        let props = PropertyMapBuilder::new().build();
        let node1 = current.create_node("Person", props.clone()).unwrap();
        let node2 = current.create_node("Person", props.clone()).unwrap();
        let edge_id = current.create_edge(node1, node2, "KNOWS", props).unwrap();

        let tx = create_test_read_tx(TxId::new(1), current);

        // Get edge
        let edge = tx.get_edge(edge_id).unwrap();
        assert_eq!(edge.id, edge_id);
        assert_eq!(edge.source, node1);
        assert_eq!(edge.target, node2);

        // Get outgoing edges
        let outgoing = tx.get_outgoing_edges(node1).unwrap();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0], edge_id);

        // Get incoming edges
        let incoming = tx.get_incoming_edges(node2).unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0], edge_id);
    }

    #[test]
    fn test_read_transaction_get_outgoing_edges_with_label() {
        let current = Arc::new(CurrentStorage::new());

        let props = PropertyMapBuilder::new().build();
        let node1 = current.create_node("Person", props.clone()).unwrap();
        let node2 = current.create_node("Person", props.clone()).unwrap();
        let node3 = current.create_node("Person", props.clone()).unwrap();

        // Create edges with different labels
        let edge1 = current
            .create_edge(node1, node2, "KNOWS", props.clone())
            .unwrap();
        let _edge2 = current.create_edge(node1, node3, "FOLLOWS", props).unwrap();

        let tx = create_test_read_tx(TxId::new(1), current);

        // Get only KNOWS edges
        let knows_edges = tx.get_outgoing_edges_with_label(node1, "KNOWS").unwrap();
        assert_eq!(knows_edges.len(), 1);
        assert_eq!(knows_edges[0], edge1);

        // Get only FOLLOWS edges
        let follows_edges = tx.get_outgoing_edges_with_label(node1, "FOLLOWS").unwrap();
        assert_eq!(follows_edges.len(), 1);
    }

    #[test]
    fn test_read_transaction_concurrent_access() {
        use std::thread;

        let current = Arc::new(CurrentStorage::new());

        // Pre-populate with data
        let props = PropertyMapBuilder::new().insert("value", 42i64).build();
        let node_id = current.create_node("Test", props).unwrap();

        // Spawn multiple reader threads
        let mut handles = vec![];
        for i in 0..10 {
            let current_clone = Arc::clone(&current);
            let handle = thread::spawn(move || {
                let tx = create_test_read_tx(TxId::new(i), current_clone);
                let node = tx.get_node(node_id).unwrap();
                assert_eq!(
                    node.get_property("value").and_then(|v| v.as_int()),
                    Some(42)
                );
            });
            handles.push(handle);
        }

        // Wait for all readers
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn read_transactions_never_enter_the_active_set() {
        // The active set answers exactly one question: was the transaction that
        // *created* this version still in flight when the snapshot was taken? A
        // read transaction creates nothing, so its membership can never be
        // tested -- registering it was unobservable work on the hottest path
        // there is. This pins that it stays out, on both create and drop.
        let db = crate::AletheiaDB::new().expect("db");
        let before = db.visibility_manager.active_count();

        {
            let _tx = db.read_transaction().expect("read tx");
            assert_eq!(
                db.visibility_manager.active_count(),
                before,
                "a read transaction must not register itself as active"
            );
        }

        assert_eq!(
            db.visibility_manager.active_count(),
            before,
            "and must leave the active set exactly as it found it"
        );
    }

    #[test]
    fn write_transactions_still_register_and_deregister() {
        // The counterpart: writers DO create versions, so they must still be
        // tracked -- removing reads from the set must not have removed writes.
        let db = crate::AletheiaDB::new().expect("db");
        let before = db.visibility_manager.active_count();

        let tx = db.write_transaction().expect("write tx");
        assert_eq!(
            db.visibility_manager.active_count(),
            before + 1,
            "a write transaction must register as active"
        );

        drop(tx);
        assert_eq!(
            db.visibility_manager.active_count(),
            before,
            "and must deregister when it goes away"
        );
    }

    #[test]
    fn test_read_transaction_find_nodes_by_property() {
        let current = Arc::new(CurrentStorage::new());

        let alice_id = current
            .create_node(
                "Person",
                PropertyMapBuilder::new()
                    .insert("name", "Alice")
                    .insert("age", 30i64)
                    .build(),
            )
            .unwrap();
        let _bob_id = current
            .create_node(
                "Person",
                PropertyMapBuilder::new()
                    .insert("name", "Bob")
                    .insert("age", 25i64)
                    .build(),
            )
            .unwrap();

        let tx = create_test_read_tx(TxId::new(1), current);

        let results = tx.find_nodes_by_property(
            "Person",
            "name",
            &crate::core::property::PropertyValue::String("Alice".into()),
        );
        assert_eq!(results, vec![alice_id]);
    }

    // Issue #359: edge-listing methods return Result so callers can
    // distinguish "node doesn't exist" (Err) from "node has no edges" (Ok(empty)).

    #[test]
    fn test_get_outgoing_edges_nonexistent_node_errors() {
        let current = Arc::new(CurrentStorage::new());
        let tx = create_test_read_tx(TxId::new(1), current);

        let missing = NodeId::new(999).unwrap();
        let result = tx.get_outgoing_edges(missing);
        assert!(
            matches!(
                result,
                Err(crate::core::error::Error::Storage(
                    crate::core::error::StorageError::NodeNotFound(id)
                )) if id == missing
            ),
            "get_outgoing_edges on a nonexistent node must return Err(NodeNotFound), got {result:?}"
        );
    }

    #[test]
    fn test_get_incoming_edges_nonexistent_node_errors() {
        let current = Arc::new(CurrentStorage::new());
        let tx = create_test_read_tx(TxId::new(1), current);

        let result = tx.get_incoming_edges(NodeId::new(999).unwrap());
        assert!(
            result.is_err(),
            "get_incoming_edges on a nonexistent node must return Err, got {result:?}"
        );
    }

    #[test]
    fn test_get_outgoing_edges_with_label_nonexistent_node_errors() {
        let current = Arc::new(CurrentStorage::new());
        let tx = create_test_read_tx(TxId::new(1), current);

        let result = tx.get_outgoing_edges_with_label(NodeId::new(999).unwrap(), "KNOWS");
        assert!(
            result.is_err(),
            "get_outgoing_edges_with_label on a nonexistent node must return Err, got {result:?}"
        );
    }

    #[test]
    fn test_get_outgoing_edges_existing_node_no_edges_ok_empty() {
        let current = Arc::new(CurrentStorage::new());
        let props = PropertyMapBuilder::new().build();
        let node = current.create_node("Person", props).unwrap();

        let tx = create_test_read_tx(TxId::new(1), current);
        let edges = tx
            .get_outgoing_edges(node)
            .expect("existing node with no edges must be Ok");
        assert!(edges.is_empty(), "expected Ok(empty), got {edges:?}");
    }

    #[test]
    fn test_get_incoming_edges_existing_node_no_edges_ok_empty() {
        let current = Arc::new(CurrentStorage::new());
        let props = PropertyMapBuilder::new().build();
        let node = current.create_node("Person", props).unwrap();

        let tx = create_test_read_tx(TxId::new(1), current);
        let edges = tx
            .get_incoming_edges(node)
            .expect("existing node with no edges must be Ok");
        assert!(edges.is_empty(), "expected Ok(empty), got {edges:?}");
    }

    #[test]
    fn test_get_outgoing_edges_with_label_no_match_ok_empty() {
        let current = Arc::new(CurrentStorage::new());
        let props = PropertyMapBuilder::new().build();
        let node1 = current.create_node("Person", props.clone()).unwrap();
        let node2 = current.create_node("Person", props.clone()).unwrap();
        current.create_edge(node1, node2, "KNOWS", props).unwrap();

        let tx = create_test_read_tx(TxId::new(1), current);
        // Node exists and has edges, but none with this label: the label is a
        // filter, not an existence check, so this is Ok(empty) and not an error.
        let edges = tx
            .get_outgoing_edges_with_label(node1, "FOLLOWS")
            .expect("existing node with no matching label must be Ok");
        assert!(edges.is_empty(), "expected Ok(empty), got {edges:?}");
    }

    #[test]
    fn test_read_transaction_find_nodes_by_property_empty() {
        let current = Arc::new(CurrentStorage::new());
        let tx = create_test_read_tx(TxId::new(1), current);

        let results = tx.find_nodes_by_property(
            "Person",
            "name",
            &crate::core::property::PropertyValue::String("Nobody".into()),
        );
        assert!(results.is_empty());
    }
}
