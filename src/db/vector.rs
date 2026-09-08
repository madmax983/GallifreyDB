//! Vector index management and search operations.
//!
//! Methods for creating vector indexes, semantic search, and similarity tracking.
use crate::core::error::{Result, ResultExt};
use crate::core::id::NodeId;
use crate::core::temporal::Timestamp;
use crate::db::AletheiaDB;
use crate::db::similarity_query::{SimilarityQuery, SimilaritySource};
use crate::db::vector_builder::VectorIndexBuilder;
use crate::index::vector::hnsw::HnswConfig;
use crate::index::vector::temporal::{TemporalVectorConfig, VectorIndexObserver};
use std::sync::Arc;

impl AletheiaDB {
    /// Enable vector indexing for a specific property.
    ///
    /// Once enabled, nodes with the specified property will be automatically
    /// indexed for similarity search. The property must contain vector values.
    ///
    /// # Arguments
    ///
    /// * `property_name` - Name of the property containing vectors
    /// * `config` - HNSW index configuration (dimensions, metric, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aletheiadb::index::vector::{HnswConfig, DistanceMetric};
    ///
    /// let config = HnswConfig::new(384, DistanceMetric::Cosine);
    /// db.enable_vector_index("embedding", config)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if vector indexing is already enabled.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn enable_vector_index(&self, property_name: &str, config: HnswConfig) -> Result<()> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_index_span("enable_vector_index", property_name).entered();
        self.current
            .enable_vector_index(property_name, config)
            .record_error_metric()
    }

    /// Rebuild a vector index from the vector properties of current nodes.
    ///
    /// This is the recovery path for a vector index that was **skipped at
    /// startup** because its persisted files (`meta.idx`, `mappings.idx`,
    /// `current.usearch`, or the `current.usearch.mappings` sidecar) were
    /// corrupted or unreadable (Issue #451). It enables the index with
    /// `config` if it is not registered, then backfills it by scanning every
    /// current node and indexing the node's `property_name` vector.
    ///
    /// # Why not just `enable_vector_index`?
    ///
    /// [`enable_vector_index`](Self::enable_vector_index) creates an **empty**
    /// index with no backfill. If you re-enable a skipped index without
    /// rebuilding, the next persistence cycle (background worker, shutdown,
    /// or `persist_indexes`) **overwrites the on-disk files with that empty
    /// index**, permanently losing the previously indexed vectors. Use this
    /// method instead to restore searchability for pre-existing nodes.
    ///
    /// If an index is already registered for `property_name`, `config` is
    /// ignored and the backfill runs against the existing index (entries are
    /// updated in place, so the call is idempotent). The rebuild covers
    /// **current state only** — temporal vector indexes are not rebuilt.
    ///
    /// Returns the number of node vectors indexed by the backfill.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aletheiadb::index::vector::{HnswConfig, DistanceMetric};
    ///
    /// // Startup skipped the "embedding" index (corrupted files on disk):
    /// let indexed = db.rebuild_vector_index(
    ///     "embedding",
    ///     HnswConfig::new(384, DistanceMetric::Cosine),
    /// )?;
    /// println!("re-indexed {indexed} vectors");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be created (e.g. the maximum
    /// number of vector-indexed properties is reached) or if indexing a
    /// node's vector fails (e.g. dimension mismatch with `config`).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn rebuild_vector_index(&self, property_name: &str, config: HnswConfig) -> Result<usize> {
        #[cfg(feature = "observability")]
        let _span = crate::observability::vector_index_span("rebuild_vector_index", property_name)
            .entered();
        let indexed = self
            .current
            .rebuild_vector_index(property_name, config)
            .record_error_metric()?;
        // Mark the vector indexes dirty so the background persistence worker
        // saves the rebuilt index (otherwise it would only be persisted after
        // the next vector write or on shutdown/persist_indexes).
        if let Some(ref tracker) = self.persistence_tracker {
            tracker.record_vector_mutation();
        }
        Ok(indexed)
    }

    /// Check if vector indexing is enabled.
    pub fn is_vector_index_enabled(&self) -> bool {
        self.current.is_vector_index_enabled()
    }

    /// Check if vector indexing is enabled for a specific property.
    pub fn is_vector_index_enabled_for(&self, property_name: &str) -> bool {
        self.current.is_vector_index_enabled_for(property_name)
    }

    /// Enable temporal vector indexing for a specific property.
    ///
    /// Once enabled, vector changes will be tracked over time using snapshot-based
    /// indexing, enabling point-in-time vector queries and semantic drift tracking.
    /// This also integrates with the historical storage's observer pattern to create
    /// vector snapshots aligned with graph anchors.
    ///
    /// # Arguments
    ///
    /// * `property_name` - Name of the property containing vectors
    /// * `config` - Temporal vector index configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aletheiadb::index::vector::temporal::{TemporalVectorConfig, SnapshotStrategy};
    /// use aletheiadb::index::vector::HnswConfig;
    ///
    /// let hnsw_config = HnswConfig::new(384, DistanceMetric::Cosine);
    /// let temporal_config = TemporalVectorConfig::default_with_hnsw(hnsw_config);
    /// db.enable_temporal_vector_index("embedding", temporal_config)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Temporal vector indexing is already enabled
    /// - The historical storage lock is poisoned
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn enable_temporal_vector_index(
        &self,
        property_name: &str,
        config: TemporalVectorConfig,
    ) -> Result<()> {
        let result = (|| {
            // Resolve hnsw_config: use provided config or get from existing vector index
            let resolved_hnsw_config = if let Some(hnsw_config) = config.hnsw_config.clone() {
                // Config was provided explicitly
                hnsw_config
            } else if self.current.is_vector_index_enabled_for(property_name) {
                // No config provided, but vector index exists - use its config
                self.current.get_hnsw_config_for(property_name).ok_or_else(|| {
                    crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                        format!(
                            "Vector index exists for '{}' but could not retrieve its configuration",
                            property_name
                        ),
                    ))
                })?
            } else {
                // No config provided and no vector index exists - error
                return Err(crate::core::error::Error::Vector(
                    crate::core::error::VectorError::IndexError(
                        "HNSW configuration is required when no vector index exists. \
                     Use TemporalVectorConfig::default_with_hnsw() to provide one, \
                     or enable the vector index first with enable_vector_index()."
                            .to_string(),
                    ),
                ));
            };

            // Enable vector index if it doesn't exist yet
            if !self.current.is_vector_index_enabled_for(property_name) {
                self.current
                    .enable_vector_index(property_name, resolved_hnsw_config.clone())?;
            }

            #[cfg(feature = "observability")]
            let _span = crate::observability::vector_index_span(
                "enable_temporal_vector_index",
                property_name,
            )
            .entered();

            // Create a resolved config with the hnsw_config set
            let resolved_config = TemporalVectorConfig {
                hnsw_config: Some(resolved_hnsw_config),
                ..config
            };

            // Enable temporal vector index in current storage
            self.current
                .enable_temporal_vector_index(property_name, resolved_config)?;

            // Get the temporal vector index for this property from current storage
            let temporal_index = self
                .current
                .get_temporal_vector_index_for(property_name)
                .ok_or_else(|| {
                    crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                        "Temporal vector index not found after enabling".to_string(),
                    ))
                })?;

            // Register pre-anchor hooks with historical storage (for strong consistency)
            // Both node and edge hooks perform the same action, so we create one and clone it
            let hook: crate::storage::historical::PreAnchorHook = {
                let index = Arc::clone(&temporal_index);
                Arc::new(move |_entity_type, _entity_id, timestamp, _properties| {
                    index.create_snapshot_for_anchor(timestamp)
                })
            };

            let node_hook = Arc::clone(&hook);
            let edge_hook = hook;

            let mut historical = self.historical.write();

            historical.register_pre_node_anchor_hook(node_hook);
            historical.register_pre_edge_anchor_hook(edge_hook);

            // Create observer and register with historical storage (for extensibility)
            let observer = VectorIndexObserver::new(temporal_index);
            historical.add_observer(std::sync::Arc::new(observer));

            Ok(())
        })();
        result.record_error_metric()
    }

    /// Check if temporal vector indexing is enabled.
    pub fn is_temporal_vector_index_enabled(&self) -> bool {
        self.current.is_temporal_vector_index_enabled()
    }

    /// Set the per-entity temporal vector snapshot policy for a node (Issue #383).
    ///
    /// Temporal vector snapshots are captured by a pre-anchor hook that fires
    /// when an entity's graph anchor is created. Historically this was global —
    /// every anchor on any entity re-snapshotted the whole vector index, even
    /// for entities whose embedding never changed. This method gives
    /// **per-entity control**: marking a node [`SnapshotPolicy::Skip`] keeps its
    /// graph anchors but suppresses its vector snapshots, while
    /// [`SnapshotPolicy::Snapshot`] (the default) preserves the prior behavior.
    ///
    /// Combine with [`set_default_node_vector_snapshot_policy`](Self::set_default_node_vector_snapshot_policy)
    /// for an opt-in model (default `Skip`, opt specific nodes back in).
    ///
    /// [`SnapshotPolicy`]: crate::storage::historical::SnapshotPolicy
    /// [`SnapshotPolicy::Skip`]: crate::storage::historical::SnapshotPolicy::Skip
    /// [`SnapshotPolicy::Snapshot`]: crate::storage::historical::SnapshotPolicy::Snapshot
    pub fn set_node_vector_snapshot_policy(
        &self,
        node_id: NodeId,
        policy: crate::storage::historical::SnapshotPolicy,
    ) {
        self.historical
            .write()
            .set_node_snapshot_policy(node_id, policy);
    }

    /// Set the per-entity temporal vector snapshot policy for an edge (Issue #383).
    ///
    /// The edge counterpart of
    /// [`set_node_vector_snapshot_policy`](Self::set_node_vector_snapshot_policy).
    pub fn set_edge_vector_snapshot_policy(
        &self,
        edge_id: crate::core::id::EdgeId,
        policy: crate::storage::historical::SnapshotPolicy,
    ) {
        self.historical
            .write()
            .set_edge_snapshot_policy(edge_id, policy);
    }

    /// Remove a node's per-entity snapshot policy override, reverting it to the
    /// current default node policy (Issue #383). Returns the removed override.
    pub fn clear_node_vector_snapshot_policy(
        &self,
        node_id: NodeId,
    ) -> Option<crate::storage::historical::SnapshotPolicy> {
        self.historical.write().clear_node_snapshot_policy(node_id)
    }

    /// Remove an edge's per-entity snapshot policy override, reverting it to the
    /// current default edge policy (Issue #383). Returns the removed override.
    pub fn clear_edge_vector_snapshot_policy(
        &self,
        edge_id: crate::core::id::EdgeId,
    ) -> Option<crate::storage::historical::SnapshotPolicy> {
        self.historical.write().clear_edge_snapshot_policy(edge_id)
    }

    /// Set the fall-through default node snapshot policy (Issue #383).
    ///
    /// Flip to [`SnapshotPolicy::Skip`] for an opt-in model where only nodes
    /// explicitly set to [`SnapshotPolicy::Snapshot`] are snapshotted.
    ///
    /// [`SnapshotPolicy::Skip`]: crate::storage::historical::SnapshotPolicy::Skip
    /// [`SnapshotPolicy::Snapshot`]: crate::storage::historical::SnapshotPolicy::Snapshot
    pub fn set_default_node_vector_snapshot_policy(
        &self,
        policy: crate::storage::historical::SnapshotPolicy,
    ) {
        self.historical
            .write()
            .set_default_node_snapshot_policy(policy);
    }

    /// Set the fall-through default edge snapshot policy (Issue #383).
    pub fn set_default_edge_vector_snapshot_policy(
        &self,
        policy: crate::storage::historical::SnapshotPolicy,
    ) {
        self.historical
            .write()
            .set_default_edge_snapshot_policy(policy);
    }

    /// Resolve the effective snapshot policy for a node (override if set, else
    /// the default node policy) (Issue #383).
    pub fn node_vector_snapshot_policy(
        &self,
        node_id: NodeId,
    ) -> crate::storage::historical::SnapshotPolicy {
        self.historical.read().node_snapshot_policy(node_id)
    }

    /// Resolve the effective snapshot policy for an edge (override if set, else
    /// the default edge policy) (Issue #383).
    pub fn edge_vector_snapshot_policy(
        &self,
        edge_id: crate::core::id::EdgeId,
    ) -> crate::storage::historical::SnapshotPolicy {
        self.historical.read().edge_snapshot_policy(edge_id)
    }

    /// List all property names that have temporal vector indexes enabled.
    ///
    /// Returns a vector of property names that have temporal vector indexing configured.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let db = AletheiaDB::new();
    /// // Enable temporal indexes for two properties
    /// db.vector_index("embedding1").hnsw(config1).temporal(temporal_config).enable()?;
    /// db.vector_index("embedding2").hnsw(config2).temporal(temporal_config).enable()?;
    ///
    /// let indexes = db.list_temporal_vector_indexes();
    /// assert!(indexes.contains(&"embedding1".to_string()));
    /// assert!(indexes.contains(&"embedding2".to_string()));
    /// ```
    pub fn list_temporal_vector_indexes(&self) -> Vec<String> {
        self.current.list_temporal_vector_indexes()
    }

    /// Create a builder for configuring a vector index on a property.
    ///
    /// This provides a fluent API for enabling vector indexes with optional
    /// temporal configuration. The builder pattern ensures proper configuration
    /// before enabling the index.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property name that will contain vector embeddings
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aletheiadb::index::vector::{HnswConfig, DistanceMetric};
    ///
    /// // Basic vector index
    /// db.vector_index("embedding")
    ///     .hnsw(HnswConfig::new(384, DistanceMetric::Cosine))
    ///     .enable()?;
    ///
    /// // With temporal indexing for time-travel queries
    /// db.vector_index("embedding")
    ///     .hnsw(HnswConfig::new(384, DistanceMetric::Cosine))
    ///     .temporal(TemporalVectorConfig::default())
    ///     .enable()?;
    /// ```
    pub fn vector_index(&self, property_name: &str) -> VectorIndexBuilder<'_> {
        VectorIndexBuilder::new(self, property_name.to_string())
    }

    /// Check if a vector index is enabled for a specific property.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property name to check
    ///
    /// # Example
    ///
    /// ```ignore
    /// db.vector_index("embedding")
    ///     .hnsw(config)
    ///     .enable()?;
    ///
    /// assert!(db.has_vector_index("embedding"));
    /// assert!(!db.has_vector_index("other_property"));
    /// ```
    pub fn has_vector_index(&self, property_name: &str) -> bool {
        self.current.has_vector_index(property_name)
    }

    /// List all enabled vector indexes.
    ///
    /// Returns information about each configured vector index including
    /// the property name, dimensions, and distance metric.
    ///
    /// # Example
    ///
    /// ```ignore
    /// db.vector_index("title_embedding")
    ///     .hnsw(HnswConfig::new(384, DistanceMetric::Cosine))
    ///     .enable()?;
    ///
    /// db.vector_index("body_embedding")
    ///     .hnsw(HnswConfig::new(768, DistanceMetric::Euclidean))
    ///     .enable()?;
    ///
    /// let indexes = db.list_vector_indexes();
    /// assert_eq!(indexes.len(), 2);
    /// ```
    pub fn list_vector_indexes(&self) -> Vec<crate::storage::VectorIndexInfo> {
        self.current.list_vector_indexes()
    }

    /// Find k most similar nodes in a specific property's vector index.
    ///
    /// Use this method when you have multiple vector indexes and need to
    /// search a specific one. The query node's embedding from the specified
    /// property is used for the search.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The indexed property to search
    /// * `query_node_id` - The node to find similar nodes for
    /// * `k` - Maximum number of results to return
    ///
    /// # Example
    ///
    /// ```rust
    /// # use aletheiadb::{AletheiaDB, WriteOps, properties};
    /// # use aletheiadb::index::vector::{HnswConfig, DistanceMetric};
    /// # use aletheiadb::core::error::Error;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let db = AletheiaDB::open(dir.path())?;
    /// # db.enable_vector_index("title_embedding", HnswConfig::new(16, DistanceMetric::Cosine))?;
    /// # db.enable_vector_index("body_embedding", HnswConfig::new(16, DistanceMetric::Cosine))?;
    /// # let node_id = db.write(|tx| -> Result<_, Error> {
    /// #     let id = tx.create_node("Doc", properties! {
    /// #         "title_embedding" => vec![0.1_f32; 16],
    /// #         "body_embedding" => vec![0.2_f32; 16]
    /// #     })?;
    /// #     Ok(id)
    /// # })?;
    /// // Search title embeddings for similar nodes
    /// let similar = db.find_similar_in("title_embedding", node_id, 10)?;
    ///
    /// // Search body embeddings (different property, potentially different results)
    /// let similar_body = db.find_similar_in("body_embedding", node_id, 10)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No vector index is enabled for the specified property
    /// - Query node is not found
    /// - Query node does not have the specified vector property
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_in(
        &self,
        property_name: &str,
        query_node_id: NodeId,
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("find_similar_in", property_name).entered();
        self.current
            .find_similar_in(property_name, query_node_id, k)
            .record_error_metric()
    }

    /// Search a specific property's vector index with a raw embedding.
    ///
    /// Use this method when searching with embeddings that don't correspond to
    /// any existing node in the graph (e.g., query embeddings from external sources).
    ///
    /// # Arguments
    ///
    /// * `property_name` - The indexed property to search
    /// * `embedding` - The query embedding vector
    /// * `k` - Maximum number of results to return
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Search with external embedding
    /// let query = embed_text("search query");
    /// let results = db.search_vectors_in("title_embedding", &query, 10)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No vector index is enabled for the specified property
    /// - Embedding dimensions don't match the index configuration
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn search_vectors_in(
        &self,
        property_name: &str,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("search_vectors_in", property_name).entered();
        self.current
            .search_vectors_in(property_name, embedding, k)
            .record_error_metric()
    }

    /// Run a vector similarity search described by a [`SimilarityQuery`].
    ///
    /// This is the unified entry point for vector search. It consolidates the
    /// previously separate `find_similar*` methods (node-based, embedding-based,
    /// label-filtered, and temporal) behind a single extensible builder, so that
    /// adding a new filter does not require adding a new database method.
    ///
    /// # Arguments
    ///
    /// * `query` - A [`SimilarityQuery`] built via [`SimilarityQuery::from_node`]
    ///   or [`SimilarityQuery::from_embedding`] and refined with `.k()`,
    ///   `.with_label()`, and `.at_time()`.
    ///
    /// # Returns
    ///
    /// A list of `(NodeId, similarity_score)` pairs sorted by similarity (highest first).
    ///
    /// # Errors
    ///
    /// Returns an error if the appropriate vector index is not enabled, the query
    /// node/embedding is invalid, or the requested combination of filters is not
    /// supported (currently: a node-based temporal search, or a label-filtered
    /// temporal search).
    ///
    /// Note: a temporal search built with `.at_time(..)` has no property
    /// selector; with multiple temporal vector indexes enabled it queries the
    /// alphabetically first indexed property. Use
    /// [`find_similar_as_of_in`](Self::find_similar_as_of_in) to target a
    /// specific property.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use aletheiadb::{AletheiaDB, WriteOps, properties, SimilarityQuery};
    /// # use aletheiadb::index::vector::{HnswConfig, DistanceMetric};
    /// # use aletheiadb::index::vector::temporal::TemporalVectorConfig;
    /// # use aletheiadb::core::error::Error;
    /// # use aletheiadb::core::temporal::time::now;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let db = AletheiaDB::open(dir.path())?;
    /// # db.enable_vector_index("embedding", HnswConfig::new(16, DistanceMetric::Cosine))?;
    /// # db.enable_temporal_vector_index("embedding", TemporalVectorConfig::default_with_hnsw(HnswConfig::new(16, DistanceMetric::Cosine)))?;
    /// # let doc_id = db.write(|tx| -> Result<_, Error> {
    /// #     let id = tx.create_node("Document", properties! {
    /// #         "embedding" => vec![0.1_f32; 16]
    /// #     })?;
    /// #     Ok(id)
    /// # })?;
    /// # let ts = now();
    /// # let query_embedding = vec![0.2_f32; 16];
    /// // Find similar Documents to an existing node.
    /// let results = db.similarity_search(
    ///     SimilarityQuery::from_node(doc_id).k(5).with_label("Document"),
    /// )?;
    ///
    /// // Find similar nodes to a raw embedding at a point in time.
    /// let results = db.similarity_search(
    ///     SimilarityQuery::from_embedding(&query_embedding).k(10).at_time(ts),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn similarity_search(&self, query: SimilarityQuery) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span = crate::observability::vector_search_span("similarity_search", "").entered();

        let k = query.limit();
        let label = query.label();
        let timestamp = query.timestamp();

        let result = match (query.source(), label, timestamp) {
            (SimilaritySource::Node(node_id), None, None) => self.current.find_similar(*node_id, k),
            (SimilaritySource::Node(node_id), Some(label), None) => {
                self.current.find_similar_with_label(*node_id, label, k)
            }
            (SimilaritySource::Embedding(embedding), None, None) => {
                self.current.find_similar_by_embedding(embedding, k)
            }
            (SimilaritySource::Embedding(embedding), Some(label), None) => self
                .current
                .find_similar_by_embedding_with_label(embedding, label, k),
            (SimilaritySource::Embedding(embedding), None, Some(ts)) => {
                self.current.find_similar_as_of(embedding, k, ts)
            }
            (SimilaritySource::Node(_), _, Some(_)) => Err(crate::core::error::Error::Vector(
                crate::core::error::VectorError::IndexError(
                    "Temporal similarity search from a node is not supported. \
                         Use SimilarityQuery::from_embedding(...).at_time(...) instead."
                        .to_string(),
                ),
            )),
            (SimilaritySource::Embedding(_), Some(_), Some(_)) => Err(
                crate::core::error::Error::Vector(crate::core::error::VectorError::IndexError(
                    "Label-filtered temporal similarity search is not supported.".to_string(),
                )),
            ),
        };

        result.record_error_metric()
    }

    /// Find k most similar nodes to a query node based on vector similarity.
    ///
    /// Returns a list of (NodeId, score) pairs sorted by similarity (highest first).
    /// The query node itself is excluded from results.
    ///
    /// # Arguments
    ///
    /// * `query_node_id` - The node to find similar nodes for
    /// * `k` - Maximum number of results to return
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Find the 5 most similar documents to a given document
    /// let results = db.find_similar(doc_id, 5)?;
    /// for (node_id, score) in results {
    ///     println!("Similar node {} with score {}", node_id, score);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vector index is not enabled
    /// - Query node is not found
    /// - Query node does not have the indexed vector property
    #[deprecated(
        since = "0.1.0",
        note = "use db.similarity_search(SimilarityQuery::from_node(id).k(k)) instead"
    )]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar(&self, query_node_id: NodeId, k: usize) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span = crate::observability::vector_search_span("find_similar", "").entered();
        self.current
            .find_similar(query_node_id, k)
            .record_error_metric()
    }

    /// Find k most similar nodes with a specific label.
    ///
    /// This is useful for finding similar nodes within a category, e.g.,
    /// "find similar documents" or "find similar users".
    ///
    /// # Arguments
    ///
    /// * `query_node_id` - The node to find similar nodes for
    /// * `label` - Only return nodes with this label
    /// * `k` - Maximum number of results to return
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Find similar Person nodes only
    /// let similar_people = db.find_similar_with_label(person_id, "Person", 10)?;
    /// ```
    #[deprecated(
        since = "0.1.0",
        note = "use db.similarity_search(SimilarityQuery::from_node(id).k(k).with_label(label)) instead"
    )]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_with_label(
        &self,
        query_node_id: NodeId,
        label: &str,
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("find_similar_with_label", "").entered();
        self.current
            .find_similar_with_label(query_node_id, label, k)
            .record_error_metric()
    }

    /// Find k most similar nodes to a raw embedding vector.
    ///
    /// This is useful when searching with embeddings that don't correspond to any
    /// existing node in the graph, such as query embeddings from external sources
    /// or user input.
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
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Search with an embedding from external source (e.g., user query)
    /// let query_embedding = get_embedding_from_llm("rust programming");
    /// let similar = db.find_similar_by_embedding(&query_embedding, 10)?;
    /// for (node_id, similarity) in similar {
    ///     println!("Node {:?} has similarity {}", node_id, similarity);
    /// }
    /// ```
    #[deprecated(
        since = "0.1.0",
        note = "use db.similarity_search(SimilarityQuery::from_embedding(emb).k(k)) instead"
    )]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_by_embedding(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("find_similar_by_embedding", "").entered();
        self.current
            .find_similar_by_embedding(embedding, k)
            .record_error_metric()
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
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Find similar documents only
    /// let query_embedding = get_embedding_from_llm("rust programming");
    /// let similar_docs = db.find_similar_by_embedding_with_label(
    ///     &query_embedding,
    ///     "Document",
    ///     5
    /// )?;
    /// ```
    #[deprecated(
        since = "0.1.0",
        note = "use db.similarity_search(SimilarityQuery::from_embedding(emb).k(k).with_label(label)) instead"
    )]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_by_embedding_with_label(
        &self,
        embedding: &[f32],
        label: &str,
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("find_similar_by_embedding_with_label", "")
                .entered();
        self.current
            .find_similar_by_embedding_with_label(embedding, label, k)
            .record_error_metric()
    }

    /// Find k most similar nodes using a custom predicate for filtering.
    ///
    /// This method allows executing a vector search where candidates are filtered
    /// by an arbitrary predicate closure. This is useful for complex filtering logic
    /// that cannot be expressed as a simple label check.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property containing the vector embeddings
    /// * `query_vector` - The query embedding vector
    /// * `k` - Maximum number of results to return
    /// * `predicate` - A closure that takes a `NodeId` and returns `true` if the node should be included
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_with_predicate<F>(
        &self,
        property_name: &str,
        query_vector: &[f32],
        k: usize,
        predicate: F,
    ) -> Result<Vec<(NodeId, f32)>>
    where
        F: Fn(&NodeId) -> bool + Send + Sync,
    {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("find_similar_with_predicate", "").entered();
        self.current
            .find_similar_with_predicate(property_name, query_vector, k, predicate)
            .record_error_metric()
    }

    /// Find k most similar nodes at a specific point in time.
    ///
    /// This method performs a temporal vector search, finding nodes with embeddings
    /// most similar to the query embedding as they existed at the specified timestamp.
    ///
    /// With multiple temporal vector indexes enabled and no property specified,
    /// the alphabetically first indexed property is queried. Prefer
    /// [`find_similar_as_of_in`](Self::find_similar_as_of_in) with an explicit
    /// property to make the target index unambiguous.
    ///
    /// # Arguments
    ///
    /// * `embedding` - Query embedding vector to search for
    /// * `k` - Maximum number of results to return
    /// * `timestamp` - Point in time to query (in microseconds since epoch)
    ///
    /// # Returns
    ///
    /// A vector of (NodeId, similarity_score) tuples, sorted by similarity in descending order.
    ///
    /// # Errors
    ///
    /// - `Error::Vector(VectorError::IndexError)` if temporal vector index is not enabled
    /// - `Error::Vector(VectorError::*)` if the query embedding is invalid
    /// - `Error::Temporal(*)` if no snapshot exists at the given timestamp
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use aletheiadb::AletheiaDB;
    /// # use aletheiadb::core::temporal::Timestamp;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = AletheiaDB::new()?;
    /// # let query_embedding = vec![0.1, 0.2, 0.3];
    /// // Find documents similar to a query at a specific point in time
    /// let timestamp_2023 = Timestamp::from(1672531200000000); // 2023-01-01 in microseconds
    /// let results = db.find_similar_as_of(&query_embedding, 10, timestamp_2023)?;
    /// # Ok(())
    /// # }
    /// ```
    #[deprecated(
        since = "0.1.0",
        note = "use db.similarity_search(SimilarityQuery::from_embedding(emb).k(k).at_time(ts)) instead"
    )]
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_as_of(
        &self,
        embedding: &[f32],
        k: usize,
        timestamp: Timestamp,
    ) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span = crate::observability::vector_search_span("find_similar_as_of", "").entered();
        self.current
            .find_similar_as_of(embedding, k, timestamp)
            .record_error_metric()
    }

    /// Find similar vectors at a specific point in time for a specific property.
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
    /// * `timestamp` - The point in time to query (in microseconds since epoch)
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
    /// ```rust,no_run
    /// # use aletheiadb::AletheiaDB;
    /// # use aletheiadb::core::temporal::Timestamp;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = AletheiaDB::new()?;
    /// # let query_embedding = vec![0.1, 0.2, 0.3];
    /// // Find documents similar to a query at a specific point in time
    /// let timestamp_2023 = Timestamp::from(1672531200000000); // 2023-01-01 in microseconds
    /// let results = db.find_similar_as_of_in(
    ///     "content_embedding",
    ///     &query_embedding,
    ///     10,
    ///     timestamp_2023
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_similar_as_of_in(
        &self,
        property_name: &str,
        embedding: &[f32],
        k: usize,
        timestamp: Timestamp,
    ) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("find_similar_as_of_in", property_name)
                .entered();
        self.current
            .find_similar_as_of_in(property_name, embedding, k, timestamp)
            .record_error_metric()
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
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use aletheiadb::AletheiaDB;
    /// # use aletheiadb::core::NodeId;
    /// use aletheiadb::core::temporal::TimeRange;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = AletheiaDB::new()?;
    /// # let node_id = NodeId::new(1)?;
    /// # let original_embedding = vec![0.1, 0.2, 0.3];
    /// // Track how a document's embedding changed from its original version
    /// let time_range = TimeRange::new(0.into(), i64::MAX.into()).unwrap();
    /// let drift = db.track_drift_in(
    ///     "content_embedding",
    ///     node_id,
    ///     &original_embedding,
    ///     time_range
    /// )?;
    ///
    /// for (timestamp, distance) in drift {
    ///     println!("At {:?}: drift = {:.3}", timestamp, distance);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn track_drift_in(
        &self,
        property_name: &str,
        node_id: NodeId,
        reference_embedding: &[f32],
        time_range: crate::core::temporal::TimeRange,
    ) -> Result<Vec<(Timestamp, f32)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("track_drift_in", property_name).entered();
        self.current
            .track_drift_in(property_name, node_id, reference_embedding, time_range)
            .record_error_metric()
    }

    /// Get the semantic evolution of a node's embedding over time in a specific property.
    ///
    /// Returns the actual embedding vectors at each snapshot timestamp, allowing
    /// you to see how the node's semantic representation changed over time.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property containing the vector embeddings
    /// * `node_id` - The node to get evolution for
    /// * `time_range` - The time range to query
    ///
    /// # Returns
    ///
    /// A vector of (timestamp, embedding) pairs showing the node's embedding
    /// at each snapshot time within the range.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Temporal vector index is not enabled
    /// - The property name doesn't match the indexed property
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use aletheiadb::AletheiaDB;
    /// # use aletheiadb::core::NodeId;
    /// use aletheiadb::core::temporal::TimeRange;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = AletheiaDB::new()?;
    /// # let node_id = NodeId::new(1)?;
    /// let time_range = TimeRange::new(0.into(), i64::MAX.into()).unwrap();
    /// let evolution = db.semantic_evolution_in("content_embedding", node_id, time_range)?;
    ///
    /// for (timestamp, embedding) in evolution {
    ///     println!("At {:?}: {} dimensions", timestamp, embedding.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn semantic_evolution_in(
        &self,
        property_name: &str,
        node_id: NodeId,
        time_range: crate::core::temporal::TimeRange,
    ) -> Result<Vec<(Timestamp, std::sync::Arc<[f32]>)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("semantic_evolution_in", property_name)
                .entered();
        self.current
            .semantic_evolution_in(property_name, node_id, time_range)
            .record_error_metric()
    }

    /// Find all nodes with semantic drift above a threshold in a specific property.
    ///
    /// Scans all nodes in the temporal index and identifies those whose embeddings
    /// have changed by more than the specified threshold over the time range.
    ///
    /// # Arguments
    ///
    /// * `property_name` - The property containing the vector embeddings
    /// * `threshold` - Minimum drift distance to include in results
    /// * `time_range` - The time range to analyze
    /// * `metric` - The distance metric to use for drift calculation
    ///
    /// # Returns
    ///
    /// A vector of (node_id, drift_score) pairs for nodes exceeding the threshold,
    /// sorted by drift score in descending order.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Temporal vector index is not enabled
    /// - The property name doesn't match the indexed property
    /// - Threshold is NaN or infinite
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use aletheiadb::AletheiaDB;
    /// use aletheiadb::core::temporal::TimeRange;
    /// use aletheiadb::index::vector::temporal::DriftMetric;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = AletheiaDB::new()?;
    /// # let start_ts = 0;
    /// # let end_ts = i64::MAX;
    /// let time_range = TimeRange::new(start_ts.into(), end_ts.into()).unwrap();
    /// let drifted = db.find_drift_in(
    ///     "content_embedding",
    ///     0.3,  // threshold
    ///     time_range,
    ///     DriftMetric::Cosine
    /// )?;
    ///
    /// for (node_id, drift) in drifted {
    ///     println!("Node {} drifted by {:.3}", node_id, drift);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn find_drift_in(
        &self,
        property_name: &str,
        threshold: f32,
        time_range: crate::core::temporal::TimeRange,
        metric: crate::index::vector::temporal::DriftMetric,
    ) -> Result<Vec<(NodeId, f32)>> {
        #[cfg(feature = "observability")]
        let _span =
            crate::observability::vector_search_span("find_drift_in", property_name).entered();
        self.current
            .find_drift_in(property_name, threshold, time_range, metric)
            .record_error_metric()
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;

    #[test]
    fn test_enable_temporal_vector_index_missing_config() {
        let db = AletheiaDB::new().unwrap();
        // Don't enable vector index first
        // Provide config without HNSW config
        let config = TemporalVectorConfig::default();

        let result = db.enable_temporal_vector_index("embedding", config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("HNSW configuration is required"));
    }

    // --- Issue #383: per-entity snapshot policy wrappers, end-to-end through AletheiaDB ---
    //
    // These exercise the public `AletheiaDB` surface end-to-end (set/get/clear/
    // default, node + edge) through the historical `RwLock`. The *behavioral*
    // proof that the policy gates the pre-anchor snapshot hook per entity lives
    // in `src/storage/historical/tests.rs` (recording-hook tests), where the
    // anchor-hook trigger is observed directly and in isolation from the
    // orthogonal transaction-strategy snapshot path.

    use crate::core::id::EdgeId;
    use crate::storage::historical::SnapshotPolicy;

    /// The default resolved policy is `Snapshot` for both nodes and edges,
    /// preserving pre-#383 behavior when nothing is configured.
    #[test]
    fn test_snapshot_policy_wrappers_default_is_snapshot() {
        let db = AletheiaDB::new().unwrap();
        assert_eq!(
            db.node_vector_snapshot_policy(NodeId::new(1).unwrap()),
            SnapshotPolicy::Snapshot
        );
        assert_eq!(
            db.edge_vector_snapshot_policy(EdgeId::new(1).unwrap()),
            SnapshotPolicy::Snapshot
        );
    }

    /// Setting, reading back, and clearing a per-node override round-trips
    /// through the historical `RwLock`.
    #[test]
    fn test_snapshot_policy_wrappers_node_set_get_clear() {
        let db = AletheiaDB::new().unwrap();
        let node = NodeId::new(7).unwrap();

        db.set_node_vector_snapshot_policy(node, SnapshotPolicy::Skip);
        assert_eq!(db.node_vector_snapshot_policy(node), SnapshotPolicy::Skip);

        assert_eq!(
            db.clear_node_vector_snapshot_policy(node),
            Some(SnapshotPolicy::Skip)
        );
        assert_eq!(
            db.node_vector_snapshot_policy(node),
            SnapshotPolicy::Snapshot,
            "cleared override reverts to default"
        );
    }

    /// The default-node policy setter creates an opt-in model; edges are a
    /// separate registry and remain at their own default.
    #[test]
    fn test_snapshot_policy_wrappers_default_opt_in_node_edge_independent() {
        let db = AletheiaDB::new().unwrap();

        db.set_default_node_vector_snapshot_policy(SnapshotPolicy::Skip);
        // Every unset node now resolves to Skip...
        assert_eq!(
            db.node_vector_snapshot_policy(NodeId::new(99).unwrap()),
            SnapshotPolicy::Skip
        );
        // ...but a specific node can be opted back in.
        db.set_node_vector_snapshot_policy(NodeId::new(1).unwrap(), SnapshotPolicy::Snapshot);
        assert_eq!(
            db.node_vector_snapshot_policy(NodeId::new(1).unwrap()),
            SnapshotPolicy::Snapshot
        );

        // Edges are an independent registry — unaffected by the node default.
        assert_eq!(
            db.edge_vector_snapshot_policy(EdgeId::new(1).unwrap()),
            SnapshotPolicy::Snapshot
        );
        db.set_default_edge_vector_snapshot_policy(SnapshotPolicy::Skip);
        db.set_edge_vector_snapshot_policy(EdgeId::new(2).unwrap(), SnapshotPolicy::Snapshot);
        assert_eq!(
            db.edge_vector_snapshot_policy(EdgeId::new(1).unwrap()),
            SnapshotPolicy::Skip
        );
        assert_eq!(
            db.edge_vector_snapshot_policy(EdgeId::new(2).unwrap()),
            SnapshotPolicy::Snapshot
        );
    }
}
