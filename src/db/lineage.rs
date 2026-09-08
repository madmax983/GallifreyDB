//! Database-level derivation lineage API (Issue #3371).
//!
//! Wires the in-memory [`LineageStore`](crate::core::lineage::LineageStore)
//! into the write path and exposes upstream/downstream closure queries. A
//! writer declares, at write time, that a fact was *derived from* a set of
//! existing fact versions; the closure queries then answer, in both
//! directions, "what was this derived from?" and "what has been derived from
//! this?" — the citation and blast-radius questions.
//!
//! # Reference validation
//!
//! Every declared source reference is resolved against historical storage
//! *before* the write commits: an unknown version, or a version whose entity
//! does not match the reference, fails the write with
//! [`LineageError::SourceNotFound`] and nothing is written. Because version
//! ids are immutable and append-only, a source that resolves at validation
//! time cannot disappear before the record is written.
//!
//! # Status resolution
//!
//! Closure entries carry a [`FactStatus`] resolved against *current* state:
//! `Current` (the referenced version is the entity's live version),
//! `Superseded` (a newer version exists), or `Absent` (the entity is gone from
//! current state — deleted or retracted). Lineage records are never mutated by
//! those downstream events (Issue #3371 immutability), so the closure over a
//! retracted input still resolves and marks that input `Absent`.

use crate::api::transaction::{WriteOps, WriteRequestOptions};
use crate::core::error::{Error, Result, ResultExt};
use crate::core::id::{EdgeId, EntityId, NodeId, VersionId};
use crate::core::lineage::{
    LineageClosure, LineageError, LineageQueryOptions, LineageRecord, LineageRef,
};
use crate::core::property::PropertyMap;
use crate::db::AletheiaDB;

/// The current-state status of a fact version referenced by a lineage record.
///
/// Resolved at query time against current storage; a lineage record itself is
/// immutable and does not change when the fact it points at is superseded or
/// retracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactStatus {
    /// The referenced version is the entity's live current version.
    Current,
    /// The entity still exists but has a newer version than the one
    /// referenced (the referenced version was superseded by an update).
    Superseded,
    /// The entity is absent from current state (deleted or retracted). The
    /// lineage reference still resolves against recorded history.
    Absent,
}

impl FactStatus {
    /// Lowercase string form for JSON/display surfaces.
    pub const fn as_str(self) -> &'static str {
        match self {
            FactStatus::Current => "current",
            FactStatus::Superseded => "superseded",
            FactStatus::Absent => "absent",
        }
    }
}

/// One entry in a resolved lineage closure: a referenced fact version, the
/// minimum hop distance from the query root, and the referenced fact's
/// current-state status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineageViewEntry {
    /// The reached fact version (entity + version).
    pub reference: LineageRef,
    /// Minimum hop distance from the query root (`1` = direct
    /// parent/child).
    pub depth: usize,
    /// Current-state status of the referenced fact.
    pub status: FactStatus,
}

/// A resolved lineage closure returned by the database API.
///
/// Like [`LineageClosure`] but with each entry's current-state
/// [`FactStatus`] resolved. Entries are ordered breadth-first (increasing
/// depth) and exclude the query root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageView {
    /// The fact the closure was computed from.
    pub root: LineageRef,
    /// The reachable fact versions with depth and status.
    pub entries: Vec<LineageViewEntry>,
    /// `true` if traversal stopped at the entry `limit` or `max_depth` bound
    /// with more of the graph unexplored (mirrors the `has_more` pagination
    /// convention, Issue #3226).
    pub has_more: bool,
}

impl LineageView {
    /// Number of distinct fact versions in the closure (excludes the root).
    #[inline]
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

impl AletheiaDB {
    /// Create a node declared as derived from `derived_from` — a set of
    /// version-pinned references to existing facts (Issue #3371).
    ///
    /// The node is created exactly as [`create_node`](Self::create_node)
    /// would, then a lineage record is written binding the new version to the
    /// declared sources. An empty `derived_from` is identical to
    /// `create_node` (no lineage recorded).
    ///
    /// # Errors
    ///
    /// - [`LineageError::SourceNotFound`] (as [`Error::Lineage`])
    ///   if any reference does not resolve to an existing fact version — the
    ///   node is **not** created.
    /// - Any error [`create_node`](Self::create_node) can return.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn create_node_with_lineage(
        &self,
        label: &str,
        properties: PropertyMap,
        derived_from: &[LineageRef],
    ) -> Result<NodeId> {
        self.create_node_with_options_and_lineage(
            label,
            properties,
            WriteRequestOptions::new(),
            derived_from,
        )
        .map(|(node_id, _version)| node_id)
    }

    /// Create an edge declared as derived from `derived_from` (Issue #3371).
    ///
    /// See [`create_node_with_lineage`](Self::create_node_with_lineage).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn create_edge_with_lineage(
        &self,
        source: NodeId,
        target: NodeId,
        label: &str,
        properties: PropertyMap,
        derived_from: &[LineageRef],
    ) -> Result<EdgeId> {
        self.create_edge_with_options_and_lineage(
            source,
            target,
            label,
            properties,
            WriteRequestOptions::new(),
            derived_from,
        )
        .map(|(edge_id, _version)| edge_id)
    }

    /// Update a node, declaring the new version as derived from `derived_from`
    /// (Issue #3371).
    ///
    /// The lineage is recorded against the *new* version produced by the
    /// update; the pre-update version's own lineage (if any) is untouched.
    /// This is how a correction records which inputs justified it.
    ///
    /// # Errors
    ///
    /// See [`create_node_with_lineage`](Self::create_node_with_lineage).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn update_node_with_lineage(
        &self,
        node_id: NodeId,
        properties: PropertyMap,
        derived_from: &[LineageRef],
    ) -> Result<VersionId> {
        self.update_node_with_options_and_lineage(
            node_id,
            properties,
            WriteRequestOptions::new(),
            derived_from,
        )
    }

    /// Update an edge, declaring the new version as derived from
    /// `derived_from` (Issue #3371).
    ///
    /// See [`update_node_with_lineage`](Self::update_node_with_lineage).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub fn update_edge_with_lineage(
        &self,
        edge_id: EdgeId,
        properties: PropertyMap,
        derived_from: &[LineageRef],
    ) -> Result<VersionId> {
        self.update_edge_with_options_and_lineage(
            edge_id,
            properties,
            WriteRequestOptions::new(),
            derived_from,
        )
    }

    /// Create a node with a full [`WriteRequestOptions`] bundle (backdated
    /// `valid_from` and/or write-time provenance) *and* a derivation lineage
    /// declaration, returning both the new id and the exact version produced
    /// (Issue #3371).
    ///
    /// This is the combined write+lineage entry point the MCP surface uses so
    /// `derived_from` composes with `valid_time`/`provenance`. The lineage
    /// record binds to the **exact** [`VersionId`] this write produced — read
    /// out of the transaction buffer before commit, never re-derived from a
    /// later "current version" read, so two concurrent same-entity writes each
    /// attribute lineage to their own version with no post-commit races.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub(crate) fn create_node_with_options_and_lineage(
        &self,
        label: &str,
        properties: PropertyMap,
        options: WriteRequestOptions,
        derived_from: &[LineageRef],
    ) -> Result<(NodeId, VersionId)> {
        self.validate_sources(derived_from)?;
        let (node_id, version) = self.write(|tx| {
            let node_id = tx.create_node_with_options(label, properties, options)?;
            let version = tx.buffered_node_version(node_id).ok_or_else(|| {
                Error::from(crate::core::error::StorageError::NodeNotFound(node_id))
            })?;
            Ok::<_, Error>((node_id, version))
        })?;
        self.record_lineage(EntityId::Node(node_id), version, derived_from)?;
        Ok((node_id, version))
    }

    /// Edge counterpart of
    /// [`create_node_with_options_and_lineage`](Self::create_node_with_options_and_lineage).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub(crate) fn create_edge_with_options_and_lineage(
        &self,
        source: NodeId,
        target: NodeId,
        label: &str,
        properties: PropertyMap,
        options: WriteRequestOptions,
        derived_from: &[LineageRef],
    ) -> Result<(EdgeId, VersionId)> {
        self.validate_sources(derived_from)?;
        let (edge_id, version) = self.write(|tx| {
            let edge_id =
                tx.create_edge_with_options(source, target, label, properties, options)?;
            let version = tx.buffered_edge_version(edge_id).ok_or_else(|| {
                Error::from(crate::core::error::StorageError::EdgeNotFound(edge_id))
            })?;
            Ok::<_, Error>((edge_id, version))
        })?;
        self.record_lineage(EntityId::Edge(edge_id), version, derived_from)?;
        Ok((edge_id, version))
    }

    /// Update a node with a full [`WriteRequestOptions`] bundle and a
    /// derivation lineage declaration, returning the exact new version
    /// (Issue #3371).
    ///
    /// See
    /// [`create_node_with_options_and_lineage`](Self::create_node_with_options_and_lineage)
    /// for why the version is captured from the transaction rather than
    /// re-read.
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub(crate) fn update_node_with_options_and_lineage(
        &self,
        node_id: NodeId,
        properties: PropertyMap,
        options: WriteRequestOptions,
        derived_from: &[LineageRef],
    ) -> Result<VersionId> {
        self.validate_sources(derived_from)?;
        let version = self.write(|tx| {
            tx.update_node_with_options(node_id, properties, options)?;
            tx.buffered_node_version(node_id)
                .ok_or_else(|| Error::from(crate::core::error::StorageError::NodeNotFound(node_id)))
        })?;
        self.record_lineage(EntityId::Node(node_id), version, derived_from)?;
        Ok(version)
    }

    /// Edge counterpart of
    /// [`update_node_with_options_and_lineage`](Self::update_node_with_options_and_lineage).
    #[must_use = "this Result must be used; ignoring errors can lead to silent failures"]
    pub(crate) fn update_edge_with_options_and_lineage(
        &self,
        edge_id: EdgeId,
        properties: PropertyMap,
        options: WriteRequestOptions,
        derived_from: &[LineageRef],
    ) -> Result<VersionId> {
        self.validate_sources(derived_from)?;
        let version = self.write(|tx| {
            tx.update_edge_with_options(edge_id, properties, options)?;
            tx.buffered_edge_version(edge_id)
                .ok_or_else(|| Error::from(crate::core::error::StorageError::EdgeNotFound(edge_id)))
        })?;
        self.record_lineage(EntityId::Edge(edge_id), version, derived_from)?;
        Ok(version)
    }

    /// The lineage reference for a node's *current* version, or `None` if the
    /// node has no current version. Convenience for building a closure root
    /// or a `derived_from` list from live entities.
    pub fn node_lineage_ref(&self, node_id: NodeId) -> Option<LineageRef> {
        self.historical
            .read()
            .get_current_node_version(node_id)
            .map(|version| LineageRef::new(node_id, version))
    }

    /// The lineage reference for an edge's *current* version, or `None`.
    pub fn edge_lineage_ref(&self, edge_id: EdgeId) -> Option<LineageRef> {
        self.historical
            .read()
            .get_current_edge_version(edge_id)
            .map(|version| LineageRef::new(edge_id, version))
    }

    /// The direct lineage record for a fact version, if one was declared.
    pub fn lineage_record(&self, version: VersionId) -> Option<LineageRecord> {
        self.lineage.record_for(version)
    }

    /// Upstream lineage closure of `root`: transitively, "what was this fact
    /// derived from?" (Issue #3371).
    ///
    /// Bounded by `options` (depth, entry limit, and optional `AS OF`
    /// transaction-time coordinate). Each entry's current-state
    /// [`FactStatus`] is resolved.
    pub fn upstream_lineage(&self, root: LineageRef, options: LineageQueryOptions) -> LineageView {
        let closure = self.lineage.upstream_closure(root, options);
        self.resolve_view(closure)
    }

    /// Downstream lineage closure of `root`: transitively, "what has been
    /// derived from this fact?" — the retraction blast-radius report
    /// (Issue #3371).
    ///
    /// Bounded by `options`. Each entry carries its minimum hop depth and
    /// current-state [`FactStatus`], so a caller can enumerate every affected
    /// downstream fact when an input is found wrong or retracted.
    pub fn downstream_lineage(
        &self,
        root: LineageRef,
        options: LineageQueryOptions,
    ) -> LineageView {
        let closure = self.lineage.downstream_closure(root, options);
        self.resolve_view(closure)
    }

    // ---- internal helpers ----

    /// Validate that every declared source reference resolves to an existing
    /// fact version whose entity matches the reference.
    fn validate_sources(&self, sources: &[LineageRef]) -> Result<()> {
        if sources.is_empty() {
            return Ok(());
        }
        let historical = self.historical.read();
        for reference in sources {
            let resolved = match reference.entity {
                EntityId::Node(node_id) => historical
                    .get_node_version(reference.version)
                    .is_some_and(|v| v.node_id == node_id),
                EntityId::Edge(edge_id) => historical
                    .get_edge_version(reference.version)
                    .is_some_and(|v| v.edge_id == edge_id),
            };
            if !resolved {
                return Err(LineageError::SourceNotFound {
                    reference: *reference,
                }
                .into());
            }
        }
        Ok(())
    }

    /// Record lineage for a freshly written version. `recorded_at` is the
    /// version's own commit (transaction) time so `AS OF` lineage queries
    /// align with the write's transaction time.
    fn record_lineage(
        &self,
        entity: EntityId,
        version: VersionId,
        sources: &[LineageRef],
    ) -> Result<()> {
        if sources.is_empty() {
            return Ok(());
        }
        let recorded_at = self.version_commit_time(entity, version);
        let derived = LineageRef { entity, version };
        self.lineage
            .record(derived, sources.to_vec(), recorded_at)
            .map_err(crate::core::error::Error::from)
            .record_error_metric()
    }

    /// The commit (transaction) time of a written version, falling back to the
    /// current wallclock if metadata cannot be read (never fails the write).
    fn version_commit_time(
        &self,
        entity: EntityId,
        version: VersionId,
    ) -> crate::core::temporal::Timestamp {
        let historical = self.historical.read();
        let ts = match entity {
            EntityId::Node(_) => historical
                .get_node_version(version)
                .map(|v| v.commit_timestamp),
            EntityId::Edge(_) => historical
                .get_edge_version(version)
                .map(|v| v.commit_timestamp),
        };
        ts.unwrap_or_else(crate::core::temporal::time::now)
    }

    /// Resolve each closure entry's current-state status.
    ///
    /// `Absent` means the entity is gone from *current* state (deleted or
    /// retracted) — this is checked against current storage, because a
    /// retraction/deletion leaves a tombstone as the entity's latest
    /// historical version (so the historical current-version pointer stays
    /// non-`None`). Among live entities, the version pointer distinguishes
    /// `Current` from `Superseded`.
    fn resolve_view(&self, closure: LineageClosure) -> LineageView {
        let historical = self.historical.read();
        let entries = closure
            .entries
            .into_iter()
            .map(|entry| {
                let status = match entry.reference.entity {
                    EntityId::Node(node_id) => {
                        if !self.current.contains_node(node_id.as_u64()) {
                            FactStatus::Absent
                        } else {
                            match historical.get_current_node_version(node_id) {
                                Some(current) if current == entry.reference.version => {
                                    FactStatus::Current
                                }
                                _ => FactStatus::Superseded,
                            }
                        }
                    }
                    EntityId::Edge(edge_id) => {
                        if self.current.get_edge(edge_id).is_err() {
                            FactStatus::Absent
                        } else {
                            match historical.get_current_edge_version(edge_id) {
                                Some(current) if current == entry.reference.version => {
                                    FactStatus::Current
                                }
                                _ => FactStatus::Superseded,
                            }
                        }
                    }
                };
                LineageViewEntry {
                    reference: entry.reference,
                    depth: entry.depth,
                    status,
                }
            })
            .collect();
        LineageView {
            root: closure.root,
            entries,
            has_more: closure.has_more,
        }
    }
}
