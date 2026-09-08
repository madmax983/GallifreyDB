//! Building a signed audit export from live database history (Issue #3358).
//!
//! This is a read-only path: it consumes the existing history reads
//! (`get_node_history` / `get_edge_history`) plus current adjacency for the
//! neighborhood scope, and never mutates the database — so it takes no
//! write-path locks.

use std::collections::BTreeSet;

use crate::audit::canonical;
use crate::audit::error::{AuditError, AuditResult};
use crate::audit::model::{
    AUDIT_FORMAT_VERSION, AuditExport, ChainAnchor, ChainProof, CoordinateRecord, EntityIdRecord,
    ExportMetadata, ExportedEntity, ExportedProperty, ExportedProvenance, ExportedValue,
    ExportedVersion, HASH_ALGORITHM, NotIncludedRecord, SIGNATURE_ALGORITHM, SIGNED_DESCRIPTION,
    ScopeRecord, SignatureBlock, f64_to_token,
};
use crate::audit::signing::AuditSigningKey;
use crate::core::history::{EntityHistory, VersionInfo};
use crate::core::interning::GLOBAL_INTERNER;
use crate::core::property::PropertyValue;
use crate::core::temporal::{TIMESTAMP_MAX, Timestamp, time};
use crate::core::{EdgeId, NodeId};
use crate::db::AletheiaDB;

/// A reference to a single node or edge for export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityRef {
    /// A node, by id.
    Node(NodeId),
    /// An edge, by id.
    Edge(EdgeId),
}

/// What to export (AC5): a single entity, an explicit set, or a neighborhood.
#[derive(Debug, Clone)]
pub enum AuditScope {
    /// A single node or edge.
    SingleEntity(EntityRef),
    /// An explicit list of entities.
    EntitySet(Vec<EntityRef>),
    /// A node plus its N-hop neighborhood at a bi-temporal coordinate.
    Neighborhood {
        /// The entity to start from.
        start: NodeId,
        /// Hop radius.
        hops: u32,
        /// Valid-time coordinate (defaults to now when omitted).
        valid_time: Option<Timestamp>,
        /// Transaction-time coordinate (defaults to now when omitted).
        transaction_time: Option<Timestamp>,
    },
}

impl AuditScope {
    /// Convenience: export a single node.
    #[must_use]
    pub fn node(id: NodeId) -> Self {
        AuditScope::SingleEntity(EntityRef::Node(id))
    }

    /// Convenience: export a single edge.
    #[must_use]
    pub fn edge(id: EdgeId) -> Self {
        AuditScope::SingleEntity(EntityRef::Edge(id))
    }
}

/// Options controlling an export: database identity + redaction (AC6/AC8).
#[derive(Debug, Clone)]
pub struct ExportOptions {
    database_id: String,
    redact_keys: Vec<String>,
}

impl ExportOptions {
    /// New options with the given operator-supplied database identity.
    #[must_use]
    pub fn new(database_id: impl Into<String>) -> Self {
        Self {
            database_id: database_id.into(),
            redact_keys: Vec::new(),
        }
    }

    /// Redact the given property keys at export (their values are omitted, but
    /// the redaction is recorded and verifiable — never silent).
    #[must_use]
    pub fn redact<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.redact_keys
            .extend(keys.into_iter().map(std::convert::Into::into));
        self
    }
}

impl AletheiaDB {
    /// Produce a signed, self-contained audit export of an entity's (or a set
    /// or neighborhood's) complete bi-temporal history (Issue #3358).
    ///
    /// The result verifies OFFLINE with only the signer's public key. See the
    /// [`crate::audit`] module docs and `docs/guides/audit-export.md` for the
    /// artifact format and verification contract.
    ///
    /// # Errors
    /// Returns [`AuditError::NoHistory`] if a single requested entity has no
    /// retrievable history, or [`AuditError::DatabaseRead`] on a read failure.
    pub fn audit_export(
        &self,
        scope: AuditScope,
        signing_key: &AuditSigningKey,
        options: &ExportOptions,
    ) -> AuditResult<AuditExport> {
        let redact: BTreeSet<String> = options.redact_keys.iter().cloned().collect();

        let (kind_str, requested, entities, not_included, coordinate, hops) =
            self.resolve_scope(&scope, &redact)?;

        let included: Vec<EntityIdRecord> = entities
            .iter()
            .map(|e| EntityIdRecord {
                kind: e.kind.clone(),
                id: e.id,
            })
            .collect();

        let version_count: usize = entities.iter().map(|e| e.versions.len()).sum();
        let mut redacted_keys: Vec<String> = redact.into_iter().collect();
        redacted_keys.sort();

        let metadata = ExportMetadata {
            database_id: options.database_id.clone(),
            exported_at: format_timestamp(time::now()),
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
            scope_description: describe_scope(&scope),
            scope: ScopeRecord {
                kind: kind_str,
                requested,
                included,
                not_included,
                coordinate,
                hops,
            },
            chain_anchor: ChainAnchor {
                source_lsn: self.stats().wal.current_lsn,
                description: "WAL LSN at export time (chain-head position, #3351 anchor)"
                    .to_string(),
            },
            redacted_keys,
            entity_count: entities.len(),
            version_count,
            hash_algorithm: HASH_ALGORITHM.to_string(),
            signature_algorithm: SIGNATURE_ALGORITHM.to_string(),
        };

        // Integrity: per-version leaves + folded chain root, then sign the root.
        let (leaves, root) = canonical::compute_chain(&metadata, &entities);
        let signature = signing_key.sign_root(&root);
        let public_key = signing_key.public_key();

        Ok(AuditExport {
            format_version: AUDIT_FORMAT_VERSION,
            metadata,
            entities,
            chain: ChainProof {
                algorithm: HASH_ALGORITHM.to_string(),
                leaves: leaves.iter().map(canonical::to_hex).collect(),
                root: canonical::to_hex(&root),
            },
            signature: SignatureBlock {
                algorithm: SIGNATURE_ALGORITHM.to_string(),
                public_key: public_key.to_hex(),
                signature: crate::core::hex::encode(&signature),
                signed: SIGNED_DESCRIPTION.to_string(),
            },
        })
    }

    /// Resolve a scope into the concrete entities to export plus scope metadata.
    #[allow(clippy::type_complexity)]
    fn resolve_scope(
        &self,
        scope: &AuditScope,
        redact: &BTreeSet<String>,
    ) -> AuditResult<(
        String,
        Vec<EntityIdRecord>,
        Vec<ExportedEntity>,
        Vec<NotIncludedRecord>,
        Option<CoordinateRecord>,
        Option<u32>,
    )> {
        match scope {
            AuditScope::SingleEntity(entity_ref) => {
                let kind = match entity_ref {
                    EntityRef::Node(_) => "single_node",
                    EntityRef::Edge(_) => "single_edge",
                };
                let requested = vec![entity_id_record(*entity_ref)];
                let exported = self
                    .export_one(*entity_ref, redact)?
                    .ok_or_else(|| AuditError::NoHistory(describe_entity(*entity_ref)))?;
                Ok((
                    kind.to_string(),
                    requested,
                    vec![exported],
                    Vec::new(),
                    None,
                    None,
                ))
            }
            AuditScope::EntitySet(refs) => {
                let requested: Vec<EntityIdRecord> =
                    refs.iter().map(|r| entity_id_record(*r)).collect();
                let mut entities = Vec::new();
                let mut not_included = Vec::new();
                for r in refs {
                    match self.export_one(*r, redact)? {
                        Some(e) => entities.push(e),
                        None => {
                            let rec = entity_id_record(*r);
                            not_included.push(NotIncludedRecord {
                                kind: rec.kind,
                                id: rec.id,
                                reason: "no_history".to_string(),
                            });
                        }
                    }
                }
                Ok((
                    "entity_set".to_string(),
                    requested,
                    entities,
                    not_included,
                    None,
                    None,
                ))
            }
            AuditScope::Neighborhood {
                start,
                hops,
                valid_time,
                transaction_time,
            } => {
                let vt = valid_time.unwrap_or_else(time::now);
                let tt = transaction_time.unwrap_or_else(time::now);
                let (node_ids, edge_ids) = self.collect_neighborhood(*start, *hops, vt, tt);

                let mut entities = Vec::new();
                for n in node_ids {
                    if let Some(e) = self.export_one(EntityRef::Node(n), redact)? {
                        entities.push(e);
                    }
                }
                for ed in edge_ids {
                    if let Some(e) = self.export_one(EntityRef::Edge(ed), redact)? {
                        entities.push(e);
                    }
                }
                let coordinate = Some(CoordinateRecord {
                    valid_time: Some(format_timestamp(vt)),
                    transaction_time: Some(format_timestamp(tt)),
                });
                Ok((
                    "neighborhood".to_string(),
                    vec![entity_id_record(EntityRef::Node(*start))],
                    entities,
                    Vec::new(),
                    coordinate,
                    Some(*hops),
                ))
            }
        }
    }

    /// Export a single entity's full history, or `None` if it has no history.
    fn export_one(
        &self,
        entity_ref: EntityRef,
        redact: &BTreeSet<String>,
    ) -> AuditResult<Option<ExportedEntity>> {
        match entity_ref {
            EntityRef::Node(id) => {
                let history = match self.get_node_history(id) {
                    Ok(h) => h,
                    Err(_) => return Ok(None),
                };
                if history.version_count() == 0 {
                    return Ok(None);
                }
                let label = history
                    .current_version()
                    .or_else(|| history.first_version())
                    .map(|v| v.label.clone());
                Ok(Some(ExportedEntity {
                    kind: "node".to_string(),
                    id: id.as_u64(),
                    label,
                    source: None,
                    target: None,
                    versions: convert_versions(&history, redact),
                }))
            }
            EntityRef::Edge(id) => {
                let history = match self.get_edge_history(id) {
                    Ok(h) => h,
                    Err(_) => return Ok(None),
                };
                if history.version_count() == 0 {
                    return Ok(None);
                }
                let label = history
                    .current_version()
                    .or_else(|| history.first_version())
                    .map(|v| v.label.clone());
                // Endpoints are informational; recover them if the edge still
                // resolves (deleted edges leave them unknown → None).
                let (source, target) = match self.get_edge(id) {
                    Ok(edge) => (Some(edge.source.as_u64()), Some(edge.target.as_u64())),
                    Err(_) => (None, None),
                };
                Ok(Some(ExportedEntity {
                    kind: "edge".to_string(),
                    id: id.as_u64(),
                    label,
                    source,
                    target,
                    versions: convert_versions(&history, redact),
                }))
            }
        }
    }

    /// BFS the current graph from `start` up to `hops`, keeping only nodes/edges
    /// valid at the given bi-temporal coordinate. Returns `(nodes, edges)`.
    ///
    /// ⚡ Bolt Optimization: Pre-allocates and reuses `next` vector via `swap` and `clear` to eliminate O(hops) intermediate heap allocations during BFS traversal.
    fn collect_neighborhood(
        &self,
        start: NodeId,
        hops: u32,
        vt: Timestamp,
        tt: Timestamp,
    ) -> (Vec<NodeId>, Vec<EdgeId>) {
        let mut nodes: BTreeSet<u64> = BTreeSet::new();
        let mut edges: BTreeSet<u64> = BTreeSet::new();
        let mut frontier: Vec<NodeId> = Vec::new();
        let mut next: Vec<NodeId> = Vec::new();

        if self.get_node_at_time(start, vt, tt).is_ok() {
            nodes.insert(start.as_u64());
            frontier.push(start);
        }

        for _ in 0..hops {
            next.clear();
            for node in &frontier {
                for edge_id in self
                    .get_outgoing_edges(*node)
                    .into_iter()
                    .chain(self.get_incoming_edges(*node))
                {
                    if self.get_edge_at_time(edge_id, vt, tt).is_err() {
                        continue;
                    }
                    edges.insert(edge_id.as_u64());
                    for endpoint in [
                        self.get_edge_source(edge_id).ok(),
                        self.get_edge_target(edge_id).ok(),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        if self.get_node_at_time(endpoint, vt, tt).is_ok()
                            && nodes.insert(endpoint.as_u64())
                        {
                            next.push(endpoint);
                        }
                    }
                }
            }
            std::mem::swap(&mut frontier, &mut next);
        }

        let node_ids = nodes
            .into_iter()
            .filter_map(|id| NodeId::new(id).ok())
            .collect();
        let edge_ids = edges
            .into_iter()
            .filter_map(|id| EdgeId::new(id).ok())
            .collect();
        (node_ids, edge_ids)
    }
}

/// Convert an entity's history into exported versions (oldest first).
fn convert_versions(history: &EntityHistory, redact: &BTreeSet<String>) -> Vec<ExportedVersion> {
    let now = time::now();
    history
        .versions
        .iter()
        .map(|v| convert_version(v, redact, now))
        .collect()
}

fn convert_version(v: &VersionInfo, redact: &BTreeSet<String>, now: Timestamp) -> ExportedVersion {
    let valid = v.temporal.valid_time();
    let tx = v.temporal.transaction_time();

    let (valid_to_micros, valid_to_logical) = closed_bound(valid.end());
    let (tx_to_micros, tx_to_logical) = closed_bound(tx.end());
    let is_current = tx.is_current() && valid.contains(now);

    let mut properties: Vec<ExportedProperty> = v
        .properties
        .iter()
        .map(|(key, value)| {
            let key_str = GLOBAL_INTERNER
                .resolve_with(*key, std::string::ToString::to_string)
                .unwrap_or_else(|| "<unknown-key>".to_string());
            if redact.contains(&key_str) {
                ExportedProperty {
                    key: key_str,
                    value: None,
                    redacted: true,
                }
            } else {
                ExportedProperty {
                    key: key_str,
                    value: Some(property_value_to_exported(value)),
                    redacted: false,
                }
            }
        })
        .collect();
    properties.sort_by(|a, b| a.key.cmp(&b.key));

    ExportedVersion {
        version_number: v.version_number,
        version_id: v.version_id.as_u64(),
        label: v.label.clone(),
        valid_from_micros: valid.start().wallclock(),
        valid_from_logical: valid.start().logical(),
        valid_to_micros,
        valid_to_logical,
        transaction_from_micros: tx.start().wallclock(),
        transaction_from_logical: tx.start().logical(),
        transaction_to_micros: tx_to_micros,
        transaction_to_logical: tx_to_logical,
        is_current,
        provenance: v.provenance.as_ref().map(convert_provenance),
        properties,
    }
}

/// `None` for an open-ended (`TIMESTAMP_MAX`) bound, else its `(micros, logical)`.
fn closed_bound(end: Timestamp) -> (Option<i64>, Option<u32>) {
    if end == TIMESTAMP_MAX {
        (None, None)
    } else {
        (Some(end.wallclock()), Some(end.logical()))
    }
}

fn convert_provenance(p: &crate::core::provenance::Provenance) -> ExportedProvenance {
    ExportedProvenance {
        source: p.source().map(std::string::ToString::to_string),
        confidence: p.confidence(),
        note: p.note().map(std::string::ToString::to_string),
        correlation_id: p.correlation_id().map(std::string::ToString::to_string),
        principal: p.principal().map(std::string::ToString::to_string),
    }
}

fn property_value_to_exported(value: &PropertyValue) -> ExportedValue {
    match value {
        PropertyValue::Null => ExportedValue::Null,
        PropertyValue::Bool(b) => ExportedValue::Bool { value: *b },
        PropertyValue::Int(i) => ExportedValue::Int { value: *i },
        PropertyValue::Float(f) => ExportedValue::Float { value: *f },
        PropertyValue::String(s) => ExportedValue::String {
            value: s.to_string(),
        },
        PropertyValue::Bytes(b) => ExportedValue::Bytes {
            hex: crate::core::hex::encode(b),
        },
        PropertyValue::Array(values) => ExportedValue::Array {
            values: values.iter().map(property_value_to_exported).collect(),
        },
        PropertyValue::Vector(v) => ExportedValue::Vector {
            values: v.iter().map(|f| f64_to_token(f64::from(*f))).collect(),
        },
        PropertyValue::SparseVector(sv) => ExportedValue::SparseVector {
            dimension: sv.dimension(),
            indices: sv.indices().to_vec(),
            values: sv
                .values()
                .iter()
                .map(|f| f64_to_token(f64::from(*f)))
                .collect(),
        },
    }
}

fn entity_id_record(r: EntityRef) -> EntityIdRecord {
    match r {
        EntityRef::Node(id) => EntityIdRecord {
            kind: "node".to_string(),
            id: id.as_u64(),
        },
        EntityRef::Edge(id) => EntityIdRecord {
            kind: "edge".to_string(),
            id: id.as_u64(),
        },
    }
}

fn describe_entity(r: EntityRef) -> String {
    match r {
        EntityRef::Node(id) => format!("node {}", id.as_u64()),
        EntityRef::Edge(id) => format!("edge {}", id.as_u64()),
    }
}

fn describe_scope(scope: &AuditScope) -> String {
    match scope {
        AuditScope::SingleEntity(EntityRef::Node(id)) => {
            format!("single_node: node {}", id.as_u64())
        }
        AuditScope::SingleEntity(EntityRef::Edge(id)) => {
            format!("single_edge: edge {}", id.as_u64())
        }
        AuditScope::EntitySet(refs) => format!("entity_set: {} entities requested", refs.len()),
        AuditScope::Neighborhood { start, hops, .. } => {
            format!("neighborhood: {} hops around node {}", hops, start.as_u64())
        }
    }
}

/// Format a timestamp as RFC 3339 (UTC), falling back to raw micros if it is
/// outside chrono's representable range.
fn format_timestamp(ts: Timestamp) -> String {
    crate::audit::model::rfc3339_micros(ts.wallclock())
}
