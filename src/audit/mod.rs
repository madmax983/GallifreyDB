//! Signed audit export of entity history for compliance (Issue #3358).
//!
//! Turns AletheiaDB's storage-layer trust into portable, offline-verifiable
//! evidence: a single-file artifact containing an entity's complete bi-temporal
//! history plus provenance, signed so a third party with only the signer's
//! **public key** can verify — with no database, no network, and no trust in the
//! operator — that (a) no content was altered after export and (b) the content
//! is complete (nothing was added, removed, or reordered).
//!
//! # Why Ed25519 (asymmetric) rather than an HMAC
//!
//! Verification must be possible for a party who has *only the signer's public
//! key*. A symmetric MAC would require the verifier to hold the same secret the
//! signer holds — so the operator could forge artifacts and a third party could
//! not independently trust them. Ed25519 detached signatures over a SHA-256 hash
//! chain give offline, public-key-only verification with reproducible signatures.
//!
//! # Artifact + verification contract
//!
//! The artifact is JSON ([`AuditExport`]). Its signature is computed over a
//! deterministic canonical *binary* encoding of the typed content (not the JSON
//! text), so reformatting is safe but any content change is caught. See the
//! `canonical` module and `docs/guides/audit-export.md` for the fully
//! documented, independently-implementable format.
//!
//! # Quick start
//!
//! ```no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::audit::{AuditScope, AuditSigningKey, ExportOptions, verify_json_bytes};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! let node = db.create_node("Person", aletheiadb::PropertyMap::new())?;
//!
//! let key = AuditSigningKey::generate();
//! let export = db.audit_export(AuditScope::node(node), &key, &ExportOptions::new("prod-db"))?;
//! let bytes = export.to_json_bytes()?;
//!
//! // Offline: only the bytes + the public key.
//! let report = verify_json_bytes(&bytes, Some(&key.public_key()))?;
//! assert!(report.passed);
//! # Ok(())
//! # }
//! ```
//!
//! # Known limitation (#3427)
//!
//! Delete/retract operations do not yet stamp an authenticated principal into
//! provenance. The export surfaces provenance faithfully — including its absence
//! on delete tombstones — and never fabricates attribution. This is documented
//! honestly rather than papered over.

mod build;
mod canonical;
mod error;
mod model;
mod signing;
mod verify;

#[cfg(test)]
mod tests;

pub use build::{AuditScope, EntityRef, ExportOptions};
pub use error::{AuditError, AuditResult};
pub use model::{
    AUDIT_FORMAT_VERSION, AuditExport, ChainAnchor, ChainProof, CoordinateRecord, EntityIdRecord,
    ExportMetadata, ExportedEntity, ExportedProperty, ExportedProvenance, ExportedValue,
    ExportedVersion, NotIncludedRecord, ScopeRecord, SignatureBlock,
};
pub use signing::{AuditPublicKey, AuditSigningKey, SIGNING_KEY_ENV};
pub use verify::{AuditVerification, verify_json_bytes};
