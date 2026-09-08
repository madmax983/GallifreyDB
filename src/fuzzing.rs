//! Fuzz-only support hooks.
//!
//! This module is compiled only with `--cfg fuzzing` or the internal `fuzzing`
//! feature. It exposes narrow wrappers around parser internals so cargo-fuzz
//! targets can exercise critical storage paths without making those internals
//! part of the normal public API.

use crate::core::hlc::HybridTimestamp;
use crate::core::id::{EdgeId, MAX_VALID_ID, NodeId, VersionId};
use crate::core::temporal::MAX_VALID_TIMESTAMP;
use crate::storage::wal::{LSN, WalEntry};

/// WAL parser and serializer hooks used by fuzz targets.
pub mod wal {
    use super::*;
    use crate::core::error::Result;
    use crate::storage::wal::{segment_reader, serialization};

    /// Parse one WAL entry from `bytes` using the current plaintext WAL version.
    ///
    /// The version passed here MUST equal the newest plaintext WAL version that
    /// [`serialize_entry`] writes, because the round-trip fuzz target
    /// (`wal_entry_parsing`) serializes a fresh entry with the newest payload
    /// shape and then parses those bytes back with this function. Parsing at a
    /// stale version skips the trailing bytes the serializer wrote — the
    /// transaction-framing markers (#3413, v7) and the delete/retract
    /// `version_id` (#3406, v9) — leaving the buffer misaligned so the entry
    /// checksum fails (see `wal_entry_parsing` fuzz regressions below).
    ///
    /// We derive the version from `segment_reader::newest_plaintext_wal_version`
    /// (itself derived from `WAL_VERSION_MAX`) rather than hardcoding a byte, so
    /// this harness can never lag a future WAL format bump.
    ///
    /// The wrapper intentionally starts at offset zero. Fuzz targets that need to
    /// test segment headers should use the public segment-reader API instead.
    pub fn parse_current_entry(bytes: &[u8]) -> Result<(WalEntry, usize)> {
        segment_reader::parse_entry_at(bytes, 0, segment_reader::newest_plaintext_wal_version())
    }

    /// Serialize a parsed WAL entry back to bytes with a fresh checksum.
    pub fn serialize_entry(entry: &WalEntry) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        serialization::serialize_operation_into(
            entry.lsn,
            entry.timestamp,
            &entry.operation,
            &mut buffer,
        )?;
        Ok(buffer)
    }
}

fn bounded_valid_id(raw: u64) -> u64 {
    raw % MAX_VALID_ID
}

impl<'a> arbitrary::Arbitrary<'a> for LSN {
    fn arbitrary(unstructured: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(LSN(u64::arbitrary(unstructured)?))
    }
}

impl<'a> arbitrary::Arbitrary<'a> for NodeId {
    fn arbitrary(unstructured: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let raw = bounded_valid_id(u64::arbitrary(unstructured)?);
        Ok(NodeId::new(raw).expect("bounded fuzz node ID must be valid"))
    }
}

impl<'a> arbitrary::Arbitrary<'a> for EdgeId {
    fn arbitrary(unstructured: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let raw = bounded_valid_id(u64::arbitrary(unstructured)?);
        Ok(EdgeId::new(raw).expect("bounded fuzz edge ID must be valid"))
    }
}

impl<'a> arbitrary::Arbitrary<'a> for VersionId {
    fn arbitrary(unstructured: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let raw = bounded_valid_id(u64::arbitrary(unstructured)?);
        Ok(VersionId::new(raw).expect("bounded fuzz version ID must be valid"))
    }
}

impl<'a> arbitrary::Arbitrary<'a> for HybridTimestamp {
    fn arbitrary(unstructured: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let max = MAX_VALID_TIMESTAMP as u64 + 1;
        let wallclock = (u64::arbitrary(unstructured)? % max) as i64;
        let logical = u32::arbitrary(unstructured)?;
        Ok(HybridTimestamp::new(wallclock, logical).expect("bounded fuzz timestamp must be valid"))
    }
}

#[cfg(test)]
mod tests {
    use super::wal::{parse_current_entry, serialize_entry};
    use crate::core::NodeId;
    use crate::core::id::{EdgeId, VersionId};
    use crate::core::interning::InternedString;
    use crate::core::property::PropertyMap;
    use crate::core::provenance::Provenance;
    use crate::core::temporal::time;
    use crate::storage::wal::entry::{LSN, WalEntry, WalOperation};

    /// Round-trip every WAL operation the fuzz harness exercises through
    /// `serialize_entry` -> `parse_current_entry` and assert byte-count and
    /// payload fidelity — the exact invariant `wal_entry_parsing` checks.
    ///
    /// Regression for Issue #3467: `parse_current_entry` pinned the parse
    /// version at `WAL_VERSION_PROVENANCE_PRINCIPAL` (v5) while the serializer
    /// writes the newest plaintext shape (v9), which added the #3413
    /// transaction-framing markers (v7) and the #3406 delete/retract
    /// `version_id` (v9). Parsing at v5 skips those trailing bytes, misaligns
    /// the buffer, and fails the entry checksum. `DeleteNode`/`DeleteEdge`/
    /// `RetractNode`/`RetractEdge` (op_type 6/7/10/11) are the ops that carry
    /// the version-gated `version_id`, so they fail before the fix and pass
    /// after; the Create/Update ops guard against re-introducing the older
    /// #3224/#3350 provenance skew.
    #[test]
    fn every_op_roundtrips_at_serialize_version() {
        // Capture a single timestamp so every op's temporal field is
        // deterministic within this test run (avoids per-op `time::now()`).
        let now = time::now();
        let ops = vec![
            WalOperation::CreateNode {
                node_id: NodeId::new(1).unwrap(),
                label: InternedString::from_raw(0),
                properties: PropertyMap::new(),
                valid_from: now,
                provenance: None,
            },
            WalOperation::CreateEdge {
                edge_id: EdgeId::new(2).unwrap(),
                source: NodeId::new(1).unwrap(),
                target: NodeId::new(3).unwrap(),
                label: InternedString::from_raw(0),
                properties: PropertyMap::new(),
                valid_from: now,
                provenance: None,
            },
            WalOperation::UpdateNode {
                node_id: NodeId::new(1).unwrap(),
                version_id: VersionId::new(4).unwrap(),
                label: InternedString::from_raw(0),
                properties: PropertyMap::new(),
                valid_from: now,
                provenance: None,
            },
            WalOperation::UpdateEdge {
                edge_id: EdgeId::new(2).unwrap(),
                version_id: VersionId::new(5).unwrap(),
                label: InternedString::from_raw(0),
                properties: PropertyMap::new(),
                valid_from: now,
                provenance: None,
            },
            // The version-gated ops (#3406) — the crux of #3467.
            WalOperation::DeleteNode {
                node_id: NodeId::new(1).unwrap(),
                valid_from: now,
                version_id: Some(VersionId::new(6).unwrap()),
                provenance: None,
            },
            WalOperation::DeleteEdge {
                edge_id: EdgeId::new(2).unwrap(),
                valid_from: now,
                version_id: Some(VersionId::new(7).unwrap()),
                provenance: None,
            },
            WalOperation::RetractNode {
                node_id: NodeId::new(1).unwrap(),
                valid_to: now,
                version_id: Some(VersionId::new(8).unwrap()),
                provenance: None,
            },
            WalOperation::RetractEdge {
                edge_id: EdgeId::new(2).unwrap(),
                valid_to: now,
                version_id: Some(VersionId::new(9).unwrap()),
                provenance: None,
            },
            // Transaction-framing markers (#3413).
            WalOperation::BeginTx { tx_id: 42 },
            WalOperation::CommitTx {
                tx_id: 42,
                entry_count: 3,
                commit_timestamp: now,
            },
        ];

        for (i, op) in ops.into_iter().enumerate() {
            let entry = WalEntry::new(LSN((i as u64) + 1), op);
            let canonical = serialize_entry(&entry).expect("entry must serialize");
            let (roundtrip, consumed) = parse_current_entry(&canonical).unwrap_or_else(|e| {
                panic!(
                    "op #{i} ({:?}) must re-parse cleanly, got: {e}",
                    entry.operation
                )
            });
            assert_eq!(
                consumed,
                canonical.len(),
                "op #{i} ({:?}) must consume all serialized bytes",
                entry.operation
            );
            assert_eq!(
                roundtrip.operation, entry.operation,
                "op #{i} payload must round-trip identically"
            );
            assert_eq!(roundtrip.lsn, entry.lsn);
            assert_eq!(roundtrip.timestamp, entry.timestamp);
        }
    }

    /// Byte-literal regression for the CI fuzz crash on Issue #3467: an
    /// `OP_RETRACT_NODE` (op_type 10) entry at LSN 16276538888567251274. The
    /// serializer writes the v9 retraction `version_id`; parsing the canonical
    /// bytes back at the stale v5 pin misaligned the buffer and failed the
    /// checksum. Any input that parses must re-serialize and re-parse
    /// losslessly.
    #[test]
    fn fuzz_crash_retract_node_v9_roundtrips() {
        let entry = WalEntry::new(
            LSN(16_276_538_888_567_251_274),
            WalOperation::RetractNode {
                node_id: NodeId::new(1).unwrap(),
                valid_to: time::now(),
                version_id: Some(VersionId::new(1).unwrap()),
                provenance: None,
            },
        );
        let canonical = serialize_entry(&entry).expect("retract entry must serialize");
        let (roundtrip, consumed) =
            parse_current_entry(&canonical).expect("serialized retract entry must parse");
        assert_eq!(consumed, canonical.len());
        assert_eq!(roundtrip.operation, entry.operation);
        assert_eq!(roundtrip.lsn, entry.lsn);

        let canonical2 = serialize_entry(&roundtrip).expect("roundtrip must serialize");
        assert_eq!(
            canonical2, canonical,
            "WAL serialization must be idempotent"
        );
    }

    /// Regression test for a bug where `parse_current_entry` hardcoded the
    /// pre-provenance `WAL_VERSION` (1) even though `serialize_entry` always
    /// writes the provenance presence byte for Create/Update ops (Issue
    /// #3224). Parsing at the stale version left a trailing byte unconsumed,
    /// breaking round-trip byte-count fidelity for every Create/Update op --
    /// exactly the invariant `fuzz/fuzz_targets/wal_entry_parsing.rs` checks.
    ///
    /// The same bug recurred with the Issue #3350 principal field (v5): the
    /// shim kept parsing at `WAL_VERSION_PROVENANCE` (3) while the serializer
    /// moved to the principal-carrying shape, so the unconsumed trailing
    /// principal byte(s) made our own canonical bytes fail their checksum.
    /// This entry carries `Some(provenance)` and therefore exercises exactly
    /// that skew: it fails with a checksum mismatch whenever
    /// `parse_current_entry`'s version lags the serializer.
    #[test]
    fn create_node_roundtrip_consumes_all_bytes() {
        let entry = WalEntry::new(
            LSN(1),
            WalOperation::CreateNode {
                node_id: NodeId::new(1).unwrap(),
                label: InternedString::from_raw(0),
                properties: PropertyMap::new(),
                valid_from: time::now(),
                provenance: Some(
                    Provenance::builder()
                        .source("test")
                        .confidence(0.9)
                        .build()
                        .unwrap(),
                ),
            },
        );

        let canonical = serialize_entry(&entry).expect("entry must serialize");
        let (roundtrip, roundtrip_consumed) =
            parse_current_entry(&canonical).expect("serialized entry must parse");

        assert_eq!(roundtrip_consumed, canonical.len());
        assert_eq!(roundtrip.lsn, entry.lsn);
    }

    /// A provenance bundle carrying the Issue #3350 `principal` field must
    /// round-trip byte-identically through the fuzz shim, exactly as the
    /// `wal_entry_parsing` fuzz target asserts.
    #[test]
    fn principal_provenance_roundtrip_is_idempotent() {
        let entry = WalEntry::new(
            LSN(7),
            WalOperation::CreateNode {
                node_id: NodeId::new(7).unwrap(),
                label: InternedString::from_raw(0),
                properties: PropertyMap::new(),
                valid_from: time::now(),
                provenance: Some(
                    Provenance::builder()
                        .source("http")
                        .principal("svc-writer")
                        .build()
                        .unwrap(),
                ),
            },
        );

        let canonical = serialize_entry(&entry).expect("entry must serialize");
        let (roundtrip, consumed) =
            parse_current_entry(&canonical).expect("serialized entry must parse");
        assert_eq!(consumed, canonical.len());

        let canonical2 = serialize_entry(&roundtrip).expect("roundtrip entry must serialize");
        assert_eq!(
            canonical2, canonical,
            "WAL serialization must be idempotent"
        );
    }

    /// Byte-literal regression for the CI fuzz crash
    /// `crash-dece516e7f71f4e5ef5e31620a1410868b267eb1` (PR #3421 / Issue
    /// #3350): these bytes parsed successfully at the stale
    /// `WAL_VERSION_PROVENANCE` with a `Some(provenance)` payload, but the
    /// re-serialized canonical form (which always carries the v5 principal
    /// slot) then failed its own checksum when parsed back at the stale
    /// version. Mirrors the `wal_entry_parsing` fuzz harness exactly: any
    /// input that parses must re-serialize and re-parse losslessly.
    #[test]
    fn fuzz_crash_dece516e_roundtrip() {
        const CRASH_INPUT: [u8; 64] = [
            160, 255, 255, 160, 160, 160, 167, 167, 167, 167, 170, 167, 170, 167, 167, 160, 160,
            255, 255, 160, 212, 63, 250, 194, 1, 1, 1, 1, 1, 228, 1, 160, 134, 1, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 160, 0, 0, 0, 0, 67, 111, 111, 167, 0, 167,
        ];

        if let Ok((entry, consumed)) = parse_current_entry(&CRASH_INPUT) {
            assert!(consumed <= CRASH_INPUT.len());
            let canonical = serialize_entry(&entry).expect("parsed WAL entry must serialize");
            let (roundtrip, roundtrip_consumed) =
                parse_current_entry(&canonical).expect("serialized WAL entry must parse");
            assert_eq!(roundtrip_consumed, canonical.len());
            assert_eq!(roundtrip.lsn, entry.lsn);
            assert_eq!(roundtrip.timestamp, entry.timestamp);
            let canonical2 =
                serialize_entry(&roundtrip).expect("double-serialized WAL entry must serialize");
            assert_eq!(
                canonical2, canonical,
                "WAL serialization must be idempotent"
            );
        }
    }
}
