//! Tests for historical storage mechanisms.
//!
//! This module verifies the anchor+delta compression, bitemporal versioning,
//! and time-travel query resolution within the `HistoricalStorage` component.

use super::*;
use crate::core::interning::GLOBAL_INTERNER;
use crate::core::observer::{StorageEvent, StorageObserver};
use crate::core::property::PropertyMapBuilder;
use crate::core::temporal::TIMESTAMP_MAX;

#[test]
fn test_create_first_version() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();
    let temporal = BiTemporalInterval::current(1000.into());
    let props = PropertyMapBuilder::new().insert("name", "Alice").build();

    storage
        .add_node_version(
            node_id,
            version_id,
            temporal.valid_time().start(),
            temporal.transaction_time().start(),
            label,
            props,
            false, // not a tombstone
        )
        .unwrap();

    // First version should be an anchor
    let version = storage.get_node_version(version_id).unwrap();
    assert!(version.is_anchor());
    assert_eq!(version.node_id, node_id);
    assert_eq!(version.prev_version, None);
}

#[test]
fn test_version_chain() {
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    });

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Create 5 versions
    let mut version_ids = Vec::new();
    for i in 0..5 {
        let version_id = VersionId::new(100 + i).unwrap();
        let temporal = BiTemporalInterval::current((1000 + (i as i64) * 100).into());
        let props = PropertyMapBuilder::new()
            .insert("name", "Alice")
            .insert("age", i as i64)
            .build();

        storage
            .add_node_version(
                node_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();

        version_ids.push(version_id);
    }

    // Check version types
    // v0: anchor (first)
    // v1: delta
    // v2: delta
    // v3: anchor (interval = 3)
    // v4: delta

    assert!(
        storage
            .get_node_version(version_ids[0])
            .unwrap()
            .is_anchor()
    );
    assert!(storage.get_node_version(version_ids[1]).unwrap().is_delta());
    assert!(storage.get_node_version(version_ids[2]).unwrap().is_delta());
    assert!(
        storage
            .get_node_version(version_ids[3])
            .unwrap()
            .is_anchor()
    );
    assert!(storage.get_node_version(version_ids[4]).unwrap().is_delta());
}

#[test]
fn test_property_reconstruction() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Version 1: name=Alice, age=30
    let v1 = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("name", "Alice")
                .insert("age", 30i64)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 2: name=Alice, age=31 (delta)
    let v2 = VersionId::new(2).unwrap();
    storage
        .add_node_version(
            node_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("name", "Alice")
                .insert("age", 31i64)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Reconstruct v2 properties
    let props = storage.reconstruct_node_properties(v2).unwrap();
    assert_eq!(props.get("name").and_then(|v| v.as_str()), Some("Alice"));
    assert_eq!(props.get("age").and_then(|v| v.as_int()), Some(31.into()));
}

#[test]
fn test_find_version_at_time() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Create versions at different times
    let v1 = VersionId::new(1).unwrap();
    let v2 = VersionId::new(2).unwrap();
    let v3 = VersionId::new(3).unwrap();

    storage
        .add_node_version(
            node_id,
            v1,
            0.into(),
            0.into(),
            label,
            PropertyMapBuilder::new().insert("age", 30i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    storage
        .add_node_version(
            node_id,
            v2,
            1000.into(),
            0.into(),
            label,
            PropertyMapBuilder::new().insert("age", 31i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    storage
        .add_node_version(
            node_id,
            v3,
            2000.into(),
            0.into(),
            label,
            PropertyMapBuilder::new().insert("age", 32i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Query at different times
    assert_eq!(
        storage.find_node_version_at_time(node_id, 500.into(), 100.into()),
        Some(v1)
    );
    assert_eq!(
        storage.find_node_version_at_time(node_id, 1500.into(), 100.into()),
        Some(v2)
    );
    assert_eq!(
        storage.find_node_version_at_time(node_id, 2500.into(), 100.into()),
        Some(v3)
    );
}

#[test]
fn test_retention_policy_node_limit() {
    // Create storage with small retention limit
    let mut storage = HistoricalStorage::with_config_and_retention(
        AnchorConfig::default(),
        RetentionPolicy::new(3, i64::MAX), // Max 3 versions per entity
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Add 3 versions - should succeed
    for i in 0..3 {
        storage
            .add_node_version(
                node_id,
                VersionId::new(i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new().build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Try to add 4th version - should fail
    let result = storage.add_node_version(
        node_id,
        VersionId::new(3).unwrap(),
        1300.into(),
        1300.into(),
        label,
        PropertyMapBuilder::new().build(),
        false, // not a tombstone
    );

    assert!(result.is_err());
    match result.unwrap_err() {
        crate::core::error::Error::Storage(StorageError::CapacityExceeded {
            resource,
            current,
            limit,
        }) => {
            assert!(resource.contains("node"));
            assert_eq!(current, 3);
            assert_eq!(limit, 3);
        }
        _ => panic!("Expected CapacityExceeded error"),
    }
}

#[test]
fn test_retention_policy_edge_limit() {
    // Create storage with small retention limit
    let mut storage = HistoricalStorage::with_config_and_retention(
        AnchorConfig::default(),
        RetentionPolicy::new(2, i64::MAX), // Max 2 versions per entity
    );

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // Add 2 versions - should succeed
    for i in 0..2 {
        storage
            .add_edge_version(
                edge_id,
                VersionId::new(i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                source,
                target,
                PropertyMapBuilder::new().build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Try to add 3rd version - should fail
    let result = storage.add_edge_version(
        edge_id,
        VersionId::new(2).unwrap(),
        1200.into(),
        1200.into(),
        label,
        source,
        target,
        PropertyMapBuilder::new().build(),
        false, // not a tombstone
    );

    assert!(result.is_err());
    match result.unwrap_err() {
        crate::core::error::Error::Storage(StorageError::CapacityExceeded {
            resource,
            current,
            limit,
        }) => {
            assert!(resource.contains("edge"));
            assert_eq!(current, 2);
            assert_eq!(limit, 2);
        }
        _ => panic!("Expected CapacityExceeded error"),
    }
}

#[test]
fn test_stats() {
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 2,
        max_delta_chain: 10,
    });

    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Add 3 node versions (anchor, delta, anchor)
    for i in 0..3 {
        storage
            .add_node_version(
                NodeId::new(1).unwrap(),
                VersionId::new(i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new().build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    let stats = storage.stats();
    assert_eq!(stats.total_node_versions, 3);
    assert_eq!(stats.node_anchor_count, 2);
    assert_eq!(stats.node_delta_count, 1);
    assert_eq!(stats.unique_nodes, 1);

    // Compression ratio should be 2/3 ≈ 0.67
    assert!((stats.compression_ratio() - 0.6666).abs() < 0.01);
}

// ============================================================
// Average Delta Chain Length (Issue #366)
// ============================================================

#[test]
fn test_calculate_avg_delta_chain_hand_computed() {
    // Issue #366: hand-computed expected average over known constructed chains.
    // With anchor_interval=10, the first version of each entity is an anchor and
    // the following versions (up to the interval) are deltas.
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 10,
        max_delta_chain: 20,
    });
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Node 1: 4 versions -> 1 anchor + 3 deltas
    for i in 0..4u64 {
        storage
            .add_node_version(
                NodeId::new(1).unwrap(),
                VersionId::new(100 + i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new().insert("v", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Node 2: 1 version -> 1 anchor + 0 deltas
    storage
        .add_node_version(
            NodeId::new(2).unwrap(),
            VersionId::new(200).unwrap(),
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    // Expected: (3 deltas + 0 deltas) / 2 anchors = 1.5
    assert!((storage.calculate_avg_delta_chain() - 1.5).abs() < f64::EPSILON);
}

#[test]
fn test_calculate_avg_delta_chain_empty_storage_fallback() {
    // Issue #366: empty historical storage falls back to the default estimate 5.0.
    let storage = HistoricalStorage::new();
    assert!((storage.calculate_avg_delta_chain() - 5.0).abs() < f64::EPSILON);
}

#[test]
fn test_calculate_avg_delta_chain_includes_edge_versions() {
    // Issue #366: the average considers both node and edge version chains.
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 10,
        max_delta_chain: 20,
    });
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Node chain: 3 versions -> 1 anchor + 2 deltas
    for i in 0..3u64 {
        storage
            .add_node_version(
                NodeId::new(1).unwrap(),
                VersionId::new(100 + i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new().insert("v", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Edge chain: 3 versions -> 1 anchor + 2 deltas
    for i in 0..3u64 {
        storage
            .add_edge_version(
                EdgeId::new(1).unwrap(),
                VersionId::new(300 + i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                NodeId::new(1).unwrap(),
                NodeId::new(2).unwrap(),
                PropertyMapBuilder::new().insert("w", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Expected: (2 node deltas + 2 edge deltas) / (1 + 1 anchors) = 2.0
    assert!((storage.calculate_avg_delta_chain() - 2.0).abs() < f64::EPSILON);
}

#[test]
fn test_calculate_avg_delta_chain_anchors_only_is_zero() {
    // Anchors with no deltas: the actual average chain length is 0.0
    // (every lookup hits an anchor directly), not the 5.0 fallback.
    let mut storage = HistoricalStorage::new();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    storage
        .add_node_version(
            NodeId::new(1).unwrap(),
            VersionId::new(100).unwrap(),
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    assert!(storage.calculate_avg_delta_chain().abs() < f64::EPSILON);
}

// ============================================================
// Vector Property Tests (VS-012)
// ============================================================
//
// Note on floating-point equality:
// These tests use exact equality (assert_eq!) which works because vectors
// are hardcoded values without computation. PropertyValue::Vector uses
// derived PartialEq (bitwise comparison). For tests involving computed
// vectors (normalization, etc.), use approximate equality instead:
//
//   fn vectors_approx_equal(a: &[f32], b: &[f32], epsilon: f32) -> bool {
//       a.len() == b.len() &&
//       a.iter().zip(b).all(|(x, y)| (x - y).abs() < epsilon)
//   }
//
// See PropertyValue::Vector documentation at src/core/property.rs for details.

#[test]
fn test_create_node_version_with_vector_property() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("Document").unwrap();
    let temporal = BiTemporalInterval::current(1000.into());

    // Create node with vector embedding
    let embedding = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
    let props = PropertyMapBuilder::new()
        .insert("title", "Test Document")
        .insert_vector("embedding", &embedding)
        .build();

    storage
        .add_node_version(
            node_id,
            version_id,
            temporal.valid_time().start(),
            temporal.transaction_time().start(),
            label,
            props,
            false, // not a tombstone
        )
        .unwrap();

    // First version should be an anchor
    let version = storage.get_node_version(version_id).unwrap();
    assert!(version.is_anchor());

    // Verify vector can be reconstructed
    let reconstructed = storage.reconstruct_node_properties(version_id).unwrap();
    assert_eq!(
        reconstructed.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding[..])
    );
}

#[test]
fn test_delta_computation_with_vector_change() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Document").unwrap();

    // Version 1: Initial embedding
    let v1 = VersionId::new(1).unwrap();
    let embedding_v1 = vec![0.1f32, 0.2, 0.3];
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("title", "Doc")
                .insert_vector("embedding", &embedding_v1)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 2: Updated embedding (should create delta)
    let v2 = VersionId::new(2).unwrap();
    let embedding_v2 = vec![0.4f32, 0.5, 0.6];
    storage
        .add_node_version(
            node_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("title", "Doc")
                .insert_vector("embedding", &embedding_v2)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // V2 should be a delta since we're within anchor interval
    let version = storage.get_node_version(v2).unwrap();
    assert!(version.is_delta());

    // Verify both versions reconstruct correctly
    let props_v1 = storage.reconstruct_node_properties(v1).unwrap();
    assert_eq!(
        props_v1.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding_v1[..])
    );

    let props_v2 = storage.reconstruct_node_properties(v2).unwrap();
    assert_eq!(
        props_v2.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding_v2[..])
    );
}

#[test]
fn test_delta_only_vector_changes() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Document").unwrap();

    // Version 1: title + embedding
    let v1 = VersionId::new(1).unwrap();
    let embedding_v1 = vec![0.1f32, 0.2];
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("title", "Same Title")
                .insert_vector("embedding", &embedding_v1)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 2: Only embedding changes
    let v2 = VersionId::new(2).unwrap();
    let embedding_v2 = vec![0.9f32, 0.8];
    storage
        .add_node_version(
            node_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("title", "Same Title") // Unchanged
                .insert_vector("embedding", &embedding_v2) // Changed
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Verify delta captures only the vector change
    let version = storage.get_node_version(v2).unwrap();
    assert!(version.is_delta());

    // Reconstruct and verify
    let props = storage.reconstruct_node_properties(v2).unwrap();
    assert_eq!(
        props.get("title").and_then(|v| v.as_str()),
        Some("Same Title")
    );
    assert_eq!(
        props.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding_v2[..])
    );
}

#[test]
fn test_vector_unchanged_between_versions() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Document").unwrap();

    // Same embedding for both versions
    let embedding = vec![0.5f32, 0.5, 0.5];

    // Version 1
    let v1 = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("title", "V1 Title")
                .insert_vector("embedding", &embedding)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 2: Same embedding, different title
    let v2 = VersionId::new(2).unwrap();
    storage
        .add_node_version(
            node_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("title", "V2 Title")
                .insert_vector("embedding", &embedding) // Unchanged
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Both should have correct embeddings
    let props_v1 = storage.reconstruct_node_properties(v1).unwrap();
    let props_v2 = storage.reconstruct_node_properties(v2).unwrap();

    assert_eq!(
        props_v1.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding[..])
    );
    assert_eq!(
        props_v2.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding[..])
    );

    // Titles should differ
    assert_eq!(
        props_v1.get("title").and_then(|v| v.as_str()),
        Some("V1 Title")
    );
    assert_eq!(
        props_v2.get("title").and_then(|v| v.as_str()),
        Some("V2 Title")
    );
}

#[test]
fn test_anchor_creation_with_vector() {
    // Configure anchor interval of 2 to force anchor creation
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 2,
        max_delta_chain: 10,
    });

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Document").unwrap();

    // Create 3 versions with different embeddings
    let embeddings = [vec![0.1f32, 0.2], vec![0.3f32, 0.4], vec![0.5f32, 0.6]];

    for (i, emb) in embeddings.iter().enumerate() {
        storage
            .add_node_version(
                node_id,
                VersionId::new(i as u64).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new()
                    .insert_vector("embedding", emb)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // V0: anchor (first), V1: delta, V2: anchor (interval=2)
    assert!(
        storage
            .get_node_version(VersionId::new(0).unwrap())
            .unwrap()
            .is_anchor()
    );
    assert!(
        storage
            .get_node_version(VersionId::new(1).unwrap())
            .unwrap()
            .is_delta()
    );
    assert!(
        storage
            .get_node_version(VersionId::new(2).unwrap())
            .unwrap()
            .is_anchor()
    );

    // Verify each version reconstructs correctly
    for (i, emb) in embeddings.iter().enumerate() {
        let props = storage
            .reconstruct_node_properties(VersionId::new(i as u64).unwrap())
            .unwrap();
        assert_eq!(
            props.get("embedding").and_then(|v| v.as_vector()),
            Some(&emb[..])
        );
    }
}

#[test]
fn test_edge_version_with_vector() {
    let mut storage = HistoricalStorage::new();

    let edge_id = EdgeId::new(1).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("SIMILAR_TO").unwrap();
    let temporal = BiTemporalInterval::current(1000.into());
    let source = NodeId::new(10).unwrap();
    let target = NodeId::new(20).unwrap();

    // Edge with relationship embedding
    let embedding = vec![0.8f32, 0.1, 0.1];
    let props = PropertyMapBuilder::new()
        .insert("weight", 0.95f64)
        .insert_vector("embedding", &embedding)
        .build();

    storage
        .add_edge_version(
            edge_id,
            version_id,
            temporal.valid_time().start(),
            temporal.transaction_time().start(),
            label,
            source,
            target,
            props,
            false, // not a tombstone
        )
        .unwrap();

    // Verify edge version
    let version = storage.get_edge_version(version_id).unwrap();
    assert!(version.is_anchor());

    // Verify properties
    let reconstructed = storage.reconstruct_edge_properties(version_id).unwrap();
    assert_eq!(
        reconstructed.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding[..])
    );
    assert_eq!(
        reconstructed.get("weight").and_then(|v| v.as_float()),
        Some(0.95)
    );
}

#[test]
fn test_edge_delta_with_vector_change() {
    let mut storage = HistoricalStorage::new();

    let edge_id = EdgeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("SIMILAR_TO").unwrap();
    let source = NodeId::new(10).unwrap();
    let target = NodeId::new(20).unwrap();

    // Version 1: Initial edge
    let v1 = VersionId::new(1).unwrap();
    let embedding_v1 = vec![0.5f32, 0.5];
    storage
        .add_edge_version(
            edge_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new()
                .insert("weight", 0.5f64)
                .insert_vector("embedding", &embedding_v1)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 2: Updated embedding and weight
    let v2 = VersionId::new(2).unwrap();
    let embedding_v2 = vec![0.9f32, 0.1];
    storage
        .add_edge_version(
            edge_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new()
                .insert("weight", 0.9f64)
                .insert_vector("embedding", &embedding_v2)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // V2 should be delta
    assert!(storage.get_edge_version(v2).unwrap().is_delta());

    // Verify reconstruction
    let props_v2 = storage.reconstruct_edge_properties(v2).unwrap();
    assert_eq!(
        props_v2.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding_v2[..])
    );
    assert_eq!(props_v2.get("weight").and_then(|v| v.as_float()), Some(0.9));
}

#[test]
fn test_high_dimensional_vector_versioning() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Embedding").unwrap();

    // High-dimensional embedding (like OpenAI's 1536-dim)
    const DIMENSIONS: usize = 1536;
    let embedding: Vec<f32> = (0..DIMENSIONS)
        .map(|i| (i as f32) / DIMENSIONS as f32)
        .collect();

    let v1 = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert_vector("embedding", &embedding)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Verify reconstruction preserves all dimensions
    let props = storage.reconstruct_node_properties(v1).unwrap();
    let retrieved = props
        .get("embedding")
        .and_then(|v| v.as_vector())
        .expect("Should have embedding");

    assert_eq!(retrieved.len(), DIMENSIONS);
    assert_eq!(retrieved, &embedding[..]);
}

#[test]
fn test_version_time_travel_with_vectors() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Document").unwrap();

    // Create versions at different times with different embeddings
    let embeddings = [
        (0, 500, vec![0.1f32, 0.0]),                          // valid 0-500
        (500, 1000, vec![0.2f32, 0.0]),                       // valid 500-1000
        (1000, TIMESTAMP_MAX.wallclock(), vec![0.3f32, 0.0]), // valid 1000+
    ];

    for (i, (start, _end, emb)) in embeddings.iter().enumerate() {
        storage
            .add_node_version(
                node_id,
                VersionId::new(i as u64).unwrap(),
                (*start).into(),
                0.into(),
                label,
                PropertyMapBuilder::new()
                    .insert_vector("embedding", emb)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Query at different times
    let v_at_250 = storage.find_node_version_at_time(node_id, 250.into(), 0.into());
    let v_at_750 = storage.find_node_version_at_time(node_id, 750.into(), 0.into());
    let v_at_1500 = storage.find_node_version_at_time(node_id, 1500.into(), 0.into());

    assert_eq!(v_at_250, Some(VersionId::new(0).unwrap()));
    assert_eq!(v_at_750, Some(VersionId::new(1).unwrap()));
    assert_eq!(v_at_1500, Some(VersionId::new(2).unwrap()));

    // Verify each has correct embedding
    for (vid, expected_emb) in [
        (v_at_250.unwrap(), &embeddings[0].2),
        (v_at_750.unwrap(), &embeddings[1].2),
        (v_at_1500.unwrap(), &embeddings[2].2),
    ] {
        let props = storage.reconstruct_node_properties(vid).unwrap();
        assert_eq!(
            props.get("embedding").and_then(|v| v.as_vector()),
            Some(&expected_emb[..])
        );
    }
}

// ============================================================
// Edge Case Tests
// ============================================================

#[test]
fn test_empty_vector_versioning() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("EmptyEmbedding").unwrap();

    // Empty vector should work with delta compression
    let empty_vec: Vec<f32> = vec![];

    // Version 1: empty vector
    let v1 = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("name", "empty")
                .insert_vector("embedding", &empty_vec)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 2: still empty (should be excluded from delta as unchanged)
    let v2 = VersionId::new(2).unwrap();
    storage
        .add_node_version(
            node_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("name", "updated")
                .insert_vector("embedding", &empty_vec)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Both versions should have empty embedding
    let props_v1 = storage.reconstruct_node_properties(v1).unwrap();
    let props_v2 = storage.reconstruct_node_properties(v2).unwrap();

    assert_eq!(
        props_v1.get("embedding").and_then(|v| v.as_vector()),
        Some(&empty_vec[..])
    );
    assert_eq!(
        props_v2.get("embedding").and_then(|v| v.as_vector()),
        Some(&empty_vec[..])
    );
}

#[test]
fn test_vector_with_special_float_values() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("SpecialFloats").unwrap();

    // Note: NaN and Infinity are allowed in storage (validation is optional).
    // However, NaN != NaN per IEEE 754, so delta computation treats NaN
    // as always changed. This test documents that behavior.

    let special_vec = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];

    let v1 = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert_vector("embedding", &special_vec)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Verify special values round-trip correctly
    let props = storage.reconstruct_node_properties(v1).unwrap();
    let retrieved = props
        .get("embedding")
        .and_then(|v| v.as_vector())
        .expect("Should have embedding");

    assert!(retrieved[0].is_infinite() && retrieved[0].is_sign_positive());
    assert!(retrieved[1].is_infinite() && retrieved[1].is_sign_negative());
    assert_eq!(retrieved[2], 0.0);
    assert_eq!(retrieved[3], -0.0);
}

#[test]
fn test_nan_in_vector_delta_behavior() {
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("NaNTest").unwrap();

    // NaN != NaN per IEEE 754, so same NaN values will be detected as
    // "changed" in delta computation. This is documented behavior.
    let nan_vec = vec![f32::NAN, 1.0];

    let v1 = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert_vector("embedding", &nan_vec)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Same NaN values - will be treated as changed due to NaN != NaN
    let v2 = VersionId::new(2).unwrap();
    storage
        .add_node_version(
            node_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new()
                .insert_vector("embedding", &nan_vec)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Both should reconstruct with NaN values
    let props_v1 = storage.reconstruct_node_properties(v1).unwrap();
    let props_v2 = storage.reconstruct_node_properties(v2).unwrap();

    let vec1 = props_v1
        .get("embedding")
        .and_then(|v| v.as_vector())
        .unwrap();
    let vec2 = props_v2
        .get("embedding")
        .and_then(|v| v.as_vector())
        .unwrap();

    // Both should have NaN at index 0
    assert!(vec1[0].is_nan());
    assert!(vec2[0].is_nan());
    assert_eq!(vec1[1], 1.0);
    assert_eq!(vec2[1], 1.0);
}

// ============================================================
// Cache Tests
// ============================================================

#[test]
fn test_cache_hit_on_second_read() {
    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(1).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();
    let timestamp = 1000.into();
    let props = PropertyMapBuilder::new()
        .insert("name", "Alice")
        .insert("age", 30i64)
        .build();

    storage
        .add_node_version(
            node_id,
            version_id,
            timestamp,
            timestamp,
            label,
            props.clone(),
            false, // not a tombstone
        )
        .unwrap();

    // First read - cache miss, populates cache
    let result1 = storage.reconstruct_node_properties(version_id).unwrap();
    assert_eq!(result1.get("name").and_then(|v| v.as_str()), Some("Alice"));

    // Check cache was populated
    let stats = storage.stats();
    assert_eq!(stats.node_cache_entries, 1);

    // Second read - should hit cache
    let result2 = storage.reconstruct_node_properties(version_id).unwrap();
    assert_eq!(result2.get("name").and_then(|v| v.as_str()), Some("Alice"));

    // Cache size shouldn't change
    let stats = storage.stats();
    assert_eq!(stats.node_cache_entries, 1);
}

#[test]
fn test_cache_populates_delta_chain() {
    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Create anchor + 3 deltas
    let v1 = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().insert("value", 1i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    let v2 = VersionId::new(2).unwrap();
    storage
        .add_node_version(
            node_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new().insert("value", 2i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    let v3 = VersionId::new(3).unwrap();
    storage
        .add_node_version(
            node_id,
            v3,
            3000.into(),
            3000.into(),
            label,
            PropertyMapBuilder::new().insert("value", 3i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    let v4 = VersionId::new(4).unwrap();
    storage
        .add_node_version(
            node_id,
            v4,
            4000.into(),
            4000.into(),
            label,
            PropertyMapBuilder::new().insert("value", 4i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Reconstruct v4 (latest delta) - should populate entire chain
    let result = storage.reconstruct_node_properties(v4).unwrap();
    assert_eq!(result.get("value").and_then(|v| v.as_int()), Some(4.into()));

    // Cache should have all versions in the chain
    let stats = storage.stats();
    assert!(stats.node_cache_entries >= 4);
}

#[test]
fn test_cache_with_custom_size() {
    let storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig::default(),
        RetentionPolicy::default(),
        100, // Small cache size
    );

    let stats = storage.stats();
    assert_eq!(stats.node_cache_entries, 0);
    assert_eq!(stats.edge_cache_entries, 0);
}

#[test]
fn test_edge_cache_functionality() {
    let mut storage = HistoricalStorage::new();
    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(10).unwrap();
    let target = NodeId::new(20).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    let temporal = BiTemporalInterval::current(1000.into());
    let props = PropertyMapBuilder::new().insert("since", 2020i64).build();

    storage
        .add_edge_version(
            edge_id,
            version_id,
            temporal.valid_time().start(),
            temporal.transaction_time().start(),
            label,
            source,
            target,
            props,
            false, // not a tombstone
        )
        .unwrap();

    // First read - cache miss
    let result1 = storage.reconstruct_edge_properties(version_id).unwrap();
    assert_eq!(
        result1.get("since").and_then(|v| v.as_int()),
        Some(2020.into())
    );

    // Check cache was populated
    let stats = storage.stats();
    assert_eq!(stats.edge_cache_entries, 1);

    // Second read - should hit cache
    let result2 = storage.reconstruct_edge_properties(version_id).unwrap();
    assert_eq!(
        result2.get("since").and_then(|v| v.as_int()),
        Some(2020.into())
    );

    // Cache size shouldn't change
    let stats = storage.stats();
    assert_eq!(stats.edge_cache_entries, 1);
}

#[test]
fn test_cache_stats_accuracy() {
    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(1).unwrap();
    let edge_id = EdgeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();
    let timestamp = 1000.into();

    // Create 5 node versions
    for i in 0..5 {
        let version_id = VersionId::new(i).unwrap();
        storage
            .add_node_version(
                node_id,
                version_id,
                timestamp,
                timestamp,
                label,
                PropertyMapBuilder::new().insert("value", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
        // Reconstruct to populate cache
        storage.reconstruct_node_properties(version_id).unwrap();
    }

    // Create 3 edge versions
    for i in 0..3 {
        let version_id = VersionId::new(100 + i).unwrap();
        storage
            .add_edge_version(
                edge_id,
                version_id,
                timestamp,
                timestamp,
                label,
                node_id,
                node_id,
                PropertyMapBuilder::new().insert("value", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
        // Reconstruct to populate cache
        storage.reconstruct_edge_properties(version_id).unwrap();
    }

    let stats = storage.stats();
    assert_eq!(stats.node_cache_entries, 5);
    assert_eq!(stats.edge_cache_entries, 3);
}

#[test]
fn test_cache_with_large_properties() {
    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(1).unwrap();
    let version_id = VersionId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Document").unwrap();

    // Create large property map with vector
    let large_vector: Vec<f32> = (0..1536).map(|i| i as f32 / 1536.0).collect();
    let props = PropertyMapBuilder::new()
        .insert("title", "Large Document")
        .insert_vector("embedding", &large_vector)
        .insert("content", "x".repeat(10000).as_str())
        .build();

    storage
        .add_node_version(
            node_id,
            version_id,
            1000.into(),
            1000.into(),
            label,
            props,
            false,
        )
        .unwrap();

    // First read
    let result1 = storage.reconstruct_node_properties(version_id).unwrap();
    assert_eq!(
        result1
            .get("embedding")
            .and_then(|v| v.as_vector())
            .map(|v| v.len()),
        Some(1536)
    );

    // Second read should hit cache
    let result2 = storage.reconstruct_node_properties(version_id).unwrap();
    assert_eq!(result1.get("title"), result2.get("title"));

    let stats = storage.stats();
    assert_eq!(stats.node_cache_entries, 1);
}

#[test]
fn test_extract_node_version_data() {
    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(1).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();
    let temporal = BiTemporalInterval::current(1000.into());
    let props = PropertyMapBuilder::new().insert("name", "Bob").build();

    storage
        .add_node_version(
            node_id,
            version_id,
            temporal.valid_time().start(),
            temporal.transaction_time().start(),
            label,
            props,
            false, // not a tombstone
        )
        .unwrap();

    let (vid, nid, lbl, data) = storage.extract_node_version_data(version_id).unwrap();
    assert_eq!(vid, version_id);
    assert_eq!(nid, node_id);
    assert_eq!(lbl, label);

    // Verify data can be used for copy-out reconstruction
    match data {
        VersionData::Anchor { properties, .. } => {
            assert_eq!(properties.get("name").and_then(|v| v.as_str()), Some("Bob"));
        }
        _ => panic!("Expected anchor"),
    }
}

#[test]
fn test_extract_edge_version_data() {
    let mut storage = HistoricalStorage::new();
    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(10).unwrap();
    let target = NodeId::new(20).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    let temporal = BiTemporalInterval::current(1000.into());
    let props = PropertyMapBuilder::new().insert("since", 2021i64).build();

    storage
        .add_edge_version(
            edge_id,
            version_id,
            temporal.valid_time().start(),
            temporal.transaction_time().start(),
            label,
            source,
            target,
            props,
            false, // not a tombstone
        )
        .unwrap();

    let (vid, eid, lbl, src, tgt, data) = storage.extract_edge_version_data(version_id).unwrap();
    assert_eq!(vid, version_id);
    assert_eq!(eid, edge_id);
    assert_eq!(lbl, label);
    assert_eq!(src, source);
    assert_eq!(tgt, target);

    // Verify data
    match data {
        VersionData::Anchor { properties, .. } => {
            assert_eq!(
                properties.get("since").and_then(|v| v.as_int()),
                Some(2021.into())
            );
        }
        _ => panic!("Expected anchor"),
    }
}

// ============================================================
// Observer Pattern Tests (VS-047)
// ============================================================

use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Mock observer that counts anchor events
struct CountingObserver {
    anchor_count: AtomicUsize,
    version_count: AtomicUsize,
}

impl StorageObserver for CountingObserver {
    fn on_event(&self, event: &StorageEvent) -> Result<()> {
        match event {
            StorageEvent::NodeAnchorCreated { .. } | StorageEvent::EdgeAnchorCreated { .. } => {
                self.anchor_count.fetch_add(1, Ordering::SeqCst);
            }
            StorageEvent::NodeVersionCreated { .. } | StorageEvent::EdgeVersionCreated { .. } => {
                self.version_count.fetch_add(1, Ordering::SeqCst);
            }
        }
        Ok(())
    }
}

/// Mock observer that only cares about node anchors
struct NodeAnchorObserver {
    count: AtomicUsize,
}

impl StorageObserver for NodeAnchorObserver {
    fn on_event(&self, _event: &StorageEvent) -> Result<()> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn interested_in(&self, event: &StorageEvent) -> bool {
        matches!(event, StorageEvent::NodeAnchorCreated { .. })
    }
}

/// Mock observer that collects events
struct CollectingObserver {
    events: StdMutex<Vec<StorageEvent>>,
}

impl StorageObserver for CollectingObserver {
    fn on_event(&self, event: &StorageEvent) -> Result<()> {
        self.events.lock().unwrap().push(event.clone());
        Ok(())
    }
}

#[test]
fn test_observer_triggered_on_node_anchor_creation() {
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    });

    let observer = Arc::new(CountingObserver {
        anchor_count: AtomicUsize::new(0),
        version_count: AtomicUsize::new(0),
    });
    storage.add_observer(observer.clone());

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create 5 versions: anchor, delta, delta, anchor, delta
    for i in 0..5 {
        storage
            .add_node_version(
                node_id,
                VersionId::new(i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new().insert("value", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Should have 2 anchors (v0 and v3)
    assert_eq!(observer.anchor_count.load(Ordering::SeqCst), 2);
    // Should have 5 total version events
    assert_eq!(observer.version_count.load(Ordering::SeqCst), 5);
}

#[test]
fn test_observer_triggered_on_edge_anchor_creation() {
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 2,
        max_delta_chain: 10,
    });

    let observer = Arc::new(CountingObserver {
        anchor_count: AtomicUsize::new(0),
        version_count: AtomicUsize::new(0),
    });
    storage.add_observer(observer.clone());

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(10).unwrap();
    let target = NodeId::new(20).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // Create 3 versions: anchor, delta, anchor
    for i in 0..3 {
        storage
            .add_edge_version(
                edge_id,
                VersionId::new(i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                source,
                target,
                PropertyMapBuilder::new().build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Should have 2 anchors
    assert_eq!(observer.anchor_count.load(Ordering::SeqCst), 2);
}

#[test]
fn test_observer_filtering() {
    let mut storage = HistoricalStorage::new();

    // Observer only interested in node anchors
    let observer = Arc::new(NodeAnchorObserver {
        count: AtomicUsize::new(0),
    });
    storage.add_observer(observer.clone());

    let node_id = NodeId::new(1).unwrap();
    let edge_id = EdgeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create node anchor
    storage
        .add_node_version(
            node_id,
            VersionId::new(1).unwrap(),
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    // Create edge anchor
    storage
        .add_edge_version(
            edge_id,
            VersionId::new(2).unwrap(),
            2000.into(),
            2000.into(),
            label,
            node_id,
            node_id,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    // Should only count node anchor (not edge anchor)
    assert_eq!(observer.count.load(Ordering::SeqCst), 1);
}

#[test]
fn test_observer_receives_correct_event_data() {
    let mut storage = HistoricalStorage::new();

    let collector = Arc::new(CollectingObserver {
        events: StdMutex::new(Vec::new()),
    });
    storage.add_observer(collector.clone());

    let node_id = NodeId::new(42).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();
    let timestamp = 5000i64;

    storage
        .add_node_version(
            node_id,
            version_id,
            timestamp.into(),
            timestamp.into(),
            label,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    let events = collector.events.lock().unwrap();
    assert_eq!(events.len(), 2); // NodeAnchorCreated + NodeVersionCreated

    // Check anchor event
    let anchor_event = events
        .iter()
        .find(|e| matches!(e, StorageEvent::NodeAnchorCreated { .. }))
        .expect("Should have NodeAnchorCreated event");

    match anchor_event {
        StorageEvent::NodeAnchorCreated {
            version_id: vid,
            node_id: nid,
            timestamp: ts,
        } => {
            assert_eq!(*vid, version_id);
            assert_eq!(*nid, node_id);
            assert_eq!(*ts, timestamp.into());
        }
        _ => panic!("Wrong event type"),
    }
}

#[test]
fn test_multiple_observers() {
    let mut storage = HistoricalStorage::new();

    let observer1 = Arc::new(CountingObserver {
        anchor_count: AtomicUsize::new(0),
        version_count: AtomicUsize::new(0),
    });
    let observer2 = Arc::new(CountingObserver {
        anchor_count: AtomicUsize::new(0),
        version_count: AtomicUsize::new(0),
    });

    storage.add_observer(observer1.clone());
    storage.add_observer(observer2.clone());

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    storage
        .add_node_version(
            node_id,
            VersionId::new(1).unwrap(),
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    // Both observers should be notified
    assert_eq!(observer1.anchor_count.load(Ordering::SeqCst), 1);
    assert_eq!(observer2.anchor_count.load(Ordering::SeqCst), 1);
}

#[test]
fn test_observer_error_doesnt_block_storage() {
    /// Observer that always returns an error
    struct FailingObserver;

    impl StorageObserver for FailingObserver {
        fn on_event(&self, _event: &StorageEvent) -> Result<()> {
            Err(crate::core::error::Error::Storage(
                StorageError::InconsistentState {
                    reason: "Test error".to_string(),
                },
            ))
        }
    }

    let mut storage = HistoricalStorage::new();
    storage.add_observer(Arc::new(FailingObserver));

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Should succeed even though observer fails
    let result = storage.add_node_version(
        node_id,
        VersionId::new(1).unwrap(),
        1000.into(),
        1000.into(),
        label,
        PropertyMapBuilder::new().build(),
        false, // not a tombstone
    );

    assert!(result.is_ok());

    // Verify version was created
    let version = storage.get_node_version(VersionId::new(1).unwrap());
    assert!(version.is_some());
}

// ========================================================================
// Pre-Anchor Hook Tests
// ========================================================================

#[test]
fn test_pre_anchor_hook_called_before_storage() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut storage = HistoricalStorage::new();
    let hook_called = Arc::new(AtomicBool::new(false));
    let hook_called_clone = Arc::clone(&hook_called);

    // Hook that sets flag when called
    let hook: PreAnchorHook = Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| {
        hook_called_clone.store(true, Ordering::SeqCst);
        Ok(Some(42))
    });

    storage.register_pre_node_anchor_hook(hook);

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create anchor (first version is always anchor)
    storage
        .add_node_version(
            node_id,
            VersionId::new(1).unwrap(),
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    // Hook should have been called
    assert!(hook_called.load(Ordering::SeqCst));
}

#[test]
fn test_pre_anchor_hook_returns_snapshot_id() {
    let mut storage = HistoricalStorage::new();

    // Hook that returns snapshot ID 123
    let hook: PreAnchorHook =
        Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| Ok(Some(123)));

    storage.register_pre_node_anchor_hook(hook);

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create anchor
    storage
        .add_node_version(
            node_id,
            VersionId::new(1).unwrap(),
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    // Verify snapshot ID was stored in anchor
    let version = storage
        .get_node_version(VersionId::new(1).unwrap())
        .unwrap();
    assert!(version.is_anchor());
    assert_eq!(version.data.get_vector_snapshot_id(), Some(123));
}

#[test]
fn test_pre_anchor_hook_none_handling() {
    let mut storage = HistoricalStorage::new();

    // Hook that returns None (no snapshot needed)
    let hook: PreAnchorHook =
        Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| Ok(None));

    storage.register_pre_node_anchor_hook(hook);

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create anchor - should succeed even with None
    storage
        .add_node_version(
            node_id,
            VersionId::new(1).unwrap(),
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    // Verify anchor created without snapshot ID
    let version = storage
        .get_node_version(VersionId::new(1).unwrap())
        .unwrap();
    assert!(version.is_anchor());
    assert_eq!(version.data.get_vector_snapshot_id(), None);
}

#[test]
fn test_pre_anchor_hook_error_graceful_degradation() {
    let mut storage = HistoricalStorage::new();

    // Hook that always fails
    let hook: PreAnchorHook = Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| {
        Err(crate::core::error::Error::Storage(
            StorageError::InconsistentState {
                reason: "Test hook error".to_string(),
            },
        ))
    });

    storage.register_pre_node_anchor_hook(hook);

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create anchor - should succeed despite hook failure (graceful degradation)
    let result = storage.add_node_version(
        node_id,
        VersionId::new(1).unwrap(),
        1000.into(),
        1000.into(),
        label,
        PropertyMapBuilder::new().build(),
        false, // not a tombstone
    );

    assert!(result.is_ok());

    // Verify anchor created without snapshot ID
    let version = storage
        .get_node_version(VersionId::new(1).unwrap())
        .unwrap();
    assert!(version.is_anchor());
    assert_eq!(version.data.get_vector_snapshot_id(), None);
}

#[test]
fn test_pre_anchor_hook_not_called_for_delta() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    });

    let hook_call_count = Arc::new(AtomicUsize::new(0));
    let hook_call_count_clone = Arc::clone(&hook_call_count);

    // Hook that counts calls
    let hook: PreAnchorHook = Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| {
        hook_call_count_clone.fetch_add(1, Ordering::SeqCst);
        Ok(Some(42))
    });

    storage.register_pre_node_anchor_hook(hook);

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create 5 versions (anchor at v0, deltas at v1-v2, anchor at v3, delta at v4)
    for i in 0..5 {
        storage
            .add_node_version(
                node_id,
                VersionId::new(100 + i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new().build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Hook should be called only for anchors (v0 and v3)
    assert_eq!(hook_call_count.load(Ordering::SeqCst), 2);
}

#[test]
fn test_pre_anchor_hook_node_and_edge_separate() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut storage = HistoricalStorage::new();

    let node_hook_count = Arc::new(AtomicUsize::new(0));
    let node_hook_count_clone = Arc::clone(&node_hook_count);
    let node_hook: PreAnchorHook =
        Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| {
            node_hook_count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(Some(1))
        });

    let edge_hook_count = Arc::new(AtomicUsize::new(0));
    let edge_hook_count_clone = Arc::clone(&edge_hook_count);
    let edge_hook: PreAnchorHook =
        Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| {
            edge_hook_count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(Some(2))
        });

    storage.register_pre_node_anchor_hook(node_hook);
    storage.register_pre_edge_anchor_hook(edge_hook);

    let node1_id = NodeId::new(1).unwrap();
    let node2_id = NodeId::new(2).unwrap();
    let edge_id = EdgeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create node version
    storage
        .add_node_version(
            node1_id,
            VersionId::new(1).unwrap(),
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    // Create edge version
    storage
        .add_edge_version(
            edge_id,
            VersionId::new(2).unwrap(),
            2000.into(),
            2000.into(),
            label,
            node1_id,
            node2_id,
            PropertyMapBuilder::new().build(),
            false, // not a tombstone
        )
        .unwrap();

    // Each hook should be called once
    assert_eq!(node_hook_count.load(Ordering::SeqCst), 1);
    assert_eq!(edge_hook_count.load(Ordering::SeqCst), 1);

    // Verify snapshot IDs are different
    let node_version = storage
        .get_node_version(VersionId::new(1).unwrap())
        .unwrap();
    let edge_version = storage
        .get_edge_version(VersionId::new(2).unwrap())
        .unwrap();
    assert_eq!(node_version.data.get_vector_snapshot_id(), Some(1));
    assert_eq!(edge_version.data.get_vector_snapshot_id(), Some(2));
}

// ------------------------------------------------------------------------
// Issue #354: PreAnchorHook failure-recovery coverage (test-only).
//
// The node Err graceful-degradation path is already covered above by
// `test_pre_anchor_hook_error_graceful_degradation`. The tests below close
// the remaining gaps: the EDGE Err path, that subsequent writes stay correct
// after a failed anchor, that failure is per-invocation (recovery on a later
// anchor), that a very large property map does not panic, and (feature-gated)
// that the failure is logged at WARN level.
// ------------------------------------------------------------------------

/// Edge analogue of `test_pre_anchor_hook_error_graceful_degradation`: a
/// failing pre-edge-anchor hook must not fail the edge write; the anchor is
/// still created, just without a vector snapshot id (graceful degradation).
#[test]
fn test_pre_anchor_hook_error_graceful_degradation_edge() {
    let mut storage = HistoricalStorage::new();

    // Hook that always fails.
    let hook: PreAnchorHook = Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| {
        Err(crate::core::error::Error::Storage(
            StorageError::InconsistentState {
                reason: "Test edge hook error".to_string(),
            },
        ))
    });

    storage.register_pre_edge_anchor_hook(hook);

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // First edge version is always an anchor -> hook fires and fails.
    let result = storage.add_edge_version(
        edge_id,
        VersionId::new(1).unwrap(),
        2000.into(),
        2000.into(),
        label,
        source,
        target,
        PropertyMapBuilder::new().build(),
        false, // not a tombstone
    );

    // Write succeeds despite the hook failure (graceful degradation).
    assert!(result.is_ok());

    // Anchor exists, is an anchor, and carries no snapshot id.
    let version = storage
        .get_edge_version(VersionId::new(1).unwrap())
        .unwrap();
    assert!(version.is_anchor());
    assert_eq!(version.data.get_vector_snapshot_id(), None);
}

/// A failed anchor hook must not corrupt subsequent writes: after the failing
/// anchor, later delta versions still reconstruct their properties correctly.
///
/// NOTE (per #3504 coordination): this asserts only on property reconstruction,
/// snapshot ids, version existence, and anchor-ness -- never on a superseded
/// version's exact `valid_to`/interval-close values.
#[test]
fn test_writes_continue_correctly_after_hook_failure() {
    // Large anchor interval so that v2/v3 following the failed anchor are
    // deltas (their correctness is what proves the failure didn't corrupt
    // subsequent writes).
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 5,
        max_delta_chain: 10,
    });

    // Hook that always fails.
    let hook: PreAnchorHook = Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| {
        Err(crate::core::error::Error::Storage(
            StorageError::InconsistentState {
                reason: "Test hook error".to_string(),
            },
        ))
    });

    storage.register_pre_node_anchor_hook(hook);

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // v1: anchor (hook fires and fails), name=Alice age=30
    let v1 = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("name", "Alice")
                .insert("age", 30i64)
                .build(),
            false,
        )
        .unwrap();

    // v2: delta, age=31
    let v2 = VersionId::new(2).unwrap();
    storage
        .add_node_version(
            node_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("name", "Alice")
                .insert("age", 31i64)
                .build(),
            false,
        )
        .unwrap();

    // v3: delta, age=32
    let v3 = VersionId::new(3).unwrap();
    storage
        .add_node_version(
            node_id,
            v3,
            3000.into(),
            3000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("name", "Alice")
                .insert("age", 32i64)
                .build(),
            false,
        )
        .unwrap();

    // The failed anchor is still an anchor and carries no snapshot id.
    let anchor = storage.get_node_version(v1).unwrap();
    assert!(anchor.is_anchor());
    assert_eq!(anchor.data.get_vector_snapshot_id(), None);

    // Subsequent deltas reconstruct their properties correctly despite the
    // failed anchor -- the write path was not corrupted.
    let props_v1 = storage.reconstruct_node_properties(v1).unwrap();
    assert_eq!(props_v1.get("name").and_then(|v| v.as_str()), Some("Alice"));
    assert_eq!(
        props_v1.get("age").and_then(|v| v.as_int()),
        Some(30.into())
    );

    let props_v2 = storage.reconstruct_node_properties(v2).unwrap();
    assert_eq!(props_v2.get("name").and_then(|v| v.as_str()), Some("Alice"));
    assert_eq!(
        props_v2.get("age").and_then(|v| v.as_int()),
        Some(31.into())
    );

    let props_v3 = storage.reconstruct_node_properties(v3).unwrap();
    assert_eq!(props_v3.get("name").and_then(|v| v.as_str()), Some("Alice"));
    assert_eq!(
        props_v3.get("age").and_then(|v| v.as_int()),
        Some(32.into())
    );
}

/// Hook failure is per-invocation: a hook that fails on its first call but
/// succeeds on a later call leaves the first anchor without a snapshot id and
/// the later anchor with one. Proves recovery works on subsequent anchors.
#[test]
fn test_pre_anchor_hook_recovers_on_later_anchor() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // anchor_interval: 1 => every version is an anchor, so each write fires
    // the hook.
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 1,
        max_delta_chain: 10,
    });

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = Arc::clone(&call_count);

    // First invocation fails; every later invocation succeeds with id 77.
    let hook: PreAnchorHook = Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| {
        let prev = call_count_clone.fetch_add(1, Ordering::SeqCst);
        if prev == 0 {
            Err(crate::core::error::Error::Storage(
                StorageError::InconsistentState {
                    reason: "First-call hook failure".to_string(),
                },
            ))
        } else {
            Ok(Some(77))
        }
    });

    storage.register_pre_node_anchor_hook(hook);

    let node1 = NodeId::new(1).unwrap();
    let node2 = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // v1: anchor, hook call #1 -> Err -> no snapshot id.
    let v1 = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node1,
            v1,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false,
        )
        .unwrap();

    // v2: anchor (different node), hook call #2 -> Ok(Some(77)) -> snapshot id.
    let v2 = VersionId::new(2).unwrap();
    storage
        .add_node_version(
            node2,
            v2,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false,
        )
        .unwrap();

    // Hook was invoked twice (both versions are anchors).
    assert_eq!(call_count.load(Ordering::SeqCst), 2);

    let first_anchor = storage.get_node_version(v1).unwrap();
    assert!(first_anchor.is_anchor());
    assert_eq!(first_anchor.data.get_vector_snapshot_id(), None);

    let second_anchor = storage.get_node_version(v2).unwrap();
    assert!(second_anchor.is_anchor());
    assert_eq!(second_anchor.data.get_vector_snapshot_id(), Some(77));
}

/// A registered hook plus a very large property map on the anchor must not
/// panic; the anchor is created normally.
#[test]
fn test_pre_anchor_hook_large_property_map_no_panic() {
    let mut storage = HistoricalStorage::new();

    // Hook that returns None (no snapshot) -- exercises the hook path without
    // affecting the large-map assertion.
    let hook: PreAnchorHook =
        Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| Ok(None));
    storage.register_pre_node_anchor_hook(hook);

    // Build a property map with 1000 distinct string keys/values.
    let mut builder = PropertyMapBuilder::new();
    for i in 0..1000 {
        builder = builder.insert(&format!("key_{i}"), format!("value_{i}"));
    }

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Must not panic.
    storage
        .add_node_version(
            node_id,
            VersionId::new(1).unwrap(),
            1000.into(),
            1000.into(),
            label,
            builder.build(),
            false,
        )
        .unwrap();

    // Anchor created normally.
    let version = storage
        .get_node_version(VersionId::new(1).unwrap())
        .unwrap();
    assert!(version.is_anchor());
}

// ========================================================================
// Issue #3525: hook hardening — multi-hook API, panic isolation, timeout.
// ========================================================================

/// Helper: create the first (anchor) node version on `storage`, returning the
/// stored anchor's vector snapshot id (if any).
fn add_first_node_anchor(storage: &mut HistoricalStorage, id: u64) -> Option<usize> {
    let node_id = NodeId::new(id).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();
    let vid = VersionId::new(id).unwrap();
    storage
        .add_node_version(
            node_id,
            vid,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().build(),
            false,
        )
        .unwrap();
    let version = storage.get_node_version(vid).unwrap();
    assert!(version.is_anchor());
    version.data.get_vector_snapshot_id()
}

/// Multiple hooks registered via `add_pre_node_anchor_hook` fire in
/// registration order, and the last `Ok(Some(id))` wins the single snapshot
/// slot.
#[test]
fn test_multiple_pre_anchor_hooks_fire_in_registration_order() {
    use std::sync::Mutex;

    let mut storage = HistoricalStorage::new();
    let order = Arc::new(Mutex::new(Vec::<usize>::new()));

    for i in 0..3usize {
        let order_clone = Arc::clone(&order);
        let hook: PreAnchorHook =
            Arc::new(move |_entity_type, _entity_id, _timestamp, _properties| {
                order_clone.lock().unwrap().push(i);
                // Each hook returns a distinct snapshot id; last should win.
                Ok(Some(100 + i))
            });
        storage.add_pre_node_anchor_hook(hook);
    }

    let snapshot = add_first_node_anchor(&mut storage, 1);

    assert_eq!(*order.lock().unwrap(), vec![0, 1, 2], "hooks ran in order");
    assert_eq!(snapshot, Some(102), "last Ok(Some) wins the snapshot slot");

    let metrics = storage.hook_metrics();
    assert_eq!(metrics.invocations, 3);
    assert_eq!(metrics.successes, 3);
    assert_eq!(metrics.failures, 0);
    assert_eq!(metrics.panics, 0);
    assert_eq!(metrics.timeouts, 0);
}

/// A panicking hook is isolated: the write succeeds, later hooks still run, and
/// the panic is counted.
#[test]
fn test_pre_anchor_hook_panic_isolated() {
    let mut storage = HistoricalStorage::new();

    // Hook 0 panics.
    let panicking: PreAnchorHook = Arc::new(|_entity_type, _entity_id, _timestamp, _properties| {
        panic!("intentional test panic in pre-anchor hook");
    });
    // Hook 1 succeeds after the panic.
    let ok_hook: PreAnchorHook =
        Arc::new(|_entity_type, _entity_id, _timestamp, _properties| Ok(Some(99)));

    storage.add_pre_node_anchor_hook(panicking);
    storage.add_pre_node_anchor_hook(ok_hook);

    let snapshot = add_first_node_anchor(&mut storage, 1);

    // Second hook still ran and its snapshot id was stored despite the panic.
    assert_eq!(snapshot, Some(99));

    let metrics = storage.hook_metrics();
    assert_eq!(metrics.invocations, 2);
    assert_eq!(metrics.panics, 1);
    assert_eq!(metrics.successes, 1);
    assert_eq!(metrics.failures, 0);
}

/// Panic isolation also applies on the inline (no-timeout) path with a single
/// hook: the write is not unwound.
#[test]
fn test_pre_anchor_hook_panic_isolated_inline_single_hook() {
    let mut storage = HistoricalStorage::new();
    assert_eq!(storage.pre_anchor_hook_timeout(), None);

    let panicking: PreAnchorHook = Arc::new(|_entity_type, _entity_id, _timestamp, _properties| {
        panic!("boom");
    });
    storage.register_pre_node_anchor_hook(panicking);

    // Write must succeed (no unwind through the write path) and produce no snapshot.
    let snapshot = add_first_node_anchor(&mut storage, 1);
    assert_eq!(snapshot, None);

    let metrics = storage.hook_metrics();
    assert_eq!(metrics.invocations, 1);
    assert_eq!(metrics.panics, 1);
}

/// An erroring hook is isolated across multiple hooks: successful hooks apply,
/// the failure is counted, and last `Ok(Some)` wins.
#[test]
fn test_pre_anchor_hook_error_isolated_across_multiple() {
    let mut storage = HistoricalStorage::new();

    let ok1: PreAnchorHook =
        Arc::new(|_entity_type, _entity_id, _timestamp, _properties| Ok(Some(1)));
    let failing: PreAnchorHook = Arc::new(|_entity_type, _entity_id, _timestamp, _properties| {
        Err(crate::core::error::Error::Storage(
            StorageError::InconsistentState {
                reason: "middle hook failed".to_string(),
            },
        ))
    });
    let ok3: PreAnchorHook =
        Arc::new(|_entity_type, _entity_id, _timestamp, _properties| Ok(Some(3)));

    storage.add_pre_node_anchor_hook(ok1);
    storage.add_pre_node_anchor_hook(failing);
    storage.add_pre_node_anchor_hook(ok3);

    let snapshot = add_first_node_anchor(&mut storage, 1);
    assert_eq!(
        snapshot,
        Some(3),
        "last successful hook wins over the failure"
    );

    let metrics = storage.hook_metrics();
    assert_eq!(metrics.invocations, 3);
    assert_eq!(metrics.successes, 2);
    assert_eq!(metrics.failures, 1);
    assert_eq!(metrics.panics, 0);
}

/// A hook that exceeds the configured timeout is reported and degraded without
/// deadlocking or holding the write path for the full hook duration.
#[test]
fn test_pre_anchor_hook_timeout_degrades() {
    use std::time::{Duration, Instant};

    let mut storage = HistoricalStorage::new();
    storage.set_pre_anchor_hook_timeout(Some(Duration::from_millis(50)));
    assert_eq!(
        storage.pre_anchor_hook_timeout(),
        Some(Duration::from_millis(50))
    );

    // Hook sleeps far longer than the timeout, then would return a snapshot.
    let slow: PreAnchorHook = Arc::new(|_entity_type, _entity_id, _timestamp, _properties| {
        std::thread::sleep(Duration::from_millis(1000));
        Ok(Some(5))
    });
    storage.register_pre_node_anchor_hook(slow);

    let start = Instant::now();
    let snapshot = add_first_node_anchor(&mut storage, 1);
    let elapsed = start.elapsed();

    // Degraded: no snapshot stored, anchor still created.
    assert_eq!(snapshot, None);
    // The write path stopped waiting near the timeout, well before the 1s hook.
    assert!(
        elapsed < Duration::from_millis(800),
        "write path should not block for the full hook duration (elapsed={elapsed:?})"
    );

    let metrics = storage.hook_metrics();
    assert_eq!(metrics.invocations, 1);
    assert_eq!(metrics.timeouts, 1);
    assert_eq!(metrics.successes, 0);
}

/// With a generous timeout, a fast hook completes normally on the watchdog path
/// and its snapshot id is stored.
#[test]
fn test_pre_anchor_hook_timeout_fast_hook_succeeds() {
    use std::time::Duration;

    let mut storage = HistoricalStorage::new();
    storage.set_pre_anchor_hook_timeout(Some(Duration::from_secs(5)));

    let fast: PreAnchorHook =
        Arc::new(|_entity_type, _entity_id, _timestamp, _properties| Ok(Some(7)));
    storage.register_pre_node_anchor_hook(fast);

    let snapshot = add_first_node_anchor(&mut storage, 1);
    assert_eq!(snapshot, Some(7));

    let metrics = storage.hook_metrics();
    assert_eq!(metrics.invocations, 1);
    assert_eq!(metrics.successes, 1);
    assert_eq!(metrics.timeouts, 0);
    assert_eq!(metrics.panics, 0);
}

/// Backward compatibility: `register_*` replaces (old single-hook "set THE
/// hook" semantics) while `add_*` appends.
#[test]
fn test_register_replaces_add_appends() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut storage = HistoricalStorage::new();

    let count = Arc::new(AtomicUsize::new(0));
    let make = |c: &Arc<AtomicUsize>, id: usize| -> PreAnchorHook {
        let c = Arc::clone(c);
        Arc::new(move |_e, _i, _t, _p| {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(Some(id))
        })
    };

    // Append two hooks.
    storage.add_pre_node_anchor_hook(make(&count, 1));
    storage.add_pre_node_anchor_hook(make(&count, 2));
    // register replaces both with a single hook.
    storage.register_pre_node_anchor_hook(make(&count, 3));

    let snapshot = add_first_node_anchor(&mut storage, 1);

    // Only the single replacing hook ran (count == 1), and its snapshot won.
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(snapshot, Some(3));
    assert_eq!(storage.hook_metrics().invocations, 1);
}

/// Node and edge hook registries are independent under the multi-hook API.
#[test]
fn test_multi_hook_node_and_edge_independent() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut storage = HistoricalStorage::new();

    let node_calls = Arc::new(AtomicUsize::new(0));
    let edge_calls = Arc::new(AtomicUsize::new(0));

    let nc = Arc::clone(&node_calls);
    storage.add_pre_node_anchor_hook(Arc::new(move |_e, _i, _t, _p| {
        nc.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }));
    let ec = Arc::clone(&edge_calls);
    storage.add_pre_edge_anchor_hook(Arc::new(move |_e, _i, _t, _p| {
        ec.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }));

    add_first_node_anchor(&mut storage, 1);

    let edge_id = EdgeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    storage
        .add_edge_version(
            edge_id,
            VersionId::new(50).unwrap(),
            2000.into(),
            2000.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new().build(),
            false,
        )
        .unwrap();

    assert_eq!(node_calls.load(Ordering::SeqCst), 1);
    assert_eq!(edge_calls.load(Ordering::SeqCst), 1);
}

// ========================================================================
// Tests for Issue #17: Recursion depth limit in version reconstruction
// ========================================================================

use super::{MAX_RECONSTRUCTION_DEPTH, RetentionPolicy};

#[test]
fn test_reconstruction_depth_limit_exceeded_for_nodes() {
    // Create storage with very high anchor interval to force delta creation
    // and cache_size=0 to prevent caching from defeating the depth test
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: (MAX_RECONSTRUCTION_DEPTH * 2) as u32, // Won't create anchors
            max_delta_chain: (MAX_RECONSTRUCTION_DEPTH * 2) as u32,
        },
        RetentionPolicy {
            max_versions_per_entity: MAX_RECONSTRUCTION_DEPTH * 2, // Allow more versions than depth limit
            max_age_ms: i64::MAX,
        },
        0, // Disable cache to test full depth traversal
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create first version (anchor)
    let v0 = VersionId::new(0).unwrap();
    storage
        .add_node_version(
            node_id,
            v0,
            0.into(),
            0.into(),
            label,
            PropertyMapBuilder::new().insert("counter", 0i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Create 100 more versions (deltas) to exceed the depth limit
    // With >= check, depth 100 triggers error, so 100 deltas will exceed
    for i in 1..=MAX_RECONSTRUCTION_DEPTH {
        let vid = VersionId::new(i as u64).unwrap();
        storage
            .add_node_version(
                node_id,
                vid,
                (i as i64 * 1000).into(),
                (i as i64 * 1000).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("counter", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Reconstruction should fail with MaxDepthExceeded
    let last_version_id = VersionId::new(MAX_RECONSTRUCTION_DEPTH as u64).unwrap();
    let result = storage.reconstruct_node_properties(last_version_id);

    assert!(result.is_err(), "Expected MaxDepthExceeded error");
    let err = result.unwrap_err();
    match err {
        crate::core::error::Error::Temporal(
            crate::core::error::TemporalError::MaxDepthExceeded { max_depth, .. },
        ) => {
            assert_eq!(max_depth, MAX_RECONSTRUCTION_DEPTH);
        }
        other => panic!("Expected MaxDepthExceeded error, got: {:?}", other),
    }
}

#[test]
fn test_reconstruction_within_depth_limit_works_for_nodes() {
    // Create storage with high anchor interval to force delta creation
    // and cache_size=0 to test full depth traversal
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 200,
            max_delta_chain: 200,
        },
        RetentionPolicy::default(),
        0, // Disable cache to test full depth traversal
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create first version (anchor)
    let v0 = VersionId::new(0).unwrap();
    storage
        .add_node_version(
            node_id,
            v0,
            0.into(),
            0.into(),
            label,
            PropertyMapBuilder::new().insert("counter", 0i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Create exactly MAX_RECONSTRUCTION_DEPTH - 1 versions (should be within limit)
    // With >= check, 99 deltas means max depth of 99 which is < 100
    for i in 1..MAX_RECONSTRUCTION_DEPTH {
        let vid = VersionId::new(i as u64).unwrap();
        storage
            .add_node_version(
                node_id,
                vid,
                (i as i64 * 1000).into(),
                (i as i64 * 1000).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("counter", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Reconstruction should succeed for version at depth limit
    let last_version_id = VersionId::new((MAX_RECONSTRUCTION_DEPTH - 1) as u64).unwrap();
    let result = storage.reconstruct_node_properties(last_version_id);

    assert!(
        result.is_ok(),
        "Expected successful reconstruction within depth limit"
    );
    let props = result.unwrap();
    assert_eq!(
        props.get("counter").and_then(|v| v.as_int()),
        Some((MAX_RECONSTRUCTION_DEPTH - 1) as i64)
    );
}

#[test]
fn test_reconstruction_depth_limit_exceeded_for_edges() {
    // Create storage with very high anchor interval to force delta creation
    // and cache_size=0 to prevent caching from defeating the depth test
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: (MAX_RECONSTRUCTION_DEPTH * 2) as u32, // Won't create anchors
            max_delta_chain: (MAX_RECONSTRUCTION_DEPTH * 2) as u32,
        },
        RetentionPolicy {
            max_versions_per_entity: MAX_RECONSTRUCTION_DEPTH * 2, // Allow more versions than depth limit
            max_age_ms: i64::MAX,
        },
        0, // Disable cache to test full depth traversal
    );

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(100).unwrap();
    let target = NodeId::new(200).unwrap();
    let label = GLOBAL_INTERNER.intern("TestEdge").unwrap();

    // Create first version (anchor)
    let v0 = VersionId::new(0).unwrap();
    storage
        .add_edge_version(
            edge_id,
            v0,
            0.into(),
            0.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new().insert("weight", 0.0f64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Create 100 more versions (deltas) to exceed the depth limit
    // With >= check, depth 100 triggers error, so 100 deltas will exceed
    for i in 1..=MAX_RECONSTRUCTION_DEPTH {
        let vid = VersionId::new(i as u64).unwrap();
        storage
            .add_edge_version(
                edge_id,
                vid,
                (i as i64 * 1000).into(),
                (i as i64 * 1000).into(),
                label,
                source,
                target,
                PropertyMapBuilder::new().insert("weight", i as f64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Reconstruction should fail with MaxDepthExceeded
    let last_version_id = VersionId::new(MAX_RECONSTRUCTION_DEPTH as u64).unwrap();
    let result = storage.reconstruct_edge_properties(last_version_id);

    assert!(result.is_err(), "Expected MaxDepthExceeded error");
    let err = result.unwrap_err();
    match err {
        crate::core::error::Error::Temporal(
            crate::core::error::TemporalError::MaxDepthExceeded { max_depth, .. },
        ) => {
            assert_eq!(max_depth, MAX_RECONSTRUCTION_DEPTH);
        }
        other => panic!("Expected MaxDepthExceeded error, got: {:?}", other),
    }
}

#[test]
fn test_reconstruction_within_depth_limit_works_for_edges() {
    // Create storage with high anchor interval to force delta creation
    // and cache_size=0 to test full depth traversal
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 200,
            max_delta_chain: 200,
        },
        RetentionPolicy::default(),
        0, // Disable cache to test full depth traversal
    );

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(100).unwrap();
    let target = NodeId::new(200).unwrap();
    let label = GLOBAL_INTERNER.intern("TestEdge").unwrap();

    // Create first version (anchor)
    let v0 = VersionId::new(0).unwrap();
    storage
        .add_edge_version(
            edge_id,
            v0,
            0.into(),
            0.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new().insert("weight", 0.0f64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Create exactly MAX_RECONSTRUCTION_DEPTH - 1 versions (should be within limit)
    // With >= check, 99 deltas means max depth of 99 which is < 100
    for i in 1..MAX_RECONSTRUCTION_DEPTH {
        let vid = VersionId::new(i as u64).unwrap();
        storage
            .add_edge_version(
                edge_id,
                vid,
                (i as i64 * 1000).into(),
                (i as i64 * 1000).into(),
                label,
                source,
                target,
                PropertyMapBuilder::new().insert("weight", i as f64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Reconstruction should succeed for version at depth limit
    let last_version_id = VersionId::new((MAX_RECONSTRUCTION_DEPTH - 1) as u64).unwrap();
    let result = storage.reconstruct_edge_properties(last_version_id);

    assert!(
        result.is_ok(),
        "Expected successful reconstruction within depth limit"
    );
    let props = result.unwrap();
    assert_eq!(
        props.get("weight").and_then(|v| v.as_float()),
        Some((MAX_RECONSTRUCTION_DEPTH - 1) as f64)
    );
}

// ========================================================================
// Improvement #2: Cache Pre-population Tests
// ========================================================================

#[test]
fn test_anchor_properties_are_cached_immediately_for_nodes() {
    // Test that when a node anchor is created, its properties are immediately in the cache
    let mut storage = HistoricalStorage::new();

    let node_id = NodeId::new(1).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();
    let timestamp = 1000.into();
    let props = PropertyMapBuilder::new()
        .insert("name", "Alice")
        .insert("age", 30i64)
        .build();

    // Create the first version (which is always an anchor)
    storage
        .add_node_version(
            node_id,
            version_id,
            timestamp,
            timestamp,
            label,
            props.clone(),
            false, // not a tombstone
        )
        .unwrap();

    // Verify the version is an anchor
    let version = storage.get_node_version(version_id).unwrap();
    assert!(version.is_anchor(), "First version should be an anchor");

    // Check that the anchor properties are in the cache
    // We can verify this by checking the cache stats
    let stats = storage.stats();
    assert_eq!(
        stats.node_cache_entries, 1,
        "Anchor properties should be cached immediately"
    );

    // Verify cache hit by reconstructing properties
    // If cached, this should be O(1) instead of O(N)
    let reconstructed = storage.reconstruct_node_properties(version_id).unwrap();
    assert_eq!(reconstructed, props);

    // Cache entries should still be 1 (cache hit, not a new entry)
    let stats_after = storage.stats();
    assert_eq!(
        stats_after.node_cache_entries, 1,
        "Cache should still have 1 entry after hit"
    );
}

#[test]
fn test_anchor_properties_are_cached_immediately_for_edges() {
    // Test that when an edge anchor is created, its properties are immediately in the cache
    let mut storage = HistoricalStorage::new();

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(100).unwrap();
    let target = NodeId::new(200).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    let timestamp = 1000.into();
    let props = PropertyMapBuilder::new()
        .insert("since", 2020i64)
        .insert("weight", 0.8f64)
        .build();

    // Create the first version (which is always an anchor)
    storage
        .add_edge_version(
            edge_id,
            version_id,
            timestamp,
            timestamp,
            label,
            source,
            target,
            props.clone(),
            false, // not a tombstone
        )
        .unwrap();

    // Verify the version is an anchor
    let version = storage.get_edge_version(version_id).unwrap();
    assert!(version.is_anchor(), "First version should be an anchor");

    // Check that the anchor properties are in the cache
    let stats = storage.stats();
    assert_eq!(
        stats.edge_cache_entries, 1,
        "Anchor properties should be cached immediately"
    );

    // Verify cache hit by reconstructing properties
    let reconstructed = storage.reconstruct_edge_properties(version_id).unwrap();
    assert_eq!(reconstructed, props);

    // Cache entries should still be 1 (cache hit, not a new entry)
    let stats_after = storage.stats();
    assert_eq!(
        stats_after.edge_cache_entries, 1,
        "Cache should still have 1 entry after hit"
    );
}

#[test]
fn test_subsequent_anchors_are_also_cached() {
    // Test that multiple anchors created in sequence are all cached
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 5, // Create anchor every 5 versions
        max_delta_chain: 5,
    });

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Create 11 versions (will create anchors at v0, v5, v10)
    for i in 0..11 {
        let version_id = VersionId::new(i).unwrap();
        let temporal = BiTemporalInterval::current((i as i64 * 1000).into());
        let props = PropertyMapBuilder::new()
            .insert("counter", i as i64)
            .build();

        storage
            .add_node_version(
                node_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();
    }

    // Verify we have 3 anchors
    let stats = storage.stats();
    assert_eq!(stats.node_anchor_count, 3, "Should have 3 anchors");
    assert_eq!(stats.node_delta_count, 8, "Should have 8 deltas");

    // All 11 versions should be in cache (3 anchors + 8 deltas populated during reconstruction)
    // Actually, initially only anchors should be pre-cached, deltas are cached on-demand
    // So we should have at least 3 entries (the anchors)
    assert!(
        stats.node_cache_entries >= 3,
        "At least the 3 anchors should be cached, got {}",
        stats.node_cache_entries
    );

    // Verify anchor versions are cached
    let anchor_v0 = VersionId::new(0).unwrap();
    let anchor_v5 = VersionId::new(5).unwrap();
    let anchor_v10 = VersionId::new(10).unwrap();

    // These should be cache hits
    storage.reconstruct_node_properties(anchor_v0).unwrap();
    storage.reconstruct_node_properties(anchor_v5).unwrap();
    storage.reconstruct_node_properties(anchor_v10).unwrap();
}

// ========================================================================
// Improvement #1: Anchor-Based Caching Tests
// ========================================================================

#[test]
fn test_anchor_cache_survives_delta_cache_pressure() {
    // Test that anchors remain cached even when delta versions fill up the regular cache
    // Use a small cache size to force evictions
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 10, // Anchor every 10 versions
            max_delta_chain: 10,
        },
        RetentionPolicy::default(),
        5, // Very small cache - only 5 entries
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Create 21 versions (anchors at v0, v10, v20 + 18 deltas)
    for i in 0..21 {
        let version_id = VersionId::new(i).unwrap();
        let temporal = BiTemporalInterval::current((i as i64 * 1000).into());
        let props = PropertyMapBuilder::new()
            .insert("counter", i as i64)
            .build();

        storage
            .add_node_version(
                node_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();
    }

    // Verify we have 3 anchors
    let stats = storage.stats();
    assert_eq!(stats.node_anchor_count, 3, "Should have 3 anchors");

    // Access many delta versions to create cache pressure
    // This should fill up the small cache (5 entries) with delta reconstructions
    for i in 1..10 {
        let version_id = VersionId::new(i).unwrap();
        storage.reconstruct_node_properties(version_id).unwrap();
    }

    // Despite cache pressure, all anchors should still be quickly accessible
    // because they're in the dedicated anchor cache
    let anchor_v0 = VersionId::new(0).unwrap();
    let anchor_v10 = VersionId::new(10).unwrap();
    let anchor_v20 = VersionId::new(20).unwrap();

    // These should be fast cache hits from the anchor cache
    let props0 = storage.reconstruct_node_properties(anchor_v0).unwrap();
    let props10 = storage.reconstruct_node_properties(anchor_v10).unwrap();
    let props20 = storage.reconstruct_node_properties(anchor_v20).unwrap();

    assert_eq!(
        props0.get("counter").and_then(|v| v.as_int()),
        Some(0.into())
    );
    assert_eq!(
        props10.get("counter").and_then(|v| v.as_int()),
        Some(10.into())
    );
    assert_eq!(
        props20.get("counter").and_then(|v| v.as_int()),
        Some(20.into())
    );
}

#[test]
fn test_delta_reconstruction_uses_anchor_cache() {
    // Test that delta reconstruction benefits from the anchor cache
    // When reconstructing a delta, we should use the cached anchor as the base
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 5,
            max_delta_chain: 5,
        },
        RetentionPolicy::default(),
        100, // Reasonable cache size
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Document").unwrap();

    // Create 8 versions (anchor at v0, v5, deltas at v1-v4, v6-v7)
    for i in 0..8 {
        let version_id = VersionId::new(i).unwrap();
        let temporal = BiTemporalInterval::current((i as i64 * 1000).into());
        let props = PropertyMapBuilder::new()
            .insert("version", i as i64)
            .insert("data", format!("content_{}", i))
            .build();

        storage
            .add_node_version(
                node_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();
    }

    // Verify anchors are in cache
    let stats = storage.stats();
    assert!(
        stats.node_cache_entries >= 2,
        "Both anchors should be cached"
    );

    // Reconstruct a delta version (v7) - should use anchor cache for v5
    let v7 = VersionId::new(7).unwrap();
    let props = storage.reconstruct_node_properties(v7).unwrap();
    assert_eq!(
        props.get("version").and_then(|v| v.as_int()),
        Some(7.into())
    );
    assert_eq!(
        props.get("data").and_then(|v| v.as_str()),
        Some("content_7")
    );
}

#[test]
fn test_anchor_cache_improves_multi_version_reconstruction() {
    // Test that multiple delta versions can reuse the same cached anchor
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 5,
        max_delta_chain: 5,
    });

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Entity").unwrap();

    // Create 10 versions (anchors at v0, v5)
    for i in 0..10 {
        let version_id = VersionId::new(i).unwrap();
        let temporal = BiTemporalInterval::current((i as i64 * 1000).into());
        let props = PropertyMapBuilder::new().insert("value", i as i64).build();

        storage
            .add_node_version(
                node_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();
    }

    // Reconstruct all deltas between v5 and v9
    // All should benefit from the cached anchor at v5
    for i in 6..10 {
        let version_id = VersionId::new(i).unwrap();
        let props = storage.reconstruct_node_properties(version_id).unwrap();
        assert_eq!(props.get("value").and_then(|v| v.as_int()), Some(i as i64));
    }

    // All reconstructions should have succeeded efficiently using the anchor cache
    let stats = storage.stats();
    assert_eq!(stats.node_anchor_count, 2, "Should have 2 anchors");
}

#[test]
fn test_anchor_cache_size_calculation() {
    // Test that anchor cache is properly sized relative to main cache

    // Small cache: 100 entries -> anchor cache should be max(100/5, 100) = 100
    let storage_small = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig::default(),
        RetentionPolicy::default(),
        100,
    );
    // We can't directly access cache capacity, but we can verify it works correctly
    // by checking that anchors are cached even with small cache
    assert_eq!(storage_small.node_property_cache.len(), 0);

    // Medium cache: 1000 entries -> anchor cache should be max(1000/5, 100) = 200
    let storage_medium = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig::default(),
        RetentionPolicy::default(),
        1000,
    );
    assert_eq!(storage_medium.node_property_cache.len(), 0);

    // Large cache: 10000 entries -> anchor cache should be max(10000/5, 100) = 2000
    let storage_large = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig::default(),
        RetentionPolicy::default(),
        10000,
    );
    assert_eq!(storage_large.node_property_cache.len(), 0);

    // Very small cache: 10 entries -> anchor cache should be max(10/5, 100) = 100 (minimum)
    let storage_tiny = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig::default(),
        RetentionPolicy::default(),
        10,
    );
    assert_eq!(storage_tiny.node_property_cache.len(), 0);
}

// ========================================================================
// Improvement #3: Adaptive Cache Sizing Tests
// ========================================================================

#[test]
fn test_should_resize_cache_recommends_growth_on_low_hit_rate() {
    // Test that `should_resize_cache` recommends resizing when hit rate is low.
    // Start with a very small cache to force low hit rate
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 10,
            max_delta_chain: 10,
        },
        RetentionPolicy::default(),
        10, // Very small cache - will have low hit rate
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Create many versions to stress the cache
    for i in 0..50 {
        let version_id = VersionId::new(i).unwrap();
        let temporal = BiTemporalInterval::current((i as i64 * 1000).into());
        let props = PropertyMapBuilder::new()
            .insert("counter", i as i64)
            .insert("data", format!("value_{}", i))
            .build();

        storage
            .add_node_version(
                node_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();
    }

    // Access many different versions to create cache misses
    for i in 0..50 {
        let version_id = VersionId::new(i).unwrap();
        storage.reconstruct_node_properties(version_id).unwrap();
    }

    // Check cache metrics
    let metrics = storage.cache_metrics();
    assert!(
        metrics.total_operations() > 0,
        "Should have cache operations"
    );

    // Check if adaptive resizing recommends increasing cache size
    // With only 10 cache slots and 50 versions, hit rate should be low
    let resize_recommendation = storage.should_resize_cache(0.8, 10);
    assert!(
        resize_recommendation.is_some(),
        "should_resize_cache should recommend resizing with low hit rate"
    );

    let hit_rate = resize_recommendation.unwrap();
    assert!(hit_rate < 0.8, "Hit rate should be below the threshold");

    println!(
        "Cache hit rate {:.2}% is below threshold, resize recommended",
        hit_rate * 100.0
    );

    let stats = storage.stats();
    assert!(
        stats.node_cache_entries > 0,
        "Cache should have some entries"
    );
}

#[test]
fn test_cache_hit_rate_tracking() {
    // Test that we can track cache hit rate metrics
    let mut storage = HistoricalStorage::with_config(AnchorConfig::default());

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("TestNode").unwrap();

    // Create some versions
    for i in 0..20 {
        let version_id = VersionId::new(i).unwrap();
        let temporal = BiTemporalInterval::current((i as i64 * 1000).into());
        let props = PropertyMapBuilder::new().insert("value", i as i64).build();

        storage
            .add_node_version(
                node_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();
    }

    // Access the same version multiple times (should be cache hits after first)
    let v5 = VersionId::new(5).unwrap();
    for _ in 0..10 {
        storage.reconstruct_node_properties(v5).unwrap();
    }

    // The cache should have entries
    let stats = storage.stats();
    assert!(
        stats.node_cache_entries > 0,
        "Cache should have entries after reconstruction"
    );

    // Check hit rate - with repeated access, should be reasonable
    let hit_rate = storage.cache_hit_rate();
    assert!(hit_rate.is_some(), "Should have cache hit rate data");

    // Note (Issue #211): After switching to iterative reconstruction, the cache
    // behavior changed. The recursive implementation cached intermediate versions
    // during reconstruction, while the iterative approach only caches the final
    // result. This reduces memory allocations (O(1) vs O(anchor_interval)) at
    // the cost of slightly lower cache hit rates in some scenarios.
    //
    // With 20 versions created (anchors at 0, 10, 20) and 10 accesses to v5:
    // - Version creation triggers ~18 reconstructions (for deltas)
    // - First access to v5: 1 reconstruction (if not already cached)
    // - Next 9 accesses to v5: 9 cache hits
    // - Expected hit rate: ~9/28 = ~32% (lower bound)
    //
    // The exact hit rate depends on which versions were cached during creation.
    // We verify it's reasonable (>20%) rather than the old >50% expectation.
    assert!(
        hit_rate.unwrap() > 0.20,
        "Hit rate should be > 20% with some repeated access, got {:?}",
        hit_rate
    );
}

#[test]
fn test_cache_resize_maintains_correctness() {
    // Test that even if cache is resized, data correctness is maintained
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 5,
            max_delta_chain: 5,
        },
        RetentionPolicy::default(),
        100,
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Data").unwrap();

    // Create test data
    for i in 0..30 {
        let version_id = VersionId::new(i).unwrap();
        let temporal = BiTemporalInterval::current((i as i64 * 1000).into());
        let props = PropertyMapBuilder::new()
            .insert("id", i as i64)
            .insert("name", format!("item_{}", i))
            .build();

        storage
            .add_node_version(
                node_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();
    }

    // Verify all data is correct regardless of cache state
    for i in 0..30 {
        let version_id = VersionId::new(i).unwrap();
        let props = storage.reconstruct_node_properties(version_id).unwrap();
        assert_eq!(props.get("id").and_then(|v| v.as_int()), Some(i as i64));
        assert_eq!(
            props.get("name").and_then(|v| v.as_str()),
            Some(format!("item_{}", i).as_str())
        );
    }

    // Verify cache metrics are being tracked
    let metrics = storage.cache_metrics();
    assert!(
        metrics.total_operations() > 0,
        "Should have cache operations"
    );

    // With good cache size (100) and sequential access, hit rate should be decent
    if let Some(hit_rate) = storage.cache_hit_rate() {
        println!("Cache hit rate: {:.2}%", hit_rate * 100.0);
    }
}

// ============================================================
// Edge Version Chain Tests (TDD for Issue #345)
// ============================================================
// These tests ensure edge version functionality has parity with
// node version functionality before refactoring to eliminate
// duplicate code.

#[test]
fn test_edge_version_chain() {
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    });

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // Create 5 versions
    let mut version_ids = Vec::new();
    for i in 0..5 {
        let version_id = VersionId::new(100 + i).unwrap();
        let temporal = BiTemporalInterval::current((1000 + (i as i64) * 100).into());
        let props = PropertyMapBuilder::new()
            .insert("weight", i as i64)
            .insert("since", "2024")
            .build();

        storage
            .add_edge_version(
                edge_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                source,
                target,
                props,
                false, // not a tombstone
            )
            .unwrap();

        version_ids.push(version_id);
    }

    // Check version types - should follow same pattern as nodes:
    // v0: anchor (first)
    // v1: delta
    // v2: delta
    // v3: anchor (interval = 3)
    // v4: delta

    assert!(
        storage
            .get_edge_version(version_ids[0])
            .unwrap()
            .is_anchor()
    );
    assert!(storage.get_edge_version(version_ids[1]).unwrap().is_delta());
    assert!(storage.get_edge_version(version_ids[2]).unwrap().is_delta());
    assert!(
        storage
            .get_edge_version(version_ids[3])
            .unwrap()
            .is_anchor()
    );
    assert!(storage.get_edge_version(version_ids[4]).unwrap().is_delta());
}

#[test]
fn test_edge_property_reconstruction() {
    let mut storage = HistoricalStorage::new();

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // Version 1: weight=10, since=2020
    let v1 = VersionId::new(1).unwrap();
    storage
        .add_edge_version(
            edge_id,
            v1,
            1000.into(),
            1000.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new()
                .insert("weight", 10i64)
                .insert("since", "2020")
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 2: weight=20, since=2020 (delta - only weight changes)
    let v2 = VersionId::new(2).unwrap();
    storage
        .add_edge_version(
            edge_id,
            v2,
            2000.into(),
            2000.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new()
                .insert("weight", 20i64)
                .insert("since", "2020")
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Reconstruct v1 properties
    let props_v1 = storage.reconstruct_edge_properties(v1).unwrap();
    assert_eq!(props_v1.get("weight").and_then(|v| v.as_int()), Some(10));
    assert_eq!(props_v1.get("since").and_then(|v| v.as_str()), Some("2020"));

    // Reconstruct v2 properties
    let props_v2 = storage.reconstruct_edge_properties(v2).unwrap();
    assert_eq!(props_v2.get("weight").and_then(|v| v.as_int()), Some(20));
    assert_eq!(props_v2.get("since").and_then(|v| v.as_str()), Some("2020"));
}

#[test]
fn test_edge_find_version_at_time() {
    let mut storage = HistoricalStorage::new();

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // Create versions at different times
    let v1 = VersionId::new(1).unwrap();
    let v2 = VersionId::new(2).unwrap();
    let v3 = VersionId::new(3).unwrap();

    storage
        .add_edge_version(
            edge_id,
            v1,
            0.into(),
            0.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new().insert("weight", 10i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    storage
        .add_edge_version(
            edge_id,
            v2,
            1000.into(),
            0.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new().insert("weight", 20i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    storage
        .add_edge_version(
            edge_id,
            v3,
            2000.into(),
            0.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new().insert("weight", 30i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Query at different times
    assert_eq!(
        storage.find_edge_version_at_time(edge_id, 500.into(), 100.into()),
        Some(v1)
    );
    assert_eq!(
        storage.find_edge_version_at_time(edge_id, 1500.into(), 100.into()),
        Some(v2)
    );
    assert_eq!(
        storage.find_edge_version_at_time(edge_id, 2500.into(), 100.into()),
        Some(v3)
    );
}

#[test]
fn test_edge_version_chain_links() {
    // Test that version chains are properly linked (prev/next)
    let mut storage = HistoricalStorage::new();

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    let v1 = VersionId::new(1).unwrap();
    let v2 = VersionId::new(2).unwrap();
    let v3 = VersionId::new(3).unwrap();

    for (i, vid) in [v1, v2, v3].iter().enumerate() {
        storage
            .add_edge_version(
                edge_id,
                *vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                source,
                target,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Check linking
    let version1 = storage.get_edge_version(v1).unwrap();
    assert_eq!(version1.prev_version, None);
    assert_eq!(version1.next_version, Some(v2));

    let version2 = storage.get_edge_version(v2).unwrap();
    assert_eq!(version2.prev_version, Some(v1));
    assert_eq!(version2.next_version, Some(v3));

    let version3 = storage.get_edge_version(v3).unwrap();
    assert_eq!(version3.prev_version, Some(v2));
    assert_eq!(version3.next_version, None);
}

#[test]
fn test_first_edge_version_is_anchor() {
    let mut storage = HistoricalStorage::new();

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    let temporal = BiTemporalInterval::current(1000.into());
    let props = PropertyMapBuilder::new().insert("weight", 5i64).build();

    storage
        .add_edge_version(
            edge_id,
            version_id,
            temporal.valid_time().start(),
            temporal.transaction_time().start(),
            label,
            source,
            target,
            props,
            false, // not a tombstone
        )
        .unwrap();

    // First version should always be an anchor
    let version = storage.get_edge_version(version_id).unwrap();
    assert!(version.is_anchor());
    assert_eq!(version.edge_id, edge_id);
    assert_eq!(version.prev_version, None);
    assert_eq!(version.source, source);
    assert_eq!(version.target, target);
}

#[test]
fn test_independent_node_edge_anchor_intervals() {
    // Verify that node and edge version chains maintain separate anchor counters
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    });

    let node_id = NodeId::new(1).unwrap();
    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(2).unwrap();
    let target = NodeId::new(3).unwrap();
    let node_label = GLOBAL_INTERNER.intern("Person").unwrap();
    let edge_label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // Create interleaved node and edge versions to ensure they don't interfere
    // Node pattern: anchor(0), delta(1), delta(2), anchor(3), delta(4)
    // Edge pattern: anchor(100), delta(101), delta(102), anchor(103), delta(104)
    let mut node_version_ids = Vec::new();
    let mut edge_version_ids = Vec::new();

    for i in 0..5 {
        // Add node version
        let node_vid = VersionId::new(i).unwrap();
        storage
            .add_node_version(
                node_id,
                node_vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                node_label,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
        node_version_ids.push(node_vid);

        // Add edge version (interleaved)
        let edge_vid = VersionId::new(100 + i).unwrap();
        storage
            .add_edge_version(
                edge_id,
                edge_vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                edge_label,
                source,
                target,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
        edge_version_ids.push(edge_vid);
    }

    // Verify node version pattern: anchor, delta, delta, anchor, delta
    assert!(
        storage
            .get_node_version(node_version_ids[0])
            .unwrap()
            .is_anchor()
    );
    assert!(
        storage
            .get_node_version(node_version_ids[1])
            .unwrap()
            .is_delta()
    );
    assert!(
        storage
            .get_node_version(node_version_ids[2])
            .unwrap()
            .is_delta()
    );
    assert!(
        storage
            .get_node_version(node_version_ids[3])
            .unwrap()
            .is_anchor()
    );
    assert!(
        storage
            .get_node_version(node_version_ids[4])
            .unwrap()
            .is_delta()
    );

    // Verify edge version pattern is the same (independent counter)
    assert!(
        storage
            .get_edge_version(edge_version_ids[0])
            .unwrap()
            .is_anchor()
    );
    assert!(
        storage
            .get_edge_version(edge_version_ids[1])
            .unwrap()
            .is_delta()
    );
    assert!(
        storage
            .get_edge_version(edge_version_ids[2])
            .unwrap()
            .is_delta()
    );
    assert!(
        storage
            .get_edge_version(edge_version_ids[3])
            .unwrap()
            .is_anchor()
    );
    assert!(
        storage
            .get_edge_version(edge_version_ids[4])
            .unwrap()
            .is_delta()
    );
}

#[test]
fn test_count_versions_since_anchor_generic() {
    // Direct test of the generic count_versions_since_anchor helper
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    });

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Create anchor(0), delta(1), delta(2)
    let mut version_ids = Vec::new();
    for i in 0..3 {
        let vid = VersionId::new(i).unwrap();
        storage
            .add_node_version(
                node_id,
                vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
        version_ids.push(vid);
    }

    // Test counting from version 2 (delta) - should find 2 deltas before anchor
    assert_eq!(storage.count_versions_since_anchor_node(version_ids[2]), 2);

    // Test counting from version 1 (delta) - should find 1 delta before anchor
    assert_eq!(storage.count_versions_since_anchor_node(version_ids[1]), 1);

    // Test counting from version 0 (anchor) - should find 0 deltas
    assert_eq!(storage.count_versions_since_anchor_node(version_ids[0]), 0);

    // Create more versions to get anchor(3), delta(4)
    for i in 3..5 {
        let vid = VersionId::new(i).unwrap();
        storage
            .add_node_version(
                node_id,
                vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
        version_ids.push(vid);
    }

    // Test counting from version 4 (delta after new anchor) - should find 1 delta
    assert_eq!(storage.count_versions_since_anchor_node(version_ids[4]), 1);

    // Test counting from version 3 (new anchor) - should find 0 deltas
    assert_eq!(storage.count_versions_since_anchor_node(version_ids[3]), 0);
}

#[test]
fn test_version_counter_cache() {
    // Test that the version counter cache correctly tracks versions since last anchor
    // This test verifies the fix for issue #208
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 5,
        max_delta_chain: 10,
    });

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Add first version - should be anchor, counter should be 0
    let v0 = VersionId::new(0).unwrap();
    storage
        .add_node_version(
            node_id,
            v0,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().insert("version", 0i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Verify first version is an anchor
    assert!(storage.get_node_version(v0).unwrap().is_anchor());

    // Add 4 more versions (deltas)
    for i in 1..5 {
        let vid = VersionId::new(i).unwrap();
        storage
            .add_node_version(
                node_id,
                vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();

        // All should be deltas
        assert!(storage.get_node_version(vid).unwrap().is_delta());
    }

    // Add 5th version - should trigger anchor creation (interval = 5)
    let v5 = VersionId::new(5).unwrap();
    storage
        .add_node_version(
            node_id,
            v5,
            1600.into(),
            1600.into(),
            label,
            PropertyMapBuilder::new().insert("version", 5i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 5 should be an anchor
    assert!(storage.get_node_version(v5).unwrap().is_anchor());

    // Add one more version - should be delta, counter should reset
    let v6 = VersionId::new(6).unwrap();
    storage
        .add_node_version(
            node_id,
            v6,
            1700.into(),
            1700.into(),
            label,
            PropertyMapBuilder::new().insert("version", 6i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 6 should be a delta
    assert!(storage.get_node_version(v6).unwrap().is_delta());

    // Test with multiple entities to ensure counters are independent
    let node_id2 = NodeId::new(2).unwrap();

    // Add versions to second entity
    for i in 0..3 {
        let vid = VersionId::new(100 + i).unwrap();
        storage
            .add_node_version(
                node_id2,
                vid,
                (2000 + (i as i64) * 100).into(),
                (2000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // First version of second entity should be anchor
    assert!(
        storage
            .get_node_version(VersionId::new(100).unwrap())
            .unwrap()
            .is_anchor()
    );
    // Next two should be deltas
    assert!(
        storage
            .get_node_version(VersionId::new(101).unwrap())
            .unwrap()
            .is_delta()
    );
    assert!(
        storage
            .get_node_version(VersionId::new(102).unwrap())
            .unwrap()
            .is_delta()
    );
}

#[test]
fn test_edge_version_counter_cache() {
    // Test that the version counter cache works for edges too
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    });

    let edge_id = EdgeId::new(1).unwrap();
    let from = NodeId::new(1).unwrap();
    let to = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // Add first version - should be anchor
    let v0 = VersionId::new(0).unwrap();
    storage
        .add_edge_version(
            edge_id,
            v0,
            1000.into(),
            1000.into(),
            label,
            from,
            to,
            PropertyMapBuilder::new().insert("version", 0i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    assert!(storage.get_edge_version(v0).unwrap().is_anchor());

    // Add 2 deltas
    for i in 1..3 {
        let vid = VersionId::new(i).unwrap();
        storage
            .add_edge_version(
                edge_id,
                vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                from,
                to,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();

        assert!(storage.get_edge_version(vid).unwrap().is_delta());
    }

    // Add 3rd delta - should trigger anchor creation (interval = 3)
    let v3 = VersionId::new(3).unwrap();
    storage
        .add_edge_version(
            edge_id,
            v3,
            1400.into(),
            1400.into(),
            label,
            from,
            to,
            PropertyMapBuilder::new().insert("version", 3i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 3 should be an anchor
    assert!(storage.get_edge_version(v3).unwrap().is_anchor());
}

#[test]
fn test_counter_cache_rebuilt_after_persistence_restore() {
    // Test for issue #208 fix: Verify that counter cache is correctly
    // rebuilt when loading from persistence
    let config = AnchorConfig {
        anchor_interval: 5,
        max_delta_chain: 10,
    };
    let mut original = HistoricalStorage::with_config(config.clone());

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Create 7 versions: anchor(0), delta(1), delta(2), delta(3), delta(4), anchor(5), delta(6)
    for i in 0..7 {
        let vid = VersionId::new(i).unwrap();
        original
            .add_node_version(
                node_id,
                vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Verify anchor pattern before persistence
    assert!(
        original
            .get_node_version(VersionId::new(0).unwrap())
            .unwrap()
            .is_anchor()
    ); // anchor
    assert!(
        original
            .get_node_version(VersionId::new(1).unwrap())
            .unwrap()
            .is_delta()
    ); // delta
    assert!(
        original
            .get_node_version(VersionId::new(4).unwrap())
            .unwrap()
            .is_delta()
    ); // delta
    assert!(
        original
            .get_node_version(VersionId::new(5).unwrap())
            .unwrap()
            .is_anchor()
    ); // anchor
    assert!(
        original
            .get_node_version(VersionId::new(6).unwrap())
            .unwrap()
            .is_delta()
    ); // delta

    // Extract all versions to simulate persistence save/load
    let saved_versions: Vec<NodeVersion> = original.node_versions.values().cloned().collect();

    // Create new storage and restore versions (simulating load from disk)
    let mut restored = HistoricalStorage::with_config(config);

    // Insert all restored versions
    for version in saved_versions {
        restored.insert_restored_node_version(version).unwrap();
    }

    // Rebuild version chains and counter cache
    restored.rebuild_version_chains();

    // Verify counter cache was rebuilt correctly
    // After version 6 (delta), counter should be 1 (one delta since last anchor at v5)
    let counter = restored
        .node_versions_since_anchor
        .get(&node_id)
        .copied()
        .unwrap_or(0);
    assert_eq!(
        counter, 1,
        "Counter should be 1 after version 6 (one delta since anchor at v5)"
    );

    // Now add more versions and verify anchor/delta pattern continues correctly
    // Add versions 7-8 (should be deltas)
    for i in 7..9 {
        let vid = VersionId::new(i).unwrap();
        restored
            .add_node_version(
                node_id,
                vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();

        // Both should be deltas
        assert!(
            restored.get_node_version(vid).unwrap().is_delta(),
            "Version {} should be delta",
            i
        );
    }

    // Add version 9 (should be delta, counter becomes 4)
    let v9 = VersionId::new(9).unwrap();
    restored
        .add_node_version(
            node_id,
            v9,
            1900.into(),
            1900.into(),
            label,
            PropertyMapBuilder::new().insert("version", 9i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    assert!(
        restored.get_node_version(v9).unwrap().is_delta(),
        "Version 9 should be delta"
    );

    // Add version 10 - should trigger anchor (5 deltas since v5: v6,v7,v8,v9,v10)
    let v10 = VersionId::new(10).unwrap();
    restored
        .add_node_version(
            node_id,
            v10,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new().insert("version", 10i64).build(),
            false, // not a tombstone
        )
        .unwrap();

    // Version 10 should be an anchor
    assert!(
        restored.get_node_version(v10).unwrap().is_anchor(),
        "Version 10 should be anchor after 5 deltas"
    );

    // Verify counter was reset to 0
    let counter_after = restored
        .node_versions_since_anchor
        .get(&node_id)
        .copied()
        .unwrap_or(0);
    assert_eq!(
        counter_after, 0,
        "Counter should be reset to 0 after creating anchor"
    );
}

#[test]
fn test_edge_counter_cache_rebuilt_after_restore() {
    // Test edge counter cache rebuilding after persistence restore
    let config = AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    };
    let mut original = HistoricalStorage::with_config(config.clone());

    let edge_id = EdgeId::new(1).unwrap();
    let from = NodeId::new(1).unwrap();
    let to = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // Create 5 versions: anchor(0), delta(1), delta(2), anchor(3), delta(4)
    for i in 0..5 {
        let vid = VersionId::new(i).unwrap();
        original
            .add_edge_version(
                edge_id,
                vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                from,
                to,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Extract versions
    let saved_versions: Vec<EdgeVersion> = original.edge_versions.values().cloned().collect();

    // Restore to new storage
    let mut restored = HistoricalStorage::with_config(config);
    for version in saved_versions {
        restored.insert_restored_edge_version(version).unwrap();
    }
    restored.rebuild_version_chains();

    // Verify counter is 1 (version 4 is delta after anchor 3)
    let counter = restored
        .edge_versions_since_anchor
        .get(&edge_id)
        .copied()
        .unwrap_or(0);
    assert_eq!(counter, 1, "Edge counter should be 1 after restore");

    // Add two more versions - should create anchor at v6
    for i in 5..7 {
        let vid = VersionId::new(i).unwrap();
        restored
            .add_edge_version(
                edge_id,
                vid,
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                from,
                to,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // v5 should be delta, v6 should be anchor
    assert!(
        restored
            .get_edge_version(VersionId::new(5).unwrap())
            .unwrap()
            .is_delta()
    );
    assert!(
        restored
            .get_edge_version(VersionId::new(6).unwrap())
            .unwrap()
            .is_anchor()
    );
}

// ========================================================================
// Tests for Issue #211: Iterative reconstruction (TDD)
// ========================================================================

#[test]
fn test_node_reconstruction_with_long_delta_chain() {
    // Test reconstruction with a long chain of deltas to verify
    // iterative approach handles deep chains efficiently
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 50, // Won't create anchors until 50 versions
            max_delta_chain: 50,
        },
        RetentionPolicy::default(),
        0, // Disable cache to test full reconstruction
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("TestNode").unwrap();

    // Create anchor version with initial properties
    let v0 = VersionId::new(0).unwrap();
    storage
        .add_node_version(
            node_id,
            v0,
            0.into(),
            0.into(),
            label,
            PropertyMapBuilder::new()
                .insert("counter", 0i64)
                .insert("name", "test")
                .insert("active", true)
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Create 40 delta versions, each modifying different properties
    // Track current values to build complete property maps
    let mut current_name = "test".to_string();
    let mut current_active = true;

    for i in 1..=40 {
        let vid = VersionId::new(i).unwrap();

        // Update properties based on iteration
        if i % 3 == 0 {
            current_name = format!("test_{}", i);
        }
        if i % 5 == 0 {
            current_active = i % 2 == 0;
        }

        // Always include all properties (complete state, not just deltas)
        storage
            .add_node_version(
                node_id,
                vid,
                (i as i64 * 1000).into(),
                (i as i64 * 1000).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("counter", i as i64)
                    .insert("name", current_name.clone())
                    .insert("active", current_active)
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Reconstruct the final version (should traverse 40 deltas)
    let final_version = VersionId::new(40).unwrap();
    let props = storage.reconstruct_node_properties(final_version).unwrap();

    // Verify final properties are correct
    assert_eq!(props.get("counter").and_then(|v| v.as_int()), Some(40));
    assert_eq!(props.get("name").and_then(|v| v.as_str()), Some("test_39")); // Last change at v39 (39 % 3 == 0)
    assert_eq!(props.get("active").and_then(|v| v.as_bool()), Some(true)); // Last change at v40 (40 % 5 == 0, 40 % 2 == 0 = true)

    // Verify reconstruction happened (cache was disabled)
    let metrics = storage.cache_metrics();
    assert!(
        metrics.full_reconstructions > 0,
        "Should have performed reconstruction"
    );
}

#[test]
fn test_edge_reconstruction_with_long_delta_chain() {
    // Test edge reconstruction with a long chain of deltas
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 50,
            max_delta_chain: 50,
        },
        RetentionPolicy::default(),
        0, // Disable cache to test full reconstruction
    );

    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(100).unwrap();
    let target = NodeId::new(200).unwrap();
    let label = GLOBAL_INTERNER.intern("TestEdge").unwrap();

    // Create anchor version
    let v0 = VersionId::new(0).unwrap();
    storage
        .add_edge_version(
            edge_id,
            v0,
            0.into(),
            0.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new()
                .insert("weight", 0.0f64)
                .insert("type", "initial")
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Create 40 delta versions
    let mut current_type = "initial".to_string();

    for i in 1..=40 {
        let vid = VersionId::new(i).unwrap();

        if i % 7 == 0 {
            current_type = format!("updated_{}", i);
        }

        storage
            .add_edge_version(
                edge_id,
                vid,
                (i as i64 * 1000).into(),
                (i as i64 * 1000).into(),
                label,
                source,
                target,
                PropertyMapBuilder::new()
                    .insert("weight", i as f64)
                    .insert("type", current_type.clone())
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Reconstruct the final version
    let final_version = VersionId::new(40).unwrap();
    let props = storage.reconstruct_edge_properties(final_version).unwrap();

    // Verify final properties
    assert_eq!(props.get("weight").and_then(|v| v.as_float()), Some(40.0));
    assert_eq!(
        props.get("type").and_then(|v| v.as_str()),
        Some("updated_35")
    ); // Last change at v35 (35 % 7 == 0)

    // Verify reconstruction happened
    let metrics = storage.cache_metrics();
    assert!(
        metrics.full_reconstructions > 0,
        "Should have performed reconstruction"
    );
}

#[test]
fn test_reconstruction_correctness_at_various_depths() {
    // Test that reconstruction is correct at various depths in the delta chain
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 20,
            max_delta_chain: 20,
        },
        RetentionPolicy::default(),
        0, // Disable cache
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create versions with predictable property values
    for i in 0..15 {
        let vid = VersionId::new(i).unwrap();
        storage
            .add_node_version(
                node_id,
                vid,
                (i as i64 * 1000).into(),
                (i as i64 * 1000).into(),
                label,
                PropertyMapBuilder::new()
                    .insert("version", i as i64)
                    .insert("sum", (i * (i + 1) / 2) as i64) // Cumulative sum for verification
                    .build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Verify reconstruction at various depths
    for i in 0..15 {
        let vid = VersionId::new(i).unwrap();
        let props = storage.reconstruct_node_properties(vid).unwrap();

        assert_eq!(
            props.get("version").and_then(|v| v.as_int()),
            Some(i as i64),
            "Version {} should have version={}",
            i,
            i
        );
        assert_eq!(
            props.get("sum").and_then(|v| v.as_int()),
            Some((i * (i + 1) / 2) as i64),
            "Version {} should have correct sum",
            i
        );
    }
}

#[test]
fn test_reconstruction_with_property_deletion() {
    // Test that reconstruction correctly handles property deletions in deltas
    let mut storage = HistoricalStorage::with_config_retention_and_cache_size(
        AnchorConfig {
            anchor_interval: 10,
            max_delta_chain: 10,
        },
        RetentionPolicy::default(),
        0,
    );

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // v0: Create node with multiple properties
    storage
        .add_node_version(
            node_id,
            VersionId::new(0).unwrap(),
            0.into(),
            0.into(),
            label,
            PropertyMapBuilder::new()
                .insert("a", "value_a")
                .insert("b", "value_b")
                .insert("c", "value_c")
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // v1: Update node, remove property 'b' (delta will have a, c but not b)
    storage
        .add_node_version(
            node_id,
            VersionId::new(1).unwrap(),
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("a", "value_a")
                .insert("c", "new_value_c")
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // v2: Add property 'd', keep a, c
    storage
        .add_node_version(
            node_id,
            VersionId::new(2).unwrap(),
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new()
                .insert("a", "value_a")
                .insert("c", "new_value_c")
                .insert("d", "value_d")
                .build(),
            false, // not a tombstone
        )
        .unwrap();

    // Verify v1 doesn't have 'b'
    let props_v1 = storage
        .reconstruct_node_properties(VersionId::new(1).unwrap())
        .unwrap();
    assert!(
        props_v1.get("b").is_none(),
        "v1 should not have property 'b'"
    );
    assert_eq!(
        props_v1.get("c").and_then(|v| v.as_str()),
        Some("new_value_c")
    );

    // Verify v2 has correct properties
    let props_v2 = storage
        .reconstruct_node_properties(VersionId::new(2).unwrap())
        .unwrap();
    assert!(
        props_v2.get("b").is_none(),
        "v2 should not have property 'b'"
    );
    assert_eq!(props_v2.get("d").and_then(|v| v.as_str()), Some("value_d"));
}

#[test]
fn test_reconstruction_with_anchor_interval() {
    // Test that reconstruction works correctly across anchor boundaries
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 5,
        max_delta_chain: 5,
    });

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create 12 versions (anchors at 0, 5, 10)
    for i in 0..12 {
        let vid = VersionId::new(i).unwrap();
        storage
            .add_node_version(
                node_id,
                vid,
                (i as i64 * 1000).into(),
                (i as i64 * 1000).into(),
                label,
                PropertyMapBuilder::new().insert("value", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Test reconstruction for versions in different delta chains
    // v3 is in first chain (anchor at v0)
    let props_v3 = storage
        .reconstruct_node_properties(VersionId::new(3).unwrap())
        .unwrap();
    assert_eq!(props_v3.get("value").and_then(|v| v.as_int()), Some(3));

    // v7 is in second chain (anchor at v5)
    let props_v7 = storage
        .reconstruct_node_properties(VersionId::new(7).unwrap())
        .unwrap();
    assert_eq!(props_v7.get("value").and_then(|v| v.as_int()), Some(7));

    // v11 is in third chain (anchor at v10)
    let props_v11 = storage
        .reconstruct_node_properties(VersionId::new(11).unwrap())
        .unwrap();
    assert_eq!(props_v11.get("value").and_then(|v| v.as_int()), Some(11));

    // Verify anchors themselves
    let props_v5 = storage
        .reconstruct_node_properties(VersionId::new(5).unwrap())
        .unwrap();
    assert_eq!(props_v5.get("value").and_then(|v| v.as_int()), Some(5));
}

// ============================================================
// Cached Stats Counter Tests (Issue #212)
// ============================================================

#[test]
fn test_stats_uses_cached_counters() {
    // Issue #212: Verify stats() returns cached counters without iterating
    // through all versions, making it O(1) instead of O(versions)
    let config = AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    };
    let mut storage = HistoricalStorage::with_config(config);

    let label = GLOBAL_INTERNER.intern("Test").unwrap();
    let node_id = NodeId::new(1).unwrap();
    let edge_id = EdgeId::new(1).unwrap();

    // Create 7 node versions: anchor(0), delta(1), delta(2), anchor(3), delta(4), delta(5), anchor(6)
    for i in 0..7 {
        storage
            .add_node_version(
                node_id,
                VersionId::new(i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new().insert("value", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Create 5 edge versions: anchor(0), delta(1), delta(2), anchor(3), delta(4)
    for i in 0..5 {
        storage
            .add_edge_version(
                edge_id,
                VersionId::new(100 + i).unwrap(),
                (2000 + (i as i64) * 100).into(),
                (2000 + (i as i64) * 100).into(),
                label,
                node_id,
                node_id,
                PropertyMapBuilder::new().insert("value", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Get stats - should return cached counters in O(1)
    let stats = storage.stats();

    // Verify node counts (7 total: 3 anchors, 4 deltas)
    assert_eq!(stats.total_node_versions, 7);
    assert_eq!(stats.node_anchor_count, 3, "Should have 3 node anchors");
    assert_eq!(stats.node_delta_count, 4, "Should have 4 node deltas");

    // Verify edge counts (5 total: 2 anchors, 3 deltas)
    assert_eq!(stats.total_edge_versions, 5);
    assert_eq!(stats.edge_anchor_count, 2, "Should have 2 edge anchors");
    assert_eq!(stats.edge_delta_count, 3, "Should have 3 edge deltas");

    // Verify other stats remain correct
    assert_eq!(stats.unique_nodes, 1);
    assert_eq!(stats.unique_edges, 1);
}

#[test]
fn test_stats_counters_with_multiple_entities() {
    // Issue #212: Test that stats counters remain accurate across multiple entities
    let config = AnchorConfig {
        anchor_interval: 2,
        max_delta_chain: 10,
    };
    let mut storage = HistoricalStorage::with_config(config);

    let label = GLOBAL_INTERNER.intern("Test").unwrap();

    // Create versions for 3 different nodes
    for node_idx in 1..=3 {
        let node_id = NodeId::new(node_idx).unwrap();
        // Each node gets 4 versions: anchor(0), delta(1), anchor(2), delta(3)
        for i in 0..4 {
            storage
                .add_node_version(
                    node_id,
                    VersionId::new(node_idx * 100 + i).unwrap(),
                    (1000 + (i as i64) * 100).into(),
                    (1000 + (i as i64) * 100).into(),
                    label,
                    PropertyMapBuilder::new().insert("value", i as i64).build(),
                    false, // not a tombstone
                )
                .unwrap();
        }
    }

    // Create versions for 2 different edges
    for edge_idx in 1..=2 {
        let edge_id = EdgeId::new(edge_idx).unwrap();
        // Each edge gets 3 versions: anchor(0), delta(1), anchor(2)
        for i in 0..3 {
            storage
                .add_edge_version(
                    edge_id,
                    VersionId::new(edge_idx * 1000 + i).unwrap(),
                    (2000 + (i as i64) * 100).into(),
                    (2000 + (i as i64) * 100).into(),
                    label,
                    NodeId::new(1).unwrap(),
                    NodeId::new(2).unwrap(),
                    PropertyMapBuilder::new().insert("value", i as i64).build(),
                    false, // not a tombstone
                )
                .unwrap();
        }
    }

    let stats = storage.stats();

    // 3 nodes × 4 versions = 12 node versions (6 anchors, 6 deltas)
    assert_eq!(stats.total_node_versions, 12);
    assert_eq!(stats.node_anchor_count, 6, "Should have 6 node anchors");
    assert_eq!(stats.node_delta_count, 6, "Should have 6 node deltas");

    // 2 edges × 3 versions = 6 edge versions (4 anchors, 2 deltas)
    assert_eq!(stats.total_edge_versions, 6);
    assert_eq!(stats.edge_anchor_count, 4, "Should have 4 edge anchors");
    assert_eq!(stats.edge_delta_count, 2, "Should have 2 edge deltas");

    assert_eq!(stats.unique_nodes, 3);
    assert_eq!(stats.unique_edges, 2);
}

#[test]
fn test_stats_counters_remain_accurate_after_persistence_restore() {
    // Issue #212: Verify counters are correctly restored after persistence
    let config = AnchorConfig {
        anchor_interval: 3,
        max_delta_chain: 10,
    };
    let mut original = HistoricalStorage::with_config(config.clone());

    let label = GLOBAL_INTERNER.intern("Test").unwrap();
    let node_id = NodeId::new(1).unwrap();

    // Create 5 versions: anchor(0), delta(1), delta(2), anchor(3), delta(4)
    for i in 0..5 {
        original
            .add_node_version(
                node_id,
                VersionId::new(i).unwrap(),
                (1000 + (i as i64) * 100).into(),
                (1000 + (i as i64) * 100).into(),
                label,
                PropertyMapBuilder::new().insert("value", i as i64).build(),
                false, // not a tombstone
            )
            .unwrap();
    }

    // Verify stats before restore
    let stats_before = original.stats();
    assert_eq!(stats_before.total_node_versions, 5);
    assert_eq!(stats_before.node_anchor_count, 2);
    assert_eq!(stats_before.node_delta_count, 3);

    // Extract and restore versions
    let saved_versions: Vec<NodeVersion> = original.node_versions.values().cloned().collect();
    let mut restored = HistoricalStorage::with_config(config);
    for version in saved_versions {
        restored.insert_restored_node_version(version).unwrap();
    }
    restored.rebuild_version_chains();

    // Verify stats after restore match original
    let stats_after = restored.stats();
    assert_eq!(stats_after.total_node_versions, 5);
    assert_eq!(
        stats_after.node_anchor_count, 2,
        "Anchor count should be preserved after restore"
    );
    assert_eq!(
        stats_after.node_delta_count, 3,
        "Delta count should be preserved after restore"
    );
}

/// Test for Issue #210: Delta creation should not reconstruct previous version properties
///
/// When creating a delta version, we need the previous version's properties to compute
/// the diff. However, we just finished adding that previous version moments ago with its
/// full properties known. Currently, we reconstruct those properties even though we
/// just had them.
///
/// This test verifies that after caching properties at write-time (not just for anchors),
/// we avoid unnecessary reconstructions during consecutive delta writes.
#[test]
fn test_delta_creation_caches_properties() {
    // Use a large anchor interval to ensure we create many deltas
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 100,
        max_delta_chain: 200,
    });

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Add 10 consecutive versions - first will be anchor, rest will be deltas
    for i in 0..10 {
        let version_id = VersionId::new(100 + i).unwrap();
        let temporal = BiTemporalInterval::current((1000 + (i as i64) * 100).into());
        let props = PropertyMapBuilder::new()
            .insert("name", "Alice")
            .insert("version", i as i64)
            .build();

        storage
            .add_node_version(
                node_id,
                version_id,
                temporal.valid_time().start(),
                temporal.transaction_time().start(),
                label,
                props,
                false, // not a tombstone
            )
            .unwrap();
    }

    // Get cache metrics
    let metrics = storage.cache_metrics();

    // After the fix, we expect ZERO full reconstructions because:
    // - Version 0: anchor (no reconstruction needed)
    // - Version 1: delta, needs version 0 properties (anchor, already cached)
    // - Version 2: delta, needs version 1 properties (should be cached from write)
    // - Version 3: delta, needs version 2 properties (should be cached from write)
    // ... and so on
    //
    // BEFORE the fix: We would see full_reconstructions > 0 because when creating
    // delta N, we reconstruct version N-1's properties even though we just added them.
    //
    // AFTER the fix: We should see full_reconstructions == 0 because we cache
    // the NEW properties when adding each version.

    assert_eq!(
        metrics.full_reconstructions, 0,
        "Expected 0 full reconstructions when adding consecutive deltas, but got {}. \
             This indicates we're reconstructing properties we just added. \
             Issue #210: Cache properties at write-time to avoid this.",
        metrics.full_reconstructions
    );

    // Verify we can still reconstruct all properties correctly
    for i in 0..10 {
        let version_id = VersionId::new(100 + i).unwrap();
        let props = storage.reconstruct_node_properties(version_id).unwrap();
        assert_eq!(
            props.get("version").unwrap().as_int().unwrap(),
            i as i64,
            "Property reconstruction failed for version {}",
            i
        );
    }
}

/// Test for Issue #210: Edge delta creation should also cache properties
///
/// Same as test_delta_creation_caches_properties but for edges.
#[test]
fn test_edge_delta_creation_caches_properties() {
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 100,
        max_delta_chain: 200,
    });

    let edge_id = EdgeId::new(1).unwrap();
    let from = NodeId::new(1).unwrap();
    let to = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    // Add 10 consecutive versions
    for i in 0..10 {
        let version_id = VersionId::new(100 + i).unwrap();
        let timestamp = (1000 + (i as i64) * 100).into();
        let props = PropertyMapBuilder::new()
            .insert("strength", i as i64)
            .build();

        storage
            .add_edge_version(
                edge_id, version_id, timestamp, timestamp, label, from, to, props,
                false, // not a tombstone
            )
            .unwrap();
    }

    let metrics = storage.cache_metrics();

    // Same expectation as for nodes: 0 reconstructions after the fix
    assert_eq!(
        metrics.full_reconstructions, 0,
        "Expected 0 full reconstructions for edge deltas, but got {}. \
             Issue #210: Cache edge properties at write-time.",
        metrics.full_reconstructions
    );

    // Verify correctness
    for i in 0..10 {
        let version_id = VersionId::new(100 + i).unwrap();
        let props = storage.reconstruct_edge_properties(version_id).unwrap();
        assert_eq!(props.get("strength").unwrap().as_int().unwrap(), i as i64);
    }
}

#[test]
fn test_sentry_find_node_version_at_time_cycle_detection() {
    // 🛡️ Sentry Test: Verify infinite loop detection in find_node_version_at_time.
    // Manually constructs a cycle (v1 -> v2 -> v1) and asserts the search terminates.

    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let mut storage = HistoricalStorage::new();
        let node_id = NodeId::new(1).unwrap();
        let label = GLOBAL_INTERNER.intern("CycleTest").unwrap();

        // Version 1: Initial
        let v1_id = VersionId::new(1).unwrap();
        let v1 = NodeVersion::new_anchor(
            v1_id,
            node_id,
            BiTemporalInterval::current(1000.into()),
            label,
            PropertyMapBuilder::new().build(),
        );
        storage.insert_restored_node_version(v1).unwrap();

        // Version 2: Delta pointing to v1
        let v2_id = VersionId::new(2).unwrap();
        let v2 = NodeVersion::new_delta(
            v2_id,
            node_id,
            BiTemporalInterval::current(2000.into()),
            label,
            &PropertyMapBuilder::new().build(),
            &PropertyMapBuilder::new().build(),
            v1_id, // Points to v1
        );
        storage.insert_restored_node_version(v2).unwrap();

        // MANUALLY CREATE CYCLE: Point v1 to v2 as previous version
        // This is corruption/illegal state, but we must handle it without hanging.
        if let Some(v1_mut) = storage.node_versions.get_mut(&v1_id) {
            v1_mut.prev_version = Some(v2_id);
        }

        // Trigger linear scan (no temporal index)
        // Search for a time that requires traversal.
        // v1 starts at 1000, v2 starts at 2000.
        // Searching for 500 should force it to walk back past both v2 and v1.
        // With cycle v2 -> v1 -> v2, it should loop forever.
        let result = storage.find_node_version_at_time(
            node_id,
            500.into(), // valid_time
            500.into(), // tx_time
        );

        tx.send(result).unwrap();
    });

    // If it hangs, recv_timeout will fail.
    // If it terminates (fixed), it should return within timeout.
    let result = rx.recv_timeout(Duration::from_millis(500));

    match result {
        Ok(found) => {
            // If fixed, it should probably return None because no version matches exactly
            // (or maybe v2 if valid/tx time matches, but the point is it returns).
            // Actually, with the cycle, it will never find "previous" if it keeps looking back.
            // If we break the loop, we return None (not found).
            assert!(
                found.is_none(),
                "Should return None on cycle detection/exhaustion"
            );
        }
        Err(_) => {
            panic!("find_node_version_at_time hung (infinite loop detected by timeout)");
        }
    }
}

// ============================================================================
// Temporal Invariant Violation Tests (Issue #350)
// ============================================================================
//
// These tests verify that temporal errors are detected and reported correctly.
// They cover three categories:
//
// 1. Version chain corruption (CorruptedVersionChain, MissingAnchor)
// 2. Temporal paradox detection (competing valid_time ranges)
// 3. Historical reconstruction with broken chains

#[test]
fn test_corrupted_version_chain_delta_no_prev_version() {
    // Verifies that a delta version with a None prev_version link is detected
    // as a corrupted chain during property reconstruction.
    //
    // This exercises the error path in reconstruct_node_properties_iterative
    // at the "Delta version has no previous version" check.
    use crate::core::version::NodeVersion;

    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(42).unwrap();
    let label = GLOBAL_INTERNER.intern("CorruptTest").unwrap();

    // Create a valid anchor version first
    let v0_id = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v0_id,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().insert("name", "Alice").build(),
            false,
        )
        .unwrap();

    // Manually create a delta version then corrupt it by setting prev_version = None.
    // This simulates a broken chain where the delta has no link back to the anchor.
    let v1_id = VersionId::new(2).unwrap();
    let mut v1 = NodeVersion::new_delta(
        v1_id,
        node_id,
        BiTemporalInterval::current(2000.into()),
        label,
        &PropertyMapBuilder::new().insert("name", "Alice").build(),
        &PropertyMapBuilder::new().insert("name", "Bob").build(),
        v0_id,
    );
    v1.prev_version = None; // Deliberately corrupt: remove backward link

    storage.insert_restored_node_version(v1).unwrap();
    storage.__test_clear_property_cache();

    // Reconstruction of v1 must fail with a specific corruption error
    let result = storage.reconstruct_node_properties(v1_id);
    assert!(result.is_err(), "Expected error for corrupted chain");
    match result.unwrap_err() {
        crate::core::error::Error::Temporal(
            crate::core::error::TemporalError::CorruptedVersionChain { reason, .. },
        ) => {
            assert!(
                reason.contains("no previous version"),
                "Expected 'no previous version' in reason, got: {reason}"
            );
        }
        err => panic!("Expected CorruptedVersionChain, got: {err:?}"),
    }
}

#[test]
fn test_missing_anchor_detected_after_anchor_deletion() {
    // Verifies that when the anchor version of a node's chain is removed from
    // storage, attempting to reconstruct a downstream delta returns MissingAnchor
    // rather than a generic VersionNotFound.
    //
    // MissingAnchor is semantically more informative: it tells the caller that
    // the chain exists but the base snapshot is gone, which is distinct from
    // the requested version never existing at all.
    //
    // Chain structure: v0 (anchor) ← v1 (delta)
    // Action: remove v0 from hot storage
    // Expected: reconstruct(v1) → MissingAnchor, not VersionNotFound

    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(99).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();

    // Build anchor → delta chain
    let v0_id = VersionId::new(10).unwrap();
    storage
        .add_node_version(
            node_id,
            v0_id,
            1000.into(),
            1000.into(),
            label,
            PropertyMapBuilder::new().insert("name", "Alice").build(),
            false,
        )
        .unwrap();

    let v1_id = VersionId::new(11).unwrap();
    storage
        .add_node_version(
            node_id,
            v1_id,
            2000.into(),
            2000.into(),
            label,
            PropertyMapBuilder::new().insert("name", "Bob").build(),
            false,
        )
        .unwrap();

    assert!(storage.get_node_version(v1_id).unwrap().is_delta());
    assert_eq!(
        storage.get_node_version(v1_id).unwrap().prev_version,
        Some(v0_id)
    );

    // Pre-populate the reconstruction cache so that __test_clear_property_cache
    // becomes load-bearing: a mutant that makes it a no-op would leave the cache
    // populated and the second reconstruct call would return Ok (cached) instead
    // of MissingAnchor, which would fail the test below.
    let warm = storage.reconstruct_node_properties(v1_id).unwrap();
    assert_eq!(
        warm.get("name").and_then(|v| v.as_str()),
        Some("Bob"),
        "pre-deletion reconstruction must succeed"
    );

    // Simulate anchor loss (e.g. migration to cold storage that is then unavailable)
    storage.__test_remove_node_version(v0_id);
    storage.__test_clear_property_cache();

    // Kill the anchor-cache-clear mutation: v0 is an anchor so add_node_version put it
    // in node_anchor_cache. After __test_clear_property_cache, the anchor cache is empty.
    // A mutant that makes node_anchor_cache.clear() a no-op leaves v0 in the cache;
    // reconstruct_node_properties_with_depth would then return Ok via the anchor-cache
    // fallback instead of falling through to the iterative path that discovers the gap.
    assert!(
        storage.reconstruct_node_properties(v0_id).is_err(),
        "Anchor must not be reconstructable via cache after clear and deletion"
    );

    let result = storage.reconstruct_node_properties(v1_id);
    assert!(result.is_err(), "Expected error when anchor is missing");
    match result.unwrap_err() {
        crate::core::error::Error::Temporal(crate::core::error::TemporalError::MissingAnchor {
            entity_id,
        }) => {
            // entity_id must come from the version's node_id field (via
            // get_node_version_any_tier), not the generic "version V" fallback.
            // This assertion kills the mutation that removes the .and_then() lookup.
            assert!(
                entity_id == "Node(99)",
                "MissingAnchor entity_id must identify the node, got: {entity_id}"
            );
        }
        err => panic!("Expected MissingAnchor, got: {err:?}"),
    }
}

#[test]
fn test_edge_missing_anchor_detected_after_anchor_deletion() {
    // Mirrors test_missing_anchor_detected_after_anchor_deletion for edge versions.
    //
    // Chain: e0 (anchor) ← e1 (delta). Remove e0. Reconstruct(e1) → MissingAnchor.

    let mut storage = HistoricalStorage::new();
    let edge_id = crate::core::id::EdgeId::new(55).unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();

    let e0_id = VersionId::new(20).unwrap();
    storage
        .add_edge_version(
            edge_id,
            e0_id,
            1000.into(),
            1000.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new().insert("weight", 1i64).build(),
            false,
        )
        .unwrap();

    let e1_id = VersionId::new(21).unwrap();
    storage
        .add_edge_version(
            edge_id,
            e1_id,
            2000.into(),
            2000.into(),
            label,
            source,
            target,
            PropertyMapBuilder::new().insert("weight", 2i64).build(),
            false,
        )
        .unwrap();

    assert!(storage.get_edge_version(e1_id).unwrap().is_delta());

    // Pre-populate the edge reconstruction cache so the primary-cache clear is load-bearing.
    // A mutant that makes edge_property_cache.clear() a no-op leaves the cached Ok for e1;
    // the subsequent reconstruct would return it instead of discovering MissingAnchor.
    let warm = storage.reconstruct_edge_properties(e1_id).unwrap();
    assert_eq!(
        warm.get("weight").and_then(|v| v.as_int()),
        Some(2),
        "pre-deletion reconstruction must succeed"
    );

    // Remove the anchor
    storage.__test_remove_edge_version(e0_id);
    storage.__test_clear_edge_property_cache();

    // Kill the anchor-cache-clear mutation: e0 is an anchor so add_edge_version put it
    // in edge_anchor_cache. After __test_clear_edge_property_cache, the anchor cache is
    // empty. A mutant that makes edge_anchor_cache.clear() a no-op leaves e0 there;
    // reconstruct_edge_properties_with_depth would return Ok via the anchor-cache fallback.
    assert!(
        storage.reconstruct_edge_properties(e0_id).is_err(),
        "Edge anchor must not be reconstructable via cache after clear and deletion"
    );

    let result = storage.reconstruct_edge_properties(e1_id);
    assert!(
        result.is_err(),
        "Expected error when edge anchor is missing"
    );
    match result.unwrap_err() {
        crate::core::error::Error::Temporal(crate::core::error::TemporalError::MissingAnchor {
            entity_id,
        }) => {
            // entity_id must come from the version's edge_id field (via
            // get_edge_version_any_tier), not the generic "version V" fallback.
            assert!(
                entity_id == "Edge(55)",
                "MissingAnchor entity_id must identify the edge, got: {entity_id}"
            );
        }
        err => panic!("Expected MissingAnchor, got: {err:?}"),
    }
}

#[test]
fn test_version_chain_reconstruction_multi_hop_deltas() {
    // Verifies correct property reconstruction across a four-version chain:
    // v0 (anchor: name=Alice) → v1 (delta: name=Bob) → v2 (delta: name=Carol) → v3 (delta: name=Dave)
    //
    // The anchor_interval is set to 10 so all updates create deltas rather than new anchors.
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 10,
        max_delta_chain: 100,
    });

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();
    let names = ["Alice", "Bob", "Carol", "Dave"];
    let mut version_ids = Vec::new();

    for (i, &name) in names.iter().enumerate() {
        let vid = VersionId::new(100 + i as u64).unwrap();
        storage
            .add_node_version(
                node_id,
                vid,
                (1000 + i as i64 * 100).into(),
                (1000 + i as i64 * 100).into(),
                label,
                PropertyMapBuilder::new().insert("name", name).build(),
                false,
            )
            .unwrap();
        version_ids.push(vid);
    }

    // v0 is anchor, v1/v2/v3 are deltas (interval=10, only first is anchor)
    assert!(
        storage
            .get_node_version(version_ids[0])
            .unwrap()
            .is_anchor()
    );
    assert!(storage.get_node_version(version_ids[1]).unwrap().is_delta());
    assert!(storage.get_node_version(version_ids[2]).unwrap().is_delta());
    assert!(storage.get_node_version(version_ids[3]).unwrap().is_delta());

    storage.__test_clear_property_cache();

    for (i, &vid) in version_ids.iter().enumerate() {
        let props = storage.reconstruct_node_properties(vid).unwrap();
        assert_eq!(
            props.get("name").and_then(|v| v.as_str()),
            Some(names[i]),
            "Version {i} should have name={}",
            names[i]
        );
    }
}

#[test]
fn test_competing_valid_times_stored_and_queried_by_bitemporal_interval() {
    // Bi-temporal databases legitimately store versions with overlapping or
    // "competing" valid_time ranges. The transaction_time dimension resolves
    // which version was "known" at any given point.
    //
    // This test verifies that:
    // 1. Two versions for the same node with different valid_from times can coexist.
    // 2. Querying at each valid_time returns the correct version.
    // 3. No error is returned for "competing" ranges — they are valid state.

    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(7).unwrap();
    let label = GLOBAL_INTERNER.intern("Event").unwrap();

    // v1: valid_from=3000, recorded at tx_time=1000
    let v1_id = VersionId::new(1).unwrap();
    storage
        .add_node_version(
            node_id,
            v1_id,
            3000.into(), // valid_from: event started at t=3000
            1000.into(), // tx_time: we recorded this at t=1000
            label,
            PropertyMapBuilder::new()
                .insert("status", "scheduled")
                .build(),
            false,
        )
        .unwrap();

    // v2: valid_from=2000, recorded at tx_time=2000 (later knowledge: event started earlier)
    let v2_id = VersionId::new(2).unwrap();
    storage
        .add_node_version(
            node_id,
            v2_id,
            2000.into(), // valid_from: corrected — event actually started at t=2000
            2000.into(), // tx_time: we learned this at t=2000
            label,
            PropertyMapBuilder::new()
                .insert("status", "started")
                .build(),
            false,
        )
        .unwrap();

    // Both versions should exist without error
    assert!(storage.get_node_version(v1_id).is_some());
    assert!(storage.get_node_version(v2_id).is_some());

    // As of tx_time=1500 (after v1 but before v2), only v1 was known.
    // Query at valid_time=3500 should find v1.
    let found = storage.find_node_version_at_time(node_id, 3500.into(), 1500.into());
    assert_eq!(
        found,
        Some(v1_id),
        "At tx_time=1500, only v1 was known for valid_time=3500"
    );

    // As of tx_time=2500 (after both versions), v2 is the latest knowledge.
    // Query at valid_time=2500 should find v2 (valid_from=2000 covers this).
    let found_v2 = storage.find_node_version_at_time(node_id, 2500.into(), 2500.into());
    assert_eq!(
        found_v2,
        Some(v2_id),
        "At tx_time=2500, v2 should be visible at valid_time=2500"
    );
}

#[test]
fn test_reconstruct_nonexistent_node_version_returns_version_not_found() {
    // When version_ids is empty (the very first lookup fails), the code must return
    // StorageError::VersionNotFound, NOT TemporalError::MissingAnchor.
    //
    // A mutant that removes the `!version_ids.is_empty()` guard in the match arm would
    // cause this call to return MissingAnchor for any missing version — this test
    // catches that mutation by checking for the exact VersionNotFound error type.
    let storage = HistoricalStorage::new();
    let nonexistent = VersionId::new(9999).unwrap();
    let result = storage.reconstruct_node_properties(nonexistent);
    assert!(result.is_err(), "Non-existent version must return an error");
    match result.unwrap_err() {
        crate::core::error::Error::Storage(StorageError::VersionNotFound(id)) => assert_eq!(id, nonexistent),
        err => panic!("Expected VersionNotFound for a never-added version, got: {:?}", err),
    }
}

#[test]
fn test_reconstruct_nonexistent_edge_version_returns_version_not_found() {
    // Mirror of test_reconstruct_nonexistent_node_version_returns_version_not_found
    // for the edge reconstruction path.
    //
    // Kills the `!version_ids.is_empty()` guard mutation in
    // reconstruct_edge_properties_iterative.
    let storage = HistoricalStorage::new();
    let nonexistent = VersionId::new(9998).unwrap();
    let result = storage.reconstruct_edge_properties(nonexistent);
    assert!(
        result.is_err(),
        "Non-existent edge version must return an error"
    );
    match result.unwrap_err() {
        crate::core::error::Error::Storage(StorageError::VersionNotFound(id)) => assert_eq!(id, nonexistent),
        err => panic!("Expected VersionNotFound for a never-added edge version, got: {err:?}"),
    }
}

#[test]
fn test_get_nodes_at_time_with_label_filters_on_version_label_before_reconstruction() {
    // Unit test for the label-aware batch lookup (Issue #3236): the label
    // check runs on the version record itself, so off-label and not-visible
    // candidates are skipped (no `None` placeholders) and only label
    // matches come back, reconstructed at the queried coordinate.
    let mut storage = HistoricalStorage::new();

    let person = GLOBAL_INTERNER.intern("Person").unwrap();
    let company = GLOBAL_INTERNER.intern("Company").unwrap();

    let person_id = NodeId::new(1).unwrap();
    let company_id = NodeId::new(2).unwrap();
    let late_person_id = NodeId::new(3).unwrap();
    let never_versioned = NodeId::new(4).unwrap();

    // Two nodes visible from t=1000, one Person created only at t=3000.
    storage
        .add_node_version(
            person_id,
            VersionId::new(100).unwrap(),
            1000.into(),
            1000.into(),
            person,
            PropertyMapBuilder::new().insert("name", "Alice").build(),
            false,
        )
        .unwrap();
    storage
        .add_node_version(
            company_id,
            VersionId::new(101).unwrap(),
            1000.into(),
            1000.into(),
            company,
            PropertyMapBuilder::new().insert("name", "Acme").build(),
            false,
        )
        .unwrap();
    storage
        .add_node_version(
            late_person_id,
            VersionId::new(102).unwrap(),
            3000.into(),
            3000.into(),
            person,
            PropertyMapBuilder::new().insert("name", "Bob").build(),
            false,
        )
        .unwrap();

    let candidates = [person_id, company_id, late_person_id, never_versioned];

    // At t=2000 only the first Person matches: the Company is filtered by
    // label, the late Person is not yet visible, and the never-versioned id
    // is skipped without error.
    let found = storage
        .get_nodes_at_time_with_label(&candidates, person, 2000.into(), 2000.into())
        .unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].id, person_id);
    assert_eq!(found[0].label, person);
    assert_eq!(
        found[0].properties.get("name"),
        Some(&crate::core::property::PropertyValue::from("Alice"))
    );

    // At t=4000 both Person nodes match, in input order.
    let found = storage
        .get_nodes_at_time_with_label(&candidates, person, 4000.into(), 4000.into())
        .unwrap();
    assert_eq!(
        found.iter().map(|n| n.id).collect::<Vec<_>>(),
        vec![person_id, late_person_id]
    );

    // The Company label sees only the company node.
    let found = storage
        .get_nodes_at_time_with_label(&candidates, company, 2000.into(), 2000.into())
        .unwrap();
    assert_eq!(
        found.iter().map(|n| n.id).collect::<Vec<_>>(),
        vec![company_id]
    );
}

// ============================================================================
// Version read metadata: provenance + bi-temporal interval (Issue #3232)
// ============================================================================

#[test]
fn test_get_node_version_read_metadata_hot_tier_hit() {
    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(1).unwrap();
    let version_id = VersionId::new(100).unwrap();
    let label = GLOBAL_INTERNER.intern("Person").unwrap();
    let props = PropertyMapBuilder::new().insert("name", "Alice").build();
    let provenance = Arc::new(
        Provenance::builder()
            .source("test-suite")
            .confidence(0.9)
            .build()
            .unwrap(),
    );

    storage
        .add_node_version_with_provenance(
            node_id,
            version_id,
            1_000.into(),
            2_000.into(),
            label,
            props,
            false,
            Some(Arc::clone(&provenance)),
        )
        .unwrap();

    let (prov, interval) = storage
        .get_node_version_read_metadata(version_id)
        .unwrap()
        .expect("hot-tier version must be found");
    let prov = prov.expect("provenance stored on the version must be returned");
    assert_eq!(prov.source(), Some("test-suite"));
    assert_eq!(prov.confidence(), Some(0.9));
    assert_eq!(interval.valid_time().start(), 1_000.into());
    assert_eq!(interval.transaction_time().start(), 2_000.into());
    assert!(interval.transaction_time().is_current());
}

#[test]
fn test_get_node_version_read_metadata_missing_version_returns_none() {
    // A version id that exists in no tier (and with no tiered storage
    // configured) must resolve to Ok(None) via the tiered fallback path,
    // never an error: the MCP layer relies on this to distinguish
    // "no metadata" from "metadata lookup failed".
    let storage = HistoricalStorage::new();
    let missing = VersionId::new(424_242).unwrap();
    assert!(
        storage
            .get_node_version_read_metadata(missing)
            .unwrap()
            .is_none()
    );
}

#[test]
fn test_get_edge_version_read_metadata_hot_tier_hit() {
    let mut storage = HistoricalStorage::new();
    let edge_id = EdgeId::new(10).unwrap();
    let version_id = VersionId::new(200).unwrap();
    let label = GLOBAL_INTERNER.intern("KNOWS").unwrap();
    let props = PropertyMapBuilder::new().insert("weight", 1.0f64).build();
    let provenance = Arc::new(
        Provenance::builder()
            .source("edge-test-suite")
            .note("hot tier")
            .build()
            .unwrap(),
    );

    storage
        .add_edge_version_with_provenance(
            edge_id,
            version_id,
            3_000.into(),
            4_000.into(),
            label,
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            props,
            false,
            Some(Arc::clone(&provenance)),
        )
        .unwrap();

    let (prov, interval) = storage
        .get_edge_version_read_metadata(version_id)
        .unwrap()
        .expect("hot-tier version must be found");
    let prov = prov.expect("provenance stored on the version must be returned");
    assert_eq!(prov.source(), Some("edge-test-suite"));
    assert_eq!(prov.note(), Some("hot tier"));
    assert_eq!(interval.valid_time().start(), 3_000.into());
    assert_eq!(interval.transaction_time().start(), 4_000.into());
    assert!(interval.transaction_time().is_current());
}

#[test]
fn test_get_edge_version_read_metadata_missing_version_returns_none() {
    // Edge mirror of the node missing-version case above.
    let storage = HistoricalStorage::new();
    let missing = VersionId::new(424_243).unwrap();
    assert!(
        storage
            .get_edge_version_read_metadata(missing)
            .unwrap()
            .is_none()
    );
}

#[test]
fn test_get_node_version_read_metadata_cold_tier_fallback() {
    use crate::storage::redb_cold_storage::RedbColdStorage;
    use crate::storage::tiered_storage::TieredStorage;

    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("cold.redb");
    let cold = Arc::new(RedbColdStorage::with_default_config(&db_path).unwrap());
    let tiered = Arc::new(TieredStorage::with_default_config(cold));

    let version_id = VersionId::new(7_100).unwrap();
    let temporal = BiTemporalInterval::with_valid_time(1_000.into(), 2_000.into());
    let version = NodeVersion::new_anchor(
        version_id,
        NodeId::new(7).unwrap(),
        temporal,
        GLOBAL_INTERNER.intern("Person").unwrap(),
        PropertyMapBuilder::new()
            .insert("name", "Cold Alice")
            .build(),
    );
    tiered.store_node_version(&version).unwrap();

    let mut storage = HistoricalStorage::new();
    storage.set_tiered_storage(tiered);

    // The version was never added to the hot tier, so the lookup must fall
    // back through the tiered path and hit cold storage.
    let (prov, interval) = storage
        .get_node_version_read_metadata(version_id)
        .unwrap()
        .expect("cold-tier version must be found");
    assert!(prov.is_none(), "anchor was stored without provenance");
    assert_eq!(interval.valid_time().start(), 1_000.into());
    assert_eq!(interval.transaction_time().start(), 2_000.into());
}

#[test]
fn test_get_edge_version_read_metadata_cold_tier_fallback() {
    use crate::storage::redb_cold_storage::RedbColdStorage;
    use crate::storage::tiered_storage::TieredStorage;

    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("cold.redb");
    let cold = Arc::new(RedbColdStorage::with_default_config(&db_path).unwrap());
    let tiered = Arc::new(TieredStorage::with_default_config(cold));

    let version_id = VersionId::new(7_200).unwrap();
    let temporal = BiTemporalInterval::with_valid_time(3_000.into(), 4_000.into());
    let version = EdgeVersion::new_anchor(
        version_id,
        EdgeId::new(70).unwrap(),
        temporal,
        GLOBAL_INTERNER.intern("KNOWS").unwrap(),
        NodeId::new(7).unwrap(),
        NodeId::new(8).unwrap(),
        PropertyMapBuilder::new().insert("weight", 2.0f64).build(),
    );
    tiered.store_edge_version(&version).unwrap();

    let mut storage = HistoricalStorage::new();
    storage.set_tiered_storage(tiered);

    // Edge mirror of the node cold-tier fallback case above.
    let (prov, interval) = storage
        .get_edge_version_read_metadata(version_id)
        .unwrap()
        .expect("cold-tier version must be found");
    assert!(prov.is_none(), "anchor was stored without provenance");
    assert_eq!(interval.valid_time().start(), 3_000.into());
    assert_eq!(interval.transaction_time().start(), 4_000.into());
}

// ============================================================================
// Issue #3387: rebuild_version_chains preserves restored (persisted) state
// ============================================================================

/// Restored chain links pointing at a version ABSENT from the hot map
/// (cold-migrated) must survive `rebuild_version_chains`: the heuristic must
/// fill only MISSING links, never rewire hot versions around the gap, and
/// must not touch an already-closed transaction time.
///
/// Chain on disk: v1 <-> v2 <-> v3, with v2 cold-migrated (absent here).
/// v1: tx [1000, 2000) CLOSED, next = Some(v2)
/// v3: tx [3000, open),        prev = Some(v2)
#[test]
fn test_rebuild_version_chains_preserves_restored_links_across_cold_gap() {
    let mut storage = HistoricalStorage::new();
    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("ColdGapNode").unwrap();

    let v1_id = VersionId::new(1).unwrap();
    let v2_id = VersionId::new(2).unwrap(); // cold-migrated: never inserted
    let v3_id = VersionId::new(3).unwrap();

    let v1_tx_end: Timestamp = 2000.into();
    let mut v1 = NodeVersion::new_anchor(
        v1_id,
        node_id,
        BiTemporalInterval::current(1000.into())
            .close_transaction_time(v1_tx_end)
            .unwrap(),
        label,
        PropertyMapBuilder::new().insert("name", "v1").build(),
    );
    v1.next_version = Some(v2_id);

    let mut v3 = NodeVersion::new_anchor(
        v3_id,
        node_id,
        BiTemporalInterval::current(3000.into()),
        label,
        PropertyMapBuilder::new().insert("name", "v3").build(),
    );
    v3.prev_version = Some(v2_id);

    storage.insert_restored_node_version(v1).unwrap();
    storage.insert_restored_node_version(v3).unwrap();

    storage.rebuild_version_chains();

    let v1 = storage.node_versions.get(&v1_id).unwrap();
    assert_eq!(
        v1.next_version,
        Some(v2_id),
        "restored next link into the cold tier must not be rewired to v3"
    );
    assert_eq!(v1.prev_version, None, "v1 is the oldest: prev stays None");
    assert_eq!(
        v1.temporal.transaction_time().end(),
        v1_tx_end,
        "already-closed tx end must not be re-closed at v3's tx start"
    );

    let v3 = storage.node_versions.get(&v3_id).unwrap();
    assert_eq!(
        v3.prev_version,
        Some(v2_id),
        "restored prev link into the cold tier must not be rewired to v1"
    );
    assert_eq!(v3.next_version, None, "v3 is the head: next stays None");
    assert!(v3.temporal.transaction_time().is_current());

    // Head still resolves to the latest-tx version.
    assert_eq!(storage.get_current_node_version(node_id), Some(v3_id));
}

/// Edge mirror of
/// [`test_rebuild_version_chains_preserves_restored_links_across_cold_gap`]:
/// the edge section of `rebuild_version_chains` has its own fill-only-missing
/// guards that a node-only test would not exercise.
#[test]
fn test_rebuild_edge_version_chains_preserves_restored_links_across_cold_gap() {
    let mut storage = HistoricalStorage::new();
    let edge_id = EdgeId::new(1).unwrap();
    let source = NodeId::new(1).unwrap();
    let target = NodeId::new(2).unwrap();
    let label = GLOBAL_INTERNER.intern("COLD_GAP_EDGE").unwrap();

    let v1_id = VersionId::new(11).unwrap();
    let v2_id = VersionId::new(12).unwrap(); // cold-migrated: never inserted
    let v3_id = VersionId::new(13).unwrap();

    let v1_tx_end: Timestamp = 2000.into();
    let mut v1 = EdgeVersion::new_anchor(
        v1_id,
        edge_id,
        BiTemporalInterval::current(1000.into())
            .close_transaction_time(v1_tx_end)
            .unwrap(),
        label,
        source,
        target,
        PropertyMapBuilder::new().insert("w", 1i64).build(),
    );
    v1.next_version = Some(v2_id);

    let mut v3 = EdgeVersion::new_anchor(
        v3_id,
        edge_id,
        BiTemporalInterval::current(3000.into()),
        label,
        source,
        target,
        PropertyMapBuilder::new().insert("w", 3i64).build(),
    );
    v3.prev_version = Some(v2_id);

    storage.insert_restored_edge_version(v1).unwrap();
    storage.insert_restored_edge_version(v3).unwrap();

    storage.rebuild_version_chains();

    let v1 = storage.edge_versions.get(&v1_id).unwrap();
    assert_eq!(
        v1.next_version,
        Some(v2_id),
        "restored next link into the cold tier must not be rewired to v3"
    );
    assert_eq!(
        v1.temporal.transaction_time().end(),
        v1_tx_end,
        "already-closed tx end must not be re-closed at v3's tx start"
    );

    let v3 = storage.edge_versions.get(&v3_id).unwrap();
    assert_eq!(
        v3.prev_version,
        Some(v2_id),
        "restored prev link into the cold tier must not be rewired to v1"
    );
    assert_eq!(v3.next_version, None);
    assert_eq!(storage.get_current_edge_version(edge_id), Some(v3_id));
}

// ============================================================================
// Issue #383: Per-entity temporal vector snapshot policy
//
// These tests exercise the per-entity snapshot-policy gate that decides, per
// node/edge, whether that entity's anchor triggers the pre-anchor snapshot
// hooks (the #3525 multi-hook API). They use a *recording* hook that captures
// the `entity_id` it is invoked with, so we can assert that the temporal vector
// index is triggered for **exactly** the entities whose policy opted in.
// ============================================================================

/// A pre-anchor hook that records every `entity_id` it is invoked for and
/// returns a fixed snapshot id, so tests can observe which entities' anchors
/// actually triggered the snapshot hooks.
fn recording_hook(sink: &Arc<std::sync::Mutex<Vec<u64>>>, snapshot_id: usize) -> PreAnchorHook {
    let sink = Arc::clone(sink);
    Arc::new(move |_entity_type, entity_id, _timestamp, _properties| {
        sink.lock().unwrap().push(entity_id);
        Ok(Some(snapshot_id))
    })
}

/// Create a first (anchor) edge version for `id` and return its stored vector
/// snapshot id (mirrors `add_first_node_anchor`).
fn add_first_edge_anchor(storage: &mut HistoricalStorage, id: u64) -> Option<usize> {
    let edge_id = EdgeId::new(id).unwrap();
    let label = GLOBAL_INTERNER.intern("LINKS").unwrap();
    let vid = VersionId::new(10_000 + id).unwrap();
    storage
        .add_edge_version(
            edge_id,
            vid,
            1000.into(),
            1000.into(),
            label,
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            PropertyMapBuilder::new().build(),
            false,
        )
        .unwrap();
    let version = storage.get_edge_version(vid).unwrap();
    assert!(version.is_anchor());
    version.data.get_vector_snapshot_id()
}

/// Backward compatibility: with no policy configured the default is `Snapshot`,
/// so every anchor triggers the hook exactly as before Issue #383.
#[test]
fn test_snapshot_policy_default_is_backward_compat() {
    let mut storage = HistoricalStorage::new();
    let seen = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    storage.add_pre_node_anchor_hook(recording_hook(&seen, 7));

    // Default resolves to Snapshot for any entity.
    assert_eq!(
        storage.node_snapshot_policy(NodeId::new(1).unwrap()),
        SnapshotPolicy::Snapshot
    );
    assert_eq!(
        storage.default_node_snapshot_policy(),
        SnapshotPolicy::Snapshot
    );

    let s1 = add_first_node_anchor(&mut storage, 1);
    let s2 = add_first_node_anchor(&mut storage, 2);

    assert_eq!(s1, Some(7));
    assert_eq!(s2, Some(7));
    assert_eq!(
        *seen.lock().unwrap(),
        vec![1, 2],
        "both anchors triggered the hook"
    );
    assert_eq!(storage.hook_metrics().invocations, 2);
}

/// A node marked `Skip` still forms its graph anchor, but the snapshot hooks are
/// not run for it: no snapshot id is stored and no invocation is counted.
#[test]
fn test_snapshot_policy_skip_node_suppresses_hook() {
    let mut storage = HistoricalStorage::new();
    let seen = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    storage.add_pre_node_anchor_hook(recording_hook(&seen, 42));

    storage.set_node_snapshot_policy(NodeId::new(1).unwrap(), SnapshotPolicy::Skip);
    assert_eq!(
        storage.node_snapshot_policy(NodeId::new(1).unwrap()),
        SnapshotPolicy::Skip
    );

    let snapshot = add_first_node_anchor(&mut storage, 1);

    // Graph anchor was still created (asserted inside the helper), but with no
    // vector snapshot and no hook invocation.
    assert_eq!(
        snapshot, None,
        "Skip entity must not capture a vector snapshot"
    );
    assert!(
        seen.lock().unwrap().is_empty(),
        "Skip entity must not invoke the hook"
    );
    assert_eq!(storage.hook_metrics().invocations, 0);
}

/// Two entities with different policies are snapshotted independently: the
/// temporal index is triggered for exactly the opted-in entity.
#[test]
fn test_snapshot_policy_two_nodes_independent() {
    let mut storage = HistoricalStorage::new();
    let seen = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    storage.add_pre_node_anchor_hook(recording_hook(&seen, 5));

    storage.set_node_snapshot_policy(NodeId::new(1).unwrap(), SnapshotPolicy::Skip);
    // Node 2 keeps the default (Snapshot).

    let s1 = add_first_node_anchor(&mut storage, 1);
    let s2 = add_first_node_anchor(&mut storage, 2);

    assert_eq!(s1, None);
    assert_eq!(s2, Some(5));
    assert_eq!(
        *seen.lock().unwrap(),
        vec![2],
        "exactly the opted-in node triggered a snapshot"
    );
    assert_eq!(storage.hook_metrics().invocations, 1);
}

/// Flipping the default to `Skip` gives an opt-in model: only nodes explicitly
/// set to `Snapshot` trigger the hook.
#[test]
fn test_snapshot_policy_default_skip_opt_in() {
    let mut storage = HistoricalStorage::new();
    let seen = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    storage.add_pre_node_anchor_hook(recording_hook(&seen, 3));

    storage.set_default_node_snapshot_policy(SnapshotPolicy::Skip);
    assert_eq!(storage.default_node_snapshot_policy(), SnapshotPolicy::Skip);
    storage.set_node_snapshot_policy(NodeId::new(5).unwrap(), SnapshotPolicy::Snapshot);

    let s5 = add_first_node_anchor(&mut storage, 5);
    let s6 = add_first_node_anchor(&mut storage, 6);

    assert_eq!(s5, Some(3), "opted-in node snapshotted");
    assert_eq!(s6, None, "default-Skip node not snapshotted");
    assert_eq!(*seen.lock().unwrap(), vec![5]);
    assert_eq!(storage.hook_metrics().invocations, 1);
}

/// The gate wraps the whole ordered multi-hook run: a `Snapshot` entity runs all
/// hooks in registration order (last `Ok(Some)` wins), while a `Skip` entity
/// runs none of them.
#[test]
fn test_snapshot_policy_multi_hook_ordering_preserved_and_gated() {
    use std::sync::Mutex;

    let mut storage = HistoricalStorage::new();
    let order = Arc::new(Mutex::new(Vec::<(u64, usize)>::new()));

    for i in 0..2usize {
        let order_clone = Arc::clone(&order);
        let hook: PreAnchorHook =
            Arc::new(move |_entity_type, entity_id, _timestamp, _properties| {
                order_clone.lock().unwrap().push((entity_id, i));
                Ok(Some(100 + i))
            });
        storage.add_pre_node_anchor_hook(hook);
    }

    storage.set_node_snapshot_policy(NodeId::new(2).unwrap(), SnapshotPolicy::Skip);

    let s1 = add_first_node_anchor(&mut storage, 1); // Snapshot (default)
    let s2 = add_first_node_anchor(&mut storage, 2); // Skip

    assert_eq!(s1, Some(101), "last hook wins for the Snapshot node");
    assert_eq!(s2, None, "Skip node runs no hooks");
    assert_eq!(
        *order.lock().unwrap(),
        vec![(1, 0), (1, 1)],
        "both hooks ran in order for node 1 only; node 2 ran none",
    );
    // Two invocations total (node 1's two hooks); node 2 contributed none.
    assert_eq!(storage.hook_metrics().invocations, 2);
}

/// Node and edge policies are independent registries: making nodes default-Skip
/// does not affect edge snapshotting, and vice versa.
#[test]
fn test_snapshot_policy_node_and_edge_independent() {
    let mut storage = HistoricalStorage::new();
    let node_seen = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    let edge_seen = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    storage.add_pre_node_anchor_hook(recording_hook(&node_seen, 1));
    storage.add_pre_edge_anchor_hook(recording_hook(&edge_seen, 2));

    // Nodes default to Skip; edges keep the Snapshot default.
    storage.set_default_node_snapshot_policy(SnapshotPolicy::Skip);

    let ns = add_first_node_anchor(&mut storage, 1);
    let es = add_first_edge_anchor(&mut storage, 1);

    assert_eq!(ns, None, "node default Skip suppresses node snapshot");
    assert_eq!(
        es,
        Some(2),
        "edge default (Snapshot) is unaffected by node policy"
    );
    assert!(node_seen.lock().unwrap().is_empty());
    assert_eq!(*edge_seen.lock().unwrap(), vec![1]);
}

/// An edge marked `Skip` suppresses the edge snapshot hook, symmetric with nodes.
#[test]
fn test_snapshot_policy_edge_skip_suppresses() {
    let mut storage = HistoricalStorage::new();
    let edge_seen = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    storage.add_pre_edge_anchor_hook(recording_hook(&edge_seen, 9));

    storage.set_edge_snapshot_policy(EdgeId::new(1).unwrap(), SnapshotPolicy::Skip);
    assert_eq!(
        storage.edge_snapshot_policy(EdgeId::new(1).unwrap()),
        SnapshotPolicy::Skip
    );

    let es = add_first_edge_anchor(&mut storage, 1);

    assert_eq!(es, None);
    assert!(edge_seen.lock().unwrap().is_empty());
    assert_eq!(storage.hook_metrics().invocations, 0);
}

/// `clear_node_snapshot_policy` reverts a node to the current default; the
/// resolved getter reflects overrides and their removal.
#[test]
fn test_snapshot_policy_clear_reverts_to_default() {
    let mut storage = HistoricalStorage::new();
    let node = NodeId::new(1).unwrap();

    storage.set_node_snapshot_policy(node, SnapshotPolicy::Skip);
    assert_eq!(storage.node_snapshot_policy(node), SnapshotPolicy::Skip);

    assert_eq!(
        storage.clear_node_snapshot_policy(node),
        Some(SnapshotPolicy::Skip)
    );
    assert_eq!(
        storage.node_snapshot_policy(node),
        SnapshotPolicy::Snapshot,
        "cleared node reverts to the default policy",
    );
    // Clearing again is a no-op.
    assert_eq!(storage.clear_node_snapshot_policy(node), None);
}

/// Lock-order / no-deadlock smoke test: a sequence of many anchors with mixed
/// per-entity policies (and a configured hook timeout) completes normally,
/// exercising the gate on the write path under the historical lock.
#[test]
fn test_snapshot_policy_mixed_sequence_completes() {
    let mut storage = HistoricalStorage::new();
    let seen = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    storage.add_pre_node_anchor_hook(recording_hook(&seen, 1));
    // A generous timeout exercises the detached-thread hook path under the gate.
    storage.set_pre_anchor_hook_timeout(Some(std::time::Duration::from_secs(5)));

    // Even ids Skip, odd ids Snapshot.
    for id in 1..=20u64 {
        if id % 2 == 0 {
            storage.set_node_snapshot_policy(NodeId::new(id).unwrap(), SnapshotPolicy::Skip);
        }
    }
    for id in 1..=20u64 {
        add_first_node_anchor(&mut storage, id);
    }

    let seen = seen.lock().unwrap();
    // Exactly the 10 odd ids triggered the snapshot hook.
    let expected: Vec<u64> = (1..=20u64).filter(|id| id % 2 == 1).collect();
    assert_eq!(*seen, expected);
    assert_eq!(storage.hook_metrics().invocations, 10);
}

// ============================================================================
// Issue #383 (follow-up): the SECOND vector-snapshot trigger — the
// `VectorIndexObserver` path — must ALSO honor the per-entity snapshot policy.
//
// The observer fires from `notify_observers(NodeAnchorCreated{node_id,..})`
// right after an anchor is stored (same critical section as the pre-anchor
// hook) and calls `create_snapshot_for_anchor`. These tests use a *recording
// observer* that captures the entity id of each `*AnchorCreated` event (and
// separately the entity id of each `*VersionCreated` event), so we can assert
// the anchor-event (vector-snapshot) trigger is delivered for EXACTLY the
// entities whose policy opted in, while the general per-version event is never
// over-suppressed.
// ============================================================================

/// Observer that records which entity ids it received anchor events and
/// version events for. Mirrors `recording_hook` but on the post-commit
/// observer path (the second snapshot trigger).
struct RecordingObserver {
    node_anchors: StdMutex<Vec<u64>>,
    edge_anchors: StdMutex<Vec<u64>>,
    node_versions: StdMutex<Vec<u64>>,
    edge_versions: StdMutex<Vec<u64>>,
}

impl RecordingObserver {
    fn new() -> Self {
        Self {
            node_anchors: StdMutex::new(Vec::new()),
            edge_anchors: StdMutex::new(Vec::new()),
            node_versions: StdMutex::new(Vec::new()),
            edge_versions: StdMutex::new(Vec::new()),
        }
    }
}

impl StorageObserver for RecordingObserver {
    fn on_event(&self, event: &StorageEvent) -> Result<()> {
        match event {
            StorageEvent::NodeAnchorCreated { node_id, .. } => {
                self.node_anchors.lock().unwrap().push(node_id.as_u64());
            }
            StorageEvent::EdgeAnchorCreated { edge_id, .. } => {
                self.edge_anchors.lock().unwrap().push(edge_id.as_u64());
            }
            StorageEvent::NodeVersionCreated { node_id, .. } => {
                self.node_versions.lock().unwrap().push(node_id.as_u64());
            }
            StorageEvent::EdgeVersionCreated { edge_id, .. } => {
                self.edge_versions.lock().unwrap().push(edge_id.as_u64());
            }
        }
        Ok(())
    }
}

/// RED: a `Skip` node must NOT deliver a `NodeAnchorCreated` event to the
/// observer (the second snapshot trigger), so the observer's
/// `create_snapshot_for_anchor` never runs for it — mirroring the gated hook.
#[test]
fn test_snapshot_policy_skip_node_suppresses_observer() {
    let mut storage = HistoricalStorage::new();
    let observer = Arc::new(RecordingObserver::new());
    storage.add_observer(observer.clone());

    storage.set_node_snapshot_policy(NodeId::new(1).unwrap(), SnapshotPolicy::Skip);

    add_first_node_anchor(&mut storage, 1);

    assert!(
        observer.node_anchors.lock().unwrap().is_empty(),
        "Skip entity must not deliver a NodeAnchorCreated (no observer snapshot)"
    );
    // The general per-version event is NOT gated: metrics/audit observers still
    // see the write.
    assert_eq!(
        *observer.node_versions.lock().unwrap(),
        vec![1],
        "the general NodeVersionCreated event must NOT be over-suppressed"
    );
}

/// A default (Snapshot) node still delivers the anchor event to the observer.
#[test]
fn test_snapshot_policy_default_node_delivers_observer_anchor() {
    let mut storage = HistoricalStorage::new();
    let observer = Arc::new(RecordingObserver::new());
    storage.add_observer(observer.clone());

    add_first_node_anchor(&mut storage, 1);
    add_first_node_anchor(&mut storage, 2);

    assert_eq!(
        *observer.node_anchors.lock().unwrap(),
        vec![1, 2],
        "default Snapshot policy delivers every anchor to the observer"
    );
}

/// Two nodes with different policies are gated independently on the observer
/// path: the anchor event reaches the observer for exactly the opted-in node.
#[test]
fn test_snapshot_policy_two_nodes_independent_observer() {
    let mut storage = HistoricalStorage::new();
    let observer = Arc::new(RecordingObserver::new());
    storage.add_observer(observer.clone());

    storage.set_node_snapshot_policy(NodeId::new(1).unwrap(), SnapshotPolicy::Skip);
    // Node 2 keeps the default (Snapshot).

    add_first_node_anchor(&mut storage, 1);
    add_first_node_anchor(&mut storage, 2);

    assert_eq!(
        *observer.node_anchors.lock().unwrap(),
        vec![2],
        "exactly the opted-in node reaches the observer"
    );
    // Both writes still produce a general version event.
    assert_eq!(*observer.node_versions.lock().unwrap(), vec![1, 2]);
}

/// Default flipped to `Skip` gives an opt-in model on the observer path too.
#[test]
fn test_snapshot_policy_default_skip_opt_in_observer() {
    let mut storage = HistoricalStorage::new();
    let observer = Arc::new(RecordingObserver::new());
    storage.add_observer(observer.clone());

    storage.set_default_node_snapshot_policy(SnapshotPolicy::Skip);
    storage.set_node_snapshot_policy(NodeId::new(5).unwrap(), SnapshotPolicy::Snapshot);

    for id in [4u64, 5, 6] {
        add_first_node_anchor(&mut storage, id);
    }

    assert_eq!(
        *observer.node_anchors.lock().unwrap(),
        vec![5],
        "only the explicitly opted-in node reaches the observer"
    );
}

/// BOTH snapshot paths agree: with a recording hook AND a recording observer
/// registered together, a `Skip` node triggers NEITHER and a `Snapshot` node
/// triggers BOTH, for the same entity id.
#[test]
fn test_snapshot_policy_hook_and_observer_agree() {
    let mut storage = HistoricalStorage::new();
    let seen_hook = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    storage.add_pre_node_anchor_hook(recording_hook(&seen_hook, 9));
    let observer = Arc::new(RecordingObserver::new());
    storage.add_observer(observer.clone());

    storage.set_node_snapshot_policy(NodeId::new(1).unwrap(), SnapshotPolicy::Skip);
    // Node 2 keeps default Snapshot.

    let s1 = add_first_node_anchor(&mut storage, 1); // Skip
    let s2 = add_first_node_anchor(&mut storage, 2); // Snapshot

    // Skip node: neither trigger fired.
    assert_eq!(s1, None, "Skip node stores no hook snapshot id");
    assert!(
        !seen_hook.lock().unwrap().contains(&1),
        "Skip node ran no hook"
    );
    assert!(
        !observer.node_anchors.lock().unwrap().contains(&1),
        "Skip node reached no observer anchor"
    );

    // Snapshot node: both triggers fired for the same entity.
    assert_eq!(s2, Some(9), "Snapshot node stores the hook snapshot id");
    assert!(
        seen_hook.lock().unwrap().contains(&2),
        "Snapshot node ran the hook"
    );
    assert!(
        observer.node_anchors.lock().unwrap().contains(&2),
        "Snapshot node reached the observer anchor"
    );
}

/// Edge symmetry: a `Skip` edge suppresses the edge anchor event to the observer.
#[test]
fn test_snapshot_policy_skip_edge_suppresses_observer() {
    let mut storage = HistoricalStorage::new();
    let observer = Arc::new(RecordingObserver::new());
    storage.add_observer(observer.clone());

    storage.set_edge_snapshot_policy(EdgeId::new(1).unwrap(), SnapshotPolicy::Skip);
    // Edge 2 keeps the default (Snapshot).

    add_first_edge_anchor(&mut storage, 1);
    add_first_edge_anchor(&mut storage, 2);

    assert_eq!(
        *observer.edge_anchors.lock().unwrap(),
        vec![2],
        "exactly the opted-in edge reaches the observer"
    );
    assert_eq!(*observer.edge_versions.lock().unwrap(), vec![1, 2]);
}

/// Delta versions never deliver an anchor event, regardless of policy — the
/// observer gate rides on `is_anchor`, unchanged.
#[test]
fn test_snapshot_policy_observer_delta_never_anchors() {
    let mut storage = HistoricalStorage::with_config(AnchorConfig {
        anchor_interval: 100, // deltas after the first version
        max_delta_chain: 200,
    });
    let observer = Arc::new(RecordingObserver::new());
    storage.add_observer(observer.clone());

    let node_id = NodeId::new(1).unwrap();
    let label = GLOBAL_INTERNER.intern("Test").unwrap();
    // Snapshot policy (default) — proves deltas are excluded by is_anchor, not policy.
    for v in 1..=4u64 {
        storage
            .add_node_version(
                node_id,
                VersionId::new(v).unwrap(),
                (1000 * v as i64).into(),
                (1000 * v as i64).into(),
                label,
                PropertyMapBuilder::new().insert("v", v as i64).build(),
                false,
            )
            .unwrap();
    }

    // Only the first version is an anchor.
    assert_eq!(
        *observer.node_anchors.lock().unwrap(),
        vec![1],
        "only the anchor version delivers a NodeAnchorCreated"
    );
    assert_eq!(
        observer.node_versions.lock().unwrap().len(),
        4,
        "every version delivers a general NodeVersionCreated"
    );
}

/// Lock-order / no-deadlock smoke: a mixed-policy anchor sequence with an
/// observer registered completes and delivers anchor events for exactly the
/// opted-in entities.
#[test]
fn test_snapshot_policy_observer_mixed_sequence_completes() {
    let mut storage = HistoricalStorage::new();
    let observer = Arc::new(RecordingObserver::new());
    storage.add_observer(observer.clone());

    for id in 1..=20u64 {
        if id % 2 == 0 {
            storage.set_node_snapshot_policy(NodeId::new(id).unwrap(), SnapshotPolicy::Skip);
        }
    }
    for id in 1..=20u64 {
        add_first_node_anchor(&mut storage, id);
    }

    let expected: Vec<u64> = (1..=20u64).filter(|id| id % 2 == 1).collect();
    assert_eq!(*observer.node_anchors.lock().unwrap(), expected);
}
