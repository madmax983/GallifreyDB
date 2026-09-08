use super::*;
use crate::core::error::StorageError;
use crate::properties;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::Arc;

#[test]
fn test_property_value_types() {
    assert!(PropertyValue::Null.is_null());
    assert_eq!(PropertyValue::Bool(true).as_bool(), Some(true));
    assert_eq!(PropertyValue::Int(42).as_int(), Some(42));
    assert_eq!(PropertyValue::Float(2.5).as_float(), Some(2.5));

    let s = PropertyValue::string("hello");
    assert_eq!(s.as_str(), Some("hello"));

    let b = PropertyValue::bytes([1, 2, 3]);
    assert_eq!(b.as_bytes(), Some(&[1u8, 2, 3][..]));

    let arr = PropertyValue::array(vec![PropertyValue::Int(1), PropertyValue::Int(2)]);
    assert_eq!(arr.as_array().unwrap().len(), 2);
}

#[test]
fn test_property_value_from() {
    let _: PropertyValue = true.into();
    let _: PropertyValue = 42i64.into();
    let _: PropertyValue = 42i32.into();
    let _: PropertyValue = 2.5f64.into();
    let _: PropertyValue = "hello".into();
    let _: PropertyValue = String::from("world").into();
    let _: PropertyValue = vec![1u8, 2, 3].into();
}

#[test]
fn test_string_to_property_value_efficient_conversion() {
    // Issue #200: Verify that From<String> for PropertyValue uses
    // efficient conversion without unnecessary copying.
    // The implementation should use Arc::from(s) which leverages
    // String → Box<str> → Arc<str> conversion chain.

    // Create an owned String
    let content = "test string for efficient conversion";
    let original = String::from(content);

    // Convert to PropertyValue - should consume the String
    let prop_value: PropertyValue = original.into();

    // Verify the value is stored correctly
    assert_eq!(
        prop_value.as_str(),
        Some(content),
        "PropertyValue should contain the original string content"
    );

    // Verify it's a String variant
    assert!(matches!(prop_value, PropertyValue::String(_)));

    // Additional test: Verify the conversion works in PropertyMapBuilder
    let test_string = String::from("builder test");
    let map = PropertyMapBuilder::new().insert("key", test_string).build();

    assert_eq!(
        map.get("key").and_then(|v: &PropertyValue| v.as_str()),
        Some("builder test"),
        "PropertyMapBuilder should handle owned String efficiently"
    );
}

#[test]
fn test_vec_u8_to_property_value_efficient_conversion() {
    // Issue #200, #201: Verify that From<Vec<u8>> for PropertyValue uses
    // efficient conversion without unnecessary copying.
    // The implementation should use Arc::from(v) which leverages
    // Vec<u8> → Box<[u8]> → Arc<[u8]> conversion chain.

    // Create an owned Vec<u8>
    let content: &[u8] = &[1u8, 2, 3, 4, 5, 42, 255];
    let original = content.to_vec();

    // Convert to PropertyValue - should consume the Vec
    let prop_value: PropertyValue = original.into();

    // Verify the value is stored correctly
    assert_eq!(
        prop_value.as_bytes(),
        Some(content),
        "PropertyValue should contain the original byte content"
    );

    // Verify it's a Bytes variant
    assert!(matches!(prop_value, PropertyValue::Bytes(_)));
}

#[test]
fn test_vec_u8_to_property_value_consumes_vec() {
    // Issue #201: Verify that From<Vec<u8>> efficiently consumes the Vec
    // rather than copying from a slice.
    //
    // The implementation uses Arc::from(vec) which consumes the Vec and can
    // potentially reuse its buffer, rather than Arc::from(vec.as_slice())
    // which would copy from the slice while leaving the Vec to be dropped.
    //
    // Note: Arc<[u8]> requires its own allocation (for ref count + data),
    // so we can't test for pointer equality. What we verify is that:
    // 1. The Vec is consumed (move semantics)
    // 2. Data is preserved correctly
    // 3. The conversion works for large payloads without double-allocation

    // Create a large Vec to make efficient conversion meaningful
    let size = 10_000;
    let mut original = Vec::with_capacity(size);
    for i in 0..size {
        original.push((i % 256) as u8);
    }

    // Convert to PropertyValue - this should consume the Vec (move semantics)
    let prop_value: PropertyValue = original.into();
    // Note: `original` is now moved and cannot be used

    // Extract the Arc<[u8]> from the PropertyValue
    if let PropertyValue::Bytes(ref arc_bytes) = prop_value {
        // Verify the length is correct
        assert_eq!(
            arc_bytes.len(),
            size,
            "PropertyValue should contain all elements"
        );

        // Verify data content is preserved correctly
        for (i, &byte) in arc_bytes.iter().enumerate() {
            assert_eq!(
                byte,
                (i % 256) as u8,
                "Data should be preserved correctly at index {i}"
            );
        }
    } else {
        panic!("PropertyValue should be Bytes variant");
    }
}

#[test]
fn test_vec_u8_empty_conversion() {
    // Issue #201: Verify edge case of empty Vec<u8> conversion
    let empty_vec: Vec<u8> = Vec::new();
    let prop_value: PropertyValue = empty_vec.into();

    assert_eq!(
        prop_value.as_bytes(),
        Some(&[] as &[u8]),
        "Empty Vec should convert to empty Bytes"
    );
    assert!(matches!(prop_value, PropertyValue::Bytes(_)));
}

#[test]
fn test_vec_u8_large_payload_conversion() {
    // Issue #201: Verify that large binary payloads convert efficiently
    // without unnecessary copying. The implementation should use Arc::from(vec)
    // which consumes the Vec, rather than Arc::from(vec.as_slice()) which
    // would copy the data.
    let size = 1_000_000; // 1MB
    let large_vec: Vec<u8> = vec![0x42; size];

    let prop_value: PropertyValue = large_vec.into();

    if let PropertyValue::Bytes(arc_bytes) = &prop_value {
        // Verify the large payload is stored correctly
        assert_eq!(arc_bytes.len(), size);
        assert!(arc_bytes.iter().all(|&b| b == 0x42));
    } else {
        panic!("PropertyValue should be Bytes variant");
    }
}

#[test]
fn test_property_map_creation() {
    let map = PropertyMap::new();
    assert!(map.is_empty());
    assert_eq!(map.len(), 0);

    let map = PropertyMap::with_capacity(10);
    assert!(map.is_empty());
}

#[test]
fn test_property_map_builder() {
    let map = PropertyMapBuilder::new()
        .insert("name", "Alice")
        .insert("age", 30i64)
        .insert("active", true)
        .build();

    assert_eq!(map.len(), 3);
    assert_eq!(
        map.get("name").and_then(|v: &PropertyValue| v.as_str()),
        Some("Alice")
    );
    assert_eq!(
        map.get("age").and_then(|v: &PropertyValue| v.as_int()),
        Some(30)
    );
    assert_eq!(map.get("active").and_then(|v| v.as_bool()), Some(true));
}

#[test]
fn test_property_map_copy_on_write() {
    let map1 = PropertyMapBuilder::new().insert("key", "value1").build();

    // Clone is cheap (just Arc increment)
    let map2 = map1.clone();
    assert_eq!(map1, map2);

    // Modify map2 (should not affect map1 due to copy-on-write)
    let map2 = map2.builder().insert("key", "value2").build();

    assert_ne!(map1, map2);
    assert_eq!(
        map1.get("key").and_then(|v: &PropertyValue| v.as_str()),
        Some("value1")
    );
    assert_eq!(
        map2.get("key").and_then(|v: &PropertyValue| v.as_str()),
        Some("value2")
    );
}

#[test]
fn test_property_map_iteration() {
    let map = PropertyMapBuilder::new()
        .insert("a", 1i64)
        .insert("b", 2i64)
        .insert("c", 3i64)
        .build();

    let keys: Vec<_> = map.keys().cloned().collect();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&GLOBAL_INTERNER.intern("a").unwrap()));
    assert!(keys.contains(&GLOBAL_INTERNER.intern("b").unwrap()));
    assert!(keys.contains(&GLOBAL_INTERNER.intern("c").unwrap()));

    let values: Vec<_> = map.values().cloned().collect();
    assert_eq!(values.len(), 3);
}

#[test]
fn test_properties_macro() {
    let map = properties! {
        "name" => "Bob",
        "age" => 25,
        "score" => 98.5,
    };

    assert_eq!(map.len(), 3);
    assert_eq!(
        map.get("name").and_then(|v: &PropertyValue| v.as_str()),
        Some("Bob")
    );
    assert_eq!(
        map.get("age").and_then(|v: &PropertyValue| v.as_int()),
        Some(25)
    );
    assert_eq!(
        map.get("score").and_then(|v: &PropertyValue| v.as_float()),
        Some(98.5)
    );
}

#[test]
fn test_property_value_display() {
    assert_eq!(format!("{}", PropertyValue::Null), "null");
    assert_eq!(format!("{}", PropertyValue::Bool(true)), "true");
    assert_eq!(format!("{}", PropertyValue::Int(42)), "42");
    assert_eq!(format!("{}", PropertyValue::Float(2.5)), "2.5");
    assert_eq!(format!("{}", PropertyValue::string("hello")), "\"hello\"");

    let arr = PropertyValue::array(vec![PropertyValue::Int(1), PropertyValue::Int(2)]);
    assert_eq!(format!("{}", arr), "[1, 2]");
}

#[test]
fn test_arc_sharing() {
    // Create a property value with a large string
    let large_string = "x".repeat(1000);
    let prop1 = PropertyValue::string(&large_string);
    let prop2 = prop1.clone();

    // Both should point to the same Arc
    if let (PropertyValue::String(s1), PropertyValue::String(s2)) = (&prop1, &prop2) {
        assert!(Arc::ptr_eq(s1, s2), "Arc should be shared");
    }
}

// ========== Vector tests ==========

#[test]
fn test_vector_constructor() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let vec_prop = PropertyValue::vector(data);

    assert_eq!(vec_prop.as_vector(), Some(&data[..]));
    assert_eq!(vec_prop.type_name(), "vector");
}

#[test]
fn test_vector_from_vec() {
    let data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let vec_prop: PropertyValue = data.clone().into();

    assert_eq!(vec_prop.as_vector(), Some(&data[..]));
}

#[test]
fn test_vector_from_slice() {
    let data = [1.5f32, 2.5, 3.5];
    let vec_prop: PropertyValue = (&data[..]).into();

    assert_eq!(vec_prop.as_vector(), Some(&data[..]));
}

#[test]
fn test_vector_display() {
    let vec_prop = PropertyValue::vector([1.0f32, 2.0, 3.0]);
    assert_eq!(format!("{}", vec_prop), "<vector[3]>");

    // Test with common embedding dimensions
    let embedding_384 = vec![0.0f32; 384];
    let vec_prop = PropertyValue::vector(&embedding_384);
    assert_eq!(format!("{}", vec_prop), "<vector[384]>");

    let embedding_1536 = vec![0.0f32; 1536];
    let vec_prop = PropertyValue::vector(&embedding_1536);
    assert_eq!(format!("{}", vec_prop), "<vector[1536]>");
}

#[test]
fn test_vector_arc_sharing() {
    // Create a large vector (typical embedding size)
    let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
    let prop1 = PropertyValue::vector(&embedding);
    let prop2 = prop1.clone();

    // Both should point to the same Arc (cheap clone)
    if let (PropertyValue::Vector(v1), PropertyValue::Vector(v2)) = (&prop1, &prop2) {
        assert!(
            Arc::ptr_eq(v1, v2),
            "Vector Arc should be shared after clone"
        );
    }

    // Values should be equal
    assert_eq!(prop1, prop2);
    assert_eq!(prop1.as_vector(), prop2.as_vector());
}

#[test]
fn test_vector_empty() {
    let empty: Vec<f32> = vec![];
    let vec_prop = PropertyValue::vector(&empty);

    assert_eq!(vec_prop.as_vector(), Some(&[][..]));
    assert_eq!(format!("{}", vec_prop), "<vector[0]>");
}

#[test]
#[should_panic(expected = "Vector dimension")]
fn test_vector_excessive_dimensions() {
    // Test that creating a vector with excessive dimensions panics
    let oversized: Vec<f32> = vec![0.0; MAX_VECTOR_DIMENSIONS + 1];
    let _ = PropertyValue::vector(oversized);
}

#[test]
fn test_vector_max_dimensions_allowed() {
    // Test that MAX_VECTOR_DIMENSIONS is exactly the limit
    let max_size: Vec<f32> = vec![0.0; MAX_VECTOR_DIMENSIONS];
    let vec_prop = PropertyValue::vector(max_size);
    assert_eq!(vec_prop.as_vector().unwrap().len(), MAX_VECTOR_DIMENSIONS);
}

#[test]
fn test_vector_accessor_wrong_type() {
    // as_vector should return None for non-vector types
    assert_eq!(PropertyValue::Null.as_vector(), None);
    assert_eq!(PropertyValue::Bool(true).as_vector(), None);
    assert_eq!(PropertyValue::Int(42).as_vector(), None);
    assert_eq!(PropertyValue::Float(1.5).as_vector(), None);
    assert_eq!(PropertyValue::string("hello").as_vector(), None);
    assert_eq!(PropertyValue::bytes([1, 2, 3]).as_vector(), None);
    assert_eq!(
        PropertyValue::array(vec![PropertyValue::Int(1)]).as_vector(),
        None
    );
}

#[test]
fn test_vector_equality() {
    let v1 = PropertyValue::vector([1.0f32, 2.0, 3.0]);
    let v2 = PropertyValue::vector([1.0f32, 2.0, 3.0]);
    let v3 = PropertyValue::vector([1.0f32, 2.0, 4.0]);

    assert_eq!(v1, v2);
    assert_ne!(v1, v3);
}

#[test]
fn test_vector_in_property_map() {
    let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
    let map = PropertyMapBuilder::new()
        .insert("name", "test_node")
        .insert("embedding", PropertyValue::vector(&embedding))
        .build();

    assert_eq!(map.len(), 2);
    assert_eq!(
        map.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding[..])
    );
}

// ========== Serialization Tests ==========

#[test]
fn test_serialize_null() {
    let value = PropertyValue::Null;
    let bytes = value.serialize().expect("Serialization failed");
    assert_eq!(bytes, vec![TAG_NULL]);

    let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(deserialized, value);
    assert_eq!(consumed, 1);
}

#[test]
fn test_serialize_bool() {
    // Test true
    let value = PropertyValue::Bool(true);
    let bytes = value.serialize().expect("Serialization failed");
    assert_eq!(bytes, vec![TAG_BOOL, 1]);

    let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(deserialized, value);
    assert_eq!(consumed, 2);

    // Test false
    let value = PropertyValue::Bool(false);
    let bytes = value.serialize().expect("Serialization failed");
    assert_eq!(bytes, vec![TAG_BOOL, 0]);

    let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(deserialized, value);
    assert_eq!(consumed, 2);
}

#[test]
fn test_deserialize_bool_non_canonical_true() {
    // Rust boolean is strict (0 or 1), but serialized format allows loose "!= 0" check.
    // This targets mutants that might enforce strict "== 1" check.
    // If the implementation uses `!= 0`, then 2 should be true.
    // If it uses `== 1` (mutant), then 2 would be false.
    let bytes = vec![TAG_BOOL, 2];
    let (val, _) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(val, PropertyValue::Bool(true));
}

#[test]
fn test_serialize_int() {
    let test_values = [0i64, 1, -1, i64::MAX, i64::MIN, 42, -12345];
    for &v in &test_values {
        let value = PropertyValue::Int(v);
        let bytes = value.serialize().expect("Serialization failed");

        assert_eq!(bytes[0], TAG_INT);
        assert_eq!(bytes.len(), 9);

        let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
        assert_eq!(deserialized, value);
        assert_eq!(consumed, 9);
    }
}

#[test]
fn test_serialize_float() {
    let test_values = [0.0f64, 1.0, -1.0, f64::MAX, f64::MIN, 1.5, -2.5];
    for &v in &test_values {
        let value = PropertyValue::Float(v);
        let bytes = value.serialize().expect("Serialization failed");

        assert_eq!(bytes[0], TAG_FLOAT);
        assert_eq!(bytes.len(), 9);

        let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
        assert_eq!(deserialized, value);
        assert_eq!(consumed, 9);
    }
}

#[test]
fn test_serialize_float_special_values() {
    // Test infinity
    let value = PropertyValue::Float(f64::INFINITY);
    let bytes = value.serialize().expect("Serialization failed");
    let (deserialized, _) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(deserialized.as_float(), Some(f64::INFINITY));

    // Test negative infinity
    let value = PropertyValue::Float(f64::NEG_INFINITY);
    let bytes = value.serialize().expect("Serialization failed");
    let (deserialized, _) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(deserialized.as_float(), Some(f64::NEG_INFINITY));

    // Test NaN - special case, NaN != NaN
    let value = PropertyValue::Float(f64::NAN);
    let bytes = value.serialize().expect("Serialization failed");
    let (deserialized, _) = PropertyValue::deserialize(&bytes).unwrap();
    assert!(deserialized.as_float().unwrap().is_nan());
}

#[test]
fn test_serialize_string() {
    let test_values = ["", "hello", "world", "hello world!", "こんにちは", "🎉"];
    for s in test_values {
        let value = PropertyValue::string(s);
        let bytes = value.serialize().expect("Serialization failed");

        assert_eq!(bytes[0], TAG_STRING);

        let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
        assert_eq!(deserialized, value);
        assert_eq!(consumed, bytes.len());
    }
}

#[test]
fn test_serialize_bytes() {
    let test_values: &[&[u8]] = &[&[], &[1], &[1, 2, 3], &[0, 255, 128]];
    for &b in test_values {
        let value = PropertyValue::bytes(b);
        let bytes = value.serialize().expect("Serialization failed");

        assert_eq!(bytes[0], TAG_BYTES);

        let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
        assert_eq!(deserialized, value);
        assert_eq!(consumed, bytes.len());
    }
}

#[test]
fn test_serialize_array() {
    // Empty array
    let value = PropertyValue::array(vec![]);
    let bytes = value.serialize().expect("Serialization failed");
    let (deserialized, _) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(deserialized, value);

    // Array with mixed types
    let value = PropertyValue::array(vec![
        PropertyValue::Int(42),
        PropertyValue::string("hello"),
        PropertyValue::Bool(true),
        PropertyValue::Float(1.5),
    ]);
    let bytes = value.serialize().expect("Serialization failed");
    let (deserialized, _) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(deserialized, value);

    // Nested array
    let inner = PropertyValue::array(vec![PropertyValue::Int(1), PropertyValue::Int(2)]);
    let value = PropertyValue::array(vec![inner, PropertyValue::Int(3)]);
    let bytes = value.serialize().expect("Serialization failed");
    let (deserialized, _) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(deserialized, value);
}

#[test]
fn test_serialize_vector_basic() {
    let data = [1.0f32, 2.0, 3.0];
    let value = PropertyValue::vector(data);
    let bytes = value.serialize().expect("Serialization failed");

    // Check format: tag (1) + dimension (4) + 3*4 bytes
    assert_eq!(bytes[0], TAG_VECTOR);
    assert_eq!(bytes.len(), 1 + 4 + 3 * 4);

    let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(deserialized, value);
    assert_eq!(consumed, bytes.len());
}

#[test]
fn test_deserialize_property_value_errors() {
    // Empty buffer
    let result = PropertyValue::deserialize(&[]);
    assert!(result.is_err());

    // Unknown type tag
    let result = PropertyValue::deserialize(&[255]);
    assert!(result.is_err());

    // Truncated Int
    let result = PropertyValue::deserialize(&[TAG_INT, 1, 2, 3]);
    assert!(result.is_err());

    // Truncated String (length says 100, but no data)
    let result = PropertyValue::deserialize(&[TAG_STRING, 100, 0, 0, 0]);
    assert!(result.is_err());
}

#[test]
fn test_serialize_property_map_empty() {
    let map = PropertyMap::new();
    let bytes = map.serialize().expect("Serialization should succeed");

    // Empty map: just count (4 bytes) = [0, 0, 0, 0]
    assert_eq!(bytes, vec![0, 0, 0, 0]);

    let (deserialized, consumed) = PropertyMap::deserialize(&bytes).unwrap();
    assert_eq!(deserialized, map);
    assert_eq!(consumed, 4);
}

#[test]
fn test_serialize_property_map_round_trip() {
    let map = PropertyMapBuilder::new()
        .insert("name", "Alice")
        .insert("age", 30i64)
        .insert("active", true)
        .insert("score", 98.5)
        .build();

    let bytes = map.serialize().expect("Serialization should succeed");
    let (deserialized, _) = PropertyMap::deserialize(&bytes).unwrap();

    // Check all values match
    assert_eq!(deserialized.len(), 4);
    assert_eq!(
        deserialized
            .get("name")
            .and_then(|v: &PropertyValue| v.as_str()),
        Some("Alice")
    );
    assert_eq!(
        deserialized
            .get("age")
            .and_then(|v: &PropertyValue| v.as_int()),
        Some(30)
    );
    assert_eq!(
        deserialized.get("active").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        deserialized
            .get("score")
            .and_then(|v: &PropertyValue| v.as_float()),
        Some(98.5)
    );
}

#[test]
fn test_serialize_property_map_with_vector() {
    let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
    let map = PropertyMapBuilder::new()
        .insert("name", "node1")
        .insert("embedding", PropertyValue::vector(&embedding))
        .build();

    let bytes = map.serialize().expect("Serialization should succeed");
    let (deserialized, _) = PropertyMap::deserialize(&bytes).unwrap();

    assert_eq!(deserialized.len(), 2);
    assert_eq!(
        deserialized
            .get("name")
            .and_then(|v: &PropertyValue| v.as_str()),
        Some("node1")
    );
    assert_eq!(
        deserialized.get("embedding").and_then(|v| v.as_vector()),
        Some(&embedding[..])
    );
}

#[test]
fn test_serialize_property_map_with_nested_array() {
    let map = PropertyMapBuilder::new()
        .insert(
            "tags",
            PropertyValue::array(vec![
                PropertyValue::string("rust"),
                PropertyValue::string("database"),
            ]),
        )
        .build();

    let bytes = map.serialize().expect("Serialization should succeed");
    let (deserialized, _) = PropertyMap::deserialize(&bytes).unwrap();

    let tags = deserialized.get("tags").and_then(|v| v.as_array()).unwrap();
    assert_eq!(tags.len(), 2);
    assert_eq!(tags[0].as_str(), Some("rust"));
    assert_eq!(tags[1].as_str(), Some("database"));
}

#[test]
fn test_property_map_deserialize_errors() {
    // Empty buffer
    let result = PropertyMap::deserialize(&[]);
    assert!(result.is_err());

    // Truncated count
    let result = PropertyMap::deserialize(&[1, 2]);
    assert!(result.is_err());

    // Says 1 key-value but no data
    let result = PropertyMap::deserialize(&[1, 0, 0, 0]);
    assert!(result.is_err());
}

#[test]
fn test_serialize_into_efficiency() {
    // Test that serialize_into appends to existing buffer correctly
    let mut buffer = vec![0xAA, 0xBB]; // Some existing data
    let value = PropertyValue::Int(42);
    value.serialize_into(&mut buffer).unwrap();

    assert_eq!(buffer[0], 0xAA);
    assert_eq!(buffer[1], 0xBB);
    assert_eq!(buffer[2], TAG_INT);
    // The rest should be 42i64 in little-endian
}

#[test]
fn test_all_property_types_round_trip() {
    // Comprehensive test of all PropertyValue types
    let values = vec![
        PropertyValue::Null,
        PropertyValue::Bool(true),
        PropertyValue::Bool(false),
        PropertyValue::Int(0),
        PropertyValue::Int(i64::MAX),
        PropertyValue::Int(i64::MIN),
        PropertyValue::Float(0.0),
        PropertyValue::Float(f64::MAX),
        PropertyValue::String(Arc::from("hello")),
        PropertyValue::String(Arc::from("")),
        PropertyValue::Bytes(Arc::from([1u8, 2, 3].as_slice())),
        PropertyValue::Bytes(Arc::from([].as_slice())),
        PropertyValue::Array(Arc::new(vec![
            PropertyValue::Int(1),
            PropertyValue::string("two"),
        ])),
        PropertyValue::Array(Arc::new(vec![])),
        PropertyValue::Vector(Arc::from([1.0f32, 2.0, 3.0].as_slice())),
        PropertyValue::Vector(Arc::from([].as_slice())),
    ];

    for value in values {
        let bytes = value.serialize().expect("Serialization failed");
        let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
        assert_eq!(
            consumed,
            bytes.len(),
            "Consumed bytes should match serialized length for {:?}",
            value.type_name()
        );

        // Special handling for NaN values
        if let PropertyValue::Float(f) = &value
            && f.is_nan()
        {
            assert!(deserialized.as_float().unwrap().is_nan());
            continue;
        }
        assert_eq!(
            deserialized,
            value,
            "Round-trip failed for {:?}",
            value.type_name()
        );
    }
}

#[test]
fn test_endianness() {
    // Verify little-endian serialization
    let value = PropertyValue::Int(0x0102030405060708i64);
    let bytes = value.serialize().expect("Serialization failed");

    // Little-endian: least significant byte first
    assert_eq!(bytes[0], TAG_INT);
    assert_eq!(bytes[1], 0x08);
    assert_eq!(bytes[2], 0x07);
    assert_eq!(bytes[3], 0x06);
    assert_eq!(bytes[4], 0x05);
    assert_eq!(bytes[5], 0x04);
    assert_eq!(bytes[6], 0x03);
    assert_eq!(bytes[7], 0x02);
    assert_eq!(bytes[8], 0x01);
}

// ========================================================================
// PropertyKey Interning Tests (Issue #16)
// ========================================================================

#[test]
fn test_property_key_interning_serialization_round_trip() {
    // Create a property map with interned keys
    let map = PropertyMapBuilder::new()
        .insert("name", "Alice")
        .insert("age", 30i64)
        .insert("active", true)
        .build();

    // Serialize
    let bytes = map.serialize().expect("Serialization should succeed");

    // Deserialize
    let (deserialized, _) =
        PropertyMap::deserialize(&bytes).expect("Deserialization should succeed");

    // Verify all values match
    assert_eq!(
        deserialized
            .get("name")
            .and_then(|v: &PropertyValue| v.as_str()),
        Some("Alice")
    );
    assert_eq!(
        deserialized
            .get("age")
            .and_then(|v: &PropertyValue| v.as_int()),
        Some(30)
    );
    assert_eq!(
        deserialized.get("active").and_then(|v| v.as_bool()),
        Some(true)
    );

    // Verify keys are interned (same ID as original)
    let original_keys: Vec<_> = map.keys().cloned().collect();
    let deserialized_keys: Vec<_> = deserialized.keys().cloned().collect();

    assert_eq!(original_keys.len(), deserialized_keys.len());
    for key in &original_keys {
        assert!(
            deserialized_keys.contains(key),
            "Key should be interned with same ID"
        );
    }
}

#[test]
fn test_property_key_memory_efficiency() {
    use std::mem::size_of;

    // Verify InternedString is indeed smaller than String
    assert_eq!(
        size_of::<InternedString>(),
        4,
        "InternedString should be 4 bytes"
    );
    assert_eq!(size_of::<String>(), 24, "String should be 24 bytes");

    // Create multiple maps with the same keys
    let maps: Vec<_> = (0..100)
        .map(|i| {
            PropertyMapBuilder::new()
                .insert("test_mem_name", format!("Person{}", i))
                .insert("test_mem_age", i as i64)
                .insert("test_mem_id", i as i64)
                .build()
        })
        .collect();

    // Verify all maps use the same interned key IDs (order may vary)
    use std::collections::HashSet;
    let first_keys: HashSet<_> = maps[0].keys().cloned().collect();
    for map in &maps[1..] {
        let map_keys: HashSet<_> = map.keys().cloned().collect();
        assert_eq!(
            first_keys, map_keys,
            "All maps should share the same interned key IDs"
        );
    }

    // Verify we have exactly 3 unique keys across all maps
    assert_eq!(first_keys.len(), 3, "Should have exactly 3 unique keys");

    // Verify the specific keys exist and can be resolved
    let expected_keys = ["test_mem_name", "test_mem_age", "test_mem_id"];
    for key_str in &expected_keys {
        let exists = first_keys.iter().any(|key| {
            GLOBAL_INTERNER
                .resolve_with(*key, |s| s == *key_str)
                .unwrap_or(false)
        });
        assert!(
            exists,
            "Key '{}' should exist in the property maps",
            key_str
        );
    }
}

#[test]
fn test_invalid_interned_string_serialization() {
    // Create an InternedString with a raw ID that doesn't exist in the interner
    let invalid_key = InternedString::from_raw(999999);

    // Create a property map and manually insert with invalid key
    let mut inner_map = HashMap::with_hasher(BuildHasherDefault::default());
    inner_map.insert(invalid_key, PropertyValue::Int(42));
    let map = PropertyMap {
        inner: Arc::new(inner_map),
        cached_size: 4, // Ignored
    };

    // Serialization should return an error, not panic
    let result = map.serialize();
    assert!(
        result.is_err(),
        "Serialization should fail for invalid InternedString"
    );

    match result {
        Err(crate::core::error::Error::Storage(StorageError::InconsistentState { reason })) => {
            assert!(
                reason.contains("not found in interner"),
                "Error message should indicate missing key in interner"
            );
        }
        _ => panic!("Expected StorageError::InconsistentState"),
    }
}

#[test]
fn test_serialize_with_invalid_key() {
    // This explicitly verifies the error path when the interner is missing a key
    // Uses the same logic as test_invalid_interned_string_serialization but
    // ensures the new `ok_or_else` path is hit correctly.

    let invalid_key = InternedString::from_raw(888888);
    let mut inner_map = HashMap::with_hasher(BuildHasherDefault::default());
    inner_map.insert(invalid_key, PropertyValue::Bool(true));
    let map = PropertyMap {
        inner: Arc::new(inner_map),
        cached_size: 4, // Ignored
    };

    let mut buffer = Vec::new();
    let result = map.serialize_into(&mut buffer);

    assert!(result.is_err(), "Should return error for missing key");
    match result {
        Err(crate::core::error::Error::Storage(StorageError::InconsistentState { reason })) => {
            assert!(
                reason.contains("888888"),
                "Error message should contain the invalid key ID"
            );
        }
        _ => panic!("Expected InconsistentState error"),
    }
}

#[test]
fn test_concurrent_property_key_access() {
    use std::sync::Arc;
    use std::thread;

    // Create a property map
    let map = Arc::new(
        PropertyMapBuilder::new()
            .insert("shared_key", "value")
            .insert("count", 0i64)
            .build(),
    );

    // Spawn multiple threads accessing the same property keys
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let map_clone = Arc::clone(&map);
            thread::spawn(move || {
                // Each thread accesses properties multiple times
                for _ in 0..100 {
                    assert_eq!(
                        map_clone
                            .get("shared_key")
                            .and_then(|v: &PropertyValue| v.as_str()),
                        Some("value")
                    );
                    assert!(map_clone.contains_key("count"));
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
}

#[test]
fn test_property_key_get_efficiency() {
    // Pre-populate interner with a key
    let interned_key = GLOBAL_INTERNER.intern("test_key").unwrap();

    let map = PropertyMapBuilder::new()
        .insert("test_key", "value")
        .build();

    // get() should auto-intern the key
    let value1 = map.get("test_key");
    assert_eq!(
        value1.and_then(|v: &PropertyValue| v.as_str()),
        Some("value")
    );

    // get_by_interned_key() should be more efficient for repeated lookups
    let value2 = map.get_by_interned_key(&interned_key);
    assert_eq!(
        value2.and_then(|v: &PropertyValue| v.as_str()),
        Some("value")
    );

    // Both methods should return the same result
    assert_eq!(value1, value2);
}

// ========================================================================
// SparseVector PropertyValue Tests
// ========================================================================

#[test]
fn test_sparse_vector_property_value_creation() {
    use crate::core::vector::SparseVec;

    let sparse = SparseVec::new(vec![0, 2, 5], vec![1.0, 2.0, 3.0], 10).unwrap();
    let prop = PropertyValue::sparse_vector(sparse);

    assert_eq!(prop.type_name(), "sparse_vector");
    assert!(prop.as_sparse_vector().is_some());
    assert!(prop.as_vector().is_none()); // Should not match dense vector
}

#[test]
fn test_sparse_vector_property_value_accessors() {
    use crate::core::vector::SparseVec;

    let sparse = SparseVec::new(vec![1, 4], vec![1.5, 2.5], 6).unwrap();
    let prop = PropertyValue::sparse_vector(sparse);

    let retrieved = prop.as_sparse_vector().unwrap();
    assert_eq!(retrieved.nnz(), 2);
    assert_eq!(retrieved.dimension(), 6);
    assert_eq!(retrieved.indices(), &[1, 4]);
    assert_eq!(retrieved.values(), &[1.5, 2.5]);
}

#[test]
fn test_sparse_vector_from_conversion() {
    use crate::core::vector::SparseVec;

    let sparse = SparseVec::new(vec![0, 3], vec![1.0, 2.0], 5).unwrap();
    let prop: PropertyValue = sparse.into();

    assert_eq!(prop.type_name(), "sparse_vector");
    assert!(prop.as_sparse_vector().is_some());
}

#[test]
fn test_sparse_vector_display() {
    use crate::core::vector::SparseVec;

    let sparse = SparseVec::new(vec![0, 2], vec![1.0, 2.0], 10).unwrap();
    let prop = PropertyValue::sparse_vector(sparse);

    let display = format!("{}", prop);
    assert_eq!(display, "<sparse_vector[dim=10, nnz=2]>");
}

#[test]
fn test_sparse_vector_arc_sharing() {
    use crate::core::vector::SparseVec;

    let sparse = SparseVec::new(vec![0, 50, 100], vec![1.0, 2.0, 3.0], 1000).unwrap();
    let prop1 = PropertyValue::sparse_vector(sparse);
    let prop2 = prop1.clone();

    // Both should point to the same Arc (cheap clone)
    if let (PropertyValue::SparseVector(sv1), PropertyValue::SparseVector(sv2)) = (&prop1, &prop2) {
        assert!(
            Arc::ptr_eq(sv1, sv2),
            "SparseVector Arc should be shared after clone"
        );
    } else {
        panic!("Expected SparseVector variants");
    }

    assert_eq!(prop1, prop2);
}

#[test]
fn test_sparse_vector_in_property_map() {
    use crate::core::vector::SparseVec;

    let sparse = SparseVec::new(vec![10, 42, 100], vec![2.5, 1.8, 3.2], 1000).unwrap();
    let map = PropertyMapBuilder::new()
        .insert("name", "document1")
        .insert("sparse_embedding", PropertyValue::sparse_vector(sparse))
        .build();

    assert_eq!(map.len(), 2);
    let retrieved_sparse = map
        .get("sparse_embedding")
        .and_then(|v| v.as_sparse_vector());
    assert!(retrieved_sparse.is_some());
    assert_eq!(retrieved_sparse.unwrap().nnz(), 3);
    assert_eq!(retrieved_sparse.unwrap().dimension(), 1000);
}

#[test]
fn test_property_map_contains_vector_with_sparse() {
    use crate::core::vector::SparseVec;

    // Map with sparse vector
    let sparse = SparseVec::new(vec![0], vec![1.0], 10).unwrap();
    let map = PropertyMapBuilder::new()
        .insert("sparse", PropertyValue::sparse_vector(sparse))
        .build();
    assert!(map.contains_vector());

    // Map with dense vector
    let map = PropertyMapBuilder::new()
        .insert("dense", PropertyValue::vector([1.0f32, 2.0, 3.0]))
        .build();
    assert!(map.contains_vector());

    // Map without vectors
    let map = PropertyMapBuilder::new()
        .insert("name", "test")
        .insert("count", 42i64)
        .build();
    assert!(!map.contains_vector());
}

// ========================================================================
// SparseVector Serialization Tests
// ========================================================================

#[test]
fn test_serialize_sparse_vector_basic() {
    use crate::core::vector::SparseVec;

    let sparse = SparseVec::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 5).unwrap();
    let prop = PropertyValue::sparse_vector(sparse);
    let bytes = prop.serialize().expect("Serialization failed");

    assert_eq!(bytes[0], TAG_SPARSE_VECTOR);

    let (deserialized, consumed) = PropertyValue::deserialize(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());

    match deserialized {
        PropertyValue::SparseVector(ref sv) => {
            assert_eq!(sv.nnz(), 3);
            assert_eq!(sv.dimension(), 5);
            assert_eq!(sv.indices(), &[0, 2, 4]);
            assert_eq!(sv.values(), &[1.0, 2.0, 3.0]);
        }
        _ => panic!("Expected SparseVector"),
    }
}

#[test]
fn test_serialize_sparse_vector_in_property_map() {
    use crate::core::vector::SparseVec;

    let sparse = SparseVec::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0], 20).unwrap();
    let map = PropertyMapBuilder::new()
        .insert("id", 123i64)
        .insert("sparse_vec", PropertyValue::sparse_vector(sparse))
        .build();

    let bytes = map.serialize().expect("Serialization should succeed");
    let (deserialized, _) = PropertyMap::deserialize(&bytes).unwrap();

    assert_eq!(deserialized.len(), 2);
    assert_eq!(
        deserialized
            .get("id")
            .and_then(|v: &PropertyValue| v.as_int()),
        Some(123)
    );

    let sparse_result = deserialized
        .get("sparse_vec")
        .and_then(|v| v.as_sparse_vector());
    assert!(sparse_result.is_some());
    let sv = sparse_result.unwrap();
    assert_eq!(sv.nnz(), 3);
    assert_eq!(sv.dimension(), 20);
}

#[test]
fn test_sparse_vector_type_mismatch() {
    use crate::core::vector::SparseVec;

    let sparse = SparseVec::new(vec![0, 1], vec![1.0, 2.0], 5).unwrap();
    let prop = PropertyValue::sparse_vector(sparse);

    // Sparse vector should not match other types
    assert!(prop.as_int().is_none());
    assert!(prop.as_str().is_none());
    assert!(prop.as_vector().is_none()); // Should not match dense vector
    assert!(prop.as_array().is_none());
}

// ========== Issue #188: Zero-copy vector access tests ==========

#[test]
fn test_as_arc_vector_returns_arc() {
    // Test that as_arc_vector returns a cloned Arc without copying the data
    let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
    let prop = PropertyValue::vector(&embedding);

    // Get the Arc via as_arc_vector
    let arc = prop.as_arc_vector().expect("Should return Some for Vector");

    // The Arc should point to the same data
    assert_eq!(&*arc, &embedding[..]);
    assert_eq!(arc.len(), 384);
}

#[test]
fn test_as_arc_vector_shares_data_with_original() {
    // Verify that as_arc_vector returns the same underlying Arc (not a copy)
    let embedding: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let prop = PropertyValue::vector(&embedding);

    // Get two Arcs - they should point to the same data
    let arc1 = prop.as_arc_vector().unwrap();
    let arc2 = prop.as_arc_vector().unwrap();

    // Both Arcs should point to the same allocation (same pointer)
    assert!(
        Arc::ptr_eq(&arc1, &arc2),
        "Multiple calls to as_arc_vector should return Arcs to the same data"
    );
}

#[test]
fn test_as_arc_vector_returns_none_for_non_vector() {
    use crate::core::vector::SparseVec;

    // as_arc_vector should return None for non-vector types
    assert!(PropertyValue::Null.as_arc_vector().is_none());
    assert!(PropertyValue::Bool(true).as_arc_vector().is_none());
    assert!(PropertyValue::Int(42).as_arc_vector().is_none());
    assert!(PropertyValue::Float(2.5).as_arc_vector().is_none());
    assert!(PropertyValue::string("test").as_arc_vector().is_none());
    assert!(PropertyValue::bytes([1, 2, 3]).as_arc_vector().is_none());
    assert!(PropertyValue::array(vec![]).as_arc_vector().is_none());

    // SparseVector is a different type - should not match dense Vector
    let sparse = SparseVec::new(vec![0, 1], vec![1.0, 2.0], 5).unwrap();
    assert!(
        PropertyValue::sparse_vector(sparse)
            .as_arc_vector()
            .is_none()
    );
}

#[test]
fn test_as_arc_vector_does_not_copy_data() {
    // Create a large vector to ensure we'd notice if data was copied
    let large_embedding: Vec<f32> = (0..4096).map(|i| i as f32).collect();
    let prop = PropertyValue::vector(&large_embedding);

    // Get the internal Arc pointer before as_arc_vector
    let internal_arc = if let PropertyValue::Vector(arc) = &prop {
        arc.clone()
    } else {
        panic!("Expected Vector variant");
    };

    // Get Arc via as_arc_vector
    let returned_arc = prop.as_arc_vector().unwrap();

    // They should point to the exact same allocation
    assert!(
        Arc::ptr_eq(&internal_arc, &returned_arc),
        "as_arc_vector should return the same Arc, not copy the data"
    );
}

// ========================================================================
// Heap Size Estimation Tests
// ========================================================================

#[test]
fn test_estimated_heap_size_primitives() {
    // Primitives should have zero heap size
    assert_eq!(PropertyValue::Null.estimated_heap_size(), 0);
    assert_eq!(PropertyValue::Bool(true).estimated_heap_size(), 0);
    assert_eq!(PropertyValue::Bool(false).estimated_heap_size(), 0);
    assert_eq!(PropertyValue::Int(42).estimated_heap_size(), 0);
    assert_eq!(PropertyValue::Int(i64::MAX).estimated_heap_size(), 0);
    assert_eq!(PropertyValue::Float(1.5).estimated_heap_size(), 0);
    assert_eq!(PropertyValue::Float(f64::MAX).estimated_heap_size(), 0);
}

#[test]
fn test_estimated_heap_size_string() {
    // String heap size should equal string length
    let empty_string = PropertyValue::string("");
    assert_eq!(empty_string.estimated_heap_size(), 0);

    let hello = PropertyValue::string("hello");
    assert_eq!(hello.estimated_heap_size(), 5);

    let long_string = PropertyValue::string("hello world, this is a longer string");
    // 36 characters
    assert_eq!(long_string.estimated_heap_size(), 36);
}

#[test]
fn test_estimated_heap_size_bytes() {
    // Bytes heap size should equal byte array length
    let empty_bytes = PropertyValue::bytes([]);
    assert_eq!(empty_bytes.estimated_heap_size(), 0);

    let some_bytes = PropertyValue::bytes([1, 2, 3, 4, 5]);
    assert_eq!(some_bytes.estimated_heap_size(), 5);

    let large_bytes: Vec<u8> = vec![0; 1000];
    let large = PropertyValue::bytes(large_bytes);
    assert_eq!(large.estimated_heap_size(), 1000);
}

#[test]
fn test_estimated_heap_size_vector() {
    // Vector heap size should be len * sizeof(f32)
    let empty_vec = PropertyValue::vector::<[f32; 0]>([]);
    assert_eq!(empty_vec.estimated_heap_size(), 0);

    let small_vec = PropertyValue::vector([1.0f32, 2.0, 3.0, 4.0]);
    assert_eq!(
        small_vec.estimated_heap_size(),
        4 * std::mem::size_of::<f32>()
    );

    let embedding = PropertyValue::vector((0..384).map(|i| i as f32).collect::<Vec<_>>());
    assert_eq!(
        embedding.estimated_heap_size(),
        384 * std::mem::size_of::<f32>()
    );
}

#[test]
fn test_estimated_heap_size_sparse_vector() {
    use crate::core::vector::SparseVec;

    // Sparse vector heap size: nnz * (sizeof(u32) + sizeof(f32)) + sizeof(usize)
    let sparse = SparseVec::new(vec![0, 10, 100], vec![1.0, 2.0, 3.0], 1000).unwrap();
    let prop = PropertyValue::sparse_vector(sparse);

    let expected = 3 * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>())
        + std::mem::size_of::<usize>();
    assert_eq!(prop.estimated_heap_size(), expected);
}

#[test]
fn test_estimated_heap_size_array() {
    // Empty array - should be 0 since no elements
    let empty_array = PropertyValue::array(vec![]);
    assert_eq!(empty_array.estimated_heap_size(), 0);

    // Array with primitives - includes Vec overhead but values have no heap size
    let primitive_array = PropertyValue::array(vec![
        PropertyValue::Int(1),
        PropertyValue::Int(2),
        PropertyValue::Int(3),
    ]);
    assert!(primitive_array.estimated_heap_size() > 0);

    // Array with strings - should include string lengths
    let string_array = PropertyValue::array(vec![
        PropertyValue::string("hello"),
        PropertyValue::string("world"),
    ]);
    // Should include at least the string lengths (5 + 5)
    assert!(string_array.estimated_heap_size() >= 10);
}

#[test]
fn test_property_map_estimated_heap_size_empty() {
    let map = PropertyMap::new();
    // Empty map should have zero heap overhead
    let size = map.estimated_heap_size();
    assert_eq!(size, 0, "Empty map heap size should be zero");
}

#[test]
fn test_property_map_estimated_heap_size_with_values() {
    // Create a map with sufficient capacity to ensure no resizing
    let map_empty = PropertyMap::with_capacity(100);
    let empty_size = map_empty.estimated_heap_size();

    let large_string = "x".repeat(10000); // 10KB
    let map_with_string = map_empty
        .builder()
        .insert("large", large_string.as_str())
        .build();
    let string_size = map_with_string.estimated_heap_size();

    // The size difference should be exactly the string length.
    let delta = string_size - empty_size;
    assert_eq!(
        delta,
        large_string.len(),
        "Heap size delta should be exactly string length"
    );

    // Also check actual total size bounds.
    // It must be at least the string length + HashMap capacity overhead
    let min_expected_size =
        large_string.len() + (100 * std::mem::size_of::<(PropertyKey, PropertyValue)>());
    assert!(
        string_size >= min_expected_size,
        "string_size {} < min_expected {}",
        string_size,
        min_expected_size
    );
}

#[test]
fn test_property_map_estimated_heap_size_with_vector() {
    let map_empty = PropertyMap::with_capacity(100);
    let empty_size = map_empty.estimated_heap_size();

    let embedding = vec![0.1f32; 384];
    let map_with_vector = map_empty
        .builder()
        .insert_vector("embedding", &embedding)
        .build();
    let vector_size = map_with_vector.estimated_heap_size();

    let delta = vector_size - empty_size;
    let expected_delta = embedding.len() * std::mem::size_of::<f32>();

    assert_eq!(
        delta, expected_delta,
        "Heap size delta should be exactly vector data size"
    );

    let min_expected_size =
        expected_delta + (100 * std::mem::size_of::<(PropertyKey, PropertyValue)>());
    assert!(
        vector_size >= min_expected_size,
        "vector_size {} < min_expected {}",
        vector_size,
        min_expected_size
    );
}

#[test]
fn test_serialized_size_matches_actual() {
    let values = vec![
        (PropertyValue::Null, 1),                           // tag
        (PropertyValue::Bool(true), 2),                     // tag + 1 byte
        (PropertyValue::Int(123), 9),                       // tag + 8 bytes
        (PropertyValue::Float(123.456), 9),                 // tag + 8 bytes
        (PropertyValue::string("test string"), 1 + 4 + 11), // tag + len + "test string"
        (PropertyValue::bytes([1, 2, 3]), 1 + 4 + 3),       // tag + len + 3 bytes
        (
            PropertyValue::array(vec![PropertyValue::Int(1), PropertyValue::string("nested")]),
            1 + 4 + 9 + (1 + 4 + 6), // tag + count + (tag + int) + (tag + len + "nested")
        ),
        (PropertyValue::vector([1.0f32, 2.0, 3.0]), 1 + 4 + 12), // tag + len + 3 * 4 bytes
    ];

    for (value, expected_size) in values {
        let predicted = value.serialized_size().expect("Size calculation failed");
        let serialized = value.serialize().expect("Serialization failed");

        assert_eq!(
            predicted,
            expected_size,
            "Predicted size mismatch for {:?}",
            value.type_name()
        );

        assert_eq!(
            serialized.len(),
            expected_size,
            "Actual serialization length mismatch for {:?}",
            value.type_name()
        );
    }
}

#[test]
fn test_property_map_serialized_size() {
    let map = PropertyMapBuilder::new()
        .insert("name", "Alice")
        .insert("age", 30i64)
        .build();

    // Expected format (little-endian):
    // 1. Entry count (u32): 2 (4 bytes)
    // 2. Entries (sorted by key):
    //    - "age":
    //      - Key length (u32): 3 (4 bytes)
    //      - Key bytes: b"age" (3 bytes)
    //      - Value Tag: TAG_INT (1 byte)
    //      - Value Data: 30 (8 bytes)
    //    - "name":
    //      - Key length (u32): 4 (4 bytes)
    //      - Key bytes: b"name" (4 bytes)
    //      - Value Tag: TAG_STRING (1 byte)
    //      - Value Data: length (4 bytes) + b"Alice" (5 bytes)
    let expected_size = 4 + (4 + 3 + 1 + 8) + (4 + 4 + 1 + 4 + 5);

    assert_eq!(
        map.serialized_size(),
        expected_size,
        "Serialized size should match exact byte format calculation"
    );

    let serialized = map.serialize().unwrap();
    assert_eq!(
        serialized.len(),
        expected_size,
        "Actual serialization length should match exact byte format calculation"
    );

    // Exact byte checks
    let count = u32::from_le_bytes(serialized[0..4].try_into().unwrap());
    assert_eq!(count, 2);
}

#[test]
fn test_concurrent_property_map_creation() {
    use std::thread;

    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                let mut builder = PropertyMapBuilder::new();
                // Insert shared keys (stress concurrent reads on interner)
                builder = builder.insert("shared_key_1", "value1");
                builder = builder.insert("shared_key_2", 42i64);

                // Insert unique keys (stress concurrent writes/interning)
                let unique_key = format!("unique_key_{}", i);
                builder = builder.insert(&unique_key, i as i64);

                let map = builder.build();
                assert_eq!(map.len(), 3);
                assert_eq!(
                    map.get("shared_key_1")
                        .and_then(|v: &PropertyValue| v.as_str()),
                    Some("value1")
                );
                assert_eq!(
                    map.get(&unique_key)
                        .and_then(|v: &PropertyValue| v.as_int()),
                    Some(i as i64)
                );
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_cached_size_tracking() {
    let builder = PropertyMapBuilder::new();

    // Initial size (count: 4)
    let map = builder.build();
    assert_eq!(map.serialized_size(), 4);

    // Insert
    let builder = map.builder().insert("key1", 100i64); // key1 (4+4) + 100 (9) = 17 + 4 = 21
    let map = builder.build();
    assert_eq!(map.cached_size, map.serialized_size());

    // Another insert
    let builder = map.builder().insert("key2", "hello"); // key2 (4+4) + "hello" (1+4+5) = 8 + 10 = 18. Total 39.
    let map = builder.build();
    assert_eq!(map.cached_size, map.serialized_size());

    // Update (replace value)
    let builder = map.builder().insert("key1", 200i64); // Same size
    let map = builder.build();
    assert_eq!(map.cached_size, map.serialized_size());

    // Remove
    let builder = map.builder().remove("key2");
    let map = builder.build();
    assert_eq!(map.cached_size, map.serialized_size());
}

#[test]
fn test_cached_size_invariant() {
    // Property-based test: size should always match actual serialization
    let map = PropertyMapBuilder::new()
        .insert("a", 1)
        .insert("b", "test")
        .remove("a")
        .insert("c", vec![1.0f32, 2.0])
        .build();

    let serialized = map.serialize().unwrap();
    assert_eq!(map.serialized_size(), serialized.len());
    assert_eq!(map.cached_size, serialized.len());
}

#[test]
fn test_from_iter_duplicate_keys() {
    // Test that FromIterator handles duplicate keys correctly with size tracking
    let items = vec![
        (
            GLOBAL_INTERNER.intern("key").unwrap(),
            PropertyValue::Int(1),
        ),
        (
            GLOBAL_INTERNER.intern("key").unwrap(),
            PropertyValue::Int(2),
        ), // duplicate!
    ];
    let map: PropertyMap = items.into_iter().collect();

    // Logical result: {"key": 2}
    // Size: 4 (count) + 4 (len) + 3 ("key") + 9 (Int) = 20

    let serialized = map.serialize().unwrap();
    assert_eq!(map.cached_size, serialized.len());
    assert_eq!(map.len(), 1); // Should only have one entry
    assert_eq!(
        map.get("key").and_then(|v: &PropertyValue| v.as_int()),
        Some(2)
    );
}

#[test]
fn test_deserialize_duplicate_keys_errors() {
    // Construct a buffer with duplicate keys manually
    // [count: 2][key: "a"][val: 1][key: "a"][val: 2]
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&2u32.to_le_bytes()); // Count: 2

    // Entry 1: "a" -> 1
    let key = "a";
    buffer.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buffer.extend_from_slice(key.as_bytes());
    PropertyValue::Int(1).serialize_into(&mut buffer).unwrap();

    // Entry 2: "a" -> 2 (Duplicate!)
    buffer.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buffer.extend_from_slice(key.as_bytes());
    PropertyValue::Int(2).serialize_into(&mut buffer).unwrap();

    // Deserialization should fail
    let result = PropertyMap::deserialize(&buffer);
    assert!(result.is_err());
    match result {
        Err(crate::core::error::Error::Storage(StorageError::CorruptedData(msg))) => {
            assert!(msg.contains("Duplicate property key"));
        }
        _ => panic!("Expected CorruptedData error"),
    }
}

#[test]
fn test_deserialize_recursion_limit() {
    // Construct a deeply nested array exceeding the recursion limit
    // Format: [TAG_ARRAY][count:1][TAG_ARRAY][count:1]...[TAG_NULL]
    let depth = MAX_RECURSION_DEPTH + 1;
    let mut bytes = Vec::new();

    for _ in 0..depth {
        bytes.push(TAG_ARRAY);
        bytes.extend_from_slice(&(1u32).to_le_bytes()); // Count = 1
    }

    // Terminate with a Null value
    bytes.push(TAG_NULL);

    // Try to deserialize
    let result = PropertyValue::deserialize(&bytes);

    // Should fail with recursion limit error
    assert!(result.is_err());
    match result {
        Err(crate::core::error::Error::Storage(StorageError::CorruptedData(msg))) => {
            assert!(msg.contains("recursion depth limit exceeded"));
        }
        _ => panic!("Expected CorruptedData error for recursion limit"),
    }
}

#[test]
fn test_deserialize_recursion_limit_boundary() {
    // Construct a deeply nested array exactly AT the limit (should succeed)
    let depth = MAX_RECURSION_DEPTH;
    let mut bytes = Vec::new();

    for _ in 0..depth {
        bytes.push(TAG_ARRAY);
        bytes.extend_from_slice(&(1u32).to_le_bytes()); // Count = 1
    }

    // Terminate with a Null value
    bytes.push(TAG_NULL);

    // Try to deserialize
    let result = PropertyValue::deserialize(&bytes);
    assert!(result.is_ok(), "Should succeed at recursion limit boundary");
}

#[test]
fn test_deserialize_truncated_after_tag() {
    // Buffer containing only a tag but no data
    let bytes = vec![TAG_STRING]; // String expects length prefix
    let result = PropertyValue::deserialize(&bytes);
    assert!(result.is_err());
    match result {
        Err(crate::core::error::Error::Storage(StorageError::CorruptedData(msg))) => {
            assert!(msg.contains("Buffer too short"));
        }
        _ => panic!("Expected CorruptedData error"),
    }
}

#[test]
fn test_estimated_heap_size_nested_array() {
    // Create a nested array: [[[[...]]]] (depth 10) containing a string at the bottom
    let mut value = PropertyValue::string("data");
    for _ in 0..10 {
        value = PropertyValue::array(vec![value]);
    }

    let size = value.estimated_heap_size();

    // Size should be:
    // 10 * (Vec capacity overhead + sizeof(PropertyValue)) + string length
    // Vec capacity is at least 1.
    let min_vec_size = std::mem::size_of::<PropertyValue>();
    let expected_min = 10 * min_vec_size + 4; // "data".len() = 4

    assert!(size >= expected_min);
}

#[test]
fn test_property_map_debug_sorting() {
    // Use unsorted insert order: b, a, c
    let map = PropertyMapBuilder::new()
        .insert("b", 2)
        .insert("a", 1)
        .insert("c", 3)
        .build();

    let debug_str = format!("{:?}", map);
    // Should sort keys alphabetically: a, b, c
    // The output format is standard debug map: {"key": value, ...}
    // We look for "a": ... "b": ... "c": ... in that order
    let pos_a = debug_str.find("\"a\"").unwrap();
    let pos_b = debug_str.find("\"b\"").unwrap();
    let pos_c = debug_str.find("\"c\"").unwrap();

    assert!(
        pos_a < pos_b,
        "Debug output should be sorted: 'a' before 'b'"
    );
    assert!(
        pos_b < pos_c,
        "Debug output should be sorted: 'b' before 'c'"
    );
}

#[test]
fn test_property_map_debug_fallback() {
    // Create a PropertyMap with a raw unresolved key
    // We must bypass PropertyMapBuilder because it validates keys against the interner
    let mut map = HashMap::with_hasher(BuildHasherDefault::default());
    let raw_key = InternedString::from_raw(u32::MAX);
    map.insert(raw_key, PropertyValue::Int(42));

    let prop_map = PropertyMap {
        inner: Arc::new(map),
        cached_size: 0, // Not used for Debug
    };

    let debug_str = format!("{:?}", prop_map);
    // Fallback format for unknown key: InternedString(4294967295)
    assert!(
        debug_str.contains("InternedString(4294967295)"),
        "Debug output should fallback for unknown key"
    );
}

#[cfg(test)]
mod sentry_tests {
    use super::*;

    use crate::core::error::StorageError;
    use std::sync::Arc;

    /// 🎯 Target: PropertyMap::from_iter
    /// 💣 Risk: Panics when recursion depth limit is exceeded.
    /// 🧪 Strategy: Construct a deeply nested structure and try to create a PropertyMap from it using collect().
    /// 🔬 Verification: Expect NO panic (Warden fix), but subsequent serialize() should fail.
    #[test]
    fn test_property_map_from_iter_no_panic_on_deep_recursion() {
        // Construct a deeply nested value: Array(Array(...Array(Int(42))...))
        // Depth: MAX_RECURSION_DEPTH + 1
        let mut value = PropertyValue::Int(42);
        // Nest it MAX_RECURSION_DEPTH + 1 times
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }

        // This should NOT panic (Warden fix)
        let map: PropertyMap = vec![(GLOBAL_INTERNER.intern("deep").unwrap(), value)]
            .into_iter()
            .collect();

        // But serialization MUST fail because it re-checks depth
        let result = map.serialize();
        assert!(
            result.is_err(),
            "Serialization should fail due to recursion limit"
        );

        // Verify cached_size includes penalty
        assert!(map.serialized_size() > 10 * 1024 * 1024);
    }

    /// 🎯 Target: PropertyMapBuilder::try_insert
    /// 💣 Risk: Should fail gracefully (return Err) instead of panicking on deep recursion.
    /// 🧪 Strategy: Try to insert a deeply nested structure using try_insert.
    /// 🔬 Verification: Check that Result is Err and contains the recursion limit message.
    #[test]
    fn test_property_map_builder_try_insert_returns_error_on_deep_recursion() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }

        // try_insert should return an error, not panic
        let result = PropertyMapBuilder::new().try_insert("deep", value);

        assert!(result.is_err(), "Expected error, got Ok");
        let err = result.err().unwrap();
        let err_msg = format!("{}", err);
        assert!(
            err_msg.contains("recursion depth limit exceeded"),
            "Unexpected error message: {}",
            err_msg
        );
    }

    /// 🎯 Target: PropertyMapBuilder::try_remove
    /// 💣 Risk: Should fail gracefully (return Err) instead of panicking on deep recursion when computing old value size.
    /// 🧪 Strategy: Force a deeply nested structure into the map and attempt to remove it via try_remove.
    /// 🔬 Verification: Check that Result is Err and contains the recursion limit message.
    #[test]
    fn test_property_map_builder_try_remove_returns_error_on_deep_recursion() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }

        let mut builder = PropertyMapBuilder::new();
        let key = GLOBAL_INTERNER.intern("deep").unwrap();
        // Insert directly to bypass try_insert's checks
        builder.map.insert(key, value);

        // try_remove should return an error
        let result = builder.try_remove("deep");

        assert!(result.is_err(), "Expected error, got Ok");
        let err = result.err().unwrap();
        let err_msg = format!("{}", err);
        assert!(
            err_msg.contains("recursion depth limit exceeded"),
            "Unexpected error message: {}",
            err_msg
        );
    }

    /// 🎯 Target: PropertyMapBuilder::try_remove_by_key
    /// 💣 Risk: Should fail gracefully (return Err) instead of panicking on deep recursion when computing old value size.
    /// 🧪 Strategy: Force a deeply nested structure into the map and attempt to remove it via try_remove_by_key.
    /// 🔬 Verification: Check that Result is Err and contains the recursion limit message.
    #[test]
    fn test_property_map_builder_try_remove_by_key_returns_error_on_deep_recursion() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }

        let mut builder = PropertyMapBuilder::new();
        let key = GLOBAL_INTERNER.intern("deep").unwrap();
        // Insert directly to bypass try_insert's checks
        builder.map.insert(key, value);

        // try_remove_by_key should return an error
        let result = builder.try_remove_by_key(&key);

        assert!(result.is_err(), "Expected error, got Ok");
        let err = result.err().unwrap();
        let err_msg = format!("{}", err);
        assert!(
            err_msg.contains("recursion depth limit exceeded"),
            "Unexpected error message: {}",
            err_msg
        );
    }

    /// 🎯 Target: PropertyMapBuilder::try_insert_by_key
    /// 💣 Risk: Should fail gracefully (return Err) instead of panicking on deep recursion when using an interned key.
    /// 🧪 Strategy: Try to insert a deeply nested structure using try_insert_by_key.
    /// 🔬 Verification: Check that Result is Err and contains the recursion limit message.
    #[test]
    fn test_property_map_builder_try_insert_by_key_returns_error_on_deep_recursion() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }

        let key = GLOBAL_INTERNER.intern("deep").unwrap();

        // try_insert_by_key should return an error, not panic
        let result = PropertyMapBuilder::new().try_insert_by_key(key, value);

        assert!(result.is_err(), "Expected error, got Ok");
        let err = result.err().unwrap();
        let err_msg = format!("{}", err);
        assert!(
            err_msg.contains("recursion depth limit exceeded"),
            "Unexpected error message: {}",
            err_msg
        );
    }

    /// 🎯 Target: PropertyValue::estimated_heap_size
    /// 💣 Risk: Calculation failure should default to a "penalty" size to prevent cache monopolization by malicious inputs.
    /// 🧪 Strategy: Call estimated_heap_size on a deeply nested structure.
    /// 🔬 Verification: Verify result is 10MB (10 * 1024 * 1024).
    #[test]
    fn test_property_value_estimated_heap_size_penalty() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }

        // Should swallow the recursion error and return the penalty size
        let size = value.estimated_heap_size();

        // 10MB penalty
        assert_eq!(size, 10 * 1024 * 1024);
    }

    /// 🎯 Target: PropertyMap::deserialize
    /// 💣 Risk: Malformed UTF-8 in property keys could cause panic or incorrect behavior.
    /// 🧪 Strategy: Manually construct a serialized buffer with invalid UTF-8 in the key section.
    /// 🔬 Verification: Verify deserialize returns a CorruptedData error with "Invalid UTF-8" message.
    #[test]
    fn test_property_map_deserialize_invalid_utf8_key() {
        // Construct a buffer manually
        // Format: [count: 4 bytes][key_len: 4 bytes][key_bytes][value...]
        let mut buffer = Vec::new();

        // Count: 1 entry
        buffer.extend_from_slice(&1u32.to_le_bytes());

        // Key length: 1 byte
        buffer.extend_from_slice(&1u32.to_le_bytes());

        // Key bytes: 0xFF is invalid in UTF-8
        buffer.push(0xFF);

        // Value: Tag Null (0), no payload
        buffer.push(TAG_NULL);

        let result = PropertyMap::deserialize(&buffer);

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{}", err);

        // Should be StorageError::CorruptedData wrapping the utf8 error
        assert!(
            err_msg.contains("Invalid UTF-8") || err_msg.contains("Corrupted data"),
            "Unexpected error message: {}",
            err_msg
        );
    }

    /// 🎯 Target: PropertyMapBuilder::insert_by_key
    /// 💣 Risk: In debug builds, inserting an invalid key should panic to catch bugs.
    /// 🧪 Strategy: Try to insert an invalid key (one not present in the interner).
    /// 🔬 Verification: Expect panic with "missing from interner" message.
    #[test]
    #[cfg(debug_assertions)] // Only runs in debug mode where debug_assert! panics
    #[should_panic(expected = "missing from interner")]
    fn test_builder_insert_by_key_panic() {
        // Create a raw key that definitely doesn't exist in the interner
        let invalid_key = InternedString::from_raw(u32::MAX);
        let builder = PropertyMapBuilder::new();
        // This should trigger debug_assert!(false) inside try_insert_by_key
        builder.insert_by_key(invalid_key, PropertyValue::Int(1));
    }

    /// 🎯 Target: PropertyValue equality with NaN
    /// 💣 Risk: Users might expect NaN == NaN, but IEEE 754 says no.
    /// 🧪 Strategy: Create PropertyValues with NaN and check equality.
    /// 🔬 Verification: Ensure behavior matches optimization (Identity -> Equal, Value -> Not Equal).
    #[test]
    fn test_property_value_nan_inequality() {
        // Dense vector with NaN
        let dense_nan = PropertyValue::vector([f32::NAN]);

        // OPTIMIZATION CHANGE: Shared pointers are now considered equal even with NaN
        assert_eq!(
            dense_nan, dense_nan,
            "Dense vector with NaN SHOULD equal itself (identity optimization)"
        );
        assert_eq!(
            dense_nan,
            dense_nan.clone(),
            "Cloned (shared) dense vector with NaN SHOULD equal itself (identity optimization)"
        );

        // But distinct allocations with same data should NOT be equal (standard IEEE 754 behavior)
        let dense_nan_2 = PropertyValue::vector([f32::NAN]);
        assert_ne!(
            dense_nan, dense_nan_2,
            "Distinct dense vectors with NaN should NOT be equal"
        );

        // Sparse vector with NaN
        // Note: SparseVec::new returns error for NaN, so we need to construct it carefully or expect error
        let result = crate::core::vector::SparseVec::new(vec![0], vec![f32::NAN], 10);
        assert!(result.is_err(), "SparseVec should reject NaN");

        // However, f32::NAN is valid f32. PropertyValue::Float(NaN) is possible.
        // Float is a primitive variant, so no pointer optimization possible.
        let float_nan = PropertyValue::Float(f64::NAN);
        assert_ne!(float_nan, float_nan, "Float NaN should not equal itself");

        // Vector property doesn't check for NaN in constructor!
        // PropertyValue::vector calls PropertyValue::try_vector -> validate_vector_dimensions.
        // It does NOT check values for NaN.
        let vec_nan = PropertyValue::vector([f32::NAN]);
        // Optimized behavior:
        assert_eq!(
            vec_nan, vec_nan,
            "Vector with NaN SHOULD equal itself (identity)"
        );
    }

    /// 🎯 Target: PropertyMapBuilder::insert panic
    /// 💣 Risk: Incorrect panic message or behavior on recursion limit.
    /// 🧪 Strategy: Trigger recursion limit and catch panic message.
    /// 🔬 Verification: expect specific string.
    #[test]
    #[should_panic(expected = "recursion depth limit exceeded")]
    fn test_property_map_builder_insert_panic_message() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }
        PropertyMapBuilder::new().insert("deep", value);
    }

    /// 🎯 Target: PropertyMap::deserialize (MAX_PROPERTY_MAP_CAPACITY)
    /// 💣 Risk: Deserialization should fail if the count exceeds the maximum allowed capacity.
    /// 🧪 Strategy: Manually construct a serialized buffer claiming to have MAX_PROPERTY_MAP_CAPACITY + 1 elements.
    /// 🔬 Verification: Expect error "exceeds maximum allowed".
    #[test]
    fn test_property_map_capacity_limit() {
        let mut bytes = Vec::new();
        let count = (MAX_PROPERTY_MAP_CAPACITY + 1) as u32;
        bytes.extend_from_slice(&count.to_le_bytes());

        // Don't need payload, should fail immediately on count check
        let result = PropertyMap::deserialize(&bytes);
        assert!(result.is_err());
        match result {
            Err(crate::core::error::Error::Storage(StorageError::CorruptedData(msg))) => {
                assert!(
                    msg.contains("exceeds maximum allowed"),
                    "Unexpected error message: {}",
                    msg
                );
            }
            _ => panic!("Expected CorruptedData error"),
        }
    }

    /// 🎯 Target: PropertyMapBuilder::remove correctness
    /// 💣 Risk: Removing a key should actually remove it and update size/len correctly.
    /// 🧪 Strategy: Insert keys, remove one, verify map state.
    /// 🔬 Verification: Check len(), contains_key(), and cached_size().
    #[test]
    fn test_property_map_builder_remove_correctness() {
        let builder = PropertyMapBuilder::new()
            .insert("keep", 1)
            .insert("remove_me", 2);

        let map_before = builder.build();
        assert_eq!(map_before.len(), 2);
        assert!(map_before.contains_key("remove_me"));

        let before_size = map_before.serialized_size();

        // Use clone() to keep map_before for comparison
        let builder = map_before.clone().builder().remove("remove_me");
        let map_after = builder.build();

        assert_eq!(map_after.len(), 1);
        assert!(!map_after.contains_key("remove_me"));
        assert!(map_after.contains_key("keep"));

        // Verify size is updated
        assert!(map_after.serialized_size() < before_size);
        assert_eq!(
            map_after.serialized_size(),
            map_after.serialize().unwrap().len()
        );
    }

    /// 🎯 Target: PropertyMap::deserialize trailing bytes
    /// 💣 Risk: Deserialization should consume exactly what is needed and return the count, ignoring trailing data.
    /// 🧪 Strategy: Serialize a valid map, append garbage, deserialize.
    /// 🔬 Verification: Check consumed bytes matches map size, not buffer size.
    #[test]
    fn test_property_map_deserialize_trailing_bytes() {
        let map = PropertyMapBuilder::new().insert("key", "value").build();
        let mut bytes = map.serialize().unwrap();
        let expected_size = bytes.len();

        // Append trailing garbage
        bytes.extend_from_slice(&[0xFF, 0xEE, 0xDD]);

        let (deserialized, consumed) = PropertyMap::deserialize(&bytes).unwrap();

        assert_eq!(deserialized, map);
        assert_eq!(consumed, expected_size);
        assert!(consumed < bytes.len());
    }

    /// 🎯 Target: PropertyMap::deserialize pre-allocation check
    /// 💣 Risk: Large count with small buffer could trigger massive allocation (DoS).
    /// 🧪 Strategy: Construct buffer with large valid count but insufficient data length.
    /// 🔬 Verification: Expect error "Insufficient buffer size".
    #[test]
    fn test_property_map_deserialize_insufficient_buffer_preallocation() {
        let mut bytes = Vec::new();
        let count = 50_000u32; // Valid count (< MAX=100_000) but requires ~250KB buffer
        bytes.extend_from_slice(&count.to_le_bytes());
        // No payload, so buffer is just 4 bytes

        let result = PropertyMap::deserialize(&bytes);
        assert!(result.is_err());
        match result {
            Err(crate::core::error::Error::Storage(StorageError::CorruptedData(msg))) => {
                assert!(
                    msg.contains("Insufficient buffer size"),
                    "Unexpected error message: {}",
                    msg
                );
            }
            _ => panic!("Expected CorruptedData error"),
        }
    }

    #[test]
    fn test_array_max_elements_boundary() {
        // Construct a buffer with exactly MAX_ARRAY_ELEMENTS
        let mut bytes = Vec::new();
        bytes.push(TAG_ARRAY);
        let count = MAX_ARRAY_ELEMENTS as u32;
        bytes.extend_from_slice(&count.to_le_bytes());

        // We can't actually allocate MAX_ARRAY_ELEMENTS (10M) * 1 byte in a test without it being slow/heavy.
        // However, we can check that it passes the initial count check and fails on buffer size check
        // (which is O(1)) OR if we provide enough data, it starts deserializing.
        //
        // So if we provide count = MAX, it should NOT return "exceeds maximum allowed".
        // It might return "Insufficient buffer size" if we don't provide data, which confirms the count passed.

        let result = PropertyValue::deserialize(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Insufficient buffer size"),
            "Should pass max check and fail on buffer size: {}",
            err
        );

        // If we provide count = MAX + 1, it MUST return "exceeds maximum allowed"
        let mut bytes_overflow = Vec::new();
        bytes_overflow.push(TAG_ARRAY);
        let count_overflow = (MAX_ARRAY_ELEMENTS + 1) as u32;
        bytes_overflow.extend_from_slice(&count_overflow.to_le_bytes());

        let result_overflow = PropertyValue::deserialize(&bytes_overflow);
        assert!(result_overflow.is_err());
        let err_overflow = result_overflow.unwrap_err();
        assert!(
            err_overflow.to_string().contains("exceeds maximum allowed"),
            "Should fail max check: {}",
            err_overflow
        );
    }

    #[test]
    fn test_property_map_capacity_boundary() {
        // Similar strategy for PropertyMap
        let mut bytes = Vec::new();
        let count = MAX_PROPERTY_MAP_CAPACITY as u32;
        bytes.extend_from_slice(&count.to_le_bytes());

        // Check boundary exact hit (should fail on buffer size, not capacity limit)
        let result = PropertyMap::deserialize(&bytes);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Insufficient buffer size")
        );

        // Check boundary violation (MAX + 1)
        let mut bytes_overflow = Vec::new();
        let count_overflow = (MAX_PROPERTY_MAP_CAPACITY + 1) as u32;
        bytes_overflow.extend_from_slice(&count_overflow.to_le_bytes());

        let result_overflow = PropertyMap::deserialize(&bytes_overflow);
        assert!(result_overflow.is_err());
        assert!(
            result_overflow
                .unwrap_err()
                .to_string()
                .contains("exceeds maximum allowed")
        );
    }

    #[test]
    fn test_contains_vector_nested() {
        // Document that contains_vector does NOT check nested arrays
        let embedding = vec![0.1f32; 4];
        let vec_val = PropertyValue::vector(&embedding);
        let array_val = PropertyValue::array(vec![vec_val]);

        let map = PropertyMapBuilder::new()
            .insert("nested_vector", array_val)
            .build();

        assert!(
            !map.contains_vector(),
            "contains_vector should ignore nested vectors (current limitation)"
        );
    }

    /// 🎯 Target: PropertyValue::semantically_equal
    /// 💣 Risk: Spurious diffs when values are NaN (because NaN != NaN).
    /// 🧪 Strategy: Compare NaN values using semantically_equal vs PartialEq.
    /// 🔬 Verification: semantically_equal returns true, == returns false (or true if shared).
    #[test]
    fn test_semantically_equal_handles_nan() {
        // Float(NaN) - No optimization for Float (primitive)
        let nan_float = PropertyValue::Float(f64::NAN);
        assert_ne!(nan_float, nan_float, "PartialEq should treat NaN != NaN");
        assert!(
            nan_float.semantically_equal(&nan_float),
            "semantically_equal should treat NaN == NaN"
        );

        // Vector with NaN
        let nan_vec = PropertyValue::vector([1.0f32, f32::NAN, 2.0f32]);

        // OPTIMIZATION CHANGE: Shared pointers are equal
        assert_eq!(
            nan_vec, nan_vec,
            "PartialEq should treat shared vector with NaN == itself"
        );

        // But distinct pointers are NOT equal
        let nan_vec_2 = PropertyValue::vector([1.0f32, f32::NAN, 2.0f32]);
        assert_ne!(
            nan_vec, nan_vec_2,
            "PartialEq should treat distinct vectors with NaN != each other"
        );

        // semantically_equal should be true for both
        assert!(
            nan_vec.semantically_equal(&nan_vec),
            "semantically_equal should treat vector with NaN == itself"
        );
        assert!(
            nan_vec.semantically_equal(&nan_vec_2),
            "semantically_equal should treat distinct vectors with NaN as equal"
        );

        // Mixed types (just to be safe)
        let other = PropertyValue::Int(42);
        assert!(!nan_float.semantically_equal(&other));
    }

    /// 🎯 Target: GLOBAL_INTERNER concurrency
    /// 💣 Risk: Race conditions when interning the same new keys from multiple threads.
    /// 🧪 Strategy: Spawn threads that all insert the SAME set of new keys.
    /// 🔬 Verification: All threads succeed and see correct data.
    #[test]
    fn test_concurrent_interning_stress() {
        use std::sync::Arc;
        use std::thread;

        // Number of threads and keys
        const NUM_THREADS: usize = 10;
        const NUM_KEYS: usize = 100;

        // Barrier to synchronize start
        let barrier = Arc::new(std::sync::Barrier::new(NUM_THREADS));

        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|t_id| {
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();

                    // Each thread inserts the same set of keys, forcing contention on interning
                    let mut builder = PropertyMapBuilder::new();
                    for i in 0..NUM_KEYS {
                        let key = format!("concurrent_stress_key_{}", i);
                        // Also mix in some thread-specific values to ensure no data corruption
                        builder = builder.insert(&key, (t_id * 1000 + i) as i64);
                    }
                    let map = builder.build();

                    // Verify the map
                    assert_eq!(map.len(), NUM_KEYS);
                    for i in 0..NUM_KEYS {
                        let key = format!("concurrent_stress_key_{}", i);
                        assert_eq!(
                            map.get(&key).and_then(|v: &PropertyValue| v.as_int()),
                            Some((t_id * 1000 + i) as i64)
                        );
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    /// 🎯 Target: MAX_VECTOR_DIMENSIONS boundary
    /// 💣 Risk: Off-by-one errors might cause rejection of valid vectors exactly at the limit.
    /// 🧪 Strategy: Serialize a vector with exactly MAX_VECTOR_DIMENSIONS.
    /// 🔬 Verification: Succeeds without panic.
    #[test]
    fn test_serialize_vector_into_at_limit() {
        let max_vector = vec![0.0f32; MAX_VECTOR_DIMENSIONS];
        let mut buffer = Vec::new();
        // Should NOT panic
        serialize_vector_into(&max_vector, &mut buffer);

        // Verify it was serialized correctly
        let (deserialized, _) = deserialize_vector(&buffer).unwrap();
        assert_eq!(deserialized.len(), MAX_VECTOR_DIMENSIONS);
    }

    /// 🎯 Target: PropertyValue::semantically_equal (Arrays)
    /// 💣 Risk: Arrays containing NaN should be semantically equal even if distinct allocations.
    /// 🧪 Strategy: Create two distinct arrays with NaN and compare.
    /// 🔬 Verification: semantically_equal returns true.
    #[test]
    fn test_semantically_equal_nested_array_nan_distinct_allocations() {
        // Create two distinct arrays containing NaN
        let nan_array_1 = PropertyValue::Array(Arc::new(vec![
            PropertyValue::Float(1.0),
            PropertyValue::Float(f64::NAN),
        ]));

        let nan_array_2 = PropertyValue::Array(Arc::new(vec![
            PropertyValue::Float(1.0),
            PropertyValue::Float(f64::NAN),
        ]));

        // Ensure they are NOT pointer equal
        if let (PropertyValue::Array(a), PropertyValue::Array(b)) = (&nan_array_1, &nan_array_2) {
            assert!(!Arc::ptr_eq(a, b), "Arrays should be distinct allocations");
        }

        // It should be semantically equal, treating NaN as equal
        assert!(
            nan_array_1.semantically_equal(&nan_array_2),
            "Distinct arrays containing NaN should be semantically equal"
        );
    }

    #[test]
    fn test_deserialize_recursion_exact_boundary() {
        // Test exactly AT the limit (should succeed)
        // [TAG_ARRAY, 1, TAG_ARRAY, 1, ..., TAG_NULL]
        // Depth 0: Array(1 element) -> Depth 1
        // ...
        // Depth 99: Array(1 element) -> Depth 100
        // Depth 100: Null
        // Total depth = 100 (allowed)

        let mut bytes = Vec::new();
        let depth = MAX_RECURSION_DEPTH;

        for _ in 0..depth {
            bytes.push(TAG_ARRAY);
            bytes.extend_from_slice(&1u32.to_le_bytes());
        }
        bytes.push(TAG_NULL);

        let result = PropertyValue::deserialize(&bytes);
        assert!(
            result.is_ok(),
            "Should succeed at exactly MAX_RECURSION_DEPTH ({})",
            MAX_RECURSION_DEPTH
        );

        // Test exactly ONE OVER the limit (should fail)
        let mut bytes_over = Vec::new();
        let depth_over = MAX_RECURSION_DEPTH + 1;

        for _ in 0..depth_over {
            bytes_over.push(TAG_ARRAY);
            bytes_over.extend_from_slice(&1u32.to_le_bytes());
        }
        bytes_over.push(TAG_NULL);

        let result_over = PropertyValue::deserialize(&bytes_over);
        assert!(
            result_over.is_err(),
            "Should fail at MAX_RECURSION_DEPTH + 1"
        );
        let err = result_over.unwrap_err();
        assert!(
            err.to_string().contains("recursion depth limit exceeded"),
            "Error should be specific: {}",
            err
        );
    }

    #[test]
    fn test_deserialize_buffer_boundary_conditions() {
        // Bool: Needs 2 bytes [TAG, VAL]
        assert!(PropertyValue::deserialize(&[TAG_BOOL]).is_err());
        assert!(PropertyValue::deserialize(&[TAG_BOOL, 0]).is_ok());

        // Int: Needs 9 bytes [TAG, 8 bytes]
        let mut int_bytes = vec![TAG_INT];
        int_bytes.extend_from_slice(&[0u8; 7]); // 8 bytes total
        assert!(PropertyValue::deserialize(&int_bytes).is_err());
        int_bytes.push(0); // 9 bytes total
        assert!(PropertyValue::deserialize(&int_bytes).is_ok());

        // Float: Needs 9 bytes
        let mut float_bytes = vec![TAG_FLOAT];
        float_bytes.extend_from_slice(&[0u8; 7]);
        assert!(PropertyValue::deserialize(&float_bytes).is_err());
        float_bytes.push(0);
        assert!(PropertyValue::deserialize(&float_bytes).is_ok());

        // String: Needs 5 bytes min [TAG, LEN(4)]
        let mut str_bytes = vec![TAG_STRING];
        str_bytes.extend_from_slice(&[0u8; 3]);
        assert!(PropertyValue::deserialize(&str_bytes).is_err()); // 4 bytes
        str_bytes.push(0); // 5 bytes (len=0)
        assert!(PropertyValue::deserialize(&str_bytes).is_ok());

        // String with data: [TAG, LEN=1, DATA] -> 6 bytes
        let mut str_data_bytes = vec![TAG_STRING];
        str_data_bytes.extend_from_slice(&1u32.to_le_bytes());
        assert!(PropertyValue::deserialize(&str_data_bytes).is_err()); // 5 bytes, need 6
        str_data_bytes.push(b'a');
        assert!(PropertyValue::deserialize(&str_data_bytes).is_ok());

        // Bytes: Needs 5 bytes min
        let mut b_bytes = vec![TAG_BYTES];
        b_bytes.extend_from_slice(&[0u8; 3]);
        assert!(PropertyValue::deserialize(&b_bytes).is_err());
        b_bytes.push(0); // len=0
        assert!(PropertyValue::deserialize(&b_bytes).is_ok());

        // Array: Needs 5 bytes min
        let mut arr_bytes = vec![TAG_ARRAY];
        arr_bytes.extend_from_slice(&[0u8; 3]);
        assert!(PropertyValue::deserialize(&arr_bytes).is_err());
        arr_bytes.push(0); // count=0
        assert!(PropertyValue::deserialize(&arr_bytes).is_ok());

        // Vector: Needs 5 bytes (TAG + DIM) + data
        // Empty vector (dim=0): 5 bytes
        let mut vec_bytes = vec![TAG_VECTOR];
        vec_bytes.extend_from_slice(&[0u8; 3]);
        assert!(PropertyValue::deserialize(&vec_bytes).is_err());
        vec_bytes.push(0); // dim=0
        assert!(PropertyValue::deserialize(&vec_bytes).is_ok());

        // Vector with dim=1: 5 + 4 = 9 bytes
        let mut vec_1_bytes = vec![TAG_VECTOR];
        vec_1_bytes.extend_from_slice(&1u32.to_le_bytes()); // dim=1
        vec_1_bytes.extend_from_slice(&[0u8; 3]); // 3 bytes data
        assert!(PropertyValue::deserialize(&vec_1_bytes).is_err()); // 8 bytes
        vec_1_bytes.push(0); // 4 bytes data
        assert!(PropertyValue::deserialize(&vec_1_bytes).is_ok());
    }

    #[test]
    fn test_property_map_duplicate_key_rejection() {
        // Construct a buffer with duplicate keys manually
        // [count: 2][key: "a"][val: 1][key: "a"][val: 2]
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&2u32.to_le_bytes()); // Count: 2

        // Entry 1: "a" -> 1
        let key = "a";
        buffer.extend_from_slice(&(key.len() as u32).to_le_bytes());
        buffer.extend_from_slice(key.as_bytes());
        PropertyValue::Int(1).serialize_into(&mut buffer).unwrap();

        // Entry 2: "a" -> 2 (Duplicate!)
        buffer.extend_from_slice(&(key.len() as u32).to_le_bytes());
        buffer.extend_from_slice(key.as_bytes());
        PropertyValue::Int(2).serialize_into(&mut buffer).unwrap();

        // Deserialization MUST fail
        let result = PropertyMap::deserialize(&buffer);
        assert!(
            result.is_err(),
            "Deserialization must fail on duplicate keys to prevent data corruption/ambiguity"
        );
        match result {
            Err(crate::core::error::Error::Storage(StorageError::CorruptedData(msg))) => {
                assert!(
                    msg.contains("Duplicate property key"),
                    "Error should mention duplicate key"
                );
            }
            _ => panic!("Expected CorruptedData error, got {:?}", result),
        }
    }

    #[test]
    fn test_property_value_partial_eq_nan_semantics() {
        // Ensure PartialEq behaves as expected (IEEE 754: NaN != NaN)
        // This is crucial because some mutants might try to relax this.
        let f1 = PropertyValue::Float(f64::NAN);
        let f2 = PropertyValue::Float(f64::NAN);
        assert_ne!(f1, f2, "PartialEq must respect IEEE 754 NaN != NaN");

        // But semantic equality treats them as equal
        assert!(
            f1.semantically_equal(&f2),
            "semantically_equal should treat NaN as equal"
        );
    }

    #[test]
    #[should_panic(expected = "Property insertion failed (recursion depth limit exceeded)")]
    fn test_property_map_builder_insert_panics_on_deep_recursion() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }
        let _ = PropertyMapBuilder::new().insert("deep", value);
    }

    #[test]
    #[should_panic(expected = "Property insertion failed (recursion depth limit exceeded)")]
    fn test_property_map_builder_insert_by_key_panics_on_deep_recursion() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }
        let key = GLOBAL_INTERNER.intern("deep").unwrap();
        let _ = PropertyMapBuilder::new().insert_by_key(key, value);
    }

    #[test]
    #[should_panic(expected = "Property removal failed")]
    fn test_property_map_builder_remove_panics_on_deep_recursion() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }

        let mut builder = PropertyMapBuilder::new();
        // Insert directly to bypass try_insert's checks
        let key = GLOBAL_INTERNER.intern("deep").unwrap();
        builder.map.insert(key, value);

        let _ = builder.remove("deep");
    }

    #[test]
    #[should_panic(expected = "Property removal failed")]
    fn test_property_map_builder_remove_by_key_panics_on_deep_recursion() {
        let mut value = PropertyValue::Int(42);
        for _ in 0..MAX_RECURSION_DEPTH + 1 {
            value = PropertyValue::Array(Arc::new(vec![value]));
        }

        let mut builder = PropertyMapBuilder::new();
        // Insert directly to bypass try_insert's checks
        let key = GLOBAL_INTERNER.intern("deep").unwrap();
        builder.map.insert(key, value);

        let _ = builder.remove_by_key(&key);
    }
}
