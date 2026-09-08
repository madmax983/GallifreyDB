//! Experimental module for generating human-readable narratives from temporal history.
//!
//! Converts raw graph mutations into readable event summaries.
use crate::AletheiaDB;
#[cfg(feature = "semantic-temporal")]
use crate::core::GLOBAL_INTERNER;
use crate::core::error::Result;
#[cfg(feature = "semantic-temporal")]
use crate::core::history::{VersionDiff, VersionInfo};
use crate::core::id::NodeId;
#[cfg(feature = "semantic-temporal")]
use crate::core::interning::InternedString;
#[cfg(feature = "semantic-temporal")]
use crate::core::temporal::time;

/// A single event in the narrative history of an entity.
#[derive(Debug, Clone)]
pub struct NarrativeEvent {
    /// ISO 8601 timestamp of when the event was recorded (transaction time).
    pub timestamp: String,
    /// Sequential version number.
    pub version_number: u64,
    /// High-level description of what happened.
    pub description: String,
    /// Detailed list of changes (if any).
    pub changes: Vec<String>,
}

/// Generator for creating natural language narratives from temporal history.
#[cfg(feature = "semantic-temporal")]
/// Generates human-readable summaries of temporal graph mutations.
///
/// # Why?
/// When analyzing a `BiTemporalInterval`, users often need to understand
/// the sequence of events (e.g., "Alice liked the post, then Bob deleted it").
/// This struct translates raw `VersionMetadata` into a chronological narrative.
pub struct NarrativeGenerator<'a> {
    db: &'a AletheiaDB,
}

#[cfg(not(feature = "semantic-temporal"))]
/// Generator for creating natural language narratives from temporal history.
#[deprecated(
    note = "NarrativeGenerator requires the 'nova' feature. Add 'features = [\"nova\"]' to your Cargo.toml."
)]
/// Generates human-readable summaries of temporal graph mutations.
///
/// # Why?
/// When analyzing a `BiTemporalInterval`, users often need to understand
/// the sequence of events (e.g., "Alice liked the post, then Bob deleted it").
/// This struct translates raw `VersionMetadata` into a chronological narrative.
pub struct NarrativeGenerator<'a> {
    _marker: std::marker::PhantomData<&'a AletheiaDB>,
}

#[cfg(feature = "semantic-temporal")]
impl<'a> NarrativeGenerator<'a> {
    /// Create a new narrative generator.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Helper to resolve interned keys to strings.
    fn resolve_key(key_id: InternedString) -> String {
        GLOBAL_INTERNER
            .resolve_with(key_id, |s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Generate a narrative for a specific node.
    ///
    /// This reconstructs the history of the node and generates a sequence of
    /// human-readable events describing how it evolved over time.
    ///
    /// # Behavior for Empty History
    /// If a node has no history (e.g. invalid ID or deleted), this returns an empty vector
    /// or an error if the node ID is invalid, consistent with `get_node_history`.
    ///
    /// # Property Removal
    /// Properties set to `null` are interpreted as "Removed property".
    ///
    /// Note: Edge narrative generation is planned for a future release.
    pub fn generate_node_narrative(&self, node_id: NodeId) -> Result<Vec<NarrativeEvent>> {
        let history = self.db.get_node_history(node_id)?;
        let mut events = Vec::new();

        let mut prev_version: Option<&VersionInfo> = None;

        for version in &history.versions {
            let timestamp = time::to_iso8601(version.temporal.transaction_time().start());
            let version_number = version.version_number;
            let mut changes = Vec::new();
            let description;

            if let Some(prev) = prev_version {
                // Determine changes between versions
                let diff = VersionDiff::compute(
                    &prev.properties,
                    &version.properties,
                    prev.version_id,
                    version.version_id,
                );

                description = format!("Version {} updated properties.", version_number);

                // Pre-allocate vector to avoid frequent reallocations
                changes.reserve(diff.added.len() + diff.removed.len() + diff.modified.len());

                // Added properties
                for (key_id, val) in diff.added.iter() {
                    let key = Self::resolve_key(*key_id);
                    changes.push(format!("Added property '{}' with value '{}'", key, val));
                }

                // Removed properties (explicitly missing keys)
                for (key_id, val) in diff.removed.iter() {
                    let key = Self::resolve_key(*key_id);
                    changes.push(format!("Removed property '{}' (was '{}')", key, val));
                }

                // Modified properties
                for (key_id, old_val, new_val) in &diff.modified {
                    let key = Self::resolve_key(*key_id);
                    if new_val.is_null() {
                        // Treat setting to null as removal
                        changes.push(format!("Removed property '{}' (was '{}')", key, old_val));
                    } else {
                        changes.push(format!(
                            "Modified property '{}' from '{}' to '{}'",
                            key, old_val, new_val
                        ));
                    }
                }
            } else {
                // First version (Creation)
                description = format!("Node created with label '{}'.", version.label);
                changes.reserve(version.properties.len());
                for (key_id, val) in version.properties.iter() {
                    let key = Self::resolve_key(*key_id);
                    changes.push(format!("Initial property '{}': '{}'", key, val));
                }
            }

            events.push(NarrativeEvent {
                timestamp,
                version_number,
                description,
                changes,
            });

            prev_version = Some(version);
        }

        Ok(events)
    }
}

#[cfg(all(test, feature = "semantic-temporal"))]
mod tests {
    use super::*;
    use crate::api::transaction::WriteOps;
    use crate::core::property::{PropertyMapBuilder, PropertyValue};

    #[test]
    fn test_node_narrative_generation() {
        let db = AletheiaDB::new().unwrap();

        // 1. Create Node
        let props1 = PropertyMapBuilder::new()
            .insert("name", "Alice")
            .insert("age", 30i64)
            .build();
        let node_id = db.create_node("Person", props1).unwrap();

        // 2. Update Node
        db.write(|tx| {
            let props2 = PropertyMapBuilder::new()
                .insert("name", "Alice")
                .insert("age", 31i64)
                .insert("city", "London")
                .build();
            tx.update_node(node_id, props2)
        })
        .unwrap();

        // 3. Generate Narrative
        let generator = NarrativeGenerator::new(&db);
        let narrative = generator.generate_node_narrative(node_id).unwrap();

        assert_eq!(narrative.len(), 2);

        // Verify First Event (Creation)
        let event1 = &narrative[0];
        assert_eq!(event1.version_number, 1);
        assert!(event1.description.contains("Node created"));
        // PropertyValue::String display format is "Alice" (quoted)
        assert!(
            event1
                .changes
                .iter()
                .any(|s| s.contains("Initial property 'name': '\"Alice\"'"))
        );
        assert!(
            event1
                .changes
                .iter()
                .any(|s| s.contains("Initial property 'age': '30'"))
        );

        // Verify Second Event (Update)
        let event2 = &narrative[1];
        assert_eq!(event2.version_number, 2);
        assert!(event2.description.contains("updated properties"));

        // age changed
        assert!(
            event2
                .changes
                .iter()
                .any(|s| s.contains("Modified property 'age' from '30' to '31'"))
        );
        // city added
        assert!(
            event2
                .changes
                .iter()
                .any(|s| s.contains("Added property 'city' with value '\"London\"'"))
        );
    }

    #[test]
    fn test_property_removal_narrative() {
        let db = AletheiaDB::new().unwrap();

        // 1. Create Node with properties
        let props1 = PropertyMapBuilder::new()
            .insert("name", "Bob")
            .insert("temp_data", "delete_me")
            .build();
        let node_id = db.create_node("Person", props1).unwrap();

        // 2. Remove property by setting to Null (Simulated removal)
        db.write(|tx| {
            let props2 = PropertyMapBuilder::new()
                .insert("temp_data", PropertyValue::Null)
                .build();
            tx.update_node(node_id, props2)
        })
        .unwrap();

        // 3. Generate Narrative
        let generator = NarrativeGenerator::new(&db);
        let narrative = generator.generate_node_narrative(node_id).unwrap();

        assert_eq!(narrative.len(), 2);

        let event2 = &narrative[1];
        assert!(
            event2
                .changes
                .iter()
                .any(|s| s.contains("Removed property 'temp_data'")),
            "Expected removal message for temp_data, found: {:?}",
            event2.changes
        );
        assert!(
            event2
                .changes
                .iter()
                .any(|s| s.contains("was '\"delete_me\"'")),
            "Expected original value in removal message"
        );
    }
}
