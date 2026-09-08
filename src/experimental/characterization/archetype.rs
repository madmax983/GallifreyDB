//! Archetype: Semantic concept extraction and purity scoring.
//!
//! # The "Nova" Philosophy 🌟
//!
//! **Archetype** answers the question: "What is the Platonic ideal of these nodes, and how well does each node embody it?"
//! By calculating the centroid of a set of vectors (the Archetype), we can score the semantic "purity"
//! of individual members, finding outliers or core representatives.
//!
//! # Use Cases
//! - **Concept Extraction**: Identifying the central theme of a group of articles.
//! - **Anomaly Detection**: Finding nodes that claim to be part of a group but lack semantic similarity.
//! - **Representative Sampling**: Finding the most "pure" examples of a cluster.

use crate::AletheiaDB;
use crate::api::transaction::ReadOps;
use crate::core::error::{Error, Result};
use crate::core::id::NodeId;
use crate::core::vector::ops;

/// Result of an archetype analysis for a specific node.
#[derive(Debug, Clone)]
pub struct ArchetypeNodeResult {
    /// The node analyzed.
    pub node_id: NodeId,
    /// How closely the node aligns with the archetype (1.0 = perfect match, 0.0 = orthogonal/opposite).
    /// Calculated via cosine similarity to the centroid.
    pub purity_score: f32,
}

/// The overall result of extracting an archetype from a set of nodes.
#[derive(Debug, Clone)]
pub struct ArchetypeResult {
    /// The computed "Platonic ideal" vector (centroid).
    pub centroid: Vec<f32>,
    /// The purity scores for all analyzed nodes, sorted by purity (highest first).
    pub nodes: Vec<ArchetypeNodeResult>,
}

/// The Archetype Engine.
pub struct Archetype<'a> {
    db: &'a AletheiaDB,
}

impl<'a> Archetype<'a> {
    /// Create a new Archetype Engine.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Analyze a set of nodes to find their common archetype and score their purity.
    ///
    /// # Arguments
    /// * `nodes` - The list of node IDs to analyze.
    /// * `property_name` - The vector property to use for semantic analysis.
    ///
    /// # Returns
    /// An `ArchetypeResult` containing the centroid and sorted purity scores.
    pub fn analyze(&self, nodes: &[NodeId], property_name: &str) -> Result<ArchetypeResult> {
        if nodes.is_empty() {
            return Err(Error::other("Cannot analyze empty node list"));
        }

        let mut vectors = Vec::new();
        let mut valid_nodes = Vec::new();

        self.db.read(|tx| {
            for &node_id in nodes {
                if let Ok(node) = tx.get_node(node_id) {
                    #[allow(clippy::collapsible_if)]
                    if let Some(prop) = node.get_property(property_name).and_then(|p| p.as_vector())
                    {
                        vectors.push(prop.to_vec());
                        valid_nodes.push(node_id);
                    }
                }
            }
            Ok::<(), Error>(())
        })?;

        if vectors.is_empty() {
            return Err(Error::other(
                "None of the provided nodes have the specified vector property",
            ));
        }

        // 1. Calculate the Centroid (Archetype)
        let mut centroid = Self::average_vectors(&vectors)?;
        ops::normalize_in_place(&mut centroid);

        // 2. Calculate Purity Scores
        let mut node_results = Vec::with_capacity(valid_nodes.len());
        for (i, vec) in vectors.iter().enumerate() {
            let mut normalized_vec = vec.clone();
            ops::normalize_in_place(&mut normalized_vec);

            let similarity = ops::cosine_similarity(&centroid, &normalized_vec)?;
            // Cosine similarity ranges from -1.0 to 1.0. We normalize it to 0.0 - 1.0 for purity.
            let purity_score = ((similarity + 1.0) / 2.0).clamp(0.0, 1.0);

            node_results.push(ArchetypeNodeResult {
                node_id: valid_nodes[i],
                purity_score,
            });
        }

        // Sort by purity descending
        node_results.sort_by(|a, b| b.purity_score.total_cmp(&a.purity_score));

        Ok(ArchetypeResult {
            centroid,
            nodes: node_results,
        })
    }

    /// Helper to average a list of vectors.
    fn average_vectors(vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Err(Error::other("Cannot average empty vector list"));
        }

        let dim = vectors[0].len();
        let mut sum = vec![0.0; dim];

        for vec in vectors {
            if vec.len() != dim {
                return Err(Error::other("Vector dimensions do not match"));
            }
            for i in 0..dim {
                sum[i] += vec[i];
            }
        }

        let count = vectors.len() as f32;
        for val in &mut sum {
            *val /= count;
        }

        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::transaction::WriteOps;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_archetype_centroid_calculation() {
        let db = AletheiaDB::new().unwrap();

        let mut n1 = NodeId::new(0).unwrap();
        let mut n2 = NodeId::new(0).unwrap();

        db.write(|tx| {
            n1 = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("embedding", &[1.0, 0.0])
                        .build(),
                )
                .unwrap();

            n2 = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("embedding", &[0.0, 1.0])
                        .build(),
                )
                .unwrap();
            Ok::<(), Error>(())
        })
        .unwrap();

        let archetype = Archetype::new(&db);
        let result = archetype.analyze(&[n1, n2], "embedding").unwrap();

        // The centroid of [1.0, 0.0] and [0.0, 1.0] should be [0.5, 0.5] before normalization
        // After normalization, it should be approximately [0.707, 0.707]
        let expected_val = 0.5_f32.sqrt(); // 1 / sqrt(2)

        assert!((result.centroid[0] - expected_val).abs() < 0.001);
        assert!((result.centroid[1] - expected_val).abs() < 0.001);
    }

    #[test]
    fn test_node_purity_score() {
        let db = AletheiaDB::new().unwrap();

        let mut center = NodeId::new(0).unwrap();
        let mut similar1 = NodeId::new(0).unwrap();
        let mut outlier = NodeId::new(0).unwrap();

        db.write(|tx| {
            // These two form the core idea
            center = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("embedding", &[1.0, 0.0])
                        .build(),
                )
                .unwrap();

            similar1 = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("embedding", &[0.9, 0.1])
                        .build(),
                )
                .unwrap();

            // This one is completely different
            outlier = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("embedding", &[-1.0, 0.0])
                        .build(),
                )
                .unwrap();
            Ok::<(), Error>(())
        })
        .unwrap();

        let archetype = Archetype::new(&db);
        let result = archetype
            .analyze(&[center, similar1, outlier], "embedding")
            .unwrap();

        assert_eq!(result.nodes.len(), 3);

        // Find the scores
        let score_center = result
            .nodes
            .iter()
            .find(|n| n.node_id == center)
            .unwrap()
            .purity_score;
        let score_similar = result
            .nodes
            .iter()
            .find(|n| n.node_id == similar1)
            .unwrap()
            .purity_score;
        let score_outlier = result
            .nodes
            .iter()
            .find(|n| n.node_id == outlier)
            .unwrap()
            .purity_score;

        // Center and similar should have high purity
        assert!(score_center > 0.8);
        assert!(score_similar > 0.8);

        // Outlier should have very low purity
        assert!(score_outlier < 0.2);

        // Nodes should be sorted by purity (highest first)
        assert_eq!(result.nodes[2].node_id, outlier);
    }

    #[test]
    fn test_archetype_nan_vectors_do_not_panic() {
        let db = AletheiaDB::new().unwrap();

        let mut n1 = NodeId::new(0).unwrap();
        let mut n2 = NodeId::new(0).unwrap();

        db.write(|tx| {
            n1 = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("embedding", &[1.0, f32::NAN])
                        .build(),
                )
                .unwrap();

            n2 = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("embedding", &[f32::NAN, 1.0])
                        .build(),
                )
                .unwrap();
            Ok::<(), Error>(())
        })
        .unwrap();

        let archetype = Archetype::new(&db);
        let result = archetype.analyze(&[n1, n2], "embedding").unwrap();

        assert_eq!(result.nodes.len(), 2);
    }
}
