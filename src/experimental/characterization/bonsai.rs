//! Bonsai: Semantic Graph Pruner.
//!
//! "Trim the noise, keep the essence."
//!
//! Bonsai traverses an ego-graph but actively prunes branches where the
//! semantic meaning drifts too far from the root node. It evaluates the
//! vector similarity between the root node and traversed nodes, cutting
//! off paths that fall below a given similarity threshold.
//!
//! This is extremely useful for generating tight, focused context windows
//! for LLMs, ensuring that graph traversal doesn't wander off-topic.

use crate::AletheiaDB;
use crate::core::error::{Error, Result};
use crate::core::id::NodeId;
use crate::core::vector::cosine_similarity;
use std::collections::{HashSet, VecDeque};

/// The Bonsai Semantic Pruning Engine.
pub struct Bonsai<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Bonsai<'a> {
    /// Create a new Bonsai pruner.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Prunes the graph starting from `root_node`.
    ///
    /// Traverses up to `max_depth`. A node is only included (and its children explored)
    /// if its vector similarity to the `root_node` is >= `similarity_threshold`.
    /// Nodes without the specified vector property are evaluated based on `keep_missing`.
    pub fn prune(
        &self,
        root_node: NodeId,
        vector_property: &str,
        similarity_threshold: f32,
        max_depth: usize,
        keep_missing: bool,
    ) -> Result<HashSet<NodeId>> {
        let mut included = HashSet::new();
        let mut queue = VecDeque::new();
        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();

        let root = self.db.get_node(root_node)?;

        let root_vector = root
            .get_property(vector_property)
            .and_then(|v| v.as_vector());
        if root_vector.is_none() && !keep_missing {
            return Err(Error::other(format!(
                "Root node missing vector property '{}'",
                vector_property
            )));
        }

        queue.push_back((root_node, 0));
        visited_nodes.insert(root_node);
        included.insert(root_node);

        while let Some((current_node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let edges = self.db.get_outgoing_edges(current_node);
            for edge_id in edges {
                if !visited_edges.insert(edge_id) {
                    continue;
                }

                if let Ok(edge) = self.db.get_edge(edge_id) {
                    let target = edge.target;

                    if !visited_nodes.insert(target) {
                        continue;
                    }

                    let target_node = self.db.get_node(target)?;
                    let target_vector = target_node
                        .get_property(vector_property)
                        .and_then(|v| v.as_vector());

                    let should_include = match (root_vector.as_ref(), target_vector) {
                        (Some(rv), Some(tv)) => match cosine_similarity(rv, tv) {
                            Ok(sim) => sim >= similarity_threshold,
                            Err(_) => false,
                        },
                        _ => keep_missing,
                    };

                    if should_include {
                        included.insert(target);
                        queue.push_back((target, depth + 1));
                    }
                }
            }
        }

        Ok(included)
    }
}

#[cfg(all(test, feature = "semantic-characterization"))]
mod tests {
    use super::*;
    use crate::WriteOps;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_bonsai_pruning() {
        let db = AletheiaDB::new().unwrap();

        let root = db
            .write(|tx| {
                let n1 = tx
                    .create_node(
                        "Concept",
                        PropertyMapBuilder::new()
                            .insert("name", "Artificial Intelligence")
                            .insert_vector("embedding", &[1.0, 0.0, 0.0])
                            .build(),
                    )
                    .unwrap();

                let n2 = tx
                    .create_node(
                        "Concept",
                        PropertyMapBuilder::new()
                            .insert("name", "Machine Learning")
                            .insert_vector("embedding", &[0.99, 0.14, 0.0])
                            .build(),
                    )
                    .unwrap();

                let n3 = tx
                    .create_node(
                        "Concept",
                        PropertyMapBuilder::new()
                            .insert("name", "Baking Bread")
                            .insert_vector("embedding", &[0.0, 1.0, 0.0])
                            .build(),
                    )
                    .unwrap();

                let n4 = tx
                    .create_node(
                        "Concept",
                        PropertyMapBuilder::new()
                            .insert("name", "Deep Learning")
                            .insert_vector("embedding", &[0.95, 0.31, 0.0])
                            .build(),
                    )
                    .unwrap();

                tx.create_edge(n1, n2, "RELATES_TO", Default::default())
                    .unwrap();
                tx.create_edge(n2, n3, "TANGENT", Default::default())
                    .unwrap();
                tx.create_edge(n2, n4, "RELATES_TO", Default::default())
                    .unwrap();

                Ok::<_, crate::core::error::Error>(n1)
            })
            .unwrap();

        let bonsai = Bonsai::new(&db);

        // Threshold 0.9: Should include AI (1.0), ML (0.99), DL (0.95).
        // Bread (0.0) should be pruned.
        let pruned = bonsai.prune(root, "embedding", 0.9, 3, false).unwrap();

        assert_eq!(pruned.len(), 3);
    }
}
