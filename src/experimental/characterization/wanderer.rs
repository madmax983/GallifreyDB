//! Wanderer: Semantic Random Walk
//!
//! "Drift through the graph."
//!
//! The Wanderer performs a semantic random walk through the graph. At each step,
//! it evaluates the outgoing neighbors of the current node and computes their
//! vector similarity to a target concept. It then moves probabilistically, where
//! the transition probabilities are modulated by a temperature parameter.

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::vector::cosine_similarity;
use rand::Rng;

/// The Wanderer Engine.
#[cfg(feature = "semantic-characterization")]
pub struct Wanderer<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Wanderer<'a> {
    /// Create a new Wanderer engine.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Drifts through the graph starting at `start_node`, moving to neighbors
    /// probabilistically based on their similarity to the `target_embedding`.
    ///
    /// Higher `temperature` makes the walk more random, while lower `temperature`
    /// makes it strictly follow the gradient of similarity.
    pub fn drift(
        &self,
        start_node: NodeId,
        target_embedding: &[f32],
        vector_property: &str,
        steps: usize,
        temperature: f32,
    ) -> Result<Vec<NodeId>> {
        let mut path = vec![start_node];
        let mut current_node = start_node;
        let mut rng = rand::thread_rng();

        for _ in 0..steps {
            let edges = self.db.get_outgoing_edges(current_node);
            if edges.is_empty() {
                break;
            }

            let mut neighbors = Vec::new();
            let mut similarities = Vec::new();

            for edge_id in edges {
                if let Ok(edge) = self.db.get_edge(edge_id) {
                    let target = edge.target;
                    let node = self.db.get_node(target)?;

                    #[allow(clippy::collapsible_if)]
                    if let Some(prop) = node.get_property(vector_property) {
                        if let Some(vec) = prop.as_vector() {
                            let sim = cosine_similarity(target_embedding, vec)?;
                            neighbors.push(target);
                            similarities.push(sim);
                        }
                    }
                }
            }

            if neighbors.is_empty() {
                break;
            }

            // Softmax with temperature
            let mut exp_sims: Vec<f32> = similarities
                .iter()
                .map(|s| (s / temperature).exp())
                .collect();

            let sum_exp: f32 = exp_sims.iter().sum();

            // Normalize probabilities
            for p in &mut exp_sims {
                *p /= sum_exp;
            }

            // Cumulative probabilities for sampling
            let mut cum_prob = 0.0;
            let rand_val: f32 = rng.r#gen();
            let mut selected = neighbors[0];

            for (i, p) in exp_sims.iter().enumerate() {
                cum_prob += p;
                if rand_val <= cum_prob {
                    selected = neighbors[i];
                    break;
                }
            }

            path.push(selected);
            current_node = selected;
        }

        Ok(path)
    }
}

#[cfg(all(test, feature = "semantic-characterization"))]
mod tests {
    use super::*;
    use crate::api::transaction::WriteOps;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_wanderer_drift() {
        let db = AletheiaDB::new().unwrap();

        let mut n1 = NodeId::new(1).unwrap();
        let mut n2 = NodeId::new(2).unwrap();
        let mut n3 = NodeId::new(3).unwrap();

        db.write(|tx| {
            n1 = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("vec", &[1.0, 0.0])
                        .build(),
                )
                .unwrap();

            n2 = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("vec", &[0.9, 0.1])
                        .build(),
                )
                .unwrap();

            n3 = tx
                .create_node(
                    "Node",
                    PropertyMapBuilder::new()
                        .insert_vector("vec", &[0.0, 1.0])
                        .build(),
                )
                .unwrap();

            tx.create_edge(n1, n2, "LINK", Default::default()).unwrap();
            tx.create_edge(n1, n3, "LINK", Default::default()).unwrap();

            Ok::<(), crate::core::error::Error>(())
        })
        .unwrap();

        let wanderer = Wanderer::new(&db);

        // Target is closest to n2. With very low temperature, it should almost certainly pick n2.
        let path = wanderer.drift(n1, &[1.0, 0.0], "vec", 1, 0.01).unwrap();

        assert_eq!(path.len(), 2);
        assert_eq!(path[0], n1);
        assert_eq!(path[1], n2);
    }
}
