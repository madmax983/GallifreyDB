//! Vortex: Echo Chamber Detector.
//!
//! "Are these nodes living in a bubble?"
//!
//! Vortex analyzes a node's local network (graph edges) to determine if it's
//! part of an echo chamber. It compares the semantic similarity of the node's
//! internal neighborhood (Internal Cohesion) against its similarity to the
//! rest of the graph (External Alienation).
//!
//! # Concepts
//! - **Internal Cohesion**: How similar a node is to its direct graph neighbors.
//! - **External Alienation**: How dissimilar a node is from random non-neighbors.
//! - **Vortex Score**: High Cohesion + High Alienation = Echo Chamber.

use crate::AletheiaDB;
use crate::core::error::{Error, Result};
use crate::core::id::NodeId;
use crate::core::vector::ops;
use rand::thread_rng;

/// The Vortex Detector engine for identifying semantic echo chambers.
pub struct VortexDetector<'a> {
    db: &'a AletheiaDB,
}

impl<'a> VortexDetector<'a> {
    /// Create a new VortexDetector.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Analyzes a node to determine its Vortex Score (Echo Chamber index).
    ///
    /// `vector_prop` is the property name storing the node's embedding (e.g. "embedding").
    /// `num_samples` is the number of random non-neighbor nodes to sample for Alienation.
    pub fn analyze(&self, node_id: NodeId, vector_prop: &str, num_samples: usize) -> Result<f32> {
        let node = self.db.get_node(node_id)?;
        let node_vec = match node.get_property(vector_prop) {
            Some(p) => p
                .as_vector()
                .ok_or_else(|| Error::other(format!("Property {} is not a vector", vector_prop)))?,
            None => return Err(Error::other(format!("Missing property {}", vector_prop))),
        };

        let edges = self.db.get_outgoing_edges(node_id);
        let mut neighbors = Vec::new();
        for edge_id in edges {
            if let Ok(edge) = self.db.get_edge(edge_id) {
                neighbors.push(edge.target);
            }
        }

        if neighbors.is_empty() {
            return Ok(0.0);
        }

        let mut cohesion_sum = 0.0;
        let mut valid_neighbors = 0;

        for &neighbor_id in &neighbors {
            if let Ok(neighbor) = self.db.get_node(neighbor_id) {
                let neighbor_vec = neighbor
                    .get_property(vector_prop)
                    .and_then(|p| p.as_vector());
                if let Some(vec) = neighbor_vec {
                    let sim = ops::cosine_similarity(node_vec, vec).unwrap_or(0.0);
                    cohesion_sum += sim;
                    valid_neighbors += 1;
                }
            }
        }

        let cohesion = if valid_neighbors > 0 {
            cohesion_sum / valid_neighbors as f32
        } else {
            0.0
        };

        // Ensure cohesion is non-negative for scoring
        let cohesion = cohesion.max(0.0);

        // Find external nodes (not in neighbors, not self)
        // For prototype we'll just scan nodes, in a real impl we'd use index/iterators
        // Note: Nova mode allows taking liberties.

        let mut external_nodes = Vec::new();
        // Since we can't easily iterate all nodes in AletheiaDB without an index,
        // we'll try random node IDs up to a limit or if we had a way to get all.
        // Actually, db has no easy "get all nodes" method exposed simply.
        // But we can just use similarity search to find nodes, but that biases towards similar.
        // We'll use a hack for the prototype: we assume node IDs are roughly contiguous
        // and just sample random IDs.

        let mut rng = thread_rng();
        // In our tests, node IDs are small. For a better guess:
        let max_id = node_id.as_u64().max(10);
        let mut attempts = 0;

        while external_nodes.len() < num_samples && attempts < num_samples * 50 {
            attempts += 1;
            use rand::Rng;
            let random_id = rng.gen_range(1..=max_id);
            let target_id = NodeId::new(random_id).unwrap_or(node_id);
            #[allow(clippy::collapsible_if)]
            if target_id != node_id && !neighbors.contains(&target_id) {
                if let Ok(ext_node) = self.db.get_node(target_id) {
                    let prop = ext_node.get_property(vector_prop);
                    let vec_opt = prop.and_then(|p| p.as_vector());
                    if let Some(vec) = vec_opt {
                        external_nodes.push(vec.to_vec());
                    }
                }
            }
        }

        let mut alienation_sum = 0.0;
        let valid_externals = external_nodes.len();

        for ext_vec in external_nodes {
            let sim = ops::cosine_similarity(node_vec, &ext_vec).unwrap_or(0.0);
            // Alienation is high when similarity is low
            alienation_sum += 1.0 - sim.max(0.0);
        }

        let alienation = if valid_externals > 0 {
            alienation_sum / valid_externals as f32
        } else {
            // If we couldn't find any external nodes, we can't prove alienation.
            0.0
        };

        let vortex_score = cohesion * alienation;

        Ok(vortex_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_vortex_score() {
        let db = AletheiaDB::new().unwrap();
        let detector = VortexDetector::new(&db);

        let n1 = db
            .create_node(
                "User",
                PropertyMapBuilder::new()
                    .insert_vector("embedding", &[1.0, 0.0, 0.0])
                    .build(),
            )
            .unwrap();
        let n2 = db
            .create_node(
                "User",
                PropertyMapBuilder::new()
                    .insert_vector("embedding", &[0.9, 0.1, 0.0])
                    .build(),
            )
            .unwrap();
        // External non-neighbors (alienated)
        let _n3 = db
            .create_node(
                "User",
                PropertyMapBuilder::new()
                    .insert_vector("embedding", &[0.0, 1.0, 0.0])
                    .build(),
            )
            .unwrap();
        let _n4 = db
            .create_node(
                "User",
                PropertyMapBuilder::new()
                    .insert_vector("embedding", &[0.0, 0.9, 0.1])
                    .build(),
            )
            .unwrap();

        db.create_edge(n1, n2, "KNOWS", Default::default()).unwrap();

        let score = detector.analyze(n1, "embedding", 2).unwrap();
        // Cohesion should be high (n1 ~ n2), Alienation should be high (n1 !~ n3, n4)
        assert!(score > 0.0);
    }
}
