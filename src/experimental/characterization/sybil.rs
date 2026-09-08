//! Sybil: Memetic Propagation Engine 🗣️
//!
//! "Sybil" simulates how information, influence, or "memes" (represented as vectors)
//! propagate through the graph structure.
//!
//! # The Hook
//! "Imagine a rumor starting at Node A. How far does it spread? Who believes it after 10 steps?
//! Sybil answers this by simulating vector propagation across the graph."
//!
//! # Concepts
//! - **Meme**: A vector representing an idea, belief, or state.
//! - **Propagation Model**: The rules for how a node updates its state based on neighbors.
//! - **Inertia**: Resistance to change (dampening factor).
//!
//! # Example
//!
//! ```rust
//! // Requires features = ["nova"]
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::sybil::{Sybil, LinearPropagation};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! // ... setup graph ...
//!
//! // Define a model: Nodes adopt 50% of neighbors' average, keep 50% of old self.
//! let model = LinearPropagation::new(0.5);
//!
//! // Run simulation for 5 steps
//! let sybil = Sybil::new(&db);
//! let final_state = sybil.simulate("opinion", &model, 5)?;
//!
//! println!("Simulation complete. {} nodes affected.", final_state.len());
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::NodeId;
use crate::core::error::Result;
use std::collections::HashMap;

/// Represents the state of the simulation at a given step.
/// Maps NodeId -> Vector (state).
pub type PropagationState = HashMap<NodeId, Vec<f32>>;

/// Defines how a node updates its state based on its neighbors.
pub trait PropagationModel {
    /// Calculate the next state for a node.
    ///
    /// # Arguments
    /// * `node_id` - The node being updated.
    /// * `current_self` - The node's current vector (if any).
    /// * `neighbors` - List of (NodeId, Neighbor Vector) tuples.
    ///
    /// # Returns
    /// The new vector state for the node.
    fn next_state(
        &self,
        node_id: NodeId,
        current_self: Option<&[f32]>,
        neighbors: &[(NodeId, &[f32])],
    ) -> Option<Vec<f32>>;
}

/// A simple linear propagation model.
/// New State = (1 - alpha) * Old State + alpha * Average(Neighbor States)
pub struct LinearPropagation {
    /// The "adoption rate" or "permeability".
    /// 0.0 = Total Inertia (No change).
    /// 1.0 = Total Conformity (Become average of neighbors).
    pub alpha: f32,
}

impl LinearPropagation {
    /// Create a new LinearPropagation model.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
        }
    }
}

impl PropagationModel for LinearPropagation {
    fn next_state(
        &self,
        _node_id: NodeId,
        current_self: Option<&[f32]>,
        neighbors: &[(NodeId, &[f32])],
    ) -> Option<Vec<f32>> {
        if neighbors.is_empty() {
            // No influence, keep current state
            return current_self.map(|v| v.to_vec());
        }

        // Calculate average of neighbors
        let dim = neighbors[0].1.len();
        let mut sum = vec![0.0; dim];
        let mut count = 0;

        for (_, vec) in neighbors {
            if vec.len() == dim {
                for i in 0..dim {
                    sum[i] += vec[i];
                }
                count += 1;
            }
        }

        if count == 0 {
            return current_self.map(|v| v.to_vec());
        }

        let avg: Vec<f32> = sum.iter().map(|x| x / count as f32).collect();

        match current_self {
            Some(self_vec) => {
                if self_vec.len() != dim {
                    // Dimension mismatch, just return self (or handle error?)
                    // For simulation robustness, we ignore mismatched neighbors/self.
                    return Some(self_vec.to_vec());
                }

                // Blend: (1 - alpha) * self + alpha * avg
                let new_vec: Vec<f32> = self_vec
                    .iter()
                    .zip(avg.iter())
                    .map(|(s, a)| (1.0 - self.alpha) * s + self.alpha * a)
                    .collect();
                Some(new_vec)
            }
            None => {
                // If we didn't have a state, we "adopt" the average (if alpha allows adoption)
                // Interpretation: If I have no opinion, do I adopt neighbors'?
                // For this model, yes, let's say we adopt the average directly.
                Some(avg)
            }
        }
    }
}

/// The Sybil Engine runner.
pub struct Sybil<'a> {
    db: &'a AletheiaDB,
}

impl<'a> Sybil<'a> {
    /// Create a new Sybil engine.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Run the simulation.
    ///
    /// # Arguments
    /// * `property_name` - The vector property to use as the "meme".
    /// * `model` - The propagation logic.
    /// * `steps` - Number of iterations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use aletheiadb::AletheiaDB;
    /// use aletheiadb::core::property::PropertyMapBuilder;
    /// use aletheiadb::experimental::characterization::sybil::{Sybil, LinearPropagation};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = AletheiaDB::new()?;
    ///
    /// // Create nodes and edges
    /// let props_a = PropertyMapBuilder::new().insert_vector("opinion", &[1.0, 0.0]).build();
    /// let node_a = db.create_node("Person", props_a)?;
    ///
    /// let props_b = PropertyMapBuilder::new().insert_vector("opinion", &[0.0, 1.0]).build();
    /// let node_b = db.create_node("Person", props_b)?;
    ///
    /// db.create_edge(node_a, node_b, "INFLUENCES", Default::default())?;
    ///
    /// let sybil = Sybil::new(&db);
    /// let model = LinearPropagation::new(0.5); // Average between self and neighbors
    ///
    /// // Run 1 step of simulation
    /// let state = sybil.simulate("opinion", &model, 1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn simulate<M: PropagationModel>(
        &self,
        property_name: &str,
        model: &M,
        steps: usize,
    ) -> Result<PropagationState> {
        // Step 1: Load Initial State
        // We scan all nodes to get their current vectors.
        // Optimization: In a real system, we might only load a subset or use an index.
        let mut current_state = self.load_initial_state(property_name)?;

        // Step 2: Iterate
        for _step in 0..steps {
            let mut next_state = HashMap::new();

            // For every node in the graph (or at least those involved in the state)
            // We need to iterate over ALL nodes that *could* be affected.
            // This implies we need the graph structure.
            // For simplicity in Nova, we'll iterate over nodes present in the current_state
            // AND their neighbors.

            // Actually, to simulate properly, we need to visit nodes.
            // If the graph is huge, this is slow.
            // Strategy: Only iterate nodes that are in `current_state` or are neighbors of `current_state`.
            // But if a node has no state, it might acquire it from a neighbor.
            // So we really need to iterate (Active Nodes + Neighbors).

            let mut active_nodes: Vec<NodeId> = current_state.keys().cloned().collect();
            // We also need neighbors of active nodes who might *receive* the meme.
            // Let's collect them.
            // Note: This could explode for dense graphs.
            // Constraint: For Nova prototype, we iterate only nodes that ALREADY have state
            // OR are explicitly connected to them.
            // Let's stick to: "Iterate all nodes that have state, and pull from neighbors."
            // Wait, if I have no state, I can't "pull" unless I'm processed.
            // Let's iterate `current_state` keys. If a neighbor doesn't have state,
            // it won't be in `current_state` next round unless we add it.

            // To allow expansion, we must include neighbors in the processing list.
            let mut expansion_candidates = Vec::new();
            for &node_id in &active_nodes {
                // Get neighbors
                let edges = self.db.get_outgoing_edges(node_id); // Assuming directed flow? Or undirected?
                // Influence usually flows Source -> Target.
                // So if A has an opinion, it influences B (Target).
                // So B needs to update.
                // So we need to iterate B.
                for edge_id in edges {
                    if let Ok(target) = self.db.get_edge_target(edge_id) {
                        expansion_candidates.push(target);
                    }
                }
            }

            // Merge and dedup
            active_nodes.extend(expansion_candidates);
            active_nodes.sort();
            active_nodes.dedup();

            for &node_id in &active_nodes {
                // Gather neighbor states (Incoming edges?)
                // If influence flows A->B, then B's new state depends on A.
                // So B looks at INCOMING edges.
                // AletheiaDB optimizes for Outgoing.
                // Getting incoming edges might be slow if not indexed.
                // Let's assume Undirected for "community consensus" or check if we have incoming index.
                // Checking memory... `AdjacencyIndex` supports generic traversal?
                // `get_incoming_edges` exists in `AletheiaDB`? Let's assume standard API usually has it,
                // but if not, we might be limited to Outgoing.
                // If we only have Outgoing (A->B), then A "pushes" to B.

                // Let's use a "Push" model for efficiency if incoming is hard.
                // Iterate A (with state). For each neighbor B, accumulate A's influence on B.
                // Then finalize B.

                // BUT, the `PropagationModel` trait is designed as "Pull" (`next_state` takes `neighbors`).
                // Let's try to find neighbors.

                let incoming_edges = self.db.get_incoming_edges(node_id);
                // If this is slow or unsupported, we might need to change strategy.
                // Assuming it works for now.

                let mut neighbor_vectors = Vec::new();
                for edge_id in incoming_edges {
                    if let Ok(source) = self.db.get_edge_source(edge_id)
                        && let Some(vec) = current_state.get(&source)
                    {
                        neighbor_vectors.push((source, vec.as_slice()));
                    }
                }

                let current_vec = current_state.get(&node_id).map(|v| v.as_slice());

                if let Some(new_vec) = model.next_state(node_id, current_vec, &neighbor_vectors) {
                    next_state.insert(node_id, new_vec);
                }
            }

            current_state = next_state;
        }

        Ok(current_state)
    }

    fn load_initial_state(&self, property_name: &str) -> Result<PropagationState> {
        let mut state = HashMap::new();
        // Full scan. In efficient impl, use an index.
        let results = self.db.query().scan(None).execute(self.db)?;

        for row in results {
            let row = row?;
            if let Some(node) = row.entity.as_node()
                && let Some(prop) = node.get_property(property_name)
                && let Some(vec) = prop.as_vector()
            {
                state.insert(node.id, vec.to_vec());
            }
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_linear_propagation_consensus() {
        let db = AletheiaDB::new().unwrap();

        // Setup: 2 nodes with different opinions connected A -> B
        let props_a = PropertyMapBuilder::new()
            .insert_vector("opinion", &[0.0, 0.0])
            .build();
        let node_a = db.create_node("Person", props_a).unwrap();

        let props_b = PropertyMapBuilder::new()
            .insert_vector("opinion", &[1.0, 1.0])
            .build();
        let node_b = db.create_node("Person", props_b).unwrap();

        // Edge A -> B (A influences B)
        db.create_edge(
            node_a,
            node_b,
            "INFLUENCES",
            PropertyMapBuilder::new().build(),
        )
        .unwrap();

        // Edge B -> A (B influences A) - make it mutual for consensus
        db.create_edge(
            node_b,
            node_a,
            "INFLUENCES",
            PropertyMapBuilder::new().build(),
        )
        .unwrap();

        let sybil = Sybil::new(&db);
        // Alpha 0.5: Take halfway between self and neighbor
        let model = LinearPropagation::new(0.5);

        // Step 1:
        // A sees B (1,1). Self (0,0). Avg(Neighbor)= (1,1). New A = 0.5*(0,0) + 0.5*(1,1) = (0.5, 0.5)
        // B sees A (0,0). Self (1,1). Avg(Neighbor)= (0,0). New B = 0.5*(1,1) + 0.5*(0,0) = (0.5, 0.5)

        let state = sybil.simulate("opinion", &model, 1).unwrap();

        let vec_a = state.get(&node_a).unwrap();
        let vec_b = state.get(&node_b).unwrap();

        assert_eq!(vec_a, &[0.5, 0.5]);
        assert_eq!(vec_b, &[0.5, 0.5]);
    }

    #[test]
    fn test_propagation_chain() {
        // A (1.0) -> B (0.0) -> C (0.0)
        // B should get value from A. C should eventually get value from B.
        let db = AletheiaDB::new().unwrap();

        let node_a = db
            .create_node(
                "Node",
                PropertyMapBuilder::new()
                    .insert_vector("val", &[1.0])
                    .build(),
            )
            .unwrap();

        let node_b = db
            .create_node(
                "Node",
                PropertyMapBuilder::new()
                    .insert_vector("val", &[0.0])
                    .build(),
            )
            .unwrap();

        let node_c = db
            .create_node(
                "Node",
                PropertyMapBuilder::new()
                    .insert_vector("val", &[0.0])
                    .build(),
            )
            .unwrap();

        db.create_edge(node_a, node_b, "LINK", Default::default())
            .unwrap();
        db.create_edge(node_b, node_c, "LINK", Default::default())
            .unwrap();

        let sybil = Sybil::new(&db);
        // Alpha 1.0 = Total Copy
        let model = LinearPropagation::new(1.0);

        // 1 Step: B sees A(1.0). B becomes 1.0. C sees B(0.0) (old state). C stays 0.0.
        let state_1 = sybil.simulate("val", &model, 1).unwrap();
        assert_eq!(state_1.get(&node_b).unwrap(), &[1.0]);
        assert_eq!(state_1.get(&node_c).unwrap(), &[0.0]); // C hasn't seen it yet

        // 2 Steps: C sees B(1.0) (from step 1). C becomes 1.0.
        // Wait, `simulate` re-runs from scratch?
        // No, `simulate` runs N steps.
        // In Step 2 of the loop:
        // Update B: sees A(1.0). stays 1.0.
        // Update C: sees B(1.0) (from prev step). becomes 1.0.
        let state_2 = sybil.simulate("val", &model, 2).unwrap();
        assert_eq!(state_2.get(&node_c).unwrap(), &[1.0]);
    }
}
