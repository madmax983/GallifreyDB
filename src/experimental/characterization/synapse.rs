//! Synapse: Adaptive Graph Hebbian Learning.
//!
//! "Cells that fire together, wire together."
//!
//! Synapse implements a mechanism for the graph to "learn" from traversal patterns.
//! It tracks usage statistics (traversals) for edges and adjusts their effective "weight"
//! dynamically. Frequently used paths become "cheaper" to traverse, while unused paths
//! decay over time.
//!
//! This enables **Adaptive Semantic Pathfinding**: finding paths that are both
//! semantically relevant AND popular/proven.
//!
//! # Concepts
//! - **Synaptic Weight**: A derived value from usage history. High usage = Low Cost.
//! - **Hebbian Learning**: `observe(edge)` strengthens the connection.
//! - **Forgetting**: `decay()` weakens all connections, simulating memory fading.
//!
//! # Usage
//!
//! ```rust
//! // Requires features = ["nova"]
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::synapse::{Synapse, SynapseContext};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! // Create a persistent context to hold weights
//! let context = SynapseContext::new();
//! let synapse = Synapse::new(&db, &context);
//!
//! // ... perform traversals ...
//! // synapse.observe(edge_id);
//!
//! // Find an adaptive path
//! // let path = synapse.adaptive_semantic_path(start, end, "embedding")?;
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::{Error, Result};
// `IdHashBuilder` for the DashMap below (its shard selector reads the high bits
// of the hash, so an identity hash collapses every entry into shard 0);
// `IdentityHasher` for the plain `HashMap`s in the pathfinding code, which have
// no shards to spread across and where sequential ids mapping to sequential
// buckets is good locality.
use crate::core::hasher::{IdHashBuilder, IdentityHasher};
use crate::core::id::{EdgeId, NodeId};
use crate::core::vector::cosine_similarity;
use dashmap::DashMap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Shared state for Synapse weights.
/// This structure holds the "memory" of the graph's usage.
/// It is designed to be shared across multiple Synapse instances (e.g., in different threads).
pub struct SynapseContext {
    /// Maps EdgeId -> Usage Count.
    /// Usage count is monotonic increasing (until manual reset/decay handling).
    weights: DashMap<EdgeId, AtomicU64, IdHashBuilder>,
}

impl Default for SynapseContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SynapseContext {
    /// Create a new empty Synapse context.
    pub fn new() -> Self {
        Self {
            weights: DashMap::with_hasher(BuildHasherDefault::default()),
        }
    }

    /// Get the raw usage count for an edge.
    pub fn get_usage(&self, edge_id: EdgeId) -> u64 {
        match self.weights.get(&edge_id) {
            Some(val) => val.load(AtomicOrdering::Relaxed),
            None => 0,
        }
    }

    /// Increment usage for an edge.
    pub fn observe(&self, edge_id: EdgeId) {
        // Optimistic update
        // If entry exists, fetch_add. If not, insert 1.
        // DashMap's entry API is the cleanest way.
        self.weights
            .entry(edge_id)
            .and_modify(|val| {
                val.fetch_add(1, AtomicOrdering::Relaxed);
            })
            .or_insert(AtomicU64::new(1));
    }

    /// Apply decay to all weights.
    ///
    /// This multiplies all usage counts by `factor` (0.0 to 1.0).
    /// Since we store integers, this is `(usage * factor) as u64`.
    pub fn decay(&self, factor: f32) {
        if factor >= 1.0 {
            return;
        }
        // Iterate and update.
        // DashMap iter() gives read access, but we access Atomics via &AtomicU64.
        // Atomics can be mutated via shared reference.
        for entry in self.weights.iter() {
            let current = entry.value().load(AtomicOrdering::Relaxed);
            let new_val = (current as f32 * factor) as u64;
            entry.value().store(new_val, AtomicOrdering::Relaxed);
        }
    }

    /// Clear all memory.
    pub fn clear(&self) {
        self.weights.clear();
    }
}

/// The Synapse Engine.
pub struct Synapse<'a> {
    db: &'a AletheiaDB,
    context: &'a SynapseContext,
}

#[derive(Clone, Copy, PartialEq)]
struct State {
    cost: f32,
    node: NodeId,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Synapse<'a> {
    /// Create a new Synapse engine linked to a DB and a Context.
    pub fn new(db: &'a AletheiaDB, context: &'a SynapseContext) -> Self {
        Self { db, context }
    }

    /// Record a traversal event.
    pub fn observe(&self, edge_id: EdgeId) {
        self.context.observe(edge_id);
    }

    /// Find an adaptive path that balances Semantic Similarity and Popularity.
    ///
    /// Cost Function:
    /// `Cost = (1.0 - Similarity) * WeightFactor`
    ///
    /// Where `WeightFactor = 1.0 / (1.0 + log2(1.0 + Usage))`
    ///
    /// - If Usage = 0, WeightFactor = 1.0 (Standard Semantic Cost).
    /// - If Usage = 100, WeightFactor ≈ 1/7. Cost is 7x cheaper.
    ///
    /// This encourages the algorithm to choose paths that are semantically "good enough"
    /// but highly popular, over paths that are semantically perfect but unknown.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use aletheiadb::AletheiaDB;
    /// use aletheiadb::core::property::PropertyMapBuilder;
    /// use aletheiadb::experimental::characterization::synapse::{Synapse, SynapseContext};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = AletheiaDB::new()?;
    /// let context = SynapseContext::new();
    /// let synapse = Synapse::new(&db, &context);
    ///
    /// let props_a = PropertyMapBuilder::new().insert_vector("vec", &[1.0, 0.0]).build();
    /// let a = db.create_node("Node", props_a)?;
    ///
    /// let props_b = PropertyMapBuilder::new().insert_vector("vec", &[0.8, 0.6]).build();
    /// let b = db.create_node("Node", props_b)?;
    ///
    /// let edge_id = db.create_edge(a, b, "NEXT", Default::default())?;
    ///
    /// // Simulate traversing the edge
    /// synapse.observe(edge_id);
    ///
    /// // Find path
    /// let path = synapse.adaptive_semantic_path(a, b, "vec")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn adaptive_semantic_path(
        &self,
        start: NodeId,
        end: NodeId,
        vector_prop: &str,
    ) -> Result<Vec<NodeId>> {
        // Validate Start/End and get Goal Vector for heuristic
        let start_node = self.db.get_node(start)?;
        let end_node = self.db.get_node(end)?;

        let _start_vec = start_node
            .properties
            .get(vector_prop)
            .and_then(|v| v.as_arc_vector())
            .ok_or_else(|| {
                Error::other(format!(
                    "Start node {} missing vector property '{}'",
                    start, vector_prop
                ))
            })?;

        // Used for validation but not strictly needed for pathfinding (Dijkstra)
        // Kept for consistency and potential heuristic addition
        let _end_vec = end_node
            .properties
            .get(vector_prop)
            .and_then(|v| v.as_arc_vector())
            .ok_or_else(|| {
                Error::other(format!(
                    "End node {} missing vector property '{}'",
                    end, vector_prop
                ))
            })?;

        // Dijkstra / A*
        let mut open_set = BinaryHeap::new();
        open_set.push(State {
            cost: 0.0,
            node: start,
        });

        let mut came_from: HashMap<NodeId, NodeId, BuildHasherDefault<IdentityHasher>> =
            HashMap::with_hasher(BuildHasherDefault::default());
        let mut g_score: HashMap<NodeId, f32, BuildHasherDefault<IdentityHasher>> =
            HashMap::with_hasher(BuildHasherDefault::default());
        g_score.insert(start, 0.0);

        while let Some(State {
            cost: _current_f,
            node: current,
        }) = open_set.pop()
        {
            if current == end {
                return Ok(self.reconstruct_path(came_from, current));
            }

            let current_node = if current == start {
                start_node.clone()
            } else {
                self.db.get_node(current)?
            };

            let current_vec = current_node
                .properties
                .get(vector_prop)
                .and_then(|v| v.as_arc_vector());

            for edge_id in self.db.get_outgoing_edges(current) {
                let neighbor = self.db.get_edge_target(edge_id)?;

                let neighbor_node = self.db.get_node(neighbor)?;
                let neighbor_vec = neighbor_node
                    .properties
                    .get(vector_prop)
                    .and_then(|v| v.as_arc_vector());

                // 1. Calculate Semantic Cost (Base Cost)
                let semantic_cost = match (&current_vec, &neighbor_vec) {
                    (Some(a), Some(b)) => (1.0 - cosine_similarity(a, b)?).max(0.001), // Ensure > 0
                    _ => 1.0, // Penalize missing vectors
                };

                // 2. Calculate Synaptic Modulator (Popularity Bonus)
                let usage = self.context.get_usage(edge_id);
                // Formula: 1.0 / (1.0 + log2(1 + usage))
                // Usage 0 -> 1.0
                // Usage 1 -> 1.0 / (1 + 1) = 0.5
                // Usage 3 -> 1.0 / (1 + 2) = 0.33
                let weight_factor = 1.0 / (1.0 + ((1 + usage) as f32).log2());

                let edge_cost = semantic_cost * weight_factor;

                let tentative_g = g_score.get(&current).unwrap_or(&f32::INFINITY) + edge_cost;

                if tentative_g < *g_score.get(&neighbor).unwrap_or(&f32::INFINITY) {
                    came_from.insert(neighbor, current);
                    g_score.insert(neighbor, tentative_g);

                    // H = 0 (Dijkstra)
                    let h_score = 0.0;

                    let f = tentative_g + h_score;
                    open_set.push(State {
                        cost: f,
                        node: neighbor,
                    });
                }
            }
        }

        Err(Error::other("No path found"))
    }

    fn reconstruct_path(
        &self,
        came_from: HashMap<NodeId, NodeId, BuildHasherDefault<IdentityHasher>>,
        mut current: NodeId,
    ) -> Vec<NodeId> {
        let mut total_path = vec![current];
        while let Some(&prev) = came_from.get(&current) {
            current = prev;
            total_path.push(current);
        }
        total_path.reverse();
        total_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_synapse_hebbian_learning() {
        let db = AletheiaDB::new().unwrap();
        // A -> B -> D (Path 1)
        // A -> C -> D (Path 2)
        // All vectors identical (Semantic cost 0).
        // Initially, pathfinding should be ambiguous or pick based on ID order.
        // Then we strengthen Path 1.

        let props = PropertyMapBuilder::new()
            .insert_vector("vec", &[1.0, 0.0])
            .build();
        let a = db.create_node("Node", props.clone()).unwrap();
        let b = db.create_node("Node", props.clone()).unwrap();
        let c = db.create_node("Node", props.clone()).unwrap();
        let d = db.create_node("Node", props.clone()).unwrap();

        let e_ab = db
            .create_edge(a, b, "NEXT", PropertyMapBuilder::new().build())
            .unwrap();
        let e_bd = db
            .create_edge(b, d, "NEXT", PropertyMapBuilder::new().build())
            .unwrap();

        let _e_ac = db
            .create_edge(a, c, "NEXT", PropertyMapBuilder::new().build())
            .unwrap();
        let _e_cd = db
            .create_edge(c, d, "NEXT", PropertyMapBuilder::new().build())
            .unwrap();

        let context = SynapseContext::new();
        let synapse = Synapse::new(&db, &context);

        // Train Path 1 (A->B->D)
        for _ in 0..10 {
            synapse.observe(e_ab);
            synapse.observe(e_bd);
        }

        // Find path. Should prefer A->B->D because of low cost.
        let path = synapse.adaptive_semantic_path(a, d, "vec").unwrap();

        assert_eq!(path, vec![a, b, d]);
    }

    #[test]
    fn test_synapse_semantic_vs_popularity_tradeoff() {
        let db = AletheiaDB::new().unwrap();
        // A -> B -> Target (Short, but semantically bad)
        // A -> C -> Target (Longer/Equal length, but semantically perfect)

        // A: [1, 0]
        // B: [0, 1] (Opposite/Orthogonal) -> Sim 0.0 -> Cost 1.0
        // C: [0.8, 0.6] -> Sim 0.8 -> Cost 0.2
        // Target: [1, 0]

        let props_a = PropertyMapBuilder::new()
            .insert_vector("vec", &[1.0, 0.0])
            .build();
        let a = db.create_node("Node", props_a).unwrap();

        let props_b = PropertyMapBuilder::new()
            .insert_vector("vec", &[0.0, 1.0])
            .build();
        let b = db.create_node("Node", props_b).unwrap();

        let props_c = PropertyMapBuilder::new()
            .insert_vector("vec", &[0.8, 0.6]) // Sim 0.8 -> Cost 0.2
            .build();
        let c = db.create_node("Node", props_c).unwrap();

        let props_target = PropertyMapBuilder::new()
            .insert_vector("vec", &[1.0, 0.0])
            .build();
        let target = db.create_node("Node", props_target).unwrap();

        let e_ab = db
            .create_edge(a, b, "NEXT", PropertyMapBuilder::new().build())
            .unwrap();
        let e_bt = db
            .create_edge(b, target, "NEXT", PropertyMapBuilder::new().build())
            .unwrap();

        let _e_ac = db
            .create_edge(a, c, "NEXT", PropertyMapBuilder::new().build())
            .unwrap();
        let _e_ct = db
            .create_edge(c, target, "NEXT", PropertyMapBuilder::new().build())
            .unwrap();

        let context = SynapseContext::new();
        let synapse = Synapse::new(&db, &context);

        // Initially, A->C->Target should be preferred (better semantics).
        let initial_path = synapse.adaptive_semantic_path(a, target, "vec").unwrap();
        assert_eq!(
            initial_path,
            vec![a, c, target],
            "Should prefer semantic match initially"
        );

        // Boost Path B massively
        // Usage 10,000 ensures boost factor ~0.07.
        // Cost(B) = 1.0 * 0.07 = 0.07.
        // Cost(C) = 0.2.
        // B wins comfortably.

        for _ in 0..10_000 {
            synapse.observe(e_ab);
            synapse.observe(e_bt);
        }

        let adapted_path = synapse.adaptive_semantic_path(a, target, "vec").unwrap();
        assert_eq!(
            adapted_path,
            vec![a, b, target],
            "Should prefer popular path after training"
        );
    }

    #[test]
    fn test_synapse_decay() {
        let context = SynapseContext::new();
        let edge = EdgeId::new(1).unwrap();

        context.observe(edge);
        context.observe(edge);
        // Usage = 2

        assert_eq!(context.get_usage(edge), 2);

        context.decay(0.5);
        // 2 * 0.5 = 1
        assert_eq!(context.get_usage(edge), 1);

        context.decay(0.5);
        // 1 * 0.5 = 0.5 -> 0 (integer truncation)
        assert_eq!(context.get_usage(edge), 0);
    }
}
