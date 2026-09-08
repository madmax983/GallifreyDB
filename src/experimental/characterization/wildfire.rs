//! Wildfire: Temperature-Modulated Memetic Propagation 🔥
//!
//! "How fast will this idea spread in a volatile environment?"
//!
//! Wildfire combines `Thermos` (Semantic Volatility) and `Sybil` (Memetic Propagation).
//! Instead of a fixed adoption rate, a node's susceptibility to new ideas is
//! proportional to its recent semantic temperature.
//!
//! # Concepts
//! - **Hot Nodes**: Highly volatile nodes adopt neighbor ideas quickly.
//! - **Cold Nodes**: Stable nodes resist change (high inertia).
//! - **Ignition**: The propagation multiplier that scales temperature to adoption rate.
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::wildfire::WildfirePropagation;
//! use aletheiadb::experimental::sybil::Sybil;
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! // ... setup graph and measure temperatures via Thermos ...
//!
//! let mut temperatures = HashMap::new();
//! // N1 is highly volatile, N2 is stable
//! # let n1 = aletheiadb::core::id::NodeId::new(1)?;
//! # let n2 = aletheiadb::core::id::NodeId::new(2)?;
//! temperatures.insert(n1, 5.0);
//! temperatures.insert(n2, 0.1);
//!
//! // Base adoption is 0.1, but temperature adds 0.2 * Temp
//! let model = WildfirePropagation::new(temperatures, 0.1, 0.2);
//!
//! let sybil = Sybil::new(&db);
//! let final_state = sybil.simulate("opinion", &model, 3)?;
//! # Ok(())
//! # }
//! ```

use crate::core::id::NodeId;
use crate::experimental::sybil::PropagationModel;
use std::collections::HashMap;

/// A temperature-modulated propagation model.
/// New State = (1 - dynamic_alpha) * Old State + dynamic_alpha * Average(Neighbor States)
/// where `dynamic_alpha = clamp(base_alpha + temp_multiplier * temperature, 0.0, 1.0)`
pub struct WildfirePropagation {
    /// Maps node ID to its computed semantic temperature.
    pub temperatures: HashMap<NodeId, f32>,
    /// The base adoption rate for a node with 0 temperature.
    pub base_alpha: f32,
    /// Multiplier mapping temperature units to alpha increase.
    pub temp_multiplier: f32,
}

impl WildfirePropagation {
    /// Create a new WildfirePropagation model.
    pub fn new(temperatures: HashMap<NodeId, f32>, base_alpha: f32, temp_multiplier: f32) -> Self {
        Self {
            temperatures,
            base_alpha: base_alpha.clamp(0.0, 1.0),
            temp_multiplier: temp_multiplier.max(0.0),
        }
    }

    /// Calculate the dynamic alpha for a specific node based on its temperature.
    fn get_dynamic_alpha(&self, node_id: NodeId) -> f32 {
        let temp = self.temperatures.get(&node_id).copied().unwrap_or(0.0);
        let alpha = self.base_alpha + (self.temp_multiplier * temp);
        alpha.clamp(0.0, 1.0)
    }
}

impl PropagationModel for WildfirePropagation {
    fn next_state(
        &self,
        node_id: NodeId,
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

        for val in sum.iter_mut().take(dim) {
            *val /= count as f32;
        }

        // Apply dynamic alpha blending
        let alpha = self.get_dynamic_alpha(node_id);

        match current_self {
            Some(self_vec) if self_vec.len() == dim => {
                let mut new_vec = vec![0.0; dim];
                for i in 0..dim {
                    new_vec[i] = (1.0 - alpha) * self_vec[i] + alpha * sum[i];
                }
                Some(new_vec)
            }
            // If the node has no current vector, it fully adopts the neighbors' average
            _ => Some(sum),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AletheiaDB;
    use crate::core::property::PropertyMapBuilder;
    use crate::experimental::sybil::Sybil;

    #[test]
    fn test_wildfire_propagation() {
        let db = AletheiaDB::new().unwrap();

        // Setup: 3 nodes with opinions
        // A (1.0) -> B (0.0) -> C (0.0)
        let props_a = PropertyMapBuilder::new()
            .insert_vector("opinion", &[1.0])
            .build();
        let node_a = db.create_node("Person", props_a).unwrap();

        let props_b = PropertyMapBuilder::new()
            .insert_vector("opinion", &[0.0])
            .build();
        let node_b = db.create_node("Person", props_b).unwrap();

        let props_c = PropertyMapBuilder::new()
            .insert_vector("opinion", &[0.0])
            .build();
        let node_c = db.create_node("Person", props_c).unwrap();

        // Edges: A -> B, B -> C
        db.create_edge(node_a, node_b, "INFLUENCES", Default::default())
            .unwrap();
        db.create_edge(node_b, node_c, "INFLUENCES", Default::default())
            .unwrap();

        // Define temperatures:
        // Node B is very "hot" (high temp)
        // Node C is "cold" (low temp)
        let mut temps = HashMap::new();
        temps.insert(node_b, 10.0);
        temps.insert(node_c, 0.0);

        // Base alpha 0.0, multiplier 0.1.
        // B's alpha = 0.0 + (10.0 * 0.1) = 1.0 (Full Conformity)
        // C's alpha = 0.0 + (0.0 * 0.1)  = 0.0 (Full Inertia)
        let model = WildfirePropagation::new(temps, 0.0, 0.1);

        let sybil = Sybil::new(&db);
        let state = sybil.simulate("opinion", &model, 2).unwrap();

        let vec_b = state.get(&node_b).unwrap();
        let vec_c = state.get(&node_c).unwrap();

        // B completely adopted A's vector because it's hot
        assert_eq!(vec_b[0], 1.0);

        // C completely ignored B's vector because it's cold
        assert_eq!(vec_c[0], 0.0);
    }
}
