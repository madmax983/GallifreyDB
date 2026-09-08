//! Gravity: Semantic Mass and Orbit Analysis.
//!
//! "Who are the real influencers?"
//!
//! The Gravity engine analyzes the semantic influence of nodes over time.
//! It measures how much a node's neighbors are "pulled" towards it in vector space.
//!
//! # Concepts
//! - **Mass**: The degree centrality of a node (number of connections).
//! - **Orbit**: The trajectory of a neighbor relative to the center node.
//! - **Velocity**: The rate at which a neighbor is moving towards (attraction) or away from (repulsion) the center.
//!
//! # Use Cases
//! - **Influence Mapping**: Identifying nodes that drive semantic change in their community.
//! - **Community Dynamics**: Detecting if a community is tightening (converging) or dispersing (diverging).
//! - **Trend Analysis**: Measuring the velocity of adoption for new concepts.
//! - **Simulation**: Predicting how vectors should move if influenced by "Gravity".

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::temporal::{TimeRange, Timestamp, time};
use crate::core::vector::{magnitude, normalize};

/// Metrics describing the orbital dynamics of a neighbor relative to a center node.
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitMetrics {
    /// The neighbor node being analyzed.
    pub neighbor_id: NodeId,
    /// Semantic distance at the start of the window.
    pub start_distance: Option<f32>,
    /// Semantic distance at the end of the window.
    pub end_distance: Option<f32>,
    /// Velocity of semantic movement (change in distance per second).
    /// Negative = Attraction (moving closer).
    /// Positive = Repulsion (moving away).
    pub velocity: Option<f32>,
}

/// The Gravity engine for analyzing semantic influence.
pub struct GravityWell<'a> {
    db: &'a AletheiaDB,
}

impl<'a> GravityWell<'a> {
    /// Create a new GravityWell.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Analyze the orbit of neighbors around a center node.
    ///
    /// # Arguments
    ///
    /// * `center_id` - The central node (the "Sun").
    /// * `property` - The vector property to analyze (e.g., "embedding").
    /// * `window` - The time range for analysis.
    ///
    /// # Returns
    ///
    /// A list of `OrbitMetrics` for each neighbor connected at `window.end`.
    pub fn analyze_orbit(
        &self,
        center_id: NodeId,
        property: &str,
        window: TimeRange,
    ) -> Result<Vec<OrbitMetrics>> {
        // Fix transaction time to "now" for consistent view
        let tx_time = time::now();

        // 1. Get neighbors at the end of the window
        // We only care about nodes that are currently connected (or at window end)
        let edge_ids = self
            .db
            .get_outgoing_edges_at_time(center_id, window.end(), tx_time);

        if edge_ids.is_empty() {
            return Ok(Vec::new());
        }

        let edges = self
            .db
            .get_edges_at_time(&edge_ids, window.end(), tx_time)?;

        // Extract valid target IDs
        let target_ids: Vec<NodeId> = edges
            .into_iter()
            .filter_map(|(_, edge_opt)| edge_opt.map(|e| e.target))
            .collect();

        // 2. Get Center Node vectors at start and end
        let center_start_vec = self.get_vector_at(center_id, property, window.start(), tx_time)?;
        let center_end_vec = self.get_vector_at(center_id, property, window.end(), tx_time)?;

        let duration_secs = window.duration_micros().unwrap_or(0) as f32 / 1_000_000.0;
        let mut metrics = Vec::with_capacity(target_ids.len());

        for target_id in target_ids {
            let target_start_vec =
                self.get_vector_at(target_id, property, window.start(), tx_time)?;
            let target_end_vec = self.get_vector_at(target_id, property, window.end(), tx_time)?;

            let start_dist = if let (Some(c), Some(t)) = (&center_start_vec, &target_start_vec) {
                Some(cosine_distance(c, t))
            } else {
                None
            };

            let end_dist = if let (Some(c), Some(t)) = (&center_end_vec, &target_end_vec) {
                Some(cosine_distance(c, t))
            } else {
                None
            };

            let velocity = if let (Some(s), Some(e)) = (start_dist, end_dist) {
                if duration_secs > 0.0 {
                    Some((e - s) / duration_secs)
                } else {
                    Some(0.0)
                }
            } else {
                None
            };

            metrics.push(OrbitMetrics {
                neighbor_id: target_id,
                start_distance: start_dist,
                end_distance: end_dist,
                velocity,
            });
        }

        Ok(metrics)
    }

    /// Helper to get vector property at a specific time.
    fn get_vector_at(
        &self,
        node_id: NodeId,
        property: &str,
        valid_time: Timestamp,
        tx_time: Timestamp,
    ) -> Result<Option<Vec<f32>>> {
        // 1. Try full history scan (most robust for experimental analysis)
        // This ensures we find the exact version even if temporal indices are lagging.
        // We use is_valid_at (ignoring transaction time) to find *any* version that covers the valid time.
        // We iterate through all versions and keep the last match, effectively preferring the
        // most recent version (highest ID) that claims validity for that time.
        // This works around issues where past versions might be transactionally closed (corrections)
        // but still represent the best knowledge of that time period.
        if let Ok(history) = self.db.get_node_history(node_id) {
            let mut best_match: Option<Vec<f32>> = None;
            for version in history.versions {
                if version.temporal.is_valid_at(valid_time)
                    && let Some(val) = version.properties.get(property)
                    && let Some(vec) = val.as_vector()
                {
                    best_match = Some(vec.to_vec());
                }
            }
            if let Some(vec) = best_match {
                return Ok(Some(vec));
            }
        }

        // 2. Try standard temporal lookup (fast path)
        if let Ok(node) = self.db.get_node_at_time(node_id, valid_time, tx_time) {
            return Self::extract_vector(&node, property);
        }

        // 3. Fallback: Current state (latest)
        if let Ok(node) = self.db.get_node(node_id) {
            return Self::extract_vector(&node, property);
        }

        Ok(None)
    }

    fn extract_vector(node: &crate::core::graph::Node, property: &str) -> Result<Option<Vec<f32>>> {
        if let Some(val) = node.properties.get(property)
            && let Some(vec) = val.as_vector()
        {
            return Ok(Some(vec.to_vec()));
        }
        Ok(None)
    }
}

/// The Gravity Simulator applies semantic forces to propose new vector positions.
pub struct GravitySimulator<'a> {
    db: &'a AletheiaDB,
}

impl<'a> GravitySimulator<'a> {
    /// Create a new Gravity Simulator.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Simulate gravitational pull from a center node on its neighbors.
    ///
    /// # Arguments
    ///
    /// * `center_id` - The node exerting the force.
    /// * `property` - The vector property to simulate.
    /// * `mass` - The "mass" of the center node (strength of pull).
    /// * `step_size` - Simulation step size (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// A list of `(NodeId, Vec<f32>)` pairs representing the proposed new vector positions for neighbors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use aletheiadb::AletheiaDB;
    /// use aletheiadb::core::property::PropertyMapBuilder;
    /// use aletheiadb::experimental::characterization::gravity::GravitySimulator;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = AletheiaDB::new()?;
    ///
    /// let center_props = PropertyMapBuilder::new().insert_vector("vec", &[1.0, 0.0]).build();
    /// let center = db.create_node("Sun", center_props)?;
    ///
    /// let neighbor_props = PropertyMapBuilder::new().insert_vector("vec", &[0.0, 1.0]).build();
    /// let neighbor = db.create_node("Planet", neighbor_props)?;
    ///
    /// db.create_edge(center, neighbor, "ORBITS", Default::default())?;
    ///
    /// let sim = GravitySimulator::new(&db);
    /// let updates = sim.simulate_pull(center, "vec", 10.0, 0.1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn simulate_pull(
        &self,
        center_id: NodeId,
        property: &str,
        mass: f32,
        step_size: f32,
    ) -> Result<Vec<(NodeId, Vec<f32>)>> {
        // Get Center Vector (Current State)
        let center_node = self.db.get_node(center_id)?;
        let center_vec = match center_node
            .properties
            .get(property)
            .and_then(|v| v.as_vector())
        {
            Some(v) => v,
            None => return Ok(Vec::new()), // No gravity without a vector
        };

        // Get Neighbors
        let edge_ids = self.db.get_outgoing_edges(center_id);
        if edge_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut proposed_updates = Vec::new();

        for edge_id in edge_ids {
            let target_id = match self.db.get_edge_target(edge_id) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let target_node = match self.db.get_node(target_id) {
                Ok(n) => n,
                Err(_) => continue,
            };

            let target_vec = match target_node
                .properties
                .get(property)
                .and_then(|v| v.as_vector())
            {
                Some(v) => v,
                None => continue,
            };

            // Physics Calculation
            // 1. Direction Vector (Center - Neighbor)
            // Note: Vectors might have different lengths if database is messy, assuming same dim.
            if center_vec.len() != target_vec.len() {
                continue;
            }

            let mut diff = Vec::with_capacity(center_vec.len());
            for (c, t) in center_vec.iter().zip(target_vec.iter()) {
                diff.push(c - t);
            }

            let dist = magnitude(&diff);

            // Avoid division by zero or extreme forces at close range
            let effective_dist = dist.max(0.0001);

            // Force Magnitude: F = G * M / r^2 (Assume G=1, neighbor mass=1)
            // Damping factor to prevent explosion
            let force = mass / (effective_dist * effective_dist + 0.1);

            // Apply force
            // New = Old + (Direction * Force * Step)
            // Direction = Diff / Dist (Unit vector)
            // So: New = Old + (Diff / Dist) * Force * Step
            //     New = Old + Diff * (Force * Step / Dist)

            let scale = (force * step_size) / effective_dist;

            let mut new_vec = Vec::with_capacity(target_vec.len());
            for (t, d) in target_vec.iter().zip(diff.iter()) {
                new_vec.push(t + d * scale);
            }

            // Normalize result to keep it on the hypersphere
            let normalized_vec = normalize(&new_vec);

            proposed_updates.push((target_id, normalized_vec));
        }

        Ok(proposed_updates)
    }
}

/// Calculate Cosine Distance (1.0 - Cosine Similarity).
/// Range: [0.0, 2.0]. 0.0 = identical, 1.0 = orthogonal, 2.0 = opposite.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    crate::core::vector::cosine_similarity(a, b)
        .map(|sim| 1.0 - sim)
        .unwrap_or(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::transaction::WriteOps;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_gravity_attraction() {
        let db = AletheiaDB::new().unwrap();

        // 1. Setup Center Node (The Sun) at [1.0, 0.0]
        let center_props = PropertyMapBuilder::new()
            .insert_vector("vec", &[1.0, 0.0])
            .build();
        let center = db.create_node("Sun", center_props).unwrap();

        // 2. Setup Neighbor (The Planet) at [0.0, 1.0] (Orthogonal, dist = 1.0)
        let neighbor_props = PropertyMapBuilder::new()
            .insert_vector("vec", &[0.0, 1.0])
            .build();
        let neighbor = db.create_node("Planet", neighbor_props).unwrap();

        // Connect them
        db.create_edge(
            center,
            neighbor,
            "ORBITS",
            PropertyMapBuilder::new().build(),
        )
        .unwrap();

        // Ensure start time is strictly after creation
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t_start = time::now();

        // Wait a bit to establish duration
        std::thread::sleep(std::time::Duration::from_millis(50));

        // 3. Move Neighbor closer to [1.0, 0.0] -> [0.7, 0.7] (approx 45 deg)
        let update_props = PropertyMapBuilder::new()
            .insert_vector("vec", &[0.707, 0.707])
            .build();

        db.write(|tx| tx.update_node(neighbor, update_props))
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(50));
        let t_end = time::now();

        // 4. Analyze Orbit
        let gravity = GravityWell::new(&db);
        let window = TimeRange::new(t_start, t_end).unwrap();

        let metrics = gravity.analyze_orbit(center, "vec", window).unwrap();

        assert_eq!(metrics.len(), 1);
        let m = &metrics[0];

        assert!(m.start_distance.is_some(), "Start distance is None");
        assert!(m.end_distance.is_some(), "End distance is None");

        // Start dist (orthogonal) ~= 1.0
        // End dist (45 deg) ~= 1.0 - 0.707 = 0.293
        assert!(
            m.start_distance.unwrap() > 0.9,
            "Start distance should be high"
        );
        assert!(
            m.end_distance.unwrap() < 0.4,
            "End distance should be lower"
        );

        // Velocity should be negative (attraction)
        assert!(
            m.velocity.unwrap() < 0.0,
            "Velocity should be negative (attraction)"
        );
    }

    #[test]
    fn test_gravity_simulator() {
        let db = AletheiaDB::new().unwrap();

        // Center: [1.0, 0.0]
        let center = db
            .create_node(
                "Sun",
                PropertyMapBuilder::new()
                    .insert_vector("vec", &[1.0, 0.0])
                    .build(),
            )
            .unwrap();

        // Neighbor: [0.0, 1.0] (Orthogonal)
        let neighbor = db
            .create_node(
                "Planet",
                PropertyMapBuilder::new()
                    .insert_vector("vec", &[0.0, 1.0])
                    .build(),
            )
            .unwrap();

        // Connect
        db.create_edge(
            center,
            neighbor,
            "ORBITS",
            PropertyMapBuilder::new().build(),
        )
        .unwrap();

        let sim = GravitySimulator::new(&db);

        // Simulate
        // Mass = 10.0, Step = 0.1
        // Force will be significant.
        // Diff = [1, -1]
        // Dist = sqrt(2) = 1.414
        // Force = 10 / (2 + 0.1) = 4.76
        // Scale = 4.76 * 0.1 / 1.414 = 0.33
        // NewVec = [0, 1] + [1, -1] * 0.33 = [0.33, 0.67]
        // Normalize([0.33, 0.67]) -> [0.44, 0.89] approx
        // It moved towards [1, 0] (x increased from 0)

        let updates = sim.simulate_pull(center, "vec", 10.0, 0.1).unwrap();

        assert_eq!(updates.len(), 1);
        let (id, new_vec) = &updates[0];
        assert_eq!(*id, neighbor);

        // Check X component increased (moved towards 1.0)
        assert!(
            new_vec[0] > 0.1,
            "X component should increase (got {})",
            new_vec[0]
        );
        // Check Y component decreased (moved away from 1.0)
        assert!(
            new_vec[1] < 0.95,
            "Y component should decrease (got {})",
            new_vec[1]
        );

        // Verify normalization
        let mag = new_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-5, "Vector should be normalized");
    }
}
