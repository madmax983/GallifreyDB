//! Kaleidoscope: A Semantic Force-Directed Layout Engine.
//!
//! "What does the graph *look* like?"
//!
//! Kaleidoscope projects high-dimensional semantic data combined with graph topology
//! into a 2D plane. It uses a physics simulation where:
//! - Nodes are charged particles (Repulsion).
//! - Edges are springs (Attraction).
//! - Vector Similarity is "Gravity" (Semantic Attraction).
//!
//! This allows you to *see* clusters that form not just from edges, but from shared meaning.

use crate::core::id::NodeId;
use std::collections::HashMap;

/// A point in 2D space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
}

impl Point {
    /// Create a new point.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Calculate Euclidean distance to another point.
    pub fn distance(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    fn add(&self, other: &Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }

    fn sub(&self, other: &Point) -> Point {
        Point::new(self.x - other.x, self.y - other.y)
    }

    fn scale(&self, factor: f32) -> Point {
        Point::new(self.x * factor, self.y * factor)
    }

    fn magnitude(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    fn normalize(&self) -> Point {
        let mag = self.magnitude();
        if mag == 0.0 {
            Point::new(0.0, 0.0)
        } else {
            self.scale(1.0 / mag)
        }
    }
}

/// Configuration for the simulation.
///
/// # Examples
///
/// ```rust
/// use aletheiadb::experimental::kaleidoscope::LayoutConfig;
///
/// let config = LayoutConfig {
///     iterations: 200,
///     semantic_strength: 50.0,
///     ..LayoutConfig::default()
/// };
/// assert_eq!(config.iterations, 200);
/// assert_eq!(config.semantic_strength, 50.0);
/// ```
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Ideal distance between nodes (k).
    pub optimal_distance: f32,
    /// Strength of repulsion force.
    pub repulsion_strength: f32,
    /// Strength of spring attraction (topology).
    pub spring_strength: f32,
    /// Strength of semantic gravity (vectors).
    pub semantic_strength: f32,
    /// Cooling factor (0.0 to 1.0).
    pub cooling_factor: f32,
    /// Max iterations.
    pub iterations: usize,
    /// Width of the canvas (for centering).
    pub width: f32,
    /// Height of the canvas (for centering).
    pub height: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            optimal_distance: 50.0,
            repulsion_strength: 1.0,
            spring_strength: 0.1,
            semantic_strength: 100.0, // Semantic pull needs to be strong to compete with topology
            cooling_factor: 0.95,
            iterations: 100,
            width: 1000.0,
            height: 1000.0,
        }
    }
}

/// The Layout Engine.
///
/// # Examples
///
/// ```rust
/// use aletheiadb::experimental::kaleidoscope::{LayoutEngine, LayoutConfig};
/// use aletheiadb::core::id::NodeId;
///
/// let mut engine = LayoutEngine::new(LayoutConfig::default());
///
/// let n1 = NodeId::new(1).unwrap();
/// let n2 = NodeId::new(2).unwrap();
/// let n3 = NodeId::new(3).unwrap();
///
/// engine.add_node(n1);
/// engine.add_node(n2);
/// engine.add_node(n3);
///
/// engine.add_edge(n1, n2); // Topological pull
/// engine.add_semantic_link(n1, n3, 0.9); // Semantic pull
///
/// engine.run();
///
/// let positions = engine.get_positions();
/// assert_eq!(positions.len(), 3);
/// ```
pub struct LayoutEngine {
    config: LayoutConfig,
    nodes: Vec<NodeId>,
    positions: HashMap<NodeId, Point>,
    velocities: HashMap<NodeId, Point>,
    edges: Vec<(NodeId, NodeId)>,
    /// Map of (NodeA, NodeB) -> Similarity Score (0.0 to 1.0)
    semantic_links: HashMap<(NodeId, NodeId), f32>,
    /// Current temperature.
    temperature: f32,
}

impl LayoutEngine {
    /// Create a new LayoutEngine with the given configuration.
    pub fn new(config: LayoutConfig) -> Self {
        let width = config.width;
        Self {
            temperature: width / 10.0, // Start hot
            config,
            nodes: Vec::new(),
            positions: HashMap::new(),
            velocities: HashMap::new(),
            edges: Vec::new(),
            semantic_links: HashMap::new(),
        }
    }

    /// Add a node to the simulation.
    pub fn add_node(&mut self, id: NodeId) {
        if !self.positions.contains_key(&id) {
            self.nodes.push(id);
            // Deterministic random start position based on ID
            // Simple Linear Congruential Generator
            let seed = id.as_u64();
            let x = ((seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407))
                % 1000) as f32;
            let y = ((seed
                .wrapping_mul(1442695040888963407)
                .wrapping_add(6364136223846793005))
                % 1000) as f32;

            self.positions.insert(id, Point::new(x, y));
            self.velocities.insert(id, Point::new(0.0, 0.0));
        }
    }

    /// Add an edge (Topology constraint).
    pub fn add_edge(&mut self, a: NodeId, b: NodeId) {
        self.add_node(a);
        self.add_node(b);
        self.edges.push((a, b));
    }

    /// Add a semantic link (Vector constraint).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::NodeId;
    /// use aletheiadb::experimental::characterization::kaleidoscope::{LayoutConfig, LayoutEngine};
    ///
    /// let mut engine = LayoutEngine::new(LayoutConfig::default());
    /// let n1 = NodeId::new(1).unwrap();
    /// let n2 = NodeId::new(2).unwrap();
    ///
    /// // Add a semantic link with a similarity score of 0.8
    /// engine.add_semantic_link(n1, n2, 0.8);
    /// ```
    pub fn add_semantic_link(&mut self, a: NodeId, b: NodeId, similarity: f32) {
        if similarity > 0.0 {
            self.add_node(a);
            self.add_node(b);
            // Store ordered pair to avoid duplicates if caller isn't careful,
            // though Hash keys handle it if we are consistent.
            let key = if a < b { (a, b) } else { (b, a) };
            self.semantic_links.insert(key, similarity);
        }
    }

    /// Run the simulation.
    pub fn run(&mut self) {
        for _ in 0..self.config.iterations {
            self.step();
            if self.temperature < 0.1 {
                break;
            }
        }
    }

    /// Execute one step of physics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use aletheiadb::core::id::NodeId;
    /// use aletheiadb::experimental::characterization::kaleidoscope::{LayoutConfig, LayoutEngine};
    ///
    /// let mut engine = LayoutEngine::new(LayoutConfig::default());
    /// let n1 = NodeId::new(1).unwrap();
    /// let n2 = NodeId::new(2).unwrap();
    /// engine.add_edge(n1, n2);
    ///
    /// // Run one step of the physics simulation manually
    /// engine.step();
    /// ```
    pub fn step(&mut self) {
        let k = self.config.optimal_distance;
        let mut forces: HashMap<NodeId, Point> = self
            .nodes
            .iter()
            .map(|&id| (id, Point::new(0.0, 0.0)))
            .collect();

        // 1. Repulsion (Coulomb) - All pairs
        // Optimizing: only check pairs? O(N^2) is fine for small N (< 500).
        for i in 0..self.nodes.len() {
            let u = self.nodes[i];
            let pos_u = self.positions[&u];

            for j in (i + 1)..self.nodes.len() {
                let v = self.nodes[j];
                let pos_v = self.positions[&v];

                let delta = pos_u.sub(&pos_v);
                let dist = delta.magnitude();

                // Avoid division by zero
                let dist = if dist < 0.1 { 0.1 } else { dist };

                // F = k^2 / d
                let repulsion = (k * k) / dist * self.config.repulsion_strength;
                let force_vec = delta.normalize().scale(repulsion);

                // Apply to u (push away)
                let f_u = forces.get_mut(&u).unwrap();
                *f_u = f_u.add(&force_vec);

                // Apply to v (push away - opposite direction)
                let f_v = forces.get_mut(&v).unwrap();
                *f_v = f_v.sub(&force_vec);
            }
        }

        // 2. Attraction (Springs) - Edges
        for &(u, v) in &self.edges {
            let pos_u = self.positions[&u];
            let pos_v = self.positions[&v];

            let delta = pos_u.sub(&pos_v);
            let dist = delta.magnitude();

            // F = d^2 / k
            let attraction = (dist * dist) / k * self.config.spring_strength;
            let force_vec = delta.normalize().scale(attraction);

            // Pull u towards v (subtract delta which points u->v? No, delta is u-v)
            // If u is at (10,0) and v is at (0,0), delta is (10,0).
            // We want u to move left (-). So we subtract force_vec.
            let f_u = forces.get_mut(&u).unwrap();
            *f_u = f_u.sub(&force_vec);

            let f_v = forces.get_mut(&v).unwrap();
            *f_v = f_v.add(&force_vec);
        }

        // 3. Semantic Gravity - Vector Similarity
        for (&(u, v), &similarity) in &self.semantic_links {
            let pos_u = self.positions[&u];
            let pos_v = self.positions[&v];

            let delta = pos_u.sub(&pos_v);
            // Linear attraction based on similarity
            // Stronger similarity = Stronger pull
            let gravity = similarity * self.config.semantic_strength;
            let force_vec = delta.normalize().scale(gravity);

            let f_u = forces.get_mut(&u).unwrap();
            *f_u = f_u.sub(&force_vec);

            let f_v = forces.get_mut(&v).unwrap();
            *f_v = f_v.add(&force_vec);
        }

        // 4. Update Positions
        for &id in &self.nodes {
            let force = forces[&id];

            // Cap movement by temperature
            let mag = force.magnitude();
            let limit = self.temperature;

            let movement = if mag > limit {
                force.normalize().scale(limit)
            } else {
                force
            };

            let pos = self.positions.get_mut(&id).unwrap();
            *pos = pos.add(&movement);

            // Bound within canvas (optional, but keeps things sane)
            // pos.x = pos.x.clamp(0.0, self.config.width);
            // pos.y = pos.y.clamp(0.0, self.config.height);
        }

        // Cool down
        self.temperature *= self.config.cooling_factor;
    }

    /// Get the final positions.
    pub fn get_positions(&self) -> &HashMap<NodeId, Point> {
        &self.positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_convergence() {
        let config = LayoutConfig {
            iterations: 50,
            ..LayoutConfig::default()
        };
        let mut engine = LayoutEngine::new(config);

        let n1 = NodeId::new(1).unwrap();
        let n2 = NodeId::new(2).unwrap();
        let n3 = NodeId::new(3).unwrap();

        // 1 and 2 are connected
        engine.add_edge(n1, n2);
        // 3 is far away (implicitly initialized randomly)
        engine.add_node(n3);

        // Run
        engine.run();

        let pos = engine.get_positions();
        let p1 = pos[&n1];
        let p2 = pos[&n2];
        let p3 = pos[&n3];

        let d12 = p1.distance(&p2);
        let d13 = p1.distance(&p3);

        // Connected nodes should be closer than unconnected nodes
        // (Probabilistic, but with 50 iterations and high spring strength, highly likely)
        assert!(
            d12 < d13,
            "Connected nodes (dist={}) should be closer than unconnected (dist={})",
            d12,
            d13
        );
    }

    #[test]
    fn test_semantic_gravity() {
        let config = LayoutConfig {
            iterations: 50,
            semantic_strength: 200.0, // Crank it up
            ..LayoutConfig::default()
        };
        let mut engine = LayoutEngine::new(config);

        let n1 = NodeId::new(1).unwrap();
        let n2 = NodeId::new(2).unwrap(); // Semantic Twin of 1
        let n3 = NodeId::new(3).unwrap(); // Random stranger

        // No topological edges!
        engine.add_node(n1);
        engine.add_node(n2);
        engine.add_node(n3);

        // Add strong semantic link between 1 and 2
        engine.add_semantic_link(n1, n2, 1.0);
        // Weak/No link to 3

        engine.run();

        let pos = engine.get_positions();
        let p1 = pos[&n1];
        let p2 = pos[&n2];
        let p3 = pos[&n3];

        let d12 = p1.distance(&p2);
        let d13 = p1.distance(&p3);

        assert!(
            d12 < d13,
            "Semantically similar nodes (dist={}) should be closer than others (dist={})",
            d12,
            d13
        );
    }

    #[test]
    fn test_visual_output() {
        // This test prints an ASCII map. It always passes, but allows manual verification.
        let config = LayoutConfig {
            width: 100.0,
            height: 40.0,
            iterations: 100,
            ..LayoutConfig::default()
        };
        let mut engine = LayoutEngine::new(config);

        // Create a "Star" graph: 1 is center, 2,3,4,5 surround it.
        let center = NodeId::new(1).unwrap();
        for i in 2..=6 {
            let sat = NodeId::new(i).unwrap();
            engine.add_edge(center, sat);
        }

        // Create a separate cluster: 10-11-12
        let c2_1 = NodeId::new(10).unwrap();
        let c2_2 = NodeId::new(11).unwrap();
        let c2_3 = NodeId::new(12).unwrap();
        engine.add_edge(c2_1, c2_2);
        engine.add_edge(c2_2, c2_3);
        engine.add_edge(c2_3, c2_1);

        engine.run();

        let pos = engine.get_positions();

        // Render
        let w = 80;
        let h = 20;
        let mut grid = vec![vec![' '; w]; h];

        // Find bounds to normalize
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for p in pos.values() {
            if p.x < min_x {
                min_x = p.x;
            }
            if p.x > max_x {
                max_x = p.x;
            }
            if p.y < min_y {
                min_y = p.y;
            }
            if p.y > max_y {
                max_y = p.y;
            }
        }

        let range_x = max_x - min_x;
        let range_y = max_y - min_y;

        println!("\nKaleidoscope Layout:");
        println!(
            "Bounds: X[{:.1}, {:.1}] Y[{:.1}, {:.1}]",
            min_x, max_x, min_y, max_y
        );

        for (id, p) in pos {
            // Map to grid
            let gx = ((p.x - min_x) / range_x * (w as f32 - 1.0)) as usize;
            let gy = ((p.y - min_y) / range_y * (h as f32 - 1.0)) as usize;

            if gx < w && gy < h {
                let c = if id.as_u64() == 1 {
                    '*'
                } else if id.as_u64() >= 10 {
                    '#'
                } else {
                    'o'
                };
                grid[gy][gx] = c;
            }
        }

        println!("+{}+", "-".repeat(w));
        for row in grid {
            let s: String = row.into_iter().collect();
            println!("|{}|", s);
        }
        println!("+{}+", "-".repeat(w));
        println!("Legend: * = Center, o = Satellite, # = Cluster 2");
    }
}
