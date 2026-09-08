//! Leviathan: Global Semantic Centroid & Outlier Detection.
//!
//! "Find the heart of the ocean, and the monsters at its edges."
//!
//! Leviathan computes the "Center of Mass" (centroid) of a specific vector property
//! across the entire graph. It then identifies the "Core" (the node closest to the centroid)
//! and the "Fringe" (nodes furthest from the centroid, i.e., outliers).
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::characterization::leviathan::Leviathan;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! let leviathan = Leviathan::new(&db);
//!
//! if let Some((core_node, _score)) = leviathan.find_core("embedding")? {
//!     println!("The most representative node is {}", core_node.as_u64());
//! }
//!
//! let fringe = leviathan.find_fringe("embedding", 5)?;
//! println!("The top outliers are: {:?}", fringe);
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::vector::ops::cosine_similarity;

/// The Leviathan Engine.
pub struct Leviathan<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Leviathan<'a> {
    /// Create a new Leviathan instance.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Calculate the global centroid for a given vector property.
    pub fn compute_centroid(&self, property_name: &str) -> Result<Option<Vec<f32>>> {
        let mut sum = Vec::new();
        let mut count = 0;

        let results = self.db.query().scan(None).execute(self.db)?;
        for row in results {
            let row = row?;
            let Some(node) = row.entity.as_node() else { continue };
            let Some(prop) = node.get_property(property_name) else { continue };
            let Some(vec) = prop.as_vector() else { continue };

            if sum.is_empty() {
                sum = vec![0.0; vec.len()];
            } else if sum.len() != vec.len() {
                continue;
            }
            for (i, &v) in vec.iter().enumerate() {
                sum[i] += v;
            }
            count += 1;
        }

        if count == 0 {
            return Ok(None);
        }

        for v in sum.iter_mut() {
            *v /= count as f32;
        }

        Ok(Some(sum))
    }

    /// Finds the node whose vector is closest to the global centroid.
    pub fn find_core(&self, property_name: &str) -> Result<Option<(NodeId, f32)>> {
        let centroid = match self.compute_centroid(property_name)? {
            Some(c) => c,
            None => return Ok(None),
        };

        if self.db.has_vector_index(property_name) {
            let results = self.db.search_vectors_in(property_name, &centroid, 1)?;
            if let Some(first) = results.first() {
                return Ok(Some((first.0, first.1)));
            }
        }

        // Fallback to full scan
        let mut best_node = None;
        let mut highest_sim = f32::NEG_INFINITY;

        let results = self.db.query().scan(None).execute(self.db)?;
        for row in results {
            let row = row?;
            let Some(node) = row.entity.as_node() else { continue };
            let Some(prop) = node.get_property(property_name) else { continue };
            let Some(vec) = prop.as_vector() else { continue };

            #[allow(clippy::collapsible_if)]
            if vec.len() == centroid.len() {
                #[allow(clippy::collapsible_if)]
                if let Ok(sim) = cosine_similarity(&centroid, vec) {
                    #[allow(clippy::collapsible_if)]
                    if sim > highest_sim {
                        highest_sim = sim;
                        best_node = Some(node.id);
                    }
                }
            }
        }

        Ok(best_node.map(|id| (id, highest_sim)))
    }

    /// Finds the nodes whose vectors are furthest from the global centroid.
    pub fn find_fringe(&self, property_name: &str, k: usize) -> Result<Vec<(NodeId, f32)>> {
        let centroid = match self.compute_centroid(property_name)? {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        let mut all_sims = Vec::new();

        let results = self.db.query().scan(None).execute(self.db)?;
        for row in results {
            let row = row?;
            let Some(node) = row.entity.as_node() else { continue };
            let Some(prop) = node.get_property(property_name) else { continue };
            let Some(vec) = prop.as_vector() else { continue };

            #[allow(clippy::collapsible_if)]
            if vec.len() == centroid.len() {
                #[allow(clippy::collapsible_if)]
                if let Ok(sim) = cosine_similarity(&centroid, vec) {
                    all_sims.push((node.id, sim));
                }
            }
        }

        all_sims.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(all_sims.into_iter().take(k).collect())
    }
}

#[cfg(all(test, feature = "semantic-characterization"))]
mod tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;
    use crate::index::vector::{DistanceMetric, HnswConfig};

    #[test]
    fn test_leviathan_centroid_and_core() {
        let db = AletheiaDB::new().unwrap();
        db.enable_vector_index("vec", HnswConfig::new(2, DistanceMetric::Cosine))
            .unwrap();

        // Node A: [1.0, 0.0]
        let a = db
            .create_node(
                "Node",
                PropertyMapBuilder::new()
                    .insert_vector("vec", &[1.0, 0.0])
                    .build(),
            )
            .unwrap();

        // Node B: [1.0, 0.0] (Same to skew centroid towards [1, 0])
        let b = db
            .create_node(
                "Node",
                PropertyMapBuilder::new()
                    .insert_vector("vec", &[1.0, 0.0])
                    .build(),
            )
            .unwrap();

        // Node C: [0.0, 1.0] (Outlier)
        let c = db
            .create_node(
                "Node",
                PropertyMapBuilder::new()
                    .insert_vector("vec", &[0.0, 1.0])
                    .build(),
            )
            .unwrap();

        let leviathan = Leviathan::new(&db);

        // Centroid should be [(1+1+0)/3, (0+0+1)/3] = [0.666, 0.333]
        let centroid = leviathan.compute_centroid("vec").unwrap().unwrap();
        assert!((centroid[0] - 0.666).abs() < 0.01);
        assert!((centroid[1] - 0.333).abs() < 0.01);

        // Core should be A or B (since they are closest to [0.66, 0.33])
        let (core, _) = leviathan.find_core("vec").unwrap().unwrap();
        assert!(core == a || core == b);

        // Fringe should be C
        let fringe = leviathan.find_fringe("vec", 1).unwrap();
        assert_eq!(fringe[0].0, c);
    }
}
