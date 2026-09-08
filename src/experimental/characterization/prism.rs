//! Prism: Semantic Spectroscopy for Vectors.
//!
//! "Unbox the Black Box."
//!
//! The Prism decomposes a vector embedding into named semantic components (facets).
//! It allows you to understand *why* a vector is positioned where it is by projecting
//! it onto a set of user-defined axes (concepts).
//!
//! # Use Cases
//! - **Explainability**: Why did the search retrieve this? "It's 80% 'Sci-Fi' and 20% 'Romance'."
//! - **Debugging**: Detect if a vector has drifted into an unwanted semantic region (e.g., "NSFW").
//! - **Filtering**: "Show me nodes that are high in 'Innovation' but low in 'Risk'."
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::prism::Prism;
//! use aletheiadb::core::property::PropertyMapBuilder;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//!
//! // Create dummy nodes for the example
//! let props = PropertyMapBuilder::new().insert_vector("embedding", &[1.0, 0.0]).build();
//! let tech_node_id = db.create_node("Concept", props)?;
//!
//! let props = PropertyMapBuilder::new().insert_vector("embedding", &[0.0, 1.0]).build();
//! let magic_node_id = db.create_node("Concept", props)?;
//!
//! let props = PropertyMapBuilder::new().insert_vector("embedding", &[0.5, 0.5]).build();
//! let target_node_id = db.create_node("Target", props)?;
//!
//! let mut prism = Prism::new(&db).with_vector_property("embedding");
//!
//! // Define the "lens"
//! prism.add_axis_from_node("Tech", tech_node_id)?;
//! prism.add_axis_from_node("Magic", magic_node_id)?;
//!
//! // Analyze a target
//! let spectrum = prism.analyze_node(target_node_id)?;
//!
//! println!("Tech: {:.2}", spectrum.get("Tech").unwrap_or(&0.0));
//! println!("Magic: {:.2}", spectrum.get("Magic").unwrap_or(&0.0));
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::{Error, Result, VectorError};
use crate::core::id::NodeId;
use crate::core::temporal::{TimeRange, Timestamp};
use crate::core::vector::ops::{dot_product, magnitude, normalize};
use std::collections::HashMap;

/// A point in the semantic evolution of a node.
#[derive(Debug, Clone)]
pub struct EvolutionPoint {
    /// The timestamp when this state became valid.
    pub timestamp: Timestamp,
    /// The semantic scores at this time.
    pub scores: HashMap<String, f32>,
}

/// A named axis in the semantic space.
#[derive(Debug, Clone)]
struct Axis {
    name: String,
    vector: Vec<f32>,
}

/// The Prism engine for vector decomposition.
pub struct Prism<'a> {
    db: &'a AletheiaDB,
    axes: Vec<Axis>,
    vector_property: Option<String>,
}

impl<'a> Prism<'a> {
    /// Create a new Prism.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self {
            db,
            axes: Vec::new(),
            vector_property: None,
        }
    }

    /// Configure the vector property Prism should use for node-based analysis.
    ///
    /// When set, `add_axis_from_node` and `analyze_node` always read this property.
    /// This avoids nondeterministic behavior on nodes with multiple vector properties.
    pub fn with_vector_property(mut self, property: &str) -> Self {
        self.vector_property = Some(property.to_string());
        self
    }

    fn resolve_node_vector<'n>(
        &self,
        node_id: NodeId,
        properties: &'n crate::core::property::PropertyMap,
        explicit_property: Option<&str>,
        context: &str,
    ) -> Result<&'n [f32]> {
        let selected_property = explicit_property.or(self.vector_property.as_deref());
        if let Some(property) = selected_property {
            let value = properties.get(property).ok_or_else(|| {
                Error::Vector(VectorError::IndexError(format!(
                    "Node {} is missing vector property '{}'",
                    node_id, property
                )))
            })?;

            return value.as_vector().ok_or_else(|| {
                Error::Vector(VectorError::IndexError(format!(
                    "Node {} property '{}' is not a dense vector",
                    node_id, property
                )))
            });
        }

        let mut first_vector: Option<&[f32]> = None;
        let mut vector_keys = Vec::new();

        for (key, value) in properties.iter() {
            if let Some(vector) = value.as_vector() {
                if first_vector.is_none() {
                    first_vector = Some(vector);
                }
                vector_keys.push(key.to_string());
            }
        }

        match vector_keys.len() {
            0 => Err(Error::Vector(VectorError::IndexError(format!(
                "Node {} has no vector properties to {}",
                node_id, context
            )))),
            1 => match first_vector {
                Some(vector) => Ok(vector),
                None => unreachable!("vector_keys length and first_vector state diverged"),
            },
            _ => {
                vector_keys.sort();
                Err(Error::Vector(VectorError::IndexError(format!(
                    "Node {} has multiple vector properties [{}]. Configure Prism with \
                     with_vector_property(...) or use *_with_property APIs for deterministic selection.",
                    node_id,
                    vector_keys.join(", ")
                ))))
            }
        }
    }

    /// Add an axis defined by a raw vector.
    /// The vector will be normalized to unit length.
    pub fn add_axis(&mut self, name: &str, vector: Vec<f32>) {
        // We normalize axes to ensure consistent projection values (cosine similarity-like)
        let normalized = normalize(&vector);
        self.axes.push(Axis {
            name: name.to_string(),
            vector: normalized,
        });
    }

    /// Add an axis defined by a node's vector.
    ///
    /// Selection rules:
    /// - Uses the configured property from `with_vector_property`, if set.
    /// - Otherwise requires exactly one dense vector property on the node.
    /// - Returns an error if the node has multiple vector properties.
    pub fn add_axis_from_node(&mut self, name: &str, node_id: NodeId) -> Result<()> {
        let node = self.db.get_node(node_id)?;
        let vector = self.resolve_node_vector(node_id, &node.properties, None, "use as an axis")?;

        self.add_axis(name, vector.to_vec());
        Ok(())
    }

    /// Add an axis defined by a specific vector property on a node.
    pub fn add_axis_from_node_with_property(
        &mut self,
        name: &str,
        node_id: NodeId,
        property: &str,
    ) -> Result<()> {
        let node = self.db.get_node(node_id)?;
        let vector =
            self.resolve_node_vector(node_id, &node.properties, Some(property), "use as an axis")?;
        self.add_axis(name, vector.to_vec());
        Ok(())
    }

    /// Orthogonalize the current axes using the Gram-Schmidt process.
    ///
    /// This ensures that the axes are mutually perpendicular, which makes
    /// the decomposition strictly additive (Component A + Component B = Target).
    ///
    /// Priority is given to axes added earlier (they keep their direction).
    /// Later axes have the components of earlier axes removed from them.
    pub fn orthogonalize(&mut self) {
        let mut ortho_axes: Vec<Axis> = Vec::with_capacity(self.axes.len());

        for axis in &self.axes {
            let mut v = axis.vector.clone();

            // Subtract projections onto existing orthogonal axes
            for u_axis in &ortho_axes {
                let u = &u_axis.vector;
                // proj_u(v) = (v . u) / (u . u) * u
                // Since u is normalized, u . u = 1.0
                // So proj_u(v) = (v . u) * u
                let dot = dot_product(&v, u).unwrap_or(0.0);

                // v = v - (dot * u)
                for (vi, ui) in v.iter_mut().zip(u.iter()) {
                    *vi -= dot * ui;
                }
            }

            // If the remaining vector is significant, add it
            if magnitude(&v) > 1e-6 {
                let normalized = normalize(&v);
                ortho_axes.push(Axis {
                    name: axis.name.clone(),
                    vector: normalized,
                });
            } else {
                // The axis was redundant (linear combination of previous axes)
                // We keep it but it will be effectively zero or we drop it?
                // Let's drop it to maintain a clean basis, but this might confuse users who added it.
                // Better to keep it? If it's zero, we can't normalize.
                // Let's skip it.
            }
        }

        self.axes = ortho_axes;
    }

    /// Analyze a raw vector against the axes.
    ///
    /// Returns a map of `Axis Name -> Score`.
    /// The score is the scalar projection of the target onto the axis.
    /// If axes are normalized, this is the dot product.
    pub fn analyze(&self, target: &[f32]) -> Result<HashMap<String, f32>> {
        let mut spectrum = HashMap::new();

        for axis in &self.axes {
            if axis.vector.len() != target.len() {
                // Skip mismatching dimensions or return error?
                // Returning error is safer.
                return Err(Error::Vector(VectorError::DimensionMismatch {
                    expected: axis.vector.len(),
                    actual: target.len(),
                }));
            }

            let score = dot_product(target, &axis.vector)?;
            spectrum.insert(axis.name.clone(), score);
        }

        Ok(spectrum)
    }

    /// Analyze a node's vector.
    ///
    /// Selection rules mirror `add_axis_from_node`.
    pub fn analyze_node(&self, node_id: NodeId) -> Result<HashMap<String, f32>> {
        let node = self.db.get_node(node_id)?;
        let vector = self.resolve_node_vector(node_id, &node.properties, None, "analyze")?;
        self.analyze(vector)
    }

    /// Analyze a node using a specific vector property.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use aletheiadb::AletheiaDB;
    /// use aletheiadb::core::property::PropertyMapBuilder;
    /// use aletheiadb::experimental::characterization::prism::Prism;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = AletheiaDB::new()?;
    /// let mut prism = Prism::new(&db);
    /// prism.add_axis("Logic", vec![1.0, 0.0]);
    ///
    /// let props = PropertyMapBuilder::new().insert_vector("custom_vec", &[1.0, 0.5]).build();
    /// let node = db.create_node("Concept", props)?;
    ///
    /// let spectrum = prism.analyze_node_with_property(node, "custom_vec")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn analyze_node_with_property(
        &self,
        node_id: NodeId,
        property: &str,
    ) -> Result<HashMap<String, f32>> {
        let node = self.db.get_node(node_id)?;
        let vector =
            self.resolve_node_vector(node_id, &node.properties, Some(property), "analyze")?;
        self.analyze(vector)
    }

    /// Analyze the semantic evolution of a node over a time range.
    ///
    /// This method retrieves the node's history and projects the vector at each version
    /// onto the defined axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use aletheiadb::AletheiaDB;
    /// use aletheiadb::core::property::PropertyMapBuilder;
    /// use aletheiadb::core::temporal::{TimeRange, time};
    /// use aletheiadb::api::transaction::WriteOps;
    /// use aletheiadb::experimental::characterization::prism::Prism;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = AletheiaDB::new()?;
    /// let mut prism = Prism::new(&db).with_vector_property("vec");
    /// prism.add_axis("Logic", vec![1.0, 0.0]);
    ///
    /// let t1 = time::from_millis(1000);
    /// let props1 = PropertyMapBuilder::new().insert_vector("vec", &[1.0, 0.0]).build();
    /// let node_id = db.write(|tx| tx.create_node_with_valid_time("Concept", props1, Some(t1)))?;
    ///
    /// let range = TimeRange::new(time::from_millis(0), time::from_millis(2000)).unwrap();
    /// let evolution = prism.analyze_evolution(node_id, range)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn analyze_evolution(
        &self,
        node_id: NodeId,
        time_range: TimeRange,
    ) -> Result<Vec<EvolutionPoint>> {
        let history = self.db.get_node_history(node_id)?;
        let mut points = Vec::new();

        for version in history.versions {
            // Check if the version is relevant to the time range
            if version.temporal.valid_time().overlaps(&time_range) {
                // Try to extract the vector
                // We ignore errors here (e.g. missing vector property in old versions)
                // because we want to return points for versions that DO have vectors.
                if let Ok(vector) = self.resolve_node_vector(
                    node_id,
                    &version.properties,
                    None,
                    "analyze evolution",
                ) {
                    let scores = self.analyze(vector)?;
                    points.push(EvolutionPoint {
                        timestamp: version.temporal.valid_time().start(),
                        scores,
                    });
                }
            }
        }

        Ok(points)
    }

    /// Calculate the "Residual" energy.
    ///
    /// Returns the magnitude of the part of the vector *not* explained by the current axes.
    /// `Residual = || Vector - Sum(Projection_i) ||`
    ///
    /// NOTE: This is only mathematically rigorous if the axes are orthogonal.
    pub fn residual(&self, target: &[f32]) -> Result<f32> {
        // Reconstruct the vector from components
        let mut reconstruction = vec![0.0; target.len()];

        for axis in &self.axes {
            let score = dot_product(target, &axis.vector)?;

            // Add (score * axis_vector) to reconstruction
            for (r, a) in reconstruction.iter_mut().zip(axis.vector.iter()) {
                *r += score * a;
            }
        }

        // Calculate difference
        let diff: Vec<f32> = target
            .iter()
            .zip(reconstruction.iter())
            .map(|(t, r)| t - r)
            .collect();

        Ok(magnitude(&diff))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;
    use crate::index::vector::{DistanceMetric, HnswConfig};

    #[test]
    fn test_prism_analysis() {
        let db = AletheiaDB::new().unwrap();
        // Enable vector index to ensure vector logic works (though Prism works on raw props too)
        let config = HnswConfig::new(2, DistanceMetric::Cosine);
        db.enable_vector_index("embedding", config).unwrap();

        let mut prism = Prism::new(&db);

        // Axis 1: X-axis (e.g., "Logic")
        prism.add_axis("Logic", vec![1.0, 0.0]);
        // Axis 2: Y-axis (e.g., "Emotion")
        prism.add_axis("Emotion", vec![0.0, 1.0]);

        // Target: [1.0, 1.0] (Logic + Emotion)
        let target = vec![1.0, 1.0];
        let spectrum = prism.analyze(&target).unwrap();

        assert!((spectrum.get("Logic").unwrap() - 1.0).abs() < 1e-5);
        assert!((spectrum.get("Emotion").unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prism_orthogonalization() {
        let db = AletheiaDB::new().unwrap();
        let mut prism = Prism::new(&db);

        // Axis 1: [1.0, 0.0]
        prism.add_axis("X", vec![1.0, 0.0]);
        // Axis 2: [1.0, 1.0] (Not orthogonal to X)
        prism.add_axis("Diag", vec![1.0, 1.0]);

        // Orthogonalize
        prism.orthogonalize();

        // Check axes
        // X should remain [1.0, 0.0]
        let x_axis = prism.axes.iter().find(|a| a.name == "X").unwrap();
        assert!((x_axis.vector[0] - 1.0).abs() < 1e-5);
        assert!((x_axis.vector[1] - 0.0).abs() < 1e-5);

        // Diag should become [0.0, 1.0] (Y-axis)
        // Original Diag was [0.707, 0.707] (normalized).
        // Wait, add_axis normalizes input!
        // Input: [1.0, 1.0] -> Normalized: [0.707, 0.707]
        // Proj_X(Diag) = (Diag . X) * X = 0.707 * [1, 0] = [0.707, 0.0]
        // Remainder = [0.707, 0.707] - [0.707, 0.0] = [0.0, 0.707]
        // Normalized Remainder = [0.0, 1.0]
        let diag_axis = prism.axes.iter().find(|a| a.name == "Diag").unwrap();
        assert!((diag_axis.vector[0] - 0.0).abs() < 1e-5);
        assert!((diag_axis.vector[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prism_residual() {
        let db = AletheiaDB::new().unwrap();
        let mut prism = Prism::new(&db);

        // Basis: Just X axis
        prism.add_axis("X", vec![1.0, 0.0]);

        // Target: [1.0, 1.0]
        // Explained: [1.0, 0.0]
        // Unexplained: [0.0, 1.0]
        // Residual Magnitude: 1.0

        let res = prism.residual(&[1.0, 1.0]).unwrap();
        assert!((res - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prism_from_node() {
        let db = AletheiaDB::new().unwrap();
        let config = HnswConfig::new(2, DistanceMetric::Cosine);
        db.enable_vector_index("vec", config).unwrap();

        let props = PropertyMapBuilder::new()
            .insert_vector("vec", &[1.0, 0.0])
            .build();
        let node = db.create_node("Basis", props).unwrap();

        let mut prism = Prism::new(&db);
        prism.add_axis_from_node("BasisNode", node).unwrap();

        let spectrum = prism.analyze(&[1.0, 0.0]).unwrap();
        assert!((spectrum.get("BasisNode").unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prism_rejects_ambiguous_vector_selection() {
        let db = AletheiaDB::new().unwrap();
        let props = PropertyMapBuilder::new()
            .insert_vector("text_vec", &[1.0, 0.0])
            .insert_vector("image_vec", &[0.0, 1.0])
            .build();
        let node = db.create_node("Doc", props).unwrap();

        let mut prism = Prism::new(&db);
        let err = prism.add_axis_from_node("DocAxis", node).unwrap_err();
        let err_msg = format!("{err}");
        assert!(err_msg.contains("multiple vector properties"));
        assert!(err_msg.contains("image_vec"));
        assert!(err_msg.contains("text_vec"));
    }

    #[test]
    fn test_prism_explicit_property_selection_is_deterministic() {
        let db = AletheiaDB::new().unwrap();
        let axis_props = PropertyMapBuilder::new()
            .insert_vector("text_vec", &[1.0, 0.0])
            .insert_vector("image_vec", &[0.0, 1.0])
            .build();
        let axis_node = db.create_node("Axis", axis_props).unwrap();

        let target_props = PropertyMapBuilder::new()
            .insert_vector("text_vec", &[1.0, 0.0])
            .insert_vector("image_vec", &[0.0, 1.0])
            .build();
        let target_node = db.create_node("Target", target_props).unwrap();

        let mut prism = Prism::new(&db).with_vector_property("text_vec");
        prism.add_axis_from_node("TextAxis", axis_node).unwrap();

        let spectrum = prism.analyze_node(target_node).unwrap();
        assert!((spectrum.get("TextAxis").unwrap() - 1.0).abs() < 1e-5);

        let mut by_property = Prism::new(&db);
        by_property
            .add_axis_from_node_with_property("ImageAxis", axis_node, "image_vec")
            .unwrap();
        let image_spectrum = by_property
            .analyze_node_with_property(target_node, "image_vec")
            .unwrap();
        assert!((image_spectrum.get("ImageAxis").unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prism_analyze_evolution() {
        use crate::api::transaction::WriteOps;
        use crate::core::temporal::{TimeRange, time};

        let db = AletheiaDB::new().unwrap();
        // Enable vector index so vectors are allowed
        let config = HnswConfig::new(2, DistanceMetric::Cosine);
        db.enable_vector_index("vec", config).unwrap();

        let t1 = time::from_millis(1000);
        let t2 = time::from_millis(2000);

        // 1. Create Node at T1: [1.0, 0.0] (Pure Logic)
        let props1 = PropertyMapBuilder::new()
            .insert_vector("vec", &[1.0, 0.0])
            .build();
        let node_id = db
            .write(|tx| tx.create_node_with_valid_time("Concept", props1, Some(t1)))
            .unwrap();

        // 2. Update Node at T2: [0.0, 1.0] (Pure Emotion)
        let props2 = PropertyMapBuilder::new()
            .insert_vector("vec", &[0.0, 1.0])
            .build();
        db.write(|tx| tx.update_node_with_valid_time(node_id, props2, Some(t2)))
            .unwrap();

        // 3. Setup Prism
        let mut prism = Prism::new(&db).with_vector_property("vec");
        prism.add_axis("Logic", vec![1.0, 0.0]);
        prism.add_axis("Emotion", vec![0.0, 1.0]);

        // 4. Analyze
        let range = TimeRange::new(time::from_millis(0), time::from_millis(3000)).unwrap();
        let points = prism.analyze_evolution(node_id, range).unwrap();

        // 5. Verify
        assert_eq!(points.len(), 2);

        // Point 1: T1, Logic=1, Emotion=0
        let p1 = &points[0];
        assert_eq!(p1.timestamp, t1);
        assert!((p1.scores.get("Logic").unwrap() - 1.0).abs() < 1e-5);
        assert!((p1.scores.get("Emotion").unwrap() - 0.0).abs() < 1e-5);

        // Point 2: T2, Logic=0, Emotion=1
        let p2 = &points[1];
        assert_eq!(p2.timestamp, t2);
        assert!((p2.scores.get("Logic").unwrap() - 0.0).abs() < 1e-5);
        assert!((p2.scores.get("Emotion").unwrap() - 1.0).abs() < 1e-5);
    }
}
