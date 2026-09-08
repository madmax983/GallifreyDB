//! Temporal Resonance Engine ("Echo").
//!
//! # The Spark
//! The "Echo" engine generates a `TemporalFingerprint` for a node based on its
//! mutation history (when it was updated) and compares these fingerprints
//! to find "resonant" nodes.
//!
//! # The Details
//! This module implements "Temporal Resonance", a way to find nodes that share
//! similar historical activity patterns.
//!
//! # The Hook
//! "Find me every sensor that failed with the same stuttering pattern as Sensor X."

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::history::EntityHistory;
use crate::core::id::NodeId;
use crate::core::temporal::time;

/// A normalized representation of a node's temporal activity.
///
/// # The Details
/// Temporal fingerprints are used to compute the similarity of update histories between nodes.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "semantic-temporal")]
/// # fn main() {
/// use aletheiadb::experimental::echo::TemporalFingerprint;
///
/// let fp1 = TemporalFingerprint { bins: vec![1.0, 0.0], resolution_us: 1000 };
/// let fp2 = TemporalFingerprint { bins: vec![0.0, 1.0], resolution_us: 1000 };
/// let fp3 = TemporalFingerprint { bins: vec![1.0, 0.0], resolution_us: 1000 };
///
/// // fp1 and fp2 are orthogonal
/// assert!(fp1.similarity(&fp2) < 0.01);
///
/// // fp1 and fp3 are identical
/// assert!(fp1.similarity(&fp3) > 0.99);
/// # }
/// # #[cfg(not(feature = "semantic-temporal"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone)]
pub struct TemporalFingerprint {
    /// Normalized activity bins (sum of squares = 1.0 for cosine similarity).
    pub bins: Vec<f32>,
    /// The resolution of each bin in microseconds.
    pub resolution_us: i64,
}

impl TemporalFingerprint {
    /// Compute the Cosine Similarity between this fingerprint and another.
    /// Returns a score between 0.0 (orthogonal) and 1.0 (identical).
    ///
    /// # Panics
    /// Panics if fingerprints have different lengths.
    pub fn similarity(&self, other: &Self) -> f32 {
        if self.bins.len() != other.bins.len() {
            // In a real system we might handle this gracefully, but for Nova we expect
            // the Resonator to produce consistent fingerprints.
            panic!(
                "Fingerprint length mismatch: {} vs {}",
                self.bins.len(),
                other.bins.len()
            );
        }

        let dot_product: f32 = self
            .bins
            .iter()
            .zip(other.bins.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Since bins are pre-normalized (unit vectors), cosine similarity is just the dot product.
        // We clamp to [0.0, 1.0] to handle floating point errors.
        dot_product.clamp(0.0, 1.0)
    }
}

/// A resonator that measures activity density over a fixed time window.
///
/// # The Details
/// It looks at the "Last N" microseconds of history and bins updates into
/// a histogram.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "semantic-temporal")]
/// # fn main() {
/// use aletheiadb::experimental::echo::ActivityDensityResonator;
///
/// let resonator = ActivityDensityResonator {
///     window_size_us: 3600 * 1_000_000, // 1 hour
///     num_bins: 60,                     // 1 minute resolution
/// };
/// # }
/// # #[cfg(not(feature = "semantic-temporal"))]
/// # fn main() {}
/// ```
pub struct ActivityDensityResonator {
    /// Total time window to consider (in microseconds).
    pub window_size_us: i64,
    /// Number of bins to divide the window into.
    pub num_bins: usize,
}

impl Default for ActivityDensityResonator {
    fn default() -> Self {
        Self {
            window_size_us: 3600 * 1_000_000, // 1 hour
            num_bins: 60,                     // 1 minute resolution
        }
    }
}

impl ActivityDensityResonator {
    /// Generate a fingerprint from the given history.
    pub fn resonate(&self, history: &EntityHistory) -> TemporalFingerprint {
        let mut bins = vec![0.0; self.num_bins];
        let bin_size_us = self.window_size_us / self.num_bins as i64;

        // We anchor the window to "now" (or the latest possible time).
        // Actually, to be deterministic for historical data, let's anchor to the *latest version*
        // in the history, OR use `time::now()`.
        // Using `time::now()` makes it sensitive to *when* you run the query (decaying echo).
        // Using `latest version` makes it relative to the entity's own timeline.
        //
        // "Echo" implies matching patterns in absolute time usually (e.g. "did everything crash at 2pm?").
        // So let's anchor to `time::now()`.
        let now = time::now().wallclock();
        let start_time = now - self.window_size_us;

        for version in &history.versions {
            let ts = version.temporal.valid_time().start().wallclock();

            if ts >= start_time && ts <= now {
                let offset = ts - start_time;
                let bin_idx = (offset / bin_size_us) as usize;
                if bin_idx < self.num_bins {
                    bins[bin_idx] += 1.0;
                }
            }
        }

        // Normalize (L2 Norm)
        let magnitude: f32 = bins.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for x in &mut bins {
                *x /= magnitude;
            }
        }

        TemporalFingerprint {
            bins,
            resolution_us: bin_size_us,
        }
    }
}

#[cfg(feature = "semantic-temporal")]
/// The Echo Chamber finds resonant nodes.
///
/// # The Details
/// Finds other nodes that have similar temporal update patterns (e.g., nodes that
/// are frequently updated at the same times).
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "semantic-temporal")]
/// # fn main() {
/// use aletheiadb::AletheiaDB;
/// use aletheiadb::experimental::echo::EchoChamber;
/// use aletheiadb::core::id::NodeId;
///
/// let db = AletheiaDB::new().unwrap();
/// let chamber = EchoChamber::new(&db);
///
/// let target_node = NodeId::new(1).unwrap();
/// let candidates = vec![NodeId::new(2).unwrap(), NodeId::new(3).unwrap()];
///
/// // chamber.find_echoes(target_node, &candidates);
/// # }
/// # #[cfg(not(feature = "semantic-temporal"))]
/// # fn main() {}
/// ```
/// A simulator for testing semantic drift and reinforcement loops.
///
/// # Why?
/// In recommender systems, repeatedly suggesting content similar to a user's
/// previous interactions creates an echo chamber. This struct simulates that
/// process over time to measure the degradation of content diversity.
pub struct EchoChamber<'a> {
    db: &'a AletheiaDB,
    resonator: ActivityDensityResonator,
}

#[cfg(not(feature = "semantic-temporal"))]
/// The Echo Chamber finds resonant nodes.
///
/// # The Details
/// Finds other nodes that have similar temporal update patterns (e.g., nodes that
/// are frequently updated at the same times).
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "semantic-temporal")]
/// # fn main() {
/// use aletheiadb::AletheiaDB;
/// use aletheiadb::experimental::echo::EchoChamber;
/// use aletheiadb::core::id::NodeId;
///
/// let db = AletheiaDB::new().unwrap();
/// let chamber = EchoChamber::new(&db);
///
/// let target_node = NodeId::new(1).unwrap();
/// let candidates = vec![NodeId::new(2).unwrap(), NodeId::new(3).unwrap()];
///
/// // chamber.find_echoes(target_node, &candidates);
/// # }
/// # #[cfg(not(feature = "semantic-temporal"))]
/// # fn main() {}
/// ```
#[deprecated(
    note = "EchoChamber requires the 'nova' feature. Add 'features = [\"nova\"]' to your Cargo.toml."
)]
/// A simulator for testing semantic drift and reinforcement loops.
///
/// # Why?
/// In recommender systems, repeatedly suggesting content similar to a user's
/// previous interactions creates an echo chamber. This struct simulates that
/// process over time to measure the degradation of content diversity.
pub struct EchoChamber<'a> {
    _marker: std::marker::PhantomData<&'a AletheiaDB>,
}

#[cfg(feature = "semantic-temporal")]
impl<'a> EchoChamber<'a> {
    /// Create a new EchoChamber with default settings.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self {
            db,
            resonator: ActivityDensityResonator::default(),
        }
    }

    /// Configure the EchoChamber with a custom resonator.
    pub fn with_resonator(mut self, resonator: ActivityDensityResonator) -> Self {
        self.resonator = resonator;
        self
    }

    /// Find nodes that resonate with the target node.
    pub fn find_echoes(&self, target: NodeId, candidates: &[NodeId]) -> Result<Vec<(NodeId, f32)>> {
        // 1. Get Target History & Fingerprint
        let target_history = self.db.get_node_history(target)?;
        let target_fp = self.resonator.resonate(&target_history);

        let mut results = Vec::with_capacity(candidates.len());

        // 2. Scan Candidates
        // Note: In a real implementation, we would want an index for this.
        // For Nova, a linear scan over provided candidates is acceptable.
        for &candidate_id in candidates {
            if candidate_id == target {
                continue;
            }

            // We accept that history might be missing for some candidates
            if let Ok(history) = self.db.get_node_history(candidate_id) {
                let fp = self.resonator.resonate(&history);
                let score = target_fp.similarity(&fp);
                if score > 0.0 {
                    results.push((candidate_id, score));
                }
            }
        }

        // 3. Sort by Score (Descending)
        // unwrap_or is safe because we clamped scores to 0.0-1.0 (no NaNs ideally, but safety first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}

#[cfg(not(feature = "semantic-temporal"))]
#[allow(deprecated)]
impl<'a> EchoChamber<'a> {
    /// Create a new EchoChamber with default settings.
    ///
    /// # Panics
    ///
    /// This method panics if the `nova` feature is not enabled.
    #[allow(unused_variables)]
    #[track_caller]
    pub fn new(db: &'a AletheiaDB) -> Self {
        panic!(
            "EchoChamber requires the 'nova' feature. Add 'features = [\"nova\"]' to your Cargo.toml."
        );
    }

    /// Configure the EchoChamber with a custom resonator.
    ///
    /// # Panics
    ///
    /// This method panics if the `nova` feature is not enabled.
    #[allow(unused_variables)]
    #[track_caller]
    pub fn with_resonator(self, resonator: ActivityDensityResonator) -> Self {
        panic!(
            "EchoChamber requires the 'nova' feature. Add 'features = [\"nova\"]' to your Cargo.toml."
        );
    }

    /// Find nodes that resonate with the target node.
    ///
    /// # Panics
    ///
    /// This method panics if the `nova` feature is not enabled.
    #[allow(unused_variables)]
    #[track_caller]
    pub fn find_echoes(&self, target: NodeId, candidates: &[NodeId]) -> Result<Vec<(NodeId, f32)>> {
        panic!(
            "EchoChamber requires the 'nova' feature. Add 'features = [\"nova\"]' to your Cargo.toml."
        );
    }
}

#[cfg(all(test, feature = "semantic-temporal"))]
mod tests {
    use super::*;
    use crate::api::transaction::WriteOps;
    use crate::core::hlc::HybridTimestamp;
    use crate::core::property::PropertyMapBuilder;
    use crate::core::temporal::{Timestamp, time};

    #[test]
    fn test_temporal_fingerprint_similarity() {
        let fp1 = TemporalFingerprint {
            bins: vec![1.0, 0.0], // Unit vector
            resolution_us: 1000,
        };
        let fp2 = TemporalFingerprint {
            bins: vec![0.0, 1.0], // Orthogonal
            resolution_us: 1000,
        };
        let fp3 = TemporalFingerprint {
            bins: vec![
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ], // ~45 degrees
            resolution_us: 1000,
        };

        assert!(fp1.similarity(&fp1) > 0.99); // Identity
        assert!(fp1.similarity(&fp2) < 0.01); // Orthogonal
        assert!(fp1.similarity(&fp3) > 0.6 && fp1.similarity(&fp3) < 0.8); // 45 deg
    }

    #[test]
    fn test_echo_chamber_integration() {
        let db = AletheiaDB::new().unwrap();

        // Define a "Now" for our test universe
        let now_wallclock = time::now().wallclock();
        // Window: Last 100 seconds
        let window = 100 * 1_000_000;

        // Create 3 nodes with specific histories

        // Node A (Target): Updates at T-80s, T-50s
        // Node B (Resonant): Updates at T-80s, T-50s (Identical pattern)
        // Node C (Noise): Updates at T-20s (Different pattern)

        let t_minus_80 = HybridTimestamp::new(now_wallclock - 80 * 1_000_000, 0).unwrap();
        let t_minus_50 = HybridTimestamp::new(now_wallclock - 50 * 1_000_000, 0).unwrap();
        let t_minus_20 = HybridTimestamp::new(now_wallclock - 20 * 1_000_000, 0).unwrap();

        let props = PropertyMapBuilder::new().insert("val", 0).build();

        // Helper to populate history
        let create_node_with_history = |timestamps: Vec<Timestamp>| -> NodeId {
            assert!(
                !timestamps.is_empty(),
                "history requires at least one timestamp"
            );

            // Create and commit the base version first.
            let id = {
                let mut tx = db.write_transaction().unwrap();
                let id = tx
                    .create_node_with_valid_time("Node", props.clone(), Some(timestamps[0]))
                    .unwrap();
                tx.commit().unwrap();
                id
            };

            // Apply subsequent versions as committed updates.
            for &ts in &timestamps[1..] {
                let mut tx = db.write_transaction().unwrap();
                tx.update_node_with_valid_time(id, props.clone(), Some(ts))
                    .unwrap();
                tx.commit().unwrap();
            }

            id
        };

        let node_a = create_node_with_history(vec![t_minus_80, t_minus_50]);
        let node_b = create_node_with_history(vec![t_minus_80, t_minus_50]);
        let node_c = create_node_with_history(vec![t_minus_20]);

        // Configure Resonator: 100s window, 10 bins (10s per bin)
        let resonator = ActivityDensityResonator {
            window_size_us: window,
            num_bins: 10,
        };

        let chamber = EchoChamber::new(&db).with_resonator(resonator);

        // Find echoes for Node A among B and C
        let echoes = chamber.find_echoes(node_a, &[node_b, node_c]).unwrap();

        // Expectation:
        // Node B should be ~1.0 (Identical)
        // Node C should be ~0.0 (Different bins)

        assert_eq!(echoes.len(), 1, "Should only return non-zero similarity"); // C might be 0.0

        let (id, score) = echoes[0];
        assert_eq!(id, node_b);
        assert!(
            score > 0.9,
            "Node B should resonate strongly (score: {})",
            score
        );

        // Verify C is not in results or very low
        let c_score = echoes.iter().find(|(id, _)| *id == node_c);
        if let Some((_, score)) = c_score {
            assert!(
                *score < 0.1,
                "Node C should not resonate (score: {})",
                score
            );
        }
    }
}

#[cfg(all(test, not(feature = "semantic-temporal")))]
#[allow(deprecated)]
mod stub_tests {
    use super::*;

    #[test]
    #[should_panic(
        expected = "EchoChamber requires the 'nova' feature. Add 'features = [\"nova\"]' to your Cargo.toml."
    )]
    fn test_stub_panic_on_new() {
        let db = AletheiaDB::new().unwrap();
        let _ = EchoChamber::new(&db);
    }

    #[test]
    #[should_panic(
        expected = "EchoChamber requires the 'nova' feature. Add 'features = [\"nova\"]' to your Cargo.toml."
    )]
    fn test_stub_panic_on_with_resonator() {
        let chamber = EchoChamber {
            _marker: std::marker::PhantomData,
        };
        // We need a dummy resonator. Since Resonator trait is public but ActivityDensityResonator is only stubbed out...
        // Actually ActivityDensityResonator is not gated?
        // Let's check the code above. ActivityDensityResonator struct definition is NOT gated.
        // So we can use it.
        let resonator = ActivityDensityResonator::default();
        let _ = chamber.with_resonator(resonator);
    }

    #[test]
    #[should_panic(
        expected = "EchoChamber requires the 'nova' feature. Add 'features = [\"nova\"]' to your Cargo.toml."
    )]
    fn test_stub_panic_on_find_echoes() {
        let chamber = EchoChamber {
            _marker: std::marker::PhantomData,
        };
        let _ = chamber.find_echoes(NodeId::new(0).unwrap(), &[]);
    }
}
