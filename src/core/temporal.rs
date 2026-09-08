//! Temporal primitives for bi-temporal graph database.
//!
//! This module implements the core temporal types that enable bi-temporality:
//! - Valid time: when facts were true in the real world
//! - Transaction time: when facts were recorded in the database
//!
//! Every graph element (node/edge version) has a BiTemporalInterval that tracks
//! both dimensions of time.
//!
//! # The Bi-Temporal Story
//!
//! Imagine a company updates an employee's salary retroactively.
//! The `Valid Time` captures *when* that raise actually goes into effect
//! in the real world. But what if someone queries the database asking,
//! "What was their salary reported *yesterday*?"
//!
//! By tracking `Transaction Time`, the database knows *when* it learned the fact.
//! This allows AletheiaDB to accurately reconstruct what the database "knew"
//! at any arbitrary point in history, protecting against overwritten realities.
//!
//! # Gotchas & Corner Cases ⚠️
//!
//! Temporal logic is tricky. Here are common pitfalls to avoid:
//!
//! 1. **MAX_VALID_TIMESTAMP**: Timestamps are capped at [`MAX_VALID_TIMESTAMP`] (`i64::MAX - 1000`).
//!    Attempts to create a `TimeRange` exceeding this will fail. This prevents DoS attacks and
//!    ensures reserved space for sentinels.
//!
//! 2. **Range Containment**: `TimeRange::contains_range(other)` is reflexive (a range contains itself)
//!    and handles exact matches. Be careful with "off-by-one" errors at boundaries.
//!    - `[100, 200)` contains `[100, 200)` -> true
//!    - `[100, 200)` contains `[100, 101)` -> true
//!
//! 3. **Visibility Logic**: [`BiTemporalInterval::is_visible_at`] requires *both* dimensions to be satisfied.
//!    A fact might be valid in the real world (valid time) but not yet known to the database
//!    (transaction time) at a specific query point.
//!
//! 4. **Half-Open Intervals**: All ranges are `[start, end)`, meaning `start` is inclusive and
//!    `end` is exclusive. `contains(end)` will always return false.
//!
//! # Common Pitfalls 🕳️
//!
//! ## ❌ OFF-BY-ONE: Range Exclusion
//! The end of a `TimeRange` is **exclusive**.
//!
//! ```rust
//! # use aletheiadb::core::temporal::{TimeRange, time};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let start = time::from_secs(100);
//! let end = time::from_secs(200);
//! let range = TimeRange::new(start, end)?;
//!
//! // ❌ Pitfall: Expecting the end timestamp to be included
//! assert_eq!(range.contains(end), false);
//!
//! // ✅ Correct: Use end-1 or check < end
//! assert_eq!(range.contains(time::from_secs(199)), true);
//! # Ok(())
//! # }
//! ```
//!
//! ## ❌ MAX_VALID_TIMESTAMP Violation
//! Timestamps cannot exceed `MAX_VALID_TIMESTAMP`.
//!
//! ```rust
//! # use aletheiadb::core::temporal::{TimeRange, MAX_VALID_TIMESTAMP, time};
//! # use aletheiadb::core::hlc::HybridTimestamp;
//! # fn main() {
//! // ❌ Pitfall: Trying to create a timestamp > MAX_VALID_TIMESTAMP
//! let too_big_val = i64::MAX;
//! // This will return an error (except for internal sentinels)
//! assert!(HybridTimestamp::new(too_big_val, 0).is_err());
//!
//! // ✅ Correct: Use TimeRange::from(start) for open ranges
//! let start = time::from_secs(100);
//! let range = TimeRange::from(start); // Automatically uses TIMESTAMP_MAX
//! assert!(range.is_current());
//! # }
//! ```

use std::fmt;

use crate::core::error::{StorageError, TemporalError};
use crate::core::hlc::HybridTimestamp;

/// Timestamp represented as Hybrid Logical Clock (HLC).
///
/// Phase 2: Migrated from i64 to HybridTimestamp for distributed temporal consistency.
///
/// HybridTimestamp combines:
/// - Wallclock component (i64 microseconds since Unix epoch)
/// - Logical counter (u32) for ordering events at same wallclock
///
/// This provides:
/// - Monotonic ordering despite clock skew
/// - Causality preservation across distributed nodes
/// - Human-readable wallclock semantics for temporal queries
pub type Timestamp = HybridTimestamp;

/// Sentinel value representing "infinity" or "current" timestamp.
///
/// Used for open-ended time ranges that extend to the present.
/// For HybridTimestamp, this is (i64::MAX, 0) - maximum wallclock, zero logical counter.
pub const TIMESTAMP_MAX: Timestamp = HybridTimestamp::new_unchecked(i64::MAX, 0);

/// Maximum valid timestamp wallclock value for user data.
///
/// Similar to MAX_VALID_ID, this reserves the upper 1000 i64 values for internal use
/// (sentinel values, metadata). Timestamps exceeding this value are rejected to prevent
/// DoS attacks and ensure system integrity.
///
/// The reserved range provides a safety margin without meaningfully restricting the
/// timestamp space (still covers ~290,000 years before/after epoch).
///
/// Note: This is the maximum wallclock value. The logical counter can be any u32 value.
pub const MAX_VALID_TIMESTAMP: i64 = i64::MAX - 1000;

/// Represents a continuous range of time [start, end).
///
/// The range includes the start timestamp but excludes the end timestamp.
/// An end value of TIMESTAMP_MAX represents an open-ended range (still current).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeRange {
    start: Timestamp,
    end: Timestamp,
}

impl TimeRange {
    /// Create a new time range.
    ///
    /// # Errors
    /// Returns `TemporalError::InvalidTimeRange` if start > end.
    #[inline]
    pub fn new(start: Timestamp, end: Timestamp) -> Result<Self, TemporalError> {
        if start > end {
            return Err(TemporalError::InvalidTimeRange { start, end });
        }

        // Warden: Validate that timestamps don't exceed MAX_VALID_TIMESTAMP
        // This prevents invalid timestamps from entering via unchecked constructors
        if start.wallclock() > MAX_VALID_TIMESTAMP && start != TIMESTAMP_MAX {
            return Err(TemporalError::InvalidTimestamp {
                timestamp: start,
                reason: format!(
                    "Start timestamp exceeds MAX_VALID_TIMESTAMP ({})",
                    MAX_VALID_TIMESTAMP
                ),
            });
        }

        if end.wallclock() > MAX_VALID_TIMESTAMP && end != TIMESTAMP_MAX {
            return Err(TemporalError::InvalidTimestamp {
                timestamp: end,
                reason: format!(
                    "End timestamp exceeds MAX_VALID_TIMESTAMP ({})",
                    MAX_VALID_TIMESTAMP
                ),
            });
        }

        Ok(TimeRange { start, end })
    }

    /// Create a time range that starts at the given timestamp and is still current.
    ///
    /// # Panics
    /// Panics if start > MAX_VALID_TIMESTAMP and start != TIMESTAMP_MAX.
    #[inline]
    pub fn from(start: Timestamp) -> Self {
        if start.wallclock() > MAX_VALID_TIMESTAMP && start != TIMESTAMP_MAX {
            panic!(
                "Start timestamp exceeds MAX_VALID_TIMESTAMP ({})",
                MAX_VALID_TIMESTAMP
            );
        }
        TimeRange {
            start,
            end: TIMESTAMP_MAX,
        }
    }

    /// Create a time range that is bounded on both ends.
    #[inline]
    pub fn between(start: Timestamp, end: Timestamp) -> Result<Self, TemporalError> {
        Self::new(start, end)
    }

    /// Create a point-in-time range (instant with zero duration).
    ///
    /// # Panics
    /// Panics if timestamp > MAX_VALID_TIMESTAMP and timestamp != TIMESTAMP_MAX.
    #[inline]
    pub fn at(timestamp: Timestamp) -> Self {
        if timestamp.wallclock() > MAX_VALID_TIMESTAMP && timestamp != TIMESTAMP_MAX {
            panic!(
                "Timestamp exceeds MAX_VALID_TIMESTAMP ({})",
                MAX_VALID_TIMESTAMP
            );
        }
        TimeRange {
            start: timestamp,
            end: timestamp,
        }
    }

    /// Get the start timestamp.
    #[inline]
    pub const fn start(&self) -> Timestamp {
        self.start
    }

    /// Get the end timestamp.
    #[inline]
    pub const fn end(&self) -> Timestamp {
        self.end
    }

    /// Returns true if this range is currently open-ended (end == TIMESTAMP_MAX).
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn is_current(&self) -> bool {
        self.end == TIMESTAMP_MAX
    }

    /// Returns true if this range has been closed (end < TIMESTAMP_MAX).
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.end < TIMESTAMP_MAX
    }

    /// Returns true if the given timestamp is contained within this range [start, end).
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn contains(&self, timestamp: Timestamp) -> bool {
        timestamp >= self.start && timestamp < self.end
    }

    /// Returns true if the given timestamp is at or after the start of this range.
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn contains_or_after(&self, timestamp: Timestamp) -> bool {
        timestamp >= self.start
    }

    /// Returns true if the range is empty (start == end).
    ///
    /// An empty range contains no timestamps and has zero duration.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Returns true if this range overlaps with another range.
    ///
    /// Two ranges overlap if there exists any timestamp that is in both ranges.
    /// Note: Returns false if either range is empty, as an empty set cannot overlap anything.
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        self.start < other.end && other.start < self.end
    }

    /// Returns true if this range completely contains another range.
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn contains_range(&self, other: &TimeRange) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Close this range at the given timestamp.
    ///
    /// Returns a new TimeRange with the same start but the specified end.
    ///
    /// # Errors
    /// Returns `TemporalError::InvalidTimeRange` if end < start.
    /// Returns `TemporalError::InvalidTimestamp` if end > MAX_VALID_TIMESTAMP.
    #[inline]
    pub fn close_at(self, end: Timestamp) -> Result<Self, TemporalError> {
        if end < self.start {
            return Err(TemporalError::InvalidTimeRange {
                start: self.start,
                end,
            });
        }

        if end.wallclock() > MAX_VALID_TIMESTAMP && end != TIMESTAMP_MAX {
            return Err(TemporalError::InvalidTimestamp {
                timestamp: end,
                reason: format!(
                    "End timestamp exceeds MAX_VALID_TIMESTAMP ({})",
                    MAX_VALID_TIMESTAMP
                ),
            });
        }

        Ok(TimeRange {
            start: self.start,
            end,
        })
    }

    /// Returns the duration of this range in microseconds.
    ///
    /// Returns None if the range is open-ended (current).
    /// Duration is calculated using wallclock components only.
    /// Note: Phase 2 - removed const due to is_current() needing HybridTimestamp comparison.
    #[inline]
    pub fn duration_micros(&self) -> Option<i64> {
        if self.is_current() {
            None
        } else {
            // Saturate at i64::MAX on overflow (e.g. extremely old start time)
            // This prevents panics when start is close to i64::MIN
            self.end
                .wallclock()
                .checked_sub(self.start.wallclock())
                .or(Some(i64::MAX))
        }
    }

    /// Serialize this TimeRange to bytes.
    ///
    /// # Binary Format (Phase 2: HybridTimestamp)
    /// ```text
    /// [start.wallclock:8][start.logical:4][end.wallclock:8][end.logical:4]
    /// ```
    /// Total: 24 bytes (2 HybridTimestamps × 12 bytes each)
    pub fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(24);
        self.serialize_into(&mut buffer);
        buffer
    }

    /// Serialize into an existing buffer.
    pub fn serialize_into(&self, buffer: &mut Vec<u8>) {
        self.start.serialize_into(buffer);
        self.end.serialize_into(buffer);
    }

    /// Deserialize a TimeRange from bytes.
    ///
    /// Returns the TimeRange and number of bytes consumed (always 24).
    pub fn deserialize(bytes: &[u8]) -> Result<(Self, usize), StorageError> {
        if bytes.len() < 24 {
            return Err(StorageError::CorruptedData(format!(
                "Buffer too short for TimeRange: {} bytes (need 24)",
                bytes.len()
            )));
        }
        let (start, _) = HybridTimestamp::deserialize(&bytes[0..12])?;
        let (end, _) = HybridTimestamp::deserialize(&bytes[12..24])?;

        if start > end {
            return Err(StorageError::CorruptedData(format!(
                "Deserialized TimeRange invalid: start {} > end {}",
                start, end
            )));
        }

        Ok((TimeRange { start, end }, 24))
    }
}

impl fmt::Display for TimeRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_current() {
            write!(f, "[{}, current)", self.start)
        } else {
            write!(f, "[{}, {})", self.start, self.end)
        }
    }
}

/// Bi-temporal interval tracking both valid time and transaction time.
///
/// This is the core temporal primitive that enables time-traveling queries:
/// - **Valid time**: When the fact was true in the real world
/// - **Transaction time**: When the fact was recorded in the database
///
/// This creates a 2D time space that allows queries like:
/// - "What did we know about X at time T?" (transaction time)
/// - "What was true about X in reality at time T?" (valid time)
/// - "When did we record that X was true at time T?" (both)
///
/// # Example
///
/// ```rust
/// use aletheiadb::core::temporal::{BiTemporalInterval, time};
///
/// // Create a bi-temporal interval
/// // - Valid from T=100 (in reality)
/// // - Recorded at T=200 (in database)
/// let valid_from = time::from_secs(100);
/// let recorded_at = time::from_secs(200);
/// let interval = BiTemporalInterval::with_valid_time(valid_from, recorded_at);
///
/// // Query at different times
/// let t150 = time::from_secs(150);
/// let t250 = time::from_secs(250);
///
/// // 1. Was it valid at T=150? YES.
/// assert!(interval.is_valid_at(t150));
///
/// // 2. Did we KNOW it was valid at T=150? NO (recorded at 200).
/// assert!(!interval.is_visible_at(t150, t150));
///
/// // 3. Do we KNOW it was valid at T=150 if we query at T=250? YES.
/// assert!(interval.is_visible_at(t150, t250));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BiTemporalInterval {
    /// When the fact was true in the real world.
    valid_time: TimeRange,
    /// When the fact was recorded in the database.
    transaction_time: TimeRange,
}

impl BiTemporalInterval {
    /// Create a new bi-temporal interval.
    #[inline]
    pub const fn new(valid_time: TimeRange, transaction_time: TimeRange) -> Self {
        BiTemporalInterval {
            valid_time,
            transaction_time,
        }
    }

    /// Create a bi-temporal interval where both dimensions start at the same time
    /// and are currently open.
    #[inline]
    pub fn current(timestamp: Timestamp) -> Self {
        let range = TimeRange::from(timestamp);
        BiTemporalInterval {
            valid_time: range,
            transaction_time: range,
        }
    }

    /// Create a bi-temporal interval for a fact that is currently valid and was just recorded.
    ///
    /// This is the most common case: recording a fact that is true now.
    #[inline]
    pub fn now(valid_start: Timestamp, tx_timestamp: Timestamp) -> Self {
        BiTemporalInterval {
            valid_time: TimeRange::from(valid_start),
            transaction_time: TimeRange::from(tx_timestamp),
        }
    }

    /// Create a bi-temporal interval with separate valid_from and transaction_time.
    ///
    /// This enables true bi-temporal semantics where:
    /// - `valid_from`: when the fact became true in reality (user-controlled)
    /// - `tx_time`: when the fact was recorded in the database (system-managed)
    ///
    /// Both dimensions are open-ended (current), but start at different times.
    /// This allows backdating facts - recording at time T2 that something was
    /// valid since time T1 (where T1 < T2).
    ///
    /// # Example
    /// ```
    /// use aletheiadb::core::temporal::BiTemporalInterval;
    /// use aletheiadb::core::hlc::HybridTimestamp;
    ///
    /// // Record today that Alice joined the company last month
    /// let last_month = HybridTimestamp::new(1704067200_000_000, 0).unwrap();
    /// let today = HybridTimestamp::new(1706745600_000_000, 0).unwrap();
    ///
    /// let interval = BiTemporalInterval::with_valid_time(last_month, today);
    ///
    /// // Valid time starts last month
    /// assert_eq!(interval.valid_time().start(), last_month);
    /// // Transaction time starts today
    /// assert_eq!(interval.transaction_time().start(), today);
    /// ```
    #[inline]
    pub fn with_valid_time(valid_from: Timestamp, tx_time: Timestamp) -> Self {
        BiTemporalInterval {
            valid_time: TimeRange::from(valid_from),
            transaction_time: TimeRange::from(tx_time),
        }
    }

    /// Get the valid time range.
    #[inline]
    pub const fn valid_time(&self) -> TimeRange {
        self.valid_time
    }

    /// Get the transaction time range.
    #[inline]
    pub const fn transaction_time(&self) -> TimeRange {
        self.transaction_time
    }

    /// Returns true if this interval is currently valid (valid time is open).
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn is_currently_valid(&self) -> bool {
        self.valid_time.is_current()
    }

    /// Returns true if this interval is currently in the database (transaction time is open).
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn is_currently_recorded(&self) -> bool {
        self.transaction_time.is_current()
    }

    /// Returns true if this interval is current in both dimensions.
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn is_current(&self) -> bool {
        self.is_currently_valid() && self.is_currently_recorded()
    }

    /// Check if this interval is visible at the given valid time.
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn is_valid_at(&self, timestamp: Timestamp) -> bool {
        self.valid_time.contains(timestamp)
    }

    /// Check if this interval was recorded by the given transaction time.
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn is_recorded_at(&self, timestamp: Timestamp) -> bool {
        self.transaction_time.contains(timestamp)
    }

    /// Check if this interval is visible in both dimensions at the given times.
    ///
    /// This answers: "At transaction time T1, did we believe this fact was true at valid time T2?"
    /// Note: Phase 2 - removed const due to HybridTimestamp comparison.
    #[inline]
    pub fn is_visible_at(&self, valid_time: Timestamp, tx_time: Timestamp) -> bool {
        self.valid_time.contains(valid_time) && self.transaction_time.contains(tx_time)
    }

    /// Close the valid time dimension at the given timestamp.
    ///
    /// This marks a fact as no longer valid in the real world.
    #[inline]
    pub fn close_valid_time(self, end: Timestamp) -> Result<Self, TemporalError> {
        Ok(BiTemporalInterval {
            valid_time: self.valid_time.close_at(end)?,
            transaction_time: self.transaction_time,
        })
    }

    /// Close the transaction time dimension at the given timestamp.
    ///
    /// This marks when we stopped believing this version of the fact.
    #[inline]
    pub fn close_transaction_time(self, end: Timestamp) -> Result<Self, TemporalError> {
        Ok(BiTemporalInterval {
            valid_time: self.valid_time,
            transaction_time: self.transaction_time.close_at(end)?,
        })
    }

    /// Close both time dimensions.
    #[inline]
    pub fn close_both(
        self,
        valid_end: Timestamp,
        tx_end: Timestamp,
    ) -> Result<Self, TemporalError> {
        Ok(BiTemporalInterval {
            valid_time: self.valid_time.close_at(valid_end)?,
            transaction_time: self.transaction_time.close_at(tx_end)?,
        })
    }

    /// Serialize this BiTemporalInterval to bytes.
    ///
    /// # Binary Format (Phase 2: HybridTimestamp)
    /// ```text
    /// [valid_time: 24 bytes][transaction_time: 24 bytes]
    /// ```
    /// Total: 48 bytes (2 TimeRanges × 24 bytes each)
    pub fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(48);
        self.serialize_into(&mut buffer);
        buffer
    }

    /// Serialize into an existing buffer.
    pub fn serialize_into(&self, buffer: &mut Vec<u8>) {
        self.valid_time.serialize_into(buffer);
        self.transaction_time.serialize_into(buffer);
    }

    /// Deserialize a BiTemporalInterval from bytes.
    ///
    /// Returns the BiTemporalInterval and number of bytes consumed (always 48).
    pub fn deserialize(bytes: &[u8]) -> Result<(Self, usize), StorageError> {
        if bytes.len() < 48 {
            return Err(StorageError::CorruptedData(format!(
                "Buffer too short for BiTemporalInterval: {} bytes (need 48)",
                bytes.len()
            )));
        }
        let (valid_time, _) = TimeRange::deserialize(&bytes[0..24])?;
        let (transaction_time, _) = TimeRange::deserialize(&bytes[24..48])?;
        Ok((
            BiTemporalInterval {
                valid_time,
                transaction_time,
            },
            48,
        ))
    }
}

impl fmt::Display for BiTemporalInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BiTemporal[valid: {}, tx: {}]",
            self.valid_time, self.transaction_time
        )
    }
}

/// Helper functions for working with timestamps.
pub mod time {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Get the current system time as a Timestamp (HybridTimestamp).
    ///
    /// Returns HybridTimestamp with current wallclock and logical counter = 0.
    /// For monotonic HLC generation with causality, use HLC's send() method.
    ///
    /// When the `simulation` feature is active (or in unit-test builds) and a
    /// `SimulatedClock` injection guard is live on the current thread, returns
    /// the simulated time instead.
    ///
    /// # Panics
    /// Panics if the system clock is set before Unix epoch. Use [`try_now`] for
    /// a fallible version that returns `Result` instead.
    pub fn now() -> Timestamp {
        #[cfg(any(test, feature = "simulation"))]
        if let Some(micros) = crate::simulation::clock::thread_local_now() {
            return HybridTimestamp::new_unchecked(micros, 0);
        }
        try_now().expect("System clock is before Unix epoch")
    }

    /// Fallible version of [`now`] that returns `Result` instead of panicking.
    ///
    /// Returns an error if the system clock is set before Unix epoch.
    /// Most callers should use [`now`] instead, since a pre-epoch clock
    /// is an unrecoverable system-level error.
    pub fn try_now() -> Result<Timestamp, crate::core::error::TemporalError> {
        let wallclock = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| crate::core::error::TemporalError::TemporalParadox {
                reason: "System clock is before Unix epoch".to_string(),
            })?
            .as_micros() as i64;

        // Return HybridTimestamp with logical counter = 0
        // Use new_unchecked for performance (system clock is trusted)
        Ok(HybridTimestamp::new_unchecked(wallclock, 0))
    }

    /// Convert a Timestamp to a human-readable ISO 8601 string (UTC).
    ///
    /// Returns "current" for TIMESTAMP_MAX.
    /// Uses the wallclock component for display.
    pub fn to_iso8601(timestamp: Timestamp) -> String {
        if timestamp == TIMESTAMP_MAX {
            return "current".to_string();
        }

        let wallclock = timestamp.wallclock();
        // Build the offset from the epoch without panicking on negative (pre-1970) or
        // extreme wallclocks: `unsigned_abs` handles `i64::MIN`, and checked add/sub guard
        // against SystemTime overflow. A non-representable instant falls back to a raw
        // microsecond form rather than panicking.
        let dur = std::time::Duration::from_micros(wallclock.unsigned_abs());
        let datetime = if wallclock >= 0 {
            UNIX_EPOCH.checked_add(dur)
        } else {
            UNIX_EPOCH.checked_sub(dur)
        };
        match datetime {
            // Simplified conversion - for production use the chrono crate.
            Some(dt) => format!("{:?}", dt),
            None => format!("{wallclock}us"),
        }
    }

    /// Create a timestamp from seconds since Unix epoch.
    /// Logical counter is set to 0.
    #[inline]
    pub const fn from_secs(secs: i64) -> Timestamp {
        HybridTimestamp::new_unchecked(secs * 1_000_000, 0)
    }

    /// Create a timestamp from milliseconds since Unix epoch.
    /// Logical counter is set to 0.
    #[inline]
    pub const fn from_millis(millis: i64) -> Timestamp {
        HybridTimestamp::new_unchecked(millis * 1_000, 0)
    }

    /// Convert a timestamp to seconds since Unix epoch.
    /// Uses the wallclock component.
    #[inline]
    pub const fn to_secs(timestamp: Timestamp) -> i64 {
        timestamp.wallclock() / 1_000_000
    }

    /// Convert a timestamp to milliseconds since Unix epoch.
    /// Uses the wallclock component.
    #[inline]
    pub const fn to_millis(timestamp: Timestamp) -> i64 {
        timestamp.wallclock() / 1_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_iso8601_handles_pre_epoch_without_panicking() {
        // A pre-1970 (negative wallclock) timestamp must not panic during formatting.
        let pre_epoch = HybridTimestamp::new_unchecked(-1_000_000, 0);
        let s = time::to_iso8601(pre_epoch);
        assert!(!s.is_empty());
        // Extreme negative must also be total.
        let extreme = HybridTimestamp::new_unchecked(i64::MIN, 0);
        let _ = time::to_iso8601(extreme);
        // TIMESTAMP_MAX still renders as "current".
        assert_eq!(time::to_iso8601(TIMESTAMP_MAX), "current");
    }

    #[test]
    fn test_time_range_creation() {
        let range = TimeRange::new(100.into(), 200.into()).unwrap();
        assert_eq!(range.start(), 100.into());
        assert_eq!(range.end(), 200.into());
        assert!(!range.is_current());
        assert!(range.is_closed());
    }

    #[test]
    fn test_time_range_current() {
        let range = TimeRange::from(100.into());
        assert_eq!(range.start(), 100.into());
        assert_eq!(range.end(), TIMESTAMP_MAX);
        assert!(range.is_current());
        assert!(!range.is_closed());
    }

    #[test]
    fn test_time_range_contains() {
        let range = TimeRange::new(100.into(), 200.into()).unwrap();
        assert!(!range.contains(99.into()));
        assert!(range.contains(100.into()));
        assert!(range.contains(150.into()));
        assert!(range.contains(199.into()));
        assert!(!range.contains(200.into())); // Exclusive end
    }

    #[test]
    fn test_time_range_overlaps() {
        let r1 = TimeRange::new(100.into(), 200.into()).unwrap();
        let r2 = TimeRange::new(150.into(), 250.into()).unwrap();
        let r3 = TimeRange::new(200.into(), 300.into()).unwrap();
        let r4 = TimeRange::new(50.into(), 75.into()).unwrap();

        assert!(r1.overlaps(&r2));
        assert!(r2.overlaps(&r1));
        assert!(!r1.overlaps(&r3)); // Touching but not overlapping
        assert!(!r1.overlaps(&r4));
    }

    #[test]
    fn test_time_range_overlaps_touching_repro() {
        let r1 = TimeRange::new(100.into(), 200.into()).unwrap();
        let r2 = TimeRange::new(200.into(), 300.into()).unwrap();

        // Ensure symmetry: neither should overlap the other
        // This prevents the mutant where "start < end" becomes "start <= end"
        // which would cause r2.overlaps(r1) to be true (200 <= 200)
        assert!(!r1.overlaps(&r2));
        assert!(!r2.overlaps(&r1));
    }

    #[test]
    fn test_time_range_contains_range() {
        let outer = TimeRange::new(100.into(), 300.into()).unwrap();
        let inner = TimeRange::new(150.into(), 250.into()).unwrap();
        let overlapping = TimeRange::new(150.into(), 350.into()).unwrap();

        assert!(outer.contains_range(&inner));
        assert!(!inner.contains_range(&outer));
        assert!(!outer.contains_range(&overlapping));
    }

    #[test]
    fn test_time_range_close_at() {
        let open = TimeRange::from(100.into());
        let closed = open.close_at(200.into()).unwrap();

        assert!(open.is_current());
        assert!(!closed.is_current());
        assert_eq!(closed.start(), 100.into());
        assert_eq!(closed.end(), 200.into());
    }

    #[test]
    fn test_time_range_close_at_invalid() {
        let open = TimeRange::from(100.into());
        let result = open.close_at(50.into());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TemporalError::InvalidTimeRange { .. }
        ));
    }

    #[test]
    fn test_time_range_duration() {
        let range = TimeRange::new(100.into(), 500.into()).unwrap();
        assert_eq!(range.duration_micros(), Some(400.into()));

        let open = TimeRange::from(100.into());
        assert_eq!(open.duration_micros(), None);
    }

    #[test]
    fn test_bitemporal_current() {
        let interval = BiTemporalInterval::current(1000.into());
        assert!(interval.is_currently_valid());
        assert!(interval.is_currently_recorded());
        assert!(interval.is_current());
    }

    #[test]
    fn test_bitemporal_now() {
        let interval = BiTemporalInterval::now(1000.into(), 2000.into());
        assert_eq!(interval.valid_time().start(), 1000.into());
        assert_eq!(interval.transaction_time().start(), 2000.into());
        assert!(interval.is_currently_valid());
        assert!(interval.is_currently_recorded());
    }

    #[test]
    fn test_bitemporal_visibility() {
        let interval = BiTemporalInterval::new(
            TimeRange::new(1000.into(), 2000.into()).unwrap(), // Valid from 1000 to 2000
            TimeRange::new(3000.into(), 4000.into()).unwrap(), // Recorded from 3000 to 4000
        );

        // Visible if both dimensions are in range
        assert!(interval.is_visible_at(1500.into(), 3500.into()));
        assert!(!interval.is_visible_at(500.into(), 3500.into())); // Before valid time
        assert!(!interval.is_visible_at(1500.into(), 2500.into())); // Before transaction time
        assert!(!interval.is_visible_at(2500.into(), 3500.into())); // After valid time
        assert!(!interval.is_visible_at(1500.into(), 4500.into())); // After transaction time

        // 🛡️ Sentry Test: Verify partial matches to kill mutant that changes && to ||
        // Valid time is in range, but transaction time is not
        assert!(!interval.is_visible_at(1500.into(), 2000.into()));
        // Transaction time is in range, but valid time is not
        assert!(!interval.is_visible_at(500.into(), 3500.into()));
    }

    #[test]
    fn test_bitemporal_close() {
        let interval = BiTemporalInterval::now(1000.into(), 2000.into());

        let closed_valid = interval.close_valid_time(1500.into()).unwrap();
        assert!(!closed_valid.is_currently_valid());
        assert!(closed_valid.is_currently_recorded());
        assert_eq!(closed_valid.valid_time().end(), 1500.into());

        let closed_tx = interval.close_transaction_time(2500.into()).unwrap();
        assert!(closed_tx.is_currently_valid());
        assert!(!closed_tx.is_currently_recorded());
        assert_eq!(closed_tx.transaction_time().end(), 2500.into());

        let closed_both = interval.close_both(1500.into(), 2500.into()).unwrap();
        assert!(!closed_both.is_currently_valid());
        assert!(!closed_both.is_currently_recorded());
        assert_eq!(closed_both.valid_time().end(), 1500.into());
        assert_eq!(closed_both.transaction_time().end(), 2500.into());
    }

    #[test]
    fn test_time_helpers() {
        let secs = 1234567890i64;
        let timestamp = time::from_secs(secs);
        assert_eq!(time::to_secs(timestamp), secs);

        let millis = 1234567890123i64;
        let timestamp = time::from_millis(millis);
        assert_eq!(time::to_millis(timestamp), millis);
    }

    #[test]
    fn test_time_now() {
        let timestamp = time::now();
        // Should be after 2020-01-01 and before 2100-01-01
        assert!(timestamp > time::from_secs(1577836800));
        assert!(timestamp < time::from_secs(4102444800));
    }

    #[test]
    fn test_time_range_invalid_returns_error() {
        // TimeRange::new should return an error for invalid ranges (start > end)
        let result = TimeRange::new(200.into(), 100.into());
        assert!(result.is_err());
        if let Err(crate::core::error::TemporalError::InvalidTimeRange { start, end }) = result {
            assert_eq!(start, 200.into());
            assert_eq!(end, 100.into());
        } else {
            panic!("Expected InvalidTimeRange error");
        }
    }

    #[test]
    fn test_time_range_valid_returns_ok() {
        // TimeRange::new should return Ok for valid ranges
        let result = TimeRange::new(100.into(), 200.into());
        assert!(result.is_ok());
        let range = result.unwrap();
        assert_eq!(range.start(), 100.into());
        assert_eq!(range.end(), 200.into());
    }

    #[test]
    fn test_time_range_equal_start_end_returns_ok() {
        // TimeRange::new should return Ok when start == end (point-in-time)
        let result = TimeRange::new(100.into(), 100.into());
        assert!(result.is_ok());
        let range = result.unwrap();
        assert_eq!(range.start(), 100.into());
        assert_eq!(range.end(), 100.into());
    }

    // Serialization tests

    #[test]
    fn test_timerange_serialize_roundtrip() {
        let ranges = [
            TimeRange::new(100.into(), 200.into()).unwrap(),
            TimeRange::from(1000.into()),
            TimeRange::at(500.into()),
            TimeRange::new(i64::MIN.into(), i64::MAX.into()).unwrap(),
            TimeRange::new(0.into(), 0.into()).unwrap(),
        ];
        for range in ranges {
            let bytes = range.serialize();
            assert_eq!(bytes.len(), 24); // Phase 2: 2 x 12-byte HybridTimestamp
            let (deserialized, consumed) = TimeRange::deserialize(&bytes).unwrap();
            assert_eq!(deserialized, range);
            assert_eq!(consumed, 24);
        }
    }

    #[test]
    fn test_timerange_serialize_into() {
        let range = TimeRange::new(100.into(), 200.into()).unwrap();
        let mut buffer = Vec::new();
        range.serialize_into(&mut buffer);
        assert_eq!(buffer.len(), 24); // Phase 2: 2 x 12-byte HybridTimestamp
        let (deserialized, _) = TimeRange::deserialize(&buffer).unwrap();
        assert_eq!(deserialized, range);
    }

    #[test]
    fn test_timerange_deserialize_truncated() {
        let result = TimeRange::deserialize(&[0; 23]); // Phase 2: 24 bytes needed
        assert!(result.is_err());
    }

    #[test]
    fn test_bitemporal_serialize_roundtrip() {
        let intervals = [
            BiTemporalInterval::current(1000.into()),
            BiTemporalInterval::now(500.into(), 600.into()),
            BiTemporalInterval::new(
                TimeRange::new(100.into(), 200.into()).unwrap(),
                TimeRange::new(300.into(), 400.into()).unwrap(),
            ),
            BiTemporalInterval::new(TimeRange::from(0.into()), TimeRange::from(0.into())),
        ];
        for interval in intervals {
            let bytes = interval.serialize();
            assert_eq!(bytes.len(), 48); // Phase 2: 2 x 24-byte TimeRange
            let (deserialized, consumed) = BiTemporalInterval::deserialize(&bytes).unwrap();
            assert_eq!(deserialized, interval);
            assert_eq!(consumed, 48);
        }
    }

    #[test]
    fn test_bitemporal_serialize_into() {
        let interval = BiTemporalInterval::new(
            TimeRange::new(100.into(), 200.into()).unwrap(),
            TimeRange::new(300.into(), 400.into()).unwrap(),
        );
        let mut buffer = Vec::new();
        interval.serialize_into(&mut buffer);
        assert_eq!(buffer.len(), 48); // Phase 2: 2 x 24-byte TimeRange
        let (deserialized, _) = BiTemporalInterval::deserialize(&buffer).unwrap();
        assert_eq!(deserialized, interval);
    }

    #[test]
    fn test_bitemporal_deserialize_truncated() {
        let result = BiTemporalInterval::deserialize(&[0; 47]); // Phase 2: 48 bytes needed
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_endianness() {
        // Phase 2: Verify HybridTimestamp little-endian format
        // Format: [start_wallclock:8][start_logical:4][end_wallclock:8][end_logical:4]
        let range =
            TimeRange::new(0x0102030405060708i64.into(), 0x1112131415161718i64.into()).unwrap();
        let bytes = range.serialize();

        assert_eq!(bytes.len(), 24); // Total size: 24 bytes

        // Start timestamp wallclock (bytes 0-7) - little-endian
        assert_eq!(bytes[0], 0x08, "start.wallclock LSB");
        assert_eq!(bytes[7], 0x01, "start.wallclock MSB");

        // Start timestamp logical (bytes 8-11) - should be 0
        assert_eq!(&bytes[8..12], &[0, 0, 0, 0], "start.logical should be 0");

        // End timestamp wallclock (bytes 12-19) - little-endian
        assert_eq!(bytes[12], 0x18, "end.wallclock LSB");
        assert_eq!(bytes[19], 0x11, "end.wallclock MSB");

        // End timestamp logical (bytes 20-23) - should be 0
        assert_eq!(&bytes[20..24], &[0, 0, 0, 0], "end.logical should be 0");
    }

    // =========================================================================
    // Phase 2: HybridTimestamp Integration Tests
    // =========================================================================

    #[test]
    fn test_timestamp_is_hybrid_timestamp() {
        // After Phase 2, Timestamp should be HybridTimestamp
        // time::now() should return HybridTimestamp
        let ts = time::now();

        // Should have wallclock component
        assert!(ts.wallclock() > 0);

        // Should have logical component (starts at 0)
        assert_eq!(ts.logical(), 0);
    }

    #[test]
    fn test_time_range_with_hybrid_timestamps() {
        use crate::core::hlc::HybridTimestamp;

        // Create timestamps with different wallclocks
        let ts1 = HybridTimestamp::new(1000, 0).unwrap();
        let ts2 = HybridTimestamp::new(2000, 0).unwrap();

        // TimeRange should work with HybridTimestamp
        let range = TimeRange::new(ts1, ts2).unwrap();

        assert_eq!(range.start(), ts1);
        assert_eq!(range.end(), ts2);
    }

    #[test]
    fn test_time_range_ordering_with_logical_component() {
        use crate::core::hlc::HybridTimestamp;

        // Same wallclock, different logical counters
        let ts1 = HybridTimestamp::new(1000, 0).unwrap();
        let ts2 = HybridTimestamp::new(1000, 1).unwrap();
        let ts3 = HybridTimestamp::new(1000, 2).unwrap();

        // HybridTimestamp ordering is lexicographic: wallclock first, then logical
        assert!(ts1 < ts2);
        assert!(ts2 < ts3);

        // TimeRange should respect this ordering
        let range = TimeRange::new(ts1, ts3).unwrap();
        assert!(range.contains(ts1));
        assert!(range.contains(ts2));
        assert!(!range.contains(ts3)); // Exclusive end
    }

    #[test]
    fn test_bitemporal_with_hybrid_timestamps() {
        use crate::core::hlc::HybridTimestamp;

        let valid_start = HybridTimestamp::new(1000, 0).unwrap();
        let tx_start = HybridTimestamp::new(2000, 0).unwrap();

        let interval = BiTemporalInterval::now(valid_start, tx_start);

        assert_eq!(interval.valid_time().start(), valid_start);
        assert_eq!(interval.transaction_time().start(), tx_start);
    }

    #[test]
    fn test_hybrid_timestamp_deserialize_errors() {
        use crate::core::hlc::HybridTimestamp;
        // 1. Buffer too short
        let short_buffer = [0u8; 11];
        let result = HybridTimestamp::deserialize(&short_buffer);
        assert!(
            matches!(result, Err(crate::core::error::StorageError::CorruptedData(ref msg)) if msg.contains("Buffer too short")),
            "Expected CorruptedData for short buffer"
        );

        // 2. Wallclock > MAX_VALID_TIMESTAMP
        let mut buffer = [0u8; 12];
        let invalid_wallclock: i64 = MAX_VALID_TIMESTAMP + 1;
        buffer[0..8].copy_from_slice(&invalid_wallclock.to_le_bytes());
        let result = HybridTimestamp::deserialize(&buffer);
        assert!(
            matches!(result, Err(crate::core::error::StorageError::CorruptedData(ref msg)) if msg.contains("exceeds MAX_VALID_TIMESTAMP")),
            "Expected CorruptedData for exceeding MAX_VALID_TIMESTAMP"
        );

        // 3. Wallclock == i64::MAX but logical != 0
        let mut buffer2 = [0u8; 12];
        buffer2[0..8].copy_from_slice(&i64::MAX.to_le_bytes());
        buffer2[8..12].copy_from_slice(&1u32.to_le_bytes()); // logical = 1
        let result2 = HybridTimestamp::deserialize(&buffer2);
        assert!(
            matches!(result2, Err(crate::core::error::StorageError::CorruptedData(ref msg)) if msg.contains("non-zero logical counter")),
            "Expected CorruptedData for i64::MAX with non-zero logical"
        );
    }

    #[test]
    fn test_hybrid_timestamp_new_validation() {
        use crate::core::hlc::HybridTimestamp;
        let invalid_wallclock: i64 = MAX_VALID_TIMESTAMP + 1;
        let result = HybridTimestamp::new(invalid_wallclock, 0);
        assert!(
            matches!(result, Err(TemporalError::InvalidTimestamp { .. })),
            "Expected InvalidTimestamp when wallclock exceeds MAX_VALID_TIMESTAMP"
        );
    }

    #[test]
    fn test_hybrid_timestamp_serialization_size() {
        use crate::core::hlc::HybridTimestamp;

        // HybridTimestamp is 12 bytes (8 wallclock + 4 logical)
        let ts = HybridTimestamp::new(1000, 5).unwrap();
        let serialized = ts.serialize();
        assert_eq!(serialized.len(), 12);

        // TimeRange with HybridTimestamp should be 24 bytes (2 × 12)
        let range = TimeRange::new(
            HybridTimestamp::new(1000, 0).unwrap(),
            HybridTimestamp::new(2000, 0).unwrap(),
        )
        .unwrap();
        let serialized = range.serialize();
        assert_eq!(serialized.len(), 24);

        // BiTemporalInterval should be 48 bytes (4 × 12)
        let interval = BiTemporalInterval::now(
            HybridTimestamp::new(1000, 0).unwrap(),
            HybridTimestamp::new(2000, 0).unwrap(),
        );
        let serialized = interval.serialize();
        assert_eq!(serialized.len(), 48);
    }

    #[test]
    fn test_timestamp_max_with_hybrid_timestamp() {
        // TIMESTAMP_MAX should be representable as HybridTimestamp
        // It represents "infinity" or "current"
        assert_eq!(TIMESTAMP_MAX.wallclock(), i64::MAX);
        assert_eq!(TIMESTAMP_MAX.logical(), 0);
    }

    #[test]
    fn test_contains_or_after_behavior() {
        use crate::core::hlc::HybridTimestamp;
        let start = HybridTimestamp::new(100, 0).unwrap();
        let end = HybridTimestamp::new(200, 0).unwrap();
        let range = TimeRange::new(start, end).unwrap();

        // Verify contains_or_after works as expected
        assert!(range.contains_or_after(start));
        assert!(range.contains_or_after(HybridTimestamp::new(150, 0).unwrap()));
        assert!(range.contains_or_after(end)); // Should be true (after start)
        assert!(range.contains_or_after(HybridTimestamp::new(300, 0).unwrap())); // Should be true
        assert!(!range.contains_or_after(HybridTimestamp::new(99, 0).unwrap())); // Should be false
    }

    #[test]
    fn test_close_at_start_time() {
        use crate::core::hlc::HybridTimestamp;
        let start = HybridTimestamp::new(100, 0).unwrap();
        let range = TimeRange::from(start);

        // Should be able to close at start time (creating empty range)
        let closed = range.close_at(start).unwrap();
        assert_eq!(closed.start(), start);
        assert_eq!(closed.end(), start);
        assert!(closed.is_closed());
    }

    #[test]
    fn test_max_valid_timestamp_boundary() {
        use crate::core::hlc::HybridTimestamp;
        let max_valid = HybridTimestamp::new(MAX_VALID_TIMESTAMP, 0).unwrap();

        // Should be able to create range with MAX_VALID_TIMESTAMP as start and end
        let range = TimeRange::new(max_valid, max_valid).unwrap();
        assert_eq!(range.start(), max_valid);

        let range2 = TimeRange::from(max_valid);
        assert_eq!(range2.start(), max_valid);

        // Should be able to create range ending at MAX_VALID_TIMESTAMP
        let almost_max = HybridTimestamp::new(MAX_VALID_TIMESTAMP - 100, 0).unwrap();
        let range3 = TimeRange::new(almost_max, max_valid).unwrap();
        assert_eq!(range3.end(), max_valid);
    }

    // =========================================================================
    // Phase 1: True Bi-Temporal Support Tests (with_valid_time)
    // =========================================================================

    #[test]
    fn test_with_valid_time_creates_separate_dimensions() {
        use crate::core::hlc::HybridTimestamp;

        let valid_from = HybridTimestamp::new(1000, 0).unwrap();
        let tx_time = HybridTimestamp::new(2000, 0).unwrap();

        let interval = BiTemporalInterval::with_valid_time(valid_from, tx_time);

        assert_eq!(interval.valid_time().start(), valid_from);
        assert_eq!(interval.transaction_time().start(), tx_time);
        assert_ne!(
            interval.valid_time().start(),
            interval.transaction_time().start()
        );
    }

    #[test]
    fn test_with_valid_time_creates_open_ended_ranges() {
        use crate::core::hlc::HybridTimestamp;

        let valid_from = HybridTimestamp::new(1000, 0).unwrap();
        let tx_time = HybridTimestamp::new(2000, 0).unwrap();

        let interval = BiTemporalInterval::with_valid_time(valid_from, tx_time);

        // Both should be open-ended (end = TIMESTAMP_MAX)
        assert!(interval.valid_time().is_current());
        assert!(interval.transaction_time().is_current());
    }

    #[test]
    fn test_with_valid_time_backdated_visibility() {
        use crate::core::hlc::HybridTimestamp;

        // Create interval: valid from Jan 1, recorded on Feb 1
        let jan_1 = HybridTimestamp::new(1_704_067_200_000_000, 0).unwrap(); // 2024-01-01
        let feb_1 = HybridTimestamp::new(1_706_745_600_000_000, 0).unwrap(); // 2024-02-01
        let jan_15 = HybridTimestamp::new(1_705_276_800_000_000, 0).unwrap(); // 2024-01-15

        let interval = BiTemporalInterval::with_valid_time(jan_1, feb_1);

        // Visible at valid_time=Jan 15, tx_time=Feb 1 (after recording)
        assert!(interval.is_visible_at(jan_15, feb_1));

        // NOT visible at valid_time=Jan 15, tx_time=Jan 15 (before recording)
        assert!(!interval.is_visible_at(jan_15, jan_15));
    }

    #[test]
    fn test_contains_range_excludes_partial_overlap_start() {
        // Outer: [100, 200)
        let start = HybridTimestamp::new(100, 0).unwrap();
        let end = HybridTimestamp::new(200, 0).unwrap();
        let outer = TimeRange::new(start, end).unwrap();

        // Inner: [50, 150) - Starts before outer
        // This targets the potential missing `self.start <= other.start` check
        let inner_start = HybridTimestamp::new(50, 0).unwrap();
        let inner_end = HybridTimestamp::new(150, 0).unwrap();
        let inner = TimeRange::new(inner_start, inner_end).unwrap();

        // Should return false because inner.start < outer.start
        assert!(
            !outer.contains_range(&inner),
            "Range starting before outer should not be contained"
        );
    }

    #[test]
    fn test_time_range_rejects_timestamps_exceeding_max_valid() {
        use crate::core::hlc::HybridTimestamp;

        // Test with MAX_VALID_TIMESTAMP + 1
        let invalid_ts = HybridTimestamp::new_unchecked(MAX_VALID_TIMESTAMP + 1, 0);
        let valid_ts = HybridTimestamp::new(100, 0).unwrap();

        // Test start timestamp validation
        // Use invalid_ts for both start and end to avoid InvalidTimeRange (start > end)
        let result = TimeRange::new(invalid_ts, invalid_ts);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TemporalError::InvalidTimestamp { .. }
        ));

        // Test end timestamp validation
        let result = TimeRange::new(valid_ts, invalid_ts);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TemporalError::InvalidTimestamp { .. }
        ));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating valid wallclock values within safe bounds.
    /// We use a range well within MAX_VALID_TIMESTAMP to avoid overflow in
    /// arithmetic operations and to stay away from sentinel values.
    fn valid_wallclock() -> impl Strategy<Value = i64> {
        0..=(MAX_VALID_TIMESTAMP / 2)
    }

    /// Strategy for generating HybridTimestamps with valid values.
    fn valid_timestamp() -> impl Strategy<Value = Timestamp> {
        (valid_wallclock(), any::<u32>())
            .prop_map(|(wc, lc)| HybridTimestamp::new_unchecked(wc, lc))
    }

    /// Strategy for generating three ordered wallclock values (a <= b <= c).
    fn three_ordered_wallclocks() -> impl Strategy<Value = (i64, i64, i64)> {
        (valid_wallclock(), valid_wallclock(), valid_wallclock()).prop_map(|(a, b, c)| {
            let mut vals = [a, b, c];
            vals.sort();
            (vals[0], vals[1], vals[2])
        })
    }

    proptest! {
        // ====================================================================
        // Timestamp Ordering Properties
        // ====================================================================

        /// Property: Timestamp ordering is transitive.
        /// If a < b and b < c then a < c.
        #[test]
        fn prop_timestamp_ordering_is_transitive(
            (wc_a, wc_b, wc_c) in three_ordered_wallclocks()
        ) {
            let a = HybridTimestamp::new_unchecked(wc_a, 0);
            let b = HybridTimestamp::new_unchecked(wc_b, 0);
            let c = HybridTimestamp::new_unchecked(wc_c, 0);

            if a < b && b < c {
                prop_assert!(a < c, "Transitivity violated: {:?} < {:?} < {:?} but a >= c", a, b, c);
            }
        }

        /// Property: Timestamp ordering is transitive with logical counters.
        #[test]
        fn prop_timestamp_ordering_transitive_with_logical(
            wc in valid_wallclock(),
            lc_a in 0u32..1000,
            lc_b in 0u32..1000,
            lc_c in 0u32..1000,
        ) {
            let a = HybridTimestamp::new_unchecked(wc, lc_a);
            let b = HybridTimestamp::new_unchecked(wc, lc_b);
            let c = HybridTimestamp::new_unchecked(wc, lc_c);

            if a < b && b < c {
                prop_assert!(a < c, "Logical counter transitivity violated");
            }
        }

        /// Property: Timestamp ordering is antisymmetric.
        /// If a <= b and b <= a then a == b.
        #[test]
        fn prop_timestamp_ordering_antisymmetric(
            a in valid_timestamp(),
            b in valid_timestamp(),
        ) {
            if a <= b && b <= a {
                prop_assert_eq!(a, b);
            }
        }

        /// Property: Timestamp ordering is total.
        /// For any a, b: exactly one of a < b, a == b, a > b holds.
        #[test]
        fn prop_timestamp_ordering_total(
            a in valid_timestamp(),
            b in valid_timestamp(),
        ) {
            let lt = a < b;
            let eq = a == b;
            let gt = a > b;
            let exactly_one = (lt as u8 + eq as u8 + gt as u8) == 1;
            prop_assert!(exactly_one, "Total ordering violated: lt={}, eq={}, gt={}", lt, eq, gt);
        }

        // ====================================================================
        // Timestamp Monotonicity Properties
        // ====================================================================

        /// Property: time::now() produces timestamps after a known past time.
        #[test]
        fn prop_time_now_is_after_epoch(_dummy in 0..10u8) {
            let ts = time::now();
            let epoch = HybridTimestamp::new_unchecked(0, 0);
            prop_assert!(ts > epoch, "time::now() should be after epoch");
        }

        /// Property: Sequential calls to time::now() produce non-decreasing timestamps.
        #[test]
        fn prop_time_now_monotonic(_dummy in 0..10u8) {
            let t1 = time::now();
            let t2 = time::now();
            prop_assert!(t2 >= t1, "Sequential now() calls should be non-decreasing: {:?} vs {:?}", t1, t2);
        }

        // ====================================================================
        // TimeRange Properties
        // ====================================================================

        /// Property: Valid time ranges always have start <= end.
        #[test]
        fn prop_time_range_well_formed(
            start_wc in valid_wallclock(),
            end_wc in valid_wallclock(),
        ) {
            let start = HybridTimestamp::new_unchecked(start_wc, 0);
            let end = HybridTimestamp::new_unchecked(end_wc, 0);
            let result = TimeRange::new(start, end);

            if start <= end {
                let range = result.expect("Valid range should succeed");
                prop_assert!(range.start() <= range.end(),
                    "TimeRange should have start <= end");
            } else {
                prop_assert!(result.is_err(),
                    "TimeRange::new should reject start > end");
            }
        }

        /// Property: TimeRange::from() always creates a current (open-ended) range.
        #[test]
        fn prop_time_range_from_is_current(ts in valid_timestamp()) {
            let range = TimeRange::from(ts);
            prop_assert!(range.is_current(), "TimeRange::from should be current");
            prop_assert_eq!(range.start(), ts);
            prop_assert_eq!(range.end(), TIMESTAMP_MAX);
        }

        /// Property: TimeRange::at() creates a point-in-time range where start == end.
        #[test]
        fn prop_time_range_at_is_point(ts in valid_timestamp()) {
            let range = TimeRange::at(ts);
            prop_assert_eq!(range.start(), range.end());
            prop_assert_eq!(range.start(), ts);
        }

        /// Property: TimeRange contains its start but not its end (half-open interval).
        #[test]
        fn prop_time_range_half_open(
            (wc_start, wc_end) in (0i64..1_000_000, 1i64..1_000_000)
                .prop_map(|(s, delta)| (s, s + delta))
        ) {
            let start = HybridTimestamp::new_unchecked(wc_start, 0);
            let end = HybridTimestamp::new_unchecked(wc_end, 0);
            let range = TimeRange::new(start, end).unwrap();

            prop_assert!(range.contains(start), "Range should contain its start");
            prop_assert!(!range.contains(end), "Range should not contain its end (exclusive)");
        }

        /// Property: TimeRange::contains is consistent with the [start, end) semantics.
        #[test]
        fn prop_time_range_contains_semantics(
            wc_start in 0i64..500_000,
            wc_end in 500_001i64..1_000_000,
            wc_point in 0i64..1_000_000,
        ) {
            let start = HybridTimestamp::new_unchecked(wc_start, 0);
            let end = HybridTimestamp::new_unchecked(wc_end, 0);
            let point = HybridTimestamp::new_unchecked(wc_point, 0);
            let range = TimeRange::new(start, end).unwrap();

            let expected = point >= start && point < end;
            prop_assert_eq!(range.contains(point), expected,
                "contains({:?}) mismatch for range [{:?}, {:?})", point, start, end);
        }

        /// Property: TimeRange overlaps is symmetric.
        #[test]
        fn prop_time_range_overlaps_symmetric(
            wc_a in 0i64..500_000,
            len_a in 1i64..500_000,
            wc_b in 0i64..500_000,
            len_b in 1i64..500_000,
        ) {
            let start_a = HybridTimestamp::new_unchecked(wc_a, 0);
            let end_a = HybridTimestamp::new_unchecked(wc_a + len_a, 0);
            let start_b = HybridTimestamp::new_unchecked(wc_b, 0);
            let end_b = HybridTimestamp::new_unchecked(wc_b + len_b, 0);

            let range_a = TimeRange::new(start_a, end_a).unwrap();
            let range_b = TimeRange::new(start_b, end_b).unwrap();

            prop_assert_eq!(range_a.overlaps(&range_b), range_b.overlaps(&range_a),
                "overlaps should be symmetric");
        }

        /// Property: A range always contains itself (reflexive containment).
        #[test]
        fn prop_time_range_contains_self(
            wc_start in 0i64..500_000,
            wc_end in 500_001i64..1_000_000,
        ) {
            let start = HybridTimestamp::new_unchecked(wc_start, 0);
            let end = HybridTimestamp::new_unchecked(wc_end, 0);
            let range = TimeRange::new(start, end).unwrap();

            prop_assert!(range.contains_range(&range),
                "A range should always contain itself");
        }

        /// Property: TimeRange serialization roundtrip preserves values.
        #[test]
        fn prop_time_range_serialization_roundtrip(
            wc_start in valid_wallclock(),
            wc_end in valid_wallclock(),
        ) {
            let (s, e) = if wc_start <= wc_end { (wc_start, wc_end) } else { (wc_end, wc_start) };
            let start = HybridTimestamp::new_unchecked(s, 0);
            let end = HybridTimestamp::new_unchecked(e, 0);
            let range = TimeRange::new(start, end).unwrap();

            let bytes = range.serialize();
            prop_assert_eq!(bytes.len(), 24);
            let (deserialized, consumed) = TimeRange::deserialize(&bytes).unwrap();
            prop_assert_eq!(consumed, 24);
            prop_assert_eq!(deserialized, range);
        }

        // ====================================================================
        // BiTemporalInterval Properties
        // ====================================================================

        /// Property: BiTemporalInterval::current() is always current in both dimensions.
        #[test]
        fn prop_bitemporal_current_is_current(ts in valid_timestamp()) {
            let interval = BiTemporalInterval::current(ts);
            prop_assert!(interval.is_currently_valid());
            prop_assert!(interval.is_currently_recorded());
            prop_assert!(interval.is_current());
        }

        /// Property: BiTemporalInterval serialization roundtrip preserves values.
        #[test]
        fn prop_bitemporal_serialization_roundtrip(
            wc_vs in valid_wallclock(),
            wc_ve in valid_wallclock(),
            wc_ts in valid_wallclock(),
            wc_te in valid_wallclock(),
        ) {
            let (vs, ve) = if wc_vs <= wc_ve { (wc_vs, wc_ve) } else { (wc_ve, wc_vs) };
            let (ts, te) = if wc_ts <= wc_te { (wc_ts, wc_te) } else { (wc_te, wc_ts) };

            let vt = TimeRange::new(
                HybridTimestamp::new_unchecked(vs, 0),
                HybridTimestamp::new_unchecked(ve, 0),
            ).unwrap();
            let tt = TimeRange::new(
                HybridTimestamp::new_unchecked(ts, 0),
                HybridTimestamp::new_unchecked(te, 0),
            ).unwrap();

            let interval = BiTemporalInterval::new(vt, tt);
            let bytes = interval.serialize();
            prop_assert_eq!(bytes.len(), 48);
            let (deserialized, consumed) = BiTemporalInterval::deserialize(&bytes).unwrap();
            prop_assert_eq!(consumed, 48);
            prop_assert_eq!(deserialized, interval);
        }

        /// Property: Closing valid time makes the interval no longer currently valid.
        #[test]
        fn prop_closing_valid_time_makes_not_current(
            wc_start in 0i64..500_000,
            wc_close in 500_001i64..1_000_000,
        ) {
            let start = HybridTimestamp::new_unchecked(wc_start, 0);
            let close = HybridTimestamp::new_unchecked(wc_close, 0);

            let interval = BiTemporalInterval::current(start);
            prop_assert!(interval.is_currently_valid());

            let closed = interval.close_valid_time(close).unwrap();
            prop_assert!(!closed.is_currently_valid());
            prop_assert_eq!(closed.valid_time().end(), close);
        }

        /// Property: Closing transaction time makes the interval no longer currently recorded.
        #[test]
        fn prop_closing_tx_time_makes_not_recorded(
            wc_start in 0i64..500_000,
            wc_close in 500_001i64..1_000_000,
        ) {
            let start = HybridTimestamp::new_unchecked(wc_start, 0);
            let close = HybridTimestamp::new_unchecked(wc_close, 0);

            let interval = BiTemporalInterval::current(start);
            prop_assert!(interval.is_currently_recorded());

            let closed = interval.close_transaction_time(close).unwrap();
            prop_assert!(!closed.is_currently_recorded());
            prop_assert_eq!(closed.transaction_time().end(), close);
        }

        /// Property: TimeRange duration is non-negative for valid ranges.
        #[test]
        fn prop_time_range_duration_non_negative(
            wc_start in valid_wallclock(),
            wc_end in valid_wallclock(),
        ) {
            let (s, e) = if wc_start <= wc_end { (wc_start, wc_end) } else { (wc_end, wc_start) };
            let start = HybridTimestamp::new_unchecked(s, 0);
            let end = HybridTimestamp::new_unchecked(e, 0);
            let range = TimeRange::new(start, end).unwrap();

            if let Some(duration) = range.duration_micros() {
                prop_assert!(duration >= 0,
                    "Duration should be non-negative, got {}", duration);
            }
        }

        /// Property: time::from_secs roundtrips through to_secs.
        #[test]
        fn prop_time_secs_roundtrip(secs in 0i64..1_000_000_000) {
            let ts = time::from_secs(secs);
            let result = time::to_secs(ts);
            prop_assert_eq!(result, secs);
        }

        /// Property: time::from_millis roundtrips through to_millis.
        #[test]
        fn prop_time_millis_roundtrip(millis in 0i64..1_000_000_000_000) {
            let ts = time::from_millis(millis);
            let result = time::to_millis(ts);
            prop_assert_eq!(result, millis);
        }
    }
}

#[cfg(test)]
mod sentry_tests {
    use super::*;

    #[test]
    fn test_sentry_bitemporal_is_current_mixed_state() {
        // 🛡️ Sentry Test: Verify BiTemporalInterval::is_current() correctly handles mixed states.
        // This test ensures that if only one dimension is open (current), is_current() returns false.
        // It specifically targets mutants that might replace `&&` with `||` in the implementation.

        let valid_start = 1000.into();
        let valid_end = 2000.into();
        let tx_start = 3000.into();

        let interval = BiTemporalInterval::new(
            TimeRange::new(valid_start, valid_end).unwrap(), // Closed (not current)
            TimeRange::from(tx_start),                       // Open (current)
        );

        assert!(!interval.is_currently_valid());
        assert!(interval.is_currently_recorded());

        // Assert that is_current() is false. If implementation used OR, this would be true.
        assert!(
            !interval.is_current(),
            "is_current() should be false if one dimension is closed"
        );
    }

    #[test]
    fn test_sentry_iso8601_format_content() {
        // 🛡️ Sentry Test: Verify time::to_iso8601 produces expected content.
        // This targets arithmetic mutants (e.g., replacing / with %) that would produce
        // wildly incorrect second values in the output string.

        let secs = 1609459200; // 2021-01-01 00:00:00 UTC
        let ts = time::from_secs(secs);
        let output = time::to_iso8601(ts);

        if cfg!(windows) {
            // On Windows, SystemTime debug format is "SystemTime { intervals: <count> }"
            // intervals are 100ns ticks since 1601-01-01
            // 1609459200 seconds (Unix epoch to 2021) + 11644473600 seconds (1601 to 1970)
            // = 13253932800 seconds total
            // * 10,000,000 (ticks per second) = 132539328000000000
            let expected = "132539328000000000";
            assert!(
                output.contains(expected),
                "to_iso8601 output should contain expected intervals on Windows. Got: {}",
                output
            );
        } else {
            // On Unix-like systems, Debug format usually contains "tv_sec: <seconds>"
            assert!(
                output.contains(&secs.to_string()),
                "to_iso8601 output should contain the seconds timestamp. Got: {}",
                output
            );
        }
    }

    #[test]
    fn test_sentry_time_range_display_format() {
        // 🛡️ Sentry Test: Verify Display implementation for TimeRange.
        // Ensures proper bracketing [ ) for ranges.

        let start = 100.into();
        let end = 200.into();

        // Closed range
        let closed = TimeRange::new(start, end).unwrap();
        let closed_str = format!("{}", closed);
        assert!(closed_str.starts_with("["));
        assert!(closed_str.ends_with(")"));
        assert!(closed_str.contains(", "));

        // Open range
        let open = TimeRange::from(start);
        let open_str = format!("{}", open);
        assert!(open_str.starts_with("["));
        assert!(open_str.ends_with(")"));
        assert!(open_str.contains("current"));
    }

    #[test]
    fn test_sentry_overlaps_strict_inequality() {
        // 🛡️ Sentry Test: Verify strict inequality in overlaps().
        // Explicitly check the touching case from the other direction.

        let r1 = TimeRange::new(100.into(), 200.into()).unwrap();
        let r2 = TimeRange::new(200.into(), 300.into()).unwrap();

        // r2 overlaps r1? 200 < 200 is False.
        // If logic was <=, it would be True.
        assert!(
            !r2.overlaps(&r1),
            "Touching ranges should not overlap (checking symmetry)"
        );
    }

    #[test]
    fn test_sentry_contains_range_strict_inequality() {
        // 🛡️ Sentry Test: Verify strict inequality in contains_range().
        // Specifically check exact end boundary match.

        let outer = TimeRange::new(100.into(), 300.into()).unwrap();
        let inner = TimeRange::new(200.into(), 300.into()).unwrap();

        // outer.end (300) == inner.end (300)
        // Logic requires inner.end <= outer.end.
        // If logic was <, this would fail.
        assert!(
            outer.contains_range(&inner),
            "Should contain range ending at exact same time"
        );
    }

    #[test]
    fn test_sentry_iso8601_precision() {
        // 🛡️ Sentry Test: Verify sub-second precision in ISO 8601 output.
        // This targets mutants that break nanosecond calculation.

        let secs = 1609459200;
        let micros = 123456;
        let ts = HybridTimestamp::new_unchecked(secs * 1_000_000 + micros, 0);
        let output = time::to_iso8601(ts);

        // Expected nanoseconds: 123456000
        if cfg!(windows) {
            // On Windows, SystemTime debug format is "SystemTime { intervals: <count> }"
            // intervals are 100ns ticks since 1601-01-01.
            // Base seconds (1601 to 1970) = 11644473600.
            // Target seconds (1970 to 2021) = 1609459200.
            // Total seconds = 13253932800.
            // Total ticks from seconds = 13253932800 * 10_000_000 = 132539328000000000.
            // Ticks from microseconds = 123456 * 10 = 1234560.
            // Total ticks = 132539328001234560.
            // We check for the fractional part contribution or the exact tick count.
            // Since the output observed is "intervals: 132539328001234560", we check for that suffix.
            assert!(
                output.contains("1234560"),
                "Output should contain the fractional ticks (1234560): {}",
                output
            );
        } else {
            // On Unix-like systems, Debug format usually contains "tv_nsec: 123456000".
            assert!(
                output.contains("123456000"),
                "Output should contain nanoseconds (123456000): {}",
                output
            );
        }
    }

    #[test]
    fn test_sentry_max_valid_timestamp_value() {
        // 🛡️ Sentry Test: Verify MAX_VALID_TIMESTAMP allows reasonably large values.
        // This targets mutants that drastically reduce MAX_VALID_TIMESTAMP (e.g. replace - with /).

        // i64::MAX is ~9e18. i64::MAX / 1000 is ~9e15.
        // We want to ensure we can store something larger than 9e15.
        // 9e15 micros is ~285 years.
        // Wait, i64::MAX micros is ~292,000 years.
        // If we mutate to i64::MAX / 1000, we limit to ~292 years.
        // Current time is ~1.7e15 micros (2024).
        // So i64::MAX / 1000 (9e15) is still future (year ~2250).
        // But we want to support timestamps far in the future (e.g. year 3000).

        // Let's just pick a value close to i64::MAX, e.g. i64::MAX - 2000.
        // If MAX_VALID_TIMESTAMP is i64::MAX / 1000, this will fail.

        let large_val = i64::MAX - 2000;
        let ts = HybridTimestamp::new_unchecked(large_val, 0);

        // This should be accepted by TimeRange::new if MAX_VALID_TIMESTAMP is correct.
        // But wait, TimeRange::new checks against MAX_VALID_TIMESTAMP.
        // If MAX_VALID_TIMESTAMP is correct (i64::MAX - 1000), then i64::MAX - 2000 < MAX_VALID_TIMESTAMP. OK.
        // If MAX_VALID_TIMESTAMP is mutated to (i64::MAX / 1000), then i64::MAX - 2000 > MAX_VALID_TIMESTAMP. Error.

        let result = TimeRange::new(ts, ts);
        assert!(
            result.is_ok(),
            "Should accept large timestamp close to i64::MAX"
        );
    }

    #[test]
    fn test_sentry_contains_range_boundary_conditions() {
        // 🛡️ Sentry Test: Verify contains_range handles identical start/end points.
        // This targets mutants that replace <= with < in the implementation.

        // Outer: [100, 200)
        let outer = TimeRange::new(100.into(), 200.into()).unwrap();

        // Same start: [100, 150)
        let same_start = TimeRange::new(100.into(), 150.into()).unwrap();
        assert!(
            outer.contains_range(&same_start),
            "Should contain range with same start timestamp"
        );

        // Same end: [150, 200)
        let same_end = TimeRange::new(150.into(), 200.into()).unwrap();
        assert!(
            outer.contains_range(&same_end),
            "Should contain range with same end timestamp"
        );

        // Same range: [100, 200)
        let same_range = TimeRange::new(100.into(), 200.into()).unwrap();
        assert!(
            outer.contains_range(&same_range),
            "Should contain itself (reflexive)"
        );
    }

    #[test]
    fn test_sentry_max_valid_timestamp_constant() {
        // 🛡️ Sentry Test: Verify MAX_VALID_TIMESTAMP is exactly i64::MAX - 1000.
        // This pins the reserved range size to prevent accidental drift.
        assert_eq!(
            MAX_VALID_TIMESTAMP,
            i64::MAX - 1000,
            "MAX_VALID_TIMESTAMP must be exactly i64::MAX - 1000 to preserve sentinel space"
        );
    }

    #[test]
    fn test_sentry_point_range_is_empty() {
        // 🛡️ Sentry Test: Verify point range behavior.
        // A point range [T, T) is empty and contains nothing.
        let point = TimeRange::at(100.into());
        assert!(point.is_empty());
        assert!(!point.contains(100.into()));
        assert_eq!(point.duration_micros(), Some(0));
    }

    #[test]
    fn test_sentry_point_range_no_overlap() {
        // 🛡️ Sentry Test: Verify empty range overlap logic.
        // An empty range cannot overlap any other range (intersection is empty).
        // Before Elenchus fix, this would return true.
        let point = TimeRange::at(100.into());
        let wide = TimeRange::new(0.into(), 200.into()).unwrap();

        assert!(
            !point.overlaps(&wide),
            "Empty range should not overlap anything"
        );
        assert!(
            !wide.overlaps(&point),
            "Range should not overlap empty range"
        );
        assert!(
            !point.overlaps(&point),
            "Empty range should not overlap itself"
        );
    }

    #[test]
    fn test_sentry_duration_micros_overflow() {
        // 🛡️ Sentry Test: Verify duration_micros saturates instead of panicking on overflow.
        // This targets mutants that replace checked_sub with sub.

        let start = HybridTimestamp::new_unchecked(i64::MIN, 0);
        let end = HybridTimestamp::new_unchecked(MAX_VALID_TIMESTAMP, 0);

        let range =
            TimeRange::new(start, end).expect("Should create valid range with extreme start");

        assert_eq!(
            range.duration_micros(),
            Some(i64::MAX),
            "Duration should saturate at i64::MAX on overflow"
        );
    }

    #[test]
    fn test_sentry_range_is_not_empty_for_valid_interval() {
        // 🛡️ Sentry Test: Verify is_empty() returns false for non-empty ranges.
        // This targets mutants that make is_empty() always return true.

        let range = TimeRange::new(100.into(), 200.into()).unwrap();
        assert!(!range.is_empty(), "Range [100, 200) should not be empty");
    }

    #[test]
    fn test_sentry_timerange_at_timestamp_max() {
        // 🛡️ Sentry Test: Verify TimeRange::at() accepts TIMESTAMP_MAX.
        // This targets the mutant `if ... timestamp != TIMESTAMP_MAX` -> `if ... timestamp == TIMESTAMP_MAX`.
        // If the check logic is inverted, this call will panic because TIMESTAMP_MAX > MAX_VALID_TIMESTAMP.
        let range = TimeRange::at(TIMESTAMP_MAX);
        assert_eq!(range.start(), TIMESTAMP_MAX);
        assert_eq!(range.end(), TIMESTAMP_MAX);
        assert!(range.is_empty());
    }

    #[test]
    fn test_sentry_timerange_new_timestamp_max() {
        // 🛡️ Sentry Test: Verify TimeRange::new() accepts TIMESTAMP_MAX for start/end.
        // This targets the mutants in TimeRange::new() that invert the sentinel check logic.
        let range = TimeRange::new(TIMESTAMP_MAX, TIMESTAMP_MAX).unwrap();
        assert_eq!(range.start(), TIMESTAMP_MAX);
        assert_eq!(range.end(), TIMESTAMP_MAX);
    }

    #[test]
    fn test_sentry_timerange_close_at_timestamp_max() {
        // 🛡️ Sentry Test: Verify TimeRange::close_at() accepts TIMESTAMP_MAX.
        // This targets the mutant in TimeRange::close_at() that inverts the sentinel check logic.
        let range = TimeRange::from(100.into());
        let closed = range.close_at(TIMESTAMP_MAX).unwrap();
        assert_eq!(closed.end(), TIMESTAMP_MAX);
        assert!(closed.is_current());
    }

    #[test]
    fn test_sentry_deserialize_rejects_inverted_range() {
        // 🛡️ Sentry Test: Verify TimeRange::deserialize rejects inverted ranges (start > end).
        // This targets mutants that remove or weaken the validation check in deserialize.
        // HybridTimestamp is 12 bytes: 8 bytes wallclock (i64 le), 4 bytes logical (u32 le).

        let mut bytes = Vec::new();

        // Start: 200
        bytes.extend_from_slice(&200i64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        // End: 100
        bytes.extend_from_slice(&100i64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        let result = TimeRange::deserialize(&bytes);

        assert!(result.is_err(), "Should reject range where start > end");

        // Check error message
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("Deserialized TimeRange invalid"));
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_VALID_TIMESTAMP")]
    fn test_sentry_timerange_from_exceeds_max_valid() {
        // 🛡️ Sentry Test: Verify TimeRange::from panics if start > MAX_VALID_TIMESTAMP
        let ts = HybridTimestamp::new_unchecked(MAX_VALID_TIMESTAMP + 1, 0);
        let _ = TimeRange::from(ts);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_VALID_TIMESTAMP")]
    fn test_sentry_timerange_at_exceeds_max_valid() {
        // 🛡️ Sentry Test: Verify TimeRange::at panics if timestamp > MAX_VALID_TIMESTAMP
        let ts = HybridTimestamp::new_unchecked(MAX_VALID_TIMESTAMP + 1, 0);
        let _ = TimeRange::at(ts);
    }

    #[test]
    fn test_sentry_timerange_close_at_exceeds_max_valid() {
        // 🛡️ Sentry Test: Verify TimeRange::close_at returns error if end > MAX_VALID_TIMESTAMP
        let open = TimeRange::from(100.into());
        let ts = HybridTimestamp::new_unchecked(MAX_VALID_TIMESTAMP + 1, 0);
        let result = open.close_at(ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_sentry_overlaps_empty_range_inside_non_empty() {
        // 🛡️ Sentry Test: Verify overlaps() returns false when one range is empty, even if inside the other
        // This targets the || mutant in is_empty() checks
        let wide = TimeRange::new(100.into(), 200.into()).unwrap();
        let point = TimeRange::at(150.into());

        assert!(
            !wide.overlaps(&point),
            "Non-empty range should not overlap an empty range inside it"
        );
        assert!(
            !point.overlaps(&wide),
            "Empty range should not overlap a non-empty range"
        );
    }

    #[test]
    fn test_sentry_bitemporal_deserialize_excess_bytes() {
        // 🛡️ Sentry Test: Verify BiTemporalInterval::deserialize returns correct consumed length
        // even if given a buffer larger than required (kills < vs == mutant)
        let interval = BiTemporalInterval::now(100.into(), 200.into());
        let mut bytes = interval.serialize();
        bytes.push(0xFF); // length is now 49

        let (parsed, consumed) = BiTemporalInterval::deserialize(&bytes).unwrap();
        assert_eq!(parsed, interval);
        assert_eq!(consumed, 48, "Should consume exactly 48 bytes");
    }
}
