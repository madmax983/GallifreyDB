//! SQL:2011 Temporal Clause Handling.
//!
//! This module handles parsing and conversion of SQL:2011 temporal clauses:
//! - `FOR SYSTEM_TIME AS OF timestamp` - Point-in-time transaction time query
//! - `FOR SYSTEM_TIME BETWEEN t1 AND t2` - Transaction time range query
//! - `FOR VALID_TIME AS OF timestamp` - Point-in-time valid time query
//! - Combined temporal specifications

use crate::core::temporal::{TimeRange, Timestamp};
use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, Utc};

use super::error::SqlError;

/// Represents a parsed SQL:2011 temporal clause.
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalClause {
    /// Point-in-time query for system (transaction) time.
    /// `FOR SYSTEM_TIME AS OF timestamp`
    SystemTimeAsOf(Timestamp),

    /// Time range query for system (transaction) time.
    /// `FOR SYSTEM_TIME BETWEEN t1 AND t2`
    SystemTimeBetween(TimeRange),

    /// Point-in-time query for valid (application) time.
    /// `FOR VALID_TIME AS OF timestamp`
    ValidTimeAsOf(Timestamp),

    /// Time range query for valid (application) time.
    /// `FOR VALID_TIME BETWEEN t1 AND t2`
    ValidTimeBetween(TimeRange),

    /// Combined bi-temporal query.
    BiTemporal {
        /// System time specification
        system_time: Box<TemporalClause>,
        /// Valid time specification
        valid_time: Box<TemporalClause>,
    },
}

impl TemporalClause {
    /// Validates that a string matches the SQL timestamp format strictly.
    ///
    /// Accepts:
    /// - `YYYY-MM-DD HH:MM:SS` (exactly 19 characters)
    /// - `YYYY-MM-DD HH:MM:SS.f` (20-26 characters, where f is 1-6 digits)
    ///
    /// This prevents parse_from_str from accepting invalid inputs with trailing characters.
    /// Uses byte-level validation to avoid allocations.
    fn is_valid_sql_timestamp(s: &str) -> bool {
        let bytes = s.as_bytes();
        let len = bytes.len();

        // Must be at least 19 characters for "YYYY-MM-DD HH:MM:SS"
        if len < 19 {
            return false;
        }

        // Must be at most 26 characters for "YYYY-MM-DD HH:MM:SS.ffffff"
        if len > 26 {
            return false;
        }

        // Check structure: YYYY-MM-DD HH:MM:SS
        if bytes[4] != b'-'
            || bytes[7] != b'-'
            || bytes[10] != b' '
            || bytes[13] != b':'
            || bytes[16] != b':'
        {
            return false;
        }

        // Check digit positions
        let digit_positions = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18];
        for &pos in &digit_positions {
            if !bytes[pos].is_ascii_digit() {
                return false;
            }
        }

        // Check fractional seconds if present
        if len > 19 {
            if bytes[19] != b'.' {
                return false;
            }
            for &byte in &bytes[20..] {
                if !byte.is_ascii_digit() {
                    return false;
                }
            }
        }

        true
    }

    /// Parse a timestamp string.
    ///
    /// Supports the following formats:
    /// - **Unix microseconds**: `1705315200000000` (fast path)
    /// - **ISO 8601 UTC**: `2024-01-15T10:00:00Z`
    /// - **ISO 8601 with offset**: `2024-01-15T15:30:00+05:30`
    /// - **SQL timestamp**: `2024-01-15 10:00:00` (assumes UTC)
    /// - **Date only**: `2024-01-15` (assumes midnight UTC)
    ///
    /// All formats support optional single or double quotes and whitespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use aletheiadb::sql::TemporalClause;
    ///
    /// // Unix microseconds
    /// let ts = TemporalClause::parse_timestamp("1705315200000000").unwrap();
    ///
    /// // ISO 8601 UTC
    /// let ts = TemporalClause::parse_timestamp("2024-01-15T10:00:00Z").unwrap();
    ///
    /// // ISO 8601 with timezone
    /// let ts = TemporalClause::parse_timestamp("2024-01-15T15:30:00+05:30").unwrap();
    ///
    /// // SQL timestamp format
    /// let ts = TemporalClause::parse_timestamp("2024-01-15 10:00:00").unwrap();
    ///
    /// // Date only
    /// let ts = TemporalClause::parse_timestamp("2024-01-15").unwrap();
    /// ```
    pub fn parse_timestamp(s: &str) -> Result<Timestamp, SqlError> {
        let trimmed = s.trim().trim_matches('\'').trim_matches('"');

        // Fast path: Try parsing as Unix microseconds first
        if let Ok(micros) = trimmed.parse::<i64>() {
            return Ok(Timestamp::from(micros));
        }

        // Try parsing as ISO 8601 with UTC timezone (e.g., "2024-01-15T10:00:00Z")
        if let Ok(dt) = trimmed.parse::<DateTime<Utc>>() {
            return Ok(Timestamp::from(dt.timestamp_micros()));
        }

        // Try parsing as ISO 8601 with timezone offset (e.g., "2024-01-15T15:30:00+05:30")
        if let Ok(dt) = trimmed.parse::<DateTime<FixedOffset>>() {
            return Ok(Timestamp::from(dt.with_timezone(&Utc).timestamp_micros()));
        }

        // Try parsing as a naive datetime with ISO-like format (T separator)
        // e.g., "2024-01-15T10:00:00" or "2024-01-15T10:00:00.123456"
        // This parse is strict and does not allow trailing characters.
        // We assume UTC for naive timestamps.
        if let Ok(dt) = trimmed.parse::<NaiveDateTime>() {
            return Ok(Timestamp::from(dt.and_utc().timestamp_micros()));
        }

        // Try parsing as SQL timestamp format with space separator (e.g., "2024-01-15 10:00:00")
        // NaiveDateTime::FromStr doesn't support space separator, so we use parse_from_str.
        // To ensure strictness, we validate the format by checking length and characters.
        if Self::is_valid_sql_timestamp(trimmed) {
            // Try with fractional seconds first (more specific)
            if let Ok(dt) = NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S%.f") {
                return Ok(Timestamp::from(dt.and_utc().timestamp_micros()));
            }
            // Try without fractional seconds
            if let Ok(dt) = NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S") {
                return Ok(Timestamp::from(dt.and_utc().timestamp_micros()));
            }
        }

        // Try parsing as date-only format (e.g., "2024-01-15").
        // This parse is strict.
        // Assume midnight UTC.
        if let Ok(date) = trimmed.parse::<NaiveDate>() {
            // `and_hms_opt` is safe and will not panic.
            if let Some(dt) = date.and_hms_opt(0, 0, 0) {
                return Ok(Timestamp::from(dt.and_utc().timestamp_micros()));
            }
        }

        // None of the formats worked, return detailed error
        Err(SqlError::InvalidTimestamp(format!(
            "Cannot parse timestamp '{}'. Supported formats:\n\
             - Unix microseconds: 1705315200000000\n\
             - ISO 8601 UTC: 2024-01-15T10:00:00Z\n\
             - ISO 8601 with offset: 2024-01-15T15:30:00+05:30\n\
             - SQL timestamp: 2024-01-15 10:00:00 (assumes UTC)\n\
             - Date only: 2024-01-15 (assumes midnight UTC)",
            s
        )))
    }

    /// Create a SYSTEM_TIME AS OF clause.
    pub fn system_time_as_of(timestamp: Timestamp) -> Self {
        TemporalClause::SystemTimeAsOf(timestamp)
    }

    /// Create a VALID_TIME AS OF clause.
    pub fn valid_time_as_of(timestamp: Timestamp) -> Self {
        TemporalClause::ValidTimeAsOf(timestamp)
    }

    /// Create a SYSTEM_TIME BETWEEN clause.
    pub fn system_time_between(start: Timestamp, end: Timestamp) -> Result<Self, SqlError> {
        let range = TimeRange::new(start, end)
            .map_err(|e| SqlError::InvalidTemporalClause(format!("Invalid time range: {}", e)))?;
        Ok(TemporalClause::SystemTimeBetween(range))
    }

    /// Create a VALID_TIME BETWEEN clause.
    pub fn valid_time_between(start: Timestamp, end: Timestamp) -> Result<Self, SqlError> {
        let range = TimeRange::new(start, end)
            .map_err(|e| SqlError::InvalidTemporalClause(format!("Invalid time range: {}", e)))?;
        Ok(TemporalClause::ValidTimeBetween(range))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_validate_sql_timestamps_correctly() {
        struct TestCase<'a> {
            input: &'a str,
            expected: bool,
            desc: &'a str,
        }

        let cases = vec![
            TestCase {
                input: "2024-01-15 10:00:00",
                expected: true,
                desc: "Valid exact length timestamp",
            },
            TestCase {
                input: "2024-01-15 10:00:00.1",
                expected: true,
                desc: "Valid 1 fractional digit",
            },
            TestCase {
                input: "2024-01-15 10:00:00.123456",
                expected: true,
                desc: "Valid 6 fractional digits",
            },
            TestCase {
                input: "2024-01-15 10:00:00.1234567",
                expected: false,
                desc: "Invalid 7 fractional digits (too long)",
            },
            TestCase {
                input: "2024-01-15 10:00:0",
                expected: false,
                desc: "Invalid short length",
            },
            TestCase {
                input: "2024/01/15 10:00:00",
                expected: false,
                desc: "Invalid separator (slash instead of dash)",
            },
            TestCase {
                input: "2024-01-15T10:00:00",
                expected: false,
                desc: "Invalid separator (T instead of space)",
            },
            TestCase {
                input: "2024-01-15 10-00-00",
                expected: false,
                desc: "Invalid time separator",
            },
            TestCase {
                input: "abcd-ef-gh ij:kl:mn",
                expected: false,
                desc: "Invalid non-digit characters",
            },
            TestCase {
                input: "2024-01-15 10:00:00x123",
                expected: false,
                desc: "Invalid fractional separator",
            },
            TestCase {
                input: "2024-01-15 10:00:00.abc",
                expected: false,
                desc: "Invalid fractional non-digits",
            },
            TestCase {
                input: "2024-01-15 10:00:00 invalid",
                expected: false,
                desc: "Invalid trailing characters",
            },
        ];

        for case in cases {
            assert_eq!(
                TemporalClause::is_valid_sql_timestamp(case.input),
                case.expected,
                "Failed on: {}",
                case.desc
            );
        }
    }

    #[test]
    fn should_return_error_when_timestamp_format_unsupported() {
        let result = TemporalClause::parse_timestamp("unsupported_format");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            SqlError::InvalidTimestamp(msg) => {
                assert!(msg.contains("Cannot parse timestamp 'unsupported_format'"));
                assert!(msg.contains("Supported formats:"));
            }
            _ => panic!("Expected SqlError::InvalidTimestamp"),
        }
    }

    #[test]
    fn should_create_as_of_clauses_correctly() {
        let ts = Timestamp::from(1000);

        let sys_clause = TemporalClause::system_time_as_of(ts);
        assert_eq!(sys_clause, TemporalClause::SystemTimeAsOf(ts));

        let valid_clause = TemporalClause::valid_time_as_of(ts);
        assert_eq!(valid_clause, TemporalClause::ValidTimeAsOf(ts));
    }

    #[test]
    fn should_create_between_clauses_correctly() {
        let start = Timestamp::from(1000);
        let end = Timestamp::from(2000);

        let sys_clause = TemporalClause::system_time_between(start, end).unwrap();
        assert!(matches!(sys_clause, TemporalClause::SystemTimeBetween(_)));
        if let TemporalClause::SystemTimeBetween(range) = sys_clause {
            assert_eq!(range.start(), start);
            assert_eq!(range.end(), end);
        }

        let valid_clause = TemporalClause::valid_time_between(start, end).unwrap();
        assert!(matches!(valid_clause, TemporalClause::ValidTimeBetween(_)));
        if let TemporalClause::ValidTimeBetween(range) = valid_clause {
            assert_eq!(range.start(), start);
            assert_eq!(range.end(), end);
        }
    }

    #[test]
    fn should_return_error_when_between_clause_range_invalid() {
        let start = Timestamp::from(2000);
        let end = Timestamp::from(1000); // end before start

        let sys_err = TemporalClause::system_time_between(start, end).unwrap_err();
        match sys_err {
            SqlError::InvalidTemporalClause(msg) => {
                assert!(msg.contains("Invalid time range:"));
            }
            _ => panic!("Expected SqlError::InvalidTemporalClause"),
        }

        let valid_err = TemporalClause::valid_time_between(start, end).unwrap_err();
        match valid_err {
            SqlError::InvalidTemporalClause(msg) => {
                assert!(msg.contains("Invalid time range:"));
            }
            _ => panic!("Expected SqlError::InvalidTemporalClause"),
        }
    }

    #[test]
    fn test_sentry_temporal_parse_timestamp_validates_sql_format_accurately() {
        assert!(TemporalClause::parse_timestamp("1").is_ok());
        assert!(TemporalClause::parse_timestamp("2024-01-15T10:00:00").is_ok());
        assert!(TemporalClause::parse_timestamp("2024-01-15 10:00:00").is_ok());
        assert!(TemporalClause::parse_timestamp("2024-01-15").is_ok());

        let invalid = TemporalClause::parse_timestamp("2024-01-15 10:00:0");
        assert!(invalid.is_err());
    }
}
