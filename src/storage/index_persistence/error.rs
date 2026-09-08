//! Error types for index persistence operations.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during index persistence operations.
#[derive(Debug, Error)]
pub enum IndexPersistenceError {
    /// Index file is corrupted or invalid
    #[error("Index file corrupted: {path}")]
    Corrupted {
        /// Path to the corrupted file
        path: PathBuf,
        #[source]
        /// Source error
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// String interner index mismatch during restoration
    #[error("String interner mismatch: expected index {expected}, got {got}")]
    InternerMismatch {
        /// Expected index value
        expected: u32,
        /// Actual index value received
        got: u32,
    },

    /// Manifest version not supported
    #[error("Manifest version {found} not supported (max supported: {supported})")]
    UnsupportedVersion {
        /// Version found in manifest
        found: u16,
        /// Maximum supported version
        supported: u16,
    },

    /// Required index file is missing
    #[error("Missing required index file: {name}")]
    MissingIndex {
        /// Name of missing index file
        name: String,
    },

    /// Invalid magic bytes in file header
    #[error("Invalid magic bytes in {path}: expected {expected:?}, got {got:?}")]
    InvalidMagic {
        /// Path to file with invalid magic bytes
        path: PathBuf,
        /// Expected magic bytes
        expected: [u8; 4],
        /// Actual magic bytes found
        got: [u8; 4],
    },

    /// Size limit exceeded (DoS protection)
    #[error("Size limit exceeded: {message}")]
    SizeLimitExceeded {
        /// Description of the size limit violation
        message: String,
    },

    /// The process-global string interner is at capacity while serializing a
    /// property VALUE at persist time (configurable interner cap).
    ///
    /// A **structured** signal (as opposed to a `Serialization(String)` wrap of
    /// the interner's `CapacityExceeded` Display) so the background-persist
    /// worker can classify this failure as *deterministic* — suspend the affected
    /// index until the process restarts with a raised
    /// `persistence.max_interned_strings` — without brittle Display-text matching.
    #[error("String interner at capacity ({current}/{limit}) while persisting a property value")]
    InternerCapacityExceeded {
        /// The interner id that could not be admitted (== the size at the breach).
        current: usize,
        /// The configured cap that was hit.
        limit: usize,
    },

    /// IO error during persistence operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Bitcode serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<bitcode::Error> for IndexPersistenceError {
    fn from(e: bitcode::Error) -> Self {
        IndexPersistenceError::Serialization(e.to_string())
    }
}

impl IndexPersistenceError {
    /// Check if this error is due to a file not being found.
    pub fn is_not_found(&self) -> bool {
        matches!(
            self,
            IndexPersistenceError::Io(e) if e.kind() == std::io::ErrorKind::NotFound
        )
    }

    /// Convert this persistence error into a core [`Error`](crate::core::error::Error)
    /// for the write/persist path, PRESERVING a structured interner-capacity
    /// signal (configurable interner cap).
    ///
    /// [`IndexPersistenceError::InternerCapacityExceeded`] becomes a typed
    /// [`StorageError::CapacityExceeded`](crate::core::error::StorageError::CapacityExceeded)
    /// so the background-persist worker classifies it as *deterministic* (suspend
    /// until restart) by matching the variant — never by scraping Display text.
    /// Every other variant is wrapped as a `PersistenceError(String)` carrying
    /// `context` (a transient I/O failure, a not-yet-supported property kind),
    /// exactly as before, so those stay on the normal retry cadence.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn into_persist_storage_error(
        self,
        context: &str,
    ) -> crate::core::error::StorageError {
        match self {
            IndexPersistenceError::InternerCapacityExceeded { current, limit } => {
                crate::core::error::StorageError::CapacityExceeded {
                    resource: "string interner".to_string(),
                    current,
                    limit,
                }
            }
            other => {
                crate::core::error::StorageError::PersistenceError(format!("{context}: {other}"))
            }
        }
    }
}

/// Result type for index persistence operations.
pub type Result<T> = std::result::Result<T, IndexPersistenceError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn should_return_true_for_not_found_io_error() {
        let err =
            IndexPersistenceError::Io(io::Error::new(io::ErrorKind::NotFound, "file not found"));
        assert!(err.is_not_found());
    }

    #[test]
    fn should_return_false_for_other_io_errors() {
        let err = IndexPersistenceError::Io(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "permission denied",
        ));
        assert!(!err.is_not_found());
    }

    #[test]
    fn should_return_false_for_non_io_errors() {
        let err = IndexPersistenceError::MissingIndex {
            name: "test".to_string(),
        };
        assert!(!err.is_not_found());
    }

    #[test]
    fn should_convert_from_bitcode_error() {
        // Generate a bitcode error by trying to decode empty bytes into a u32
        let bitcode_err = bitcode::decode::<u32>(&[]).unwrap_err();
        let err: IndexPersistenceError = bitcode_err.into();

        match err {
            IndexPersistenceError::Serialization(msg) => {
                assert!(
                    !msg.is_empty(),
                    "Serialization error message should not be empty"
                );
            }
            _ => panic!("Expected Serialization error"),
        }
    }
}
