//! AletheiaDB Prelude
//!
//! The prelude module re-exports the most commonly used traits and types
//! to make starting with AletheiaDB easier.
//!
//! # Usage
//!
//! ```rust
//! use aletheiadb::prelude::*;
//! ```
//!
//! > **Note:** This module exports a custom `Result` type (`aletheiadb::Result<T>`)
//! > which shadows `std::result::Result`. If your code specifies `Result` with two
//! > generic parameters (e.g., `Result<(), Box<dyn std::error::Error>>`), you must
//! > fully qualify it as `std::result::Result` to avoid compiler errors.

pub use crate::AletheiaDB;
pub use crate::api::{ReadOps, ReadTransaction, WriteOps, WriteTransaction};
pub use crate::core::error::{Error, Result};
pub use crate::core::id::{EdgeId, NodeId, VersionId};
pub use crate::core::interning::InternedString;
pub use crate::core::property::{PropertyMap, PropertyMapBuilder, PropertyValue};
pub use crate::core::temporal::{BiTemporalInterval, TimeRange, Timestamp, time};
// Re-export the properties! macro. It is exported at the crate root because of #[macro_export],
// so we re-export it from there.
pub use crate::properties;
pub use crate::storage::wal::{DurabilityMode, WriteOptions};
