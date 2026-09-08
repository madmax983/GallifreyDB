//! Characterization: describe concepts + LLM/visualization export.
//!
//! Modules that compute properties of nodes/subgraphs (mass, archetype purity,
//! spectral decomposition, propagation behaviour) and that emit graph data in
//! formats consumed by downstream tools (Mermaid, force-directed layouts,
//! Markdown for LLM context).
//!
//! Experimental — gated by `features = ["semantic-characterization"]` (or the `nova` umbrella).

pub mod archetype;
pub mod graph_context;
pub mod gravity;
pub mod kaleidoscope;
pub mod papyrus;
pub mod prism;
pub mod sybil;
pub mod synapse;

// `wildfire` is a stub awaiting revival — see CHANGELOG / ADR-0050.
// pub mod wildfire;
#[cfg(feature = "semantic-characterization")]
pub mod bonsai;
pub mod starlight;
#[cfg(feature = "semantic-characterization")]
pub use bonsai::*;
