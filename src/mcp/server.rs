//! AletheiaDB MCP Server implementation.
//!
//! This module implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server
//! for AletheiaDB, enabling Large Language Models (LLMs) to interact with the database.
//!
//! # Overview
//!
//! The MCP server exposes AletheiaDB's capabilities as a set of "tools" that an LLM can invoke.
//! This allows AI agents to:
//! - **Query Knowledge**: Retrieve nodes, edges, and their properties.
//! - **Modify Graph**: Create, update, and delete nodes and edges.
//! - **Semantic Search**: Find similar nodes using vector embeddings.
//! - **Time Travel**: Query the graph as it existed at any point in history.
//! - **Hybrid Search**: Combine graph traversal, vector search, and temporal filtering.
//!
//! # Available Tools
//!
//! The server exposes over 20 tools, categorized by function:
//!
//! | Category | Key Tools | Description |
//! |----------|-----------|-------------|
//! | **Nodes** | `get_node`, `create_node`, `update_node` | CRUD operations for nodes |
//! | **Edges** | `get_edge`, `create_edge`, `traverse` | CRUD and traversal for edges |
//! | **Vector** | `find_similar`, `enable_vector_index` | Semantic similarity search |
//! | **Temporal** | `get_node_at_time`, `find_nodes_at_time`, `get_node_history`, `temporal_extent` | Time-travel queries, history, and dataset extent |
//! | **Hybrid** | `hybrid_query` | Combined graph + vector + temporal queries |
//! | **Schema** | `get_schema` | Discover node labels, edge types, and property keys (optionally bi-temporal) |
//!
//! # Examples
//!
//! Below are examples of how to interact with the key tools using JSON-RPC requests (the underlying protocol of MCP).
//!
//! ## 1. Creating a Node
//!
//! **Request:** `create_node`
//! ```json
//! {
//!   "label": "Person",
//!   "properties": {
//!     "name": "Alice",
//!     "age": 30,
//!     "interests": ["Rust", "Graphs"]
//!   }
//! }
//! ```
//!
//! **Response:**
//! ```json
//! {
//!   "id": 1,
//!   "label": "Person",
//!   "properties": {
//!     "name": "Alice",
//!     "age": 30,
//!     "interests": ["Rust", "Graphs"]
//!   }
//! }
//! ```
//!
//! ## 2. Vector Search
//!
//! **Request:** `find_similar`
//! ```json
//! {
//!   "property_name": "embedding",
//!   "embedding": [0.1, 0.2, 0.3, ...],
//!   "k": 5
//! }
//! ```
//!
//! **Response:**
//! ```json
//! {
//!   "results": [
//!     {
//!       "node": {
//!         "id": 42,
//!         "label": "Document",
//!         "properties": { "title": "Rust Guide", ... }
//!       },
//!       "score": 0.98
//!     },
//!     ...
//!   ],
//!   "count": 5
//! }
//! ```
//!
//! ## 3. Hybrid Query (Graph + Vector + Time)
//!
//! **Request:** `hybrid_query`
//! ```json
//! {
//!   "start_node_id": 1,
//!   "traverse_edge": "KNOWS",
//!   "query_embedding": [0.1, 0.2, ...],
//!   "valid_time": "2024-01-01T00:00:00Z"
//! }
//! ```
//!
//! # Usage
//!
//! The server is typically run as a standalone binary communicating over stdio:
//!
//! ```bash
//! cargo run --bin aletheia-mcp --features mcp-server
//! ```
//!
//! Or embedded within a larger application:
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::mcp::AletheiaMcpServer;
//! // Requires 'rmcp' dependency
//! use rmcp::{ServiceExt, transport::stdio};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let db = Arc::new(AletheiaDB::new()?);
//! let server = AletheiaMcpServer::new(db);
//!
//! // Run the server over stdio (blocks until stdin closes)
//! let service = server.serve(stdio()).await?;
//! service.waiting().await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use base64::Engine;
use chrono::{DateTime, Utc};
use rmcp::{
    ErrorData as RmcpErrorData, ServerHandler,
    model::{
        CallToolRequestParams, CallToolResult, Content, Implementation, ListToolsResult,
        PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
    },
    service::{RequestContext, RoleServer},
};
use serde_json::json;

use crate::api::transaction::WriteOps;
use crate::core::changefeed::ChangeCursor;
use crate::core::changefeed_subscription::{ChangeFilter, RecvError, Subscription};
use crate::core::temporal::time;
use crate::core::{
    ChangeFeedQuery, ChangeType, EdgeId, GLOBAL_INTERNER, NodeId, PropertyMap, PropertyMapBuilder,
    PropertyValue, Provenance, ProvenanceFilter, Timestamp, VersionId,
};
use crate::db::AletheiaDB;
use crate::index::vector::{DistanceMetric, HnswConfig};
use crate::query::executor::{EntityId as ResultEntityId, EntityResult};
use crate::query::limits::QueryResourceLimits;

use super::auth::{McpAuthConfig, SessionAuth, read_only_replica_error, tool_access_class};
use super::batch::ApplyBatchRequest;
use super::budget;
use super::cursor::{CursorManager, CursorPayload};
use super::error::{McpError, McpErrorCode, query_kind_classification};
use super::limits::{
    EffectiveQueryLimits, LimitCounters, LimitCountsSnapshot, LimitDimension, QueryLimitsConfig,
};
use super::tools::*;
use crate::auth::{AccessClass, AuthMode, Principal};

// ============================================================================
// Resource Limits (to prevent DoS attacks)
// ============================================================================

/// Maximum traversal depth to prevent stack overflow and excessive computation.
/// Increased from 10 to 20 to support business scenarios:
/// - Social network analysis (6 degrees of separation)
/// - Supply chain tracking (can exceed 10 hops)
/// - Genealogy and multi-generation queries
///
///   Still provides DoS protection via query timeouts and result limits.
const MAX_TRAVERSAL_DEPTH: usize = 20;

/// Maximum number of results to return in a single query.
const MAX_RESULT_LIMIT: usize = 10_000;

/// Default number of results to return.
const DEFAULT_RESULT_LIMIT: usize = 100;

/// Maximum pagination offset to prevent excessive memory usage.
/// Since we fetch limit+offset rows and then skip, this limits total rows fetched.
const MAX_PAGINATION_OFFSET: usize = 10_000;

/// Maximum k for vector similarity search.
const MAX_VECTOR_K: usize = 1000;

/// Node-expansion budget per unit of `max_depth` for the `semantic_path` tool
/// (Issue #2907). The unbounded A* search is a DoS risk on large/adversarial
/// graphs when the endpoints are attacker-controlled; the MCP surface therefore
/// derives a bounded expansion budget (`max_depth * this`, both clamped) and
/// aborts the search once it is exhausted.
#[cfg(feature = "semantic-search")]
const SEMANTIC_PATH_EXPANSIONS_PER_DEPTH: usize = 1_000;

/// Default k for vector similarity search.
const DEFAULT_VECTOR_K: usize = 10;

/// Default transaction time placeholder string.
const TRANSACTION_TIME_NOW: &str = "now";

/// Maximum UTF-8 byte length of a single text input accepted by the embedding
/// tools (Issue #2906). Bounds per-request embedding work and memory so an
/// oversized input is rejected with `INVALID_ARGUMENT` before any model runs.
///
/// Only referenced by the feature-on embedding handlers; gated so the
/// feature-off build (whose handlers return the unavailable-feature error
/// without validating) does not carry a dead constant.
#[cfg(feature = "embeddings")]
const MAX_EMBED_TEXT_BYTES: usize = 64 * 1024;

/// Maximum number of text strings accepted by `embed_text` in one call
/// (Issue #2906).
#[cfg(feature = "embeddings")]
const MAX_EMBED_TEXTS: usize = 256;

/// Maximum number of per-chunk embeddings `embed_text` returns in one call
/// (Issue #2906). Also the ceiling `max_chunks` is clamped to.
#[cfg(feature = "embeddings")]
const MAX_EMBED_CHUNKS: usize = 512;

/// Character window size used to chunk-expand each `embed_text` input document
/// before embedding (Issue #2906). A long document is split into contiguous
/// windows of at most this many Unicode scalar values, so it yields MULTIPLE
/// aligned embeddings instead of being silently truncated to a single vector.
/// Matches `embed_anything`'s default document chunk size.
#[cfg(feature = "embeddings")]
const EMBED_CHUNK_SIZE_CHARS: usize = 1000;

/// Split `text` into contiguous character windows of at most `chunk_size_chars`
/// Unicode scalar values each (Issue #2906 chunk expansion).
///
/// A deterministic, model-independent splitter so a long `embed_text` input
/// document expands into MULTIPLE chunks (each subsequently embedded and
/// aligned to its own [`EmbedData`](crate::embeddings::EmbedData)) rather than
/// being silently truncated to a single vector. Splits on Unicode scalar
/// boundaries, so every chunk is valid UTF-8. Empty input yields a single empty
/// chunk (callers validate non-empty input upstream). `chunk_size_chars` must
/// be non-zero.
#[cfg(feature = "embeddings")]
pub(crate) fn split_text_into_chunks(text: &str, chunk_size_chars: usize) -> Vec<String> {
    debug_assert!(chunk_size_chars > 0, "chunk size must be non-zero");
    let chunk_size_chars = chunk_size_chars.max(1);
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut count = 0usize;
    for ch in text.chars() {
        current.push(ch);
        count += 1;
        if count == chunk_size_chars {
            chunks.push(std::mem::take(&mut current));
            count = 0;
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Default maximum number of operations accepted by a single `apply_batch`
/// call (Issue #3231). Deliberately far below the core transaction buffer's
/// own DoS cap (`WriteBuffer::DEFAULT_MAX_OPERATIONS` = 50,000), so the MCP
/// surface bound is always the one that fires, with the limit echoed in the
/// rejection per the #3226 completeness convention.
pub(crate) const DEFAULT_MAX_BATCH_OPERATIONS: usize = 1000;

/// Default maximum number of designation targets accepted by a single
/// `designate_subject` call (Issue #3701 hardening). Mirrors the `apply_batch`
/// cap: an over-cap request is rejected up front with `INVALID_ARGUMENT`
/// (limit echoed per the #3226 convention) before any registry mutation, so an
/// unbounded `targets` array cannot be used as a DoS vector. Configurable per
/// server via [`AletheiaMcpServer::with_max_designate_targets`].
pub(crate) const DEFAULT_MAX_DESIGNATE_TARGETS: usize = 1000;

/// AletheiaDB MCP Server.
///
/// Exposes AletheiaDB's graph, vector, and temporal capabilities through the Model Context Protocol.
/// This struct implements `rmcp::ServerHandler` to handle tool calls from an MCP client (typically an LLM).
///
/// # Resource Limits
///
/// To prevent Denial of Service (DoS) attacks and excessive resource usage, the server enforces several limits:
/// - **Max Traversal Depth**: 20 hops (prevents infinite loops and stack overflows)
/// - **Max Results**: 10,000 items (prevents OOM on large result sets)
/// - **Max Vector K**: 1,000 nearest neighbors (limits expensive similarity calculations)
///
/// # Thread Safety
///
/// The server is thread-safe and can handle concurrent requests. It holds an `Arc` to the underlying
/// database instance, which handles its own concurrency control via MVCC and lock-free structures.
#[derive(Clone)]
pub struct AletheiaMcpServer {
    db: Arc<AletheiaDB>,
    /// Maximum operations accepted by one `apply_batch` call (Issue #3231).
    pub(crate) max_batch_operations: usize,
    /// Maximum number of designation targets accepted by one
    /// `designate_subject` call (Issue #3701 hardening). An over-cap request is
    /// rejected up front with `INVALID_ARGUMENT` before any registry mutation.
    pub(crate) max_designate_targets: usize,
    /// Maximum number of entries accepted in a `priority_properties` budget
    /// array (Issue #3583). Bounds the one-time cost of building the protected-
    /// key set; an over-cap array is rejected with `INVALID_ARGUMENT`.
    pub(crate) max_priority_properties: usize,
    auth: SessionAuth,
    /// Snapshot-anchored keyset continuation cursors (Issue #3360). Shared
    /// (one manager == one MCP connection) so its live-cursor cap is a
    /// per-connection cap.
    cursors: Arc<CursorManager>,
    /// Per-query resource limits for the `query` tool (Issue #3368): wall-clock
    /// timeout, result-row cap, and result-byte cap, each with a server default,
    /// an operator ceiling, and a per-call override.
    query_limits: Arc<QueryLimitsConfig>,
    /// Per-dimension counters of over-limit terminations (Issue #3368
    /// observability). Shared so a cloned server (used for the timeout
    /// thread-race) increments the same counters.
    limit_counters: Arc<LimitCounters>,
    /// Number of `query` calls currently occupying a timeout-race worker
    /// thread (Issue #3368 DoS guard). Bounded by
    /// [`QueryLimitsConfig::max_in_flight_queries`]; a worker holds its slot
    /// for its whole (non-cancellable) lifetime — even after the caller has
    /// timed out and discarded it — so a still-running discarded query keeps
    /// occupying capacity. Shared across clones so the cloned server used for
    /// the race sees the same live count.
    in_flight_queries: Arc<AtomicUsize>,
    /// Optional embedding-model handle (Issue #2906). Present only when the
    /// `embeddings` feature is compiled AND a model has been configured via
    /// [`with_embedder`](Self::with_embedder). `None` -> the embedding tools
    /// (`embed_query`, `embed_text`, `semantic_search`,
    /// `create_node_with_embedding`, `update_node_embedding`) return
    /// `FAILED_PRECONDITION` ("no embedding model configured"). The tools are
    /// advertised unconditionally (Design A); when the feature is *not*
    /// compiled they return a structured unavailable-feature error instead.
    #[cfg(feature = "embeddings")]
    embedder: Option<Arc<crate::embeddings::Embedder>>,
}

/// Outcome of the `await_changes` synchronous prelude (Issue #3673): either an
/// immediate result (bad args / subscribe error / catch-up hit) or a live
/// [`Subscription`] to wait on for `timeout`. Shared by the sync
/// [`AletheiaMcpServer::handle_await_changes`] (blocking Condvar) and the async
/// [`AletheiaMcpServer::dispatch_await_changes_async`] (event-driven Notify) so
/// both run identical subscribe + catch-up logic and differ only in HOW they
/// wait. The `Immediate` result is boxed so the enum's variants stay
/// size-balanced (`CallToolResult` is far larger than a `Subscription`).
enum AwaitChangesStep {
    Immediate(Box<CallToolResult>),
    Block {
        sub: Subscription,
        timeout: std::time::Duration,
    },
}

/// RAII slot in the bounded in-flight-query pool (Issue #3368). Decrements the
/// shared counter on drop — which happens inside the spawned worker closure
/// when the WORKER finishes (never when the caller merely times out), so a
/// discarded-but-still-running query keeps holding its slot until it actually
/// completes.
struct InFlightGuard {
    counter: Arc<AtomicUsize>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Outcome of racing a query computation against its wall-clock deadline
/// (Issue #3368). Distinguishes a real timeout (retriable `RESOURCE_EXHAUSTED`)
/// from a worker that died — e.g. panicked — mid-computation (non-retriable
/// `INTERNAL`), so a deterministic panic is never mislabeled as an invisible,
/// infinitely-retriable timeout.
enum RaceOutcome {
    /// The worker finished within the deadline; carries its result.
    Completed(CallToolResult),
    /// The deadline elapsed before the worker finished.
    TimedOut,
    /// The worker's sender was dropped without a result (it panicked or was
    /// otherwise torn down) before the deadline.
    WorkerDied,
}

impl AletheiaMcpServer {
    /// Create a new MCP server wrapping a AletheiaDB instance.
    ///
    /// # Authentication
    ///
    /// This constructor is the **embedded/programmatic** entry point and is
    /// deliberately source-compatible with pre-#3350 behavior: it runs in
    /// anonymous mode (no authentication — a Rust caller holding the
    /// `Arc<AletheiaDB>` can already do anything the tools can). Serving
    /// deployments should use [`with_auth`](Self::with_auth); the
    /// `aletheia-mcp` binary requires authentication by default.
    ///
    /// # Arguments
    ///
    /// * `db` - An `Arc` wrapping the initialized database instance.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use aletheiadb::AletheiaDB;
    /// use aletheiadb::mcp::AletheiaMcpServer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = Arc::new(AletheiaDB::new()?);
    /// let server = AletheiaMcpServer::new(db);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(db: Arc<AletheiaDB>) -> Self {
        Self {
            db,
            max_batch_operations: DEFAULT_MAX_BATCH_OPERATIONS,
            max_designate_targets: DEFAULT_MAX_DESIGNATE_TARGETS,
            max_priority_properties: budget::DEFAULT_MAX_PRIORITY_PROPERTIES,
            auth: SessionAuth::Anonymous,
            cursors: Arc::new(CursorManager::new()),
            query_limits: Arc::new(QueryLimitsConfig::default()),
            limit_counters: Arc::new(LimitCounters::default()),
            in_flight_queries: Arc::new(AtomicUsize::new(0)),
            #[cfg(feature = "embeddings")]
            embedder: None,
        }
    }

    /// Configure the embedding-model handle used by the embedding tools
    /// (Issue #2906).
    ///
    /// The embedding tools are advertised unconditionally, but they only
    /// produce embeddings once a model is configured here; until then they
    /// return `FAILED_PRECONDITION`. A model is deliberately NOT built
    /// implicitly (e.g. from an env var) inside [`new`](Self::new) /
    /// [`with_auth`](Self::with_auth) because loading a model can download
    /// weights and block — unacceptable for the many embedded/test callers of
    /// those constructors. Serving deployments call this explicitly after
    /// building the [`Embedder`](crate::embeddings::Embedder).
    #[cfg(feature = "embeddings")]
    #[must_use = "with_embedder returns a new server; discarding it drops the configured model"]
    pub fn with_embedder(mut self, embedder: Arc<crate::embeddings::Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Override the per-query resource limits for the `query` tool (Issue
    /// #3368): the wall-clock timeout, result-row, and result-byte defaults +
    /// operator ceilings that a per-call `limits` override is folded against.
    ///
    /// Defaults are [`QueryLimitsConfig::default`] (enabled, generous). Pass
    /// [`QueryLimitsConfig::disabled`] to turn all enforcement off (unlimited,
    /// per-call overrides ignored).
    #[must_use]
    pub fn with_query_limits(mut self, limits: QueryLimitsConfig) -> Self {
        self.query_limits = Arc::new(limits);
        self
    }

    /// Snapshot of the per-dimension over-limit termination counters (Issue
    /// #3368 observability): how many `query` calls were terminated by the
    /// wall-clock timeout or rejected by the response-byte cap, and how many
    /// per-call overrides were rejected for exceeding an operator ceiling.
    ///
    /// Surfacing these through `database_stats` needs a storage-layer counter
    /// and is a cross-lane follow-up; this MCP-surface accessor is the interim.
    #[must_use]
    pub fn limit_termination_counts(&self) -> LimitCountsSnapshot {
        self.limit_counters.snapshot()
    }

    /// Override the cursor lifecycle bounds (Issue #3360): the continuation
    /// cursor TTL and the per-connection cap on concurrently live cursors.
    /// Defaults are a 5-minute TTL and 128 live cursors.
    #[must_use]
    pub fn with_cursor_config(mut self, ttl: std::time::Duration, max_live_cursors: usize) -> Self {
        self.cursors = Arc::new(CursorManager::with_config(ttl, max_live_cursors));
        self
    }

    /// Override the maximum number of operations accepted by a single
    /// `apply_batch` call (default: 1000, Issue #3231). An over-limit batch
    /// is rejected before any operation runs, with the limit echoed in the
    /// structured error's `details` (per the #3226 completeness convention).
    ///
    /// The cap is an MCP-surface payload bound; the core transaction buffer
    /// keeps its own independent DoS ceiling (50,000 ops).
    #[must_use]
    pub fn with_max_batch_operations(mut self, max_batch_operations: usize) -> Self {
        self.max_batch_operations = max_batch_operations;
        self
    }

    /// Override the maximum number of designation targets accepted by a single
    /// `designate_subject` call (default: 1000, Issue #3701). An over-limit
    /// request is rejected before any registry mutation, with the limit echoed
    /// in the structured error's `details` (per the #3226 completeness
    /// convention).
    ///
    /// The cap is an MCP-surface payload bound mirroring
    /// [`with_max_batch_operations`](Self::with_max_batch_operations), guarding
    /// against an unbounded `targets` array being used as a DoS vector.
    #[must_use]
    pub fn with_max_designate_targets(mut self, max_designate_targets: usize) -> Self {
        self.max_designate_targets = max_designate_targets;
        self
    }

    /// Override the maximum number of entries accepted in a `priority_properties`
    /// token-budget array (default: 1024, Issue #3583).
    ///
    /// `priority_properties` (Issue #3353) names the property keys a budgeted
    /// read protects from elision. It is consulted for every property of every
    /// returned entity, so an unbounded array is a denial-of-service vector:
    /// this cap rejects an over-long array up front with a structured
    /// `INVALID_ARGUMENT` error (naming the cap and the given length) before any
    /// shaping runs, keeping response-shaping cost bounded regardless of caller
    /// input. The per-key lookup itself is O(1); this cap additionally bounds the
    /// one-time cost of validating the array and building the lookup set.
    #[must_use = "with_max_priority_properties returns a new server; discarding it drops the configured limit"]
    pub fn with_max_priority_properties(mut self, max_priority_properties: usize) -> Self {
        self.max_priority_properties = max_priority_properties;
        self
    }

    /// Create an MCP server with authentication and role-based
    /// authorization (Issue #3350, Phase 2).
    ///
    /// The MCP transport is stdio, so the credential is **session-scoped**:
    /// supplied once at construction (see
    /// [`McpAuthConfig::with_credential`]) and re-verified against the
    /// [`AuthStore`](crate::auth::AuthStore) on every tool call, so a
    /// revocation in the (possibly HTTP-shared) store takes effect on the
    /// next call.
    ///
    /// Behavior per [`AuthMode`]:
    ///
    /// - `Required` + valid credential → every tool call is authorized
    ///   against the principal's role (see the matrix in
    ///   `docs/guides/access-control-matrix.md`).
    /// - `Required` + missing/invalid/revoked credential → the server still
    ///   serves, but **every** tool call (including unknown tool names)
    ///   returns the uniform `UNAUTHENTICATED` error.
    /// - `Anonymous` → full access, exactly like [`new`](Self::new); a
    ///   prominent warning is emitted on stderr (never stdout — that is the
    ///   MCP protocol channel).
    pub fn with_auth(db: Arc<AletheiaDB>, auth: McpAuthConfig) -> Self {
        if auth.mode() == AuthMode::Anonymous {
            // PROMINENT warning: the operator explicitly opted out of auth.
            // stderr only — stdout carries the MCP protocol.
            eprintln!(
                "WARNING: AUTHENTICATION IS DISABLED (auth mode: anonymous). \
                 Every MCP tool call has full, unauthenticated access to the \
                 database. Do not expose this server to untrusted callers."
            );
        }
        Self {
            db,
            max_batch_operations: DEFAULT_MAX_BATCH_OPERATIONS,
            max_designate_targets: DEFAULT_MAX_DESIGNATE_TARGETS,
            max_priority_properties: budget::DEFAULT_MAX_PRIORITY_PROPERTIES,
            auth: SessionAuth::from(auth),
            cursors: Arc::new(CursorManager::new()),
            query_limits: Arc::new(QueryLimitsConfig::default()),
            limit_counters: Arc::new(LimitCounters::default()),
            in_flight_queries: Arc::new(AtomicUsize::new(0)),
            #[cfg(feature = "embeddings")]
            embedder: None,
        }
    }

    /// The verified session principal, if any.
    ///
    /// Re-verifies the session credential against the store, so a revoked
    /// key yields `None`. Always `None` in anonymous mode. Principal `id`
    /// and `name` are safe to log or stamp into provenance; key material
    /// never reaches this type.
    pub fn session_principal(&self) -> Option<Principal> {
        self.auth.principal()
    }

    /// Get a reference to the underlying database.
    ///
    /// Useful if you need to access the database directly for operations not exposed via MCP.
    pub fn db(&self) -> &Arc<AletheiaDB> {
        &self.db
    }

    // ========================================================================
    // Public Tool Methods (for testing and direct access)
    // ========================================================================

    /// Extract text content from a CallToolResult.
    ///
    /// Returns an error JSON string if the result contains no text content.
    pub(crate) fn extract_text(result: CallToolResult) -> String {
        result
            .content
            .first()
            .and_then(|c| c.as_text().map(|s| s.text.clone()))
            .unwrap_or_else(|| {
                json!({
                    "error": McpError::new(McpErrorCode::Internal, "No content in response")
                        .to_json()
                })
                .to_string()
            })
    }

    /// Get a node by its ID.
    ///
    /// Returns the node's label and all properties in JSON format.
    ///
    /// # Output Format
    ///
    /// ```json
    /// {
    ///   "id": 123,
    ///   "label": "Person",
    ///   "properties": {
    ///     "name": "Alice",
    ///     "age": 30
    ///   }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a JSON object with a structured "error" object if the node is
    /// not found (Issue #3234 error contract):
    /// ```json
    /// {
    ///   "error": {
    ///     "code": "NOT_FOUND",
    ///     "message": "Storage error: Node not found: Node(123)",
    ///     "retriable": false
    ///   }
    /// }
    /// ```
    pub fn get_node(&self, req: GetNodeRequest) -> String {
        Self::extract_text(self.handle_get_node(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Create a new node.
    ///
    /// Creates a node with the specified label and optional properties.
    /// Returns the created node's ID and details in JSON format.
    ///
    /// # Example Request
    ///
    /// ```rust
    /// use aletheiadb::mcp::CreateNodeRequest;
    /// use std::collections::HashMap;
    /// use serde_json::json;
    ///
    /// let mut props = HashMap::new();
    /// props.insert("name".to_string(), json!("Alice"));
    ///
    /// let req = CreateNodeRequest {
    ///     label: "Person".to_string(),
    ///     properties: Some(props),
    ///     valid_time: None,
    ///     provenance: None,
    ///     derived_from: None,
    ///     namespace: None,
    /// };
    /// ```
    ///
    /// # Output Format
    ///
    /// Returns the complete node object including the newly assigned ID.
    ///
    /// ```json
    /// {
    ///   "id": 1,
    ///   "label": "Person",
    ///   "properties": { "name": "Alice" }
    /// }
    /// ```
    pub fn create_node(&self, req: CreateNodeRequest) -> String {
        Self::extract_text(self.handle_create_node(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Update a node's properties.
    ///
    /// Merges the provided properties with existing ones.
    /// - New keys are added.
    /// - Existing keys are updated.
    /// - Keys set to `null` are removed (future feature, currently sets to Null).
    ///
    /// # Output Format
    ///
    /// Returns the updated node object.
    pub fn update_node(&self, req: UpdateNodeRequest) -> String {
        Self::extract_text(self.handle_update_node(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Generate a single dense embedding from text (Issue #2906).
    ///
    /// Returns the tool's JSON response as a string. When the `embeddings`
    /// feature is not compiled (or no model is configured) this returns the
    /// structured unavailable/precondition error rather than an embedding.
    pub fn embed_query(&self, req: EmbedQueryRequest) -> String {
        Self::extract_text(self.handle_embed_query(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Generate per-chunk dense embeddings for multiple texts (Issue #2906).
    pub fn embed_text(&self, req: EmbedTextRequest) -> String {
        Self::extract_text(self.handle_embed_text(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Create a node whose embedding is generated from text (Issue #2906).
    pub fn create_node_with_embedding(&self, req: CreateNodeWithEmbeddingRequest) -> String {
        Self::extract_text(self.handle_create_node_with_embedding(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Regenerate a node's embedding property from text, preserving all other
    /// properties (Issue #2906).
    pub fn update_node_embedding(&self, req: UpdateNodeEmbeddingRequest) -> String {
        Self::extract_text(self.handle_update_node_embedding(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Delete a node (safe-by-default).
    ///
    /// Permanently removes the node from the current state. Historical versions
    /// remain accessible via time-travel queries.
    ///
    /// Mirrors Cypher's `DETACH DELETE` contract (Issue #3209): if the node has
    /// connected edges and `detach` is not `true`, the deletion is **refused**
    /// and the JSON response reports `connected_edges` (the number of edges that
    /// would be orphaned) so the caller can decide. Pass `detach: true` to delete
    /// the node together with all connected edges; the response then reports
    /// `edges_removed`. A node with no connected edges always deletes cleanly.
    pub fn delete_node(&self, req: DeleteNodeRequest) -> String {
        Self::extract_text(self.handle_delete_node(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Retract a node as of a valid time (Issue #3230).
    ///
    /// Closes the node's valid-time interval at `valid_time` (default: now)
    /// WITHOUT deleting its history: `AS OF VALID_TIME` queries strictly
    /// before that instant still return the node, and `AS OF SYSTEM_TIME`
    /// queries positioned before the retraction still show it open-ended.
    ///
    /// Mirrors the `delete_node` DETACH contract (Issue #3209): if the node
    /// has connected edges and `detach` is not `true`, the retraction is
    /// **refused** and the JSON response reports `connected_edges`. Pass
    /// `detach: true` to co-retract every connected edge at the same valid
    /// time; the response then reports `edges_retracted`. Re-retracting an
    /// already-retracted node is an idempotent no-op returning the existing
    /// interval with `already_retracted: true`.
    pub fn retract_node(&self, req: RetractNodeRequest) -> String {
        Self::extract_text(self.handle_retract_node(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Retract an edge as of a valid time (Issue #3230).
    ///
    /// See [`retract_node`](Self::retract_node) for the bi-temporal
    /// semantics and idempotency; edges have no detach concern.
    pub fn retract_edge(&self, req: RetractEdgeRequest) -> String {
        Self::extract_text(self.handle_retract_edge(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Delete a node and all its connected edges (cascade delete).
    ///
    /// Removes the node and any edges connected to it, maintaining referential integrity.
    pub fn delete_node_cascade(&self, req: DeleteNodeCascadeRequest) -> String {
        Self::extract_text(self.handle_delete_node_cascade(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// List nodes with optional filtering.
    ///
    /// Supports filtering by label and pagination (limit/offset).
    /// Note: Listing all nodes without filters can be expensive on large graphs.
    pub fn list_nodes(&self, req: ListNodesRequest) -> String {
        Self::extract_text(self.handle_list_nodes(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Count nodes.
    ///
    /// Returns the total number of nodes in the graph, or nodes matching a specific label.
    pub fn count_nodes(&self, req: CountNodesRequest) -> String {
        Self::extract_text(self.handle_count_nodes(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Dispatch a tool by name from a raw JSON arguments object, returning the
    /// tool's JSON string result.
    ///
    /// Unlike the per-tool typed methods (e.g. [`get_edge`](Self::get_edge)),
    /// which call their `handle_*` directly, this routes through
    /// [`dispatch_tool`](Self::dispatch_tool) so the raw-argument-driven
    /// cross-cutting read features apply: the Issue #3353 token-budget shaping
    /// (`max_response_tokens` / `max_response_bytes` / `priority_properties`)
    /// and the Issue #3360 snapshot-anchored cursor paging (`use_cursor` /
    /// `cursor`), both of which are read off the raw arguments and are therefore
    /// invisible to the typed request structs. On the embedded/anonymous
    /// [`AletheiaMcpServer::new`] path the built-in tool authorization is a
    /// no-op; a caller that has authorized the tool out-of-band (e.g. the
    /// autumn-web HTTP surface's own RBAC gate) gets the same result a direct
    /// `handle_*` would produce when neither budget nor cursor is present.
    ///
    // exposed for autumn-web migration (Issue #3524)
    pub fn dispatch_tool_json(&self, name: &str, args: serde_json::Value) -> String {
        Self::extract_text(self.dispatch_tool(name, args))
    }

    /// Every advertised tool's classified [`AccessClass`], exposed for
    /// external conformance sweeps (e.g. the Issue #3355 replica read-only
    /// enforcement test) without depending on the crate-private
    /// `TOOL_ACCESS_CLASSES` table directly.
    #[must_use]
    pub fn tool_access_classes() -> Vec<(&'static str, AccessClass)> {
        super::auth::TOOL_ACCESS_CLASSES.to_vec()
    }

    /// Get an edge by its ID.
    ///
    /// Returns the edge's source, target, label, and properties.
    pub fn get_edge(&self, req: GetEdgeRequest) -> String {
        Self::extract_text(self.handle_get_edge(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Create a new edge.
    ///
    /// Establishes a relationship between two existing nodes.
    /// Fails if either source or target node does not exist.
    pub fn create_edge(&self, req: CreateEdgeRequest) -> String {
        Self::extract_text(self.handle_create_edge(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Update an edge's properties.
    ///
    /// Merges properties similar to `update_node`.
    pub fn update_edge(&self, req: UpdateEdgeRequest) -> String {
        Self::extract_text(self.handle_update_edge(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Delete an edge.
    ///
    /// Removes the relationship between two nodes.
    pub fn delete_edge(&self, req: DeleteEdgeRequest) -> String {
        Self::extract_text(self.handle_delete_edge(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// List edges.
    ///
    /// Lists edges with pagination. Note that filtering edges by label without a start node is not currently efficient.
    pub fn list_edges(&self, req: ListEdgesRequest) -> String {
        Self::extract_text(self.handle_list_edges(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Count edges.
    ///
    /// Returns the total number of edges in the graph.
    pub fn count_edges(&self, req: CountEdgesRequest) -> String {
        Self::extract_text(self.handle_count_edges(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Get outgoing edges from a node.
    ///
    /// Returns all edges starting from the specified node. Can be filtered by edge label.
    pub fn get_outgoing_edges(&self, req: GetOutgoingEdgesRequest) -> String {
        Self::extract_text(self.handle_get_outgoing_edges(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Get incoming edges to a node.
    ///
    /// Returns all edges ending at the specified node. Can be filtered by edge label.
    pub fn get_incoming_edges(&self, req: GetIncomingEdgesRequest) -> String {
        Self::extract_text(self.handle_get_incoming_edges(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Get the complete version history of a node.
    ///
    /// Returns all versions in chronological order (oldest first), including
    /// each version's write-time provenance bundle when present (Issue #3224).
    pub fn get_node_history(&self, req: GetNodeHistoryRequest) -> String {
        Self::extract_text(self.handle_get_node_history(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Get the complete version history of an edge.
    ///
    /// Returns all versions in chronological order (oldest first), including
    /// each version's write-time provenance bundle when present (Issue #3224).
    pub fn get_edge_history(&self, req: GetEdgeHistoryRequest) -> String {
        Self::extract_text(self.handle_get_edge_history(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Traverse the graph.
    ///
    /// Performs a multi-hop traversal starting from a node, following edges of a specific type.
    /// Returns the path and the final nodes found.
    pub fn traverse(&self, req: TraverseRequest) -> String {
        Self::extract_text(self.handle_traverse(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Create (register) an agent-scoped namespace (Issue #3349, PR3b).
    ///
    /// Returns the created `{name, description, created_at}`; a duplicate name
    /// (or the implicit `default`) is a `CONFLICT`, a malformed/reserved name is
    /// `INVALID_ARGUMENT`.
    pub fn create_namespace(&self, req: CreateNamespaceRequest) -> String {
        Self::extract_text(self.handle_create_namespace(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// List all registered namespaces with per-namespace counts (Issue #3349,
    /// PR3b).
    pub fn list_namespaces(&self, req: ListNamespacesRequest) -> String {
        Self::extract_text(self.handle_list_namespaces(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Describe a single namespace by name (Issue #3349, PR3b). An unregistered
    /// name is `NOT_FOUND` (`details.namespace`).
    pub fn describe_namespace(&self, req: DescribeNamespaceRequest) -> String {
        Self::extract_text(self.handle_describe_namespace(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// ADMIN. Designate a GDPR erasure subject over one or more targets
    /// (Issue #3359, Slice 4b). The first Admin-class MCP tool.
    pub fn designate_subject(&self, req: DesignateSubjectRequest) -> String {
        Self::extract_text(self.handle_designate_subject(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// ADMIN. Irreversibly erase a designated subject and return a signed
    /// erasure attestation (Issue #3359, Slice 4b).
    pub fn erase_subject(&self, req: EraseSubjectRequest) -> String {
        Self::extract_text(self.handle_erase_subject(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Find similar nodes.
    ///
    /// Performs a K-Nearest Neighbors (k-NN) search using vector embeddings.
    ///
    /// # Prerequisites
    ///
    /// A vector index must be enabled on the target property using `enable_vector_index`
    /// before this method can be used.
    ///
    /// # Output Format
    ///
    /// Returns a list of matches with their similarity scores.
    ///
    /// ```json
    /// {
    ///   "results": [
    ///     {
    ///       "node": {
    ///         "id": 123,
    ///         "label": "Document",
    ///         "properties": { "title": "..." }
    ///       },
    ///       "score": 0.95
    ///     }
    ///   ],
    ///   "count": 1
    /// }
    /// ```
    pub fn find_similar(&self, req: FindSimilarRequest) -> String {
        Self::extract_text(self.handle_find_similar(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Enable vector index.
    ///
    /// Configures and builds an HNSW index on a specific property, enabling semantic search.
    /// This is a prerequisite for `find_similar`.
    ///
    /// # Arguments
    ///
    /// * `property_name`: The property to index (e.g., "embedding").
    /// * `dimensions`: The size of the vector (e.g., 1536 for OpenAI).
    /// * `distance_metric`: "cosine" (default), "euclidean", or "dot".
    ///
    /// # Output Format
    ///
    /// Returns a success confirmation.
    ///
    /// ```json
    /// {
    ///   "success": true,
    ///   "property_name": "embedding",
    ///   "dimensions": 1536,
    ///   "distance_metric": "cosine"
    /// }
    /// ```
    pub fn enable_vector_index(&self, req: EnableVectorIndexRequest) -> String {
        Self::extract_text(self.handle_enable_vector_index(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// List vector indexes.
    ///
    /// Returns a list of all active vector indexes and their configuration (dimensions, metric).
    ///
    /// # Output Format
    ///
    /// ```json
    /// {
    ///   "indexes": [
    ///     {
    ///       "property_name": "embedding",
    ///       "dimensions": 1536,
    ///       "distance_metric": "Cosine"
    ///     }
    ///   ],
    ///   "count": 1
    /// }
    /// ```
    pub fn list_vector_indexes(&self, req: ListVectorIndexesRequest) -> String {
        Self::extract_text(self.handle_list_vector_indexes(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Enable a uniqueness constraint on a label+property pair.
    pub fn enable_unique_constraint(&self, req: EnableUniqueConstraintRequest) -> String {
        Self::extract_text(self.handle_enable_unique_constraint(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// List all active uniqueness constraints.
    pub fn list_unique_constraints(&self, req: ListUniqueConstraintsRequest) -> String {
        Self::extract_text(self.handle_list_unique_constraints(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Get node at a specific time.
    ///
    /// Performs a time-travel query to retrieve the state of a node at a specific valid time and transaction time.
    pub fn get_node_at_time(&self, req: GetNodeAtTimeRequest) -> String {
        Self::extract_text(self.handle_get_node_at_time(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Get edge at a specific time.
    ///
    /// Performs a time-travel query to retrieve the state of an edge at a specific valid time and transaction time.
    pub fn get_edge_at_time(&self, req: GetEdgeAtTimeRequest) -> String {
        Self::extract_text(self.handle_get_edge_at_time(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Find nodes by label (and optional exact property match) as of a
    /// bi-temporal point (Issue #3236).
    ///
    /// The entry-point resolver for "the Person named Alice, as of
    /// 2024-01-01" when no `NodeId` is known: each returned node is
    /// reconstructed as it existed at `(valid_time, transaction_time)`.
    pub fn find_nodes_at_time(&self, req: FindNodesAtTimeRequest) -> String {
        Self::extract_text(self.handle_find_nodes_at_time(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Discover the graph's schema: distinct node labels and edge types,
    /// their counts, and the property keys observed on each.
    ///
    /// With no `as_of_*` fields, returns the current-state schema. With
    /// either field set, returns the schema as it existed at that bi-temporal
    /// instant. Never errors on an empty database.
    pub fn get_schema(&self, req: GetSchemaRequest) -> String {
        Self::extract_text(self.handle_get_schema(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Report the dataset's queryable bi-temporal extent: earliest/latest
    /// valid-time and transaction-time bounds across all recorded history
    /// (including expired/superseded versions), as RFC3339 strings.
    ///
    /// Pass `by_label: true` for a per-node-label / per-edge-type breakdown.
    /// An empty database returns explicit `null` bounds, never epoch 0.
    pub fn temporal_extent(&self, req: TemporalExtentRequest) -> String {
        Self::extract_text(self.handle_temporal_extent(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Query the upstream derivation lineage of a fact (Issue #3371): "what
    /// was this fact derived from?", transitively — the evidence chain.
    pub fn lineage_upstream(&self, req: crate::mcp::tools::LineageQueryRequest) -> String {
        Self::extract_text(self.handle_lineage_upstream(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Query the downstream derivation lineage of a fact (Issue #3371): "what
    /// has been derived from this fact?", transitively — the blast radius.
    pub fn lineage_downstream(&self, req: crate::mcp::tools::LineageQueryRequest) -> String {
        Self::extract_text(self.handle_lineage_downstream(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Get a holistic database statistics snapshot.
    ///
    /// Returns current graph size, bi-temporal depth (version counts and
    /// anchor/delta compression), storage-tier presence/distribution
    /// (hot / warm-cache / cold), and WAL durability state in a single call.
    /// Thin aggregator over [`AletheiaDB::stats`] — all values are O(1)/cached
    /// counter reads, never a version scan.
    pub fn database_stats(&self, req: DatabaseStatsRequest) -> String {
        // Native/MCP path: derive the per-principal breakdown visibility from
        // the MCP session principal (anonymous mode is fully privileged →
        // included). The HTTP route authenticates independently and must call
        // `database_stats_with_visibility` with its own principal's role.
        self.database_stats_with_visibility(req, self.caller_is_admin())
    }

    /// `database_stats` with an explicit decision on whether the changefeed
    /// per-principal identity breakdown is included (Issue #3678).
    ///
    /// The HTTP `GET /database_stats` route authenticates independently of the
    /// MCP session, so it computes admin-ness from its own
    /// `Authorized<MetricsClass>` principal and threads it here. A non-admin
    /// caller receives the scalar aggregates but not the per-principal roster.
    #[must_use]
    pub fn database_stats_with_visibility(
        &self,
        req: DatabaseStatsRequest,
        include_per_principal: bool,
    ) -> String {
        Self::extract_text(self.handle_database_stats(
            serde_json::to_value(req).expect("request serialization should not fail"),
            include_per_principal,
        ))
    }

    /// Whether the current MCP session principal may see admin-gated details
    /// (Issue #3678). Anonymous mode has no session principal and is fully
    /// privileged, so it returns `true`; an authenticated session returns
    /// `true` only for the `admin` role.
    fn caller_is_admin(&self) -> bool {
        self.session_principal()
            .is_none_or(|p| p.role.allows(crate::auth::AccessClass::Admin))
    }

    /// Test-only access to the raw `database_stats` handler, bypassing typed
    /// request construction so tests can exercise wire-level argument edge
    /// cases (null arguments, non-object arguments, unknown keys) exactly as
    /// `call_tool` delivers them. Admin-visible (per-principal breakdown
    /// included) so the argument-edge assertions are unaffected by the #3678
    /// gate.
    #[cfg(test)]
    pub(crate) fn database_stats_raw(&self, args: serde_json::Value) -> String {
        Self::extract_text(self.handle_database_stats(args, true))
    }

    /// List graph-wide changes within a transaction-time window.
    ///
    /// Enumerates the nodes and edges whose versions were committed in `[tx_from, tx_to)`,
    /// with optional valid-time and label filtering and stable cursor pagination. This is the
    /// discovery primitive an agent reaches for to answer "what changed since `<time>`?"
    /// without already knowing any entity IDs.
    pub fn list_changes(&self, req: ListChangesRequest) -> String {
        Self::extract_text(self.handle_list_changes(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Long-poll for the next committed changes (the push changefeed's blocking
    /// surface, Issue #3375).
    ///
    /// Stateless per call: subscribe to the push feed, optionally catch up from a
    /// prior `from_token` via [`list_changes`](Self::list_changes), and otherwise
    /// block up to `timeout_ms` (default 25000, hard cap 60000) for the next
    /// matching commit. A timeout returns an empty `changes` array with
    /// `timed_out: true` and a `resume_token` to poll again; a lagged
    /// subscription returns a retriable `RESOURCE_EXHAUSTED` carrying the
    /// `resume_token` to resume losslessly via `list_changes`.
    ///
    /// This tool is deliberately excluded from the #3368 per-read timeout /
    /// #3353 token-budget / #3360 cursor wrappers — a long-poll is expected to
    /// block, so those cross-cutting read features would truncate or abort it.
    pub fn await_changes(&self, req: AwaitChangesRequest) -> String {
        self.await_changes_for_principal(req, None)
    }

    /// Long-poll for changes on behalf of an explicit principal (Issue #3678).
    ///
    /// Used by the HTTP `/changes/await` projection, which authenticates the
    /// caller at the route and threads that principal id here so the per-principal
    /// changefeed quota is enforced on the HTTP surface too (the native stdio path
    /// resolves the principal from the MCP session instead, via
    /// [`session_principal`](Self::session_principal)). Passing `None` falls back
    /// to the session principal (or the shared `"anonymous"` bucket).
    pub fn await_changes_for_principal(
        &self,
        req: AwaitChangesRequest,
        principal_key: Option<String>,
    ) -> String {
        Self::extract_text(self.handle_await_changes(
            serde_json::to_value(req).expect("request serialization should not fail"),
            principal_key,
        ))
    }

    /// Event-driven async counterpart to [`await_changes_for_principal`]
    /// (Issue #3673).
    ///
    /// Waits on the changefeed via the `recv_async` Notify path, so the long-poll
    /// pins **no** Tokio worker for its duration and releases its per-principal
    /// slot immediately when the caller's future is dropped (HTTP client
    /// disconnect). Used by the HTTP `/changes/await` projection; the native MCP
    /// `call_tool` seam routes through [`dispatch_await_changes_async`](Self::dispatch_await_changes_async)
    /// directly. Behavior (delivery / resume / timeout / lagged semantics) is
    /// identical to the sync entry — only the wait mechanism differs.
    pub async fn await_changes_for_principal_async(
        &self,
        req: AwaitChangesRequest,
        principal_key: Option<String>,
    ) -> String {
        Self::extract_text(
            self.dispatch_await_changes_async(
                serde_json::to_value(req).expect("request serialization should not fail"),
                principal_key,
            )
            .await,
        )
    }

    /// Execute a hybrid query.
    ///
    /// Combines **graph traversal**, **vector similarity**, and **temporal filtering** into a single query.
    /// This is the most powerful tool for "reasoning" about data.
    ///
    /// # Capabilities
    ///
    /// - **Start**: From a specific node (`start_node_id`) OR by vector search (`query_embedding`).
    /// - **Traverse**: Follow edges (`traverse_edge`) up to `traverse_depth`.
    /// - **Filter**: By time (`valid_time`) or label (`filter_label`).
    /// - **Rank**: Re-rank results by vector similarity (`query_embedding`).
    ///
    /// # Example Scenarios
    ///
    /// 1. **"Find similar documents written by Alice"**
    ///    - Start at Alice (`start_node_id`)
    ///    - Traverse `WROTE` edges
    ///    - Rank by similarity to `query_embedding`
    ///
    /// 2. **"Find papers about AI published last year"**
    ///    - Vector search (`query_embedding`)
    ///    - Filter by `valid_time`
    ///
    /// # Output Format
    ///
    /// Returns a list of hybrid results containing the node, optional score, path, and timestamp.
    ///
    /// ```json
    /// {
    ///   "results": [
    ///     {
    ///       "node": { "id": 10, "label": "Document", "properties": {...} },
    ///       "similarity_score": 0.92,
    ///       "traversal_path": [1, 5, 10],
    ///       "timestamp": "2023-01-01T12:00:00Z"
    ///     }
    ///   ],
    ///   "count": 1
    /// }
    /// ```
    pub fn hybrid_query(&self, req: HybridQueryRequest) -> String {
        Self::extract_text(self.handle_hybrid_query(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    /// Execute a read-only declarative query (Cypher or AQL).
    ///
    /// Returns the engine's structured rows plus column metadata as JSON, or a
    /// structured `{error:{kind,code,retriable,message,clause?,language}}`
    /// payload (`kind` per the query tool's own contract, `code`/`retriable`
    /// per the uniform Issue #3234 error contract). Mutating statements are
    /// rejected before execution and never write.
    pub fn query(&self, req: QueryRequest) -> String {
        Self::extract_text(self.handle_query(
            serde_json::to_value(req).expect("request serialization should not fail"),
        ))
    }

    // ========================================================================
    // Helper methods for converting between AletheiaDB and MCP types
    // ========================================================================

    fn interned_to_string(&self, interned: crate::core::InternedString) -> String {
        GLOBAL_INTERNER.resolve_or_else(interned, || format!("<unknown:{}>", interned.as_u32()))
    }

    /// Look up the provenance bundle and temporal bounds for the *exact*
    /// version already captured in `version_id` (a `Node`/`Edge` snapshot's
    /// `current_version`), rather than re-resolving "whichever version is
    /// current now". This keeps the returned metadata consistent with the
    /// properties already read from that same snapshot, even under a
    /// concurrent write -- and makes the bounds correct for point-in-time
    /// reads, which set `current_version` to the matched historical version.
    ///
    /// A lookup failure (e.g. a corrupted or unreachable cold-storage record)
    /// is logged rather than silently treated as "no metadata": that
    /// distinction matters because an MCP caller has no other way to tell
    /// the two cases apart. Best-effort here (returning `None`s rather than
    /// failing the whole response) is deliberate: a single-node lookup or a
    /// bulk endpoint like `list_nodes`/`traverse` should not fail entirely
    /// because one entry's metadata couldn't be read.
    ///
    /// `now` is the request-scoped wallclock captured once per tool call
    /// (Issue #3391), so every entity in one response evaluates `is_current`
    /// against the same instant.
    fn lookup_node_read_metadata(
        &self,
        version_id: VersionId,
        now: Timestamp,
    ) -> (Option<Provenance>, Option<TemporalBounds>) {
        match self.db.get_node_version_read_metadata(version_id) {
            Ok(Some((provenance, interval))) => (
                provenance,
                Some(TemporalBounds::from_interval_at(&interval, now)),
            ),
            Ok(None) => {
                eprintln!(
                    "Warning: node version {} not found in any tier; omitting provenance/temporal",
                    version_id.as_u64()
                );
                (None, None)
            }
            Err(e) => {
                eprintln!(
                    "Warning: failed to load metadata for node version {}: {}",
                    version_id.as_u64(),
                    e
                );
                (None, None)
            }
        }
    }

    /// Edge counterpart of [`lookup_node_read_metadata`](Self::lookup_node_read_metadata).
    fn lookup_edge_read_metadata(
        &self,
        version_id: VersionId,
        now: Timestamp,
    ) -> (Option<Provenance>, Option<TemporalBounds>) {
        match self.db.get_edge_version_read_metadata(version_id) {
            Ok(Some((provenance, interval))) => (
                provenance,
                Some(TemporalBounds::from_interval_at(&interval, now)),
            ),
            Ok(None) => {
                eprintln!(
                    "Warning: edge version {} not found in any tier; omitting provenance/temporal",
                    version_id.as_u64()
                );
                (None, None)
            }
            Err(e) => {
                eprintln!(
                    "Warning: failed to load metadata for edge version {}: {}",
                    version_id.as_u64(),
                    e
                );
                (None, None)
            }
        }
    }

    fn node_to_response(
        &self,
        node: &crate::core::Node,
        include_vectors: bool,
        now: Timestamp,
    ) -> NodeResponse {
        let (provenance, temporal) = self.lookup_node_read_metadata(node.current_version, now);
        NodeResponse {
            id: node.id.as_u64(),
            label: self.interned_to_string(node.label),
            properties: self.property_map_to_json(&node.properties, include_vectors),
            namespace: Self::namespace_field(node.namespace()),
            provenance,
            temporal,
        }
    }

    /// Render an entity's namespace as the first-class response field
    /// (Issue #3349): `None` for the implicit `default` namespace — so a
    /// single-agent (`default`-only) response stays byte-identical to
    /// pre-namespace behavior — and `Some(name)` for any non-default namespace.
    fn namespace_field(ns: crate::core::namespace::Namespace) -> Option<String> {
        if ns.is_default() {
            None
        } else {
            Some(ns.into_string())
        }
    }

    fn edge_to_response(
        &self,
        edge: &crate::core::Edge,
        include_vectors: bool,
        now: Timestamp,
    ) -> EdgeResponse {
        let (provenance, temporal) = self.lookup_edge_read_metadata(edge.current_version, now);
        EdgeResponse {
            id: edge.id.as_u64(),
            source_id: edge.source.as_u64(),
            target_id: edge.target.as_u64(),
            label: self.interned_to_string(edge.label),
            properties: self.property_map_to_json(&edge.properties, include_vectors),
            namespace: Self::namespace_field(edge.namespace()),
            provenance,
            temporal,
        }
    }

    /// Test-only accessor for the private JSON serializer (crypto-shred PR-1b
    /// erased-marker test lives in `db::crypto_shred::integration_tests`).
    #[cfg(test)]
    pub(crate) fn property_map_to_json_for_test(
        &self,
        props: &PropertyMap,
        include_vectors: bool,
    ) -> HashMap<String, serde_json::Value> {
        self.property_map_to_json(props, include_vectors)
    }

    fn property_map_to_json(
        &self,
        props: &PropertyMap,
        include_vectors: bool,
    ) -> HashMap<String, serde_json::Value> {
        let mut result = HashMap::new();
        for (key, value) in props.iter() {
            let key_str = self.interned_to_string(*key);
            // Elide engine-reserved ride-along keys (the namespace marker,
            // #3349, and crypto-shred markers) — they are surfaced as
            // first-class fields, never as user properties.
            if crate::core::namespace::is_reserved_property_key(&key_str) {
                continue;
            }
            result.insert(key_str, self.property_value_to_json(value, include_vectors));
        }
        result
    }

    fn property_value_to_json(
        &self,
        value: &PropertyValue,
        include_vectors: bool,
    ) -> serde_json::Value {
        match value {
            PropertyValue::Null => serde_json::Value::Null,
            PropertyValue::Bool(b) => serde_json::Value::Bool(*b),
            PropertyValue::Int(i) => json!(*i),
            PropertyValue::Float(f) => json!(*f),
            PropertyValue::String(s) => serde_json::Value::String(s.to_string()),
            PropertyValue::Bytes(b) => {
                // GDPR crypto-shred (Issue #3359, PR-1b): a sealed `SUBJ` envelope
                // that reaches the MCP funnel belongs to a known erasure subject.
                // Active-subject values are already unsealed to plaintext at the db
                // read boundary, so a surviving envelope is (almost always) erased —
                // render the erased descriptor (analogous to the #3220 vector-elision
                // shape) instead of leaking opaque ciphertext as base64. A `Bytes`
                // value whose subject the keyring does not recognize is treated as
                // ordinary user bytes.
                #[cfg(feature = "audit-export")]
                if let Some(descriptor) = self.sealed_value_descriptor(b) {
                    return descriptor;
                }
                serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(b))
            }
            PropertyValue::Array(arr) => serde_json::Value::Array(
                arr.iter()
                    .map(|v| self.property_value_to_json(v, include_vectors))
                    .collect(),
            ),
            PropertyValue::Vector(v) => {
                if include_vectors {
                    serde_json::Value::Array(
                        v.iter()
                            .map(|&f| serde_json::Value::from(f as f64))
                            .collect(),
                    )
                } else {
                    json!({"type": "vector", "dim": v.len(), "elided": true})
                }
            }
            PropertyValue::SparseVector(sv) => {
                if include_vectors {
                    json!({
                        "indices": sv.indices(),
                        "values": sv.values()
                    })
                } else {
                    json!({
                        "type": "sparse_vector",
                        "dim": sv.dimension(),
                        "nnz": sv.nnz(),
                        "elided": true
                    })
                }
            }
        }
    }

    /// If `bytes` is a sealed crypto-shred `SUBJ` envelope for a subject the
    /// keyring recognizes, return its MCP descriptor (Issue #3359, PR-1b);
    /// otherwise `None` (render as ordinary bytes). An erased subject yields
    /// `{"type":"sealed","erased":true,"subject_id":<id>}`; an active subject
    /// that somehow reached here un-unsealed yields `erased:false` — never the
    /// plaintext and never the raw ciphertext.
    #[cfg(feature = "audit-export")]
    fn sealed_value_descriptor(&self, bytes: &[u8]) -> Option<serde_json::Value> {
        use crate::db::crypto_shred::envelope;
        if !envelope::is_envelope(bytes) {
            return None;
        }
        let header = envelope::parse_header(bytes).ok()?;
        let erased = self.db.crypto_shred.is_erased(&header.subject_id);
        let active = self.db.crypto_shred.is_active(&header.subject_id);
        if !erased && !active {
            // Not a subject we know — treat as ordinary user bytes.
            return None;
        }
        Some(json!({
            "type": "sealed",
            "erased": erased,
            "subject_id": header.subject_id,
        }))
    }

    pub(crate) fn json_to_property_map(
        &self,
        json: &HashMap<String, serde_json::Value>,
    ) -> Result<PropertyMap, crate::core::error::Error> {
        let mut builder = PropertyMapBuilder::new();
        for (key, value) in json {
            if let Some(pv) = self.json_to_property_value(value) {
                // `try_insert` interns the property KEY. A capacity exhaustion
                // there is a TYPED `StorageError::CapacityExceeded` that must
                // survive to the call site so it renders as an actionable
                // FAILED_PRECONDITION (configurable interner cap), NOT a generic
                // INVALID_ARGUMENT. Propagate the typed error verbatim.
                builder = builder.try_insert(key.as_str(), pv)?;
            }
        }
        Ok(builder.build())
    }

    /// Render a property-map construction error to a `CallToolResult`.
    ///
    /// A string-interner capacity exhaustion (a property-KEY interning breach,
    /// configurable interner cap) routes through [`Self::db_error`] so it
    /// surfaces as an actionable `FAILED_PRECONDITION` naming
    /// `persistence.max_interned_strings` with structured
    /// `{resource, current, limit}` details — the same envelope the node-LABEL
    /// path already produces. Every other property error (bad value, recursion
    /// depth) stays a caller-fault `INVALID_ARGUMENT`, preserving prior behavior.
    pub(crate) fn property_map_error(&self, e: crate::core::error::Error) -> CallToolResult {
        if matches!(
            &e,
            crate::core::error::Error::Storage(
                crate::core::error::StorageError::CapacityExceeded { .. }
            )
        ) {
            self.db_error(e)
        } else {
            self.invalid_argument(&format!("Invalid properties: {}", e))
        }
    }

    fn json_to_property_value(&self, value: &serde_json::Value) -> Option<PropertyValue> {
        match value {
            serde_json::Value::Null => Some(PropertyValue::Null),
            serde_json::Value::Bool(b) => Some(PropertyValue::Bool(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(PropertyValue::Int(i))
                } else {
                    n.as_f64().map(PropertyValue::Float)
                }
            }
            serde_json::Value::String(s) => Some(PropertyValue::String(Arc::from(s.as_str()))),
            serde_json::Value::Array(arr) => {
                if arr.iter().all(|v| v.is_number()) && !arr.is_empty() {
                    let floats: Vec<f32> = arr
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    if floats.len() == arr.len() {
                        return Some(PropertyValue::Vector(Arc::from(floats)));
                    }
                }
                let values: Vec<PropertyValue> = arr
                    .iter()
                    .filter_map(|v| self.json_to_property_value(v))
                    .collect();
                Some(PropertyValue::Array(Arc::new(values)))
            }
            serde_json::Value::Object(_) => None,
        }
    }

    pub(crate) fn parse_timestamp(&self, s: &str) -> Result<Timestamp, String> {
        // Try parsing as ISO 8601 timestamp first
        if let Ok(dt) = s.parse::<DateTime<Utc>>() {
            let micros = dt.timestamp_micros();
            return Ok(Timestamp::from(micros));
        }

        // Also try parsing ISO 8601 without timezone (assume UTC)
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
            let micros = dt.and_utc().timestamp_micros();
            return Ok(Timestamp::from(micros));
        }

        // Fall back to microseconds since epoch
        if let Ok(micros) = s.parse::<i64>() {
            return Ok(Timestamp::from(micros));
        }

        Err(format!(
            "Invalid timestamp format: '{}'. Expected ISO 8601 (e.g., '2024-01-15T10:00:00Z') or microseconds since epoch.",
            s
        ))
    }

    /// Parse an optional timestamp argument, returning a structured INVALID_ARGUMENT error result on a parse failure.
    ///
    /// Collapses the otherwise-duplicated "if present, parse, else None" handling for the
    /// changefeed's optional time bounds.
    // clippy::result_large_err: the Err is rmcp's `CallToolResult` (~176B), the
    // MCP error-response type carried across the whole tool surface; boxing it
    // would ripple through the entire MCP `Result` API and every `?` call site.
    // Cold error path. Allowed pending a deliberate Box refactor.
    #[allow(clippy::result_large_err)]
    fn parse_opt_timestamp(
        &self,
        label: &str,
        value: &Option<String>,
    ) -> std::result::Result<Option<Timestamp>, CallToolResult> {
        value
            .as_deref()
            .map(|s| self.parse_timestamp(s))
            .transpose()
            .map_err(|e| self.invalid_argument(&format!("Invalid {label}: {e}")))
    }

    /// Parse an optional namespace **write** argument (Issue #3349, PR3a),
    /// returning a structured `INVALID_ARGUMENT` error result on a malformed
    /// name. `None`/absent ⇒ the default namespace (unchanged behavior). A
    /// supplied `"default"` resolves to the default namespace (equivalent to
    /// omitting it).
    // clippy::result_large_err: Err is rmcp's `CallToolResult` (~176B); see
    // `parse_opt_timestamp`.
    #[allow(clippy::result_large_err)]
    fn parse_opt_namespace(
        &self,
        value: &Option<String>,
    ) -> std::result::Result<Option<crate::core::namespace::Namespace>, CallToolResult> {
        match value.as_deref() {
            None => Ok(None),
            Some(s) => match crate::core::namespace::Namespace::new(s) {
                Ok(ns) => Ok(Some(ns)),
                Err(e) => Err(self.db_error(crate::core::error::Error::Namespace(e))),
            },
        }
    }

    /// Reject an explicit namespace supplied to a **non-create** write
    /// (update / delete — Issue #3349, PR3a): a namespace is immutable after
    /// creation, so supplying one is `INVALID_ARGUMENT`
    /// ([`NamespaceError::Immutable`]) rather than a silent no-op. `None` ⇒ OK.
    // clippy::result_large_err: Err is rmcp's `CallToolResult`; see above.
    #[allow(clippy::result_large_err)]
    fn reject_namespace_on_write_update(
        &self,
        value: &Option<String>,
    ) -> std::result::Result<(), CallToolResult> {
        if value.is_some() {
            return Err(self.db_error(crate::core::error::Error::Namespace(
                crate::core::namespace::NamespaceError::Immutable,
            )));
        }
        Ok(())
    }

    /// Parse an optional namespace **read scope** argument (Issue #3349, PR3a).
    ///
    /// Accepts three JSON shapes:
    /// - a **string** namespace name (single-namespace scope), or the special
    ///   selector `"all"` (no filter);
    /// - an **array of string** namespace names (the union scope);
    /// - absent ⇒ `Ok(None)`, meaning "no explicit scope supplied". Each scoped
    ///   read tool resolves this to [`NamespaceScope::default`] — the `default`
    ///   namespace ONLY (isolated-by-default, #3349 FIX2) — via
    ///   `.unwrap_or_default()`, then routes through its `*_scoped` path. This is
    ///   exact back-compat for any pre-namespace database (all data is `default`,
    ///   so an omitted read returns all of it) and only isolates once non-default
    ///   namespaces exist. It is therefore equivalent to an explicit
    ///   `"default"`, and distinct from `"all"` (which imposes no filter). The
    ///   non-scoping tools (`list_edges`/`hybrid_query`/`query`) instead treat an
    ///   omitted/`"all"` scope as a no-op via [`reject_unsupported_scope`].
    ///
    /// An **empty array** is `INVALID_ARGUMENT` (a scope that silently matches
    /// nothing is forbidden). A malformed namespace name is `INVALID_ARGUMENT`
    /// with `details.namespace`. An unknown (unregistered) namespace is not
    /// rejected here — it surfaces as `NOT_FOUND` when the scoped read validates
    /// the scope against the registry (`details.namespace`).
    // clippy::result_large_err: Err is rmcp's `CallToolResult`; see above.
    #[allow(clippy::result_large_err)]
    fn parse_opt_scope(
        &self,
        value: &Option<serde_json::Value>,
    ) -> std::result::Result<Option<crate::core::namespace::NamespaceScope>, CallToolResult> {
        use crate::core::error::Error;
        use crate::core::namespace::{Namespace, NamespaceScope};
        let Some(value) = value else {
            return Ok(None);
        };
        // An explicit JSON null is treated as "not supplied".
        if value.is_null() {
            return Ok(None);
        }
        let scope = match value {
            serde_json::Value::String(s) if s == "all" => NamespaceScope::All,
            serde_json::Value::String(s) => match Namespace::new(s.as_str()) {
                Ok(ns) => NamespaceScope::Single(ns),
                Err(e) => return Err(self.db_error(Error::Namespace(e))),
            },
            serde_json::Value::Array(arr) => {
                let mut namespaces = Vec::with_capacity(arr.len());
                for el in arr {
                    let Some(name) = el.as_str() else {
                        return Err(
                            self.invalid_argument("namespace scope array elements must be strings")
                        );
                    };
                    match Namespace::new(name) {
                        Ok(ns) => namespaces.push(ns),
                        Err(e) => return Err(self.db_error(Error::Namespace(e))),
                    }
                }
                // Empty list ⇒ INVALID_ARGUMENT (never-silently-match-nothing).
                match NamespaceScope::list(namespaces) {
                    Ok(scope) => scope,
                    Err(e) => return Err(self.db_error(Error::Namespace(e))),
                }
            }
            _ => {
                return Err(self.invalid_argument(
                    "namespace must be a string, an array of strings, or \"all\"",
                ));
            }
        };
        Ok(Some(scope))
    }

    /// For a read tool that does **not** yet support a namespace scope in v1
    /// (`list_edges`, `hybrid_query`, `query`): accept an absent scope or the
    /// no-op `"all"` selector (both leave behavior unchanged), but reject any
    /// narrowing scope with a structured `INVALID_ARGUMENT` — never a silently
    /// unscoped result (never-silently-wrong). A malformed / empty scope still
    /// surfaces its own `parse_opt_scope` error.
    // clippy::result_large_err: Err is rmcp's `CallToolResult`; see above.
    #[allow(clippy::result_large_err)]
    fn reject_unsupported_scope(
        &self,
        value: &Option<serde_json::Value>,
        message: &str,
    ) -> std::result::Result<(), CallToolResult> {
        match self.parse_opt_scope(value)? {
            None | Some(crate::core::namespace::NamespaceScope::All) => Ok(()),
            Some(_) => Err(self.invalid_argument(message)),
        }
    }

    /// Resolve a pair of independently-optional `as_of_valid_time` /
    /// `as_of_transaction_time` request fields into a single bi-temporal
    /// coordinate, shared by every MCP tool that supports point-in-time
    /// queries (`get_schema`, `traverse`, ...).
    ///
    /// Returns `None` when both fields are absent (current-state / no
    /// temporal filtering). When either field is supplied, the other
    /// defaults to the current time -- e.g. `as_of_valid_time` alone answers
    /// "using everything recorded so far, what was valid at this instant"
    /// (transaction_time defaults to now), matching the Rust API's
    /// `get_node_at_valid_time`/`get_node_at_transaction_time` convenience
    /// methods, which default the unspecified dimension the same way.
    // clippy::result_large_err: Err is rmcp's `CallToolResult` (~176B); see
    // `parse_opt_timestamp` — boxing would cascade through the MCP `Result` API.
    #[allow(clippy::result_large_err)]
    fn resolve_bitemporal_as_of(
        &self,
        as_of_valid_time: &Option<String>,
        as_of_transaction_time: &Option<String>,
    ) -> std::result::Result<Option<(Timestamp, Timestamp)>, CallToolResult> {
        let valid_time = self.parse_opt_timestamp("as_of_valid_time", as_of_valid_time)?;
        let transaction_time =
            self.parse_opt_timestamp("as_of_transaction_time", as_of_transaction_time)?;
        Ok(match (valid_time, transaction_time) {
            (None, None) => None,
            (vt, tt) => Some((vt.unwrap_or_else(time::now), tt.unwrap_or_else(time::now))),
        })
    }

    /// Validate and convert an optional MCP [`ProvenanceRequest`] into a
    /// core [`Provenance`](crate::core::provenance::Provenance), stamping
    /// the authenticated session principal (Issue #3350).
    ///
    /// Mirrors [`parse_opt_timestamp`](Self::parse_opt_timestamp): returns
    /// `Err(invalid_argument(...))` with a clear message when `confidence` is out
    /// of `[0.0, 1.0]` (Issue #3224), rather than a generic deserialization
    /// error. An entirely empty bundle (all fields omitted) is normalized to
    /// `None` -- never persisted as a fabricated empty object.
    ///
    /// **Principal stamping**: when the session has a verified principal
    /// (see [`session_principal`](Self::session_principal)), its *name* is
    /// recorded as the bundle's `principal` field -- composing with (never
    /// replacing) whatever `source`/`confidence`/`note`/`correlation_id`
    /// the caller supplied. A write with no caller-supplied provenance
    /// still records a principal-only bundle. The principal is
    /// server-stamped from the verified credential; [`ProvenanceRequest`]
    /// deliberately has no `principal` field, so callers cannot forge it.
    /// Anonymous-mode sessions (and the embedded `new()` constructor)
    /// record no principal -- the field is absent, not an empty string.
    // clippy::result_large_err: Err is rmcp's `CallToolResult` (~176B); see
    // `parse_opt_timestamp` — boxing would cascade through the MCP `Result` API.
    #[allow(clippy::result_large_err)]
    pub(crate) fn parse_opt_provenance(
        &self,
        value: Option<crate::mcp::tools::ProvenanceRequest>,
    ) -> std::result::Result<Option<Provenance>, CallToolResult> {
        let principal = self.session_principal().map(|p| p.name);
        let supplied = match value {
            Some(req) => {
                let provenance = Provenance::from_parts(
                    req.source,
                    req.confidence,
                    req.note,
                    req.correlation_id,
                    None,
                )
                .map_err(|e| {
                    self.invalid_argument(&format!(
                        "Invalid provenance: confidence must be between 0.0 and 1.0 ({e})"
                    ))
                })?;
                // Normalize an all-empty caller bundle away *before*
                // stamping, so "caller sent {}" and "caller sent nothing"
                // behave identically.
                if provenance.is_empty() {
                    None
                } else {
                    Some(provenance)
                }
            }
            None => None,
        };
        Ok(match (supplied, principal) {
            (Some(p), Some(name)) => Some(p.with_principal(name)),
            (Some(p), None) => Some(p),
            // A principal-only bundle cannot fail validation (only
            // `confidence` is validated, and it is unset here); `.ok()`
            // keeps this non-panicking regardless.
            (None, Some(name)) => Provenance::builder().principal(name).build().ok(),
            (None, None) => None,
        })
    }

    /// Parse the optional provenance filter parameters (Issue #3348) off the
    /// raw tool arguments, returning `Ok(None)` when no filter is active
    /// (byte-identical legacy behavior) or a validated [`ProvenanceFilter`].
    ///
    /// Reading straight off the raw `Value` (rather than a typed struct field)
    /// lets the offset path and the snapshot-anchored cursor path (#3360)
    /// share one parser, and keeps the filter available on tools whose cursor
    /// handlers never build the typed request. The four recognized keys are
    /// `provenance_source` (scalar, exact), `provenance_sources` (array,
    /// any-of), `min_confidence` (inclusive lower bound in `[0,1]`), and
    /// `include_unattributed` (bool).
    ///
    /// Invalid values fail closed with a structured `INVALID_ARGUMENT` whose
    /// `details.field` names the offending parameter (AC5) — never a silent
    /// empty result.
    // clippy::result_large_err: Err is rmcp's `CallToolResult` (~176B); see
    // `parse_opt_timestamp` — boxing would cascade through the MCP `Result` API.
    #[allow(clippy::result_large_err)]
    pub(crate) fn parse_provenance_filter(
        &self,
        args: &serde_json::Value,
    ) -> std::result::Result<Option<ProvenanceFilter>, CallToolResult> {
        let Some(obj) = args.as_object() else {
            return Ok(None);
        };

        // Type-check each key up front so a wrong JSON type is a clear
        // caller-fault error rather than a silently-ignored filter.
        let provenance_source = match obj.get("provenance_source") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(_) => {
                return Err(self.provenance_filter_type_error("provenance_source", "a string"));
            }
        };

        let provenance_sources = match obj.get("provenance_sources") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Array(arr)) => {
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    match item {
                        serde_json::Value::String(s) => out.push(s.clone()),
                        _ => {
                            return Err(self.provenance_filter_type_error(
                                "provenance_sources",
                                "an array of strings",
                            ));
                        }
                    }
                }
                Some(out)
            }
            Some(_) => {
                return Err(
                    self.provenance_filter_type_error("provenance_sources", "an array of strings")
                );
            }
        };

        let min_confidence = match obj.get("min_confidence") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Number(n)) => match n.as_f64() {
                Some(f) => Some(f),
                None => {
                    return Err(self
                        .provenance_filter_type_error("min_confidence", "a number in [0.0, 1.0]"));
                }
            },
            Some(_) => {
                return Err(
                    self.provenance_filter_type_error("min_confidence", "a number in [0.0, 1.0]")
                );
            }
        };

        let include_unattributed = match obj.get("include_unattributed") {
            None | Some(serde_json::Value::Null) => false,
            Some(serde_json::Value::Bool(b)) => *b,
            Some(_) => {
                return Err(self.provenance_filter_type_error("include_unattributed", "a boolean"));
            }
        };

        ProvenanceFilter::validated(
            provenance_source,
            provenance_sources,
            min_confidence,
            include_unattributed,
        )
        .map_err(|e| {
            self.error_result(
                McpError::new(McpErrorCode::InvalidArgument, e.to_string())
                    .details(json!({ "field": e.field() })),
            )
        })
    }

    /// Structured `INVALID_ARGUMENT` for a wrong-typed provenance-filter
    /// parameter (Issue #3348), naming the field so the caller can repair it.
    fn provenance_filter_type_error(&self, field: &str, expected: &str) -> CallToolResult {
        self.error_result(
            McpError::new(
                McpErrorCode::InvalidArgument,
                format!("Invalid '{field}': expected {expected}"),
            )
            .details(json!({ "field": field })),
        )
    }

    /// Parse the optional Issue #3372 `fusion_policy` object into a validated
    /// [`FusionPolicy`](crate::db::fusion::FusionPolicy). Returns `Ok(None)`
    /// when the param is absent/null (unchanged behavior). A wrong JSON type or
    /// an invalid weight/parameter maps to `INVALID_ARGUMENT` (#3234,
    /// `retriable:false`) — `FusionPolicyError` is not a `db::Error`, so it is
    /// mapped directly here.
    // clippy::result_large_err: Err is rmcp's `CallToolResult`; see
    // `parse_provenance_filter`.
    #[cfg(feature = "semantic-retrieval-fusion")]
    #[allow(clippy::result_large_err)]
    pub(crate) fn parse_fusion_policy(
        &self,
        args: &serde_json::Value,
    ) -> std::result::Result<Option<crate::db::fusion::FusionPolicy>, CallToolResult> {
        let Some(obj) = args.as_object() else {
            return Ok(None);
        };
        let policy_val = match obj.get("fusion_policy") {
            None | Some(serde_json::Value::Null) => return Ok(None),
            Some(serde_json::Value::Object(m)) => m,
            Some(_) => {
                return Err(self.invalid_argument(
                    "Invalid 'fusion_policy': expected an object with optional numeric weights",
                ));
            }
        };

        // Read an optional finite f64 field, rejecting a wrong JSON type here so
        // the caller gets a field-named INVALID_ARGUMENT rather than a silently
        // ignored setting.
        let read_num = |key: &str| -> std::result::Result<Option<f64>, CallToolResult> {
            match policy_val.get(key) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(serde_json::Value::Number(n)) => Ok(Some(n.as_f64().unwrap_or(f64::NAN))),
                Some(_) => Err(self.invalid_argument(&format!(
                    "Invalid 'fusion_policy.{key}': expected a number"
                ))),
            }
        };

        let mut builder = crate::db::fusion::FusionPolicy::builder();
        if let Some(v) = read_num("w_similarity")? {
            builder = builder.w_similarity(v);
        }
        if let Some(v) = read_num("w_confidence")? {
            builder = builder.w_confidence(v);
        }
        if let Some(v) = read_num("w_recency")? {
            builder = builder.w_recency(v);
        }
        if let Some(v) = read_num("neutral_confidence")? {
            builder = builder.neutral_confidence(v);
        }
        if let Some(v) = read_num("recency_half_life_secs")? {
            builder = builder.recency_half_life_secs(v);
        }

        builder.build().map(Some).map_err(|e| {
            self.error_result(
                McpError::new(McpErrorCode::InvalidArgument, e.to_string())
                    .details(json!({ "param": "fusion_policy" })),
            )
        })
    }

    /// Serialize a [`FusionBreakdown`](crate::db::fusion::FusionBreakdown) into
    /// the per-result `score_breakdown` JSON (Issue #3372).
    #[cfg(feature = "semantic-retrieval-fusion")]
    fn fusion_breakdown_to_json(b: &crate::db::fusion::FusionBreakdown) -> serde_json::Value {
        json!({
            "similarity": b.similarity,
            "confidence": b.confidence,
            "confidence_defaulted": b.confidence_defaulted,
            "recency": b.recency,
            "recency_defaulted": b.recency_defaulted,
            "fused": b.fused,
        })
    }

    /// Parse a single MCP [`LineageRefRequest`] into a core
    /// [`LineageRef`](crate::core::lineage::LineageRef) (Issue #3371).
    ///
    /// Validates `entity_kind` (`"node"`/`"edge"`, case-insensitive) and that
    /// the id and version are in range. Structural validity only — whether the
    /// version actually *exists* is checked by the write path
    /// (`validate_sources`) so a dangling reference becomes a `NOT_FOUND`
    /// rather than an `INVALID_ARGUMENT`.
    // clippy::result_large_err: Err is rmcp's `CallToolResult` (~176B); see
    // `parse_opt_timestamp` — boxing would cascade through the MCP `Result` API.
    #[allow(clippy::result_large_err)]
    fn parse_lineage_ref(
        &self,
        req: &crate::mcp::tools::LineageRefRequest,
    ) -> std::result::Result<crate::core::lineage::LineageRef, CallToolResult> {
        let version = VersionId::new(req.version)
            .map_err(|e| self.invalid_argument(&format!("Invalid derived_from version: {e}")))?;
        let entity = match req.entity_kind.trim().to_ascii_lowercase().as_str() {
            "node" => crate::core::id::EntityId::Node(NodeId::new(req.id).map_err(|e| {
                self.invalid_argument(&format!("Invalid derived_from node id: {e}"))
            })?),
            "edge" => crate::core::id::EntityId::Edge(EdgeId::new(req.id).map_err(|e| {
                self.invalid_argument(&format!("Invalid derived_from edge id: {e}"))
            })?),
            other => {
                return Err(self.invalid_argument(&format!(
                    "Invalid derived_from entity_kind '{other}': expected 'node' or 'edge'"
                )));
            }
        };
        Ok(crate::core::lineage::LineageRef { entity, version })
    }

    /// Parse the optional `derived_from` list on a write request into core
    /// [`LineageRef`](crate::core::lineage::LineageRef)s (Issue #3371). `None`
    /// or an empty list yields an empty vec (no lineage recorded).
    // clippy::result_large_err: Err is rmcp's `CallToolResult` (~176B); see
    // `parse_opt_timestamp` — boxing would cascade through the MCP `Result` API.
    // Also covers the inner `map(|r| self.parse_lineage_ref(r))` closure.
    #[allow(clippy::result_large_err)]
    fn parse_derived_from(
        &self,
        value: &Option<Vec<crate::mcp::tools::LineageRefRequest>>,
    ) -> std::result::Result<Vec<crate::core::lineage::LineageRef>, CallToolResult> {
        match value {
            None => Ok(Vec::new()),
            Some(list) => list.iter().map(|r| self.parse_lineage_ref(r)).collect(),
        }
    }

    /// Parse an optional transaction time, returning the current time if not specified.
    fn parse_optional_tx_time(&self, tx_time: Option<&str>) -> Result<Timestamp, String> {
        match tx_time {
            Some(tx) => self.parse_timestamp(tx),
            None => Ok(time::now()),
        }
    }

    /// Format transaction time for response, using the constant for current time.
    fn format_tx_time_response(tx_time: Option<String>) -> String {
        tx_time.unwrap_or_else(|| TRANSACTION_TIME_NOW.to_string())
    }

    /// Serialize a timestamp as an RFC 3339 `Z`-suffixed microsecond-precision
    /// JSON string — the interval-bound convention used by retraction
    /// responses (half-open `[start, end)`; `null` would denote an
    /// open-ended bound). Delegates to
    /// [`format_timestamp_rfc3339`](Self::format_timestamp_rfc3339) so all
    /// MCP tools share one formatter.
    fn timestamp_to_rfc3339_micros(ts: Timestamp) -> serde_json::Value {
        json!(Self::format_timestamp_rfc3339(ts))
    }

    /// Format a resolved bi-temporal coordinate as an RFC 3339 UTC string
    /// with microsecond precision and a `Z` suffix (e.g.
    /// `2024-01-15T10:00:00.000000Z`).
    ///
    /// Total: a wallclock value outside chrono's representable range degrades
    /// to the raw microsecond count rather than silently rendering the 1970
    /// epoch.
    fn format_timestamp_rfc3339(ts: Timestamp) -> String {
        match DateTime::<Utc>::from_timestamp_micros(ts.wallclock()) {
            Some(dt) => dt.to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
            None => ts.wallclock().to_string(),
        }
    }

    fn matches_label(&self, interned: crate::core::InternedString, label: &str) -> bool {
        GLOBAL_INTERNER
            .resolve_with(interned, |s| s == label)
            .unwrap_or(false)
    }

    /// Get the expected dimensions for a vector index property.
    /// Returns None if the index doesn't exist.
    fn get_vector_index_dimensions(&self, property_name: &str) -> Option<usize> {
        self.db
            .list_vector_indexes()
            .into_iter()
            .find(|info| info.property_name == property_name)
            .map(|info| info.dimensions)
    }

    /// Validate embedding dimensions against the expected dimensions for an index.
    fn validate_embedding_dimensions(
        &self,
        embedding: &[f32],
        property_name: &str,
    ) -> Result<(), String> {
        if let Some(expected_dims) = self.get_vector_index_dimensions(property_name)
            && embedding.len() != expected_dims
        {
            return Err(format!(
                "Embedding dimension mismatch: expected {} dimensions for property '{}', got {}",
                expected_dims,
                property_name,
                embedding.len()
            ));
        }
        Ok(())
    }

    pub(crate) fn success_json(&self, value: serde_json::Value) -> CallToolResult {
        CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
        )])
    }

    /// Serialize a structured [`McpError`] into an error `CallToolResult`
    /// with the Issue #3234 shape:
    /// `{"error": {"code", "message", "retriable", "details"?}}`.
    pub(crate) fn error_result(&self, err: McpError) -> CallToolResult {
        Self::error_result_with_top_level(err, serde_json::Map::new())
    }

    /// Like [`error_result`](Self::error_result), but preserving additional
    /// legacy top-level fields alongside the structured `error` object (e.g.
    /// the #3209 DETACH refusal's `connected_edges`), so pre-#3234 consumers
    /// lose nothing.
    pub(crate) fn error_result_with_top_level(
        err: McpError,
        mut top_level: serde_json::Map<String, serde_json::Value>,
    ) -> CallToolResult {
        top_level.insert("error".to_string(), err.to_json());
        let value = serde_json::Value::Object(top_level);
        // Compact serialization, matching both the pre-#3234 error payloads
        // and the query tool's error path (success payloads stay pretty).
        CallToolResult::error(vec![Content::text(
            serde_json::to_string(&value).unwrap_or_else(|_| value.to_string()),
        )])
    }

    /// Caller-fault error: malformed arguments, bad IDs, unparseable
    /// timestamps, inconsistent parameter combinations. Never retriable.
    pub(crate) fn invalid_argument(&self, msg: &str) -> CallToolResult {
        self.error_result(McpError::new(McpErrorCode::InvalidArgument, msg))
    }

    /// The request is well-formed but the system is not in the required
    /// state (e.g. a vector index is not enabled). Never retriable as-is.
    fn failed_precondition(&self, msg: &str) -> CallToolResult {
        self.error_result(McpError::new(McpErrorCode::FailedPrecondition, msg))
    }

    /// Classify an internal database error into the structured MCP shape.
    ///
    /// A `UniqueViolation` additionally carries its constraint metadata under
    /// `error.details` and (for backward compatibility) as legacy top-level
    /// fields, exactly as before #3234.
    fn db_error(&self, e: impl Into<crate::core::error::Error>) -> CallToolResult {
        let e = e.into();
        if let Some(crate::core::error::ConstraintError::UniqueViolation {
            label,
            property,
            value,
            existing_node_id,
        }) = e.as_constraint()
        {
            let details = json!({
                "label": label,
                "property": property,
                "value": value,
                "existing_node_id": existing_node_id.as_u64()
            });
            let mut top_level = serde_json::Map::new();
            top_level.insert("success".to_string(), json!(false));
            top_level.insert("constraint_violation".to_string(), json!(true));
            top_level.insert("label".to_string(), json!(label));
            top_level.insert("property".to_string(), json!(property));
            top_level.insert("value".to_string(), json!(value));
            top_level.insert(
                "existing_node_id".to_string(),
                json!(existing_node_id.as_u64()),
            );
            return Self::error_result_with_top_level(
                McpError::from_db_error(&e).details(details),
                top_level,
            );
        }
        self.error_result(McpError::from_db_error(&e))
    }

    /// Attach result-completeness signals (Issue #3226) to a bounded read
    /// tool's response object so a single MCP call reveals whether the returned
    /// page is the whole truth.
    ///
    /// - `has_more` is always set: `true` when at least one matching result
    ///   exists beyond the returned page, `false` otherwise.
    /// - `next_offset` (the value to pass as `offset` for the next page) is set
    ///   only when `has_more` is `true`; it is omitted otherwise.
    /// - `total_matching` is set only when a matching total is cheaply known;
    ///   it is omitted (never faked) when computing it would require an
    ///   expensive full scan.
    ///
    /// `consumed` is the number of underlying candidates the page advanced
    /// past -- i.e. the requested page window (e.g. `limit` or `k`), NOT the
    /// number of items actually returned in the body. These differ when an
    /// id resolved from an index (property lookup, vector search) turns out
    /// to be a since-deleted node and is dropped: the page still consumed a
    /// full window of candidates, so `next_offset` must advance by the window
    /// size, not the smaller returned count, or the next call would re-skip
    /// into already-consumed candidates and duplicate a row across pages.
    fn attach_completeness(
        value: &mut serde_json::Value,
        offset: usize,
        consumed: usize,
        has_more: bool,
        total_matching: Option<usize>,
    ) {
        if let Some(obj) = value.as_object_mut() {
            obj.insert("has_more".to_string(), json!(has_more));
            if has_more {
                obj.insert(
                    "next_offset".to_string(),
                    json!(offset.saturating_add(consumed)),
                );
            }
            if let Some(total) = total_matching {
                obj.insert("total_matching".to_string(), json!(total));
            }
        }
    }

    // ========================================================================
    // Snapshot-anchored cursor continuation (Issue #3360)
    // ========================================================================

    /// Whether the raw arguments opt into cursor paging -- either a `cursor`
    /// continuation token or `use_cursor: true` on the first page. The cursor
    /// parameters are additive and read directly off the arguments so the
    /// per-tool request structs (and their many struct-literal call sites)
    /// stay unchanged.
    fn cursor_requested(args: &serde_json::Value) -> bool {
        args.get("cursor").and_then(|v| v.as_str()).is_some()
            || args.get("use_cursor").and_then(|v| v.as_bool()) == Some(true)
    }

    /// Whether the raw arguments carry any (non-null) Issue #3348 provenance
    /// filter parameter. Used to fail closed when a filter is combined with a
    /// path that does not yet honor it (v1: cursor paging), rather than
    /// silently returning unfiltered results.
    fn has_provenance_filter_args(args: &serde_json::Value) -> bool {
        let Some(obj) = args.as_object() else {
            return false;
        };
        ["provenance_source", "provenance_sources", "min_confidence"]
            .iter()
            .any(|k| matches!(obj.get(*k), Some(v) if !v.is_null()))
    }

    /// Fail-closed error for combining a provenance filter with cursor paging
    /// (Issue #3348 v1): the snapshot-anchored cursor token does not yet carry
    /// the filter, so rather than silently ignore it, reject with guidance to
    /// use offset paging.
    fn provenance_filter_cursor_unsupported(&self) -> CallToolResult {
        self.error_result(
            McpError::new(
                McpErrorCode::InvalidArgument,
                "A provenance filter is not supported together with cursor paging \
                 (use_cursor/cursor) in this version; use offset paging (limit/offset) \
                 to combine provenance filtering with pagination.",
            )
            .details(json!({ "field": "cursor" })),
        )
    }

    /// Read an optional non-empty string argument.
    fn arg_str(args: &serde_json::Value, key: &str) -> Option<String> {
        args.get(key).and_then(|v| v.as_str()).map(str::to_string)
    }

    /// Read an optional non-null JSON argument (e.g. a `property_value` that
    /// may be a string, number, or bool).
    fn arg_value(args: &serde_json::Value, key: &str) -> Option<serde_json::Value> {
        match args.get(key) {
            None | Some(serde_json::Value::Null) => None,
            Some(v) => Some(v.clone()),
        }
    }

    /// Read and clamp the per-page `limit`, matching the bounded read tools'
    /// convention (default 100, at least 1, capped at `MAX_RESULT_LIMIT`).
    fn arg_limit(args: &serde_json::Value) -> usize {
        args.get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .clamp(1, MAX_RESULT_LIMIT)
    }

    /// Emit one snapshot-anchored keyset page over a set of node candidates
    /// (shared by `list_nodes` and `find_nodes_at_time` in cursor mode).
    ///
    /// `candidates` MUST be sorted ascending by node id (both underlying `db`
    /// finders guarantee this) and reconstructed at `snapshot`. The page
    /// returns the first `limit` candidates whose id is strictly greater than
    /// `after` -- a keyset filter that avoids re-emitting prior result pages (no
    /// dup/gap). Candidate enumeration is still O(total) per page in v1 (the
    /// full candidate scan re-runs each page); a true depth-independent keyset
    /// seek is a follow-up. When more candidates remain, a continuation `cursor` is
    /// issued carrying the same pinned `snapshot`, so the union of all pages
    /// equals exactly the unbounded result at that one bi-temporal moment.
    ///
    /// `parent_cid` is the cursor id of the page being resumed (empty for the
    /// first page): passing it through keeps the whole scan on one registry
    /// slot instead of consuming the live-cursor cap once per page.
    #[allow(clippy::too_many_arguments)]
    fn emit_node_cursor_page(
        &self,
        tool: &str,
        snapshot: (Timestamp, Timestamp),
        after: Option<u64>,
        limit: usize,
        include_vectors: bool,
        filters: serde_json::Value,
        candidates: crate::db::ops::NodesAtTime,
        parent_cid: String,
        now: Timestamp,
    ) -> CallToolResult {
        let sampled = candidates.sampled;
        let nodes = candidates.nodes;
        let total_matching = nodes.len();

        // Keyset seek: everything strictly past the last id returned so far
        // (`after == None` on the first page returns from the beginning).
        let past = |id: u64| after.is_none_or(|a| id > a);
        let remaining = nodes.iter().filter(|n| past(n.id.as_u64())).count();
        let page: Vec<&crate::core::Node> = nodes
            .iter()
            .filter(|n| past(n.id.as_u64()))
            .take(limit)
            .collect();
        let has_more = remaining > page.len();
        let last_id = page.last().map(|n| n.id.as_u64());

        let responses: Vec<NodeResponse> = page
            .iter()
            .map(|n| self.node_to_response(n, include_vectors, now))
            .collect();

        let mut obj = serde_json::Map::new();
        obj.insert(
            "nodes".to_string(),
            serde_json::to_value(&responses).unwrap_or_else(|_| json!([])),
        );
        obj.insert("count".to_string(), json!(responses.len()));
        obj.insert("total_matching".to_string(), json!(total_matching));
        obj.insert("sampled".to_string(), json!(sampled));
        obj.insert("has_more".to_string(), json!(has_more));
        // Disclose the pinned snapshot so the caller can see exactly which
        // bi-temporal moment the whole scan is answering at.
        obj.insert(
            "snapshot_valid_time".to_string(),
            json!(Self::format_timestamp_rfc3339(snapshot.0)),
        );
        obj.insert(
            "snapshot_transaction_time".to_string(),
            json!(Self::format_timestamp_rfc3339(snapshot.1)),
        );
        obj.insert("paging".to_string(), json!("cursor"));

        if has_more {
            // `last_id` is Some whenever the page is non-empty; a full page
            // with more remaining always has a last id.
            if let Some(last_id) = last_id {
                let mut payload = CursorPayload::seed(
                    tool,
                    (snapshot.0.wallclock(), snapshot.1.wallclock()),
                    limit,
                    filters,
                );
                // Continuation key = last id ACTUALLY EMITTED on this page.
                // When #3353 token budgets land and can trim a page short of
                // `limit`, `after` MUST still be derived from the last row that
                // survived the trim (not the limit-th candidate), or the next
                // page would skip the trimmed-off rows.
                payload.after = Some(last_id);
                payload.cid = parent_cid;
                match self.cursors.issue(payload) {
                    Ok(token) => {
                        obj.insert("cursor".to_string(), json!(token));
                        obj.insert(
                            "cursor_ttl_seconds".to_string(),
                            json!(self.cursors.ttl().as_secs()),
                        );
                    }
                    Err(e) => return self.error_result(e),
                }
            }
        }

        self.success_json(serde_json::Value::Object(obj))
    }

    /// Fetch the node candidate set for a cursored scan at `snapshot`,
    /// applying the same optional exact-property filter `list_nodes` /
    /// `find_nodes_at_time` support. Returns a structured error result on a
    /// bad property value.
    // clippy::result_large_err: Err is rmcp's `CallToolResult` (~176B); see
    // `parse_opt_timestamp` — boxing would cascade through the MCP `Result` API.
    #[allow(clippy::result_large_err)]
    fn fetch_node_candidates(
        &self,
        label: &str,
        property_key: &Option<String>,
        property_value: &Option<serde_json::Value>,
        snapshot: (Timestamp, Timestamp),
    ) -> Result<crate::db::ops::NodesAtTime, CallToolResult> {
        let (vt, tt) = snapshot;
        match (property_key, property_value) {
            (Some(key), Some(val)) => {
                let pv = match self.json_to_property_value(val) {
                    Some(v) => v,
                    None => {
                        return Err(self.invalid_argument(
                            "Unsupported property_value type. Use strings, numbers, booleans, or null.",
                        ));
                    }
                };
                self.db
                    .find_nodes_by_property_at(label, key, &pv, vt, tt)
                    .map_err(|e| self.db_error(e))
            }
            _ => self
                .db
                .find_nodes_at_time(label, vt, tt)
                .map_err(|e| self.db_error(e)),
        }
    }

    /// Build the opaque `filters` blob embedded in a node-scan cursor so a
    /// continuation reconstructs the exact same query with no extra params.
    fn node_scan_filters(
        label: &str,
        property_key: &Option<String>,
        property_value: &Option<serde_json::Value>,
        include_vectors: bool,
    ) -> serde_json::Value {
        json!({
            "label": label,
            "property_key": property_key,
            "property_value": property_value,
            "include_vectors": include_vectors,
        })
    }

    /// Emit one snapshot-anchored keyset page over a node's adjacency (shared
    /// by `get_outgoing_edges` / `get_incoming_edges` in cursor mode), ordered
    /// by edge id. The adjacency is read as of the pinned snapshot via the
    /// bi-temporal `get_*_edges_at_time` path, so paging is consistent under
    /// concurrent writes exactly like the node scans.
    #[allow(clippy::too_many_arguments)]
    fn emit_adjacency_cursor_page(
        &self,
        tool: &str,
        node_id: NodeId,
        incoming: bool,
        label: &Option<String>,
        snapshot: (Timestamp, Timestamp),
        after: Option<u64>,
        limit: usize,
        include_vectors: bool,
        parent_cid: String,
        now: Timestamp,
    ) -> CallToolResult {
        let (vt, tt) = snapshot;
        let edge_ids = if incoming {
            self.db.get_incoming_edges_at_time(node_id, vt, tt)
        } else {
            self.db.get_outgoing_edges_at_time(node_id, vt, tt)
        };

        // Resolve each candidate edge once as of the snapshot, applying the
        // optional label filter, then order by edge id for a stable keyset.
        let mut resolved: Vec<crate::core::Edge> = edge_ids
            .into_iter()
            .filter_map(|eid| self.get_edge_maybe_at(eid, Some((vt, tt))).ok())
            .filter(|e| {
                label
                    .as_ref()
                    .map(|l| self.matches_label(e.label, l))
                    .unwrap_or(true)
            })
            .collect();
        resolved.sort_by_key(|e| e.id.as_u64());

        let past = |id: u64| after.is_none_or(|a| id > a);
        let total_matching = resolved.len();
        let remaining = resolved.iter().filter(|e| past(e.id.as_u64())).count();
        let page: Vec<&crate::core::Edge> = resolved
            .iter()
            .filter(|e| past(e.id.as_u64()))
            .take(limit)
            .collect();
        let has_more = remaining > page.len();
        let last_id = page.last().map(|e| e.id.as_u64());

        let responses: Vec<EdgeResponse> = page
            .iter()
            .map(|e| self.edge_to_response(e, include_vectors, now))
            .collect();

        let mut obj = serde_json::Map::new();
        obj.insert(
            "edges".to_string(),
            serde_json::to_value(&responses).unwrap_or_else(|_| json!([])),
        );
        obj.insert("count".to_string(), json!(responses.len()));
        obj.insert("total_matching".to_string(), json!(total_matching));
        obj.insert("has_more".to_string(), json!(has_more));
        obj.insert(
            "snapshot_valid_time".to_string(),
            json!(Self::format_timestamp_rfc3339(snapshot.0)),
        );
        obj.insert(
            "snapshot_transaction_time".to_string(),
            json!(Self::format_timestamp_rfc3339(snapshot.1)),
        );
        obj.insert("paging".to_string(), json!("cursor"));

        if has_more && let Some(last_id) = last_id {
            let filters = json!({
                "node_id": node_id.as_u64(),
                "label": label,
                "incoming": incoming,
                "include_vectors": include_vectors,
            });
            let mut payload = CursorPayload::seed(
                tool,
                (snapshot.0.wallclock(), snapshot.1.wallclock()),
                limit,
                filters,
            );
            // Continuation key = last edge id ACTUALLY EMITTED on this page.
            // When #3353 token budgets land and can trim a page short of
            // `limit`, `after` MUST still be derived from the last row that
            // survived the trim (not the limit-th candidate), or the next page
            // would skip the trimmed-off edges.
            payload.after = Some(last_id);
            payload.cid = parent_cid;
            match self.cursors.issue(payload) {
                Ok(token) => {
                    obj.insert("cursor".to_string(), json!(token));
                    obj.insert(
                        "cursor_ttl_seconds".to_string(),
                        json!(self.cursors.ttl().as_secs()),
                    );
                }
                Err(e) => return self.error_result(e),
            }
        }

        self.success_json(serde_json::Value::Object(obj))
    }

    /// Cursor-mode dispatch shared by `get_outgoing_edges` /
    /// `get_incoming_edges` (Issue #3360). `incoming` selects the direction and
    /// the tool name the cursor is bound to.
    fn handle_adjacency_cursor(
        &self,
        tool: &str,
        incoming: bool,
        args: &serde_json::Value,
    ) -> CallToolResult {
        let now = time::now();

        if let Some(token) = args.get("cursor").and_then(|v| v.as_str()) {
            let payload = match self.cursors.decode(token, tool) {
                Ok(p) => p,
                Err(e) => return self.error_result(e),
            };
            let node_id = match NodeId::new(payload.filters["node_id"].as_u64().unwrap_or(0)) {
                Ok(id) => id,
                Err(e) => return self.invalid_argument(&e.to_string()),
            };
            let label = payload.filters["label"].as_str().map(str::to_string);
            let include_vectors = payload.filters["include_vectors"]
                .as_bool()
                .unwrap_or(false);
            let snapshot = (Timestamp::from(payload.svt), Timestamp::from(payload.stt));
            return self.emit_adjacency_cursor_page(
                tool,
                node_id,
                incoming,
                &label,
                snapshot,
                payload.after,
                payload.limit,
                include_vectors,
                payload.cid,
                now,
            );
        }

        let node_id = match NodeId::new(args.get("node_id").and_then(|v| v.as_u64()).unwrap_or(0)) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        let label = Self::arg_str(args, "label");
        let include_vectors = args
            .get("include_vectors")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let limit = Self::arg_limit(args);
        let snapshot = (now, now);
        self.emit_adjacency_cursor_page(
            tool,
            node_id,
            incoming,
            &label,
            snapshot,
            None,
            limit,
            include_vectors,
            String::new(),
            now,
        )
    }

    // ========================================================================
    // Tool Implementations
    // ========================================================================

    fn handle_get_node(&self, args: serde_json::Value) -> CallToolResult {
        // Issue #3348: parse the optional provenance filter off the raw args
        // before consuming them into the typed request.
        let prov_filter = match self.parse_provenance_filter(&args) {
            Ok(f) => f,
            Err(result) => return result,
        };

        let req: GetNodeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        // Issue #3349: a namespace scope narrows visibility (an out-of-scope
        // node is reported NOT_FOUND, indistinguishable from missing). Omitted ⇒
        // the `default` namespace only (isolated-by-default, #3349 FIX2), routed
        // through the scoped read; `all` imposes no filter (byte-identical
        // unscoped fast path). For a pre-namespace database (all data is
        // `default`) an omitted read still returns all of it.
        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();
        let node_result = match &scope {
            crate::core::namespace::NamespaceScope::All => self.db.get_node(node_id),
            scope => self.db.get_node_scoped(node_id, scope),
        };

        match node_result {
            Ok(node) => {
                let now = time::now();
                let response =
                    self.node_to_response(&node, req.include_vectors.unwrap_or(false), now);
                // Issue #3348: when a provenance filter is supplied, a node
                // that exists but does not satisfy it is not in the filtered
                // view. Surface an honest, non-retriable NOT_FOUND (never a
                // fabricated node), naming the constraint via details.
                if let Some(filter) = &prov_filter
                    && !filter.matches(response.provenance.as_ref())
                {
                    return self.error_result(
                        McpError::new(
                            McpErrorCode::NotFound,
                            format!(
                                "Node {} exists but does not satisfy the provenance filter",
                                node_id.as_u64()
                            ),
                        )
                        .details(json!({
                            "node_id": node_id.as_u64(),
                            "reason": "provenance_filter_excluded"
                        })),
                    );
                }
                self.success_json(
                    serde_json::to_value(&response)
                        .expect("response serialization should not fail"),
                )
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_create_node(&self, args: serde_json::Value) -> CallToolResult {
        let req: CreateNodeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let properties = match req.properties {
            Some(p) => match self.json_to_property_map(&p) {
                Ok(map) => map,
                Err(e) => return self.property_map_error(e),
            },
            None => PropertyMap::default(),
        };

        let valid_from = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v,
            Err(result) => return result,
        };
        let provenance = match self.parse_opt_provenance(req.provenance) {
            Ok(p) => p,
            Err(result) => return result,
        };

        let derived_from = match self.parse_derived_from(&req.derived_from) {
            Ok(refs) => refs,
            Err(result) => return result,
        };

        let namespace = match self.parse_opt_namespace(&req.namespace) {
            Ok(ns) => ns,
            Err(result) => return result,
        };

        let mut options = crate::api::transaction::WriteRequestOptions::new();
        if let Some(valid_from) = valid_from {
            options = options.with_valid_from(valid_from);
        }
        if let Some(provenance) = provenance {
            options = options.with_provenance(provenance);
        }
        if let Some(ns) = namespace.clone() {
            options = options.with_namespace(ns);
        }

        let created = if derived_from.is_empty() {
            self.db
                .create_node_with_options(&req.label, properties, options)
        } else {
            self.db
                .create_node_with_options_and_lineage(
                    &req.label,
                    properties,
                    options,
                    &derived_from,
                )
                .map(|(node_id, _version)| node_id)
        };

        // Auto-register the (non-default) namespace only AFTER a successful
        // commit, mirroring `create_node_in_namespace` (a failed write must not
        // leave a durable phantom namespace).
        if let (Ok(_), Some(ns)) = (&created, &namespace)
            && let Err(e) = self.db.register_namespace_on_write(ns)
        {
            return self.db_error(e);
        }

        match created {
            Ok(node_id) => match self.db.get_node(node_id) {
                Ok(node) => {
                    let now = time::now();
                    let response = self.node_to_response(&node, true, now);
                    self.success_json(
                        serde_json::to_value(&response)
                            .expect("response serialization should not fail"),
                    )
                }
                Err(e) => self.db_error(e),
            },
            Err(e) => self.db_error(e),
        }
    }

    fn handle_update_node(&self, args: serde_json::Value) -> CallToolResult {
        let req: UpdateNodeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        // A namespace is immutable after creation (Issue #3349): supplying one
        // on an update is INVALID_ARGUMENT, never a silent no-op.
        if let Err(result) = self.reject_namespace_on_write_update(&req.namespace) {
            return result;
        }

        let properties = match self.json_to_property_map(&req.properties) {
            Ok(map) => map,
            Err(e) => return self.property_map_error(e),
        };

        let valid_from = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v,
            Err(result) => return result,
        };
        let provenance = match self.parse_opt_provenance(req.provenance) {
            Ok(p) => p,
            Err(result) => return result,
        };

        let derived_from = match self.parse_derived_from(&req.derived_from) {
            Ok(refs) => refs,
            Err(result) => return result,
        };

        let mut options = crate::api::transaction::WriteRequestOptions::new();
        if let Some(valid_from) = valid_from {
            options = options.with_valid_from(valid_from);
        }
        if let Some(provenance) = provenance {
            options = options.with_provenance(provenance);
        }

        let updated = if derived_from.is_empty() {
            self.db
                .update_node_with_options(node_id, properties, options)
        } else {
            self.db
                .update_node_with_options_and_lineage(node_id, properties, options, &derived_from)
                .map(|_version| ())
        };

        match updated {
            Ok(()) => match self.db.get_node(node_id) {
                Ok(node) => {
                    let now = time::now();
                    let response = self.node_to_response(&node, true, now);
                    self.success_json(
                        serde_json::to_value(&response)
                            .expect("response serialization should not fail"),
                    )
                }
                Err(e) => self.db_error(e),
            },
            Err(e) => self.db_error(e),
        }
    }

    fn handle_delete_node(&self, args: serde_json::Value) -> CallToolResult {
        let req: DeleteNodeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        // A namespace is immutable after creation (Issue #3349): supplying one
        // on a delete is INVALID_ARGUMENT, never a silent no-op.
        if let Err(result) = self.reject_namespace_on_write_update(&req.namespace) {
            return result;
        }

        let detach = req.detach.unwrap_or(false);

        let valid_from = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v,
            Err(result) => return result,
        };

        if detach && valid_from.is_some() {
            return self.invalid_argument(
                "valid_time is not supported together with detach:true; cascade delete does \
                 not support backdating. Delete the connected edges individually with \
                 valid_time, or omit valid_time to cascade-delete at now.",
            );
        }

        // Issue #3427: stamp the authenticated session principal onto the
        // tombstone version's provenance. These destructive tools take NO
        // caller-supplied provenance (pass `None`), so the bundle is
        // principal-only and unforgeable from request fields — the principal
        // is server-derived by `parse_opt_provenance`, never read from the
        // request JSON. Anonymous sessions yield `None` (no principal field).
        let provenance = match self.parse_opt_provenance(None) {
            Ok(p) => p,
            Err(result) => return result,
        };
        let mut options = crate::api::transaction::WriteRequestOptions::new();
        if let Some(valid_from) = valid_from {
            options = options.with_valid_from(valid_from);
        }
        if let Some(provenance) = provenance {
            options = options.with_provenance(provenance);
        }

        // Perform the connected-edge check and the deletion inside a single write
        // transaction so they observe the same storage state. Splitting the count
        // into a separate transaction (or doing it before opening one) leaves a
        // check-then-act gap in which a concurrent writer could add an edge after
        // the count but before the delete, silently orphaning it. Keeping both in
        // one closure removes that cross-transaction gap (Issue #3209).
        enum Outcome {
            Refused { connected_edges: usize },
            Deleted { edges_removed: usize },
        }

        let outcome = self.db.write(|tx| -> crate::core::error::Result<Outcome> {
            // `count_connected_edges` reads the same `current` storage the
            // transaction's edge traversal uses, and also verifies the node
            // exists (errors propagate via `?`).
            let connected_edges = self.db.count_connected_edges(node_id)?;

            // Refuse-by-default: never report a bare success while silently
            // orphaning edges. The caller must opt into destruction via `detach`.
            if connected_edges > 0 && !detach {
                return Ok(Outcome::Refused { connected_edges });
            }

            if detach {
                // Cascade-equivalent delete: remove the node and all connected
                // edges, reporting exactly how many edges were removed. The
                // options (principal provenance) stamp every co-deleted edge
                // tombstone too, not just the node (Issue #3427).
                tx.delete_node_cascade_with_options(node_id, options.clone())?;
                Ok(Outcome::Deleted {
                    edges_removed: connected_edges,
                })
            } else {
                // No connected edges: a plain delete cannot orphan anything.
                // `options` carries the backdated valid_from (if any) and the
                // acting principal's provenance (Issue #3427).
                tx.delete_node_with_options(node_id, options.clone())?;
                Ok(Outcome::Deleted { edges_removed: 0 })
            }
        });

        let outcome = match outcome {
            Ok(o) => o,
            Err(e) => return self.db_error(e),
        };

        match outcome {
            // The #3209 refusal in the #3234 structured shape:
            // `FAILED_PRECONDITION` with `details.connected_edges`, while the
            // legacy top-level fields are preserved additively (no loss).
            Outcome::Refused { connected_edges } => {
                let message = format!(
                    "Node {} has {} connected edge(s); refusing to delete. \
                     Pass `detach: true` to delete the node and its connected edges, \
                     or remove the edges first.",
                    req.node_id, connected_edges
                );
                let mut top_level = serde_json::Map::new();
                top_level.insert("success".to_string(), json!(false));
                top_level.insert("node_id".to_string(), json!(req.node_id));
                top_level.insert("connected_edges".to_string(), json!(connected_edges));
                top_level.insert("detach_required".to_string(), json!(true));
                Self::error_result_with_top_level(
                    McpError::new(McpErrorCode::FailedPrecondition, message).details(json!({
                        "node_id": req.node_id,
                        "connected_edges": connected_edges,
                        "detach_required": true
                    })),
                    top_level,
                )
            }
            Outcome::Deleted { edges_removed } => self.success_json(json!({
                "success": true,
                "deleted_node_id": req.node_id,
                "detached": detach,
                "edges_removed": edges_removed
            })),
        }
    }

    fn handle_retract_node(&self, args: serde_json::Value) -> CallToolResult {
        let req: RetractNodeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (see handle_delete_node).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let detach = req.detach.unwrap_or(false);

        // Matching the #3221 convention: valid_time defaults to now.
        let valid_to = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v.unwrap_or_else(time::now),
            Err(result) => return result,
        };

        // Issue #3427: principal-only, unforgeable provenance (see
        // handle_delete_node) stamped onto the retraction version — and onto
        // every co-retracted edge in the detach branch below.
        let provenance = match self.parse_opt_provenance(None) {
            Ok(p) => p,
            Err(result) => return result,
        };

        // Perform the connected-edge check and the retraction inside a single
        // write transaction so they observe the same storage state — no
        // check-then-act gap for a concurrent writer to slip an edge into
        // (same rationale as handle_delete_node, Issue #3209).
        enum Outcome {
            Refused { connected_edges: usize },
            Retracted(crate::api::transaction::RetractionResult),
        }

        let outcome = self.db.write(|tx| -> crate::core::error::Result<Outcome> {
            use crate::api::transaction::ReadOps;

            // The connected-edge contract only applies to a currently-present
            // node; an already-retracted node short-circuits below to the
            // idempotent result (and a nonexistent one to NOT_FOUND). All
            // reads go through the transaction itself (buffer-aware,
            // snapshot-isolated) rather than back through `self.db`.
            if tx.get_node(node_id).is_ok() {
                // Enumerate DISTINCT connected edges once — the refusal
                // count and the detach co-retraction share the same
                // sort/dedup list, so `connected_edges` always equals what
                // `detach: true` would retract (a self-loop appears in both
                // adjacency directions but is one edge).
                let mut edge_ids = tx.get_outgoing_edges(node_id)?;
                edge_ids.extend(tx.get_incoming_edges(node_id)?);
                edge_ids.sort_unstable();
                edge_ids.dedup();
                let connected_edges = edge_ids.len();

                // Refuse-by-default: never report a bare success that leaves
                // edges pointing at a retracted node. The caller must opt
                // into co-retraction via `detach`.
                if connected_edges > 0 && !detach {
                    return Ok(Outcome::Refused { connected_edges });
                }

                if detach && connected_edges > 0 {
                    // Co-retract every connected edge at the same valid time,
                    // stamping the SAME acting principal onto each edge's
                    // retraction version as the node's (Issue #3427).
                    let mut edges_retracted = 0;
                    for edge_id in edge_ids {
                        let edge_result =
                            tx.retract_edge_with_provenance(edge_id, valid_to, provenance.clone())?;
                        if !edge_result.already_retracted {
                            edges_retracted += 1;
                        }
                    }

                    let mut result =
                        tx.retract_node_with_provenance(node_id, valid_to, provenance.clone())?;
                    result.edges_retracted = edges_retracted;
                    return Ok(Outcome::Retracted(result));
                }
            }

            Ok(Outcome::Retracted(tx.retract_node_with_provenance(
                node_id,
                valid_to,
                provenance.clone(),
            )?))
        });

        let outcome = match outcome {
            Ok(o) => o,
            Err(e) => return self.db_error(e),
        };

        match outcome {
            // The refusal in the #3234 structured shape: FAILED_PRECONDITION
            // with `details.connected_edges`, legacy top-level fields
            // preserved additively (byte-for-byte parallel to the
            // handle_delete_node refusal).
            Outcome::Refused { connected_edges } => {
                let message = format!(
                    "Node {} has {} connected edge(s); refusing to retract. \
                     Pass `detach: true` to retract the node and its connected edges \
                     at the same valid time, or retract the edges first.",
                    req.node_id, connected_edges
                );
                let mut top_level = serde_json::Map::new();
                top_level.insert("success".to_string(), json!(false));
                top_level.insert("node_id".to_string(), json!(req.node_id));
                top_level.insert("connected_edges".to_string(), json!(connected_edges));
                top_level.insert("detach_required".to_string(), json!(true));
                Self::error_result_with_top_level(
                    McpError::new(McpErrorCode::FailedPrecondition, message).details(json!({
                        "node_id": req.node_id,
                        "connected_edges": connected_edges,
                        "detach_required": true
                    })),
                    top_level,
                )
            }
            Outcome::Retracted(result) => self.success_json(json!({
                "success": true,
                "node_id": req.node_id,
                "retracted": true,
                "already_retracted": result.already_retracted,
                "valid_from": Self::timestamp_to_rfc3339_micros(result.valid_from),
                "valid_to": Self::timestamp_to_rfc3339_micros(result.valid_to),
                "edges_retracted": result.edges_retracted
            })),
        }
    }

    fn handle_retract_edge(&self, args: serde_json::Value) -> CallToolResult {
        let req: RetractEdgeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let edge_id = match EdgeId::new(req.edge_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (see handle_delete_node).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        // Matching the #3221 convention: valid_time defaults to now.
        let valid_to = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v.unwrap_or_else(time::now),
            Err(result) => return result,
        };

        // Issue #3427: principal-only, unforgeable provenance (see
        // handle_delete_node) stamped onto the retraction version.
        let provenance = match self.parse_opt_provenance(None) {
            Ok(p) => p,
            Err(result) => return result,
        };

        match self
            .db
            .retract_edge_with_provenance(edge_id, valid_to, provenance)
        {
            Ok(result) => self.success_json(json!({
                "success": true,
                "edge_id": req.edge_id,
                "retracted": true,
                "already_retracted": result.already_retracted,
                "valid_from": Self::timestamp_to_rfc3339_micros(result.valid_from),
                "valid_to": Self::timestamp_to_rfc3339_micros(result.valid_to)
            })),
            Err(e) => self.db_error(e),
        }
    }

    /// Map a [`CryptoShredError`] to the MCP #3234 error envelope via its stable
    /// `.code()` (Issue #3359, Slice 4b).
    ///
    /// `CryptoShredError` is NOT a `crate::core::error::Error` variant, so
    /// routing it through [`Self::db_error`] would collapse `INVALID_ARGUMENT` /
    /// `CONFLICT` to `INTERNAL` (its `From<..> for Error` only preserves
    /// `FAILED_PRECONDITION`). We map directly instead. Its `Display` never
    /// contains key bytes or plaintext (guaranteed by the error type), so the
    /// message is safe to surface verbatim.
    fn crypto_shred_error(&self, e: crate::db::CryptoShredError) -> CallToolResult {
        let code = match e.code() {
            "INVALID_ARGUMENT" => McpErrorCode::InvalidArgument,
            "FAILED_PRECONDITION" => McpErrorCode::FailedPrecondition,
            "CONFLICT" => McpErrorCode::Conflict,
            "NOT_FOUND" => McpErrorCode::NotFound,
            _ => McpErrorCode::Internal,
        };
        self.error_result(McpError::new(code, e.to_string()).retriable(e.retriable()))
    }

    /// ADMIN. Designate a GDPR erasure subject over one or more targets
    /// (Issue #3359, Slice 4b).
    fn handle_designate_subject(&self, args: serde_json::Value) -> CallToolResult {
        use crate::db::DesignationTarget;

        // DoS guard: reject an over-cap `targets` array up front, on the RAW
        // JSON value BEFORE `serde_json::from_value` deserializes/allocates the
        // typed Vec, mirroring the `apply_batch` cap (limit echoed per the #3226
        // convention).
        if let Some(arr) = args.get("targets").and_then(|t| t.as_array())
            && arr.len() > self.max_designate_targets
        {
            return self.error_result(
                McpError::new(
                    McpErrorCode::InvalidArgument,
                    format!(
                        "designate targets length {} exceeds maximum of {}",
                        arr.len(),
                        self.max_designate_targets
                    ),
                )
                .details(json!({
                    "limit": self.max_designate_targets,
                    "submitted": arr.len(),
                })),
            );
        }

        let req: DesignateSubjectRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let mut targets = Vec::with_capacity(req.targets.len());
        for t in req.targets {
            let has_keys = t.keys.as_ref().is_some_and(|k| !k.is_empty());
            let target = match (t.entity_kind.as_str(), has_keys) {
                ("node", false) => DesignationTarget::WholeNode(t.id),
                ("node", true) => DesignationTarget::NodeProperties(t.id, t.keys.unwrap()),
                ("edge", false) => DesignationTarget::WholeEdge(t.id),
                ("edge", true) => DesignationTarget::EdgeProperties(t.id, t.keys.unwrap()),
                (other, _) => {
                    return self.invalid_argument(&format!(
                        "invalid entity_kind '{other}' (expected 'node' or 'edge')"
                    ));
                }
            };
            targets.push(target);
        }

        let count = targets.len();
        match self.db.designate_subject(req.subject_id.clone(), targets) {
            Ok(()) => self.success_json(json!({
                "success": true,
                "subject_id": req.subject_id,
                "targets_designated": count,
            })),
            Err(e) => self.crypto_shred_error(e),
        }
    }

    /// ADMIN. Irreversibly erase a designated subject and return a signed
    /// erasure attestation (Issue #3359, Slice 4b). The response carries only
    /// the subject id, entity count, timestamp, and signature/pubkey hex —
    /// never key material or plaintext.
    fn handle_erase_subject(&self, args: serde_json::Value) -> CallToolResult {
        let req: EraseSubjectRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        match self.db.erase_subject(req.subject_id) {
            Ok(att) => {
                let timestamp =
                    Self::format_timestamp_rfc3339(Timestamp::from(att.timestamp_micros));
                self.success_json(json!({
                    "success": true,
                    "subject_id": att.subject_id,
                    "entity_count": att.entity_count,
                    "timestamp": timestamp,
                    "timestamp_micros": att.timestamp_micros,
                    "signature": crate::core::hex::encode(&att.signature),
                    "signer_public_key": att.signer_public_key.to_hex(),
                }))
            }
            Err(e) => self.crypto_shred_error(e),
        }
    }

    fn handle_delete_node_cascade(&self, args: serde_json::Value) -> CallToolResult {
        let req: DeleteNodeCascadeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        // Issue #3427: principal-only, unforgeable provenance (see
        // handle_delete_node) stamped onto the node's tombstone AND every
        // co-deleted edge's tombstone.
        let provenance = match self.parse_opt_provenance(None) {
            Ok(p) => p,
            Err(result) => return result,
        };
        let mut options = crate::api::transaction::WriteRequestOptions::new();
        if let Some(provenance) = provenance {
            options = options.with_provenance(provenance);
        }

        match self
            .db
            .write(|tx| tx.delete_node_cascade_with_options(node_id, options.clone()))
        {
            Ok(()) => self.success_json(json!({
                "success": true,
                "deleted_node_id": req.node_id,
                "cascade": true
            })),
            Err(e) => self.db_error(e),
        }
    }

    /// Cursor-mode `list_nodes` (Issue #3360): snapshot-anchored keyset paging.
    ///
    /// The scan is pinned to the transaction time captured on the first page
    /// and every page is reconstructed as of that coordinate via the same
    /// point-in-time machinery `find_nodes_at_time` uses, so concurrent writes
    /// after the anchor are invisible: the union of all pages equals exactly a
    /// single unbounded `list_nodes` at the anchor moment. Requires `label`
    /// (an unlabeled list has no enumerable, ordered candidate set).
    fn handle_list_nodes_cursor(&self, args: &serde_json::Value) -> CallToolResult {
        let now = time::now();

        // Resume path: everything discriminating is baked into the token.
        if let Some(token) = args.get("cursor").and_then(|v| v.as_str()) {
            let payload = match self.cursors.decode(token, "list_nodes") {
                Ok(p) => p,
                Err(e) => return self.error_result(e),
            };
            let label = payload.filters["label"].as_str().unwrap_or("").to_string();
            let property_key = payload.filters["property_key"].as_str().map(str::to_string);
            let property_value = match &payload.filters["property_value"] {
                serde_json::Value::Null => None,
                v => Some(v.clone()),
            };
            let include_vectors = payload.filters["include_vectors"]
                .as_bool()
                .unwrap_or(false);
            let snapshot = (Timestamp::from(payload.svt), Timestamp::from(payload.stt));
            let candidates = match self.fetch_node_candidates(
                &label,
                &property_key,
                &property_value,
                snapshot,
            ) {
                Ok(c) => c,
                Err(result) => return result,
            };
            return self.emit_node_cursor_page(
                "list_nodes",
                snapshot,
                payload.after,
                payload.limit,
                include_vectors,
                payload.filters.clone(),
                candidates,
                payload.cid,
                now,
            );
        }

        // First page: validate and pin the snapshot at "now".
        let property_key = Self::arg_str(args, "property_key");
        let property_value = Self::arg_value(args, "property_value");
        if property_key.is_some() != property_value.is_some() {
            return self.invalid_argument(
                "Both 'property_key' and 'property_value' are required together",
            );
        }
        let label = match Self::arg_str(args, "label") {
            Some(l) => l,
            None => {
                return self.invalid_argument(
                    "Cursor paging requires 'label' (an unlabeled node list has no ordered, \
                     enumerable candidate set to page over).",
                );
            }
        };
        let limit = Self::arg_limit(args);
        let include_vectors = args
            .get("include_vectors")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let snapshot = (now, now);
        let candidates =
            match self.fetch_node_candidates(&label, &property_key, &property_value, snapshot) {
                Ok(c) => c,
                Err(result) => return result,
            };
        let filters =
            Self::node_scan_filters(&label, &property_key, &property_value, include_vectors);
        self.emit_node_cursor_page(
            "list_nodes",
            snapshot,
            None,
            limit,
            include_vectors,
            filters,
            candidates,
            String::new(),
            now,
        )
    }

    fn handle_list_nodes(&self, args: serde_json::Value) -> CallToolResult {
        // Snapshot-anchored cursor paging (Issue #3360) is a distinct path
        // from the legacy offset paging below, which stays unchanged for
        // backward compatibility. The cursor parameters are additive and read
        // straight off the raw arguments (the request structs stay unchanged).
        if Self::cursor_requested(&args) {
            // Issue #3348: fail closed rather than silently ignore a filter
            // the cursor path does not yet honor.
            if Self::has_provenance_filter_args(&args) {
                return self.provenance_filter_cursor_unsupported();
            }
            // Issue #3349: a narrowing namespace scope does not yet compose with
            // the #3360 cursor path — fail closed rather than page an unscoped
            // scan under a scope the caller asked for.
            if let Err(result) = self.reject_unsupported_scope(
                &args.get("namespace").cloned(),
                "namespace scope does not compose with cursor paging (use_cursor) in v1; page the \
                 scoped scan with offset/limit instead, or use \"all\".",
            ) {
                return result;
            }
            return self.handle_list_nodes_cursor(&args);
        }

        // Issue #3348: parse the optional provenance filter before consuming
        // the raw args into the typed request.
        let prov_filter = match self.parse_provenance_filter(&args) {
            Ok(f) => f,
            Err(result) => return result,
        };

        let req: ListNodesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        // Apply resource limits. A page must be able to carry a continuation
        // cursor, so the limit is at least 1 (limit:0 would otherwise report
        // has_more:true with next_offset==offset, a non-progressing page that
        // traps a paginating caller in an infinite loop).
        let limit = req
            .limit
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .clamp(1, MAX_RESULT_LIMIT);
        let offset = req.offset.unwrap_or(0).min(MAX_PAGINATION_OFFSET);

        // Validate property filter: both key and value are required together with label
        if req.property_key.is_some() != req.property_value.is_some() {
            return self.invalid_argument(
                "Both 'property_key' and 'property_value' are required together",
            );
        }
        if req.property_key.is_some() && req.label.is_none() {
            return self.invalid_argument("Property filtering requires 'label' to be specified");
        }

        // Issue #3349 (FIX2): an omitted scope resolves to the `default`
        // namespace only (isolated-by-default) and routes through the scoped
        // membership index, exactly like an explicit narrowing scope. For a
        // pre-namespace database (all data is `default`) this still returns all
        // of it. Only the `all` selector falls through to the full-featured
        // unscoped path below (byte-identical to the current unscoped behavior).
        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();
        if !matches!(scope, crate::core::namespace::NamespaceScope::All) {
            return self.handle_list_nodes_scoped(&req, &scope, prov_filter.as_ref());
        }

        // One request-scoped wallclock for every entity in the response
        // (Issue #3391).
        let now = time::now();

        // Property-based lookup: label + property_key + property_value
        if let (Some(label), Some(prop_key), Some(prop_val)) =
            (&req.label, &req.property_key, &req.property_value)
        {
            let property_value =
                match self.json_to_property_value(prop_val) {
                    Some(v) => v,
                    None => return self.invalid_argument(
                        "Unsupported property_value type. Use strings, numbers, booleans, or null.",
                    ),
                };

            let node_ids = self
                .db
                .find_nodes_by_property(label, prop_key, &property_value);

            // The full matching id list is already materialized, so the total
            // is cheap to report and `has_more` is exact.
            let total_matching = node_ids.len();
            let include_vectors = req.include_vectors.unwrap_or(false);
            let mut nodes = Vec::with_capacity(limit);
            for node_id in node_ids.into_iter().skip(offset).take(limit) {
                if let Ok(node) = self.db.get_node(node_id) {
                    let resp = self.node_to_response(&node, include_vectors, now);
                    // Issue #3348: the provenance filter is applied per-page.
                    // Non-matching rows are dropped from this page while
                    // `offset`/`next_offset` still advance over the unfiltered
                    // candidate window, so paging stays gap-free/dup-free (a
                    // page may therefore hold fewer than `limit` rows).
                    if prov_filter
                        .as_ref()
                        .is_none_or(|f| f.matches(resp.provenance.as_ref()))
                    {
                        nodes.push(resp);
                    }
                }
            }

            // `has_more`/`next_offset` are derived from the requested window
            // (`limit`) against `total_matching`, not from `nodes.len()`: a
            // stale property-index entry pointing at a since-deleted node is
            // still one of the `limit` ids this page consumed, so basing
            // `next_offset` on the (possibly smaller) resolved count would
            // re-skip into already-consumed ids and duplicate a row on the
            // next page.
            let has_more = offset.saturating_add(limit) < total_matching;
            let mut response = json!({
                "nodes": nodes,
                "count": nodes.len(),
                "offset": offset,
                "limit": limit
            });
            // With a provenance filter active, `total_matching` would be the
            // *unfiltered* candidate count and therefore misleading, so it is
            // omitted; `has_more` still carries the completeness signal.
            let reported_total = if prov_filter.is_some() {
                None
            } else {
                Some(total_matching)
            };
            Self::attach_completeness(&mut response, offset, limit, has_more, reported_total);
            return self.success_json(response);
        }

        // Label-only scan
        if let Some(label) = &req.label {
            let builder = crate::query::QueryBuilder::new().scan_label(label);

            // Note: We fetch offset+limit rows then skip offset. We fetch one
            // extra row (offset+limit+1) purely to detect whether more matching
            // nodes exist beyond this page (`has_more`) without paying for a
            // full-scan count. Offset is capped to prevent excessive memory use.
            match builder.limit(limit + offset + 1).execute(&self.db) {
                Ok(results) => {
                    // Use iterator-based approach to avoid allocating full Vec
                    let include_vectors = req.include_vectors.unwrap_or(false);
                    let mut nodes = Vec::with_capacity(limit);
                    let mut skipped = 0;
                    let mut has_more = false;

                    // `page_window` counts rows consumed from the unfiltered
                    // scan for this page (so `next_offset` advances correctly
                    // even when the provenance filter drops some of them).
                    let mut page_window = 0usize;
                    for row_result in results {
                        match row_result {
                            Ok(row) => {
                                if skipped < offset {
                                    skipped += 1;
                                    continue;
                                }
                                if let EntityResult::Node(node) = row.entity {
                                    if page_window >= limit {
                                        // The extra (offset+limit+1)th matching
                                        // row proves more results remain.
                                        has_more = true;
                                        break;
                                    }
                                    page_window += 1;
                                    let resp = self.node_to_response(&node, include_vectors, now);
                                    // Issue #3348: per-page provenance filter.
                                    if prov_filter
                                        .as_ref()
                                        .is_none_or(|f| f.matches(resp.provenance.as_ref()))
                                    {
                                        nodes.push(resp);
                                    }
                                }
                            }
                            Err(e) => return self.db_error(e),
                        }
                    }

                    // A label scan cannot cheaply know the matching total
                    // (that needs a full scan), so `total_matching` is omitted;
                    // `has_more` alone carries the completeness signal.
                    // `next_offset` advances by the unfiltered `page_window`,
                    // not `nodes.len()`, so a provenance-filtered page never
                    // re-skips or duplicates rows on the next page.
                    let mut response = json!({
                        "nodes": nodes,
                        "count": nodes.len(),
                        "offset": offset,
                        "limit": limit
                    });
                    Self::attach_completeness(&mut response, offset, page_window, has_more, None);
                    self.success_json(response)
                }
                Err(e) => self.db_error(e),
            }
        } else {
            // Without a label filter, we cannot efficiently list all nodes
            // Return a helpful message
            let mut response = json!({
                "message": "Use 'label' filter to list nodes by type, or use 'count_nodes' for total count",
                "total_count": self.db.node_count(),
                "nodes": [],
                "count": 0,
                "offset": offset,
                "limit": limit
            });
            Self::attach_completeness(&mut response, offset, 0, false, None);
            self.success_json(response)
        }
    }

    /// Namespace-scoped `list_nodes` page (Issue #3349): candidates come from
    /// the membership index (not a full scan), filtered to the given
    /// (narrowing) `scope`, then offset/limit-paginated and rendered exactly
    /// like the unscoped path (temporal bounds, vector elision, first-class
    /// `namespace` field, provenance filter). Unlike the unscoped path this does
    /// not compose with the #3360 cursor or #3353 token-budget shaping in v1
    /// (offset paging applies).
    fn handle_list_nodes_scoped(
        &self,
        req: &ListNodesRequest,
        scope: &crate::core::namespace::NamespaceScope,
        prov_filter: Option<&crate::core::ProvenanceFilter>,
    ) -> CallToolResult {
        let limit = req
            .limit
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .clamp(1, MAX_RESULT_LIMIT);
        let offset = req.offset.unwrap_or(0).min(MAX_PAGINATION_OFFSET);
        let include_vectors = req.include_vectors.unwrap_or(false);
        let now = time::now();

        // No label filter: reproduce the unscoped path's guidance response
        // exactly (byte-identical back-compat for an omitted/default-scope read —
        // #3349 FIX2). The unscoped path declines to enumerate every node without
        // a label; we mirror that, reporting the **in-scope** node count as
        // `total_count` (for a pre-namespace / `default`-only database this
        // equals `node_count()`, so the response is byte-identical). `list_nodes_
        // scoped` also validates the scope, so an unknown namespace still surfaces
        // as NOT_FOUND(details.namespace) here.
        if req.label.is_none() && req.property_key.is_none() {
            let total = match self.db.list_nodes_scoped(None, scope) {
                Ok(ids) => ids.len(),
                Err(e) => return self.db_error(e),
            };
            let mut response = json!({
                "message": "Use 'label' filter to list nodes by type, or use 'count_nodes' for total count",
                "total_count": total,
                "nodes": [],
                "count": 0,
                "offset": offset,
                "limit": limit
            });
            Self::attach_completeness(&mut response, offset, 0, false, None);
            return self.success_json(response);
        }

        // Resolve the candidate id set within scope (validates the scope against
        // the registry — an unknown namespace ⇒ NOT_FOUND(details.namespace)).
        let ids_result = if let (Some(label), Some(prop_key), Some(prop_val)) =
            (&req.label, &req.property_key, &req.property_value)
        {
            let property_value = match self.json_to_property_value(prop_val) {
                Some(v) => v,
                None => {
                    return self.invalid_argument(
                        "Unsupported property_value type. Use strings, numbers, booleans, or null.",
                    );
                }
            };
            self.db
                .find_nodes_by_property_scoped(label, prop_key, &property_value, scope)
        } else {
            self.db.list_nodes_scoped(req.label.as_deref(), scope)
        };
        let node_ids = match ids_result {
            Ok(ids) => ids,
            Err(e) => return self.db_error(e),
        };

        let total_matching = node_ids.len();
        let mut nodes = Vec::with_capacity(limit.min(total_matching.saturating_sub(offset)));
        for node_id in node_ids.into_iter().skip(offset).take(limit) {
            if let Ok(node) = self.db.get_node(node_id) {
                let resp = self.node_to_response(&node, include_vectors, now);
                if prov_filter.is_none_or(|f| f.matches(resp.provenance.as_ref())) {
                    nodes.push(resp);
                }
            }
        }

        let has_more = offset.saturating_add(limit) < total_matching;
        let mut response = json!({
            "nodes": nodes,
            "count": nodes.len(),
            "offset": offset,
            "limit": limit,
        });
        // Match the unscoped path's total-reporting exactly (byte-identical
        // back-compat for an omitted/default-scope read — #3349 FIX2): a
        // property lookup reports the materialized `total_matching` (as the
        // unscoped property path does), while a label-only / no-filter scan
        // omits it (as the unscoped label-only path does). A provenance filter
        // makes the materialized total the *unfiltered* candidate count, so it
        // is likewise omitted. `has_more` always carries the completeness signal.
        let reported_total = if prov_filter.is_some() || req.property_key.is_none() {
            None
        } else {
            Some(total_matching)
        };
        Self::attach_completeness(&mut response, offset, limit, has_more, reported_total);
        self.success_json(response)
    }

    fn handle_count_nodes(&self, args: serde_json::Value) -> CallToolResult {
        let req: CountNodesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        if let Some(label) = &req.label {
            // Use QueryBuilder to count by label efficiently without collecting all rows
            let builder = crate::query::QueryBuilder::new().scan_label(label);
            match builder.execute(&self.db) {
                Ok(mut results) => {
                    // Efficiently count without allocating a Vec
                    match results.try_fold(0usize, |acc, row| row.map(|_| acc + 1)) {
                        Ok(count) => self.success_json(json!({"count": count, "label": label})),
                        Err(e) => self.error_result(McpError::from_db_error(&e).with_message(
                            format!("Error counting nodes with label '{}': {}", label, e),
                        )),
                    }
                }
                Err(e) => self.error_result(McpError::from_db_error(&e).with_message(format!(
                    "Error executing count query for label '{}': {}",
                    label, e
                ))),
            }
        } else {
            self.success_json(json!({"count": self.db.node_count()}))
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Namespace management (Issue #3349, PR3b)
    // ────────────────────────────────────────────────────────────────────────

    /// Render a namespace's metadata (without counts) as a JSON object:
    /// `{name, description, created_at}` (created_at RFC 3339).
    fn namespace_info_json(info: &crate::db::NamespaceInfo) -> serde_json::Value {
        json!({
            "name": info.name,
            "description": info.description,
            "created_at": time::to_iso8601(info.created_at),
        })
    }

    /// Look up the current-state `(node_count, edge_count)` for a namespace name
    /// from the O(1) membership-index-backed [`AletheiaDB::namespace_counts`]
    /// snapshot. Missing (a registered-but-unpopulated namespace not folded into
    /// the counts snapshot) defaults to `(0, 0)`.
    fn namespace_counts_for(counts: &[crate::db::NamespaceCount], name: &str) -> (usize, usize) {
        counts
            .iter()
            .find(|c| c.name == name)
            .map_or((0, 0), |c| (c.node_count, c.edge_count))
    }

    fn handle_create_namespace(&self, args: serde_json::Value) -> CallToolResult {
        let req: CreateNamespaceRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        match self.db.create_namespace(&req.name, req.description) {
            Ok(info) => self.success_json(Self::namespace_info_json(&info)),
            Err(e) => self.db_error(e),
        }
    }

    fn handle_list_namespaces(&self, args: serde_json::Value) -> CallToolResult {
        let _req: ListNamespacesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        let counts = self.db.namespace_counts();
        let namespaces: Vec<serde_json::Value> = self
            .db
            .list_namespaces()
            .iter()
            .map(|info| {
                let (node_count, edge_count) = Self::namespace_counts_for(&counts, &info.name);
                let mut obj = Self::namespace_info_json(info);
                if let Some(map) = obj.as_object_mut() {
                    map.insert("node_count".to_string(), json!(node_count));
                    map.insert("edge_count".to_string(), json!(edge_count));
                }
                obj
            })
            .collect();
        let count = namespaces.len();
        self.success_json(json!({ "namespaces": namespaces, "count": count }))
    }

    fn handle_describe_namespace(&self, args: serde_json::Value) -> CallToolResult {
        let req: DescribeNamespaceRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        match self.db.get_namespace(&req.name) {
            Ok(info) => {
                let counts = self.db.namespace_counts();
                let (node_count, edge_count) = Self::namespace_counts_for(&counts, &info.name);
                let mut obj = Self::namespace_info_json(&info);
                if let Some(map) = obj.as_object_mut() {
                    map.insert("node_count".to_string(), json!(node_count));
                    map.insert("edge_count".to_string(), json!(edge_count));
                }
                self.success_json(obj)
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_get_edge(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetEdgeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let edge_id = match EdgeId::new(req.edge_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        // Issue #3349: a namespace scope narrows visibility. Omitted ⇒ the
        // `default` namespace only (isolated-by-default, #3349 FIX2); `all`
        // imposes no filter (unscoped fast path).
        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();
        let edge_result = match &scope {
            crate::core::namespace::NamespaceScope::All => self.db.get_edge(edge_id),
            scope => self.db.get_edge_scoped(edge_id, scope),
        };

        match edge_result {
            Ok(edge) => {
                let now = time::now();
                let response =
                    self.edge_to_response(&edge, req.include_vectors.unwrap_or(false), now);
                self.success_json(
                    serde_json::to_value(&response)
                        .expect("response serialization should not fail"),
                )
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_create_edge(&self, args: serde_json::Value) -> CallToolResult {
        let req: CreateEdgeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let source_id = match NodeId::new(req.source_id) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&format!("Invalid source_id: {}", e)),
        };

        let target_id = match NodeId::new(req.target_id) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&format!("Invalid target_id: {}", e)),
        };

        let properties = match req.properties {
            Some(p) => match self.json_to_property_map(&p) {
                Ok(map) => map,
                Err(e) => return self.property_map_error(e),
            },
            None => PropertyMap::default(),
        };

        let valid_from = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v,
            Err(result) => return result,
        };
        let provenance = match self.parse_opt_provenance(req.provenance) {
            Ok(p) => p,
            Err(result) => return result,
        };

        let derived_from = match self.parse_derived_from(&req.derived_from) {
            Ok(refs) => refs,
            Err(result) => return result,
        };

        let namespace = match self.parse_opt_namespace(&req.namespace) {
            Ok(ns) => ns,
            Err(result) => return result,
        };

        let mut options = crate::api::transaction::WriteRequestOptions::new();
        if let Some(valid_from) = valid_from {
            options = options.with_valid_from(valid_from);
        }
        if let Some(provenance) = provenance {
            options = options.with_provenance(provenance);
        }
        if let Some(ns) = namespace.clone() {
            options = options.with_namespace(ns);
        }

        let created = if derived_from.is_empty() {
            self.db
                .create_edge_with_options(source_id, target_id, &req.label, properties, options)
        } else {
            self.db
                .create_edge_with_options_and_lineage(
                    source_id,
                    target_id,
                    &req.label,
                    properties,
                    options,
                    &derived_from,
                )
                .map(|(edge_id, _version)| edge_id)
        };

        // Auto-register the (non-default) namespace only AFTER a successful
        // commit (see handle_create_node).
        if let (Ok(_), Some(ns)) = (&created, &namespace)
            && let Err(e) = self.db.register_namespace_on_write(ns)
        {
            return self.db_error(e);
        }

        match created {
            Ok(edge_id) => match self.db.get_edge(edge_id) {
                Ok(edge) => {
                    let now = time::now();
                    let response = self.edge_to_response(&edge, true, now);
                    self.success_json(
                        serde_json::to_value(&response)
                            .expect("response serialization should not fail"),
                    )
                }
                Err(e) => self.db_error(e),
            },
            Err(e) => self.db_error(e),
        }
    }

    fn handle_update_edge(&self, args: serde_json::Value) -> CallToolResult {
        let req: UpdateEdgeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let edge_id = match EdgeId::new(req.edge_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        // A namespace is immutable after creation (Issue #3349): supplying one
        // on an update is INVALID_ARGUMENT, never a silent no-op.
        if let Err(result) = self.reject_namespace_on_write_update(&req.namespace) {
            return result;
        }

        let properties = match self.json_to_property_map(&req.properties) {
            Ok(map) => map,
            Err(e) => return self.property_map_error(e),
        };

        let valid_from = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v,
            Err(result) => return result,
        };
        let provenance = match self.parse_opt_provenance(req.provenance) {
            Ok(p) => p,
            Err(result) => return result,
        };

        let derived_from = match self.parse_derived_from(&req.derived_from) {
            Ok(refs) => refs,
            Err(result) => return result,
        };

        let mut options = crate::api::transaction::WriteRequestOptions::new();
        if let Some(valid_from) = valid_from {
            options = options.with_valid_from(valid_from);
        }
        if let Some(provenance) = provenance {
            options = options.with_provenance(provenance);
        }

        let updated = if derived_from.is_empty() {
            self.db
                .update_edge_with_options(edge_id, properties, options)
        } else {
            self.db
                .update_edge_with_options_and_lineage(edge_id, properties, options, &derived_from)
                .map(|_version| ())
        };

        match updated {
            Ok(()) => match self.db.get_edge(edge_id) {
                Ok(edge) => {
                    let now = time::now();
                    let response = self.edge_to_response(&edge, true, now);
                    self.success_json(
                        serde_json::to_value(&response)
                            .expect("response serialization should not fail"),
                    )
                }
                Err(e) => self.db_error(e),
            },
            Err(e) => self.db_error(e),
        }
    }

    fn handle_delete_edge(&self, args: serde_json::Value) -> CallToolResult {
        let req: DeleteEdgeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let edge_id = match EdgeId::new(req.edge_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        // A namespace is immutable after creation (Issue #3349): supplying one
        // on a delete is INVALID_ARGUMENT, never a silent no-op.
        if let Err(result) = self.reject_namespace_on_write_update(&req.namespace) {
            return result;
        }

        let valid_from = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v,
            Err(result) => return result,
        };

        // Issue #3427: principal-only, unforgeable provenance (see
        // handle_delete_node) stamped onto the tombstone version.
        let provenance = match self.parse_opt_provenance(None) {
            Ok(p) => p,
            Err(result) => return result,
        };
        let mut options = crate::api::transaction::WriteRequestOptions::new();
        if let Some(valid_from) = valid_from {
            options = options.with_valid_from(valid_from);
        }
        if let Some(provenance) = provenance {
            options = options.with_provenance(provenance);
        }

        match self
            .db
            .write(|tx| tx.delete_edge_with_options(edge_id, options.clone()))
        {
            Ok(()) => self.success_json(json!({
                "success": true,
                "deleted_edge_id": req.edge_id
            })),
            Err(e) => self.db_error(e),
        }
    }

    fn handle_list_edges(&self, args: serde_json::Value) -> CallToolResult {
        // Cursor paging (Issue #3360) is not supported here: `list_edges` does
        // not enumerate edges (there is no global edge scan). Rather than
        // silently ignore the flag, direct the caller to the cursor-paged
        // adjacency tools. (No-silent-fallback culture.)
        if Self::cursor_requested(&args) {
            return self.error_result(
                McpError::new(
                    McpErrorCode::InvalidArgument,
                    "list_edges does not enumerate edges and is not cursorable. Use \
                     get_outgoing_edges or get_incoming_edges from a known node -- both support \
                     snapshot-anchored cursor paging (use_cursor / cursor).",
                )
                .details(json!({ "cursorable_alternatives": ["get_outgoing_edges", "get_incoming_edges"] })),
            );
        }

        // Issue #3349: list_edges does not enumerate edges by namespace in v1;
        // a narrowing scope is INVALID_ARGUMENT (never silently unscoped).
        if let Err(result) = self.reject_unsupported_scope(
            &args.get("namespace").cloned(),
            "list_edges does not enumerate edges by namespace in v1; a namespace scope is not \
             supported here. Use the scoped adjacency/traversal reads instead, or \"all\".",
        ) {
            return result;
        }

        let req: ListEdgesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        // Apply resource limits
        let limit = req
            .limit
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .min(MAX_RESULT_LIMIT);
        let offset = req.offset.unwrap_or(0);

        // Edges cannot be efficiently listed without knowing source/target nodes.
        // Provide helpful guidance to use get_outgoing_edges or get_incoming_edges.
        let mut response = json!({
            "message": "Use 'get_outgoing_edges' or 'get_incoming_edges' from a known node to list edges",
            "total_count": self.db.edge_count(),
            "edges": [],
            "count": 0,
            "offset": offset,
            "limit": limit,
            "label_filter": req.label
        });
        Self::attach_completeness(&mut response, offset, 0, false, None);
        self.success_json(response)
    }

    fn handle_count_edges(&self, args: serde_json::Value) -> CallToolResult {
        let req: CountEdgesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        // Note: Counting by label is not efficiently supported without iterating all edges.
        // For now, we only support total count.
        if req.label.is_some() {
            self.success_json(json!({
                "message": "Counting edges by label is not supported. Use total_count instead.",
                "total_count": self.db.edge_count(),
                "count": null
            }))
        } else {
            self.success_json(json!({"count": self.db.edge_count()}))
        }
    }

    fn handle_get_outgoing_edges(&self, args: serde_json::Value) -> CallToolResult {
        // Snapshot-anchored cursor paging (Issue #3360); the full-adjacency
        // path below is unchanged for backward compatibility.
        if Self::cursor_requested(&args) {
            if Self::has_provenance_filter_args(&args) {
                return self.provenance_filter_cursor_unsupported();
            }
            return self.handle_adjacency_cursor("get_outgoing_edges", false, &args);
        }

        // Issue #3348: parse the optional provenance filter before consuming args.
        let prov_filter = match self.parse_provenance_filter(&args) {
            Ok(f) => f,
            Err(result) => return result,
        };

        let req: GetOutgoingEdgesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let edge_ids = if let Some(label) = &req.label {
            self.db.get_outgoing_edges_with_label(node_id, label)
        } else {
            self.db.get_outgoing_edges(node_id)
        };

        let include_vectors = req.include_vectors.unwrap_or(false);
        // One request-scoped wallclock for every entity in the response
        // (Issue #3391).
        let now = time::now();
        let edges: Vec<EdgeResponse> = edge_ids
            .into_iter()
            .filter_map(|eid| self.db.get_edge(eid).ok())
            .map(|e| self.edge_to_response(&e, include_vectors, now))
            // Issue #3348: keep only edges whose per-version provenance passes.
            .filter(|resp| {
                prov_filter
                    .as_ref()
                    .is_none_or(|f| f.matches(resp.provenance.as_ref()))
            })
            .collect();

        // This handler returns the complete adjacency (no limit/offset), so the
        // result is never truncated: `has_more` is always false and
        // `total_matching` equals the returned count.
        let count = edges.len();
        let mut response = json!({
            "edges": edges,
            "count": count
        });
        Self::attach_completeness(&mut response, 0, 0, false, Some(count));
        self.success_json(response)
    }

    fn handle_get_incoming_edges(&self, args: serde_json::Value) -> CallToolResult {
        // Snapshot-anchored cursor paging (Issue #3360); the full-adjacency
        // path below is unchanged for backward compatibility.
        if Self::cursor_requested(&args) {
            if Self::has_provenance_filter_args(&args) {
                return self.provenance_filter_cursor_unsupported();
            }
            return self.handle_adjacency_cursor("get_incoming_edges", true, &args);
        }

        // Issue #3348: parse the optional provenance filter before consuming args.
        let prov_filter = match self.parse_provenance_filter(&args) {
            Ok(f) => f,
            Err(result) => return result,
        };

        let req: GetIncomingEdgesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let edge_ids = self.db.get_incoming_edges(node_id);

        // Filter by label if provided
        let include_vectors = req.include_vectors.unwrap_or(false);
        // One request-scoped wallclock for every entity in the response
        // (Issue #3391).
        let now = time::now();
        let edges: Vec<EdgeResponse> = edge_ids
            .into_iter()
            .filter_map(|eid| self.db.get_edge(eid).ok())
            .filter(|e| {
                req.label
                    .as_ref()
                    .map(|l| self.matches_label(e.label, l))
                    .unwrap_or(true)
            })
            .map(|e| self.edge_to_response(&e, include_vectors, now))
            // Issue #3348: keep only edges whose per-version provenance passes.
            .filter(|resp| {
                prov_filter
                    .as_ref()
                    .is_none_or(|f| f.matches(resp.provenance.as_ref()))
            })
            .collect();

        // Complete adjacency (no limit/offset): never truncated, so
        // `has_more` is always false and `total_matching` equals the count.
        let count = edges.len();
        let mut response = json!({
            "edges": edges,
            "count": count
        });
        Self::attach_completeness(&mut response, 0, 0, false, Some(count));
        self.success_json(response)
    }

    /// Fetch a node, optionally as of a bi-temporal coordinate.
    fn get_node_maybe_at(
        &self,
        node_id: NodeId,
        temporal: Option<(Timestamp, Timestamp)>,
    ) -> Result<crate::core::Node, crate::core::error::Error> {
        match temporal {
            Some((vt, tt)) => self.db.get_node_at_time(node_id, vt, tt),
            None => self.db.get_node(node_id),
        }
    }

    /// Fetch an edge, optionally as of a bi-temporal coordinate.
    fn get_edge_maybe_at(
        &self,
        edge_id: EdgeId,
        temporal: Option<(Timestamp, Timestamp)>,
    ) -> Result<crate::core::Edge, crate::core::error::Error> {
        match temporal {
            Some((vt, tt)) => self.db.get_edge_at_time(edge_id, vt, tt),
            None => self.db.get_edge(edge_id),
        }
    }

    /// Fetch one candidate edge exactly once, check its label (unless the id
    /// list is already known to be label-filtered), and resolve the "next"
    /// node for this hop via `next_of` -- e.g. `|e| e.target` for an edge
    /// reached through the outgoing side, `|e| e.source` for the incoming
    /// side. Folding the label check and the source/target extraction into
    /// a single fetch avoids reconstructing the same edge twice per hop on
    /// the temporal path (each `get_edge_at_time` call is an index lookup
    /// plus a property reconstruction), and resolving `next_of` per-edge
    /// (rather than from the caller's top-level `direction` string) is what
    /// keeps a `"both"`-direction traversal correct: an edge discovered via
    /// the incoming side must resolve to `edge.source`, never `edge.target`.
    fn resolve_edge_hop(
        &self,
        edge_id: EdgeId,
        edge_label: &str,
        already_label_filtered: bool,
        next_of: impl Fn(&crate::core::Edge) -> NodeId,
        temporal: Option<(Timestamp, Timestamp)>,
    ) -> Option<NodeId> {
        let edge = self.get_edge_maybe_at(edge_id, temporal).ok()?;
        if !already_label_filtered && !self.matches_label(edge.label, edge_label) {
            return None;
        }
        Some(next_of(&edge))
    }

    /// Enumerate the next-hop node ids for a traversal step, optionally as
    /// of a bi-temporal coordinate. Current-state outgoing lookups are
    /// pre-filtered by label at the storage layer; every other combination
    /// (incoming, or the temporal index -- which has no label-aware
    /// incoming variant and, for outgoing, an existing
    /// `TemporalAdjacencyIndex::get_outgoing_with_label_at_time` that isn't
    /// yet plumbed through `HistoricalStorage`/`AletheiaDB`'s public API) is
    /// filtered by fetching each candidate edge once via `resolve_edge_hop`.
    fn traversal_next_hops(
        &self,
        current_id: NodeId,
        edge_label: &str,
        direction: &str,
        temporal: Option<(Timestamp, Timestamp)>,
    ) -> Vec<NodeId> {
        let outgoing = || -> Vec<NodeId> {
            match temporal {
                Some((vt, tt)) => self
                    .db
                    .get_outgoing_edges_at_time(current_id, vt, tt)
                    .into_iter()
                    .filter_map(|eid| {
                        self.resolve_edge_hop(eid, edge_label, false, |e| e.target, temporal)
                    })
                    .collect(),
                None => self
                    .db
                    .get_outgoing_edges_with_label(current_id, edge_label)
                    .into_iter()
                    .filter_map(|eid| {
                        self.resolve_edge_hop(eid, edge_label, true, |e| e.target, temporal)
                    })
                    .collect(),
            }
        };
        let incoming = || -> Vec<NodeId> {
            match temporal {
                Some((vt, tt)) => self.db.get_incoming_edges_at_time(current_id, vt, tt),
                None => self.db.get_incoming_edges(current_id),
            }
            .into_iter()
            .filter_map(|eid| self.resolve_edge_hop(eid, edge_label, false, |e| e.source, temporal))
            .collect()
        };

        match direction {
            "incoming" => incoming(),
            "both" => {
                let mut hops = outgoing();
                hops.extend(incoming());
                hops
            }
            _ => outgoing(),
        }
    }

    fn handle_traverse(&self, args: serde_json::Value) -> CallToolResult {
        // Snapshot-anchored cursor paging (Issue #3360). Unlike the id-keyset
        // node/adjacency scans, a DFS result order is not a simple id keyset,
        // so traverse's cursor pins the bi-temporal snapshot (making every
        // continuation page consistent -- AC2) and continues by an internal
        // offset over the deterministic DFS order (v1; a depth-independent
        // keyset traversal is a documented follow-up). The offset path below
        // is unchanged for backward compatibility.
        if Self::cursor_requested(&args) {
            if Self::has_provenance_filter_args(&args) {
                return self.provenance_filter_cursor_unsupported();
            }
            // Issue #3349: a narrowing namespace scope does not compose with the
            // #3360 cursor path in v1 — fail closed.
            if let Err(result) = self.reject_unsupported_scope(
                &args.get("namespace").cloned(),
                "namespace scope does not compose with cursor paging (use_cursor) in v1; use \
                 offset/limit paging with the scope instead, or \"all\".",
            ) {
                return result;
            }
            return self.handle_traverse_cursor(&args);
        }

        // Issue #3348: parse the optional provenance filter before consuming args.
        let prov_filter = match self.parse_provenance_filter(&args) {
            Ok(f) => f,
            Err(result) => return result,
        };

        let req: TraverseRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let start_id = match NodeId::new(req.start_node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let temporal = match self
            .resolve_bitemporal_as_of(&req.as_of_valid_time, &req.as_of_transaction_time)
        {
            Ok(t) => t,
            Err(result) => return result,
        };

        // Apply resource limits to prevent DoS. A page must be able to carry a
        // continuation cursor, so the limit is at least 1 (limit:0 would
        // otherwise report has_more:true with next_offset==offset, a
        // non-progressing page that traps a paginating caller in a loop).
        let depth = req.depth.unwrap_or(1).min(MAX_TRAVERSAL_DEPTH);
        let limit = req
            .limit
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .clamp(1, MAX_RESULT_LIMIT);
        let offset = req.offset.unwrap_or(0).min(MAX_PAGINATION_OFFSET);
        let direction = req.direction.as_deref().unwrap_or("outgoing");
        // One request-scoped wallclock for every entity in the response
        // (Issue #3391).
        let now = time::now();

        // Issue #3349 (FIX2 / PR3d): resolve an omitted scope to the `default`
        // namespace only (isolated-by-default). An explicit *non-default*
        // narrowing scope routes through the boundary-enforcing scoped BFS (a hop
        // is crossed only if the edge's own namespace AND the far node's
        // namespace are in scope, so a scope cannot leak via transit through an
        // out-of-scope node) — now for all three directions (outgoing, incoming,
        // both; PR3d). The `default` scope — whether omitted or explicit — and
        // the `all` selector run the full unscoped DFS below, which preserves the
        // path/depth result shape and every traversal direction; the `default`
        // case then drops any non-default node from the returned page so an
        // omitted read stays default-isolated (a pre-namespace database is all
        // `default`, so nothing is dropped and the result is byte-identical).
        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();
        let default_only = match &scope {
            crate::core::namespace::NamespaceScope::All => false,
            crate::core::namespace::NamespaceScope::Single(ns) if ns.is_default() => true,
            _ => {
                return self.handle_traverse_scoped(
                    &req,
                    &scope,
                    start_id,
                    direction,
                    depth,
                    limit,
                    offset,
                    temporal,
                    prov_filter.as_ref(),
                    now,
                );
            }
        };

        let (mut results, has_more) = self.run_traversal(
            start_id,
            &req.edge_label,
            direction,
            depth,
            limit,
            offset,
            temporal,
            req.include_vectors.unwrap_or(false),
            now,
        );

        // Issue #3348: apply the provenance filter to this page's returned
        // nodes (each carries its per-version provenance, resolved at the
        // temporal coordinate when `as_of_*` is set -- AC3). `page_window` is
        // the pre-filter page size, so `next_offset` still advances over the
        // unfiltered DFS order (gap-free/dup-free); a page may hold fewer than
        // `limit` results. The filter constrains the returned nodes, not which
        // edges the DFS follows (edge-provenance-gated path pruning is a
        // documented follow-up).
        let page_window = results.len();
        // Default-scope isolation (#3349 FIX2): drop any non-default node from
        // the page (its response carries `Some(namespace)`; the default
        // namespace is elided to `None`). `page_window` above is the pre-filter
        // DFS page size, so `next_offset` still advances over the unfiltered DFS
        // order (gap-free/dup-free), exactly like the provenance filter.
        // Semantics note (#3349): this `default` path filters returned nodes, not
        // the crossed edges, so it preserves the path/depth DFS result shape for
        // every direction. An explicit narrowing scope instead routes through the
        // boundary-enforcing scoped BFS above (all directions, PR3d), which never
        // crosses an out-of-scope edge/node; the two paths differ in shape by
        // design (see `handle_traverse_scoped`).
        if default_only {
            results.retain(|r| r.node.namespace.is_none());
        }
        if let Some(filter) = &prov_filter {
            results.retain(|r| filter.matches(r.node.provenance.as_ref()));
        }

        let count = results.len();
        let mut response = match temporal {
            Some((vt, tt)) => json!({
                "results": results,
                "count": count,
                "as_of_valid_time": time::to_iso8601(vt),
                "as_of_transaction_time": time::to_iso8601(tt),
            }),
            None => json!({
                "results": results,
                "count": count
            }),
        };
        // The matching total would require exhausting the traversal, so
        // `total_matching` is omitted; `has_more`/`next_offset` carry the
        // completeness signal.
        Self::attach_completeness(&mut response, offset, page_window, has_more, None);
        self.success_json(response)
    }

    /// Namespace-scoped `traverse` (Issue #3349). Routes through the
    /// boundary-enforcing scoped BFS (`traverse_scoped` / `traverse_scoped_as_of`)
    /// so an edge is crossed only when the edge's own namespace AND the target
    /// node's namespace are in scope — a scope can never leak across the
    /// boundary or via transit through an out-of-scope node.
    ///
    /// All three directions (`outgoing`, `incoming`, `both`) are supported and
    /// boundary-scoped (Issue #3349, PR3d): the boundary rule (edge.ns ∧
    /// far-node.ns ∈ scope) applies symmetrically. The scoped BFS returns the
    /// reachable in-scope node **set** (sorted by id), so each result carries its
    /// `node` while `depth`/`path` are omitted — the boundary-correct set, not
    /// per-path detail (unchanged from the shipped outgoing scoped path). It does
    /// not compose with the #3360 cursor / #3353 token-budget shaping (offset
    /// paging applies).
    #[allow(clippy::too_many_arguments)]
    fn handle_traverse_scoped(
        &self,
        req: &TraverseRequest,
        scope: &crate::core::namespace::NamespaceScope,
        start_id: NodeId,
        direction: &str,
        depth: usize,
        limit: usize,
        offset: usize,
        temporal: Option<(Timestamp, Timestamp)>,
        prov_filter: Option<&crate::core::ProvenanceFilter>,
        now: Timestamp,
    ) -> CallToolResult {
        // Issue #3349 (PR3d): all three traversal directions are boundary-scoped.
        // The boundary rule (edge.ns ∧ far-node.ns ∈ scope) applies symmetrically
        // for outgoing (→ target), incoming (source →), and both.
        let traverse_direction = match direction {
            "outgoing" => crate::db::namespace_query::TraverseDirection::Outgoing,
            "incoming" => crate::db::namespace_query::TraverseDirection::Incoming,
            "both" => crate::db::namespace_query::TraverseDirection::Both,
            other => {
                return self.invalid_argument(&format!(
                    "invalid traverse direction {other:?}; expected \"outgoing\", \"incoming\", \
                     or \"both\""
                ));
            }
        };
        // The scoped BFS takes an optional edge label; the MCP tool always
        // supplies one (a required field), so restrict to that relationship.
        let edge_label = Some(req.edge_label.as_str());
        let reachable = match temporal {
            Some((vt, tt)) => self.db.traverse_scoped_as_of_directed(
                start_id,
                edge_label,
                traverse_direction,
                depth,
                vt,
                tt,
                scope,
            ),
            None => self.db.traverse_scoped_directed(
                start_id,
                edge_label,
                traverse_direction,
                depth,
                scope,
            ),
        };
        let reachable = match reachable {
            Ok(ids) => ids,
            Err(e) => return self.db_error(e),
        };

        let total = reachable.len();
        let include_vectors = req.include_vectors.unwrap_or(false);
        let mut results: Vec<serde_json::Value> = Vec::new();
        for node_id in reachable.into_iter().skip(offset).take(limit) {
            let node = match temporal {
                Some((vt, tt)) => self.db.get_node_at_time(node_id, vt, tt),
                None => self.db.get_node(node_id),
            };
            if let Ok(node) = node {
                let resp = self.node_to_response(&node, include_vectors, now);
                if prov_filter.is_none_or(|f| f.matches(resp.provenance.as_ref())) {
                    results.push(json!({ "node": resp }));
                }
            }
        }

        let has_more = offset.saturating_add(limit) < total;
        let count = results.len();
        let mut response = match temporal {
            Some((vt, tt)) => json!({
                "results": results,
                "count": count,
                "as_of_valid_time": time::to_iso8601(vt),
                "as_of_transaction_time": time::to_iso8601(tt),
            }),
            None => json!({
                "results": results,
                "count": count,
            }),
        };
        Self::attach_completeness(&mut response, offset, limit, has_more, Some(total));
        self.success_json(response)
    }

    /// The DFS core of `traverse`, shared by the offset path and the
    /// snapshot-anchored cursor path (Issue #3360). Returns the page of
    /// results (after skipping `offset`, taking `limit`) and whether more
    /// remain. Behavior is identical to the pre-refactor inline loop.
    #[allow(clippy::too_many_arguments)]
    fn run_traversal(
        &self,
        start_id: NodeId,
        edge_label: &str,
        direction: &str,
        depth: usize,
        limit: usize,
        offset: usize,
        temporal: Option<(Timestamp, Timestamp)>,
        include_vectors: bool,
        now: Timestamp,
    ) -> (Vec<TraversalResult>, bool) {
        // Use depth-first search (DFS) traversal.
        // DFS is chosen for memory efficiency: it processes nodes immediately rather than
        // queuing all nodes at each level. For large graphs with high branching factors,
        // this significantly reduces peak memory usage compared to BFS.
        let mut results: Vec<TraversalResult> = Vec::new();
        let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut frontier: Vec<(NodeId, Vec<u64>, usize)> =
            vec![(start_id, vec![start_id.as_u64()], 0)];
        // Caches whether a node exists as of `temporal`, so a node already
        // resolved (via the results-push lookup below, or a prior visit) is
        // never re-fetched just to decide whether to keep expanding past it.
        // Only consulted on the temporal path: the current-state path never
        // gated edge-following on node existence (an edge left dangling by a
        // non-cascade delete_node is documented, pre-existing behavior --
        // see CLAUDE.md's "Orphaned Edges" section -- and stays unchanged).
        let mut node_exists_cache: std::collections::HashMap<u64, bool> =
            std::collections::HashMap::new();
        // Offset pagination + completeness: `produced` counts every resolved
        // result node in traversal order; the first `offset` are skipped (a
        // prior page), the next `limit` are collected. Once the page is full
        // we expand the last collected node's own out-edges as usual (so
        // graph structure discovery for this hop isn't cut short), then stop:
        // `has_more` is read off whatever remains on `frontier` rather than by
        // resolving one further node. That peek-by-resolution would otherwise
        // force expansion through dangling/orphaned edges (current-state mode
        // keeps expanding past unresolvable nodes -- see the Err(_) arm below)
        // in search of a single confirmable result, which can walk the entire
        // remaining reachable subgraph just to answer `has_more`. Reading the
        // frontier instead is O(1) and conservative: it may say `true` when
        // the remaining candidates are all dangling and the next page would
        // in fact be empty, but it never requires unbounded work and never
        // under-reports.
        let mut produced: usize = 0;
        let mut has_more = false;

        while let Some((current_id, path, current_depth)) = frontier.pop() {
            let mut current_exists = true;
            if current_depth > 0 && !visited.contains(&current_id.as_u64()) {
                visited.insert(current_id.as_u64());
                match self.get_node_maybe_at(current_id, temporal) {
                    Ok(node) => {
                        produced += 1;
                        if produced > offset && results.len() < limit {
                            results.push(TraversalResult {
                                node: self.node_to_response(&node, include_vectors, now),
                                path: path.clone(),
                                depth: current_depth,
                            });
                        }
                    }
                    // Non-temporal: preserve the historical "keep expanding
                    // regardless" behavior. Temporal: a node absent at the
                    // coordinate must not have its edges followed further --
                    // otherwise nodes only reachable through it would
                    // surface despite being excluded from `results`.
                    Err(_) => current_exists = temporal.is_none(),
                }
                if temporal.is_some() {
                    node_exists_cache.insert(current_id.as_u64(), current_exists);
                }
            } else if temporal.is_some() {
                current_exists = *node_exists_cache
                    .entry(current_id.as_u64())
                    .or_insert_with(|| self.get_node_maybe_at(current_id, temporal).is_ok());
            }

            if current_depth < depth && current_exists {
                let next_ids =
                    self.traversal_next_hops(current_id, edge_label, direction, temporal);

                for next_id in next_ids {
                    if !visited.contains(&next_id.as_u64()) {
                        let mut new_path = path.clone();
                        new_path.push(next_id.as_u64());
                        frontier.push((next_id, new_path, current_depth + 1));
                    }
                }
            }

            if results.len() >= limit {
                has_more = !frontier.is_empty();
                break;
            }
        }

        (results, has_more)
    }

    /// Cursor-mode `traverse` (Issue #3360): snapshot-pinned offset
    /// continuation. On the first page the bi-temporal snapshot is pinned
    /// (to the request's `as_of_*` coordinate, or to "now" if none was given,
    /// so a current-state traversal still becomes a consistent point-in-time
    /// scan for the duration of the cursor). Every continuation re-walks the
    /// deterministic DFS as of that pinned snapshot and skips the already-seen
    /// prefix, so all pages reflect one consistent moment.
    fn handle_traverse_cursor(&self, args: &serde_json::Value) -> CallToolResult {
        let now = time::now();

        // Resolve page parameters, start node, and pinned snapshot, from the
        // token when resuming or from the request on the first page.
        let (
            start_id,
            edge_label,
            direction,
            depth,
            limit,
            offset,
            include_vectors,
            snapshot,
            parent_cid,
        ) = if let Some(token) = args.get("cursor").and_then(|v| v.as_str()) {
            let payload = match self.cursors.decode(token, "traverse") {
                Ok(p) => p,
                Err(e) => return self.error_result(e),
            };
            let f = &payload.filters;
            let start_id = match NodeId::new(f["start_node_id"].as_u64().unwrap_or(0)) {
                Ok(id) => id,
                Err(e) => return self.invalid_argument(&e.to_string()),
            };
            (
                start_id,
                f["edge_label"].as_str().unwrap_or("").to_string(),
                f["direction"].as_str().unwrap_or("outgoing").to_string(),
                f["depth"].as_u64().unwrap_or(1) as usize,
                payload.limit,
                payload.off as usize,
                f["include_vectors"].as_bool().unwrap_or(false),
                (Timestamp::from(payload.svt), Timestamp::from(payload.stt)),
                payload.cid,
            )
        } else {
            // First page: parse the request and pin the snapshot. If no as_of
            // was supplied, anchor at "now" so the whole scan is consistent.
            let start_id = match NodeId::new(
                args.get("start_node_id")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            ) {
                Ok(id) => id,
                Err(e) => return self.invalid_argument(&e.to_string()),
            };
            let temporal = match self.resolve_bitemporal_as_of(
                &Self::arg_str(args, "as_of_valid_time"),
                &Self::arg_str(args, "as_of_transaction_time"),
            ) {
                Ok(t) => t,
                Err(result) => return result,
            };
            let snapshot = temporal.unwrap_or((now, now));
            (
                start_id,
                Self::arg_str(args, "edge_label").unwrap_or_default(),
                Self::arg_str(args, "direction").unwrap_or_else(|| "outgoing".into()),
                args.get("depth")
                    .and_then(|v| v.as_u64())
                    .map(|d| d as usize)
                    .unwrap_or(1)
                    .min(MAX_TRAVERSAL_DEPTH),
                Self::arg_limit(args),
                0usize,
                args.get("include_vectors")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                snapshot,
                String::new(),
            )
        };

        let (results, has_more) = self.run_traversal(
            start_id,
            &edge_label,
            &direction,
            depth,
            limit,
            offset,
            Some(snapshot),
            include_vectors,
            now,
        );
        let count = results.len();

        let mut obj = serde_json::Map::new();
        obj.insert(
            "results".to_string(),
            serde_json::to_value(&results).unwrap_or_else(|_| json!([])),
        );
        obj.insert("count".to_string(), json!(count));
        obj.insert(
            "snapshot_valid_time".to_string(),
            json!(Self::format_timestamp_rfc3339(snapshot.0)),
        );
        obj.insert(
            "snapshot_transaction_time".to_string(),
            json!(Self::format_timestamp_rfc3339(snapshot.1)),
        );
        obj.insert("has_more".to_string(), json!(has_more));
        obj.insert("paging".to_string(), json!("cursor"));

        if has_more {
            let filters = json!({
                "start_node_id": start_id.as_u64(),
                "edge_label": edge_label,
                "direction": direction,
                "depth": depth,
                "include_vectors": include_vectors,
            });
            let mut payload = CursorPayload::seed(
                "traverse",
                (snapshot.0.wallclock(), snapshot.1.wallclock()),
                limit,
                filters,
            );
            // Continuation offset advances by the number of rows ACTUALLY
            // EMITTED (`count`). When #3353 token budgets land and can trim a
            // page short of `limit`, `off` MUST advance by the post-trim row
            // count (not `limit`), or the next page would skip trimmed rows.
            payload.off = (offset + count) as u64;
            payload.cid = parent_cid;
            match self.cursors.issue(payload) {
                Ok(token) => {
                    obj.insert("cursor".to_string(), json!(token));
                    obj.insert(
                        "cursor_ttl_seconds".to_string(),
                        json!(self.cursors.ttl().as_secs()),
                    );
                }
                Err(e) => return self.error_result(e),
            }
        }

        self.success_json(serde_json::Value::Object(obj))
    }

    // ========================================================================
    // Semantic-search analysis tools (Issue #2907)
    //
    // Advertised unconditionally (Design A). The real handler bodies live under
    // `#[cfg(feature = "semantic-search")]`; a `#[cfg(not(...))]` twin returns
    // the structured `semantic_search_unavailable` FAILED_PRECONDITION so a
    // caller on a build without the feature gets an actionable error rather than
    // an "unknown tool".
    // ========================================================================

    /// Structured `FAILED_PRECONDITION` returned by every semantic-search tool
    /// when the `semantic-search` feature is not compiled into this build
    /// (Issue #2907). Mirrors the cypher `language_unavailable` pattern.
    #[cfg(not(feature = "semantic-search"))]
    fn semantic_search_unavailable(&self, tool: &str) -> CallToolResult {
        self.error_result(
            McpError::new(
                McpErrorCode::FailedPrecondition,
                format!(
                    "Tool '{tool}' requires the `semantic-search` feature, which is not compiled \
                     into this build. Rebuild AletheiaDB with `--features semantic-search` to \
                     enable it."
                ),
            )
            .details(json!({
                "tool": tool,
                "required_feature": "semantic-search",
            })),
        )
    }

    #[cfg(not(feature = "semantic-search"))]
    fn handle_semantic_path(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_search_unavailable("semantic_path")
    }

    #[cfg(feature = "semantic-search")]
    fn handle_semantic_path(&self, args: serde_json::Value) -> CallToolResult {
        let req: SemanticPathRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        let start = match NodeId::new(req.start) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        let end = match NodeId::new(req.end) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        if !self.db.is_vector_index_enabled_for(&req.property_name) {
            return self.failed_precondition(&format!(
                "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                req.property_name
            ));
        }
        // Clamp depth and derive a bounded expansion budget (DoS protection —
        // the plain A* search is unbounded).
        let max_depth = req
            .max_depth
            .unwrap_or(MAX_TRAVERSAL_DEPTH)
            .clamp(1, MAX_TRAVERSAL_DEPTH);
        let max_expansions = max_depth.saturating_mul(SEMANTIC_PATH_EXPANSIONS_PER_DEPTH);

        let navigator =
            crate::semantic_search::semantic_navigator::SemanticNavigator::new(&self.db);
        use crate::semantic_search::semantic_navigator::PathSearchOutcome;
        match navigator.find_path_bounded(start, end, &req.property_name, max_expansions) {
            Ok(PathSearchOutcome::Found(path)) => {
                let ids: Vec<u64> = path.iter().map(|n| n.as_u64()).collect();
                self.success_json(json!({
                    "path": ids,
                    "length": path.len(),
                    "start": req.start,
                    "end": req.end,
                    "property_name": req.property_name,
                }))
            }
            // A genuinely disconnected pair is a NORMAL outcome, not a server
            // bug: report NOT_FOUND (non-retriable) rather than INTERNAL, so an
            // LLM treats it as "no such route" instead of escalating.
            Ok(PathSearchOutcome::NoPath) => self.error_result(McpError::new(
                McpErrorCode::NotFound,
                format!(
                    "no path found between nodes {} and {} within max_depth {}",
                    req.start, req.end, max_depth
                ),
            )),
            // The DoS-protection expansion budget was exhausted before a path
            // was found. This is a resource-limit refusal (FAILED_PRECONDITION,
            // matching StorageError::CapacityExceeded's classification), not an
            // internal error: the caller can lower max_depth or pick closer
            // endpoints and retry a smaller search.
            Ok(PathSearchOutcome::BudgetExhausted { max_expansions }) => self.error_result(
                McpError::new(
                    McpErrorCode::FailedPrecondition,
                    format!(
                        "semantic path search hit its expansion budget of {max_expansions} \
                             node expansions before finding a path; lower max_depth or choose \
                             closer endpoints"
                    ),
                )
                .details(json!({
                    "reason": "expansion_budget_exceeded",
                    "max_expansions": max_expansions,
                })),
            ),
            Err(e) => self.db_error(e),
        }
    }

    #[cfg(not(feature = "semantic-search"))]
    fn handle_concept_analogy(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_search_unavailable("concept_analogy")
    }

    #[cfg(feature = "semantic-search")]
    fn handle_concept_analogy(&self, args: serde_json::Value) -> CallToolResult {
        let req: ConceptAnalogyRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        let a = match NodeId::new(req.a) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        let b = match NodeId::new(req.b) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        let c = match NodeId::new(req.c) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        if !self.db.is_vector_index_enabled_for(&req.property_name) {
            return self.failed_precondition(&format!(
                "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                req.property_name
            ));
        }
        let k = req.k.unwrap_or(DEFAULT_VECTOR_K).clamp(1, MAX_VECTOR_K);
        let algebra = crate::semantic_search::concept_algebra::ConceptAlgebra::new(&self.db)
            .with_property(&req.property_name);
        match algebra.analogy(a, b, c, k) {
            Ok(results) => self.success_json(Self::rank_response(&results, &req.property_name)),
            Err(e) => self.db_error(e),
        }
    }

    #[cfg(not(feature = "semantic-search"))]
    fn handle_concept_mean(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_search_unavailable("concept_mean")
    }

    #[cfg(feature = "semantic-search")]
    fn handle_concept_mean(&self, args: serde_json::Value) -> CallToolResult {
        let req: ConceptMeanRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        if req.nodes.len() > MAX_RESULT_LIMIT {
            return self.invalid_argument(&format!(
                "Too many nodes ({}); at most {} are accepted.",
                req.nodes.len(),
                MAX_RESULT_LIMIT
            ));
        }
        let mut nodes = Vec::with_capacity(req.nodes.len());
        for raw in &req.nodes {
            match NodeId::new(*raw) {
                Ok(id) => nodes.push(id),
                Err(e) => return self.invalid_argument(&e.to_string()),
            }
        }
        if !self.db.is_vector_index_enabled_for(&req.property_name) {
            return self.failed_precondition(&format!(
                "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                req.property_name
            ));
        }
        let k = req.k.unwrap_or(DEFAULT_VECTOR_K).clamp(1, MAX_VECTOR_K);
        let algebra = crate::semantic_search::concept_algebra::ConceptAlgebra::new(&self.db)
            .with_property(&req.property_name);
        match algebra.mean(&nodes, k) {
            Ok(results) => self.success_json(Self::rank_response(&results, &req.property_name)),
            Err(e) => self.db_error(e),
        }
    }

    #[cfg(not(feature = "semantic-search"))]
    fn handle_find_duplicate_candidates(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_search_unavailable("find_duplicate_candidates")
    }

    #[cfg(feature = "semantic-search")]
    fn handle_find_duplicate_candidates(&self, args: serde_json::Value) -> CallToolResult {
        let req: FindDuplicateCandidatesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        // v1: the underlying SimilarityQuery has no property selector, so the
        // search resolves against the node's own indexed embedding. We still
        // validate that a vector index exists for `property_name` so a caller
        // gets a clear precondition error rather than a silent surprise.
        if !self.db.is_vector_index_enabled_for(&req.property_name) {
            return self.failed_precondition(&format!(
                "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                req.property_name
            ));
        }
        let threshold = req.threshold.unwrap_or(0.9);
        if !(0.0..=1.0).contains(&threshold) {
            return self
                .invalid_argument(&format!("threshold must be in [0, 1] (got {threshold})."));
        }
        let limit = req.limit.unwrap_or(DEFAULT_VECTOR_K).clamp(1, MAX_VECTOR_K);
        let detector = crate::semantic_search::highlander::HighlanderDetector::new(&self.db);
        match detector.find_duplicates(node_id, threshold, limit) {
            Ok(results) => {
                // The underlying similarity query resolves against the target's
                // own embedding, so it can return the target itself (similarity
                // ~= 1.0). A node is never its own duplicate — filter it out.
                let candidates: Vec<serde_json::Value> = results
                    .iter()
                    .filter(|(id, _)| *id != node_id)
                    .map(|(id, sim)| json!({ "node_id": id.as_u64(), "similarity": sim }))
                    .collect();
                let count = candidates.len();
                self.success_json(json!({
                    "candidates": candidates,
                    "count": count,
                    "threshold": threshold,
                    "target": req.node_id,
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    #[cfg(not(feature = "semantic-search"))]
    fn handle_semantic_horizon(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_search_unavailable("semantic_horizon")
    }

    #[cfg(feature = "semantic-search")]
    fn handle_semantic_horizon(&self, args: serde_json::Value) -> CallToolResult {
        let req: SemanticHorizonRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        let seed = match NodeId::new(req.seed) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        if !(0.0..=1.0).contains(&req.threshold) {
            return self.invalid_argument(&format!(
                "threshold must be in [0, 1] (got {}).",
                req.threshold
            ));
        }
        if !self.db.is_vector_index_enabled_for(&req.property_name) {
            return self.failed_precondition(&format!(
                "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                req.property_name
            ));
        }
        let max_depth = req
            .max_depth
            .unwrap_or(MAX_TRAVERSAL_DEPTH)
            .clamp(1, MAX_TRAVERSAL_DEPTH);
        let engine = crate::semantic_search::horizon::HorizonEngine::new(&self.db);
        match engine.map_horizon(seed, &req.property_name, req.threshold, max_depth) {
            Ok(result) => {
                let mut interior: Vec<u64> = result.interior.iter().map(|n| n.as_u64()).collect();
                interior.sort_unstable();
                let mut horizon: Vec<u64> = result.horizon.iter().map(|n| n.as_u64()).collect();
                horizon.sort_unstable();
                self.success_json(json!({
                    "seed": req.seed,
                    "threshold": req.threshold,
                    "max_depth": max_depth,
                    "interior_count": interior.len(),
                    "horizon_count": horizon.len(),
                    "interior": interior,
                    "horizon": horizon,
                }))
            }
            // The visited-node DoS cap (a resource-limit refusal) surfaces as a
            // FAILED_PRECONDITION naming the cap, not an INTERNAL error, so a
            // caller knows to narrow the query (lower max_depth / raise
            // threshold) rather than escalate a server bug.
            Err(crate::core::error::Error::Storage(
                crate::core::error::StorageError::CapacityExceeded { limit, .. },
            )) => self.error_result(
                McpError::new(
                    McpErrorCode::FailedPrecondition,
                    format!(
                        "semantic horizon search hit its visited-node cap of {limit} nodes; \
                         narrow the query by lowering max_depth or raising threshold"
                    ),
                )
                .details(json!({
                    "reason": "visit_cap_exceeded",
                    "max_visited_nodes": limit,
                })),
            ),
            Err(e) => self.db_error(e),
        }
    }

    #[cfg(not(feature = "semantic-search"))]
    fn handle_context_aspects(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_search_unavailable("context_aspects")
    }

    #[cfg(feature = "semantic-search")]
    fn handle_context_aspects(&self, args: serde_json::Value) -> CallToolResult {
        let req: ContextAspectsRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        if !self.db.is_vector_index_enabled_for(&req.property_name) {
            return self.failed_precondition(&format!(
                "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                req.property_name
            ));
        }
        let k = req.k.unwrap_or(DEFAULT_VECTOR_K).clamp(1, MAX_VECTOR_K);
        let include_vectors = req.include_vectors.unwrap_or(false);
        let chameleon = crate::semantic_search::chameleon::Chameleon::new(&self.db);
        match chameleon.analyze_context(node_id, &req.property_name, k) {
            Ok(aspects) => {
                let aspects_json: Vec<serde_json::Value> = aspects
                    .iter()
                    .map(|aspect| {
                        // Sort exemplars by ascending node id for deterministic,
                        // reproducible LLM-facing output (Chameleon clustering
                        // order is not guaranteed stable). Mirrors the sorted
                        // interior/horizon sets in semantic_horizon.
                        let mut exemplars: Vec<u64> =
                            aspect.exemplars.iter().map(|n| n.as_u64()).collect();
                        exemplars.sort_unstable();
                        // Elide the centroid vector by default (#3220).
                        let centroid = if include_vectors {
                            json!(aspect.centroid)
                        } else {
                            json!({
                                "type": "vector",
                                "dim": aspect.centroid.len(),
                                "elided": true,
                            })
                        };
                        json!({
                            "weight": aspect.weight,
                            "exemplars": exemplars,
                            "centroid": centroid,
                        })
                    })
                    .collect();
                self.success_json(json!({
                    "node_id": req.node_id,
                    "count": aspects_json.len(),
                    "aspects": aspects_json,
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    /// Shared response shape for the ranked concept-algebra tools
    /// (`concept_analogy`, `concept_mean`): `{results:[{node_id, score}], count,
    /// property_name}`.
    #[cfg(feature = "semantic-search")]
    fn rank_response(results: &[(NodeId, f32)], property_name: &str) -> serde_json::Value {
        let entries: Vec<serde_json::Value> = results
            .iter()
            .map(|(id, score)| json!({ "node_id": id.as_u64(), "score": score }))
            .collect();
        json!({
            "results": entries,
            "count": results.len(),
            "property_name": property_name,
        })
    }

    fn handle_find_similar(&self, args: serde_json::Value) -> CallToolResult {
        // Issue #3348: parse the optional provenance filter before consuming args.
        let prov_filter = match self.parse_provenance_filter(&args) {
            Ok(f) => f,
            Err(result) => return result,
        };

        // Issue #3372: when a `fusion_policy` is supplied, re-rank by the fused
        // score and return here; otherwise fall through to the unchanged
        // similarity-only path below. Feature-gated: the param is only
        // advertised/parsed when `semantic-retrieval-fusion` is compiled.
        #[cfg(feature = "semantic-retrieval-fusion")]
        {
            match self.parse_fusion_policy(&args) {
                Ok(Some(policy)) => {
                    return self.handle_find_similar_fused(&args, &policy, prov_filter.as_ref());
                }
                Ok(None) => {}
                Err(result) => return result,
            }
        }

        let req: FindSimilarRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        // Apply resource limits. A page must be able to carry a continuation
        // cursor, so k is at least 1 (k:0 would otherwise report
        // has_more:true with next_offset==offset, a non-progressing page).
        let k = req.k.unwrap_or(DEFAULT_VECTOR_K).clamp(1, MAX_VECTOR_K);
        // Bound the total pagination window (offset+k) by MAX_VECTOR_K so the
        // over-fetch below never asks the vector index for more than
        // MAX_VECTOR_K+1 candidates regardless of offset -- otherwise a large
        // offset could force a search far past the MAX_VECTOR_K resource
        // budget the cap is meant to enforce. Vector-similarity pagination is
        // therefore bounded to the top MAX_VECTOR_K matches; an offset beyond
        // that horizon returns an empty, complete (`has_more: false`) page.
        let offset = req
            .offset
            .unwrap_or(0)
            .min(MAX_PAGINATION_OFFSET)
            .min(MAX_VECTOR_K.saturating_sub(k));

        if !self.db.is_vector_index_enabled_for(&req.property_name) {
            return self.failed_precondition(&format!(
                "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                req.property_name
            ));
        }

        // Validate embedding dimensions
        if let Err(e) = self.validate_embedding_dimensions(&req.embedding, &req.property_name) {
            return self.invalid_argument(&e);
        }

        // Issue #3349 (FIX2): an omitted scope resolves to the `default`
        // namespace only (isolated-by-default) and routes through the
        // filter-complete scoped k-NN (over-fetches until it has k genuinely
        // in-scope results — never k-then-drop), exactly like an explicit
        // narrowing scope. For a pre-namespace database (all data is `default`)
        // this returns the same top-k. Only the `all` selector falls through to
        // the full unscoped search below.
        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();
        if !matches!(scope, crate::core::namespace::NamespaceScope::All) {
            return self.handle_find_similar_scoped(&req, &scope, k, offset, prov_filter.as_ref());
        }

        // Over-fetch one past the requested page (offset + k + 1, capped at
        // MAX_VECTOR_K + 1 by the offset bound above) so we can tell whether
        // more similar nodes exist beyond this page (`has_more`) without a
        // second query. The matching total would need a full index scan, so
        // `total_matching` is omitted.
        //
        // Issue #3348 (AC6): when a provenance filter is active, over-fetch the
        // full candidate horizon (MAX_VECTOR_K + 1) and apply the filter to
        // CANDIDATES so the returned top-k are all filter-passing -- never a
        // post-truncation that returns fewer than k when >= k passing
        // candidates exist within the horizon. The horizon (MAX_VECTOR_K) is
        // the same resource bound the unfiltered path already enforces.
        let fetch_k = if prov_filter.is_some() {
            MAX_VECTOR_K.saturating_add(1)
        } else {
            k.saturating_add(offset).saturating_add(1)
        };
        match self
            .db
            .similarity_search(crate::SimilarityQuery::from_embedding(req.embedding).k(fetch_k))
        {
            Ok(results) => {
                let include_vectors = req.include_vectors.unwrap_or(false);
                // One request-scoped wallclock for every entity in the
                // response (Issue #3391).
                let now = time::now();

                let (similarity_results, has_more) = if let Some(filter) = &prov_filter {
                    // Build each candidate (in descending score order), keep
                    // only filter-passing ones, THEN page. Ranking order is
                    // preserved; only per-candidate inclusion changes (AC6).
                    let passing: Vec<SimilarityResult> = results
                        .into_iter()
                        .filter_map(|(node_id, score)| {
                            self.db.get_node(node_id).ok().map(|node| SimilarityResult {
                                node: self.node_to_response(&node, include_vectors, now),
                                score,
                                score_breakdown: None,
                            })
                        })
                        .filter(|r| filter.matches(r.node.provenance.as_ref()))
                        .collect();
                    let has_more = passing.len() > offset.saturating_add(k);
                    let page = passing.into_iter().skip(offset).take(k).collect();
                    (page, has_more)
                } else {
                    let has_more = results.len() > offset.saturating_add(k);
                    let page: Vec<SimilarityResult> = results
                        .into_iter()
                        .skip(offset)
                        .take(k)
                        .filter_map(|(node_id, score)| {
                            self.db.get_node(node_id).ok().map(|node| SimilarityResult {
                                node: self.node_to_response(&node, include_vectors, now),
                                score,
                                score_breakdown: None,
                            })
                        })
                        .collect();
                    (page, has_more)
                };

                let count = similarity_results.len();
                let mut response = json!({
                    "results": similarity_results,
                    "count": count
                });
                // `next_offset` advances by the requested window `k`, not the
                // (possibly smaller) resolved `count`: a since-deleted node
                // behind a stale vector-index entry is still one of the `k`
                // candidates this page consumed, so basing next_offset on
                // `count` would re-skip into already-consumed candidates and
                // duplicate a row on the next page.
                Self::attach_completeness(&mut response, offset, k, has_more, None);
                self.success_json(response)
            }
            Err(e) => self.db_error(e),
        }
    }

    /// Namespace-scoped `find_similar` (Issue #3349). Routes through the
    /// filter-complete scoped k-NN (`find_similar_by_embedding_scoped`), which
    /// over-fetches until it has the requested number of genuinely in-scope
    /// results (never k-then-drop); scores and ranking order match the unscoped
    /// search. Offset paging applies; it does not compose with the #3360 cursor
    /// or #3353 token-budget shaping in v1.
    fn handle_find_similar_scoped(
        &self,
        req: &FindSimilarRequest,
        scope: &crate::core::namespace::NamespaceScope,
        k: usize,
        offset: usize,
        prov_filter: Option<&crate::core::ProvenanceFilter>,
    ) -> CallToolResult {
        // Over-fetch one past the page window so `has_more` is exact, capped at
        // the MAX_VECTOR_K resource horizon. When a provenance filter is active
        // fetch the full horizon so the returned top-k are all filter-passing.
        let fetch_k = if prov_filter.is_some() {
            MAX_VECTOR_K.saturating_add(1)
        } else {
            offset
                .saturating_add(k)
                .saturating_add(1)
                .min(MAX_VECTOR_K.saturating_add(1))
        };
        let ranked = match self
            .db
            .find_similar_by_embedding_scoped(&req.embedding, fetch_k, scope)
        {
            Ok(r) => r,
            Err(e) => return self.db_error(e),
        };

        let include_vectors = req.include_vectors.unwrap_or(false);
        let now = time::now();
        let built: Vec<SimilarityResult> = ranked
            .into_iter()
            .filter_map(|(node_id, score)| {
                self.db.get_node(node_id).ok().map(|node| SimilarityResult {
                    node: self.node_to_response(&node, include_vectors, now),
                    score,
                    score_breakdown: None,
                })
            })
            .filter(|r| prov_filter.is_none_or(|f| f.matches(r.node.provenance.as_ref())))
            .collect();

        let has_more = built.len() > offset.saturating_add(k);
        let page: Vec<SimilarityResult> = built.into_iter().skip(offset).take(k).collect();
        let count = page.len();
        let mut response = json!({
            "results": page,
            "count": count,
        });
        Self::attach_completeness(&mut response, offset, k, has_more, None);
        self.success_json(response)
    }

    /// Provenance-weighted fused `find_similar` (Issue #3372). Re-ranks the
    /// candidate set (scoped or unscoped, per #3349) by the fused score of
    /// similarity × provenance-confidence × temporal-recency, attaching a
    /// `score_breakdown` to each result. Composes with the #3348 provenance
    /// filter (applied per-candidate before paging) and #3349 namespace scoping.
    ///
    /// v1: `find_similar` has no temporal params, so recency is evaluated
    /// against the current wallclock (AS-OF fusion is satisfied on
    /// `hybrid_query`). The queried index must use the Cosine metric.
    #[cfg(feature = "semantic-retrieval-fusion")]
    fn handle_find_similar_fused(
        &self,
        args: &serde_json::Value,
        policy: &crate::db::fusion::FusionPolicy,
        prov_filter: Option<&crate::core::ProvenanceFilter>,
    ) -> CallToolResult {
        use crate::db::fusion::fused_horizon;

        let req: FindSimilarRequest = match serde_json::from_value(args.clone()) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let k = req.k.unwrap_or(DEFAULT_VECTOR_K).clamp(1, MAX_VECTOR_K);
        let offset = req
            .offset
            .unwrap_or(0)
            .min(MAX_PAGINATION_OFFSET)
            .min(MAX_VECTOR_K.saturating_sub(k));

        if !self.db.is_vector_index_enabled_for(&req.property_name) {
            return self.failed_precondition(&format!(
                "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                req.property_name
            ));
        }
        if let Err(e) = self.validate_embedding_dimensions(&req.embedding, &req.property_name) {
            return self.invalid_argument(&e);
        }
        // Fusion's similarity term assumes an already-`[0,1]` score, true only
        // for Cosine; reject other metrics rather than distorting the term
        // (mirrors `FusionError::UnsupportedMetric`).
        if let Some(err) = self.fusion_metric_precondition(&req.property_name) {
            return err;
        }

        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();

        // Over-fetch a `k`-scaled horizon (never a fixed page window) so a
        // geometrically-far but high-trust candidate can still win (AC4).
        let horizon = fused_horizon(k);
        let candidates: Vec<(NodeId, f32)> =
            if matches!(scope, crate::core::namespace::NamespaceScope::All) {
                match self.db.similarity_search(
                    crate::SimilarityQuery::from_embedding(req.embedding.clone()).k(horizon),
                ) {
                    Ok(r) => r,
                    Err(e) => return self.db_error(e),
                }
            } else {
                match self
                    .db
                    .find_similar_by_embedding_scoped(&req.embedding, horizon, &scope)
                {
                    Ok(r) => r,
                    Err(e) => return self.db_error(e),
                }
            };

        let include_vectors = req.include_vectors.unwrap_or(false);
        let now = time::now();
        let reference_now = now.wallclock();

        // Fuse each candidate, keeping the fused score alongside the result so we
        // can sort by it. Provenance-filter per-candidate (before paging).
        let mut scored: Vec<(f64, NodeId, SimilarityResult)> = Vec::with_capacity(candidates.len());
        for (node_id, similarity) in candidates {
            let Ok(node) = self.db.get_node(node_id) else {
                continue;
            };
            let response = self.node_to_response(&node, include_vectors, now);
            if !prov_filter.is_none_or(|f| f.matches(response.provenance.as_ref())) {
                continue;
            }
            let (confidence, valid_from_micros, recency_defaulted) =
                self.node_fusion_inputs(node.current_version);
            let recency = if recency_defaulted {
                crate::db::fusion::DEFAULT_NEUTRAL_RECENCY
            } else {
                policy.recency(reference_now, valid_from_micros)
            };
            let mut breakdown = policy.fuse(f64::from(similarity), confidence, recency);
            breakdown.recency_defaulted = recency_defaulted;
            let result = SimilarityResult {
                node: response,
                score: similarity,
                score_breakdown: Some(Self::fusion_breakdown_to_json(&breakdown)),
            };
            scored.push((breakdown.fused, node_id, result));
        }

        // Total-order sort by fused score (descending), stable node-id tie-break.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let has_more = scored.len() > offset.saturating_add(k);
        let page: Vec<SimilarityResult> = scored
            .into_iter()
            .skip(offset)
            .take(k)
            .map(|(_, _, r)| r)
            .collect();
        let count = page.len();
        let mut response = json!({
            "results": page,
            "count": count,
        });
        Self::attach_completeness(&mut response, offset, k, has_more, None);
        self.success_json(response)
    }

    /// Read a candidate node's fusion inputs — `(confidence, valid_from_micros,
    /// recency_defaulted)` — from its current version (Issue #3372). Mirrors the
    /// Rust-API `node_fusion_metadata`; when metadata cannot be loaded,
    /// `recency_defaulted` is set so the caller substitutes a **neutral**
    /// recency (never a maximum-recency boost).
    #[cfg(feature = "semantic-retrieval-fusion")]
    fn node_fusion_inputs(&self, version_id: VersionId) -> (Option<f64>, i64, bool) {
        match self.db.get_node_version_read_metadata(version_id) {
            Ok(Some((provenance, interval))) => (
                provenance.and_then(|p| p.confidence()),
                interval.valid_time().start().wallclock(),
                false,
            ),
            _ => (None, 0, true),
        }
    }

    // ========================================================================
    // Embedding generation & text semantic search (Issue #2906)
    //
    // Design A: all five tools are advertised and dispatched unconditionally.
    // The real work lives under `#[cfg(feature = "embeddings")]`; under
    // `#[cfg(not(feature = "embeddings"))]` each handler returns a structured
    // unavailable-feature error (mirroring how `handle_query` returns
    // `language_unavailable` when the `cypher` feature is off), so the tool
    // inventory is a single flat count regardless of feature selection.
    // ========================================================================

    /// Structured error returned by the embedding tools when the `embeddings`
    /// feature was not compiled into this build (Design A, Issue #2906).
    #[cfg(not(feature = "embeddings"))]
    fn embeddings_unavailable(&self) -> CallToolResult {
        self.error_result(
            McpError::new(
                McpErrorCode::FailedPrecondition,
                "Embedding generation is unavailable: this server was built without the \
                 `embeddings` feature. Rebuild with `--features embeddings` (and configure a \
                 model) to use embed_query, embed_text, semantic_search, \
                 create_node_with_embedding, and update_node_embedding.",
            )
            .details(json!({ "feature": "embeddings", "available": false })),
        )
    }

    /// Run an embedding future to completion on the ambient multi-threaded
    /// Tokio runtime, bridging the synchronous MCP dispatch path to the async
    /// `embed_anything` API (Issue #2906).
    ///
    /// Returns `Err(CallToolResult)` — a structured `UNAVAILABLE` — instead of
    /// panicking when there is no current runtime or the runtime is
    /// single-threaded. `tokio::task::block_in_place` (which lets us block the
    /// current worker on the embedding future without stalling the whole
    /// runtime) is only legal on a **multi-thread** runtime worker; it panics on
    /// a current-thread runtime. That is exactly why the current-thread flavor
    /// is pre-checked here and short-circuited to `UNAVAILABLE` rather than
    /// allowed to reach `block_in_place`. The `aletheia-mcp` binary and
    /// `aletheia-server` both run on a multi-thread runtime, so the happy path
    /// always applies there.
    #[cfg(feature = "embeddings")]
    #[allow(clippy::result_large_err)]
    fn block_on_embedding<F>(&self, fut: F) -> std::result::Result<F::Output, CallToolResult>
    where
        F: std::future::Future,
    {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                Ok(tokio::task::block_in_place(move || handle.block_on(fut)))
            }
            _ => Err(self.error_result(McpError::new(
                McpErrorCode::Unavailable,
                "No multi-threaded async runtime is available to run embedding generation. \
                 Serve the MCP/HTTP surface on a multi-threaded Tokio runtime.",
            ))),
        }
    }

    /// The configured embedder, or a structured `FAILED_PRECONDITION` result
    /// when none has been set (Issue #2906).
    #[cfg(feature = "embeddings")]
    #[allow(clippy::result_large_err)]
    fn require_embedder(
        &self,
    ) -> std::result::Result<Arc<crate::embeddings::Embedder>, CallToolResult> {
        self.embedder.as_ref().map(Arc::clone).ok_or_else(|| {
            self.failed_precondition(
                "No embedding model is configured on this server. Configure one via \
                 AletheiaMcpServer::with_embedder (or the server's embedding-model option) \
                 before using the embedding tools.",
            )
        })
    }

    /// Embed a single string into one dense vector (Issue #2906).
    #[cfg(feature = "embeddings")]
    fn handle_embed_query(&self, args: serde_json::Value) -> CallToolResult {
        let req: EmbedQueryRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        if req.text.len() > MAX_EMBED_TEXT_BYTES {
            return self.invalid_argument(&format!(
                "text exceeds the maximum of {} bytes (got {})",
                MAX_EMBED_TEXT_BYTES,
                req.text.len()
            ));
        }
        let embedder = match self.require_embedder() {
            Ok(e) => e,
            Err(result) => return result,
        };

        let text = req.text;
        let embedded = match self.block_on_embedding(async move {
            crate::embeddings::embed_query(&[text.as_str()], embedder.as_ref(), None).await
        }) {
            Ok(r) => r,
            Err(result) => return result,
        };
        let data = match embedded {
            Ok(d) => d,
            Err(e) => {
                // Do not leak upstream provider/model/path detail to the caller.
                eprintln!("embed_query: embedding generation failed: {e}");
                return self.error_result(McpError::new(
                    McpErrorCode::Internal,
                    "embedding generation failed",
                ));
            }
        };
        match crate::embeddings::embed_data_to_dense_iter(data, Some(1)).next() {
            Some(Ok(dense)) => {
                let dim = dense.embedding.len();
                self.success_json(json!({ "embedding": dense.embedding, "dim": dim }))
            }
            Some(Err(e)) => self.failed_precondition(&format!(
                "Configured embedding model produced an unsupported (non-dense) result: {e}"
            )),
            None => self.error_result(McpError::new(
                McpErrorCode::Internal,
                "Embedding model returned no result",
            )),
        }
    }

    /// Embed multiple texts with real chunk expansion (Issue #2906): each input
    /// document is split into contiguous character windows
    /// ([`split_text_into_chunks`]) and every chunk is embedded independently
    /// via [`process_chunks`](crate::embeddings::process_chunks), so a long
    /// document produces MULTIPLE embeddings rather than one truncated vector.
    /// Results are aligned to their source chunk through
    /// [`EmbedData`](crate::embeddings::EmbedData) (`.text`/`.metadata`), never
    /// a positional zip; `metadata` carries the originating `source_index` and
    /// `chunk_index`. `max_chunks` is a hard cap on the returned embeddings; the
    /// response sets `truncated: true` when the cap trims the expansion. A
    /// `max_chunks` of `0` is rejected as INVALID_ARGUMENT.
    #[cfg(feature = "embeddings")]
    fn handle_embed_text(&self, args: serde_json::Value) -> CallToolResult {
        let req: EmbedTextRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        if req.texts.is_empty() {
            return self.invalid_argument("texts must contain at least one string");
        }
        if req.texts.len() > MAX_EMBED_TEXTS {
            return self.invalid_argument(&format!(
                "texts exceeds the maximum of {} entries (got {})",
                MAX_EMBED_TEXTS,
                req.texts.len()
            ));
        }
        if let Some(oversized) = req.texts.iter().find(|t| t.len() > MAX_EMBED_TEXT_BYTES) {
            return self.invalid_argument(&format!(
                "a text entry exceeds the maximum of {} bytes (got {})",
                MAX_EMBED_TEXT_BYTES,
                oversized.len()
            ));
        }
        // A zero cap is a caller error: it can never return anything useful.
        if req.max_chunks == Some(0) {
            return self.invalid_argument("max_chunks must be greater than zero");
        }
        let chunk_cap = req
            .max_chunks
            .map_or(MAX_EMBED_CHUNKS, |m| m.min(MAX_EMBED_CHUNKS));
        let embedder = match self.require_embedder() {
            Ok(e) => e,
            Err(result) => return result,
        };

        // Chunk-expand each input document into contiguous character windows,
        // preserving source alignment via metadata (never a positional zip).
        // Each chunk is embedded independently and re-aligned to its own
        // `EmbedData` below, so a long document yields MULTIPLE embeddings
        // instead of being silently truncated to one vector (Issue #2906 AC).
        let mut chunk_texts: Vec<String> = Vec::new();
        let mut chunk_meta: Vec<Option<std::collections::HashMap<String, String>>> = Vec::new();
        for (source_index, text) in req.texts.iter().enumerate() {
            for (chunk_index, chunk) in split_text_into_chunks(text, EMBED_CHUNK_SIZE_CHARS)
                .into_iter()
                .enumerate()
            {
                let mut meta = std::collections::HashMap::new();
                meta.insert("source_index".to_string(), source_index.to_string());
                meta.insert("chunk_index".to_string(), chunk_index.to_string());
                chunk_texts.push(chunk);
                chunk_meta.push(Some(meta));
            }
        }
        // Honor `max_chunks` as a hard cap on the returned embeddings, and
        // disclose when the cap trimmed the expansion.
        let truncated = chunk_texts.len() > chunk_cap;
        if truncated {
            chunk_texts.truncate(chunk_cap);
            chunk_meta.truncate(chunk_cap);
        }

        let embedded = match self.block_on_embedding(async move {
            crate::embeddings::process_chunks(&chunk_texts, &chunk_meta, &embedder, None, None)
                .await
        }) {
            Ok(r) => r,
            Err(result) => return result,
        };
        let data = match embedded {
            Ok(d) => d,
            Err(e) => {
                // Do not leak upstream provider/model/path detail to the caller.
                eprintln!("embed_text: embedding generation failed: {e}");
                return self.error_result(McpError::new(
                    McpErrorCode::Internal,
                    "embedding generation failed",
                ));
            }
        };

        // `process_chunks` returns a fresh `Arc<Vec<EmbedData>>` (refcount 1),
        // so `try_unwrap` recovers the owned vector without cloning.
        let data = Arc::try_unwrap(data).unwrap_or_else(|arc| arc.as_ref().clone());
        let mut chunks = Vec::new();
        for item in crate::embeddings::embed_data_to_dense_iter(data, None) {
            match item {
                Ok(dense) => {
                    let dim = dense.embedding.len();
                    chunks.push(json!({
                        "text": dense.text,
                        "metadata": dense.metadata,
                        "embedding": dense.embedding,
                        "dim": dim,
                    }));
                }
                Err(e) => {
                    return self.failed_precondition(&format!(
                        "Configured embedding model produced an unsupported (non-dense) result: {e}"
                    ));
                }
            }
        }
        let count = chunks.len();
        self.success_json(json!({ "chunks": chunks, "count": count, "truncated": truncated }))
    }

    /// Embed `query_text` and reuse the exact `find_similar` path so the
    /// response envelope is byte-identical (Issue #2906).
    #[cfg(feature = "embeddings")]
    fn handle_semantic_search(&self, args: serde_json::Value) -> CallToolResult {
        let req: SemanticSearchRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        if req.query_text.len() > MAX_EMBED_TEXT_BYTES {
            return self.invalid_argument(&format!(
                "query_text exceeds the maximum of {} bytes (got {})",
                MAX_EMBED_TEXT_BYTES,
                req.query_text.len()
            ));
        }
        // Validate the target index up front (cheap, no model needed) so a
        // missing index is a clear FAILED_PRECONDITION even before embedding.
        if !self.db.is_vector_index_enabled_for(&req.property_name) {
            return self.failed_precondition(&format!(
                "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                req.property_name
            ));
        }
        let embedder = match self.require_embedder() {
            Ok(e) => e,
            Err(result) => return result,
        };

        let query_text = req.query_text;
        let embedded = match self.block_on_embedding(async move {
            crate::embeddings::embed_query(&[query_text.as_str()], embedder.as_ref(), None).await
        }) {
            Ok(r) => r,
            Err(result) => return result,
        };
        let data = match embedded {
            Ok(d) => d,
            Err(e) => {
                // Do not leak upstream provider/model/path detail to the caller.
                eprintln!("semantic_search: embedding generation failed: {e}");
                return self.error_result(McpError::new(
                    McpErrorCode::Internal,
                    "embedding generation failed",
                ));
            }
        };
        let embedding =
            match crate::embeddings::to_dense_iter(data.into_iter().map(|d| d.embedding), Some(1))
                .next()
            {
                Some(Ok(v)) => v,
                Some(Err(e)) => {
                    return self.failed_precondition(&format!(
                        "Configured embedding model produced an unsupported (non-dense) result: {e}"
                    ));
                }
                None => {
                    return self.error_result(McpError::new(
                        McpErrorCode::Internal,
                        "Embedding model returned no result",
                    ));
                }
            };
        // Dimension mismatch is a caller-fault INVALID_ARGUMENT.
        if let Err(e) = self.validate_embedding_dimensions(&embedding, &req.property_name) {
            return self.invalid_argument(&e);
        }

        // Delegate to the exact find_similar handler with a rebuilt request so
        // the success envelope (results/score/temporal/#3220 elision/#3226
        // pagination) is byte-identical.
        let mut find_args = serde_json::Map::new();
        find_args.insert("property_name".to_string(), json!(req.property_name));
        find_args.insert("embedding".to_string(), json!(embedding));
        if let Some(k) = req.k {
            find_args.insert("k".to_string(), json!(k));
        }
        if let Some(offset) = req.offset {
            find_args.insert("offset".to_string(), json!(offset));
        }
        if let Some(include_vectors) = req.include_vectors {
            find_args.insert("include_vectors".to_string(), json!(include_vectors));
        }
        self.handle_find_similar(serde_json::Value::Object(find_args))
    }

    /// Embed `text` and store it as a vector property on a new node, reusing
    /// the standard `create_node_with_options` write path (Issue #2906).
    #[cfg(feature = "embeddings")]
    fn handle_create_node_with_embedding(&self, args: serde_json::Value) -> CallToolResult {
        let req: CreateNodeWithEmbeddingRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        if req.text.len() > MAX_EMBED_TEXT_BYTES {
            return self.invalid_argument(&format!(
                "text exceeds the maximum of {} bytes (got {})",
                MAX_EMBED_TEXT_BYTES,
                req.text.len()
            ));
        }
        let embedder = match self.require_embedder() {
            Ok(e) => e,
            Err(result) => return result,
        };

        let valid_from = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v,
            Err(result) => return result,
        };
        let provenance = match self.parse_opt_provenance(req.provenance) {
            Ok(p) => p,
            Err(result) => return result,
        };

        let embedding = match self.embed_one(&req.text, embedder) {
            Ok(v) => v,
            Err(result) => return result,
        };

        // Build the property map (non-embedding props) then attach the
        // generated vector under `embedding_property`.
        let base = match &req.properties {
            Some(p) => match self.json_to_property_map(p) {
                Ok(map) => map,
                Err(e) => return self.property_map_error(e),
            },
            None => PropertyMap::default(),
        };
        let properties = match base
            .builder()
            .try_insert_vector(&req.embedding_property, &embedding)
        {
            Ok(b) => b.build(),
            Err(e) => {
                return self.invalid_argument(&format!("Invalid embedding property: {}", e));
            }
        };

        let mut options = crate::api::transaction::WriteRequestOptions::new();
        if let Some(valid_from) = valid_from {
            options = options.with_valid_from(valid_from);
        }
        if let Some(provenance) = provenance {
            options = options.with_provenance(provenance);
        }

        match self
            .db
            .create_node_with_options(&req.label, properties, options)
        {
            Ok(node_id) => match self.db.get_node(node_id) {
                Ok(node) => {
                    let now = time::now();
                    // Write path returns the full vector it just wrote.
                    let response = self.node_to_response(&node, true, now);
                    self.success_json(
                        serde_json::to_value(&response)
                            .expect("response serialization should not fail"),
                    )
                }
                Err(e) => self.db_error(e),
            },
            Err(e) => self.db_error(e),
        }
    }

    /// Embed `text` and update ONLY the embedding property of an existing node,
    /// preserving all other properties (Issue #2906).
    ///
    /// The embedding is generated FIRST (a slow, network/CPU-bound step holding
    /// no snapshot or lock), then the read-merge-write runs inside a SINGLE
    /// [`write`](crate::db::AletheiaDB::write) transaction: `update_node`
    /// replaces every property, so the existing properties are re-read from the
    /// transaction's own snapshot and merged before overriding the embedding.
    /// Doing the read and the write in one transaction closes the lost-update
    /// race — a concurrent writer that commits in the window is caught by
    /// commit-time conflict detection instead of being silently overwritten.
    #[cfg(feature = "embeddings")]
    fn handle_update_node_embedding(&self, args: serde_json::Value) -> CallToolResult {
        let req: UpdateNodeEmbeddingRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        if req.text.len() > MAX_EMBED_TEXT_BYTES {
            return self.invalid_argument(&format!(
                "text exceeds the maximum of {} bytes (got {})",
                MAX_EMBED_TEXT_BYTES,
                req.text.len()
            ));
        }
        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        let embedder = match self.require_embedder() {
            Ok(e) => e,
            Err(result) => return result,
        };

        let valid_from = match self.parse_opt_timestamp("valid_time", &req.valid_time) {
            Ok(v) => v,
            Err(result) => return result,
        };

        // Embed FIRST (slow, network/CPU-bound), holding no snapshot or lock,
        // so the transaction window below stays as short as possible.
        let embedding = match self.embed_one(&req.text, embedder) {
            Ok(v) => v,
            Err(result) => return result,
        };

        // Stamp the authenticated principal (if any) onto the version.
        let provenance = match self.parse_opt_provenance(None) {
            Ok(p) => p,
            Err(result) => return result,
        };

        // Read + merge + write inside ONE transaction so the merge sees the
        // transaction's own snapshot and commit-time conflict detection guards
        // against a concurrent writer being silently lost (Issue #2906 fix).
        let embedding_property = req.embedding_property;

        /// Local error type bridging the property-builder failure (a caller
        /// fault) and storage errors through the `db.write` closure.
        enum UpdateEmbeddingError {
            Db(crate::core::error::Error),
            Vector(String),
        }
        impl From<crate::core::error::Error> for UpdateEmbeddingError {
            fn from(e: crate::core::error::Error) -> Self {
                UpdateEmbeddingError::Db(e)
            }
        }

        let write_result = self.db.write(|tx| {
            use crate::api::transaction::ReadOps;
            let node = tx.get_node(node_id)?;
            let properties = node
                .properties
                .builder()
                .try_insert_vector(&embedding_property, &embedding)
                .map_err(|e| {
                    UpdateEmbeddingError::Vector(format!("Invalid embedding property: {e}"))
                })?
                .build();

            let mut options = crate::api::transaction::WriteRequestOptions::new();
            if let Some(valid_from) = valid_from {
                options = options.with_valid_from(valid_from);
            }
            if let Some(provenance) = provenance {
                options = options.with_provenance(provenance);
            }
            tx.update_node_with_options(node_id, properties, options)?;
            Ok::<(), UpdateEmbeddingError>(())
        });

        match write_result {
            Ok(()) => match self.db.get_node(node_id) {
                Ok(node) => {
                    let now = time::now();
                    // Write path returns the full vector it just wrote.
                    let response = self.node_to_response(&node, true, now);
                    self.success_json(
                        serde_json::to_value(&response)
                            .expect("response serialization should not fail"),
                    )
                }
                Err(e) => self.db_error(e),
            },
            Err(UpdateEmbeddingError::Vector(msg)) => self.invalid_argument(&msg),
            Err(UpdateEmbeddingError::Db(e)) => self.db_error(e),
        }
    }

    /// Embed one text into a single dense vector, mapping upstream/non-dense
    /// failures to structured error results (Issue #2906 shared helper).
    #[cfg(feature = "embeddings")]
    #[allow(clippy::result_large_err)]
    fn embed_one(
        &self,
        text: &str,
        embedder: Arc<crate::embeddings::Embedder>,
    ) -> std::result::Result<Vec<f32>, CallToolResult> {
        let owned = text.to_string();
        let embedded = self.block_on_embedding(async move {
            crate::embeddings::embed_query(&[owned.as_str()], embedder.as_ref(), None).await
        })?;
        let data = embedded.map_err(|e| {
            // Do not leak upstream provider/model/path detail to the caller.
            eprintln!("embed_one: embedding generation failed: {e}");
            self.error_result(McpError::new(
                McpErrorCode::Internal,
                "embedding generation failed",
            ))
        })?;
        match crate::embeddings::to_dense_iter(data.into_iter().map(|d| d.embedding), Some(1))
            .next()
        {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(self.failed_precondition(&format!(
                "Configured embedding model produced an unsupported (non-dense) result: {e}"
            ))),
            None => Err(self.error_result(McpError::new(
                McpErrorCode::Internal,
                "Embedding model returned no result",
            ))),
        }
    }

    // Feature-off variants: return the structured unavailable-feature error.

    #[cfg(not(feature = "embeddings"))]
    fn handle_embed_query(&self, _args: serde_json::Value) -> CallToolResult {
        self.embeddings_unavailable()
    }

    #[cfg(not(feature = "embeddings"))]
    fn handle_embed_text(&self, _args: serde_json::Value) -> CallToolResult {
        self.embeddings_unavailable()
    }

    #[cfg(not(feature = "embeddings"))]
    fn handle_semantic_search(&self, _args: serde_json::Value) -> CallToolResult {
        self.embeddings_unavailable()
    }

    #[cfg(not(feature = "embeddings"))]
    fn handle_create_node_with_embedding(&self, _args: serde_json::Value) -> CallToolResult {
        self.embeddings_unavailable()
    }

    #[cfg(not(feature = "embeddings"))]
    fn handle_update_node_embedding(&self, _args: serde_json::Value) -> CallToolResult {
        self.embeddings_unavailable()
    }

    fn handle_enable_vector_index(&self, args: serde_json::Value) -> CallToolResult {
        let req: EnableVectorIndexRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let distance_metric = match req.distance_metric.as_deref().unwrap_or("cosine") {
            "euclidean" => DistanceMetric::Euclidean,
            "dot" | "dot_product" => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        };

        let config = HnswConfig::new(req.dimensions, distance_metric);

        match self.db.enable_vector_index(&req.property_name, config) {
            Ok(()) => self.success_json(json!({
                "success": true,
                "property_name": req.property_name,
                "dimensions": req.dimensions,
                "distance_metric": req.distance_metric.unwrap_or_else(|| "cosine".to_string())
            })),
            Err(e) => self.db_error(e),
        }
    }

    fn handle_list_vector_indexes(&self, _args: serde_json::Value) -> CallToolResult {
        let indexes = self.db.list_vector_indexes();
        let index_list: Vec<serde_json::Value> = indexes
            .into_iter()
            .map(|info| {
                json!({
                    "property_name": info.property_name,
                    "dimensions": info.dimensions,
                    "distance_metric": format!("{:?}", info.distance_metric)
                })
            })
            .collect();
        self.success_json(json!({
            "indexes": index_list,
            "count": index_list.len()
        }))
    }

    fn handle_enable_unique_constraint(&self, args: serde_json::Value) -> CallToolResult {
        let req: EnableUniqueConstraintRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        match self
            .db
            .unique_constraint(&req.label, &req.property)
            .enable()
        {
            Ok(()) => self.success_json(json!({
                "success": true,
                "label": req.label,
                "property": req.property
            })),
            Err(e) => self.db_error(e),
        }
    }

    fn handle_list_unique_constraints(&self, _args: serde_json::Value) -> CallToolResult {
        let constraints = self.db.list_unique_constraints();
        let list: Vec<serde_json::Value> = constraints
            .into_iter()
            .map(|(label, property)| json!({ "label": label, "property": property }))
            .collect();
        self.success_json(json!({
            "constraints": list,
            "count": list.len()
        }))
    }

    fn handle_get_node_at_time(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetNodeAtTimeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let valid_time = match self.parse_timestamp(&req.valid_time) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&e),
        };

        let tx_time = match self.parse_optional_tx_time(req.transaction_time.as_deref()) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&e),
        };

        // Issue #3349: reconstruct at the coordinate first, then filter by the
        // (immutable) namespace scope; out of scope ⇒ NOT_FOUND. Omitted ⇒ the
        // `default` namespace only (isolated-by-default, #3349 FIX2); `all`
        // imposes no filter.
        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();
        let node_result = match &scope {
            crate::core::namespace::NamespaceScope::All => {
                self.db.get_node_at_time(node_id, valid_time, tx_time)
            }
            scope => self
                .db
                .get_node_at_time_scoped(node_id, valid_time, tx_time, scope),
        };

        match node_result {
            Ok(node) => {
                let now = time::now();
                let response = self.node_to_response(&node, true, now);
                self.success_json(json!({
                    "node": response,
                    "valid_time": req.valid_time,
                    "transaction_time": Self::format_tx_time_response(req.transaction_time)
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_get_edge_at_time(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetEdgeAtTimeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let edge_id = match EdgeId::new(req.edge_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let valid_time = match self.parse_timestamp(&req.valid_time) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&e),
        };

        let tx_time = match self.parse_optional_tx_time(req.transaction_time.as_deref()) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&e),
        };

        // Issue #3349: reconstruct then filter by the (immutable) namespace
        // scope. Omitted ⇒ the `default` namespace only (isolated-by-default,
        // #3349 FIX2); `all` imposes no filter.
        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();
        let edge_result = match &scope {
            crate::core::namespace::NamespaceScope::All => {
                self.db.get_edge_at_time(edge_id, valid_time, tx_time)
            }
            scope => self
                .db
                .get_edge_at_time_scoped(edge_id, valid_time, tx_time, scope),
        };

        match edge_result {
            Ok(edge) => {
                let now = time::now();
                let response = self.edge_to_response(&edge, true, now);
                self.success_json(json!({
                    "edge": response,
                    "valid_time": req.valid_time,
                    "transaction_time": Self::format_tx_time_response(req.transaction_time)
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    /// Cursor-mode `find_nodes_at_time` (Issue #3360): snapshot-anchored keyset
    /// paging. The snapshot is the caller's requested `(valid_time,
    /// transaction_time)` -- already a point-in-time read, so consistency is
    /// native; continuation just seeks by node id past the last returned.
    fn handle_find_nodes_at_time_cursor(&self, args: &serde_json::Value) -> CallToolResult {
        let now = time::now();

        if let Some(token) = args.get("cursor").and_then(|v| v.as_str()) {
            let payload = match self.cursors.decode(token, "find_nodes_at_time") {
                Ok(p) => p,
                Err(e) => return self.error_result(e),
            };
            let label = payload.filters["label"].as_str().unwrap_or("").to_string();
            let property_key = payload.filters["property_key"].as_str().map(str::to_string);
            let property_value = match &payload.filters["property_value"] {
                serde_json::Value::Null => None,
                v => Some(v.clone()),
            };
            let include_vectors = payload.filters["include_vectors"]
                .as_bool()
                .unwrap_or(false);
            let snapshot = (Timestamp::from(payload.svt), Timestamp::from(payload.stt));
            let candidates = match self.fetch_node_candidates(
                &label,
                &property_key,
                &property_value,
                snapshot,
            ) {
                Ok(c) => c,
                Err(result) => return result,
            };
            return self.emit_node_cursor_page(
                "find_nodes_at_time",
                snapshot,
                payload.after,
                payload.limit,
                include_vectors,
                payload.filters.clone(),
                candidates,
                payload.cid,
                now,
            );
        }

        // First page: validate filter combo and pin the requested coordinate.
        let label = Self::arg_str(args, "label").unwrap_or_default();
        let property_key = Self::arg_str(args, "property_key");
        let property_value = Self::arg_value(args, "property_value");
        if property_key.is_some() != property_value.is_some() {
            return self.invalid_argument(
                "Both 'property_key' and 'property_value' are required together",
            );
        }
        let valid_time_str = Self::arg_str(args, "valid_time").unwrap_or_default();
        let valid_time = match self.parse_timestamp(&valid_time_str) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&format!("Invalid valid_time: {}", e)),
        };
        let tx_time = match self
            .parse_optional_tx_time(Self::arg_str(args, "transaction_time").as_deref())
        {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&format!("Invalid transaction_time: {}", e)),
        };
        let limit = Self::arg_limit(args);
        let include_vectors = args
            .get("include_vectors")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let snapshot = (valid_time, tx_time);
        let candidates =
            match self.fetch_node_candidates(&label, &property_key, &property_value, snapshot) {
                Ok(c) => c,
                Err(result) => return result,
            };
        let filters =
            Self::node_scan_filters(&label, &property_key, &property_value, include_vectors);
        self.emit_node_cursor_page(
            "find_nodes_at_time",
            snapshot,
            None,
            limit,
            include_vectors,
            filters,
            candidates,
            String::new(),
            now,
        )
    }

    /// Find nodes by label (and optional exact property match) as of a
    /// bi-temporal point (Issue #3236).
    ///
    /// Validation mirrors `handle_list_nodes` (both-or-neither property
    /// filter, same limit/offset clamps); the temporal reconstruction is
    /// delegated to `AletheiaDB::find_nodes_at_time` /
    /// `find_nodes_by_property_at`, which reconstruct each candidate from
    /// the historical version visible at the queried coordinate -- so nodes
    /// deleted from current state are still found when both dimensions
    /// anchor before the deletion. The candidate set is capped at the same
    /// `max_schema_as_of_entities` limit bi-temporal `get_schema` uses; when
    /// truncated, the response discloses it via `sampled: true` and
    /// `total_matching`/`has_more` count matches within the sampled
    /// candidate set only.
    fn handle_find_nodes_at_time(&self, args: serde_json::Value) -> CallToolResult {
        // Snapshot-anchored cursor paging (Issue #3360); offset paging below
        // is unchanged for backward compatibility.
        if Self::cursor_requested(&args) {
            // Issue #3349: a narrowing namespace scope does not compose with the
            // #3360 cursor path in v1 — fail closed.
            if let Err(result) = self.reject_unsupported_scope(
                &args.get("namespace").cloned(),
                "namespace scope does not compose with cursor paging (use_cursor) in v1; use \
                 offset/limit paging with the scope instead, or \"all\".",
            ) {
                return result;
            }
            return self.handle_find_nodes_at_time_cursor(&args);
        }

        let req: FindNodesAtTimeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        // Issue #3349 (FIX2): an omitted scope resolves to the `default`
        // namespace only (isolated-by-default) and routes each candidate
        // (reconstructed at the coordinate) through the immutable namespace
        // filter, exactly like an explicit narrowing scope. For a pre-namespace
        // database (all data is `default`) the candidate set is unchanged. Only
        // the `all` selector falls through unscoped.
        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();
        let scope = if matches!(scope, crate::core::namespace::NamespaceScope::All) {
            None
        } else {
            Some(scope)
        };

        // Validate property filter: both key and value are required together
        // (mirroring list_nodes).
        if req.property_key.is_some() != req.property_value.is_some() {
            return self.invalid_argument(
                "Both 'property_key' and 'property_value' are required together",
            );
        }

        let valid_time = match self.parse_timestamp(&req.valid_time) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&format!("Invalid valid_time: {}", e)),
        };
        let tx_time = match self.parse_optional_tx_time(req.transaction_time.as_deref()) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&format!("Invalid transaction_time: {}", e)),
        };

        // Apply resource limits exactly like list_nodes: a page must be able
        // to carry a continuation cursor, so the limit is at least 1.
        let limit = req
            .limit
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .clamp(1, MAX_RESULT_LIMIT);
        let offset = req.offset.unwrap_or(0).min(MAX_PAGINATION_OFFSET);

        let matches =
            if let (Some(prop_key), Some(prop_val)) = (&req.property_key, &req.property_value) {
                let property_value = match self.json_to_property_value(prop_val) {
                    Some(v) => v,
                    None => return self.invalid_argument(
                        "Unsupported property_value type. Use strings, numbers, booleans, or null.",
                    ),
                };
                match &scope {
                    Some(scope) => self.db.find_nodes_by_property_at_scoped(
                        &req.label,
                        prop_key,
                        &property_value,
                        valid_time,
                        tx_time,
                        scope,
                    ),
                    None => self.db.find_nodes_by_property_at(
                        &req.label,
                        prop_key,
                        &property_value,
                        valid_time,
                        tx_time,
                    ),
                }
            } else {
                match &scope {
                    Some(scope) => self
                        .db
                        .find_nodes_at_time_scoped(&req.label, valid_time, tx_time, scope),
                    None => self.db.find_nodes_at_time(&req.label, valid_time, tx_time),
                }
            };

        match matches {
            Ok(matches) => {
                // The matching set is already materialized (sorted by node
                // id for stable pagination), so the total is cheap to report
                // and `has_more` is exact *within the candidate set*. When
                // `sampled` is true the candidate enumeration was truncated
                // at the configured cap, so `total_matching` is honest only
                // about the sampled candidates -- the flag discloses that.
                let sampled = matches.sampled;
                let matches = matches.nodes;
                let total_matching = matches.len();
                let include_vectors = req.include_vectors.unwrap_or(false);
                // One request-scoped wallclock for every entity in the
                // response (Issue #3391).
                let now = time::now();
                let nodes: Vec<NodeResponse> = matches
                    .iter()
                    .skip(offset)
                    .take(limit)
                    .map(|node| self.node_to_response(node, include_vectors, now))
                    .collect();

                let has_more = offset.saturating_add(limit) < total_matching;
                let mut response = json!({
                    "nodes": nodes,
                    "count": nodes.len(),
                    "offset": offset,
                    "limit": limit,
                    // Candidate-set truncation disclosure, mirroring
                    // get_schema's `sampled` (same underlying cap).
                    "sampled": sampled,
                    // The resolved coordinate this answer holds at -- the
                    // omitted transaction_time resolves to a concrete "now".
                    "valid_time": Self::format_timestamp_rfc3339(valid_time),
                    "transaction_time": Self::format_timestamp_rfc3339(tx_time),
                });
                Self::attach_completeness(
                    &mut response,
                    offset,
                    limit,
                    has_more,
                    Some(total_matching),
                );
                self.success_json(response)
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_list_changes(&self, args: serde_json::Value) -> CallToolResult {
        let req: ListChangesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let tx_from = match self.parse_timestamp(&req.tx_from) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&format!("Invalid tx_from: {}", e)),
        };
        let tx_to = match self.parse_timestamp(&req.tx_to) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&format!("Invalid tx_to: {}", e)),
        };

        let valid_from = match self.parse_opt_timestamp("valid_from", &req.valid_from) {
            Ok(v) => v,
            Err(resp) => return resp,
        };
        let valid_to = match self.parse_opt_timestamp("valid_to", &req.valid_to) {
            Ok(v) => v,
            Err(resp) => return resp,
        };

        // A page must be able to carry a continuation cursor, so the limit is at least 1.
        let limit = req
            .limit
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .clamp(1, MAX_RESULT_LIMIT);

        let query = ChangeFeedQuery {
            tx_from,
            tx_to,
            valid_from,
            valid_to,
            label: req.label.clone(),
            limit,
            cursor: req.cursor.clone(),
        };

        match self.db.list_changes(&query) {
            Ok(page) => {
                let changes: Vec<serde_json::Value> = page
                    .changes
                    .iter()
                    .map(Self::changefeed_change_json)
                    .collect();

                self.success_json(json!({
                    "changes": changes,
                    "count": page.changes.len(),
                    "next_cursor": page.next_cursor,
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    /// Serialize a single [`ChangeRecord`](crate::core::ChangeRecord) into the
    /// changefeed row JSON shared by `list_changes` and `await_changes`, so both
    /// surfaces emit a byte-identical change shape (Issue #3375).
    fn changefeed_change_json(record: &crate::core::ChangeRecord) -> serde_json::Value {
        json!({
            "entity_id": record.entity_id,
            "version_id": record.version_id,
            "kind": record.kind.as_str(),
            "change_type": record.change_type.as_str(),
            "label": record.label,
            "namespace": record.namespace.as_str(),
            "transaction_time": time::to_iso8601(record.transaction_time()),
            "transaction_time_range": {
                "start": time::to_iso8601(record.transaction_time_range.start()),
                "end": time::to_iso8601(record.transaction_time_range.end()),
            },
            "valid_time_range": {
                "start": time::to_iso8601(record.valid_time_range.start()),
                "end": time::to_iso8601(record.valid_time_range.end()),
            },
        })
    }

    /// Build the `await_changes` success envelope from a set of changes plus the
    /// resume/timeout/has-more signals.
    fn await_changes_success(
        &self,
        changes: &[crate::core::ChangeRecord],
        resume_token: Option<String>,
        timed_out: bool,
        has_more: bool,
    ) -> CallToolResult {
        let rows: Vec<serde_json::Value> =
            changes.iter().map(Self::changefeed_change_json).collect();
        self.success_json(json!({
            "changes": rows,
            "count": changes.len(),
            "resume_token": resume_token,
            "timed_out": timed_out,
            "has_more": has_more,
        }))
    }

    /// Parse the request's `change_types` strings into [`ChangeType`]s, returning
    /// the offending token's error message for any unrecognized value (the
    /// caller wraps it in a structured `INVALID_ARGUMENT`). Returns a small `Err`
    /// (`String`) rather than a `CallToolResult` to keep the `Result` compact.
    fn parse_change_types(raw: &[String]) -> Result<Vec<ChangeType>, String> {
        raw.iter()
            .map(|s| match s.as_str() {
                "created" => Ok(ChangeType::Created),
                "modified" => Ok(ChangeType::Modified),
                "deleted" => Ok(ChangeType::Deleted),
                other => Err(format!(
                    "Invalid change_type '{other}': expected one of created, modified, deleted"
                )),
            })
            .collect()
    }

    /// Long-poll for the next committed changes (the push changefeed's blocking
    /// surface, Issue #3375).
    ///
    /// Stateless per call: subscribe (capturing the committed frontier so no
    /// change between catch-up and blocking is lost), optionally catch up from a
    /// prior `from_token` via `list_changes` (returning immediately if any change
    /// already exists), otherwise block up to the clamped `timeout_ms`. Error
    /// mappings (Issue #3234): a lagged subscription → retriable
    /// `RESOURCE_EXHAUSTED` with `details.resume_token`; a subscribe cap breach →
    /// retriable `UNAVAILABLE`; a malformed `from_token` → `INVALID_ARGUMENT`.
    ///
    /// This synchronous entry is now reached only via `dispatch_tool`
    /// (embedded / programmatic / test callers): the native MCP `call_tool`
    /// seam intercepts `await_changes` and routes it through the event-driven
    /// [`Self::dispatch_await_changes_async`] instead (Issue #3673), so the
    /// `block_in_place` worker bridge below no longer runs on the live MCP
    /// server surface.
    fn handle_await_changes(
        &self,
        args: serde_json::Value,
        principal_override: Option<String>,
    ) -> CallToolResult {
        match self.await_changes_prelude(args, principal_override) {
            AwaitChangesStep::Immediate(result) => *result,
            AwaitChangesStep::Block { sub, timeout } => {
                // Synchronous Condvar long-poll for embedded/programmatic callers
                // (and any non-`call_tool` sync dispatch). SAFETY/why:
                // `recv_timeout` parks this worker for up to 60s. On a multi-thread
                // Tokio runtime, run it inside `block_in_place` so Tokio can spin up
                // a replacement worker; on a current-thread runtime or with no
                // runtime (embedded/programmatic callers) `block_in_place` would
                // panic, so call inline. The event-driven native `call_tool` and
                // HTTP `/changes/await` paths instead use the async
                // [`Self::dispatch_await_changes_async`] wait, which pins no worker
                // and releases promptly on client disconnect (Issue #3673).
                let recv_result = match tokio::runtime::Handle::try_current() {
                    Ok(handle)
                        if handle.runtime_flavor()
                            == tokio::runtime::RuntimeFlavor::MultiThread =>
                    {
                        tokio::task::block_in_place(|| sub.recv_timeout(timeout))
                    }
                    _ => sub.recv_timeout(timeout),
                };
                self.finish_await_recv(&sub, recv_result)
            }
        }
    }

    /// Event-driven async counterpart to [`handle_await_changes`] (Issue #3673).
    ///
    /// Runs the identical subscribe + catch-up prelude, then — for the blocking
    /// leg — awaits [`Subscription::recv_async`] instead of parking a worker on
    /// the synchronous `recv_timeout`. Because the wait is a suspended future, N
    /// concurrent long-polls pin **zero** worker threads (fixing the AC(a)
    /// `block_in_place` pin), and dropping this future on client disconnect drops
    /// the `Subscription` immediately, freeing its per-principal slot without
    /// waiting out the timeout. Used by the native MCP `call_tool` seam and the
    /// HTTP `/changes/await` projection.
    pub(crate) async fn dispatch_await_changes_async(
        &self,
        args: serde_json::Value,
        principal_override: Option<String>,
    ) -> CallToolResult {
        match self.await_changes_prelude(args, principal_override) {
            AwaitChangesStep::Immediate(result) => *result,
            AwaitChangesStep::Block { sub, timeout } => {
                let recv_result = sub.recv_async(timeout).await;
                self.finish_await_recv(&sub, recv_result)
            }
        }
    }

    /// The synchronous prelude shared by [`handle_await_changes`] (sync Condvar
    /// wait) and [`dispatch_await_changes_async`] (event-driven Notify wait),
    /// Issue #3673: it resolves the principal bucket, builds the filter,
    /// subscribes (capturing the frontier), and runs the optional catch-up — all
    /// fast, non-blocking work — returning either an [`AwaitChangesStep::Immediate`]
    /// result or a live [`Subscription`] to wait on. Factoring it out guarantees
    /// both surfaces run byte-identical subscribe/catch-up logic and differ ONLY
    /// in how they wait, so delivery semantics stay unchanged (AC c).
    fn await_changes_prelude(
        &self,
        args: serde_json::Value,
        principal_override: Option<String>,
    ) -> AwaitChangesStep {
        let req: AwaitChangesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => {
                return AwaitChangesStep::Immediate(Box::new(
                    self.invalid_argument(&format!("Invalid arguments: {}", e)),
                ));
            }
        };

        // Resolve the per-principal quota bucket (Issue #3678): the surface-supplied
        // override (HTTP route), else the authenticated MCP session principal, else
        // the shared "anonymous" bucket — so every surface enforces the quota and no
        // single (even anonymous) caller can exhaust the global cap and starve others.
        let principal_key = principal_override
            .or_else(|| self.session_principal().map(|p| p.id))
            .unwrap_or_else(|| "anonymous".to_string());

        // Build the change filter (label/type/change-type dimensions).
        let mut filter = ChangeFilter::all();
        if let Some(labels) = &req.node_labels {
            filter = filter.with_node_labels(labels.iter().cloned());
        }
        if let Some(types) = &req.edge_types {
            filter = filter.with_edge_types(types.iter().cloned());
        }
        if let Some(change_types) = &req.change_types {
            let parsed = match Self::parse_change_types(change_types) {
                Ok(v) => v,
                Err(msg) => {
                    return AwaitChangesStep::Immediate(Box::new(self.invalid_argument(&msg)));
                }
            };
            filter = filter.with_change_types(parsed);
        }

        // Namespace scope (Issue #3349, PR3c). An omitted scope resolves to the
        // `default` namespace only (isolated-by-default, mirroring the PR3a read
        // tools) via `.unwrap_or_default()`; `"all"` imposes no namespace filter.
        // A malformed / empty scope is a structured INVALID_ARGUMENT (never a
        // silently-unscoped subscription).
        let scope = match self.parse_opt_scope(&req.namespace) {
            Ok(opt) => opt.unwrap_or_default(),
            Err(result) => return AwaitChangesStep::Immediate(Box::new(result)),
        };
        filter = filter.with_namespace_scope(scope);

        let limit = req
            .limit
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .clamp(1, MAX_RESULT_LIMIT);

        // Subscribe FIRST so the frontier is captured before the catch-up read:
        // any change committed after this point is buffered in the subscription,
        // so the catch-up→block handoff is gap-free. Clone the filter into the
        // subscription so the handler retains a copy to post-filter the catch-up
        // page with (the blocking leg is filtered by the subscription itself, so
        // the catch-up leg must apply the SAME predicate — Fix 1).
        let sub = match self
            .db
            .subscribe_changes_for_principal(Some(&principal_key), filter.clone())
        {
            Ok(s) => s,
            Err(e) => {
                // A per-principal quota breach (Issue #3678) falls through to
                // `db_error`, which classifies it as a retriable RESOURCE_EXHAUSTED
                // with `details {principal, current, limit}`. A GLOBAL cap breach is
                // transient (another consumer may disconnect): override the default
                // FAILED_PRECONDITION with a retriable UNAVAILABLE carrying the
                // capacity metadata.
                if let crate::core::error::Error::Storage(
                    crate::core::error::StorageError::CapacityExceeded {
                        resource,
                        current,
                        limit,
                    },
                ) = &e
                {
                    return AwaitChangesStep::Immediate(Box::new(
                        self.error_result(
                            McpError::new(McpErrorCode::Unavailable, e.to_string())
                                .retriable(true)
                                .details(json!({
                                    "resource": resource,
                                    "current": current,
                                    "limit": limit,
                                })),
                        ),
                    ));
                }
                return AwaitChangesStep::Immediate(Box::new(self.db_error(e)));
            }
        };

        // Catch-up path: if the caller has a resume token, pull everything
        // committed strictly after it via the durable `list_changes` feed and
        // return immediately when anything is available.
        if let Some(token) = req.from_token.as_deref() {
            let decoded = match ChangeCursor::decode(token) {
                Ok(c) => c,
                Err(_) => {
                    return AwaitChangesStep::Immediate(Box::new(
                        self.invalid_argument("Invalid from_token: malformed continuation token"),
                    ));
                }
            };
            let query = ChangeFeedQuery {
                tx_from: Timestamp::from(decoded.tx_wallclock),
                tx_to: time::now(),
                valid_from: None,
                valid_to: None,
                label: None,
                limit,
                cursor: Some(token.to_string()),
            };
            match self.db.list_changes(&query) {
                // The window scanned rows: post-filter them with the SAME
                // ChangeFilter the blocking leg's subscription applies, so a
                // filtered subscription's resume path never returns
                // non-matching changes (Fix 1). The resume token is the last
                // SCANNED cursor (page.next_cursor, or the last scanned row's
                // cursor when the scan reached the window's end) — NOT the last
                // matching row — so a page that filters down to nothing still
                // advances the caller past every scanned row (no re-scan/stall).
                Ok(page) if !page.changes.is_empty() => {
                    let filtered: Vec<crate::core::ChangeRecord> = page
                        .changes
                        .iter()
                        .filter(|r| filter.matches(r))
                        .cloned()
                        .collect();
                    let resume = page
                        .next_cursor
                        .clone()
                        .or_else(|| page.changes.last().map(|r| r.cursor().encode()));
                    return AwaitChangesStep::Immediate(Box::new(self.await_changes_success(
                        &filtered,
                        resume,
                        false,
                        page.next_cursor.is_some(),
                    )));
                }
                Ok(_) => { /* nothing buffered yet — fall through to block */ }
                Err(e) => return AwaitChangesStep::Immediate(Box::new(self.db_error(e))),
            }
        }

        // Block for the next change up to the clamped timeout (default 25s, hard
        // cap 60s). `recv_*(0)` returns instantly (Ok(empty)).
        let timeout_ms = req.timeout_ms.unwrap_or(25_000).min(60_000);
        let timeout = std::time::Duration::from_millis(timeout_ms);
        AwaitChangesStep::Block { sub, timeout }
    }

    /// Format a completed `recv` (sync `recv_timeout` or async `recv_async`) into
    /// the `await_changes` response envelope, Issue #3673 — shared by both wait
    /// paths so they return a byte-identical result and only the wait mechanism
    /// differs.
    fn finish_await_recv(
        &self,
        sub: &Subscription,
        recv_result: std::result::Result<Vec<crate::core::ChangeRecord>, RecvError>,
    ) -> CallToolResult {
        match recv_result {
            Ok(recs) if !recs.is_empty() => {
                let resume = recs.last().map(|r| r.cursor().encode());
                self.await_changes_success(&recs, resume, false, false)
            }
            Ok(_) => {
                // Timed out with nothing buffered: an honest empty result plus a
                // resume anchor (the subscribe-time baseline if nothing drained).
                self.await_changes_success(&[], sub.resume_token(), true, false)
            }
            Err(RecvError::Lagged { resume_token }) => {
                let details = match &resume_token {
                    Some(tok) => json!({ "reason": "changefeed_lagged", "resume_token": tok }),
                    None => json!({ "reason": "changefeed_lagged" }),
                };
                self.error_result(
                    McpError::new(
                        McpErrorCode::ResourceExhausted,
                        "The changefeed subscription lagged and was disconnected; resume \
                         losslessly by calling list_changes with the provided resume_token.",
                    )
                    .retriable(true)
                    .details(details),
                )
            }
        }
    }

    // ============================================================================
    // Phase 11: Independent Dimension Temporal Queries & Version History
    // ============================================================================

    fn handle_get_node_at_valid_time(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetNodeAtValidTimeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let valid_time = match self.parse_timestamp(&req.valid_time) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&e),
        };

        match self.db.get_node_at_valid_time(node_id, valid_time) {
            Ok(node) => {
                let now = time::now();
                let response = self.node_to_response(&node, true, now);
                self.success_json(json!({
                    "node": response,
                    "valid_time": req.valid_time
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_get_node_at_transaction_time(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetNodeAtTransactionTimeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let tx_time = match self.parse_timestamp(&req.transaction_time) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&e),
        };

        match self.db.get_node_at_transaction_time(node_id, tx_time) {
            Ok(node) => {
                let now = time::now();
                let response = self.node_to_response(&node, true, now);
                self.success_json(json!({
                    "node": response,
                    "transaction_time": req.transaction_time
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_get_node_history(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetNodeHistoryRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        match self.db.get_node_history(node_id) {
            Ok(history) => {
                let versions: Vec<_> = history
                    .versions
                    .iter()
                    .map(|v| self.version_info_to_response(v))
                    .collect();

                self.success_json(json!({
                    "node_id": req.node_id,
                    "versions": versions,
                    "version_count": versions.len()
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_diff_node_versions(&self, args: serde_json::Value) -> CallToolResult {
        let req: DiffNodeVersionsRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let node_id = match NodeId::new(req.node_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let from_version = match crate::core::id::VersionId::new(req.from_version) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let to_version = match crate::core::id::VersionId::new(req.to_version) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        match self
            .db
            .diff_node_versions(node_id, from_version, to_version)
        {
            Ok(diff) => {
                let response = self.version_diff_to_response(&diff);
                self.success_json(json!(response))
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_get_edge_at_valid_time(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetEdgeAtValidTimeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let edge_id = match EdgeId::new(req.edge_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let valid_time = match self.parse_timestamp(&req.valid_time) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&e),
        };

        match self.db.get_edge_at_valid_time(edge_id, valid_time) {
            Ok(edge) => {
                let now = time::now();
                let response = self.edge_to_response(&edge, true, now);
                self.success_json(json!({
                    "edge": response,
                    "valid_time": req.valid_time
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_get_edge_at_transaction_time(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetEdgeAtTransactionTimeRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let edge_id = match EdgeId::new(req.edge_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let tx_time = match self.parse_timestamp(&req.transaction_time) {
            Ok(t) => t,
            Err(e) => return self.invalid_argument(&e),
        };

        match self.db.get_edge_at_transaction_time(edge_id, tx_time) {
            Ok(edge) => {
                let now = time::now();
                let response = self.edge_to_response(&edge, true, now);
                self.success_json(json!({
                    "edge": response,
                    "transaction_time": req.transaction_time
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_get_edge_history(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetEdgeHistoryRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let edge_id = match EdgeId::new(req.edge_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        match self.db.get_edge_history(edge_id) {
            Ok(history) => {
                let versions: Vec<_> = history
                    .versions
                    .iter()
                    .map(|v| self.version_info_to_response(v))
                    .collect();

                self.success_json(json!({
                    "edge_id": req.edge_id,
                    "versions": versions,
                    "version_count": versions.len()
                }))
            }
            Err(e) => self.db_error(e),
        }
    }

    fn handle_diff_edge_versions(&self, args: serde_json::Value) -> CallToolResult {
        let req: DiffEdgeVersionsRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let edge_id = match EdgeId::new(req.edge_id) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let from_version = match crate::core::id::VersionId::new(req.from_version) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        let to_version = match crate::core::id::VersionId::new(req.to_version) {
            Ok(id) => id,
            // An out-of-range ID is a caller fault; emit the bare
            // `StorageError` text verbatim (`db_error` would wrap it in
            // `Error::Storage`, prefixing "Storage error: " — a message
            // regression vs pre-#3234 responses).
            Err(e) => return self.invalid_argument(&e.to_string()),
        };

        match self
            .db
            .diff_edge_versions(edge_id, from_version, to_version)
        {
            Ok(diff) => {
                let response = self.version_diff_to_response(&diff);
                self.success_json(json!(response))
            }
            Err(e) => self.db_error(e),
        }
    }

    // Helper methods for converting internal types to response types

    fn version_info_to_response(&self, info: &crate::query::VersionInfo) -> serde_json::Value {
        use crate::core::temporal::TIMESTAMP_MAX;

        let properties = self.property_map_to_json(&info.properties, true);

        let valid_to = {
            let end = info.temporal.valid_time().end();
            if end == TIMESTAMP_MAX {
                None
            } else {
                Some(end.wallclock().to_string())
            }
        };

        let transaction_to = {
            let end = info.temporal.transaction_time().end();
            if end == TIMESTAMP_MAX {
                None
            } else {
                Some(end.wallclock().to_string())
            }
        };

        let mut value = json!({
            "version_number": info.version_number,
            "version_id": info.version_id.as_u64(),
            "valid_from": info.temporal.valid_time().start().wallclock().to_string(),
            "valid_to": valid_to,
            "transaction_from": info.temporal.transaction_time().start().wallclock().to_string(),
            "transaction_to": transaction_to,
            "properties": properties,
            "label": info.label
        });

        // Provenance is omitted entirely when absent -- never a fabricated
        // default (Issue #3224).
        if let Some(provenance) = &info.provenance
            && let Some(obj) = value.as_object_mut()
        {
            obj.insert(
                "provenance".to_string(),
                serde_json::to_value(provenance).unwrap_or(serde_json::Value::Null),
            );
        }

        value
    }

    fn version_diff_to_response(&self, diff: &crate::query::VersionDiff) -> serde_json::Value {
        let added = self.property_map_to_json(&diff.added, true);
        let removed = self.property_map_to_json(&diff.removed, true);

        let modified: Vec<_> = diff
            .modified
            .iter()
            .map(|(key, old_val, new_val)| {
                let key_str = GLOBAL_INTERNER
                    .resolve_with(*key, |s| s.to_string())
                    .unwrap_or_else(|| format!("{:?}", key));

                json!({
                    "key": key_str,
                    "old_value": self.property_value_to_json(old_val, true),
                    "new_value": self.property_value_to_json(new_val, true)
                })
            })
            .collect();

        json!({
            "from_version": diff.from_version.as_u64(),
            "to_version": diff.to_version.as_u64(),
            "added": added,
            "removed": removed,
            "modified": modified,
            "has_changes": diff.has_changes(),
            "change_count": diff.change_count()
        })
    }

    /// Convert a [`crate::db::GraphSchema`] into its JSON wire representation.
    fn schema_to_json(&self, schema: &crate::db::GraphSchema) -> serde_json::Value {
        let node_labels: Vec<serde_json::Value> = schema
            .node_labels
            .iter()
            .map(|l| {
                json!({
                    "label": l.label,
                    "count": l.count,
                    "property_keys": l.property_keys,
                })
            })
            .collect();

        let edge_types: Vec<serde_json::Value> = schema
            .edge_types
            .iter()
            .map(|e| {
                json!({
                    "edge_type": e.edge_type,
                    "count": e.count,
                    "property_keys": e.property_keys,
                })
            })
            .collect();

        // Per-namespace current node/edge counts (Issue #3349, PR3a): one
        // `{name, node_count, edge_count}` entry per registered-or-populated
        // namespace, sorted by name. Empty for a bi-temporal (`as_of`) schema
        // (the membership index is a current-state acceleration structure).
        // NamespaceCount.name is a user-facing namespace name, never the elided
        // ride-along key.
        let namespaces: Vec<serde_json::Value> = schema
            .namespaces
            .iter()
            .map(|n| {
                json!({
                    "name": n.name,
                    "node_count": n.node_count,
                    "edge_count": n.edge_count,
                })
            })
            .collect();

        json!({
            "node_labels": node_labels,
            "edge_types": edge_types,
            "total_nodes": schema.total_nodes,
            "total_edges": schema.total_edges,
            "sampled": schema.sampled,
            "namespaces": namespaces,
            "as_of": schema.as_of.map(|instant| json!({
                "valid_time": time::to_iso8601(instant.valid_time),
                "transaction_time": time::to_iso8601(instant.transaction_time),
            })),
        })
    }

    /// Discover the graph's schema (labels, edge types, property keys),
    /// optionally as of a bi-temporal instant. Never errors on an empty
    /// database — returns a well-formed, empty summary instead.
    fn handle_get_schema(&self, args: serde_json::Value) -> CallToolResult {
        let req: GetSchemaRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let temporal = match self
            .resolve_bitemporal_as_of(&req.as_of_valid_time, &req.as_of_transaction_time)
        {
            Ok(t) => t,
            Err(result) => return result,
        };

        let result = match temporal {
            None => self.db.schema(),
            Some((vt, tt)) => self.db.schema_as_of(vt, tt),
        };

        match result {
            Ok(schema) => self.success_json(self.schema_to_json(&schema)),
            Err(e) => self.db_error(e),
        }
    }

    /// Render a finite timestamp as an RFC3339 string (UTC, microsecond
    /// precision). Falls back to a raw microsecond form for coordinates
    /// outside chrono's representable range rather than panicking.
    fn timestamp_to_rfc3339(ts: Timestamp) -> String {
        let micros = ts.wallclock();
        DateTime::<Utc>::from_timestamp_micros(micros)
            .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Micros, true))
            .unwrap_or_else(|| format!("{micros}us"))
    }

    /// Serialize one dimension's bounds; `None` bounds become explicit JSON
    /// `null`s (never `0`/epoch), so an empty database is unambiguous.
    fn time_bounds_to_json(bounds: &crate::db::TimeBounds) -> serde_json::Value {
        json!({
            "earliest": bounds.earliest.map(Self::timestamp_to_rfc3339),
            "latest": bounds.latest.map(Self::timestamp_to_rfc3339),
        })
    }

    /// Convert a [`crate::db::TemporalExtent`] into its JSON wire
    /// representation. Serialization only — all bounds are computed by
    /// [`AletheiaDB::temporal_extent`]/[`AletheiaDB::temporal_extent_by_label`].
    fn temporal_extent_to_json(extent: &crate::db::TemporalExtent) -> serde_json::Value {
        let mut value = json!({
            "valid_time": Self::time_bounds_to_json(&extent.valid_time),
            "transaction_time": Self::time_bounds_to_json(&extent.transaction_time),
        });

        let label_extents_to_json = |extents: &[crate::db::LabelExtent], key: &str| {
            extents
                .iter()
                .map(|e| {
                    json!({
                        key: e.label,
                        "valid_time": Self::time_bounds_to_json(&e.valid_time),
                        "transaction_time": Self::time_bounds_to_json(&e.transaction_time),
                    })
                })
                .collect::<Vec<_>>()
        };

        if let Some(obj) = value.as_object_mut() {
            if let Some(node_labels) = &extent.node_labels {
                obj.insert(
                    "node_labels".to_string(),
                    serde_json::Value::Array(label_extents_to_json(node_labels, "label")),
                );
            }
            if let Some(edge_types) = &extent.edge_types {
                obj.insert(
                    "edge_types".to_string(),
                    serde_json::Value::Array(label_extents_to_json(edge_types, "edge_type")),
                );
            }
        }

        value
    }

    /// Report the dataset's queryable bi-temporal extent (Issue #3238).
    ///
    /// The handler only serializes: bounds come from the public
    /// `AletheiaDB::temporal_extent` / `temporal_extent_by_label` API.
    fn handle_temporal_extent(&self, args: serde_json::Value) -> CallToolResult {
        // The tool has no required arguments: a call with no `arguments`
        // object at all must behave like `{}`.
        let args = if args.is_null() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            args
        };

        let req: TemporalExtentRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let result = if req.by_label.unwrap_or(false) {
            self.db.temporal_extent_by_label()
        } else {
            self.db.temporal_extent()
        };

        match result {
            Ok(extent) => self.success_json(Self::temporal_extent_to_json(&extent)),
            Err(e) => self.db_error(e),
        }
    }

    /// Convert a core [`EntityId`](crate::core::id::EntityId) into the
    /// `(entity_kind, id)` pair used in lineage JSON responses.
    fn lineage_entity_parts(entity: crate::core::id::EntityId) -> (&'static str, u64) {
        match entity {
            crate::core::id::EntityId::Node(id) => ("node", id.as_u64()),
            crate::core::id::EntityId::Edge(id) => ("edge", id.as_u64()),
        }
    }

    /// Serialize one resolved lineage entry (version-pinned ref + depth +
    /// current-state status) for a lineage query response (Issue #3371).
    fn lineage_entry_to_json(entry: &crate::db::LineageViewEntry) -> serde_json::Value {
        let (entity_kind, id) = Self::lineage_entity_parts(entry.reference.entity);
        json!({
            "entity_kind": entity_kind,
            "id": id,
            "version": entry.reference.version.as_u64(),
            "depth": entry.depth,
            "status": entry.status.as_str(),
        })
    }

    /// Shared implementation of the `lineage_upstream` / `lineage_downstream`
    /// tools (Issue #3371): resolve the root, run the closure in `direction`,
    /// paginate, and shape the response with `has_more`/`next_offset` (#3226).
    fn handle_lineage_query(&self, args: serde_json::Value, upstream: bool) -> CallToolResult {
        let req: crate::mcp::tools::LineageQueryRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let root_req = crate::mcp::tools::LineageRefRequest {
            entity_kind: req.entity_kind.clone(),
            id: req.id,
            version: req.version,
        };
        let root = match self.parse_lineage_ref(&root_req) {
            Ok(r) => r,
            Err(result) => return result,
        };

        let as_of =
            match self.parse_opt_timestamp("as_of_transaction_time", &req.as_of_transaction_time) {
                Ok(v) => v,
                Err(result) => return result,
            };

        let page_limit = req.limit.unwrap_or(100);
        let offset = req.offset.unwrap_or(0);
        // Fetch enough to cover the requested page window; slicing happens
        // below so `offset` paginates a stable breadth-first ordering.
        let fetch_limit = offset.saturating_add(page_limit);

        let mut options = crate::core::lineage::LineageQueryOptions::new().with_limit(fetch_limit);
        if let Some(max_depth) = req.max_depth {
            options = options.with_max_depth(max_depth);
        }
        if let Some(as_of) = as_of {
            options = options.with_as_of(as_of);
        }

        let view = if upstream {
            self.db.upstream_lineage(root, options)
        } else {
            self.db.downstream_lineage(root, options)
        };

        let page: Vec<serde_json::Value> = view
            .entries
            .iter()
            .skip(offset)
            .take(page_limit)
            .map(Self::lineage_entry_to_json)
            .collect();
        // The store already bounded the fetch to `fetch_limit`; anything it
        // dropped (limit or depth cap) is reported via `has_more`.
        let has_more = view.has_more;
        let returned = page.len();

        let (root_kind, root_id) = Self::lineage_entity_parts(root.entity);
        let mut response = json!({
            "direction": if upstream { "upstream" } else { "downstream" },
            "root": {
                "entity_kind": root_kind,
                "id": root_id,
                "version": root.version.as_u64(),
            },
            "entries": page,
            "count": returned,
        });
        if let Some(obj) = response.as_object_mut() {
            obj.insert("has_more".to_string(), json!(has_more));
            if has_more {
                obj.insert(
                    "next_offset".to_string(),
                    json!(offset.saturating_add(returned)),
                );
            }
        }
        self.success_json(response)
    }

    /// Handle the `lineage_upstream` tool (Issue #3371): "what was this fact
    /// derived from?" — the transitive evidence chain.
    fn handle_lineage_upstream(&self, args: serde_json::Value) -> CallToolResult {
        self.handle_lineage_query(args, true)
    }

    /// Handle the `lineage_downstream` tool (Issue #3371): "what has been
    /// derived from this fact?" — the retraction blast-radius report.
    fn handle_lineage_downstream(&self, args: serde_json::Value) -> CallToolResult {
        self.handle_lineage_query(args, false)
    }

    // ========================================================================
    // Belief-revision audit (Issue #3362)
    //
    // Advertised unconditionally (Design A). The real handler body lives under
    // `#[cfg(feature = "semantic-temporal")]`; a `#[cfg(not(...))]` twin returns
    // a structured `FAILED_PRECONDITION` (mirroring `semantic_search_unavailable`)
    // so a caller on a build without the feature gets an actionable error rather
    // than an "unknown tool".
    // ========================================================================

    /// Structured `FAILED_PRECONDITION` returned by `get_belief_revisions` when
    /// the `semantic-temporal` feature is not compiled into this build.
    #[cfg(not(feature = "semantic-temporal"))]
    fn handle_get_belief_revisions(&self, _args: serde_json::Value) -> CallToolResult {
        self.error_result(
            McpError::new(
                McpErrorCode::FailedPrecondition,
                "Tool 'get_belief_revisions' requires the `semantic-temporal` feature, which is \
                 not compiled into this build. Rebuild AletheiaDB with `--features \
                 semantic-temporal` to enable it."
                    .to_string(),
            )
            .details(json!({
                "tool": "get_belief_revisions",
                "required_feature": "semantic-temporal",
            })),
        )
    }

    /// Handle the `get_belief_revisions` tool (Issue #3362): classify an
    /// entity's stored belief revisions and return the §9 JSON shape.
    #[cfg(feature = "semantic-temporal")]
    fn handle_get_belief_revisions(&self, args: serde_json::Value) -> CallToolResult {
        use crate::core::id::EntityId;
        use crate::experimental::temporal::belief_revision::RevisionOptions;

        let req: GetBeliefRevisionsRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        // Resolve the entity id (an out-of-range id is a caller fault →
        // INVALID_ARGUMENT with the bare id-validation message).
        let entity = match req.entity_kind.as_str() {
            "node" => match NodeId::new(req.id) {
                Ok(id) => EntityId::Node(id),
                Err(e) => return self.invalid_argument(&e.to_string()),
            },
            "edge" => match EdgeId::new(req.id) {
                Ok(id) => EntityId::Edge(id),
                Err(e) => return self.invalid_argument(&e.to_string()),
            },
            other => {
                return self.invalid_argument(&format!(
                    "entity_kind must be 'node' or 'edge', got '{other}'"
                ));
            }
        };

        let mut options = RevisionOptions::new();
        if let Some(ref key) = req.property_key {
            options = options.with_property_key(key.clone());
        }
        if let Some(ref ts_str) = req.as_of_transaction_time {
            let ts = match self.parse_timestamp(ts_str) {
                Ok(t) => t,
                Err(e) => return self.invalid_argument(&e),
            };
            options = options.with_as_of_transaction_time(ts);
        }
        if let Some(limit) = req.limit {
            options = options.with_limit(limit);
        }

        match self.db.belief_revisions(entity, &options) {
            Ok(log) => self.success_json(self.belief_revision_log_to_json(&log)),
            Err(e) => self.db_error(e),
        }
    }

    /// Serialize a [`BeliefRevisionLog`] into the Issue #3362 §9 JSON shape.
    #[cfg(feature = "semantic-temporal")]
    fn belief_revision_log_to_json(
        &self,
        log: &crate::experimental::temporal::belief_revision::BeliefRevisionLog,
    ) -> serde_json::Value {
        use crate::core::id::EntityId;

        let (kind, id) = match log.entity {
            EntityId::Node(nid) => ("node", nid.as_u64()),
            EntityId::Edge(eid) => ("edge", eid.as_u64()),
        };

        let revisions: Vec<serde_json::Value> = log
            .revisions
            .iter()
            .map(|r| self.revision_to_json(r))
            .collect();

        let confidence_trajectory: Vec<serde_json::Value> = log
            .revisions
            .iter()
            .map(|r| match r.confidence {
                Some(c) => json!(c),
                None => serde_json::Value::Null,
            })
            .collect();

        json!({
            "entity": { "kind": kind, "id": id },
            "property_key": log.property_key,
            "as_of_transaction_time": log
                .as_of_transaction_time
                .map(Self::format_timestamp_rfc3339),
            "revisions": revisions,
            "confidence_trajectory": confidence_trajectory,
            "has_more": log.has_more,
        })
    }

    /// Serialize a single [`Revision`] entry (Issue #3362 §9).
    #[cfg(feature = "semantic-temporal")]
    fn revision_to_json(
        &self,
        rev: &crate::experimental::temporal::belief_revision::Revision,
    ) -> serde_json::Value {
        let changes: Vec<serde_json::Value> = rev
            .changes
            .iter()
            .map(|c| {
                json!({
                    "key": c.key,
                    "prior": c.prior.as_ref().map(Self::property_value_to_exported_json),
                    "new": c.new.as_ref().map(Self::property_value_to_exported_json),
                })
            })
            .collect();

        let provenance = rev.provenance.as_ref().map(|p| {
            json!({
                "source": p.source(),
                "confidence": p.confidence(),
                "note": p.note(),
                "correlation_id": p.correlation_id(),
                "principal": p.principal(),
            })
        });

        json!({
            "version_number": rev.version_number,
            "version_id": rev.version_id.as_u64(),
            "transaction_time": Self::format_timestamp_rfc3339(rev.transaction_time),
            "valid_from": Self::format_timestamp_rfc3339(rev.valid_from),
            "valid_to": rev.valid_to.map(Self::format_timestamp_rfc3339),
            "classification": rev.class.as_str(),
            "changes": changes,
            "provenance": provenance,
            "confidence": rev.confidence,
        })
    }

    /// Serialize a [`PropertyValue`] into the self-describing tagged shape used
    /// by belief-revision `changes` (mirrors `audit::model::ExportedValue`:
    /// `{"type": "...", "value": ...}`), independent of the `audit-export`
    /// feature so it is available under a bare `semantic-temporal` build.
    #[cfg(feature = "semantic-temporal")]
    fn property_value_to_exported_json(value: &PropertyValue) -> serde_json::Value {
        match value {
            PropertyValue::Null => json!({ "type": "null" }),
            PropertyValue::Bool(b) => json!({ "type": "bool", "value": b }),
            PropertyValue::Int(i) => json!({ "type": "int", "value": i }),
            PropertyValue::Float(f) => json!({ "type": "float", "value": f.to_string() }),
            PropertyValue::String(s) => json!({ "type": "string", "value": s.to_string() }),
            PropertyValue::Bytes(b) => {
                json!({ "type": "bytes", "hex": crate::core::hex::encode(b) })
            }
            PropertyValue::Array(values) => json!({
                "type": "array",
                "values": values
                    .iter()
                    .map(Self::property_value_to_exported_json)
                    .collect::<Vec<_>>(),
            }),
            PropertyValue::Vector(v) => json!({
                "type": "vector",
                "values": v.iter().map(|f| f64::from(*f).to_string()).collect::<Vec<_>>(),
            }),
            PropertyValue::SparseVector(sv) => json!({
                "type": "sparse_vector",
                "dimension": sv.dimension(),
                "indices": sv.indices().to_vec(),
                "values": sv.values().iter().map(std::string::ToString::to_string).collect::<Vec<_>>(),
            }),
        }
    }

    // ========================================================================
    // Temporal drift-alarm management (Issue #3367 / PR #3728).
    //
    // Advertised unconditionally (Design A). Real handlers gate on
    // `semantic-temporal`; a `#[cfg(not(...))]` twin returns a structured
    // FAILED_PRECONDITION with `{tool, required_feature}`.
    // ========================================================================

    /// Structured FAILED_PRECONDITION for a `semantic-temporal`-gated tool when
    /// the feature is not compiled into this build.
    #[cfg(not(feature = "semantic-temporal"))]
    fn semantic_temporal_unavailable(&self, tool: &str) -> CallToolResult {
        self.error_result(
            McpError::new(
                McpErrorCode::FailedPrecondition,
                format!(
                    "Tool '{tool}' requires the `semantic-temporal` feature, which is not \
                     compiled into this build. Rebuild AletheiaDB with `--features \
                     semantic-temporal` to enable it."
                ),
            )
            .details(json!({ "tool": tool, "required_feature": "semantic-temporal" })),
        )
    }

    #[cfg(not(feature = "semantic-temporal"))]
    fn handle_create_drift_monitor(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_temporal_unavailable("create_drift_monitor")
    }

    #[cfg(not(feature = "semantic-temporal"))]
    fn handle_list_drift_monitors(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_temporal_unavailable("list_drift_monitors")
    }

    #[cfg(not(feature = "semantic-temporal"))]
    fn handle_delete_drift_monitor(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_temporal_unavailable("delete_drift_monitor")
    }

    #[cfg(not(feature = "semantic-temporal"))]
    fn handle_query_drift_alarms(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_temporal_unavailable("query_drift_alarms")
    }

    #[cfg(not(feature = "semantic-temporal"))]
    fn handle_resolve_drift_alarm(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_temporal_unavailable("resolve_drift_alarm")
    }

    #[cfg(not(feature = "semantic-temporal"))]
    fn handle_contradiction_genealogy(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_temporal_unavailable("contradiction_genealogy")
    }

    #[cfg(not(feature = "semantic-temporal"))]
    fn handle_find_contradictions(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_temporal_unavailable("find_contradictions")
    }

    #[cfg(not(feature = "semantic-temporal"))]
    fn handle_counterfactual_replay(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_temporal_unavailable("counterfactual_replay")
    }

    /// Parse a `cosine` / `euclidean` / `angular` metric token.
    #[cfg(feature = "semantic-temporal")]
    fn parse_drift_metric(
        &self,
        s: &str,
    ) -> std::result::Result<crate::index::vector::temporal::DriftMetric, String> {
        use crate::index::vector::temporal::DriftMetric;
        match s.trim().to_ascii_lowercase().as_str() {
            "cosine" => Ok(DriftMetric::Cosine),
            "euclidean" => Ok(DriftMetric::Euclidean),
            "angular" => Ok(DriftMetric::Angular),
            other => Err(format!(
                "metric must be 'cosine', 'euclidean', or 'angular', got '{other}'"
            )),
        }
    }

    /// Lowercase token for a drift metric.
    #[cfg(feature = "semantic-temporal")]
    fn drift_metric_str(m: crate::index::vector::temporal::DriftMetric) -> &'static str {
        use crate::index::vector::temporal::DriftMetric;
        match m {
            DriftMetric::Cosine => "cosine",
            DriftMetric::Euclidean => "euclidean",
            DriftMetric::Angular => "angular",
        }
    }

    /// Serialize a [`DriftMonitor`] to its documented JSON shape.
    #[cfg(feature = "semantic-temporal")]
    fn drift_monitor_to_json(
        &self,
        m: &crate::experimental::temporal::drift_alarm::DriftMonitor,
    ) -> serde_json::Value {
        use crate::experimental::temporal::drift_alarm::EvalMode;
        let spec = &m.spec;
        let (mode, interval): (&str, serde_json::Value) = match spec.mode {
            EvalMode::OnWrite => ("on_write", serde_json::Value::Null),
            EvalMode::Scheduled { interval } => (
                "scheduled",
                json!(u64::try_from(interval.as_micros()).unwrap_or(u64::MAX)),
            ),
        };
        json!({
            "id": m.id.get(),
            "created_at": Self::format_timestamp_rfc3339(m.created_at),
            "spec": {
                "property_key": spec.property_key,
                "label": spec.label,
                "entities": spec
                    .entities
                    .as_ref()
                    .map(|v| v.iter().map(|n| n.as_u64()).collect::<Vec<_>>()),
                "metric": Self::drift_metric_str(spec.metric),
                "threshold": spec.threshold,
                "window_micros": u64::try_from(spec.window.as_micros()).unwrap_or(u64::MAX),
                "target": spec.target.as_str(),
                "mode": mode,
                "scheduled_interval_micros": interval,
            },
        })
    }

    /// Serialize a [`DriftAlarm`] to its documented JSON shape.
    #[cfg(feature = "semantic-temporal")]
    fn drift_alarm_to_json(
        &self,
        a: &crate::experimental::temporal::drift_alarm::DriftAlarm,
    ) -> serde_json::Value {
        json!({
            "alarm_id": a.alarm_id.as_u64(),
            "monitor_id": a.monitor_id.get(),
            "entity": a.entity.map(|n| n.as_u64()),
            "label": a.label,
            "measured_distance": a.measured_distance,
            "threshold": a.threshold,
            "metric": Self::drift_metric_str(a.metric),
            "compared_now": Self::format_timestamp_rfc3339(a.compared_now),
            "compared_past": Self::format_timestamp_rfc3339(a.compared_past),
            "from_version": a.from_version.map(|v| v.as_u64()),
            "to_version": a.to_version.map(|v| v.as_u64()),
            "resolved": a.resolved,
            "fired_at": Self::format_timestamp_rfc3339(a.fired_at),
        })
    }

    /// Handle `create_drift_monitor` (Write, Issue #3367).
    #[cfg(feature = "semantic-temporal")]
    fn handle_create_drift_monitor(&self, args: serde_json::Value) -> CallToolResult {
        use crate::experimental::temporal::drift_alarm::{DriftMonitorSpec, DriftTarget, EvalMode};

        let req: CreateDriftMonitorRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let metric = match self.parse_drift_metric(&req.metric) {
            Ok(m) => m,
            Err(msg) => return self.invalid_argument(&msg),
        };
        let target = match req.target.as_str() {
            "per_entity" => DriftTarget::PerEntity,
            "label_centroid" => DriftTarget::LabelCentroid,
            other => {
                return self.invalid_argument(&format!(
                    "target must be 'per_entity' or 'label_centroid', got '{other}'"
                ));
            }
        };
        let mode = match req.mode.as_str() {
            "on_write" => EvalMode::OnWrite,
            "scheduled" => match req.scheduled_interval_micros {
                Some(us) => EvalMode::Scheduled {
                    interval: std::time::Duration::from_micros(us),
                },
                None => {
                    return self
                        .invalid_argument("mode 'scheduled' requires scheduled_interval_micros");
                }
            },
            other => {
                return self.invalid_argument(&format!(
                    "mode must be 'on_write' or 'scheduled', got '{other}'"
                ));
            }
        };
        let entities = match req.entities {
            Some(ids) => {
                let mut out = Vec::with_capacity(ids.len());
                for raw in ids {
                    match NodeId::new(raw) {
                        Ok(id) => out.push(id),
                        Err(e) => return self.invalid_argument(&e.to_string()),
                    }
                }
                Some(out)
            }
            None => None,
        };

        let spec = DriftMonitorSpec {
            property_key: req.property_key,
            label: req.label,
            entities,
            metric,
            threshold: req.threshold,
            window: std::time::Duration::from_micros(req.window_micros),
            target,
            mode,
        };

        match self.db.create_drift_monitor(spec) {
            Ok(monitor) => self.success_json(self.drift_monitor_to_json(&monitor)),
            Err(e) => self.db_error(e),
        }
    }

    /// Handle `list_drift_monitors` (Read, Issue #3367).
    #[cfg(feature = "semantic-temporal")]
    fn handle_list_drift_monitors(&self, args: serde_json::Value) -> CallToolResult {
        let _req: ListDriftMonitorsRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        let monitors = self.db.list_drift_monitors();
        let list: Vec<serde_json::Value> = monitors
            .iter()
            .map(|m| self.drift_monitor_to_json(m))
            .collect();
        self.success_json(json!({ "monitors": list, "count": monitors.len() }))
    }

    /// Handle `delete_drift_monitor` (Write, Issue #3367).
    #[cfg(feature = "semantic-temporal")]
    fn handle_delete_drift_monitor(&self, args: serde_json::Value) -> CallToolResult {
        use crate::experimental::temporal::drift_alarm::MonitorId;
        let req: DeleteDriftMonitorRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        match self.db.delete_drift_monitor(MonitorId::new(req.id)) {
            Ok(()) => self.success_json(json!({ "deleted": true, "id": req.id })),
            Err(e) => self.db_error(e),
        }
    }

    /// Handle `query_drift_alarms` (Read, Issue #3367).
    #[cfg(feature = "semantic-temporal")]
    fn handle_query_drift_alarms(&self, args: serde_json::Value) -> CallToolResult {
        use crate::experimental::temporal::drift_alarm::{
            DEFAULT_ALARM_QUERY_LIMIT, DriftAlarmFilter, MonitorId,
        };
        let req: QueryDriftAlarmsRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let time_range = match (&req.time_range_start, &req.time_range_end) {
            (Some(s), Some(e)) => {
                let start = match self.parse_timestamp(s) {
                    Ok(t) => t,
                    Err(m) => return self.invalid_argument(&m),
                };
                let end = match self.parse_timestamp(e) {
                    Ok(t) => t,
                    Err(m) => return self.invalid_argument(&m),
                };
                Some((start, end))
            }
            (None, None) => None,
            _ => {
                return self.invalid_argument(
                    "time_range_start and time_range_end must be provided together",
                );
            }
        };

        let filter = DriftAlarmFilter {
            monitor_id: req.monitor_id.map(MonitorId::new),
            label: req.label,
            resolved: req.resolved,
            time_range,
            limit: req.limit.unwrap_or(DEFAULT_ALARM_QUERY_LIMIT),
        };

        match self.db.query_drift_alarms(&filter) {
            Ok(alarms) => {
                let list: Vec<serde_json::Value> =
                    alarms.iter().map(|a| self.drift_alarm_to_json(a)).collect();
                self.success_json(json!({ "alarms": list, "count": alarms.len() }))
            }
            Err(e) => self.db_error(e),
        }
    }

    /// Handle `resolve_drift_alarm` (Write, Issue #3367).
    #[cfg(feature = "semantic-temporal")]
    fn handle_resolve_drift_alarm(&self, args: serde_json::Value) -> CallToolResult {
        let req: ResolveDriftAlarmRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };
        let alarm_id = match NodeId::new(req.alarm_id) {
            Ok(id) => id,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        match self.db.resolve_drift_alarm(alarm_id) {
            Ok(a) => self.success_json(self.drift_alarm_to_json(&a)),
            Err(e) => self.db_error(e),
        }
    }

    // ========================================================================
    // Contradiction genealogy (Issue #3352 / PR #3742).
    // ========================================================================

    /// Serialize a [`ClaimRef`] to `{entity_kind, id, version}`.
    #[cfg(feature = "semantic-temporal")]
    fn claim_ref_to_json(
        r: &crate::experimental::temporal::contradiction_genealogy::ClaimRef,
    ) -> serde_json::Value {
        let (kind, id) = Self::lineage_entity_parts(r.entity);
        json!({ "entity_kind": kind, "id": id, "version": r.version.as_u64() })
    }

    /// Serialize a bi-temporal coordinate.
    #[cfg(feature = "semantic-temporal")]
    fn bitemporal_coord_to_json(
        c: crate::experimental::temporal::contradiction_genealogy::BiTemporalCoordinate,
    ) -> serde_json::Value {
        json!({
            "transaction_time": Self::format_timestamp_rfc3339(c.transaction_time),
            "valid_time": Self::format_timestamp_rfc3339(c.valid_time),
        })
    }

    /// Serialize a competing claim.
    #[cfg(feature = "semantic-temporal")]
    fn competing_claim_to_json(
        &self,
        c: &crate::experimental::temporal::contradiction_genealogy::CompetingClaim,
    ) -> serde_json::Value {
        json!({
            "claim": Self::claim_ref_to_json(&c.claim),
            "value_display": c.value_display,
            "valid_from": Self::format_timestamp_rfc3339(c.valid_from),
            "valid_to": c.valid_to.map(Self::format_timestamp_rfc3339),
            "transaction_from": Self::format_timestamp_rfc3339(c.transaction_from),
            "transaction_to": c.transaction_to.map(Self::format_timestamp_rfc3339),
            "is_current": c.is_current,
            "provenance": c.provenance.as_ref().map(|p| json!({
                "source": p.source,
                "confidence": p.confidence,
                "note": p.note,
            })),
            "origin": c.origin.as_str(),
            "supersedes": c.supersedes.map(|v| v.as_u64()),
            "superseded_by": c.superseded_by.map(|v| v.as_u64()),
        })
    }

    /// Serialize a divergence pair.
    #[cfg(feature = "semantic-temporal")]
    fn divergence_pair_to_json(
        p: &crate::experimental::temporal::contradiction_genealogy::DivergencePair,
    ) -> serde_json::Value {
        json!({
            "earlier": Self::claim_ref_to_json(&p.earlier),
            "later": Self::claim_ref_to_json(&p.later),
            "kind": p.kind.as_str(),
            "coordinate": Self::bitemporal_coord_to_json(p.coordinate),
            "overlapping_valid_from": Self::format_timestamp_rfc3339(p.overlapping_valid_from),
            "overlapping_valid_to": p.overlapping_valid_to.map(Self::format_timestamp_rfc3339),
        })
    }

    /// Serialize a per-source summary.
    #[cfg(feature = "semantic-temporal")]
    fn source_summary_to_json(
        s: &crate::experimental::temporal::contradiction_genealogy::SourceSummary,
    ) -> serde_json::Value {
        json!({
            "source": s.source,
            "backs_values": s.backs_values,
            "claim_count": s.claim_count,
            "min_confidence": s.min_confidence,
            "max_confidence": s.max_confidence,
            "latest_confidence": s.latest_confidence,
            "most_recent_assertion": Self::format_timestamp_rfc3339(s.most_recent_assertion),
        })
    }

    /// Serialize a full [`ContradictionGenealogy`].
    #[cfg(feature = "semantic-temporal")]
    fn contradiction_genealogy_to_json(
        &self,
        g: &crate::experimental::temporal::contradiction_genealogy::ContradictionGenealogy,
    ) -> serde_json::Value {
        json!({
            "entity": g.entity.map(|e| {
                let (kind, id) = Self::lineage_entity_parts(e);
                json!({ "entity_kind": kind, "id": id })
            }),
            "property": g.property,
            "claims": g
                .claims
                .iter()
                .map(|c| self.competing_claim_to_json(c))
                .collect::<Vec<_>>(),
            "pairs": g.pairs.iter().map(Self::divergence_pair_to_json).collect::<Vec<_>>(),
            "divergence_point": g.divergence_point.map(Self::bitemporal_coord_to_json),
            "sources": g.sources.iter().map(Self::source_summary_to_json).collect::<Vec<_>>(),
            "narrative": g.narrative,
            "truncated": g.truncated,
        })
    }

    /// Serialize a [`ContradictionScan`].
    #[cfg(feature = "semantic-temporal")]
    fn contradiction_scan_to_json(
        &self,
        scan: &crate::experimental::temporal::contradiction_genealogy::ContradictionScan,
    ) -> serde_json::Value {
        let contradictions: Vec<serde_json::Value> = scan
            .contradictions
            .iter()
            .map(|s| {
                let (kind, id) = Self::lineage_entity_parts(s.entity);
                json!({
                    "entity": { "entity_kind": kind, "id": id },
                    "property": s.property,
                    "claim_count": s.claim_count,
                    "divergence_point": Self::bitemporal_coord_to_json(s.divergence_point),
                    "classification": s.classification.as_str(),
                })
            })
            .collect();
        json!({
            "contradictions": contradictions,
            "scanned_entities": scan.scanned_entities,
            "sampled": scan.sampled,
            "has_more": scan.has_more,
            "next_offset": scan.next_offset,
        })
    }

    /// Parse an `entity_kind` + id into an [`EntityId`], returning a structured
    /// `INVALID_ARGUMENT` `CallToolResult` on a bad kind or out-of-range id.
    // clippy::result_large_err: Err is rmcp's `CallToolResult` (~176B); this is
    // the established pattern for the MCP arg-parsing helpers (see the
    // `parse_lineage_ref` cluster above).
    #[cfg(feature = "semantic-temporal")]
    #[allow(clippy::result_large_err)]
    fn parse_entity_kind_ct(
        &self,
        kind: &str,
        id: u64,
    ) -> std::result::Result<crate::core::id::EntityId, CallToolResult> {
        use crate::core::id::EntityId;
        match kind.trim().to_ascii_lowercase().as_str() {
            "node" => NodeId::new(id)
                .map(EntityId::Node)
                .map_err(|e| self.invalid_argument(&e.to_string())),
            "edge" => EdgeId::new(id)
                .map(EntityId::Edge)
                .map_err(|e| self.invalid_argument(&e.to_string())),
            other => Err(self.invalid_argument(&format!(
                "entity_kind must be 'node' or 'edge', got '{other}'"
            ))),
        }
    }

    /// Handle `contradiction_genealogy` (Read, Issue #3352).
    #[cfg(feature = "semantic-temporal")]
    fn handle_contradiction_genealogy(&self, args: serde_json::Value) -> CallToolResult {
        use crate::experimental::temporal::contradiction_genealogy::{
            ClaimRef, ContradictionTarget, GenealogyOptions,
        };

        let req: ContradictionGenealogyRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let target = if let Some(claims) = req.claims {
            let mut refs = Vec::with_capacity(claims.len());
            for c in &claims {
                let entity = match self.parse_entity_kind_ct(&c.entity_kind, c.id) {
                    Ok(e) => e,
                    Err(r) => return r,
                };
                let version = match VersionId::new(c.version) {
                    Ok(v) => v,
                    Err(e) => return self.invalid_argument(&e.to_string()),
                };
                refs.push(ClaimRef { entity, version });
            }
            ContradictionTarget::Claims(refs)
        } else {
            let kind = match &req.entity_kind {
                Some(s) => s.as_str(),
                None => {
                    return self.invalid_argument(
                        "provide either 'claims' or 'entity_kind' + 'id' + 'property'",
                    );
                }
            };
            let id = match req.id {
                Some(i) => i,
                None => {
                    return self
                        .invalid_argument("'id' is required when targeting a single entity");
                }
            };
            let property = match req.property {
                Some(p) => p,
                None => {
                    return self
                        .invalid_argument("'property' is required when targeting a single entity");
                }
            };
            let entity = match self.parse_entity_kind_ct(kind, id) {
                Ok(e) => e,
                Err(r) => return r,
            };
            ContradictionTarget::EntityProperty { entity, property }
        };

        let mut options = GenealogyOptions::new();
        if let Some(ts) = req.as_of_transaction_time {
            let t = match self.parse_timestamp(&ts) {
                Ok(t) => t,
                Err(m) => return self.invalid_argument(&m),
            };
            options = options.with_as_of_transaction_time(t);
        }
        if let Some(m) = req.max_claims {
            options = options.with_max_claims(m);
        }
        if let Some(m) = req.max_sources {
            options = options.with_max_sources(m);
        }

        match self.db.contradiction_genealogy(target, &options) {
            Ok(g) => self.success_json(self.contradiction_genealogy_to_json(&g)),
            Err(e) => self.db_error(e),
        }
    }

    /// Handle `find_contradictions` (Read, Issue #3352).
    #[cfg(feature = "semantic-temporal")]
    fn handle_find_contradictions(&self, args: serde_json::Value) -> CallToolResult {
        use crate::experimental::temporal::contradiction_genealogy::{
            ContradictionScope, DEFAULT_CONTRADICTION_LIMIT, EntityKindScope,
        };

        let req: FindContradictionsRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let entity_kind = match req.entity_kind.trim().to_ascii_lowercase().as_str() {
            "nodes" => EntityKindScope::Nodes,
            "edges" => EntityKindScope::Edges,
            "both" => EntityKindScope::Both,
            other => {
                return self.invalid_argument(&format!(
                    "entity_kind must be 'nodes', 'edges', or 'both', got '{other}'"
                ));
            }
        };

        let valid_time_window = match (&req.valid_time_start, &req.valid_time_end) {
            (Some(s), Some(e)) => {
                let start = match self.parse_timestamp(s) {
                    Ok(t) => t,
                    Err(m) => return self.invalid_argument(&m),
                };
                let end = match self.parse_timestamp(e) {
                    Ok(t) => t,
                    Err(m) => return self.invalid_argument(&m),
                };
                Some((start, end))
            }
            (None, None) => None,
            _ => {
                return self.invalid_argument(
                    "valid_time_start and valid_time_end must be provided together",
                );
            }
        };
        let transaction_time_window = match (&req.transaction_time_start, &req.transaction_time_end)
        {
            (Some(s), Some(e)) => {
                let start = match self.parse_timestamp(s) {
                    Ok(t) => t,
                    Err(m) => return self.invalid_argument(&m),
                };
                let end = match self.parse_timestamp(e) {
                    Ok(t) => t,
                    Err(m) => return self.invalid_argument(&m),
                };
                Some((start, end))
            }
            (None, None) => None,
            _ => {
                return self.invalid_argument(
                    "transaction_time_start and transaction_time_end must be provided together",
                );
            }
        };

        let scope = ContradictionScope {
            entity_kind,
            label: req.label,
            property: req.property,
            valid_time_window,
            transaction_time_window,
            limit: req.limit.unwrap_or(DEFAULT_CONTRADICTION_LIMIT),
            offset: req.offset.unwrap_or(0),
        };

        match self.db.find_contradictions(&scope) {
            Ok(scan) => self.success_json(self.contradiction_scan_to_json(&scan)),
            Err(e) => self.db_error(e),
        }
    }

    // ========================================================================
    // Counterfactual replay (Issue #3357 / PR #3743).
    // ========================================================================

    /// Serialize a [`DivergenceReport`], stamping the AC8 `counterfactual: true`
    /// per-response marker.
    #[cfg(feature = "semantic-temporal")]
    fn counterfactual_report_to_json(
        &self,
        r: &crate::experimental::temporal::counterfactual::DivergenceReport,
    ) -> serde_json::Value {
        let entity_json = |e: &crate::core::id::EntityId| {
            let (kind, id) = Self::lineage_entity_parts(*e);
            json!({ "entity_kind": kind, "id": id })
        };
        json!({
            "counterfactual": true,
            "excluded_writes": r.excluded_writes(),
            "unattributed_writes_encountered": r.unattributed_writes_encountered(),
            "orphaned_updates": r.orphaned_updates(),
            "entities_changed": r.entities_changed(),
            "entities_removed": r.entities_removed(),
            "changed_entities": r.changed_entities().iter().map(entity_json).collect::<Vec<_>>(),
            "removed_entities": r.removed_entities().iter().map(entity_json).collect::<Vec<_>>(),
        })
    }

    /// Handle `counterfactual_replay` (Read, Issue #3357).
    #[cfg(feature = "semantic-temporal")]
    fn handle_counterfactual_replay(&self, args: serde_json::Value) -> CallToolResult {
        use crate::experimental::temporal::counterfactual::{
            CounterfactualConfig, CounterfactualError, ExclusionPredicate,
        };

        let CounterfactualReplayRequest {
            name,
            exclude_source,
            exclude_sources,
            within_transaction_from,
            within_transaction_to,
            max_replay_versions,
        } = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let mut predicate = match (exclude_source, exclude_sources) {
            (Some(s), None) => ExclusionPredicate::source(s),
            (None, Some(list)) => {
                if list.is_empty() {
                    return self.invalid_argument("exclude_sources must be a non-empty array");
                }
                ExclusionPredicate::sources(list)
            }
            (None, None) => {
                return self.invalid_argument(
                    "provide exactly one of 'exclude_source' or 'exclude_sources'",
                );
            }
            (Some(_), Some(_)) => {
                return self.invalid_argument(
                    "provide only one of 'exclude_source' or 'exclude_sources', not both",
                );
            }
        };

        let from =
            match self.parse_opt_timestamp("within_transaction_from", &within_transaction_from) {
                Ok(v) => v,
                Err(r) => return r,
            };
        let to = match self.parse_opt_timestamp("within_transaction_to", &within_transaction_to) {
            Ok(v) => v,
            Err(r) => return r,
        };
        if from.is_some() || to.is_some() {
            predicate = predicate.within_transaction_time(from, to);
        }

        let mut config = CounterfactualConfig::default();
        if let Some(cap) = max_replay_versions {
            config.max_replay_versions = cap;
        }

        match self.db.counterfactual_replay(name, predicate, config) {
            Ok(view) => self.success_json(self.counterfactual_report_to_json(view.report())),
            Err(CounterfactualError::HistoryTooLarge { versions, cap }) => self.error_result(
                McpError::new(
                    McpErrorCode::FailedPrecondition,
                    format!(
                        "recorded history too large for counterfactual replay: {versions} \
                         versions exceeds cap of {cap}"
                    ),
                )
                .details(json!({ "versions": versions, "cap": cap })),
            ),
            Err(CounterfactualError::NotFound(m)) => {
                self.error_result(McpError::new(McpErrorCode::NotFound, m))
            }
            Err(CounterfactualError::Internal(m)) => {
                self.error_result(McpError::new(McpErrorCode::Internal, m))
            }
        }
    }

    // ========================================================================
    // Trust propagation (Issue #3382 / PR #3748). Gated on `semantic-reasoning`.
    // ========================================================================

    /// Structured FAILED_PRECONDITION for a `semantic-reasoning`-gated tool when
    /// the feature is not compiled into this build.
    #[cfg(not(feature = "semantic-reasoning"))]
    fn semantic_reasoning_unavailable(&self, tool: &str) -> CallToolResult {
        self.error_result(
            McpError::new(
                McpErrorCode::FailedPrecondition,
                format!(
                    "Tool '{tool}' requires the `semantic-reasoning` feature, which is not \
                     compiled into this build. Rebuild AletheiaDB with `--features \
                     semantic-reasoning` to enable it."
                ),
            )
            .details(json!({ "tool": tool, "required_feature": "semantic-reasoning" })),
        )
    }

    #[cfg(not(feature = "semantic-reasoning"))]
    fn handle_trust_breakdown(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_reasoning_unavailable("trust_breakdown")
    }

    #[cfg(not(feature = "semantic-reasoning"))]
    fn handle_list_trust_policies(&self, _args: serde_json::Value) -> CallToolResult {
        self.semantic_reasoning_unavailable("list_trust_policies")
    }

    /// Serialize a [`TrustBreakdown`] tree recursively.
    #[cfg(feature = "semantic-reasoning")]
    fn trust_breakdown_to_json(
        &self,
        b: &crate::experimental::reasoning::trust_propagation::TrustBreakdown,
    ) -> serde_json::Value {
        let (kind, id) = Self::lineage_entity_parts(b.reference.entity);
        json!({
            "reference": {
                "entity_kind": kind,
                "id": id,
                "version": b.reference.version.as_u64(),
            },
            "status": b.status.as_str(),
            "confidence": b.confidence,
            "source": b.source.as_str(),
            "combinator": b.combinator.map(|c| c.as_str()),
            "truncated": b.truncated,
            "children": b
                .children
                .iter()
                .map(|c| self.trust_breakdown_to_json(c))
                .collect::<Vec<_>>(),
        })
    }

    /// Whether any node in a breakdown tree was truncated.
    #[cfg(feature = "semantic-reasoning")]
    fn trust_breakdown_has_more(
        b: &crate::experimental::reasoning::trust_propagation::TrustBreakdown,
    ) -> bool {
        b.truncated || b.children.iter().any(Self::trust_breakdown_has_more)
    }

    /// Handle `trust_breakdown` (Read, Issue #3382).
    #[cfg(feature = "semantic-reasoning")]
    fn handle_trust_breakdown(&self, args: serde_json::Value) -> CallToolResult {
        use crate::core::id::EntityId;
        use crate::core::lineage::LineageRef;
        use crate::experimental::reasoning::trust_propagation::TrustOptions;

        let req: TrustBreakdownRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let entity = match req.entity_kind.trim().to_ascii_lowercase().as_str() {
            "node" => match NodeId::new(req.id) {
                Ok(id) => EntityId::Node(id),
                Err(e) => return self.invalid_argument(&e.to_string()),
            },
            "edge" => match EdgeId::new(req.id) {
                Ok(id) => EntityId::Edge(id),
                Err(e) => return self.invalid_argument(&e.to_string()),
            },
            other => {
                return self.invalid_argument(&format!(
                    "entity_kind must be 'node' or 'edge', got '{other}'"
                ));
            }
        };
        let version = match VersionId::new(req.version) {
            Ok(v) => v,
            Err(e) => return self.invalid_argument(&e.to_string()),
        };
        let root = LineageRef::new(entity, version);

        let mut options = TrustOptions::new();
        if let Some(d) = req.max_depth {
            options = options.with_max_depth(d);
        }
        if let Some(n) = req.limit {
            options = options.with_max_nodes(n);
        }
        if let Some(ts) = req.as_of_transaction_time {
            let t = match self.parse_timestamp(&ts) {
                Ok(t) => t,
                Err(m) => return self.invalid_argument(&m),
            };
            options = options.with_as_of(t);
        }

        let breakdown = self.db.trust_breakdown(root, &options);
        let mut response = self.trust_breakdown_to_json(&breakdown);
        if let Some(obj) = response.as_object_mut() {
            obj.insert(
                "has_more".to_string(),
                json!(Self::trust_breakdown_has_more(&breakdown)),
            );
        }
        self.success_json(response)
    }

    /// Handle `list_trust_policies` (Read, Issue #3382).
    #[cfg(feature = "semantic-reasoning")]
    fn handle_list_trust_policies(&self, args: serde_json::Value) -> CallToolResult {
        use crate::experimental::reasoning::trust_propagation::{
            MissingConfidencePolicy, TrustPolicy,
        };

        let _req: ListTrustPoliciesRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let view = self.db.list_trust_policies();
        let missing_str = |m: MissingConfidencePolicy| match m {
            MissingConfidencePolicy::Zero => "zero",
            MissingConfidencePolicy::Neutral => "neutral",
            MissingConfidencePolicy::Ignore => "ignore",
        };
        let policy_json = |p: &TrustPolicy| json!({ "combinator": p.combinator.as_str(), "missing": missing_str(p.missing) });
        let labels: Vec<serde_json::Value> = view
            .labels
            .iter()
            .map(|(label, policy)| json!({ "label": label, "policy": policy_json(policy) }))
            .collect();
        self.success_json(json!({
            "default": policy_json(&view.default),
            "labels": labels,
        }))
    }

    /// Handle the `audit_export` tool (Issue #3358).
    ///
    /// Produces a signed, offline-verifiable evidence artifact of an entity's
    /// complete bi-temporal history. The Ed25519 signing key is operator-
    /// provided out of band via the `ALETHEIADB_AUDIT_SIGNING_KEY` environment
    /// variable (a 32-byte hex seed); the secret is never returned or logged —
    /// only the public key travels in the artifact.
    fn handle_audit_export(&self, args: serde_json::Value) -> CallToolResult {
        use crate::audit::{AuditScope, AuditSigningKey, ExportOptions, SIGNING_KEY_ENV};

        let req: AuditExportRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        let scope = match req.entity_type.as_str() {
            "node" => match NodeId::new(req.entity_id) {
                Ok(id) => AuditScope::node(id),
                Err(e) => return self.invalid_argument(&e.to_string()),
            },
            "edge" => match crate::core::id::EdgeId::new(req.entity_id) {
                Ok(id) => AuditScope::edge(id),
                Err(e) => return self.invalid_argument(&e.to_string()),
            },
            other => {
                return self.invalid_argument(&format!(
                    "entity_type must be 'node' or 'edge', got '{other}'"
                ));
            }
        };

        // The signing key is a precondition supplied by the operator, not a
        // caller argument — a missing key is a FAILED_PRECONDITION, never a
        // silent unsigned export.
        let signing_key = match AuditSigningKey::from_env(SIGNING_KEY_ENV) {
            Ok(k) => k,
            Err(_) => {
                return self.failed_precondition(&format!(
                    "audit export requires an operator-provided Ed25519 signing key in the \
                     {SIGNING_KEY_ENV} environment variable (32-byte hex seed)"
                ));
            }
        };

        let mut options =
            ExportOptions::new(req.database_id.unwrap_or_else(|| "aletheiadb".to_string()));
        if !req.redact_keys.is_empty() {
            options = options.redact(req.redact_keys);
        }

        match self.db.audit_export(scope, &signing_key, &options) {
            Ok(export) => match serde_json::to_value(&export) {
                Ok(artifact) => self.success_json(json!({
                    "artifact": artifact,
                    "public_key": signing_key.public_key().to_hex(),
                    "entity_count": export.entity_count(),
                    "version_count": export.version_count(),
                    "chain_root": export.chain.root,
                })),
                Err(e) => self.error_result(McpError::new(
                    McpErrorCode::Internal,
                    format!("failed to serialize artifact: {e}"),
                )),
            },
            Err(crate::audit::AuditError::NoHistory(msg)) => self.error_result(McpError::new(
                McpErrorCode::NotFound,
                format!("no exportable history: {msg}"),
            )),
            Err(e) => self.error_result(McpError::new(
                McpErrorCode::Internal,
                format!("audit export failed: {e}"),
            )),
        }
    }

    /// Handle the `database_stats` tool (Issue #3222).
    ///
    /// Thin aggregator: delegates entirely to the public
    /// [`AletheiaDB::stats`] snapshot and serializes it — no storage logic
    /// lives here. The underlying getters are all O(1)/cached (see
    /// `src/db/stats.rs`), so this never triggers a version scan.
    ///
    /// `include_per_principal` gates the changefeed per-principal breakdown
    /// (Issue #3678): the scalar aggregates (`active_subscriptions`, the
    /// `#3368 resource_limits` counters, everything else) stay at the
    /// `database_stats` tool's `metrics` access tier, but the
    /// `changefeed.per_principal` identity list — which would let a
    /// lowest-privilege `metrics`/`reader` credential enumerate every other
    /// principal's id + live count — is **omitted** unless the caller is
    /// admin. Both surfaces route here: the MCP dispatch passes
    /// [`caller_is_admin`](Self::caller_is_admin) (session-derived), the HTTP
    /// route passes its own authenticated principal's admin-ness via
    /// [`database_stats_with_visibility`](Self::database_stats_with_visibility).
    fn handle_database_stats(
        &self,
        args: serde_json::Value,
        include_per_principal: bool,
    ) -> CallToolResult {
        // The tool takes no required arguments; clients may send no
        // `arguments` at all (surfaced here as JSON null) or an empty
        // object. Normalize null so both forms are accepted.
        let args = if args.is_null() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            args
        };
        let _req: DatabaseStatsRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        match serde_json::to_value(self.db.stats()) {
            Ok(mut value) => {
                // Surface the Issue #3368 over-limit termination counters
                // alongside the storage-layer stats. Additive: the
                // `DatabaseStats` struct and storage layer are untouched; these
                // are MCP-surface atomics folded in here. Row-cap breaches are
                // NOT terminations (they self-disclose via `truncated`/
                // `has_more`), so there is deliberately no row counter.
                if let Some(obj) = value.as_object_mut() {
                    let snap = self.limit_counters.snapshot();
                    // Engine-lane counters (Issue #3368 engine lane): terminations
                    // from the executor's cooperative `ResourceGuardIterator`
                    // (the `query` tool's memory budget + cooperative timeout, and
                    // every Rust-API `db.query()....with_*` caller). A distinct
                    // family from the MCP-surface atoms above, nested under
                    // `engine` so the two are never conflated or double-counted.
                    let engine = self.db.query_limit_counters();
                    obj.insert(
                        "resource_limits".to_string(),
                        json!({
                            "timeout_terminations": snap.wall_clock_timeout,
                            "byte_cap_terminations": snap.result_bytes,
                            "memory_terminations": snap.memory_bytes,
                            "override_rejections": snap.override_rejected,
                            "engine": {
                                "timeout_terminations": engine.wall_clock_timeout,
                                "row_cap_terminations": engine.result_rows,
                                "memory_terminations": engine.memory_bytes,
                                "override_rejections": engine.override_rejected,
                            },
                        }),
                    );
                }
                // Admin-gate the per-principal changefeed identity roster
                // (Issue #3678). For a non-admin caller the whole
                // `changefeed.per_principal` key is removed — not emptied — so
                // the response carries NO other principal's identity, while the
                // scalar `changefeed.active_subscriptions` aggregate remains
                // for monitoring at the metrics tier. Expressed as a single
                // `if let` over a combinator to avoid a nested (collapsible)
                // `if` without introducing a `let`-chain.
                if let Some(changefeed) = (!include_per_principal)
                    .then(|| {
                        value
                            .get_mut("changefeed")
                            .and_then(serde_json::Value::as_object_mut)
                    })
                    .flatten()
                {
                    changefeed.remove("per_principal");
                }
                self.success_json(value)
            }
            Err(e) => self.error_result(McpError::new(
                McpErrorCode::Internal,
                format!("Failed to serialize database stats: {}", e),
            )),
        }
    }

    /// Handle the `verify_chain` tool (Issue #3351): verify the tamper-evident
    /// provenance hash chain — full, entity-scoped, or against an exported
    /// anchor. Read-only; when the chain is not enabled the request is a
    /// structured `FAILED_PRECONDITION` (never a silent empty pass).
    fn handle_verify_chain(&self, args: serde_json::Value) -> CallToolResult {
        let args = if args.is_null() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            args
        };
        let req: VerifyChainRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        // Precedence: anchor extension > entity-scoped > full.
        let (verification, scope) = if let Some(anchor_value) = req.against {
            let anchor: crate::provenance_chain::ChainHead =
                match serde_json::from_value(anchor_value) {
                    Ok(a) => a,
                    Err(e) => {
                        return self.invalid_argument(&format!(
                            "`against` is not a valid exported chain head (as returned by \
                             export_chain_head): {e}"
                        ));
                    }
                };
            match self.db.verify_chain_against(&anchor) {
                Ok(v) => (v, "anchor"),
                Err(e) => return self.chain_not_enabled(e),
            }
        } else if req.entity_kind.is_some() || req.id.is_some() {
            let kind = match req.entity_kind.as_deref() {
                Some(k) => match k.trim().to_ascii_lowercase().as_str() {
                    "node" => crate::provenance_chain::EntityKind::Node,
                    "edge" => crate::provenance_chain::EntityKind::Edge,
                    other => {
                        return self.invalid_argument(&format!(
                            "entity_kind must be 'node' or 'edge', got '{other}'"
                        ));
                    }
                },
                None => {
                    return self.invalid_argument("entity_kind is required when `id` is supplied");
                }
            };
            let id = match req.id {
                Some(id) => id,
                None => {
                    return self
                        .invalid_argument("`id` is required when `entity_kind` is supplied");
                }
            };
            match self.db.verify_entity_chain(kind, id) {
                Ok(v) => (v, "entity"),
                Err(e) => return self.chain_not_enabled(e),
            }
        } else {
            match self.db.verify_chain() {
                Ok(v) => (v, "full"),
                Err(e) => return self.chain_not_enabled(e),
            }
        };

        self.success_json(json!({
            "scope": scope,
            "passed": verification.passed,
            "head_seq": verification.head_seq,
            "head_digest": verification.head_digest_hex,
            "earliest_broken_seq": verification.earliest_broken_seq,
            "reason": verification.reason,
            "transactions_checked": verification.transactions_checked,
        }))
    }

    /// Handle the `export_chain_head` tool (Issue #3351): export the current
    /// chain head as an external anchor for offline storage and later
    /// fork/rollback detection via `verify_chain`'s `against` argument.
    fn handle_export_chain_head(&self, args: serde_json::Value) -> CallToolResult {
        let args = if args.is_null() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            args
        };
        let _req: ExportChainHeadRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        match self.db.export_chain_head() {
            Ok(head) => match serde_json::to_value(&head) {
                Ok(value) => self.success_json(value),
                Err(e) => self.error_result(McpError::new(
                    McpErrorCode::Internal,
                    format!("Failed to serialize chain head: {}", e),
                )),
            },
            Err(e) => self.chain_not_enabled(e),
        }
    }

    /// Map the "chain not enabled" database error (Issue #3351) to a structured
    /// `FAILED_PRECONDITION` so a caller learns the chain must be enabled for
    /// this data dir rather than misreading a bare failure.
    fn chain_not_enabled(&self, e: crate::core::error::Error) -> CallToolResult {
        self.failed_precondition(&e.to_string())
    }

    /// Thread a namespace read scope into a [`QueryBuilder`] (Issue #3349, PR3d).
    ///
    /// `All` becomes the explicit no-op `in_all_namespaces` (identical to the
    /// pre-scope path); a single/union scope threads through
    /// `in_namespace`/`in_namespaces` so the executor applies the source-leaf
    /// filter + traversal boundary. Generic over the builder state so it composes
    /// at any point in the fluent chain.
    fn scoped_builder<S: crate::query::builder::QueryState>(
        builder: crate::query::QueryBuilder<S>,
        scope: &crate::core::namespace::NamespaceScope,
    ) -> crate::query::QueryBuilder<S> {
        use crate::core::namespace::NamespaceScope;
        match scope {
            NamespaceScope::All => builder.in_all_namespaces(),
            NamespaceScope::Single(ns) => builder.in_namespace(ns.clone()),
            NamespaceScope::List(list) => builder.in_namespaces(list.iter().cloned()),
        }
    }

    fn handle_hybrid_query(&self, args: serde_json::Value) -> CallToolResult {
        // Issue #3349 (PR3d): real namespace scoping. Parse the optional
        // `namespace` read scope (omitted ⇒ `default`-only, `"all"` ⇒ no filter),
        // then validate it up front so unknown ns ⇒ NOT_FOUND / empty list ⇒
        // INVALID_ARGUMENT is returned even on the single-node paths that don't
        // route through the scope-validating executor.
        let scope = match self.parse_opt_scope(&args.get("namespace").cloned()) {
            Ok(s) => s,
            Err(result) => return result,
        }
        .unwrap_or_default();
        if let Err(e) = self.db.validate_scope(&scope) {
            return self.db_error(e);
        }
        let scope_is_restricting = !matches!(scope, crate::core::namespace::NamespaceScope::All);

        // Issue #3348: parse the optional provenance filter before consuming args.
        let prov_filter = match self.parse_provenance_filter(&args) {
            Ok(f) => f,
            Err(result) => return result,
        };

        // Issue #3372: parse the optional fusion policy (feature-gated) before
        // consuming args. When present, results are re-ranked by the fused score
        // and each carries a `score_breakdown`; omitting it is unchanged.
        #[cfg(feature = "semantic-retrieval-fusion")]
        let fusion_policy = match self.parse_fusion_policy(&args) {
            Ok(p) => p,
            Err(result) => return result,
        };

        let req: HybridQueryRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => return self.invalid_argument(&format!("Invalid arguments: {}", e)),
        };

        // Apply resource limits
        let limit = req
            .limit
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .min(MAX_RESULT_LIMIT);
        let depth = req.traverse_depth.unwrap_or(1).min(MAX_TRAVERSAL_DEPTH);
        let k = req.top_k.unwrap_or(DEFAULT_VECTOR_K).min(MAX_VECTOR_K);

        // Parse temporal parameters if provided
        let valid_time = if let Some(ref vt) = req.valid_time {
            match self.parse_timestamp(vt) {
                Ok(t) => Some(t),
                Err(e) => return self.invalid_argument(&format!("Invalid valid_time: {}", e)),
            }
        } else {
            None
        };

        let tx_time = if let Some(ref tt) = req.transaction_time {
            match self.parse_timestamp(tt) {
                Ok(t) => Some(t),
                Err(e) => {
                    return self.invalid_argument(&format!("Invalid transaction_time: {}", e));
                }
            }
        } else {
            None
        };

        let include_vectors = req.include_vectors.unwrap_or(false);

        // One request-scoped wallclock for every entity in the response
        // (Issue #3391).
        let now = time::now();

        // Helper to convert rows to hybrid results with temporal info. Issue
        // #3348: rows whose per-version provenance fails the filter are
        // dropped here; ranked (similarity) order among survivors is
        // preserved (no reordering).
        let prov_filter_ref = prov_filter.as_ref();
        let rows_to_results =
            |rows: Vec<crate::query::executor::QueryRow>| -> Vec<HybridQueryResult> {
                rows.into_iter()
                    .filter_map(|row| {
                        if let EntityResult::Node(node) = row.entity {
                            let node = self.node_to_response(&node, include_vectors, now);
                            if !prov_filter_ref.is_none_or(|f| f.matches(node.provenance.as_ref()))
                            {
                                return None;
                            }
                            Some(HybridQueryResult {
                                node,
                                similarity_score: row.score,
                                traversal_path: row.path.map(|p| {
                                    p.iter()
                                        .map(|e| match e {
                                            ResultEntityId::Node(id) => id.as_u64(),
                                            ResultEntityId::Edge(id) => id.as_u64(),
                                        })
                                        .collect()
                                }),
                                timestamp: row.timestamp.map(|t| t.wallclock().to_string()),
                                score_breakdown: None,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            };

        // Use QueryBuilder for hybrid queries
        if let Some(start_id) = req.start_node_id {
            let node_id = match NodeId::new(start_id) {
                Ok(id) => id,
                // Bare StorageError text verbatim — see the note on the other
                // ID-validation sites.
                Err(e) => return self.invalid_argument(&e.to_string()),
            };

            // If temporal filtering requested, use temporal query
            if let (Some(vt), Some(tt)) = (valid_time, tx_time) {
                // Temporal query for a single node
                return match self.db.get_node_at_time(node_id, vt, tt) {
                    Ok(node) => {
                        // Issue #3349 (PR3d): a start node outside the read scope
                        // is filtered out (empty result), never leaked.
                        let in_scope = !scope_is_restricting || scope.contains(&node.namespace());
                        let response = self.node_to_response(&node, include_vectors, now);
                        // Issue #3348: apply the provenance filter (evaluated on
                        // the version resolved at this coordinate) to the single
                        // result; a fail yields an empty result set, never a
                        // fabricated row.
                        #[allow(unused_mut)]
                        let mut results: Vec<HybridQueryResult> = if in_scope
                            && prov_filter_ref
                                .is_none_or(|f| f.matches(response.provenance.as_ref()))
                        {
                            vec![HybridQueryResult {
                                node: response,
                                similarity_score: None,
                                traversal_path: Some(vec![node_id.as_u64()]),
                                timestamp: Some(vt.wallclock().to_string()),
                                score_breakdown: None,
                            }]
                        } else {
                            Vec::new()
                        };
                        // Issue #3372 (MED-3): attach the fused `score_breakdown`
                        // on the single-node AS-OF path too. Ranking is a no-op
                        // for one result; AC6 confidence/recency are resolved at
                        // the (valid, tx) coordinate here.
                        #[cfg(feature = "semantic-retrieval-fusion")]
                        if let Some(policy) = fusion_policy.as_ref() {
                            self.apply_fusion_to_hybrid(&mut results, policy, valid_time, tx_time);
                        }
                        self.success_json(json!({
                            "results": results,
                            "count": results.len(),
                            "temporal_query": {
                                "valid_time": req.valid_time,
                                "transaction_time": req.transaction_time
                            }
                        }))
                    }
                    Err(e) => self.db_error(e),
                };
            }

            // Graph-first query with optional vector ranking
            let builder = crate::query::QueryBuilder::new().start(node_id);

            let builder = if let Some(ref edge_label) = req.traverse_edge {
                if depth > 1 {
                    builder.traverse_n(edge_label, depth)
                } else {
                    builder.traverse(edge_label)
                }
            } else {
                // Just return the start node
                return match self.db.get_node(node_id) {
                    Ok(node) => {
                        // Issue #3349 (PR3d): filter an out-of-scope start node.
                        let in_scope = !scope_is_restricting || scope.contains(&node.namespace());
                        let response = self.node_to_response(&node, include_vectors, now);
                        // Issue #3348: filter the single start node.
                        #[allow(unused_mut)]
                        let mut results: Vec<HybridQueryResult> = if in_scope
                            && prov_filter_ref
                                .is_none_or(|f| f.matches(response.provenance.as_ref()))
                        {
                            vec![HybridQueryResult {
                                node: response,
                                similarity_score: None,
                                traversal_path: Some(vec![node_id.as_u64()]),
                                timestamp: None,
                                score_breakdown: None,
                            }]
                        } else {
                            Vec::new()
                        };
                        // Issue #3372 (MED-3): attach the fused `score_breakdown`
                        // on the single-node (no-traverse) path too. Ranking is a
                        // no-op for one result; with no full (valid, tx) AS-OF
                        // set here, confidence/recency resolve from the current
                        // version (documented partial-coordinate limitation).
                        #[cfg(feature = "semantic-retrieval-fusion")]
                        if let Some(policy) = fusion_policy.as_ref() {
                            self.apply_fusion_to_hybrid(&mut results, policy, valid_time, tx_time);
                        }
                        self.success_json(json!({
                            "results": results,
                            "count": results.len(),
                        }))
                    }
                    Err(e) => self.db_error(e),
                };
            };

            // Execute and collect results. Issue #3349 (PR3d): thread the read
            // scope through the executor (source-leaf filter + traversal
            // boundary) so a graph-first hybrid never crosses an out-of-scope
            // edge/node.
            let builder = Self::scoped_builder(builder, &scope);
            match builder.limit(limit).execute(&self.db) {
                Ok(results) => match results.collect_all() {
                    Ok(rows) => {
                        #[allow(unused_mut)]
                        let mut hybrid_results = rows_to_results(rows);
                        #[cfg(feature = "semantic-retrieval-fusion")]
                        if let Some(policy) = fusion_policy.as_ref() {
                            self.apply_fusion_to_hybrid(
                                &mut hybrid_results,
                                policy,
                                valid_time,
                                tx_time,
                            );
                        }
                        self.success_json(json!({
                            "results": hybrid_results,
                            "count": hybrid_results.len()
                        }))
                    }
                    Err(e) => self.db_error(e),
                },
                Err(e) => self.db_error(e),
            }
        } else if let Some(ref embedding) = req.query_embedding {
            // Vector-first query
            // Use vector_property if specified
            let property_name = req.vector_property.as_deref().unwrap_or("embedding");

            // Check if vector index is enabled for the property
            if !self.db.is_vector_index_enabled_for(property_name) {
                return self.failed_precondition(&format!(
                    "Vector index not enabled for property '{}'. Use enable_vector_index first.",
                    property_name
                ));
            }

            // Validate embedding dimensions
            if let Err(e) = self.validate_embedding_dimensions(embedding, property_name) {
                return self.invalid_argument(&e);
            }

            // Issue #3372 (MED-2): with a fusion policy, reject a non-Cosine
            // index up front — mirroring `handle_find_similar_fused` — rather
            // than feeding a distance whose scores are not in [0,1] into the
            // fused score.
            #[cfg(feature = "semantic-retrieval-fusion")]
            if fusion_policy.is_some()
                && let Some(err) = self.fusion_metric_precondition(property_name)
            {
                return err;
            }

            // Issue #3372 (HIGH-1): when a fusion policy is present, over-fetch
            // the `k`-scaled fused horizon (never the similarity top-k) so a
            // geometrically-far but high-trust candidate can still enter the
            // fused ranking, rather than degenerating into a post-hoc re-sort of
            // the similarity top-k. Mirrors `handle_find_similar_fused`'s
            // `fused_horizon(k)` over-fetch.
            #[cfg(feature = "semantic-retrieval-fusion")]
            let fusion_fetch = fusion_policy
                .as_ref()
                .map(|_| crate::db::fusion::fused_horizon(k).min(MAX_VECTOR_K));
            #[cfg(not(feature = "semantic-retrieval-fusion"))]
            let fusion_fetch: Option<usize> = None;

            // Issue #3348 (AC6) / #3349 (PR3d): with a provenance filter OR a
            // restricting namespace scope, over-fetch the candidate horizon so
            // the returned top rows are all filter-passing (filter-complete)
            // rather than a short post-truncation — mirroring
            // `find_similar_scoped`. Ranked order is preserved; the page is
            // truncated back to `limit` after filtering. The namespace filter is
            // applied by the executor (source-leaf scope filter) and stays
            // ignorant of *why* a vector is absent (e.g. a crypto-shredded
            // embedding is simply never a candidate), composing with GDPR HNSW
            // exclusion.
            let (fetch_k, fetch_limit) = if prov_filter.is_some() || scope_is_restricting {
                (MAX_VECTOR_K, MAX_VECTOR_K)
            } else if let Some(horizon) = fusion_fetch {
                (horizon, horizon)
            } else {
                (k, limit)
            };
            let builder = crate::query::QueryBuilder::new().find_similar(embedding, fetch_k);
            let builder = Self::scoped_builder(builder, &scope);

            match builder.limit(fetch_limit).execute(&self.db) {
                Ok(results) => match results.collect_all() {
                    Ok(rows) => {
                        let mut hybrid_results = rows_to_results(rows);
                        // Issue #3372: re-rank the candidate set by the fused
                        // score BEFORE truncating to `limit`, so a high-trust
                        // candidate below the similarity-only page still surfaces.
                        #[cfg(feature = "semantic-retrieval-fusion")]
                        if let Some(policy) = fusion_policy.as_ref() {
                            self.apply_fusion_to_hybrid(
                                &mut hybrid_results,
                                policy,
                                valid_time,
                                tx_time,
                            );
                        }
                        hybrid_results.truncate(limit);
                        self.success_json(json!({
                            "results": hybrid_results,
                            "count": hybrid_results.len(),
                            "vector_property": property_name
                        }))
                    }
                    Err(e) => self.db_error(e),
                },
                Err(e) => self.db_error(e),
            }
        } else if let Some(ref label) = req.filter_label {
            // Label scan query. Issue #3349 (PR3d): thread the read scope.
            let builder = crate::query::QueryBuilder::new().scan_label(label);
            let builder = Self::scoped_builder(builder, &scope);

            match builder.limit(limit).execute(&self.db) {
                Ok(results) => match results.collect_all() {
                    Ok(rows) => {
                        #[allow(unused_mut)]
                        let mut hybrid_results = rows_to_results(rows);
                        #[cfg(feature = "semantic-retrieval-fusion")]
                        if let Some(policy) = fusion_policy.as_ref() {
                            self.apply_fusion_to_hybrid(
                                &mut hybrid_results,
                                policy,
                                valid_time,
                                tx_time,
                            );
                        }
                        self.success_json(json!({
                            "results": hybrid_results,
                            "count": hybrid_results.len()
                        }))
                    }
                    Err(e) => self.db_error(e),
                },
                Err(e) => self.db_error(e),
            }
        } else {
            self.invalid_argument(
                "Must specify either start_node_id, query_embedding, or filter_label",
            )
        }
    }

    /// Reject a non-Cosine vector index for a provenance-weighted fused search
    /// (Issue #3372). Fusion's similarity term assumes an already-`[0,1]` score,
    /// true only for Cosine; a Euclidean/dot index would silently distort the
    /// fused score. Returns the structured `FAILED_PRECONDITION`
    /// (`FusionError::UnsupportedMetric`) when the queried property's index uses
    /// another metric, else `None`. Shared by the `find_similar` and
    /// `hybrid_query` fused paths.
    #[cfg(feature = "semantic-retrieval-fusion")]
    fn fusion_metric_precondition(&self, property_name: &str) -> Option<CallToolResult> {
        if let Some(info) = self
            .db
            .list_vector_indexes()
            .into_iter()
            .find(|i| i.property_name == property_name)
            && info.distance_metric != DistanceMetric::Cosine
        {
            return Some(self.failed_precondition(&format!(
                "provenance-weighted fusion requires a Cosine vector index; the index for '{}' \
                 uses {:?}, whose scores are not in [0,1] (v1 supports Cosine only)",
                property_name, info.distance_metric
            )));
        }
        None
    }

    /// Re-rank `results` by the provenance-weighted fused score (Issue #3372)
    /// and attach a `score_breakdown` to each. Confidence/recency are read from
    /// the version resolved at the bi-temporal coordinate when a full `(valid,
    /// tx)` `AS OF` is set (AC6), else from the current version. Recency is
    /// evaluated against the `valid_time` coordinate when set, else `now`.
    #[cfg(feature = "semantic-retrieval-fusion")]
    fn apply_fusion_to_hybrid(
        &self,
        results: &mut Vec<HybridQueryResult>,
        policy: &crate::db::fusion::FusionPolicy,
        valid_time: Option<Timestamp>,
        tx_time: Option<Timestamp>,
    ) {
        let reference_now = valid_time.map_or_else(|| time::now().wallclock(), |t| t.wallclock());
        let mut scored: Vec<(f64, HybridQueryResult)> = std::mem::take(results)
            .into_iter()
            .map(|mut r| {
                let (confidence, valid_from_micros, recency_defaulted) =
                    NodeId::new(r.node.id).ok().map_or((None, 0, true), |id| {
                        self.hybrid_fusion_inputs(id, valid_time, tx_time)
                    });
                let recency = if recency_defaulted {
                    crate::db::fusion::DEFAULT_NEUTRAL_RECENCY
                } else {
                    policy.recency(reference_now, valid_from_micros)
                };
                let similarity = r.similarity_score.map_or(0.0, f64::from);
                let mut breakdown = policy.fuse(similarity, confidence, recency);
                breakdown.recency_defaulted = recency_defaulted;
                r.score_breakdown = Some(Self::fusion_breakdown_to_json(&breakdown));
                (breakdown.fused, r)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.node.id.cmp(&b.1.node.id))
        });
        *results = scored.into_iter().map(|(_, r)| r).collect();
    }

    /// Resolve a hybrid result node's fusion inputs (Issue #3372): when a full
    /// bi-temporal `(valid, tx)` coordinate is set, read from the version at
    /// that coordinate (AC6); otherwise from the current version.
    #[cfg(feature = "semantic-retrieval-fusion")]
    fn hybrid_fusion_inputs(
        &self,
        id: NodeId,
        valid_time: Option<Timestamp>,
        tx_time: Option<Timestamp>,
    ) -> (Option<f64>, i64, bool) {
        let version_id = if let (Some(vt), Some(tt)) = (valid_time, tx_time) {
            match self.db.get_node_at_time(id, vt, tt) {
                Ok(node) => node.current_version,
                Err(_) => return (None, 0, true),
            }
        } else {
            match self.db.get_node(id) {
                Ok(node) => node.current_version,
                Err(_) => return (None, 0, true),
            }
        };
        self.node_fusion_inputs(version_id)
    }

    // ========================================================================
    // Declarative query tool (read-only Cypher / AQL) -- Issue #3213
    // ========================================================================

    /// Build a structured query-tool error payload.
    ///
    /// Distinct `kind`s let an LLM self-correct on retry: `invalid_request`,
    /// `read_only_violation`, `language_unavailable`, `parse_error`,
    /// `unsupported_construct`, `invalid_params`, `runtime_error`.
    ///
    /// The `kind` field is the query tool's own published contract (Issue
    /// #3213) and is preserved verbatim; the uniform `code`/`retriable`
    /// fields (Issue #3234) are added additively, derived from `kind`.
    fn query_error(
        &self,
        kind: &str,
        message: &str,
        clause: Option<&str>,
        language: Option<&str>,
    ) -> CallToolResult {
        let (code, retriable) = query_kind_classification(kind);
        self.query_error_classified(kind, code, retriable, message, clause, language)
    }

    /// [`query_error`](Self::query_error) with an explicit `code`/`retriable`
    /// classification, for callers that can classify more precisely than the
    /// kind-derived default (e.g. a `runtime_error` caused by a timeout is
    /// `UNAVAILABLE`/retriable rather than `INTERNAL`).
    fn query_error_classified(
        &self,
        kind: &str,
        code: McpErrorCode,
        retriable: bool,
        message: &str,
        clause: Option<&str>,
        language: Option<&str>,
    ) -> CallToolResult {
        let mut obj = serde_json::Map::new();
        obj.insert("kind".to_string(), json!(kind));
        obj.insert("code".to_string(), json!(code.as_str()));
        obj.insert("retriable".to_string(), json!(retriable));
        obj.insert("message".to_string(), json!(message));
        if let Some(clause) = clause {
            obj.insert("clause".to_string(), json!(clause));
        }
        if let Some(language) = language {
            obj.insert("language".to_string(), json!(language));
        }
        CallToolResult::error(vec![Content::text(
            json!({ "error": serde_json::Value::Object(obj) }).to_string(),
        )])
    }

    /// [`query_error_classified`](Self::query_error_classified) plus a
    /// structured `details` object (Issue #3368): the resource-limit errors
    /// carry `details: {dimension, limit, consumed?}` (or `{dimension,
    /// requested, ceiling}` for a rejected override) so an LLM can read the
    /// violated dimension and its bound and self-correct without parsing the
    /// human message.
    fn query_error_with_details(
        &self,
        kind: &str,
        code: McpErrorCode,
        retriable: bool,
        message: &str,
        language: Option<&str>,
        details: serde_json::Value,
    ) -> CallToolResult {
        let mut obj = serde_json::Map::new();
        obj.insert("kind".to_string(), json!(kind));
        obj.insert("code".to_string(), json!(code.as_str()));
        obj.insert("retriable".to_string(), json!(retriable));
        obj.insert("message".to_string(), json!(message));
        if let Some(language) = language {
            obj.insert("language".to_string(), json!(language));
        }
        obj.insert("details".to_string(), details);
        CallToolResult::error(vec![Content::text(
            json!({ "error": serde_json::Value::Object(obj) }).to_string(),
        )])
    }

    /// Map an engine error from query execution into a structured query-tool error.
    fn map_query_error(&self, error: crate::core::error::Error, language: &str) -> CallToolResult {
        use crate::core::error::{Error, QueryError};
        use crate::core::namespace::NamespaceError;
        match error {
            Error::Query(QueryError::SyntaxError { message }) => {
                self.query_error("parse_error", &message, None, Some(language))
            }
            // Issue #3349 (PR3d): a namespace scope error surfaces through the
            // query envelope carrying `details.namespace` (the offending name),
            // so a caller self-corrects exactly as on the structured scoped read
            // tools. All namespace errors are caller faults (non-retriable).
            Error::Namespace(ns_err) => {
                let (kind, code, details) = match &ns_err {
                    NamespaceError::NotFound { namespace } => (
                        "runtime_error",
                        McpErrorCode::NotFound,
                        json!({ "namespace": namespace }),
                    ),
                    NamespaceError::InvalidName { name, .. } => (
                        "invalid_params",
                        McpErrorCode::InvalidArgument,
                        json!({ "namespace": name }),
                    ),
                    // Issue #3349: a programmatic scope colliding with an
                    // in-query `USE / IN NAMESPACE` clause is a malformed
                    // *request* (the caller specified the scope in two
                    // conflicting places) — `invalid_request`, not
                    // `unsupported_construct` (the clause IS supported now).
                    // No `details.namespace` is carried: the message and
                    // details must never disclose an isolation boundary.
                    NamespaceError::ScopeConflict => (
                        "invalid_request",
                        McpErrorCode::InvalidArgument,
                        serde_json::Value::Null,
                    ),
                    NamespaceError::AlreadyExists { .. }
                    | NamespaceError::ReservedPropertyKey { .. }
                    | NamespaceError::Immutable => (
                        "invalid_params",
                        McpErrorCode::InvalidArgument,
                        serde_json::Value::Null,
                    ),
                };
                let message = Error::Namespace(ns_err).to_string();
                self.query_error_with_details(kind, code, false, &message, Some(language), details)
            }
            Error::Query(QueryError::UnsupportedFeature { feature }) => self.query_error(
                "unsupported_construct",
                &format!("Unsupported query construct: {feature}"),
                None,
                Some(language),
            ),
            Error::Query(QueryError::InvalidParameter { parameter, reason }) => self.query_error(
                "invalid_params",
                &format!("Invalid parameter '{parameter}': {reason}"),
                None,
                Some(language),
            ),
            // A type mismatch is a caller-repairable argument fault (e.g. a
            // provenance accessor compared to the wrong type, `confidence(r) =
            // 'high'`, Issue #3354a) -- surface it as `invalid_params` per the
            // query tool's structured-error contract, never a silent empty
            // result.
            Error::Query(QueryError::TypeMismatch { expected, actual }) => self.query_error(
                "invalid_params",
                &format!("Type mismatch: expected {expected}, got {actual}"),
                None,
                Some(language),
            ),
            Error::Query(QueryError::ExecutionError { message }) => {
                self.query_error("runtime_error", &message, None, Some(language))
            }
            // A per-query engine-lane resource limit was breached mid-drain
            // (Issue #3368 cooperative enforcement): the executor's
            // `ResourceGuardIterator` aborted the scan cooperatively rather
            // than letting it run to completion. Surfaced with the same
            // structured `details.{dimension, limit, consumed}` shape the
            // MCP-surface timeout/byte-cap builders already use, so a caller
            // branches on `details.dimension` identically regardless of
            // which layer (MCP-surface or engine) caught the breach.
            // `retriable` is threaded straight from the engine error: `true`
            // only for the wall-clock dimension (a read-only re-run is
            // safe), `false` for the memory-budget dimension (a re-run
            // deterministically breaches again).
            Error::Query(QueryError::ResourceExhausted {
                dimension,
                limit,
                consumed,
                retriable,
            }) => {
                let message = Error::Query(QueryError::ResourceExhausted {
                    dimension,
                    limit,
                    consumed,
                    retriable,
                })
                .to_string();
                self.query_error_with_details(
                    "runtime_error",
                    McpErrorCode::ResourceExhausted,
                    retriable,
                    &message,
                    Some(language),
                    json!({
                        "dimension": dimension,
                        "limit": limit,
                        "consumed": consumed,
                    }),
                )
            }
            // Anything else keeps kind "runtime_error" (the tool's own
            // contract) but classifies code/retriable from the actual error,
            // so e.g. a timeout is UNAVAILABLE/retriable, not INTERNAL.
            other => {
                let classified = McpError::from_db_error(&other);
                self.query_error_classified(
                    "runtime_error",
                    classified.code(),
                    classified.is_retriable(),
                    &other.to_string(),
                    None,
                    Some(language),
                )
            }
        }
    }

    /// Serialize a single query row (entity + score/path/timestamp) to JSON.
    ///
    /// Query rows carry only id/label/properties -- no provenance or
    /// temporal block -- so entities are serialized directly instead of
    /// through `node_to_response`/`edge_to_response`, which would pay a
    /// per-entity version-metadata lookup just to discard the result
    /// (Issue #3391).
    /// Serialize a single query-result entity (node/edge/id/null) to JSON.
    ///
    /// Shared by the single-entity row path and the multi-variable binding path
    /// (#549) so both render nodes/edges identically. `include_vectors: true` --
    /// query rows carry stored properties; vector elision is not applied here.
    fn query_entity_to_json(&self, entity: &EntityResult) -> serde_json::Value {
        match entity {
            EntityResult::Node(node) => json!({
                "type": "node",
                "id": node.id.as_u64(),
                "label": self.interned_to_string(node.label),
                "properties": self.property_map_to_json(&node.properties, true),
            }),
            EntityResult::Edge(edge) => json!({
                "type": "edge",
                "id": edge.id.as_u64(),
                "source_id": edge.source.as_u64(),
                "target_id": edge.target.as_u64(),
                "label": self.interned_to_string(edge.label),
                "properties": self.property_map_to_json(&edge.properties, true),
            }),
            EntityResult::NodeId(id) => json!({"type": "node", "id": id.as_u64()}),
            EntityResult::EdgeId(id) => json!({"type": "edge", "id": id.as_u64()}),
            // Null binding from an unmatched OPTIONAL MATCH pattern: surface
            // as JSON null so an LLM/caller sees the preserved row explicitly.
            EntityResult::Null => serde_json::Value::Null,
        }
    }

    fn query_row_to_json(&self, row: crate::query::executor::QueryRow) -> serde_json::Value {
        // Multi-variable binding row (#549): `MATCH (a),(b) RETURN a,b` binds
        // several variables, which the single `entity` field cannot represent
        // (its `entity` is `EntityResult::Null`). Serialize each bound variable
        // under its name, MERGED with any scalar `columns` (property/alias
        // projections). Checked BEFORE the columns-only branch because a binding
        // row can also carry columns. Without this, the row would render as a
        // lossy `{"entity": null}` and drop every bound entity.
        if let Some(bindings) = row.bindings {
            let mut obj = serde_json::Map::with_capacity(
                bindings.len() + row.columns.as_ref().map_or(0, Vec::len),
            );
            for (name, entity) in bindings {
                obj.insert(name, self.query_entity_to_json(&entity));
            }
            if let Some(columns) = row.columns {
                for (name, value) in columns {
                    obj.insert(name, self.property_value_to_json(&value, true));
                }
            }
            return serde_json::Value::Object(obj);
        }
        // Computed/aggregate row (e.g. `RETURN count(*)`, `RETURN n.dept,
        // count(*)`): the meaningful payload lives in `row.columns`, not
        // `row.entity` (which is `EntityResult::Null`). Render each named column
        // via `property_value_to_json` so an LLM sees the aggregate/group value
        // instead of a lossy `entity: null`. Closes the #558 MCP-surface
        // follow-up. `include_vectors: true` -- computed aggregate values are
        // not stored embeddings, so nothing should be elided.
        if let Some(columns) = row.columns {
            let mut obj = serde_json::Map::with_capacity(columns.len());
            for (name, value) in columns {
                obj.insert(name, self.property_value_to_json(&value, true));
            }
            return serde_json::Value::Object(obj);
        }
        let entity = self.query_entity_to_json(&row.entity);
        json!({
            "entity": entity,
            "score": row.score,
            "path": row.path.map(|p| {
                p.iter()
                    .map(|e| match e {
                        ResultEntityId::Node(id) => json!({"type": "node", "id": id.as_u64()}),
                        ResultEntityId::Edge(id) => json!({"type": "edge", "id": id.as_u64()}),
                    })
                    .collect::<Vec<_>>()
            }),
            "timestamp": row.timestamp.map(|t| t.wallclock().to_string()),
        })
    }

    /// Convert JSON parameter bindings into Cypher parameter values.
    #[cfg(feature = "cypher")]
    fn json_to_cypher_params(
        &self,
        params: Option<&HashMap<String, serde_json::Value>>,
    ) -> std::result::Result<HashMap<String, crate::cypher::CypherParameterValue>, (String, String)>
    {
        use crate::cypher::CypherParameterValue;
        let mut out = HashMap::new();
        let Some(map) = params else {
            return Ok(out);
        };
        for (key, value) in map {
            let pv = match value {
                serde_json::Value::Null => CypherParameterValue::Null,
                serde_json::Value::Bool(b) => CypherParameterValue::Bool(*b),
                serde_json::Value::Number(n) => {
                    // Check is_f64() first so that JSON floats (e.g. 1.0) are not
                    // silently coerced to Int by as_i64(), which would succeed for
                    // whole-number floats in some representations.
                    if n.is_f64() {
                        CypherParameterValue::Float(n.as_f64().unwrap())
                    } else if let Some(i) = n.as_i64() {
                        CypherParameterValue::Int(i)
                    } else if let Some(f) = n.as_f64() {
                        CypherParameterValue::Float(f)
                    } else {
                        return Err((key.clone(), "unsupported numeric value".to_string()));
                    }
                }
                serde_json::Value::String(s) => CypherParameterValue::String(s.clone()),
                serde_json::Value::Array(arr) => {
                    if arr.is_empty() {
                        return Err((
                            key.clone(),
                            "array parameters must not be empty; embeddings require at least \
                             one dimension"
                                .to_string(),
                        ));
                    }
                    let mut floats = Vec::with_capacity(arr.len());
                    for element in arr {
                        match element.as_f64() {
                            Some(f) => floats.push(f as f32),
                            None => {
                                return Err((
                                    key.clone(),
                                    "array parameters must contain only numbers (numeric arrays \
                                     are treated as embeddings)"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                    CypherParameterValue::Embedding(Arc::from(floats))
                }
                serde_json::Value::Object(_) => {
                    return Err((
                        key.clone(),
                        "object parameters are not supported".to_string(),
                    ));
                }
            };
            out.insert(key.clone(), pv);
        }
        Ok(out)
    }

    fn handle_query(&self, args: serde_json::Value) -> CallToolResult {
        // Extract language early so it can appear in error payloads even when
        // full deserialization fails (language is not yet known at that point).
        let raw_language = args
            .get("language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase());

        // Cursor paging is not supported for the declarative query tool in v1
        // (Issue #3360); captured before `args` is consumed by deserialization.
        let cursor_requested = Self::cursor_requested(&args);

        // Issue #3349: real declarative (AQL/Cypher) namespace scoping. Parse the
        // optional `namespace` read scope as the *programmatic* scope `P`. It is
        // kept as an `Option` (NOT collapsed to `default` here): the executor's
        // `reconcile_namespace_scope` distinguishes an omitted parameter (`None`
        // — let any in-query `USE / IN NAMESPACE` clause govern, else the
        // fail-closed `default` namespace) from an explicit one (`Some` — a hard
        // ceiling that must match any in-query clause). `"all"` imposes no
        // filter. Captured before `args` is consumed by deserialization;
        // validated (unknown ns ⇒ NOT_FOUND, empty list ⇒ INVALID_ARGUMENT)
        // inside the scoped executor.
        let scope = match self.parse_opt_scope(&args.get("namespace").cloned()) {
            Ok(s) => s,
            Err(result) => return result,
        };

        let req: QueryRequest = match serde_json::from_value(args) {
            Ok(r) => r,
            Err(e) => {
                return self.query_error(
                    "invalid_request",
                    &format!("Invalid arguments: {e}"),
                    None,
                    raw_language.as_deref(),
                );
            }
        };

        let language = req.language.to_ascii_lowercase();
        if language != "cypher" && language != "aql" {
            return self.query_error(
                "invalid_request",
                &format!(
                    "Unsupported query language '{}'. Use \"cypher\" or \"aql\".",
                    req.language
                ),
                None,
                Some(&language),
            );
        }

        // Cursor paging (Issue #3360) is not supported for the declarative
        // query tool in v1: arbitrary result shapes (projections, aggregates,
        // ordering) have no snapshot-anchored keyset to page over. Return a
        // structured `unsupported_construct` error rather than silently
        // serving a single truncated page (AC7: no silent fallback). Callers
        // needing consistent, resumable scans use `list_nodes` /
        // `find_nodes_at_time`, which are cursor-paged.
        if cursor_requested {
            return self.query_error(
                "unsupported_construct",
                "Cursor paging is not supported for the `query` tool in v1. Use `list_nodes` or \
                 `find_nodes_at_time` for snapshot-anchored, resumable cursor scans; the `query` \
                 tool returns a single (optionally `limit`-bounded) result set.",
                None,
                Some(&language),
            );
        }

        // Read-only guard: reject mutating statements BEFORE any execution.
        // Runs for every language so the tool can never write, even if the
        // grammars later gain write support.
        if let Some(clause) = detect_mutating_clause(&req.query) {
            return self.query_error(
                "read_only_violation",
                &format!(
                    "The `query` tool is read-only; the `{clause}` clause would mutate state and \
                     is rejected before execution."
                ),
                Some(clause),
                Some(&language),
            );
        }

        // Resolve the per-query resource limits (Issue #3368): fold the server
        // defaults, operator ceilings, and the per-call `limits` override. An
        // override that exceeds a ceiling is rejected up front, BEFORE any
        // execution, as a structured `invalid_params`/INVALID_ARGUMENT error
        // whose `details` name the dimension, the requested value, and the
        // ceiling so the caller can shrink and retry.
        let effective = match self.query_limits.effective(req.limits.as_ref()) {
            Ok(eff) => eff,
            Err(over) => {
                self.limit_counters.record_override_rejected();
                return self.query_error_with_details(
                    "invalid_params",
                    McpErrorCode::InvalidArgument,
                    false,
                    &format!(
                        "Per-call {} limit override ({}) exceeds the operator ceiling ({}).",
                        over.dimension.as_str(),
                        over.requested,
                        over.ceiling,
                    ),
                    Some(&language),
                    json!({
                        "dimension": over.dimension.as_str(),
                        "requested": over.requested,
                        "ceiling": over.ceiling,
                    }),
                );
            }
        };

        // The caller's own `limit` param keeps its existing contract (default
        // 100, absolute hard cap `MAX_RESULT_LIMIT`); the effective row cap
        // tightens it further when configured. With default limits the two are
        // equal, so behavior is unchanged.
        let requested = req
            .limit
            .unwrap_or(DEFAULT_RESULT_LIMIT)
            .min(MAX_RESULT_LIMIT);
        let row_limit = if effective.max_result_rows > 0 {
            requested.min(effective.max_result_rows)
        } else {
            requested
        };

        self.run_query_under_timeout(effective, language, req, row_limit, scope)
    }

    /// Run the `query` tool's execution under the effective wall-clock timeout
    /// (Issue #3368).
    ///
    /// When `timeout_ms` is `0` (unlimited), the query runs inline with no
    /// overhead. Otherwise it runs on a dedicated thread raced against the
    /// deadline: if the deadline elapses first the caller gets a prompt,
    /// structured `RESOURCE_EXHAUSTED` timeout error (retriable — the `query`
    /// tool is read-only, so a tightened retry is always sound). Mirroring the
    /// HTTP `/query` surface, the underlying computation is **not** cancelled
    /// (it runs to completion on its thread and is then discarded); this bounds
    /// the caller's wait without engine-level cooperative cancellation, which
    /// is the query-executor lane's concern. The read-only guard has already
    /// run, so a discarded computation can never leave a partial write.
    fn run_query_under_timeout(
        &self,
        effective: EffectiveQueryLimits,
        language: String,
        req: QueryRequest,
        row_limit: usize,
        scope: Option<crate::core::namespace::NamespaceScope>,
    ) -> CallToolResult {
        if effective.timeout_ms == 0 {
            // Inline (unlimited-timeout) path: no worker thread is spawned, so
            // the in-flight guard does not apply (Issue #3368).
            return self.execute_query_core(effective, &language, &req, row_limit, &scope);
        }

        // Bounded in-flight-query DoS guard (Issue #3368): the timeout race
        // spawns a detached, non-cancellable worker; without a cap a caller
        // could pile up thousands of still-running expensive queries. Acquire a
        // slot before spawning; at the cap, reject with a retriable UNAVAILABLE
        // rather than spawning yet another worker.
        let cap = self.query_limits.max_in_flight_queries;
        let guard = match self.try_acquire_in_flight(cap) {
            Some(guard) => guard,
            None => return self.in_flight_capacity_error(&language, cap),
        };

        let server = self.clone();
        let lang = language.clone();
        let outcome = self.race_deadline(effective.timeout_ms, guard, move || {
            server.execute_query_core(effective, &lang, &req, row_limit, &scope)
        });
        match outcome {
            RaceOutcome::Completed(result) => result,
            RaceOutcome::TimedOut => self.wall_clock_timeout_error(&language, effective.timeout_ms),
            RaceOutcome::WorkerDied => self.worker_died_error(&language),
        }
    }

    /// Try to reserve a slot in the bounded in-flight-query pool (Issue #3368).
    ///
    /// Returns `Some(guard)` on success (the guard releases the slot on drop),
    /// or `None` when the pool is already at `cap` (the caller then rejects the
    /// query). A `cap` of `0` means unbounded — a slot is always granted (still
    /// accounted, so [`AletheiaMcpServer`] can observe the live count). Uses a
    /// CAS loop so a spurious over-count can never leak a slot.
    fn try_acquire_in_flight(&self, cap: usize) -> Option<InFlightGuard> {
        let mut current = self.in_flight_queries.load(Ordering::Acquire);
        loop {
            if cap != 0 && current >= cap {
                return None;
            }
            match self.in_flight_queries.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(InFlightGuard {
                        counter: Arc::clone(&self.in_flight_queries),
                    });
                }
                Err(actual) => current = actual,
            }
        }
    }

    /// Race a computation `f` against a wall-clock deadline (Issue #3368).
    ///
    /// Returns [`RaceOutcome::Completed`] if `f` finished within `timeout_ms`,
    /// [`RaceOutcome::TimedOut`] if the deadline elapsed first, or
    /// [`RaceOutcome::WorkerDied`] if the worker's sender was dropped without a
    /// result (a panic in `f` or anything it calls) — the latter is reported as
    /// a non-retriable INTERNAL error rather than a retriable timeout, so a
    /// deterministic panic can never masquerade as an infinitely-retriable
    /// timeout.
    ///
    /// `f` runs on a dedicated thread and is **not** cancelled on timeout — it
    /// runs to completion and its result is discarded — so the caller's wait is
    /// bounded without engine-level cancellation (mirroring the HTTP `/query`
    /// surface). `f` must therefore be free of externally-visible side effects
    /// that would be unsafe to discard; the `query` tool's read-only guard runs
    /// before this, so a discarded query can never leave a partial write.
    /// `guard` is moved into the worker so its in-flight slot is released when
    /// the WORKER finishes (not when the caller times out).
    fn race_deadline<F>(&self, timeout_ms: u64, guard: InFlightGuard, f: F) -> RaceOutcome
    where
        F: FnOnce() -> CallToolResult + Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Hold the in-flight slot for the worker's whole lifetime; it is
            // released when `_guard` drops at the end of this closure — even if
            // the caller already timed out and discarded the result.
            let _guard = guard;
            // A send error only means the receiver already timed out and went
            // away; the computed result is simply discarded.
            let _ = tx.send(f());
        });
        match rx.recv_timeout(std::time::Duration::from_millis(timeout_ms)) {
            Ok(result) => RaceOutcome::Completed(result),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => RaceOutcome::TimedOut,
            // The sender dropped without sending — the worker panicked (or was
            // otherwise torn down). Do NOT treat this as a timeout.
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => RaceOutcome::WorkerDied,
        }
    }

    /// Build the structured worker-died error (Issue #3368): the timeout-race
    /// worker's sender dropped without a result, i.e. the computation panicked.
    /// Reported as a non-retriable `INTERNAL` (kind `runtime_error`) — a
    /// deterministic panic must not be retried forever as a "timeout". Carries
    /// no engine-internal detail (no secret/panic payload) beyond the fact of
    /// the failure. Not counted as a timeout termination.
    fn worker_died_error(&self, language: &str) -> CallToolResult {
        self.query_error_classified(
            "runtime_error",
            McpErrorCode::Internal,
            false,
            "The query worker terminated unexpectedly before producing a result (internal \
             error). This is not a timeout; retrying the identical query is unlikely to help.",
            None,
            Some(language),
        )
    }

    /// Build the structured in-flight-capacity error (Issue #3368): the bounded
    /// worker pool is full, so the query was rejected instead of spawning yet
    /// another non-cancellable worker. Retriable `UNAVAILABLE` — the caller
    /// should back off and retry once load subsides.
    fn in_flight_capacity_error(&self, language: &str, cap: usize) -> CallToolResult {
        self.query_error_with_details(
            "runtime_error",
            McpErrorCode::Unavailable,
            true,
            &format!(
                "The server is at its cap of {cap} concurrently executing timed queries and \
                 cannot admit another right now. Retry with backoff, or run without a wall-clock \
                 timeout only in a trusted/embedded deployment.",
            ),
            Some(language),
            json!({
                "reason": "in_flight_query_cap",
                "max_in_flight_queries": cap,
            }),
        )
    }

    /// Build the structured wall-clock-timeout error and record the termination
    /// (Issue #3368). `retriable: true` — the `query` tool is read-only, so a
    /// tightened retry is always sound.
    fn wall_clock_timeout_error(&self, language: &str, timeout_ms: u64) -> CallToolResult {
        self.limit_counters
            .record_termination(LimitDimension::WallClockTimeout);
        self.query_error_with_details(
            "runtime_error",
            McpErrorCode::ResourceExhausted,
            true,
            &format!(
                "Query exceeded the wall-clock timeout of {timeout_ms} ms and was terminated. \
                 Retry with a narrower query (smaller depth/LIMIT/time window) or a larger \
                 `limits.timeout_ms` (up to the operator ceiling).",
            ),
            Some(language),
            json!({
                "dimension": LimitDimension::WallClockTimeout.as_str(),
                "limit": timeout_ms,
            }),
        )
    }

    /// Derive the engine-lane [`QueryResourceLimits`] (Issue #3368 engine
    /// core) that `execute_query_core` threads into the reconciled `_limited`
    /// entry points, from the MCP surface's already-resolved `effective`
    /// limits.
    ///
    /// **Deadline (cooperative timeout).** The caller-facing timeout is, and
    /// remains, [`race_deadline`](Self::race_deadline)'s deterministic
    /// `timeout_ms` race — every existing timeout test keeps observing exactly
    /// that boundary. This method computes a SEPARATE, LATER deadline
    /// (`engine_deadline_ms = timeout_ms + max(timeout_ms / 2, 100)`) handed to
    /// the executor's [`ResourceGuardIterator`](crate::query::executor::ResourceGuardIterator)
    /// so the DETACHED WORKER itself cooperatively self-cancels within a
    /// bounded grace period after the caller has already timed out, instead of
    /// running a (possibly pathological) query to completion on an abandoned
    /// thread. This releases the worker's CPU and its in-flight slot promptly
    /// (AC3 "releases its resources", AC8 neighbor protection) without
    /// changing what the caller observes. `timeout_ms == 0` (unlimited) yields
    /// `deadline: None`.
    ///
    /// **Memory.** `max_memory_bytes` is folded from the MCP-surface memory
    /// budget (`effective.max_query_memory_bytes`, default-off / `0` =
    /// unlimited under the default config — zero behavior change unless an
    /// operator opts in), enforced by the SAME `ResourceGuardIterator` via its
    /// real per-row [`estimate_row_bytes`](crate::query::limits::estimate_row_bytes)
    /// accounting during the drain — a cooperative, mid-scan enforcement
    /// distinct from (and additive to) the post-hoc serialized-response-size
    /// proxy [`enforce_memory_budget`](Self::enforce_memory_budget) applies to
    /// the wrapped read tools.
    ///
    /// **Rows.** Always `None` here: the `query` tool's existing
    /// `take_n(row_limit + 1)` / `truncated: true` contract already owns row
    /// truncation as a disclosed, successful response — turning that into an
    /// engine-level error would be a behavior change, not an addition.
    fn engine_query_limits(effective: EffectiveQueryLimits) -> QueryResourceLimits {
        let deadline = if effective.timeout_ms > 0 {
            let engine_deadline_ms = effective
                .timeout_ms
                .saturating_add(effective.timeout_ms.saturating_div(2).max(100));
            Some(Instant::now() + Duration::from_millis(engine_deadline_ms))
        } else {
            None
        };
        let max_memory_bytes = if effective.max_query_memory_bytes > 0 {
            Some(effective.max_query_memory_bytes)
        } else {
            None
        };
        QueryResourceLimits {
            deadline,
            max_rows: None,
            max_memory_bytes,
        }
    }

    /// Execute the query, collect up to `row_limit` rows (disclosing truncation
    /// via `truncated`), serialize, and enforce the effective result-byte cap
    /// (Issue #3368).
    ///
    /// The byte cap is a **protective**, fail-closed bound (over cap →
    /// `RESOURCE_EXHAUSTED`, non-retriable), enforced here BEFORE the response
    /// returns to [`dispatch_tool`](Self::dispatch_tool)'s #3353 token-budget
    /// shaper — so the two compose without double-truncation: a byte-cap breach
    /// short-circuits to an error (which the budget shaper passes through
    /// untouched), while a within-cap response is then optionally shaped by the
    /// caller's own token budget with the #3353 disclosed ladder.
    ///
    /// The declarative execution itself now runs under the engine-lane
    /// cooperative [`QueryResourceLimits`] derived by
    /// [`engine_query_limits`](Self::engine_query_limits) (Issue #3368): a
    /// breach surfaces as `Error::Query(QueryError::ResourceExhausted { .. })`,
    /// mapped by [`map_query_error`](Self::map_query_error) into the same
    /// structured `details.{dimension, limit, consumed}` envelope the
    /// MCP-surface timeout/byte-cap builders already use.
    fn execute_query_core(
        &self,
        effective: EffectiveQueryLimits,
        language: &str,
        req: &QueryRequest,
        row_limit: usize,
        scope: &Option<crate::core::namespace::NamespaceScope>,
    ) -> CallToolResult {
        let has_params = req.params.as_ref().is_some_and(|p| !p.is_empty());
        let engine_limits = Self::engine_query_limits(effective);

        // Issue #3349: the query executes under the *reconciled* namespace read
        // scope. `scope` is the programmatic parameter `P` (`None` when omitted);
        // the executor's `reconcile_namespace_scope` folds it with any in-query
        // `USE / IN NAMESPACE` clause `C` (omitted+none ⇒ `default`-only,
        // omitted+clause ⇒ the clause, explicit ⇒ a ceiling that must match the
        // clause), then rides the `Query` IR into the same executor source-leaf
        // filter + traversal boundary the Rust builder uses.
        let execution = match language {
            "aql" => {
                if has_params {
                    return self.query_error(
                        "invalid_request",
                        "AQL does not support parameter bindings; inline literal values or use \
                         language \"cypher\" with $params.",
                        None,
                        Some("aql"),
                    );
                }
                self.db
                    .execute_aql_reconciled_limited(&req.query, scope.clone(), engine_limits)
            }
            "cypher" => {
                #[cfg(feature = "cypher")]
                {
                    match self.json_to_cypher_params(req.params.as_ref()) {
                        Ok(params) if params.is_empty() => {
                            self.db.execute_cypher_reconciled_limited(
                                &req.query,
                                scope.clone(),
                                engine_limits,
                            )
                        }
                        Ok(params) => self.db.execute_cypher_with_params_reconciled_limited(
                            &req.query,
                            params,
                            scope.clone(),
                            engine_limits,
                        ),
                        Err((parameter, reason)) => {
                            return self.query_error(
                                "invalid_params",
                                &format!("Invalid parameter '{parameter}': {reason}"),
                                None,
                                Some("cypher"),
                            );
                        }
                    }
                }
                #[cfg(not(feature = "cypher"))]
                {
                    return self.query_error(
                        "language_unavailable",
                        "Cypher support is not compiled in (enable the `cypher` feature). Use \
                         language \"aql\" instead.",
                        None,
                        Some("cypher"),
                    );
                }
            }
            _ => unreachable!("language already validated above"),
        };

        let results = match execution {
            Ok(results) => results,
            Err(e) => return self.map_query_error(e, language),
        };

        // Collect one extra row to detect (and report) truncation at the cap.
        let collected = match results.take_n(row_limit.saturating_add(1)) {
            Ok(rows) => rows,
            Err(e) => return self.map_query_error(e, language),
        };
        let truncated = collected.len() > row_limit;
        // Detect computed/aggregate rows (#558): a row carrying named `columns`
        // (e.g. `RETURN count(*)`, `RETURN n.dept, count(*)`) reports its own
        // column schema in projection order. Ordinary entity rows leave this
        // `None`, so the static entity/score/path/timestamp schema is retained
        // byte-for-byte.
        // A multi-variable binding row (#549) advertises its columns
        // dynamically too: the bound variable names (in binding order) followed
        // by any scalar projection columns -- mirroring how aggregate rows
        // derive their dynamic schema -- so a caller can map row keys to
        // columns instead of seeing the static entity/score/path schema.
        // Build rows one at a time, measuring serialized size incrementally so
        // peak working memory stays ≈ the byte cap (Issue #3368). Each row's
        // compact serialized length is a strict lower bound on that row's
        // contribution to the final pretty-printed response (pretty-printing
        // only adds whitespace/indentation, and the response envelope adds
        // more), so once the running row bytes alone exceed the cap the full
        // response is guaranteed over cap: we stop building further rows and
        // fail closed immediately instead of materializing the whole (up to
        // 100k-row) result and then rejecting it.
        let cap = effective.max_response_bytes;
        let mut computed_columns: Option<Vec<String>> = None;
        let mut rows: Vec<serde_json::Value> = Vec::new();
        let mut running_bytes: usize = 0;
        for row in collected.into_iter().take(row_limit) {
            if computed_columns.is_none() {
                if let Some(bindings) = &row.bindings {
                    let mut names: Vec<String> =
                        bindings.iter().map(|(name, _)| name.clone()).collect();
                    if let Some(cols) = &row.columns {
                        names.extend(cols.iter().map(|(name, _)| name.clone()));
                    }
                    computed_columns = Some(names);
                } else if let Some(cols) = &row.columns {
                    computed_columns = Some(cols.iter().map(|(name, _)| name.clone()).collect());
                }
            }
            let row_json = self.query_row_to_json(row);
            if cap > 0 {
                // Compact length is the cheapest strict lower bound on the row's
                // pretty-printed footprint.
                running_bytes = running_bytes
                    .saturating_add(serde_json::to_string(&row_json).map_or(0, |s| s.len()));
                if running_bytes > cap {
                    // Provably over cap already — reject without building the
                    // rest of the result (peak allocation ≈ cap).
                    return self.result_bytes_error(language, running_bytes, cap);
                }
            }
            rows.push(row_json);
        }
        let row_count = rows.len();

        let columns = match &computed_columns {
            Some(names) => computed_query_columns(names),
            None => query_columns(),
        };

        let value = json!({
            "language": language,
            "columns": columns,
            "rows": rows,
            "row_count": row_count,
            "truncated": truncated,
        });

        // Protective result-byte cap (Issue #3368): the incremental guard above
        // bounds peak allocation; here we take the exact serialized bytes the
        // response will carry (the pretty-printed form `success_json` emits) and
        // fail closed if the envelope + whitespace pushes a within-row-budget
        // response over the cap. Non-retriable — the same query would produce
        // the same oversized result; the caller must narrow it or raise
        // `limits.max_response_bytes`.
        let serialized = serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
        if cap > 0 && serialized.len() > cap {
            return self.result_bytes_error(language, serialized.len(), cap);
        }

        CallToolResult::success(vec![Content::text(serialized)])
    }

    /// Build the structured result-byte-cap error and record the termination
    /// (Issue #3368). `consumed` is the measured (or provably-lower-bounded)
    /// serialized size, always `> cap`. Non-retriable `RESOURCE_EXHAUSTED`.
    fn result_bytes_error(&self, language: &str, consumed: usize, cap: usize) -> CallToolResult {
        self.limit_counters
            .record_termination(LimitDimension::ResultBytes);
        self.query_error_with_details(
            "runtime_error",
            McpErrorCode::ResourceExhausted,
            false,
            &format!(
                "Query response ({consumed} bytes) exceeds the result-byte cap of {cap} bytes. \
                 Narrow the query (smaller LIMIT / fewer returned properties) or raise \
                 `limits.max_response_bytes` (up to the operator ceiling).",
            ),
            Some(language),
            json!({
                "dimension": LimitDimension::ResultBytes.as_str(),
                "limit": cap,
                "consumed": consumed,
            }),
        )
    }

    /// Test-only accessor returning the names of all advertised tools.
    #[cfg(test)]
    pub(crate) fn list_tools_for_test(&self) -> Vec<String> {
        tool_definitions()
            .iter()
            .map(|tool| tool.name.to_string())
            .collect()
    }

    /// Test-only accessor returning the advertised input schema for a tool,
    /// so tests can pin schema contents (e.g. the apply_batch op variants)
    /// against silent drift.
    #[cfg(test)]
    pub(crate) fn tool_input_schema_for_test(&self, name: &str) -> Option<serde_json::Value> {
        tool_definitions()
            .iter()
            .find(|tool| tool.name == name)
            .map(|tool| serde_json::Value::Object((*tool.input_schema).clone()))
    }

    /// Dispatch a tool call by name to its handler.
    ///
    /// Shared between [`ServerHandler::call_tool`] and tests so that every
    /// advertised tool (including ones added later) can be driven through the
    /// exact same dispatch table the MCP transport uses — e.g. the
    /// registry-driven error-shape test iterates [`tool_definitions`] and
    /// calls this for each tool, guaranteeing new tools are automatically
    /// covered by the structured-error contract (Issue #3234).
    ///
    /// # Authentication & authorization (Issue #3350)
    ///
    /// This is the single enforcement point for the MCP surface: the session
    /// credential is (re-)verified and the tool's access class checked
    /// against the principal's role **before** any handler runs — including
    /// before tool-name resolution, so an unknown tool name cannot bypass
    /// authentication or probe the tool inventory. The per-tool public Rust
    /// methods (e.g. [`get_node`](Self::get_node)) are the embedded API and
    /// are not gated — a Rust caller already holds the `Arc<AletheiaDB>`.
    pub(crate) fn dispatch_tool(&self, name: &str, args: serde_json::Value) -> CallToolResult {
        if let Err(err) = self.auth.authorize_tool(name) {
            return self.error_result(err);
        }
        // Read-only replica enforcement (Issue #3355, Slice A). Runs after
        // authorization (so an unauthenticated/unauthorized caller still
        // gets the uniform auth error first) but before any handler: a
        // write/admin-class tool called against a replica-role node is
        // refused uniformly here, regardless of which handler would have
        // run. Read and Metrics-class tools (including an unclassified/
        // unknown tool name) are unaffected.
        if self.db.is_replica()
            && matches!(
                tool_access_class(name),
                Some(AccessClass::Write) | Some(AccessClass::Admin)
            )
        {
            return self.error_result(read_only_replica_error());
        }
        // Token-budget-aware response shaping (Issue #3353). For the read tools
        // listed in `BUDGETABLE_READ_TOOLS`, an optional `max_response_tokens` /
        // `max_response_bytes` shapes the successful response to fit the stated
        // budget with a disclosed truncation contract. Omitting the budget
        // parameters leaves behavior completely unchanged.
        if is_budgetable_read_tool(name) {
            match budget::parse_budget(&args, self.max_priority_properties) {
                Ok(Some(budget_req)) => {
                    // Retain the original arguments so the rung-4 truncation
                    // handle can emit a concrete offset-based resume call
                    // (Issue #3353 F1/F5).
                    let orig_args = args.clone();
                    // Resource limits (#3368) run first — the byte cap short-
                    // circuits an oversized response to an error before the
                    // #3353 budget shaper sees it — so a within-cap response is
                    // what gets shaped to the caller's token budget.
                    let result = self.dispatch_read_tool_limited(name, args);
                    return self.apply_budget(name, result, &budget_req, &orig_args);
                }
                Ok(None) => {}
                Err(err) => return self.error_result(err),
            }
        }
        self.dispatch_read_tool_limited(name, args)
    }

    /// Apply the parsed token budget to a handler's result. Errors pass through
    /// untouched; a successful object response is shaped to fit and
    /// re-serialized, or replaced with the structured `INVALID_ARGUMENT`
    /// too-small-budget error (Issue #3353 AC6). Non-object / non-JSON success
    /// payloads cannot degrade along the entity ladder but are still held to the
    /// byte cap with a disclosed truncation marker — the "guaranteed to fit"
    /// contract is unconditional (Issue #3353 F6).
    fn apply_budget(
        &self,
        name: &str,
        result: CallToolResult,
        budget_req: &budget::BudgetRequest,
        args: &serde_json::Value,
    ) -> CallToolResult {
        if result.is_error.unwrap_or(false) {
            return result;
        }
        let text = Self::extract_text(result);
        let value: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            // Non-JSON payload: still enforce the byte cap on the raw text.
            Err(_) => {
                return match budget::enforce_raw_cap(text, budget_req) {
                    Ok(capped) => self.success_json(serde_json::Value::String(capped)),
                    Err(err) => self.error_result(err),
                };
            }
        };
        if !value.is_object() {
            // JSON scalar/array: not shapeable along the entity ladder, but the
            // serialized form is still capped so an unbounded array cannot
            // bypass the budget.
            let serialized =
                serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
            if serialized.len() as u64 <= budget_req.effective_bytes() {
                return self.success_json(value);
            }
            return match budget::enforce_raw_cap(serialized, budget_req) {
                Ok(capped) => self.success_json(serde_json::Value::String(capped)),
                Err(err) => self.error_result(err),
            };
        }
        match budget::shape_response(value, budget_req, name, args) {
            Ok(shaped) => self.success_json(shaped),
            Err(err) => self.error_result(err),
        }
    }

    /// The tool-name dispatch table shared between the budgeted and unbudgeted
    /// paths.
    fn dispatch_read_tool(&self, name: &str, args: serde_json::Value) -> CallToolResult {
        match name {
            "get_node" => self.handle_get_node(args),
            "create_node" => self.handle_create_node(args),
            "update_node" => self.handle_update_node(args),
            "delete_node" => self.handle_delete_node(args),
            "retract_node" => self.handle_retract_node(args),
            "retract_edge" => self.handle_retract_edge(args),
            "delete_node_cascade" => self.handle_delete_node_cascade(args),
            "list_nodes" => self.handle_list_nodes(args),
            "count_nodes" => self.handle_count_nodes(args),
            "get_edge" => self.handle_get_edge(args),
            "create_edge" => self.handle_create_edge(args),
            "update_edge" => self.handle_update_edge(args),
            "delete_edge" => self.handle_delete_edge(args),
            "apply_batch" => self.handle_apply_batch(args),
            "list_edges" => self.handle_list_edges(args),
            "count_edges" => self.handle_count_edges(args),
            "get_outgoing_edges" => self.handle_get_outgoing_edges(args),
            "get_incoming_edges" => self.handle_get_incoming_edges(args),
            "traverse" => self.handle_traverse(args),
            "find_similar" => self.handle_find_similar(args),
            // Embedding generation & text semantic search (Issue #2906).
            "embed_query" => self.handle_embed_query(args),
            "embed_text" => self.handle_embed_text(args),
            "semantic_search" => self.handle_semantic_search(args),
            "create_node_with_embedding" => self.handle_create_node_with_embedding(args),
            "update_node_embedding" => self.handle_update_node_embedding(args),
            // Semantic-search analysis tools (Issue #2907). Advertised and
            // dispatched unconditionally (Design A); the handler bodies gate on
            // the `semantic-search` feature.
            "semantic_path" => self.handle_semantic_path(args),
            "concept_analogy" => self.handle_concept_analogy(args),
            "concept_mean" => self.handle_concept_mean(args),
            "find_duplicate_candidates" => self.handle_find_duplicate_candidates(args),
            "semantic_horizon" => self.handle_semantic_horizon(args),
            "context_aspects" => self.handle_context_aspects(args),
            "enable_vector_index" => self.handle_enable_vector_index(args),
            "list_vector_indexes" => self.handle_list_vector_indexes(args),
            "enable_unique_constraint" => self.handle_enable_unique_constraint(args),
            "list_unique_constraints" => self.handle_list_unique_constraints(args),
            "get_node_at_time" => self.handle_get_node_at_time(args),
            "get_edge_at_time" => self.handle_get_edge_at_time(args),
            "find_nodes_at_time" => self.handle_find_nodes_at_time(args),
            "list_changes" => self.handle_list_changes(args),
            "await_changes" => self.handle_await_changes(args, None),
            "get_node_at_valid_time" => self.handle_get_node_at_valid_time(args),
            "get_node_at_transaction_time" => self.handle_get_node_at_transaction_time(args),
            "get_node_history" => self.handle_get_node_history(args),
            "diff_node_versions" => self.handle_diff_node_versions(args),
            "get_edge_at_valid_time" => self.handle_get_edge_at_valid_time(args),
            "get_edge_at_transaction_time" => self.handle_get_edge_at_transaction_time(args),
            "get_edge_history" => self.handle_get_edge_history(args),
            "diff_edge_versions" => self.handle_diff_edge_versions(args),
            // Belief-revision audit (Issue #3362). Advertised and dispatched
            // unconditionally (Design A); the handler body gates on the
            // `semantic-temporal` feature.
            "get_belief_revisions" => self.handle_get_belief_revisions(args),
            // Temporal drift-alarm management (Issue #3367). Advertised and
            // dispatched unconditionally (Design A); handler bodies gate on the
            // `semantic-temporal` feature.
            "create_drift_monitor" => self.handle_create_drift_monitor(args),
            "list_drift_monitors" => self.handle_list_drift_monitors(args),
            "delete_drift_monitor" => self.handle_delete_drift_monitor(args),
            "query_drift_alarms" => self.handle_query_drift_alarms(args),
            "resolve_drift_alarm" => self.handle_resolve_drift_alarm(args),
            // Contradiction genealogy (Issue #3352).
            "contradiction_genealogy" => self.handle_contradiction_genealogy(args),
            "find_contradictions" => self.handle_find_contradictions(args),
            // Counterfactual replay (Issue #3357).
            "counterfactual_replay" => self.handle_counterfactual_replay(args),
            // Trust propagation (Issue #3382). Handler bodies gate on the
            // `semantic-reasoning` feature.
            "trust_breakdown" => self.handle_trust_breakdown(args),
            "list_trust_policies" => self.handle_list_trust_policies(args),
            "hybrid_query" => self.handle_hybrid_query(args),
            "query" => self.handle_query(args),
            "get_schema" => self.handle_get_schema(args),
            "temporal_extent" => self.handle_temporal_extent(args),
            "lineage_upstream" => self.handle_lineage_upstream(args),
            "lineage_downstream" => self.handle_lineage_downstream(args),
            "audit_export" => self.handle_audit_export(args),
            "database_stats" => self.handle_database_stats(args, self.caller_is_admin()),
            "verify_chain" => self.handle_verify_chain(args),
            "export_chain_head" => self.handle_export_chain_head(args),
            // Namespace management (Issue #3349, PR3b).
            "create_namespace" => self.handle_create_namespace(args),
            "list_namespaces" => self.handle_list_namespaces(args),
            "describe_namespace" => self.handle_describe_namespace(args),
            // GDPR crypto-shred — ADMIN-class (Issue #3359, Slice 4b).
            "designate_subject" => self.handle_designate_subject(args),
            "erase_subject" => self.handle_erase_subject(args),
            _ => self.error_result(
                McpError::new(McpErrorCode::NotFound, format!("Unknown tool: {}", name))
                    .details(json!({ "tool": name })),
            ),
        }
    }

    /// Dispatch a read tool under the per-query resource limits (Issue #3368
    /// residue): the wall-clock timeout and the result-byte cap, reusing the
    /// exact machinery the `query` tool self-enforces (the timeout thread-race,
    /// the bounded in-flight-worker DoS guard, and the structured
    /// `RESOURCE_EXHAUSTED` error builders + termination counters).
    ///
    /// Tools not in [`RESOURCE_LIMITED_READ_TOOLS`] — including `query`, which
    /// self-enforces inline — pass straight through to
    /// [`dispatch_read_tool`](Self::dispatch_read_tool) unchanged. For the
    /// covered tools this resolves the server-default effective limits (no
    /// per-call override in v1):
    ///
    /// - **Timeout.** The zero-overhead inline path (no thread spawn, no clone,
    ///   no in-flight guard) applies **only when the effective timeout is `0`**
    ///   — i.e. the `disabled()` config. Under the *default* config the
    ///   effective timeout is 30_000 ms (not 0), so each covered call takes the
    ///   timeout-race worker path (reserve an in-flight slot — rejected
    ///   `UNAVAILABLE` at the cap — then race the handler against the deadline
    ///   on a detached worker, exactly like the `query` tool): a `TimedOut`
    ///   outcome maps to the retriable `RESOURCE_EXHAUSTED` timeout error, a
    ///   `WorkerDied` (panic) outcome to the non-retriable INTERNAL error. The
    ///   response bytes are unchanged versus a bare `dispatch_read_tool`, but
    ///   the per-call worker spawn is real overhead on hot-path reads
    ///   (including cheap `get_node_at_time` / `get_edge_at_time`); a
    ///   quantifying micro-benchmark is a deferred (Lane-2) follow-up.
    /// - **Byte cap.** A *non-error* response is then held to the effective
    ///   response-byte cap post-hoc (see
    ///   [`enforce_result_byte_cap`](Self::enforce_result_byte_cap)); errors
    ///   pass through untouched.
    ///
    /// Ordering (Issue #3360 / #3353): the cursor page is resolved inside the
    /// handler, the resource byte cap is applied here, and the #3353 token
    /// budget shaper (in [`dispatch_tool`](Self::dispatch_tool)) then runs
    /// *after* on the within-cap response — never before the cap.
    fn dispatch_read_tool_limited(&self, name: &str, args: serde_json::Value) -> CallToolResult {
        if !is_resource_limited_read_tool(name) {
            return self.dispatch_read_tool(name, args);
        }

        // No per-call override for these tools in v1, so `effective(None)` is
        // infallible (an override is the only thing that can be rejected). Fall
        // back defensively to "unlimited" rather than `unwrap`, keeping the
        // production path panic-free per the coding standards.
        let eff = self
            .query_limits
            .effective(None)
            .unwrap_or_else(|_| EffectiveQueryLimits::unlimited());
        let cap = eff.max_response_bytes;
        let mem_cap = eff.max_query_memory_bytes;

        let result = if eff.timeout_ms == 0 {
            // Inline (unlimited-timeout) fast path: no worker thread, no guard,
            // no clone — zero overhead over a bare `dispatch_read_tool` call.
            self.dispatch_read_tool(name, args)
        } else {
            let in_flight_cap = self.query_limits.max_in_flight_queries;
            let guard = match self.try_acquire_in_flight(in_flight_cap) {
                Some(guard) => guard,
                None => return self.in_flight_capacity_error(name, in_flight_cap),
            };
            let server = self.clone();
            let tool = name.to_string();
            match self.race_deadline(eff.timeout_ms, guard, move || {
                server.dispatch_read_tool(&tool, args)
            }) {
                RaceOutcome::Completed(result) => result,
                RaceOutcome::TimedOut => {
                    return self.read_tool_timeout_error(name, eff.timeout_ms);
                }
                RaceOutcome::WorkerDied => return self.worker_died_error(name),
            }
        };

        let result = self.enforce_result_byte_cap(name, result, cap);
        self.enforce_memory_budget(name, result, mem_cap)
    }

    /// Enforce the result-byte cap on a resource-limited read tool's response
    /// (Issue #3368 residue), post-hoc.
    ///
    /// A `cap` of `0` (unlimited) and any error response pass straight through.
    /// Otherwise, if the serialized response text exceeds `cap`, the response is
    /// replaced with the structured non-retriable `RESOURCE_EXHAUSTED`
    /// result-byte error (which also records the termination counter); a
    /// within-cap response is returned unchanged. The single text content item
    /// each read handler produces *is* the serialized JSON response, so its
    /// byte length is the response size — measured in place without consuming
    /// the result, so a within-cap response is returned byte-for-byte identical.
    fn enforce_result_byte_cap(
        &self,
        name: &str,
        result: CallToolResult,
        cap: usize,
    ) -> CallToolResult {
        if cap == 0 || result.is_error.unwrap_or(false) {
            return result;
        }
        let len = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.len()))
            .unwrap_or(0);
        if len > cap {
            return self.read_tool_byte_cap_error(name, len, cap);
        }
        result
    }

    /// Enforce the estimated-memory budget on a resource-limited read tool's
    /// response (Issue #3368 memory-budget dimension), post-hoc and default-off.
    ///
    /// A `mem_cap` of `0` (unlimited — the default config) and any error
    /// response pass straight through, so this is a no-op unless an operator
    /// opts in. Otherwise the working-memory proxy is estimated as the
    /// serialized response length scaled by [`MEMORY_WORKING_SET_EXPANSION`]:
    /// the in-RAM materialized representation that produced the response (parsed
    /// entities, property maps, vectors, hashmap/enum/capacity overhead) is a
    /// documented multiple of the compact serialized JSON. If that estimate
    /// exceeds `mem_cap`, the response is replaced with the non-retriable
    /// `RESOURCE_EXHAUSTED` / `memory_bytes` error (recording the termination
    /// counter). This is an honest *proxy*, not true per-task allocation
    /// accounting — the same class of post-hoc honesty as the byte cap; see the
    /// design note in `docs/guides/mcp-query-tool.md`.
    fn enforce_memory_budget(
        &self,
        name: &str,
        result: CallToolResult,
        mem_cap: usize,
    ) -> CallToolResult {
        if mem_cap == 0 || result.is_error.unwrap_or(false) {
            return result;
        }
        let len = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.len()))
            .unwrap_or(0);
        let estimated = len.saturating_mul(MEMORY_WORKING_SET_EXPANSION);
        if estimated > mem_cap {
            return self.read_tool_memory_error(name, estimated, mem_cap);
        }
        result
    }

    /// Tool-agnostic wall-clock-timeout error for the six wrapped read tools
    /// (Issue #3368 residue).
    ///
    /// Unlike the `query` tool's
    /// [`wall_clock_timeout_error`](Self::wall_clock_timeout_error) — whose
    /// `kind: "runtime_error"` and `language` fields are meaningful for a query
    /// *language* — these tools are not a query language and expose no per-call
    /// `limits` override in v1. Reusing the query builder would emit a
    /// semantically wrong `language: "<toolname>"`, a spurious `kind`, and
    /// remediation telling the caller to raise `limits.timeout_ms`, which they
    /// cannot set. This emitter instead produces the neutral #3234 envelope
    /// (`{error:{code,message,retriable,details}}` — no `kind`, no `language`)
    /// with tool-neutral remediation. Records the `WallClockTimeout`
    /// termination exactly once; retriable, since these tools are read-only so
    /// a narrower retry is always sound.
    fn read_tool_timeout_error(&self, name: &str, timeout_ms: u64) -> CallToolResult {
        self.limit_counters
            .record_termination(LimitDimension::WallClockTimeout);
        self.error_result(
            McpError::new(
                McpErrorCode::ResourceExhausted,
                format!(
                    "Tool '{name}' exceeded the wall-clock timeout of {timeout_ms} ms; narrow \
                     the request (smaller depth/limit/time window) and retry."
                ),
            )
            .retriable(true)
            .details(json!({
                "dimension": LimitDimension::WallClockTimeout.as_str(),
                "limit": timeout_ms,
            })),
        )
    }

    /// Tool-agnostic result-byte-cap error for the six wrapped read tools
    /// (Issue #3368 residue). The neutral #3234 counterpart to the `query`
    /// tool's [`result_bytes_error`](Self::result_bytes_error): no `kind`, no
    /// `language`, and no unactionable `limits.max_response_bytes` advice (these
    /// tools have no per-call override). Records the `ResultBytes` termination
    /// exactly once; non-retriable — the same request yields the same oversized
    /// response, so the caller must narrow it.
    fn read_tool_byte_cap_error(&self, name: &str, consumed: usize, cap: usize) -> CallToolResult {
        self.limit_counters
            .record_termination(LimitDimension::ResultBytes);
        self.error_result(
            McpError::new(
                McpErrorCode::ResourceExhausted,
                format!(
                    "Tool '{name}' response of {consumed} bytes exceeded the {cap}-byte cap; \
                     narrow the request (smaller depth/limit/k) and retry."
                ),
            )
            .retriable(false)
            .details(json!({
                "dimension": LimitDimension::ResultBytes.as_str(),
                "limit": cap,
                "consumed": consumed,
            })),
        )
    }

    /// Tool-agnostic estimated-memory-budget error for the wrapped read tools
    /// (Issue #3368 memory-budget dimension). The neutral #3234 counterpart to
    /// [`read_tool_byte_cap_error`](Self::read_tool_byte_cap_error): no `kind`,
    /// no `language`, no per-call `limits` advice. `consumed` is the *estimated*
    /// working-memory proxy (a documented multiple of the serialized size), not
    /// a measured allocation. Records the `Memory` termination exactly once;
    /// non-retriable — the same request materializes the same estimate, so the
    /// caller must narrow it.
    fn read_tool_memory_error(&self, name: &str, consumed: usize, cap: usize) -> CallToolResult {
        self.limit_counters
            .record_termination(LimitDimension::Memory);
        self.error_result(
            McpError::new(
                McpErrorCode::ResourceExhausted,
                format!(
                    "Tool '{name}' estimated working memory of {consumed} bytes exceeded the \
                     {cap}-byte memory budget; narrow the request (smaller depth/limit/k) and \
                     retry."
                ),
            )
            .retriable(false)
            .details(json!({
                "dimension": LimitDimension::Memory.as_str(),
                "limit": cap,
                "consumed": consumed,
            })),
        )
    }
}

/// In-memory expansion factor for the Issue #3368 memory-budget proxy.
///
/// The working-set memory used to materialize a read response (parsed graph
/// entities, `PropertyMap` hashmaps, `Vec<f32>` embeddings, `String` keys, plus
/// capacity slack and enum/hashmap overhead) is empirically a small multiple of
/// the compact serialized JSON that leaves the seam. `4×` is the documented v1
/// estimate of that multiple; it is a fixed constant, not response-shape
/// adaptive (a Lane-2 follow-up). See the design note in
/// `docs/guides/mcp-query-tool.md`.
pub(crate) const MEMORY_WORKING_SET_EXPANSION: usize = 4;

/// The read tools that honor the Issue #3353 token budget (`max_response_tokens`
/// / `max_response_bytes`). Kept as a single source of truth so the dispatch
/// path, the schema-documentation path, and the CI conformance sweep cannot
/// drift apart.
pub(crate) const BUDGETABLE_READ_TOOLS: &[&str] = &[
    "get_node",
    "list_nodes",
    "get_edge",
    "list_edges",
    "get_outgoing_edges",
    "get_incoming_edges",
    "traverse",
    "find_similar",
    "semantic_search",
    "hybrid_query",
    "query",
    "find_nodes_at_time",
    "get_node_history",
    "get_schema",
    // Belief-revision audit (Issue #3362) — array-returning read, enrolled in
    // the token budget for parity with its `get_node_history` sibling (#3353).
    "get_belief_revisions",
    // Deferred MCP-registry batch reads (Issue #3367 / #3352 / #3382) — each has
    // a budgetable sibling and can return large arrays/trees, so budgetable.
    // `counterfactual_replay` is deliberately EXCLUDED (its AC8 `counterfactual:
    // true` marker must never be stripped by budget-ladder truncation), and
    // `list_trust_policies` is small/bounded (like `list_vector_indexes`).
    "list_drift_monitors",
    "query_drift_alarms",
    "contradiction_genealogy",
    "find_contradictions",
    "trust_breakdown",
    // Semantic-search analysis tools (Issue #2907) — all read-only and
    // potentially large, so budgetable.
    "semantic_path",
    "concept_analogy",
    "concept_mean",
    "find_duplicate_candidates",
    "semantic_horizon",
    "context_aspects",
];

/// Does this tool honor the token budget parameters (Issue #3353)?
pub(crate) fn is_budgetable_read_tool(name: &str) -> bool {
    BUDGETABLE_READ_TOOLS.contains(&name)
}

/// The read tools whose responses are governed by the per-query resource
/// limits (Issue #3368 residue): the wall-clock timeout and the result-byte
/// cap. The `query` tool self-enforces the same limits inline (see
/// [`AletheiaMcpServer::handle_query`]) and is deliberately *not* listed here,
/// so it is never double-wrapped. Kept as a single source of truth so the
/// dispatch wrapper and any future schema/documentation path cannot drift.
///
/// v1 scope (Lane-2 follow-ups deferred): these tools honor only the server
/// **defaults** — there is no per-call `limits` override for them (unlike the
/// `query` tool), no memory-budget dimension, and no engine-level cooperative
/// cancellation; the timed computation runs to completion on a detached worker
/// and its result is discarded on timeout (identical to the `query` tool's
/// non-cancellable race). The byte cap is enforced **post-hoc** on the fully
/// serialized response (the `query` tool's incremental row-by-row guard is not
/// reused here); a within-cap response is then optionally shaped by the #3353
/// token budget.
pub(crate) const RESOURCE_LIMITED_READ_TOOLS: &[&str] = &[
    "traverse",
    "hybrid_query",
    "find_similar",
    "get_node_at_time",
    "get_edge_at_time",
    "find_nodes_at_time",
    // Contradiction analysis (Issue #3352): `find_contradictions` runs an
    // O(entities * versions^2) scan and `contradiction_genealogy` an
    // O(versions^2) reconstruction — both potentially slow, so they enroll for
    // the uniform wall-clock-timeout + result-byte-cap coverage (Issue #3368).
    "contradiction_genealogy",
    "find_contradictions",
    // Semantic-search analysis tools (Issue #2907): read-only, potentially slow
    // graph+vector scans. They carry their own per-operation bounds, but enroll
    // here for the uniform wall-clock-timeout + result-byte-cap coverage every
    // other slow read gets (Issue #3368). At the default 30s timeout their
    // responses are unchanged.
    "semantic_path",
    "concept_analogy",
    "concept_mean",
    "find_duplicate_candidates",
    "semantic_horizon",
    "context_aspects",
];

/// Does this tool have its response governed by the per-query resource limits
/// (Issue #3368 residue)?
pub(crate) fn is_resource_limited_read_tool(name: &str) -> bool {
    RESOURCE_LIMITED_READ_TOOLS.contains(&name)
}

/// Read tools that honor the Issue #3348 provenance filter parameters
/// (`provenance_source`, `provenance_sources`, `min_confidence`,
/// `include_unattributed`). This is the single source of truth for schema
/// advertisement; each listed tool's handler parses the filter via
/// [`AletheiaMcpServer::parse_provenance_filter`].
pub(crate) const PROVENANCE_FILTERABLE_TOOLS: &[&str] = &[
    "get_node",
    "list_nodes",
    "get_outgoing_edges",
    "get_incoming_edges",
    "traverse",
    "find_similar",
    "hybrid_query",
];

/// Does this tool honor the provenance filter parameters (Issue #3348)?
pub(crate) fn is_provenance_filterable_tool(name: &str) -> bool {
    PROVENANCE_FILTERABLE_TOOLS.contains(&name)
}

fn make_input_schema<T: rmcp::schemars::JsonSchema>()
-> Arc<serde_json::Map<String, serde_json::Value>> {
    let schema = rmcp::schemars::schema_for!(T);
    let value = serde_json::to_value(schema).expect("JSON schema serialization should not fail");
    match value {
        serde_json::Value::Object(map) => Arc::new(map),
        _ => Arc::new(serde_json::Map::new()),
    }
}

/// Inject the snapshot-anchored cursor parameters (`use_cursor`, `cursor`) into
/// a generated tool `inputSchema` (Issue #3360).
///
/// The five cursorable read tools (`list_nodes`, `find_nodes_at_time`,
/// `get_outgoing_edges`, `get_incoming_edges`, `traverse`) read `use_cursor` /
/// `cursor` directly off the raw JSON arguments rather than through their typed
/// request structs, so those two parameters are absent from the derived schema
/// and would otherwise be undiscoverable to a client/LLM. This programmatically
/// adds them (both optional) to `inputSchema.properties` with clear
/// descriptions -- mirroring the sibling budget-param injection (#3353) rather
/// than threading the fields through every request-struct construction site.
fn inject_cursor_schema_params(
    schema: Arc<serde_json::Map<String, serde_json::Value>>,
) -> Arc<serde_json::Map<String, serde_json::Value>> {
    let mut map = (*schema).clone();
    let props = map
        .entry("properties")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if let Some(props) = props.as_object_mut() {
        props.insert(
            "use_cursor".to_string(),
            json!({
                "type": "boolean",
                "description": "Optional. Set true on the FIRST call to open a \
                    snapshot-anchored cursor scan; the response then carries an opaque \
                    `cursor` token to resume paging. Consistent, duplicate-free and \
                    gap-free under concurrent writes. Omit (or false) for the default \
                    offset pagination."
            }),
        );
        props.insert(
            "cursor".to_string(),
            json!({
                "type": "string",
                "description": "Optional. The opaque continuation token returned by a \
                    prior cursor page. Echo it back VERBATIM with no other arguments to \
                    fetch the next page. When the response omits `cursor` (has_more is \
                    false) the scan is complete."
            }),
        );
    }
    Arc::new(map)
}

/// [`make_input_schema`] for a cursorable read tool, with the `use_cursor` /
/// `cursor` parameters injected (see [`inject_cursor_schema_params`]).
fn make_cursor_input_schema<T: rmcp::schemars::JsonSchema>()
-> Arc<serde_json::Map<String, serde_json::Value>> {
    inject_cursor_schema_params(make_input_schema::<T>())
}

// The read-only statement guard (`detect_mutating_clause`) moved to
// `crate::query::read_only` so the HTTP surface can reuse it for RBAC
// classification (Issue #3350). Re-imported here to keep call sites stable.
use crate::query::read_only::detect_mutating_clause;

/// Column metadata describing the structured shape of each `query` result row.
///
/// `QueryRow` does not carry the `RETURN` aliases, so the contract is the fixed
/// row schema rather than per-query projection names. Constructed once and
/// cloned cheaply on each call via `OnceLock`.
fn query_columns() -> serde_json::Value {
    use std::sync::OnceLock;
    static COLUMNS: OnceLock<serde_json::Value> = OnceLock::new();
    COLUMNS
        .get_or_init(|| {
            json!([
                {
                    "name": "entity",
                    "type": "node|edge",
                    "description": "The node or edge bound by the RETURN clause (with label and properties)."
                },
                {
                    "name": "score",
                    "type": "number|null",
                    "description": "Vector similarity score, present for ranked queries."
                },
                {
                    "name": "path",
                    "type": "array<{type:string,id:number}>|null",
                    "description": "Typed entity references along the traversal path (each element has `type` and `id`), when applicable."
                },
                {
                    "name": "timestamp",
                    "type": "string|null",
                    "description": "Bi-temporal point this row represents, when applicable."
                }
            ])
        })
        .clone()
}

/// Build the column schema for a computed/aggregate `query` result (#558).
///
/// When a result carries named `QueryRow::columns` (e.g. `RETURN count(*)` or
/// `RETURN n.dept, count(*)`), the response advertises those column names in
/// projection order instead of the static entity/score/path/timestamp schema,
/// so a caller can map each row's values to their columns.
fn computed_query_columns(names: &[String]) -> serde_json::Value {
    serde_json::Value::Array(
        names
            .iter()
            .map(|name| {
                json!({
                    "name": name,
                    "type": "value",
                    "description": "Computed column from a RETURN projection or aggregation."
                })
            })
            .collect(),
    )
}

/// Build the list of tool definitions advertised by this MCP server.
///
/// Shared between [`ServerHandler::list_tools`] and test helpers so the advertised tool
/// set and the `call_tool` dispatch table cannot silently drift apart.
fn tool_definitions() -> Vec<Tool> {
    let mut tools = vec![
        Tool::new(
            "get_node",
            "Get a node by its ID. Returns the node's label and properties. \
                     Vector/embedding properties are elided by default (replaced with a \
                     `{type, dim, elided:true}` descriptor) to protect LLM context; pass \
                     `include_vectors: true` to receive the full float array.",
            make_input_schema::<GetNodeRequest>(),
        ),
        Tool::new(
            "create_node",
            "Create a new node with a label and optional properties. Optionally pass \
                     `valid_time` (ISO 8601 or microseconds since epoch) to record when this fact \
                     became true in the real world (backdating/future-dating); omit it to default \
                     to the transaction time. Transaction time is always system-assigned.",
            make_input_schema::<CreateNodeRequest>(),
        ),
        Tool::new(
            "update_node",
            "Update an existing node's properties. Optionally pass `valid_time` (ISO 8601 or \
                     microseconds since epoch) to record when this update became true in the real \
                     world; omit it to default to the transaction time. Transaction time is always \
                     system-assigned.",
            make_input_schema::<UpdateNodeRequest>(),
        ),
        Tool::new(
            "delete_node",
            "Delete a node by its ID (safe-by-default). If the node has connected \
                     edges and `detach` is not true, the deletion is refused and the response \
                     reports `connected_edges`. Pass `detach: true` to delete the node together \
                     with all connected edges; the response then reports `edges_removed`. \
                     Optionally pass `valid_time` to record when this fact stopped being true in \
                     the real world; omit it to default to the transaction time. Not supported \
                     together with `detach: true` (cascade delete does not support backdating).",
            make_input_schema::<DeleteNodeRequest>(),
        ),
        Tool::new(
            "retract_node",
            "Retract a node as of a valid time (bi-temporal, safe-by-default): close its \
                     valid-time interval at `valid_time` (ISO 8601 or microseconds since epoch; \
                     defaults to now) WITHOUT deleting its history. Unlike delete_node, queries \
                     AS OF a valid time strictly before `valid_time` still return the node, and \
                     AS OF SYSTEM_TIME queries positioned before the retraction still show it \
                     open-ended. If the node has connected edges and `detach` is not true, the \
                     retraction is refused and the response reports `connected_edges`; pass \
                     `detach: true` to co-retract every connected edge at the same valid time \
                     (`edges_retracted` reports how many). Re-retracting is an idempotent no-op \
                     returning the existing interval with `already_retracted: true`. The response \
                     carries the closed half-open interval as `valid_from`/`valid_to` RFC 3339 \
                     strings. Transaction time is always system-assigned.",
            make_input_schema::<RetractNodeRequest>(),
        ),
        Tool::new(
            "retract_edge",
            "Retract an edge as of a valid time (bi-temporal): close its valid-time interval \
                     at `valid_time` (ISO 8601 or microseconds since epoch; defaults to now) \
                     WITHOUT deleting its history. Queries AS OF a valid time strictly before \
                     `valid_time` still return the edge; AS OF SYSTEM_TIME queries positioned \
                     before the retraction still show it open-ended. Re-retracting is an \
                     idempotent no-op returning the existing interval with \
                     `already_retracted: true`. The response carries the closed half-open \
                     interval as `valid_from`/`valid_to` RFC 3339 strings. Transaction time is \
                     always system-assigned.",
            make_input_schema::<RetractEdgeRequest>(),
        ),
        Tool::new(
            "delete_node_cascade",
            "Delete a node and all its connected edges (cascade delete). \
                     This maintains referential integrity by removing orphaned edges.",
            make_input_schema::<DeleteNodeCascadeRequest>(),
        ),
        Tool::new(
            "list_nodes",
            "List nodes with optional label filter and pagination. The response carries \
                     `has_more` (true when more matching nodes exist beyond this page — always \
                     check it before trusting a count); when true, `next_offset` gives the \
                     `offset` to pass for the next page. `total_matching` is included when cheap \
                     (property-filtered queries) and omitted for plain label scans. \
                     Vector/embedding properties are elided by default (replaced with a \
                     `{type, dim, elided:true}` descriptor) to protect LLM context; pass \
                     `include_vectors: true` to receive the full float arrays. \
                     Pass `use_cursor: true` for snapshot-anchored cursor paging; echo the \
                     returned `cursor` back verbatim (no other args) for the next page; when \
                     the response omits `cursor` / `has_more` is false, the scan is complete.",
            make_cursor_input_schema::<ListNodesRequest>(),
        ),
        Tool::new(
            "count_nodes",
            "Count the total number of nodes.",
            make_input_schema::<CountNodesRequest>(),
        ),
        Tool::new(
            "get_edge",
            "Get an edge by its ID. Vector/embedding properties are elided by default \
                     (replaced with a `{type, dim, elided:true}` descriptor) to protect LLM \
                     context; pass `include_vectors: true` to receive the full float array.",
            make_input_schema::<GetEdgeRequest>(),
        ),
        Tool::new(
            "create_edge",
            "Create a new edge between two nodes. Optionally pass `valid_time` (ISO 8601 or \
                     microseconds since epoch) to record when this relationship became true in the \
                     real world (backdating/future-dating); omit it to default to the transaction \
                     time. Transaction time is always system-assigned.",
            make_input_schema::<CreateEdgeRequest>(),
        ),
        Tool::new(
            "update_edge",
            "Update an existing edge's properties. Optionally pass `valid_time` (ISO 8601 or \
                     microseconds since epoch) to record when this update became true in the real \
                     world; omit it to default to the transaction time. Transaction time is always \
                     system-assigned.",
            make_input_schema::<UpdateEdgeRequest>(),
        ),
        Tool::new(
            "delete_edge",
            "Delete an edge by its ID. Optionally pass `valid_time` to record when this \
                     relationship stopped being true in the real world; omit it to default to the \
                     transaction time.",
            make_input_schema::<DeleteEdgeRequest>(),
        ),
        Tool::new(
            "apply_batch",
            "Apply an ordered list of write operations ATOMICALLY (all-or-nothing) in one \
                     call. Each operation is a tagged object (`op`: create_node, create_edge, \
                     update_node, update_edge, delete_node, delete_edge) mirroring the single-op \
                     tools, including optional per-op `valid_time`. A `create_node` may carry a \
                     `ref` alias; later edge operations may reference batch-created nodes as \
                     '$alias' (or positionally as '$<index>') wherever a node id is accepted as \
                     an endpoint — forward references are rejected. If ANY operation fails \
                     (validation, unknown ref, constraint violation, detach refusal), NONE of \
                     the batch's writes take effect: atomicity holds for every acknowledged \
                     outcome and all non-crash failures (narrow caveat: a process crash during \
                     the commit flush can persist a prefix of a batch until WAL transaction \
                     framing lands, issue #3413), and the error reports \
                     `details.failed_op_index`. `delete_node` honors the safe-by-default DETACH \
                     contract against committed AND batch-created edges. On success the response \
                     returns per-operation results in input order (ids and version ids for \
                     creates/updates) plus `ref_map` mapping every alias to its committed real \
                     id. Batch size is capped (default 1000; the limit is echoed on rejection). \
                     v1 limits: an op may not update/delete a node created in the same batch, \
                     and each committed entity accepts at most one write per batch.",
            make_input_schema::<ApplyBatchRequest>(),
        ),
        Tool::new(
            "list_edges",
            "List edges with optional label filter and pagination. Edges cannot be listed \
                     without a start node, so this returns guidance (use get_outgoing_edges / \
                     get_incoming_edges) with `has_more: false` for response-shape consistency. \
                     Vector/embedding properties are elided by default (replaced with a \
                     `{type, dim, elided:true}` descriptor) to protect LLM context; pass \
                     `include_vectors: true` to receive the full float arrays.",
            make_input_schema::<ListEdgesRequest>(),
        ),
        Tool::new(
            "count_edges",
            "Count the total number of edges.",
            make_input_schema::<CountEdgesRequest>(),
        ),
        Tool::new(
            "get_outgoing_edges",
            "Get all outgoing edges from a node. Returns the complete set (never \
                     truncated) in the default full-adjacency mode, so the response carries \
                     `has_more: false` and `total_matching` equal to `count`; with \
                     `use_cursor: true` the adjacency is paged and `has_more` may be true with a \
                     continuation `cursor`. Vector/embedding properties are elided \
                     by default (replaced with a `{type, dim, elided:true}` descriptor) to \
                     protect LLM context; pass `include_vectors: true` to receive the full float \
                     arrays. Pass `use_cursor: true` for snapshot-anchored cursor paging; echo \
                     the returned `cursor` back verbatim (no other args) for the next page; when \
                     the response omits `cursor` / `has_more` is false, the scan is complete.",
            make_cursor_input_schema::<GetOutgoingEdgesRequest>(),
        ),
        Tool::new(
            "get_incoming_edges",
            "Get all incoming edges to a node. Returns the complete set (never truncated) \
                     in the default full-adjacency mode, so the response carries \
                     `has_more: false` and `total_matching` equal to `count`; with \
                     `use_cursor: true` the adjacency is paged and `has_more` may be true with a \
                     continuation `cursor`. Vector/embedding properties are elided by \
                     default (replaced with a `{type, dim, elided:true}` descriptor) to protect \
                     LLM context; pass `include_vectors: true` to receive the full float arrays. \
                     Pass `use_cursor: true` for snapshot-anchored cursor paging; echo the \
                     returned `cursor` back verbatim (no other args) for the next page; when the \
                     response omits `cursor` / `has_more` is false, the scan is complete.",
            make_cursor_input_schema::<GetIncomingEdgesRequest>(),
        ),
        Tool::new(
            "traverse",
            "Traverse the graph starting from a node. The response carries `has_more` \
                     (true when the traversal was truncated by `limit` — check it before trusting \
                     `count`); when true, `next_offset` gives the `offset` to pass for the next \
                     page. Vector/embedding properties are \
                     elided by default (replaced with a `{type, dim, elided:true}` descriptor) to \
                     protect LLM context; pass `include_vectors: true` to receive the full float \
                     arrays. Accepts optional as_of_valid_time / as_of_transaction_time (ISO 8601 \
                     or microseconds since epoch) to walk the graph as it existed at that \
                     bi-temporal instant instead of the current state -- e.g. \"Alice's KNOWS \
                     network as of last year\". Edges/nodes not valid at that instant are \
                     excluded; omitting both parameters reproduces today's current-state behavior \
                     exactly. Pass `use_cursor: true` for snapshot-anchored cursor paging; echo \
                     the returned `cursor` back verbatim (no other args) for the next page; when \
                     the response omits `cursor` / `has_more` is false, the scan is complete.",
            make_cursor_input_schema::<TraverseRequest>(),
        ),
        Tool::new(
            "find_similar",
            "Find nodes similar to a query embedding. The response carries `has_more` \
                     (true when more similar nodes exist beyond the returned `k`); when true, \
                     `next_offset` gives the `offset` to pass for the next page. Pagination via \
                     `offset` only reaches the top-ranked matches up to the server's k cap \
                     (`offset + k` bounded); querying past that horizon returns an empty page with \
                     `has_more: false` rather than an unbounded search. The similarity \
                     `score` is always returned in full. Vector/embedding properties on the \
                     returned nodes are elided by default (replaced with a \
                     `{type, dim, elided:true}` descriptor) to protect LLM context; pass \
                     `include_vectors: true` to receive the full float arrays.",
            make_input_schema::<FindSimilarRequest>(),
        ),
        Tool::new(
            "embed_query",
            "Generate a single dense embedding vector from a text string using the server's \
                     configured embedding model (Issue #2906). Returns `{embedding, dim}`. Use it \
                     to build a query vector for find_similar, or as a general text-to-vector \
                     utility. Requires the server to be built with the `embeddings` feature and to \
                     have a model configured; otherwise returns a structured \
                     FAILED_PRECONDITION error. Input size is bounded.",
            make_input_schema::<EmbedQueryRequest>(),
        ),
        Tool::new(
            "embed_text",
            "Generate dense embeddings for multiple texts with real chunk expansion (Issue \
                     #2906): each input document is split into contiguous character windows and \
                     every chunk is embedded independently, so a long document yields MULTIPLE \
                     embeddings instead of one truncated vector. Returns per-chunk results \
                     `{text, metadata, embedding, dim}` aligned to their source chunk via the \
                     model's own chunk output (never a positional zip); `metadata` carries the \
                     originating `source_index` and `chunk_index`. An optional `max_chunks` caps \
                     the returned embeddings (a value of 0 is rejected as INVALID_ARGUMENT); the \
                     response sets `truncated: true` when the cap trims the expansion. Requires \
                     the `embeddings` feature and a configured model; otherwise returns a \
                     structured FAILED_PRECONDITION error. Input count and size are bounded.",
            make_input_schema::<EmbedTextRequest>(),
        ),
        Tool::new(
            "semantic_search",
            "Text semantic search (Issue #2906): embed `query_text` with the server's \
                     configured model, then run the exact find_similar k-NN path against the vector \
                     index on `property_name`, so the response envelope is identical to \
                     find_similar (results, score, temporal block, #3220 vector elision, #3226 \
                     pagination). Refuses with FAILED_PRECONDITION if the index/property does not \
                     exist, INVALID_ARGUMENT on an embedding-dimension mismatch. Requires the \
                     `embeddings` feature and a configured model.",
            make_input_schema::<SemanticSearchRequest>(),
        ),
        Tool::new(
            "create_node_with_embedding",
            "Create a node whose embedding is generated from `text` (Issue #2906): embed the \
                     text with the server's configured model and store the vector under \
                     `embedding_property` (alongside any other `properties`), compatible with \
                     enable_vector_index / find_similar / semantic_search. Supports optional \
                     `valid_time` and `provenance`. Requires the `embeddings` feature and a \
                     configured model.",
            make_input_schema::<CreateNodeWithEmbeddingRequest>(),
        ),
        Tool::new(
            "update_node_embedding",
            "Regenerate a node's embedding from `text` and update ONLY the embedding property \
                     (Issue #2906). Because update_node replaces all properties, this first reads \
                     the node and MERGES its existing properties, then overrides `embedding_property` \
                     with the freshly generated vector — every other property is preserved. \
                     Supports optional `valid_time`. Requires the `embeddings` feature and a \
                     configured model.",
            make_input_schema::<UpdateNodeEmbeddingRequest>(),
        ),
        Tool::new(
            "enable_vector_index",
            "Enable vector indexing on a property.",
            make_input_schema::<EnableVectorIndexRequest>(),
        ),
        Tool::new(
            "list_vector_indexes",
            "List all enabled vector indexes.",
            make_input_schema::<ListVectorIndexesRequest>(),
        ),
        Tool::new(
            "enable_unique_constraint",
            "Enable a uniqueness constraint on a label+property pair. Fails fast if existing duplicates are found.",
            make_input_schema::<EnableUniqueConstraintRequest>(),
        ),
        Tool::new(
            "list_unique_constraints",
            "List all active uniqueness constraints.",
            make_input_schema::<ListUniqueConstraintsRequest>(),
        ),
        Tool::new(
            "get_node_at_time",
            "Get node state at a specific time.",
            make_input_schema::<GetNodeAtTimeRequest>(),
        ),
        Tool::new(
            "get_edge_at_time",
            "Get edge state at a specific time.",
            make_input_schema::<GetEdgeAtTimeRequest>(),
        ),
        Tool::new(
            "find_nodes_at_time",
            "Find nodes by label (and optional exact property_key/property_value match) as of \
                     a bi-temporal point -- resolve e.g. \"the Person named Alice as of \
                     2024-01-01\" in one call, without knowing any node ID. `valid_time` is \
                     required (ISO 8601 or microseconds since epoch); `transaction_time` is \
                     optional and defaults to now. Each returned node is reconstructed AS IT \
                     EXISTED at that coordinate (not its current state); nodes that did not \
                     exist, or whose property value did not hold, at that point are excluded. \
                     Recalling a since-deleted node requires anchoring BOTH dimensions before \
                     the deletion. Results are sorted by node id; the response echoes the \
                     resolved valid_time/transaction_time and carries `has_more`/`next_offset`/\
                     `total_matching` pagination metadata. On databases with very large \
                     bi-temporal history the candidate scan is capped; the response then sets \
                     `sampled: true` and `total_matching` counts matches within the sampled \
                     candidate set only. Vector/embedding properties are \
                     elided by default; pass `include_vectors: true` for full arrays. \
                     Pass `use_cursor: true` for snapshot-anchored cursor paging; echo the \
                     returned `cursor` back verbatim (no other args) for the next page; when the \
                     response omits `cursor` / `has_more` is false, the scan is complete.",
            make_cursor_input_schema::<FindNodesAtTimeRequest>(),
        ),
        Tool::new(
            "list_changes",
            "List graph-wide changes (node & edge versions) committed in a transaction-time window, with optional valid-time and label filters and stable cursor pagination. Discover what changed without knowing entity IDs.",
            make_input_schema::<ListChangesRequest>(),
        ),
        Tool::new(
            "await_changes",
            "Long-poll for the next committed changes matching an optional node-label / edge-type / change-type filter. Blocks up to timeout_ms (default 25000, max 60000); resume losslessly by passing the prior resume_token back as from_token. The streaming counterpart to list_changes.",
            make_input_schema::<AwaitChangesRequest>(),
        ),
        Tool::new(
            "get_node_at_valid_time",
            "Get node state at a specific valid time (independent dimension query).",
            make_input_schema::<GetNodeAtValidTimeRequest>(),
        ),
        Tool::new(
            "get_node_at_transaction_time",
            "Get node state at a specific transaction time (independent dimension query).",
            make_input_schema::<GetNodeAtTransactionTimeRequest>(),
        ),
        Tool::new(
            "get_node_history",
            "Get complete version history of a node.",
            make_input_schema::<GetNodeHistoryRequest>(),
        ),
        Tool::new(
            "diff_node_versions",
            "Compute the difference between two versions of a node.",
            make_input_schema::<DiffNodeVersionsRequest>(),
        ),
        Tool::new(
            "get_edge_at_valid_time",
            "Get edge state at a specific valid time (independent dimension query).",
            make_input_schema::<GetEdgeAtValidTimeRequest>(),
        ),
        Tool::new(
            "get_edge_at_transaction_time",
            "Get edge state at a specific transaction time (independent dimension query).",
            make_input_schema::<GetEdgeAtTransactionTimeRequest>(),
        ),
        Tool::new(
            "get_edge_history",
            "Get complete version history of an edge.",
            make_input_schema::<GetEdgeHistoryRequest>(),
        ),
        Tool::new(
            "diff_edge_versions",
            "Compute the difference between two versions of an edge.",
            make_input_schema::<DiffEdgeVersionsRequest>(),
        ),
        Tool::new(
            "get_belief_revisions",
            "Audit when and why the database changed its mind about a node or edge \
             (Issue #3362): classify each stored version transition as \
             initial_assertion / correction / world_change / retraction / reaffirmation, \
             with the provenance and confidence trajectory. Optionally scope to one \
             property key or a transaction-time coordinate. Requires the \
             `semantic-temporal` feature.",
            make_input_schema::<GetBeliefRevisionsRequest>(),
        ),
        Tool::new(
            "create_drift_monitor",
            "Declare a semantic-drift monitor (Issue #3367): watch a vector \
             property and fire a durable, queryable alarm when meaning drifts \
             past a threshold over a time window (per-entity or label-centroid; \
             on-write or scheduled). Requires the `semantic-temporal` feature.",
            make_input_schema::<CreateDriftMonitorRequest>(),
        ),
        Tool::new(
            "list_drift_monitors",
            "List all declared drift monitors and their specs (Issue #3367). \
             Requires the `semantic-temporal` feature.",
            make_input_schema::<ListDriftMonitorsRequest>(),
        ),
        Tool::new(
            "delete_drift_monitor",
            "Delete a drift monitor by id, removing it from future evaluation \
             (Issue #3367). Requires the `semantic-temporal` feature.",
            make_input_schema::<DeleteDriftMonitorRequest>(),
        ),
        Tool::new(
            "query_drift_alarms",
            "Query fired drift alarms (Issue #3367), filtered by monitor, label, \
             resolved state, and fire-time window. Each alarm carries the measured \
             distance, threshold, metric, both compared coordinates, and version \
             refs. Requires the `semantic-temporal` feature.",
            make_input_schema::<QueryDriftAlarmsRequest>(),
        ),
        Tool::new(
            "resolve_drift_alarm",
            "Resolve a drift alarm (Issue #3367) as a recorded, AS OF-stable \
             update (the alarm is never deleted). Requires the `semantic-temporal` \
             feature.",
            make_input_schema::<ResolveDriftAlarmRequest>(),
        ),
        Tool::new(
            "contradiction_genealogy",
            "Reconstruct how conflicting claims about one fact evolved across \
             bi-temporal history and provenance (Issue #3352): every competing \
             claim's valid/transaction intervals, provenance, supersession chain, \
             the divergence point, per-source trust summary, and a prose narrative. \
             Requires the `semantic-temporal` feature.",
            make_input_schema::<ContradictionGenealogyRequest>(),
        ),
        Tool::new(
            "find_contradictions",
            "Scan the database (by label / property / time window, paginated) for \
             entity+property pairs holding conflicting values over overlapping \
             valid-time intervals (Issue #3352). Requires the `semantic-temporal` \
             feature.",
            make_input_schema::<FindContradictionsRequest>(),
        ),
        Tool::new(
            "counterfactual_replay",
            "Materialize a read-only counterfactual view that excludes a source's \
             writes and report the blast radius — which entities changed or were \
             removed (Issue #3357). The real database is never mutated; the \
             response carries a `counterfactual: true` marker. Requires the \
             `semantic-temporal` feature.",
            make_input_schema::<CounterfactualReplayRequest>(),
        ),
        Tool::new(
            "trust_breakdown",
            "Explain a fact's computed confidence as a tree over its derivation \
             lineage (Issue #3382): each node's version-pinned reference, status, \
             computed confidence, classification, and combinator. Requires the \
             `semantic-reasoning` feature.",
            make_input_schema::<TrustBreakdownRequest>(),
        ),
        Tool::new(
            "list_trust_policies",
            "List the active trust-propagation policies (Issue #3382): the \
             database default combinator + missing-confidence rule and per-label \
             overrides. Requires the `semantic-reasoning` feature.",
            make_input_schema::<ListTrustPoliciesRequest>(),
        ),
        Tool::new(
            "hybrid_query",
            "Execute a hybrid query combining graph traversal, vector similarity, and \
                     temporal filtering. The `similarity_score` is always returned in full. \
                     Vector/embedding properties on returned nodes are elided by default \
                     (replaced with a `{type, dim, elided:true}` descriptor) to protect LLM \
                     context; pass `include_vectors: true` to receive the full float arrays.",
            make_input_schema::<HybridQueryRequest>(),
        ),
        Tool::new(
            "query",
            "Execute a single READ-ONLY declarative query and get structured rows back, \
             replacing a chain of get_node/traverse/filter calls. \
             `language` is \"cypher\" or \"aql\"; pass the statement in `query` and optional \
             `$param` bindings in `params` (Cypher only; numeric arrays are treated as \
             embeddings). \
             Supported read-only subset: MATCH patterns with node/edge labels and inline \
             property filters, variable-depth traversal (-[:REL*1..3]->), directions \
             (->, <-, -), WHERE, RETURN [DISTINCT] / AS aliases, ORDER BY, SKIP/LIMIT, WITH \
             chaining, vector similarity ranking, and bi-temporal scoping \
             (AS OF TIMESTAMP/VALID_TIME/SYSTEM_TIME, FOR SYSTEM_TIME AS OF, BETWEEN ... AND ...). \
             AQL WHERE clauses can filter on write-time provenance (Issue #3354a) via the \
             read-only accessors source(x)/confidence(x)/reason(x) (e.g. \
             WHERE confidence(n) >= 0.9 AND source(n) = 'hr-system') and provenance(x) IS [NOT] NULL; \
             provenance is evaluated on the version visible at the query's AS OF coordinate, \
             unattributed versions are excluded (select them with provenance(x) IS NULL). \
             Mutating statements (CREATE/MERGE/SET/DELETE/REMOVE/DETACH/DROP/CALL/FOREACH/LOAD) \
             are rejected before execution and never write. Results are capped (default 100, \
             max 10000 rows; `truncated` indicates a cap hit). Per-query resource limits \
             (Issue #3368) bound each call: a wall-clock timeout, a result-row cap (disclosed \
             via `truncated`), and a result-byte cap. Pass optional `limits` \
             ({timeout_ms, max_result_rows, max_response_bytes}) to tighten within the operator \
             ceilings; a timeout or byte-cap breach fails closed with a RESOURCE_EXHAUSTED \
             error (timeouts are retriable, byte-cap breaches are not), and an over-ceiling \
             override is rejected INVALID_ARGUMENT. Errors are returned as a \
             structured {error:{kind,code,retriable,message,clause?,language,details?}} payload \
             (kinds: invalid_request, read_only_violation, language_unavailable, parse_error, \
             unsupported_construct, invalid_params, runtime_error; code/retriable follow the \
             uniform MCP error contract) so callers can self-correct.",
            make_input_schema::<QueryRequest>(),
        ),
        Tool::new(
            "get_schema",
            "Discover the graph's schema: distinct node labels and edge/relationship types, \
             each with an entity count and the union of property keys observed. Call this \
             first to learn what labels/edge-types/properties exist before guessing names in \
             list_nodes, traverse, or query. Accepts optional as_of_valid_time / \
             as_of_transaction_time (ISO 8601 or microseconds since epoch) to return the \
             schema as it existed at that bi-temporal instant instead of the current state. \
             Returns a well-formed empty summary on an empty database, never an error.",
            make_input_schema::<GetSchemaRequest>(),
        ),
        Tool::new(
            "temporal_extent",
            "Report the dataset's queryable bi-temporal extent: valid_time {earliest, latest} \
             and transaction_time {earliest, latest} as RFC3339 strings, in one call. Call \
             this BEFORE issuing AS OF queries so the target instant lands inside recorded \
             data — an AS OF before valid_time.earliest returns empty results that are \
             otherwise indistinguishable from 'the fact never existed'. Bounds cover ALL \
             recorded history, including expired/superseded versions and deletions (a 2019 \
             fact later corrected still counts toward earliest); this is a calendar RANGE, \
             not a current-state count. Convention: earliest = the minimum interval start in \
             that dimension; latest = the maximum of interval starts and CLOSED interval \
             ends. Open-ended intervals (still-valid facts / still-current records) \
             contribute only their start, so latest is the newest finite recorded event \
             coordinate, never +infinity. An empty database returns explicit nulls for every \
             bound, never 0/epoch-1970. Bounds only ever widen while the server runs, so \
             callers can cache the result for a session. Coverage caveat: bounds span all \
             history recorded during the current process lifetime plus hot-tier history \
             restored at startup; on databases with cold-storage migration, versions migrated \
             to cold storage before the last restart are not reflected. Pass by_label: true \
             to also receive the same bounds per node label (node_labels) and per edge type \
             (edge_types); per-label bounds are computed from hot-tier history only and may \
             be narrower than the overall bounds (or a label absent) after cold migration.",
            make_input_schema::<TemporalExtentRequest>(),
        ),
        Tool::new(
            "lineage_upstream",
            "Query the UPSTREAM derivation lineage of a fact (Issue #3371): 'what was this fact \
             derived from?', transitively — the evidence chain for citation-grade answers. \
             Arguments: entity_kind ('node'|'edge'), id, version (lineage is version-pinned; use \
             the exact version whose evidence you want), optional max_depth (transitive hop cap; \
             1 = direct parents), optional limit (default 100) and offset for pagination, and \
             optional as_of_transaction_time (ISO 8601 / RFC 3339 or integer microseconds) to see \
             lineage as it was recorded by that transaction time. Returns {direction:'upstream', \
             root, entries[], count, has_more, next_offset?}; each entry is a version-pinned ref \
             {entity_kind, id, version} plus depth (min hops from the root) and status \
             ('current'|'superseded'|'absent'). Lineage records are immutable, so a source that \
             was later retracted/deleted still resolves and is marked 'absent'. Declare lineage at \
             write time via the create_node/create_edge/update_node/update_edge `derived_from` \
             parameter.",
            make_input_schema::<LineageQueryRequest>(),
        ),
        Tool::new(
            "lineage_downstream",
            "Query the DOWNSTREAM derivation lineage of a fact (Issue #3371): 'what has been \
             derived from this fact?', transitively — the retraction BLAST RADIUS. When an input \
             fact is found wrong or retracted, one call enumerates every transitively derived \
             fact so you can assess contamination. Arguments: entity_kind ('node'|'edge'), id, \
             version, optional max_depth (transitive hop cap; 1 = direct children), optional limit \
             (default 100) and offset for pagination, and optional as_of_transaction_time (ISO \
             8601 / RFC 3339 or integer microseconds). Returns {direction:'downstream', root, \
             entries[], count, has_more, next_offset?}; each entry is a version-pinned ref \
             {entity_kind, id, version} plus depth (min hops from the root) and status \
             ('current'|'superseded'|'absent'). Version-pinned: the closure reflects exactly the \
             versions that declared this fact as a source.",
            make_input_schema::<LineageQueryRequest>(),
        ),
        Tool::new(
            "audit_export",
            "Produce a SIGNED, self-contained audit export of a single entity's (node or edge) \
             complete bi-temporal history plus provenance — a portable evidence artifact a \
             third party can verify OFFLINE with only the signer's public key (no database, no \
             network). Use this to answer 'prove what the system knew about entity X, and \
             when' for compliance, audits, GDPR/CCPA subject-access requests, or legal \
             discovery. Arguments: entity_type ('node'|'edge'), entity_id, optional \
             database_id (recorded in the artifact), optional redact_keys (property keys whose \
             VALUES are omitted while the redaction stays recorded and verifiable). The \
             artifact contains every version across both time dimensions (including superseded \
             versions and delete tombstones), per-version provenance/principal where recorded, \
             a per-version SHA-256 hash chain, and an Ed25519 signature over the chain root. \
             The Ed25519 signing key is operator-provided out of band via the \
             ALETHEIADB_AUDIT_SIGNING_KEY environment variable (32-byte hex seed); the secret \
             is never returned — only the public key travels in the artifact. Returns the \
             artifact JSON plus public_key, chain_root, and entity/version counts. Note: \
             delete/retract operations do not yet stamp an authenticated principal (#3427); \
             the export surfaces provenance faithfully, including its absence, and never \
             fabricates attribution.",
            make_input_schema::<AuditExportRequest>(),
        ),
        Tool::new(
            "database_stats",
            "Get a holistic database statistics snapshot in one call (no arguments). Use it \
             to orient yourself before querying: how big is the dataset, how much history \
             exists to time-travel through, where that history lives, and what durability \
             is active. Returns: `current` {node_count, edge_count} (current-state graph \
             size); `historical` {total_node_versions, total_edge_versions, unique_nodes, \
             unique_edges, anchor_count, delta_count, node/edge anchor+delta breakdowns, \
             compression_ratio} (bi-temporal depth of the in-RAM store; anchors are full \
             snapshots, deltas are changes — anchor_count + delta_count always equals total \
             versions, and compression_ratio = anchors/total, lower is better); \
             `cold_storage` — `{enabled: false}` when no cold (disk) tier is configured \
             (this means NOT CONFIGURED, never '0 cold versions'), or `{enabled: true, \
             node_versions_stored, edge_versions_stored, compression_ratio, tier_access: \
             {hot_hits, warm_hits, cold_hits, misses}}` showing how many versions live on \
             disk (counts persist across restarts) and how reads distribute across the \
             hot/warm-cache/cold tiers (compression_ratio and tier_access count activity \
             since the current process opened the database); `wal` {enabled, \
             durability_mode (synchronous|async|group_commit|async_batched), current_lsn \
             (the NEXT log sequence number to be allocated), total_appends, healthy}. All \
             values are O(1)/cached counter reads — this call never scans versions and is \
             safe to call frequently. Counts only, point-in-time: for the calendar RANGE \
             of stored history (earliest/latest timestamps) use temporal_extent; \
             for per-label breakdowns use get_schema.",
            make_input_schema::<DatabaseStatsRequest>(),
        ),
        // ── Semantic-search analysis tools (Issue #2907) ──────────────────
        Tool::new(
            "semantic_path",
            "Find a path between two nodes guided by vector similarity (A* over the \
             `semantic-search` cohort's Semantic Navigator). Returns the node-id path plus its \
             length. Requires a vector index on `property_name`; both endpoints must carry that \
             embedding. The search is bounded (a node-expansion budget derived from `max_depth`, \
             clamped to 20) so it is safe on large graphs. Requires the `semantic-search` \
             feature; otherwise returns a FAILED_PRECONDITION.",
            make_input_schema::<SemanticPathRequest>(),
        ),
        Tool::new(
            "concept_analogy",
            "Solve a vector analogy `a : b :: c : ?` over node embeddings (Concept Algebra): \
             returns the top-k nodes nearest `b - a + c`, each with a similarity score. Requires \
             a vector index on `property_name`. Requires the `semantic-search` feature; otherwise \
             returns a FAILED_PRECONDITION.",
            make_input_schema::<ConceptAnalogyRequest>(),
        ),
        Tool::new(
            "concept_mean",
            "Rank the top-k nodes nearest the centroid (mean embedding) of a set of nodes \
             (Concept Algebra). Useful for 'find things like this whole group'. Requires a vector \
             index on `property_name`; the node set is capped. Requires the `semantic-search` \
             feature; otherwise returns a FAILED_PRECONDITION.",
            make_input_schema::<ConceptMeanRequest>(),
        ),
        Tool::new(
            "find_duplicate_candidates",
            "Find near-duplicate candidates for a node by embedding similarity (Highlander \
             entity-resolution detector): returns candidates at or above `threshold`, each with a \
             similarity score. v1 note: the search uses the node's own indexed embedding; \
             `property_name` selects the vector index whose existence is validated, not an \
             arbitrary property vector. Requires the `semantic-search` feature; otherwise returns \
             a FAILED_PRECONDITION.",
            make_input_schema::<FindDuplicateCandidatesRequest>(),
        ),
        Tool::new(
            "semantic_horizon",
            "Map a node's semantic event horizon (Horizon engine): starting from `seed`, expand \
             through neighbours whose similarity is at or above `threshold` (the interior); the \
             first neighbours to fall below form the horizon (boundary). Returns sorted interior \
             and horizon node-id sets with counts. `threshold` must be in [0, 1]; `max_depth` is \
             clamped to 20. Requires a vector index on `property_name` and the `semantic-search` \
             feature; otherwise returns a FAILED_PRECONDITION.",
            make_input_schema::<SemanticHorizonRequest>(),
        ),
        Tool::new(
            "context_aspects",
            "Decompose a node's neighbourhood into distinct semantic aspects (Chameleon): each \
             aspect carries a weight, representative exemplar node ids, and a centroid vector \
             (elided by default per #3220 — pass `include_vectors: true` for the full array). \
             Useful for disentangling multi-faceted context. Requires a vector index on \
             `property_name` and the `semantic-search` feature; otherwise returns a \
             FAILED_PRECONDITION.",
            make_input_schema::<ContextAspectsRequest>(),
        ),
        Tool::new(
            "verify_chain",
            "Verify the tamper-evident provenance hash chain (Issue #3351) — proof that the \
             recorded bi-temporal history has not been altered. Three modes: (1) FULL (no \
             arguments) walks the whole chain from genesis and, on tamper, reports the \
             `earliest_broken_seq`; (2) ENTITY-SCOPED (pass `entity_kind` 'node'|'edge' and \
             `id`) recomputes only that entity's contribution; (3) ANCHOR EXTENSION (pass \
             `against`, a previously exported chain head from export_chain_head) proves the \
             current chain append-only-extends that anchor, detecting rollback (truncation) \
             and fork (divergence). Read-only. Returns {scope, passed, head_seq, head_digest, \
             earliest_broken_seq, reason, transactions_checked}. Requires the chain to be \
             enabled for this database; if it is not, returns a FAILED_PRECONDITION error \
             (never a silent empty pass).",
            make_input_schema::<VerifyChainRequest>(),
        ),
        Tool::new(
            "export_chain_head",
            "Export the current provenance hash chain head as an external anchor (Issue \
             #3351). Store the returned checkpoint offsite; later pass it back as \
             verify_chain's `against` argument to prove the chain has only been appended to \
             (detecting rollback and fork). No arguments. Returns the chain head {seq, digest, \
             commit_ts, anchor_lsn, genesis_digest} with digests as lowercase hex. Requires \
             the chain to be enabled; otherwise returns a FAILED_PRECONDITION error.",
            make_input_schema::<ExportChainHeadRequest>(),
        ),
        // Namespace management (Issue #3349, PR3b).
        Tool::new(
            "create_namespace",
            "Create (register) an agent-scoped namespace so it is listable/describable even \
             when empty (Issue #3349). Pass `name` (e.g. 'agent:planner'; charset \
             [A-Za-z0-9._:/-], max 128 bytes) and an optional `description`. Returns the created \
             namespace {name, description, created_at}. A duplicate name (or the implicit \
             'default') is a CONFLICT; a malformed name or the reserved 'all' selector is \
             INVALID_ARGUMENT. Registering up front is optional — writing to an unknown namespace \
             auto-registers it — but doing so lets you record a description and detect typos via \
             list_namespaces.",
            make_input_schema::<CreateNamespaceRequest>(),
        ),
        Tool::new(
            "list_namespaces",
            "List all registered namespaces (the implicit 'default' first, then others in \
             creation order), so a caller can discover every scope and catch a mistyped \
             auto-registered namespace (Issue #3349). No arguments. Returns \
             {namespaces:[{name, description, created_at, node_count, edge_count}], count}; the \
             per-namespace current-state node/edge counts are O(1) membership-index reads.",
            make_input_schema::<ListNamespacesRequest>(),
        ),
        Tool::new(
            "describe_namespace",
            "Describe a single namespace by name (Issue #3349). Pass `name`; returns \
             {name, description, created_at, node_count, edge_count}. The implicit 'default' \
             namespace always resolves; an unregistered name returns NOT_FOUND with \
             details.namespace.",
            make_input_schema::<DescribeNamespaceRequest>(),
        ),
        // GDPR crypto-shred — ADMIN-class (Issue #3359, Slice 4b). The first
        // Admin-class MCP tools.
        Tool::new(
            "designate_subject",
            "ADMIN. Designate a GDPR erasure subject over one or more targets — whole \
             nodes/edges and/or specific property keys (Issue #3359). Pass `subject_id` \
             (non-empty, <=256 bytes, no control chars) and a non-empty `targets` array of \
             {entity_kind:'node'|'edge', id, keys?} — `keys` present seals only those \
             properties, absent/empty seals the whole entity. Designating an already-active \
             subject merges the new targets in. Requires encryption configured, else \
             FAILED_PRECONDITION. Erasure of the designated subject later renders its sealed \
             payload permanently undecryptable.",
            make_input_schema::<DesignateSubjectRequest>(),
        ),
        Tool::new(
            "erase_subject",
            "ADMIN. Irreversibly erase a designated GDPR subject: destroy its key material so \
             its sealed payload becomes permanently undecryptable, and return a signed erasure \
             attestation {subject_id, entity_count, timestamp, timestamp_micros, signature (hex), \
             signer_public_key (hex)} — never any property content or key material. Pass \
             `subject_id`. Erasing an undesignated subject is FAILED_PRECONDITION; re-erasing is \
             an idempotent no-op returning the recorded attestation.",
            make_input_schema::<EraseSubjectRequest>(),
        ),
    ];

    // Advertise the Issue #3353 token budget on every budgetable read tool by
    // (a) injecting the three budget parameters into its generated
    // `inputSchema.properties` so they are machine-discoverable, and (b)
    // appending a uniform prose hint to its description as a secondary,
    // human-readable pointer. Both are kept in lockstep with
    // `BUDGETABLE_READ_TOOLS` (the dispatch-path source of truth) without
    // editing each tool literal.
    for tool in tools.iter_mut() {
        if is_budgetable_read_tool(&tool.name) {
            inject_budget_schema_params(&mut tool.input_schema);
            let mut desc = tool.description.as_deref().unwrap_or("").to_string();
            desc.push_str(BUDGET_TOOL_HINT);
            tool.description = Some(std::borrow::Cow::Owned(desc));
        }
        // Advertise the Issue #3348 provenance filter on every filterable read
        // tool, kept in lockstep with `PROVENANCE_FILTERABLE_TOOLS`.
        if is_provenance_filterable_tool(&tool.name) {
            inject_provenance_filter_schema_params(&mut tool.input_schema);
            let mut desc = tool.description.as_deref().unwrap_or("").to_string();
            desc.push_str(PROVENANCE_FILTER_TOOL_HINT);
            tool.description = Some(std::borrow::Cow::Owned(desc));
        }
        // Advertise the Issue #3372 provenance-weighted fusion policy on
        // `find_similar` / `hybrid_query`, only when the feature is compiled
        // (so an unusable param is never advertised). Count-neutral: adds a
        // param, not a tool.
        #[cfg(feature = "semantic-retrieval-fusion")]
        if is_fusion_policy_tool(&tool.name) {
            inject_fusion_policy_schema_params(&mut tool.input_schema);
            let mut desc = tool.description.as_deref().unwrap_or("").to_string();
            desc.push_str(FUSION_POLICY_TOOL_HINT);
            tool.description = Some(std::borrow::Cow::Owned(desc));
        }
    }
    tools
}

/// The two tools that accept an optional Issue #3372 `fusion_policy` param.
#[cfg(feature = "semantic-retrieval-fusion")]
pub(crate) const FUSION_POLICY_TOOLS: &[&str] = &["find_similar", "hybrid_query"];

/// Whether `name` accepts the Issue #3372 `fusion_policy` param.
#[cfg(feature = "semantic-retrieval-fusion")]
pub(crate) fn is_fusion_policy_tool(name: &str) -> bool {
    FUSION_POLICY_TOOLS.contains(&name)
}

/// Inject the optional Issue #3372 `fusion_policy` object parameter into a
/// tool's generated JSON `inputSchema.properties`. Optional, so `required` is
/// untouched. Idempotent.
#[cfg(feature = "semantic-retrieval-fusion")]
fn inject_fusion_policy_schema_params(
    schema: &mut Arc<serde_json::Map<String, serde_json::Value>>,
) {
    let schema = Arc::make_mut(schema);
    let props = schema
        .entry("properties".to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    let Some(props) = props.as_object_mut() else {
        return;
    };
    props.entry("fusion_policy".to_string()).or_insert_with(|| {
        json!({
            "type": "object",
            "description": "Optional provenance-weighted fusion policy (Issue #3372): re-rank \
                candidates by a weighted mean of vector similarity, provenance confidence, and \
                temporal recency instead of similarity alone, attaching a per-result \
                `score_breakdown`. Omit for unchanged (similarity-only) behavior. Invalid weights \
                return INVALID_ARGUMENT.",
            "properties": {
                "w_similarity": { "type": "number", "minimum": 0 },
                "w_confidence": { "type": "number", "minimum": 0 },
                "w_recency": { "type": "number", "minimum": 0 },
                "neutral_confidence": { "type": "number", "minimum": 0, "maximum": 1 },
                "recency_half_life_secs": { "type": "number", "exclusiveMinimum": 0 }
            }
        })
    });
}

/// Uniform description suffix documenting the `fusion_policy` parameter
/// (Issue #3372), appended to `find_similar` / `hybrid_query`.
#[cfg(feature = "semantic-retrieval-fusion")]
const FUSION_POLICY_TOOL_HINT: &str = " Optional provenance-weighted fusion (Issue #3372): pass \
    `fusion_policy` (an object with any of `w_similarity`, `w_confidence`, `w_recency` \
    (each >= 0), `neutral_confidence` (in [0,1]), `recency_half_life_secs` (> 0)) to re-rank \
    results by a weighted mean of vector similarity, provenance confidence, and temporal recency; \
    each result then carries a `score_breakdown`. Invalid weights return INVALID_ARGUMENT. Omit \
    for unchanged similarity-only behavior.";

/// Inject the four optional Issue #3348 provenance-filter parameters into a
/// tool's generated JSON `inputSchema.properties`, so a client introspecting
/// the schema discovers them with correct types. All are optional, so
/// `required` is untouched. Idempotent.
fn inject_provenance_filter_schema_params(
    schema: &mut Arc<serde_json::Map<String, serde_json::Value>>,
) {
    let schema = Arc::make_mut(schema);
    let props = schema
        .entry("properties".to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    let Some(props) = props.as_object_mut() else {
        return;
    };
    props
        .entry("provenance_source".to_string())
        .or_insert_with(|| {
            json!({
                "type": "string",
                "description": "Optional provenance filter (Issue #3348): keep only facts whose \
                    write-time provenance source exactly equals this value. Combine with \
                    'provenance_sources' (any-of) and/or 'min_confidence' (AND).",
            })
        });
    props
        .entry("provenance_sources".to_string())
        .or_insert_with(|| {
            json!({
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional provenance filter (Issue #3348): keep facts whose \
                    provenance source is ANY of these values (unioned with 'provenance_source'). \
                    An empty list is rejected as INVALID_ARGUMENT.",
            })
        });
    props
        .entry("min_confidence".to_string())
        .or_insert_with(|| {
            json!({
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Optional provenance filter (Issue #3348): keep only facts whose \
                    recorded confidence is >= this INCLUSIVE threshold, in [0.0, 1.0]. \
                    Out-of-range/NaN is rejected as INVALID_ARGUMENT.",
            })
        });
    props
        .entry("include_unattributed".to_string())
        .or_insert_with(|| {
            json!({
                "type": "boolean",
                "description": "Optional provenance filter (Issue #3348): when true, facts with NO \
                    recorded provenance bundle are re-included despite an active provenance filter \
                    (default false = excluded).",
            })
        });
}

/// Uniform description suffix documenting the provenance-filter parameters
/// (Issue #3348), appended to every filterable read tool.
const PROVENANCE_FILTER_TOOL_HINT: &str = " Optional provenance filter (Issue #3348): pass \
    `provenance_source` (exact) and/or `provenance_sources` (any-of) and/or `min_confidence` \
    (inclusive lower bound in [0,1]) to return only facts whose write-time provenance meets your \
    bar; the constraints are ANDed. The filter is evaluated per-version (so it composes with \
    `as_of_*` temporal reads). Facts with no recorded provenance are excluded unless \
    `include_unattributed: true`. Invalid values (confidence out of [0,1], empty source list) \
    return INVALID_ARGUMENT naming the field. Omit for unchanged behavior.";

/// Inject the three optional Issue #3353 token-budget parameters into a tool's
/// generated JSON `inputSchema.properties`, so a client that introspects the
/// schema (not just the prose description) discovers them with correct types.
/// All three are optional, so `required` is deliberately left untouched. Idempotent.
fn inject_budget_schema_params(schema: &mut Arc<serde_json::Map<String, serde_json::Value>>) {
    let schema = Arc::make_mut(schema);
    let props = schema
        .entry("properties".to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    let Some(props) = props.as_object_mut() else {
        return;
    };
    props
        .entry("max_response_tokens".to_string())
        .or_insert_with(|| {
            json!({
                "type": "integer",
                "minimum": 1,
                "description": "Optional (Issue #3353): cap the response at approximately this \
                    many tokens (estimated as ceil(utf8_bytes/4)). The serialized response, \
                    including its truncation metadata, is guaranteed to fit.",
            })
        });
    props
        .entry("max_response_bytes".to_string())
        .or_insert_with(|| {
            json!({
                "type": "integer",
                "minimum": 1,
                "description": "Optional (Issue #3353): byte-exact cap on the serialized response. \
                    When both this and max_response_tokens are given, the tighter bound wins.",
            })
        });
    props
        .entry("priority_properties".to_string())
        .or_insert_with(|| {
            json!({
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional (Issue #3353): property keys to protect from elision at \
                    every degradation rung when a response budget is active.",
            })
        });
}

/// Uniform description suffix documenting the token-budget parameters
/// (Issue #3353), appended to every budgetable read tool.
const BUDGET_TOOL_HINT: &str = " Optional token budget (Issue #3353): pass \
    `max_response_tokens` (estimated as ceil(utf8_bytes/4)) or the byte-exact \
    `max_response_bytes` to cap the response size; the serialized response \
    (including its truncation metadata) is guaranteed to fit. Over budget, the \
    response degrades deterministically (elide bulky property values → per-entity \
    summaries → counts-plus-handles), carrying a `budget` block naming the rung \
    applied and a fetch handle at every elision site. Protect specific properties \
    with `priority_properties`. A budget too small for the minimal response \
    returns INVALID_ARGUMENT stating the minimum viable budget. Omit for \
    unchanged (row-limit) behavior.";

impl ServerHandler for AletheiaMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::from_build_env())
            .with_instructions(
                "AletheiaDB MCP Server - A bi-temporal graph database with vector search. \
                 Use the provided tools to query and manipulate graph data with full \
                 temporal versioning and vector similarity search capabilities.",
            )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, RmcpErrorData> {
        Ok(ListToolsResult {
            tools: tool_definitions(),
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, RmcpErrorData> {
        let args = request
            .arguments
            .map(serde_json::Value::Object)
            .unwrap_or(serde_json::Value::Null);

        // `await_changes` is an event-driven long-poll (Issue #3673): route it
        // through the async dispatch so the suspended future pins NO worker thread
        // and drops (freeing its per-principal slot) the moment the caller's
        // future is cancelled on disconnect. It is deliberately excluded from the
        // #3353/#3368/#3360 wrappers `dispatch_tool` applies, so bypassing them
        // changes nothing but the wait mechanism — but auth must still be enforced
        // here to mirror `dispatch_tool`.
        if request.name.as_ref() == "await_changes" {
            if let Err(err) = self.auth.authorize_tool("await_changes") {
                return Ok(self.error_result(err));
            }
            return Ok(self.dispatch_await_changes_async(args, None).await);
        }

        Ok(self.dispatch_tool(request.name.as_ref(), args))
    }
}

#[cfg(test)]
mod server_unit_tests {
    use std::sync::Arc;

    use super::{AletheiaMcpServer, CallToolResult, Content};
    use crate::core::PropertyValue;
    use crate::core::error::{Error, QueryError};
    use crate::core::id::{EdgeId, NodeId};
    use crate::db::AletheiaDB;
    use crate::query::executor::{EntityResult, QueryRow};

    fn make_server() -> AletheiaMcpServer {
        AletheiaMcpServer::new(Arc::new(AletheiaDB::new().expect("db init")))
    }

    /// Configurable interner cap (finding 1): a REAL `create_node` whose property
    /// KEY tips the interner over must render as an actionable
    /// `FAILED_PRECONDITION` with `{resource,current,limit}` details naming
    /// `persistence.max_interned_strings` — NOT a generic `INVALID_ARGUMENT`.
    ///
    /// Lowers the process-global interner cap, so it runs in a SUBPROCESS to
    /// isolate the mutation from concurrent tests.
    #[test]
    fn interner_cap_property_key_breach_is_failed_precondition_via_subprocess() {
        use std::process::Command;
        use std::time::{Duration, Instant};

        let exe = std::env::current_exe().expect("failed to locate current test binary");
        let mut child = Command::new(exe)
            .args([
                "--ignored",
                "--exact",
                "mcp::server::server_unit_tests::interner_cap_property_key_breach_helper",
            ])
            .spawn()
            .expect("failed to spawn subprocess for interner-cap property-key test");

        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    assert!(status.success(), "interner-cap property-key helper failed");
                    break;
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        panic!("interner-cap property-key helper did not complete");
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => panic!("failed while polling subprocess: {e}"),
            }
        }
    }

    #[test]
    #[ignore]
    fn interner_cap_property_key_breach_helper() {
        use super::CreateNodeRequest;
        use crate::core::GLOBAL_INTERNER;
        use std::collections::HashMap;

        let server = make_server();

        // Pin the cap exactly at the live interner size, so the NEXT new intern —
        // the fresh property KEY below, interned by `json_to_property_map` before
        // the node label — tips it over. (No capacity rollbacks have occurred in
        // this fresh subprocess, so next_id == len().)
        let base = GLOBAL_INTERNER.len();
        GLOBAL_INTERNER.set_max_capacity(base);

        let mut props = HashMap::new();
        props.insert(
            "uniq_property_key_that_tips_the_interner_over".to_string(),
            serde_json::json!("some_value"),
        );
        let req = CreateNodeRequest {
            label: "TipLabel".to_string(),
            properties: Some(props),
            valid_time: None,
            provenance: None,
            derived_from: None,
            namespace: None,
        };

        let resp = server.create_node(req);
        let val: serde_json::Value =
            serde_json::from_str(&resp).expect("create_node must return JSON");

        // AFTER the fix: FAILED_PRECONDITION with structured details + actionable
        // message. (BEFORE the fix this was INVALID_ARGUMENT with no details.)
        assert_eq!(
            val["error"]["code"], "FAILED_PRECONDITION",
            "property-KEY interner breach must be FAILED_PRECONDITION, got: {resp}"
        );
        assert_eq!(val["error"]["retriable"], false);
        assert_eq!(val["error"]["details"]["resource"], "string interner");
        assert!(
            val["error"]["details"]["limit"].is_number(),
            "details must carry a numeric limit, got: {resp}"
        );
        assert!(
            val["error"]["details"]["current"].is_number(),
            "details must carry a numeric current, got: {resp}"
        );
        assert!(
            val["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("persistence.max_interned_strings"),
            "message must name the knob, got: {resp}"
        );
    }

    fn error_kind(server: &AletheiaMcpServer, err: Error) -> String {
        let result = server.map_query_error(err, "aql");
        let text = AletheiaMcpServer::extract_text(result);
        let val: serde_json::Value = serde_json::from_str(&text).unwrap();
        val["error"]["kind"].as_str().unwrap_or("").to_string()
    }

    /// Like [`error_kind`], but returning the whole serialized `error` object
    /// so tests can assert `code`/`retriable` alongside `kind`.
    fn error_payload(server: &AletheiaMcpServer, err: Error) -> serde_json::Value {
        let result = server.map_query_error(err, "aql");
        let text = AletheiaMcpServer::extract_text(result);
        let val: serde_json::Value = serde_json::from_str(&text).unwrap();
        val["error"].clone()
    }

    // ===================================================================
    // Per-query resource limits (Issue #3368) — low-level unit tests of the
    // timeout race + timeout-error shape. These access private methods, so
    // they live in the descendant `server_unit_tests` module (the sibling
    // `mcp::tests` integration tests cover the end-to-end behavior).
    // ===================================================================

    /// Acquire an unbounded (cap 0) in-flight slot for the low-level
    /// `race_deadline` unit tests that exercise the race in isolation.
    fn test_guard(server: &AletheiaMcpServer) -> super::InFlightGuard {
        server
            .try_acquire_in_flight(0)
            .expect("unbounded acquire always succeeds")
    }

    /// A computation that finishes within the deadline is returned intact.
    #[test]
    fn race_deadline_returns_fast_result() {
        let server = make_server();
        let out = server.race_deadline(1_000, test_guard(&server), || {
            CallToolResult::success(vec![Content::text("done".to_string())])
        });
        match out {
            super::RaceOutcome::Completed(result) => {
                assert_eq!(AletheiaMcpServer::extract_text(result), "done");
            }
            _ => panic!("fast computation must complete within the deadline"),
        }
    }

    /// A computation that overruns the deadline yields `TimedOut` (the caller
    /// then synthesizes the timeout error). Deterministic: 300 ms work vs 10 ms
    /// deadline.
    #[test]
    fn race_deadline_times_out_slow_result() {
        let server = make_server();
        let out = server.race_deadline(10, test_guard(&server), || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            CallToolResult::success(vec![Content::text("late".to_string())])
        });
        assert!(
            matches!(out, super::RaceOutcome::TimedOut),
            "a computation exceeding the deadline must race to TimedOut"
        );
    }

    /// FIX 1 (Issue #3368): a worker that panics mid-computation drops its
    /// sender, which `race_deadline` must surface as `WorkerDied` — NOT a
    /// timeout. The 1 s deadline is far longer than the immediate panic, so a
    /// `TimedOut` here would prove the Disconnected/Timeout collapse bug.
    #[test]
    fn race_deadline_panicking_worker_is_worker_died_not_timeout() {
        let server = make_server();
        let out = server.race_deadline(1_000, test_guard(&server), || {
            panic!("deterministic worker panic");
        });
        assert!(
            matches!(out, super::RaceOutcome::WorkerDied),
            "a panicking worker must race to WorkerDied, never TimedOut"
        );
    }

    /// FIX 1 (Issue #3368): end-to-end, a panicking query worker yields a
    /// non-retriable INTERNAL error (not a retriable RESOURCE_EXHAUSTED
    /// timeout) and does NOT bump the wall-clock-timeout counter.
    #[test]
    fn query_panicking_worker_yields_internal_not_timeout() {
        let server = make_server();
        assert_eq!(server.limit_termination_counts().wall_clock_timeout, 0);
        // A generous 60 s deadline; the worker panics immediately, so any
        // timeout classification would be a bug, not a race artifact.
        let effective = super::EffectiveQueryLimits {
            timeout_ms: 60_000,
            max_result_rows: 0,
            max_response_bytes: 0,
            max_query_memory_bytes: 0,
        };
        let out = server.race_deadline(effective.timeout_ms, test_guard(&server), || {
            panic!("boom");
        });
        let result = match out {
            super::RaceOutcome::WorkerDied => server.worker_died_error("aql"),
            super::RaceOutcome::TimedOut => panic!("panic must not be classified as a timeout"),
            super::RaceOutcome::Completed(_) => panic!("panicking worker cannot complete"),
        };
        assert_eq!(result.is_error, Some(true));
        let val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(result)).unwrap();
        let err = &val["error"];
        assert_eq!(err["kind"].as_str(), Some("runtime_error"));
        assert_eq!(err["code"].as_str(), Some("INTERNAL"));
        assert_eq!(err["retriable"].as_bool(), Some(false));
        // The panic must NOT have been counted as a wall-clock timeout.
        assert_eq!(server.limit_termination_counts().wall_clock_timeout, 0);
    }

    /// FIX 2 (Issue #3368): at the in-flight cap, an extra timed query is
    /// rejected `UNAVAILABLE` (retriable) instead of spawning another worker;
    /// once the occupying workers finish the slots are released and a
    /// subsequent timed query succeeds again.
    #[test]
    fn in_flight_cap_rejects_then_releases() {
        use crate::core::property::PropertyMap;
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        db.create_node("Widget", PropertyMap::new()).unwrap();
        // Cap of 1 live worker, with a generous timeout so the guard (not the
        // deadline) is what rejects.
        let server = AletheiaMcpServer::new(db).with_query_limits(super::QueryLimitsConfig {
            max_in_flight_queries: 1,
            default_timeout_ms: 60_000,
            max_timeout_ms: 0,
            ..super::QueryLimitsConfig::default()
        });

        // Manually occupy the only slot to simulate a still-running worker.
        let held = server
            .try_acquire_in_flight(server.query_limits.max_in_flight_queries)
            .expect("first slot acquires");

        // A timed query now finds the pool full and is rejected UNAVAILABLE.
        let rejected = server.handle_query(serde_json::json!({
            "language": "aql",
            "query": "MATCH (n:Widget) RETURN n",
        }));
        assert_eq!(rejected.is_error, Some(true));
        let val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(rejected)).unwrap();
        assert_eq!(val["error"]["code"].as_str(), Some("UNAVAILABLE"), "{val}");
        assert_eq!(val["error"]["retriable"].as_bool(), Some(true), "{val}");
        assert_eq!(
            val["error"]["details"]["max_in_flight_queries"].as_u64(),
            Some(1),
            "{val}"
        );

        // Release the held slot; the next timed query now admits and succeeds.
        drop(held);
        let ok = server.handle_query(serde_json::json!({
            "language": "aql",
            "query": "MATCH (n:Widget) RETURN n",
        }));
        let val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(ok)).unwrap();
        assert!(val.get("error").is_none(), "slot released, query ok: {val}");
        assert_eq!(val["row_count"].as_u64(), Some(1));
        // No worker is left occupying a slot after completion. The worker
        // releases its slot when its closure ends (the `InFlightGuard` drops),
        // which is ordered strictly *after* it sends the result the caller
        // already received above — so under parallel test load the guard-drop
        // can lag the received result by a scheduling quantum. Wait
        // deterministically (bounded) for the count to fall back to 0 rather
        // than racing the worker with an immediate assert; a real slot leak
        // still fails the test when the deadline elapses.
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(5);
        while server.in_flight_queries.load(super::Ordering::Acquire) != 0 {
            if start.elapsed() >= timeout {
                panic!("workers must release their slot on completion (still occupied after 5s)");
            }
            std::thread::yield_now();
        }
    }

    /// FIX 2 (Issue #3368): the inline `timeout_ms == 0` path never spawns a
    /// worker and is therefore unaffected by the in-flight guard — even with
    /// the pool artificially saturated, an unlimited-timeout query runs inline
    /// and succeeds.
    #[test]
    fn in_flight_cap_does_not_affect_inline_zero_timeout_path() {
        use crate::core::property::PropertyMap;
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        db.create_node("Widget", PropertyMap::new()).unwrap();
        // Unlimited timeout (default_timeout_ms 0) + cap 1, and saturate it.
        let server = AletheiaMcpServer::new(db).with_query_limits(super::QueryLimitsConfig {
            max_in_flight_queries: 1,
            default_timeout_ms: 0,
            max_timeout_ms: 0,
            ..super::QueryLimitsConfig::default()
        });
        let _held = server
            .try_acquire_in_flight(server.query_limits.max_in_flight_queries)
            .expect("first slot acquires");
        // Inline path: no guard consulted, no spawn, succeeds despite full pool.
        let ok = server.handle_query(serde_json::json!({
            "language": "aql",
            "query": "MATCH (n:Widget) RETURN n",
        }));
        let val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(ok)).unwrap();
        assert!(
            val.get("error").is_none(),
            "inline zero-timeout path is unguarded: {val}"
        );
        assert_eq!(val["row_count"].as_u64(), Some(1));
    }

    /// The wall-clock-timeout error is a retriable RESOURCE_EXHAUSTED with
    /// `details.dimension = wall_clock_timeout`, and it bumps the counter.
    #[test]
    fn wall_clock_timeout_error_shape_and_counter() {
        let server = make_server();
        assert_eq!(server.limit_termination_counts().wall_clock_timeout, 0);
        let result = server.wall_clock_timeout_error("aql", 25);
        assert_eq!(result.is_error, Some(true));
        let val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(result)).unwrap();
        let err = &val["error"];
        assert_eq!(err["kind"].as_str(), Some("runtime_error"));
        assert_eq!(err["code"].as_str(), Some("RESOURCE_EXHAUSTED"));
        assert_eq!(err["retriable"].as_bool(), Some(true));
        assert_eq!(
            err["details"]["dimension"].as_str(),
            Some("wall_clock_timeout")
        );
        assert_eq!(err["details"]["limit"].as_u64(), Some(25));
        assert_eq!(server.limit_termination_counts().wall_clock_timeout, 1);
    }

    /// A configured wall-clock timeout wraps real execution: `execute_query_core`
    /// runs on a raced thread and a tight deadline over a non-trivial dataset
    /// terminates with the structured timeout error. Uses a 1 ms timeout with a
    /// seeded dataset so the parse+execute+serialize pipeline reliably overruns.
    #[test]
    fn query_wall_clock_timeout_terminates_with_resource_exhausted() {
        use crate::core::property::PropertyMap;
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        for _ in 0..4_000 {
            db.create_node("Widget", PropertyMap::new()).unwrap();
        }
        let server = AletheiaMcpServer::new(db).with_query_limits(super::QueryLimitsConfig {
            default_timeout_ms: 1,
            max_timeout_ms: 0,
            ..super::QueryLimitsConfig::default()
        });
        let result = server.handle_query(serde_json::json!({
            "language": "aql",
            "query": "MATCH (n:Widget) RETURN n",
            "limit": 4_000,
        }));
        // With a 1 ms budget over 4k rows the query is terminated; assert the
        // structured timeout error (the discarded worker thread never wrote).
        assert_eq!(result.is_error, Some(true));
        let val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(result)).unwrap();
        assert_eq!(val["error"]["code"].as_str(), Some("RESOURCE_EXHAUSTED"));
        assert_eq!(
            val["error"]["details"]["dimension"].as_str(),
            Some("wall_clock_timeout")
        );
        assert_eq!(val["error"]["retriable"].as_bool(), Some(true));
        assert!(server.limit_termination_counts().wall_clock_timeout >= 1);
    }

    /// EXPLAIN (Issue #562) flows through the MCP `query` tool's
    /// computed-columns path unchanged: it is not a mutating clause, and its
    /// single `plan`-column row renders like any aggregate/computed row.
    #[cfg(feature = "cypher")]
    #[test]
    fn handle_query_explain_returns_plan_column() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        db.create_node("Person", crate::core::property::PropertyMap::new())
            .unwrap();
        let server = AletheiaMcpServer::new(db);

        let result = server.handle_query(serde_json::json!({
            "language": "cypher",
            "query": "EXPLAIN MATCH (n:Person) RETURN n",
        }));
        let text = AletheiaMcpServer::extract_text(result);
        let val: serde_json::Value = serde_json::from_str(&text).unwrap();

        assert!(
            val.get("error").is_none(),
            "EXPLAIN of a read query must not be rejected: {val}"
        );
        assert_eq!(val["row_count"], 1, "EXPLAIN yields one row: {val}");
        let rows = val["rows"].as_array().expect("rows array");
        let plan = rows[0]["plan"]
            .as_str()
            .expect("the row carries a `plan` string column");
        assert!(
            plan.contains("NodeScan"),
            "the plan text is surfaced through MCP: {plan}"
        );
        let columns = val["columns"].as_array().expect("columns array");
        assert!(
            columns.iter().any(|c| c["name"] == "plan"),
            "column metadata names the `plan` column: {val}"
        );
    }

    /// End-to-end (Issue #3354b): a Cypher provenance `WHERE` accessor filters
    /// rows through the MCP `query` tool exactly like the AQL side, using the
    /// shared executor evaluation. Alice (source `hr-system`) is kept; Bob
    /// (source `crm-sync`) is excluded.
    #[cfg(feature = "cypher")]
    #[test]
    fn handle_query_cypher_provenance_where_filters() {
        use crate::api::transaction::WriteRequestOptions;
        use crate::core::property::PropertyMapBuilder;
        use crate::core::provenance::Provenance;

        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let alice_prov = Provenance::builder()
            .source("hr-system")
            .build()
            .expect("prov");
        db.create_node_with_options(
            "Person",
            PropertyMapBuilder::new().insert("name", "alice").build(),
            WriteRequestOptions::new().with_provenance(alice_prov),
        )
        .expect("create alice");
        let bob_prov = Provenance::builder()
            .source("crm-sync")
            .build()
            .expect("prov");
        db.create_node_with_options(
            "Person",
            PropertyMapBuilder::new().insert("name", "bob").build(),
            WriteRequestOptions::new().with_provenance(bob_prov),
        )
        .expect("create bob");
        let server = AletheiaMcpServer::new(db);

        let result = server.handle_query(serde_json::json!({
            "language": "cypher",
            "query": "MATCH (n:Person) WHERE source(n) = 'hr-system' RETURN n",
        }));
        let val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(result)).unwrap();
        assert!(
            val.get("error").is_none(),
            "provenance WHERE must not error: {val}"
        );
        assert_eq!(val["row_count"], 1, "only alice matches: {val}");
        let rows = val["rows"].as_array().expect("rows array");
        assert_eq!(
            rows[0]["entity"]["properties"]["name"], "alice",
            "the kept row is alice: {val}"
        );
    }

    /// A Cypher provenance accessor compared to the wrong type surfaces as the
    /// structured `invalid_params` kind (never a silent empty result), mirroring
    /// the AQL side (Issue #3354b).
    #[cfg(feature = "cypher")]
    #[test]
    fn handle_query_cypher_provenance_type_mismatch_is_invalid_params() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let server = AletheiaMcpServer::new(db);
        let result = server.handle_query(serde_json::json!({
            "language": "cypher",
            "query": "MATCH (n:Person) WHERE confidence(n) = 'high' RETURN n",
        }));
        let val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(result)).unwrap();
        assert_eq!(
            val["error"]["kind"].as_str(),
            Some("invalid_params"),
            "type mismatch is a caller-repairable argument fault: {val}"
        );
    }

    /// The ordering-op-on-string rejection (fail-closed) surfaces as a
    /// structured `parse_error` through the MCP `query` tool (Issue #3354b).
    #[cfg(feature = "cypher")]
    #[test]
    fn handle_query_cypher_provenance_ordering_op_rejected() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let server = AletheiaMcpServer::new(db);
        let result = server.handle_query(serde_json::json!({
            "language": "cypher",
            "query": "MATCH (n:Person) WHERE source(n) < 'x' RETURN n",
        }));
        let val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(result)).unwrap();
        assert_eq!(
            val["error"]["kind"].as_str(),
            Some("parse_error"),
            "ordering op on a string accessor is rejected: {val}"
        );
    }

    /// Read-only invariant (Issue #3354b): a provenance predicate cannot smuggle
    /// a write. A pure provenance-`WHERE` read is accepted, but the same query
    /// with an appended mutating clause is rejected `read_only_violation`
    /// *before* execution by the existing statement guard.
    #[cfg(feature = "cypher")]
    #[test]
    fn handle_query_cypher_provenance_stays_read_only() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let server = AletheiaMcpServer::new(db);

        // Pure provenance read: not a mutating clause, accepted.
        let ok = server.handle_query(serde_json::json!({
            "language": "cypher",
            "query": "MATCH (n:Person) WHERE provenance(n) IS NOT NULL RETURN n",
        }));
        let ok_val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(ok)).unwrap();
        assert!(
            ok_val.get("error").is_none(),
            "a read-only provenance query must not be rejected: {ok_val}"
        );

        // Provenance WHERE with an appended write is rejected before execution.
        let bad = server.handle_query(serde_json::json!({
            "language": "cypher",
            "query": "MATCH (n:Person) WHERE source(n) = 'hr-system' DELETE n",
        }));
        let bad_val: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(bad)).unwrap();
        assert_eq!(
            bad_val["error"]["kind"].as_str(),
            Some("read_only_violation"),
            "a smuggled write is rejected read-only: {bad_val}"
        );
    }

    #[test]
    fn map_query_error_unsupported_feature_yields_unsupported_construct() {
        let server = make_server();
        let err = Error::Query(QueryError::UnsupportedFeature {
            feature: "DISTINCT".to_string(),
        });
        assert_eq!(error_kind(&server, err), "unsupported_construct");
    }

    #[test]
    fn map_query_error_scope_conflict_yields_invalid_request() {
        use crate::core::namespace::NamespaceError;
        let server = make_server();
        let payload = error_payload(&server, Error::Namespace(NamespaceError::ScopeConflict));
        assert_eq!(payload["kind"].as_str(), Some("invalid_request"));
        assert_eq!(payload["code"].as_str(), Some("INVALID_ARGUMENT"));
        assert_eq!(payload["retriable"].as_bool(), Some(false));
        // Must not disclose namespace contents.
        assert!(
            payload["details"].is_null(),
            "no namespace details: {payload}"
        );
    }

    #[test]
    fn map_query_error_invalid_parameter_yields_invalid_params() {
        let server = make_server();
        let err = Error::Query(QueryError::InvalidParameter {
            parameter: "p".to_string(),
            reason: "out of range".to_string(),
        });
        assert_eq!(error_kind(&server, err), "invalid_params");
    }

    #[test]
    fn map_query_error_execution_error_yields_runtime_error() {
        let server = make_server();
        let err = Error::Query(QueryError::ExecutionError {
            message: "boom".to_string(),
        });
        assert_eq!(error_kind(&server, err), "runtime_error");
    }

    #[test]
    fn map_query_error_type_mismatch_yields_invalid_params() {
        // A provenance accessor compared to the wrong type (e.g.
        // `confidence(r) = 'high'`, Issue #3354a) surfaces as `invalid_params`,
        // never a silent empty result (AC5).
        let server = make_server();
        let err = Error::Query(QueryError::TypeMismatch {
            expected: "number".to_string(),
            actual: "string".to_string(),
        });
        assert_eq!(error_kind(&server, err), "invalid_params");
    }

    #[test]
    fn map_query_error_other_variant_yields_runtime_error() {
        let server = make_server();
        // Error::Other is a variant not matched by any specific arm — falls through to `other`.
        let err = Error::Other("unexpected situation".to_string());
        assert_eq!(error_kind(&server, err), "runtime_error");
    }

    #[test]
    fn map_query_error_timeout_yields_retriable_unavailable_runtime_error() {
        // A timeout keeps the query tool's own `kind` contract
        // ("runtime_error") but is classified UNAVAILABLE/retriable from the
        // underlying engine error — and `retriable: true` must survive the
        // query-tool serialization path, not just the in-memory struct.
        let server = make_server();
        let error = error_payload(
            &server,
            Error::Query(QueryError::Timeout { duration_ms: 5000 }),
        );
        assert_eq!(error["kind"], "runtime_error", "got: {error}");
        assert_eq!(error["code"], "UNAVAILABLE", "got: {error}");
        assert_eq!(error["retriable"], true, "got: {error}");
    }

    /// A `QueryError::ResourceExhausted` for the memory dimension maps to a
    /// non-retriable `RESOURCE_EXHAUSTED` `runtime_error` carrying
    /// `details.{dimension, limit, consumed}` (Issue #3368 cooperative
    /// enforcement).
    #[test]
    fn map_query_error_resource_exhausted_memory_yields_resource_exhausted_non_retriable() {
        let server = make_server();
        let payload = error_payload(
            &server,
            Error::Query(QueryError::ResourceExhausted {
                dimension: "memory_bytes",
                limit: 1024,
                consumed: 2048,
                retriable: false,
            }),
        );
        assert_eq!(payload["kind"].as_str(), Some("runtime_error"), "{payload}");
        assert_eq!(
            payload["code"].as_str(),
            Some("RESOURCE_EXHAUSTED"),
            "{payload}"
        );
        assert_eq!(payload["retriable"].as_bool(), Some(false), "{payload}");
        assert_eq!(
            payload["details"]["dimension"].as_str(),
            Some("memory_bytes"),
            "{payload}"
        );
        assert_eq!(
            payload["details"]["limit"].as_u64(),
            Some(1024),
            "{payload}"
        );
        assert_eq!(
            payload["details"]["consumed"].as_u64(),
            Some(2048),
            "{payload}"
        );
    }

    /// The wall-clock-timeout dimension of the same engine error is retriable
    /// (a read-only re-run is safe), unlike the memory dimension above (Issue
    /// #3368).
    #[test]
    fn map_query_error_resource_exhausted_timeout_yields_retriable() {
        let server = make_server();
        let payload = error_payload(
            &server,
            Error::Query(QueryError::ResourceExhausted {
                dimension: "wall_clock_timeout",
                limit: 500,
                consumed: 501,
                retriable: true,
            }),
        );
        assert_eq!(
            payload["code"].as_str(),
            Some("RESOURCE_EXHAUSTED"),
            "{payload}"
        );
        assert_eq!(payload["retriable"].as_bool(), Some(true), "{payload}");
        assert_eq!(
            payload["details"]["dimension"].as_str(),
            Some("wall_clock_timeout"),
            "{payload}"
        );
    }

    #[test]
    fn query_row_to_json_node_id_variant() {
        let server = make_server();
        let row = QueryRow::from_entity(EntityResult::NodeId(NodeId::new(42).unwrap()));
        let json = server.query_row_to_json(row);
        assert_eq!(json["entity"]["type"].as_str(), Some("node"));
        assert_eq!(json["entity"]["id"].as_u64(), Some(42));
    }

    #[test]
    fn query_row_to_json_edge_id_variant() {
        let server = make_server();
        let row = QueryRow::from_entity(EntityResult::EdgeId(EdgeId::new(99).unwrap()));
        let json = server.query_row_to_json(row);
        assert_eq!(json["entity"]["type"].as_str(), Some("edge"));
        assert_eq!(json["entity"]["id"].as_u64(), Some(99));
    }

    // --- #558 MCP surface: computed/aggregate rows render their named columns.
    // These are deliberately feature-independent (no `#[cfg(feature = "cypher")]`)
    // so the coverage job -- which compiles without `cypher` -- still exercises
    // the aggregate branches of `query_row_to_json`, `handle_query`'s column
    // selection, and `computed_query_columns`. `QueryRow::from_columns` builds
    // exactly the `entity: Null, columns: Some(..)` shape those branches key on.

    #[test]
    fn query_row_to_json_renders_computed_columns() {
        let server = make_server();
        let row = QueryRow::from_columns(vec![("count(*)".to_string(), PropertyValue::Int(5))]);
        let json = server.query_row_to_json(row);
        // The aggregate value is surfaced under its column name, not lost.
        assert_eq!(json["count(*)"].as_i64(), Some(5), "got: {json}");
        // Computed rows use the bare column-map shape: no entity/score/path/timestamp keys.
        let obj = json
            .as_object()
            .expect("computed row must be a JSON object");
        assert!(!obj.contains_key("entity"), "no entity key: {json}");
        assert!(!obj.contains_key("score"), "no score key: {json}");
        assert!(!obj.contains_key("path"), "no path key: {json}");
        assert!(!obj.contains_key("timestamp"), "no timestamp key: {json}");
    }

    #[test]
    fn query_row_to_json_renders_multi_column_computed_row() {
        let server = make_server();
        let row = QueryRow::from_columns(vec![
            ("n.dept".to_string(), PropertyValue::String("Eng".into())),
            ("c".to_string(), PropertyValue::Int(3)),
        ]);
        let json = server.query_row_to_json(row);
        assert_eq!(json["n.dept"].as_str(), Some("Eng"), "got: {json}");
        assert_eq!(json["c"].as_i64(), Some(3), "got: {json}");
    }

    #[test]
    fn computed_query_columns_names_the_columns() {
        let names = vec!["count(*)".to_string(), "c".to_string()];
        let cols = super::computed_query_columns(&names);
        let arr = cols.as_array().expect("columns must be a JSON array");
        assert_eq!(arr.len(), 2, "one entry per column, in order: {cols}");
        assert_eq!(arr[0]["name"].as_str(), Some("count(*)"), "got: {cols}");
        assert_eq!(arr[1]["name"].as_str(), Some("c"), "got: {cols}");
        // Each entry carries the response schema shape (name/type/description).
        assert!(arr[0].get("type").is_some(), "type present: {cols}");
        assert!(
            arr[0].get("description").is_some(),
            "description present: {cols}"
        );
    }

    #[test]
    fn handle_enable_unique_constraint_invalid_json_returns_error() {
        // Covers the `Err(e) => return self.invalid_argument(...)` parse-error arm of
        // handle_enable_unique_constraint (added for Issue #3218).  The public
        // `enable_unique_constraint(req)` API always serialises a valid struct,
        // so this arm is only reachable via the internal handle_ function.
        let server = make_server();
        let result = server.handle_enable_unique_constraint(serde_json::Value::Null);
        // Must be an error CallToolResult (is_error = Some(true))
        assert!(
            result.is_error.unwrap_or(false),
            "Null JSON input must produce an error result"
        );
    }

    #[test]
    fn timestamp_to_rfc3339_out_of_chrono_range_falls_back_to_micros() {
        // Coordinates outside chrono's representable range must render as
        // the raw-microseconds fallback instead of panicking. Timestamps up
        // to MAX_VALID_TIMESTAMP (i64::MAX - 1000 µs) are storable but far
        // beyond chrono's ~year-262143 ceiling.
        let ts = crate::core::temporal::Timestamp::from(i64::MAX - 1000);
        let rendered = AletheiaMcpServer::timestamp_to_rfc3339(ts);
        assert_eq!(rendered, format!("{}us", i64::MAX - 1000));

        // Sanity: an in-range coordinate renders as RFC3339, not the fallback.
        let ts = crate::core::temporal::Timestamp::from(1_614_556_800_000_000); // 2021-03-01
        let rendered = AletheiaMcpServer::timestamp_to_rfc3339(ts);
        assert_eq!(rendered, "2021-03-01T00:00:00.000000Z");
    }

    #[test]
    fn handle_temporal_extent_invalid_by_label_type_routes_through_invalid_argument() {
        // A mistyped argument (string instead of bool) must produce the
        // structured Issue #3234 error payload with code INVALID_ARGUMENT
        // (Issue #3238).
        let server = make_server();
        let result = server.handle_temporal_extent(serde_json::json!({"by_label": "yes"}));
        assert!(
            result.is_error.unwrap_or(false),
            "mistyped by_label must produce an error result"
        );
        let text = AletheiaMcpServer::extract_text(result);
        let val: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            val["error"]["code"], "INVALID_ARGUMENT",
            "mistyped by_label must classify as INVALID_ARGUMENT: {val}"
        );
        assert_eq!(
            val["error"]["retriable"], false,
            "INVALID_ARGUMENT must not be retriable: {val}"
        );
        assert!(
            val["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("Invalid arguments")),
            "message must preserve the free-text detail: {val}"
        );
    }

    #[test]
    fn handle_temporal_extent_null_args_behaves_like_no_arguments() {
        // The tool has no required arguments: an MCP call with the
        // `arguments` object omitted entirely (routed here as Null) must
        // succeed exactly like `{}`.
        let server = make_server();
        let result = server.handle_temporal_extent(serde_json::Value::Null);
        assert!(
            !result.is_error.unwrap_or(false),
            "temporal_extent with no arguments must succeed"
        );
        let text = AletheiaMcpServer::extract_text(result);
        let val: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(val["valid_time"]["earliest"].is_null());
        assert!(val["transaction_time"]["latest"].is_null());
    }

    #[test]
    fn query_row_to_json_null_binding_serializes_entity_as_json_null() {
        // A null binding from an unmatched OPTIONAL MATCH pattern must
        // surface as an explicit JSON null entity (row preserved).
        let server = make_server();
        let value = server.query_row_to_json(QueryRow::from_entity(EntityResult::Null));
        assert!(
            value["entity"].is_null(),
            "null binding must serialize as JSON null: {value}"
        );
        assert!(value["score"].is_null());
        assert!(value["path"].is_null());
    }

    // --- #549 MCP surface: multi-variable binding rows serialize each bound
    // entity under its variable name (never a lossy `entity: null`).

    #[test]
    fn query_row_to_json_bindings_serializes_each_entity() {
        use crate::core::graph::Node;
        use crate::core::id::VersionId;
        use crate::core::{GLOBAL_INTERNER, PropertyMapBuilder};

        let server = make_server();
        let mk = |id: u64, name: &str| {
            let label = GLOBAL_INTERNER.intern("Person").unwrap();
            Node::new(
                NodeId::new(id).unwrap(),
                label,
                PropertyMapBuilder::new().insert("name", name).build(),
                VersionId::new(1).unwrap(),
            )
        };
        let row = QueryRow::from_bindings(
            vec![
                ("a".to_string(), EntityResult::Node(mk(1, "Alice"))),
                ("b".to_string(), EntityResult::Node(mk(2, "Bob"))),
            ],
            None,
        );
        let json = server.query_row_to_json(row);
        let obj = json
            .as_object()
            .expect("bindings row must be a JSON object");
        // Both variables surface as node objects, NOT a lossy null.
        assert_eq!(obj["a"]["type"].as_str(), Some("node"), "got: {json}");
        assert_eq!(obj["a"]["id"].as_u64(), Some(1), "got: {json}");
        assert_eq!(
            obj["a"]["properties"]["name"].as_str(),
            Some("Alice"),
            "got: {json}"
        );
        assert_eq!(obj["b"]["type"].as_str(), Some("node"), "got: {json}");
        assert_eq!(obj["b"]["id"].as_u64(), Some(2), "got: {json}");
    }

    #[test]
    fn query_row_to_json_bindings_merged_with_columns() {
        use crate::core::graph::Node;
        use crate::core::id::VersionId;
        use crate::core::{GLOBAL_INTERNER, PropertyMapBuilder};

        let server = make_server();
        let label = GLOBAL_INTERNER.intern("Person").unwrap();
        let node = Node::new(
            NodeId::new(7).unwrap(),
            label,
            PropertyMapBuilder::new().insert("name", "Alice").build(),
            VersionId::new(1).unwrap(),
        );
        let row = QueryRow::from_bindings(
            vec![("a".to_string(), EntityResult::Node(node))],
            Some(vec![(
                "a.name".to_string(),
                PropertyValue::String("Alice".into()),
            )]),
        );
        let json = server.query_row_to_json(row);
        // The bound entity and the scalar column are both present, merged.
        assert_eq!(json["a"]["type"].as_str(), Some("node"), "got: {json}");
        assert_eq!(json["a.name"].as_str(), Some("Alice"), "got: {json}");
    }

    #[cfg(feature = "cypher")]
    #[test]
    #[ignore = "Multi-pattern matching is currently disabled in v1 due to namespace scoping constraints"]
    fn handle_query_multi_pattern_returns_non_null_bindings() {
        use crate::core::PropertyMapBuilder;
        let server = make_server();
        server
            .db
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
            )
            .unwrap();
        server
            .db
            .create_node(
                "Company",
                PropertyMapBuilder::new().insert("name", "Acme").build(),
            )
            .unwrap();
        // A multi-variable-pattern MATCH rejects any *restricting* namespace
        // scope (v1, src/db/query.rs), and the MCP `query` tool defaults an
        // omitted `namespace` to `default`-only (fail-closed / isolated-by-
        // default, #3349 / PR3d #3731). This test must therefore pass
        // `"namespace": "all"` to run the cartesian product unscoped.
        let result = server.handle_query(serde_json::json!({
            "language": "cypher",
            "query": "MATCH (a:Person),(b:Company) RETURN a,b",
            "namespace": "all"
        }));
        let text = AletheiaMcpServer::extract_text(result);
        let val: serde_json::Value = serde_json::from_str(&text).unwrap();
        let rows = val["rows"].as_array().expect("rows array");
        assert_eq!(rows.len(), 1, "got: {val}");
        assert_eq!(rows[0]["a"]["type"].as_str(), Some("node"), "got: {val}");
        assert_eq!(rows[0]["b"]["type"].as_str(), Some("node"), "got: {val}");
        // The response columns are derived dynamically from the binding names.
        let cols: Vec<String> = val["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["name"].as_str().unwrap().to_string())
            .collect();
        assert!(
            cols.contains(&"a".to_string()) && cols.contains(&"b".to_string()),
            "columns must name the bound variables: {val}"
        );
    }

    /// Fail-closed guard (companion to
    /// `handle_query_multi_pattern_returns_non_null_bindings`): the SAME
    /// multi-variable-pattern MATCH with NO `namespace` arg must be rejected,
    /// not silently unscoped. An omitted `namespace` defaults to `default`-only
    /// (#3349 / PR3d #3731), and a multi-variable-pattern MATCH rejects any
    /// restricting scope (v1, src/db/query.rs), so the tool returns a
    /// structured `unsupported_construct` / `INVALID_ARGUMENT` error. This pins
    /// the fail-closed semantics so a future change can't silently unscope it.
    #[cfg(feature = "cypher")]
    #[test]
    fn handle_query_multi_pattern_omitted_scope_is_rejected() {
        use crate::core::PropertyMapBuilder;
        let server = make_server();
        server
            .db
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
            )
            .unwrap();
        server
            .db
            .create_node(
                "Company",
                PropertyMapBuilder::new().insert("name", "Acme").build(),
            )
            .unwrap();
        // No `namespace` arg -> defaults to `default`-only (fail-closed).
        let result = server.handle_query(serde_json::json!({
            "language": "cypher",
            "query": "MATCH (a:Person),(b:Company) RETURN a,b"
        }));
        let text = AletheiaMcpServer::extract_text(result);
        let val: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            val["error"]["code"].as_str(),
            Some("INVALID_ARGUMENT"),
            "got: {val}"
        );
        assert_eq!(
            val["error"]["kind"].as_str(),
            Some("unsupported_construct"),
            "got: {val}"
        );
    }

    // ===================================================================
    // Per-query resource limits extended to the read tools (Issue #3368
    // residue): wall-clock timeout + result-byte cap on
    // traverse/hybrid_query/find_similar/get_node_at_time/get_edge_at_time/
    // find_nodes_at_time, plus the `database_stats.resource_limits` counters.
    // These drive the full `dispatch_tool` seam (auth -> #3368 wrapper ->
    // #3353 budget), so they exercise the wrapper end to end.
    // ===================================================================

    use crate::core::property::PropertyMap;

    /// Seed a star graph: one root with `leaves` outgoing `LINK` edges to
    /// leaf nodes. Returns the root's id. A wide star makes a single-hop
    /// traverse do real (multi-node) work, so a 1 ms wall-clock timeout races
    /// against a computation that reliably overruns it.
    fn seed_star(db: &Arc<AletheiaDB>, leaves: usize) -> u64 {
        let root = db.create_node("Root", PropertyMap::new()).expect("root");
        for _ in 0..leaves {
            let leaf = db.create_node("Leaf", PropertyMap::new()).expect("leaf");
            db.create_edge(root, leaf, "LINK", PropertyMap::new())
                .expect("edge");
        }
        root.as_u64()
    }

    fn parse(result: CallToolResult) -> serde_json::Value {
        serde_json::from_str(&AletheiaMcpServer::extract_text(result)).expect("json response")
    }

    /// The wrapper covers the residue read tools plus the enrolled #2907
    /// semantic-search analysis tools, and never the self-enforcing `query`
    /// tool.
    #[test]
    fn resource_limited_read_tools_membership() {
        for t in [
            "traverse",
            "hybrid_query",
            "find_similar",
            "get_node_at_time",
            "get_edge_at_time",
            "find_nodes_at_time",
            // Issue #2907 semantic-search tools enrolled for uniform coverage.
            "semantic_path",
            "concept_analogy",
            "concept_mean",
            "find_duplicate_candidates",
            "semantic_horizon",
            "context_aspects",
        ] {
            assert!(
                super::is_resource_limited_read_tool(t),
                "{t} must be covered"
            );
        }
        // `query` self-enforces inline; it must NOT be double-wrapped.
        assert!(!super::is_resource_limited_read_tool("query"));
        assert!(!super::is_resource_limited_read_tool("get_node"));
        assert!(!super::is_resource_limited_read_tool("database_stats"));
    }

    /// A pathological `traverse` under a 1 ms wall-clock timeout is terminated
    /// with a retriable RESOURCE_EXHAUSTED whose `details.dimension` is
    /// `wall_clock_timeout`, and the timeout counter is bumped exactly once.
    #[test]
    fn traverse_wall_clock_timeout_is_resource_exhausted() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        // Enough leaves that a single-hop traverse + serialization dwarfs 1 ms.
        let root = seed_star(&db, 3_000);
        let server = AletheiaMcpServer::new(db).with_query_limits(super::QueryLimitsConfig {
            default_timeout_ms: 1,
            ..super::QueryLimitsConfig::default()
        });
        assert_eq!(server.limit_termination_counts().wall_clock_timeout, 0);

        let result = server.dispatch_tool(
            "traverse",
            serde_json::json!({
                "start_node_id": root,
                "edge_label": "LINK",
                "depth": 1,
                "limit": 10_000,
            }),
        );
        assert_eq!(result.is_error, Some(true), "expected a timeout error");
        let val = parse(result);
        let err = &val["error"];
        assert_eq!(err["code"].as_str(), Some("RESOURCE_EXHAUSTED"), "{val}");
        assert_eq!(err["retriable"].as_bool(), Some(true), "{val}");
        assert_eq!(
            err["details"]["dimension"].as_str(),
            Some("wall_clock_timeout"),
            "{val}"
        );
        // Tool-agnostic envelope (Issue #3368 residue): NO query-tool `kind` /
        // `language` fields (a wrapped read tool is not a query language), and
        // no unactionable `limits.timeout_ms` remediation (these tools have no
        // per-call override).
        assert!(err.get("kind").is_none(), "no kind field: {val}");
        assert!(err.get("language").is_none(), "no language field: {val}");
        let msg = err["message"].as_str().unwrap_or_default();
        assert!(
            !msg.contains("limits.timeout_ms"),
            "message must not reference limits.timeout_ms: {val}"
        );
        assert!(
            msg.contains("traverse"),
            "message must name the tool: {val}"
        );
        assert_eq!(server.limit_termination_counts().wall_clock_timeout, 1);
    }

    /// A `traverse` response that exceeds the result-byte cap fails closed with
    /// a non-retriable RESOURCE_EXHAUSTED / `result_bytes` error whose
    /// `details.consumed` exceeds the cap, bumping the byte-cap counter. Uses
    /// the inline (unlimited-timeout) path so only the post-hoc byte cap is
    /// under test.
    #[test]
    fn traverse_result_byte_cap_fails_closed() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let root = seed_star(&db, 5);
        let server = AletheiaMcpServer::new(db).with_query_limits(super::QueryLimitsConfig {
            default_timeout_ms: 0,          // inline path (no timeout race)
            default_max_response_bytes: 64, // tiny cap the response exceeds
            max_response_bytes: 0,          // no ceiling to interfere
            ..super::QueryLimitsConfig::default()
        });
        assert_eq!(server.limit_termination_counts().result_bytes, 0);

        let result = server.dispatch_tool(
            "traverse",
            serde_json::json!({
                "start_node_id": root,
                "edge_label": "LINK",
                "depth": 1,
                "limit": 100,
            }),
        );
        assert_eq!(result.is_error, Some(true), "expected a byte-cap error");
        let val = parse(result);
        let err = &val["error"];
        assert_eq!(err["code"].as_str(), Some("RESOURCE_EXHAUSTED"), "{val}");
        assert_eq!(err["retriable"].as_bool(), Some(false), "{val}");
        assert_eq!(
            err["details"]["dimension"].as_str(),
            Some("result_bytes"),
            "{val}"
        );
        assert_eq!(err["details"]["limit"].as_u64(), Some(64), "{val}");
        assert!(
            err["details"]["consumed"].as_u64().unwrap_or(0) > 64,
            "consumed must exceed the cap: {val}"
        );
        // Tool-agnostic envelope (Issue #3368 residue): NO query-tool `kind` /
        // `language` fields, and no unactionable `limits.max_response_bytes`
        // remediation.
        assert!(err.get("kind").is_none(), "no kind field: {val}");
        assert!(err.get("language").is_none(), "no language field: {val}");
        let msg = err["message"].as_str().unwrap_or_default();
        assert!(
            !msg.contains("limits.max_response_bytes"),
            "message must not reference limits.max_response_bytes: {val}"
        );
        assert!(
            msg.contains("traverse"),
            "message must name the tool: {val}"
        );
        assert_eq!(server.limit_termination_counts().result_bytes, 1);
    }

    /// A large-but-within-limits result is a success that self-discloses
    /// incompleteness via `has_more`/`next_offset` (row-cap breaches are the
    /// tool's own `limit`, NOT a resource-limit termination): no counter moves.
    #[test]
    fn row_limited_result_is_success_and_uncounted() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let root = seed_star(&db, 5);
        // Default limits (generous timeout + byte cap) over the seeded db.
        let server = AletheiaMcpServer::new(db);

        let result = server.dispatch_tool(
            "traverse",
            serde_json::json!({
                "start_node_id": root,
                "edge_label": "LINK",
                "depth": 1,
                "limit": 1, // fewer than the 5 leaves -> a partial page
            }),
        );
        assert_ne!(result.is_error, Some(true), "row limit is not an error");
        let val = parse(result);
        assert_eq!(val["count"].as_u64(), Some(1), "{val}");
        assert_eq!(val["has_more"].as_bool(), Some(true), "{val}");
        assert!(val.get("next_offset").is_some(), "{val}");
        let counts = server.limit_termination_counts();
        assert_eq!(counts.wall_clock_timeout, 0);
        assert_eq!(counts.result_bytes, 0);
    }

    /// `database_stats` surfaces the over-limit termination counters under an
    /// additive `resource_limits` block: zero on a fresh server, and reflecting
    /// forced terminations afterwards.
    #[test]
    fn database_stats_surfaces_resource_limit_counters() {
        let server = make_server();

        // Fresh: block present, all zero.
        let val = parse(server.dispatch_tool("database_stats", serde_json::json!({})));
        let rl = &val["resource_limits"];
        assert_eq!(rl["timeout_terminations"].as_u64(), Some(0), "{val}");
        assert_eq!(rl["byte_cap_terminations"].as_u64(), Some(0), "{val}");
        assert_eq!(rl["memory_terminations"].as_u64(), Some(0), "{val}");
        assert_eq!(rl["override_rejections"].as_u64(), Some(0), "{val}");

        // Force one of each termination through the shared counters (the exact
        // tool-agnostic builders the wrapper invokes on TimedOut / over-cap).
        let _ = server.read_tool_timeout_error("traverse", 1);
        let _ = server.read_tool_byte_cap_error("traverse", 4096, 64);
        let _ = server.read_tool_memory_error("traverse", 4096, 64);

        let val = parse(server.dispatch_tool("database_stats", serde_json::json!({})));
        let rl = &val["resource_limits"];
        assert_eq!(rl["timeout_terminations"].as_u64(), Some(1), "{val}");
        assert_eq!(rl["byte_cap_terminations"].as_u64(), Some(1), "{val}");
        assert_eq!(rl["memory_terminations"].as_u64(), Some(1), "{val}");
    }

    /// A `traverse` response whose estimated working memory (serialized length ×
    /// the documented expansion factor) exceeds a configured
    /// `default_max_query_memory_bytes` fails closed with a non-retriable
    /// RESOURCE_EXHAUSTED / `memory_bytes` error whose `details.consumed`
    /// exceeds the cap, bumping the memory-termination counter. Uses the inline
    /// (unlimited-timeout) path and no byte cap so only the post-hoc memory
    /// budget is under test.
    #[test]
    fn traverse_memory_budget_fails_closed() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let root = seed_star(&db, 5);
        let server = AletheiaMcpServer::new(db).with_query_limits(super::QueryLimitsConfig {
            default_timeout_ms: 0,              // inline path (no timeout race)
            default_max_response_bytes: 0,      // no byte cap to interfere
            max_response_bytes: 0,              // no ceiling to interfere
            default_max_query_memory_bytes: 64, // tiny memory budget
            max_query_memory_bytes: 0,          // no ceiling to interfere
            ..super::QueryLimitsConfig::default()
        });
        assert_eq!(server.limit_termination_counts().memory_bytes, 0);

        let result = server.dispatch_tool(
            "traverse",
            serde_json::json!({
                "start_node_id": root,
                "edge_label": "LINK",
                "depth": 1,
                "limit": 100,
            }),
        );
        assert_eq!(
            result.is_error,
            Some(true),
            "expected a memory-budget error"
        );
        let val = parse(result);
        let err = &val["error"];
        assert_eq!(err["code"].as_str(), Some("RESOURCE_EXHAUSTED"), "{val}");
        assert_eq!(err["retriable"].as_bool(), Some(false), "{val}");
        assert_eq!(
            err["details"]["dimension"].as_str(),
            Some("memory_bytes"),
            "{val}"
        );
        assert_eq!(err["details"]["limit"].as_u64(), Some(64), "{val}");
        assert!(
            err["details"]["consumed"].as_u64().unwrap_or(0) > 64,
            "estimated consumed must exceed the cap: {val}"
        );
        // Tool-agnostic envelope: NO query-tool `kind` / `language` fields.
        assert!(err.get("kind").is_none(), "no kind field: {val}");
        assert!(err.get("language").is_none(), "no language field: {val}");
        assert!(
            err["message"]
                .as_str()
                .unwrap_or_default()
                .contains("traverse"),
            "message must name the tool: {val}"
        );
        assert_eq!(server.limit_termination_counts().memory_bytes, 1);
    }

    /// Default config is memory-budget-OFF: a normal `traverse` succeeds and the
    /// memory-termination counter never moves (zero behavior change unless an
    /// operator opts in).
    #[test]
    fn default_config_leaves_memory_budget_off() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let root = seed_star(&db, 5);
        let server = AletheiaMcpServer::new(db); // default config

        let result = server.dispatch_tool(
            "traverse",
            serde_json::json!({
                "start_node_id": root,
                "edge_label": "LINK",
                "depth": 1,
                "limit": 100,
            }),
        );
        assert_ne!(
            result.is_error,
            Some(true),
            "default config must not engage the memory budget"
        );
        // Self-evidently a non-trivial response (the five LINK leaves are
        // present) — so "a response that would exceed a small cap" needs no
        // cross-reference to the sibling fails-closed test.
        let text = AletheiaMcpServer::extract_text(result);
        assert!(
            text.contains("Leaf"),
            "response must carry the traversed leaf nodes: {text}"
        );
        assert_eq!(server.limit_termination_counts().memory_bytes, 0);
    }

    /// Pins the ×4 expansion factor itself (the actual novel arithmetic).
    ///
    /// `enforce_memory_budget` is fed a synthetic success response of a *known*
    /// text length N with a cap set STRICTLY BETWEEN N and N×4, so the budget
    /// can only trip because of the expansion multiply: the raw serialized size
    /// (N) is within cap, but the estimated working set (N×4) is not. The
    /// assertions lock `consumed == N×4` and `limit == cap` to the literal
    /// values, so this test goes red if the factor is changed to anything other
    /// than 4 (e.g. 1 leaves the estimate within cap and no error is produced at
    /// all; 2 or 3 likewise fail the strict-between trip; any factor still emits
    /// a `consumed` that no longer equals 400).
    #[test]
    fn memory_budget_pins_expansion_factor() {
        // N = 100 raw bytes; cap = 250 is strictly between 100 and 100×4 = 400.
        const N: usize = 100;
        const CAP: usize = 250;
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let server = AletheiaMcpServer::new(db);

        let synthetic = CallToolResult::success(vec![Content::text("a".repeat(N))]);
        assert_eq!(server.limit_termination_counts().memory_bytes, 0);

        let result = server.enforce_memory_budget("traverse", synthetic, CAP);
        assert_eq!(
            result.is_error,
            Some(true),
            "N×4 must exceed the strictly-between cap"
        );
        let val = parse(result);
        let details = &val["error"]["details"];
        assert_eq!(details["dimension"].as_str(), Some("memory_bytes"), "{val}");
        // The load-bearing assertion: consumed is EXACTLY N×4, locking factor 4.
        assert_eq!(details["consumed"].as_u64(), Some(400), "{val}");
        assert_eq!(details["limit"].as_u64(), Some(250), "{val}");
        assert_eq!(server.limit_termination_counts().memory_bytes, 1);

        // Control: the same response under a cap >= N×4 passes through untouched
        // (proves the trip above is the multiply, not an always-on rejection).
        let ok = server.enforce_memory_budget(
            "traverse",
            CallToolResult::success(vec![Content::text("a".repeat(N))]),
            N * super::MEMORY_WORKING_SET_EXPANSION,
        );
        assert_ne!(ok.is_error, Some(true), "estimate == cap must not trip");
        assert_eq!(server.limit_termination_counts().memory_bytes, 1);
    }

    /// Ordering + no-double-count invariant: when a single response would trip
    /// BOTH the result-byte cap AND the memory budget, the byte cap runs first
    /// (its error response short-circuits `enforce_memory_budget` via the
    /// `is_error` guard), so the caller sees `dimension == "result_bytes"`, the
    /// byte-cap counter increments, and the memory counter stays 0 — never
    /// double-counted. Guards against a future reorder or guard-drop.
    #[test]
    fn byte_cap_precedes_memory_budget_no_double_count() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let root = seed_star(&db, 5);
        let server = AletheiaMcpServer::new(db).with_query_limits(super::QueryLimitsConfig {
            default_timeout_ms: 0,              // inline path (no timeout race)
            default_max_response_bytes: 64,     // byte cap trips (runs first)
            max_response_bytes: 0,              // no ceiling to interfere
            default_max_query_memory_bytes: 64, // memory budget would ALSO trip
            max_query_memory_bytes: 0,          // no ceiling to interfere
            ..super::QueryLimitsConfig::default()
        });
        assert_eq!(server.limit_termination_counts().result_bytes, 0);
        assert_eq!(server.limit_termination_counts().memory_bytes, 0);

        let result = server.dispatch_tool(
            "traverse",
            serde_json::json!({
                "start_node_id": root,
                "edge_label": "LINK",
                "depth": 1,
                "limit": 100,
            }),
        );
        assert_eq!(result.is_error, Some(true), "both caps would trip");
        let val = parse(result);
        assert_eq!(
            val["error"]["details"]["dimension"].as_str(),
            Some("result_bytes"),
            "byte cap runs first and wins: {val}"
        );
        // Exactly one termination recorded, on the byte-cap dimension only.
        assert_eq!(server.limit_termination_counts().result_bytes, 1, "{val}");
        assert_eq!(
            server.limit_termination_counts().memory_bytes,
            0,
            "memory budget must not double-count after the byte cap fired: {val}"
        );
    }

    /// Fast path: with limits disabled (unlimited timeout + no byte cap) the
    /// wrapper runs the handler inline and returns a response byte-for-byte
    /// identical to the unwrapped `dispatch_read_tool` — no behavior change.
    #[test]
    fn zero_timeout_fast_path_is_identical_to_unwrapped() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let root = seed_star(&db, 4);
        let server =
            AletheiaMcpServer::new(db).with_query_limits(super::QueryLimitsConfig::disabled());
        let args = serde_json::json!({
            "start_node_id": root,
            "edge_label": "LINK",
            "depth": 1,
            "limit": 100,
        });

        let wrapped =
            AletheiaMcpServer::extract_text(server.dispatch_tool("traverse", args.clone()));
        let bare = AletheiaMcpServer::extract_text(server.dispatch_read_tool("traverse", args));
        assert_eq!(
            wrapped, bare,
            "the disabled-limits fast path must not alter the response"
        );
    }

    // =====================================================================
    // `query` tool: cooperative engine-lane enforcement (Issue #3368 residue)
    //
    // Unlike the wrapped read tools above (post-hoc, response-size-based
    // `enforce_memory_budget`), the `query` tool threads its effective limits
    // straight into the executor's `ResourceGuardIterator` via the `_limited`
    // reconciled entry points, so a breach is caught mid-drain with real
    // per-row accounting and surfaces through `map_query_error`'s
    // `ResourceExhausted` arm (query-tool `kind`/`language` envelope, unlike
    // the neutral read-tool envelope).
    // =====================================================================

    /// A tiny operator-configured `default_max_query_memory_bytes` trips the
    /// engine-lane guard on the very first row of an otherwise-ordinary AQL
    /// `query` tool call, surfacing as a non-retriable `RESOURCE_EXHAUSTED`
    /// error whose `details.dimension == "memory_bytes"`. Uses the inline
    /// (`timeout_ms: 0`) path so only the memory dimension is under test.
    #[test]
    fn query_tool_engine_memory_budget_fails_closed() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let _root = seed_star(&db, 5);
        let server = AletheiaMcpServer::new(db).with_query_limits(super::QueryLimitsConfig {
            default_timeout_ms: 0,             // inline path (no timeout race)
            default_max_response_bytes: 0,     // no byte cap to interfere
            max_response_bytes: 0,             // no ceiling to interfere
            default_max_query_memory_bytes: 1, // tiny budget -> trips on row 1
            max_query_memory_bytes: 0,         // no ceiling to interfere
            ..super::QueryLimitsConfig::default()
        });

        let result = server.handle_query(serde_json::json!({
            "language": "aql",
            "query": "MATCH (n:Leaf) RETURN n",
        }));
        assert_eq!(
            result.is_error,
            Some(true),
            "expected an engine-lane memory-budget error"
        );
        let val = parse(result);
        let err = &val["error"];
        assert_eq!(err["kind"].as_str(), Some("runtime_error"), "{val}");
        assert_eq!(err["code"].as_str(), Some("RESOURCE_EXHAUSTED"), "{val}");
        assert_eq!(err["retriable"].as_bool(), Some(false), "{val}");
        assert_eq!(
            err["details"]["dimension"].as_str(),
            Some("memory_bytes"),
            "{val}"
        );
        assert_eq!(err["details"]["limit"].as_u64(), Some(1), "{val}");
        assert!(
            err["details"]["consumed"].as_u64().unwrap_or(0) > 1,
            "consumed must exceed the 1-byte cap: {val}"
        );
        assert_eq!(err["language"].as_str(), Some("aql"), "{val}");
    }

    /// Default config leaves the engine-lane memory budget off (`0` =
    /// unlimited): an ordinary `query` tool call over the same seeded graph
    /// succeeds and returns real rows, with zero behavior change from before
    /// this feature landed.
    #[test]
    fn query_tool_default_config_memory_budget_off_returns_normal_results() {
        let db = Arc::new(AletheiaDB::new().expect("db init"));
        let _root = seed_star(&db, 5);
        let server = AletheiaMcpServer::new(db); // default config

        let result = server.handle_query(serde_json::json!({
            "language": "aql",
            "query": "MATCH (n:Leaf) RETURN n",
        }));
        assert_ne!(
            result.is_error,
            Some(true),
            "default config must not engage the engine memory budget"
        );
        let val = parse(result);
        assert_eq!(val["row_count"].as_u64(), Some(5), "{val}");
        assert_eq!(val["truncated"].as_bool(), Some(false), "{val}");
    }

    // =====================================================================
    // In-query `USE / IN NAMESPACE` clause through the MCP/HTTP `query` tool
    // (Issue #3349 follow-up). The tool reconciles the `namespace` request
    // parameter (programmatic scope `P`) with an in-statement clause (`C`):
    //   (None,   None)   -> Single("default")        (fail-closed default)
    //   (None,   Some C) -> C                         (honor the clause)
    //   (Some P, None)   -> P                         (unchanged)
    //   (Some P, Some C) -> P iff semantically equal to C, else INVALID_ARGUMENT
    // Never widen past `P`. Covered for BOTH Cypher and AQL, incl. EXPLAIN /
    // PROFILE, with an explicit cross-namespace leak sweep.
    // =====================================================================
    mod namespace_query_reconcile_tests {
        use super::*;

        /// Seed one Person per namespace: `Legacy` (default), `Alice`
        /// (agent:a), `Bob` (agent:b).
        fn seed(server: &AletheiaMcpServer) {
            use crate::core::property::PropertyMapBuilder;
            let mk = |name: &str| PropertyMapBuilder::new().insert("name", name).build();
            server.db.create_node("Person", mk("Legacy")).unwrap();
            server
                .db
                .create_node_in_namespace("Person", mk("Alice"), "agent:a")
                .unwrap();
            server
                .db
                .create_node_in_namespace("Person", mk("Bob"), "agent:b")
                .unwrap();
        }

        fn run(server: &AletheiaMcpServer, args: serde_json::Value) -> serde_json::Value {
            let text = AletheiaMcpServer::extract_text(server.handle_query(args));
            serde_json::from_str(&text).unwrap()
        }

        /// Sorted `name` properties across all returned entity rows.
        fn names(val: &serde_json::Value) -> Vec<String> {
            let mut out: Vec<String> = val["rows"]
                .as_array()
                .map(|rows| {
                    rows.iter()
                        .filter_map(|r| {
                            r["entity"]["properties"]["name"]
                                .as_str()
                                .map(str::to_string)
                        })
                        .collect()
                })
                .unwrap_or_default();
            out.sort();
            out
        }

        fn assert_conflict(val: &serde_json::Value) {
            assert_eq!(
                val["error"]["code"].as_str(),
                Some("INVALID_ARGUMENT"),
                "a scope conflict is INVALID_ARGUMENT: {val}"
            );
            assert_eq!(
                val["error"]["retriable"].as_bool(),
                Some(false),
                "a scope conflict is never retriable: {val}"
            );
            assert_eq!(
                val["error"]["kind"].as_str(),
                Some("invalid_request"),
                "the clause IS supported now, so the kind is invalid_request, \
                 not unsupported_construct: {val}"
            );
            assert!(
                val.get("rows").is_none(),
                "a refused query returns no rows: {val}"
            );
            // The error message must NOT leak namespace contents.
            let msg = val["error"]["message"].as_str().unwrap_or_default();
            assert!(
                !msg.contains("agent:a") && !msg.contains("agent:b"),
                "the conflict message must not disclose namespace names: {val}"
            );
        }

        // ---- (1) no param + `USE NAMESPACE foo` -> only ns-foo data --------
        #[test]
        fn aql_omitted_param_honors_in_query_clause() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                }),
            );
            assert!(val.get("error").is_none(), "clause must be honored: {val}");
            assert_eq!(names(&val), vec!["Alice".to_string()]);
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_omitted_param_honors_in_query_clause() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                }),
            );
            assert!(val.get("error").is_none(), "clause must be honored: {val}");
            assert_eq!(names(&val), vec!["Alice".to_string()]);
        }

        // ---- (2) no param + no clause -> only default ----------------------
        #[test]
        fn aql_omitted_param_no_clause_is_default_only() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({ "language": "aql", "query": "MATCH (n:Person) RETURN n" }),
            );
            assert_eq!(names(&val), vec!["Legacy".to_string()]);
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_omitted_param_no_clause_is_default_only() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({ "language": "cypher", "query": "MATCH (n:Person) RETURN n" }),
            );
            assert_eq!(names(&val), vec!["Legacy".to_string()]);
        }

        // ---- (3) param=foo + no clause -> foo ------------------------------
        #[test]
        fn aql_param_only_scopes() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            assert_eq!(names(&val), vec!["Alice".to_string()]);
        }

        // ---- (4) param=foo + identical clause -> allow, foo ----------------
        #[test]
        fn aql_param_plus_identical_clause_allowed() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            assert!(
                val.get("error").is_none(),
                "identical scope is allowed: {val}"
            );
            assert_eq!(names(&val), vec!["Alice".to_string()]);
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_param_plus_identical_clause_allowed() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            assert!(
                val.get("error").is_none(),
                "identical scope is allowed: {val}"
            );
            assert_eq!(names(&val), vec!["Alice".to_string()]);
        }

        // ---- (5) param=foo + disjoint clause -> refuse ---------------------
        #[test]
        fn aql_param_plus_disjoint_clause_refused() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE NAMESPACE 'agent:b' MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            assert_conflict(&val);
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_param_plus_disjoint_clause_refused() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "USE NAMESPACE 'agent:b' MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            assert_conflict(&val);
        }

        // ---- (6) param=foo + wider `USE ALL NAMESPACES` -> refuse ----------
        #[test]
        fn aql_param_plus_wider_all_clause_refused() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE ALL NAMESPACES MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            assert_conflict(&val);
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_param_plus_wider_all_clause_refused() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "USE ALL NAMESPACES MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            assert_conflict(&val);
        }

        // ---- (7) param=ALL + narrower clause -> refuse (safest shape) ------
        #[test]
        fn aql_param_all_plus_narrower_clause_refused() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                    "namespace": "all",
                }),
            );
            assert_conflict(&val);
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_param_all_plus_narrower_clause_refused() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                    "namespace": "all",
                }),
            );
            assert_conflict(&val);
        }

        // ---- (8) no param + `USE ALL NAMESPACES` -> All, >= 2 namespaces ---
        #[test]
        fn aql_omitted_param_all_clause_spans_all_namespaces() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE ALL NAMESPACES MATCH (n:Person) RETURN n",
                }),
            );
            assert!(val.get("error").is_none(), "ALL clause is allowed: {val}");
            assert_eq!(
                names(&val),
                vec!["Alice".to_string(), "Bob".to_string(), "Legacy".to_string()],
            );
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_omitted_param_all_clause_spans_all_namespaces() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "USE ALL NAMESPACES MATCH (n:Person) RETURN n",
                }),
            );
            assert!(val.get("error").is_none(), "ALL clause is allowed: {val}");
            assert_eq!(
                names(&val),
                vec!["Alice".to_string(), "Bob".to_string(), "Legacy".to_string()],
            );
        }

        // ---- (9) List order-insensitive; strict-subset refused ------------
        #[test]
        fn aql_list_param_vs_clause_order_insensitive_allowed() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE NAMESPACE 'agent:b', 'agent:a' MATCH (n:Person) RETURN n",
                    "namespace": ["agent:a", "agent:b"],
                }),
            );
            assert!(
                val.get("error").is_none(),
                "same set, any order -> allow: {val}"
            );
            assert_eq!(names(&val), vec!["Alice".to_string(), "Bob".to_string()]);
        }

        #[test]
        fn aql_list_param_strict_subset_clause_refused() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE NAMESPACE 'agent:a', 'agent:b' MATCH (n:Person) RETURN n",
                    "namespace": ["agent:a"],
                }),
            );
            assert_conflict(&val);
        }

        // ---- (10) normalization: case is NOT a bypass ---------------------
        #[test]
        fn aql_case_differing_names_are_not_equal_refused() {
            let server = make_server();
            seed(&server);
            // `Agent:A` is a distinct (valid, case-sensitive) namespace name.
            let val = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE NAMESPACE 'Agent:A' MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            assert_conflict(&val);
        }

        // ---- (11) EXPLAIN (cypher only) -----------------------------------
        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_explain_omitted_param_honors_clause_no_leak() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "EXPLAIN USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                }),
            );
            assert!(
                val.get("error").is_none(),
                "scoped EXPLAIN must plan: {val}"
            );
            let rows = val["rows"].as_array().expect("plan rows");
            assert_eq!(rows.len(), 1, "EXPLAIN yields one plan row: {val}");
            let plan = rows[0]["plan"].as_str().unwrap_or_default();
            // The plan text must not surface other namespaces' data.
            assert!(
                !plan.contains("Bob") && !plan.contains("Legacy"),
                "EXPLAIN must not surface cross-namespace data: {plan}"
            );
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_explain_param_plus_conflicting_clause_refused() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "EXPLAIN USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                    "namespace": "agent:b",
                }),
            );
            assert_conflict(&val);
        }

        // ---- (12) PROFILE (cypher only): scoped counters, no leak ---------
        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_profile_omitted_param_honors_clause_scoped_counts() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "PROFILE USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                }),
            );
            assert!(val.get("error").is_none(), "scoped PROFILE must run: {val}");
            let plan = val["rows"][0]["plan"].as_str().unwrap_or_default();
            assert!(
                plan.contains("actual rows: 1"),
                "scoped PROFILE reports the scoped count (1): {plan}"
            );
            // The unscoped total across the 3 seeded namespaces is 3; a scoped
            // PROFILE must never leak that cardinality side channel.
            assert!(
                !plan.contains("actual rows: 3"),
                "scoped PROFILE must not leak the unscoped total: {plan}"
            );
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_profile_param_plus_conflicting_clause_refused() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "PROFILE USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                    "namespace": "agent:b",
                }),
            );
            assert_conflict(&val);
        }

        // ---- (13) multi-pattern omitted scope stays rejected --------------
        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_multi_pattern_omitted_scope_still_rejected() {
            let server = make_server();
            seed(&server);
            let val = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "MATCH (a:Person),(b:Person) RETURN a,b",
                }),
            );
            // Default-only (fail-closed) is restricting; the multi-variable
            // evaluator cannot thread it -> unsupported_construct.
            assert_eq!(
                val["error"]["code"].as_str(),
                Some("INVALID_ARGUMENT"),
                "{val}"
            );
            assert_eq!(
                val["error"]["kind"].as_str(),
                Some("unsupported_construct"),
                "{val}"
            );
        }

        // ---- (14) HTTP `/query` shares handle_query (dispatch parity) -----
        #[test]
        fn http_dispatch_shares_reconcile_outcome_aql() {
            let server = make_server();
            seed(&server);
            let args = serde_json::json!({
                "language": "aql",
                "query": "USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
            });
            let via_handle = AletheiaMcpServer::extract_text(server.handle_query(args.clone()));
            let via_dispatch = server.dispatch_tool_json("query", args);
            assert_eq!(
                via_handle, via_dispatch,
                "the autumn-server /query dispatch path must be byte-identical to handle_query"
            );
            let val: serde_json::Value = serde_json::from_str(&via_dispatch).unwrap();
            assert_eq!(names(&val), vec!["Alice".to_string()]);
        }

        // ---- (15) cross-namespace leak sweep: scope A never returns B -----
        #[test]
        fn aql_scope_a_never_leaks_b_or_default() {
            let server = make_server();
            seed(&server);
            // Every composition path that resolves to agent:a.
            let via_clause = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                }),
            );
            let via_param = run(
                &server,
                serde_json::json!({
                    "language": "aql",
                    "query": "MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            for val in [&via_clause, &via_param] {
                assert_eq!(names(val), vec!["Alice".to_string()], "{val}");
                let all = serde_json::to_string(val).unwrap();
                assert!(
                    !all.contains("Bob") && !all.contains("Legacy"),
                    "leak: {val}"
                );
            }
        }

        #[cfg(feature = "cypher")]
        #[test]
        fn cypher_scope_a_never_leaks_b_or_default() {
            let server = make_server();
            seed(&server);
            let via_clause = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n",
                }),
            );
            let via_param = run(
                &server,
                serde_json::json!({
                    "language": "cypher",
                    "query": "MATCH (n:Person) RETURN n",
                    "namespace": "agent:a",
                }),
            );
            for val in [&via_clause, &via_param] {
                assert_eq!(names(val), vec!["Alice".to_string()], "{val}");
                let all = serde_json::to_string(val).unwrap();
                assert!(
                    !all.contains("Bob") && !all.contains("Legacy"),
                    "leak: {val}"
                );
            }
            // EXPLAIN + PROFILE scoped to agent:a must not surface B/Legacy.
            for directive in ["EXPLAIN", "PROFILE"] {
                let val = run(
                    &server,
                    serde_json::json!({
                        "language": "cypher",
                        "query": format!(
                            "{directive} USE NAMESPACE 'agent:a' MATCH (n:Person) RETURN n"
                        ),
                    }),
                );
                let plan = val["rows"][0]["plan"].as_str().unwrap_or_default();
                assert!(
                    !plan.contains("Bob") && !plan.contains("Legacy"),
                    "{directive} leaked cross-namespace data: {plan}"
                );
            }
        }
    }
}
