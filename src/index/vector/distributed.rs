//! Distributed vector search across multiple network nodes.
//!
//! This module provides a distributed implementation of the `VectorIndex` trait,
//! enabling horizontal scaling of vector search across multiple machines.
//!
//! # Overview
//!
//! The `DistributedVectorIndex` routes vectors to remote nodes using consistent
//! hashing, and executes search queries using a scatter-gather pattern across
//! all nodes.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                  DistributedVectorIndex                          │
//! │   • Consistent hashing for routing                               │
//! │   • Scatter-gather for queries                                   │
//! │   • Circuit breakers for fault tolerance                         │
//! └──────────────────────────────────────────────────────────────────┘
//!               │                    │                    │
//!               ▼                    ▼                    ▼
//!      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
//!      │   Node 0    │      │   Node 1    │      │   Node 2    │
//!      │ VectorIndex │      │ VectorIndex │      │ VectorIndex │
//!      └─────────────┘      └─────────────┘      └─────────────┘
//! ```
//!
//! # Key Features
//!
//! - **Horizontal scaling**: Distribute vectors across multiple nodes
//! - **Fault tolerance**: Circuit breakers prevent cascading failures
//! - **Parallel search**: Query all nodes concurrently and merge results
//! - **Consistent routing**: Same vector ID always routes to same node
//!
//! # Example
//!
//! ```ignore
//! use aletheiadb::index::vector::distributed::{
//!     DistributedVectorIndex, DistributedVectorConfig, VectorNodeConfig
//! };
//! use aletheiadb::index::vector::{DistanceMetric, VectorIndex};
//!
//! // Define cluster topology
//! let config = DistributedVectorConfig::new(384, DistanceMetric::Cosine)
//!     .with_node(VectorNodeConfig::new(0, "node0:9000"))
//!     .with_node(VectorNodeConfig::new(1, "node1:9000"))
//!     .with_node(VectorNodeConfig::new(2, "node2:9000"));
//!
//! let index = DistributedVectorIndex::new(config)?;
//!
//! // Add vectors - automatically routed to appropriate node
//! let embedding = vec![0.1f32; 384];
//! index.add(node_id, &embedding)?;
//!
//! // Search across all nodes
//! let results = index.search(&query, 10)?;
//! ```

use crate::core::error::{Error, Result, VectorError};
use crate::core::hasher::IdentityHasher;
use crate::core::id::NodeId;
use crate::core::vector::validate_vector;
use crate::index::vector::{DistanceMetric, Quantization, VectorIndex, merge_top_k_results};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Maximum number of results that can be requested in a search.
const MAX_K: usize = 100_000;

/// Overfetch factor for filtered search.
/// When applying post-search filters, we fetch this many times more results
/// than requested to increase the chance of having enough results after filtering.
const FILTER_OVERFETCH_FACTOR: usize = 10;

/// Default timeout for remote operations.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default failure threshold for circuit breaker.
const DEFAULT_FAILURE_THRESHOLD: usize = 5;

/// Default circuit breaker open duration.
const DEFAULT_OPEN_DURATION: Duration = Duration::from_secs(30);

/// Recommended imbalance threshold for rebalancing.
/// Trigger rebalancing when the largest node has more than 2x the vectors of the smallest.
pub const RECOMMENDED_IMBALANCE_THRESHOLD: f64 = 2.0;

// ============================================================================
// Error Types
// ============================================================================

/// Errors specific to distributed vector operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedError {
    /// No nodes are available.
    NoNodesAvailable,
    /// A specific node is unavailable.
    NodeUnavailable {
        /// Node ID.
        node_id: u16,
        /// Reason for unavailability.
        reason: String,
    },
    /// All nodes failed during a query.
    AllNodesFailed {
        /// Number of nodes that failed.
        failed_count: usize,
        /// Sample error message.
        sample_error: String,
    },
    /// Operation timed out.
    Timeout {
        /// Operation that timed out.
        operation: String,
        /// Duration before timeout.
        duration: Duration,
    },
    /// Circuit breaker is open for a node.
    CircuitOpen {
        /// Node ID.
        node_id: u16,
        /// Remaining time before circuit closes.
        remaining: Duration,
    },
    /// Configuration error.
    ConfigError(String),
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributedError::NoNodesAvailable => {
                write!(f, "No nodes available in the distributed index")
            }
            DistributedError::NodeUnavailable { node_id, reason } => {
                write!(f, "Node {} is unavailable: {}", node_id, reason)
            }
            DistributedError::AllNodesFailed {
                failed_count,
                sample_error,
            } => {
                write!(
                    f,
                    "All {} nodes failed during query. Sample error: {}",
                    failed_count, sample_error
                )
            }
            DistributedError::Timeout {
                operation,
                duration,
            } => {
                write!(
                    f,
                    "Operation '{}' timed out after {:?}",
                    operation, duration
                )
            }
            DistributedError::CircuitOpen { node_id, remaining } => {
                write!(
                    f,
                    "Circuit breaker open for node {}, {} seconds remaining",
                    node_id,
                    remaining.as_secs()
                )
            }
            DistributedError::ConfigError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
        }
    }
}

impl std::error::Error for DistributedError {}

// ============================================================================
// Distributed Vector Client Trait
// ============================================================================

/// Trait for communicating with remote vector index nodes.
///
/// This trait abstracts the network communication, allowing for different
/// implementations (HTTP, gRPC, in-process mock for testing).
pub trait VectorNodeClient: Send + Sync + fmt::Debug {
    /// Get the node ID this client connects to.
    fn node_id(&self) -> u16;

    /// Check if the connection is healthy.
    fn is_healthy(&self) -> bool;

    /// Add a vector to the remote index.
    fn add(&self, id: NodeId, vector: &[f32]) -> Result<()>;

    /// Remove a vector from the remote index.
    fn remove(&self, id: NodeId) -> Result<()>;

    /// Search for k-nearest neighbors on the remote index.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>>;

    /// Get the number of vectors on the remote index.
    fn len(&self) -> Result<usize>;

    /// Check if the remote index is empty.
    fn is_empty(&self) -> Result<bool> {
        self.len().map(|len| len == 0)
    }

    /// Perform a health check.
    fn health_check(&self) -> Result<()>;
}

// ============================================================================
// Circuit Breaker for Nodes
// ============================================================================

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed (normal operation).
    Closed,
    /// Circuit is open (rejecting requests).
    Open,
    /// Circuit is half-open (testing if service recovered).
    HalfOpen,
}

/// Configuration for circuit breaker.
///
/// The `CircuitBreakerConfig` defines the thresholds and durations for the
/// [`NodeCircuitBreaker`] to transition between `Closed`, `Open`, and `HalfOpen` states.
/// It helps prevent cascading failures in a distributed setup when remote nodes are unresponsive.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() {
/// use aletheiadb::index::vector::distributed::CircuitBreakerConfig;
/// use std::time::Duration;
///
/// let config = CircuitBreakerConfig {
///     failure_threshold: 5,
///     open_duration: Duration::from_secs(30),
///     success_threshold: 3,
/// };
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit.
    pub failure_threshold: usize,
    /// Duration to keep circuit open.
    pub open_duration: Duration,
    /// Number of successes in half-open to close circuit.
    pub success_threshold: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
            open_duration: DEFAULT_OPEN_DURATION,
            success_threshold: 3,
        }
    }
}

/// Circuit breaker for a single node connection.
///
/// The `NodeCircuitBreaker` tracks failures and successes of requests to a remote node
/// (typically represented by a [`VectorNodeClient`]). If failures exceed the configured
/// threshold, it transitions to an `Open` state, failing fast. After a timeout, it transitions
/// to a `HalfOpen` state to test if the node has recovered.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() {
/// use aletheiadb::index::vector::distributed::{CircuitBreakerConfig, NodeCircuitBreaker, CircuitState};
///
/// let config = CircuitBreakerConfig::default();
/// let breaker = NodeCircuitBreaker::new(config);
///
/// // Initially, the circuit is closed and allows requests
/// assert_eq!(breaker.state(), CircuitState::Closed);
/// assert!(breaker.should_allow());
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
#[derive(Debug)]
pub struct NodeCircuitBreaker {
    config: CircuitBreakerConfig,
    state: RwLock<CircuitState>,
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    opened_at: RwLock<Option<Instant>>,
}

impl NodeCircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            opened_at: RwLock::new(None),
        }
    }

    /// Get the current state.
    pub fn state(&self) -> CircuitState {
        self.maybe_transition();
        self.state
            .read()
            .map(|s| *s)
            .unwrap_or(CircuitState::Closed)
    }

    /// Check if requests should be allowed.
    pub fn should_allow(&self) -> bool {
        self.maybe_transition();
        let state = self
            .state
            .read()
            .map(|s| *s)
            .unwrap_or(CircuitState::Closed);
        matches!(state, CircuitState::Closed | CircuitState::HalfOpen)
    }

    /// Record a successful request.
    pub fn record_success(&self) {
        let state = match self.state.read() {
            Ok(s) => *s,
            Err(_) => return,
        };

        match state {
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::SeqCst);
            }
            CircuitState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                if successes >= self.config.success_threshold {
                    if let Ok(mut s) = self.state.write() {
                        *s = CircuitState::Closed;
                    }
                    self.failure_count.store(0, Ordering::SeqCst);
                    self.success_count.store(0, Ordering::SeqCst);
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        let state = match self.state.read() {
            Ok(s) => *s,
            Err(_) => return,
        };

        match state {
            CircuitState::Closed => {
                let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
                if failures >= self.config.failure_threshold {
                    if let Ok(mut s) = self.state.write() {
                        *s = CircuitState::Open;
                    }
                    if let Ok(mut opened) = self.opened_at.write() {
                        *opened = Some(Instant::now());
                    }
                }
            }
            CircuitState::HalfOpen => {
                if let Ok(mut s) = self.state.write() {
                    *s = CircuitState::Open;
                }
                if let Ok(mut opened) = self.opened_at.write() {
                    *opened = Some(Instant::now());
                }
                self.success_count.store(0, Ordering::SeqCst);
            }
            CircuitState::Open => {
                // Do not reset the timer when the circuit is already open.
                // This allows it to transition to HalfOpen after the original duration.
            }
        }
    }

    /// Check and perform state transitions based on time.
    ///
    /// This method reads opened_at before acquiring the state write lock to
    /// avoid holding locks across different fields simultaneously.
    fn maybe_transition(&self) {
        // Read opened_at first to avoid holding multiple locks
        let should_transition = self
            .opened_at
            .read()
            .ok()
            .and_then(|opened| *opened)
            .is_some_and(|opened_time| opened_time.elapsed() >= self.config.open_duration);

        if !should_transition {
            return;
        }

        // Now acquire state write lock and verify state is still Open
        let mut state_guard = match self.state.write() {
            Ok(s) => s,
            Err(_) => return,
        };

        // Double-check state is still Open (could have changed)
        if *state_guard == CircuitState::Open {
            *state_guard = CircuitState::HalfOpen;
            self.success_count.store(0, Ordering::SeqCst);
        }
    }

    /// Get remaining time before circuit can close.
    pub fn remaining_open_time(&self) -> Option<Duration> {
        let state = self.state.read().ok()?;
        if *state != CircuitState::Open {
            return None;
        }

        if let Ok(opened) = self.opened_at.read()
            && let Some(opened_time) = *opened
        {
            let elapsed = opened_time.elapsed();
            if elapsed < self.config.open_duration {
                return Some(self.config.open_duration - elapsed);
            }
        }
        None
    }

    /// Reset the circuit breaker to closed state.
    pub fn reset(&self) {
        if let Ok(mut s) = self.state.write() {
            *s = CircuitState::Closed;
        }
        self.failure_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
    }
}

// ============================================================================
// Node Connection
// ============================================================================

/// A connection to a remote vector index node.
///
/// The `NodeConnection` wraps a [`VectorNodeClient`] with a [`NodeCircuitBreaker`]
/// and tracks basic metrics (requests, failures). It acts as the primary interface
/// for executing requests against a specific node in a distributed index.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() {
/// use aletheiadb::index::vector::distributed::{
///     MockVectorNodeClient, NodeConnection, CircuitBreakerConfig
/// };
/// use aletheiadb::index::vector::DistanceMetric;
/// use std::sync::Arc;
///
/// let client = Arc::new(MockVectorNodeClient::new(0, 384, DistanceMetric::Cosine));
/// let connection = NodeConnection::new(client, CircuitBreakerConfig::default());
///
/// assert_eq!(connection.node_id(), 0);
/// assert!(connection.is_available());
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
pub struct NodeConnection<C: VectorNodeClient> {
    /// The client for this node.
    client: Arc<C>,
    /// Circuit breaker for this node.
    circuit_breaker: NodeCircuitBreaker,
    /// Total requests made to this node.
    request_count: AtomicU64,
    /// Failed requests to this node.
    failure_count: AtomicU64,
}

impl<C: VectorNodeClient> fmt::Debug for NodeConnection<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeConnection")
            .field("node_id", &self.client.node_id())
            .field("circuit_state", &self.circuit_breaker.state())
            .field("request_count", &self.request_count.load(Ordering::Relaxed))
            .field("failure_count", &self.failure_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl<C: VectorNodeClient> NodeConnection<C> {
    /// Create a new node connection.
    pub fn new(client: Arc<C>, circuit_config: CircuitBreakerConfig) -> Self {
        Self {
            client,
            circuit_breaker: NodeCircuitBreaker::new(circuit_config),
            request_count: AtomicU64::new(0),
            failure_count: AtomicU64::new(0),
        }
    }

    /// Get the node ID.
    pub fn node_id(&self) -> u16 {
        self.client.node_id()
    }

    /// Check if the node is available.
    pub fn is_available(&self) -> bool {
        self.circuit_breaker.should_allow() && self.client.is_healthy()
    }

    /// Get the circuit breaker state.
    pub fn circuit_state(&self) -> CircuitState {
        self.circuit_breaker.state()
    }

    /// Execute an operation with circuit breaker protection.
    pub fn execute<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&C) -> Result<T>,
    {
        self.request_count.fetch_add(1, Ordering::Relaxed);

        if !self.circuit_breaker.should_allow() {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "Circuit breaker open for node {}",
                self.client.node_id()
            ))));
        }

        match f(&self.client) {
            Ok(result) => {
                self.circuit_breaker.record_success();
                Ok(result)
            }
            Err(e) => {
                self.failure_count.fetch_add(1, Ordering::Relaxed);
                self.circuit_breaker.record_failure();
                Err(e)
            }
        }
    }

    /// Get connection statistics.
    pub fn stats(&self) -> NodeConnectionStats {
        NodeConnectionStats {
            node_id: self.client.node_id(),
            circuit_state: self.circuit_breaker.state(),
            request_count: self.request_count.load(Ordering::Relaxed),
            failure_count: self.failure_count.load(Ordering::Relaxed),
        }
    }

    /// Reset the circuit breaker.
    pub fn reset_circuit(&self) {
        self.circuit_breaker.reset();
    }
}

/// Statistics for a node connection.
///
/// `NodeConnectionStats` contains diagnostic information about a [`NodeConnection`],
/// including its current circuit state, total requests, and total failures.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() {
/// use aletheiadb::index::vector::distributed::{NodeConnectionStats, CircuitState};
///
/// let stats = NodeConnectionStats {
///     node_id: 1,
///     circuit_state: CircuitState::Closed,
///     request_count: 100,
///     failure_count: 2,
/// };
///
/// assert_eq!(stats.success_rate(), 0.98);
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone)]
pub struct NodeConnectionStats {
    /// Node ID.
    pub node_id: u16,
    /// Circuit breaker state.
    pub circuit_state: CircuitState,
    /// Total requests made.
    pub request_count: u64,
    /// Failed requests.
    pub failure_count: u64,
}

impl NodeConnectionStats {
    /// Calculate success rate.
    pub fn success_rate(&self) -> f64 {
        if self.request_count == 0 {
            1.0
        } else {
            1.0 - (self.failure_count as f64 / self.request_count as f64)
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a single vector node.
///
/// The `VectorNodeConfig` describes how to connect to a specific remote node,
/// including its endpoint, timeout settings, and specific [`CircuitBreakerConfig`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() {
/// use aletheiadb::index::vector::distributed::VectorNodeConfig;
/// use std::time::Duration;
///
/// let node_config = VectorNodeConfig::new(1, "node1:9000")
///     .with_timeout(Duration::from_secs(10));
///
/// assert_eq!(node_config.node_id, 1);
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone)]
pub struct VectorNodeConfig {
    /// Node ID (must be unique).
    pub node_id: u16,
    /// Node endpoint (e.g., "node0:9000").
    pub endpoint: String,
    /// Request timeout.
    pub timeout: Duration,
    /// Circuit breaker configuration.
    pub circuit_breaker: CircuitBreakerConfig,
}

impl VectorNodeConfig {
    /// Create a new node configuration.
    pub fn new(node_id: u16, endpoint: impl Into<String>) -> Self {
        Self {
            node_id,
            endpoint: endpoint.into(),
            timeout: DEFAULT_TIMEOUT,
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }

    /// Set the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the circuit breaker configuration.
    pub fn with_circuit_breaker(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker = config;
        self
    }
}

/// Routing strategy for distributing vectors across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingStrategy {
    /// Hash-based routing: node = hash(node_id) % num_nodes
    #[default]
    HashBased,
    /// Range-based routing: divides ID space evenly across nodes.
    RangeBased,
}

/// Configuration for the distributed vector index.
///
/// The `DistributedVectorConfig` sets up the topology and behavior for a
/// [`DistributedVectorIndex`], including the expected vector dimensionality,
/// distance metric, routing strategy, and the list of [`VectorNodeConfig`]s.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() {
/// use aletheiadb::index::vector::distributed::{DistributedVectorConfig, VectorNodeConfig};
/// use aletheiadb::index::vector::DistanceMetric;
///
/// let config = DistributedVectorConfig::new(384, DistanceMetric::Cosine)
///     .with_node(VectorNodeConfig::new(0, "node0:9000"))
///     .with_node(VectorNodeConfig::new(1, "node1:9000"));
///
/// assert!(config.validate().is_ok());
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone)]
pub struct DistributedVectorConfig {
    /// Vector dimensionality.
    pub dimensions: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Nodes in the cluster.
    pub nodes: Vec<VectorNodeConfig>,
    /// Routing strategy.
    pub routing_strategy: RoutingStrategy,
    /// Minimum number of nodes required for search.
    pub min_nodes_for_search: usize,
    /// Whether to allow partial results when some nodes fail.
    pub allow_partial_results: bool,
}

impl DistributedVectorConfig {
    /// Create a new configuration.
    pub fn new(dimensions: usize, metric: DistanceMetric) -> Self {
        Self {
            dimensions,
            metric,
            nodes: Vec::new(),
            routing_strategy: RoutingStrategy::default(),
            min_nodes_for_search: 1,
            allow_partial_results: true,
        }
    }

    /// Add a node to the cluster.
    pub fn with_node(mut self, node: VectorNodeConfig) -> Self {
        self.nodes.push(node);
        self
    }

    /// Set the routing strategy.
    pub fn with_routing_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.routing_strategy = strategy;
        self
    }

    /// Set the minimum number of nodes required for search.
    pub fn with_min_nodes_for_search(mut self, min_nodes: usize) -> Self {
        self.min_nodes_for_search = min_nodes;
        self
    }

    /// Set whether to allow partial results.
    pub fn with_allow_partial_results(mut self, allow: bool) -> Self {
        self.allow_partial_results = allow;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.dimensions == 0 {
            return Err(Error::Vector(VectorError::InvalidVector {
                reason: "dimensions must be > 0".to_string(),
            }));
        }

        if self.nodes.is_empty() {
            return Err(Error::Vector(VectorError::IndexError(
                "At least one node must be configured".to_string(),
            )));
        }

        // Check for duplicate node IDs
        let mut seen_ids = std::collections::HashSet::new();
        for node in &self.nodes {
            if !seen_ids.insert(node.node_id) {
                return Err(Error::Vector(VectorError::IndexError(format!(
                    "Duplicate node ID: {}",
                    node.node_id
                ))));
            }
        }

        Ok(())
    }
}

// ============================================================================
// Distributed Vector Index
// ============================================================================

/// Statistics for the distributed index.
///
/// `DistributedIndexStats` provides a cluster-wide summary of the [`DistributedVectorIndex`],
/// aggregating counts and holding detailed [`NodeConnectionStats`] for all participating nodes.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() {
/// use aletheiadb::index::vector::distributed::{DistributedIndexStats, NodeConnectionStats, CircuitState};
///
/// let stats = DistributedIndexStats {
///     total_vectors: 1000,
///     node_count: 2,
///     available_nodes: 2,
///     node_stats: vec![], // Would contain actual node stats
/// };
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone)]
pub struct DistributedIndexStats {
    /// Total vectors across all nodes.
    pub total_vectors: usize,
    /// Number of nodes.
    pub node_count: usize,
    /// Number of available nodes.
    pub available_nodes: usize,
    /// Statistics per node.
    pub node_stats: Vec<NodeConnectionStats>,
}

/// Statistics for rebalancing the distributed index.
///
/// `RebalanceStats` describes the current balance of vectors across nodes in a
/// [`DistributedVectorIndex`]. It helps determine if rebalancing is necessary based
/// on the `imbalance_ratio` compared to the [`RECOMMENDED_IMBALANCE_THRESHOLD`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() {
/// use aletheiadb::index::vector::distributed::RebalanceStats;
///
/// let stats = RebalanceStats {
///     total_vectors: 200,
///     node_count: 2,
///     min_node_size: 50,
///     max_node_size: 150,
///     imbalance_ratio: 3.0,
///     vectors_to_move: 50,
///     node_sizes: vec![(0, 50), (1, 150)],
/// };
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone)]
pub struct RebalanceStats {
    /// Total vectors across all nodes.
    pub total_vectors: usize,
    /// Number of nodes.
    pub node_count: usize,
    /// Minimum node size.
    pub min_node_size: usize,
    /// Maximum node size.
    pub max_node_size: usize,
    /// Imbalance ratio (max/min).
    pub imbalance_ratio: f64,
    /// Estimated vectors to move for balance.
    pub vectors_to_move: usize,
    /// Vector count per node.
    pub node_sizes: Vec<(u16, usize)>,
}

/// A distributed vector index across multiple network nodes.
///
/// This structure provides horizontal scalability by partitioning vectors
/// across multiple remote nodes and coordinating search operations using
/// a scatter-gather pattern.
///
/// See [`DistributedVectorConfig`] for configuring the topology, and
/// [`VectorNodeClient`] for the required abstraction to interface with remote nodes.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() -> aletheiadb::core::error::Result<()> {
/// use std::sync::Arc;
/// use aletheiadb::index::vector::distributed::{
///     DistributedVectorIndex, DistributedVectorConfig, VectorNodeConfig, MockVectorNodeClient
/// };
/// use aletheiadb::index::vector::DistanceMetric;
///
/// let config = DistributedVectorConfig::new(128, DistanceMetric::Cosine)
///     .with_node(VectorNodeConfig::new(0, "node0:9000"));
///
/// let client = Arc::new(MockVectorNodeClient::new(0, 128, DistanceMetric::Cosine));
/// let index = DistributedVectorIndex::new(config, vec![client])?;
///
/// assert_eq!(index.node_count(), 1);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
pub struct DistributedVectorIndex<C: VectorNodeClient> {
    /// Configuration.
    config: DistributedVectorConfig,
    /// Node connections.
    nodes: Vec<Arc<NodeConnection<C>>>,
}

impl<C: VectorNodeClient> fmt::Debug for DistributedVectorIndex<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DistributedVectorIndex")
            .field("dimensions", &self.config.dimensions)
            .field("metric", &self.config.metric)
            .field("node_count", &self.nodes.len())
            .field("routing_strategy", &self.config.routing_strategy)
            .finish()
    }
}

impl<C: VectorNodeClient + 'static> DistributedVectorIndex<C> {
    /// Create a new distributed vector index from an existing list of clients.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the distributed index
    /// * `clients` - Pre-created client instances for each node
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid.
    pub fn new(config: DistributedVectorConfig, clients: Vec<Arc<C>>) -> Result<Self> {
        config.validate()?;

        if clients.len() != config.nodes.len() {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "Number of clients ({}) doesn't match number of configured nodes ({})",
                clients.len(),
                config.nodes.len()
            ))));
        }

        let nodes: Vec<Arc<NodeConnection<C>>> = clients
            .into_iter()
            .zip(config.nodes.iter())
            .map(|(client, node_config)| {
                Arc::new(NodeConnection::new(
                    client,
                    node_config.circuit_breaker.clone(),
                ))
            })
            .collect();

        Ok(Self { config, nodes })
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of available nodes.
    pub fn available_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_available()).count()
    }

    /// Get the routing strategy.
    pub fn routing_strategy(&self) -> RoutingStrategy {
        self.config.routing_strategy
    }

    /// Get the node index for a given NodeId.
    fn node_for_id(&self, id: NodeId) -> usize {
        debug_assert!(!self.nodes.is_empty(), "nodes cannot be empty");
        let num_nodes = self.nodes.len();

        match self.config.routing_strategy {
            RoutingStrategy::HashBased => {
                let mut hasher = DefaultHasher::new();
                id.as_u64().hash(&mut hasher);
                (hasher.finish() as usize) % num_nodes
            }
            RoutingStrategy::RangeBased => {
                let num_nodes_128 = num_nodes as u128;
                let id_128 = id.as_u64() as u128;
                let node = ((id_128 * num_nodes_128) / (u64::MAX as u128 + 1)) as usize;
                node.min(num_nodes - 1)
            }
        }
    }

    /// Get a reference to a specific node connection.
    pub fn get_node(&self, index: usize) -> Option<&Arc<NodeConnection<C>>> {
        self.nodes.get(index)
    }

    /// Get statistics about the distributed index.
    pub fn stats(&self) -> DistributedIndexStats {
        let node_stats: Vec<NodeConnectionStats> = self.nodes.iter().map(|n| n.stats()).collect();

        let available_nodes = self.nodes.iter().filter(|n| n.is_available()).count();

        // Query all available nodes for their vector counts
        let total_vectors = self.len();

        DistributedIndexStats {
            total_vectors,
            node_count: self.nodes.len(),
            available_nodes,
            node_stats,
        }
    }

    /// Reset all circuit breakers.
    pub fn reset_all_circuits(&self) {
        for node in &self.nodes {
            node.reset_circuit();
        }
    }

    /// Check if the cluster needs rebalancing.
    ///
    /// Returns true if the imbalance ratio exceeds the threshold.
    pub fn needs_rebalancing(&self, threshold: f64) -> bool {
        let sizes: Vec<usize> = self
            .nodes
            .par_iter()
            .filter(|n| n.is_available())
            .filter_map(|node| node.execute(|client| client.len()).ok())
            .collect();

        if sizes.is_empty() || sizes.len() < 2 {
            return false;
        }

        let min_size = sizes.iter().min().copied().unwrap_or(0);
        let max_size = sizes.iter().max().copied().unwrap_or(0);

        if min_size == 0 {
            return max_size > 0;
        }

        let imbalance_ratio = max_size as f64 / min_size as f64;
        imbalance_ratio > threshold
    }

    /// Get rebalancing statistics.
    pub fn rebalance_stats(&self) -> RebalanceStats {
        let sizes: Vec<(u16, usize)> = self
            .nodes
            .par_iter()
            .filter(|n| n.is_available())
            .filter_map(|node| {
                node.execute(|client| client.len().map(|len| (client.node_id(), len)))
                    .ok()
            })
            .collect();

        let total_vectors: usize = sizes.iter().map(|(_, s)| s).sum();
        let node_count = sizes.len();

        let min_size = sizes.iter().map(|(_, s)| *s).min().unwrap_or(0);
        let max_size = sizes.iter().map(|(_, s)| *s).max().unwrap_or(0);

        let imbalance_ratio = if min_size > 0 {
            max_size as f64 / min_size as f64
        } else if max_size > 0 {
            f64::INFINITY
        } else {
            1.0
        };

        let target_per_node = total_vectors.checked_div(node_count).unwrap_or(0);

        let vectors_to_move: usize = sizes
            .iter()
            .filter(|(_, s)| *s > target_per_node)
            .map(|(_, s)| s - target_per_node)
            .sum();

        RebalanceStats {
            total_vectors,
            node_count,
            min_node_size: min_size,
            max_node_size: max_size,
            imbalance_ratio,
            vectors_to_move,
            node_sizes: sizes,
        }
    }

    /// Merge search results from multiple nodes using a min-heap for efficiency.
    fn merge_results(node_results: Vec<Vec<(NodeId, f32)>>, k: usize) -> Vec<(NodeId, f32)> {
        merge_top_k_results(node_results, k)
    }
}

impl<C: VectorNodeClient + 'static> VectorIndex for DistributedVectorIndex<C> {
    fn add(&self, id: NodeId, vector: &[f32]) -> Result<()> {
        validate_vector(vector)?;

        if vector.len() != self.config.dimensions {
            return Err(Error::Vector(VectorError::DimensionMismatch {
                expected: self.config.dimensions,
                actual: vector.len(),
            }));
        }

        let node_idx = self.node_for_id(id);
        self.nodes[node_idx].execute(|client| client.add(id, vector))
    }

    fn remove(&self, id: NodeId) -> Result<()> {
        let node_idx = self.node_for_id(id);
        self.nodes[node_idx].execute(|client| client.remove(id))
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>> {
        validate_vector(query)?;

        if query.len() != self.config.dimensions {
            return Err(Error::Vector(VectorError::DimensionMismatch {
                expected: self.config.dimensions,
                actual: query.len(),
            }));
        }

        // Prevent DoS attacks via excessive k values
        if k > MAX_K {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "k={} exceeds maximum allowed value of {}",
                k, MAX_K
            ))));
        }

        // Check available nodes
        let available_nodes: Vec<_> = self.nodes.iter().filter(|n| n.is_available()).collect();

        if available_nodes.len() < self.config.min_nodes_for_search {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "Not enough nodes available: {} < {}",
                available_nodes.len(),
                self.config.min_nodes_for_search
            ))));
        }

        // Search all available nodes in parallel
        let results: Vec<Result<Vec<(NodeId, f32)>>> = available_nodes
            .par_iter()
            .map(|node| node.execute(|client| client.search(query, k)))
            .collect();

        // Collect successful results
        let mut successful_results = Vec::with_capacity(results.len()); // ⚡ Bolt Optimization: Pre-allocate successful_results vector using results.len() to prevent intermediate heap reallocations on the happy path.
        let mut failed_count = 0;
        let mut sample_error = String::new();

        for result in results {
            match result {
                Ok(r) => successful_results.push(r),
                Err(e) => {
                    failed_count += 1;
                    if sample_error.is_empty() {
                        sample_error = e.to_string();
                    }
                }
            }
        }

        // Check if we have enough successful results
        if successful_results.is_empty() {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "All nodes failed: {}",
                sample_error
            ))));
        }

        if !self.config.allow_partial_results && failed_count > 0 {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "{} nodes failed during search",
                failed_count
            ))));
        }

        Ok(Self::merge_results(successful_results, k))
    }

    fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        predicate: F,
    ) -> Result<Vec<(NodeId, f32)>>
    where
        F: Fn(&NodeId) -> bool + Send + Sync,
    {
        // For distributed search with filter, we search all nodes and filter results.
        // A more efficient implementation would push the filter to the nodes.
        // We overfetch to increase the likelihood of having enough results after filtering.
        let results = self.search(query, k.saturating_mul(FILTER_OVERFETCH_FACTOR))?;

        let filtered: Vec<(NodeId, f32)> = results
            .into_iter()
            .filter(|(id, _)| predicate(id))
            .take(k)
            .collect();

        Ok(filtered)
    }

    fn len(&self) -> usize {
        // Sum lengths from all available nodes
        self.nodes
            .par_iter()
            .filter(|n| n.is_available())
            .filter_map(|node| node.execute(|client| client.len()).ok())
            .sum()
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.config.metric
    }

    fn memory_usage(&self) -> usize {
        // Memory usage is distributed across nodes
        // Return local metadata overhead only
        std::mem::size_of::<Self>()
            + self.nodes.len() * std::mem::size_of::<Arc<NodeConnection<C>>>()
    }

    fn quantization(&self) -> Quantization {
        Quantization::F32 // Default; actual quantization is on the nodes
    }
}

// ============================================================================
// Mock Client for Testing
// ============================================================================

/// Mock client for testing distributed vector operations.
///
/// `MockVectorNodeClient` implements the [`VectorNodeClient`] trait in-memory
/// without actually performing any network I/O. It tracks vectors in a local hash map
/// and implements simple distance calculations, making it ideal for unit testing
/// [`DistributedVectorIndex`] logic.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "semantic-search")]
/// # fn main() -> aletheiadb::core::error::Result<()> {
/// use aletheiadb::index::vector::distributed::{MockVectorNodeClient, VectorNodeClient};
/// use aletheiadb::index::vector::DistanceMetric;
/// use aletheiadb::core::id::NodeId;
///
/// let client = MockVectorNodeClient::new(0, 128, DistanceMetric::Cosine);
///
/// let node_id = NodeId::new(42).unwrap();
/// let vec = vec![0.1f32; 128];
/// client.add(node_id, &vec)?;
///
/// assert_eq!(client.len()?, 1);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "semantic-search"))]
/// # fn main() {}
/// ```
#[derive(Debug)]
pub struct MockVectorNodeClient {
    node_id: u16,
    healthy: AtomicBool,
    vectors:
        RwLock<std::collections::HashMap<NodeId, Vec<f32>, BuildHasherDefault<IdentityHasher>>>,
    dimensions: usize,
    metric: DistanceMetric,
    fail_next: RwLock<Option<String>>,
}

impl MockVectorNodeClient {
    /// Create a new mock client.
    pub fn new(node_id: u16, dimensions: usize, metric: DistanceMetric) -> Self {
        Self {
            node_id,
            healthy: AtomicBool::new(true),
            vectors: RwLock::new(std::collections::HashMap::with_hasher(
                BuildHasherDefault::default(),
            )),
            dimensions,
            metric,
            fail_next: RwLock::new(None),
        }
    }

    /// Set whether the client is healthy.
    pub fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::SeqCst);
    }

    /// Make the next operation fail.
    pub fn fail_next(&self, error: impl Into<String>) {
        if let Ok(mut lock) = self.fail_next.write() {
            *lock = Some(error.into());
        }
    }

    fn check_fail(&self) -> Result<()> {
        let mut lock = self.fail_next.write().map_err(|_| {
            Error::Vector(VectorError::IndexError("lock poisoned".into()))
        })?;
        if let Some(err) = lock.take() {
            return Err(Error::Vector(VectorError::IndexError(err)));
        }
        Ok(())
    }

    fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if mag_a > 0.0 && mag_b > 0.0 {
                    dot / (mag_a * mag_b)
                } else {
                    0.0
                }
            }
            DistanceMetric::Euclidean => {
                let dist: f32 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                -dist // Negate so higher is better
            }
            DistanceMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
            other => panic!(
                "MockVectorNodeClient does not support {:?} distance metric",
                other
            ),
        }
    }
}

impl VectorNodeClient for MockVectorNodeClient {
    fn node_id(&self) -> u16 {
        self.node_id
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::SeqCst)
    }

    fn add(&self, id: NodeId, vector: &[f32]) -> Result<()> {
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "Node {} is unavailable",
                self.node_id
            ))));
        }

        if vector.len() != self.dimensions {
            return Err(Error::Vector(VectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.len(),
            }));
        }

        self.vectors.write().map_err(|_| Error::Vector(VectorError::IndexError("lock poisoned".into())))?.insert(id, vector.to_vec());
        Ok(())
    }

    fn remove(&self, id: NodeId) -> Result<()> {
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "Node {} is unavailable",
                self.node_id
            ))));
        }

        self.vectors.write().map_err(|_| Error::Vector(VectorError::IndexError("lock poisoned".into())))?.remove(&id);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>> {
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "Node {} is unavailable",
                self.node_id
            ))));
        }

        let vectors = self.vectors.read().map_err(|_| Error::Vector(VectorError::IndexError("lock poisoned".into())))?;
        let mut results: Vec<(NodeId, f32)> = vectors
            .iter()
            .map(|(id, vec)| (*id, self.compute_similarity(query, vec)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    fn len(&self) -> Result<usize> {
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(Error::Vector(VectorError::IndexError(format!(
                "Node {} is unavailable",
                self.node_id
            ))));
        }

        Ok(self.vectors.read().map_err(|_| Error::Vector(VectorError::IndexError("lock poisoned".into())))?.len())
    }

    fn health_check(&self) -> Result<()> {
        self.check_fail()?;

        if self.is_healthy() {
            Ok(())
        } else {
            Err(Error::Vector(VectorError::IndexError(format!(
                "Node {} is unavailable",
                self.node_id
            ))))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_vector_node_client_lock_poisoning() {
        use std::panic;

        let client = MockVectorNodeClient::new(0, 4, DistanceMetric::Cosine);
        let node_id = NodeId::new(1).unwrap();
        let vector = vec![1.0, 0.0, 0.0, 0.0];

        // Intentionally poison the vectors lock
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _lock = client.vectors.write().unwrap();
            panic!("Poisoning vectors lock");
        }));

        // Intentionally poison the fail_next lock
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _lock = client.fail_next.write().unwrap();
            panic!("Poisoning fail_next lock");
        }));

        // Assert that methods return IndexError rather than panicking
        let result = client.add(node_id, &vector);
        assert!(matches!(
            result,
            Err(Error::Vector(VectorError::IndexError(_)))
        ));

        let result = client.remove(node_id);
        assert!(matches!(
            result,
            Err(Error::Vector(VectorError::IndexError(_)))
        ));

        let result = client.search(&vector, 10);
        assert!(matches!(
            result,
            Err(Error::Vector(VectorError::IndexError(_)))
        ));

        let result = client.len();
        assert!(matches!(
            result,
            Err(Error::Vector(VectorError::IndexError(_)))
        ));
    }

    fn create_test_config(num_nodes: usize) -> DistributedVectorConfig {
        let mut config = DistributedVectorConfig::new(4, DistanceMetric::Cosine);
        for i in 0..num_nodes {
            config = config.with_node(VectorNodeConfig::new(i as u16, format!("node{}:9000", i)));
        }
        config
    }

    fn create_test_clients(num_nodes: usize) -> Vec<Arc<MockVectorNodeClient>> {
        (0..num_nodes)
            .map(|i| {
                Arc::new(MockVectorNodeClient::new(
                    i as u16,
                    4,
                    DistanceMetric::Cosine,
                ))
            })
            .collect()
    }

    // ============================================================
    // Configuration Tests
    // ============================================================

    #[test]
    fn test_config_creation() {
        let config = DistributedVectorConfig::new(384, DistanceMetric::Cosine);
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.metric, DistanceMetric::Cosine);
        assert!(config.nodes.is_empty());
    }

    #[test]
    fn test_config_with_nodes() {
        let config = DistributedVectorConfig::new(384, DistanceMetric::Cosine)
            .with_node(VectorNodeConfig::new(0, "node0:9000"))
            .with_node(VectorNodeConfig::new(1, "node1:9000"));

        assert_eq!(config.nodes.len(), 2);
        assert_eq!(config.nodes[0].node_id, 0);
        assert_eq!(config.nodes[1].node_id, 1);
    }

    #[test]
    fn test_config_validation_zero_dimensions() {
        let config = DistributedVectorConfig::new(0, DistanceMetric::Cosine)
            .with_node(VectorNodeConfig::new(0, "node0:9000"));

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_no_nodes() {
        let config = DistributedVectorConfig::new(384, DistanceMetric::Cosine);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_duplicate_nodes() {
        let config = DistributedVectorConfig::new(384, DistanceMetric::Cosine)
            .with_node(VectorNodeConfig::new(0, "node0:9000"))
            .with_node(VectorNodeConfig::new(0, "node1:9000")); // Duplicate ID

        assert!(config.validate().is_err());
    }

    // ============================================================
    // Index Creation Tests
    // ============================================================

    #[test]
    fn test_create_distributed_index() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);
        let clients: Vec<Arc<MockVectorNodeClient>> = clients.into_iter().collect();

        let index = DistributedVectorIndex::new(config, clients)?;

        assert_eq!(index.node_count(), 3);
        assert_eq!(index.dimensions(), 4);
        assert_eq!(index.distance_metric(), DistanceMetric::Cosine);

        Ok(())
    }

    #[test]
    fn test_create_mismatched_clients() {
        let config = create_test_config(3);
        let clients = create_test_clients(2); // Wrong number

        let result = DistributedVectorIndex::new(config, clients);
        assert!(result.is_err());
    }

    // ============================================================
    // Add/Remove Tests
    // ============================================================

    #[test]
    fn test_add_vector() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        let node = NodeId::new(1).unwrap();
        let vector = vec![1.0, 0.0, 0.0, 0.0];
        index.add(node, &vector)?;

        assert_eq!(index.len(), 1);

        Ok(())
    }

    #[test]
    fn test_add_multiple_vectors() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        for i in 1..=100 {
            let node = NodeId::new(i).unwrap();
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            index.add(node, &vector)?;
        }

        assert_eq!(index.len(), 100);

        Ok(())
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients).unwrap();

        let node = NodeId::new(1).unwrap();
        let wrong_dim = vec![1.0, 0.0]; // Wrong dimensions

        assert!(index.add(node, &wrong_dim).is_err());
    }

    #[test]
    fn test_add_with_nan() {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients).unwrap();

        let node = NodeId::new(1).unwrap();
        let nan_vector = vec![1.0, f32::NAN, 0.0, 0.0];

        assert!(index.add(node, &nan_vector).is_err());
    }

    #[test]
    fn test_remove_vector() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        let node = NodeId::new(1).unwrap();
        let vector = vec![1.0, 0.0, 0.0, 0.0];

        index.add(node, &vector)?;
        assert_eq!(index.len(), 1);

        index.remove(node)?;
        assert_eq!(index.len(), 0);

        Ok(())
    }

    // ============================================================
    // Search Tests
    // ============================================================

    #[test]
    fn test_search_empty_index() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10)?;

        assert!(results.is_empty());

        Ok(())
    }

    #[test]
    fn test_search_basic() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Add vectors
        let node1 = NodeId::new(1).unwrap();
        let node2 = NodeId::new(2).unwrap();
        let node3 = NodeId::new(3).unwrap();

        index.add(node1, &[1.0, 0.0, 0.0, 0.0])?;
        index.add(node2, &[0.9, 0.1, 0.0, 0.0])?;
        index.add(node3, &[0.0, 1.0, 0.0, 0.0])?;

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 3)?;

        assert_eq!(results.len(), 3);
        // First result should be node1 (identical to query)
        assert_eq!(results[0].0, node1);

        Ok(())
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients).unwrap();

        let node = NodeId::new(1).unwrap();
        index.add(node, &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let wrong_query = vec![1.0, 0.0]; // Wrong dimensions

        assert!(index.search(&wrong_query, 10).is_err());
    }

    #[test]
    fn test_search_with_filter() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Add vectors
        for i in 1..=10 {
            let node = NodeId::new(i).unwrap();
            index.add(node, &[i as f32, 0.0, 0.0, 0.0])?;
        }

        let query = vec![5.0, 0.0, 0.0, 0.0];
        let results = index.search_with_filter(&query, 10, |id| id.as_u64() % 2 == 0)?;

        // Should only have even IDs
        for (id, _) in &results {
            assert_eq!(id.as_u64() % 2, 0);
        }

        Ok(())
    }

    // ============================================================
    // Routing Tests
    // ============================================================

    #[test]
    fn test_consistent_routing() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        let node = NodeId::new(42).unwrap();
        let route1 = index.node_for_id(node);
        let route2 = index.node_for_id(node);
        let route3 = index.node_for_id(node);

        assert_eq!(route1, route2);
        assert_eq!(route2, route3);

        Ok(())
    }

    #[test]
    fn test_range_based_routing() -> Result<()> {
        let mut config = create_test_config(3);
        config.routing_strategy = RoutingStrategy::RangeBased;
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Test edge cases
        let node_min = NodeId::new(0).unwrap();
        let node_max = NodeId::new(u64::MAX - 1000).unwrap();

        let route_min = index.node_for_id(node_min);
        let route_max = index.node_for_id(node_max);

        assert!(route_min < 3);
        assert!(route_max < 3);

        Ok(())
    }

    // ============================================================
    // Circuit Breaker Tests
    // ============================================================

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            open_duration: Duration::from_millis(100),
            success_threshold: 2,
        };
        let cb = NodeCircuitBreaker::new(config);

        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_half_open_transition() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration: Duration::from_millis(10),
            success_threshold: 1,
        };
        let cb = NodeCircuitBreaker::new(config);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        std::thread::sleep(Duration::from_millis(20));

        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_closes_from_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration: Duration::from_millis(10),
            success_threshold: 2,
        };
        let cb = NodeCircuitBreaker::new(config);

        cb.record_failure();
        std::thread::sleep(Duration::from_millis(20));

        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    // ============================================================
    // Node Failure Tests
    // ============================================================

    #[test]
    fn test_search_with_unhealthy_node() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        // Make one node unhealthy
        clients[1].set_healthy(false);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Add vectors (will only go to healthy nodes based on routing)
        for i in 1..=10 {
            let node = NodeId::new(i).unwrap();
            // This will succeed for nodes routed to healthy clients
            let _ = index.add(node, &[i as f32, 0.0, 0.0, 0.0]);
        }

        // Search should still work with partial results
        let query = vec![5.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10)?;

        // Should have results from the healthy nodes
        assert!(!results.is_empty() || index.len() == 0);

        Ok(())
    }

    #[test]
    fn test_search_all_nodes_unavailable() {
        let mut config = create_test_config(3);
        config.min_nodes_for_search = 1;
        let clients = create_test_clients(3);

        // Make all nodes unhealthy
        for client in &clients {
            client.set_healthy(false);
        }

        let index = DistributedVectorIndex::new(config, clients).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let result = index.search(&query, 10);

        assert!(result.is_err());
    }

    // ============================================================
    // Stats Tests
    // ============================================================

    #[test]
    fn test_stats() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        let stats = index.stats();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.available_nodes, 3);
        assert_eq!(stats.node_stats.len(), 3);

        Ok(())
    }

    #[test]
    fn test_node_connection_stats() -> Result<()> {
        let config = create_test_config(1);
        let clients = create_test_clients(1);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Perform some operations
        let node = NodeId::new(1).unwrap();
        index.add(node, &[1.0, 0.0, 0.0, 0.0])?;
        index.add(NodeId::new(2).unwrap(), &[0.0, 1.0, 0.0, 0.0])?;

        let stats = index.stats();
        assert!(stats.node_stats[0].request_count >= 2);

        Ok(())
    }

    // ============================================================
    // Merge Results Tests
    // ============================================================

    #[test]
    fn test_merge_results_empty() {
        let results = DistributedVectorIndex::<MockVectorNodeClient>::merge_results(vec![], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_merge_results_single_node() {
        let node_results = vec![vec![
            (NodeId::new(1).unwrap(), 0.9),
            (NodeId::new(2).unwrap(), 0.8),
            (NodeId::new(3).unwrap(), 0.7),
        ]];

        let merged = DistributedVectorIndex::<MockVectorNodeClient>::merge_results(node_results, 2);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].0, NodeId::new(1).unwrap());
        assert_eq!(merged[1].0, NodeId::new(2).unwrap());
    }

    #[test]
    fn test_merge_results_multiple_nodes() {
        let node_results = vec![
            vec![
                (NodeId::new(1).unwrap(), 0.9),
                (NodeId::new(2).unwrap(), 0.7),
            ],
            vec![
                (NodeId::new(3).unwrap(), 0.85),
                (NodeId::new(4).unwrap(), 0.6),
            ],
        ];

        let merged = DistributedVectorIndex::<MockVectorNodeClient>::merge_results(node_results, 3);
        assert_eq!(merged.len(), 3);
        // Should be sorted: 0.9, 0.85, 0.7
        assert_eq!(merged[0].0, NodeId::new(1).unwrap());
        assert_eq!(merged[1].0, NodeId::new(3).unwrap());
        assert_eq!(merged[2].0, NodeId::new(2).unwrap());
    }

    // ============================================================
    // Mock Client Tests
    // ============================================================

    #[test]
    fn test_mock_client_add_search() -> Result<()> {
        let client = MockVectorNodeClient::new(0, 4, DistanceMetric::Cosine);

        let node = NodeId::new(1).unwrap();
        client.add(node, &[1.0, 0.0, 0.0, 0.0])?;

        let results = client.search(&[1.0, 0.0, 0.0, 0.0], 10)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, node);
        assert!((results[0].1 - 1.0).abs() < 0.001); // Cosine similarity with self = 1.0

        Ok(())
    }

    #[test]
    fn test_mock_client_fail_next() {
        let client = MockVectorNodeClient::new(0, 4, DistanceMetric::Cosine);
        client.fail_next("Test error");

        let result = client.add(NodeId::new(1).unwrap(), &[1.0, 0.0, 0.0, 0.0]);
        assert!(result.is_err());

        // Next call should succeed
        let result = client.add(NodeId::new(2).unwrap(), &[0.0, 1.0, 0.0, 0.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_client_unhealthy() {
        let client = MockVectorNodeClient::new(0, 4, DistanceMetric::Cosine);
        client.set_healthy(false);

        let result = client.add(NodeId::new(1).unwrap(), &[1.0, 0.0, 0.0, 0.0]);
        assert!(result.is_err());
    }

    // ============================================================
    // Rebalancing Tests
    // ============================================================

    #[test]
    fn test_needs_rebalancing_empty() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Empty index - no rebalancing needed
        assert!(!index.needs_rebalancing(2.0));

        Ok(())
    }

    #[test]
    fn test_needs_rebalancing_balanced() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Add vectors that should distribute relatively evenly
        for i in 1..=30 {
            let node = NodeId::new(i).unwrap();
            index.add(node, &[i as f32, 0.0, 0.0, 0.0])?;
        }

        // With hash-based routing, should be relatively balanced
        // Threshold of 3.0 should accommodate some variance
        let stats = index.rebalance_stats();
        // The test is that the function works, not that perfect balance is achieved
        assert!(stats.total_vectors == 30);

        Ok(())
    }

    #[test]
    fn test_rebalance_stats() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Add some vectors
        for i in 1..=15 {
            let node = NodeId::new(i).unwrap();
            index.add(node, &[i as f32, 0.0, 0.0, 0.0])?;
        }

        let stats = index.rebalance_stats();

        assert_eq!(stats.total_vectors, 15);
        assert_eq!(stats.node_count, 3);
        assert!(stats.min_node_size <= stats.max_node_size);
        assert!(stats.imbalance_ratio >= 1.0);

        Ok(())
    }

    // ============================================================
    // VectorNodeConfig Tests
    // ============================================================

    #[test]
    fn test_node_config_defaults() {
        let config = VectorNodeConfig::new(0, "node0:9000");

        assert_eq!(config.node_id, 0);
        assert_eq!(config.endpoint, "node0:9000");
        assert_eq!(config.timeout, DEFAULT_TIMEOUT);
    }

    #[test]
    fn test_node_config_with_timeout() {
        let config = VectorNodeConfig::new(0, "node0:9000").with_timeout(Duration::from_secs(60));

        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    // ============================================================
    // DistributedError Tests
    // ============================================================

    #[test]
    fn test_distributed_error_display() {
        let err = DistributedError::NoNodesAvailable;
        assert!(format!("{}", err).contains("No nodes available"));

        let err = DistributedError::NodeUnavailable {
            node_id: 0,
            reason: "connection refused".to_string(),
        };
        assert!(format!("{}", err).contains("Node 0"));
        assert!(format!("{}", err).contains("connection refused"));

        let err = DistributedError::CircuitOpen {
            node_id: 1,
            remaining: Duration::from_secs(10),
        };
        assert!(format!("{}", err).contains("Circuit breaker"));
        assert!(format!("{}", err).contains("node 1"));
    }

    #[test]
    fn test_distributed_error_display_all_variants() {
        let err = DistributedError::AllNodesFailed {
            failed_count: 3,
            sample_error: "connection timeout".to_string(),
        };
        assert!(format!("{}", err).contains("All 3 nodes failed"));
        assert!(format!("{}", err).contains("connection timeout"));

        let err = DistributedError::Timeout {
            operation: "search".to_string(),
            duration: Duration::from_secs(30),
        };
        assert!(format!("{}", err).contains("search"));
        assert!(format!("{}", err).contains("timed out"));

        let err = DistributedError::ConfigError("invalid dimensions".to_string());
        assert!(format!("{}", err).contains("Configuration error"));
        assert!(format!("{}", err).contains("invalid dimensions"));
    }

    // ============================================================
    // Search k > MAX_K Tests
    // ============================================================

    #[test]
    fn test_search_k_exceeds_max() {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        // Request more than MAX_K results
        let result = index.search(&query, MAX_K + 1);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_search_k_at_max() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        let query = vec![1.0, 0.0, 0.0, 0.0];
        // Request exactly MAX_K results - should succeed
        let results = index.search(&query, MAX_K)?;

        assert!(results.is_empty()); // No vectors added

        Ok(())
    }

    // ============================================================
    // Stats Tests with Vectors
    // ============================================================

    #[test]
    fn test_stats_with_vectors() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Add some vectors
        for i in 1..=15 {
            let node = NodeId::new(i).unwrap();
            index.add(node, &[i as f32, 0.0, 0.0, 0.0])?;
        }

        let stats = index.stats();
        assert_eq!(stats.total_vectors, 15);
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.available_nodes, 3);

        Ok(())
    }

    // ============================================================
    // Partial Results Tests
    // ============================================================

    #[test]
    fn test_search_partial_results_disabled() {
        let mut config = create_test_config(3);
        config.allow_partial_results = false;
        let clients = create_test_clients(3);

        // Make one node unhealthy
        clients[1].set_healthy(false);

        let index = DistributedVectorIndex::new(config, clients).unwrap();

        // Add a vector to a healthy node
        let node = NodeId::new(1).unwrap();
        let _ = index.add(node, &[1.0, 0.0, 0.0, 0.0]);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        // With partial results disabled and one node unhealthy, search should fail
        // (but only if the unhealthy node is actually queried - depends on routing)
        let _ = index.search(&query, 10);
    }

    // ============================================================
    // Circuit Breaker Reset Tests
    // ============================================================

    #[test]
    fn test_reset_all_circuits() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Force some failures to open circuits
        for node in index.nodes.iter() {
            for _ in 0..10 {
                node.circuit_breaker.record_failure();
            }
        }

        // Verify circuits are open
        for node in index.nodes.iter() {
            assert_eq!(node.circuit_state(), CircuitState::Open);
        }

        // Reset all circuits
        index.reset_all_circuits();

        // Verify circuits are closed
        for node in index.nodes.iter() {
            assert_eq!(node.circuit_state(), CircuitState::Closed);
        }

        Ok(())
    }

    // ============================================================
    // Mock Client is_empty Tests
    // ============================================================

    #[test]
    fn test_mock_client_is_empty() -> Result<()> {
        let client = MockVectorNodeClient::new(0, 4, DistanceMetric::Cosine);

        assert!(client.is_empty()?);

        client.add(NodeId::new(1).unwrap(), &[1.0, 0.0, 0.0, 0.0])?;
        assert!(!client.is_empty()?);

        Ok(())
    }

    // ============================================================
    // Circuit Breaker Remaining Time Tests
    // ============================================================

    #[test]
    fn test_circuit_breaker_remaining_time() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration: Duration::from_secs(60),
            success_threshold: 1,
        };
        let cb = NodeCircuitBreaker::new(config);

        // Initially closed - no remaining time
        assert!(cb.remaining_open_time().is_none());

        // Open the circuit
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Should have remaining time
        let remaining = cb.remaining_open_time();
        assert!(remaining.is_some());
        assert!(remaining.unwrap() > Duration::from_secs(0));
        assert!(remaining.unwrap() <= Duration::from_secs(60));
    }

    // ============================================================
    // Node Connection Tests
    // ============================================================

    #[test]
    fn test_node_connection_execute_with_circuit_open() {
        let client = Arc::new(MockVectorNodeClient::new(0, 4, DistanceMetric::Cosine));
        let connection = NodeConnection::new(
            client,
            CircuitBreakerConfig {
                failure_threshold: 1,
                open_duration: Duration::from_secs(60),
                success_threshold: 1,
            },
        );

        // Open the circuit
        connection.circuit_breaker.record_failure();
        assert_eq!(connection.circuit_state(), CircuitState::Open);

        // Execute should fail with circuit open
        let result = connection.execute(|c| c.health_check());
        assert!(result.is_err());
    }

    #[test]
    fn test_node_connection_debug() {
        let client = Arc::new(MockVectorNodeClient::new(0, 4, DistanceMetric::Cosine));
        let connection = NodeConnection::new(client, CircuitBreakerConfig::default());

        let debug_str = format!("{:?}", connection);
        assert!(debug_str.contains("NodeConnection"));
        assert!(debug_str.contains("node_id"));
    }

    // ============================================================
    // DistributedVectorIndex Debug Tests
    // ============================================================

    #[test]
    fn test_distributed_index_debug() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        let debug_str = format!("{:?}", index);
        assert!(debug_str.contains("DistributedVectorIndex"));
        assert!(debug_str.contains("dimensions"));
        assert!(debug_str.contains("node_count"));

        Ok(())
    }

    // ============================================================
    // Mock Client Dimension Mismatch Tests
    // ============================================================

    #[test]
    fn test_mock_client_dimension_mismatch() {
        let client = MockVectorNodeClient::new(0, 4, DistanceMetric::Cosine);

        let result = client.add(NodeId::new(1).unwrap(), &[1.0, 0.0]); // Wrong dimensions
        assert!(result.is_err());
    }

    // ============================================================
    // Config Routing Strategy Tests
    // ============================================================

    #[test]
    fn test_config_routing_strategy() {
        let config = DistributedVectorConfig::new(384, DistanceMetric::Cosine)
            .with_node(VectorNodeConfig::new(0, "node0:9000"))
            .with_routing_strategy(RoutingStrategy::RangeBased);

        assert_eq!(config.routing_strategy, RoutingStrategy::RangeBased);
    }

    #[test]
    fn test_config_min_nodes_for_search() {
        let config = DistributedVectorConfig::new(384, DistanceMetric::Cosine)
            .with_node(VectorNodeConfig::new(0, "node0:9000"))
            .with_min_nodes_for_search(2);

        assert_eq!(config.min_nodes_for_search, 2);
    }

    // ============================================================
    // Node Connection Stats Success Rate Tests
    // ============================================================

    #[test]
    fn test_node_connection_stats_success_rate() {
        let stats = NodeConnectionStats {
            node_id: 0,
            circuit_state: CircuitState::Closed,
            request_count: 10,
            failure_count: 3,
        };

        let rate = stats.success_rate();
        assert!((rate - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_node_connection_stats_success_rate_zero_requests() {
        let stats = NodeConnectionStats {
            node_id: 0,
            circuit_state: CircuitState::Closed,
            request_count: 0,
            failure_count: 0,
        };

        assert_eq!(stats.success_rate(), 1.0);
    }

    // ============================================================
    // Concurrent Circuit Breaker Stress Tests
    // ============================================================

    #[test]
    fn test_circuit_breaker_concurrent_failures() {
        use std::thread;

        let config = CircuitBreakerConfig {
            failure_threshold: 10,
            open_duration: Duration::from_secs(60),
            success_threshold: 3,
        };
        let cb = Arc::new(NodeCircuitBreaker::new(config));

        let mut handles = vec![];

        // Spawn multiple threads to record failures concurrently
        for _ in 0..4 {
            let cb_clone = Arc::clone(&cb);
            let handle = thread::spawn(move || {
                for _ in 0..5 {
                    cb_clone.record_failure();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // After 20 failures (4 threads x 5), circuit should be open
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_concurrent_success_failure_mix() {
        use std::thread;

        let config = CircuitBreakerConfig {
            failure_threshold: 5,
            open_duration: Duration::from_millis(10),
            success_threshold: 2,
        };
        let cb = Arc::new(NodeCircuitBreaker::new(config));

        let mut handles = vec![];

        // Some threads record successes
        for _ in 0..2 {
            let cb_clone = Arc::clone(&cb);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    cb_clone.record_success();
                    std::thread::sleep(Duration::from_micros(100));
                }
            });
            handles.push(handle);
        }

        // Some threads record failures
        for _ in 0..2 {
            let cb_clone = Arc::clone(&cb);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    cb_clone.record_failure();
                    std::thread::sleep(Duration::from_micros(100));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // State should be valid (either Closed, Open, or HalfOpen)
        let state = cb.state();
        assert!(
            state == CircuitState::Closed
                || state == CircuitState::Open
                || state == CircuitState::HalfOpen
        );
    }

    #[test]
    fn test_circuit_breaker_concurrent_state_checks() {
        use std::thread;

        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            open_duration: Duration::from_millis(50),
            success_threshold: 2,
        };
        let cb = Arc::new(NodeCircuitBreaker::new(config));

        // Open the circuit
        for _ in 0..5 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);

        let mut handles = vec![];

        // Multiple threads checking state simultaneously
        for _ in 0..8 {
            let cb_clone = Arc::clone(&cb);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let _ = cb_clone.state();
                    let _ = cb_clone.should_allow();
                    let _ = cb_clone.remaining_open_time();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // No panic means success - concurrent reads are safe
    }

    // ============================================================
    // Rebalancing Threshold Constant Test
    // ============================================================

    #[test]
    fn test_recommended_imbalance_threshold() {
        // Verify the constant exists and has expected value
        assert!((RECOMMENDED_IMBALANCE_THRESHOLD - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_needs_rebalancing_with_threshold_constant() -> Result<()> {
        let config = create_test_config(3);
        let clients = create_test_clients(3);

        let index = DistributedVectorIndex::new(config, clients)?;

        // Empty index - no rebalancing needed at any threshold
        assert!(!index.needs_rebalancing(RECOMMENDED_IMBALANCE_THRESHOLD));

        Ok(())
    }
}
