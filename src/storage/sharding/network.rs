//! Network abstraction layer for shard-to-shard communication.
//!
//! This module provides traits and implementations for network communication
//! between shards, including connection pooling and circuit breakers.

// Allow nested if statements - they're sometimes clearer
#![allow(clippy::collapsible_if)]

use super::types::{ShardId, ShardState};
use crate::core::hlc::HybridTimestamp;
use crate::core::id::{EdgeId, NodeId, TxId};
use parking_lot::RwLock as PlRwLock;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Error types for network operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum NetworkError {
    /// Connection failed.
    ConnectionFailed { shard_id: ShardId, reason: String },
    /// Request timed out.
    Timeout {
        shard_id: ShardId,
        operation: String,
        duration: Duration,
    },
    /// Circuit breaker is open.
    CircuitOpen {
        shard_id: ShardId,
        remaining: Duration,
    },
    /// Shard is unavailable.
    ShardUnavailable(ShardId),
    /// Serialization error.
    SerializationError(String),
    /// Protocol error.
    ProtocolError(String),
    /// Connection pool exhausted.
    PoolExhausted {
        shard_id: ShardId,
        max_connections: usize,
    },
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkError::ConnectionFailed { shard_id, reason } => {
                write!(f, "Connection to {} failed: {}", shard_id, reason)
            }
            NetworkError::Timeout {
                shard_id,
                operation,
                duration,
            } => {
                write!(
                    f,
                    "Operation '{}' on {} timed out after {:?}",
                    operation, shard_id, duration
                )
            }
            NetworkError::CircuitOpen {
                shard_id,
                remaining,
            } => {
                write!(
                    f,
                    "Circuit breaker open for {}, {} remaining",
                    shard_id,
                    remaining.as_secs()
                )
            }
            NetworkError::ShardUnavailable(shard_id) => {
                write!(f, "Shard {} is unavailable", shard_id)
            }
            NetworkError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            NetworkError::ProtocolError(msg) => {
                write!(f, "Protocol error: {}", msg)
            }
            NetworkError::PoolExhausted {
                shard_id,
                max_connections,
            } => {
                write!(
                    f,
                    "Connection pool exhausted for {} (max: {})",
                    shard_id, max_connections
                )
            }
        }
    }
}

impl std::error::Error for NetworkError {}

/// Result type for network operations.
pub type NetworkResult<T> = Result<T, NetworkError>;

/// Response from a prepare request.
#[derive(Debug, Clone)]
pub struct PrepareResponse {
    /// Whether the shard is ready to commit.
    pub ready: bool,
    /// Optional reason if not ready.
    pub reason: Option<String>,
}

/// Response from a commit request.
#[derive(Debug, Clone)]
pub struct CommitResponse {
    /// Whether the commit succeeded.
    pub success: bool,
}

/// Response from an abort request.
#[derive(Debug, Clone)]
pub struct AbortResponse {
    /// Whether the abort was acknowledged.
    pub acknowledged: bool,
}

/// Data for a single node being migrated.
#[derive(Debug, Clone)]
pub struct NodeData {
    /// Node ID.
    pub id: NodeId,
    /// Node label.
    pub label: String,
    /// Serialized properties.
    pub properties: Vec<u8>,
    /// Valid time start.
    pub valid_from: u64,
    /// Valid time end (None = current).
    pub valid_to: Option<u64>,
}

/// Data for a single edge being migrated.
#[derive(Debug, Clone)]
pub struct EdgeData {
    /// Edge ID.
    pub id: EdgeId,
    /// Source node ID.
    pub source: NodeId,
    /// Target node ID.
    pub target: NodeId,
    /// Edge label.
    pub label: String,
    /// Serialized properties.
    pub properties: Vec<u8>,
    /// Valid time start.
    pub valid_from: u64,
    /// Valid time end (None = current).
    pub valid_to: Option<u64>,
}

/// Batch of migration data.
#[derive(Debug, Clone)]
pub struct MigrationBatch {
    /// Migration ID.
    pub migration_id: u64,
    /// Batch sequence number.
    pub batch_number: u64,
    /// Is this the last batch?
    pub is_last: bool,
    /// Nodes in this batch.
    pub nodes: Vec<NodeData>,
    /// Edges in this batch.
    pub edges: Vec<EdgeData>,
    /// Checksum for verification.
    pub checksum: u64,
}

/// Response from receiving migration data.
#[derive(Debug, Clone)]
pub struct MigrationResponse {
    /// Whether the batch was accepted.
    pub accepted: bool,
    /// Number of nodes written.
    pub nodes_written: u64,
    /// Number of edges written.
    pub edges_written: u64,
    /// Error message if not accepted.
    pub error: Option<String>,
}

/// Trait for shard-to-shard communication.
///
/// This trait abstracts network communication, allowing for different
/// implementations (gRPC, HTTP, in-process for testing).
///
/// # ⚠️ Blocking API
///
/// All methods in this trait are **synchronous and blocking**.
///
/// - Implementations are expected to block the current thread until the network
///   operation completes or times out.
/// - Callers (like `QueryExecutor`) will block while waiting for responses.
/// - This simplifies error handling and state management but limits concurrency.
///
/// For high-throughput scenarios, this trait will eventually be migrated to `async`.
pub trait ShardClient: Send + Sync + fmt::Debug {
    /// Get the shard ID this client connects to.
    fn shard_id(&self) -> ShardId;

    /// Check if the connection is healthy.
    fn is_healthy(&self) -> bool;

    /// Send a prepare request for 2PC.
    fn prepare(
        &self,
        tx_id: TxId,
        operations: &[u8],
        timestamp: Option<HybridTimestamp>,
    ) -> NetworkResult<PrepareResponse>;

    /// Send a commit request for 2PC.
    fn commit(
        &self,
        tx_id: TxId,
        timestamp: Option<HybridTimestamp>,
    ) -> NetworkResult<CommitResponse>;

    /// Send an abort request for 2PC.
    fn abort(&self, tx_id: TxId) -> NetworkResult<AbortResponse>;

    /// Execute a read query on the shard.
    fn query(&self, query_id: u64, query_data: &[u8]) -> NetworkResult<Vec<u8>>;

    /// Get the current state of the shard.
    fn get_state(&self) -> NetworkResult<ShardState>;

    /// Send migration data to this shard.
    fn receive_migration_batch(&self, batch: MigrationBatch) -> NetworkResult<MigrationResponse>;

    /// Extract data from this shard for migration.
    fn extract_migration_batch(
        &self,
        migration_id: u64,
        labels: &[String],
        batch_size: usize,
        offset: u64,
    ) -> NetworkResult<MigrationBatch>;

    /// Perform a health check.
    fn health_check(&self) -> NetworkResult<()>;
}

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
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit.
    pub failure_threshold: usize,
    /// Duration to keep circuit open.
    pub open_duration: Duration,
    /// Number of successes in half-open to close circuit.
    pub success_threshold: usize,
    /// Window for counting failures.
    pub failure_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            open_duration: Duration::from_secs(30),
            success_threshold: 3,
            failure_window: Duration::from_secs(60),
        }
    }
}

/// Circuit breaker for a single shard connection.
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    // These three locks are `parking_lot::RwLock` (task-fair queuing) rather
    // than `std::sync::RwLock` (unspecified fairness). Combined with the
    // structural rule below -- `state` and `opened_at` are never held nested
    // in either order -- this guarantees the lock-inversion deadlock reported
    // in issue #3229 cannot occur regardless of reader/writer scheduling.
    state: PlRwLock<CircuitState>,
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    last_failure: PlRwLock<Option<Instant>>,
    opened_at: PlRwLock<Option<Instant>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: PlRwLock::new(CircuitState::Closed),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            last_failure: PlRwLock::new(None),
            opened_at: PlRwLock::new(None),
        }
    }

    /// Get the current state.
    pub fn state(&self) -> CircuitState {
        self.maybe_transition();
        *self.state.read()
    }

    /// Check if requests should be allowed.
    pub fn should_allow(&self) -> bool {
        self.maybe_transition();
        let state = *self.state.read();
        matches!(state, CircuitState::Closed | CircuitState::HalfOpen)
    }

    /// Record a successful request.
    pub fn record_success(&self) {
        // Snapshot the state and release the read guard immediately.
        let state = *self.state.read();

        match state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::SeqCst);
            }
            CircuitState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                if successes >= self.config.success_threshold {
                    // Transition to closed
                    *self.state.write() = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::SeqCst);
                    self.success_count.store(0, Ordering::SeqCst);
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but ignore
            }
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        // Snapshot the state and release the read guard immediately.
        let state = *self.state.read();

        match state {
            CircuitState::Closed => {
                // Window-expiry check + last_failure update must be atomic:
                // take the `last_failure` write guard ONCE and do the whole
                // check-then-act under it, so concurrent resets can't lose
                // increments (failures within the window accumulate; an
                // expired window resets before this failure counts as 1).
                {
                    let mut last_failure = self.last_failure.write();
                    if let Some(last_time) = *last_failure {
                        if last_time.elapsed() > self.config.failure_window {
                            self.failure_count.store(0, Ordering::SeqCst);
                        }
                    }
                    *last_failure = Some(Instant::now());
                }

                let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;

                if failures >= self.config.failure_threshold {
                    // Re-check the state under the write guard so only the
                    // first tripping thread stamps opened_at (concurrent trips
                    // don't keep pushing it forward). Decide the transition
                    // while holding `state`, capture whether we opened, then
                    // release `state` BEFORE touching `opened_at` -- the two
                    // locks are never held nested (issue #3229 deadlock fix).
                    let did_open = {
                        let mut s = self.state.write();
                        if *s == CircuitState::Closed {
                            *s = CircuitState::Open;
                            true
                        } else {
                            false
                        }
                    };
                    if did_open {
                        *self.opened_at.write() = Some(Instant::now());
                    }
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open goes back to open. Re-check the
                // state under the write guard so a stale HalfOpen snapshot
                // can't trip a breaker that concurrently closed. Decide the
                // transition while holding `state`, capture whether we opened,
                // then release `state` BEFORE touching `opened_at` -- never
                // held nested (issue #3229 deadlock fix).
                let did_open = {
                    let mut s = self.state.write();
                    if *s == CircuitState::HalfOpen {
                        *s = CircuitState::Open;
                        self.success_count.store(0, Ordering::SeqCst);
                        true
                    } else {
                        false
                    }
                };
                if did_open {
                    *self.opened_at.write() = Some(Instant::now());
                }
            }
            CircuitState::Open => {
                // Already open, update opened_at
                *self.opened_at.write() = Some(Instant::now());
            }
        }
    }

    /// Check and perform state transitions based on time.
    ///
    /// Deadlock-safety (issue #3229): this method must never hold `opened_at`
    /// and `state` at the same time. It snapshots the current state, releases
    /// that guard, then snapshots `opened_at` into a local and releases *that*
    /// guard before acquiring `state.write()`. Holding `opened_at.read()`
    /// across `state.write()` (the original ordering) closed a lock-inversion
    /// cycle with `remaining_open_time` (state -> opened_at) and
    /// `record_failure` (opened_at writer).
    fn maybe_transition(&self) {
        // Snapshot the state and release the read guard immediately.
        let state = *self.state.read();
        if state != CircuitState::Open {
            return;
        }

        // Snapshot `opened_at` into a local and release the guard BEFORE
        // acquiring `state.write()` -- never hold `opened_at` and `state`
        // nested in either order.
        let opened_time = *self.opened_at.read();
        let Some(opened_time) = opened_time else {
            return;
        };
        if opened_time.elapsed() < self.config.open_duration {
            return;
        }

        // Re-check the state under the write lock so we don't clobber a
        // concurrent transition (e.g. a `record_failure` that moved us
        // elsewhere between the snapshot above and acquiring this lock).
        let mut s = self.state.write();
        if *s == CircuitState::Open {
            *s = CircuitState::HalfOpen;
            self.success_count.store(0, Ordering::SeqCst);
        }
    }

    /// Get remaining time before circuit can close.
    ///
    /// Deadlock-safety (issue #3229): snapshots `state` and releases the read
    /// guard before touching `opened_at`, so `state` and `opened_at` are never
    /// held nested.
    pub fn remaining_open_time(&self) -> Option<Duration> {
        // Snapshot the state and release the read guard before touching
        // `opened_at`.
        let state = *self.state.read();
        if state != CircuitState::Open {
            return None;
        }

        if let Some(opened_time) = *self.opened_at.read() {
            let elapsed = opened_time.elapsed();
            if elapsed < self.config.open_duration {
                return Some(self.config.open_duration - elapsed);
            }
        }
        None
    }

    /// Reset the circuit breaker to closed state.
    pub fn reset(&self) {
        *self.state.write() = CircuitState::Closed;
        self.failure_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
    }
}

/// Configuration for connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum connections per shard.
    pub max_connections: usize,
    /// Minimum idle connections to maintain.
    pub min_idle: usize,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Idle timeout before closing connection.
    pub idle_timeout: Duration,
    /// Maximum lifetime of a connection.
    pub max_lifetime: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_idle: 2,
            connect_timeout: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(3600),
        }
    }
}

/// Connection pool for shard clients.
#[derive(Debug)]
pub struct ConnectionPool<C: ShardClient> {
    shard_id: ShardId,
    config: PoolConfig,
    connections: RwLock<Vec<Arc<C>>>,
    active_count: AtomicUsize,
    circuit_breaker: CircuitBreaker,
    total_requests: AtomicU64,
    failed_requests: AtomicU64,
}

impl<C: ShardClient> ConnectionPool<C> {
    /// Create a new connection pool.
    pub fn new(
        shard_id: ShardId,
        config: PoolConfig,
        circuit_config: CircuitBreakerConfig,
    ) -> Self {
        Self {
            shard_id,
            config,
            connections: RwLock::new(Vec::new()),
            active_count: AtomicUsize::new(0),
            circuit_breaker: CircuitBreaker::new(circuit_config),
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
        }
    }

    /// Get the shard ID.
    pub fn shard_id(&self) -> ShardId {
        self.shard_id
    }

    /// Check if the pool allows requests (circuit breaker).
    pub fn is_available(&self) -> bool {
        self.circuit_breaker.should_allow()
    }

    /// Get the circuit breaker state.
    pub fn circuit_state(&self) -> CircuitState {
        self.circuit_breaker.state()
    }

    /// Get a connection from the pool.
    pub fn get(&self) -> NetworkResult<Arc<C>> {
        // Check circuit breaker
        if !self.circuit_breaker.should_allow() {
            return Err(NetworkError::CircuitOpen {
                shard_id: self.shard_id,
                remaining: self
                    .circuit_breaker
                    .remaining_open_time()
                    .unwrap_or_default(),
            });
        }

        // Try to get an existing connection
        if let Ok(connections) = self.connections.read() {
            for conn in connections.iter() {
                if conn.is_healthy() {
                    self.active_count.fetch_add(1, Ordering::SeqCst);
                    return Ok(Arc::clone(conn));
                }
            }
        }

        // Check if we can create a new connection
        let active = self.active_count.load(Ordering::SeqCst);
        if active >= self.config.max_connections {
            return Err(NetworkError::PoolExhausted {
                shard_id: self.shard_id,
                max_connections: self.config.max_connections,
            });
        }

        // No available connection - caller needs to create one
        Err(NetworkError::ShardUnavailable(self.shard_id))
    }

    /// Return a connection to the pool.
    pub fn release(&self, _conn: Arc<C>) {
        self.active_count.fetch_sub(1, Ordering::SeqCst);
    }

    /// Add a connection to the pool.
    pub fn add(&self, conn: Arc<C>) {
        if let Ok(mut connections) = self.connections.write() {
            connections.push(conn);
        }
    }

    /// Record a successful operation.
    pub fn record_success(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.circuit_breaker.record_success();
    }

    /// Record a failed operation.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.circuit_breaker.record_failure();
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            shard_id: self.shard_id,
            total_connections: self.connections.read().map(|c| c.len()).unwrap_or(0),
            active_connections: self.active_count.load(Ordering::SeqCst),
            circuit_state: self.circuit_breaker.state(),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
        }
    }

    /// Reset the circuit breaker.
    pub fn reset_circuit(&self) {
        self.circuit_breaker.reset();
    }
}

/// Statistics for a connection pool.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Shard ID.
    pub shard_id: ShardId,
    /// Total connections in pool.
    pub total_connections: usize,
    /// Currently active connections.
    pub active_connections: usize,
    /// Circuit breaker state.
    pub circuit_state: CircuitState,
    /// Total requests made.
    pub total_requests: u64,
    /// Failed requests.
    pub failed_requests: u64,
}

impl PoolStats {
    /// Calculate success rate.
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            1.0
        } else {
            1.0 - (self.failed_requests as f64 / self.total_requests as f64)
        }
    }
}

/// Mock shard client for testing.
#[derive(Debug)]
pub struct MockShardClient {
    shard_id: ShardId,
    healthy: RwLock<bool>,
    prepare_response: RwLock<Option<PrepareResponse>>,
    commit_response: RwLock<Option<CommitResponse>>,
    abort_response: RwLock<Option<AbortResponse>>,
    query_response: RwLock<Vec<u8>>,
    state: RwLock<ShardState>,
    latency: RwLock<Duration>,
    fail_next: RwLock<Option<NetworkError>>,
    call_counts: RwLock<HashMap<String, usize>>,
    /// The last query data received by this mock client.
    pub query_received: RwLock<Option<Vec<u8>>>,
}

impl MockShardClient {
    /// Create a new mock client.
    pub fn new(shard_id: ShardId) -> Self {
        Self {
            shard_id,
            healthy: RwLock::new(true),
            prepare_response: RwLock::new(Some(PrepareResponse {
                ready: true,
                reason: None,
            })),
            commit_response: RwLock::new(Some(CommitResponse { success: true })),
            abort_response: RwLock::new(Some(AbortResponse { acknowledged: true })),
            query_response: RwLock::new(Vec::new()),
            state: RwLock::new(ShardState::new(shard_id)),
            latency: RwLock::new(Duration::from_micros(100)),
            fail_next: RwLock::new(None),
            call_counts: RwLock::new(HashMap::new()),
            query_received: RwLock::new(None),
        }
    }

    /// Set whether the client is healthy.
    pub fn set_healthy(&self, healthy: bool) {
        *self.healthy.write().unwrap() = healthy;
    }

    /// Set the prepare response.
    pub fn set_prepare_response(&self, response: PrepareResponse) {
        *self.prepare_response.write().unwrap() = Some(response);
    }

    /// Set the commit response.
    pub fn set_commit_response(&self, response: CommitResponse) {
        *self.commit_response.write().unwrap() = Some(response);
    }

    /// Set the abort response.
    pub fn set_abort_response(&self, response: AbortResponse) {
        *self.abort_response.write().unwrap() = Some(response);
    }

    /// Set the query response.
    pub fn set_query_response(&self, response: Vec<u8>) {
        *self.query_response.write().unwrap() = response;
    }

    /// Set the shard state.
    pub fn set_state(&self, state: ShardState) {
        *self.state.write().unwrap() = state;
    }

    /// Set simulated latency.
    pub fn set_latency(&self, latency: Duration) {
        *self.latency.write().unwrap() = latency;
    }

    /// Make the next call fail with the given error.
    pub fn fail_next(&self, error: NetworkError) {
        *self.fail_next.write().unwrap() = Some(error);
    }

    /// Get call count for a method.
    pub fn call_count(&self, method: &str) -> usize {
        self.call_counts
            .read()
            .unwrap()
            .get(method)
            .copied()
            .unwrap_or(0)
    }

    fn increment_call(&self, method: &str) {
        let mut counts = self.call_counts.write().unwrap();
        *counts.entry(method.to_string()).or_insert(0) += 1;
    }

    fn check_fail(&self) -> NetworkResult<()> {
        let mut fail = self.fail_next.write().unwrap();
        if let Some(err) = fail.take() {
            return Err(err);
        }
        Ok(())
    }

    fn simulate_latency(&self) {
        let latency = *self.latency.read().unwrap();
        if latency > Duration::ZERO {
            std::thread::sleep(latency);
        }
    }
}

impl ShardClient for MockShardClient {
    fn shard_id(&self) -> ShardId {
        self.shard_id
    }

    fn is_healthy(&self) -> bool {
        *self.healthy.read().unwrap()
    }

    fn prepare(
        &self,
        _tx_id: TxId,
        _operations: &[u8],
        _timestamp: Option<HybridTimestamp>,
    ) -> NetworkResult<PrepareResponse> {
        self.increment_call("prepare");
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(NetworkError::ShardUnavailable(self.shard_id));
        }

        self.simulate_latency();

        self.prepare_response
            .read()
            .unwrap()
            .clone()
            .ok_or(NetworkError::ProtocolError("No response configured".into()))
    }

    fn commit(
        &self,
        _tx_id: TxId,
        _timestamp: Option<HybridTimestamp>,
    ) -> NetworkResult<CommitResponse> {
        self.increment_call("commit");
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(NetworkError::ShardUnavailable(self.shard_id));
        }

        self.simulate_latency();

        self.commit_response
            .read()
            .unwrap()
            .clone()
            .ok_or(NetworkError::ProtocolError("No response configured".into()))
    }

    fn abort(&self, _tx_id: TxId) -> NetworkResult<AbortResponse> {
        self.increment_call("abort");
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(NetworkError::ShardUnavailable(self.shard_id));
        }

        self.simulate_latency();

        self.abort_response
            .read()
            .unwrap()
            .clone()
            .ok_or(NetworkError::ProtocolError("No response configured".into()))
    }

    fn query(&self, _query_id: u64, _query_data: &[u8]) -> NetworkResult<Vec<u8>> {
        self.increment_call("query");
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(NetworkError::ShardUnavailable(self.shard_id));
        }

        *self.query_received.write().unwrap() = Some(_query_data.to_vec());

        self.simulate_latency();
        Ok(self.query_response.read().unwrap().clone())
    }

    fn get_state(&self) -> NetworkResult<ShardState> {
        self.increment_call("get_state");
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(NetworkError::ShardUnavailable(self.shard_id));
        }

        self.simulate_latency();
        Ok(self.state.read().unwrap().clone())
    }

    fn receive_migration_batch(&self, batch: MigrationBatch) -> NetworkResult<MigrationResponse> {
        self.increment_call("receive_migration_batch");
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(NetworkError::ShardUnavailable(self.shard_id));
        }

        self.simulate_latency();
        Ok(MigrationResponse {
            accepted: true,
            nodes_written: batch.nodes.len() as u64,
            edges_written: batch.edges.len() as u64,
            error: None,
        })
    }

    fn extract_migration_batch(
        &self,
        migration_id: u64,
        _labels: &[String],
        batch_size: usize,
        offset: u64,
    ) -> NetworkResult<MigrationBatch> {
        self.increment_call("extract_migration_batch");
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(NetworkError::ShardUnavailable(self.shard_id));
        }

        self.simulate_latency();

        // Return empty batch (simulating end of data)
        Ok(MigrationBatch {
            migration_id,
            batch_number: offset / batch_size as u64,
            is_last: true,
            nodes: Vec::new(),
            edges: Vec::new(),
            checksum: 0,
        })
    }

    fn health_check(&self) -> NetworkResult<()> {
        self.increment_call("health_check");
        self.check_fail()?;

        if !self.is_healthy() {
            return Err(NetworkError::ShardUnavailable(self.shard_id));
        }

        self.simulate_latency();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shard_id(id: u16) -> ShardId {
        ShardId::new(id).unwrap()
    }

    // ============ NetworkError Tests ============

    #[test]
    fn test_network_error_display() {
        let err = NetworkError::ConnectionFailed {
            shard_id: make_shard_id(0),
            reason: "refused".to_string(),
        };
        assert!(format!("{}", err).contains("refused"));

        let err = NetworkError::Timeout {
            shard_id: make_shard_id(1),
            operation: "query".to_string(),
            duration: Duration::from_secs(30),
        };
        assert!(format!("{}", err).contains("query"));
        assert!(format!("{}", err).contains("timed out"));

        let err = NetworkError::CircuitOpen {
            shard_id: make_shard_id(2),
            remaining: Duration::from_secs(10),
        };
        assert!(format!("{}", err).contains("Circuit breaker"));

        let err = NetworkError::PoolExhausted {
            shard_id: make_shard_id(3),
            max_connections: 10,
        };
        assert!(format!("{}", err).contains("exhausted"));
    }

    // ============ CircuitBreaker Tests ============

    #[test]
    fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Record failures
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_success_resets_count() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.record_failure();
        cb.record_failure();
        cb.record_success(); // Should reset

        // These failures shouldn't trigger open
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure(); // Now it should open
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_half_open_transition() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration: Duration::from_millis(10),
            success_threshold: 2,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for open duration
        std::thread::sleep(Duration::from_millis(20));

        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_closes_from_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration: Duration::from_millis(10),
            success_threshold: 2,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.record_failure();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_failure_in_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration: Duration::from_millis(10),
            success_threshold: 2,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.record_failure();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_remaining_time() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration: Duration::from_secs(30),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        assert!(cb.remaining_open_time().is_none());

        cb.record_failure();
        let remaining = cb.remaining_open_time();
        assert!(remaining.is_some());
        assert!(remaining.unwrap() <= Duration::from_secs(30));
    }

    /// Regression test for issue #3229: lock-inversion deadlock on an open
    /// `CircuitBreaker`.
    ///
    /// The bug: three internal `RwLock`s (`state`, `opened_at`, `last_failure`)
    /// were held nested in conflicting orders while the breaker was Open:
    ///
    /// * `maybe_transition` held `opened_at.read()` across `state.write()`
    ///   (opened_at -> state)
    /// * `remaining_open_time` held `state.read()` across `opened_at.read()`
    ///   (state -> opened_at)
    /// * `record_failure` (Open branch) wanted `opened_at.write()`
    ///
    /// Under a fair/writer-preferring lock a queued `opened_at` writer blocks
    /// new `opened_at` readers, closing a 3-thread cycle that deadlocks the
    /// connection pool.
    ///
    /// This test hammers those exact code paths concurrently on a shared,
    /// perpetually-Open breaker and asserts that all workers make progress
    /// within a watchdog deadline. It is NOT modelled with loom -- loom does
    /// not model `RwLock` writer-preference starvation. With the structural
    /// fix (no nested `state`+`opened_at` holds) there is no cycle, so it
    /// completes near-instantly; against the buggy code the workers wedge and
    /// the watchdog fires.
    ///
    /// Honesty note: the *guarantee* is structural (the fix removes the nested
    /// holds outright). This test is a probabilistic backstop -- it reproduces
    /// the deadlock reliably because the three `CircuitBreaker` locks use
    /// `parking_lot::RwLock` (task-fair queuing), but a single scheduling run
    /// is not a proof of absence.
    #[test]
    fn test_circuit_breaker_concurrent_no_deadlock_issue_3229() {
        use std::sync::mpsc::{self, RecvTimeoutError};
        use std::thread;

        // open_duration = 0 forces `maybe_transition` to always reach the
        // (previously nested) `state.write()` path while the breaker is Open,
        // maximizing the chance of forming the lock-inversion cycle.
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration: Duration::ZERO,
            success_threshold: usize::MAX,
            ..Default::default()
        };
        let cb = Arc::new(CircuitBreaker::new(config));
        // Drive it Open to begin with.
        cb.record_failure();

        const THREADS_PER_ROLE: usize = 5;
        const ITERATIONS: usize = 100_000;
        const NUM_WORKERS: usize = THREADS_PER_ROLE * 3;
        // Generous relative to the fixed version (which finishes in well under
        // a second); long enough to be sure a real deadlock has wedged.
        const WATCHDOG: Duration = Duration::from_secs(20);

        let (tx, rx) = mpsc::channel::<()>();
        let mut handles = Vec::with_capacity(NUM_WORKERS);

        for _ in 0..THREADS_PER_ROLE {
            // Role A: maybe_transition (opened_at.read -> state.write in the bug)
            {
                let cb = Arc::clone(&cb);
                let tx = tx.clone();
                handles.push(thread::spawn(move || {
                    for _ in 0..ITERATIONS {
                        let _ = cb.state();
                    }
                    let _ = tx.send(());
                }));
            }
            // Role B: remaining_open_time (state.read -> opened_at.read in the bug)
            {
                let cb = Arc::clone(&cb);
                let tx = tx.clone();
                handles.push(thread::spawn(move || {
                    for _ in 0..ITERATIONS {
                        let _ = cb.remaining_open_time();
                    }
                    let _ = tx.send(());
                }));
            }
            // Role C: record_failure (wants opened_at.write while Open)
            {
                let cb = Arc::clone(&cb);
                let tx = tx.clone();
                handles.push(thread::spawn(move || {
                    for _ in 0..ITERATIONS {
                        cb.record_failure();
                    }
                    let _ = tx.send(());
                }));
            }
        }
        // Drop the original sender so the channel disconnects once every worker
        // (and its cloned sender) is gone.
        drop(tx);

        let deadline = Instant::now() + WATCHDOG;
        let mut completed = 0usize;
        while completed < NUM_WORKERS {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .unwrap_or(Duration::ZERO);
            match rx.recv_timeout(remaining) {
                Ok(()) => completed += 1,
                Err(RecvTimeoutError::Timeout) => panic!(
                    "issue #3229 deadlock: only {completed}/{NUM_WORKERS} circuit-breaker \
                     workers finished within {WATCHDOG:?}; the remaining threads are wedged \
                     in a lock-inversion cycle"
                ),
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        assert_eq!(
            completed, NUM_WORKERS,
            "all circuit-breaker workers should complete without deadlock"
        );

        for h in handles {
            h.join().expect("worker thread panicked");
        }
    }

    // ============ ConnectionPool Tests ============

    #[test]
    fn test_pool_creation() {
        let pool: ConnectionPool<MockShardClient> = ConnectionPool::new(
            make_shard_id(0),
            PoolConfig::default(),
            CircuitBreakerConfig::default(),
        );

        assert_eq!(pool.shard_id(), make_shard_id(0));
        assert!(pool.is_available());
        assert_eq!(pool.circuit_state(), CircuitState::Closed);
    }

    #[test]
    fn test_pool_add_and_get() {
        let shard_id = make_shard_id(0);
        let pool: ConnectionPool<MockShardClient> = ConnectionPool::new(
            shard_id,
            PoolConfig::default(),
            CircuitBreakerConfig::default(),
        );

        let client = Arc::new(MockShardClient::new(shard_id));
        pool.add(Arc::clone(&client));

        let conn = pool.get().unwrap();
        assert_eq!(conn.shard_id(), shard_id);
    }

    #[test]
    fn test_pool_circuit_breaker_integration() {
        let shard_id = make_shard_id(0);
        let pool: ConnectionPool<MockShardClient> = ConnectionPool::new(
            shard_id,
            PoolConfig::default(),
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        );

        let client = Arc::new(MockShardClient::new(shard_id));
        pool.add(client);

        pool.record_failure();
        assert!(pool.is_available());

        pool.record_failure();
        assert!(!pool.is_available());

        let result = pool.get();
        assert!(matches!(result, Err(NetworkError::CircuitOpen { .. })));
    }

    #[test]
    fn test_pool_stats() {
        let shard_id = make_shard_id(0);
        let pool: ConnectionPool<MockShardClient> = ConnectionPool::new(
            shard_id,
            PoolConfig::default(),
            CircuitBreakerConfig::default(),
        );

        let client = Arc::new(MockShardClient::new(shard_id));
        pool.add(client);

        pool.record_success();
        pool.record_success();
        pool.record_failure();

        let stats = pool.stats();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.failed_requests, 1);
        assert!((stats.success_rate() - 0.666).abs() < 0.01);
    }

    // ============ MockShardClient Tests ============

    #[test]
    fn test_mock_client_creation() {
        let client = MockShardClient::new(make_shard_id(0));
        assert!(client.is_healthy());
        assert_eq!(client.shard_id(), make_shard_id(0));
    }

    #[test]
    fn test_mock_client_prepare() {
        let client = MockShardClient::new(make_shard_id(0));

        let response = client.prepare(TxId::new(1), &[], None).unwrap();
        assert!(response.ready);
        assert_eq!(client.call_count("prepare"), 1);
    }

    #[test]
    fn test_mock_client_commit() {
        let client = MockShardClient::new(make_shard_id(0));

        let response = client.commit(TxId::new(1), None).unwrap();
        assert!(response.success);
        assert_eq!(client.call_count("commit"), 1);
    }

    #[test]
    fn test_mock_client_abort() {
        let client = MockShardClient::new(make_shard_id(0));

        let response = client.abort(TxId::new(1)).unwrap();
        assert!(response.acknowledged);
        assert_eq!(client.call_count("abort"), 1);
    }

    #[test]
    fn test_mock_client_unhealthy() {
        let client = MockShardClient::new(make_shard_id(0));
        client.set_healthy(false);

        let result = client.prepare(TxId::new(1), &[], None);
        assert!(matches!(result, Err(NetworkError::ShardUnavailable(_))));
    }

    #[test]
    fn test_mock_client_fail_next() {
        let client = MockShardClient::new(make_shard_id(0));

        client.fail_next(NetworkError::Timeout {
            shard_id: make_shard_id(0),
            operation: "test".to_string(),
            duration: Duration::from_secs(30),
        });

        let result = client.prepare(TxId::new(1), &[], None);
        assert!(matches!(result, Err(NetworkError::Timeout { .. })));

        // Next call should succeed
        let result = client.prepare(TxId::new(2), &[], None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_client_custom_responses() {
        let client = MockShardClient::new(make_shard_id(0));

        client.set_prepare_response(PrepareResponse {
            ready: false,
            reason: Some("test".to_string()),
        });

        let response = client.prepare(TxId::new(1), &[], None).unwrap();
        assert!(!response.ready);
        assert_eq!(response.reason, Some("test".to_string()));
    }

    #[test]
    fn test_mock_client_get_state() {
        let client = MockShardClient::new(make_shard_id(0));

        let mut state = ShardState::new(make_shard_id(0));
        state.node_count = 1000;
        client.set_state(state);

        let retrieved = client.get_state().unwrap();
        assert_eq!(retrieved.node_count, 1000);
    }

    #[test]
    fn test_mock_client_migration() {
        let client = MockShardClient::new(make_shard_id(0));

        let batch = MigrationBatch {
            migration_id: 1,
            batch_number: 0,
            is_last: false,
            nodes: vec![NodeData {
                id: NodeId::new(1).unwrap(),
                label: "Person".to_string(),
                properties: vec![],
                valid_from: 0,
                valid_to: None,
            }],
            edges: vec![],
            checksum: 12345,
        };

        let response = client.receive_migration_batch(batch).unwrap();
        assert!(response.accepted);
        assert_eq!(response.nodes_written, 1);
    }

    #[test]
    fn test_mock_client_extract_migration() {
        let client = MockShardClient::new(make_shard_id(0));

        let batch = client
            .extract_migration_batch(1, &["Person".to_string()], 100, 0)
            .unwrap();
        assert!(batch.is_last);
        assert!(batch.nodes.is_empty());
    }

    #[test]
    fn test_mock_client_health_check() {
        let client = MockShardClient::new(make_shard_id(0));

        assert!(client.health_check().is_ok());

        client.set_healthy(false);
        assert!(client.health_check().is_err());
    }
}
