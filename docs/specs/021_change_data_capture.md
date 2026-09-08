# 🔭 Vantage Spec: Change Data Capture (CDC) Streams

| Metadata | Details |
| :--- | :--- |
| **ID** | SPEC-021 |
| **Status** | 📝 Draft |
| **Owner** | Vantage (Product) |
| **Implementer** | Core (Engineering) |
| **Priority** | P1 (High) |
| **Related Code** | `src/storage/wal/` and `src/api/` (Proposed) |

## 1. 👤 User Stories

> **As a** Data Engineer,
> **I want to** subscribe to a real-time stream of all mutations (inserts, updates, deletes) occurring in the database,
> **So that** I can synchronize AletheiaDB with downstream systems (like Snowflake, Elasticsearch, or Apache Kafka) without running expensive polling queries.

> **As an** Event-Driven Microservices Developer,
> **I want to** trigger business logic (e.g., sending an email, updating a cache) immediately when a specific node's properties change,
> **So that** my application can react to graph changes in real-time.

> **As an** Audit/Compliance Officer,
> **I want to** pipe a cryptographically secure, sequential log of all database transactions into long-term immutable storage,
> **So that** I can maintain a pristine audit trail of exactly who changed what and when.

## 2. 🧐 The "So What?" (Business Value)

AletheiaDB is highly performant for writing data, and its bi-temporal model keeps a perfect internal history. However, getting data *out* of AletheiaDB as it changes currently requires manual, inefficient polling via the HTTP API or Rust SDK.

**The Gap:**
- **Integration Friction**: Modern data architectures rely on event streaming (Kafka, Kinesis). A database that acts as a "black box" until queried is difficult to integrate into real-time pipelines.
- **Polling Overhead**: Users writing scripts to poll for `tx_time > last_check` waste massive amounts of CPU and network bandwidth on the database.
- **Lost Temporal Resolution**: If a node changes twice between polling intervals, downstream systems miss the intermediate state.

**ROI:**
- **Enterprise Ecosystem Integration**: CDC turns AletheiaDB into a "good citizen" in the modern data stack, easily integrating with standard ETL/ELT pipelines.
- **Real-Time Applications**: Unlocks reactive architectures (e.g., updating UI dashboards instantly via WebSockets when the graph changes).
- **Reduced Load**: Replaces thousands of inefficient polling queries with a single, efficient push-based stream.

## 3. ✅ Acceptance Criteria

### Functional Requirements

1.  **WAL Tailing API**:
    - Must expose a Rust API (and subsequently an HTTP Server-Sent Events (SSE) or gRPC streaming endpoint) that allows clients to "tail" the Write-Ahead Log (WAL).
    - The stream MUST yield structured events: `TransactionBegin`, `NodeCreated`, `NodeUpdated`, `NodeDeleted`, `EdgeCreated`, `EdgeDeleted`, and `TransactionCommit`.
    - Each event MUST include the Logical Sequence Number (LSN) and the transaction timestamp (`tx_time`).

2.  **Resumability**:
    - Clients MUST be able to provide a starting `LSN` to resume streaming from a specific point in history, preventing data loss if the client disconnects.

3.  **Filtering (Phase 1 Basic)**:
    - The API SHOULD allow basic filtering at the source (e.g., "only send events for nodes with label 'Customer'"), to reduce network overhead.

### Non-Functional Requirements

-   **Performance/Impact**: Tailing the WAL must have `< 2%` performance impact on the primary write path. Reading from the CDC stream must be asynchronous and non-blocking to writers.
-   **Metric Definition**: Success = A client can stream 10,000 transaction events per second with `< 5ms` latency from the time the transaction is committed to the time the event is pushed to the client.

## 4. 🚫 Out of Scope (Phase 1)

-   **Native Kafka/Kinesis Integration**: We will build the foundational CDC API and HTTP streaming endpoints first. Dedicated sink connectors (e.g., native Kafka Producer) are out of scope for Phase 1.
-   **Complex Event Processing (CEP)**: The stream emits raw row-level changes. Aggregations (e.g., "emit an event only if the balance drops below 0 over 5 transactions") are the responsibility of the downstream consumer, not the database.
-   **Distributed Shard Aggregation**: In a sharded setup, Phase 1 requires the client to tail each shard individually. A unified, globally-ordered cluster stream is deferred to Phase 2.
