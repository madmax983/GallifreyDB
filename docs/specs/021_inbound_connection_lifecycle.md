# 🔭 Vantage Spec: Inbound Connection & MCP-Session Lifecycle Policy

| Metadata | Details |
| :--- | :--- |
| **ID** | SPEC-021 |
| **Status** | 📝 Draft |
| **Owner** | Vantage (Product) |
| **Implementer** | Core (Engineering) |
| **Priority** | P0 (Critical) |
| **Related Code** | `src/http/server.rs`, `src/mcp/` |

## 1. 👤 User Stories

> **As an** API Administrator,
> **I want to** configure rate limiting and concurrency limits natively on the HTTP and MCP surfaces,
> **So that** a malfunctioning or malicious client cannot overwhelm the embedded database process.

> **As a** DevOps Engineer,
> **I want to** enforce resource budgets (max sessions, cursor caps) per MCP agent session,
> **So that** concurrent AI agents don't exhaust memory by leaving long-running cursors or transactions open.

> **As a** Security Engineer,
> **I want to** apply rate limits directly within the application tier rather than relying solely on an external reverse proxy,
> **So that** defense-in-depth is maintained and API-key-aware rate limiting is enforced consistently.

## 2. 🧐 The "So What?" (Business Value)

Currently, AletheiaDB's HTTP layer lacks in-process rate limiting (ADR 0055 punted this to the reverse proxy due to tower middleware incompatibilities in older versions of `autumn-web`). As we migrate to a unified HTTP/MCP surface under `autumn-web` 0.5.0, these incompatibilities are resolved.

Furthermore, as we introduce MCP-over-HTTP to support multi-agent integrations, the system must protect itself from aggressive autonomous agents. A poorly written AI agent could open hundreds of concurrent sessions or leak pagination cursors, crashing the single embedded database process.

**The Gap:**
- **Rate Limiting:** Punted to reverse proxy; application is vulnerable to direct overload.
- **Resource Exhaustion:** Multi-agent MCP workflows have no bounded session concurrency or strict cursor budgets.

**ROI:**
- **Operational Stability:** Ensures the database remains responsive under heavy or abusive inbound load.
- **Simplified Deployment:** Reduces the mandatory dependency on complex external reverse proxies for basic rate limiting (superseding ADR 0055's punt).
- **Agent Safety:** Safely allows multiple autonomous agents to interact with the database concurrently without memory leaks.

## 3. ✅ Acceptance Criteria

### Functional Requirements

1. **In-Process Rate Limiting:**
   - Must implement an in-process `tower-governor` rate-limit layer.
   - Must return an HTTP `429 Too Many Requests` status code when limits are exceeded.
2. **Concurrency Control:**
   - Must implement a `ConcurrencyLimit` layer for backpressure on the inbound HTTP/MCP surface.
3. **MCP Session Budgets:**
   - Must enforce a configurable maximum number of concurrent MCP-over-HTTP sessions.
   - Must enforce a maximum cursor cap per session (defaulting to 128) and a strict cursor TTL (defaulting to 5 minutes) to prevent memory leaks from orphaned agents.
4. **Configuration Integration:**
   - The rate limits and concurrency budgets must be tunable via the public configuration struct/file (`AletheiaDBConfig`).

### Non-Functional Requirements

- **Metric Definition:** Success = System maintains < 50ms p99 latency for allowed requests while rejecting excess traffic with 429s under a load of 10,000 req/sec.

## 4. 🚫 Out of Scope (Phase 1)

- **Outbound DB Connection Pooling:** AletheiaDB is strictly an embedded database. There is no outbound connection pool to manage.
- **Distributed Rate Limiting:** State synchronization for rate limits across multiple horizontally scaled instances (e.g., via Redis) is out of scope. Limits apply per-process.
- **TCP-level DDoS Mitigation:** SYN floods and volumetric attacks must still be handled by infrastructure-level firewalls or load balancers.
