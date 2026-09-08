# 🔭 Vantage Spec: Atomic Checkpointing

| Metadata | Details |
| :--- | :--- |
| **ID** | SPEC-021 |
| **Status** | 📝 Draft |
| **Owner** | Vantage (Product) |
| **Implementer** | Core (Engineering) |
| **Priority** | P1 (Critical) |
| **Related Code** | `src/storage/checkpoint.rs`, `tests/snapshot_race_condition.rs` |

## 1. 👤 User Stories

> **As a** Database Administrator,
> **I want to** ensure that checkpoints capture a perfectly consistent snapshot of both current and historical data simultaneously,
> **So that** I don't lose data integrity or experience corrupted states upon database recovery.

## 2. 🧐 The "So What?" (Business Value)

Currently, AletheiaDB creates snapshots of `CurrentStorage` and `HistoricalStorage` sequentially without coordination. This creates a race condition window where concurrent writes can partially affect one snapshot but not the other, leading to "torn" checkpoints and data inconsistency upon recovery.

**The Gap:**
- **Data Corruption Risk:** In concurrent environments, recovering from a checkpoint might yield a database where historical versions reference current nodes that do not exist, or vice versa.
- **Reliability:** Fails to provide fundamental ACID durability/consistency guarantees during the checkpoint process.

**ROI:**
- **Trust & Reliability:** Guarantees that the database can safely recover to a consistent point in time without manual intervention or data loss.
- **Enterprise Readiness:** A fundamental requirement for production deployments.

**Metric Definition:**
- **Success:** 0 torn checkpoints under high concurrency; the `test_concurrent_write_during_snapshot_creation` test passes consistently.

## 3. ✅ Acceptance Criteria

### Functional Requirements

1.  **Atomic Coordination:**
    - The system MUST coordinate the creation of snapshots across both `CurrentStorage` and `HistoricalStorage` to ensure they represent the exact same Log Sequence Number (LSN) and the exact same set of writes.
2.  **Write Isolation:**
    - Writes MUST be safely ordered around the snapshot boundary. No concurrent write should be able to sneak in between the creation of the current and historical snapshots.

### Non-Functional Requirements

-   **Performance:** Coordination must not stall the write path for more than 1 millisecond.

## 4. 🚫 Out of Scope (Phase 1)

-   **Distributed Checkpointing:** Coordinating checkpoints across multiple sharded nodes (Phase 2).
