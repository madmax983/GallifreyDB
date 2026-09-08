# 🔭 Vantage: Spec for GDPR Hard Deletes (Evict)

## 👤 User Story
**As a** Compliance Officer or Database Administrator,
**I want** to execute a true "Hard Delete" (eviction) of a specific entity (node) and all its historical versions and connected edges across all tiers of storage,
**so that** I can comply with GDPR/CCPA "Right to be Forgotten" mandates, legal discovery purges, or remove accidentally ingested toxic data (e.g., PII in the wrong field) without destroying the entire database.

## 🧐 The "So What?" Ask
**What business problem does this solve?**
AletheiaDB is strictly append-only and bi-temporal by design, which guarantees perfect audit integrity. However, privacy laws like GDPR (General Data Protection Regulation) or CCPA legally mandate the permanent, unrecoverable deletion of user data upon request. Currently, migrating users from systems like XTDB note that `[::xt/evict]` has no equivalent in AletheiaDB, creating a severe adoption blocker for enterprise customers who operate in regulated environments. If a user cannot legally purge data, they cannot use the database.

**Success Metric Definition:**
- **Compliance:** 100% eradication of the specified node ID (and its properties) from the WAL, Hot Memory, Checkpoints, and Cold Storage tiers.
- **Verification:** A subsequent time-travel query (`AS OF SYSTEM_TIME`) for the evicted node ID returns absolutely nothing, as if the node never existed.
- **Auditability:** The eviction action itself is logged in an immutable, restricted audit trail (proving compliance) without retaining the purged payload.

## ✅ Acceptance Criteria
- Must define a new transaction operation `tx.evict_node(id)` that goes beyond a standard `delete_node` (which only adds a tombstone).
- Must permanently scrub the node's data from all storage tiers (WAL, active index, and cold tier).
- Must cascade the eviction to immediately connected edges (similar to cascade delete) to prevent dangling references, or optionally provide a mode to strictly error out if connected edges exist.
- Must execute securely, potentially requiring a higher RBAC permission level (e.g., `ROLE_COMPLIANCE_ADMIN`) than a standard write.
- Must leave a metadata trace of the eviction (e.g., `EvictedNode { id, tx_time, reason }`) to prove the data was destroyed to auditors, while strictly containing zero original properties.

## 🚫 Out of Scope
- **Selective Property Eviction:** Purging *only* the `email` property across all time while keeping the node itself. Phase 1 focuses on full node eviction.
- **Cascading Eviction across the Graph:** Evicting a user and recursively evicting all documents they authored. Phase 1 focuses strictly on the target node and its direct incident edges.
- **Instant Cold Tier Re-writing:** Re-writing historical immutable redb cold-tier files immediately on every single evict. The system may instead write a durable "Evict Filter" (a bloom filter or blocklist) to the index, effectively hiding the data immediately and deferring the heavy physical re-writing of the cold tier to a background compaction/vacuum job.
