# 🔭 Vantage: Spec for Data Retention Policies (Automated Pruning)

## 👤 User Story
**As a** Database Administrator or Data Privacy Officer,
**I want** to configure automated data retention policies (like TTL or Keep N versions) for specific node labels,
**so that** I can comply with data privacy regulations (like GDPR's Right to be Forgotten) and manage storage costs without having to manually run batch deletion jobs on historical data.

## 🧐 The "So What?" Ask
**What business problem does this solve?**
AletheiaDB's bi-temporal architecture is incredibly powerful because it keeps a full, immutable history of all changes. However, this "keep everything forever" default is fundamentally at odds with strict privacy laws (GDPR, CCPA) which mandate that certain PII (Personally Identifiable Information) must not be retained indefinitely. Furthermore, not all data holds historical value; keeping every minor state change of a high-frequency sensor wastes disk space and inflates cloud storage costs. By introducing automated Data Retention Policies directly into the storage engine, we shift the burden of compliance and cost-management from the application layer to the database layer, ensuring it is handled safely, consistently, and performantly.

**Success Metric Definition:**
- **Storage Efficiency:** Nodes configured with a "Keep N=5" policy should demonstrably reclaim disk space during WAL compaction/garbage collection for versions older than the 5th most recent.
- **Query Correctness:** Time-travel queries (`AS OF T`) requesting a version of a node that has been pruned due to a retention policy should return a clear, specific error (e.g., `DataPrunedError`) rather than returning nothing or returning incorrect adjacent data.

## ✅ Acceptance Criteria
- Must define an API to set a `RetentionPolicy` (e.g., `KeepN(count)`, `KeepDuration(time)`, `KeepForever`) on a per-Label basis.
- The background compaction/garbage collection process must identify and permanently delete historical node and edge versions that fall outside their assigned retention window.
- The system must ensure that the "current state" of a node is *never* deleted by a history-trimming retention policy, even if the last update was very old.
- Must provide a clear error message if a user attempts to time-travel query into a pruned historical period.
- Must cleanly integrate with existing Vector Index retention policies to ensure semantic search doesn't return pointers to pruned data.

## 🚫 Out of Scope
- Granular per-property retention policies (e.g., keep the "name" for 1 year, but delete the "email" after 30 days). Policies apply to the entire node version.
- "Soft Deletion" recovery. When data is pruned by a retention policy, it must be physically removed from disk (hard delete) to satisfy compliance requirements.
- Moving pruned data to a cold-storage tier. This feature focuses on deletion; Tiered Storage handles archiving.
