# Sentry's Journal 🛡️

## Distributed Transaction Durability Gap
**Learning:** The `ShardCoordinator` was using an in-memory-only commit log (`TwoPhaseCommitLog`), despite a persistent implementation (`PersistentCommitLog`) being available in the codebase. This meant distributed transactions were not durable across coordinator restarts—a violation of ACID properties (Atomicity/Durability). Furthermore, the persistent log implementation was missing the `commit_timestamp` field, which is critical for consistent recovery in a system using Hybrid Logical Clocks.

**Action:**
1.  Always verify that "persistent" components are actually wired up to configuration and initialization paths.
2.  When seeing duplicate implementations (in-memory vs persistent structs), check which one is used in production code vs tests.
3.  Wired up `PersistentCommitLog` to `ShardCoordinator` via new `wal_path` config.
4.  Updated `PersistentCommitLog` schema to include `commit_timestamp` (with backward compatibility logic for V1, though V2 enforced for new writes).
## Vector Error Path Validation
**Learning:** Rust doc tests often use `unwrap()` and this can inadvertently lead to a false sense of security where error paths inside logic components like `SparseVec::new` lack explicit test coverage. Missing explicit test cases for `VectorError` variants could lead to regressions.
**Action:** Always add exhaustive table-driven tests mapping out every possible error path (`ContainsNaN`, `InvalidSparseVector` due to zero values, dimension mismatches, etc.) when testing logic components handling input.

## IdentityHasher FNV-1a Fallback Coverage Gap
**Learning:** `IdentityHasher` implements a highly optimized pass-through hash for known primitive integers (length 1, 2, 4, 8, 16), but has a catch-all fallback (`_ =>`) using the FNV-1a algorithm for any other length byte slices. This fallback logic lacked property-based verification to ensure it accurately implements the FNV-1a algorithm for arbitrary slices and correctly chains state when prior writes have occurred. Testing this fallback logic revealed an edge case in empty slice fallback.
**Action:** Always verify "catch-all" match arms using property testing to cover all unexpected shapes and lengths, ensuring manual implementations match reference implementations.
**IdentityHasher Coverage Gap**
**Learning:** `IdentityHasher` provides optimizations for pre-hashed unique integer keys. However, large portions of `Hasher::write` branches, state mutability paths, and trait implementations (like bitwise operations in `update_state`) were not comprehensively tested, leaving them vulnerable to subtle regressions if tampered with (e.g., via mutation testing).
**Action:** Wrote exhaustive tests covering every match arm in `write`, explicitly tested the `else` branch of `update_state` (which involves a XOR mix and multiply), and checked each integer specific method (`write_u8`, `write_u16`, etc.) individually and sequentially to eliminate any remaining `cargo mutants` escapees.

## Panic Risks in Query Iterators and Mock Clients
**Learning:** `unwrap()` inside iterator implementations (like `VectorRerankIterator::next`) or trait implementations for mock clients (like `MockVectorNodeClient`) pose a significant availability risk, as a panic can crash the thread handling the query or the entire database process.
**Action:** Always gracefully handle `None` on iterators (returning `None` or propagating errors) and map lock poisoning errors (`PoisonError`) to domain-specific errors (e.g. `VectorError`) to ensure the system degrades gracefully instead of crashing during simulated faults or unexpected states.

**[Temporal TimeRange Max Timestamp and Deserialize Mutants]**
**Learning:** `cargo mutants` revealed missing test coverage for `MAX_VALID_TIMESTAMP` bounds checking in open ranges (`TimeRange::from` and `TimeRange::at`), empty range overlap edge cases (`TimeRange::overlaps` short-circuit behavior), and deserialization stringency (`BiTemporalInterval::deserialize` exact consumed length checking against buffer size `> 48`).
**Action:** Added targeted unit tests checking `#[should_panic]` when `MAX_VALID_TIMESTAMP` is exceeded, explicitly testing overlap between empty point-ranges within larger non-empty ranges, and appending excess bytes to binary formats to verify exact parser length consumption limits.

## LimitPushdown Mutants
**Learning:** `cargo mutants` revealed missing test coverage for `LimitPushdown::push_down` in several conditions around `||`. Also limits shouldn't be blindly pushed down through filters, because limits only apply after the filter reduces the row count. Tests covering the lack of modification of binary children boundaries have also been introduced.
**Action:** When adding rules like `LimitPushdown`, always ensure to write exhaustive structural test cases (verifying limits propagating, or explicitly stopping at operations like `Filter` or `Sort`).

## ProjectIterator try_insert unwrapping
**Learning:** `unwrap()` inside iterator implementations (like `ProjectIterator::next`) poses a significant panic risk when handling properties, especially dynamically sized ones where recursion depth limits can be exceeded or insertion errors can occur. In a database context, panics inside iterators will crash the entire query process rather than just returning an error to the client.
**Action:** Always gracefully handle property insertion errors using `match` or `?` and propagate them down the iterator pipeline instead of unwrapping, allowing the query to fail safely. Added tests mocking a `try_insert` failure by using an invalid property state.
**[Fix `execute_traversal` target shards bug]**
**Learning:** `plan.steps` from `route_traversal` returns empty list, and we need `involved_shards`
**Action:** Replace `steps` with `involved_shards` and update the test.
## PropertyMapBuilder Panic Testing
**Learning:** `PropertyMapBuilder::insert` and `remove` correctly wrap their fallible counterparts (`try_insert` and `try_remove`) but expected a panic via `.expect()` on recursion depth exceeded, or failures.
**Action:** Wrote explicit `#[should_panic]` unit tests ensuring that recursion limits are safely caught before crashing with an unexpected error message or causing UB.
