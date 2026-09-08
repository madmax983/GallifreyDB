👺 Havoc: Fix deadlock in distributed transaction coordination

🧊 **The Trigger:**
In `ShardCoordinator::prepare_distributed_transaction` and `commit_distributed_transaction`, a `RwLockReadGuard` on `self.connections` was held while attempting to acquire a `RwLockWriteGuard` on `self.active_transactions`.
This happened when the `connections.read()` call returned an `Err(PoisonError)`. The `Err(_)` branch of the `match` expression bound the poisoned lock error to a temporary that lived until the end of the `match` block. While this temporary was alive, the code called `self.reinsert_transaction`, which requested `active_transactions.write()`.

📉 **The Stack Trace:**
```
thread 'storage::sharding::coordinator::tests::test_havoc_deadlock' panicked at src/storage/sharding/coordinator.rs:2006:9:
Deadlock detected!
```

🧪 **Reproduction:**
Run `cargo test --lib test_havoc_deadlock` (with the modified test that correctly simulates a write lock on connections to reproduce the exact scenario). The added Loom test `test_havoc_loom_sharding_coordinator_deadlock` strictly proves the fix in bounded state space.

😈 **Comment:**
You assumed that dropping a `PoisonError` via `Err(_)` immediately released the `RwLockReadGuard` it contained. You were wrong. In Rust, temporaries in a `match` scrutinee live until the end of the `match` block. I split the matching to eagerly drop the poisoned lock guard. Don't let your locks linger.
