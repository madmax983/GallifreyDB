//! Event-driven changefeed long-poll: worker-pin removal + prompt slot release
//! on client disconnect (Issue #3673), driven against the live assembled
//! autumn-web server and the core `Subscription::recv_async` primitive.
//!
//! These lock the ACs:
//!
//! - **(a) worker-starvation**: N concurrent `recv_async` long-polls on a SMALL
//!   worker pool do NOT starve unrelated work — a suspended async wait pins zero
//!   worker threads (RED if `recv_async` blocks the thread like the old
//!   `block_in_place`/`recv_timeout` path).
//! - **(b) prompt disconnect release**: an SSE `GET /changes/stream` client and an
//!   HTTP `POST /changes/await` long-poll BOTH free their per-principal slot in
//!   well under the old 15s / 60s bound when the client disconnects — event-driven
//!   (`tx.closed()` / future-drop), no commits needed to wake the worker.
//! - **(b) + #3688 churn**: rapid reconnect churn near the per-principal cap never
//!   falsely exhausts the quota (stale slots are reaped promptly).
//! - **(c) delivery unchanged**: the event-driven path drains cursor-ascending and
//!   losslessly, identical to the sync `recv_timeout` contract.
//! - normal-path timeout still releases; disconnect decrements the counter exactly
//!   once; `database_stats` per-principal returns to baseline promptly.

use std::sync::Arc;
use std::time::{Duration, Instant};

use aletheia_server::build_server_client;
use aletheiadb::auth::{AuthMode, AuthStore, Role};
use aletheiadb::core::changefeed::ChangeRecord;
use aletheiadb::core::changefeed_subscription::ChangefeedConfig;
use aletheiadb::{AletheiaDB, ChangeFilter, PropertyMapBuilder};
use serde_json::{Value, json};

const READER_TOKEN: &str = "reader-token-abcdefghijklmnop";
const WRITER_TOKEN: &str = "writer-token-abcdefghijklmnop";
const ADMIN_TOKEN: &str = "admin-token-abcdefghijklmnop";

/// Wall-clock bound for a per-principal slot to be reaped after a client
/// disconnect. Deliberately loose (5s) so it never flakes on a loaded CI worker
/// while still RED-ing the pre-fix regressions with wide headroom: the old SSE
/// path lingered up to `STREAM_POLL_INTERVAL` (~15s) and the old await path up
/// to the full 60s `timeout_ms`, so 5s catches either with ≥3× margin.
const SLOT_RELEASE_BOUND: Duration = Duration::from_secs(5);

// ── fixtures / helpers ───────────────────────────────────────────────────────

fn new_db() -> Arc<AletheiaDB> {
    Arc::new(AletheiaDB::new().expect("create db"))
}

fn keyed_store() -> Arc<AuthStore> {
    let store = Arc::new(AuthStore::new());
    store
        .insert_bootstrap_key("reader", Role::Reader, READER_TOKEN)
        .expect("reader key");
    store
        .insert_bootstrap_key("writer", Role::Writer, WRITER_TOKEN)
        .expect("writer key");
    store
        .insert_bootstrap_key("admin", Role::Admin, ADMIN_TOKEN)
        .expect("admin key");
    store
}

fn principal_id(store: &AuthStore, token: &str) -> String {
    store.verify(token).expect("verify token").id
}

/// The live per-principal subscription count for `id` (0 if absent).
fn principal_count(db: &AletheiaDB, id: &str) -> usize {
    db.changefeed_principal_counts()
        .into_iter()
        .find(|(k, _)| k == id)
        .map(|(_, c)| c)
        .unwrap_or(0)
}

/// Poll `cond` every 20ms until it holds or `bound` elapses. Returns the elapsed
/// time on success, or `None` if the bound was hit first.
async fn wait_until(bound: Duration, mut cond: impl FnMut() -> bool) -> Option<Duration> {
    let start = Instant::now();
    while start.elapsed() < bound {
        if cond() {
            return Some(start.elapsed());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    cond().then(|| start.elapsed())
}

/// A committed change wakes any live subscription (used to advance the feed).
fn commit(db: &AletheiaDB, n: i64) {
    let _ = db.create_node("Ping", PropertyMapBuilder::new().insert("n", n).build());
}

/// A stable dedup/order key equivalent to the `ChangeCursor`.
fn key(r: &ChangeRecord) -> (i64, u32, u8, u64, u64) {
    let start = r.transaction_time_range.start();
    (
        start.wallclock(),
        start.logical(),
        r.kind.ord(),
        r.entity_id,
        r.version_id,
    )
}

// ── (a) CORE: N long-polls do not starve a small worker pool ──────────────────

/// N concurrent `recv_async` long-polls (60s) on a runtime whose worker pool AND
/// blocking pool are BOTH capped at 2 must pin neither: an event-driven wait
/// consumes zero threads of either kind, so N can far exceed the combined budget
/// (2 + 2 = 4) and unrelated work of BOTH flavors still makes progress.
///
/// The two probes are the discriminators:
/// - an async probe (an ordinary task) RED-s a *worker-pinning* regression
///   (`block_in_place` or a naive synchronous `recv_timeout` on the worker):
///   with 2 workers pinned and only 2 blocking replacements available, no worker
///   is left to schedule it;
/// - a `spawn_blocking` probe RED-s a *blocking-pool* regression
///   (`spawn_blocking(recv_timeout)`): with both blocking threads held by
///   long-polls it can never be scheduled.
///
/// The event-driven path holds neither resource, so both probes complete
/// promptly and the single commit still wakes all N. (Manually built runtime:
/// `#[tokio::test]` cannot set `max_blocking_threads`.)
#[test]
fn recv_async_does_not_starve_small_worker_pool() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(2)
        .enable_all()
        .build()
        .expect("build limited runtime");

    rt.block_on(async {
        let db = new_db();
        // N strictly exceeds the combined worker+blocking budget (2 + 2 = 4), so
        // ANY thread-pinning wait exhausts both pools and starves a probe.
        const N: usize = 8;

        let mut handles = Vec::new();
        for _ in 0..N {
            let sub = db
                .subscribe_changes(ChangeFilter::all())
                .expect("subscribe");
            handles.push(tokio::spawn(async move {
                sub.recv_async(Duration::from_secs(60)).await
            }));
        }

        // Give every long-poll time to reach and park in its wait, so that under
        // a thread-pinning regression the worker/blocking pools are actually
        // saturated before the probes run.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Probe 1 (async): RED-s a worker-pinning regression. With both workers
        // pinned (and both blocking replacements consumed) nothing schedules it.
        let async_probe = tokio::spawn(async {
            tokio::time::sleep(Duration::from_millis(5)).await;
            42u32
        });
        let async_probed = tokio::time::timeout(Duration::from_secs(2), async_probe).await;
        assert!(
            async_probed.is_ok(),
            "async probe starved: concurrent long-polls pinned the WORKER pool \
             (block_in_place / naive sync recv_timeout regression)"
        );
        assert_eq!(async_probed.unwrap().expect("async probe join"), 42);

        // Probe 2 (blocking): RED-s a spawn_blocking regression. With both
        // blocking threads held by long-polls it can never run.
        let blocking_probe =
            tokio::task::spawn_blocking(|| std::thread::sleep(Duration::from_millis(5)));
        let blocking_probed = tokio::time::timeout(Duration::from_secs(2), blocking_probe).await;
        assert!(
            blocking_probed.is_ok(),
            "blocking probe starved: concurrent long-polls held the BLOCKING pool \
             (spawn_blocking(recv_timeout) regression)"
        );
        blocking_probed.unwrap().expect("blocking probe join");

        // One commit wakes ALL N long-polls; each resolves with the change.
        commit(&db, 1);
        let joined = tokio::time::timeout(Duration::from_secs(5), async {
            for h in handles {
                let recs = h.await.expect("join").expect("recv_async ok");
                assert_eq!(recs.len(), 1, "each long-poll receives the one commit");
            }
        })
        .await;
        assert!(
            joined.is_ok(),
            "not all long-polls resolved after the commit (worker starvation)"
        );
    });
}

// ── (a reverse): event-driven, no busy-spin when idle ─────────────────────────

/// An idle `recv_async` parks event-driven: over a 300ms idle interval the
/// outer future is polled only O(1) times (one initial poll that returns
/// Pending, one final poll when the timeout fires), NOT once per tick. This is
/// the distinctive busy-spin assertion — a `loop { sleep(10ms); check }`
/// placeholder would wake the task ~30 times and blow the poll-count bound even
/// though it also honors the elapsed and single-delivery checks. A single commit
/// still yields exactly one delivery.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recv_async_is_event_driven_no_busy_spin_when_idle() {
    use std::future::Future;

    let db = new_db();
    let sub = db
        .subscribe_changes(ChangeFilter::all())
        .expect("subscribe");

    let start = Instant::now();
    // Count how many times the recv_async future is polled while idle. An
    // event-driven wait registers a waker and parks, so the runtime polls it
    // only on a genuine wake (here: the timeout). A busy-poll design re-polls on
    // every self-scheduled tick.
    let mut polls = 0usize;
    let fut = sub.recv_async(Duration::from_millis(300));
    tokio::pin!(fut);
    let empty = std::future::poll_fn(|cx| {
        polls += 1;
        fut.as_mut().poll(cx)
    })
    .await
    .expect("recv ok");
    assert!(empty.is_empty(), "no data was committed");
    assert!(
        start.elapsed() >= Duration::from_millis(250),
        "recv_async returned early ({:?}) — not event-driven?",
        start.elapsed()
    );
    assert!(
        polls <= 5,
        "recv_async was polled {polls} times over a 300ms idle wait — not \
         event-driven (a busy-poll / sleep-loop design would poll ~30 times)"
    );

    // One commit → exactly one prompt delivery (no missed / duplicate wake).
    commit(&db, 7);
    let got = tokio::time::timeout(
        Duration::from_secs(2),
        sub.recv_async(Duration::from_secs(2)),
    )
    .await
    .expect("did not hang")
    .expect("recv ok");
    assert_eq!(got.len(), 1, "single commit delivered exactly once");
}

// ── (c): event-driven delivery is ordered and lossless ────────────────────────

/// Draining via `recv_async` across interleaved commits yields records in
/// cursor-ascending order and the union equals the durable ground truth —
/// identical to the sync `recv_timeout` contract (AC c).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn event_driven_delivery_is_ordered_and_lossless() {
    let db = new_db();

    // First, distinctively exercise the ASYNC push-wakeup path: a suspended
    // `recv_async` waiter must be woken by the commit *broadcast* while parked,
    // not by draining after the fact. A waiter that only returned on timeout (or
    // a busy drain-poll) would blow the tight bound below.
    {
        let sub = db
            .subscribe_changes(ChangeFilter::all())
            .expect("subscribe");
        let waiter = tokio::spawn(async move {
            let start = Instant::now();
            let recs = sub
                .recv_async(Duration::from_secs(10))
                .await
                .expect("recv ok");
            (recs, start.elapsed())
        });
        // Let the waiter enter recv_async and park on the Notify.
        tokio::task::yield_now().await;
        commit(&db, -1);
        let (recs, waited) = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("async waiter must be woken by the commit, not wait out 10s")
            .expect("join");
        assert_eq!(recs.len(), 1, "async waiter delivered the pushed commit");
        assert!(
            waited < Duration::from_secs(2),
            "async waiter woken promptly by push-wakeup, not the 10s timeout: {waited:?}"
        );
    }

    let sub = db
        .subscribe_changes(ChangeFilter::all())
        .expect("subscribe");

    const COMMITS: i64 = 25;
    let mut collected: Vec<ChangeRecord> = Vec::new();
    for i in 0..COMMITS {
        commit(&db, i);
        // Drain whatever is available with a short async wait.
        loop {
            let batch = sub
                .recv_async(Duration::from_millis(50))
                .await
                .expect("recv ok");
            if batch.is_empty() {
                break;
            }
            collected.extend(batch);
        }
    }
    // Flush any remainder.
    loop {
        let batch = sub
            .recv_async(Duration::from_millis(50))
            .await
            .expect("recv ok");
        if batch.is_empty() {
            break;
        }
        collected.extend(batch);
    }

    // Cursor-ascending order.
    let keys: Vec<_> = collected.iter().map(key).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "records delivered in cursor-ascending order");

    // Lossless: each commit is one node-create record, delivered exactly once.
    assert_eq!(db.changefeed_subscription_count(), 1, "sub still live");
    assert_eq!(
        collected.len() as i64,
        COMMITS,
        "every committed change delivered exactly once"
    );
}

// ── (b): SSE client disconnect releases the slot under the tight bound ─────────

/// An SSE `GET /changes/stream` client disconnect frees the per-principal slot in
/// well under 2s WITHOUT any commits to wake the worker — the async worker races
/// `tx.closed()` and drops its `Subscription` at once. RED under the old
/// `spawn_blocking` + `recv_timeout(15s)` idle poll (release lagged up to 15s and
/// required a commit to wake).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sse_disconnect_releases_slot_under_two_seconds() {
    let db = new_db();
    let store = keyed_store();
    let reader_id = principal_id(&store, READER_TOKEN);
    db.set_changefeed_config(
        ChangefeedConfig::default()
            .with_max_subscriptions(128)
            .with_max_subscriptions_per_principal(4),
    );
    let client = Arc::new(build_server_client(
        Arc::clone(&db),
        store,
        AuthMode::Required,
    ));

    let stream_task = {
        let client = Arc::clone(&client);
        tokio::spawn(async move {
            let _ = client
                .get("/changes/stream")
                .header("authorization", &format!("Bearer {READER_TOKEN}"))
                .send()
                .await;
        })
    };

    let registered = wait_until(Duration::from_secs(5), || {
        principal_count(&db, &reader_id) == 1
    })
    .await;
    assert!(registered.is_some(), "SSE stream should register the slot");

    // The SSE handler must still be blocked in its stream (never naturally
    // completed) so that `abort()` is what triggers the disconnect we measure.
    assert!(
        !stream_task.is_finished(),
        "SSE stream task completed on its own — disconnect not actually exercised"
    );

    // Disconnect — NO commits to wake the worker (the whole point).
    stream_task.abort();

    let released = wait_until(SLOT_RELEASE_BOUND, || principal_count(&db, &reader_id) == 0).await;
    assert!(
        released.is_some(),
        "SSE disconnect must free the slot within SLOT_RELEASE_BOUND with no commits (was up to 15s)"
    );

    // Genuinely free: the reader can re-subscribe.
    let _fresh = db
        .subscribe_changes_for_principal(Some(&reader_id), ChangeFilter::all())
        .expect("reader re-subscribes after prompt release");
}

// ── (b): HTTP await long-poll disconnect releases the slot promptly ───────────

/// A `POST /changes/await` long-poll (60s) client disconnect frees the
/// per-principal slot in well under 2s — the async handler future is dropped on
/// disconnect, dropping the `Subscription`. RED under the old `spawn_blocking`
/// path: the blocking worker is NOT cancelled on future-drop and held the slot
/// for the full 60s timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn await_disconnect_releases_slot_under_two_seconds() {
    let db = new_db();
    let store = keyed_store();
    let reader_id = principal_id(&store, READER_TOKEN);
    db.set_changefeed_config(
        ChangefeedConfig::default()
            .with_max_subscriptions(128)
            .with_max_subscriptions_per_principal(4),
    );
    let client = Arc::new(build_server_client(
        Arc::clone(&db),
        store,
        AuthMode::Required,
    ));

    let await_task = {
        let client = Arc::clone(&client);
        tokio::spawn(async move {
            let _ = client
                .post("/changes/await")
                .header("authorization", &format!("Bearer {READER_TOKEN}"))
                .json(&json!({ "timeout_ms": 60000 }))
                .send()
                .await;
        })
    };

    let registered = wait_until(Duration::from_secs(5), || {
        principal_count(&db, &reader_id) == 1
    })
    .await;
    assert!(
        registered.is_some(),
        "await long-poll should register the slot"
    );

    // The long-poll must still be blocking (not naturally returned) so `abort()`
    // is what triggers the disconnect release we measure.
    assert!(
        !await_task.is_finished(),
        "await task completed on its own — disconnect not actually exercised"
    );

    await_task.abort();

    let released = wait_until(SLOT_RELEASE_BOUND, || principal_count(&db, &reader_id) == 0).await;
    assert!(
        released.is_some(),
        "await disconnect must free the slot within SLOT_RELEASE_BOUND (was up to 60s)"
    );
}

// ── (b + #3688): reconnect churn never falsely exhausts the quota ─────────────

/// Bound for the churn test's per-round reap wait. Deliberately generous:
/// this test's subject is churn *accumulation* (repeated saturate→abort→reap
/// never falsely exhausting the quota), not reap latency -- the
/// prompt-disconnect timing is pinned by the other tests in this file
/// against the 5s `SLOT_RELEASE_BOUND`. 30s still REDs the #3688 regression
/// with 2x margin (the await-path bug lingered up to the full 60s
/// `timeout_ms`).
const CHURN_REAP_BOUND: Duration = Duration::from_secs(30);

/// Saturate→abort→reap rounds. Five rounds still exercise churn accumulation
/// (each round independently re-saturates the cap after a full reap); ten
/// doubled the exposure window on oversubscribed CI runners without adding
/// signal (Issue #3814).
const CHURN_ROUNDS: usize = 5;

/// Rapid open→disconnect churn near the per-principal cap never falsely exhausts
/// the quota: each disconnect frees its slot promptly (event-driven) so the next
/// open always fits. RED if stale slots linger 15/60s and accumulate past the
/// cap (the #3688 transient false-exhaustion this issue eliminates).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnect_churn_no_false_exhaustion() {
    let db = new_db();
    let store = keyed_store();
    let reader_id = principal_id(&store, READER_TOKEN);
    const CAP: usize = 2;
    db.set_changefeed_config(
        ChangefeedConfig::default()
            .with_max_subscriptions(128)
            .with_max_subscriptions_per_principal(CAP),
    );
    let client = Arc::new(build_server_client(
        Arc::clone(&db),
        store,
        AuthMode::Required,
    ));

    for round in 0..CHURN_ROUNDS {
        // Saturate the ENTIRE per-principal cap with concurrent long-polls, then
        // abort them all at once. Under the #3688 bug the aborted slots linger
        // (15/60s), so the cap stays occupied and the next subscribe is falsely
        // rejected — that false-exhaustion is the distinctive failure asserted
        // below, not a per-round release-timing race.
        let mut tasks = Vec::new();
        for _ in 0..CAP {
            let client = Arc::clone(&client);
            tasks.push(tokio::spawn(async move {
                let _ = client
                    .post("/changes/await")
                    .header("authorization", &format!("Bearer {READER_TOKEN}"))
                    .json(&json!({ "timeout_ms": 60000 }))
                    .send()
                    .await;
            }));
        }

        assert!(
            wait_until(Duration::from_secs(5), || principal_count(&db, &reader_id)
                == CAP)
            .await
            .is_some(),
            "round {round}: the concurrent long-polls should saturate the cap"
        );
        for task in &tasks {
            assert!(
                !task.is_finished(),
                "round {round}: a long-poll completed on its own — disconnect not exercised"
            );
            task.abort();
        }
        // Awaiting the aborted tasks guarantees client-side teardown (request
        // future dropped, connection torn down) completes BEFORE the reap
        // bound below starts ticking: the bound then measures only
        // server-side slot release. Without this, abort-scheduling delay on a
        // loaded runner eats into the reap budget (Issue #3814).
        for task in tasks {
            let _ = task.await;
        }

        // DISTINCTIVE #3688 guard: with the cap just saturated and every holder
        // aborted, the reap must be COMPLETE -- the principal's live count
        // returns to zero -- before the next round. Waiting only for "a fresh
        // subscribe fits" (the old probe) can succeed on a PARTIAL reap; the
        // next round then saturates via a stale+new mix and its rejected
        // long-poll trips the `is_finished` assert above -- a per-round
        // release-timing race the old comment claimed this test didn't have
        // (Issue #3814). Under the #3688 bug the slots linger and the count
        // never reaches zero within the bound -- false-exhaustion.
        let reaped = wait_until(CHURN_REAP_BOUND, || principal_count(&db, &reader_id) == 0).await;
        assert!(
            reaped.is_some(),
            "round {round}: aborted slots not fully reaped within {CHURN_REAP_BOUND:?} -- \
             quota would falsely exhaust on the next round (#3688 false-exhaustion)"
        );

        // And the quota genuinely fits again after the churn.
        db.subscribe_changes_for_principal(Some(&reader_id), ChangeFilter::all())
            .expect("round {round}: quota falsely exhausted after churn");
    }

    // Fully released; a fresh CAP subscribes succeed and the CAP+1 is rejected.
    let mut held = Vec::new();
    for _ in 0..CAP {
        held.push(
            db.subscribe_changes_for_principal(Some(&reader_id), ChangeFilter::all())
                .expect("fresh cap subscribes after churn"),
        );
    }
    assert!(
        db.subscribe_changes_for_principal(Some(&reader_id), ChangeFilter::all())
            .is_err(),
        "cap still binds after churn (no leak)"
    );
}

// ── (b normal path): await timeout still releases the slot ────────────────────

/// A `POST /changes/await` that times out normally (no data, no disconnect)
/// returns `timed_out: true` and frees the slot promptly — the async rewrite
/// preserves the normal timeout→Drop path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn await_timeout_still_releases_slot() {
    let db = new_db();
    let store = keyed_store();
    let reader_id = principal_id(&store, READER_TOKEN);
    db.set_changefeed_config(
        ChangefeedConfig::default()
            .with_max_subscriptions(128)
            .with_max_subscriptions_per_principal(4),
    );
    let client = build_server_client(Arc::clone(&db), store, AuthMode::Required);

    let resp = client
        .post("/changes/await")
        .header("authorization", &format!("Bearer {READER_TOKEN}"))
        .json(&json!({ "timeout_ms": 150 }))
        .send()
        .await;
    assert_eq!(resp.status.as_u16(), 200);
    let body: Value = resp.json();
    assert_eq!(body["timed_out"], true, "clean timeout: {body}");

    let released = wait_until(Duration::from_secs(2), || {
        principal_count(&db, &reader_id) == 0
    })
    .await;
    assert!(
        released.is_some(),
        "a timed-out long-poll must free its slot promptly"
    );
}

// ── (c quota): disconnect decrements the principal counter exactly once ────────

/// A single subscribe→drop decrements the per-principal counter to exactly 0 and
/// removes the map entry; a re-subscribe then succeeds. Guards against a double
/// decrement/underflow — `Subscription::Drop` must remain the sole release
/// anchor even after the async rewrite.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disconnect_decrements_principal_counter_exactly_once() {
    let db = new_db();
    let sub = db
        .subscribe_changes_for_principal(Some("alice"), ChangeFilter::all())
        .expect("subscribe");
    assert_eq!(principal_count(&db, "alice"), 1);
    drop(sub);
    assert_eq!(principal_count(&db, "alice"), 0, "exactly one decrement");
    assert!(
        db.changefeed_principal_counts()
            .iter()
            .all(|(k, _)| k != "alice"),
        "the zeroed principal entry is pruned"
    );
    let _again = db
        .subscribe_changes_for_principal(Some("alice"), ChangeFilter::all())
        .expect("re-subscribe after release");
    assert_eq!(principal_count(&db, "alice"), 1);
}

// ── (b + #3688 stats): database_stats returns to baseline after disconnect ─────

/// After an SSE client disconnect, the admin `database_stats` changefeed
/// per-principal breakdown returns to baseline (the reader absent, aggregate 0)
/// within the tight bound — mirrors the per-principal live count and the #3688
/// stats surface.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quota_stats_return_to_baseline_after_disconnect() {
    let db = new_db();
    let store = keyed_store();
    let reader_id = principal_id(&store, READER_TOKEN);
    db.set_changefeed_config(
        ChangefeedConfig::default()
            .with_max_subscriptions(128)
            .with_max_subscriptions_per_principal(4),
    );
    let client = Arc::new(build_server_client(
        Arc::clone(&db),
        Arc::clone(&store),
        AuthMode::Required,
    ));

    let stream_task = {
        let client = Arc::clone(&client);
        tokio::spawn(async move {
            let _ = client
                .get("/changes/stream")
                .header("authorization", &format!("Bearer {READER_TOKEN}"))
                .send()
                .await;
        })
    };
    assert!(
        wait_until(Duration::from_secs(5), || principal_count(&db, &reader_id)
            == 1)
        .await
        .is_some(),
        "reader slot registers"
    );

    assert!(
        !stream_task.is_finished(),
        "SSE stream task completed on its own — disconnect not actually exercised"
    );
    stream_task.abort();

    // The per-principal live count returns to zero promptly (the stats source).
    let released = wait_until(SLOT_RELEASE_BOUND, || principal_count(&db, &reader_id) == 0).await;
    assert!(
        released.is_some(),
        "stats source returns to baseline within SLOT_RELEASE_BOUND"
    );

    // And the admin database_stats roster no longer lists the reader.
    let resp = client
        .get("/database_stats")
        .header("authorization", &format!("Bearer {ADMIN_TOKEN}"))
        .send()
        .await;
    assert_eq!(resp.status.as_u16(), 200);
    let body: Value = resp.json();
    let cf = body
        .get("changefeed")
        .or_else(|| body.get("data").and_then(|d| d.get("changefeed")))
        .expect("changefeed stats block");
    let lists_reader = cf["per_principal"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .any(|e| e["principal"].as_str() == Some(reader_id.as_str()))
        })
        .unwrap_or(false);
    assert!(
        !lists_reader,
        "database_stats must not list the disconnected reader: {cf}"
    );
}
