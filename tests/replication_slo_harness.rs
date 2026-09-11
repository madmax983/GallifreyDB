//! RPO/RTO/lag/write-overhead measurement harness for asynchronous
//! replication (Issue #3355, Slice D, AC7 + the design's success metrics).
//!
//! # What these metrics mean
//!
//! - **RPO** (Recovery Point Objective, async mode) = how much acknowledged
//!   write history can be lost on failover = the replication lag at the
//!   moment of failure. It is bounded by `entries_behind`/`lag_ms` as
//!   reported by [`AletheiaDB::replication_progress`] -- the `lag_*` tests
//!   below measure this directly by sampling progress under sustained write
//!   load. (The chaos test, `tests/replication_chaos.rs`, additionally
//!   measures the REALIZED RPO of one concrete kill -- the count of
//!   acknowledged writes that fell in the lag window at the moment of
//!   promotion.)
//! - **RTO** (Recovery Time Objective) = how long failover itself takes =
//!   the wall-clock duration of [`AletheiaDB::promote_to_primary`]. The
//!   `rto_*` tests below measure this directly.
//! - **Write overhead** = how much the primary's own write latency degrades
//!   by having a replica attached at all (the design is pull-based
//!   specifically so this should be ~0%; the `write_overhead_*` tests
//!   measure the actual delta with generous CI headroom).
//!
//! # CI-sized vs. reference variants
//!
//! Every metric has a small `*_ci_sized` `#[test]` that always runs (small
//! fixture, generous bounds well above normal CI noise) and prints its
//! measurement with a stable `SLO <metric>=<value>` prefix so a docs/CI
//! pipeline can scrape it without parsing assertions. The `*_reference_*`
//! variants are `#[ignore]`-d: they replay the issue's actual fixture size
//! (10K nodes / 50K edges) against the issue's real target (RTO < 10s). Run
//! them explicitly:
//!
//! ```text
//! cargo test --test replication_slo_harness -- --ignored
//! ```
//!
//! # Concurrency note
//!
//! All load generation here is **single-threaded** (see the "Deviation"
//! note in `tests/replication_chaos.rs`'s module doc comment): concurrent
//! multi-threaded writers before a crash/reopen can hit a pre-existing,
//! replication-independent WAL-replay bug this chaos-test effort
//! discovered (a replayed `CreateNode`'s locally-reinvented `version_id` can
//! collide with a later write's real logged `version_id`). None of these
//! measurements need concurrent writers, so they simply don't exercise that
//! landmine.
//!
//! Run with: `cargo test --test replication_slo_harness --features config-toml`

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use aletheiadb::config::{AletheiaDBConfig, WalConfigBuilder};
use aletheiadb::{
    AletheiaDB, DurabilityMode, PropertyMap, PropertyMapBuilder, ReplicationOptions,
    ReplicationServer, TcpSource,
};
use serial_test::serial;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn props(v: &str) -> PropertyMap {
    PropertyMapBuilder::new().insert("v", v).build()
}

fn loopback_addr(addr: std::net::SocketAddr) -> String {
    format!("127.0.0.1:{}", addr.port())
}

fn wait_until<F: FnMut() -> bool>(timeout: Duration, mut check: F) -> bool {
    let start = Instant::now();
    loop {
        if check() {
            return true;
        }
        if start.elapsed() >= timeout {
            return check();
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A durable primary: `Synchronous` durability, matching the rest of the
/// replication test suite (the feed only ever ships durable/flushed data).
fn build_durable_primary() -> (tempfile::TempDir, AletheiaDB) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = AletheiaDBConfig::builder()
        .wal(
            WalConfigBuilder::new()
                .wal_dir(dir.path().join("wal"))
                .durability_mode(DurabilityMode::Synchronous)
                .build(),
        )
        .build();
    let db = AletheiaDB::with_unified_config(config).expect("create primary");
    (dir, db)
}

fn fast_opts() -> ReplicationOptions {
    ReplicationOptions {
        poll_interval: Duration::from_millis(10),
        batch_max_entries: 1000,
        persist_every: None,
        persist_every_entries: 0,
    }
}

/// Populate `db` with `node_count` nodes and up to `edge_count` edges (a
/// simple chain/fan pattern -- realistic bi-temporal graph shape, not just
/// isolated nodes), single-threaded and sequential.
fn seed_fixture(
    db: &AletheiaDB,
    node_count: usize,
    edge_count: usize,
) -> Vec<aletheiadb::core::id::NodeId> {
    let mut ids = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let id = db
            .create_node("FixtureNode", props(&format!("n{i}")))
            .expect("create fixture node");
        ids.push(id);
    }
    let mut edges_made = 0usize;
    'outer: for pair in ids.windows(2) {
        for _ in 0..((edge_count / node_count.max(1)).max(1)) {
            if edges_made >= edge_count {
                break 'outer;
            }
            db.create_edge(pair[0], pair[1], "LINKS", PropertyMapBuilder::new().build())
                .expect("create fixture edge");
            edges_made += 1;
        }
    }
    ids
}

/// Percentile (nearest-rank) over a slice of already-collected `u64`
/// samples. `pct` in `[0.0, 100.0]`. Returns `0` for an empty slice.
fn percentile(samples: &mut [u64], pct: f64) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    let rank = ((pct / 100.0) * samples.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(samples.len() - 1);
    samples[idx]
}

// ---------------------------------------------------------------------------
// RTO (promotion latency)
// ---------------------------------------------------------------------------

/// Build a small fixture, replicate it, promote, and measure RTO. CI-sized:
/// generous 30s bound (the design targets <10s; this just guards against a
/// gross regression/hang under CI noise).
#[test]
#[serial(replication_slo)]
fn rto_ci_sized() {
    let (_dir, primary) = build_durable_primary();
    let primary = Arc::new(primary);
    seed_fixture(&primary, 1_000, 5_000);

    let server = ReplicationServer::start(Arc::clone(&primary), "127.0.0.1:0", "rto-tok".into())
        .expect("start replication server");
    let addr = loopback_addr(server.local_addr());

    let replica = AletheiaDB::new().expect("create replica");
    replica
        .start_replication(Box::new(TcpSource::new(addr, "rto-tok")), fast_opts())
        .expect("start replication");

    assert!(
        wait_until(Duration::from_secs(30), || replica.node_count()
            == primary.node_count()
            && replica.edge_count() == primary.edge_count()),
        "replica must converge before promotion can be meaningfully timed (primary: {}/{}, \
         replica: {}/{})",
        primary.node_count(),
        primary.edge_count(),
        replica.node_count(),
        replica.edge_count(),
    );

    let start = Instant::now();
    let report = replica.promote_to_primary().expect("promote");
    let elapsed = start.elapsed();

    println!("SLO rto_promotion_ms={}", elapsed.as_millis());
    assert!(report.applier_stopped);
    assert!(
        elapsed < Duration::from_secs(30),
        "CI-sized RTO must be well under 30s (measured {elapsed:?}); the design target is <10s \
         on the full 10K/50K reference fixture -- see rto_reference_10k_nodes_50k_edges"
    );

    server.shutdown();
}

/// Reference-sized: the issue's actual 10K nodes / 50K edges fixture,
/// asserting the issue's real RTO target (<10s).
///
/// Run explicitly: `cargo test --test replication_slo_harness -- --ignored`
#[test]
#[ignore = "reference-sized fixture (10K nodes / 50K edges); run explicitly with --ignored"]
#[serial(replication_slo)]
fn rto_reference_10k_nodes_50k_edges() {
    let (_dir, primary) = build_durable_primary();
    let primary = Arc::new(primary);
    seed_fixture(&primary, 10_000, 50_000);

    let server =
        ReplicationServer::start(Arc::clone(&primary), "127.0.0.1:0", "rto-ref-tok".into())
            .expect("start replication server");
    let addr = loopback_addr(server.local_addr());

    let replica = AletheiaDB::new().expect("create replica");
    replica
        .start_replication(
            Box::new(TcpSource::new(addr, "rto-ref-tok")),
            ReplicationOptions {
                poll_interval: Duration::from_millis(10),
                batch_max_entries: 5000,
                persist_every: None,
                persist_every_entries: 0,
            },
        )
        .expect("start replication");

    assert!(
        wait_until(Duration::from_secs(120), || replica.node_count()
            == primary.node_count()
            && replica.edge_count() == primary.edge_count()),
        "replica must converge on the reference fixture before promotion (primary: {}/{}, \
         replica: {}/{})",
        primary.node_count(),
        primary.edge_count(),
        replica.node_count(),
        replica.edge_count(),
    );

    let start = Instant::now();
    let report = replica.promote_to_primary().expect("promote");
    let elapsed = start.elapsed();

    println!("SLO rto_promotion_reference_ms={}", elapsed.as_millis());
    assert!(report.applier_stopped);
    assert!(
        elapsed < Duration::from_secs(10),
        "RTO on the 10K/50K reference fixture must be under 10s (measured {elapsed:?})"
    );

    server.shutdown();
}

// ---------------------------------------------------------------------------
// Lag (RPO proxy: replication lag under sustained load)
// ---------------------------------------------------------------------------

/// Sustained write load for `duration` while sampling `replication_progress()`
/// on `replica` every ~50ms. Returns (lag_ms samples, entries_behind samples)
/// -- only samples where the applier had already observed the primary's
/// flushed position at least once (both fields go from `None` to `Some` once
/// the first fetch completes).
fn sample_lag_under_load(
    primary: &Arc<AletheiaDB>,
    replica: &AletheiaDB,
    duration: Duration,
) -> (Vec<u64>, Vec<u64>) {
    let stop = Arc::new(AtomicBool::new(false));

    let writer_primary = Arc::clone(primary);
    let writer_stop = Arc::clone(&stop);
    let writer = std::thread::Builder::new()
        .name("slo-lag-writer".into())
        .spawn(move || {
            let mut i: u64 = 0;
            while !writer_stop.load(Ordering::Relaxed) {
                let _ = writer_primary.create_node("LagLoadNode", props(&format!("l{i}")));
                i += 1;
            }
        })
        .expect("spawn writer thread");

    let mut lag_ms_samples = Vec::new();
    let mut entries_behind_samples = Vec::new();
    let sampling_deadline = Instant::now() + duration;
    while Instant::now() < sampling_deadline {
        if let Some(p) = replica.replication_progress() {
            if let Some(lag) = p.lag_ms {
                lag_ms_samples.push(lag);
            }
            if let Some(behind) = p.entries_behind {
                entries_behind_samples.push(behind);
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    stop.store(true, Ordering::Relaxed);
    writer.join().expect("writer thread panicked");

    (lag_ms_samples, entries_behind_samples)
}

/// CI-sized sustained-load lag measurement: ~5s of single-threaded write
/// load against a streaming replica, sampling lag every ~50ms. Generous
/// bound (p99 < 10s) -- the design's steady-state lag is expected to be
/// milliseconds; this just guards against a gross regression/stall.
#[test]
#[serial(replication_slo)]
fn lag_ci_sized() {
    let (_dir, primary) = build_durable_primary();
    let primary = Arc::new(primary);

    let server = ReplicationServer::start(Arc::clone(&primary), "127.0.0.1:0", "lag-tok".into())
        .expect("start replication server");
    let addr = loopback_addr(server.local_addr());

    let replica = AletheiaDB::new().expect("create replica");
    replica
        .start_replication(Box::new(TcpSource::new(addr, "lag-tok")), fast_opts())
        .expect("start replication");

    let (mut lag_ms_samples, mut entries_behind_samples) =
        sample_lag_under_load(&primary, &replica, Duration::from_secs(5));

    assert!(
        !lag_ms_samples.is_empty(),
        "expected at least one lag sample during 5s of sustained load"
    );

    let p50 = percentile(&mut lag_ms_samples, 50.0);
    let p99 = percentile(&mut lag_ms_samples, 99.0);
    let behind_p99 = percentile(&mut entries_behind_samples, 99.0);

    println!("SLO lag_p50_ms={p50}");
    println!("SLO lag_p99_ms={p99}");
    println!("SLO lag_entries_behind_p99={behind_p99}");

    assert!(
        p99 < 10_000,
        "CI-sized lag p99 must be well under 10s (measured {p99}ms); steady-state lag is \
         expected to be milliseconds"
    );

    server.shutdown();
    // Let the replica settle so its own background thread doesn't outlive
    // the test needlessly.
    let _ = wait_until(Duration::from_secs(2), || {
        replica
            .replication_progress()
            .map(|p| p.state != "streaming")
            .unwrap_or(true)
    });
}

/// Reference-sized sustained-load lag measurement: same shape as
/// `lag_ci_sized` but over a longer window, printed for the docs slice.
/// Run explicitly: `cargo test --test replication_slo_harness -- --ignored`
#[test]
#[ignore = "longer sustained-load window; run explicitly with --ignored"]
#[serial(replication_slo)]
fn lag_reference_sustained() {
    let (_dir, primary) = build_durable_primary();
    let primary = Arc::new(primary);

    let server =
        ReplicationServer::start(Arc::clone(&primary), "127.0.0.1:0", "lag-ref-tok".into())
            .expect("start replication server");
    let addr = loopback_addr(server.local_addr());

    let replica = AletheiaDB::new().expect("create replica");
    replica
        .start_replication(Box::new(TcpSource::new(addr, "lag-ref-tok")), fast_opts())
        .expect("start replication");

    let (mut lag_ms_samples, mut entries_behind_samples) =
        sample_lag_under_load(&primary, &replica, Duration::from_secs(30));

    let p50 = percentile(&mut lag_ms_samples, 50.0);
    let p99 = percentile(&mut lag_ms_samples, 99.0);
    let behind_p99 = percentile(&mut entries_behind_samples, 99.0);

    println!("SLO lag_reference_p50_ms={p50}");
    println!("SLO lag_reference_p99_ms={p99}");
    println!("SLO lag_reference_entries_behind_p99={behind_p99}");

    server.shutdown();
}

// ---------------------------------------------------------------------------
// Write overhead (primary write latency with vs. without an attached replica)
// ---------------------------------------------------------------------------

/// Time `n` single-op `create_node` transactions against `db`, single-threaded.
fn time_n_writes(db: &AletheiaDB, n: usize, label_prefix: &str) -> Duration {
    let start = Instant::now();
    for i in 0..n {
        db.create_node("OverheadNode", props(&format!("{label_prefix}{i}")))
            .expect("create node");
    }
    start.elapsed()
}

/// A replica whose own durability cannot contend with the primary under
/// measurement (Issue #3784): [`DurabilityMode::Async`] fsyncs on a background
/// timer instead of once per applied write. The applier still streams and
/// applies every entry -- "a replica is attached" stays true in every way
/// that can touch the primary's write path -- but the replica's fsyncs stop
/// being the dominant term in the primary's write-latency measurement. (The
/// default `GroupCommit` already batches, but its flush cadence still tracks
/// the applier's write rate; a time-based background flush decouples the two
/// entirely.)
fn build_low_contention_replica() -> (tempfile::TempDir, AletheiaDB) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = AletheiaDBConfig::builder()
        .wal(
            WalConfigBuilder::new()
                .wal_dir(dir.path().join("wal"))
                .durability_mode(
                    DurabilityMode::async_mode_validated(100).expect("valid async mode"),
                )
                .build(),
        )
        .build();
    let db = AletheiaDB::with_unified_config(config).expect("create replica");
    (dir, db)
}

/// CI-sized write-overhead measurement, hardened against shared-runner noise
/// (Issue #3784).
///
/// The old design measured one 500-write burst on `primary_a` (no replica)
/// against one burst on `primary_b` (replica attached): cross-instance, single
/// shot, fixed order, and fsync-bound on both sides by the co-located
/// replica's own WAL fsyncs -- on a 2-core shared runner the ratio was
/// dominated by disk contention, not the primary's write path, and failed
/// unrelated PRs as a coin flip. Each bullet below answers one of #3784's
/// miscalibration points:
///
/// - **Same-instance A/B.** One primary is measured before and after the
///   replica attaches, so instance-level differences (tempdirs, WAL segment
///   state, allocator history) cancel instead of landing in the ratio.
/// - **Warmup + best-of-N.** A warmup burst (discarded) settles page cache /
///   allocator / WAL segment; then each side takes the *minimum* of `TRIALS`
///   timed bursts. Scheduling noise only ever slows a trial down, so the
///   minimum is the faithful estimator of the true cost; single-shot sampling
///   is gone. Both sides are measured warm, so the fixed order no longer
///   biases the ratio upward (residual second-phase warmth biases it
///   slightly *down*, i.e. away from the failure direction).
/// - **Replica-side contention excluded.** The replica runs in
///   [`DurabilityMode::Async`] (see [`build_low_contention_replica`]): its own
///   fsyncs are machine contention, not write-path overhead. The primary --
///   the thing under measurement -- stays fully `Synchronous`, and the
///   applier still streams and applies every entry.
/// - The 25% bound is unchanged: the pull-based design shows ~0% overhead by
///   construction (the feed reads flushed segment files independently of the
///   write path), and the measurement above is now quiet enough that 25% is
///   generous headroom rather than a coin flip.
///
/// Prints `SLO write_overhead_*` exactly as before for the docs/CI scrape.
#[test]
#[serial(replication_slo)]
fn write_overhead_ci_sized() {
    const N: usize = 500;
    const TRIALS: usize = 3;

    let (_dir, primary) = build_durable_primary();
    let primary = Arc::new(primary);

    // Warmup (discarded), then best-of-TRIALS baseline on the bare primary.
    time_n_writes(&primary, N, "warmup");
    let mut baseline = Duration::MAX;
    for t in 0..TRIALS {
        baseline = baseline.min(time_n_writes(&primary, N, &format!("base{t}")));
    }

    // Attach a streaming replica to the SAME primary.
    let (_replica_dir, replica) = build_low_contention_replica();
    let server =
        ReplicationServer::start(Arc::clone(&primary), "127.0.0.1:0", "overhead-tok".into())
            .expect("start replication server");
    let addr = loopback_addr(server.local_addr());
    replica
        .start_replication(Box::new(TcpSource::new(addr, "overhead-tok")), fast_opts())
        .expect("start replication");

    // Give the applier a moment to actually start streaming before timing,
    // so the measurement reflects steady-state overhead, not connection
    // setup.
    let _ = wait_until(Duration::from_secs(5), || {
        replica
            .replication_progress()
            .map(|p| p.state == "streaming" || p.state == "connecting")
            .unwrap_or(false)
    });

    // Warmup with the replica attached (discarded), then best-of-TRIALS.
    time_n_writes(&primary, N, "warmup-repl");
    let mut with_replica = Duration::MAX;
    for t in 0..TRIALS {
        with_replica = with_replica.min(time_n_writes(&primary, N, &format!("repl{t}")));
    }

    let baseline_ms = baseline.as_secs_f64() * 1000.0;
    let with_replica_ms = with_replica.as_secs_f64() * 1000.0;
    let ratio = if baseline_ms > 0.0 {
        (with_replica_ms - baseline_ms) / baseline_ms
    } else {
        0.0
    };

    println!("SLO write_overhead_baseline_ms={baseline_ms:.3}");
    println!("SLO write_overhead_with_replica_ms={with_replica_ms:.3}");
    println!("SLO write_overhead_ratio={ratio:.4}");

    // The replica runs co-located with the primary in this harness, so the
    // measured "overhead" includes plain machine contention (the applier's
    // CPU + fsyncs competing for the same cores/disk), not just the primary's
    // write path -- which has no replication hooks at all. Verification of the
    // hardened measurement on a loaded 2-core Linux runner still failed a 25%
    // hard gate 5/10 runs (ratios up to 13x), so per #3784's authorized
    // fallback the gate is log-only on all platforms: the `SLO
    // write_overhead_*` lines above keep the number visible in CI without
    // gating merges on runner weather. (This subsumes the old Windows-only
    // log-only carve-out.)
    if ratio >= 0.25 {
        println!(
            "SLO write_overhead_gate=WARN ratio {:.2}% exceeded 25% \
             (baseline={baseline_ms:.3}ms, with_replica={with_replica_ms:.3}ms) -- \
             log-only; see #3784",
            ratio * 100.0
        );
    } else {
        println!("SLO write_overhead_gate=OK");
    }

    server.shutdown();
}
