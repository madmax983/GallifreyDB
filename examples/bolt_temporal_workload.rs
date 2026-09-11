//! Deterministic point-in-time (bi-temporal) read workload for instruction-count
//! profiling of `AletheiaDB::get_node_at_time`'s full read path: the
//! `EntityTimelines`/`EntityTimeline` bi-temporal index lookup
//! (`find_node_version_at_time`) followed by anchor+delta property
//! reconstruction (`HistoricalStorage::reconstruct_node_properties`).
//!
//! `examples/bolt_workload.rs` exercises current-state reads only (`get_node`,
//! `traverse`, `find_nodes_by_property`) and never builds up version history, so
//! it never touches either of these -- the "<10ms reconstruction" and temporal
//! index paths CLAUDE.md documents. This harness fills that gap: it exercises
//! the public `AletheiaDB` API the way an LLM agent issuing "what did we know
//! about X at time T" queries does (`get_node_at_time`), through the exact
//! call shape the `get_node_at_time` MCP tool and `AS OF` Cypher/AQL both use.
//!
//! The write phase backdates each node's updates with explicit `valid_from`
//! timestamps (`create_node_with_valid_time` / `update_node_with_valid_time`),
//! producing real anchor+delta chains under the configured `anchor_interval`.
//! The read phase then issues point-in-time reads scattered across many
//! distinct (node, valid_time) pairs -- deliberately more distinct pairs than
//! the configured `reconstruction_cache_size`, so the bounded per-version
//! property cache (`HistoricalStorage::node_property_cache`) can't hold the
//! working set and most reads take the full chain-walk reconstruction path,
//! exactly as a real deployment's cache does once historical query traffic
//! exceeds its capacity.
//!
//! This is a plain binary, not a criterion benchmark: criterion's
//! iteration/warmup loop is prohibitively slow stacked under valgrind's own
//! 20-50x slowdown. Running one fixed, deterministic pass keeps instruction
//! counts reproducible run to run and keeps the profiling session tractable.
//!
//! ```text
//! cargo build --profile bench --example bolt_temporal_workload
//! valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.out \
//!     target/release/examples/bolt_temporal_workload
//! callgrind_annotate /tmp/callgrind.out | less
//!
//! valgrind --tool=dhat --dhat-out-file=/tmp/dhat.out \
//!     target/release/examples/bolt_temporal_workload
//! ```
//!
//! Env vars: `BOLT_NODES` (default 500), `BOLT_VERSIONS_PER_NODE` (default 20),
//! `BOLT_CACHE_SIZE` (default 200), `BOLT_READ_ITERS` (default 8000).

use aletheiadb::config::{AletheiaDBConfig, HistoricalConfigBuilder, WalConfigBuilder};
use aletheiadb::core::temporal::time;
use aletheiadb::prelude::*;

const DAY_SECS: i64 = 86_400;
// 2020-01-01T00:00:00Z -- comfortably in the past, so every backdated
// `valid_from` below is also in the past (the write path rejects only
// future-dated facts more than one year out).
const BASE_SECS: i64 = 1_577_836_800;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let node_count = env_usize("BOLT_NODES", 500);
    let versions_per_node = env_usize("BOLT_VERSIONS_PER_NODE", 20);
    let cache_size = env_usize("BOLT_CACHE_SIZE", 200);
    let read_iters = env_usize("BOLT_READ_ITERS", 8_000);

    // Ephemeral, tempdir-backed WAL (mirrors `AletheiaDB::new()`) so every
    // invocation starts from a clean slate instead of replaying WAL entries
    // left behind by a prior run -- the default `wal_dir` is a relative path
    // and would otherwise accumulate state across runs from the same cwd.
    let tempdir = tempfile::Builder::new()
        .prefix("bolt-temporal-workload-")
        .tempdir()
        .expect("create tempdir");
    let wal = WalConfigBuilder::new()
        .wal_dir(tempdir.path().join("wal"))
        .build();
    let historical = HistoricalConfigBuilder::new()
        .anchor_interval(10)
        .expect("anchor_interval")
        .reconstruction_cache_size(cache_size)
        .expect("reconstruction_cache_size")
        .build();
    let config = AletheiaDBConfig::builder()
        .wal(wal)
        .historical(historical)
        .build();
    let db = AletheiaDB::with_unified_config(config).expect("create db");

    // --- Write phase: back-dated updates build real anchor+delta chains. ---
    // `valid_times[n][v]` is the valid_from timestamp of node n's v-th version,
    // so the read phase below can target an exact historical version.
    let mut node_ids = Vec::with_capacity(node_count);
    let mut valid_times = Vec::with_capacity(node_count);

    for i in 0..node_count {
        let v0 = time::from_secs(BASE_SECS);
        let props = PropertyMapBuilder::new()
            .insert("name", format!("Sensor{i}"))
            .insert("reading", 0_i64)
            .insert("status", "active")
            .build();
        let node_id = db
            .create_node_with_valid_time("Sensor", props, Some(v0))
            .expect("create_node_with_valid_time");

        let mut times = Vec::with_capacity(versions_per_node);
        times.push(v0);

        for v in 1..versions_per_node {
            let vt = time::from_secs(BASE_SECS + (v as i64) * DAY_SECS);
            let props = PropertyMapBuilder::new()
                .insert("reading", v as i64)
                .insert("status", if v % 5 == 0 { "maintenance" } else { "active" })
                .build();
            db.update_node_with_valid_time(node_id, props, Some(vt))
                .expect("update_node_with_valid_time");
            times.push(vt);
        }

        node_ids.push(node_id);
        valid_times.push(times);
    }

    // --- Read phase: point-in-time reads scattered across many distinct
    // (node, version) pairs -- node_count * versions_per_node distinct
    // reconstructions, far exceeding `cache_size`, so most calls miss the
    // property cache and take the full anchor+delta chain-walk path. ---
    let now = time::now();
    let mut sink: u64 = 0;
    for i in 0..read_iters {
        let node_idx = i % node_count;
        // Scatter the version index with a coprime-ish stride so consecutive
        // iterations don't revisit the same chain position (which an LRU
        // cache would just keep hot, defeating the point of this workload).
        let version_idx = (i * 7 + i / node_count) % versions_per_node;

        let node_id = node_ids[node_idx];
        let valid_time = valid_times[node_idx][version_idx];

        if let Ok(node) = db.get_node_at_time(node_id, valid_time, now) {
            sink = sink.wrapping_add(node.properties.len() as u64);
        }
    }

    println!(
        "sink={sink} nodes={node_count} versions_per_node={versions_per_node} \
cache_size={cache_size} read_iters={read_iters}"
    );
}
