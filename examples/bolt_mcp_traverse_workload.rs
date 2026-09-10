//! Deterministic MCP `traverse` tool workload for instruction-count profiling.
//!
//! Unlike `bolt_workload.rs` (which hand-simulates multi-hop traversal via
//! raw `get_outgoing_edges`/`get_edge` calls on `AletheiaDB` directly), this
//! harness drives the actual MCP `traverse` tool entry point
//! (`AletheiaMcpServer::traverse`) -- the exact code `handle_traverse` runs
//! for every `"traverse"` tool call an LLM client makes over stdio. Calling
//! it in-process (rather than through `mcp_round_trip.rs`'s subprocess/stdio
//! transport) keeps the trace free of IPC/process overhead unrelated to the
//! tool dispatch logic itself, mirroring the "direct-call floor" pattern that
//! benchmark already uses for its own baseline.
//!
//! This is a plain binary, not a criterion benchmark, for the same reason as
//! `bolt_workload.rs`: criterion's iteration/warmup loop is prohibitively
//! slow stacked under valgrind's own 20-50x slowdown. One fixed,
//! deterministic pass keeps instruction counts reproducible run to run.
//!
//! ```text
//! cargo build --profile bench --example bolt_mcp_traverse_workload --features mcp-server
//! valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.out \
//!     target/release/examples/bolt_mcp_traverse_workload
//! callgrind_annotate /tmp/callgrind.out | less
//!
//! valgrind --tool=dhat --dhat-out-file=/tmp/dhat.out \
//!     target/release/examples/bolt_mcp_traverse_workload
//! ```
//!
//! Env vars: `BOLT_NODES` (default 1000), `BOLT_OUT_DEGREE` (default 6),
//! `BOLT_READ_ITERS` (default 3000), `BOLT_DEPTH` (default 3).

use aletheiadb::mcp::{AletheiaMcpServer, QueryLimitsConfig, TraverseRequest};
use aletheiadb::prelude::*;
use std::sync::Arc;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let node_count = env_usize("BOLT_NODES", 1_000);
    let out_degree = env_usize("BOLT_OUT_DEGREE", 6);
    let read_iters = env_usize("BOLT_READ_ITERS", 3_000);
    let depth = env_usize("BOLT_DEPTH", 3);

    let db = AletheiaDB::new().expect("create ephemeral db");

    // --- Write phase: a social-graph-shaped dataset (same shape as
    // `bolt_workload.rs`, alternating KNOWS/FOLLOWS labels on each node's
    // outgoing edges so a label-filtered traversal exercises the
    // `get_outgoing_edges_with_label` current-state pre-filtered path). ---
    let mut node_ids = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let props = PropertyMapBuilder::new()
            .insert("name", format!("Person{i}"))
            .insert("age", (i % 80) as i64)
            .build();
        node_ids.push(db.create_node("Person", props).expect("create_node"));
    }

    for i in 0..node_count {
        for j in 0..out_degree {
            let target = node_ids[(i + j + 1) % node_count];
            let label = if j % 2 == 0 { "KNOWS" } else { "FOLLOWS" };
            let props = PropertyMapBuilder::new()
                .insert("weight", (i + j) as i64)
                .build();
            db.create_edge(node_ids[i], target, label, props)
                .expect("create_edge");
        }
    }

    // Issue #3368's per-call wall-clock-timeout race spawns an OS thread per
    // `traverse` call under the default (30s) resource-limit config -- a
    // large, already-documented, separately-tracked cost ("a quantifying
    // micro-benchmark deferred to Lane-2" per the MCP server docs) that would
    // otherwise swamp an instruction-count trace of the traversal dispatch
    // logic itself. Disabling it here isolates what this harness exists to
    // profile: `handle_traverse`/`traversal_next_hops`/`resolve_edge_hop`.
    let server =
        AletheiaMcpServer::new(Arc::new(db)).with_query_limits(QueryLimitsConfig::disabled());

    // --- Read phase: the "traverse" MCP tool, current-state (no as_of),
    // outgoing direction, label-filtered -- the exact request shape an LLM
    // client sends to explore a node's KNOWS neighborhood. ---
    let mut sink: u64 = 0;
    for i in 0..read_iters {
        let start = node_ids[i % node_count];
        // `TraverseRequest` is `#[non_exhaustive]`, so build it the same way
        // `handle_traverse` itself does on the wire: deserialize from JSON.
        let req: TraverseRequest = serde_json::from_value(serde_json::json!({
            "start_node_id": start.as_u64(),
            "edge_label": "KNOWS",
            "direction": "outgoing",
            "depth": depth,
            "limit": 500,
        }))
        .expect("valid TraverseRequest");
        let response = server.traverse(req);
        sink = sink.wrapping_add(response.len() as u64);
    }

    println!(
        "sink={sink} nodes={node_count} edges={} read_iters={read_iters} depth={depth}",
        node_count * out_degree
    );
}
