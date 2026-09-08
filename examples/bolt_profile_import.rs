//! Bolt profiling harness: bulk CSV node import, driven through the public
//! `AletheiaDB::import()` entry point exactly like `benches/import_throughput.rs`.
//!
//! This is not a criterion benchmark — it runs the workload exactly once so that
//! `valgrind --tool=callgrind` / `--tool=dhat` produce a clean, deterministic
//! instruction/allocation count uncontaminated by criterion's warmup and sampling loop.
//!
//! Row count defaults to 20_000 (override with `BOLT_PROFILE_ROWS`) — large enough to
//! dominate the profile with import cost rather than process startup.
//!
//! Reproduce:
//!   cargo build --profile bench --example bolt_profile_import
//!   valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.out \
//!       ./target/release/examples/bolt_profile_import
//!   callgrind_annotate /tmp/callgrind.out

use aletheiadb::AletheiaDB;
use aletheiadb::api::import::{ColumnType, LabelSource, NodeMapping};
use std::hint::black_box;
use std::io::Write;

fn rows() -> usize {
    std::env::var("BOLT_PROFILE_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000)
}

fn make_node_csv(n: usize) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("nodes.csv");
    let mut file = std::fs::File::create(&path).expect("create csv");
    writeln!(file, "id,name,age").unwrap();
    for i in 0..n {
        writeln!(file, "n{i},Name{i},{}", i % 100).unwrap();
    }
    file.flush().unwrap();
    (dir, path)
}

fn node_mapping() -> NodeMapping {
    NodeMapping::new(LabelSource::fixed("Person"), "id")
        .property("name", "name", ColumnType::String)
        .property("age", "age", ColumnType::Int)
}

fn main() {
    let n = rows();
    let (_dir, csv_path) = make_node_csv(n);

    let db = AletheiaDB::new().expect("db");
    let mut importer = db.import();
    let report = importer
        .nodes_from_csv(&csv_path, node_mapping())
        .expect("import");

    black_box(report.nodes_imported);
    println!("imported {} nodes", report.nodes_imported);
}
