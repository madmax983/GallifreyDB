# Getting Started

This guide walks you through the core operations: creating a database, adding
nodes and edges, querying the graph, updating data, and running a time-travel query.
By the end you'll have a working mental model of how AletheiaDB works day-to-day.

**Prerequisites:** Rust 1.92+ and AletheiaDB in your `Cargo.toml`. See [Installation](installation.md).

---

## 1. Create a Database

```rust
use aletheiadb::prelude::*;

fn main() -> Result<()> {
    let db = AletheiaDB::new()?;
    // db is ready to use
    Ok(())
}
```

`AletheiaDB::new()` creates an **ephemeral**, tempdir-backed database — it
does not survive process restart. That's the right choice for this guide's
examples, tests, and scratch sessions. For a database that persists at a
path you choose, use `AletheiaDB::open(path)` instead:

```rust
let db = AletheiaDB::open("./mydb")?;
```

> **Note on leftover state**: Unlike `new()`, data at a path passed to
> `open()` survives between runs. If you run this guide's examples against
> the same `./mydb` directory more than once, each run adds its own nodes on
> top of the last rather than starting fresh. Delete `./mydb` between runs,
> or point each run at its own path, if you want a clean slate.

For custom durability settings or cold storage, use
`AletheiaDB::with_unified_config()` — see [Persistence Guide](PERSISTENCE.md).

---

## 2. Create Nodes

Nodes have a **label** (a type name) and **properties** (key/value pairs).

```rust
use aletheiadb::prelude::*;

let db = AletheiaDB::new()?;

let alice_id = db.create_node("Person", properties! {
    "name" => "Alice",
    "age"  => 30i64,
})?;

let bob_id = db.create_node("Person", properties! {
    "name" => "Bob",
    "age"  => 28i64,
})?;

println!("Alice: {:?}, Bob: {:?}", alice_id, bob_id);
```

`create_node` returns a `NodeId`. Hold onto it — you'll use it for edges and queries.

---

## 3. Create Edges

Edges connect two nodes with a **relationship type** and optional properties.

```rust
let edge_id = db.create_edge(alice_id, bob_id, "KNOWS", properties! {
    "since" => 2023i64,
})?;
```

Edges are directed: this edge goes from Alice → Bob over the `KNOWS` relationship.

---

## 4. Read Nodes and Edges

```rust
// Read a single node
let alice = db.get_node(alice_id)?;
println!("Label: {}", alice.label);                          // "Person"
println!("Name: {:?}", alice.properties.get("name"));        // Some("Alice")

// Read an edge
let edge = db.get_edge(edge_id)?;
println!("From: {:?}, To: {:?}", edge.source, edge.target);

// Traverse: get edge IDs for a specific relationship type, then fetch each edge
let edge_ids = db.get_outgoing_edges_with_label(alice_id, "KNOWS");
for eid in edge_ids {
    let e = db.get_edge(eid)?;
    println!("Alice knows node: {:?}", e.target);
}
```

---

## 5. Update Data

Use an explicit write transaction to update nodes or edges atomically.

```rust
db.write(|tx| {
    tx.update_node(alice_id, properties! {
        "age" => 31i64,   // Alice's birthday
    })
})?;
```

All operations inside `db.write(|tx| { ... })` are atomic — they either all
succeed or all roll back.

---

## 6. Multi-Operation Transactions

For operations that must succeed together:

```rust
let (carol_id, dave_id) = db.write(|tx| -> Result<(NodeId, NodeId)> {
    let carol = tx.create_node("Person", properties! { "name" => "Carol" })?;
    let dave  = tx.create_node("Person", properties! { "name" => "Dave"  })?;
    tx.create_edge(carol, dave, "WORKS_WITH", properties! {})?;
    Ok((carol, dave))
})?;
```

---

## 7. Time-Travel Queries

This is where AletheiaDB goes beyond ordinary graph databases. Every write is
recorded with two timestamps: **valid time** (when the fact was true in the real
world) and **transaction time** (when it was recorded in the database). You can
query at any point in either dimension.

```rust
use aletheiadb::time;

// Record the time before we make a change
let before_update = time::now();

// Update Alice's role
db.write(|tx| {
    tx.update_node(alice_id, properties! { "role" => "engineer" })
})?;

// Query what Alice looked like *before* the update
let historical_alice = db.get_node_at_time(
    alice_id,
    before_update,  // valid time: what was true at this moment
    before_update,  // transaction time: what the DB knew at this moment
)?;

// historical_alice.properties does not have "role" — the update hadn't happened yet
println!("{:?}", historical_alice.properties.get("role")); // None
```

See [Core Concepts](core-concepts.md) for a deeper explanation of valid time vs
transaction time.

---

## 8. Vector Search (Optional)

AletheiaDB stores dense vector embeddings as node properties and indexes them
with HNSW for fast k-NN search. Enable the index before inserting nodes.

```rust
use aletheiadb::{AletheiaDB, HnswConfig, DistanceMetric, SimilarityQuery};

let db = AletheiaDB::new()?;

// Enable HNSW indexing on the "embedding" property
db.vector_index("embedding")
    .hnsw(HnswConfig::new(384, DistanceMetric::Cosine))
    .enable()?;

let v1 = vec![0.1f32; 384];
let v2 = vec![0.9f32; 384]; // very different from v1

let doc1 = db.create_node("Document", properties! {
    "title"     => "Intro to Rust",
    "embedding" => &v1[..],
})?;

let _doc2 = db.create_node("Document", properties! {
    "title"     => "Advanced Rust",
    "embedding" => &v2[..],
})?;

// Find the 10 nodes most similar to doc1
// (doc1 itself is excluded from results)
let similar = db.similarity_search(SimilarityQuery::from_node(doc1).k(10))?;
for (node_id, score) in similar {
    println!("Node {:?} similarity: {:.3}", node_id, score);
}
```

---

## 9. Hybrid Queries (Graph + Vector + Temporal)

All three dimensions can be combined in a single query:

```rust
use aletheiadb::time;

let query_embedding = vec![0.1f32; 384];
let t = time::now();

// "Who does Alice know (graph), ranked by semantic similarity to my embedding
//  (vector), as of a specific point in time (temporal)?"
let results = db.query()
    .as_of(t, t)                                      // temporal
    .start(alice_id)                                  // graph: start node
    .traverse("KNOWS")                                // graph: hop
    .rank_by_similarity(&query_embedding, 10)         // vector: re-rank
    .execute(&db)?;

for row in results {
    let row = row?;
    println!("Found {:?} with score {:?}", row.entity, row.score);
}
```

See [Hybrid Query Guide](hybrid-query-guide.md) for the full API.

---

## 10. Delete

```rust
// Delete a specific edge
db.write(|tx| tx.delete_edge(edge_id))?;

// Delete a node AND all its connected edges (recommended)
db.write(|tx| tx.delete_node_cascade(alice_id))?;

// Delete node only (leaves orphaned edges — use carefully)
// db.write(|tx| tx.delete_node(alice_id))?;
```

Prefer `delete_node_cascade` to avoid orphaned edges. See [Known Limitations](../../README.md#known-limitations).

---

## What's Next

You now know the core loop: create nodes and edges, query and traverse, update
with transactions, look back in time. From here:

- **Understand the model** → [Core Concepts](core-concepts.md)
- **Persist data to disk** → [Persistence Guide](PERSISTENCE.md)
- **Deep vector search** → [Vector Search Integration](vector-search-integration.md)
- **Complex hybrid queries** → [Hybrid Query Guide](hybrid-query-guide.md)
- **Production observability** → [docs/OBSERVABILITY.md](../OBSERVABILITY.md)
- **Scale horizontally** → [Sharding Guide](sharding-guide.md)
- **Connect to Claude / LLMs** → Run `cargo run --bin aletheia-mcp --features mcp-server`
