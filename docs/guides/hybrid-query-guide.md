# Hybrid Query Guide

This guide covers AletheiaDB's hybrid query API, which combines **graph traversal**, **vector similarity**, and **bi-temporal queries** into a unified interface.

## Overview

Hybrid queries enable powerful patterns like:
- "Who does Alice know that's similar to Bob?" (Graph + Vector)
- "What was semantically similar to this concept in 2023?" (Vector + Temporal)
- "Who did Alice know in 2023 that was similar to Bob?" (Graph + Vector + Temporal)

## Quick Start

### Prerequisites

```rust
use aletheiadb::{AletheiaDB, PropertyMapBuilder};
use aletheiadb::index::vector::{HnswConfig, DistanceMetric};

// Create database with vector indexing
let db = AletheiaDB::new();
let config = HnswConfig::new(384, DistanceMetric::Cosine);
db.enable_vector_index("embedding", config)?;
```

### Three Ways to Query

**1. Direct Functions** (simple patterns)
```rust
use aletheiadb::query::hybrid::{traverse_and_rank, find_similar_as_of};

// Graph + Vector: Find neighbors ranked by similarity
let results = traverse_and_rank(&db, alice_id, "KNOWS", &query_embedding, 10)?;

// Temporal + Vector: Point-in-time semantic search
let results = find_similar_as_of(&db, &query_embedding, 10, timestamp)?;
```

> **Note:** With multiple temporal vector indexes enabled, the property-less
> `find_similar_as_of()` queries the alphabetically first indexed property;
> use `db.find_similar_as_of_in("property", ...)` to target a specific one.

**2. Query Builder** (complex compositions)
```rust
let results = db.query()
    .start(alice_id)
    .traverse("KNOWS")
    .rank_by_similarity(&bob_embedding, 10)
    .filter(Predicate::gt("score", 0.8))
    .execute(&db)?;
```

**3. Database Methods** (convenience)
```rust
let results = db.traverse_and_rank(alice_id, "KNOWS", &embedding, 10)?;
```

## Query Builder API

### State Machine

The query builder uses compile-time type states to prevent invalid queries:

```
┌─────────┐     start()      ┌──────────┐
│ Initial │ ───────────────► │ HasNodes │
└─────────┘                  └──────────┘
     │                            │
     │ find_similar()             │ traverse()
     ▼                            ▼
┌──────────────────┐    ┌─────────────────────┐
│ HasVectorResults │    │ HasTraversalResults │
└──────────────────┘    └─────────────────────┘
     │                            │
     │ traverse()                 │ rank_by_similarity()
     └────────────────────────────┘
```

### Source Operations

Start your query from one of these entry points:

```rust
// Single node
db.query().start(node_id)

// Multiple nodes
db.query().start_from(vec![node1, node2, node3])

// Scan all nodes
db.query().scan(None)               // All nodes
db.query().scan(Some("Person"))     // Filtered by label
db.query().scan_label("Document")   // Shorthand

// Vector search
db.query().find_similar(&embedding, 10)
db.query().find_similar_with_metric(&embedding, 10, DistanceMetric::Euclidean)
```

### Graph Operations

Traverse the graph structure:

```rust
// Single-hop outgoing edges
.traverse("KNOWS")          // With label filter
.traverse_all()             // Any label

// Multi-hop traversal
.traverse_n("KNOWS", 3)     // Exactly 3 hops

// Direction variants
.traverse_in("CREATED")     // Incoming edges
.traverse_both("LINKED")    // Both directions
```

### Vector Operations

Add semantic similarity:

```rust
// Rank current results by similarity to embedding
.rank_by_similarity(&embedding, top_k)

// Find nodes similar to another node
.similar_to(source_node_id, k)

// Advanced: custom property and label filter
.similar_to_builder(source_node_id, k)
    .property("custom_embedding")
    .label_filter("Document")
    .finish()
```

### Multi-Property Vector Queries

Query specific vector properties when you have multiple embeddings per node:

```rust
// Find similar using a specific property
db.query()
    .find_similar_builder(&query_embedding, 10)
    .property("content_embedding")  // Query content embeddings
    .metric(DistanceMetric::Cosine)
    .finish()
    .execute(&db)?;

// Traverse then rank by a specific property
db.query()
    .start(alice_id)
    .traverse("KNOWS")
    .rank_by_similarity_builder(&query_embedding, 10)
    .property("expertise_embedding")  // Rank by expertise similarity
    .finish()
    .execute(&db)?;

// Chain operations with different properties
db.query()
    .find_similar_builder(&title_embedding, 50)
    .property("title_embedding")
    .finish()
    .traverse("RELATED_TO")
    .rank_by_similarity_builder(&content_embedding, 10)
    .property("content_embedding")  // Rerank by content
    .finish()
    .execute(&db)?;
```

### Temporal Operations

Add time-travel capabilities:

```rust
// Point-in-time query (bi-temporal)
.as_of(valid_time, transaction_time)

// Time range query
.between(start_timestamp, end_timestamp)
```

### Filter Operations

Refine your results:

```rust
use aletheiadb::query::ir::Predicate;

// Property predicates
.filter(Predicate::eq("name", "Alice"))
.filter(Predicate::gt("age", 18))
.filter(Predicate::exists("email"))
.filter(Predicate::contains("bio", "engineer"))

// Label filter
.with_label("Person")

// Combine predicates
.filter(
    Predicate::eq("status", "active")
        .and(Predicate::gt("score", 0.8))
)
```

### Control Operations

Fine-tune execution:

```rust
.limit(10)              // Maximum results
.skip(20)               // Skip first N
.parallel()             // Enable parallel execution
.with_provenance()      // Include metadata (paths, timestamps)
.with_hint(IndexHint::UseVectorIndex)  // Optimizer hint
```

### Building and Executing

```rust
// Option 1: Build then execute separately
let query = db.query()
    .start(node_id)
    .traverse("KNOWS")
    .build();

let results = db.execute_query(query)?;

// Option 2: Execute directly (recommended)
let results = db.query()
    .start(node_id)
    .traverse("KNOWS")
    .execute(&db)?;
```

## Common Patterns

### Pattern 1: Find Similar Neighbors

"Who does Alice know that's most similar to Bob?"

```rust
let bob_node = db.get_node(bob_id)?;
let bob_embedding = bob_node
    .get_property("embedding")
    .and_then(|p| p.as_vector())
    .ok_or(Error::other("Property not found"))?
    .to_vec();

let results = db.query()
    .start(alice_id)
    .traverse("KNOWS")
    .rank_by_similarity(&bob_embedding, 10)
    .execute(&db)?;

for row in results {
    let row = row?;
    println!("Found: {:?}, similarity: {:?}", row.entity, row.score);
}
```

### Pattern 2: Temporal Semantic Search

"What documents were similar to this query in 2023?"

```rust
let timestamp_2023 = 1672531200000000; // 2023-01-01 in microseconds

let results = db.query()
    .as_of(timestamp_2023, timestamp_2023)
    .find_similar(&query_embedding, 10)
    .with_label("Document")
    .execute(&db)?;
```

### Pattern 3: Multi-Hop Traversal with Ranking

"Find friends-of-friends, ranked by expertise similarity"

```rust
let expertise_embedding = vec![/* ... */];

let results = db.query()
    .start(alice_id)
    .traverse_n("KNOWS", 2)  // 2-hop: friends of friends
    .with_label("Person")
    .rank_by_similarity(&expertise_embedding, 20)
    .filter(Predicate::exists("expertise"))
    .limit(10)
    .execute(&db)?;
```

### Pattern 4: Provenance Tracking

"Show how we reached each result"

```rust
let results = db.query()
    .start(alice_id)
    .traverse("KNOWS")
    .traverse("WORKS_AT")
    .with_provenance()
    .execute(&db)?;

for row in results {
    let row = row?;
    if let Some(path) = row.path {
        println!("Path: {:?}", path);
    }
}
```

### Pattern 5: Full Hybrid Query

"Who did Alice know in 2023 that was similar to Bob and had high scores?"

```rust
let results = db.query()
    .as_of(timestamp_2023, timestamp_2023)
    .start(alice_id)
    .traverse("KNOWS")
    .rank_by_similarity(&bob_embedding, 50)
    .filter(Predicate::gt("score", 0.8))
    .limit(10)
    .with_provenance()
    .execute(&db)?;
```

## Working with Results

### QueryRow Structure

```rust
pub struct QueryRow {
    pub entity: EntityResult,           // The entity found
    pub score: Option<f32>,             // Vector similarity score
    pub path: Option<Vec<EntityId>>,    // Traversal path
    pub timestamp: Option<Timestamp>,   // Temporal context
}

pub enum EntityResult {
    Node(Node),       // Full node with all properties
    Edge(Edge),       // Full edge data
    NodeId(NodeId),   // Lightweight ID reference
    EdgeId(EdgeId),
}
```

### Processing Results

```rust
let results = query.execute(&db)?;

// Iterate and handle each row
for row in results {
    let row = row?;  // Handle potential errors

    // Match on entity type
    match row.entity {
        EntityResult::Node(node) => {
            println!("Node: {} (label: {})", node.id, node.label);
            for (key, value) in node.properties.iter() {
                println!("  {}: {:?}", key, value);
            }
        }
        EntityResult::NodeId(id) => {
            // Fetch full node if needed
            let node = db.get_node(id)?;
        }
        _ => {}
    }

    // Check similarity score
    if let Some(score) = row.score {
        println!("  Similarity: {:.4}", score);
    }

    // Check provenance path
    if let Some(path) = &row.path {
        println!("  Path: {:?}", path);
    }
}
```

### Collecting All Results

```rust
// Collect into Vec
let all_results: Vec<QueryRow> = results
    .collect::<Result<Vec<_>, _>>()?;

// Get count
let count = all_results.len();
```

## Predicates Reference

### Comparison Predicates

```rust
Predicate::eq("field", value)     // Equal
Predicate::ne("field", value)     // Not equal
Predicate::gt("field", value)     // Greater than
Predicate::gte("field", value)    // Greater than or equal
Predicate::lt("field", value)     // Less than
Predicate::lte("field", value)    // Less than or equal
```

### String Predicates

```rust
Predicate::contains("field", "substring")
Predicate::starts_with("field", "prefix")
Predicate::ends_with("field", "suffix")
```

### Existence Predicates

```rust
Predicate::exists("field")        // Property exists
Predicate::not_exists("field")    // Property doesn't exist
```

### Membership Predicates

```rust
Predicate::in_list("status", vec!["active", "pending"])
```

### Logical Combinations

```rust
// AND
let p = Predicate::eq("a", 1).and(Predicate::eq("b", 2));

// OR
let p = Predicate::eq("a", 1).or(Predicate::eq("a", 2));

// NOT
let p = !Predicate::exists("deleted_at");

// Complex combinations
let p = Predicate::eq("status", "active")
    .and(
        Predicate::gt("score", 0.8)
            .or(Predicate::exists("featured"))
    );
```

### Type Conversions

Predicates accept various Rust types:

```rust
Predicate::eq("name", "Alice")        // &str
Predicate::eq("name", String::from("Alice"))  // String
Predicate::eq("age", 25i64)           // i64
Predicate::eq("active", true)         // bool
Predicate::gt("score", 0.5f64)        // f64
```

## Performance Considerations

### Choose the Right API Level

| Pattern | Recommended API | Reason |
|---------|-----------------|--------|
| Simple traverse + rank | `traverse_and_rank()` | Optimized path, minimal overhead |
| Temporal vector search | `find_similar_as_of()` | Direct temporal index access |
| Complex multi-step | Query Builder | Full optimization pipeline |
| One-off simple queries | Convenience methods | Balance of simplicity/performance |

Note that with multiple temporal vector indexes enabled, the property-less
`find_similar_as_of()` targets the alphabetically first indexed property —
use `find_similar_as_of_in("property", ...)` when you need a specific one.

### Optimize Your Queries

1. **Filter early**: Add `.filter()` and `.with_label()` as early as possible
2. **Limit results**: Always use `.limit()` when you don't need all results
3. **Use appropriate k**: Smaller k values are faster for ranking operations
4. **Avoid Variable depth**: `TraversalDepth::Variable` can be expensive

### Performance Targets

| Query Type | Target Latency | Notes |
|------------|----------------|-------|
| Single node lookup | <1µs | O(1) hash lookup |
| Single-hop traversal | <1µs/hop | CSR adjacency |
| 3-hop traversal | <100µs | Compound |
| k-NN search (k=10, 1M vectors) | <10ms | HNSW index |
| Graph + Vector hybrid | <20ms | traverse + rank |
| Full hybrid (temporal) | <30ms | as_of + traverse + rank |

## Error Handling

### Common Errors

```rust
use aletheiadb::core::error::{Error, StorageError, VectorError};

match results.next() {
    Some(Ok(row)) => { /* process row */ }
    Some(Err(Error::Storage(StorageError::NodeNotFound(id)))) => {
        eprintln!("Node {} not found", id);
    }
    Some(Err(Error::Vector(VectorError::DimensionMismatch { expected, got }))) => {
        eprintln!("Expected {} dims, got {}", expected, got);
    }
    Some(Err(e)) => {
        eprintln!("Query error: {}", e);
    }
    None => { /* no more results */ }
}
```

### Graceful Degradation

The query engine handles some issues gracefully:

- **Nodes without embeddings**: Skipped silently (warning logged)
- **Dimension mismatches**: Node skipped with warning
- **Deleted nodes in traversal**: Skipped

These behaviors ensure partial results rather than complete failure.

## Advanced Topics

### Custom Distance Metrics

```rust
// Cosine (default) - good for semantic similarity
db.query().find_similar_with_metric(&emb, 10, DistanceMetric::Cosine)

// Euclidean - good for spatial clustering
db.query().find_similar_with_metric(&emb, 10, DistanceMetric::Euclidean)

// Dot Product - good for MaxSim, ColBERT
db.query().find_similar_with_metric(&emb, 10, DistanceMetric::DotProduct)
```

### Optimizer Hints

```rust
use aletheiadb::query::plan::IndexHint;

db.query()
    .start(node_id)
    .with_hint(IndexHint::UseVectorIndex)  // Force vector index use
    .with_hint(IndexHint::UseBruteForce)   // Skip indexes
    // ...
```

### Parallel Execution

```rust
db.query()
    .start_from(many_nodes)
    .traverse("RELATED")
    .parallel()  // Enable parallel processing
    .execute(&db)?
```

## Troubleshooting

### Query Returns Empty Results

1. **Check node exists**: `db.get_node(start_id)?`
2. **Check edges exist**: `db.get_outgoing_edges(start_id)`
3. **Check embeddings**: Node must have "embedding" property
4. **Check temporal context**: Timestamp must have data

### Similarity Scores Are Wrong

1. **Normalize embeddings**: Cosine similarity works best with unit vectors
2. **Check dimensions**: Query and stored vectors must match
3. **Verify metric**: Ensure you're using the expected distance metric

### Performance Is Slow

1. **Add LIMIT**: Don't fetch more than needed
2. **Filter early**: Push filters before expensive operations
3. **Reduce k**: Smaller k = faster ranking
4. **Check index**: Ensure vector index is enabled

## Related Documentation

- [Vector Search Integration Guide](./vector-search-integration.md) - HNSW configuration
- [Vector Search Performance Guide](./vector-search-performance.md) - Tuning parameters
- [Design Document](../VECTOR_SEARCH_DESIGN.md) - Architecture overview
- [ADR-0019: Hybrid Query Planner](../adr/0019-hybrid-query-planner.md)
- [ADR-0021: Hybrid Query Execution](../adr/0021-hybrid-query-execution.md)
