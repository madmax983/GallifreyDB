//! Starlight: Semantic Graph to JSON Exporter.
//!
//! "Light up the graph."
//!
//! Starlight exports the AletheiaDB knowledge graph into a JSON format
//! compatible with popular graph visualization libraries like 3d-force-graph.
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::characterization::starlight::Starlight;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! let starlight = Starlight::new(&db);
//!
//! # let start_node = aletheiadb::core::id::NodeId::new(1).unwrap();
//! let json = starlight.export_ego_graph(start_node, 2, Some(500))?;
//! println!("{}", json);
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::interning::{GLOBAL_INTERNER, InternedString};
use serde_json::json;
use std::collections::{HashSet, VecDeque};

/// The Starlight Exporter Engine.
pub struct Starlight<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Starlight<'a> {
    /// Create a new Starlight exporter.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Exports an ego-graph centered around `start_node` up to `max_depth` hops in JSON.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use aletheiadb::AletheiaDB;
    /// use aletheiadb::core::property::PropertyMapBuilder;
    /// use aletheiadb::experimental::characterization::starlight::Starlight;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = AletheiaDB::new()?;
    ///
    /// let props_a = PropertyMapBuilder::new().insert("name", "Alice").build();
    /// let node_a = db.create_node("Person", props_a)?;
    ///
    /// let props_b = PropertyMapBuilder::new().insert("name", "Bob").build();
    /// let node_b = db.create_node("Person", props_b)?;
    ///
    /// db.create_edge(node_a, node_b, "KNOWS", Default::default())?;
    ///
    /// let starlight = Starlight::new(&db);
    /// let json = starlight.export_ego_graph(node_a, 2, Some(500))?;
    /// println!("{}", json);
    /// # Ok(())
    /// # }
    /// ```
    pub fn export_ego_graph(
        &self,
        start_node: NodeId,
        max_depth: usize,
        max_nodes: Option<usize>,
    ) -> Result<String> {
        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut queue = VecDeque::new();

        let mut nodes_json = Vec::new();
        let mut links_json = Vec::new();

        queue.push_back((start_node, 0));
        visited_nodes.insert(start_node);

        while let Some((current_node, depth)) = queue.pop_front() {
            // Write node definition
            let node = self.db.get_node(current_node)?;
            let label = Self::resolve_str(node.label);

            let name = if let Some(val) = node.get_property("name") {
                val.to_string()
            } else if let Some(val) = node.get_property("title") {
                val.to_string()
            } else if let Some(val) = node.get_property("id") {
                val.to_string()
            } else {
                label.clone()
            };

            nodes_json.push(json!({
                "id": current_node.as_u64(),
                "label": label,
                "name": name,
            }));

            if depth >= max_depth {
                continue;
            }

            // Get outgoing edges
            let edges = self.db.get_outgoing_edges(current_node);
            for edge_id in edges {
                if !visited_edges.insert(edge_id) {
                    continue;
                }

                if let Ok(edge) = self.db.get_edge(edge_id) {
                    let target = edge.target;
                    let is_new_target = !visited_nodes.contains(&target);

                    if is_new_target {
                        // Only include new nodes that are within the cap.
                        if max_nodes.is_none_or(|limit| visited_nodes.len() < limit) {
                            visited_nodes.insert(target);
                            queue.push_back((target, depth + 1));

                            links_json.push(json!({
                                "source": current_node.as_u64(),
                                "target": target.as_u64(),
                                "label": Self::resolve_str(edge.label),
                            }));
                        }
                    } else {
                        // Target already in the subgraph
                        links_json.push(json!({
                            "source": current_node.as_u64(),
                            "target": target.as_u64(),
                            "label": Self::resolve_str(edge.label),
                        }));
                    }
                }
            }
        }

        let graph_json = json!({
            "nodes": nodes_json,
            "links": links_json,
        });

        serde_json::to_string(&graph_json)
            .map_err(|e| crate::core::error::Error::other(e.to_string()))
    }

    fn resolve_str(s: InternedString) -> String {
        GLOBAL_INTERNER
            .resolve_with(s, |s| s.to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    }
}

#[cfg(all(test, feature = "semantic-characterization"))]
mod tests {
    use super::*;
    use crate::PropertyMapBuilder;
    use crate::WriteOps;

    #[test]
    fn test_starlight_json_export() {
        let db = AletheiaDB::new().unwrap();

        let mut node_a = NodeId::new(1).unwrap();
        let mut node_b = NodeId::new(2).unwrap();

        db.write(|tx| {
            // Node A: Alice
            let props_a = PropertyMapBuilder::new().insert("name", "Alice").build();
            node_a = tx.create_node("Person", props_a).unwrap();

            // Node B: Bob
            let props_b = PropertyMapBuilder::new().insert("name", "Bob").build();
            node_b = tx.create_node("Person", props_b).unwrap();

            // Edge A -> B
            tx.create_edge(node_a, node_b, "KNOWS", Default::default())
                .unwrap();
            Ok::<(), crate::core::error::Error>(())
        })
        .unwrap();

        let starlight = Starlight::new(&db);
        let json_str = starlight.export_ego_graph(node_a, 1, None).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        assert!(parsed.get("nodes").is_some());
        assert!(parsed.get("links").is_some());

        let nodes = parsed["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 2);

        let links = parsed["links"].as_array().unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0]["source"], node_a.as_u64());
        assert_eq!(links[0]["target"], node_b.as_u64());
    }
}
