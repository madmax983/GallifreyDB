//! Papyrus: Semantic Graph to Mermaid JS Exporter.
//!
//! "Visualize the invisible."
//!
//! Papyrus is an exporter that traverses the AletheiaDB knowledge graph
//! and generates a Mermaid JS flowchart. This is perfect for rendering
//! interactive graphs in Markdown, GitHub READMEs, or LLM chat interfaces.
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::papyrus::Papyrus;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! let papyrus = Papyrus::new(&db);
//!
//! # let start_node = aletheiadb::core::id::NodeId::new(1).unwrap();
//! // Export the ego-network around a specific node up to 2 hops, capped at 500 nodes
//! let mermaid_chart = papyrus.export_ego_graph(start_node, 2, Some(500))?;
//! println!("{}", mermaid_chart);
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::interning::{GLOBAL_INTERNER, InternedString};
use std::collections::{HashSet, VecDeque};
use std::fmt::Write;

/// The Papyrus Exporter Engine.
pub struct Papyrus<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Papyrus<'a> {
    /// Create a new Papyrus exporter.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Exports an ego-graph centered around `start_node` up to `max_depth` hops.
    ///
    /// `max_nodes` caps the total number of nodes included in the export. When
    /// `Some(n)` is provided the BFS stops once `n` nodes have been visited,
    /// preventing excessive memory use on dense or deeply-connected graphs.
    /// Pass `None` to visit all reachable nodes within `max_depth`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use aletheiadb::AletheiaDB;
    /// use aletheiadb::core::property::PropertyMapBuilder;
    /// use aletheiadb::experimental::characterization::papyrus::Papyrus;
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
    /// let papyrus = Papyrus::new(&db);
    /// let mermaid_chart = papyrus.export_ego_graph(node_a, 2, Some(500))?;
    /// println!("{}", mermaid_chart);
    /// # Ok(())
    /// # }
    /// ```
    pub fn export_ego_graph(
        &self,
        start_node: NodeId,
        max_depth: usize,
        max_nodes: Option<usize>,
    ) -> Result<String> {
        let mut output = String::new();
        writeln!(&mut output, "graph TD").unwrap();

        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((start_node, 0));
        visited_nodes.insert(start_node);

        while let Some((current_node, depth)) = queue.pop_front() {
            // Write node definition – propagates an error if the node is missing.
            self.write_node(&mut output, current_node)?;

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
                            self.write_edge(&mut output, current_node, target, edge.label)?;
                        }
                        // else: over the limit — omit this node and its edge.
                    } else {
                        // Target already in the subgraph; record cross/back-edges.
                        self.write_edge(&mut output, current_node, target, edge.label)?;
                    }
                }
            }
        }

        Ok(output)
    }

    fn write_node(&self, output: &mut String, node_id: NodeId) -> Result<()> {
        let node = self.db.get_node(node_id)?;
        let label = Self::resolve_str(node.label);

        // Try to find a human-readable name property
        let name = if let Some(val) = node.get_property("name") {
            val.to_string()
        } else if let Some(val) = node.get_property("title") {
            val.to_string()
        } else if let Some(val) = node.get_property("id") {
            val.to_string()
        } else {
            label.clone()
        };

        // Format: N1["Person: Alice"]
        writeln!(
            output,
            "    N{}[\"{}: {}\"]",
            node_id.as_u64(),
            label,
            Self::escape_mermaid(&name)
        )
        .unwrap();
        Ok(())
    }

    fn write_edge(
        &self,
        output: &mut String,
        source: NodeId,
        target: NodeId,
        label: InternedString,
    ) -> Result<()> {
        let label_str = Self::resolve_str(label);
        let escaped_label = Self::escape_mermaid(&label_str);
        // Format: N1 -->|"KNOWS"| N2
        writeln!(
            output,
            "    N{} -->|\"{}\"| N{}",
            source.as_u64(),
            escaped_label,
            target.as_u64()
        )
        .unwrap();
        Ok(())
    }

    fn resolve_str(s: InternedString) -> String {
        GLOBAL_INTERNER
            .resolve_with(s, |s| s.to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn escape_mermaid(s: &str) -> String {
        s.replace('"', "'").replace('\n', "<br/>")
    }
}

#[cfg(all(test, feature = "semantic-characterization"))]
mod tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_papyrus_mermaid_export() {
        let db = AletheiaDB::new().unwrap();

        // Node A: Alice
        let props_a = PropertyMapBuilder::new().insert("name", "Alice").build();
        let node_a = db.create_node("Person", props_a).unwrap();

        // Node B: Bob
        let props_b = PropertyMapBuilder::new().insert("name", "Bob").build();
        let node_b = db.create_node("Person", props_b).unwrap();

        // Edge A -> B
        db.create_edge(node_a, node_b, "KNOWS", Default::default())
            .unwrap();

        let papyrus = Papyrus::new(&db);
        let chart = papyrus.export_ego_graph(node_a, 1, None).unwrap();

        assert!(chart.contains("graph TD"));
        assert!(chart.contains(&format!("N{}[\"Person: 'Alice'\"]", node_a.as_u64())));
        assert!(chart.contains(&format!("N{}[\"Person: 'Bob'\"]", node_b.as_u64())));
        assert!(chart.contains(&format!(
            "N{} -->|\"KNOWS\"| N{}",
            node_a.as_u64(),
            node_b.as_u64()
        )));
    }

    #[test]
    fn test_papyrus_max_depth() {
        let db = AletheiaDB::new().unwrap();

        let a = db
            .create_node(
                "Node",
                PropertyMapBuilder::new().insert("name", "A").build(),
            )
            .unwrap();
        let b = db
            .create_node(
                "Node",
                PropertyMapBuilder::new().insert("name", "B").build(),
            )
            .unwrap();
        let c = db
            .create_node(
                "Node",
                PropertyMapBuilder::new().insert("name", "C").build(),
            )
            .unwrap();

        db.create_edge(a, b, "L1", Default::default()).unwrap();
        db.create_edge(b, c, "L2", Default::default()).unwrap();

        let papyrus = Papyrus::new(&db);

        // Depth 1: should only see A and B
        let chart1 = papyrus.export_ego_graph(a, 1, None).unwrap();
        assert!(chart1.contains("A"));
        assert!(chart1.contains("B"));
        assert!(!chart1.contains("C"), "Depth 1 should not include node C");

        // Depth 2: should see all
        let chart2 = papyrus.export_ego_graph(a, 2, None).unwrap();
        assert!(chart2.contains("C"));
    }

    #[test]
    fn test_papyrus_max_nodes() {
        let db = AletheiaDB::new().unwrap();

        let a = db
            .create_node(
                "Node",
                PropertyMapBuilder::new().insert("name", "A").build(),
            )
            .unwrap();
        let b = db
            .create_node(
                "Node",
                PropertyMapBuilder::new().insert("name", "B").build(),
            )
            .unwrap();
        let c = db
            .create_node(
                "Node",
                PropertyMapBuilder::new().insert("name", "C").build(),
            )
            .unwrap();

        db.create_edge(a, b, "L1", Default::default()).unwrap();
        db.create_edge(b, c, "L2", Default::default()).unwrap();

        let papyrus = Papyrus::new(&db);

        // max_nodes = 1: only the start node should appear
        let chart = papyrus.export_ego_graph(a, 2, Some(1)).unwrap();
        assert!(chart.contains("A"));
        assert!(!chart.contains("B"), "max_nodes=1 should stop before B");
        assert!(!chart.contains("C"), "max_nodes=1 should stop before C");
    }

    #[test]
    fn test_papyrus_escape_mermaid_newline() {
        let db = AletheiaDB::new().unwrap();
        let props = PropertyMapBuilder::new()
            .insert("name", "Line1\nLine2")
            .build();
        let node = db.create_node("Item", props).unwrap();
        let papyrus = Papyrus::new(&db);
        let chart = papyrus.export_ego_graph(node, 0, None).unwrap();
        assert!(
            chart.contains("<br/>"),
            "Newlines in labels must be escaped as <br/>"
        );
    }
}
