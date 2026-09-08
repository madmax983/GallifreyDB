//! Petroglyph: Semantic Graph to Graphviz DOT Exporter.
//!
//! "Carved in stone."
//!
//! Petroglyph exports the AletheiaDB knowledge graph into the Graphviz DOT format.
//! This is the undisputed standard for programmatic graph visualization, supporting
//! complex rendering tools like `dot`, `neato`, and `sfdp` for analyzing graph topologies.
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::characterization::petroglyph::Petroglyph;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! let petroglyph = Petroglyph::new(&db);
//!
//! # let start_node = aletheiadb::core::id::NodeId::new(1).unwrap();
//! // Export the ego-network around a specific node up to 2 hops, capped at 500 nodes
//! let dot_script = petroglyph.export_ego_graph(start_node, 2, Some(500))?;
//! println!("{}", dot_script);
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::interning::{GLOBAL_INTERNER, InternedString};
use std::collections::{HashSet, VecDeque};
use std::fmt::Write;

/// The Petroglyph Exporter Engine.
pub struct Petroglyph<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Petroglyph<'a> {
    /// Create a new Petroglyph exporter.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Exports an ego-graph centered around `start_node` up to `max_depth` hops.
    ///
    /// `max_nodes` caps the total number of nodes included in the export. When
    /// `Some(n)` is provided the BFS stops once `n` nodes have been visited.
    /// Pass `None` to visit all reachable nodes within `max_depth`.
    pub fn export_ego_graph(
        &self,
        start_node: NodeId,
        max_depth: usize,
        max_nodes: Option<usize>,
    ) -> Result<String> {
        let mut output = String::new();
        writeln!(&mut output, "digraph AletheiaGraph {{").unwrap();
        // Base styling for Graphviz
        writeln!(&mut output, "    node [shape=box, style=rounded];").unwrap();
        writeln!(&mut output, "    edge [fontsize=10];").unwrap();

        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((start_node, 0));
        visited_nodes.insert(start_node);

        while let Some((current_node, depth)) = queue.pop_front() {
            self.write_node(&mut output, current_node)?;

            if depth >= max_depth {
                continue;
            }

            let edges = self.db.get_outgoing_edges(current_node);
            for edge_id in edges {
                if !visited_edges.insert(edge_id) {
                    continue;
                }

                if let Ok(edge) = self.db.get_edge(edge_id) {
                    let target = edge.target;
                    let is_new_target = !visited_nodes.contains(&target);

                    if is_new_target {
                        if max_nodes.is_none_or(|limit| visited_nodes.len() < limit) {
                            visited_nodes.insert(target);
                            queue.push_back((target, depth + 1));
                            self.write_edge(&mut output, current_node, target, edge.label)?;
                        }
                    } else {
                        self.write_edge(&mut output, current_node, target, edge.label)?;
                    }
                }
            }
        }

        writeln!(&mut output, "}}").unwrap();
        Ok(output)
    }

    fn write_node(&self, output: &mut String, node_id: NodeId) -> Result<()> {
        let node = self.db.get_node(node_id)?;
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

        let escaped_name = Self::escape_dot(&name);

        // DOT format: N1 [label="Person\n\"Alice\""];
        writeln!(
            output,
            "    N{} [label=\"{}\\n{}\"];",
            node_id.as_u64(),
            label,
            escaped_name
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
        let escaped_label = Self::escape_dot(&label_str);

        // DOT format: N1 -> N2 [label="KNOWS"];
        writeln!(
            output,
            "    N{} -> N{} [label=\"{}\"];",
            source.as_u64(),
            target.as_u64(),
            escaped_label
        )
        .unwrap();
        Ok(())
    }

    fn resolve_str(s: InternedString) -> String {
        GLOBAL_INTERNER
            .resolve_with(s, |s| s.to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn escape_dot(s: &str) -> String {
        s.replace('"', "\\\"").replace('\n', "\\n")
    }
}

#[cfg(all(test, feature = "semantic-characterization"))]
mod tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_petroglyph_dot_export() {
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

        let petroglyph = Petroglyph::new(&db);
        let dot = petroglyph.export_ego_graph(node_a, 1, None).unwrap();

        assert!(dot.contains("digraph AletheiaGraph {"));
        assert!(dot.contains("node [shape=box, style=rounded];"));
        assert!(dot.contains(&format!("N{} [label=\"Person\\n\\\"Alice\\\"\"];", node_a.as_u64())));
        assert!(dot.contains(&format!("N{} [label=\"Person\\n\\\"Bob\\\"\"];", node_b.as_u64())));
        assert!(dot.contains(&format!(
            "N{} -> N{} [label=\"KNOWS\"];",
            node_a.as_u64(),
            node_b.as_u64()
        )));
    }

    #[test]
    fn test_petroglyph_max_depth() {
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

        let petroglyph = Petroglyph::new(&db);

        // Depth 1: should only see A and B
        let dot1 = petroglyph.export_ego_graph(a, 1, None).unwrap();
        assert!(dot1.contains(&format!("N{} [label=\"Node\\n\\\"A\\\"\"];", a.as_u64())));
        assert!(dot1.contains(&format!("N{} [label=\"Node\\n\\\"B\\\"\"];", b.as_u64())));
        assert!(
            !dot1.contains(&format!("N{} [label=\"Node\\n\\\"C\\\"\"];", c.as_u64())),
            "Depth 1 should not include node C"
        );

        // Depth 2: should see all
        let dot2 = petroglyph.export_ego_graph(a, 2, None).unwrap();
        assert!(dot2.contains(&format!("N{} [label=\"Node\\n\\\"C\\\"\"];", c.as_u64())));
    }

    #[test]
    fn test_petroglyph_max_nodes() {
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

        let petroglyph = Petroglyph::new(&db);

        // max_nodes = 1: only the start node should appear
        let dot = petroglyph.export_ego_graph(a, 2, Some(1)).unwrap();
        assert!(dot.contains(&format!("N{} [label=\"Node\\n\\\"A\\\"\"];", a.as_u64())));
        assert!(
            !dot.contains(&format!("N{} [label=\"Node\\n\\\"B\\\"\"];", b.as_u64())),
            "max_nodes=1 should stop before B"
        );
        assert!(
            !dot.contains(&format!("N{} [label=\"Node\\n\\\"C\\\"\"];", c.as_u64())),
            "max_nodes=1 should stop before C"
        );
    }

    #[test]
    fn test_petroglyph_escape_dot() {
        let db = AletheiaDB::new().unwrap();
        let props = PropertyMapBuilder::new()
            .insert("name", "Line1\nLine2\"Quotes\"")
            .build();
        let node = db.create_node("Item", props).unwrap();
        let petroglyph = Petroglyph::new(&db);
        let dot = petroglyph.export_ego_graph(node, 0, None).unwrap();

        assert!(
            dot.contains("Line1\\nLine2\\\"Quotes\\\""),
            "Newlines and quotes must be correctly escaped for DOT format."
        );
    }
}
