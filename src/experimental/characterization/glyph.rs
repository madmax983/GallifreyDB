//! Glyph: Semantic Graph to Graphviz DOT Exporter.
//!
//! "Trace the connections."
//!
//! Glyph exports the AletheiaDB knowledge graph into Graphviz DOT format,
//! making it easy to generate diagrams using standard tools.
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::characterization::glyph::Glyph;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! let glyph = Glyph::new(&db);
//!
//! # let start_node = aletheiadb::core::id::NodeId::new(1).unwrap();
//! let dot_str = glyph.export_ego_graph(start_node, 2, Some(500))?;
//! println!("{}", dot_str);
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::interning::{GLOBAL_INTERNER, InternedString};
use std::collections::{HashSet, VecDeque};
use std::fmt::Write;

/// The Glyph Exporter Engine.
pub struct Glyph<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Glyph<'a> {
    /// Create a new Glyph exporter.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Exports an ego-graph centered around `start_node` up to `max_depth` hops in Graphviz DOT format.
    pub fn export_ego_graph(
        &self,
        start_node: NodeId,
        max_depth: usize,
        max_nodes: Option<usize>,
    ) -> Result<String> {
        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut queue = VecDeque::new();

        let mut dot_out = String::new();
        writeln!(&mut dot_out, "digraph EgoGraph {{")
            .map_err(|e| crate::core::error::Error::other(e.to_string()))?;

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
                // If it's a fallback label, we want quotes around it to match the standard format.
                // PropertyValue::to_string() outputs strings with double quotes (e.g., "\"Alice\"").
                format!("\"{}\"", label)
            };

            let escaped_name = name.replace("\"", "\\\"");

            writeln!(
                &mut dot_out,
                "    n{} [label=\"{}: {}\"];",
                current_node.as_u64(),
                label,
                escaped_name
            )
            .map_err(|e| crate::core::error::Error::other(e.to_string()))?;

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

                    let edge_label = Self::resolve_str(edge.label);
                    let escaped_edge_label = edge_label.replace("\"", "\\\"");

                    if is_new_target {
                        // Only include new nodes that are within the cap.
                        if max_nodes.is_none_or(|limit| visited_nodes.len() < limit) {
                            visited_nodes.insert(target);
                            queue.push_back((target, depth + 1));

                            writeln!(
                                &mut dot_out,
                                "    n{} -> n{} [label=\"{}\"];",
                                current_node.as_u64(),
                                target.as_u64(),
                                escaped_edge_label
                            )
                            .map_err(|e| crate::core::error::Error::other(e.to_string()))?;
                        }
                    } else {
                        // Target already in the subgraph
                        writeln!(
                            &mut dot_out,
                            "    n{} -> n{} [label=\"{}\"];",
                            current_node.as_u64(),
                            target.as_u64(),
                            escaped_edge_label
                        )
                        .map_err(|e| crate::core::error::Error::other(e.to_string()))?;
                    }
                }
            }
        }

        writeln!(&mut dot_out, "}}")
            .map_err(|e| crate::core::error::Error::other(e.to_string()))?;
        Ok(dot_out)
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
    fn test_glyph_dot_export() {
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

        let glyph = Glyph::new(&db);
        let dot_str = glyph.export_ego_graph(node_a, 1, None).unwrap();

        assert!(dot_str.contains("digraph EgoGraph {"));
        // PropertyValue to string has quotes, e.g., "\"Alice\""
        assert!(dot_str.contains(&format!(
            "n{} [label=\"Person: \\\"Alice\\\"\"]",
            node_a.as_u64()
        )));
        assert!(dot_str.contains(&format!(
            "n{} [label=\"Person: \\\"Bob\\\"\"]",
            node_b.as_u64()
        )));
        assert!(dot_str.contains(&format!(
            "n{} -> n{} [label=\"KNOWS\"]",
            node_a.as_u64(),
            node_b.as_u64()
        )));
        assert!(dot_str.contains("}"));
    }
}
