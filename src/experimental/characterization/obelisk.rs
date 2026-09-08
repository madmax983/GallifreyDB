//! Obelisk: Semantic Graph to Graphviz DOT Exporter.
//!
//! "Cast your graph in stone."
//!
//! Obelisk exports the AletheiaDB knowledge graph into Graphviz DOT format.
//! This allows for rich, static visualizations of complex topological structures
//! directly from the temporal/semantic database.
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::characterization::obelisk::Obelisk;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! let obelisk = Obelisk::new(&db);
//!
//! # let start_node = aletheiadb::core::id::NodeId::new(1).unwrap();
//! // Export the ego-network around a specific node up to 2 hops, capped at 500 nodes
//! let dot_graph = obelisk.export_ego_graph(start_node, 2, Some(500))?;
//! println!("{}", dot_graph);
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::interning::{GLOBAL_INTERNER, InternedString};
use std::collections::{HashSet, VecDeque};
use std::fmt::Write;

/// The Obelisk Exporter Engine.
pub struct Obelisk<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Obelisk<'a> {
    /// Create a new Obelisk exporter.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Exports an ego-graph centered around `start_node` up to `max_depth` hops.
    ///
    /// `max_nodes` caps the total number of nodes included in the export. When
    /// `Some(n)` is provided the BFS stops once `n` nodes have been visited,
    /// preventing excessive memory use on dense or deeply-connected graphs.
    /// Pass `None` to visit all reachable nodes within `max_depth`.
    pub fn export_ego_graph(
        &self,
        start_node: NodeId,
        max_depth: usize,
        max_nodes: Option<usize>,
    ) -> Result<String> {
        let mut output = String::new();
        writeln!(&mut output, "digraph G {{").unwrap();
        writeln!(
            &mut output,
            "    node [shape=box, style=filled, color=lightblue];"
        )
        .unwrap();

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
            Self::strip_quotes(&val.to_string())
        } else if let Some(val) = node.get_property("title") {
            Self::strip_quotes(&val.to_string())
        } else if let Some(val) = node.get_property("id") {
            Self::strip_quotes(&val.to_string())
        } else {
            label.clone()
        };

        writeln!(
            output,
            "    N{} [label=\"{}: {}\"];",
            node_id.as_u64(),
            label,
            Self::escape_dot(&name)
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

    fn strip_quotes(s: &str) -> String {
        if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
            s[1..s.len() - 1].to_string()
        } else {
            s.to_string()
        }
    }
}

#[cfg(all(test, feature = "semantic-characterization"))]
mod tests {
    use super::*;
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_obelisk_dot_export() {
        let db = AletheiaDB::new().unwrap();

        let props_a = PropertyMapBuilder::new().insert("name", "Alice").build();
        let node_a = db.create_node("Person", props_a).unwrap();

        let props_b = PropertyMapBuilder::new().insert("name", "Bob").build();
        let node_b = db.create_node("Person", props_b).unwrap();

        db.create_edge(node_a, node_b, "KNOWS", Default::default())
            .unwrap();

        let obelisk = Obelisk::new(&db);
        let chart = obelisk.export_ego_graph(node_a, 1, None).unwrap();

        assert!(chart.contains("digraph G {"));
        assert!(chart.contains(&format!("N{} [label=\"Person: Alice\"];", node_a.as_u64())));
        assert!(chart.contains(&format!("N{} [label=\"Person: Bob\"];", node_b.as_u64())));
        assert!(chart.contains(&format!(
            "N{} -> N{} [label=\"KNOWS\"];",
            node_a.as_u64(),
            node_b.as_u64()
        )));
    }

    #[test]
    fn test_obelisk_strip_quotes() {
        let db = AletheiaDB::new().unwrap();
        let props = PropertyMapBuilder::new()
            .insert("name", "\"Quoted\"")
            .build();
        let node = db.create_node("Item", props).unwrap();
        let obelisk = Obelisk::new(&db);
        let chart = obelisk.export_ego_graph(node, 0, None).unwrap();
        // Since it's stored as `"Quoted"` (literal string containing quotes)
        // and string properties serialize to `""Quoted""`
        // the strip_quotes removes outer quotes but leaves inner quotes
        // which get escaped.
        assert!(chart.contains("\\\"Quoted\\\""));
    }
}
