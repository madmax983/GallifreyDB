//! Ledger: Semantic Graph to CSV Exporter.
//!
//! "Follow the money."
//!
//! Ledger exports the AletheiaDB knowledge graph into CSV format
//! for easy import into spreadsheets or other data analysis tools.
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::characterization::ledger::Ledger;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! let ledger = Ledger::new(&db);
//!
//! # let start_node = aletheiadb::core::id::NodeId::new(1).unwrap();
//! let (nodes_csv, edges_csv) = ledger.export_ego_graph(start_node, 2, Some(500))?;
//! println!("Nodes:\n{}", nodes_csv);
//! println!("Edges:\n{}", edges_csv);
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::interning::{GLOBAL_INTERNER, InternedString};
use std::collections::{HashSet, VecDeque};
use std::fmt::Write;

/// The Ledger Exporter Engine.
pub struct Ledger<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Ledger<'a> {
    /// Create a new Ledger exporter.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Exports an ego-graph centered around `start_node` up to `max_depth` hops.
    ///
    /// Returns a tuple of two Strings: `(nodes_csv, edges_csv)`.
    pub fn export_ego_graph(
        &self,
        start_node: NodeId,
        max_depth: usize,
        max_nodes: Option<usize>,
    ) -> Result<(String, String)> {
        let mut nodes_csv = String::new();
        let mut edges_csv = String::new();

        writeln!(&mut nodes_csv, "id,label,name").unwrap();
        writeln!(&mut edges_csv, "source,target,label").unwrap();

        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((start_node, 0));
        visited_nodes.insert(start_node);

        while let Some((current_node, depth)) = queue.pop_front() {
            // Write node definition
            self.write_node(&mut nodes_csv, current_node)?;

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
                        // Only add new nodes if we haven't reached max depth
                        if depth < max_depth {
                            // Only include new nodes that are within the cap.
                            if max_nodes.is_none_or(|limit| visited_nodes.len() < limit) {
                                visited_nodes.insert(target);
                                queue.push_back((target, depth + 1));
                                self.write_edge(&mut edges_csv, current_node, target, edge.label)?;
                            }
                        }
                    } else {
                        // Target already in the subgraph (cross-edge or back-edge)
                        self.write_edge(&mut edges_csv, current_node, target, edge.label)?;
                    }
                }
            }
        }

        Ok((nodes_csv, edges_csv))
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

        writeln!(
            output,
            "{},\"{}\",\"{}\"",
            node_id.as_u64(),
            Self::escape_csv(&label),
            Self::escape_csv(&name)
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
        writeln!(
            output,
            "{},{},\"{}\"",
            source.as_u64(),
            target.as_u64(),
            Self::escape_csv(&label_str)
        )
        .unwrap();
        Ok(())
    }

    fn resolve_str(s: InternedString) -> String {
        GLOBAL_INTERNER
            .resolve_with(s, |s| s.to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn escape_csv(s: &str) -> String {
        s.replace('"', "\"\"")
    }
}

#[cfg(all(test, feature = "semantic-characterization"))]
mod tests {
    use super::*;
    use crate::PropertyMapBuilder;
    use crate::WriteOps;

    #[test]
    fn test_ledger_csv_export() {
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

        let ledger = Ledger::new(&db);
        let (nodes_csv, edges_csv) = ledger.export_ego_graph(node_a, 1, None).unwrap();

        assert!(nodes_csv.contains("id,label,name"));
        assert!(nodes_csv.contains(&format!("{},\"Person\",\"\"\"Alice\"\"\"", node_a.as_u64())));
        assert!(nodes_csv.contains(&format!("{},\"Person\",\"\"\"Bob\"\"\"", node_b.as_u64())));

        assert!(edges_csv.contains("source,target,label"));
        assert!(edges_csv.contains(&format!(
            "{},{},\"KNOWS\"",
            node_a.as_u64(),
            node_b.as_u64()
        )));
    }

    #[test]
    fn test_ledger_escape_csv() {
        let db = AletheiaDB::new().unwrap();
        let props = PropertyMapBuilder::new()
            .insert("name", "Alice \"The Boss\" Smith")
            .build();

        let mut node_a = NodeId::new(1).unwrap();
        db.write(|tx| {
            node_a = tx.create_node("Person", props).unwrap();
            Ok::<(), crate::core::error::Error>(())
        })
        .unwrap();

        let ledger = Ledger::new(&db);
        let (nodes_csv, _) = ledger.export_ego_graph(node_a, 0, None).unwrap();
        println!("nodes_csv: {}", nodes_csv);

        assert!(nodes_csv.contains(&format!(
            "{},\"Person\",\"\"\"Alice \"\"The Boss\"\" Smith\"\"\"",
            node_a.as_u64()
        )));
    }
}
