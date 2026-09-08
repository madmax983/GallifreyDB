//! Alexandria: Obsidian Vault Exporter.
//!
//! "Build your personal library of Alexandria."
//!
//! Alexandria exports the AletheiaDB knowledge graph into a directory of Markdown
//! files, formatted as an Obsidian-compatible vault. This bridges the gap between
//! a programmatic graph database and a human-readable personal knowledge management (PKM) tool.
//!
//! # The Hook
//! Why keep your graph trapped in a database? Alexandria lets you export your
//! complex semantic relationships into an Obsidian vault, where you can visually
//! explore them, click through bidirectional links, and read the properties as frontmatter.
//!
//! # Example
//! ```rust,no_run
//! use aletheiadb::AletheiaDB;
//! use aletheiadb::experimental::characterization::alexandria::Alexandria;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = AletheiaDB::new()?;
//! let alexandria = Alexandria::new(&db);
//!
//! # let start_node = aletheiadb::core::id::NodeId::new(1).unwrap();
//! // Export an ego-graph to a Map of filename -> content
//! let vault_files = alexandria.export_vault(start_node, 2, None)?;
//!
//! // In a real app, you would write these to disk:
//! // for (filename, content) in vault_files {
//! //     std::fs::write(format!("./my_vault/{}", filename), content)?;
//! // }
//! # Ok(())
//! # }
//! ```

use crate::AletheiaDB;
use crate::core::error::Result;
use crate::core::id::NodeId;
use crate::core::interning::{GLOBAL_INTERNER, InternedString};
use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(feature = "semantic-characterization")]
use std::fmt::Write;

/// The Alexandria Exporter Engine.
pub struct Alexandria<'a> {
    db: &'a AletheiaDB,
}

#[cfg(feature = "semantic-characterization")]
impl<'a> Alexandria<'a> {
    /// Create a new Alexandria exporter.
    pub fn new(db: &'a AletheiaDB) -> Self {
        Self { db }
    }

    /// Exports an ego-graph centered around `start_node` up to `max_depth` hops.
    ///
    /// Returns a map where the key is the suggested filename (e.g., "Alice.md")
    /// and the value is the Markdown content of that file.
    pub fn export_vault(
        &self,
        start_node: NodeId,
        max_depth: usize,
        max_nodes: Option<usize>,
    ) -> Result<HashMap<String, String>> {
        let mut visited_nodes = HashSet::new();
        let mut queue = VecDeque::new();
        let mut files = HashMap::new();

        queue.push_back((start_node, 0));
        visited_nodes.insert(start_node);

        while let Some((current_node, depth)) = queue.pop_front() {
            let node = self.db.get_node(current_node)?;
            let label = Self::resolve_str(node.label);

            // Important: we remove surrounding quotes from string values if any, to avoid `"Alice"`
            let name = if let Some(val) = node.get_property("name") {
                val.to_string().replace('"', "")
            } else if let Some(val) = node.get_property("title") {
                val.to_string().replace('"', "")
            } else if let Some(val) = node.get_property("id") {
                val.to_string().replace('"', "")
            } else {
                format!("{}_{}", label, current_node.as_u64())
            };

            // Clean up name for filename (Obsidian doesn't like certain chars)
            let safe_name =
                name.replace(|c: char| !c.is_alphanumeric() && c != ' ' && c != '-', "_");
            let filename = format!("{}.md", safe_name);

            let mut content = String::new();

            // Write YAML Frontmatter
            writeln!(&mut content, "---").unwrap();
            writeln!(&mut content, "id: {}", current_node.as_u64()).unwrap();
            writeln!(&mut content, "label: {}", label).unwrap();
            for (k, v) in node.properties.iter() {
                // Skip vector properties as they are too large for frontmatter
                if v.as_vector().is_none() {
                    let val_str = v.to_string().replace('\n', " ").replace('"', "");
                    writeln!(&mut content, "{}: {}", k, val_str).unwrap();
                }
            }
            writeln!(&mut content, "---\n").unwrap();

            // Write Title
            writeln!(&mut content, "# {}\n", name).unwrap();

            // Write Outgoing Edges
            let outgoing = self.db.get_outgoing_edges(current_node);
            if !outgoing.is_empty() {
                writeln!(&mut content, "## Outgoing Connections").unwrap();
                for edge_id in outgoing {
                    if let Ok(edge) = self.db.get_edge(edge_id) {
                        let target_node = self.db.get_node(edge.target)?;
                        let target_name = if let Some(val) = target_node.get_property("name") {
                            val.to_string().replace('"', "")
                        } else {
                            format!(
                                "{}_{}",
                                Self::resolve_str(target_node.label),
                                edge.target.as_u64()
                            )
                        };
                        let target_safe_name = target_name
                            .replace(|c: char| !c.is_alphanumeric() && c != ' ' && c != '-', "_");
                        let edge_label = Self::resolve_str(edge.label);

                        writeln!(&mut content, "- [[{}]] ({})", target_safe_name, edge_label)
                            .unwrap();

                        if depth < max_depth {
                            let is_new = !visited_nodes.contains(&edge.target);
                            if is_new && max_nodes.is_none_or(|limit| visited_nodes.len() < limit) {
                                visited_nodes.insert(edge.target);
                                queue.push_back((edge.target, depth + 1));
                            }
                        }
                    }
                }
                writeln!(&mut content).unwrap();
            }

            // Write Incoming Edges
            let incoming = self.db.get_incoming_edges(current_node);
            if !incoming.is_empty() {
                writeln!(&mut content, "## Incoming Connections").unwrap();
                for edge_id in incoming {
                    if let Ok(edge) = self.db.get_edge(edge_id) {
                        let source_node = self.db.get_node(edge.source)?;
                        let source_name = if let Some(val) = source_node.get_property("name") {
                            val.to_string().replace('"', "")
                        } else {
                            format!(
                                "{}_{}",
                                Self::resolve_str(source_node.label),
                                edge.source.as_u64()
                            )
                        };
                        let source_safe_name = source_name
                            .replace(|c: char| !c.is_alphanumeric() && c != ' ' && c != '-', "_");
                        let edge_label = Self::resolve_str(edge.label);

                        writeln!(&mut content, "- [[{}]] ({})", source_safe_name, edge_label)
                            .unwrap();
                    }
                }
            }

            files.insert(filename, content);
        }

        Ok(files)
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
    use crate::core::property::PropertyMapBuilder;

    #[test]
    fn test_alexandria_export() {
        let db = AletheiaDB::new().unwrap();

        let props_a = PropertyMapBuilder::new().insert("name", "Alice").build();
        let node_a = db.create_node("Person", props_a).unwrap();

        let props_b = PropertyMapBuilder::new().insert("name", "Bob").build();
        let node_b = db.create_node("Person", props_b).unwrap();

        db.create_edge(node_a, node_b, "KNOWS", Default::default())
            .unwrap();

        let alexandria = Alexandria::new(&db);
        let vault = alexandria.export_vault(node_a, 2, None).unwrap();

        assert_eq!(vault.len(), 2);

        let alice_doc = vault.get("Alice.md").unwrap();
        assert!(alice_doc.contains("---"));
        assert!(alice_doc.contains("label: Person"));
        assert!(alice_doc.contains("name: Alice"));
        assert!(alice_doc.contains("# Alice"));
        assert!(alice_doc.contains("- [[Bob]] (KNOWS)"));

        let bob_doc = vault.get("Bob.md").unwrap();
        assert!(bob_doc.contains("## Incoming Connections"));
        assert!(bob_doc.contains("- [[Alice]] (KNOWS)"));
    }
}
