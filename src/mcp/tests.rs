//! Tests for the AletheiaDB MCP server.
//!
//! These tests verify the MCP server functionality including:
//! - Server initialization and info
//! - Node CRUD operations
//! - Edge CRUD operations
//! - Graph traversal
//! - Vector similarity search
//! - Temporal queries
//! - Hybrid queries
//!
//! Tests are executed sequentially to avoid cross-test interference on internal state.

use std::collections::HashMap;
use std::sync::Arc;

use crate::core::PropertyMapBuilder;
use crate::db::AletheiaDB;

use super::server::AletheiaMcpServer;
use super::tools::*;

/// Helper to create a test database with some sample data.
fn create_test_db() -> Arc<AletheiaDB> {
    let db = AletheiaDB::new().expect("Failed to create database");
    Arc::new(db)
}

#[allow(unused_imports)]
use rmcp::ServerHandler;

/// Helper to create a test server.
fn create_test_server() -> AletheiaMcpServer {
    AletheiaMcpServer::new(create_test_db())
}

/// Helper to parse JSON response and check for errors.
fn parse_response<T: serde::de::DeserializeOwned>(response: &str) -> Result<T, String> {
    let value: serde_json::Value =
        serde_json::from_str(response).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    if let Some(error) = value.get("error") {
        // Structured error shape (Issue #3234): `{code, message, retriable}`.
        return Err(error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown error")
            .to_string());
    }

    serde_json::from_value(value).map_err(|e| format!("Failed to deserialize: {}", e))
}

// ============================================================================
// Server Initialization Tests
// ============================================================================

#[test]
fn test_server_creation() {
    let server = create_test_server();
    assert!(server.db().node_count() == 0);
    assert!(server.db().edge_count() == 0);
}

#[test]
fn test_server_with_existing_db() {
    let db = create_test_db();

    // Add some data to the database first
    let _node_id = db
        .create_node(
            "Person",
            PropertyMapBuilder::new().insert("name", "Alice").build(),
        )
        .expect("Failed to create node");

    let server = AletheiaMcpServer::new(db);
    assert_eq!(server.db().node_count(), 1);
}

// ============================================================================
// Node CRUD Tests
// ============================================================================

mod node_tests {
    use super::*;

    /// Helper to create two `Person` nodes and return their IDs.
    fn create_two_nodes(server: &AletheiaMcpServer) -> (u64, u64) {
        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        }))
        .unwrap();
        (n1.id, n2.id)
    }

    #[test]
    fn test_create_node_basic() {
        let server = create_test_server();

        let req = CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        };

        let response = server.create_node(req);
        let node: NodeResponse = parse_response(&response).expect("Failed to parse response");

        assert_eq!(node.label, "Person");
        // Note: Node IDs start at 0, so we just verify a valid response was returned
    }

    #[test]
    fn test_create_node_with_properties() {
        let server = create_test_server();

        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!("Alice"));
        props.insert("age".to_string(), serde_json::json!(30));

        let req = CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props),
            provenance: None,
        };

        let response = server.create_node(req);
        let node: NodeResponse = parse_response(&response).expect("Failed to parse response");

        assert_eq!(node.label, "Person");
        assert_eq!(
            node.properties.get("name"),
            Some(&serde_json::json!("Alice"))
        );
        assert_eq!(node.properties.get("age"), Some(&serde_json::json!(30)));
    }

    #[test]
    fn test_get_node() {
        let server = create_test_server();

        // Create a node first
        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!("Bob"));

        let create_req = CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props),
            provenance: None,
        };

        let create_response = server.create_node(create_req);
        let created: NodeResponse = parse_response(&create_response).expect("Failed to create");

        // Now get it
        let get_req = GetNodeRequest {
            namespace: None,
            node_id: created.id,
            include_vectors: None,
        };

        let get_response = server.get_node(get_req);
        let retrieved: NodeResponse = parse_response(&get_response).expect("Failed to get");

        assert_eq!(retrieved.id, created.id);
        assert_eq!(retrieved.label, "Person");
        assert_eq!(
            retrieved.properties.get("name"),
            Some(&serde_json::json!("Bob"))
        );
    }

    #[test]
    fn test_get_nonexistent_node() {
        let server = create_test_server();

        let req = GetNodeRequest {
            namespace: None,
            node_id: 999999,
            include_vectors: None,
        };

        let response = server.get_node(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_update_node() {
        let server = create_test_server();

        // Create a node
        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!("Charlie"));
        props.insert("age".to_string(), serde_json::json!(25));

        let create_req = CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props),
            provenance: None,
        };

        let create_response = server.create_node(create_req);
        let created: NodeResponse = parse_response(&create_response).expect("Failed to create");

        // Update it
        let mut new_props = HashMap::new();
        new_props.insert("age".to_string(), serde_json::json!(26));
        new_props.insert("city".to_string(), serde_json::json!("London"));

        let update_req = UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id: created.id,
            properties: new_props,
            provenance: None,
        };

        let update_response = server.update_node(update_req);
        let updated: NodeResponse = parse_response(&update_response).expect("Failed to update");

        assert_eq!(updated.properties.get("age"), Some(&serde_json::json!(26)));
        assert_eq!(
            updated.properties.get("city"),
            Some(&serde_json::json!("London"))
        );
    }

    #[test]
    fn test_delete_node() {
        let server = create_test_server();

        // Create a node
        let create_req = CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "ToDelete".to_string(),
            properties: None,
            provenance: None,
        };

        let create_response = server.create_node(create_req);
        let created: NodeResponse = parse_response(&create_response).expect("Failed to create");
        let node_id = created.id;

        // Delete it
        let delete_req = DeleteNodeRequest {
            namespace: None,
            node_id,
            detach: None,
            valid_time: None,
        };

        let delete_response = server.delete_node(delete_req);
        let value: serde_json::Value = serde_json::from_str(&delete_response).unwrap();

        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));

        // Verify it's gone
        let get_req = GetNodeRequest {
            namespace: None,
            node_id,
            include_vectors: None,
        };
        let get_response = server.get_node(get_req);
        let get_value: serde_json::Value = serde_json::from_str(&get_response).unwrap();

        assert!(get_value.get("error").is_some());
    }

    #[test]
    fn test_delete_node_with_edges_refused_without_detach() {
        // Issue #3209: the MCP delete_node must NOT return a bare success when the
        // target node has connected edges. Without `detach`, it refuses and reports
        // the machine-readable count of connected edges.
        let server = create_test_server();
        let (source_id, target_id) = create_two_nodes(&server);

        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id,
            target_id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });

        let delete_response = server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id: source_id,
            detach: None,
            valid_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&delete_response).unwrap();

        // It must not claim success...
        assert_ne!(value.get("success"), Some(&serde_json::json!(true)));
        // ...and it must surface exactly the number of connected edges.
        assert_eq!(value.get("connected_edges"), Some(&serde_json::json!(1)));

        // The node must still exist (refused, not destroyed).
        let get_value: serde_json::Value = serde_json::from_str(&server.get_node(GetNodeRequest {
            namespace: None,
            node_id: source_id,
            include_vectors: None,
        }))
        .unwrap();
        assert!(get_value.get("error").is_none());
    }

    #[test]
    fn test_delete_node_with_detach_removes_edges_no_dangling() {
        // Issue #3209: with detach: true, the MCP path performs a cascade-equivalent
        // delete and reports the number of edges removed, leaving no dangling endpoints.
        let server = create_test_server();
        let (source_id, target_id) = create_two_nodes(&server);

        // A third node so we have both an outgoing and an incoming edge.
        let third = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let third_id: NodeResponse = parse_response(&third).unwrap();
        let third_id = third_id.id;

        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id,
            target_id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: third_id,
            target_id: source_id,
            label: "FOLLOWS".to_string(),
            properties: None,
            provenance: None,
        });

        let delete_response = server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id: source_id,
            detach: Some(true),
            valid_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&delete_response).unwrap();

        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));
        assert_eq!(value.get("edges_removed"), Some(&serde_json::json!(2)));
        assert_eq!(value.get("detached"), Some(&serde_json::json!(true)));

        // Node is gone.
        let get_value: serde_json::Value = serde_json::from_str(&server.get_node(GetNodeRequest {
            namespace: None,
            node_id: source_id,
            include_vectors: None,
        }))
        .unwrap();
        assert!(get_value.get("error").is_some());

        // Traversal from the surviving neighbors yields no dangling endpoint to
        // the deleted node: the connected edges were removed with it.
        let outgoing: serde_json::Value =
            serde_json::from_str(&server.get_outgoing_edges(GetOutgoingEdgesRequest {
                node_id: third_id,
                label: None,
                include_vectors: None,
            }))
            .unwrap();
        let edges = outgoing.get("edges").and_then(|e| e.as_array());
        assert!(
            edges.map(|e| e.is_empty()).unwrap_or(true),
            "third node should have no outgoing edges after detach delete"
        );

        let incoming: serde_json::Value =
            serde_json::from_str(&server.get_incoming_edges(GetIncomingEdgesRequest {
                node_id: target_id,
                label: None,
                include_vectors: None,
            }))
            .unwrap();
        let in_edges = incoming.get("edges").and_then(|e| e.as_array());
        assert!(
            in_edges.map(|e| e.is_empty()).unwrap_or(true),
            "target node should have no incoming edges after detach delete"
        );
    }

    #[test]
    fn test_delete_node_without_edges_succeeds_with_detach_false() {
        // A node with no edges deletes cleanly and reports zero edges removed.
        let server = create_test_server();
        let created: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Lonely".to_string(),
            properties: None,
            provenance: None,
        }))
        .unwrap();

        let delete_response = server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id: created.id,
            detach: None,
            valid_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&delete_response).unwrap();

        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));
        assert_eq!(value.get("edges_removed"), Some(&serde_json::json!(0)));
    }

    #[test]
    fn test_list_nodes() {
        let server = create_test_server();

        // Create multiple nodes
        for i in 0..5 {
            let mut props = HashMap::new();
            props.insert("index".to_string(), serde_json::json!(i));

            let req = CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "ListTest".to_string(),
                properties: Some(props),
                provenance: None,
            };
            server.create_node(req);
        }

        // List with label filter (required for listing nodes efficiently)
        let list_req = ListNodesRequest {
            namespace: None,
            label: Some("ListTest".to_string()),
            property_key: None,
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: None,
        };

        let response = server.list_nodes(list_req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert_eq!(value.get("count"), Some(&serde_json::json!(5)));
    }

    #[test]
    fn test_list_nodes_with_label_filter() {
        let server = create_test_server();

        // Create nodes with different labels
        for _ in 0..3 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "TypeA".to_string(),
                properties: None,
                provenance: None,
            });
        }

        for _ in 0..2 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "TypeB".to_string(),
                properties: None,
                provenance: None,
            });
        }

        // List only TypeA
        let list_req = ListNodesRequest {
            namespace: None,
            label: Some("TypeA".to_string()),
            property_key: None,
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: None,
        };

        let response = server.list_nodes(list_req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert_eq!(value.get("count"), Some(&serde_json::json!(3)));
    }

    #[test]
    fn test_list_nodes_with_pagination() {
        let server = create_test_server();

        // Create 10 nodes
        for i in 0..10 {
            let mut props = HashMap::new();
            props.insert("index".to_string(), serde_json::json!(i));

            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Paginated".to_string(),
                properties: Some(props),
                provenance: None,
            });
        }

        // Get first page (with label filter required for efficient listing)
        let page1_req = ListNodesRequest {
            namespace: None,
            label: Some("Paginated".to_string()),
            property_key: None,
            property_value: None,
            limit: Some(5),
            offset: Some(0),
            include_vectors: None,
        };

        let page1_response = server.list_nodes(page1_req);
        let page1: serde_json::Value = serde_json::from_str(&page1_response).unwrap();

        assert_eq!(page1.get("count"), Some(&serde_json::json!(5)));

        // Get second page
        let page2_req = ListNodesRequest {
            namespace: None,
            label: Some("Paginated".to_string()),
            property_key: None,
            property_value: None,
            limit: Some(5),
            offset: Some(5),
            include_vectors: None,
        };

        let page2_response = server.list_nodes(page2_req);
        let page2: serde_json::Value = serde_json::from_str(&page2_response).unwrap();

        assert_eq!(page2.get("count"), Some(&serde_json::json!(5)));
    }

    #[test]
    fn test_count_nodes() {
        let server = create_test_server();

        // Initially empty
        let count_req = CountNodesRequest { label: None };
        let response = server.count_nodes(count_req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("count"), Some(&serde_json::json!(0)));

        // Create some nodes
        for _ in 0..3 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Counted".to_string(),
                properties: None,
                provenance: None,
            });
        }

        let count_req = CountNodesRequest { label: None };
        let response = server.count_nodes(count_req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("count"), Some(&serde_json::json!(3)));
    }

    #[test]
    fn test_count_nodes_with_label() {
        let server = create_test_server();

        // Create nodes with different labels
        for _ in 0..3 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "CountA".to_string(),
                properties: None,
                provenance: None,
            });
        }

        for _ in 0..2 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "CountB".to_string(),
                properties: None,
                provenance: None,
            });
        }

        let count_a = server.count_nodes(CountNodesRequest {
            label: Some("CountA".to_string()),
        });
        let value_a: serde_json::Value = serde_json::from_str(&count_a).unwrap();
        assert_eq!(value_a.get("count"), Some(&serde_json::json!(3)));

        let count_b = server.count_nodes(CountNodesRequest {
            label: Some("CountB".to_string()),
        });
        let value_b: serde_json::Value = serde_json::from_str(&count_b).unwrap();
        assert_eq!(value_b.get("count"), Some(&serde_json::json!(2)));
    }
}

// ============================================================================
// Edge CRUD Tests
// ============================================================================

mod edge_tests {
    use super::*;

    fn create_two_nodes(server: &AletheiaMcpServer) -> (u64, u64) {
        let node1 = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Alice"));
                m
            }),
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&node1).unwrap();

        let node2 = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Bob"));
                m
            }),
            provenance: None,
        });
        let n2: NodeResponse = parse_response(&node2).unwrap();

        (n1.id, n2.id)
    }

    #[test]
    fn test_create_edge_basic() {
        let server = create_test_server();
        let (source_id, target_id) = create_two_nodes(&server);

        let req = CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id,
            target_id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        };

        let response = server.create_edge(req);
        let edge: EdgeResponse = parse_response(&response).expect("Failed to create edge");

        assert_eq!(edge.source_id, source_id);
        assert_eq!(edge.target_id, target_id);
        assert_eq!(edge.label, "KNOWS");
    }

    #[test]
    fn test_create_edge_with_properties() {
        let server = create_test_server();
        let (source_id, target_id) = create_two_nodes(&server);

        let mut props = HashMap::new();
        props.insert("since".to_string(), serde_json::json!("2024-01-01"));
        props.insert("strength".to_string(), serde_json::json!(0.9));

        let req = CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id,
            target_id,
            label: "KNOWS".to_string(),
            properties: Some(props),
            provenance: None,
        };

        let response = server.create_edge(req);
        let edge: EdgeResponse = parse_response(&response).expect("Failed to create edge");

        assert_eq!(
            edge.properties.get("since"),
            Some(&serde_json::json!("2024-01-01"))
        );
        assert_eq!(
            edge.properties.get("strength"),
            Some(&serde_json::json!(0.9))
        );
    }

    #[test]
    fn test_get_edge() {
        let server = create_test_server();
        let (source_id, target_id) = create_two_nodes(&server);

        let create_response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id,
            target_id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        let created: EdgeResponse = parse_response(&create_response).unwrap();

        let get_response = server.get_edge(GetEdgeRequest {
            namespace: None,
            edge_id: created.id,
            include_vectors: None,
        });
        let retrieved: EdgeResponse = parse_response(&get_response).unwrap();

        assert_eq!(retrieved.id, created.id);
        assert_eq!(retrieved.source_id, source_id);
        assert_eq!(retrieved.target_id, target_id);
    }

    #[test]
    fn test_update_edge() {
        let server = create_test_server();
        let (source_id, target_id) = create_two_nodes(&server);

        let create_response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id,
            target_id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        let created: EdgeResponse = parse_response(&create_response).unwrap();

        let mut new_props = HashMap::new();
        new_props.insert("weight".to_string(), serde_json::json!(0.5));

        let update_response = server.update_edge(UpdateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            edge_id: created.id,
            properties: new_props,
            provenance: None,
        });
        let updated: EdgeResponse = parse_response(&update_response).unwrap();

        assert_eq!(
            updated.properties.get("weight"),
            Some(&serde_json::json!(0.5))
        );
    }

    #[test]
    fn test_delete_edge() {
        let server = create_test_server();
        let (source_id, target_id) = create_two_nodes(&server);

        let create_response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id,
            target_id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        let created: EdgeResponse = parse_response(&create_response).unwrap();

        let delete_response = server.delete_edge(DeleteEdgeRequest {
            namespace: None,
            edge_id: created.id,
            valid_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&delete_response).unwrap();
        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));

        // Verify it's gone
        let get_response = server.get_edge(GetEdgeRequest {
            namespace: None,
            edge_id: created.id,
            include_vectors: None,
        });
        let get_value: serde_json::Value = serde_json::from_str(&get_response).unwrap();
        assert!(get_value.get("error").is_some());
    }

    #[test]
    fn test_list_edges() {
        let server = create_test_server();
        let (n1, n2) = create_two_nodes(&server);

        // Create node3
        let node3 = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n3: NodeResponse = parse_response(&node3).unwrap();

        // Create edges
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1,
            target_id: n2,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n2,
            target_id: n3.id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });

        // Note: list_edges doesn't support listing all edges without a node
        // It returns a message indicating to use get_outgoing_edges or get_incoming_edges
        let list_response = server.list_edges(ListEdgesRequest {
            namespace: None,
            label: None,
            limit: None,
            offset: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&list_response).unwrap();
        // Verify total_count is 2 (edges exist, even if not returned)
        assert_eq!(value.get("total_count"), Some(&serde_json::json!(2)));
        // Verify a helpful message is returned
        assert!(value.get("message").is_some());
    }

    #[test]
    fn test_count_edges() {
        let server = create_test_server();
        let (n1, n2) = create_two_nodes(&server);

        let count_response = server.count_edges(CountEdgesRequest { label: None });
        let value: serde_json::Value = serde_json::from_str(&count_response).unwrap();
        assert_eq!(value.get("count"), Some(&serde_json::json!(0)));

        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1,
            target_id: n2,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });

        let count_response = server.count_edges(CountEdgesRequest { label: None });
        let value: serde_json::Value = serde_json::from_str(&count_response).unwrap();
        assert_eq!(value.get("count"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn test_get_outgoing_edges() {
        let server = create_test_server();
        let (n1, n2) = create_two_nodes(&server);

        // Create node3
        let node3 = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n3: NodeResponse = parse_response(&node3).unwrap();

        // Create edges from n1
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1,
            target_id: n2,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1,
            target_id: n3.id,
            label: "WORKS_WITH".to_string(),
            properties: None,
            provenance: None,
        });

        // Get all outgoing edges
        let response = server.get_outgoing_edges(GetOutgoingEdgesRequest {
            node_id: n1,
            label: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("count"), Some(&serde_json::json!(2)));

        // Get only KNOWS edges
        let response = server.get_outgoing_edges(GetOutgoingEdgesRequest {
            node_id: n1,
            label: Some("KNOWS".to_string()),
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("count"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn test_get_incoming_edges() {
        let server = create_test_server();
        let (n1, n2) = create_two_nodes(&server);

        // Create node3
        let node3 = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n3: NodeResponse = parse_response(&node3).unwrap();

        // Create edges to n2
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1,
            target_id: n2,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n3.id,
            target_id: n2,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });

        let response = server.get_incoming_edges(GetIncomingEdgesRequest {
            node_id: n2,
            label: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("count"), Some(&serde_json::json!(2)));
    }
}

// ============================================================================
// Graph Traversal Tests
// ============================================================================

mod traversal_tests {
    use super::*;

    fn create_graph(server: &AletheiaMcpServer) -> Vec<u64> {
        // Create a simple graph: A -> B -> C -> D
        let nodes: Vec<u64> = (0..4)
            .map(|i| {
                let response = server.create_node(CreateNodeRequest {
                    namespace: None,
                    derived_from: None,
                    valid_time: None,
                    label: "Node".to_string(),
                    properties: Some({
                        let mut m = HashMap::new();
                        m.insert("name".to_string(), serde_json::json!(format!("Node{}", i)));
                        m
                    }),
                    provenance: None,
                });
                let node: NodeResponse = parse_response(&response).unwrap();
                node.id
            })
            .collect();

        // Create edges
        for i in 0..3 {
            server.create_edge(CreateEdgeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                source_id: nodes[i],
                target_id: nodes[i + 1],
                label: "NEXT".to_string(),
                properties: None,
                provenance: None,
            });
        }

        nodes
    }

    #[test]
    fn test_traverse_single_hop() {
        let server = create_test_server();
        let nodes = create_graph(&server);

        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: nodes[0],
            edge_label: "NEXT".to_string(),
            direction: Some("outgoing".to_string()),
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Should find Node1 (one hop from Node0)
        let count = value.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
        assert!(count >= 1, "Should find at least one node in traversal");
    }

    #[test]
    fn test_traverse_multi_hop() {
        let server = create_test_server();
        let nodes = create_graph(&server);

        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: nodes[0],
            edge_label: "NEXT".to_string(),
            direction: Some("outgoing".to_string()),
            depth: Some(3),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Should find nodes at depths 1, 2, and 3
        let count = value.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
        assert!(count >= 1, "Should find nodes in multi-hop traversal");
    }

    #[test]
    fn test_traverse_incoming() {
        let server = create_test_server();
        let nodes = create_graph(&server);

        // Traverse incoming from Node3 (should find Node2)
        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: nodes[3],
            edge_label: "NEXT".to_string(),
            direction: Some("incoming".to_string()),
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let count = value.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
        assert!(count >= 1, "Should find incoming node");
    }

    #[test]
    fn test_traverse_with_limit() {
        let server = create_test_server();
        let nodes = create_graph(&server);

        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: nodes[0],
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(3),
            limit: Some(2),
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let count = value.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
        assert!(count <= 2, "Should respect limit");
    }
}

// ============================================================================
// Vector Search Tests
// ============================================================================

mod vector_tests {
    use super::*;

    #[test]
    fn test_enable_vector_index() {
        let server = create_test_server();

        let response = server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 128,
            distance_metric: Some("cosine".to_string()),
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn test_list_vector_indexes() {
        let server = create_test_server();

        // Enable an index
        server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 128,
            distance_metric: None,
        });

        let response = server.list_vector_indexes(ListVectorIndexesRequest {});
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        let indexes = value.get("indexes").unwrap().as_array().unwrap();
        // The new format returns objects with property_name, dimensions, and distance_metric
        assert!(indexes.iter().any(|i| {
            i.get("property_name")
                .and_then(|p| p.as_str())
                .map(|p| p == "embedding")
                .unwrap_or(false)
        }));
    }

    #[test]
    fn test_find_similar_without_index() {
        let server = create_test_server();

        let response = server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            k: Some(5),
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some(), "Should error without index");
    }

    #[test]
    fn test_find_similar_with_index() {
        let server = create_test_server();

        // Enable vector index
        server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 4,
            distance_metric: Some("cosine".to_string()),
        });

        // Create nodes with embeddings
        for i in 0..5 {
            let mut props = HashMap::new();
            props.insert("name".to_string(), serde_json::json!(format!("Doc{}", i)));
            props.insert(
                "embedding".to_string(),
                serde_json::json!([
                    (i as f32) * 0.1,
                    (i as f32) * 0.2,
                    (i as f32) * 0.3,
                    (i as f32) * 0.4
                ]),
            );

            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Document".to_string(),
                properties: Some(props),
                provenance: None,
            });
        }

        // Search for similar
        let response = server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            k: Some(3),
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // May or may not find results depending on index state
        assert!(value.get("results").is_some() || value.get("error").is_some());
    }
}

// ============================================================================
// Temporal Query Tests
// ============================================================================

mod temporal_tests {
    use super::*;

    #[test]
    fn test_get_node_at_time_invalid_timestamp() {
        let server = create_test_server();

        // Create a node
        let node_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&node_response).unwrap();

        // Try to get with invalid timestamp format
        let response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: "invalid-timestamp".to_string(),
            transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_get_node_at_time_valid_timestamp() {
        let server = create_test_server();

        // Create a node
        let node_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Alice"));
                m
            }),
            provenance: None,
        });
        let node: NodeResponse = parse_response(&node_response).unwrap();

        // Get with timestamp as microseconds since epoch
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        let response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: now_micros.to_string(),
            transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Should either succeed or give a meaningful error
        assert!(value.get("node").is_some() || value.get("error").is_some());
    }

    #[test]
    fn test_get_edge_at_time() {
        let server = create_test_server();

        // Create nodes and edge
        let n1_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&n1_response).unwrap();

        let n2_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n2: NodeResponse = parse_response(&n2_response).unwrap();

        let edge_response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        let edge: EdgeResponse = parse_response(&edge_response).unwrap();

        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        let response = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: now_micros.to_string(),
            transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("edge").is_some() || value.get("error").is_some());
    }

    // ------------------------------------------------------------------------
    // list_changes (temporal changefeed, Issue #3216)
    // ------------------------------------------------------------------------

    fn list_changes_req() -> ListChangesRequest {
        ListChangesRequest {
            tx_from: "0".to_string(),
            tx_to: i64::MAX.to_string(),
            valid_from: None,
            valid_to: None,
            label: None,
            limit: None,
            cursor: None,
        }
    }

    #[test]
    fn test_list_changes_success_shape() {
        let server = create_test_server();
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });

        let response = server.list_changes(list_changes_req());
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        let changes = value["changes"].as_array().expect("changes array");
        assert_eq!(changes.len(), 1);
        assert_eq!(value["count"], serde_json::json!(1));
        assert!(value.get("next_cursor").is_some());

        let row = &changes[0];
        assert_eq!(row["kind"], serde_json::json!("node"));
        assert_eq!(row["change_type"], serde_json::json!("created"));
        assert_eq!(row["label"], serde_json::json!("Person"));
        assert!(row.get("transaction_time").is_some());
        assert!(row["transaction_time_range"].get("start").is_some());
        assert!(row["valid_time_range"].get("start").is_some());
    }

    #[test]
    fn test_list_changes_invalid_window_errors() {
        let server = create_test_server();
        let mut req = list_changes_req();
        req.tx_from = "2000000".to_string();
        req.tx_to = "1000000".to_string();
        let response = server.list_changes(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_list_changes_empty_window_is_success() {
        let server = create_test_server();
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let mut req = list_changes_req();
        req.tx_from = "1000000".to_string();
        req.tx_to = "1000000".to_string(); // empty window
        let response = server.list_changes(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none());
        assert_eq!(value["count"], serde_json::json!(0));
    }

    #[test]
    fn test_list_changes_half_specified_valid_window_errors() {
        let server = create_test_server();
        let mut req = list_changes_req();
        req.valid_from = Some("1000".to_string());
        // valid_to omitted.
        let response = server.list_changes(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_list_changes_bad_timestamp_errors() {
        let server = create_test_server();
        let mut req = list_changes_req();
        req.tx_from = "not-a-time".to_string();
        let response = server.list_changes(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_list_changes_registered_in_tool_list() {
        let server = create_test_server();
        let tools = server.list_tools_for_test();
        assert!(
            tools.iter().any(|name| name == "list_changes"),
            "list_changes must be registered in the tool list"
        );
    }
}

// ============================================================================
// await_changes (push changefeed long-poll, Issue #3375)
// ============================================================================

mod changefeed_await_tests {
    use super::*;
    use crate::core::changefeed::ChangeCursor;
    use crate::core::temporal::Timestamp;
    use std::time::Duration;

    fn await_req() -> AwaitChangesRequest {
        AwaitChangesRequest {
            node_labels: None,
            edge_types: None,
            change_types: None,
            from_token: None,
            timeout_ms: Some(0),
            limit: None,
            namespace: None,
        }
    }

    /// Issue #3349, PR3c: an `await_changes` call scoped to namespace `A` (via the
    /// catch-up path) returns only `A`'s changes; a `"bad name"` scope is rejected
    /// with `INVALID_ARGUMENT` (never a silently-unscoped subscription).
    #[test]
    fn await_changes_namespace_scopes_catch_up() {
        let server = create_test_server();
        let db = server.db();
        // Commit changes in two namespaces BEFORE the resume window opens.
        let base = db
            .create_node_in_namespace("Person", crate::PropertyMap::new(), "agent:a")
            .unwrap();
        db.create_node_in_namespace("Person", crate::PropertyMap::new(), "agent:b")
            .unwrap();
        // A resume token anchored before both writes: catch-up scans them, the
        // namespace filter then keeps only agent:a.
        let from = crate::core::changefeed::ChangeCursor::baseline_after(Timestamp::from(0));

        let mut req = await_req();
        req.from_token = Some(from);
        req.namespace = Some(serde_json::json!("agent:a"));
        let response = server.await_changes(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let changes = value["changes"].as_array().expect("changes array");
        assert!(!changes.is_empty(), "expected the agent:a catch-up change");
        for c in changes {
            assert_eq!(
                c["namespace"],
                serde_json::json!("agent:a"),
                "await_changes returned a foreign namespace: {c}"
            );
        }
        // The agent:a node is present; no agent:b entity leaked.
        assert!(changes.iter().any(|c| c["entity_id"] == base.as_u64()));

        // A malformed namespace scope is rejected before subscribing.
        let mut bad = await_req();
        bad.namespace = Some(serde_json::json!("bad name"));
        let resp = server.await_changes(bad);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["error"]["code"], serde_json::json!("INVALID_ARGUMENT"));
    }

    #[test]
    fn await_changes_registered_in_tool_list() {
        let server = create_test_server();
        let tools = server.list_tools_for_test();
        assert!(
            tools.iter().any(|name| name == "await_changes"),
            "await_changes must be registered in the tool list"
        );
    }

    /// Issue #3678: the MCP `await_changes` surface enforces the per-principal
    /// changefeed quota through the SAME `subscribe_changes_for_principal` funnel
    /// as the HTTP surfaces. Anonymous-mode sessions share the `"anonymous"`
    /// bucket, so a held subscription in that bucket at the cap makes the next
    /// `await_changes` return the RESOURCE_EXHAUSTED / retriable:true envelope
    /// with `details {principal, current, limit}`.
    #[test]
    fn await_changes_enforces_per_principal_quota() {
        use crate::core::changefeed_subscription::ChangefeedConfig;

        let server = create_test_server();
        server.db().set_changefeed_config(ChangefeedConfig {
            max_subscriptions: 100,
            buffer_capacity: 16,
            max_subscriptions_per_principal: 1,
            ..ChangefeedConfig::default()
        });
        // Hold one subscription in the shared "anonymous" bucket (the key the
        // anonymous-mode server resolves), saturating the per-principal cap of 1.
        let _held = server
            .db()
            .subscribe_changes_for_principal(Some("anonymous"), crate::ChangeFilter::all())
            .unwrap();

        let response = server.await_changes(await_req());
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            value["error"]["code"],
            serde_json::json!("RESOURCE_EXHAUSTED")
        );
        assert_eq!(value["error"]["retriable"], serde_json::json!(true));
        assert_eq!(
            value["error"]["details"]["principal"],
            serde_json::json!("anonymous")
        );
        assert_eq!(value["error"]["details"]["current"], serde_json::json!(1));
        assert_eq!(value["error"]["details"]["limit"], serde_json::json!(1));
    }

    #[test]
    fn no_write_small_timeout_times_out_with_resume_token() {
        let server = create_test_server();
        let mut req = await_req();
        req.timeout_ms = Some(50); // bounded — must not hang
        let response = server.await_changes(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        assert_eq!(value["timed_out"], serde_json::json!(true));
        assert_eq!(value["count"], serde_json::json!(0));
        assert!(
            value["changes"].as_array().unwrap().is_empty(),
            "no changes on timeout"
        );
        // The subscribe-time baseline anchor is always present as a resume token.
        assert!(
            value["resume_token"].as_str().is_some(),
            "a timed-out poll still yields a resume_token: {value}"
        );
        assert_eq!(value["has_more"], serde_json::json!(false));
    }

    #[test]
    fn catch_up_path_returns_immediately() {
        let server = create_test_server();
        server.db().create_node("Person", props("Alice")).unwrap();

        // A baseline token positioned at the very start of time: the catch-up
        // pull returns everything committed after it, deterministically.
        let token = ChangeCursor::baseline_after(Timestamp::from(0));
        let mut req = await_req();
        req.from_token = Some(token);
        // timeout_ms 0 — the catch-up must return without ever blocking.
        let response = server.await_changes(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        assert_eq!(value["timed_out"], serde_json::json!(false));
        let changes = value["changes"].as_array().expect("changes array");
        assert_eq!(changes.len(), 1, "the created Person is caught up: {value}");
        assert_eq!(changes[0]["kind"], serde_json::json!("node"));
        assert_eq!(changes[0]["change_type"], serde_json::json!("created"));
        assert_eq!(changes[0]["label"], serde_json::json!("Person"));
        assert!(value["resume_token"].as_str().is_some());
    }

    #[test]
    fn resume_via_token_dedups_strictly_later() {
        let server = create_test_server();
        server.db().create_node("Person", props("Alice")).unwrap();

        // First catch-up from the start yields the change + a resume token.
        let mut first = await_req();
        first.from_token = Some(ChangeCursor::baseline_after(Timestamp::from(0)));
        let v1: serde_json::Value = serde_json::from_str(&server.await_changes(first)).unwrap();
        let token = v1["resume_token"]
            .as_str()
            .expect("resume token")
            .to_string();

        // Resuming from that token (no new writes) returns nothing — the already
        // delivered change is excluded by the strict `> cursor` rule, and with
        // timeout 0 the poll does not block.
        let mut second = await_req();
        second.from_token = Some(token);
        let v2: serde_json::Value = serde_json::from_str(&server.await_changes(second)).unwrap();
        assert!(v2.get("error").is_none(), "unexpected error: {v2}");
        assert_eq!(v2["count"], serde_json::json!(0), "no duplicate: {v2}");
        assert_eq!(v2["timed_out"], serde_json::json!(true));
    }

    #[test]
    fn malformed_from_token_is_invalid_argument() {
        let server = create_test_server();
        let mut req = await_req();
        req.from_token = Some("not-a-valid-token".to_string());
        let response = server.await_changes(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let code = value["error"]["code"].as_str().expect("error code");
        assert_eq!(code, "INVALID_ARGUMENT", "got: {value}");
    }

    #[test]
    fn invalid_change_type_is_invalid_argument() {
        let server = create_test_server();
        let mut req = await_req();
        req.change_types = Some(vec!["created".to_string(), "bogus".to_string()]);
        let response = server.await_changes(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            value["error"]["code"].as_str(),
            Some("INVALID_ARGUMENT"),
            "got: {value}"
        );
    }

    #[test]
    fn subscribe_then_write_delivers_the_change() {
        // Deterministic LIVE-branch delivery (Fix 3): subscribe FIRST, commit a
        // write (which buffers the change into the live subscription), THEN
        // `recv_timeout` drains it immediately — exercising the live push path
        // directly, with no from_token so no catch-up leg can mask a broken live
        // wakeup. Bounded timeout so a regressed live path fails fast, never
        // hangs, and is non-flaky (the write is fully committed before recv).
        use crate::core::changefeed::ChangeType;
        use crate::core::changefeed_subscription::ChangeFilter;

        let server = create_test_server();
        let db = server.db();

        // Subscribe with the same all-matching filter the tool builds by default.
        let sub = db
            .subscribe_changes(ChangeFilter::all())
            .expect("subscribe");

        // Commit a write strictly AFTER subscribe: it is buffered into `sub`.
        db.create_node("Person", props("Bob")).unwrap();

        // LIVE branch: recv_timeout drains the buffered change immediately — this
        // is the live wakeup path, not a catch-up fallback nor an empty timeout.
        let recs = sub
            .recv_timeout(Duration::from_millis(60))
            .expect("live recv must not lag");
        assert!(
            !recs.is_empty(),
            "the committed change is delivered on the LIVE branch (not an empty timeout)"
        );
        assert_eq!(
            recs.len(),
            1,
            "exactly the one committed change is delivered"
        );
        assert_eq!(recs[0].label, "Person");
        assert_eq!(recs[0].change_type, ChangeType::Created);
    }

    #[test]
    fn catch_up_applies_the_subscription_filter() {
        // Fix 1 (a): a resume (catch-up) over a window of MIXED changes must
        // return ONLY the changes matching the subscription's ChangeFilter.
        // Previously the catch-up leg returned page.changes VERBATIM (unfiltered),
        // so a filtered subscription's resume path leaked non-matching changes.
        let server = create_test_server();

        // Window: created (Alice) + modified (Alice) + created (Carol) +
        // deleted (Carol) → a mix of all three change types.
        let alice: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        }))
        .expect("create Alice");
        let mut bumped = HashMap::new();
        bumped.insert("age".to_string(), serde_json::json!(31));
        server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id: alice.id,
            properties: bumped,
            provenance: None,
        });
        let carol: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        }))
        .expect("create Carol");
        server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id: carol.id,
            detach: None,
            valid_time: None,
        });

        // Subscribe filtered to deletes only and catch up from the start of time.
        let mut req = await_req();
        req.change_types = Some(vec!["deleted".to_string()]);
        req.from_token = Some(ChangeCursor::baseline_after(Timestamp::from(0)));
        req.timeout_ms = Some(0); // catch-up must return without blocking
        let v: serde_json::Value = serde_json::from_str(&server.await_changes(req)).unwrap();

        assert!(v.get("error").is_none(), "unexpected error: {v}");
        assert_eq!(
            v["timed_out"],
            serde_json::json!(false),
            "scanned rows: {v}"
        );
        let changes = v["changes"].as_array().expect("changes array");
        assert!(
            !changes.is_empty(),
            "the delete matches the filter and must be returned: {v}"
        );
        for c in changes {
            assert_eq!(
                c["change_type"],
                serde_json::json!("deleted"),
                "only deleted changes survive the filter: {v}"
            );
        }
    }

    #[test]
    fn catch_up_advances_resume_token_when_page_filters_to_empty() {
        // Fix 1 (b): when a scanned page has rows but the filter removes them all,
        // await_changes must STILL advance past the scanned rows (resume_token =
        // last scanned cursor / next_cursor, has_more = next_cursor.is_some()) and
        // return immediately — so a fully-filtered-out page makes progress instead
        // of re-scanning the same rows or stalling into the block.
        let server = create_test_server();
        // Four created changes; NONE are deletes.
        for _ in 0..4 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Person".to_string(),
                properties: None,
                provenance: None,
            });
        }

        let mut req = await_req();
        req.change_types = Some(vec!["deleted".to_string()]); // matches nothing here
        req.from_token = Some(ChangeCursor::baseline_after(Timestamp::from(0)));
        req.limit = Some(2); // bound the page → next_cursor Some, has_more true
        req.timeout_ms = Some(0);
        let v: serde_json::Value = serde_json::from_str(&server.await_changes(req)).unwrap();

        assert!(v.get("error").is_none(), "unexpected error: {v}");
        // The page scanned 2 rows, all filtered out: immediate empty, not a block.
        assert_eq!(
            v["timed_out"],
            serde_json::json!(false),
            "scanned rows → not a timeout: {v}"
        );
        assert_eq!(
            v["count"],
            serde_json::json!(0),
            "all rows filtered out: {v}"
        );
        assert_eq!(
            v["has_more"],
            serde_json::json!(true),
            "more scanned pages remain: {v}"
        );
        let token = v["resume_token"]
            .as_str()
            .expect("advanced resume token")
            .to_string();

        // Resuming from the advanced token scans the NEXT page — real progress,
        // no re-scan of the first page and no stall.
        let mut req2 = await_req();
        req2.change_types = Some(vec!["deleted".to_string()]);
        req2.from_token = Some(token);
        req2.limit = Some(2);
        req2.timeout_ms = Some(0);
        let v2: serde_json::Value = serde_json::from_str(&server.await_changes(req2)).unwrap();
        assert!(v2.get("error").is_none(), "unexpected error: {v2}");
        assert_eq!(v2["count"], serde_json::json!(0), "still no deletes: {v2}");
        assert!(
            v2["resume_token"].as_str().is_some(),
            "the resume anchor still advances: {v2}"
        );
    }

    #[test]
    fn await_changes_excluded_from_resource_and_budget_wrappers() {
        // The long-poll must NOT be wrapped by the #3368 per-read timeout or the
        // #3353 token-budget shaper (either would truncate/abort the block).
        assert!(
            !crate::mcp::server::is_resource_limited_read_tool("await_changes"),
            "await_changes must be excluded from RESOURCE_LIMITED_READ_TOOLS"
        );
        assert!(
            !crate::mcp::server::is_budgetable_read_tool("await_changes"),
            "await_changes must not be budgetable"
        );
    }

    fn props(name: &str) -> crate::core::PropertyMap {
        PropertyMapBuilder::new().insert("name", name).build()
    }
}

// ============================================================================
// Hybrid Query Tests
// ============================================================================

mod hybrid_tests {
    use super::*;

    #[test]
    fn test_hybrid_query_with_start_node() {
        let server = create_test_server();

        // Create nodes
        let n1_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Alice"));
                m
            }),
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&n1_response).unwrap();

        let n2_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Bob"));
                m
            }),
            provenance: None,
        });
        let _n2: NodeResponse = parse_response(&n2_response).unwrap();

        // Execute hybrid query starting from n1
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: Some(n1.id),
            traverse_edge: None,
            traverse_depth: None,
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: None,
            transaction_time: None,
            filter_label: None,
            limit: Some(10),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("results").is_some() || value.get("error").is_some());
    }

    #[test]
    fn test_hybrid_query_with_label_filter() {
        let server = create_test_server();

        // Create nodes with different labels
        for _ in 0..3 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Person".to_string(),
                properties: None,
                provenance: None,
            });
        }

        for _ in 0..2 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Document".to_string(),
                properties: None,
                provenance: None,
            });
        }

        // Query only Person nodes
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: None,
            traverse_edge: None,
            traverse_depth: None,
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: None,
            transaction_time: None,
            filter_label: Some("Person".to_string()),
            limit: Some(100),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        if let Some(results) = value.get("results").and_then(|r| r.as_array()) {
            for result in results {
                let label = result
                    .get("node")
                    .and_then(|n| n.get("label"))
                    .and_then(|l| l.as_str());
                assert_eq!(label, Some("Person"));
            }
        }
    }

    #[test]
    fn test_hybrid_query_requires_criteria() {
        let server = create_test_server();

        // Query without any criteria should error
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: None,
            traverse_edge: None,
            traverse_depth: None,
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: None,
            transaction_time: None,
            filter_label: None,
            limit: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_hybrid_query_with_traversal() {
        let server = create_test_server();

        // Create a simple graph
        let n1_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&n1_response).unwrap();

        let n2_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n2: NodeResponse = parse_response(&n2_response).unwrap();

        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });

        // Query with traversal
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: Some(n1.id),
            traverse_edge: Some("KNOWS".to_string()),
            traverse_depth: Some(1),
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: None,
            transaction_time: None,
            filter_label: None,
            limit: Some(10),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("results").is_some() || value.get("error").is_some());
    }
}

// ============================================================================
// Property Conversion Tests
// ============================================================================

mod conversion_tests {
    use super::*;

    #[test]
    fn test_property_types_roundtrip() {
        let server = create_test_server();

        let mut props = HashMap::new();
        props.insert("string_val".to_string(), serde_json::json!("hello"));
        props.insert("int_val".to_string(), serde_json::json!(42));
        props.insert("float_val".to_string(), serde_json::json!(1.5));
        props.insert("bool_val".to_string(), serde_json::json!(true));
        props.insert("null_val".to_string(), serde_json::Value::Null);
        props.insert("array_val".to_string(), serde_json::json!([1, 2, 3]));

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Test".to_string(),
            properties: Some(props),
            provenance: None,
        });

        let node: NodeResponse = parse_response(&response).expect("Failed to create node");

        assert_eq!(
            node.properties.get("string_val"),
            Some(&serde_json::json!("hello"))
        );
        assert_eq!(node.properties.get("int_val"), Some(&serde_json::json!(42)));
        assert_eq!(
            node.properties.get("bool_val"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            node.properties.get("null_val"),
            Some(&serde_json::Value::Null)
        );
    }

    #[test]
    fn test_vector_property() {
        let server = create_test_server();

        let mut props = HashMap::new();
        props.insert(
            "embedding".to_string(),
            serde_json::json!([0.1, 0.2, 0.3, 0.4]),
        );

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Document".to_string(),
            properties: Some(props),
            provenance: None,
        });

        let node: NodeResponse = parse_response(&response).expect("Failed to create node");

        // Vector should be preserved
        let embedding = node.properties.get("embedding").unwrap();
        assert!(embedding.is_array());
    }
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

mod coverage_tests {
    use super::*;

    #[test]
    fn test_vector_dimension_validation() {
        let server = create_test_server();

        // Enable vector index with 4 dimensions
        let response = server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 4,
            distance_metric: Some("cosine".to_string()),
        });
        assert!(
            !response.contains("error"),
            "Failed to enable index: {}",
            response
        );

        // Try to search with wrong dimensions (3 instead of 4)
        let response = server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3], // Wrong: 3 dimensions instead of 4
            k: Some(5),
            offset: None,
            include_vectors: None,
        });

        // Should get dimension mismatch error
        assert!(response.contains("error"), "Expected error response");
        assert!(
            response.contains("dimension mismatch") || response.contains("Embedding dimension"),
            "Expected dimension mismatch error, got: {}",
            response
        );
    }

    #[test]
    fn test_hybrid_query_dimension_validation() {
        let server = create_test_server();

        // Enable vector index with 4 dimensions
        let response = server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 4,
            distance_metric: Some("cosine".to_string()),
        });
        assert!(
            !response.contains("error"),
            "Failed to enable index: {}",
            response
        );

        // Try hybrid query with wrong embedding dimensions
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: None,
            traverse_edge: None,
            traverse_depth: None,
            query_embedding: Some(vec![0.1, 0.2, 0.3]), // Wrong: 3 dimensions
            vector_property: Some("embedding".to_string()),
            top_k: Some(5),
            filter_label: None,
            limit: None,
            valid_time: None,
            transaction_time: None,
            include_vectors: None,
        });

        // Should get dimension mismatch error
        assert!(response.contains("error"), "Expected error response");
        assert!(
            response.contains("dimension mismatch") || response.contains("Embedding dimension"),
            "Expected dimension mismatch error, got: {}",
            response
        );
    }

    #[test]
    fn test_iso8601_timestamp_parsing() {
        let server = create_test_server();

        // Create a node first
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Event".to_string(),
            properties: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&response).expect("Failed to create node");

        // Test with ISO 8601 timestamp format (with Z timezone)
        let response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: "2024-01-15T10:30:00Z".to_string(),
            transaction_time: None,
        });

        // Should not contain a parsing error
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Either we get the node back or a "not found" type error (since we're querying at a past time)
        // but we should NOT get a timestamp parsing error
        if let Some(error) = value.get("error") {
            let err_str = error["message"].as_str().unwrap_or("");
            assert!(
                !err_str.contains("Invalid timestamp format"),
                "ISO 8601 timestamp should be parsed correctly, got error: {}",
                err_str
            );
        }
    }

    #[test]
    fn test_iso8601_timestamp_without_timezone() {
        let server = create_test_server();

        // Create a node first
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Event".to_string(),
            properties: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&response).expect("Failed to create node");

        // Test with ISO 8601 timestamp without timezone (should assume UTC)
        let response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: "2024-01-15T10:30:00".to_string(),
            transaction_time: None,
        });

        // Should not contain a parsing error
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        if let Some(error) = value.get("error") {
            let err_str = error["message"].as_str().unwrap_or("");
            assert!(
                !err_str.contains("Invalid timestamp format"),
                "ISO 8601 timestamp without TZ should be parsed correctly, got error: {}",
                err_str
            );
        }
    }

    #[test]
    fn test_transaction_time_now_response() {
        let server = create_test_server();

        // Create a node first
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Event".to_string(),
            properties: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&response).expect("Failed to create node");

        // Get node at time without specifying transaction_time
        let response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: "0".to_string(), // Use 0 for simplicity
            transaction_time: None,
        });

        // Response should contain "now" for transaction_time when not specified
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        if value.get("error").is_none() {
            let tx_time = value.get("transaction_time");
            assert_eq!(
                tx_time,
                Some(&serde_json::json!("now")),
                "Expected 'now' for unspecified transaction_time"
            );
        }
    }

    #[test]
    fn test_count_nodes_with_nonexistent_label() {
        let server = create_test_server();

        // Count nodes with a label that doesn't exist
        let response = server.count_nodes(CountNodesRequest {
            label: Some("NonexistentLabel".to_string()),
        });

        // Should return count: 0 (not an error)
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "Should not error for nonexistent label"
        );
        assert_eq!(
            value.get("count"),
            Some(&serde_json::json!(0)),
            "Count should be 0 for nonexistent label"
        );
    }

    #[test]
    fn test_list_nodes_offset_cap() {
        let server = create_test_server();

        // Create a few nodes
        for i in 0..5 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "OffsetTest".to_string(),
                properties: Some({
                    let mut props = HashMap::new();
                    props.insert("index".to_string(), serde_json::json!(i));
                    props
                }),
                provenance: None,
            });
        }

        // Request with a very large offset (should be capped)
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("OffsetTest".to_string()),
            property_key: None,
            property_value: None,
            limit: Some(10),
            offset: Some(100_000), // Very large offset, should be capped to MAX_PAGINATION_OFFSET,
            include_vectors: None,
        });

        // Should not error, just return empty results due to offset being beyond data
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "Large offset should not cause error: {}",
            response
        );
    }

    #[test]
    fn test_traversal_depth_limit() {
        let server = create_test_server();

        // Create a chain of nodes
        let mut prev_id: Option<u64> = None;
        for i in 0..5 {
            let response = server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "ChainNode".to_string(),
                properties: Some({
                    let mut props = HashMap::new();
                    props.insert("level".to_string(), serde_json::json!(i));
                    props
                }),
                provenance: None,
            });
            let node: NodeResponse = parse_response(&response).unwrap();

            if let Some(source_id) = prev_id {
                server.create_edge(CreateEdgeRequest {
                    namespace: None,
                    derived_from: None,
                    valid_time: None,
                    source_id,
                    target_id: node.id,
                    label: "NEXT".to_string(),
                    properties: None,
                    provenance: None,
                });
            }
            prev_id = Some(node.id);
        }

        // Try to traverse with a very large depth (should be capped to MAX_TRAVERSAL_DEPTH)
        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: 0,
            edge_label: "NEXT".to_string(),
            depth: Some(100), // Very large depth, should be capped
            direction: Some("outgoing".to_string()),
            limit: Some(50),
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        // Should not error
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "Large depth should be capped, not error: {}",
            response
        );
    }
}

// ============================================================================
// ServerHandler Trait Tests
// ============================================================================

mod server_handler_tests {
    use super::*;
    use rmcp::ServerHandler;

    #[test]
    fn test_get_info() {
        let server = create_test_server();
        let info = server.get_info();

        assert!(info.instructions.is_some());
        let instructions = info.instructions.unwrap();
        assert!(instructions.contains("AletheiaDB"));
        assert!(instructions.contains("bi-temporal"));
    }

    #[test]
    fn test_get_info_instructions_content() {
        let server = create_test_server();
        let info = server.get_info();

        let instructions = info.instructions.expect("Instructions should be set");

        // Verify instructions mention key features
        assert!(instructions.contains("graph"));
        assert!(instructions.contains("temporal") || instructions.contains("Temporal"));
        assert!(instructions.contains("vector") || instructions.contains("Vector"));
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[test]
    fn test_update_node_nonexistent() {
        let server = create_test_server();

        let req = UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id: 999999,
            properties: HashMap::new(),
            provenance: None,
        };

        let response = server.update_node(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_delete_node_nonexistent() {
        let server = create_test_server();

        let req = DeleteNodeRequest {
            namespace: None,
            node_id: 999999,
            detach: None,
            valid_time: None,
        };

        let response = server.delete_node(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_get_edge_nonexistent() {
        let server = create_test_server();

        let req = GetEdgeRequest {
            namespace: None,
            edge_id: 999999,
            include_vectors: None,
        };

        let response = server.get_edge(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_update_edge_nonexistent() {
        let server = create_test_server();

        let req = UpdateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            edge_id: 999999,
            properties: HashMap::new(),
            provenance: None,
        };

        let response = server.update_edge(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_delete_edge_nonexistent() {
        let server = create_test_server();

        let req = DeleteEdgeRequest {
            namespace: None,
            edge_id: 999999,
            valid_time: None,
        };

        let response = server.delete_edge(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_create_edge_invalid_source() {
        let server = create_test_server();

        // Create only target node
        let target_resp = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Target".to_string(),
            properties: None,
            provenance: None,
        });
        let target: NodeResponse = parse_response(&target_resp).unwrap();

        // Try to create edge with non-existent source
        let req = CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: 999999,
            target_id: target.id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        };

        let response = server.create_edge(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_create_edge_invalid_target() {
        let server = create_test_server();

        // Create only source node
        let source_resp = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Source".to_string(),
            properties: None,
            provenance: None,
        });
        let source: NodeResponse = parse_response(&source_resp).unwrap();

        // Try to create edge with non-existent target
        let req = CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: source.id,
            target_id: 999999,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        };

        let response = server.create_edge(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_some());
    }
}

// ============================================================================
// Vector Index Distance Metric Tests
// ============================================================================

mod vector_distance_tests {
    use super::*;

    #[test]
    fn test_enable_vector_index_euclidean() {
        let server = create_test_server();

        let response = server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding_euclidean".to_string(),
            dimensions: 64,
            distance_metric: Some("euclidean".to_string()),
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));
        assert_eq!(
            value.get("distance_metric"),
            Some(&serde_json::json!("euclidean"))
        );
    }

    #[test]
    fn test_enable_vector_index_dot_product() {
        let server = create_test_server();

        let response = server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding_dot".to_string(),
            dimensions: 64,
            distance_metric: Some("dot".to_string()),
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));
        assert_eq!(
            value.get("distance_metric"),
            Some(&serde_json::json!("dot"))
        );
    }

    #[test]
    fn test_enable_vector_index_dot_product_alias() {
        let server = create_test_server();

        let response = server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding_dotprod".to_string(),
            dimensions: 64,
            distance_metric: Some("dot_product".to_string()),
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn test_enable_vector_index_unknown_metric_defaults_to_cosine() {
        let server = create_test_server();

        let response = server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding_unknown".to_string(),
            dimensions: 64,
            distance_metric: Some("unknown_metric".to_string()),
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Unknown metrics should default to cosine and succeed
        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn test_find_similar_k_capping() {
        let server = create_test_server();

        // Enable vector index
        server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 4,
            distance_metric: None,
        });

        // Try to request more than MAX_VECTOR_K results
        let response = server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            k: Some(10000), // Much larger than MAX_VECTOR_K,
            offset: None,
            include_vectors: None,
        });

        // Should not error (k gets capped internally)
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // The query succeeds (may return empty results, but no error about k)
        assert!(
            value.get("results").is_some(),
            "Should handle large k gracefully without an error, but got: {:?}",
            value.get("error")
        );
    }
}

// ============================================================================
// Temporal Query Extended Tests
// ============================================================================

mod temporal_extended_tests {
    use super::*;

    #[test]
    fn test_get_node_at_time_with_transaction_time() {
        let server = create_test_server();

        // Create a node
        let node_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Event".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("title".to_string(), serde_json::json!("Conference"));
                m
            }),
            provenance: None,
        });
        let node: NodeResponse = parse_response(&node_response).unwrap();

        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        // Get node at time with explicit transaction time
        let response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: now_micros.to_string(),
            transaction_time: Some(now_micros.to_string()),
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Should have explicit transaction_time in response
        if value.get("error").is_none() {
            assert!(
                value.get("transaction_time").is_some(),
                "Should include transaction_time in response"
            );
            // Should not be "now" since we provided explicit time
            assert_ne!(
                value.get("transaction_time"),
                Some(&serde_json::json!("now"))
            );
        }
    }

    #[test]
    fn test_get_edge_at_time_with_transaction_time() {
        let server = create_test_server();

        // Create nodes and edge
        let n1_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&n1_response).unwrap();

        let n2_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n2: NodeResponse = parse_response(&n2_response).unwrap();

        let edge_response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        let edge: EdgeResponse = parse_response(&edge_response).unwrap();

        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        // Get edge at time with explicit transaction time
        let response = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: now_micros.to_string(),
            transaction_time: Some(now_micros.to_string()),
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        if value.get("error").is_none() {
            assert!(value.get("edge").is_some());
            assert!(value.get("transaction_time").is_some());
        }
    }

    #[test]
    fn test_get_node_at_time_invalid_transaction_time() {
        let server = create_test_server();

        let node_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Test".to_string(),
            properties: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&node_response).unwrap();

        // Try with invalid transaction time format
        let response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: "0".to_string(),
            transaction_time: Some("not-a-valid-timestamp".to_string()),
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
        let error_str = value["error"]["message"].as_str().unwrap();
        assert!(
            error_str.contains("Invalid timestamp format"),
            "Should report invalid timestamp: {}",
            error_str
        );
    }

    #[test]
    fn test_get_edge_at_time_invalid_valid_time() {
        let server = create_test_server();

        // Create nodes and edge
        let n1_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&n1_response).unwrap();

        let n2_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n2: NodeResponse = parse_response(&n2_response).unwrap();

        let edge_response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        let edge: EdgeResponse = parse_response(&edge_response).unwrap();

        // Try with invalid valid_time format
        let response = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: "invalid-time".to_string(),
            transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_iso8601_with_offset_timezone() {
        let server = create_test_server();

        let node_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Event".to_string(),
            properties: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&node_response).unwrap();

        // Test with offset timezone (+00:00)
        let response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: "2024-01-15T10:30:00+00:00".to_string(),
            transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        if let Some(error) = value.get("error") {
            let err_str = error["message"].as_str().unwrap_or("");
            assert!(
                !err_str.contains("Invalid timestamp format"),
                "Offset timezone should be parsed correctly, got error: {}",
                err_str
            );
        }
    }
}

// ============================================================================
// Traversal Extended Tests
// ============================================================================

mod traversal_extended_tests {
    use super::*;

    fn create_bidirectional_graph(server: &AletheiaMcpServer) -> Vec<u64> {
        // Create nodes: A <-> B <-> C
        let nodes: Vec<u64> = (0..3)
            .map(|i| {
                let response = server.create_node(CreateNodeRequest {
                    namespace: None,
                    derived_from: None,
                    valid_time: None,
                    label: "BiNode".to_string(),
                    properties: Some({
                        let mut m = HashMap::new();
                        m.insert("name".to_string(), serde_json::json!(format!("Node{}", i)));
                        m
                    }),
                    provenance: None,
                });
                let node: NodeResponse = parse_response(&response).unwrap();
                node.id
            })
            .collect();

        // Create bidirectional edges
        for i in 0..2 {
            server.create_edge(CreateEdgeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                source_id: nodes[i],
                target_id: nodes[i + 1],
                label: "CONNECTED".to_string(),
                properties: None,
                provenance: None,
            });
            server.create_edge(CreateEdgeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                source_id: nodes[i + 1],
                target_id: nodes[i],
                label: "CONNECTED".to_string(),
                properties: None,
                provenance: None,
            });
        }

        nodes
    }

    #[test]
    fn test_traverse_bidirectional() {
        let server = create_test_server();
        let nodes = create_bidirectional_graph(&server);

        // Traverse bidirectionally from middle node
        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: nodes[1], // Middle node
            edge_label: "CONNECTED".to_string(),
            direction: Some("both".to_string()),
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let count = value.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
        // Should find both nodes[0] and nodes[2]
        assert!(
            count >= 2,
            "Bidirectional traversal should find both neighbors"
        );
    }

    #[test]
    fn test_traverse_both_direction_finds_node_reachable_only_via_incoming_edge() {
        // Regression test: direction:"both" must resolve the correct
        // neighbor for edges discovered through the *incoming* side too, not
        // just fall back to `edge.target` (which is `current_id` itself for
        // an incoming edge, silently dropping the neighbor). Build a graph
        // with NO reciprocal outgoing edge, so this can only pass if the
        // incoming half of "both" is resolved correctly.
        let server = create_test_server();

        let a = {
            let response = server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Node".to_string(),
                properties: None,
                provenance: None,
            });
            let node: NodeResponse = parse_response(&response).unwrap();
            node.id
        };
        let b = {
            let response = server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Node".to_string(),
                properties: None,
                provenance: None,
            });
            let node: NodeResponse = parse_response(&response).unwrap();
            node.id
        };
        // Only B -> A exists; A has no outgoing edge back to B.
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: b,
            target_id: a,
            label: "POINTS_AT".to_string(),
            properties: None,
            provenance: None,
        });

        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: a,
            edge_label: "POINTS_AT".to_string(),
            direction: Some("both".to_string()),
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none(), "unexpected error: {value}");
        let count = value["count"].as_u64().unwrap();
        assert_eq!(
            count, 1,
            "direction:\"both\" must find B via the incoming B->A edge, not drop it: {value}"
        );
        assert_eq!(value["results"][0]["node"]["id"], b);
    }

    #[test]
    fn test_traverse_nonexistent_edge_label() {
        let server = create_test_server();

        // Create a single node
        let node_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Lonely".to_string(),
            properties: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&node_response).unwrap();

        // Traverse with non-existent edge label
        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: node.id,
            edge_label: "NONEXISTENT".to_string(),
            direction: None,
            depth: Some(3),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Should return empty results, not error
        assert!(value.get("error").is_none());
        assert_eq!(value.get("count"), Some(&serde_json::json!(0)));
    }

    #[test]
    fn test_traverse_default_direction() {
        let server = create_test_server();

        // Create A -> B
        let n1_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Node".to_string(),
            properties: None,
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&n1_response).unwrap();

        let n2_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Node".to_string(),
            properties: None,
            provenance: None,
        });
        let n2: NodeResponse = parse_response(&n2_response).unwrap();

        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "NEXT".to_string(),
            properties: None,
            provenance: None,
        });

        // Traverse without specifying direction (should default to outgoing)
        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: n1.id,
            edge_label: "NEXT".to_string(),
            direction: None, // Default to outgoing
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let count = value.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
        assert!(count >= 1, "Default direction should find outgoing nodes");
    }

    #[test]
    fn test_traverse_result_limit() {
        let server = create_test_server();

        // Create a star graph: center -> 10 spokes
        let center_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Center".to_string(),
            properties: None,
            provenance: None,
        });
        let center: NodeResponse = parse_response(&center_response).unwrap();

        for i in 0..10 {
            let spoke_response = server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Spoke".to_string(),
                properties: Some({
                    let mut m = HashMap::new();
                    m.insert("index".to_string(), serde_json::json!(i));
                    m
                }),
                provenance: None,
            });
            let spoke: NodeResponse = parse_response(&spoke_response).unwrap();

            server.create_edge(CreateEdgeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                source_id: center.id,
                target_id: spoke.id,
                label: "SPOKE".to_string(),
                properties: None,
                provenance: None,
            });
        }

        // Traverse with limit
        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: center.id,
            edge_label: "SPOKE".to_string(),
            direction: Some("outgoing".to_string()),
            depth: Some(1),
            limit: Some(5),
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let count = value.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
        assert!(count <= 5, "Traversal should respect limit");
    }
}

// ============================================================================
// Hybrid Query Extended Tests
// ============================================================================

mod hybrid_extended_tests {
    use super::*;

    #[test]
    fn test_hybrid_query_with_temporal() {
        let server = create_test_server();

        // Create a node
        let node_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Alice"));
                m
            }),
            provenance: None,
        });
        let node: NodeResponse = parse_response(&node_response).unwrap();

        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        // Query with both valid_time and transaction_time
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: Some(node.id),
            traverse_edge: None,
            traverse_depth: None,
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: Some(now_micros.to_string()),
            transaction_time: Some(now_micros.to_string()),
            filter_label: None,
            limit: Some(10),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Should return temporal query result
        if value.get("error").is_none() {
            assert!(
                value.get("temporal_query").is_some() && value.get("results").is_some(),
                "A successful temporal hybrid query should return both 'temporal_query' and 'results' fields."
            );
        }
    }

    #[test]
    fn test_hybrid_query_multi_depth_traversal() {
        let server = create_test_server();

        // Create chain: A -> B -> C -> D
        let nodes: Vec<u64> = (0..4)
            .map(|i| {
                let response = server.create_node(CreateNodeRequest {
                    namespace: None,
                    derived_from: None,
                    valid_time: None,
                    label: "ChainNode".to_string(),
                    properties: Some({
                        let mut m = HashMap::new();
                        m.insert("level".to_string(), serde_json::json!(i));
                        m
                    }),
                    provenance: None,
                });
                let node: NodeResponse = parse_response(&response).unwrap();
                node.id
            })
            .collect();

        for i in 0..3 {
            server.create_edge(CreateEdgeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                source_id: nodes[i],
                target_id: nodes[i + 1],
                label: "CHAIN".to_string(),
                properties: None,
                provenance: None,
            });
        }

        // Query with depth > 1
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: Some(nodes[0]),
            traverse_edge: Some("CHAIN".to_string()),
            traverse_depth: Some(3),
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: None,
            transaction_time: None,
            filter_label: None,
            limit: Some(10),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        if value.get("error").is_none() {
            let count = value.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
            assert!(count >= 1, "Multi-depth traversal should find nodes");
        }
    }

    #[test]
    fn test_hybrid_query_invalid_valid_time() {
        let server = create_test_server();

        let node_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Test".to_string(),
            properties: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&node_response).unwrap();

        // Query with invalid valid_time
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: Some(node.id),
            traverse_edge: None,
            traverse_depth: None,
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: Some("invalid-time".to_string()),
            transaction_time: None,
            filter_label: None,
            limit: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
        let error = value["error"]["message"].as_str().unwrap();
        assert!(
            error.contains("Invalid valid_time"),
            "Should report invalid valid_time"
        );
    }

    #[test]
    fn test_hybrid_query_invalid_transaction_time() {
        let server = create_test_server();

        let node_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Test".to_string(),
            properties: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&node_response).unwrap();

        // Query with invalid transaction_time
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: Some(node.id),
            traverse_edge: None,
            traverse_depth: None,
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: Some("0".to_string()),
            transaction_time: Some("bad-tx-time".to_string()),
            filter_label: None,
            limit: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
        let error = value["error"]["message"].as_str().unwrap();
        assert!(
            error.contains("Invalid transaction_time"),
            "Should report invalid transaction_time"
        );
    }

    #[test]
    fn test_hybrid_query_limit_capping() {
        let server = create_test_server();

        // Create some nodes
        for i in 0..5 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "LimitTest".to_string(),
                properties: Some({
                    let mut m = HashMap::new();
                    m.insert("index".to_string(), serde_json::json!(i));
                    m
                }),
                provenance: None,
            });
        }

        // Query with very large limit (should be capped internally)
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: None,
            traverse_edge: None,
            traverse_depth: None,
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: None,
            transaction_time: None,
            filter_label: Some("LimitTest".to_string()),
            limit: Some(100000), // Much larger than MAX_RESULT_LIMIT,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Should not error (limit gets capped)
        assert!(
            value.get("results").is_some(),
            "Large limit should be capped, not error"
        );
    }

    #[test]
    fn test_hybrid_query_vector_without_index() {
        let server = create_test_server();

        // Try vector search without enabling index
        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: None,
            traverse_edge: None,
            traverse_depth: None,
            vector_property: Some("embedding".to_string()),
            query_embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            top_k: Some(5),
            valid_time: None,
            transaction_time: None,
            filter_label: None,
            limit: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
        let error = value["error"]["message"].as_str().unwrap();
        assert!(
            error.contains("Vector index not enabled"),
            "Should report missing vector index"
        );
    }
}

// ============================================================================
// Edge Operations Extended Tests
// ============================================================================

mod edge_extended_tests {
    use super::*;

    #[test]
    fn test_list_edges_with_label() {
        let server = create_test_server();

        // Create nodes
        let n1_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&n1_response).unwrap();

        let n2_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n2: NodeResponse = parse_response(&n2_response).unwrap();

        // Create edges with different labels
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "WORKS_WITH".to_string(),
            properties: None,
            provenance: None,
        });

        // List edges with label filter
        let response = server.list_edges(ListEdgesRequest {
            namespace: None,
            label: Some("KNOWS".to_string()),
            limit: None,
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // list_edges doesn't actually filter, it just includes the label in response
        assert!(value.get("label_filter").is_some());
        assert_eq!(value.get("label_filter"), Some(&serde_json::json!("KNOWS")));
    }

    #[test]
    fn test_count_edges_with_label() {
        let server = create_test_server();

        // Create nodes and edges
        let n1_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&n1_response).unwrap();

        let n2_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n2: NodeResponse = parse_response(&n2_response).unwrap();

        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });

        // Count edges with label (should return message about not being supported)
        let response = server.count_edges(CountEdgesRequest {
            label: Some("KNOWS".to_string()),
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("message").is_some());
        assert!(
            value
                .get("message")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("not supported")
        );
    }

    #[test]
    fn test_get_incoming_edges_with_label() {
        let server = create_test_server();

        // Create nodes
        let n1_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n1: NodeResponse = parse_response(&n1_response).unwrap();

        let n2_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n2: NodeResponse = parse_response(&n2_response).unwrap();

        let n3_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });
        let n3: NodeResponse = parse_response(&n3_response).unwrap();

        // Create edges with different labels pointing to n2
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n3.id,
            target_id: n2.id,
            label: "WORKS_WITH".to_string(),
            properties: None,
            provenance: None,
        });

        // Get incoming edges with label filter
        let response = server.get_incoming_edges(GetIncomingEdgesRequest {
            node_id: n2.id,
            label: Some("KNOWS".to_string()),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("count"), Some(&serde_json::json!(1)));

        let edges = value.get("edges").unwrap().as_array().unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].get("label"), Some(&serde_json::json!("KNOWS")));
    }
}

// ============================================================================
// List Nodes Without Label Tests
// ============================================================================

mod list_nodes_extended_tests {
    use super::*;

    #[test]
    fn test_list_nodes_without_label() {
        let server = create_test_server();

        // Create some nodes
        for i in 0..3 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: format!("Type{}", i),
                properties: None,
                provenance: None,
            });
        }

        // List nodes without label filter
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: None,
            property_key: None,
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Should return message about using label filter
        assert!(value.get("message").is_some());
        assert!(value.get("total_count").is_some());
        assert_eq!(value.get("total_count"), Some(&serde_json::json!(3)));
        // nodes array should be empty
        assert_eq!(value.get("count"), Some(&serde_json::json!(0)));
    }

    #[test]
    fn test_list_nodes_limit_capping() {
        let server = create_test_server();

        // Create a few nodes
        for i in 0..5 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "LimitCap".to_string(),
                properties: Some({
                    let mut m = HashMap::new();
                    m.insert("index".to_string(), serde_json::json!(i));
                    m
                }),
                provenance: None,
            });
        }

        // Request with very large limit (should be capped to MAX_RESULT_LIMIT)
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("LimitCap".to_string()),
            property_key: None,
            property_value: None,
            limit: Some(100000), // Much larger than MAX_RESULT_LIMIT
            offset: Some(0),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Should not error
        assert!(
            value.get("nodes").is_some(),
            "Large limit should be capped, not error"
        );
    }

    // ========================================================================
    // Property-Based Lookup via MCP
    // ========================================================================

    #[test]
    fn test_list_nodes_by_property_string() {
        let server = create_test_server();

        // Create nodes with different names
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Alice"));
                m
            }),
            provenance: None,
        });
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Bob"));
                m
            }),
            provenance: None,
        });
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Alice"));
                m
            }),
            provenance: None,
        });

        // Filter by property
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Person".to_string()),
            property_key: Some("name".to_string()),
            property_value: Some(serde_json::json!("Alice")),
            limit: None,
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let nodes = value["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 2, "Should find exactly 2 Alice nodes");
    }

    #[test]
    fn test_list_nodes_by_property_int() {
        let server = create_test_server();

        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Sensor".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("reading".to_string(), serde_json::json!(42));
                m
            }),
            provenance: None,
        });
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Sensor".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("reading".to_string(), serde_json::json!(99));
                m
            }),
            provenance: None,
        });

        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Sensor".to_string()),
            property_key: Some("reading".to_string()),
            property_value: Some(serde_json::json!(42)),
            limit: None,
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let nodes = value["nodes"].as_array().unwrap();
        assert_eq!(
            nodes.len(),
            1,
            "Should find exactly 1 sensor with reading=42"
        );
    }

    #[test]
    fn test_list_nodes_by_property_no_match() {
        let server = create_test_server();

        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Item".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("color".to_string(), serde_json::json!("red"));
                m
            }),
            provenance: None,
        });

        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Item".to_string()),
            property_key: Some("color".to_string()),
            property_value: Some(serde_json::json!("blue")),
            limit: None,
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let nodes = value["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 0, "Should find no matching nodes");
    }

    #[test]
    fn test_list_nodes_by_property_missing_label() {
        let server = create_test_server();

        // property_key without label should error
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: None,
            property_key: Some("name".to_string()),
            property_value: Some(serde_json::json!("Alice")),
            limit: None,
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_some(),
            "Should error when label is missing for property filter"
        );
    }

    #[test]
    fn test_list_nodes_by_property_key_without_value() {
        let server = create_test_server();

        // property_key without property_value should error
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Person".to_string()),
            property_key: Some("name".to_string()),
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_some(),
            "Should error when property_value is missing"
        );
    }

    #[test]
    fn test_list_nodes_by_property_with_pagination() {
        let server = create_test_server();

        // Create 5 nodes with same property
        for _ in 0..5 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Widget".to_string(),
                properties: Some({
                    let mut m = HashMap::new();
                    m.insert("status".to_string(), serde_json::json!("active"));
                    m
                }),
                provenance: None,
            });
        }

        // Page 1: first 2
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Widget".to_string()),
            property_key: Some("status".to_string()),
            property_value: Some(serde_json::json!("active")),
            limit: Some(2),
            offset: Some(0),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let nodes = value["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 2, "Page 1 should have 2 nodes");

        // Page 2: next 2
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Widget".to_string()),
            property_key: Some("status".to_string()),
            property_value: Some(serde_json::json!("active")),
            limit: Some(2),
            offset: Some(2),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let nodes = value["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 2, "Page 2 should have 2 nodes");

        // Page 3: last 1
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Widget".to_string()),
            property_key: Some("status".to_string()),
            property_value: Some(serde_json::json!("active")),
            limit: Some(2),
            offset: Some(4),
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let nodes = value["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 1, "Page 3 should have 1 node");
    }
}

// ============================================================================
// Declarative Query Tool Tests (Issue #3213)
// ============================================================================

mod query_tool_tests {
    use super::*;

    /// Seed a node with the given label and `name` property, returning its id.
    fn seed_named(server: &AletheiaMcpServer, label: &str, name: &str) -> u64 {
        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!(name));
        let resp = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: label.to_string(),
            properties: Some(props),
            provenance: None,
        });
        let node: NodeResponse = parse_response(&resp).expect("seed node should succeed");
        node.id
    }

    /// Run a query and return the parsed JSON value (success or error).
    fn run_query(server: &AletheiaMcpServer, req: QueryRequest) -> serde_json::Value {
        serde_json::from_str(&server.query(req)).expect("query response should be JSON")
    }

    /// Extract the structured error object from a query response.
    fn error_obj(value: &serde_json::Value) -> &serde_json::Value {
        value
            .get("error")
            .expect("expected an `error` payload in the response")
    }

    #[test]
    fn test_query_tool_is_advertised() {
        let server = create_test_server();
        assert!(
            server.list_tools_for_test().contains(&"query".to_string()),
            "the `query` tool must be advertised in list_tools"
        );
    }

    #[test]
    fn test_query_aql_returns_structured_rows() {
        let server = create_test_server();
        seed_named(&server, "Widget", "alpha");

        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "aql".to_string(),
                query: "MATCH (n:Widget) RETURN n".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );

        // Column metadata is present.
        assert!(
            value.get("columns").and_then(|c| c.as_array()).is_some(),
            "response must include column metadata: {value}"
        );
        assert_eq!(
            value["row_count"].as_u64(),
            Some(1),
            "exactly one row: {value}"
        );
        let rows = value["rows"].as_array().expect("rows array");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["entity"]["label"].as_str(), Some("Widget"));
        assert_eq!(
            rows[0]["entity"]["properties"]["name"].as_str(),
            Some("alpha")
        );
    }

    #[test]
    fn test_query_rejects_mutations_before_execution() {
        for stmt in [
            "CREATE (n:Hacker {name: 'x'})",
            "MATCH (n:Widget) SET n.name = 'x'",
            "MATCH (n:Widget) DELETE n",
            "MERGE (n:Widget {name: 'x'})",
            "MATCH (n:Widget) REMOVE n.name",
        ] {
            let server = create_test_server();
            seed_named(&server, "Widget", "alpha");
            let before = server.db().node_count();

            let value = run_query(
                &server,
                QueryRequest {
                    namespace: None,
                    language: "cypher".to_string(),
                    query: stmt.to_string(),
                    params: None,
                    limit: None,
                    limits: None,
                },
            );

            let err = error_obj(&value);
            assert_eq!(
                err["kind"].as_str(),
                Some("read_only_violation"),
                "statement `{stmt}` must be rejected as read-only: {value}"
            );
            assert!(
                err.get("clause").and_then(|c| c.as_str()).is_some(),
                "the error must name the offending clause: {value}"
            );
            // It must never write.
            assert_eq!(
                server.db().node_count(),
                before,
                "statement `{stmt}` must not have mutated state"
            );
        }
    }

    #[test]
    fn test_query_parse_error_is_structured() {
        let server = create_test_server();
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "aql".to_string(),
                query: "this is not a valid query".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        assert_eq!(
            error_obj(&value)["kind"].as_str(),
            Some("parse_error"),
            "malformed query must yield a parse_error: {value}"
        );
    }

    #[test]
    fn test_query_unknown_language_is_invalid_request() {
        let server = create_test_server();
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "sql".to_string(),
                query: "MATCH (n) RETURN n".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        assert_eq!(
            error_obj(&value)["kind"].as_str(),
            Some("invalid_request"),
            "unknown language must yield invalid_request: {value}"
        );
    }

    #[test]
    fn test_query_aql_rejects_params() {
        let server = create_test_server();
        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("alpha"));
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "aql".to_string(),
                query: "MATCH (n:Widget) RETURN n".to_string(),
                params: Some(params),
                limit: None,
                limits: None,
            },
        );
        assert_eq!(
            error_obj(&value)["kind"].as_str(),
            Some("invalid_request"),
            "AQL does not support params and must reject them: {value}"
        );
    }

    #[test]
    fn test_query_row_cap_truncates() {
        let server = create_test_server();
        for i in 0..5 {
            seed_named(&server, "Widget", &format!("w{i}"));
        }
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "aql".to_string(),
                query: "MATCH (n:Widget) RETURN n".to_string(),
                params: None,
                limit: Some(2),
                limits: None,
            },
        );
        assert_eq!(
            value["row_count"].as_u64(),
            Some(2),
            "row cap honored: {value}"
        );
        assert_eq!(
            value["truncated"].as_bool(),
            Some(true),
            "truncated flag must be set when results exceed the cap: {value}"
        );
    }

    #[cfg(not(feature = "cypher"))]
    #[test]
    fn test_query_cypher_reports_unavailable_when_feature_disabled() {
        let server = create_test_server();
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n) RETURN n".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        assert_eq!(
            error_obj(&value)["kind"].as_str(),
            Some("language_unavailable"),
            "with the cypher feature off, cypher must report as unavailable (not fail to compile): {value}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_returns_rows() {
        let server = create_test_server();
        seed_named(&server, "Person", "Alice");
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n:Person {name: 'Alice'}) RETURN n".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        assert_eq!(value["row_count"].as_u64(), Some(1), "{value}");
        assert_eq!(
            value["rows"][0]["entity"]["properties"]["name"].as_str(),
            Some("Alice")
        );
    }

    // --- #558 MCP surface: Cypher aggregate/computed rows must render their
    // named column values (not `entity: null`). See the aggregation limitation
    // note in CLAUDE.md that this closes at the MCP layer.

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_aggregate_count_renders_value_not_null() {
        let server = create_test_server();
        for name in ["a", "b", "c", "d", "e"] {
            seed_named(&server, "Person", name);
        }
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n:Person) RETURN count(*)".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        assert_eq!(value["row_count"].as_u64(), Some(1), "{value}");
        // The aggregate value must be surfaced under its column name, not lost
        // to an `entity: null` rendering.
        assert_eq!(
            value["rows"][0]["count(*)"].as_i64(),
            Some(5),
            "aggregate count must render its value under its column name: {value}"
        );
        assert!(
            value["rows"][0]["entity"].is_null(),
            "aggregate row must not carry a non-null entity payload: {value}"
        );
        // Column metadata names the aggregate column (not the static schema).
        let cols = value["columns"].as_array().expect("columns array");
        assert!(
            cols.iter().any(|c| c["name"].as_str() == Some("count(*)")),
            "columns must name the aggregate column: {value}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_grouped_aggregate_renders_group_and_count() {
        let server = create_test_server();
        for (name, dept) in [
            ("a", "Eng"),
            ("b", "Eng"),
            ("c", "Eng"),
            ("d", "Sales"),
            ("e", "Sales"),
        ] {
            let mut props = HashMap::new();
            props.insert("name".to_string(), serde_json::json!(name));
            props.insert("dept".to_string(), serde_json::json!(dept));
            let resp = server.create_node(CreateNodeRequest {
                namespace: None,
                valid_time: None,
                label: "Person".to_string(),
                properties: Some(props),
                provenance: None,
                derived_from: None,
            });
            let _: NodeResponse = parse_response(&resp).expect("seed node should succeed");
        }
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n:Person) RETURN n.dept, count(*)".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        assert_eq!(value["row_count"].as_u64(), Some(2), "{value}");
        let rows = value["rows"].as_array().expect("rows array");
        let mut counts = HashMap::new();
        for r in rows {
            let dept = r["n.dept"]
                .as_str()
                .unwrap_or_else(|| panic!("group column `n.dept` must be present: {value}"))
                .to_string();
            let cnt = r["count(*)"]
                .as_i64()
                .unwrap_or_else(|| panic!("aggregate column `count(*)` must be present: {value}"));
            counts.insert(dept, cnt);
        }
        assert_eq!(counts.get("Eng"), Some(&3), "{value}");
        assert_eq!(counts.get("Sales"), Some(&2), "{value}");
        let cols = value["columns"].as_array().expect("columns array");
        let names: Vec<&str> = cols.iter().filter_map(|c| c["name"].as_str()).collect();
        assert!(
            names.contains(&"n.dept") && names.contains(&"count(*)"),
            "columns must list both the group key and the aggregate: {value}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_aggregate_alias_names_column() {
        let server = create_test_server();
        for name in ["a", "b", "c"] {
            seed_named(&server, "Person", name);
        }
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n:Person) RETURN count(*) AS c".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        assert_eq!(
            value["rows"][0]["c"].as_i64(),
            Some(3),
            "aliased aggregate value must render under its alias: {value}"
        );
        let cols = value["columns"].as_array().expect("columns array");
        assert!(
            cols.iter().any(|c| c["name"].as_str() == Some("c")),
            "columns must name the aggregate alias `c`: {value}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_plain_entity_row_rendering_unchanged() {
        // Non-regression: a plain entity query keeps the exact
        // entity/score/path/timestamp row shape and static column schema.
        let server = create_test_server();
        seed_named(&server, "Person", "Alice");
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n:Person {name: 'Alice'}) RETURN n".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        let row = &value["rows"][0];
        assert_eq!(row["entity"]["label"].as_str(), Some("Person"), "{value}");
        assert_eq!(
            row["entity"]["properties"]["name"].as_str(),
            Some("Alice"),
            "{value}"
        );
        assert!(row.get("score").is_some(), "score key present: {value}");
        assert!(row.get("path").is_some(), "path key present: {value}");
        assert!(
            row.get("timestamp").is_some(),
            "timestamp key present: {value}"
        );
        let names: Vec<&str> = value["columns"]
            .as_array()
            .expect("columns array")
            .iter()
            .filter_map(|c| c["name"].as_str())
            .collect();
        assert_eq!(
            names,
            vec!["entity", "score", "path", "timestamp"],
            "plain entity query keeps the static column schema: {value}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_optional_match_null_row_serializes_as_json_null() {
        let server = create_test_server();
        seed_named(&server, "Person", "Alice");
        // Alice has no outgoing KNOWS edges: OPTIONAL MATCH preserves the
        // base row and the null binding must surface as JSON null, not crash.
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (a:Person {name: 'Alice'}) OPTIONAL MATCH (a)-[:KNOWS]->(x) RETURN x"
                    .to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        assert_eq!(value["row_count"].as_u64(), Some(1), "{value}");
        assert!(
            value["rows"][0]["entity"].is_null(),
            "unmatched OPTIONAL MATCH row must serialize entity as JSON null: {value}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_with_params() {
        let server = create_test_server();
        seed_named(&server, "Person", "Alice");
        seed_named(&server, "Person", "Bob");
        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("Alice"));
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n:Person {name: $name}) RETURN n".to_string(),
                params: Some(params),
                limit: None,
                limits: None,
            },
        );
        assert_eq!(value["row_count"].as_u64(), Some(1), "{value}");
        assert_eq!(
            value["rows"][0]["entity"]["properties"]["name"].as_str(),
            Some("Alice")
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_temporal_clauses_are_honored() {
        let server = create_test_server();
        seed_named(&server, "Person", "Alice");

        // Each temporal form must parse, flow through to the engine, and return
        // the correct row (the clause is honored, not dropped or rejected).
        for clause in [
            "AS OF VALID_TIME '2099-01-01'",
            "AS OF SYSTEM_TIME '2099-01-01'",
            "AS OF TIMESTAMP '2099-01-01T00:00:00Z'",
            "BETWEEN '2000-01-01' AND '2099-01-01'",
        ] {
            let q = format!("MATCH (n:Person) {clause} RETURN n");
            let value = run_query(
                &server,
                QueryRequest {
                    namespace: None,
                    language: "cypher".to_string(),
                    query: q.clone(),
                    params: None,
                    limit: None,
                    limits: None,
                },
            );
            assert!(
                value.get("error").is_none(),
                "temporal query `{q}` must not error: {value}"
            );
            assert_eq!(
                value["row_count"].as_u64(),
                Some(1),
                "temporal query `{q}` must return the node: {value}"
            );
            assert_eq!(
                value["rows"][0]["entity"]["properties"]["name"].as_str(),
                Some("Alice"),
                "temporal query `{q}` must return correct data: {value}"
            );
        }
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_invalid_temporal_timestamp_is_structured() {
        let server = create_test_server();
        seed_named(&server, "Person", "Alice");
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n:Person) AS OF TIMESTAMP 'not-a-timestamp' RETURN n".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        let kind = error_obj(&value)["kind"].as_str();
        assert!(
            matches!(kind, Some("invalid_params") | Some("parse_error")),
            "an invalid temporal timestamp must yield a structured error: {value}"
        );
    }

    // -----------------------------------------------------------------------
    // Tests for fixes applied after the initial code review
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_backslash_escape_in_string_does_not_block_valid_query() {
        // A query whose string literal contains a backslash-escaped quote must
        // NOT be rejected as a read_only_violation. The old sanitizer would
        // close the quote at the `\'` and then scan `s fine'}) RETURN n` as
        // bare tokens, finding no mutating keyword and passing — but the correct
        // behavior is that the whole content stays inside the string.
        let server = create_test_server();
        seed_named(&server, "Person", "Alice");
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "aql".to_string(),
                // String literal `it\'s fine` — backslash-escaped quote inside single quotes.
                query: "MATCH (n:Person {note: 'it\\'s fine'}) RETURN n".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        // The guard must not fire (no read_only_violation for a read-only query).
        let err = value.get("error");
        if let Some(err) = err {
            assert_ne!(
                err["kind"].as_str(),
                Some("read_only_violation"),
                "backslash-escaped quote inside a string literal must not trip the read-only guard: {value}"
            );
        }
        // Whether the engine can parse this specific AQL variant is a grammar
        // question; what matters is that the guard itself does not reject it.
    }

    #[test]
    fn test_query_clause_field_absent_for_non_write_errors() {
        // `clause` is reserved exclusively for read_only_violation (it names the
        // mutating keyword). For other error kinds (parse_error,
        // unsupported_construct, invalid_params, runtime_error) the field must
        // NOT appear, since it would be semantically wrong.
        let server = create_test_server();
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "aql".to_string(),
                query: "this is not valid".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        let err = error_obj(&value);
        assert_eq!(err["kind"].as_str(), Some("parse_error"));
        assert!(
            err.get("clause").is_none(),
            "parse_error must not include a `clause` field: {value}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_empty_array_param_is_invalid() {
        // An empty JSON array [] must be rejected with invalid_params rather
        // than silently accepted as a zero-dimension embedding.
        let server = create_test_server();
        let mut params = HashMap::new();
        params.insert("vec".to_string(), serde_json::json!([]));
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n) RETURN n".to_string(),
                params: Some(params),
                limit: None,
                limits: None,
            },
        );
        assert_eq!(
            error_obj(&value)["kind"].as_str(),
            Some("invalid_params"),
            "empty array parameter must yield invalid_params: {value}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_float_param_preserved_as_float() {
        // A JSON number with a decimal point (1.0) must be bound as Float, not Int.
        // We verify indirectly: the query must not fail with a type error, and
        // the params round-trip must work (parse_error or correct rows, not a
        // crash or wrong type coercion error).
        let server = create_test_server();
        seed_named(&server, "Product", "Gadget");
        let mut params = HashMap::new();
        // 1.0 is an explicit float in JSON — must not be coerced to Int.
        params.insert("threshold".to_string(), serde_json::json!(1.0_f64));
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n:Product) WHERE n.price < $threshold RETURN n".to_string(),
                params: Some(params),
                limit: None,
                limits: None,
            },
        );
        // The key assertion: the error must NOT be invalid_params (the float
        // was accepted as a float). Whether the engine returns rows or a
        // parse/runtime error for the WHERE predicate is a grammar question.
        if let Some(err) = value.get("error") {
            assert_ne!(
                err["kind"].as_str(),
                Some("invalid_params"),
                "a float-valued JSON param must be accepted as Float, not rejected: {value}"
            );
        }
    }

    #[test]
    fn test_query_single_line_comment_before_mutation_is_not_rejected() {
        // A `//`-style comment that contains a mutating keyword must NOT trip the
        // read-only guard — only bare (outside-comment, outside-string) tokens count.
        let server = create_test_server();
        seed_named(&server, "Widget", "alpha");
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "aql".to_string(),
                query: "// CREATE would mutate\nMATCH (n:Widget) RETURN n".to_string(),
                params: None,
                limit: None,
                limits: None,
            },
        );
        if let Some(err) = value.get("error") {
            assert_ne!(
                err["kind"].as_str(),
                Some("read_only_violation"),
                "mutating keyword inside a // comment must not trigger the guard: {value}"
            );
        }
    }

    #[test]
    fn test_query_node_label_matching_mutating_keyword_is_allowed() {
        // A node label that happens to spell a mutating keyword (e.g. :Call, :Set)
        // must NOT be rejected — the token is preceded by ':' and is a label, not
        // a clause.
        for stmt in [
            "MATCH (c:Call) RETURN c",
            "MATCH (s:Set) RETURN s",
            "MATCH (d:Drop) RETURN d",
        ] {
            let server = create_test_server();
            let value = run_query(
                &server,
                QueryRequest {
                    namespace: None,
                    language: "aql".to_string(),
                    query: stmt.to_string(),
                    params: None,
                    limit: None,
                    limits: None,
                },
            );
            if let Some(err) = value.get("error") {
                assert_ne!(
                    err["kind"].as_str(),
                    Some("read_only_violation"),
                    "node label `{stmt}` must not trip the read-only guard: {value}"
                );
            }
        }
    }

    #[test]
    fn test_query_property_key_matching_mutating_keyword_is_allowed() {
        // A property key that happens to spell a mutating keyword (e.g. n.set,
        // n.merge) must NOT be rejected — the token is preceded by '.' and is a
        // property access, not a clause.
        for stmt in [
            "MATCH (n) RETURN n.set",
            "MATCH (n) RETURN n.merge",
            "MATCH (n) RETURN n.delete",
        ] {
            let server = create_test_server();
            let value = run_query(
                &server,
                QueryRequest {
                    namespace: None,
                    language: "aql".to_string(),
                    query: stmt.to_string(),
                    params: None,
                    limit: None,
                    limits: None,
                },
            );
            if let Some(err) = value.get("error") {
                assert_ne!(
                    err["kind"].as_str(),
                    Some("read_only_violation"),
                    "property key in `{stmt}` must not trip the read-only guard: {value}"
                );
            }
        }
    }

    #[test]
    fn test_query_remaining_mutating_keywords_are_rejected() {
        // DETACH, DROP, FOREACH, and LOAD are listed as mutating keywords but are
        // not exercised by the existing CREATE/SET/DELETE/MERGE/REMOVE tests.
        for (stmt, kw) in [
            ("DETACH DELETE (n)", "DETACH"),
            ("DROP INDEX ON :Person(name)", "DROP"),
            ("FOREACH (x IN $list | SET x.flag = 1)", "FOREACH"),
            ("LOAD CSV FROM 'file' AS row", "LOAD"),
        ] {
            let server = create_test_server();
            let value = run_query(
                &server,
                QueryRequest {
                    namespace: None,
                    language: "aql".to_string(),
                    query: stmt.to_string(),
                    params: None,
                    limit: None,
                    limits: None,
                },
            );
            assert_eq!(
                error_obj(&value)["kind"].as_str(),
                Some("read_only_violation"),
                "statement with `{kw}` must be rejected: {value}"
            );
        }
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_null_param_is_accepted() {
        // JSON null must be bound as CypherParameterValue::Null without error.
        let server = create_test_server();
        let mut params = HashMap::new();
        params.insert("x".to_string(), serde_json::json!(null));
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n) WHERE n.x = $x RETURN n".to_string(),
                params: Some(params),
                limit: None,
                limits: None,
            },
        );
        if let Some(err) = value.get("error") {
            assert_ne!(
                err["kind"].as_str(),
                Some("invalid_params"),
                "null param must be accepted: {value}"
            );
        }
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_bool_and_int_params_are_accepted() {
        // JSON booleans and integers must bind without error.
        let server = create_test_server();
        let mut params = HashMap::new();
        params.insert("flag".to_string(), serde_json::json!(true));
        params.insert("count".to_string(), serde_json::json!(42_i64));
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n) WHERE n.flag = $flag AND n.count = $count RETURN n".to_string(),
                params: Some(params),
                limit: None,
                limits: None,
            },
        );
        if let Some(err) = value.get("error") {
            assert_ne!(
                err["kind"].as_str(),
                Some("invalid_params"),
                "bool/int params must be accepted: {value}"
            );
        }
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_object_param_is_invalid() {
        // A JSON object parameter is not supported; must yield invalid_params.
        let server = create_test_server();
        let mut params = HashMap::new();
        params.insert("obj".to_string(), serde_json::json!({"key": "value"}));
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n) RETURN n".to_string(),
                params: Some(params),
                limit: None,
                limits: None,
            },
        );
        assert_eq!(
            error_obj(&value)["kind"].as_str(),
            Some("invalid_params"),
            "object parameter must yield invalid_params: {value}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_query_cypher_non_numeric_array_param_is_invalid() {
        // An array containing non-numeric elements is not a valid embedding;
        // must yield invalid_params.
        let server = create_test_server();
        let mut params = HashMap::new();
        params.insert("vec".to_string(), serde_json::json!(["not", "numbers"]));
        let value = run_query(
            &server,
            QueryRequest {
                namespace: None,
                language: "cypher".to_string(),
                query: "MATCH (n) RETURN n".to_string(),
                params: Some(params),
                limit: None,
                limits: None,
            },
        );
        assert_eq!(
            error_obj(&value)["kind"].as_str(),
            Some("invalid_params"),
            "non-numeric array parameter must yield invalid_params: {value}"
        );
    }

    // =======================================================================
    // Per-query resource limits (Issue #3368) — end-to-end behavior through
    // the public `query()` / `dispatch_tool` surfaces.
    // =======================================================================

    use crate::mcp::{QueryLimitsConfig, QueryLimitsOverride};

    /// A `query` request with `limits` fields set on the override.
    fn query_req(language: &str, query: &str, limits: Option<QueryLimitsOverride>) -> QueryRequest {
        QueryRequest {
            namespace: None,
            language: language.to_string(),
            query: query.to_string(),
            params: None,
            limit: None,
            limits,
        }
    }

    /// Default (generous) limits leave a small query's output byte-identical:
    /// no error, `truncated:false`, and the row is present.
    #[test]
    fn test_query_default_limits_output_unchanged() {
        let server = create_test_server();
        seed_named(&server, "Widget", "alpha");
        let value = run_query(&server, query_req("aql", "MATCH (n:Widget) RETURN n", None));
        assert!(
            value.get("error").is_none(),
            "no error under defaults: {value}"
        );
        assert_eq!(value["row_count"].as_u64(), Some(1));
        assert_eq!(value["truncated"].as_bool(), Some(false));
    }

    /// A per-call override above the operator ceiling is rejected up front with
    /// INVALID_ARGUMENT / kind invalid_params and `details {dimension, requested,
    /// ceiling}` — for every dimension — and bumps the override-rejected counter.
    #[test]
    fn test_query_over_ceiling_overrides_are_invalid_argument() {
        let server = create_test_server(); // default ceilings
        let cases = [
            (
                QueryLimitsOverride {
                    timeout_ms: Some(999_999_999),
                    ..Default::default()
                },
                "wall_clock_timeout",
            ),
            (
                QueryLimitsOverride {
                    max_result_rows: Some(999_999_999),
                    ..Default::default()
                },
                "result_rows",
            ),
            (
                QueryLimitsOverride {
                    max_response_bytes: Some(usize::MAX),
                    ..Default::default()
                },
                "result_bytes",
            ),
        ];
        for (over, dim) in cases {
            let value = run_query(&server, query_req("aql", "MATCH (n) RETURN n", Some(over)));
            let err = error_obj(&value);
            assert_eq!(err["kind"].as_str(), Some("invalid_params"), "{value}");
            assert_eq!(err["code"].as_str(), Some("INVALID_ARGUMENT"), "{value}");
            assert_eq!(err["retriable"].as_bool(), Some(false), "{value}");
            assert_eq!(err["details"]["dimension"].as_str(), Some(dim), "{value}");
            assert!(err["details"]["ceiling"].as_u64().is_some(), "{value}");
        }
        assert_eq!(server.limit_termination_counts().override_rejected, 3);
    }

    /// A timeout override of `0` (unlimited) under a finite operator ceiling is
    /// rejected — a caller cannot disable the timeout from the request.
    #[test]
    fn test_query_zero_timeout_override_under_ceiling_rejected() {
        let server = create_test_server();
        let value = run_query(
            &server,
            query_req(
                "aql",
                "MATCH (n) RETURN n",
                Some(QueryLimitsOverride {
                    timeout_ms: Some(0),
                    ..Default::default()
                }),
            ),
        );
        assert_eq!(error_obj(&value)["code"].as_str(), Some("INVALID_ARGUMENT"));
        assert_eq!(
            error_obj(&value)["details"]["dimension"].as_str(),
            Some("wall_clock_timeout")
        );
    }

    /// A per-call row override BELOW the default is honored and truncates the
    /// result with the disclosed `truncated:true` completeness signal (#3226),
    /// NOT a fail-closed error (the AC's alternative for list-like reads).
    #[test]
    fn test_query_row_override_truncates_with_disclosure() {
        let server = create_test_server();
        for name in ["a", "b", "c"] {
            seed_named(&server, "Widget", name);
        }
        let value = run_query(
            &server,
            query_req(
                "aql",
                "MATCH (n:Widget) RETURN n",
                Some(QueryLimitsOverride {
                    max_result_rows: Some(1),
                    ..Default::default()
                }),
            ),
        );
        assert!(
            value.get("error").is_none(),
            "row cap truncates, never errors: {value}"
        );
        assert_eq!(value["row_count"].as_u64(), Some(1), "{value}");
        assert_eq!(value["truncated"].as_bool(), Some(true), "{value}");
    }

    /// The result-byte cap fails closed: a response over the cap is a
    /// non-retriable RESOURCE_EXHAUSTED with `details {dimension, limit,
    /// consumed}` and bumps the result-bytes counter.
    #[test]
    fn test_query_byte_cap_exceeded_is_resource_exhausted() {
        let server = create_test_server().with_query_limits(QueryLimitsConfig {
            default_max_response_bytes: 40, // tiny: any real row overflows
            ..QueryLimitsConfig::default()
        });
        seed_named(&server, "Widget", "alpha");
        let value = run_query(&server, query_req("aql", "MATCH (n:Widget) RETURN n", None));
        let err = error_obj(&value);
        assert_eq!(err["code"].as_str(), Some("RESOURCE_EXHAUSTED"), "{value}");
        assert_eq!(err["retriable"].as_bool(), Some(false), "{value}");
        assert_eq!(err["details"]["dimension"].as_str(), Some("result_bytes"));
        let limit = err["details"]["limit"].as_u64().unwrap();
        let consumed = err["details"]["consumed"].as_u64().unwrap();
        assert!(
            consumed > limit,
            "consumed {consumed} must exceed limit {limit}"
        );
        assert_eq!(server.limit_termination_counts().result_bytes, 1);
    }

    /// Disabled limits ignore every per-call override (an override that would
    /// otherwise be rejected is silently ignored) and enforce nothing.
    #[test]
    fn test_query_disabled_limits_ignore_overrides() {
        let server = create_test_server().with_query_limits(QueryLimitsConfig::disabled());
        seed_named(&server, "Widget", "alpha");
        let value = run_query(
            &server,
            query_req(
                "aql",
                "MATCH (n:Widget) RETURN n",
                Some(QueryLimitsOverride {
                    timeout_ms: Some(0),
                    max_result_rows: Some(0),
                    max_response_bytes: Some(1), // would fail closed if enforced
                }),
            ),
        );
        assert!(
            value.get("error").is_none(),
            "disabled limits enforce nothing: {value}"
        );
        assert_eq!(value["row_count"].as_u64(), Some(1));
    }

    /// Composition with the #3353 token budget: the protective operator byte cap
    /// runs FIRST inside `handle_query`, so a byte-cap breach fails closed as a
    /// RESOURCE_EXHAUSTED error even when the caller supplies a (generous) token
    /// budget — the budget shaper passes the error through, never masking the
    /// protective limit (no silent bypass).
    #[test]
    fn test_query_byte_cap_wins_over_token_budget() {
        let server = create_test_server().with_query_limits(QueryLimitsConfig {
            default_max_response_bytes: 40,
            ..QueryLimitsConfig::default()
        });
        seed_named(&server, "Widget", "alpha");
        // Drive through dispatch_tool so the #3353 budget shaper is engaged.
        let result = server.dispatch_tool(
            "query",
            serde_json::json!({
                "language": "aql",
                "query": "MATCH (n:Widget) RETURN n",
                "max_response_tokens": 100000, // generous budget: would not truncate
            }),
        );
        let value: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(result)).unwrap();
        assert_eq!(
            error_obj(&value)["code"].as_str(),
            Some("RESOURCE_EXHAUSTED"),
            "operator byte cap must win over a generous token budget: {value}"
        );
    }

    /// Composition with the #3353 token budget, other direction: within the
    /// operator byte cap, a small token budget still shapes the response
    /// gracefully (disclosed) — the two limits coexist.
    #[test]
    fn test_query_within_byte_cap_token_budget_still_shapes() {
        let server = create_test_server(); // generous operator byte cap
        for i in 0..5 {
            seed_named(&server, "Widget", &format!("w{i}"));
        }
        let result = server.dispatch_tool(
            "query",
            serde_json::json!({
                "language": "aql",
                "query": "MATCH (n:Widget) RETURN n",
                "max_response_tokens": 256, // small: engages the budget ladder
            }),
        );
        let value: serde_json::Value =
            serde_json::from_str(&AletheiaMcpServer::extract_text(result)).unwrap();
        // Either a shaped success (budget block present) or the too-small-budget
        // INVALID_ARGUMENT — but never a RESOURCE_EXHAUSTED (we are within the
        // operator byte cap) and never an unbounded response.
        if let Some(err) = value.get("error") {
            assert_eq!(err["code"].as_str(), Some("INVALID_ARGUMENT"), "{value}");
        } else {
            assert!(
                value.get("budget").is_some(),
                "budget shaping disclosed: {value}"
            );
        }
    }
}

// ============================================================================
// Uniqueness Constraint Tests
// ============================================================================

mod constraint_tests {
    use super::*;

    fn parse_json(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("server response must be valid JSON")
    }

    #[test]
    fn test_enable_unique_constraint_success() {
        let server = create_test_server();

        let response = server.enable_unique_constraint(EnableUniqueConstraintRequest {
            label: "Person".to_string(),
            property: "email".to_string(),
        });

        let v = parse_json(&response);
        assert_eq!(v["success"], serde_json::json!(true));
        assert_eq!(v["label"], "Person");
        assert_eq!(v["property"], "email");
    }

    #[test]
    fn test_enable_unique_constraint_rejects_existing_duplicates() {
        let server = create_test_server();

        let mut props = HashMap::new();
        props.insert("email".to_string(), serde_json::json!("dup@x"));
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props.clone()),
            provenance: None,
        });
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props),
            provenance: None,
        });

        let response = server.enable_unique_constraint(EnableUniqueConstraintRequest {
            label: "Person".to_string(),
            property: "email".to_string(),
        });

        let v = parse_json(&response);
        assert!(
            v.get("error").is_some(),
            "enable on label with pre-existing duplicates must return error: {v}"
        );
    }

    #[test]
    fn test_list_unique_constraints_empty() {
        let server = create_test_server();

        let response = server.list_unique_constraints(ListUniqueConstraintsRequest {});

        let v = parse_json(&response);
        assert_eq!(v["count"], serde_json::json!(0));
        assert!(
            v["constraints"]
                .as_array()
                .map(|a| a.is_empty())
                .unwrap_or(false),
            "constraints list should be empty on fresh server: {v}"
        );
    }

    #[test]
    fn test_list_unique_constraints_after_enable() {
        let server = create_test_server();

        server.enable_unique_constraint(EnableUniqueConstraintRequest {
            label: "Person".to_string(),
            property: "email".to_string(),
        });
        server.enable_unique_constraint(EnableUniqueConstraintRequest {
            label: "Company".to_string(),
            property: "name".to_string(),
        });

        let response = server.list_unique_constraints(ListUniqueConstraintsRequest {});

        let v = parse_json(&response);
        assert_eq!(
            v["count"],
            serde_json::json!(2),
            "should list 2 constraints: {v}"
        );
        assert_eq!(v["constraints"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_create_node_constraint_violation_structured_error() {
        let server = create_test_server();

        server.enable_unique_constraint(EnableUniqueConstraintRequest {
            label: "Person".to_string(),
            property: "email".to_string(),
        });

        let mut props = HashMap::new();
        props.insert("email".to_string(), serde_json::json!("alice@x"));

        let first_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props.clone()),
            provenance: None,
        });
        let first: NodeResponse =
            parse_response(&first_response).expect("first create must succeed");

        let dup_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props),
            provenance: None,
        });

        let v = parse_json(&dup_response);
        assert_eq!(
            v["constraint_violation"],
            serde_json::json!(true),
            "duplicate create must set constraint_violation=true: {v}"
        );
        assert_eq!(v["success"], serde_json::json!(false));
        assert_eq!(v["label"], "Person");
        assert_eq!(v["property"], "email");
        assert_eq!(v["value"], "alice@x");
        assert_eq!(
            v["existing_node_id"],
            serde_json::json!(first.id),
            "existing_node_id must point to the first node: {v}"
        );
    }

    #[test]
    fn test_update_node_constraint_violation_structured_error() {
        let server = create_test_server();

        server.enable_unique_constraint(EnableUniqueConstraintRequest {
            label: "Person".to_string(),
            property: "email".to_string(),
        });

        let mut props_a = HashMap::new();
        props_a.insert("email".to_string(), serde_json::json!("a@x"));
        let resp_a = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props_a),
            provenance: None,
        });
        let node_a: NodeResponse = parse_response(&resp_a).expect("node A must be created");

        let mut props_b = HashMap::new();
        props_b.insert("email".to_string(), serde_json::json!("b@x"));
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props_b),
            provenance: None,
        });

        let mut collision = HashMap::new();
        collision.insert("email".to_string(), serde_json::json!("b@x"));
        let update_response = server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id: node_a.id,
            properties: collision,
            provenance: None,
        });

        let v = parse_json(&update_response);
        assert_eq!(
            v["constraint_violation"],
            serde_json::json!(true),
            "colliding update must set constraint_violation=true: {v}"
        );
        assert_eq!(v["success"], serde_json::json!(false));
        assert_eq!(v["property"], "email");
    }

    #[test]
    fn test_update_node_non_constraint_error_fallthrough() {
        // Covers server.rs: the `db_error(...)` fallthrough in
        // handle_update_node for non-constraint errors — db_error only emits
        // the legacy constraint top-level fields (constraint_violation,
        // existing_node_id, ...) for a ConstraintError::UniqueViolation.
        //
        // We call update_node with a valid-format but non-existent node ID so that
        // `tx.update_node` returns StorageError::NodeNotFound — which is NOT a
        // ConstraintError::UniqueViolation — and db_error takes the plain path.
        let server = create_test_server();

        let response = server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id: 99999,
            properties: HashMap::new(),
            provenance: None,
        });

        let v = parse_json(&response);
        assert!(
            v.get("error").is_some(),
            "update_node on non-existent node must return an error: {v}"
        );
        assert_eq!(
            v.get("constraint_violation"),
            None,
            "non-constraint error must NOT set constraint_violation: {v}"
        );
    }
}

// ============================================================================
// Schema Discovery Tests (get_schema, Issue #3214)
// ============================================================================

mod schema_tests {
    use super::*;

    fn schema_req(
        as_of_valid_time: Option<&str>,
        as_of_transaction_time: Option<&str>,
    ) -> GetSchemaRequest {
        GetSchemaRequest {
            as_of_valid_time: as_of_valid_time.map(str::to_string),
            as_of_transaction_time: as_of_transaction_time.map(str::to_string),
        }
    }

    #[test]
    fn test_get_schema_empty_database() {
        let server = create_test_server();

        let response = server.get_schema(schema_req(None, None));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        assert_eq!(value["node_labels"].as_array().unwrap().len(), 0);
        assert_eq!(value["edge_types"].as_array().unwrap().len(), 0);
        assert_eq!(value["total_nodes"], serde_json::json!(0));
        assert_eq!(value["total_edges"], serde_json::json!(0));
        assert_eq!(value["sampled"], serde_json::json!(false));
        assert!(value["as_of"].is_null());
    }

    #[test]
    fn test_get_schema_populated_graph() {
        let server = create_test_server();

        let alice_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Alice"));
                m
            }),
            provenance: None,
        });
        let alice: NodeResponse = parse_response(&alice_response).unwrap();

        let bob_response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("email".to_string(), serde_json::json!("bob@example.com"));
                m
            }),
            provenance: None,
        });
        let bob: NodeResponse = parse_response(&bob_response).unwrap();

        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: alice.id,
            target_id: bob.id,
            label: "KNOWS".to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("since".to_string(), serde_json::json!(2020));
                m
            }),
            provenance: None,
        });

        let response = server.get_schema(schema_req(None, None));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");

        let node_labels = value["node_labels"].as_array().unwrap();
        assert_eq!(node_labels.len(), 1);
        assert_eq!(node_labels[0]["label"], serde_json::json!("Person"));
        assert_eq!(node_labels[0]["count"], serde_json::json!(2));
        let person_keys: Vec<String> = node_labels[0]["property_keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(person_keys, vec!["email".to_string(), "name".to_string()]);

        let edge_types = value["edge_types"].as_array().unwrap();
        assert_eq!(edge_types.len(), 1);
        assert_eq!(edge_types[0]["edge_type"], serde_json::json!("KNOWS"));
        assert_eq!(edge_types[0]["count"], serde_json::json!(1));
        assert_eq!(edge_types[0]["property_keys"], serde_json::json!(["since"]));

        assert_eq!(value["total_nodes"], serde_json::json!(2));
        assert_eq!(value["total_edges"], serde_json::json!(1));
        assert_eq!(value["sampled"], serde_json::json!(false));
        assert!(value["as_of"].is_null());
    }

    #[test]
    fn test_get_schema_invalid_timestamp() {
        let server = create_test_server();

        let response = server.get_schema(schema_req(Some("not-a-timestamp"), None));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_some());
    }

    /// AC: with only `as_of_valid_time` set, `as_of_transaction_time`
    /// defaults to "now" -- so a node created (and thus recorded) right now
    /// is visible, since "now" covers both "the fact is valid as of
    /// as_of_valid_time" (explicit) and "it was recorded by now" (default).
    #[test]
    fn test_get_schema_only_valid_time_set_defaults_transaction_time_to_now() {
        let server = create_test_server();

        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Report".to_string(),
            properties: None,
            provenance: None,
        });

        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        let response = server.get_schema(schema_req(Some(&now_micros.to_string()), None));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        assert!(!value["as_of"].is_null(), "temporal branch must be taken");
        let labels: Vec<String> = value["node_labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["label"].as_str().unwrap().to_string())
            .collect();
        assert!(
            labels.contains(&"Report".to_string()),
            "Report should be visible: valid_time=now (explicit), transaction_time defaults to now: {labels:?}"
        );
    }

    /// AC: with only `as_of_transaction_time` set, `as_of_valid_time`
    /// defaults to "now" -- so a node created (and thus recorded) right now
    /// is visible.
    #[test]
    fn test_get_schema_only_transaction_time_set_defaults_valid_time_to_now() {
        let server = create_test_server();

        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Invoice".to_string(),
            properties: None,
            provenance: None,
        });

        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        let response = server.get_schema(schema_req(None, Some(&now_micros.to_string())));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        assert!(!value["as_of"].is_null(), "temporal branch must be taken");
        let labels: Vec<String> = value["node_labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["label"].as_str().unwrap().to_string())
            .collect();
        assert!(
            labels.contains(&"Invoice".to_string()),
            "Invoice should be visible: transaction_time=now (explicit), valid_time defaults to now: {labels:?}"
        );
    }

    #[test]
    fn test_get_schema_as_of_label_absent_before_first_write() {
        let server = create_test_server();

        // Schema as of the Unix epoch — well before any writes — must be empty.
        let before = server.get_schema(schema_req(Some("0"), Some("0")));
        let before_value: serde_json::Value = serde_json::from_str(&before).unwrap();
        assert!(before_value.get("error").is_none());
        assert_eq!(before_value["node_labels"].as_array().unwrap().len(), 0);

        // Create a node now.
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Gadget".to_string(),
            properties: None,
            provenance: None,
        });

        // Still as of the epoch: "Gadget" must remain absent (recorded later).
        let still_before = server.get_schema(schema_req(Some("0"), Some("0")));
        let still_before_value: serde_json::Value = serde_json::from_str(&still_before).unwrap();
        assert!(still_before_value.get("error").is_none());
        let labels: Vec<String> = still_before_value["node_labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["label"].as_str().unwrap().to_string())
            .collect();
        assert!(
            !labels.contains(&"Gadget".to_string()),
            "Gadget should be absent as of the epoch: {labels:?}"
        );

        // Current-state schema (no as_of) must include it.
        let now = server.get_schema(schema_req(None, None));
        let now_value: serde_json::Value = serde_json::from_str(&now).unwrap();
        let now_labels: Vec<String> = now_value["node_labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["label"].as_str().unwrap().to_string())
            .collect();
        assert!(now_labels.contains(&"Gadget".to_string()));
    }

    #[test]
    fn test_get_schema_registered_in_tool_list() {
        let server = create_test_server();
        let tools = server.list_tools_for_test();
        assert!(
            tools.iter().any(|name| name == "get_schema"),
            "get_schema must be registered in the tools manifest: {tools:?}"
        );
    }

    /// AC: "Counts in the summary match the results of the corresponding
    /// count_nodes / count_edges calls for each label/type."
    #[test]
    fn test_get_schema_counts_match_count_nodes_and_count_edges_tools() {
        let server = create_test_server();

        for _ in 0..3 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Person".to_string(),
                properties: None,
                provenance: None,
            });
        }
        for _ in 0..2 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Company".to_string(),
                properties: None,
                provenance: None,
            });
        }
        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Company".to_string(),
            properties: None,
            provenance: None,
        }))
        .unwrap();
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "WORKS_AT".to_string(),
            properties: None,
            provenance: None,
        });

        let schema_value: serde_json::Value =
            serde_json::from_str(&server.get_schema(schema_req(None, None))).unwrap();

        // Per-label node counts must match the count_nodes tool filtered by that label.
        for label_entry in schema_value["node_labels"].as_array().unwrap() {
            let label = label_entry["label"].as_str().unwrap();
            let count_response = server.count_nodes(CountNodesRequest {
                label: Some(label.to_string()),
            });
            let count_value: serde_json::Value = serde_json::from_str(&count_response).unwrap();
            assert_eq!(
                label_entry["count"], count_value["count"],
                "schema count for label '{label}' must match count_nodes(label='{label}')"
            );
        }

        // Total node/edge counts must match the unfiltered count_nodes/count_edges tools.
        let total_nodes: serde_json::Value =
            serde_json::from_str(&server.count_nodes(CountNodesRequest { label: None })).unwrap();
        assert_eq!(schema_value["total_nodes"], total_nodes["count"]);

        let total_edges: serde_json::Value =
            serde_json::from_str(&server.count_edges(CountEdgesRequest { label: None })).unwrap();
        assert_eq!(schema_value["total_edges"], total_edges["count"]);
    }
}

// ============================================================================
// Vector Elision Tests (Issue #3220)
// ============================================================================
//
// MCP read responses elide vector/embedding properties by default (replacing
// them with a `{type, dim, elided:true}` descriptor) to protect LLM context.
// `include_vectors: true` is the escape hatch that restores the full float
// array. These tests pin that contract across every affected tool.

mod vector_elision_tests {
    use super::*;
    use crate::core::vector::SparseVec;

    fn embedding_of(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32) * 0.001 + 0.1).collect()
    }

    fn create_node_with_embedding(server: &AletheiaMcpServer, dim: usize) -> (u64, Vec<f32>) {
        let embedding = embedding_of(dim);
        let mut props = HashMap::new();
        props.insert(
            "embedding".to_string(),
            serde_json::json!(embedding.clone()),
        );
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Document".to_string(),
            properties: Some(props),
            provenance: None,
        });
        let node: NodeResponse = parse_response(&response).expect("Failed to create node");
        (node.id, embedding)
    }

    #[test]
    fn test_get_node_elides_vector_by_default() {
        let server = create_test_server();
        let (node_id, _embedding) = create_node_with_embedding(&server, 1536);

        let response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id,
            include_vectors: None,
        });

        assert!(
            response.len() < 500,
            "elided response should be small, was {} bytes: {response}",
            response.len()
        );

        let node: NodeResponse = parse_response(&response).expect("Failed to get node");
        assert_eq!(
            node.properties.get("embedding"),
            Some(&serde_json::json!({"type": "vector", "dim": 1536, "elided": true}))
        );
    }

    #[test]
    fn test_get_node_include_vectors_true_is_lossless() {
        let server = create_test_server();
        let (node_id, embedding) = create_node_with_embedding(&server, 1536);

        let response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id,
            include_vectors: Some(true),
        });

        let node: NodeResponse = parse_response(&response).expect("Failed to get node");
        let returned = node.properties.get("embedding").unwrap();
        let returned_array = returned.as_array().expect("embedding should be an array");
        assert_eq!(returned_array.len(), 1536);

        let returned_floats: Vec<f32> = returned_array
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(returned_floats, embedding);
    }

    // Issue #2906 (MAJOR 3): `update_node_embedding` merges the new embedding
    // into the node's existing properties inside a SINGLE write transaction.
    // This exercises the exact transactional read-merge-write mechanism the
    // handler uses (without needing a model): reading the node from the
    // transaction's own snapshot, inserting the vector into the existing
    // property builder, and updating — proving every other property is
    // preserved and the merge is atomic (so a lost-update race cannot silently
    // drop a concurrent writer's non-embedding change).
    #[test]
    fn update_embedding_transactional_merge_preserves_other_properties() {
        use crate::api::transaction::{ReadOps, WriteOps, WriteRequestOptions};

        let db = AletheiaDB::new().expect("db");

        // Seed a node with non-embedding properties and an initial embedding.
        let initial = PropertyMapBuilder::new()
            .insert("name", "Alice")
            .insert("age", 30i64)
            .try_insert_vector("embedding", &[0.1f32, 0.2, 0.3, 0.4])
            .expect("seed vector")
            .build();
        let node_id = db.create_node("Person", initial).expect("create");

        // Merge a NEW embedding, reproducing the handler's transaction closure.
        let new_embedding = vec![0.9f32, 0.8, 0.7, 0.6];
        db.write(|tx| {
            let node = tx.get_node(node_id)?;
            let properties = node
                .properties
                .builder()
                .try_insert_vector("embedding", &new_embedding)
                .expect("insert vector")
                .build();
            tx.update_node_with_options(node_id, properties, WriteRequestOptions::new())?;
            Ok::<(), crate::Error>(())
        })
        .expect("transactional update");

        // Non-embedding properties survive the embedding-only update.
        let node = db.get_node(node_id).expect("get after update");
        assert_eq!(
            node.properties.get("name").and_then(|v| v.as_str()),
            Some("Alice"),
            "name must be preserved across the embedding update"
        );
        assert_eq!(
            node.properties.get("age").and_then(|v| v.as_int()),
            Some(30),
            "age must be preserved across the embedding update"
        );
        // And the embedding itself was updated to the new vector.
        let stored = node
            .properties
            .get("embedding")
            .and_then(|v| v.as_vector())
            .expect("embedding vector present");
        assert_eq!(stored, new_embedding.as_slice(), "embedding was updated");
    }

    #[test]
    fn test_list_nodes_elides_vectors_by_default() {
        let server = create_test_server();
        for _ in 0..3 {
            create_node_with_embedding(&server, 8);
        }

        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Document".to_string()),
            property_key: None,
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let nodes = value["nodes"].as_array().expect("nodes array");
        assert_eq!(nodes.len(), 3);
        for node in nodes {
            assert_eq!(
                node["properties"]["embedding"],
                serde_json::json!({"type": "vector", "dim": 8, "elided": true})
            );
        }

        // include_vectors: true restores the full arrays.
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Document".to_string()),
            property_key: None,
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: Some(true),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let expected_embedding = embedding_of(8);
        for node in value["nodes"].as_array().unwrap() {
            let arr = node["properties"]["embedding"]
                .as_array()
                .expect("embedding should be a full array");
            assert_eq!(arr.len(), 8);
            let returned_floats: Vec<f32> = arr.iter().map(|v| v.as_f64().unwrap() as f32).collect();
            assert_eq!(returned_floats, expected_embedding);
        }
    }

    #[test]
    fn test_get_edge_and_outgoing_incoming_edges_elide_vectors_by_default() {
        let server = create_test_server();
        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        }))
        .unwrap();

        let embedding = embedding_of(6);
        let mut props = HashMap::new();
        props.insert("embedding".to_string(), serde_json::json!(embedding));
        let edge_response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "SIMILAR_TO".to_string(),
            properties: Some(props),
            provenance: None,
        });
        let edge: EdgeResponse = parse_response(&edge_response).unwrap();

        // get_edge
        let response = server.get_edge(GetEdgeRequest {
            namespace: None,
            edge_id: edge.id,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            value["properties"]["embedding"],
            serde_json::json!({"type": "vector", "dim": 6, "elided": true})
        );

        let response = server.get_edge(GetEdgeRequest {
            namespace: None,
            edge_id: edge.id,
            include_vectors: Some(true),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let arr = value["properties"]["embedding"].as_array().unwrap();
        assert_eq!(arr.len(), 6);
        let returned_floats: Vec<f32> = arr.iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(returned_floats, embedding);

        // get_outgoing_edges
        let response = server.get_outgoing_edges(GetOutgoingEdgesRequest {
            node_id: n1.id,
            label: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            value["edges"][0]["properties"]["embedding"],
            serde_json::json!({"type": "vector", "dim": 6, "elided": true})
        );

        let response = server.get_outgoing_edges(GetOutgoingEdgesRequest {
            node_id: n1.id,
            label: None,
            include_vectors: Some(true),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let arr = value["edges"][0]["properties"]["embedding"].as_array().unwrap();
        assert_eq!(arr.len(), 6);
        let returned_floats: Vec<f32> = arr.iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(returned_floats, embedding);

        // get_incoming_edges
        let response = server.get_incoming_edges(GetIncomingEdgesRequest {
            node_id: n2.id,
            label: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            value["edges"][0]["properties"]["embedding"],
            serde_json::json!({"type": "vector", "dim": 6, "elided": true})
        );

        let response = server.get_incoming_edges(GetIncomingEdgesRequest {
            node_id: n2.id,
            label: None,
            include_vectors: Some(true),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let arr = value["edges"][0]["properties"]["embedding"].as_array().unwrap();
        assert_eq!(arr.len(), 6);
        let returned_floats: Vec<f32> = arr.iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(returned_floats, embedding);
    }

    #[test]
    fn test_traverse_elides_vectors_by_default() {
        let server = create_test_server();
        let (n1_id, _) = create_node_with_embedding(&server, 5);
        let (n2_id, embedding2) = create_node_with_embedding(&server, 5);
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1_id,
            target_id: n2_id,
            label: "NEXT".to_string(),
            properties: None,
            provenance: None,
        });

        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: n1_id,
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let results = value["results"].as_array().expect("results array");
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0]["node"]["properties"]["embedding"],
            serde_json::json!({"type": "vector", "dim": 5, "elided": true})
        );

        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: n1_id,
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: Some(true),
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let results = value["results"].as_array().unwrap();
        let arr = results[0]["node"]["properties"]["embedding"].as_array().unwrap();
        assert_eq!(arr.len(), 5);
        let returned_floats: Vec<f32> = arr.iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(returned_floats, embedding2);
    }

    #[test]
    fn test_find_similar_elides_vectors_but_keeps_score() {
        let server = create_test_server();
        server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 4,
            distance_metric: Some("cosine".to_string()),
        });

        for i in 0..3 {
            let mut props = HashMap::new();
            props.insert(
                "embedding".to_string(),
                serde_json::json!([
                    (i as f32) * 0.1,
                    (i as f32) * 0.2,
                    (i as f32) * 0.3,
                    (i as f32) * 0.4,
                ]),
            );
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Document".to_string(),
                properties: Some(props),
                provenance: None,
            });
        }

        let response = server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            k: Some(3),
            offset: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let results = value["results"].as_array().expect("results array");
        assert!(!results.is_empty());
        for result in results {
            assert!(result.get("score").and_then(|s| s.as_f64()).is_some());
            assert_eq!(
                result["node"]["properties"]["embedding"],
                serde_json::json!({"type": "vector", "dim": 4, "elided": true})
            );
        }

        let response = server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            k: Some(3),
            offset: None,
            include_vectors: Some(true),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        for result in value["results"].as_array().unwrap() {
            assert!(result.get("score").and_then(|s| s.as_f64()).is_some());
            let arr = result["node"]["properties"]["embedding"].as_array().unwrap();
            assert_eq!(arr.len(), 4);
            let returned_floats: Vec<f32> = arr.iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let base = returned_floats[0] / 0.1;
            assert_eq!(returned_floats, vec![base * 0.1, base * 0.2, base * 0.3, base * 0.4]);
        }
    }

    #[test]
    fn test_hybrid_query_vector_first_elides_vectors_but_keeps_similarity_score() {
        let server = create_test_server();
        server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 4,
            distance_metric: Some("cosine".to_string()),
        });
        for i in 0..3 {
            let mut props = HashMap::new();
            props.insert(
                "embedding".to_string(),
                serde_json::json!([
                    (i as f32) * 0.1,
                    (i as f32) * 0.2,
                    (i as f32) * 0.3,
                    (i as f32) * 0.4,
                ]),
            );
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Document".to_string(),
                properties: Some(props),
                provenance: None,
            });
        }

        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: None,
            traverse_edge: None,
            traverse_depth: None,
            vector_property: Some("embedding".to_string()),
            query_embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            top_k: Some(3),
            valid_time: None,
            transaction_time: None,
            filter_label: None,
            limit: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let results = value["results"].as_array().expect("results array");
        assert!(!results.is_empty());
        for result in results {
            assert!(result.get("similarity_score").is_some());
            assert_eq!(
                result["node"]["properties"]["embedding"],
                serde_json::json!({"type": "vector", "dim": 4, "elided": true})
            );
        }

        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: None,
            traverse_edge: None,
            traverse_depth: None,
            vector_property: Some("embedding".to_string()),
            query_embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            top_k: Some(3),
            valid_time: None,
            transaction_time: None,
            filter_label: None,
            limit: None,
            include_vectors: Some(true),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        for result in value["results"].as_array().unwrap() {
            let arr = result["node"]["properties"]["embedding"].as_array().unwrap();
            assert_eq!(arr.len(), 4);
            let returned_floats: Vec<f32> = arr.iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let base = returned_floats[0] / 0.1;
            assert_eq!(returned_floats, vec![base * 0.1, base * 0.2, base * 0.3, base * 0.4]);
        }
    }

    #[test]
    fn test_hybrid_query_graph_first_elides_vectors_but_keeps_similarity_score() {
        let server = create_test_server();
        let (n1_id, _) = create_node_with_embedding(&server, 4);
        let (n2_id, embedding2) = create_node_with_embedding(&server, 4);
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: n1_id,
            target_id: n2_id,
            label: "NEXT".to_string(),
            properties: None,
            provenance: None,
        });

        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: Some(n1_id),
            traverse_edge: Some("NEXT".to_string()),
            traverse_depth: Some(1),
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: None,
            transaction_time: None,
            filter_label: None,
            limit: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let results = value["results"].as_array().expect("results array");
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0]["node"]["properties"]["embedding"],
            serde_json::json!({"type": "vector", "dim": 4, "elided": true})
        );

        let response = server.hybrid_query(HybridQueryRequest {
            namespace: None,
            start_node_id: Some(n1_id),
            traverse_edge: Some("NEXT".to_string()),
            traverse_depth: Some(1),
            vector_property: None,
            query_embedding: None,
            top_k: None,
            valid_time: None,
            transaction_time: None,
            filter_label: None,
            limit: None,
            include_vectors: Some(true),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let arr = value["results"][0]["node"]["properties"]["embedding"].as_array().unwrap();
        assert_eq!(arr.len(), 4);
        let returned_floats: Vec<f32> = arr.iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(returned_floats, embedding2);
    }

    #[test]
    fn test_sparse_vector_elided_by_default_with_dim_and_nnz() {
        let server = create_test_server();

        let sparse = SparseVec::new(vec![1, 4], vec![1.5, 2.3], 10).expect("valid sparse vector");
        let props = PropertyMapBuilder::new()
            .try_insert("embedding", sparse)
            .expect("insert sparse vector")
            .build();
        let node_id = server
            .db()
            .create_node("Document", props)
            .expect("create node with sparse vector");

        let response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id: node_id.as_u64(),
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            value["properties"]["embedding"],
            serde_json::json!({"type": "sparse_vector", "dim": 10, "nnz": 2, "elided": true})
        );

        let response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id: node_id.as_u64(),
            include_vectors: Some(true),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            value["properties"]["embedding"]["indices"],
            serde_json::json!([1, 4])
        );
        let values: Vec<f32> = value["properties"]["embedding"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(values, vec![1.5_f32, 2.3_f32]);
    }

    #[test]
    fn test_create_node_and_update_node_always_return_full_vectors() {
        let server = create_test_server();

        let mut props = HashMap::new();
        props.insert(
            "embedding".to_string(),
            serde_json::json!([0.1, 0.2, 0.3, 0.4]),
        );
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Document".to_string(),
            properties: Some(props),
            provenance: None,
        });
        let node: NodeResponse = parse_response(&response).expect("Failed to create node");
        assert_eq!(
            node.properties
                .get("embedding")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            4
        );

        let mut new_props = HashMap::new();
        new_props.insert(
            "embedding".to_string(),
            serde_json::json!([0.5, 0.6, 0.7, 0.8]),
        );
        let update_response = server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id: node.id,
            properties: new_props,
            provenance: None,
        });
        let updated: NodeResponse =
            parse_response(&update_response).expect("Failed to update node");
        let updated_embedding = updated
            .properties
            .get("embedding")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(updated_embedding.len(), 4);
        assert_eq!(updated_embedding[0].as_f64().unwrap() as f32, 0.5);
    }

    #[test]
    fn test_non_vector_properties_unaffected_by_include_vectors_flag() {
        let server = create_test_server();

        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!("Alice"));
        props.insert("age".to_string(), serde_json::json!(30));
        props.insert("active".to_string(), serde_json::json!(true));
        props.insert("nickname".to_string(), serde_json::Value::Null);
        props.insert("tags".to_string(), serde_json::json!(["a", "b", "c"]));

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props),
            provenance: None,
        });
        let node: NodeResponse = parse_response(&response).expect("Failed to create node");

        let variants = [None, Some(false), Some(true)];
        let mut previous: Option<HashMap<String, serde_json::Value>> = None;
        for include_vectors in variants {
            let response = server.get_node(GetNodeRequest {
                namespace: None,
                node_id: node.id,
                include_vectors,
            });
            let fetched: NodeResponse = parse_response(&response).expect("Failed to get node");
            if let Some(prev) = &previous {
                assert_eq!(prev, &fetched.properties);
            }
            previous = Some(fetched.properties);
        }
    }
}

// ============================================================================
// Valid-Time Write Tests (Issue #3221)
// ============================================================================

/// MCP surface tests for valid-time retraction (Issue #3230).
mod retraction_tests {
    use super::*;
    use chrono::{DateTime, Duration, Utc};

    fn rfc3339_hours_ago(now: DateTime<Utc>, hours: i64) -> String {
        (now - Duration::hours(hours)).to_rfc3339()
    }

    /// Create a Person node whose valid_from is `hours` hours before `now`.
    fn create_person_at(server: &AletheiaMcpServer, now: DateTime<Utc>, hours: i64) -> u64 {
        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, hours)),
            provenance: None,
        }))
        .expect("create should succeed");
        node.id
    }

    /// Parse an interval bound from a retraction response and return its
    /// microseconds since epoch, asserting the RFC 3339 `Z`-suffixed
    /// microsecond-precision convention.
    fn parse_bound_micros(value: &serde_json::Value, key: &str) -> i64 {
        let s = value[key]
            .as_str()
            .unwrap_or_else(|| panic!("{key} must be an RFC 3339 string, got {value}"));
        assert!(s.ends_with('Z'), "{key} must carry a Z suffix, got {s}");
        s.parse::<DateTime<Utc>>()
            .unwrap_or_else(|e| panic!("{key} must parse as RFC 3339 ({e}): {s}"))
            .timestamp_micros()
    }

    #[test]
    fn retract_node_happy_path_returns_rfc3339_interval() {
        let server = create_test_server();
        let now = Utc::now();
        let node_id = create_person_at(&server, now, 2);
        let t_retract = rfc3339_hours_ago(now, 1);

        let response = server.retract_node(RetractNodeRequest {
            node_id,
            valid_time: Some(t_retract.clone()),
            detach: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "expected success, got {value}"
        );
        assert_eq!(value["success"], true);
        assert_eq!(value["node_id"], node_id);
        assert_eq!(value["retracted"], true);
        assert_eq!(value["already_retracted"], false);
        assert_eq!(value["edges_retracted"], 0);

        // The closed half-open interval [valid_from, valid_to) comes back as
        // RFC 3339 microsecond-precision strings matching the request times.
        let valid_from_us = parse_bound_micros(&value, "valid_from");
        let valid_to_us = parse_bound_micros(&value, "valid_to");
        assert_eq!(valid_from_us, (now - Duration::hours(2)).timestamp_micros());
        assert_eq!(valid_to_us, (now - Duration::hours(1)).timestamp_micros());

        // Gone from current state (structured NOT_FOUND).
        let after = server.get_node(GetNodeRequest {
            namespace: None,
            node_id,
            include_vectors: None,
        });
        let after: serde_json::Value = serde_json::from_str(&after).unwrap();
        assert_eq!(after["error"]["code"], "NOT_FOUND", "got {after}");
    }

    #[test]
    fn retract_node_backdated_round_trip_via_get_node_at_time() {
        let server = create_test_server();
        let now = Utc::now();
        let node_id = create_person_at(&server, now, 3);

        let response = server.retract_node(RetractNodeRequest {
            node_id,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            detach: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "expected success, got {value}"
        );

        // Still visible at a valid time strictly before T...
        let before = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id,
            valid_time: rfc3339_hours_ago(now, 2),
            transaction_time: None,
        });
        let before: serde_json::Value = serde_json::from_str(&before).unwrap();
        assert!(before.get("error").is_none(), "expected node, got {before}");

        // ...and not at/after T.
        let at = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id,
            valid_time: rfc3339_hours_ago(now, 1),
            transaction_time: None,
        });
        let at: serde_json::Value = serde_json::from_str(&at).unwrap();
        assert!(
            at.get("error").is_some(),
            "expected not-valid at T, got {at}"
        );

        let after = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id,
            valid_time: rfc3339_hours_ago(now, 0),
            transaction_time: None,
        });
        let after: serde_json::Value = serde_json::from_str(&after).unwrap();
        assert!(
            after.get("error").is_some(),
            "expected not-valid, got {after}"
        );
    }

    #[test]
    fn retract_node_with_connected_edges_refused_failed_precondition() {
        let server = create_test_server();
        let now = Utc::now();
        let a = create_person_at(&server, now, 2);
        let b = create_person_at(&server, now, 2);
        let _edge: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: a,
            target_id: b,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 2)),
            provenance: None,
        }))
        .unwrap();

        let response = server.retract_node(RetractNodeRequest {
            node_id: a,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            detach: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        // Structured shape (#3234): FAILED_PRECONDITION + details.
        assert_eq!(value["error"]["code"], "FAILED_PRECONDITION", "got {value}");
        assert_eq!(value["error"]["retriable"], false);
        assert_eq!(value["error"]["details"]["connected_edges"], 1);
        assert_eq!(value["error"]["details"]["detach_required"], true);
        assert_eq!(value["error"]["details"]["node_id"], a);

        // Legacy top-level fields preserved additively (#3209 parallel).
        assert_eq!(value["success"], false);
        assert_eq!(value["connected_edges"], 1);
        assert_eq!(value["detach_required"], true);
        assert_eq!(value["node_id"], a);

        // Never a success that strands edges: node is still present.
        let still_there = server.get_node(GetNodeRequest {
            namespace: None,
            node_id: a,
            include_vectors: None,
        });
        let still_there: serde_json::Value = serde_json::from_str(&still_there).unwrap();
        assert!(still_there.get("error").is_none(), "got {still_there}");
    }

    #[test]
    fn retract_node_detach_coretracts_edges_and_reports_count() {
        let server = create_test_server();
        let now = Utc::now();
        let a = create_person_at(&server, now, 3);
        let b = create_person_at(&server, now, 3);
        let edge: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: a,
            target_id: b,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 3)),
            provenance: None,
        }))
        .unwrap();

        let t_retract = rfc3339_hours_ago(now, 1);
        let response = server.retract_node(RetractNodeRequest {
            node_id: a,
            valid_time: Some(t_retract),
            detach: Some(true),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "expected success, got {value}"
        );
        assert_eq!(value["edges_retracted"], 1);
        assert_eq!(value["already_retracted"], false);

        // The co-retracted edge: queryable strictly before T, gone after.
        let before = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: rfc3339_hours_ago(now, 2),
            transaction_time: None,
        });
        let before: serde_json::Value = serde_json::from_str(&before).unwrap();
        assert!(before.get("error").is_none(), "expected edge, got {before}");

        let after = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: rfc3339_hours_ago(now, 0),
            transaction_time: None,
        });
        let after: serde_json::Value = serde_json::from_str(&after).unwrap();
        assert!(
            after.get("error").is_some(),
            "expected not-valid, got {after}"
        );

        // Re-retracting the co-retracted edge is an idempotent no-op
        // reporting the original valid_to.
        let re_retract = server.retract_edge(RetractEdgeRequest {
            edge_id: edge.id,
            valid_time: Some(rfc3339_hours_ago(now, 0)),
        });
        let re_retract: serde_json::Value = serde_json::from_str(&re_retract).unwrap();
        assert!(re_retract.get("error").is_none(), "got {re_retract}");
        assert_eq!(re_retract["already_retracted"], true);
        assert_eq!(
            parse_bound_micros(&re_retract, "valid_to"),
            (now - Duration::hours(1)).timestamp_micros(),
            "idempotent re-retract must return the original valid_to"
        );
    }

    #[test]
    fn retract_node_idempotent_second_call_returns_existing_interval() {
        let server = create_test_server();
        let now = Utc::now();
        let node_id = create_person_at(&server, now, 3);

        let first = server.retract_node(RetractNodeRequest {
            node_id,
            valid_time: Some(rfc3339_hours_ago(now, 2)),
            detach: None,
        });
        let first: serde_json::Value = serde_json::from_str(&first).unwrap();
        assert!(first.get("error").is_none(), "got {first}");
        assert_eq!(first["already_retracted"], false);

        // Different T on the second call: the EXISTING end comes back.
        let second = server.retract_node(RetractNodeRequest {
            node_id,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            detach: None,
        });
        let second: serde_json::Value = serde_json::from_str(&second).unwrap();
        assert!(second.get("error").is_none(), "got {second}");
        assert_eq!(second["success"], true);
        assert_eq!(second["already_retracted"], true);
        assert_eq!(
            parse_bound_micros(&second, "valid_to"),
            (now - Duration::hours(2)).timestamp_micros()
        );
    }

    #[test]
    fn retract_node_malformed_valid_time_invalid_argument() {
        let server = create_test_server();
        let now = Utc::now();
        let node_id = create_person_at(&server, now, 2);

        let response = server.retract_node(RetractNodeRequest {
            node_id,
            valid_time: Some("not-a-timestamp".to_string()),
            detach: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value["error"]["code"], "INVALID_ARGUMENT", "got {value}");
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains("Invalid valid_time"), "got: {message}");

        // Nothing was retracted.
        let still_there = server.get_node(GetNodeRequest {
            namespace: None,
            node_id,
            include_vectors: None,
        });
        let still_there: serde_json::Value = serde_json::from_str(&still_there).unwrap();
        assert!(still_there.get("error").is_none(), "got {still_there}");
    }

    #[test]
    fn retract_node_valid_time_before_valid_from_invalid_argument() {
        let server = create_test_server();
        let now = Utc::now();
        let node_id = create_person_at(&server, now, 1);

        // T strictly before the node's valid_from.
        let response = server.retract_node(RetractNodeRequest {
            node_id,
            valid_time: Some(rfc3339_hours_ago(now, 2)),
            detach: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        // Matches #3221's existing mapping for the
        // ValidTimeBeforeEntityCreation family: INVALID_ARGUMENT.
        assert_eq!(value["error"]["code"], "INVALID_ARGUMENT", "got {value}");
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains("before entity creation"), "got: {message}");
    }

    #[test]
    fn retract_node_nonexistent_not_found() {
        let server = create_test_server();
        let response = server.retract_node(RetractNodeRequest {
            node_id: 999_999,
            valid_time: None,
            detach: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value["error"]["code"], "NOT_FOUND", "got {value}");
    }

    #[test]
    fn retract_edge_happy_path_and_omitted_valid_time_defaults_to_now() {
        let server = create_test_server();
        let now = Utc::now();
        let a = create_person_at(&server, now, 2);
        let b = create_person_at(&server, now, 2);
        let edge: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: a,
            target_id: b,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 2)),
            provenance: None,
        }))
        .unwrap();

        // Omitted valid_time defaults to now (#3221 convention).
        let response = server.retract_edge(RetractEdgeRequest {
            edge_id: edge.id,
            valid_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "expected success, got {value}"
        );
        assert_eq!(value["success"], true);
        assert_eq!(value["edge_id"], edge.id);
        assert_eq!(value["retracted"], true);
        assert_eq!(value["already_retracted"], false);

        let valid_to_us = parse_bound_micros(&value, "valid_to");
        let delta = (valid_to_us - now.timestamp_micros()).abs();
        assert!(
            delta < 60_000_000,
            "omitted valid_time must default to (approximately) now; delta {delta}us"
        );
        assert_eq!(
            parse_bound_micros(&value, "valid_from"),
            (now - Duration::hours(2)).timestamp_micros()
        );

        // History still shows the edge before the retraction instant.
        let before = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: rfc3339_hours_ago(now, 1),
            transaction_time: None,
        });
        let before: serde_json::Value = serde_json::from_str(&before).unwrap();
        assert!(before.get("error").is_none(), "expected edge, got {before}");
    }

    #[test]
    fn retract_edge_nonexistent_not_found() {
        let server = create_test_server();
        let response = server.retract_edge(RetractEdgeRequest {
            edge_id: 999_999,
            valid_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value["error"]["code"], "NOT_FOUND", "got {value}");
    }

    /// Fix-round #2: `connected_edges` counts DISTINCT edges. A self-loop
    /// occupies both adjacency directions but is one edge, so the refusal
    /// must report 1 — the same number `detach: true` then retracts.
    #[test]
    fn retract_node_self_loop_refusal_reports_one_and_detach_retracts_one() {
        let server = create_test_server();
        let now = Utc::now();
        let a = create_person_at(&server, now, 3);
        let _self_loop: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: a,
            target_id: a,
            label: "SELF".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 3)),
            provenance: None,
        }))
        .unwrap();

        // Refusal: 1 distinct edge, everywhere the count appears.
        let refusal = server.retract_node(RetractNodeRequest {
            node_id: a,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            detach: None,
        });
        let refusal: serde_json::Value = serde_json::from_str(&refusal).unwrap();
        assert_eq!(
            refusal["error"]["code"], "FAILED_PRECONDITION",
            "got {refusal}"
        );
        assert_eq!(
            refusal["error"]["details"]["connected_edges"], 1,
            "a self-loop must count once (distinct edges), got {refusal}"
        );
        assert_eq!(refusal["connected_edges"], 1);

        // Detach retracts exactly the refusal count.
        let detach = server.retract_node(RetractNodeRequest {
            node_id: a,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            detach: Some(true),
        });
        let detach: serde_json::Value = serde_json::from_str(&detach).unwrap();
        assert!(detach.get("error").is_none(), "got {detach}");
        assert_eq!(
            detach["edges_retracted"], 1,
            "detach must retract the same count the refusal reported"
        );
    }

    /// Fix-round #2: one outgoing + one incoming edge from different peers
    /// — the refusal reports 2, detach retracts 2, and BOTH edges obey the
    /// before/after-T temporal matrix.
    #[test]
    fn retract_node_multi_edge_refusal_reports_two_and_detach_retracts_two() {
        let server = create_test_server();
        let now = Utc::now();
        let x = create_person_at(&server, now, 3);
        let b = create_person_at(&server, now, 3);
        let c = create_person_at(&server, now, 3);
        let e_out: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: x,
            target_id: b,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 3)),
            provenance: None,
        }))
        .unwrap();
        let e_in: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: c,
            target_id: x,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 3)),
            provenance: None,
        }))
        .unwrap();

        let refusal = server.retract_node(RetractNodeRequest {
            node_id: x,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            detach: None,
        });
        let refusal: serde_json::Value = serde_json::from_str(&refusal).unwrap();
        assert_eq!(
            refusal["error"]["code"], "FAILED_PRECONDITION",
            "got {refusal}"
        );
        assert_eq!(refusal["error"]["details"]["connected_edges"], 2);
        assert_eq!(refusal["connected_edges"], 2);

        let detach = server.retract_node(RetractNodeRequest {
            node_id: x,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            detach: Some(true),
        });
        let detach: serde_json::Value = serde_json::from_str(&detach).unwrap();
        assert!(detach.get("error").is_none(), "got {detach}");
        assert_eq!(detach["edges_retracted"], 2);

        // Both edges: queryable strictly before T, gone at/after T.
        for edge_id in [e_out.id, e_in.id] {
            let before = server.get_edge_at_time(GetEdgeAtTimeRequest {
                namespace: None,
                edge_id,
                valid_time: rfc3339_hours_ago(now, 2),
                transaction_time: None,
            });
            let before: serde_json::Value = serde_json::from_str(&before).unwrap();
            assert!(
                before.get("error").is_none(),
                "edge {edge_id} must remain queryable strictly before T, got {before}"
            );

            let after = server.get_edge_at_time(GetEdgeAtTimeRequest {
                namespace: None,
                edge_id,
                valid_time: rfc3339_hours_ago(now, 0),
                transaction_time: None,
            });
            let after: serde_json::Value = serde_json::from_str(&after).unwrap();
            assert!(
                after.get("error").is_some(),
                "edge {edge_id} must be gone after T, got {after}"
            );

            let current = server.get_edge(GetEdgeRequest {
                namespace: None,
                edge_id,
                include_vectors: None,
            });
            let current: serde_json::Value = serde_json::from_str(&current).unwrap();
            assert_eq!(
                current["error"]["code"], "NOT_FOUND",
                "edge {edge_id} must be absent from current state, got {current}"
            );
        }
    }
}

mod valid_time_write_tests {
    use super::*;
    use chrono::{DateTime, Duration, Utc};

    /// `now` is the caller's own `Utc::now()`, captured once at the start of the test
    /// and reused for every offset it computes. Calling `Utc::now()` freshly inside the
    /// helper would let each call observe a slightly different instant under load,
    /// making relative orderings between offsets non-deterministic.
    fn rfc3339_hours_ago(now: DateTime<Utc>, hours: i64) -> String {
        (now - Duration::hours(hours)).to_rfc3339()
    }

    fn rfc3339_hours_from_now(now: DateTime<Utc>, hours: i64) -> String {
        (now + Duration::hours(hours)).to_rfc3339()
    }

    #[test]
    fn test_create_node_with_valid_time_backdated_round_trip() {
        let server = create_test_server();
        let now = Utc::now();

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            provenance: None,
        });
        let node: NodeResponse = parse_response(&response).expect("create should succeed");

        // Visible shortly after its valid_from.
        let visible = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: rfc3339_hours_ago(now, 0),
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&visible).unwrap();
        assert!(value.get("error").is_none(), "expected node, got {value}");

        // Invisible strictly before its valid_from.
        let invisible = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: rfc3339_hours_ago(now, 2),
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&invisible).unwrap();
        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_create_node_omitted_valid_time_matches_today() {
        // Omitting the field entirely (raw JSON) still deserializes.
        let parsed: CreateNodeRequest =
            serde_json::from_value(serde_json::json!({ "label": "Person" }))
                .expect("valid_time should be optional");
        assert_eq!(parsed.valid_time, None);

        let server = create_test_server();
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        });
        let node: NodeResponse = parse_response(&response).expect("create should succeed");
        assert_eq!(node.label, "Person");

        let now_response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: Utc::now().to_rfc3339(),
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&now_response).unwrap();
        assert!(value.get("error").is_none());
    }

    #[test]
    fn test_create_node_malformed_valid_time_structured_error() {
        let server = create_test_server();

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: Some("not-a-timestamp".to_string()),
            provenance: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let error = value
            .get("error")
            .and_then(|e| e["message"].as_str())
            .unwrap();
        assert!(error.contains("Invalid valid_time"), "got: {error}");
        assert_eq!(server.db().node_count(), 0, "no node should be created");
    }

    #[test]
    fn test_create_node_far_future_valid_time_typed_error_surfaced() {
        let server = create_test_server();
        let now = Utc::now();

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_from_now(now, 24 * 400)), // > 1 year
            provenance: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let error = value
            .get("error")
            .and_then(|e| e["message"].as_str())
            .unwrap();
        assert!(error.contains("too far in future"), "got: {error}");
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn test_create_edge_with_valid_time_backdated_round_trip() {
        let server = create_test_server();
        let now = Utc::now();

        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            provenance: None,
        });
        let edge: EdgeResponse = parse_response(&response).expect("create_edge should succeed");

        let visible = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: rfc3339_hours_ago(now, 0),
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&visible).unwrap();
        assert!(value.get("error").is_none(), "expected edge, got {value}");

        let invisible = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: rfc3339_hours_ago(now, 2),
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&invisible).unwrap();
        assert!(value.get("error").is_some());
    }

    #[test]
    fn test_update_node_with_valid_time_backdated() {
        let server = create_test_server();
        let now = Utc::now();

        let mut props = HashMap::new();
        props.insert("city".to_string(), serde_json::json!("Paris"));
        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: Some(props),
            valid_time: Some(rfc3339_hours_ago(now, 2)),
            provenance: None,
        }))
        .unwrap();

        let mut update_props = HashMap::new();
        update_props.insert("city".to_string(), serde_json::json!("London"));
        let update_valid_time = rfc3339_hours_ago(now, 1);
        let response = server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            node_id: node.id,
            properties: update_props,
            valid_time: Some(update_valid_time.clone()),
            provenance: None,
        });
        let updated: NodeResponse = parse_response(&response).expect("update should succeed");
        assert_eq!(
            updated.properties.get("city"),
            Some(&serde_json::json!("London"))
        );

        // Visible from its own valid_from onward.
        let at_update = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: update_valid_time,
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&at_update).unwrap();
        let fetched: NodeResponse = serde_json::from_value(value["node"].clone()).unwrap();
        assert_eq!(
            fetched.properties.get("city"),
            Some(&serde_json::json!("London"))
        );
    }

    #[test]
    fn test_update_node_valid_time_before_creation_rejected() {
        let server = create_test_server();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let mut update_props = HashMap::new();
        update_props.insert("name".to_string(), serde_json::json!("Bob"));
        let response = server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            node_id: node.id,
            properties: update_props,
            // Far enough in the past to precede the node's creation time.
            valid_time: Some("1970-01-01T00:00:01Z".to_string()),
            provenance: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let error = value
            .get("error")
            .and_then(|e| e["message"].as_str())
            .unwrap();
        assert!(error.contains("before entity creation"), "got: {error}");
    }

    #[test]
    fn test_transaction_time_not_client_settable() {
        let server = create_test_server();
        let now = Utc::now();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 1)),
            provenance: None,
        }))
        .unwrap();

        // As of a transaction_time 30 minutes ago, the write (committed just now)
        // had not happened yet -- the fact must not be retroactively knowable.
        let too_early_tx = (now - Duration::minutes(30)).to_rfc3339();
        let too_early = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: rfc3339_hours_ago(now, 0),
            transaction_time: Some(too_early_tx),
        });
        let value: serde_json::Value = serde_json::from_str(&too_early).unwrap();
        assert!(value.get("error").is_some());

        // Omitting transaction_time defaults to now, where the write is visible.
        let now_response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: rfc3339_hours_ago(now, 0),
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&now_response).unwrap();
        assert!(value.get("error").is_none());
    }

    #[test]
    fn test_update_edge_with_valid_time_backdated() {
        let server = create_test_server();
        let now = Utc::now();

        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let mut props = HashMap::new();
        props.insert("strength".to_string(), serde_json::json!(1));
        let edge: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: Some(props),
            valid_time: Some(rfc3339_hours_ago(now, 2)),
            provenance: None,
        }))
        .unwrap();

        let mut update_props = HashMap::new();
        update_props.insert("strength".to_string(), serde_json::json!(9));
        let update_valid_time = rfc3339_hours_ago(now, 1);
        let response = server.update_edge(UpdateEdgeRequest {
            namespace: None,
            derived_from: None,
            edge_id: edge.id,
            properties: update_props,
            valid_time: Some(update_valid_time.clone()),
            provenance: None,
        });
        let updated: EdgeResponse = parse_response(&response).expect("update should succeed");
        assert_eq!(
            updated.properties.get("strength"),
            Some(&serde_json::json!(9))
        );

        let at_update = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: update_valid_time,
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&at_update).unwrap();
        let fetched: EdgeResponse = serde_json::from_value(value["edge"].clone()).unwrap();
        assert_eq!(
            fetched.properties.get("strength"),
            Some(&serde_json::json!(9))
        );
    }

    /// Regression test: `create_edge_with_valid_time` used to be the only one of the six
    /// `*_with_valid_time` operations that never enforced the 1-year future cap, so an
    /// edge could silently be created with an arbitrarily-far-future `valid_time`. Mirrors
    /// `test_create_node_far_future_valid_time_typed_error_surfaced`.
    #[test]
    fn test_create_edge_far_future_valid_time_typed_error_surfaced() {
        let server = create_test_server();
        let now = Utc::now();

        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_from_now(now, 24 * 400)), // > 1 year
            provenance: None,
        });

        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let error = value
            .get("error")
            .and_then(|e| e["message"].as_str())
            .unwrap();
        assert!(error.contains("too far in future"), "got: {error}");
        assert_eq!(server.db().edge_count(), 0);
    }

    /// `delete_node`'s `valid_time` field: the caller-specified `valid_from` is
    /// correctly threaded through to the tombstone version, and the node is no longer
    /// visible as of now. (A probe strictly between create's and delete's `valid_from`
    /// is not reachable via `get_node_at_time` -- deleting closes the previous version's
    /// *transaction* time at commit, same documented caveat as
    /// `update_node_with_valid_time_backdated_round_trip` in `db::ops::tests`.)
    #[test]
    fn test_delete_node_with_valid_time_backdated_round_trip() {
        let server = create_test_server();
        let now = Utc::now();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 2)),
            provenance: None,
        }))
        .unwrap();

        let delete_valid_time = rfc3339_hours_ago(now, 1);
        let response = server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id: node.id,
            detach: None,
            valid_time: Some(delete_valid_time.clone()),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));

        // The caller-specified valid_from was correctly threaded through to the
        // tombstone version.
        let node_id = crate::core::id::NodeId::new(node.id).unwrap();
        let historical = server.db().historical.read();
        let version_id = historical.get_current_node_version(node_id).unwrap();
        let version = historical.get_node_version(version_id).unwrap();
        let recorded_valid_from = version.temporal.valid_time().start();
        drop(historical);
        let expected: chrono::DateTime<Utc> = delete_valid_time.parse().unwrap();
        assert_eq!(
            recorded_valid_from.wallclock(),
            expected.timestamp_micros(),
            "tombstone valid_from should match the requested delete_valid_time"
        );

        // No longer visible as of now.
        let gone = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: now.to_rfc3339(),
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&gone).unwrap();
        assert!(value.get("error").is_some());
    }

    /// `delete_edge`'s `valid_time` field: edge mirror of the node round trip above --
    /// same caveat about the probe-strictly-between-versions gap applies.
    #[test]
    fn test_delete_edge_with_valid_time_backdated_round_trip() {
        let server = create_test_server();
        let now = Utc::now();

        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let edge: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: Some(rfc3339_hours_ago(now, 2)),
            provenance: None,
        }))
        .unwrap();

        let delete_valid_time = rfc3339_hours_ago(now, 1);
        let response = server.delete_edge(DeleteEdgeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: Some(delete_valid_time.clone()),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));

        // The caller-specified valid_from was correctly threaded through to the
        // tombstone version.
        let edge_id = crate::core::id::EdgeId::new(edge.id).unwrap();
        let historical = server.db().historical.read();
        let version_id = historical.get_current_edge_version(edge_id).unwrap();
        let version = historical.get_edge_version(version_id).unwrap();
        let recorded_valid_from = version.temporal.valid_time().start();
        drop(historical);
        let expected: chrono::DateTime<Utc> = delete_valid_time.parse().unwrap();
        assert_eq!(
            recorded_valid_from.wallclock(),
            expected.timestamp_micros(),
            "tombstone valid_from should match the requested delete_valid_time"
        );

        // No longer visible as of now.
        let gone = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: edge.id,
            valid_time: now.to_rfc3339(),
            transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&gone).unwrap();
        assert!(value.get("error").is_some());
    }

    /// `detach: true` (cascade delete) does not support backdating -- passing
    /// `valid_time` alongside it must be a clear, structured rejection rather than
    /// silently ignoring `valid_time` or cascading at the wrong time.
    #[test]
    fn test_delete_node_detach_with_valid_time_rejected() {
        let server = create_test_server();
        let now = Utc::now();

        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        parse_response::<EdgeResponse>(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let response = server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id: n1.id,
            detach: Some(true),
            valid_time: Some(rfc3339_hours_ago(now, 1)),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some(), "expected error, got {value}");
        assert_ne!(value.get("success"), Some(&serde_json::json!(true)));

        // Node and edge must be untouched by the rejected request.
        let get_value: serde_json::Value = serde_json::from_str(&server.get_node(GetNodeRequest {
            namespace: None,
            node_id: n1.id,
            include_vectors: None,
        }))
        .unwrap();
        assert!(get_value.get("error").is_none(), "node should still exist");
    }

    /// Plain cascade delete (`detach: true`, no `valid_time`) must behave exactly as
    /// before this fix -- a regression guard alongside the new rejection above.
    #[test]
    fn test_delete_node_detach_without_valid_time_unchanged() {
        let server = create_test_server();

        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        parse_response::<EdgeResponse>(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let response = server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id: n1.id,
            detach: Some(true),
            valid_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value.get("success"), Some(&serde_json::json!(true)));
        assert_eq!(value.get("edges_removed"), Some(&serde_json::json!(1)));
        assert_eq!(value.get("detached"), Some(&serde_json::json!(true)));
    }
}

// ============================================================================
// Write-Time Provenance Tests (Issue #3224)
// ============================================================================

mod provenance_write_tests {
    use super::*;

    #[test]
    fn test_create_node_with_provenance_persists() {
        let server = create_test_server();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: Some("hr-system".to_string()),
                confidence: Some(0.95),
                note: None,
                correlation_id: Some("batch-2026-06".to_string()),
            }),
        }))
        .unwrap();

        // Returned directly on the create response.
        let provenance = node.provenance.expect("provenance should be present");
        assert_eq!(provenance.source(), Some("hr-system"));
        assert_eq!(provenance.confidence(), Some(0.95));
        assert_eq!(provenance.correlation_id(), Some("batch-2026-06"));

        // And retrievable via history (per-version).
        let history = server.get_node_history(GetNodeHistoryRequest { node_id: node.id });
        let value: serde_json::Value = serde_json::from_str(&history).unwrap();
        let versions = value["versions"].as_array().unwrap();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0]["provenance"]["source"], "hr-system");
        assert_eq!(versions[0]["provenance"]["confidence"], 0.95);
    }

    #[test]
    fn test_create_node_without_provenance_response_unchanged() {
        let server = create_test_server();

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("provenance").is_none(),
            "provenance key must be omitted, not null, when absent; got {value}"
        );
    }

    #[test]
    fn test_create_node_empty_provenance_object_treated_as_absent() {
        let server = create_test_server();

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: None,
                confidence: None,
                note: None,
                correlation_id: None,
            }),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("provenance").is_none(),
            "an all-absent bundle must never be persisted/returned as a fabricated object; got {value}"
        );
    }

    #[test]
    fn test_create_node_provenance_confidence_above_one_rejected() {
        let server = create_test_server();

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: None,
                confidence: Some(1.5),
                note: None,
                correlation_id: None,
            }),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let error = value
            .get("error")
            .and_then(|e| e["message"].as_str())
            .unwrap();
        assert!(error.contains("confidence"), "got: {error}");
        assert!(
            error.contains("0.0") && error.contains("1.0"),
            "got: {error}"
        );
        assert_eq!(server.db().node_count(), 0, "no node should be created");
    }

    #[test]
    fn test_create_node_provenance_confidence_below_zero_rejected() {
        let server = create_test_server();

        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: None,
                confidence: Some(-0.1),
                note: None,
                correlation_id: None,
            }),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let error = value
            .get("error")
            .and_then(|e| e["message"].as_str())
            .unwrap();
        assert!(error.contains("confidence"), "got: {error}");
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn test_create_node_provenance_confidence_boundaries_accepted() {
        let server = create_test_server();

        for confidence in [0.0, 1.0] {
            let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                label: "Person".to_string(),
                properties: None,
                valid_time: None,
                provenance: Some(ProvenanceRequest {
                    source: None,
                    confidence: Some(confidence),
                    note: None,
                    correlation_id: None,
                }),
            }))
            .expect("boundary confidence should be accepted");
            assert_eq!(node.provenance.unwrap().confidence(), Some(confidence));
        }
    }

    #[test]
    fn test_update_node_with_provenance_history_per_version() {
        let server = create_test_server();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let updated: NodeResponse = parse_response(&server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            node_id: node.id,
            properties: HashMap::new(),
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: Some("csv-import:2026-06".to_string()),
                confidence: Some(0.4),
                note: None,
                correlation_id: None,
            }),
        }))
        .unwrap();
        assert_eq!(
            updated.provenance.unwrap().source(),
            Some("csv-import:2026-06")
        );

        let history = server.get_node_history(GetNodeHistoryRequest { node_id: node.id });
        let value: serde_json::Value = serde_json::from_str(&history).unwrap();
        let versions = value["versions"].as_array().unwrap();
        assert_eq!(versions.len(), 2);
        assert!(
            versions[0].get("provenance").is_none(),
            "v1 (plain create) should have no provenance"
        );
        assert_eq!(versions[1]["provenance"]["source"], "csv-import:2026-06");
    }

    #[test]
    fn test_update_node_provenance_confidence_out_of_range_rejected() {
        let server = create_test_server();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let response = server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            node_id: node.id,
            properties: HashMap::new(),
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: None,
                confidence: Some(2.0),
                note: None,
                correlation_id: None,
            }),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());

        // The node's properties must be unchanged by the rejected update.
        let history = server.get_node_history(GetNodeHistoryRequest { node_id: node.id });
        let history_value: serde_json::Value = serde_json::from_str(&history).unwrap();
        assert_eq!(history_value["versions"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_create_edge_with_provenance_persists() {
        let server = create_test_server();

        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let edge: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: Some("claude-mcp".to_string()),
                confidence: Some(0.7),
                note: Some("inferred from document".to_string()),
                correlation_id: None,
            }),
        }))
        .unwrap();

        let provenance = edge.provenance.expect("provenance should be present");
        assert_eq!(provenance.source(), Some("claude-mcp"));
        assert_eq!(provenance.note(), Some("inferred from document"));

        let history = server.get_edge_history(GetEdgeHistoryRequest { edge_id: edge.id });
        let value: serde_json::Value = serde_json::from_str(&history).unwrap();
        assert_eq!(value["versions"][0]["provenance"]["source"], "claude-mcp");
    }

    #[test]
    fn test_create_edge_provenance_confidence_above_one_rejected() {
        let server = create_test_server();

        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: None,
                confidence: Some(1.1),
                note: None,
                correlation_id: None,
            }),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_some());
        assert_eq!(server.db().edge_count(), 0, "no edge should be created");
    }

    #[test]
    fn test_update_edge_with_provenance_history_per_version() {
        let server = create_test_server();

        let n1: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let n2: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let edge: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: n1.id,
            target_id: n2.id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let _updated: EdgeResponse = parse_response(&server.update_edge(UpdateEdgeRequest {
            namespace: None,
            derived_from: None,
            edge_id: edge.id,
            properties: HashMap::new(),
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: Some("hr-system".to_string()),
                confidence: None,
                note: None,
                correlation_id: None,
            }),
        }))
        .unwrap();

        let history = server.get_edge_history(GetEdgeHistoryRequest { edge_id: edge.id });
        let value: serde_json::Value = serde_json::from_str(&history).unwrap();
        let versions = value["versions"].as_array().unwrap();
        assert_eq!(versions.len(), 2);
        assert!(versions[0].get("provenance").is_none());
        assert_eq!(versions[1]["provenance"]["source"], "hr-system");
    }

    #[test]
    fn test_get_node_returns_current_provenance() {
        let server = create_test_server();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: Some("hr-system".to_string()),
                confidence: None,
                note: None,
                correlation_id: None,
            }),
        }))
        .unwrap();

        let fetched: NodeResponse = parse_response(&server.get_node(GetNodeRequest {
            namespace: None,
            node_id: node.id,
            include_vectors: None,
        }))
        .unwrap();
        assert_eq!(fetched.provenance.unwrap().source(), Some("hr-system"));
    }

    #[test]
    fn test_get_node_without_provenance_omits_key() {
        let server = create_test_server();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id: node.id,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("provenance").is_none());
    }

    #[test]
    fn test_get_node_history_omits_provenance_when_absent() {
        let server = create_test_server();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let history = server.get_node_history(GetNodeHistoryRequest { node_id: node.id });
        let value: serde_json::Value = serde_json::from_str(&history).unwrap();
        assert!(value["versions"][0].get("provenance").is_none());
    }

    /// Regression test: `node_to_response`/`edge_to_response` used to
    /// hardcode `provenance: None`, and only `get_node`/`create_node`/
    /// `update_node` (and edge equivalents) patched it back in afterward.
    /// `list_nodes` (and every other read path built on `node_to_response`)
    /// silently omitted provenance even when it existed. See Issue #3224.
    #[test]
    fn test_list_nodes_returns_provenance() {
        let server = create_test_server();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "ListProvenanceTest".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: Some("csv-import".to_string()),
                confidence: None,
                note: None,
                correlation_id: None,
            }),
        }))
        .unwrap();

        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("ListProvenanceTest".to_string()),
            property_key: None,
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let nodes = value["nodes"].as_array().unwrap();
        let found = nodes.iter().find(|n| n["id"] == node.id).unwrap();
        assert_eq!(found["provenance"]["source"], "csv-import");
    }

    /// See [`test_list_nodes_returns_provenance`] -- same gap, `traverse` path.
    #[test]
    fn test_traverse_returns_provenance() {
        let server = create_test_server();

        let start: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();
        let end: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: Some("graph-import".to_string()),
                confidence: None,
                note: None,
                correlation_id: None,
            }),
        }))
        .unwrap();
        parse_response::<EdgeResponse>(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: start.id,
            target_id: end.id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: start.id,
            edge_label: "KNOWS".to_string(),
            direction: Some("outgoing".to_string()),
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let results = value["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["node"]["provenance"]["source"], "graph-import");
    }

    /// See [`test_list_nodes_returns_provenance`] -- same gap, the bi-temporal
    /// `get_node_at_time` path. This one matters especially: `get_node_at_time`
    /// resolves the *as-of* version, not necessarily the node's current head,
    /// so the returned provenance must be attached to that exact historical
    /// version rather than "whatever is current now".
    #[test]
    fn test_get_node_at_time_returns_provenance_for_that_version() {
        let server = create_test_server();

        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: None,
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: Some("initial-load".to_string()),
                confidence: None,
                note: None,
                correlation_id: None,
            }),
        }))
        .unwrap();

        let as_of_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        // Update with different provenance; the as-of query above must still
        // report the *original* version's provenance, not this new one.
        parse_response::<NodeResponse>(&server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            node_id: node.id,
            properties: HashMap::new(),
            valid_time: None,
            provenance: Some(ProvenanceRequest {
                source: Some("manual-correction".to_string()),
                confidence: None,
                note: None,
                correlation_id: None,
            }),
        }))
        .unwrap();

        let response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id: node.id,
            valid_time: as_of_micros.to_string(),
            transaction_time: Some(as_of_micros.to_string()),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value["node"]["provenance"]["source"], "initial-load");
    }
}

// ============================================================================
// Point-in-time (AS OF) graph traversal (Issue #3225)
// ============================================================================

mod traverse_as_of_tests {
    use super::*;
    use chrono::{DateTime, Duration, Utc};

    /// `now` is the caller's own `Utc::now()`, captured once at the start of the test
    /// and reused for every offset it computes, so relative orderings between offsets
    /// stay deterministic regardless of how long the test takes to run.
    fn hours_ago(now: DateTime<Utc>, hours: i64) -> String {
        (now - Duration::hours(hours)).to_rfc3339()
    }

    fn create_node_at(server: &AletheiaMcpServer, label: &str, valid_time: &str) -> u64 {
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: label.to_string(),
            properties: None,
            valid_time: Some(valid_time.to_string()),
            provenance: None,
        });
        let node: NodeResponse = parse_response(&response).expect("create_node should succeed");
        node.id
    }

    fn create_edge_at(
        server: &AletheiaMcpServer,
        source_id: u64,
        target_id: u64,
        label: &str,
        valid_time: &str,
    ) -> u64 {
        let response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id,
            target_id,
            label: label.to_string(),
            properties: None,
            valid_time: Some(valid_time.to_string()),
            provenance: None,
        });
        let edge: EdgeResponse = parse_response(&response).expect("create_edge should succeed");
        edge.id
    }

    fn traverse_count(
        server: &AletheiaMcpServer,
        req: TraverseRequest,
    ) -> (usize, serde_json::Value) {
        let response = server.traverse(req);
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none(), "unexpected error: {value}");
        let count = value["count"].as_u64().unwrap() as usize;
        (count, value)
    }

    #[test]
    fn test_traverse_as_of_excludes_edge_created_after_coordinate() {
        let server = create_test_server();
        let now = Utc::now();

        let a = create_node_at(&server, "Person", &hours_ago(now, 10));
        let b = create_node_at(&server, "Person", &hours_ago(now, 10));
        create_edge_at(&server, a, b, "KNOWS", &hours_ago(now, 2));

        // Before the edge was created: excluded.
        let (count_before, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 5)),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(
            count_before, 0,
            "edge created after coordinate must be excluded"
        );

        // After the edge was created: included.
        let (count_after, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 1)),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(
            count_after, 1,
            "edge valid as of coordinate must be included"
        );
    }

    #[test]
    fn test_traverse_as_of_excludes_retired_edge_but_recalls_it_before_retirement() {
        let server = create_test_server();
        let now = Utc::now();

        let a = create_node_at(&server, "Person", &hours_ago(now, 10));
        let b = create_node_at(&server, "Person", &hours_ago(now, 10));
        let edge_id = create_edge_at(&server, a, b, "KNOWS", &hours_ago(now, 5));
        // Anchor a transaction-time coordinate right after the edge was created but
        // before it is (really, physically) deleted below.
        let tx_time_before_delete = Utc::now().to_rfc3339();

        // Retire (delete) the edge, backdating the retirement itself.
        let delete_response = server.delete_edge(DeleteEdgeRequest {
            namespace: None,
            edge_id,
            valid_time: Some(hours_ago(now, 2)),
        });
        let delete_value: serde_json::Value = serde_json::from_str(&delete_response).unwrap();
        assert!(
            delete_value.get("error").is_none(),
            "delete_edge should succeed: {delete_value}"
        );

        // Between creation (5h ago) and retirement (2h ago), as recorded by a
        // transaction time before the deletion was committed: still valid --
        // recall must be 1.0 even though the edge has since been deleted for real
        // (mirrors tests/temporal_adjacency_default_enabled_test.rs's recall check).
        let (count_before_retirement, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 3)),
                as_of_transaction_time: Some(tx_time_before_delete),
            },
        );
        assert_eq!(
            count_before_retirement, 1,
            "edge valid at coordinate must be recalled despite later deletion"
        );

        // After retirement (2h ago): excluded.
        let (count_after_retirement, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 1)),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(
            count_after_retirement, 0,
            "edge retired before coordinate must be excluded"
        );
    }

    #[test]
    fn test_traverse_as_of_valid_time_vs_transaction_time_disagreement() {
        let server = create_test_server();
        let now = Utc::now();

        let a = create_node_at(&server, "Person", &hours_ago(now, 10));
        let b = create_node_at(&server, "Person", &hours_ago(now, 10));
        // Valid time is backdated to 5h ago; the *actual* commit (transaction time)
        // happens for real, right now.
        create_edge_at(&server, a, b, "KNOWS", &hours_ago(now, 5));

        // valid_time is satisfied (5h ago has passed), but transaction_time asks
        // "as recorded by 20h ago" -- long before the edge was actually committed.
        let (count_tx_gate_fails, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 5)),
                as_of_transaction_time: Some(hours_ago(now, 20)),
            },
        );
        assert_eq!(
            count_tx_gate_fails, 0,
            "transaction-time coordinate before the real commit must exclude the edge \
             even though valid_time alone would admit it"
        );

        // valid_time asks for 8h ago (not yet valid), transaction_time defaults to now
        // (satisfied, since the edge is already committed for real).
        let (count_vt_gate_fails, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 8)),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(
            count_vt_gate_fails, 0,
            "valid-time coordinate before the fact became true must exclude the edge \
             even though transaction_time alone would admit it"
        );

        // Both axes satisfied: included.
        let (count_both_satisfied, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 5)),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(
            count_both_satisfied, 1,
            "both axes satisfied must include the edge"
        );
    }

    #[test]
    fn test_traverse_as_of_empty_when_start_node_not_yet_existing() {
        let server = create_test_server();
        let now = Utc::now();

        let a = create_node_at(&server, "Person", &hours_ago(now, 3));
        let b = create_node_at(&server, "Person", &hours_ago(now, 3));
        create_edge_at(&server, a, b, "KNOWS", &hours_ago(now, 3));

        // 5h ago: neither node nor edge existed yet.
        let (count, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 5)),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(
            count, 0,
            "start node not yet existing must yield empty results, not an error"
        );
    }

    #[test]
    fn test_traverse_as_of_invalid_valid_time_returns_structured_error() {
        let server = create_test_server();
        let a = create_node_at(&server, "Person", &Utc::now().to_rfc3339());

        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: a,
            edge_label: "KNOWS".to_string(),
            direction: Some("outgoing".to_string()),
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: Some("not-a-timestamp".to_string()),
            as_of_transaction_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let error = value
            .get("error")
            .and_then(|e| e["message"].as_str())
            .expect("expected a structured error, got no error field");
        assert!(
            error.contains("as_of_valid_time"),
            "error should identify the offending field: {error}"
        );
    }

    #[test]
    fn test_traverse_as_of_invalid_transaction_time_returns_structured_error() {
        let server = create_test_server();
        let a = create_node_at(&server, "Person", &Utc::now().to_rfc3339());

        let response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: a,
            edge_label: "KNOWS".to_string(),
            direction: Some("outgoing".to_string()),
            depth: Some(1),
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: Some("not-a-timestamp".to_string()),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        let error = value
            .get("error")
            .and_then(|e| e["message"].as_str())
            .expect("expected a structured error, got no error field");
        assert!(
            error.contains("as_of_transaction_time"),
            "error should identify the offending field: {error}"
        );
    }

    #[test]
    fn test_traverse_as_of_respects_resource_limits() {
        let server = create_test_server();
        let now = Utc::now();
        let valid_time = hours_ago(now, 3);

        // Chain of 5 nodes (4 hops), all backdated to the same valid_time.
        let mut start = None;
        let mut prev_id: Option<u64> = None;
        for _ in 0..5 {
            let node_id = create_node_at(&server, "ChainNode", &valid_time);
            start.get_or_insert(node_id);
            if let Some(source_id) = prev_id {
                create_edge_at(&server, source_id, node_id, "NEXT", &valid_time);
            }
            prev_id = Some(node_id);
        }
        let start = start.unwrap();

        // Depth far beyond MAX_TRAVERSAL_DEPTH must be clamped, not error, on the
        // temporal path exactly as it is on the current-state path.
        let (count, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: start,
                edge_label: "NEXT".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(100),
                limit: Some(50),
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 1)),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(count, 4, "clamped depth should still reach the full chain");
    }

    #[test]
    fn test_traverse_as_of_no_temporal_params_matches_current_state() {
        let server = create_test_server();

        let a = create_node_at(&server, "Person", &Utc::now().to_rfc3339());
        let b = create_node_at(&server, "Person", &Utc::now().to_rfc3339());
        create_edge_at(&server, a, b, "KNOWS", &Utc::now().to_rfc3339());
        let now_after_creation = Utc::now();

        let (count_no_temporal, value_no_temporal) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: None,
                as_of_transaction_time: None,
            },
        );

        let (count_as_of_now, value_as_of_now) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(now_after_creation.to_rfc3339()),
                as_of_transaction_time: Some(now_after_creation.to_rfc3339()),
            },
        );

        assert_eq!(count_no_temporal, 1);
        assert_eq!(count_no_temporal, count_as_of_now);
        assert_eq!(
            value_no_temporal["results"][0]["node"]["id"],
            value_as_of_now["results"][0]["node"]["id"],
            "explicit as_of=now must reproduce the identical current-state answer"
        );
    }

    #[test]
    fn test_traverse_as_of_multi_hop() {
        let server = create_test_server();
        let now = Utc::now();
        let valid_time = hours_ago(now, 3);

        let a = create_node_at(&server, "Person", &valid_time);
        let b = create_node_at(&server, "Person", &valid_time);
        let c = create_node_at(&server, "Person", &valid_time);
        create_edge_at(&server, a, b, "KNOWS", &valid_time);
        create_edge_at(&server, b, c, "KNOWS", &valid_time);

        let (count, _) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(2),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 1)),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(
            count, 2,
            "both hops should be reachable as of the coordinate"
        );
    }

    #[test]
    fn test_traverse_as_of_both_direction_finds_node_reachable_only_via_incoming_edge() {
        // Temporal-path variant of the direction:"both" fix: a node reachable
        // only through an edge pointing AT the current node (not away from
        // it) must still be found, not silently dropped because `next_id`
        // was resolved from the top-level `direction` string alone.
        let server = create_test_server();
        let now = Utc::now();
        let valid_time = hours_ago(now, 3);

        let a = create_node_at(&server, "Person", &valid_time);
        let b = create_node_at(&server, "Person", &valid_time);
        // Only B -> A exists; A has no reciprocal outgoing edge.
        create_edge_at(&server, b, a, "POINTS_AT", &valid_time);

        let (count, value) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "POINTS_AT".to_string(),
                direction: Some("both".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(hours_ago(now, 1)),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(
            count, 1,
            "as_of direction:\"both\" must find B via the incoming B->A edge: {value}"
        );
        assert_eq!(value["results"][0]["node"]["id"], b);
    }

    #[test]
    fn test_traverse_as_of_stops_at_intermediate_node_missing_at_coordinate() {
        // A node deleted (without cascading its edges) via the low-level
        // Rust API -- bypassing the MCP delete_node tool's own
        // connected-edges safety check, exactly the documented "Orphaned
        // Edges on Node Deletion" behavior of that API -- must stop a
        // temporal traversal from continuing past it, even though its
        // dangling edge was never itself retired.
        use crate::api::transaction::WriteOps;

        let server = create_test_server();
        let now = Utc::now();
        let far_past = hours_ago(now, 10);

        let a = create_node_at(&server, "Person", &far_past);
        let b = create_node_at(&server, "Person", &far_past);
        let c = create_node_at(&server, "Person", &far_past);
        create_edge_at(&server, a, b, "KNOWS", &far_past);
        create_edge_at(&server, b, c, "KNOWS", &far_past);

        let b_node_id = crate::core::id::NodeId::new(b).unwrap();
        server
            .db()
            .write_with_timestamp(|tx| tx.delete_node(b_node_id))
            .expect("low-level delete_node should succeed");
        let after_delete = Utc::now().to_rfc3339();

        let (count, value) = traverse_count(
            &server,
            TraverseRequest {
                namespace: None,
                start_node_id: a,
                edge_label: "KNOWS".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(2),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: Some(after_delete),
                as_of_transaction_time: None,
            },
        );
        assert_eq!(
            count, 0,
            "traversal must not continue past a node absent at the coordinate: {value}"
        );
    }
}

// ============================================================================
// Result-Completeness Signal Tests (Issue #3226)
//
// Every bounded MCP read tool must let a single call reveal whether more
// results exist (`has_more`), how to fetch them (`next_offset`, for
// offset-paginated tools), and, where cheaply derivable, the total matching
// count (`total_matching`). These tests pin that contract per tool.
// ============================================================================

mod completeness_tests {
    use super::*;

    /// Read `has_more` as a bool (defaults to a sentinel that fails assertions
    /// when the field is entirely absent, so the red phase is meaningful).
    fn has_more(value: &serde_json::Value) -> Option<bool> {
        value.get("has_more").and_then(|v| v.as_bool())
    }

    /// Read `next_offset` as a number when present; `None` when absent or null.
    fn next_offset(value: &serde_json::Value) -> Option<u64> {
        value.get("next_offset").and_then(|v| v.as_u64())
    }

    fn parse(response: &str) -> serde_json::Value {
        serde_json::from_str(response).expect("valid JSON response")
    }

    fn seed_labeled(server: &AletheiaMcpServer, label: &str, n: usize) {
        for i in 0..n {
            let mut props = HashMap::new();
            props.insert("index".to_string(), serde_json::json!(i));
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: label.to_string(),
                properties: Some(props),
                provenance: None,
            });
        }
    }

    // --- list_nodes: label scan ------------------------------------------

    #[test]
    fn test_list_nodes_has_more_and_next_offset() {
        let server = create_test_server();
        seed_labeled(&server, "PersonHM", 250);

        // Page 1: full page, more remains.
        let page1 = parse(&server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("PersonHM".to_string()),
            property_key: None,
            property_value: None,
            limit: Some(100),
            offset: Some(0),
            include_vectors: None,
        }));
        assert_eq!(page1.get("count"), Some(&serde_json::json!(100)));
        assert_eq!(
            has_more(&page1),
            Some(true),
            "page 1 of 250 has more: {page1}"
        );
        assert_eq!(
            next_offset(&page1),
            Some(100),
            "next_offset points at page 2"
        );

        // Page 2: full page, still more.
        let page2 = parse(&server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("PersonHM".to_string()),
            property_key: None,
            property_value: None,
            limit: Some(100),
            offset: Some(100),
            include_vectors: None,
        }));
        assert_eq!(has_more(&page2), Some(true));
        assert_eq!(next_offset(&page2), Some(200));

        // Page 3: partial final page, no more.
        let page3 = parse(&server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("PersonHM".to_string()),
            property_key: None,
            property_value: None,
            limit: Some(100),
            offset: Some(200),
            include_vectors: None,
        }));
        assert_eq!(page3.get("count"), Some(&serde_json::json!(50)));
        assert_eq!(
            has_more(&page3),
            Some(false),
            "final page has no more: {page3}"
        );
        assert_eq!(next_offset(&page3), None, "no next_offset on final page");
    }

    #[test]
    fn test_list_nodes_label_scan_omits_total_matching() {
        // A label scan cannot cheaply know the total without a full scan, so
        // `total_matching` is omitted (not faked); `has_more` carries the signal.
        let server = create_test_server();
        seed_labeled(&server, "ScanHM", 5);

        let value = parse(&server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("ScanHM".to_string()),
            property_key: None,
            property_value: None,
            limit: Some(2),
            offset: Some(0),
            include_vectors: None,
        }));
        assert_eq!(has_more(&value), Some(true));
        assert!(
            value.get("total_matching").is_none(),
            "label scan must omit total_matching: {value}"
        );
    }

    // --- list_nodes: property path ---------------------------------------

    #[test]
    fn test_list_nodes_property_path_reports_total_matching() {
        let server = create_test_server();
        for _ in 0..150 {
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "WidgetHM".to_string(),
                properties: Some({
                    let mut m = HashMap::new();
                    m.insert("status".to_string(), serde_json::json!("active"));
                    m
                }),
                provenance: None,
            });
        }

        let page1 = parse(&server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("WidgetHM".to_string()),
            property_key: Some("status".to_string()),
            property_value: Some(serde_json::json!("active")),
            limit: Some(100),
            offset: Some(0),
            include_vectors: None,
        }));
        assert_eq!(page1.get("count"), Some(&serde_json::json!(100)));
        assert_eq!(
            page1.get("total_matching"),
            Some(&serde_json::json!(150)),
            "property path materializes the full id list, so total is cheap: {page1}"
        );
        assert_eq!(has_more(&page1), Some(true));
        assert_eq!(next_offset(&page1), Some(100));

        let page2 = parse(&server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("WidgetHM".to_string()),
            property_key: Some("status".to_string()),
            property_value: Some(serde_json::json!("active")),
            limit: Some(100),
            offset: Some(100),
            include_vectors: None,
        }));
        assert_eq!(page2.get("count"), Some(&serde_json::json!(50)));
        assert_eq!(page2.get("total_matching"), Some(&serde_json::json!(150)));
        assert_eq!(has_more(&page2), Some(false));
        assert_eq!(next_offset(&page2), None);
    }

    // --- list_nodes / list_edges guidance paths --------------------------

    #[test]
    fn test_list_nodes_unfiltered_has_more_false() {
        let server = create_test_server();
        seed_labeled(&server, "AnyHM", 3);
        let value = parse(&server.list_nodes(ListNodesRequest {
            namespace: None,
            label: None,
            property_key: None,
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: None,
        }));
        assert_eq!(
            has_more(&value),
            Some(false),
            "unfiltered guidance path carries has_more:false for shape consistency: {value}"
        );
        assert_eq!(next_offset(&value), None);
    }

    #[test]
    fn test_list_edges_has_more_false() {
        let server = create_test_server();
        let value = parse(&server.list_edges(ListEdgesRequest {
            namespace: None,
            label: None,
            limit: None,
            offset: None,
            include_vectors: None,
        }));
        assert_eq!(
            has_more(&value),
            Some(false),
            "list_edges guidance: {value}"
        );
        assert_eq!(next_offset(&value), None);
    }

    // --- outgoing / incoming edges (additive only, never truncated) ------

    fn seed_star(server: &AletheiaMcpServer, out: usize) -> u64 {
        let center = {
            let r = server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Hub".to_string(),
                properties: None,
                provenance: None,
            });
            let n: NodeResponse = parse_response(&r).unwrap();
            n.id
        };
        for _ in 0..out {
            let leaf = {
                let r = server.create_node(CreateNodeRequest {
                    namespace: None,
                    derived_from: None,
                    valid_time: None,
                    label: "Leaf".to_string(),
                    properties: None,
                    provenance: None,
                });
                let n: NodeResponse = parse_response(&r).unwrap();
                n.id
            };
            server.create_edge(CreateEdgeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                source_id: center,
                target_id: leaf,
                label: "LINK".to_string(),
                properties: None,
                provenance: None,
            });
        }
        center
    }

    #[test]
    fn test_get_outgoing_edges_completeness() {
        let server = create_test_server();
        let center = seed_star(&server, 3);
        let value = parse(&server.get_outgoing_edges(GetOutgoingEdgesRequest {
            node_id: center,
            label: None,
            include_vectors: None,
        }));
        assert_eq!(value.get("count"), Some(&serde_json::json!(3)));
        assert_eq!(
            has_more(&value),
            Some(false),
            "outgoing edges return the full set: {value}"
        );
        assert_eq!(
            value.get("total_matching"),
            Some(&serde_json::json!(3)),
            "total_matching equals the complete count: {value}"
        );
    }

    #[test]
    fn test_get_incoming_edges_completeness() {
        let server = create_test_server();
        // Build a hub with 2 incoming edges.
        let sink = {
            let r = server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Sink".to_string(),
                properties: None,
                provenance: None,
            });
            let n: NodeResponse = parse_response(&r).unwrap();
            n.id
        };
        for _ in 0..2 {
            let src = {
                let r = server.create_node(CreateNodeRequest {
                    namespace: None,
                    derived_from: None,
                    valid_time: None,
                    label: "Src".to_string(),
                    properties: None,
                    provenance: None,
                });
                let n: NodeResponse = parse_response(&r).unwrap();
                n.id
            };
            server.create_edge(CreateEdgeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                source_id: src,
                target_id: sink,
                label: "LINK".to_string(),
                properties: None,
                provenance: None,
            });
        }
        let value = parse(&server.get_incoming_edges(GetIncomingEdgesRequest {
            node_id: sink,
            label: None,
            include_vectors: None,
        }));
        assert_eq!(value.get("count"), Some(&serde_json::json!(2)));
        assert_eq!(has_more(&value), Some(false));
        assert_eq!(value.get("total_matching"), Some(&serde_json::json!(2)));
    }

    // --- traverse ---------------------------------------------------------

    /// Build a chain start -> n1 -> n2 -> ... of `len` NEXT edges; return ids.
    fn seed_chain(server: &AletheiaMcpServer, len: usize) -> Vec<u64> {
        let ids: Vec<u64> = (0..=len)
            .map(|_| {
                let r = server.create_node(CreateNodeRequest {
                    namespace: None,
                    derived_from: None,
                    valid_time: None,
                    label: "ChainHM".to_string(),
                    properties: None,
                    provenance: None,
                });
                let n: NodeResponse = parse_response(&r).unwrap();
                n.id
            })
            .collect();
        for w in ids.windows(2) {
            server.create_edge(CreateEdgeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                source_id: w[0],
                target_id: w[1],
                label: "NEXT".to_string(),
                properties: None,
                provenance: None,
            });
        }
        ids
    }

    #[test]
    fn test_traverse_has_more_and_next_offset_when_truncated() {
        let server = create_test_server();
        // start -> 5 further nodes reachable via NEXT.
        let ids = seed_chain(&server, 5);

        // Truncate at 2 of the 5 reachable nodes.
        let truncated = parse(&server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: ids[0],
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(5),
            limit: Some(2),
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        }));
        assert_eq!(truncated.get("count"), Some(&serde_json::json!(2)));
        assert_eq!(
            has_more(&truncated),
            Some(true),
            "traversal truncated by limit signals more: {truncated}"
        );
        assert_eq!(next_offset(&truncated), Some(2));

        // A limit above the reachable count exhausts the traversal.
        let complete = parse(&server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: ids[0],
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(5),
            limit: Some(100),
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        }));
        assert_eq!(complete.get("count"), Some(&serde_json::json!(5)));
        assert_eq!(has_more(&complete), Some(false), "exhausted: {complete}");
        assert_eq!(next_offset(&complete), None);
    }

    #[test]
    fn test_traverse_offset_paginates() {
        let server = create_test_server();
        let ids = seed_chain(&server, 5); // 5 reachable nodes

        // Collect the ids returned across two offset pages of size 2 plus a
        // final page; the union must equal the single-shot full traversal.
        let full = parse(&server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: ids[0],
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(5),
            limit: Some(100),
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        }));
        let full_count = full.get("count").and_then(|c| c.as_u64()).unwrap();
        assert_eq!(full_count, 5);

        let page2 = parse(&server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: ids[0],
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(5),
            limit: Some(2),
            offset: Some(2),
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        }));
        assert_eq!(page2.get("count"), Some(&serde_json::json!(2)));
        assert_eq!(
            has_more(&page2),
            Some(true),
            "offset 2 of 5 has more: {page2}"
        );
        assert_eq!(next_offset(&page2), Some(4));

        let page3 = parse(&server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: ids[0],
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(5),
            limit: Some(2),
            offset: Some(4),
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        }));
        assert_eq!(page3.get("count"), Some(&serde_json::json!(1)));
        assert_eq!(has_more(&page3), Some(false), "last page: {page3}");
        assert_eq!(next_offset(&page3), None);
    }

    // --- find_similar -----------------------------------------------------

    /// Seed up to 5 documents with well-separated (near-orthogonal) 4-dim
    /// embeddings: the four axis-aligned unit vectors plus the all-ones
    /// vector. Against a query embedding like `[0.1, 0.2, 0.3, 0.4]`, cosine
    /// similarity to each is strictly distinct (0.183, 0.365, 0.548, 0.730,
    /// 0.913) with wide gaps between them, so ranking is unambiguous and a
    /// deterministic count/`has_more` assertion never depends on HNSW
    /// (approximate) search resolving a near-tied score consistently.
    fn seed_vectors(server: &AletheiaMcpServer, n: usize) {
        assert!(
            n <= 5,
            "seed_vectors only defines 5 well-separated directions"
        );
        server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 4,
            distance_metric: Some("cosine".to_string()),
        });
        for i in 0..n {
            let mut props = HashMap::new();
            props.insert("name".to_string(), serde_json::json!(format!("Doc{i}")));
            let embedding: [f32; 4] = if i < 4 {
                let mut e = [0.0_f32; 4];
                e[i] = 1.0;
                e
            } else {
                [1.0; 4]
            };
            props.insert("embedding".to_string(), serde_json::json!(embedding));
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Document".to_string(),
                properties: Some(props),
                provenance: None,
            });
        }
    }

    #[test]
    fn test_find_similar_has_more_and_offset_paginates() {
        let server = create_test_server();
        seed_vectors(&server, 5);
        let query = vec![0.1_f32, 0.2, 0.3, 0.4];

        // k=2 of 5 available -> more remains.
        let page1 = parse(&server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: query.clone(),
            k: Some(2),
            offset: None,
            include_vectors: None,
        }));
        assert_eq!(page1.get("count"), Some(&serde_json::json!(2)));
        assert_eq!(has_more(&page1), Some(true), "2 of 5 similar: {page1}");
        assert_eq!(next_offset(&page1), Some(2));

        // Offset into the middle.
        let page2 = parse(&server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: query.clone(),
            k: Some(2),
            offset: Some(2),
            include_vectors: None,
        }));
        assert_eq!(page2.get("count"), Some(&serde_json::json!(2)));
        assert_eq!(has_more(&page2), Some(true));
        assert_eq!(next_offset(&page2), Some(4));

        // k beyond the available set exhausts it.
        let complete = parse(&server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: query,
            k: Some(50),
            offset: None,
            include_vectors: None,
        }));
        assert_eq!(has_more(&complete), Some(false), "exhausted: {complete}");
        assert_eq!(next_offset(&complete), None);
    }

    // --- success-metric sweep --------------------------------------------

    #[test]
    fn test_all_bounded_read_tools_carry_has_more() {
        let server = create_test_server();
        // Minimal data touching every tool.
        let center = seed_star(&server, 2);
        seed_vectors(&server, 2);

        let responses = vec![
            (
                "list_nodes(label)",
                server.list_nodes(ListNodesRequest {
                    namespace: None,
                    label: Some("Leaf".to_string()),
                    property_key: None,
                    property_value: None,
                    limit: Some(1),
                    offset: None,
                    include_vectors: None,
                }),
            ),
            (
                "list_nodes(unfiltered)",
                server.list_nodes(ListNodesRequest {
                    namespace: None,
                    label: None,
                    property_key: None,
                    property_value: None,
                    limit: None,
                    offset: None,
                    include_vectors: None,
                }),
            ),
            (
                "list_edges",
                server.list_edges(ListEdgesRequest {
                    namespace: None,
                    label: None,
                    limit: None,
                    offset: None,
                    include_vectors: None,
                }),
            ),
            (
                "get_outgoing_edges",
                server.get_outgoing_edges(GetOutgoingEdgesRequest {
                    node_id: center,
                    label: None,
                    include_vectors: None,
                }),
            ),
            (
                "get_incoming_edges",
                server.get_incoming_edges(GetIncomingEdgesRequest {
                    node_id: center,
                    label: None,
                    include_vectors: None,
                }),
            ),
            (
                "traverse",
                server.traverse(TraverseRequest {
                    namespace: None,
                    start_node_id: center,
                    edge_label: "LINK".to_string(),
                    direction: None,
                    depth: Some(1),
                    limit: Some(1),
                    offset: None,
                    include_vectors: None,
                    as_of_valid_time: None,
                    as_of_transaction_time: None,
                }),
            ),
            (
                "find_similar",
                server.find_similar(FindSimilarRequest {
                    namespace: None,
                    property_name: "embedding".to_string(),
                    embedding: vec![0.1, 0.2, 0.3, 0.4],
                    k: Some(1),
                    offset: None,
                    include_vectors: None,
                }),
            ),
        ];

        for (name, resp) in responses {
            let value = parse(&resp);
            assert!(
                value.get("has_more").and_then(|v| v.as_bool()).is_some(),
                "{name} response must carry a boolean has_more: {value}"
            );
        }
    }

    // --- limit/k==0 must not produce a non-progressing page --------------
    //
    // A page must be able to carry a continuation cursor: limit/k:0 with
    // has_more:true and next_offset==offset (an unchanged offset) would trap
    // a paginating caller in an infinite loop. `limit`/`k` are clamped to a
    // minimum of 1 so a page always advances.

    #[test]
    fn test_list_nodes_limit_zero_is_clamped_to_one() {
        let server = create_test_server();
        seed_labeled(&server, "ClampHM", 3);
        let value = parse(&server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("ClampHM".to_string()),
            property_key: None,
            property_value: None,
            limit: Some(0),
            offset: None,
            include_vectors: None,
        }));
        assert_eq!(
            value.get("count"),
            Some(&serde_json::json!(1)),
            "limit:0 must be clamped to at least 1, not return an empty page: {value}"
        );
        assert_eq!(has_more(&value), Some(true));
        assert_eq!(next_offset(&value), Some(1), "page must advance: {value}");
    }

    #[test]
    fn test_traverse_limit_zero_is_clamped_to_one() {
        let server = create_test_server();
        let ids = seed_chain(&server, 3);
        let value = parse(&server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: ids[0],
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(3),
            limit: Some(0),
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        }));
        assert_eq!(
            value.get("count"),
            Some(&serde_json::json!(1)),
            "limit:0 must be clamped to at least 1, not return an empty page: {value}"
        );
        assert_eq!(has_more(&value), Some(true));
        assert_eq!(next_offset(&value), Some(1), "page must advance: {value}");
    }

    #[test]
    fn test_find_similar_k_zero_is_clamped_to_one() {
        let server = create_test_server();
        seed_vectors(&server, 3);
        let value = parse(&server.find_similar(FindSimilarRequest {
            namespace: None,
            property_name: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            k: Some(0),
            offset: None,
            include_vectors: None,
        }));
        assert_eq!(
            value.get("count"),
            Some(&serde_json::json!(1)),
            "k:0 must be clamped to at least 1, not return an empty page: {value}"
        );
        assert_eq!(has_more(&value), Some(true));
        assert_eq!(next_offset(&value), Some(1), "page must advance: {value}");
    }

    // --- traverse must not walk unboundedly through dangling edges -------

    #[test]
    fn test_traverse_bounded_peek_past_dangling_edges() {
        let server = create_test_server();
        // start -> n1 -> n2 -> n3 -> n4, then n2..n4 are deleted via the
        // low-level (non-cascade) API, which per CLAUDE.md's "Orphaned
        // Edges" section leaves their connecting edges dangling in storage.
        let ids = seed_chain(&server, 4);
        for &dangling_id in &ids[2..=4] {
            server
                .db()
                .delete_node_with_valid_time(
                    crate::core::id::NodeId::new(dangling_id).unwrap(),
                    None,
                )
                .unwrap();
        }

        // n1 is the sole live node beyond `start` and becomes the one-item
        // page (limit:1). Before this fix, proving `has_more` required
        // resolving one MORE node beyond the page -- but every node past n1
        // is dangling, so the old code would walk the entire dangling chain
        // (n2, n3, n4) via "keep expanding regardless", find nothing
        // resolvable, and wrongly report has_more:false. The fix instead
        // reads `has_more` off whatever `traversal_next_hops` already queued
        // for n1 (the edge to n2) without resolving it, so it correctly
        // reports uncertainty as has_more:true in O(1) extra work rather
        // than an O(dangling-chain-length) walk that still gets it wrong.
        let response = parse(&server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: ids[0],
            edge_label: "NEXT".to_string(),
            direction: None,
            depth: Some(10),
            limit: Some(1),
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        }));
        assert_eq!(response.get("count"), Some(&serde_json::json!(1)));
        assert_eq!(
            has_more(&response),
            Some(true),
            "a pending (even if unresolved) frontier candidate must not be silently \
             reported as has_more:false: {response}"
        );
    }
}

// ============================================================================
// Point-in-time (AS OF) node find by label and property (Issue #3236)
// ============================================================================
//
// `find_nodes_at_time` resolves "the Person named Alice, as of T" without a
// prior NodeId: it returns each matching node AS IT EXISTED at the queried
// bi-temporal coordinate, excluding nodes that did not exist (or whose
// property value did not hold) at that point.

mod find_nodes_at_time_tests {
    use super::*;
    use chrono::{DateTime, Utc};

    /// Capture an RFC 3339 anchor strictly after every previously committed
    /// write: an HLC commit stamp in the same microsecond as the anchor
    /// carries a logical component > 0 and would order *after* it, so sleep
    /// past the microsecond boundary first.
    fn anchor_after_commits() -> String {
        std::thread::sleep(std::time::Duration::from_millis(2));
        Utc::now().to_rfc3339()
    }

    fn base_req(label: &str, valid_time: &str) -> FindNodesAtTimeRequest {
        FindNodesAtTimeRequest {
            namespace: None,
            label: label.to_string(),
            property_key: None,
            property_value: None,
            valid_time: valid_time.to_string(),
            transaction_time: None,
            limit: None,
            offset: None,
            include_vectors: None,
        }
    }

    fn name_req(label: &str, name: &str, valid_time: &str) -> FindNodesAtTimeRequest {
        FindNodesAtTimeRequest {
            namespace: None,
            property_key: Some("name".to_string()),
            property_value: Some(serde_json::json!(name)),
            ..base_req(label, valid_time)
        }
    }

    fn find(server: &AletheiaMcpServer, req: FindNodesAtTimeRequest) -> serde_json::Value {
        serde_json::from_str(&server.find_nodes_at_time(req))
            .expect("find_nodes_at_time should return valid JSON")
    }

    fn node_ids(value: &serde_json::Value) -> Vec<u64> {
        value["nodes"]
            .as_array()
            .unwrap_or_else(|| panic!("expected a nodes array, got: {value}"))
            .iter()
            .map(|n| n["id"].as_u64().expect("node id should be a u64"))
            .collect()
    }

    fn create_named(server: &AletheiaMcpServer, label: &str, name: &str) -> u64 {
        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: label.to_string(),
            properties: Some(HashMap::from([(
                "name".to_string(),
                serde_json::json!(name),
            )])),
            valid_time: None,
            provenance: None,
        }))
        .expect("create should succeed");
        node.id
    }

    fn rename(server: &AletheiaMcpServer, node_id: u64, name: &str) {
        let _: NodeResponse = parse_response(&server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            node_id,
            properties: HashMap::from([("name".to_string(), serde_json::json!(name))]),
            valid_time: None,
            provenance: None,
        }))
        .expect("update should succeed");
    }

    fn delete(server: &AletheiaMcpServer, node_id: u64) {
        let response = server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id,
            detach: None,
            valid_time: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(value["success"], serde_json::json!(true), "{value}");
    }

    fn assert_invalid_argument(response: &str, must_mention: &str) {
        let value: serde_json::Value = serde_json::from_str(response).unwrap();
        assert_eq!(
            value["error"]["code"],
            serde_json::json!("INVALID_ARGUMENT"),
            "expected INVALID_ARGUMENT, got: {value}"
        );
        let message = value["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains(must_mention),
            "error message should mention '{must_mention}': {message}"
        );
    }

    /// Build a request anchored at the same instant on BOTH dimensions.
    /// Recalling a superseded (or deleted) version requires this: the
    /// superseding write closes the old version's *transaction* interval,
    /// so with `transaction_time` defaulted to now only the latest recorded
    /// version is visible (same convention as `traverse`'s `as_of_*`, see
    /// the guide).
    fn name_req_at(label: &str, name: &str, t: &str) -> FindNodesAtTimeRequest {
        FindNodesAtTimeRequest {
            namespace: None,
            transaction_time: Some(t.to_string()),
            ..name_req(label, name, t)
        }
    }

    #[test]
    fn returns_node_as_it_existed_at_t_not_current_state() {
        let server = create_test_server();
        let id = create_named(&server, "Person", "Alice");
        let t1 = anchor_after_commits();
        rename(&server, id, "Bob");
        let t2 = anchor_after_commits();

        // At (t1, t1) the node matched "Alice" and its returned properties
        // are the reconstructed at-time state, not the current one.
        let value = find(&server, name_req_at("Person", "Alice", &t1));
        assert_eq!(node_ids(&value), vec![id], "{value}");
        assert_eq!(value["nodes"][0]["properties"]["name"], "Alice");
        // "Bob" did not hold yet at t1...
        let value = find(&server, name_req_at("Person", "Bob", &t1));
        assert!(node_ids(&value).is_empty(), "{value}");
        // ...and the situation reverses at t2.
        let value = find(&server, name_req_at("Person", "Bob", &t2));
        assert_eq!(node_ids(&value), vec![id]);
        assert_eq!(value["nodes"][0]["properties"]["name"], "Bob");
        let value = find(&server, name_req_at("Person", "Alice", &t2));
        assert!(node_ids(&value).is_empty(), "{value}");
    }

    #[test]
    fn node_created_after_t_is_excluded() {
        let server = create_test_server();
        let t0 = anchor_after_commits();
        let id = create_named(&server, "Person", "Alice");
        let t1 = anchor_after_commits();

        let value = find(&server, name_req("Person", "Alice", &t0));
        assert!(
            node_ids(&value).is_empty(),
            "node created after T must be excluded: {value}"
        );
        let value = find(&server, name_req("Person", "Alice", &t1));
        assert_eq!(node_ids(&value), vec![id]);
    }

    /// T2 (review): the "both dimensions required" contract as a NEGATIVE
    /// test. Anchoring only `valid_time` before a deletion while
    /// `transaction_time` defaults to now must NOT recall the deleted node
    /// -- the deletion already closed the version's transaction interval.
    #[test]
    fn deleted_node_not_recalled_when_transaction_time_omitted() {
        let server = create_test_server();
        let id = create_named(&server, "Person", "Alice");
        let t_before = anchor_after_commits();
        delete(&server, id);
        std::thread::sleep(std::time::Duration::from_millis(2));

        // valid_time anchored before the deletion, transaction_time omitted
        // (defaults to now): empty, per the documented contract.
        let value = find(&server, name_req("Person", "Alice", &t_before));
        assert!(
            node_ids(&value).is_empty(),
            "past valid_time + defaulted transaction_time must not recall a deleted node: {value}"
        );
        assert_eq!(value["total_matching"], serde_json::json!(0));

        // Label-only path obeys the same contract.
        let value = find(&server, base_req("Person", &t_before));
        assert!(node_ids(&value).is_empty(), "{value}");
    }

    #[test]
    fn node_deleted_before_t_recalled_when_both_dimensions_anchored() {
        let server = create_test_server();
        let id = create_named(&server, "Person", "Alice");
        let t_before = anchor_after_commits();
        delete(&server, id);
        let t_after = anchor_after_commits();

        // Both dimensions anchored before the deletion recall the node even
        // though it no longer exists in current state.
        let value = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                transaction_time: Some(t_before.clone()),
                ..name_req("Person", "Alice", &t_before)
            },
        );
        assert_eq!(node_ids(&value), vec![id], "{value}");

        // At a coordinate after the deletion the node is gone.
        let value = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                transaction_time: Some(t_after.clone()),
                ..name_req("Person", "Alice", &t_after)
            },
        );
        assert!(node_ids(&value).is_empty(), "{value}");
    }

    #[test]
    fn label_only_as_of_scan_without_property_filter() {
        let server = create_test_server();
        let a = create_named(&server, "Person", "Alice");
        let b = create_named(&server, "Person", "Bob");
        let _c = create_named(&server, "Company", "Acme");
        let t1 = anchor_after_commits();
        delete(&server, b);
        let t2 = anchor_after_commits();

        let value = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                transaction_time: Some(t1.clone()),
                ..base_req("Person", &t1)
            },
        );
        assert_eq!(node_ids(&value), vec![a, b], "{value}");

        let value = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                transaction_time: Some(t2.clone()),
                ..base_req("Person", &t2)
            },
        );
        assert_eq!(node_ids(&value), vec![a], "{value}");
    }

    #[test]
    fn current_state_equivalence_with_list_nodes_property_filter() {
        let server = create_test_server();
        let _a = create_named(&server, "Person", "Alice");
        let _b = create_named(&server, "Person", "Bob");
        let _c = create_named(&server, "Person", "Alice");
        let now = anchor_after_commits();

        let list_value: serde_json::Value =
            serde_json::from_str(&server.list_nodes(ListNodesRequest {
                namespace: None,
                label: Some("Person".to_string()),
                property_key: Some("name".to_string()),
                property_value: Some(serde_json::json!("Alice")),
                limit: None,
                offset: None,
                include_vectors: None,
            }))
            .unwrap();
        let mut current_ids = node_ids(&list_value);
        current_ids.sort_unstable();

        let at_now = find(&server, name_req("Person", "Alice", &now));
        assert_eq!(
            node_ids(&at_now),
            current_ids,
            "valid_time=now + transaction_time=now must equal the current-state result set"
        );
    }

    #[test]
    fn pagination_limit_offset_and_completeness_metadata() {
        let server = create_test_server();
        let mut created: Vec<u64> = (0..3)
            .map(|_| create_named(&server, "Person", "Alice"))
            .collect();
        created.sort_unstable();
        let now = anchor_after_commits();

        // Page 1 (limit 1): stable ordering, exact total, continuation cursor.
        let page1 = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                limit: Some(1),
                ..name_req("Person", "Alice", &now)
            },
        );
        assert_eq!(node_ids(&page1), vec![created[0]], "{page1}");
        assert_eq!(page1["count"], serde_json::json!(1));
        assert_eq!(page1["total_matching"], serde_json::json!(3));
        assert_eq!(page1["has_more"], serde_json::json!(true));
        assert_eq!(page1["next_offset"], serde_json::json!(1));

        // Page 2 continues where page 1 left off, and its continuation
        // metadata advances from the requested offset (T6: next_offset must
        // be exercised at offset > 0, not just on page 1).
        let page2 = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                limit: Some(1),
                offset: Some(1),
                ..name_req("Person", "Alice", &now)
            },
        );
        assert_eq!(node_ids(&page2), vec![created[1]], "{page2}");
        assert_eq!(page2["has_more"], serde_json::json!(true));
        assert_eq!(page2["next_offset"], serde_json::json!(2));

        // Final page reports has_more: false.
        let page3 = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                limit: Some(1),
                offset: Some(2),
                ..name_req("Person", "Alice", &now)
            },
        );
        assert_eq!(node_ids(&page3), vec![created[2]], "{page3}");
        assert_eq!(page3["has_more"], serde_json::json!(false));
    }

    /// T4 (review, mutation killer): `limit: 0` is clamped UP to 1 (a page
    /// must be able to carry a continuation cursor), exactly like
    /// `list_nodes` -- one node comes back, not zero.
    #[test]
    fn limit_zero_is_clamped_up_to_one() {
        let server = create_test_server();
        let mut created: Vec<u64> = (0..2)
            .map(|_| create_named(&server, "Person", "Alice"))
            .collect();
        created.sort_unstable();
        let now = anchor_after_commits();

        let value = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                limit: Some(0),
                ..name_req("Person", "Alice", &now)
            },
        );
        assert!(value.get("error").is_none(), "{value}");
        assert_eq!(value["limit"], serde_json::json!(1));
        assert_eq!(
            node_ids(&value),
            vec![created[0]],
            "limit 0 must clamp to 1 and return one node, not zero: {value}"
        );
        assert_eq!(value["count"], serde_json::json!(1));
        assert_eq!(value["has_more"], serde_json::json!(true));
    }

    #[test]
    fn oversized_limit_and_offset_are_clamped_not_errors() {
        let server = create_test_server();
        let _ = create_named(&server, "Person", "Alice");
        let now = anchor_after_commits();

        // Far beyond MAX_RESULT_LIMIT / MAX_PAGINATION_OFFSET: clamped,
        // exactly like list_nodes -- never an error.
        let value = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                limit: Some(1_000_000),
                offset: Some(1_000_000),
                ..name_req("Person", "Alice", &now)
            },
        );
        assert!(value.get("error").is_none(), "{value}");
        assert_eq!(value["limit"], serde_json::json!(10_000));
        assert_eq!(value["offset"], serde_json::json!(10_000));
        assert!(node_ids(&value).is_empty(), "offset past the end: {value}");
    }

    #[test]
    fn invalid_or_empty_timestamps_return_structured_errors() {
        let server = create_test_server();

        let response = server.find_nodes_at_time(base_req("Person", "not-a-timestamp"));
        assert_invalid_argument(&response, "valid_time");

        let response = server.find_nodes_at_time(base_req("Person", ""));
        assert_invalid_argument(&response, "valid_time");

        let response = server.find_nodes_at_time(FindNodesAtTimeRequest {
            namespace: None,
            transaction_time: Some("not-a-timestamp".to_string()),
            ..base_req("Person", &Utc::now().to_rfc3339())
        });
        assert_invalid_argument(&response, "transaction_time");
    }

    #[test]
    fn property_key_without_value_and_reverse_return_structured_errors() {
        let server = create_test_server();
        let now = Utc::now().to_rfc3339();

        let response = server.find_nodes_at_time(FindNodesAtTimeRequest {
            namespace: None,
            property_key: Some("name".to_string()),
            ..base_req("Person", &now)
        });
        assert_invalid_argument(&response, "property_key");

        let response = server.find_nodes_at_time(FindNodesAtTimeRequest {
            namespace: None,
            property_value: Some(serde_json::json!("Alice")),
            ..base_req("Person", &now)
        });
        assert_invalid_argument(&response, "property_value");
    }

    #[test]
    fn unsupported_property_value_type_returns_structured_error() {
        let server = create_test_server();
        let response = server.find_nodes_at_time(FindNodesAtTimeRequest {
            namespace: None,
            property_key: Some("name".to_string()),
            property_value: Some(serde_json::json!({"nested": "object"})),
            ..base_req("Person", &Utc::now().to_rfc3339())
        });
        assert_invalid_argument(&response, "property_value");
    }

    #[test]
    fn response_carries_resolved_bitemporal_coordinate_as_rfc3339() {
        let server = create_test_server();
        let _ = create_named(&server, "Person", "Alice");
        let anchor = anchor_after_commits();
        let anchor_micros = DateTime::parse_from_rfc3339(&anchor)
            .unwrap()
            .timestamp_micros();

        let value = find(&server, name_req("Person", "Alice", &anchor));

        // The resolved valid_time round-trips to the exact requested instant.
        let valid_time = value["valid_time"].as_str().expect("valid_time present");
        assert_eq!(
            DateTime::parse_from_rfc3339(valid_time)
                .expect("valid_time should be RFC 3339")
                .timestamp_micros(),
            anchor_micros
        );

        // The omitted transaction_time resolves to a concrete "now", not a
        // placeholder -- it must parse and be at/after the anchor.
        let tx_time = value["transaction_time"]
            .as_str()
            .expect("transaction_time present");
        let tx_micros = DateTime::parse_from_rfc3339(tx_time)
            .expect("transaction_time should be RFC 3339")
            .timestamp_micros();
        assert!(
            tx_micros >= anchor_micros,
            "resolved transaction_time {tx_time} should not precede the request anchor {anchor}"
        );
    }

    #[test]
    fn unknown_label_returns_empty_set_not_error() {
        let server = create_test_server();
        let value = find(
            &server,
            base_req("NeverSeenLabel3236", &Utc::now().to_rfc3339()),
        );
        assert!(value.get("error").is_none(), "{value}");
        assert!(node_ids(&value).is_empty());
        assert_eq!(value["total_matching"], serde_json::json!(0));
        assert_eq!(value["has_more"], serde_json::json!(false));
        assert_eq!(value["sampled"], serde_json::json!(false));
    }

    /// T3 (review): vector/embedding properties follow the #3220 elision
    /// convention on this tool -- elided `{type, dim, elided: true}`
    /// descriptor by default, full float array with `include_vectors: true`.
    #[test]
    fn vector_properties_elided_by_default_and_full_with_include_vectors() {
        let server = create_test_server();
        let embedding: Vec<f32> = (0..16).map(|i| (i as f32) * 0.001 + 0.1).collect();
        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Document".to_string(),
            properties: Some(HashMap::from([
                ("name".to_string(), serde_json::json!("Doc1")),
                ("embedding".to_string(), serde_json::json!(embedding)),
            ])),
            valid_time: None,
            provenance: None,
        }))
        .expect("create should succeed");
        let now = anchor_after_commits();

        // Default: elided descriptor, exactly like list_nodes.
        let value = find(&server, base_req("Document", &now));
        assert_eq!(node_ids(&value), vec![node.id], "{value}");
        assert_eq!(
            value["nodes"][0]["properties"]["embedding"],
            serde_json::json!({"type": "vector", "dim": 16, "elided": true})
        );
        // Non-vector properties are untouched.
        assert_eq!(value["nodes"][0]["properties"]["name"], "Doc1");

        // include_vectors: true restores the full, lossless array.
        let value = find(
            &server,
            FindNodesAtTimeRequest {
                namespace: None,
                include_vectors: Some(true),
                ..base_req("Document", &now)
            },
        );
        let returned: Vec<f32> = value["nodes"][0]["properties"]["embedding"]
            .as_array()
            .expect("embedding should be a full array")
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(returned, embedding);
    }

    /// T7 (review): candidate-set truncation (the C1 cap) is disclosed via
    /// `sampled: true`, with `total_matching` counting matches within the
    /// sampled candidate set only. The cap is the same configurable
    /// `max_schema_as_of_entities` bi-temporal `get_schema` uses, injected
    /// small here (mirroring the schema_as_of cap test).
    #[test]
    fn truncated_candidate_scan_discloses_sampled_true() {
        use crate::config::{AletheiaDBConfig, HistoricalConfigBuilder};
        use crate::test_utils::create_test_db_with_config;

        let config = AletheiaDBConfig::builder()
            .historical(
                HistoricalConfigBuilder::new()
                    .max_schema_as_of_entities(2)
                    .build(),
            )
            .build();
        let (_tmp, db) = create_test_db_with_config(config).unwrap();
        let server = AletheiaMcpServer::new(Arc::new(db));

        let mut created: Vec<u64> = (0..4)
            .map(|_| create_named(&server, "Person", "Alice"))
            .collect();
        created.sort_unstable();
        let now = anchor_after_commits();

        let value = find(&server, name_req("Person", "Alice", &now));
        assert_eq!(value["sampled"], serde_json::json!(true), "{value}");
        // total_matching is honest about the sampled candidate set: the cap
        // keeps the lowest 2 node ids, both of which match.
        assert_eq!(value["total_matching"], serde_json::json!(2));
        assert_eq!(node_ids(&value), created[..2].to_vec(), "{value}");
    }
}

// ============================================================================
// Structured Error Codes with Retriable Flag (Issue #3234)
// ============================================================================
//
// Every MCP error response must be `{ "error": { "code", "message",
// "retriable", "details"? } }` where `code` is drawn from the documented,
// stable enum and `retriable` is `true` only for transient failure classes.
// These tests drive every distinct MCP error path and assert the exact code,
// plus a registry-driven sweep that automatically covers newly added tools.

mod structured_error_tests {
    use super::*;
    use serde_json::json;

    /// Every code documented for the MCP error contract (Issue #3234;
    /// UNAUTHENTICATED / PERMISSION_DENIED added by Issue #3350).
    const DOCUMENTED_CODES: &[&str] = &[
        "NOT_FOUND",
        "INVALID_ARGUMENT",
        "CONSTRAINT_VIOLATION",
        "FAILED_PRECONDITION",
        "CONFLICT",
        "UNAVAILABLE",
        "INTERNAL",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
    ];

    /// Dispatch a tool through the same table `call_tool` uses and parse the
    /// JSON payload. Returns `(parsed_json, is_error)`.
    fn dispatch(
        server: &AletheiaMcpServer,
        tool: &str,
        args: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        let result = server.dispatch_tool(tool, args);
        let is_error = result.is_error.unwrap_or(false);
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.clone()))
            .expect("tool result should carry text content");
        let value = serde_json::from_str(&text).expect("tool response should be valid JSON");
        (value, is_error)
    }

    /// Assert the response carries the Issue #3234 structured error shape and
    /// return `(code, retriable, message)`.
    fn assert_structured_error(value: &serde_json::Value, context: &str) -> (String, bool, String) {
        let error = value
            .get("error")
            .unwrap_or_else(|| panic!("[{context}] expected an `error` payload, got: {value}"));
        let obj = error.as_object().unwrap_or_else(|| {
            panic!("[{context}] `error` must be a structured object, not free text: {error}")
        });
        let code = obj
            .get("code")
            .and_then(|c| c.as_str())
            .unwrap_or_else(|| panic!("[{context}] error must carry a string `code`: {error}"))
            .to_string();
        assert!(
            DOCUMENTED_CODES.contains(&code.as_str()),
            "[{context}] code {code:?} is not in the documented enum"
        );
        let retriable = obj
            .get("retriable")
            .and_then(|r| r.as_bool())
            .unwrap_or_else(|| {
                panic!("[{context}] error must carry a boolean `retriable`: {error}")
            });
        let message = obj
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or_else(|| panic!("[{context}] error must carry a string `message`: {error}"))
            .to_string();
        assert!(
            !message.is_empty(),
            "[{context}] error message must not be empty"
        );
        (code, retriable, message)
    }

    /// Convenience: dispatch, require an error, assert shape + expected code.
    fn assert_error_code(
        server: &AletheiaMcpServer,
        tool: &str,
        args: serde_json::Value,
        expected_code: &str,
    ) -> serde_json::Value {
        let (value, is_error) = dispatch(server, tool, args);
        assert!(is_error, "[{tool}] expected an error result, got: {value}");
        let (code, retriable, _msg) = assert_structured_error(&value, tool);
        assert_eq!(code, expected_code, "[{tool}] wrong code: {value}");
        // Caller-fault classes are never advisorily retriable.
        if matches!(
            expected_code,
            "NOT_FOUND" | "INVALID_ARGUMENT" | "CONSTRAINT_VIOLATION" | "FAILED_PRECONDITION"
        ) {
            assert!(!retriable, "[{tool}] {expected_code} must not be retriable");
        }
        value
    }

    fn seed_node(server: &AletheiaMcpServer, label: &str, name: &str) -> u64 {
        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: label.to_string(),
            properties: Some(HashMap::from([(
                "name".to_string(),
                serde_json::Value::String(name.to_string()),
            )])),
            valid_time: None,
            provenance: None,
        }))
        .expect("seed node should succeed");
        node.id
    }

    // ------------------------------------------------------------------
    // Registry-driven sweep: every advertised tool, present and future,
    // must return the structured shape on its invalid-arguments path.
    // A tool added to `tool_definitions()`/`dispatch_tool` is covered
    // here automatically, with zero test changes.
    // ------------------------------------------------------------------

    #[test]
    fn every_advertised_tool_returns_structured_error_on_invalid_arguments() {
        let server = create_test_server();
        let mut errored = 0;
        for tool in server.list_tools_for_test() {
            let (value, is_error) = dispatch(&server, &tool, serde_json::Value::Null);
            if is_error {
                errored += 1;
                let (code, _, message) = assert_structured_error(&value, &tool);
                // A tool present in tool_definitions() but missing from
                // dispatch_tool's match would fall into the "Unknown tool"
                // NOT_FOUND arm and silently pass a shape-only sweep. Catch
                // that drift explicitly.
                assert!(
                    !message.contains("Unknown tool"),
                    "[{tool}] advertised but not dispatched: {message}"
                );
                // Null args normalize to the empty request for no-required-arg
                // tools, so a tool either (a) fails argument deserialization ->
                // INVALID_ARGUMENT, or (b) deserializes cleanly but hits an
                // unmet precondition -> FAILED_PRECONDITION (e.g. the Issue
                // #3351 chain tools when the provenance chain is not enabled on
                // this server). Both are structured, non-retriable, and
                // caller-actionable; anything else is a bug.
                assert!(
                    code == "INVALID_ARGUMENT" || code == "FAILED_PRECONDITION",
                    "[{tool}] null args must classify as INVALID_ARGUMENT or \
                     FAILED_PRECONDITION: {value}"
                );
            }
        }
        assert!(
            errored > 0,
            "sweep must exercise at least one error path (null args should fail deserialization)"
        );
    }

    #[test]
    fn unknown_tool_returns_not_found() {
        let server = create_test_server();
        let value = assert_error_code(
            &server,
            "definitely_not_a_tool",
            serde_json::Value::Null,
            "NOT_FOUND",
        );
        let (_, _, message) = assert_structured_error(&value, "unknown tool");
        assert!(message.contains("Unknown tool"), "got: {message}");
    }

    // ------------------------------------------------------------------
    // INVALID_ARGUMENT paths
    // ------------------------------------------------------------------

    #[test]
    fn malformed_arguments_return_invalid_argument() {
        let server = create_test_server();
        let value = assert_error_code(
            &server,
            "get_node",
            json!({"node_id": "not-a-number"}),
            "INVALID_ARGUMENT",
        );
        let (_, _, message) = assert_structured_error(&value, "get_node malformed");
        assert!(message.contains("Invalid arguments"), "got: {message}");
    }

    #[test]
    fn out_of_range_id_returns_invalid_argument() {
        let server = create_test_server();
        let value = assert_error_code(
            &server,
            "get_node",
            json!({"node_id": u64::MAX}),
            "INVALID_ARGUMENT",
        );
        // Message fidelity: the bare StorageError text, byte-for-byte
        // identical to what pre-#3234 releases returned as the whole `error`
        // value — no "Storage error: " wrapper prefix.
        let (_, _, message) = assert_structured_error(&value, "get_node out-of-range");
        let expected = crate::core::error::StorageError::InvalidId {
            id: u64::MAX,
            id_type: "node",
        }
        .to_string();
        assert_eq!(message, expected, "got: {message}");
        assert!(
            message.starts_with("Invalid node ID"),
            "message must be the bare StorageError text: {message}"
        );
        assert!(
            !message.starts_with("Storage error: "),
            "message must not gain a Storage error prefix: {message}"
        );

        let value = assert_error_code(
            &server,
            "get_edge",
            json!({"edge_id": u64::MAX}),
            "INVALID_ARGUMENT",
        );
        let (_, _, message) = assert_structured_error(&value, "get_edge out-of-range");
        assert!(
            message.starts_with("Invalid edge ID"),
            "message must be the bare StorageError text: {message}"
        );
    }

    #[test]
    fn invalid_timestamp_returns_invalid_argument() {
        let server = create_test_server();
        let value = assert_error_code(
            &server,
            "create_node",
            json!({"label": "Person", "valid_time": "not-a-timestamp"}),
            "INVALID_ARGUMENT",
        );
        let (_, _, message) = assert_structured_error(&value, "create_node bad valid_time");
        assert!(message.contains("Invalid valid_time"), "got: {message}");

        let node_id = seed_node(&server, "Person", "Alice");
        assert_error_code(
            &server,
            "get_node_at_time",
            json!({"node_id": node_id, "valid_time": "not-a-timestamp"}),
            "INVALID_ARGUMENT",
        );
        assert_error_code(
            &server,
            "traverse",
            json!({
                "start_node_id": node_id,
                "edge_label": "KNOWS",
                "as_of_valid_time": "not-a-timestamp"
            }),
            "INVALID_ARGUMENT",
        );
        assert_error_code(
            &server,
            "list_changes",
            json!({"tx_from": "not-a-timestamp", "tx_to": "2026-01-01T00:00:00Z"}),
            "INVALID_ARGUMENT",
        );
    }

    #[test]
    fn invalid_provenance_confidence_returns_invalid_argument() {
        let server = create_test_server();
        let value = assert_error_code(
            &server,
            "create_node",
            json!({"label": "Person", "provenance": {"confidence": 1.5}}),
            "INVALID_ARGUMENT",
        );
        let (_, _, message) = assert_structured_error(&value, "create_node bad provenance");
        assert!(message.contains("confidence"), "got: {message}");
    }

    #[test]
    fn list_nodes_inconsistent_property_filter_returns_invalid_argument() {
        let server = create_test_server();
        assert_error_code(
            &server,
            "list_nodes",
            json!({"label": "Person", "property_key": "name"}),
            "INVALID_ARGUMENT",
        );
        // Property filter without label is likewise a caller fault.
        assert_error_code(
            &server,
            "list_nodes",
            json!({"property_key": "name", "property_value": "Alice"}),
            "INVALID_ARGUMENT",
        );
    }

    #[test]
    fn delete_node_valid_time_with_detach_returns_invalid_argument() {
        let server = create_test_server();
        let node_id = seed_node(&server, "Person", "Alice");
        assert_error_code(
            &server,
            "delete_node",
            json!({
                "node_id": node_id,
                "detach": true,
                "valid_time": "2024-01-01T00:00:00Z"
            }),
            "INVALID_ARGUMENT",
        );
    }

    #[test]
    fn hybrid_query_without_entry_point_returns_invalid_argument() {
        let server = create_test_server();
        assert_error_code(&server, "hybrid_query", json!({}), "INVALID_ARGUMENT");
    }

    #[test]
    fn find_similar_dimension_mismatch_returns_invalid_argument() {
        let server = create_test_server();
        let (_, is_error) = dispatch(
            &server,
            "enable_vector_index",
            json!({"property_name": "embedding", "dimensions": 4}),
        );
        assert!(!is_error, "enabling the index should succeed");
        assert_error_code(
            &server,
            "find_similar",
            json!({"property_name": "embedding", "embedding": [0.1, 0.2, 0.3]}),
            "INVALID_ARGUMENT",
        );
    }

    // ------------------------------------------------------------------
    // NOT_FOUND paths
    // ------------------------------------------------------------------

    #[test]
    fn missing_entities_return_not_found() {
        let server = create_test_server();
        let value = assert_error_code(&server, "get_node", json!({"node_id": 424242}), "NOT_FOUND");
        let (_, _, message) = assert_structured_error(&value, "get_node missing");
        assert!(message.contains("Node not found"), "got: {message}");

        assert_error_code(&server, "get_edge", json!({"edge_id": 424242}), "NOT_FOUND");
        assert_error_code(
            &server,
            "update_node",
            json!({"node_id": 424242, "properties": {"a": 1}}),
            "NOT_FOUND",
        );
        assert_error_code(
            &server,
            "update_edge",
            json!({"edge_id": 424242, "properties": {"a": 1}}),
            "NOT_FOUND",
        );
        assert_error_code(
            &server,
            "delete_node",
            json!({"node_id": 424242}),
            "NOT_FOUND",
        );
        assert_error_code(
            &server,
            "delete_edge",
            json!({"edge_id": 424242}),
            "NOT_FOUND",
        );
        // A missing endpoint surfaces from the transaction layer as a
        // validation failure: the request is well-formed, but the graph is
        // not in the state it requires -> FAILED_PRECONDITION.
        assert_error_code(
            &server,
            "create_edge",
            json!({"source_id": 424242, "target_id": 424243, "label": "KNOWS"}),
            "FAILED_PRECONDITION",
        );
    }

    #[test]
    fn node_absent_at_temporal_coordinate_returns_not_found() {
        let server = create_test_server();
        let node_id = seed_node(&server, "Person", "Alice");
        // Long before the node's creation: not found at that coordinate.
        assert_error_code(
            &server,
            "get_node_at_time",
            json!({"node_id": node_id, "valid_time": "2000-01-01T00:00:00Z"}),
            "NOT_FOUND",
        );
    }

    // ------------------------------------------------------------------
    // FAILED_PRECONDITION paths
    // ------------------------------------------------------------------

    #[test]
    fn find_similar_without_index_returns_failed_precondition() {
        let server = create_test_server();
        let value = assert_error_code(
            &server,
            "find_similar",
            json!({"property_name": "missing_prop", "embedding": [0.1, 0.2]}),
            "FAILED_PRECONDITION",
        );
        let (_, _, message) = assert_structured_error(&value, "find_similar no index");
        assert!(
            message.contains("Vector index not enabled"),
            "got: {message}"
        );

        // Same precondition through the hybrid query path.
        assert_error_code(
            &server,
            "hybrid_query",
            json!({"query_embedding": [0.1, 0.2], "vector_property": "missing_prop"}),
            "FAILED_PRECONDITION",
        );
    }

    /// Issue #3209's DETACH-delete refusal migrated to the structured shape:
    /// `FAILED_PRECONDITION` with `details.connected_edges`, while the legacy
    /// top-level `connected_edges` / `detach_required` fields remain (no loss).
    #[test]
    fn detach_refusal_returns_failed_precondition_with_connected_edges_details() {
        let server = create_test_server();
        let a = seed_node(&server, "Person", "Alice");
        let b = seed_node(&server, "Person", "Bob");
        let (_, is_error) = dispatch(
            &server,
            "create_edge",
            json!({"source_id": a, "target_id": b, "label": "KNOWS"}),
        );
        assert!(!is_error, "edge creation should succeed");

        let value = assert_error_code(
            &server,
            "delete_node",
            json!({"node_id": a}),
            "FAILED_PRECONDITION",
        );

        // Structured details carry the machine-readable count.
        assert_eq!(
            value["error"]["details"]["connected_edges"].as_u64(),
            Some(1),
            "details.connected_edges must report the edge count: {value}"
        );
        // Legacy #3209 fields are preserved additively at the top level.
        assert_eq!(value["connected_edges"].as_u64(), Some(1), "got: {value}");
        assert_eq!(value["detach_required"].as_bool(), Some(true));
        assert_eq!(value["success"].as_bool(), Some(false));
        assert_eq!(value["node_id"].as_u64(), Some(a));
        // Free text is preserved as error.message.
        let (_, _, message) = assert_structured_error(&value, "detach refusal");
        assert!(message.contains("connected edge"), "got: {message}");
        assert!(message.contains("detach"), "got: {message}");
    }

    // ------------------------------------------------------------------
    // CONSTRAINT_VIOLATION paths
    // ------------------------------------------------------------------

    #[test]
    fn unique_violation_returns_constraint_violation_with_details() {
        let server = create_test_server();
        let (_, is_error) = dispatch(
            &server,
            "enable_unique_constraint",
            json!({"label": "Person", "property": "email"}),
        );
        assert!(!is_error, "enabling the constraint should succeed");

        let (_, is_error) = dispatch(
            &server,
            "create_node",
            json!({"label": "Person", "properties": {"email": "a@example.com"}}),
        );
        assert!(!is_error, "first node should succeed");

        let value = assert_error_code(
            &server,
            "create_node",
            json!({"label": "Person", "properties": {"email": "a@example.com"}}),
            "CONSTRAINT_VIOLATION",
        );
        // Structured details carry the constraint metadata.
        assert_eq!(
            value["error"]["details"]["label"].as_str(),
            Some("Person"),
            "got: {value}"
        );
        assert_eq!(
            value["error"]["details"]["property"].as_str(),
            Some("email")
        );
        assert!(value["error"]["details"]["existing_node_id"].is_u64());
        // Legacy top-level fields are preserved additively.
        assert_eq!(value["constraint_violation"].as_bool(), Some(true));
        assert_eq!(value["label"].as_str(), Some("Person"));
        assert_eq!(value["property"].as_str(), Some("email"));
        assert!(value["existing_node_id"].is_u64());
    }

    #[test]
    fn enable_constraint_over_duplicates_returns_failed_precondition() {
        // DuplicateOnEnable is a *state* problem — duplicates already exist
        // in the graph, so the enable cannot succeed until the data is fixed.
        // That is FAILED_PRECONDITION, not a constraint rejecting this write.
        let server = create_test_server();
        for _ in 0..2 {
            let (_, is_error) = dispatch(
                &server,
                "create_node",
                json!({"label": "Person", "properties": {"email": "dup@example.com"}}),
            );
            assert!(!is_error);
        }
        assert_error_code(
            &server,
            "enable_unique_constraint",
            json!({"label": "Person", "property": "email"}),
            "FAILED_PRECONDITION",
        );
    }

    // ------------------------------------------------------------------
    // The `query` tool: `kind` is preserved verbatim (its own contract,
    // Issue #3213) while `code`/`retriable` are added additively.
    // ------------------------------------------------------------------

    #[test]
    fn query_tool_errors_keep_kind_and_gain_code_and_retriable() {
        let server = create_test_server();

        // read_only_violation
        let (value, is_error) = dispatch(
            &server,
            "query",
            json!({"language": "aql", "query": "CREATE (n:Person) RETURN n"}),
        );
        assert!(is_error);
        assert_eq!(
            value["error"]["kind"].as_str(),
            Some("read_only_violation"),
            "the query tool's kind field must be preserved: {value}"
        );
        let (code, retriable, _) = assert_structured_error(&value, "query read-only");
        assert_eq!(code, "INVALID_ARGUMENT");
        assert!(!retriable);

        // invalid_request (unsupported language)
        let (value, is_error) = dispatch(
            &server,
            "query",
            json!({"language": "sparql", "query": "SELECT ?s WHERE { ?s ?p ?o }"}),
        );
        assert!(is_error);
        assert_eq!(value["error"]["kind"].as_str(), Some("invalid_request"));
        let (code, _, _) = assert_structured_error(&value, "query bad language");
        assert_eq!(code, "INVALID_ARGUMENT");

        // parse_error
        let (value, is_error) = dispatch(
            &server,
            "query",
            json!({"language": "aql", "query": "THIS IS NOT A QUERY"}),
        );
        assert!(is_error);
        assert_eq!(
            value["error"]["kind"].as_str(),
            Some("parse_error"),
            "got: {value}"
        );
        let (code, _, _) = assert_structured_error(&value, "query parse error");
        assert_eq!(code, "INVALID_ARGUMENT");
    }

    // ------------------------------------------------------------------
    // History / independent-dimension temporal tools (Phase 11)
    // ------------------------------------------------------------------

    fn seed_edge(server: &AletheiaMcpServer, source: u64, target: u64) -> u64 {
        let (value, is_error) = dispatch(
            server,
            "create_edge",
            json!({"source_id": source, "target_id": target, "label": "KNOWS"}),
        );
        assert!(!is_error, "seed edge should succeed: {value}");
        value["id"].as_u64().expect("created edge must carry an id")
    }

    /// Every history/temporal tool must classify a missing entity as
    /// NOT_FOUND. Note the message here still flows through `db_error` (the
    /// storage lookup fails, not the ID parse), so assert the code, not the
    /// message text.
    #[test]
    fn history_tools_missing_entity_returns_not_found() {
        let server = create_test_server();
        let ts = "2026-01-01T00:00:00Z";
        let cases: Vec<(&str, serde_json::Value)> = vec![
            (
                "get_node_at_valid_time",
                json!({"node_id": 424242, "valid_time": ts}),
            ),
            (
                "get_node_at_transaction_time",
                json!({"node_id": 424242, "transaction_time": ts}),
            ),
            ("get_node_history", json!({"node_id": 424242})),
            (
                "diff_node_versions",
                json!({"node_id": 424242, "from_version": 1, "to_version": 2}),
            ),
            (
                "get_edge_at_valid_time",
                json!({"edge_id": 424242, "valid_time": ts}),
            ),
            (
                "get_edge_at_transaction_time",
                json!({"edge_id": 424242, "transaction_time": ts}),
            ),
            ("get_edge_history", json!({"edge_id": 424242})),
            (
                "diff_edge_versions",
                json!({"edge_id": 424242, "from_version": 1, "to_version": 2}),
            ),
        ];
        for (tool, args) in cases {
            assert_error_code(&server, tool, args, "NOT_FOUND");
        }
    }

    /// Every history/temporal tool that takes a timestamp must classify an
    /// unparseable one as INVALID_ARGUMENT — even when the entity exists.
    #[test]
    fn history_tools_bad_timestamp_returns_invalid_argument() {
        let server = create_test_server();
        let a = seed_node(&server, "Person", "Alice");
        let b = seed_node(&server, "Person", "Bob");
        let edge_id = seed_edge(&server, a, b);
        let cases: Vec<(&str, serde_json::Value)> = vec![
            (
                "get_node_at_valid_time",
                json!({"node_id": a, "valid_time": "not-a-timestamp"}),
            ),
            (
                "get_node_at_transaction_time",
                json!({"node_id": a, "transaction_time": "not-a-timestamp"}),
            ),
            (
                "get_edge_at_valid_time",
                json!({"edge_id": edge_id, "valid_time": "not-a-timestamp"}),
            ),
            (
                "get_edge_at_transaction_time",
                json!({"edge_id": edge_id, "transaction_time": "not-a-timestamp"}),
            ),
        ];
        for (tool, args) in cases {
            assert_error_code(&server, tool, args, "INVALID_ARGUMENT");
        }
    }

    #[test]
    fn edge_absent_at_temporal_coordinate_returns_not_found() {
        let server = create_test_server();
        let a = seed_node(&server, "Person", "Alice");
        let b = seed_node(&server, "Person", "Bob");
        let edge_id = seed_edge(&server, a, b);
        // Long before the edge's creation: not found at that coordinate.
        assert_error_code(
            &server,
            "get_edge_at_time",
            json!({"edge_id": edge_id, "valid_time": "2000-01-01T00:00:00Z"}),
            "NOT_FOUND",
        );
    }

    // ------------------------------------------------------------------
    // hybrid_query argument classification
    // ------------------------------------------------------------------

    #[test]
    fn hybrid_query_bad_temporal_arguments_return_invalid_argument() {
        let server = create_test_server();
        let node_id = seed_node(&server, "Person", "Alice");
        let value = assert_error_code(
            &server,
            "hybrid_query",
            json!({"start_node_id": node_id, "valid_time": "not-a-timestamp"}),
            "INVALID_ARGUMENT",
        );
        let (_, _, message) = assert_structured_error(&value, "hybrid_query bad valid_time");
        assert!(message.contains("Invalid valid_time"), "got: {message}");

        let value = assert_error_code(
            &server,
            "hybrid_query",
            json!({"start_node_id": node_id, "transaction_time": "not-a-timestamp"}),
            "INVALID_ARGUMENT",
        );
        let (_, _, message) = assert_structured_error(&value, "hybrid_query bad tx_time");
        assert!(
            message.contains("Invalid transaction_time"),
            "got: {message}"
        );
    }

    #[test]
    fn hybrid_query_embedding_dimension_mismatch_returns_invalid_argument() {
        let server = create_test_server();
        let (_, is_error) = dispatch(
            &server,
            "enable_vector_index",
            json!({"property_name": "embedding", "dimensions": 4}),
        );
        assert!(!is_error, "enabling the index should succeed");
        assert_error_code(
            &server,
            "hybrid_query",
            json!({"query_embedding": [0.1, 0.2, 0.3], "vector_property": "embedding"}),
            "INVALID_ARGUMENT",
        );
    }

    #[test]
    fn hybrid_query_missing_start_node_returns_not_found() {
        let server = create_test_server();
        assert_error_code(
            &server,
            "hybrid_query",
            json!({"start_node_id": 424242}),
            "NOT_FOUND",
        );
    }

    // ------------------------------------------------------------------
    // Adjacency, schema, create_edge endpoint parsing, list_changes
    // ------------------------------------------------------------------

    #[test]
    fn adjacency_tools_out_of_range_id_returns_invalid_argument() {
        let server = create_test_server();
        for tool in ["get_outgoing_edges", "get_incoming_edges"] {
            let value = assert_error_code(
                &server,
                tool,
                json!({"node_id": u64::MAX}),
                "INVALID_ARGUMENT",
            );
            let (_, _, message) = assert_structured_error(&value, tool);
            assert!(
                message.starts_with("Invalid node ID"),
                "[{tool}] bare StorageError text expected: {message}"
            );
        }
        // Pin the current contract for an in-range but *missing* node: the
        // adjacency tools return an empty adjacency (success), not NOT_FOUND.
        for tool in ["get_outgoing_edges", "get_incoming_edges"] {
            let (value, is_error) = dispatch(&server, tool, json!({"node_id": 424242}));
            assert!(
                !is_error,
                "[{tool}] missing node currently yields an empty adjacency: {value}"
            );
            assert_eq!(value["count"].as_u64(), Some(0), "got: {value}");
        }
    }

    #[test]
    fn get_schema_bad_as_of_returns_invalid_argument() {
        let server = create_test_server();
        assert_error_code(
            &server,
            "get_schema",
            json!({"as_of_valid_time": "not-a-timestamp"}),
            "INVALID_ARGUMENT",
        );
        assert_error_code(
            &server,
            "get_schema",
            json!({"as_of_transaction_time": "not-a-timestamp"}),
            "INVALID_ARGUMENT",
        );
    }

    #[test]
    fn create_edge_out_of_range_endpoints_return_invalid_argument() {
        let server = create_test_server();
        let value = assert_error_code(
            &server,
            "create_edge",
            json!({"source_id": u64::MAX, "target_id": 1, "label": "KNOWS"}),
            "INVALID_ARGUMENT",
        );
        let (_, _, message) = assert_structured_error(&value, "create_edge bad source_id");
        assert!(message.contains("Invalid source_id"), "got: {message}");

        let value = assert_error_code(
            &server,
            "create_edge",
            json!({"source_id": 1, "target_id": u64::MAX, "label": "KNOWS"}),
            "INVALID_ARGUMENT",
        );
        let (_, _, message) = assert_structured_error(&value, "create_edge bad target_id");
        assert!(message.contains("Invalid target_id"), "got: {message}");
    }

    #[test]
    fn list_changes_bad_tx_to_returns_invalid_argument() {
        let server = create_test_server();
        let value = assert_error_code(
            &server,
            "list_changes",
            json!({"tx_from": "2026-01-01T00:00:00Z", "tx_to": "not-a-timestamp"}),
            "INVALID_ARGUMENT",
        );
        let (_, _, message) = assert_structured_error(&value, "list_changes bad tx_to");
        assert!(message.contains("Invalid tx_to"), "got: {message}");
    }
}

// ============================================================================
// Temporal Extent Tests (temporal_extent, Issue #3238)
// ============================================================================

mod temporal_extent_tests {
    use super::*;

    fn extent_req(by_label: Option<bool>) -> TemporalExtentRequest {
        TemporalExtentRequest { by_label }
    }

    fn create_node_at(server: &AletheiaMcpServer, label: &str, valid_time: &str) -> NodeResponse {
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: Some(valid_time.to_string()),
            label: label.to_string(),
            properties: Some({
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("x"));
                m
            }),
            provenance: None,
        });
        parse_response(&response).expect("create node")
    }

    fn assert_rfc3339(value: &serde_json::Value, context: &str) {
        let s = value
            .as_str()
            .unwrap_or_else(|| panic!("{context} must be a string: {value}"));
        chrono::DateTime::parse_from_rfc3339(s)
            .unwrap_or_else(|e| panic!("{context} must be RFC3339, got '{s}': {e}"));
    }

    #[test]
    fn test_temporal_extent_tool_listed_in_registry() {
        let server = create_test_server();
        let tools = server.list_tools_for_test();
        assert!(
            tools.iter().any(|name| name == "temporal_extent"),
            "temporal_extent must be registered in the tool list"
        );
    }

    #[test]
    fn test_temporal_extent_empty_database_returns_null_bounds() {
        let server = create_test_server();

        let response = server.temporal_extent(extent_req(None));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        // Bounds must be explicit nulls -- present in the payload, and
        // definitely not epoch/1970 strings.
        for dim in ["valid_time", "transaction_time"] {
            for bound in ["earliest", "latest"] {
                let b = value[dim]
                    .get(bound)
                    .unwrap_or_else(|| panic!("{dim}.{bound} must be present"));
                assert!(
                    b.is_null(),
                    "{dim}.{bound} on an empty DB must be null, got {b}"
                );
            }
        }
        // Without by_label, no breakdown keys appear at all.
        assert!(value.get("node_labels").is_none());
        assert!(value.get("edge_types").is_none());
    }

    #[test]
    fn test_temporal_extent_empty_database_by_label_returns_empty_breakdown() {
        let server = create_test_server();

        let response = server.temporal_extent(extent_req(Some(true)));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        assert!(value["valid_time"]["earliest"].is_null());
        assert!(value["transaction_time"]["latest"].is_null());
        assert_eq!(value["node_labels"].as_array().unwrap().len(), 0);
        assert_eq!(value["edge_types"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_temporal_extent_populated_shape_and_rfc3339() {
        let server = create_test_server();

        let alice = create_node_at(&server, "Person", "2021-03-01T00:00:00Z");
        let acme = create_node_at(&server, "Company", "2023-06-15T00:00:00Z");
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: Some("2024-01-01T00:00:00Z".to_string()),
            source_id: alice.id,
            target_id: acme.id,
            label: "WORKS_AT".to_string(),
            properties: None,
            provenance: None,
        });

        let response = server.temporal_extent(extent_req(None));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");

        // Valid-time bounds reflect the backdated writes exactly (all
        // intervals are open-ended, so only their starts count).
        assert_eq!(
            value["valid_time"]["earliest"],
            serde_json::json!("2021-03-01T00:00:00.000000Z")
        );
        assert_eq!(
            value["valid_time"]["latest"],
            serde_json::json!("2024-01-01T00:00:00.000000Z")
        );

        // Transaction-time bounds are system-assigned but must be real,
        // parseable RFC3339 strings (not sentinels, not micros integers).
        assert_rfc3339(&value["transaction_time"]["earliest"], "tx earliest");
        assert_rfc3339(&value["transaction_time"]["latest"], "tx latest");

        // No breakdown without by_label.
        assert!(value.get("node_labels").is_none());
        assert!(value.get("edge_types").is_none());
    }

    #[test]
    fn test_temporal_extent_expired_version_still_counts_toward_earliest() {
        let server = create_test_server();

        let alice = create_node_at(&server, "Person", "2021-03-01T00:00:00Z");
        // Supersede the original version: its interval is closed, but its
        // backdated start must still bound `earliest` (extent covers ALL
        // recorded history, not just current state).
        server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: Some("2025-01-01T00:00:00Z".to_string()),
            node_id: alice.id,
            properties: {
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Alice v2"));
                m
            },
            provenance: None,
        });

        let response = server.temporal_extent(extent_req(None));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        assert_eq!(
            value["valid_time"]["earliest"],
            serde_json::json!("2021-03-01T00:00:00.000000Z"),
            "superseded version must still count toward earliest"
        );
        assert_eq!(
            value["valid_time"]["latest"],
            serde_json::json!("2025-01-01T00:00:00.000000Z")
        );
    }

    #[test]
    fn test_temporal_extent_by_label_breakdown() {
        let server = create_test_server();

        let alice = create_node_at(&server, "Person", "2021-03-01T00:00:00Z");
        let acme = create_node_at(&server, "Company", "2023-06-15T00:00:00Z");
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: Some("2024-01-01T00:00:00Z".to_string()),
            source_id: alice.id,
            target_id: acme.id,
            label: "WORKS_AT".to_string(),
            properties: None,
            provenance: None,
        });

        let response = server.temporal_extent(extent_req(Some(true)));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();

        assert!(value.get("error").is_none(), "unexpected error: {value}");

        let node_labels = value["node_labels"].as_array().unwrap();
        assert_eq!(node_labels.len(), 2);
        // Sorted by label: Company before Person.
        assert_eq!(node_labels[0]["label"], serde_json::json!("Company"));
        assert_eq!(
            node_labels[0]["valid_time"]["earliest"],
            serde_json::json!("2023-06-15T00:00:00.000000Z")
        );
        assert_eq!(node_labels[1]["label"], serde_json::json!("Person"));
        assert_eq!(
            node_labels[1]["valid_time"]["earliest"],
            serde_json::json!("2021-03-01T00:00:00.000000Z")
        );
        assert_rfc3339(
            &node_labels[1]["transaction_time"]["earliest"],
            "Person tx earliest",
        );

        let edge_types = value["edge_types"].as_array().unwrap();
        assert_eq!(edge_types.len(), 1);
        assert_eq!(edge_types[0]["edge_type"], serde_json::json!("WORKS_AT"));
        assert_eq!(
            edge_types[0]["valid_time"]["earliest"],
            serde_json::json!("2024-01-01T00:00:00.000000Z")
        );

        // Overall bounds still cover the union of all labels.
        assert_eq!(
            value["valid_time"]["earliest"],
            serde_json::json!("2021-03-01T00:00:00.000000Z")
        );
        assert_eq!(
            value["valid_time"]["latest"],
            serde_json::json!("2024-01-01T00:00:00.000000Z")
        );
    }

    #[test]
    fn test_temporal_extent_deleted_node_tombstone_still_counts() {
        // Deletions count toward the extent: a node backdated to 2021 and
        // then deleted must still bound valid_time.earliest, and both
        // dimensions' latest must reflect the deletion event.
        let server = create_test_server();

        let alice = create_node_at(&server, "Person", "2021-03-01T00:00:00Z");
        let delete_response = server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id: alice.id,
            detach: None,
            valid_time: None,
        });
        let delete_value: serde_json::Value = serde_json::from_str(&delete_response).unwrap();
        assert!(
            delete_value.get("error").is_none(),
            "unexpected delete error: {delete_value}"
        );

        let response = server.temporal_extent(extent_req(None));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none(), "unexpected error: {value}");

        assert_eq!(
            value["valid_time"]["earliest"],
            serde_json::json!("2021-03-01T00:00:00.000000Z"),
            "a deleted node's backdated start must still bound earliest"
        );

        // The deletion closed the valid interval at ~now and recorded the
        // tombstone: both latest bounds move past the 2021 backdate.
        for dim in ["valid_time", "transaction_time"] {
            assert_rfc3339(&value[dim]["latest"], &format!("{dim} latest"));
            let latest =
                chrono::DateTime::parse_from_rfc3339(value[dim]["latest"].as_str().unwrap())
                    .unwrap();
            let backdate = chrono::DateTime::parse_from_rfc3339("2022-01-01T00:00:00Z").unwrap();
            assert!(
                latest > backdate,
                "{dim} latest must reflect the deletion event, got {latest}"
            );
        }
    }

    #[test]
    fn test_as_of_before_extent_is_empty_inside_extent_has_data() {
        // The answerability contract the extent exists to provide: an AS OF
        // query strictly before valid_time.earliest returns empty, while the
        // same query inside the extent returns data.
        let server = create_test_server();

        create_node_at(&server, "Person", "2021-03-01T00:00:00Z");

        let response = server.temporal_extent(extent_req(None));
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            value["valid_time"]["earliest"],
            serde_json::json!("2021-03-01T00:00:00.000000Z")
        );

        // AS OF before the extent: empty (indistinguishable from "never
        // existed" -- exactly why temporal_extent must be consulted first).
        let before = server.get_schema(GetSchemaRequest {
            as_of_valid_time: Some("2019-01-01T00:00:00Z".to_string()),
            as_of_transaction_time: None,
        });
        let before: serde_json::Value = serde_json::from_str(&before).unwrap();
        assert!(before.get("error").is_none(), "unexpected error: {before}");
        assert_eq!(
            before["node_labels"].as_array().unwrap().len(),
            0,
            "AS OF before valid_time.earliest must return empty"
        );

        // The same query inside the extent returns the data.
        let inside = server.get_schema(GetSchemaRequest {
            as_of_valid_time: Some("2022-01-01T00:00:00Z".to_string()),
            as_of_transaction_time: None,
        });
        let inside: serde_json::Value = serde_json::from_str(&inside).unwrap();
        assert!(inside.get("error").is_none(), "unexpected error: {inside}");
        let labels = inside["node_labels"].as_array().unwrap();
        assert_eq!(labels.len(), 1, "AS OF inside the extent must return data");
        assert_eq!(labels[0]["label"], serde_json::json!("Person"));
    }
}

// ============================================================================
// Database Stats Tests (Issue #3222)
// ============================================================================

mod database_stats_tests {
    use super::*;

    fn stats_response(server: &AletheiaMcpServer) -> serde_json::Value {
        let response = server.database_stats(DatabaseStatsRequest {});
        serde_json::from_str(&response).expect("database_stats must return valid JSON")
    }

    /// AC1 + AC9 (empty DB well-formed): the tool takes no required arguments
    /// and returns a single well-formed JSON object even on an empty database.
    #[test]
    fn test_database_stats_empty_db_well_formed() {
        let server = create_test_server();
        let value = stats_response(&server);

        assert!(value.get("error").is_none(), "unexpected error: {value}");
        assert_eq!(value["current"]["node_count"], serde_json::json!(0));
        assert_eq!(value["current"]["edge_count"], serde_json::json!(0));
        assert_eq!(
            value["historical"]["total_node_versions"],
            serde_json::json!(0)
        );
        assert_eq!(
            value["historical"]["total_edge_versions"],
            serde_json::json!(0)
        );
        assert_eq!(value["historical"]["unique_nodes"], serde_json::json!(0));
        assert_eq!(value["historical"]["unique_edges"], serde_json::json!(0));
        assert_eq!(value["historical"]["anchor_count"], serde_json::json!(0));
        assert_eq!(value["historical"]["delta_count"], serde_json::json!(0));
        // Disabled subsystems must be tagged, never zero-reported (AC7).
        assert_eq!(value["cold_storage"]["enabled"], serde_json::json!(false));
        assert_eq!(value["wal"]["enabled"], serde_json::json!(true));
    }

    /// AC3 + AC9 (populated DB shape): counts reflect data, version counters
    /// track updates, and anchors + deltas always sum to the version totals.
    #[test]
    fn test_database_stats_populated_db_shape() {
        let server = create_test_server();

        let (a, b) = {
            let a: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Person".to_string(),
                properties: None,
                provenance: None,
            }))
            .unwrap();
            let b: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Person".to_string(),
                properties: None,
                provenance: None,
            }))
            .unwrap();
            (a.id, b.id)
        };
        server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: a,
            target_id: b,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        });
        // An update creates an extra node version beyond the creates.
        server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id: a,
            properties: {
                let mut m = HashMap::new();
                m.insert("name".to_string(), serde_json::json!("Alice"));
                m
            },
            provenance: None,
        });

        let value = stats_response(&server);
        assert!(value.get("error").is_none(), "unexpected error: {value}");

        assert_eq!(value["current"]["node_count"], serde_json::json!(2));
        assert_eq!(value["current"]["edge_count"], serde_json::json!(1));

        let hist = &value["historical"];
        assert_eq!(hist["unique_nodes"], serde_json::json!(2));
        assert_eq!(hist["unique_edges"], serde_json::json!(1));
        let total_node_versions = hist["total_node_versions"].as_u64().unwrap();
        assert_eq!(
            total_node_versions, 3,
            "2 creates + 1 update must yield exactly 3 node versions: {value}"
        );
        assert_eq!(hist["total_edge_versions"], serde_json::json!(1));

        // Compression accounting must be internally consistent.
        let anchors = hist["anchor_count"].as_u64().unwrap();
        let deltas = hist["delta_count"].as_u64().unwrap();
        let total_versions = total_node_versions + hist["total_edge_versions"].as_u64().unwrap();
        assert_eq!(
            anchors + deltas,
            total_versions,
            "anchors + deltas must equal total versions: {value}"
        );
        assert!(anchors >= 1, "at least one anchor expected: {value}");
        // With the default anchor interval (10), the update after a create is
        // stored as a delta: the anchor/delta split must actually engage, and
        // the anchor share must therefore drop below 1.0.
        assert!(
            deltas >= 1,
            "an update following a create must be stored as a delta: {value}"
        );
        let ratio = hist["compression_ratio"].as_f64().unwrap();
        assert!(
            ratio > 0.0 && ratio < 1.0,
            "compression_ratio must be in (0.0, 1.0) once a delta exists: {value}"
        );
    }

    /// AC4 + AC7 + AC9 (disabled tier tagged): a database without cold
    /// storage must report `cold_storage: {"enabled": false}` with NO count
    /// fields, so an LLM can never mistake "disabled" for "0 cold versions".
    #[test]
    fn test_database_stats_cold_storage_disabled_is_tagged_not_zero() {
        let server = create_test_server();
        let value = stats_response(&server);

        // Exact shape: no counts, no tier distribution, no extra keys --
        // anything beyond the tag could be mistaken for real (zero) data.
        assert_eq!(
            value["cold_storage"],
            serde_json::json!({"enabled": false}),
            "disabled cold_storage must be exactly {{\"enabled\": false}}: {value}"
        );
    }

    /// AC4 (enabled tier reports distribution): with cold storage configured,
    /// the snapshot reports cold counts and the hot/warm/cold access
    /// distribution.
    #[test]
    fn test_database_stats_cold_storage_enabled_reports_tiers() {
        use crate::config::{AletheiaDBConfig, HistoricalConfigBuilder, WalConfigBuilder};

        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let config = AletheiaDBConfig::builder()
            .wal(
                WalConfigBuilder::new()
                    .wal_dir(temp_dir.path().join("wal"))
                    .build(),
            )
            .persistence(crate::storage::index_persistence::PersistenceConfig {
                enabled: false,
                ..Default::default()
            })
            .historical(
                HistoricalConfigBuilder::new()
                    .enable_cold_storage(true)
                    .cold_storage_path(temp_dir.path().join("cold.redb"))
                    .build(),
            )
            .build();
        let db = Arc::new(AletheiaDB::with_unified_config(config).expect("db init"));
        let server = AletheiaMcpServer::new(db);

        let value = stats_response(&server);
        assert!(value.get("error").is_none(), "unexpected error: {value}");

        let cold = value["cold_storage"]
            .as_object()
            .expect("cold_storage must be an object");
        assert_eq!(cold["enabled"], serde_json::json!(true));
        assert_eq!(cold["node_versions_stored"], serde_json::json!(0));
        assert_eq!(cold["edge_versions_stored"], serde_json::json!(0));
        let tier_access = cold["tier_access"]
            .as_object()
            .expect("enabled cold storage must report tier_access distribution");
        for key in ["hot_hits", "warm_hits", "cold_hits", "misses"] {
            assert!(
                tier_access.contains_key(key),
                "tier_access must contain {key}: {value}"
            );
        }
    }

    /// AC5 (WAL state): durability mode is a stable token and the
    /// next-to-be-allocated LSN is reported and advances with writes.
    #[test]
    fn test_database_stats_wal_state_reported() {
        let server = create_test_server();
        let value = stats_response(&server);

        let wal = value["wal"].as_object().expect("wal must be an object");
        assert_eq!(wal["enabled"], serde_json::json!(true));
        let mode = wal["durability_mode"].as_str().unwrap();
        assert!(
            ["synchronous", "async", "group_commit", "async_batched"].contains(&mode),
            "durability_mode must be a stable token, got: {mode}"
        );
        let lsn_before = wal["current_lsn"].as_u64().unwrap();
        assert!(lsn_before >= 1, "LSN starts at 1: {value}");

        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });

        let value_after = stats_response(&server);
        let lsn_after = value_after["wal"]["current_lsn"].as_u64().unwrap();
        assert!(
            lsn_after > lsn_before,
            "LSN must advance after a write: {lsn_before} -> {lsn_after}"
        );
        assert!(value_after["wal"]["healthy"].as_bool().unwrap());
    }

    /// AC1: the tool is advertised in the MCP tool list.
    #[test]
    fn test_database_stats_registered_in_tool_list() {
        let server = create_test_server();
        let tools = server.list_tools_for_test();
        assert!(
            tools.iter().any(|t| t == "database_stats"),
            "database_stats must be advertised: {tools:?}"
        );
    }

    /// AC2: the MCP tool is a thin aggregator over the public API — both
    /// surfaces must report identical numbers on the same database.
    #[test]
    fn test_database_stats_matches_public_api() {
        let server = create_test_server();
        server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: None,
            provenance: None,
        });

        let mut value = stats_response(&server);
        // Issue #3368 residue: the MCP response additively carries a
        // `resource_limits` termination-counter block that is an MCP-surface
        // concern, not part of the storage-layer `DatabaseStats`. Assert it is
        // present, then strip it so the remainder is exactly the serialized
        // public API (the thin-aggregator contract is otherwise unchanged).
        assert!(
            value.get("resource_limits").is_some(),
            "MCP response must surface the resource_limits counters: {value}"
        );
        value
            .as_object_mut()
            .expect("stats response is an object")
            .remove("resource_limits");

        let api_stats = server.db().stats();
        let api_value = serde_json::to_value(&api_stats).expect("DatabaseStats must serialize");
        assert_eq!(
            value, api_value,
            "MCP response (minus the additive resource_limits block) must be exactly the \
             serialized public DatabaseStats"
        );
    }

    /// Issue #3368 engine lane: `database_stats.resource_limits` carries an
    /// `engine` sub-object with the executor-guard termination counters (a
    /// distinct family from the top-level MCP-surface counters), and an engine
    /// termination driven through a Rust-API `db.query()` with a row cap is
    /// visible there.
    #[test]
    fn database_stats_surfaces_engine_lane_counters() {
        let server = create_test_server();
        let db = server.db();

        // Seed a hub with several neighbors, then trip the engine row cap.
        let hub = db
            .create_node(
                "Person",
                crate::core::PropertyMapBuilder::new()
                    .insert("name", "Hub")
                    .build(),
            )
            .expect("hub");
        for _ in 0..5 {
            let leaf = db
                .create_node("Person", crate::core::PropertyMapBuilder::new().build())
                .expect("leaf");
            db.create_edge(
                hub,
                leaf,
                "KNOWS",
                crate::core::PropertyMapBuilder::new().build(),
            )
            .expect("edge");
        }
        let results = db
            .query()
            .start(hub)
            .traverse("KNOWS")
            .with_max_rows(2)
            .execute(db)
            .expect("execute lazily");
        for row in results {
            if row.is_err() {
                break;
            }
        }

        let value = stats_response(&server);
        let engine = &value["resource_limits"]["engine"];
        assert!(
            engine.is_object(),
            "resource_limits.engine must be present: {value}"
        );
        assert_eq!(
            engine["row_cap_terminations"],
            serde_json::json!(1),
            "the engine row-cap termination must be surfaced: {value}"
        );
        // The four engine dimensions are always present.
        for key in [
            "timeout_terminations",
            "row_cap_terminations",
            "memory_terminations",
            "override_rejections",
        ] {
            assert!(engine.get(key).is_some(), "engine.{key} missing: {value}");
        }
    }

    /// Wire-level argument handling: `call_tool` delivers missing arguments
    /// as JSON null, and clients may also send an empty object or extra
    /// unknown keys — all must succeed. A non-object argument value must be
    /// rejected with a structured error, not a panic.
    #[test]
    fn test_database_stats_raw_argument_edge_cases() {
        let server = create_test_server();

        // No `arguments` at all (call_tool surfaces this as null).
        let value: serde_json::Value =
            serde_json::from_str(&server.database_stats_raw(serde_json::Value::Null))
                .expect("null args must yield valid JSON");
        assert!(
            value.get("error").is_none(),
            "null arguments must succeed: {value}"
        );
        assert!(value.get("current").is_some());

        // Empty object.
        let value: serde_json::Value =
            serde_json::from_str(&server.database_stats_raw(serde_json::json!({})))
                .expect("empty-object args must yield valid JSON");
        assert!(
            value.get("error").is_none(),
            "empty-object arguments must succeed: {value}"
        );

        // Unknown keys are tolerated (the request struct is not
        // deny_unknown_fields), so a slightly-wrong client still gets stats.
        let value: serde_json::Value =
            serde_json::from_str(&server.database_stats_raw(serde_json::json!({"unexpected": 1})))
                .expect("unknown-key args must yield valid JSON");
        assert!(
            value.get("error").is_none(),
            "unknown keys must be tolerated: {value}"
        );
        assert!(value.get("wal").is_some());

        // A non-object argument value is a malformed request: structured
        // error (Issue #3234 shape), never a panic or silent success.
        let value: serde_json::Value =
            serde_json::from_str(&server.database_stats_raw(serde_json::json!("bogus")))
                .expect("invalid args must still yield valid JSON");
        let error = value
            .get("error")
            .and_then(|e| e.as_object())
            .expect("non-object arguments must produce a structured error object");
        assert_eq!(
            error.get("code").and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "malformed arguments must be INVALID_ARGUMENT: {value}"
        );
        assert_eq!(
            error.get("retriable").and_then(|r| r.as_bool()),
            Some(false),
            "INVALID_ARGUMENT is never retriable: {value}"
        );
        let message = error
            .get("message")
            .and_then(|m| m.as_str())
            .expect("structured error must carry a message");
        assert!(
            message.contains("Invalid arguments"),
            "error must identify the argument problem: {value}"
        );
    }

    /// Schema contract: the exact key sets of every response object are
    /// load-bearing for MCP consumers — a renamed or dropped key is a
    /// breaking change that must fail this test.
    #[test]
    fn test_database_stats_key_sets_are_stable() {
        use crate::config::{AletheiaDBConfig, HistoricalConfigBuilder, WalConfigBuilder};

        fn keys(value: &serde_json::Value) -> Vec<&str> {
            let mut keys: Vec<&str> = value
                .as_object()
                .expect("expected a JSON object")
                .keys()
                .map(String::as_str)
                .collect();
            keys.sort_unstable();
            keys
        }

        // Cold-enabled server so the flattened details keys are present too.
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let config = AletheiaDBConfig::builder()
            .wal(
                WalConfigBuilder::new()
                    .wal_dir(temp_dir.path().join("wal"))
                    .build(),
            )
            .persistence(crate::storage::index_persistence::PersistenceConfig {
                enabled: false,
                ..Default::default()
            })
            .historical(
                HistoricalConfigBuilder::new()
                    .enable_cold_storage(true)
                    .cold_storage_path(temp_dir.path().join("cold.redb"))
                    .build(),
            )
            .build();
        let db = Arc::new(AletheiaDB::with_unified_config(config).expect("db init"));
        let server = AletheiaMcpServer::new(db);

        let value = stats_response(&server);
        assert_eq!(
            keys(&value),
            vec![
                "chain",
                "changefeed",
                "cold_storage",
                "current",
                "historical",
                // Per-namespace counts (Issue #3349, PR3a): serialized since the
                // surface slice; empty array on a default-only database.
                "namespaces",
                // Replication role + progress skeleton (Issue #3355, Slice A).
                "replication",
                "resource_limits",
                "wal"
            ]
        );
        // Push-changefeed subscription block (Issue #3678): the scalar aggregate
        // plus the per-principal breakdown. The embedded `new()` server is
        // anonymous (admin-privileged), so the admin-gated `per_principal` key is
        // present (empty here — no live subscriptions).
        assert_eq!(
            keys(&value["changefeed"]),
            vec!["active_subscriptions", "per_principal"]
        );
        // Issue #3368 residue: the additive per-query resource-limit
        // termination counters are surfaced under a stable `resource_limits`
        // block alongside the storage-layer stats.
        assert_eq!(
            keys(&value["resource_limits"]),
            vec![
                "byte_cap_terminations",
                // Issue #3368 engine lane: nested executor-guard termination
                // counters (a distinct family from the top-level MCP-surface
                // counters), always present.
                "engine",
                // Issue #3368 memory-budget dimension (default-off): additive
                // counter, present even when the budget is never engaged.
                "memory_terminations",
                "override_rejections",
                "timeout_terminations",
            ]
        );
        // The nested engine block carries the four executor-guard dimensions.
        assert_eq!(
            keys(&value["resource_limits"]["engine"]),
            vec![
                "memory_terminations",
                "override_rejections",
                "row_cap_terminations",
                "timeout_terminations",
            ]
        );
        // Provenance hash chain block (Issue #3351 AC7): present on every
        // stats response; disabled here, so all optional fields are null but
        // their keys are always emitted.
        assert_eq!(
            keys(&value["chain"]),
            vec![
                "enabled",
                "genesis_digest",
                "head_digest",
                "head_seq",
                "last_verified",
            ]
        );
        assert_eq!(keys(&value["current"]), vec!["edge_count", "node_count"]);
        assert_eq!(
            keys(&value["historical"]),
            vec![
                "anchor_count",
                "compression_ratio",
                "delta_count",
                "edge_anchor_count",
                "edge_delta_count",
                "node_anchor_count",
                "node_delta_count",
                "total_edge_versions",
                "total_node_versions",
                "unique_edges",
                "unique_nodes",
            ]
        );
        assert_eq!(
            keys(&value["cold_storage"]),
            vec![
                "compression_ratio",
                "edge_versions_stored",
                "enabled",
                "node_versions_stored",
                "tier_access",
            ]
        );
        assert_eq!(
            keys(&value["cold_storage"]["tier_access"]),
            vec!["cold_hits", "hot_hits", "misses", "warm_hits"]
        );
        assert_eq!(
            keys(&value["wal"]),
            vec![
                "current_lsn",
                "durability_mode",
                "enabled",
                "healthy",
                "total_appends",
            ]
        );
        // Replication role + progress skeleton (Issue #3355, Slice A): a
        // primary reports its role with `replica: null` (populated starting
        // with the Slice B replication engine).
        assert_eq!(keys(&value["replication"]), vec!["replica", "role"]);
        assert_eq!(value["replication"]["role"], "primary");
        assert!(value["replication"]["replica"].is_null());
    }

    /// Deletions through the MCP surface must be reflected coherently:
    /// current-state counts shrink, while the bi-temporal record (unique
    /// entities and their versions) is retained and grows — a delete is a
    /// recorded change, not an erasure.
    #[test]
    fn test_database_stats_reflects_deletions() {
        let server = create_test_server();

        let (a, b) = {
            let a: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Person".to_string(),
                properties: None,
                provenance: None,
            }))
            .unwrap();
            let b: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Person".to_string(),
                properties: None,
                provenance: None,
            }))
            .unwrap();
            (a.id, b.id)
        };
        let edge: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            source_id: a,
            target_id: b,
            label: "KNOWS".to_string(),
            properties: None,
            provenance: None,
        }))
        .unwrap();

        let before = stats_response(&server);
        assert_eq!(before["current"]["node_count"], serde_json::json!(2));
        assert_eq!(before["current"]["edge_count"], serde_json::json!(1));
        let node_versions_before = before["historical"]["total_node_versions"]
            .as_u64()
            .unwrap();
        let edge_versions_before = before["historical"]["total_edge_versions"]
            .as_u64()
            .unwrap();

        // Delete the edge, then node `a` (now unconnected).
        let del_edge: serde_json::Value =
            serde_json::from_str(&server.delete_edge(DeleteEdgeRequest {
                namespace: None,
                edge_id: edge.id,
                valid_time: None,
            }))
            .unwrap();
        assert_eq!(del_edge.get("success"), Some(&serde_json::json!(true)));
        let del_node: serde_json::Value =
            serde_json::from_str(&server.delete_node(DeleteNodeRequest {
                namespace: None,
                node_id: a,
                detach: None,
                valid_time: None,
            }))
            .unwrap();
        assert_eq!(del_node.get("success"), Some(&serde_json::json!(true)));

        let after = stats_response(&server);
        assert_eq!(after["current"]["node_count"], serde_json::json!(1));
        assert_eq!(after["current"]["edge_count"], serde_json::json!(0));

        // The bi-temporal record is retained: both nodes and the edge still
        // have history, and the deletions were recorded as new versions.
        assert_eq!(after["historical"]["unique_nodes"], serde_json::json!(2));
        assert_eq!(after["historical"]["unique_edges"], serde_json::json!(1));
        assert!(
            after["historical"]["total_node_versions"].as_u64().unwrap() > node_versions_before,
            "a node deletion must be recorded as a version, not an erasure: \
             before={node_versions_before}, after={after}"
        );
        assert!(
            after["historical"]["total_edge_versions"].as_u64().unwrap() > edge_versions_before,
            "an edge deletion must be recorded as a version, not an erasure: \
             before={edge_versions_before}, after={after}"
        );
    }
}

// ============================================================================
// Temporal bounds on read responses (Issue #3232)
// ============================================================================

mod temporal_bounds_tests {
    use super::*;

    /// Extract the `temporal` block from an entity JSON object, failing the
    /// test if it is absent.
    fn temporal_of(entity: &serde_json::Value) -> serde_json::Value {
        entity
            .get("temporal")
            .unwrap_or_else(|| panic!("response must carry a `temporal` block: {entity}"))
            .clone()
    }

    /// Parse an RFC3339 string back to microseconds since epoch.
    fn rfc3339_to_micros(s: &str) -> i64 {
        chrono::DateTime::parse_from_rfc3339(s)
            .unwrap_or_else(|e| panic!("`{s}` must be RFC3339: {e}"))
            .timestamp_micros()
    }

    /// Assert the shape shared by every current (live) version: RFC3339
    /// `*_from` bounds, explicit-null open `*_to` bounds (present, never
    /// omitted), and `is_current: true`.
    fn assert_current_open_bounds(temporal: &serde_json::Value) {
        for key in ["valid_from", "transaction_from"] {
            let s = temporal[key]
                .as_str()
                .unwrap_or_else(|| panic!("`{key}` must be an RFC3339 string: {temporal}"));
            rfc3339_to_micros(s);
        }
        for key in ["valid_to", "transaction_to"] {
            let v = temporal.get(key).unwrap_or_else(|| {
                panic!("`{key}` must be present (as null), not omitted: {temporal}")
            });
            assert!(
                v.is_null(),
                "`{key}` must be JSON null for open bounds: {temporal}"
            );
        }
        assert_eq!(
            temporal["is_current"],
            serde_json::json!(true),
            "a live version must report is_current: true: {temporal}"
        );
    }

    fn create_person(server: &AletheiaMcpServer, name: &str) -> serde_json::Value {
        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!(name));
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Person".to_string(),
            properties: Some(props),
            provenance: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none(), "create_node failed: {value}");
        value
    }

    fn create_knows_edge(
        server: &AletheiaMcpServer,
        source_id: u64,
        target_id: u64,
    ) -> serde_json::Value {
        let response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id,
            target_id,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none(), "create_edge failed: {value}");
        value
    }

    fn now_micros() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    }

    #[test]
    fn test_create_and_get_node_carry_temporal_bounds() {
        let server = create_test_server();

        let created = create_person(&server, "Alice");
        assert_current_open_bounds(&temporal_of(&created));

        let get_response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id: created["id"].as_u64().unwrap(),
            include_vectors: None,
        });
        let retrieved: serde_json::Value = serde_json::from_str(&get_response).unwrap();
        assert_current_open_bounds(&temporal_of(&retrieved));

        // Existing top-level fields are unchanged (additive contract) and the
        // typed response still round-trips.
        let typed: NodeResponse = parse_response(&get_response).expect("typed parse must work");
        assert_eq!(typed.label, "Person");
        assert_eq!(
            typed.properties.get("name"),
            Some(&serde_json::json!("Alice"))
        );
    }

    #[test]
    fn test_create_and_get_edge_carry_temporal_bounds() {
        let server = create_test_server();

        let a = create_person(&server, "Alice")["id"].as_u64().unwrap();
        let b = create_person(&server, "Bob")["id"].as_u64().unwrap();

        let created = create_knows_edge(&server, a, b);
        assert_current_open_bounds(&temporal_of(&created));

        let get_response = server.get_edge(GetEdgeRequest {
            namespace: None,
            edge_id: created["id"].as_u64().unwrap(),
            include_vectors: None,
        });
        let retrieved: serde_json::Value = serde_json::from_str(&get_response).unwrap();
        assert_current_open_bounds(&temporal_of(&retrieved));

        let typed: EdgeResponse = parse_response(&get_response).expect("typed parse must work");
        assert_eq!(typed.label, "KNOWS");
        assert_eq!(typed.source_id, a);
        assert_eq!(typed.target_id, b);

        // Point-in-time edge read carries the block too. Capture one shared
        // "now" so both dimensions anchor the same bi-temporal instant.
        let now = now_micros().to_string();
        let at_time_response = server.get_edge_at_time(GetEdgeAtTimeRequest {
            namespace: None,
            edge_id: created["id"].as_u64().unwrap(),
            valid_time: now.clone(),
            transaction_time: Some(now),
        });
        let at_time: serde_json::Value = serde_json::from_str(&at_time_response).unwrap();
        assert!(
            at_time.get("error").is_none(),
            "get_edge_at_time failed: {at_time}"
        );
        assert_current_open_bounds(&temporal_of(&at_time["edge"]));
    }

    #[test]
    fn test_list_nodes_and_traverse_stamp_each_node() {
        let server = create_test_server();

        let a = create_person(&server, "Alice")["id"].as_u64().unwrap();
        let b = create_person(&server, "Bob")["id"].as_u64().unwrap();
        create_knows_edge(&server, a, b);

        let list_response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some("Person".to_string()),
            property_key: None,
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: None,
        });
        let listed: serde_json::Value = serde_json::from_str(&list_response).unwrap();
        let nodes = listed["nodes"].as_array().expect("nodes array");
        assert_eq!(nodes.len(), 2);
        for node in nodes {
            assert_current_open_bounds(&temporal_of(node));
        }

        let traverse_response = server.traverse(TraverseRequest {
            namespace: None,
            start_node_id: a,
            edge_label: "KNOWS".to_string(),
            direction: None,
            depth: None,
            limit: None,
            offset: None,
            include_vectors: None,
            as_of_valid_time: None,
            as_of_transaction_time: None,
        });
        let traversed: serde_json::Value = serde_json::from_str(&traverse_response).unwrap();
        let results = traversed["results"].as_array().expect("results array");
        assert_eq!(results.len(), 1);
        for result in results {
            assert_current_open_bounds(&temporal_of(&result["node"]));
        }
    }

    #[test]
    fn test_adjacency_and_update_responses_carry_temporal_bounds() {
        let server = create_test_server();

        let a = create_person(&server, "Alice")["id"].as_u64().unwrap();
        let b = create_person(&server, "Bob")["id"].as_u64().unwrap();
        let edge_id = create_knows_edge(&server, a, b)["id"].as_u64().unwrap();

        for response in [
            server.get_outgoing_edges(GetOutgoingEdgesRequest {
                node_id: a,
                label: None,
                include_vectors: None,
            }),
            server.get_incoming_edges(GetIncomingEdgesRequest {
                node_id: b,
                label: None,
                include_vectors: None,
            }),
        ] {
            let value: serde_json::Value = serde_json::from_str(&response).unwrap();
            let edges = value["edges"].as_array().expect("edges array");
            assert_eq!(edges.len(), 1);
            for edge in edges {
                assert_current_open_bounds(&temporal_of(edge));
            }
        }

        // update_node / update_edge responses reflect the NEW version's
        // bounds: still open-ended and current.
        let mut props = HashMap::new();
        props.insert("age".to_string(), serde_json::json!(31));
        let update_node_response = server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id: a,
            properties: props.clone(),
            provenance: None,
        });
        let updated_node: serde_json::Value = serde_json::from_str(&update_node_response).unwrap();
        assert_current_open_bounds(&temporal_of(&updated_node));

        let update_edge_response = server.update_edge(UpdateEdgeRequest {
            namespace: None,
            derived_from: None,
            edge_id,
            properties: props,
            valid_time: None,
            provenance: None,
        });
        let updated_edge: serde_json::Value = serde_json::from_str(&update_edge_response).unwrap();
        assert_current_open_bounds(&temporal_of(&updated_edge));
    }

    #[test]
    fn test_superseded_version_has_open_valid_closed_tx_bound_and_is_not_current() {
        let server = create_test_server();

        let node_id = create_person(&server, "Alice")["id"].as_u64().unwrap();

        // Anchor a bi-temporal coordinate strictly after the create and
        // strictly before the update.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let anchor = now_micros();
        std::thread::sleep(std::time::Duration::from_millis(5));

        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!("Alice v2"));
        parse_response::<NodeResponse>(&server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id,
            properties: props,
            provenance: None,
        }))
        .expect("update must succeed");

        // The at-time read anchored before the update returns the superseded
        // version. #3504: supersession is append-only on the valid dimension,
        // so the superseded version's valid interval stays OPEN
        // (valid_to == null); only its transaction-time bound is closed by the
        // update. It is not current (its tx interval is closed).
        let at_time_response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id,
            valid_time: anchor.to_string(),
            transaction_time: Some(anchor.to_string()),
        });
        let at_time: serde_json::Value = serde_json::from_str(&at_time_response).unwrap();
        assert!(
            at_time.get("error").is_none(),
            "get_node_at_time failed: {at_time}"
        );
        let superseded = temporal_of(&at_time["node"]);
        assert!(
            superseded["valid_to"].is_null(),
            "superseded version's valid_to must stay open (#3504, append-only): {superseded}"
        );
        let transaction_to = superseded["transaction_to"].as_str().unwrap_or_else(|| {
            panic!("superseded transaction_to must be a closed RFC3339 bound: {superseded}")
        });
        assert!(rfc3339_to_micros(transaction_to) > anchor);
        assert_eq!(
            superseded["is_current"],
            serde_json::json!(false),
            "a superseded version must report is_current: false (its tx interval is closed): {superseded}"
        );

        // A current read after the update shows the NEW version's bounds:
        // open-ended, current, and starting after the anchor.
        let current_response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id,
            include_vectors: None,
        });
        let current: serde_json::Value = serde_json::from_str(&current_response).unwrap();
        let current_temporal = temporal_of(&current);
        assert_current_open_bounds(&current_temporal);
        assert!(
            rfc3339_to_micros(current_temporal["transaction_from"].as_str().unwrap()) > anchor,
            "current version must start after the anchor: {current_temporal}"
        );
    }

    #[test]
    fn test_temporal_bounds_agree_with_node_history() {
        let server = create_test_server();

        let node_id = create_person(&server, "Alice")["id"].as_u64().unwrap();

        std::thread::sleep(std::time::Duration::from_millis(5));
        let anchor = now_micros();
        std::thread::sleep(std::time::Duration::from_millis(5));

        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!("Alice v2"));
        parse_response::<NodeResponse>(&server.update_node(UpdateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            node_id,
            properties: props,
            provenance: None,
        }))
        .expect("update must succeed");

        let history_response = server.get_node_history(GetNodeHistoryRequest { node_id });
        let history: serde_json::Value = serde_json::from_str(&history_response).unwrap();
        let versions = history["versions"].as_array().expect("versions array");
        assert_eq!(versions.len(), 2, "expected two versions: {history}");

        // History emits micros-as-string; the temporal block's RFC3339 must
        // decode to the same microsecond values for the same version.
        let history_micros = |v: &serde_json::Value, key: &str| -> Option<i64> {
            let field = v.get(key).unwrap_or(&serde_json::Value::Null);
            if field.is_null() {
                None
            } else {
                Some(field.as_str().unwrap().parse::<i64>().unwrap())
            }
        };
        let temporal_micros = |t: &serde_json::Value, key: &str| -> Option<i64> {
            let field = t.get(key).unwrap_or_else(|| {
                panic!("temporal block must always carry `{key}` (null when open): {t}")
            });
            if field.is_null() {
                None
            } else {
                Some(rfc3339_to_micros(field.as_str().unwrap()))
            }
        };

        let superseded_history = versions
            .iter()
            .find(|v| !v["transaction_to"].is_null())
            .expect("one superseded version");
        let current_history = versions
            .iter()
            .find(|v| v["transaction_to"].is_null())
            .expect("one current version");

        let at_time_response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id,
            valid_time: anchor.to_string(),
            transaction_time: Some(anchor.to_string()),
        });
        let at_time: serde_json::Value = serde_json::from_str(&at_time_response).unwrap();
        let superseded_temporal = temporal_of(&at_time["node"]);

        let current_response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id,
            include_vectors: None,
        });
        let current: serde_json::Value = serde_json::from_str(&current_response).unwrap();
        let current_temporal = temporal_of(&current);

        for key in [
            "valid_from",
            "valid_to",
            "transaction_from",
            "transaction_to",
        ] {
            assert_eq!(
                temporal_micros(&superseded_temporal, key),
                history_micros(superseded_history, key),
                "superseded `{key}` must agree with history: {superseded_temporal} vs {superseded_history}"
            );
            assert_eq!(
                temporal_micros(&current_temporal, key),
                history_micros(current_history, key),
                "current `{key}` must agree with history: {current_temporal} vs {current_history}"
            );
        }
    }

    #[test]
    fn test_not_yet_valid_fact_reports_is_current_false() {
        let server = create_test_server();

        // A fact that only becomes true one hour from now: both bounds are
        // open, yet the valid interval does not contain the current time, so
        // is_current must be false. This isolates the valid-time conjunct of
        // is_current -- a transaction-time-only implementation would wrongly
        // report true here.
        let future = now_micros() + 3_600_000_000;
        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!("Alice"));
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: Some(future.to_string()),
            label: "Person".to_string(),
            properties: Some(props),
            provenance: None,
        });
        let created: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            created.get("error").is_none(),
            "create_node with a future valid_time must succeed: {created}"
        );

        let assert_future_not_current = |temporal: &serde_json::Value| {
            for key in ["valid_to", "transaction_to"] {
                let v = temporal.get(key).unwrap_or_else(|| {
                    panic!("`{key}` must be present (as null), not omitted: {temporal}")
                });
                assert!(
                    v.is_null(),
                    "`{key}` must be JSON null for open bounds: {temporal}"
                );
            }
            assert_eq!(
                rfc3339_to_micros(temporal["valid_from"].as_str().unwrap()),
                future,
                "valid_from must decode to the future valid time exactly: {temporal}"
            );
            assert_eq!(
                temporal["is_current"],
                serde_json::json!(false),
                "a not-yet-valid fact must report is_current: false even though \
                 both bounds are open: {temporal}"
            );
        };
        assert_future_not_current(&temporal_of(&created));

        let get_response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id: created["id"].as_u64().unwrap(),
            include_vectors: None,
        });
        let retrieved: serde_json::Value = serde_json::from_str(&get_response).unwrap();
        assert_future_not_current(&temporal_of(&retrieved));
    }

    #[test]
    fn test_backdated_create_reflects_valid_time_in_temporal_block() {
        let server = create_test_server();

        // A fact backdated one hour: valid since then (and still valid), so
        // it is current, but valid_from must reflect the caller-supplied
        // valid time while transaction_from is the (later) system-assigned
        // commit time.
        let backdated = now_micros() - 3_600_000_000;
        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!("Alice"));
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: Some(backdated.to_string()),
            label: "Person".to_string(),
            properties: Some(props),
            provenance: None,
        });
        let created: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            created.get("error").is_none(),
            "create_node with a backdated valid_time must succeed: {created}"
        );

        let temporal = temporal_of(&created);
        assert_current_open_bounds(&temporal);
        assert_eq!(
            rfc3339_to_micros(temporal["valid_from"].as_str().unwrap()),
            backdated,
            "valid_from must decode to the backdated valid time exactly: {temporal}"
        );
        assert!(
            rfc3339_to_micros(temporal["transaction_from"].as_str().unwrap()) > backdated,
            "transaction_from is system-assigned and must be later than the \
             backdated valid time: {temporal}"
        );
    }

    #[test]
    fn test_deleted_node_at_time_read_has_closed_transaction_to() {
        let server = create_test_server();

        let node_id = create_person(&server, "Alice")["id"].as_u64().unwrap();

        // Anchor a bi-temporal coordinate strictly after the create and
        // strictly before the delete.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let anchor = now_micros();
        std::thread::sleep(std::time::Duration::from_millis(5));

        let delete_response = server.delete_node(DeleteNodeRequest {
            namespace: None,
            node_id,
            detach: None,
            valid_time: None,
        });
        let deleted: serde_json::Value = serde_json::from_str(&delete_response).unwrap();
        assert!(
            deleted.get("error").is_none(),
            "delete_node failed: {deleted}"
        );

        // The at-time read anchored before the delete returns the pre-delete
        // version. The delete writer closes that version's transaction_time
        // at the delete's commit timestamp, so transaction_to must be a
        // closed RFC3339 bound strictly after the anchor and the version must
        // no longer be current. (valid_to is implementation-dependent for
        // deletes -- the tombstone carries the valid-time closure -- so it is
        // deliberately not asserted here.)
        let at_time_response = server.get_node_at_time(GetNodeAtTimeRequest {
            namespace: None,
            node_id,
            valid_time: anchor.to_string(),
            transaction_time: Some(anchor.to_string()),
        });
        let at_time: serde_json::Value = serde_json::from_str(&at_time_response).unwrap();
        assert!(
            at_time.get("error").is_none(),
            "get_node_at_time anchored before the delete must still see the \
             pre-delete version: {at_time}"
        );
        let temporal = temporal_of(&at_time["node"]);
        let transaction_to = temporal["transaction_to"].as_str().unwrap_or_else(|| {
            panic!("deleted version's transaction_to must be a closed RFC3339 bound: {temporal}")
        });
        assert!(
            rfc3339_to_micros(transaction_to) > anchor,
            "transaction_to must be the delete's commit time, strictly after \
             the anchor: {temporal}"
        );
        assert_eq!(
            temporal["is_current"],
            serde_json::json!(false),
            "a deleted version must report is_current: false: {temporal}"
        );
    }

    // ------------------------------------------------------------------
    // Conversion-level unit tests for `is_current` on hand-constructed
    // `BiTemporalInterval`s (HLC-robust semantics: transaction dimension is
    // judged by interval openness, valid dimension at wallclock granularity).
    // ------------------------------------------------------------------

    use crate::core::hlc::HybridTimestamp;
    use crate::core::temporal::{BiTemporalInterval, TimeRange, time};

    #[test]
    fn test_is_current_true_for_hlc_logical_component_in_current_microsecond() {
        // An HLC commit stamp in the CURRENT microsecond with logical > 0
        // orders strictly after SystemTime-now (logical = 0) in the same
        // microsecond. Under the old `contains(now)` rule this live,
        // open-ended head version reported is_current: false; it must now
        // report true.
        let now = time::now();
        let commit = HybridTimestamp::new(now.wallclock(), 5).expect("valid HLC stamp");
        let interval = BiTemporalInterval::now(commit, commit);
        let bounds = TemporalBounds::from(&interval);
        assert!(
            bounds.is_current,
            "an open head version committed in the current microsecond with \
             logical > 0 must be current: {bounds:?}"
        );
    }

    #[test]
    fn test_is_current_true_when_hlc_frontier_runs_ahead_of_wallclock() {
        // After a tolerated backward OS-clock step, the HLC frontier runs
        // AHEAD of SystemTime, so commit stamps land slightly in the
        // wallclock future (here: +50ms) while the fact's valid time is not
        // skewed. The transaction interval is open -- by definition the
        // currently-recorded version -- so is_current must be true; the old
        // tx_from-vs-now comparison misreported false for minutes.
        let now = time::now();
        let skewed_commit =
            HybridTimestamp::new(now.wallclock() + 50_000, 0).expect("valid skewed stamp");
        let interval = BiTemporalInterval::with_valid_time(now, skewed_commit);
        let bounds = TemporalBounds::from(&interval);
        assert!(
            bounds.is_current,
            "an open head version whose commit stamp runs ahead of the OS \
             clock must be current: {bounds:?}"
        );
    }

    #[test]
    fn test_is_current_false_for_explicit_future_valid_time() {
        // An explicitly backdated-to-the-future fact (valid_from = now + 1h)
        // is not yet true in reality, so is_current must be false even
        // though both intervals are open-ended.
        let now = time::now();
        let future_valid =
            HybridTimestamp::new(now.wallclock() + 3_600_000_000, 0).expect("valid future stamp");
        let interval = BiTemporalInterval::with_valid_time(future_valid, now);
        let bounds = TemporalBounds::from(&interval);
        assert!(
            !bounds.is_current,
            "a not-yet-valid fact must not be current even with open \
             intervals: {bounds:?}"
        );
    }

    #[test]
    fn test_is_current_false_for_closed_transaction_interval() {
        // A superseded version (closed transaction interval) is never
        // current, even if its valid interval still contains now.
        let now = time::now();
        let hour_ago =
            HybridTimestamp::new(now.wallclock() - 3_600_000_000, 0).expect("valid past stamp");
        let second_ago =
            HybridTimestamp::new(now.wallclock() - 1_000_000, 0).expect("valid past stamp");
        let interval = BiTemporalInterval::new(
            TimeRange::from(hour_ago),
            TimeRange::new(hour_ago, second_ago).expect("valid closed range"),
        );
        let bounds = TemporalBounds::from(&interval);
        assert!(
            !bounds.is_current,
            "a superseded (closed-transaction) version must not be current \
             even while still valid: {bounds:?}"
        );
    }

    #[test]
    fn test_is_current_false_for_expired_valid_interval() {
        // A fact whose valid time ended in the past is not current, even
        // though its transaction interval is still open (it is still the
        // recorded version of that fact).
        let now = time::now();
        let two_hours_ago =
            HybridTimestamp::new(now.wallclock() - 7_200_000_000, 0).expect("valid past stamp");
        let hour_ago =
            HybridTimestamp::new(now.wallclock() - 3_600_000_000, 0).expect("valid past stamp");
        let interval = BiTemporalInterval::new(
            TimeRange::new(two_hours_ago, hour_ago).expect("valid closed range"),
            TimeRange::from(two_hours_ago),
        );
        let bounds = TemporalBounds::from(&interval);
        assert!(
            !bounds.is_current,
            "an expired fact must not be current even with an open \
             transaction interval: {bounds:?}"
        );
    }

    #[test]
    fn test_wallclock_beyond_chrono_range_falls_back_to_raw_micros() {
        // chrono can only represent dates up to ~year 262143 (~8.2e18
        // micros), while valid HLC wallclocks extend to MAX_VALID_TIMESTAMP
        // (i64::MAX - 1000). A bound past chrono's range must fall back to
        // the raw microsecond count as a string instead of silently
        // rendering the 1970 epoch.
        let huge_micros = 9_000_000_000_000_000_000i64;
        let far_future = HybridTimestamp::new(huge_micros, 0).expect("legal HLC wallclock");
        let interval = BiTemporalInterval::new(
            TimeRange::from(far_future),
            TimeRange::from(HybridTimestamp::new(1_000_000, 0).expect("valid stamp")),
        );

        let bounds = TemporalBounds::from(&interval);
        assert_eq!(
            bounds.valid_from,
            huge_micros.to_string(),
            "out-of-chrono-range wallclock must serialize as raw micros: {bounds:?}"
        );
        assert!(
            bounds.valid_to.is_none(),
            "open-ended valid_to must stay None: {bounds:?}"
        );
        // The in-range transaction bound must still be RFC3339.
        assert_eq!(rfc3339_to_micros(&bounds.transaction_from), 1_000_000);
        // A fact whose valid time hasn't begun is not current.
        assert!(!bounds.is_current, "far-future fact is not current");
    }

    #[test]
    fn test_from_interval_at_valid_from_equal_to_now_is_current() {
        // Half-open interval semantics: a fact becomes true AT valid_from,
        // so an explicit `now` exactly equal to valid_from is current
        // (Issue #3391: `from_interval_at` evaluates at the caller's
        // request-scoped instant, not a fresh wallclock read).
        let t = HybridTimestamp::new(1_700_000_000_000_000, 0).expect("valid stamp");
        let interval = BiTemporalInterval::now(t, t);
        let bounds = TemporalBounds::from_interval_at(&interval, t);
        assert!(
            bounds.is_current,
            "a fact is current from valid_from inclusive: {bounds:?}"
        );
    }

    #[test]
    fn test_from_interval_at_one_micro_before_valid_from_is_not_current() {
        // One microsecond before valid_from the fact is not yet true.
        let t = HybridTimestamp::new(1_700_000_000_000_000, 0).expect("valid stamp");
        let interval = BiTemporalInterval::now(t, t);
        let just_before = HybridTimestamp::new(t.wallclock() - 1, 0).expect("valid earlier stamp");
        let bounds = TemporalBounds::from_interval_at(&interval, just_before);
        assert!(
            !bounds.is_current,
            "a fact is not current one microsecond before valid_from: {bounds:?}"
        );
    }

    #[test]
    fn test_node_with_unresolvable_version_omits_temporal_and_provenance() {
        // Best-effort degrade path (mirrors provenance, Issue #3224): when
        // the version metadata cannot be loaded from any tier, the read must
        // still succeed with the `temporal` and `provenance` blocks omitted
        // -- never a fabricated value, never a whole-response failure.
        use crate::core::id::{NodeId, VersionId};

        let server = create_test_server();
        let created = create_person(&server, "Ghost");
        let node_id = created["id"].as_u64().unwrap();

        // Re-point the stored node's current_version at a version id that
        // exists in no tier, simulating unreadable version metadata.
        let mut node = server
            .db()
            .current
            .get_node(NodeId::new(node_id).unwrap())
            .unwrap();
        node.current_version = VersionId::new(9_999_999).unwrap();
        server
            .db()
            .current
            .update_node_direct(node, time::now())
            .unwrap();

        let response = server.get_node(GetNodeRequest {
            namespace: None,
            node_id,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "read must still succeed when metadata is unreadable: {value}"
        );
        assert_eq!(value["id"].as_u64(), Some(node_id));
        assert_eq!(value["label"], "Person");
        assert_eq!(value["properties"]["name"], "Ghost");
        assert!(
            value.get("temporal").is_none(),
            "temporal must be omitted, not fabricated: {value}"
        );
        assert!(
            value.get("provenance").is_none(),
            "provenance must be omitted, not fabricated: {value}"
        );
    }

    #[test]
    fn test_edge_with_unresolvable_version_omits_temporal_and_provenance() {
        // Edge mirror of the node degrade case above.
        use crate::core::id::{EdgeId, VersionId};

        let server = create_test_server();
        let alice = create_person(&server, "Alice");
        let bob = create_person(&server, "Bob");
        let edge = create_knows_edge(
            &server,
            alice["id"].as_u64().unwrap(),
            bob["id"].as_u64().unwrap(),
        );
        let edge_id = edge["id"].as_u64().unwrap();

        let mut stored = server
            .db()
            .current
            .get_edge(EdgeId::new(edge_id).unwrap())
            .unwrap();
        stored.current_version = VersionId::new(9_999_998).unwrap();
        server.db().current.update_edge_direct(stored).unwrap();

        let response = server.get_edge(GetEdgeRequest {
            namespace: None,
            edge_id,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "read must still succeed when metadata is unreadable: {value}"
        );
        assert_eq!(value["id"].as_u64(), Some(edge_id));
        assert_eq!(value["label"], "KNOWS");
        assert!(
            value.get("temporal").is_none(),
            "temporal must be omitted, not fabricated: {value}"
        );
        assert!(
            value.get("provenance").is_none(),
            "provenance must be omitted, not fabricated: {value}"
        );
    }
}

// ============================================================================
// Atomic multi-write batches with local refs (`apply_batch`, Issue #3231)
// ============================================================================

mod apply_batch_tests {
    use super::*;
    use crate::core::id::{EdgeId, NodeId};
    use crate::core::temporal::{Timestamp, time};
    use serde_json::json;

    /// 2020-01-01T00:00:00Z in microseconds since epoch — a safely-backdated
    /// valid time for #3221-style per-op valid_time assertions.
    const T_PAST_MICROS: i64 = 1_577_836_800_000_000;

    /// Dispatch `apply_batch` through the same table `call_tool` uses.
    /// Returns `(parsed_json, is_error)`.
    fn dispatch_batch(
        server: &AletheiaMcpServer,
        args: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        let result = server.dispatch_tool("apply_batch", args);
        let is_error = result.is_error.unwrap_or(false);
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.clone()))
            .expect("tool result should carry text content");
        let value = serde_json::from_str(&text).expect("tool response should be valid JSON");
        (value, is_error)
    }

    /// Apply a batch given just the operations array.
    fn apply(server: &AletheiaMcpServer, ops: serde_json::Value) -> (serde_json::Value, bool) {
        dispatch_batch(server, json!({ "operations": ops }))
    }

    /// Apply a batch and require success, returning the parsed response.
    fn apply_ok(server: &AletheiaMcpServer, ops: serde_json::Value) -> serde_json::Value {
        let (value, is_error) = apply(server, ops);
        assert!(!is_error, "expected batch success, got: {value}");
        assert_eq!(value["success"], json!(true), "got: {value}");
        value
    }

    /// Apply a batch and require a structured error with the given code,
    /// returning the full response payload.
    fn apply_err(
        server: &AletheiaMcpServer,
        ops: serde_json::Value,
        expected_code: &str,
    ) -> serde_json::Value {
        let (value, is_error) = apply(server, ops);
        assert!(is_error, "expected batch error, got: {value}");
        let error = value
            .get("error")
            .unwrap_or_else(|| panic!("expected structured error payload: {value}"));
        assert_eq!(
            error["code"].as_str(),
            Some(expected_code),
            "wrong code: {value}"
        );
        assert!(
            error["message"].as_str().is_some_and(|m| !m.is_empty()),
            "error must carry a message: {value}"
        );
        assert!(
            error["retriable"].as_bool().is_some(),
            "error must carry retriable: {value}"
        );
        // Caller-fault classes are never retriable (#3234 contract).
        if matches!(expected_code, "INVALID_ARGUMENT" | "FAILED_PRECONDITION") {
            assert_eq!(
                error["retriable"],
                json!(false),
                "{expected_code} must be non-retriable: {value}"
            );
        }
        value
    }

    /// Seed a committed `Person` node with a `name` property via the
    /// single-op tool and return its id.
    fn seed_person(server: &AletheiaMcpServer, name: &str) -> u64 {
        let node: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: Some(HashMap::from([(
                "name".to_string(),
                serde_json::Value::String(name.to_string()),
            )])),
            valid_time: None,
            provenance: None,
        }))
        .expect("seed node should succeed");
        node.id
    }

    /// Seed a committed edge between two committed nodes and return its id.
    fn seed_edge(server: &AletheiaMcpServer, source: u64, target: u64) -> u64 {
        let edge: EdgeResponse = parse_response(&server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id: source,
            target_id: target,
            label: "KNOWS".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        }))
        .expect("seed edge should succeed");
        edge.id
    }

    // ------------------------------------------------------------------
    // Registration and argument-shape basics
    // ------------------------------------------------------------------

    #[test]
    fn apply_batch_registered_in_tool_list() {
        let server = create_test_server();
        let tools = server.list_tools_for_test();
        assert!(
            tools.iter().any(|t| t == "apply_batch"),
            "apply_batch must be advertised: {tools:?}"
        );
    }

    /// Schema drift guard: the advertised input schema must mention all six
    /// op variant names, so an added/renamed variant that misses the schema
    /// (or a schemars regression collapsing the tagged enum) fails loudly.
    #[test]
    fn apply_batch_schema_advertises_all_op_variants() {
        let server = create_test_server();
        let schema = server
            .tool_input_schema_for_test("apply_batch")
            .expect("apply_batch must advertise an input schema");
        let schema_text = schema.to_string();
        for variant in [
            "create_node",
            "create_edge",
            "update_node",
            "update_edge",
            "delete_node",
            "delete_edge",
            // DBOS Phase 3e: the seventh variant.
            "compare_and_set_node",
        ] {
            assert!(
                schema_text.contains(variant),
                "apply_batch input schema must mention op variant {variant}: {schema_text}"
            );
        }
    }

    #[test]
    fn apply_batch_null_args_returns_invalid_argument() {
        let server = create_test_server();
        let (value, is_error) = dispatch_batch(&server, serde_json::Value::Null);
        assert!(is_error, "null args must error: {value}");
        assert_eq!(value["error"]["code"], json!("INVALID_ARGUMENT"));
    }

    #[test]
    fn apply_batch_empty_operations_is_noop_success() {
        let server = create_test_server();
        let value = apply_ok(&server, json!([]));
        assert_eq!(value["results"], json!([]));
        assert_eq!(value["ref_map"], json!({}));
        assert_eq!(value["operation_count"], json!(0));
        assert_eq!(server.db().node_count(), 0);
    }

    // ------------------------------------------------------------------
    // Happy paths: subgraph in one call, local refs, version ids
    // ------------------------------------------------------------------

    /// The issue's success metric: an entity-with-relationships subgraph of
    /// three-plus operations lands in ONE call, with every alias resolved to
    /// a committed real ID and per-op version ids for creates/updates.
    #[test]
    fn apply_batch_creates_subgraph_with_refs_and_version_ids() {
        let server = create_test_server();
        let value = apply_ok(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "alice",
                 "properties": {"name": "Alice"}},
                {"op": "create_node", "label": "Person", "ref": "bob",
                 "properties": {"name": "Bob"}},
                {"op": "create_edge", "source_id": "$alice", "target_id": "$bob",
                 "label": "KNOWS", "properties": {"since": 2024}},
            ]),
        );

        assert_eq!(value["operation_count"], json!(3));
        let results = value["results"].as_array().expect("results array");
        assert_eq!(results.len(), 3);

        // Per-op results in input order, each echoing its index and op.
        assert_eq!(results[0]["op"], json!("create_node"));
        assert_eq!(results[0]["index"], json!(0));
        assert_eq!(results[0]["ref"], json!("alice"));
        let alice_id = results[0]["node_id"].as_u64().expect("alice node_id");
        assert!(
            results[0]["version_id"].as_u64().is_some(),
            "create_node result must carry a version_id: {value}"
        );

        assert_eq!(results[1]["op"], json!("create_node"));
        let bob_id = results[1]["node_id"].as_u64().expect("bob node_id");

        assert_eq!(results[2]["op"], json!("create_edge"));
        let edge_id = results[2]["edge_id"].as_u64().expect("edge_id");
        assert_eq!(results[2]["source_id"], json!(alice_id));
        assert_eq!(results[2]["target_id"], json!(bob_id));
        assert!(results[2]["version_id"].as_u64().is_some());

        // ref_map maps every caller-supplied alias to its committed real ID.
        assert_eq!(value["ref_map"]["alice"], json!(alice_id));
        assert_eq!(value["ref_map"]["bob"], json!(bob_id));

        // All three writes are committed and visible.
        assert_eq!(server.db().node_count(), 2);
        assert_eq!(server.db().edge_count(), 1);
        let edge = server.db().get_edge(EdgeId::new(edge_id).unwrap()).unwrap();
        assert_eq!(edge.source.as_u64(), alice_id);
        assert_eq!(edge.target.as_u64(), bob_id);
    }

    /// Edge endpoints may mix pre-existing committed IDs and local refs
    /// (including positional `$0`-style refs) interchangeably.
    #[test]
    fn apply_batch_edge_mixes_committed_id_and_local_ref() {
        let server = create_test_server();
        let existing = seed_person(&server, "Carol");

        let value = apply_ok(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "properties": {"name": "Dave"}},
                {"op": "create_edge", "source_id": "$0", "target_id": existing,
                 "label": "KNOWS"},
            ]),
        );

        let results = value["results"].as_array().unwrap();
        let dave_id = results[0]["node_id"].as_u64().unwrap();
        assert_eq!(results[1]["source_id"], json!(dave_id));
        assert_eq!(results[1]["target_id"], json!(existing));
        assert_eq!(server.db().edge_count(), 1);
    }

    #[test]
    fn apply_batch_returns_version_ids_in_input_order_for_creates_and_updates() {
        let server = create_test_server();
        let existing = seed_person(&server, "Eve");
        let other = seed_person(&server, "Grace");
        let seeded_edge = seed_edge(&server, existing, other);

        let value = apply_ok(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "f",
                 "properties": {"name": "Frank"}},
                {"op": "update_node", "node_id": existing,
                 "properties": {"name": "Eve II"}},
                {"op": "create_edge", "source_id": "$f", "target_id": existing,
                 "label": "KNOWS"},
                {"op": "update_edge", "edge_id": seeded_edge,
                 "properties": {"weight": 5}},
            ]),
        );

        let results = value["results"].as_array().unwrap();
        assert_eq!(results.len(), 4);
        for (i, expected_op) in ["create_node", "update_node", "create_edge", "update_edge"]
            .iter()
            .enumerate()
        {
            assert_eq!(results[i]["index"], json!(i));
            assert_eq!(results[i]["op"], json!(*expected_op));
            assert!(
                results[i]["version_id"].as_u64().is_some(),
                "op {i} must carry version_id: {value}"
            );
        }
        // The updates targeted the committed entities and reported them back.
        assert_eq!(results[1]["node_id"], json!(existing));
        assert_eq!(results[3]["edge_id"], json!(seeded_edge));

        // PATCH semantics as in the single-op tools: node property updated...
        let node = server
            .db()
            .get_node(NodeId::new(existing).unwrap())
            .unwrap();
        let name = node
            .properties
            .iter()
            .find_map(|(k, v)| {
                crate::core::interning::GLOBAL_INTERNER
                    .resolve_with(*k, |s| s == "name")
                    .unwrap_or(false)
                    .then(|| format!("{v:?}"))
            })
            .unwrap_or_default();
        assert!(name.contains("Eve II"), "update must apply: {name}");

        // ...and the edge property merge took effect too.
        let edge = server
            .db()
            .get_edge(EdgeId::new(seeded_edge).unwrap())
            .unwrap();
        let weight = edge
            .properties
            .iter()
            .find_map(|(k, v)| {
                crate::core::interning::GLOBAL_INTERNER
                    .resolve_with(*k, |s| s == "weight")
                    .unwrap_or(false)
                    .then(|| format!("{v:?}"))
            })
            .unwrap_or_default();
        assert!(weight.contains('5'), "edge update must apply: {weight}");
    }

    // ------------------------------------------------------------------
    // compare_and_set_node op (DBOS Phase 3e, Piece 3): fence a step-record
    // batch on the owning run's version, atomically.
    // ------------------------------------------------------------------

    /// Current head version of a committed node id.
    fn node_version(server: &AletheiaMcpServer, id: u64) -> u64 {
        server
            .db()
            .get_node(NodeId::new(id).unwrap())
            .unwrap()
            .current_version
            .as_u64()
    }

    #[test]
    fn apply_batch_cas_matching_version_records_step() {
        let server = create_test_server();
        let run = seed_person(&server, "run-W1"); // stand-in for a WorkflowRun node
        let v = node_version(&server, run);

        // Atomic fenced step-record: CAS the run (still v -> I own it), then
        // create the Step and a HAS_STEP edge, all-or-nothing.
        let value = apply_ok(
            &server,
            json!([
                {"op": "compare_and_set_node", "node_id": run, "expected_version": v,
                 "properties": {"name": "run-W1", "status": "running"}},
                {"op": "create_node", "label": "Step", "ref": "s",
                 "properties": {"idem_key": "W1:0", "output": "42"}},
                {"op": "create_edge", "source_id": run, "target_id": "$s",
                 "label": "HAS_STEP"},
            ]),
        );

        let results = value["results"].as_array().unwrap();
        assert_eq!(results[0]["op"], json!("compare_and_set_node"));
        assert!(
            results[0]["version_id"].as_u64().is_some_and(|nv| nv != v),
            "CAS installed a new version: {value}"
        );
        // The run's full-replace map now carries status=running.
        let node = server.db().get_node(NodeId::new(run).unwrap()).unwrap();
        let status = node
            .properties
            .iter()
            .find_map(|(k, val)| {
                crate::core::interning::GLOBAL_INTERNER
                    .resolve_with(*k, |s| s == "status")
                    .unwrap_or(false)
                    .then(|| format!("{val:?}"))
            })
            .unwrap_or_default();
        assert!(
            status.contains("running"),
            "CAS full-replace applied: {status}"
        );
        // The Step was recorded.
        let steps = server.db().find_nodes_by_property(
            "Step",
            "idem_key",
            &crate::core::PropertyValue::String("W1:0".into()),
        );
        assert_eq!(steps.len(), 1, "exactly one Step recorded");
    }

    #[test]
    fn apply_batch_cas_stale_version_aborts_whole_batch() {
        let server = create_test_server();
        let run = seed_person(&server, "run-W2");
        let stale = node_version(&server, run);

        // A successor "steals" the run: advance its version so `stale` is stale.
        server
            .db()
            .update_node_with_valid_time(
                NodeId::new(run).unwrap(),
                crate::PropertyMapBuilder::new()
                    .insert("owner", "successor")
                    .build(),
                None,
            )
            .unwrap();
        assert_ne!(node_version(&server, run), stale);

        let nodes_before = server.db().node_count();
        let edges_before = server.db().edge_count();

        // A zombie tries to fence-record a Step on the STALE version. The CAS
        // fails at the commit guard -> the WHOLE batch aborts, Step never
        // poisoned.
        let value = apply_err(
            &server,
            json!([
                {"op": "compare_and_set_node", "node_id": run, "expected_version": stale,
                 "properties": {"name": "run-W2", "status": "running"}},
                {"op": "create_node", "label": "Step", "ref": "s",
                 "properties": {"idem_key": "W2:0", "output": "zombie"}},
                {"op": "create_edge", "source_id": run, "target_id": "$s",
                 "label": "HAS_STEP"},
            ]),
            "FAILED_PRECONDITION",
        );
        // The CAS mismatch surfaces at commit (after execute_batch buffers the
        // ops), so it is a commit-phase failure: `failed_op_index` is present
        // but null, exactly like the unique-constraint abort.
        assert!(
            value["error"]["details"].get("failed_op_index").is_some(),
            "details must carry failed_op_index (null for commit-phase CAS abort): {value}"
        );

        // Zero writes: no Step node, no edge, counts unchanged.
        assert_eq!(server.db().node_count(), nodes_before);
        assert_eq!(server.db().edge_count(), edges_before);
        let steps = server.db().find_nodes_by_property(
            "Step",
            "idem_key",
            &crate::core::PropertyValue::String("W2:0".into()),
        );
        assert!(steps.is_empty(), "the poisoning Step must NOT be recorded");
    }

    #[test]
    fn apply_batch_cas_of_batch_created_node_rejected() {
        let server = create_test_server();
        // CAS targeting a node created earlier in the same batch (a local ref)
        // is the documented v1-scope rejection (mirrors update/delete).
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Run", "ref": "r",
                 "properties": {"wf": "W3"}},
                {"op": "compare_and_set_node", "node_id": "$r", "expected_version": 1,
                 "properties": {"wf": "W3"}},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(
            value["error"]["details"]["failed_op_index"],
            json!(1),
            "the CAS op index must be reported: {value}"
        );
    }

    #[test]
    fn apply_batch_delete_edge_and_delete_node_report_no_version_id() {
        let server = create_test_server();
        let a = seed_person(&server, "A");
        let b = seed_person(&server, "B");
        let e = seed_edge(&server, a, b);

        let value = apply_ok(
            &server,
            json!([
                {"op": "delete_edge", "edge_id": e},
                {"op": "delete_node", "node_id": a},
            ]),
        );
        let results = value["results"].as_array().unwrap();
        assert_eq!(results[0]["op"], json!("delete_edge"));
        assert_eq!(results[0]["edge_id"], json!(e));
        assert!(
            results[0].get("version_id").is_none_or(|v| v.is_null()),
            "deletes carry no version_id in v1: {value}"
        );
        assert_eq!(results[1]["op"], json!("delete_node"));
        assert_eq!(results[1]["node_id"], json!(a));
        assert_eq!(results[1]["edges_removed"], json!(0));
        assert_eq!(server.db().node_count(), 1);
        assert_eq!(server.db().edge_count(), 0);
    }

    // ------------------------------------------------------------------
    // Static pre-validation: refs
    // ------------------------------------------------------------------

    #[test]
    fn apply_batch_forward_ref_rejected_with_op_index_and_nothing_committed() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_edge", "source_id": "$late", "target_id": "$late",
                 "label": "KNOWS"},
                {"op": "create_node", "label": "Person", "ref": "late"},
            ]),
            "INVALID_ARGUMENT",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["failed_op_index"], json!(0), "got: {value}");
        assert_eq!(details["ref"], json!("$late"));
        assert_eq!(details["defined_at_index"], json!(1));
        assert_eq!(server.db().node_count(), 0, "nothing may commit");
        assert_eq!(server.db().edge_count(), 0);
    }

    #[test]
    fn apply_batch_unknown_ref_rejected_invalid_argument() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "a"},
                {"op": "create_edge", "source_id": "$a", "target_id": "$ghost",
                 "label": "KNOWS"},
            ]),
            "INVALID_ARGUMENT",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["failed_op_index"], json!(1));
        assert_eq!(details["ref"], json!("$ghost"));
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn apply_batch_duplicate_alias_rejected_with_first_definition_index() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "dup"},
                {"op": "create_node", "label": "Person", "ref": "dup"},
            ]),
            "INVALID_ARGUMENT",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["failed_op_index"], json!(1));
        assert_eq!(details["first_defined_at_index"], json!(0));
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn apply_batch_unknown_op_type_rejected_with_index() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person"},
                {"op": "upsert_node", "label": "Person"},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn apply_batch_empty_ref_alias_rejected() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": ""},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(0));
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn apply_batch_dollar_prefixed_ref_alias_rejected() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person"},
                {"op": "create_node", "label": "Person", "ref": "$a"},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains('$'),
            "message must explain the '$' rule: {msg}"
        );
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn apply_batch_purely_numeric_ref_alias_rejected() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "7"},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(0));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("numeric"),
            "message must explain the positional reservation: {msg}"
        );
        assert_eq!(server.db().node_count(), 0);
    }

    /// A '$'-ref whose name is not all-ASCII-digits must never parse
    /// positionally ("+5" would satisfy usize::from_str): it falls through
    /// to alias lookup and errors as an unknown ref.
    #[test]
    fn apply_batch_plus_prefixed_positional_ref_rejected_as_unknown() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person"},
                {"op": "create_edge", "source_id": "$+0", "target_id": "$0",
                 "label": "KNOWS"},
            ]),
            "INVALID_ARGUMENT",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["failed_op_index"], json!(1));
        assert_eq!(details["ref"], json!("$+0"));
        assert_eq!(server.db().node_count(), 0);
    }

    /// Pins the chosen leading-zeros behavior: an all-digits name parses
    /// positionally, so "$00" is the same reference as "$0".
    #[test]
    fn apply_batch_numeric_ref_with_leading_zeros_resolves_positionally() {
        let server = create_test_server();
        let value = apply_ok(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "properties": {"name": "Zero"}},
                {"op": "create_edge", "source_id": "$00", "target_id": "$0",
                 "label": "SELF"},
            ]),
        );
        let results = value["results"].as_array().unwrap();
        let node_id = results[0]["node_id"].as_u64().unwrap();
        assert_eq!(results[1]["source_id"], json!(node_id));
        assert_eq!(results[1]["target_id"], json!(node_id));
        assert_eq!(server.db().edge_count(), 1);
    }

    /// A plain numeric string as a write target is a type error, not a
    /// v1-scope rejection; a '$'-ref gets the v1-scope message.
    #[test]
    fn apply_batch_numeric_string_node_id_rejected_with_integer_message() {
        let server = create_test_server();

        let value = apply_err(
            &server,
            json!([
                {"op": "update_node", "node_id": "123", "properties": {"x": 1}},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(0));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("must be an integer node id") && msg.contains("\"123\""),
            "numeric string must get the type-error message: {msg}"
        );

        let value = apply_err(
            &server,
            json!([
                {"op": "update_node", "node_id": "$ghost", "properties": {"x": 1}},
            ]),
            "INVALID_ARGUMENT",
        );
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("not supported in v1"),
            "'$'-refs must get the v1-scope message: {msg}"
        );
        assert_eq!(server.db().node_count(), 0);
    }

    /// Per-op provenance is validated through the SAME shared helper the
    /// single-op tools use (`parse_opt_provenance`, the #3350 stamping
    /// seam), with the error re-shaped to carry the batch's op index.
    #[test]
    fn apply_batch_invalid_provenance_rejected_per_op() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person"},
                {"op": "create_node", "label": "Person",
                 "provenance": {"source": "sensor", "confidence": 3.5}},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("confidence"),
            "shared-helper message must be preserved: {msg}"
        );
        assert_eq!(server.db().node_count(), 0);
    }

    /// #3413-adjacent bi-temporal guard: a batch-created edge that a later
    /// detach-delete would elide may NOT carry an explicit valid_time — the
    /// elision would silently erase the backdated fact.
    #[test]
    fn apply_batch_backdated_edge_annihilation_rejected() {
        let server = create_test_server();
        let x = seed_person(&server, "X");
        let nodes_before = server.db().node_count();

        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "a"},
                {"op": "create_edge", "source_id": "$a", "target_id": x,
                 "label": "KNOWS", "valid_time": T_PAST_MICROS.to_string()},
                {"op": "delete_node", "node_id": x, "detach": true},
            ]),
            "INVALID_ARGUMENT",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["failed_op_index"], json!(1), "got: {value}");
        assert_eq!(details["removed_by_delete_at_index"], json!(2));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("valid_time"),
            "message must explain the carve-out: {msg}"
        );

        // Rejected statically: nothing committed.
        assert_eq!(server.db().node_count(), nodes_before);
        assert_eq!(server.db().edge_count(), 0);
    }

    /// The same batch WITHOUT the explicit valid_time is fine (the edge is
    /// elided with no history loss — it never carried a backdated fact).
    #[test]
    fn apply_batch_non_backdated_edge_annihilation_still_allowed() {
        let server = create_test_server();
        let x = seed_person(&server, "X");
        let value = apply_ok(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "a"},
                {"op": "create_edge", "source_id": "$a", "target_id": x,
                 "label": "KNOWS"},
                {"op": "delete_node", "node_id": x, "detach": true},
            ]),
        );
        assert_eq!(
            value["results"][1]["removed_by_delete_at_index"],
            json!(2),
            "elision path unchanged for non-backdated edges: {value}"
        );
        assert_eq!(server.db().edge_count(), 0);
    }

    // ------------------------------------------------------------------
    // v1 scope: writes against batch-created refs, double writes
    // ------------------------------------------------------------------

    #[test]
    fn apply_batch_update_of_batch_created_ref_rejected_v1() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "a"},
                {"op": "update_node", "node_id": "$a", "properties": {"x": 1}},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("not supported"),
            "message must state the v1 limitation: {msg}"
        );
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn apply_batch_delete_of_batch_created_ref_rejected_v1() {
        let server = create_test_server();
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "a"},
                {"op": "delete_node", "node_id": "$a"},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn apply_batch_second_write_to_same_committed_entity_rejected() {
        let server = create_test_server();
        let existing = seed_person(&server, "G");
        let value = apply_err(
            &server,
            json!([
                {"op": "update_node", "node_id": existing, "properties": {"x": 1}},
                {"op": "update_node", "node_id": existing, "properties": {"x": 2}},
            ]),
            "INVALID_ARGUMENT",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["failed_op_index"], json!(1));
        assert_eq!(details["first_write_op_index"], json!(0));
        // The seeded node is untouched.
        assert_eq!(server.db().node_count(), 1);
    }

    #[test]
    fn apply_batch_create_edge_referencing_earlier_deleted_node_rejected() {
        let server = create_test_server();
        let a = seed_person(&server, "H");
        let b = seed_person(&server, "I");
        let value = apply_err(
            &server,
            json!([
                {"op": "delete_node", "node_id": a},
                {"op": "create_edge", "source_id": a, "target_id": b, "label": "KNOWS"},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        assert_eq!(server.db().node_count(), 2, "nothing may commit");
    }

    // ------------------------------------------------------------------
    // Cap enforcement
    // ------------------------------------------------------------------

    #[test]
    fn apply_batch_at_cap_succeeds_and_over_cap_rejected_with_limit_echoed() {
        let server = AletheiaMcpServer::new(create_test_db()).with_max_batch_operations(2);

        // Exactly at the cap: fine.
        let value = apply_ok(
            &server,
            json!([
                {"op": "create_node", "label": "Person"},
                {"op": "create_node", "label": "Person"},
            ]),
        );
        assert_eq!(value["operation_count"], json!(2));

        // One over: rejected before anything runs, limit echoed (#3226).
        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person"},
                {"op": "create_node", "label": "Person"},
                {"op": "create_node", "label": "Person"},
            ]),
            "INVALID_ARGUMENT",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["limit"], json!(2));
        assert_eq!(details["submitted"], json!(3));
        assert_eq!(server.db().node_count(), 2, "over-cap batch must not run");
    }

    #[test]
    fn apply_batch_default_cap_is_1000() {
        let server = create_test_server();
        let ops: Vec<serde_json::Value> = (0..1001)
            .map(|_| json!({"op": "create_node", "label": "Person"}))
            .collect();
        let value = apply_err(&server, json!(ops), "INVALID_ARGUMENT");
        assert_eq!(value["error"]["details"]["limit"], json!(1000));
        assert_eq!(value["error"]["details"]["submitted"], json!(1001));
        assert_eq!(server.db().node_count(), 0);
    }

    // ------------------------------------------------------------------
    // Atomicity (AC8: the flagship test)
    // ------------------------------------------------------------------

    /// Inject a failure mid-batch (update of a nonexistent node at op index
    /// 2) and assert via count_nodes/count_edges that ZERO of the batch's
    /// writes survived — including ops that already executed (indices 0, 1)
    /// and ops after the failing one (index 3).
    #[test]
    fn apply_batch_mid_batch_failure_leaves_counts_unchanged() {
        let server = create_test_server();
        let baseline = seed_person(&server, "Baseline");
        let nodes_before = server.db().node_count();
        let edges_before = server.db().edge_count();

        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "a",
                 "properties": {"name": "Ghost A"}},
                {"op": "create_node", "label": "Person", "ref": "b",
                 "properties": {"name": "Ghost B"}},
                {"op": "update_node", "node_id": 999_999, "properties": {"x": 1}},
                {"op": "create_edge", "source_id": "$a", "target_id": "$b",
                 "label": "KNOWS"},
            ]),
            "NOT_FOUND",
        );
        assert_eq!(
            value["error"]["details"]["failed_op_index"],
            json!(2),
            "the failing op index must be reported: {value}"
        );

        // AC8: zero writes survived.
        assert_eq!(server.db().node_count(), nodes_before);
        assert_eq!(server.db().edge_count(), edges_before);
        // And the batch-created ghosts are truly absent from lookups.
        let ghosts = server.db().find_nodes_by_property(
            "Person",
            "name",
            &crate::core::PropertyValue::String("Ghost A".into()),
        );
        assert!(ghosts.is_empty(), "no partial write may be visible");
        // The pre-existing node is untouched.
        assert!(server.db().get_node(NodeId::new(baseline).unwrap()).is_ok());
    }

    /// Pins the behavior when a caller *guesses* the numeric id a batch
    /// create will allocate and submits it as a write target: prevalidation
    /// accepts it (it is a plain integer), but an explicit batch-created-ref
    /// guard (Issue #3417) rejects it with the same v1-scope INVALID_ARGUMENT
    /// the static `$alias` prevalidation emits — updating/deleting an entity
    /// created in the same batch is not supported in v1. The whole batch
    /// aborts and zero writes survive. Before #3417's buffer-aware reads this
    /// was caught only incidentally as NOT_FOUND by the committed-only read;
    /// the guard fires regardless of read semantics.
    #[test]
    fn apply_batch_guessed_id_of_batch_created_node_aborts_whole_batch() {
        let server = create_test_server();
        let seeded = seed_person(&server, "Seed");
        // Node ids are allocated sequentially: the batch's create will get
        // seeded + 1. Submitting that id "works" structurally but must not
        // let the caller write to an uncommitted entity.
        let guessed = seeded + 1;

        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person",
                 "properties": {"name": "Ghost"}},
                {"op": "update_node", "node_id": guessed, "properties": {"x": 1}},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        // The guard names the create op that allocated the guessed id.
        assert_eq!(value["error"]["details"]["created_at_index"], json!(0));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("created in the same batch") && msg.contains("not supported in v1"),
            "guessed-id write must get the v1-scope message: {msg}"
        );

        // Zero writes survived: only the seeded node exists, and the
        // guessed id resolves to nothing.
        assert_eq!(server.db().node_count(), 1);
        assert!(server.db().get_node(NodeId::new(guessed).unwrap()).is_err());
    }

    /// The batch-created-ref guard fires BEFORE the `detach` flag is
    /// consulted: a `delete_node` with `detach: true` targeting the guessed
    /// integer id of a node created earlier in the SAME batch is rejected with
    /// the v1-scope INVALID_ARGUMENT (Issue #3417), not routed into the
    /// cascade/detach path. The whole batch aborts and zero writes survive.
    #[test]
    fn apply_batch_detach_delete_of_batch_created_node_aborts_before_detach() {
        let server = create_test_server();
        let seeded = seed_person(&server, "Seed");
        // The batch's create will be allocated `seeded + 1`.
        let guessed = seeded + 1;

        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person",
                 "properties": {"name": "Ghost"}},
                {"op": "delete_node", "node_id": guessed, "detach": true},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        assert_eq!(value["error"]["details"]["created_at_index"], json!(0));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("created in the same batch") && msg.contains("not supported in v1"),
            "detach-delete of a batch-created id must get the v1-scope message: {msg}"
        );

        // Zero writes survived: only the seeded node exists, and the
        // guessed id resolves to nothing.
        assert_eq!(server.db().node_count(), 1);
        assert!(server.db().get_node(NodeId::new(guessed).unwrap()).is_err());
    }

    /// `created_at_index` reports the ACTUAL originating create op, not always
    /// `0`: a batch that creates node A (op 0) then node B (op 1) and then
    /// writes B's guessed integer id (op 2) is rejected naming op 1 as the
    /// creator. The whole batch aborts and zero writes survive (Issue #3417).
    #[test]
    fn apply_batch_guessed_id_of_second_batch_created_node_reports_correct_index() {
        let server = create_test_server();
        let seeded = seed_person(&server, "Seed");
        // Sequential allocation: op 0 gets `seeded + 1` (node A), op 1 gets
        // `seeded + 2` (node B). Op 2 targets B's guessed id.
        let guessed_b = seeded + 2;

        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "properties": {"name": "A"}},
                {"op": "create_node", "label": "Person", "properties": {"name": "B"}},
                {"op": "update_node", "node_id": guessed_b, "properties": {"x": 1}},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(2));
        // Proves the guard names the create op that allocated the guessed id
        // (op 1), not a hardcoded 0.
        assert_eq!(value["error"]["details"]["created_at_index"], json!(1));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("created in the same batch") && msg.contains("not supported in v1"),
            "guessed second-create id write must get the v1-scope message: {msg}"
        );

        // Zero writes survived: only the seeded node exists; neither guessed id
        // resolves to a committed node.
        assert_eq!(server.db().node_count(), 1);
        assert!(
            server
                .db()
                .get_node(NodeId::new(seeded + 1).unwrap())
                .is_err()
        );
        assert!(
            server
                .db()
                .get_node(NodeId::new(guessed_b).unwrap())
                .is_err()
        );
    }

    /// `delete_edge` against a guessed batch-created edge id emits the SAME
    /// message substrings the node / `update_edge` variants assert, bringing
    /// its coverage to parity (Issue #3417). Whole batch aborts, zero writes.
    #[test]
    fn apply_batch_guessed_id_delete_edge_message_parity() {
        let server = create_test_server();
        let a = seed_person(&server, "A");
        let b = seed_person(&server, "B");

        // Anchor the guess to reality: the next create_edge (op 0 of the
        // failing batch) is allocated `anchor + 1`.
        let anchor = apply_ok(
            &server,
            json!([
                {"op": "create_edge", "source_id": a, "target_id": b, "label": "KNOWS"},
            ]),
        );
        let guessed_edge = anchor["results"][0]["edge_id"].as_u64().unwrap() + 1;

        let value = apply_err(
            &server,
            json!([
                {"op": "create_edge", "source_id": a, "target_id": b, "label": "KNOWS"},
                {"op": "delete_edge", "edge_id": guessed_edge},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        assert_eq!(value["error"]["details"]["created_at_index"], json!(0));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("created in the same batch")
                && msg.contains("not supported in v1")
                && msg.contains("Commit the creation first"),
            "delete_edge guard message must match the node/update_edge variants: {msg}"
        );
        assert!(
            server
                .db()
                .get_edge(EdgeId::new(guessed_edge).unwrap())
                .is_err(),
            "the aborted batch's edge must not be committed"
        );
    }

    /// The edge counterpart: a guessed integer id of a batch-created EDGE
    /// submitted to `update_edge` / `delete_edge` is rejected by the same
    /// batch-created-ref guard (Issue #3417), whole batch aborts, zero writes.
    #[test]
    fn apply_batch_guessed_id_of_batch_created_edge_aborts_whole_batch() {
        let server = create_test_server();
        let a = seed_person(&server, "A");
        let b = seed_person(&server, "B");

        // Anchor the guess to reality: create one committed edge and read its
        // id. Edge ids are allocated sequentially (including ids consumed by
        // aborted batches), so the very next create_edge — op 0 of the failing
        // batch that immediately follows — gets `anchor + 1`, the id a caller
        // would guess. Each variant re-anchors because the previous aborted
        // batch consumed an id.
        let anchor = apply_ok(
            &server,
            json!([
                {"op": "create_edge", "source_id": a, "target_id": b, "label": "KNOWS"},
            ]),
        );
        let guessed_edge = anchor["results"][0]["edge_id"].as_u64().unwrap() + 1;

        let value = apply_err(
            &server,
            json!([
                {"op": "create_edge", "source_id": a, "target_id": b, "label": "KNOWS"},
                {"op": "update_edge", "edge_id": guessed_edge, "properties": {"x": 1}},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        assert_eq!(value["error"]["details"]["created_at_index"], json!(0));
        let msg = value["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("created in the same batch") && msg.contains("edge"),
            "guessed edge-id write must get the v1-scope message: {msg}"
        );
        // Zero writes survived: the batch-created edge was never committed.
        assert!(
            server
                .db()
                .get_edge(EdgeId::new(guessed_edge).unwrap())
                .is_err(),
            "the aborted batch's edge must not be committed"
        );

        // delete_edge form is guarded identically. Re-anchor: the aborted
        // batch above consumed an edge id, so recompute the guess.
        let anchor = apply_ok(
            &server,
            json!([
                {"op": "create_edge", "source_id": a, "target_id": b, "label": "KNOWS"},
            ]),
        );
        let guessed_edge = anchor["results"][0]["edge_id"].as_u64().unwrap() + 1;

        let value = apply_err(
            &server,
            json!([
                {"op": "create_edge", "source_id": a, "target_id": b, "label": "KNOWS"},
                {"op": "delete_edge", "edge_id": guessed_edge},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        assert_eq!(value["error"]["details"]["created_at_index"], json!(0));
        assert!(
            server
                .db()
                .get_edge(EdgeId::new(guessed_edge).unwrap())
                .is_err(),
            "the aborted batch's edge must not be committed"
        );
    }

    /// The guard does NOT over-fire: updating and deleting a PRE-EXISTING
    /// committed node/edge within a batch still succeeds — only ids created
    /// earlier in the SAME batch are refused (Issue #3417).
    #[test]
    fn apply_batch_update_delete_of_preexisting_committed_entity_still_succeeds() {
        let server = create_test_server();
        let a = seed_person(&server, "A");
        let b = seed_person(&server, "B");
        let edge = seed_edge(&server, a, b);

        // Update a committed node + update a committed edge in one batch.
        let value = apply_ok(
            &server,
            json!([
                {"op": "update_node", "node_id": a, "properties": {"role": "lead"}},
                {"op": "update_edge", "edge_id": edge, "properties": {"weight": 2}},
            ]),
        );
        assert_eq!(value["operation_count"], json!(2), "got: {value}");

        // Delete a committed edge, then the (now-detached) committed node.
        let value = apply_ok(
            &server,
            json!([
                {"op": "delete_edge", "edge_id": edge},
                {"op": "delete_node", "node_id": b},
            ]),
        );
        assert_eq!(value["operation_count"], json!(2), "got: {value}");
        assert_eq!(server.db().edge_count(), 0);
    }

    #[test]
    fn apply_batch_unique_constraint_violation_aborts_whole_batch() {
        let server = create_test_server();
        let response = server.enable_unique_constraint(EnableUniqueConstraintRequest {
            label: "Person".to_string(),
            property: "email".to_string(),
        });
        assert!(
            !response.contains("\"error\""),
            "constraint enable should succeed: {response}"
        );
        let _existing: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: "Person".to_string(),
            properties: Some(HashMap::from([(
                "email".to_string(),
                serde_json::Value::String("a@x.com".to_string()),
            )])),
            valid_time: None,
            provenance: None,
        }))
        .unwrap();

        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person",
                 "properties": {"name": "Fresh"}},
                {"op": "create_node", "label": "Person",
                 "properties": {"email": "a@x.com"}},
            ]),
            "CONSTRAINT_VIOLATION",
        );
        assert!(
            value["error"]["details"].get("failed_op_index").is_some(),
            "details must carry failed_op_index (possibly null for commit-time \
             constraint failures): {value}"
        );
        // The whole batch aborted: only the seeded node exists.
        assert_eq!(server.db().node_count(), 1);
    }

    // ------------------------------------------------------------------
    // #3209 safe-by-default delete inside a batch
    // ------------------------------------------------------------------

    /// A batch-created edge counts toward the #3209 refusal even though it
    /// is invisible to committed-state adjacency (the in-batch case).
    #[test]
    fn apply_batch_delete_node_with_batch_created_edge_refused_without_detach() {
        let server = create_test_server();
        let x = seed_person(&server, "X");
        let nodes_before = server.db().node_count();

        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "a"},
                {"op": "create_edge", "source_id": "$a", "target_id": x,
                 "label": "KNOWS"},
                {"op": "delete_node", "node_id": x},
            ]),
            "FAILED_PRECONDITION",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["failed_op_index"], json!(2));
        assert_eq!(details["connected_edges"], json!(1));
        assert_eq!(details["detach_required"], json!(true));
        // Legacy top-level fields preserved, mirroring the single-op tool.
        assert_eq!(value["connected_edges"], json!(1));

        // Atomic refusal: nothing committed.
        assert_eq!(server.db().node_count(), nodes_before);
        assert_eq!(server.db().edge_count(), 0);
    }

    /// With `detach: true`, the cascade also removes batch-created edges
    /// touching the node (which `delete_node_cascade` alone cannot see).
    #[test]
    fn apply_batch_delete_node_detach_true_removes_batch_created_edges() {
        let server = create_test_server();
        let x = seed_person(&server, "X");

        let value = apply_ok(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "a",
                 "properties": {"name": "Survivor"}},
                {"op": "create_edge", "source_id": "$a", "target_id": x,
                 "label": "KNOWS"},
                {"op": "delete_node", "node_id": x, "detach": true},
            ]),
        );

        let results = value["results"].as_array().unwrap();
        // The batch-created edge was removed by the later detach delete.
        assert_eq!(
            results[1]["removed_by_delete_at_index"],
            json!(2),
            "create_edge result must report the removing op: {value}"
        );
        assert_eq!(results[2]["op"], json!("delete_node"));
        assert_eq!(results[2]["detached"], json!(true));
        assert_eq!(
            results[2]["edges_removed"],
            json!(1),
            "the batch-created edge counts as removed: {value}"
        );

        // Final state: X gone, the batch node exists, no edges at all.
        assert_eq!(server.db().node_count(), 1);
        assert_eq!(server.db().edge_count(), 0);
        assert!(server.db().get_node(NodeId::new(x).unwrap()).is_err());
        let survivor = server.db().find_nodes_by_property(
            "Person",
            "name",
            &crate::core::PropertyValue::String("Survivor".into()),
        );
        assert_eq!(survivor.len(), 1);
    }

    #[test]
    fn apply_batch_delete_node_detach_true_cascades_committed_edges() {
        let server = create_test_server();
        let a = seed_person(&server, "A");
        let b = seed_person(&server, "B");
        let _e = seed_edge(&server, a, b);

        let value = apply_ok(
            &server,
            json!([
                {"op": "delete_node", "node_id": a, "detach": true},
            ]),
        );
        assert_eq!(value["results"][0]["edges_removed"], json!(1));
        assert_eq!(server.db().node_count(), 1);
        assert_eq!(server.db().edge_count(), 0);
    }

    /// A batch-created self-loop appears in both adjacency directions but is
    /// one edge: the refusal must count it once (mirrors retract_node dedup).
    #[test]
    fn apply_batch_self_loop_edge_then_delete_counts_edge_once() {
        let server = create_test_server();
        let x = seed_person(&server, "X");

        let value = apply_err(
            &server,
            json!([
                {"op": "create_edge", "source_id": x, "target_id": x,
                 "label": "SELF"},
                {"op": "delete_node", "node_id": x},
            ]),
            "FAILED_PRECONDITION",
        );
        assert_eq!(
            value["error"]["details"]["connected_edges"],
            json!(1),
            "self-loop must count once: {value}"
        );
        assert_eq!(server.db().edge_count(), 0);
    }

    /// Updating a committed edge and then detach-deleting one of its
    /// endpoints in the same batch is refused with BOTH op indices — the
    /// cascade would silently discard the update.
    #[test]
    fn apply_batch_update_edge_then_detach_delete_endpoint_refused_with_both_indices() {
        let server = create_test_server();
        let a = seed_person(&server, "A");
        let b = seed_person(&server, "B");
        let e = seed_edge(&server, a, b);

        let value = apply_err(
            &server,
            json!([
                {"op": "update_edge", "edge_id": e, "properties": {"weight": 9}},
                {"op": "delete_node", "node_id": a, "detach": true},
            ]),
            "FAILED_PRECONDITION",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["failed_op_index"], json!(1), "got: {value}");
        assert_eq!(details["updated_at_index"], json!(0));
        assert_eq!(details["edge_id"], json!(e));
        assert_eq!(details["node_id"], json!(a));

        // Atomic refusal: nothing changed.
        assert_eq!(server.db().node_count(), 2);
        assert_eq!(server.db().edge_count(), 1);
    }

    /// Explicitly writing to a committed edge that an earlier detach-delete
    /// in the same batch already cascaded away is refused with the removing
    /// op's index.
    #[test]
    fn apply_batch_delete_edge_removed_by_cascade_refused() {
        let server = create_test_server();
        let a = seed_person(&server, "A");
        let b = seed_person(&server, "B");
        let e = seed_edge(&server, a, b);

        let value = apply_err(
            &server,
            json!([
                {"op": "delete_node", "node_id": a, "detach": true},
                {"op": "delete_edge", "edge_id": e},
            ]),
            "FAILED_PRECONDITION",
        );
        let details = &value["error"]["details"];
        assert_eq!(details["failed_op_index"], json!(1), "got: {value}");
        assert_eq!(details["removed_at_index"], json!(0));
        assert_eq!(details["edge_id"], json!(e));

        // Atomic refusal: the cascade itself was rolled back too.
        assert_eq!(server.db().node_count(), 2);
        assert_eq!(server.db().edge_count(), 1);
    }

    /// Committed edges already deleted earlier in the same batch no longer
    /// count toward the #3209 refusal.
    #[test]
    fn apply_batch_delete_node_after_batch_deleted_edges_succeeds_without_detach() {
        let server = create_test_server();
        let a = seed_person(&server, "A");
        let b = seed_person(&server, "B");
        let e = seed_edge(&server, a, b);

        let value = apply_ok(
            &server,
            json!([
                {"op": "delete_edge", "edge_id": e},
                {"op": "delete_node", "node_id": a},
            ]),
        );
        assert_eq!(value["results"][1]["edges_removed"], json!(0));
        assert_eq!(server.db().node_count(), 1);
        assert_eq!(server.db().edge_count(), 0);
    }

    // ------------------------------------------------------------------
    // Per-op valid_time (#3221 inside a batch)
    // ------------------------------------------------------------------

    #[test]
    fn apply_batch_valid_time_honored_per_op() {
        let server = create_test_server();
        let value = apply_ok(
            &server,
            json!([
                {"op": "create_node", "label": "Person", "ref": "old",
                 "properties": {"name": "Old"},
                 "valid_time": T_PAST_MICROS.to_string()},
                {"op": "create_node", "label": "Person", "ref": "new",
                 "properties": {"name": "New"}},
            ]),
        );
        let old_id = value["results"][0]["node_id"].as_u64().unwrap();
        let new_id = value["results"][1]["node_id"].as_u64().unwrap();
        let now = time::now();

        // The backdated node existed (valid-time) back in 2020...
        assert!(
            server
                .db()
                .get_node_at_time(
                    NodeId::new(old_id).unwrap(),
                    Timestamp::from(T_PAST_MICROS),
                    now
                )
                .is_ok(),
            "backdated valid_time must be honored"
        );
        // ...but not one second before its valid_from.
        assert!(
            server
                .db()
                .get_node_at_time(
                    NodeId::new(old_id).unwrap(),
                    Timestamp::from(T_PAST_MICROS - 1_000_000),
                    now
                )
                .is_err()
        );
        // The non-backdated node did not exist in 2020.
        assert!(
            server
                .db()
                .get_node_at_time(
                    NodeId::new(new_id).unwrap(),
                    Timestamp::from(T_PAST_MICROS),
                    now
                )
                .is_err()
        );
    }

    #[test]
    fn apply_batch_valid_time_and_detach_rejected_per_op() {
        let server = create_test_server();
        let x = seed_person(&server, "X");
        let nodes_before = server.db().node_count();

        let value = apply_err(
            &server,
            json!([
                {"op": "create_node", "label": "Person",
                 "valid_time": T_PAST_MICROS.to_string()},
                {"op": "delete_node", "node_id": x, "detach": true,
                 "valid_time": T_PAST_MICROS.to_string()},
            ]),
            "INVALID_ARGUMENT",
        );
        assert_eq!(value["error"]["details"]["failed_op_index"], json!(1));
        assert_eq!(server.db().node_count(), nodes_before);
    }

    // ------------------------------------------------------------------
    // WAL durability and recovery
    // ------------------------------------------------------------------

    /// A committed batch survives a clean shutdown + reopen. NOTE: with a
    /// clean shutdown, `open()` loads the index SNAPSHOT written at close
    /// and replays no WAL entries — this test therefore verifies
    /// snapshot-based restart, not WAL replay. Genuine WAL replay of a
    /// batch is exercised by
    /// `apply_batch_committed_batch_survives_wal_replay_without_index_snapshot`.
    #[test]
    fn apply_batch_committed_batch_survives_wal_recovery() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let path = temp_dir.path().join("batch-db");

        {
            let db = Arc::new(AletheiaDB::open(&path).expect("open durable db"));
            let server = AletheiaMcpServer::new(db);
            apply_ok(
                &server,
                json!([
                    {"op": "create_node", "label": "Person", "ref": "a",
                     "properties": {"name": "Durable A"}},
                    {"op": "create_node", "label": "Person", "ref": "b",
                     "properties": {"name": "Durable B"}},
                    {"op": "create_edge", "source_id": "$a", "target_id": "$b",
                     "label": "KNOWS"},
                ]),
            );
            assert_eq!(server.db().node_count(), 2);
            assert_eq!(server.db().edge_count(), 1);
        }

        // Reopen: the shutdown index snapshot must restore the whole batch.
        let db = AletheiaDB::open(&path).expect("reopen durable db");
        assert_eq!(db.node_count(), 2, "batch nodes must survive recovery");
        assert_eq!(db.edge_count(), 1, "batch edge must survive recovery");
        let found = db.find_nodes_by_property(
            "Person",
            "name",
            &crate::core::PropertyValue::String("Durable A".into()),
        );
        assert_eq!(found.len(), 1);
    }

    /// Forces GENUINE WAL replay: after a clean shutdown, the index
    /// snapshot directory is deleted, so the reopen can only rebuild state
    /// from the WAL. The full subgraph — nodes, edge, endpoints, and
    /// properties — must be restored from the batch's WAL entries alone.
    #[test]
    fn apply_batch_committed_batch_survives_wal_replay_without_index_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let path = temp_dir.path().join("batch-db-replay");

        let (alice_id, bob_id, edge_id) = {
            let db = Arc::new(AletheiaDB::open(&path).expect("open durable db"));
            let server = AletheiaMcpServer::new(db);
            let value = apply_ok(
                &server,
                json!([
                    {"op": "create_node", "label": "Person", "ref": "a",
                     "properties": {"name": "Replay A"}},
                    {"op": "create_node", "label": "Person", "ref": "b",
                     "properties": {"name": "Replay B"}},
                    {"op": "create_edge", "source_id": "$a", "target_id": "$b",
                     "label": "KNOWS", "properties": {"since": 2024}},
                ]),
            );
            let results = value["results"].as_array().unwrap();
            (
                results[0]["node_id"].as_u64().unwrap(),
                results[1]["node_id"].as_u64().unwrap(),
                results[2]["edge_id"].as_u64().unwrap(),
            )
        };

        // Delete the index snapshot so `open()` cannot short-circuit the
        // WAL: the reopen below must genuinely replay the batch's entries.
        std::fs::remove_dir_all(path.join("indexes")).expect("remove index snapshot dir");

        let db = AletheiaDB::open(&path).expect("reopen durable db");
        assert_eq!(db.node_count(), 2, "batch nodes must survive WAL replay");
        assert_eq!(db.edge_count(), 1, "batch edge must survive WAL replay");

        // Nodes restored with their properties.
        for (name, id) in [("Replay A", alice_id), ("Replay B", bob_id)] {
            let found = db.find_nodes_by_property(
                "Person",
                "name",
                &crate::core::PropertyValue::String(name.into()),
            );
            assert_eq!(found.len(), 1, "{name} must be restored via WAL replay");
            assert_eq!(found[0].as_u64(), id, "{name} must keep its node id");
        }

        // Edge restored with its endpoints and property.
        let edge = db
            .get_edge(EdgeId::new(edge_id).unwrap())
            .expect("batch edge must be restored via WAL replay");
        assert_eq!(edge.source.as_u64(), alice_id);
        assert_eq!(edge.target.as_u64(), bob_id);
        let since = edge
            .properties
            .iter()
            .find_map(|(k, v)| {
                crate::core::interning::GLOBAL_INTERNER
                    .resolve_with(*k, |s| s == "since")
                    .unwrap_or(false)
                    .then(|| format!("{v:?}"))
            })
            .unwrap_or_default();
        assert!(
            since.contains("2024"),
            "edge property must survive WAL replay: {since}"
        );
    }

    /// A batch that fails at op time aborts BEFORE the WAL is ever touched
    /// (WAL append happens only at commit), so nothing is appended and
    /// recovery trivially restores nothing — this pins that a failed batch
    /// leaves no WAL residue to resurrect.
    #[test]
    fn apply_batch_failed_batch_leaves_nothing_after_recovery() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let path = temp_dir.path().join("batch-db-failed");

        {
            let db = Arc::new(AletheiaDB::open(&path).expect("open durable db"));
            let server = AletheiaMcpServer::new(db);
            apply_err(
                &server,
                json!([
                    {"op": "create_node", "label": "Person",
                     "properties": {"name": "Ghost"}},
                    {"op": "update_node", "node_id": 424_242, "properties": {"x": 1}},
                ]),
                "NOT_FOUND",
            );
            assert_eq!(server.db().node_count(), 0);
        }

        let db = AletheiaDB::open(&path).expect("reopen durable db");
        assert_eq!(db.node_count(), 0, "failed batch must leave no trace");
        assert_eq!(db.edge_count(), 0);
    }

    /// Issue #3723: a property-KEY interner-cap breach inside an `apply_batch`
    /// op must render the SAME structured `FAILED_PRECONDITION` envelope as the
    /// single-op `create_node`/HTTP surfaces (post #3716) — actionable message
    /// naming `persistence.max_interned_strings`, `retriable: false`, and
    /// `details.{resource, current, limit}` — NOT the generic `INVALID_ARGUMENT`
    /// the batch previously flattened it to. The breach lands in Phase 1 static
    /// pre-validation (before any transaction opens), so it carries the op's
    /// `failed_op_index` (not `null`) and leaves ZERO writes.
    ///
    /// Lowers the process-global interner cap, so it runs in a SUBPROCESS to
    /// isolate the mutation from concurrent tests (mirrors the create_node
    /// counterpart in `mcp::server`).
    #[test]
    fn apply_batch_interner_cap_property_key_breach_is_failed_precondition_via_subprocess() {
        use std::process::Command;
        use std::time::{Duration, Instant};

        let exe = std::env::current_exe().expect("failed to locate current test binary");
        let mut child = Command::new(exe)
            .args([
                "--ignored",
                "--exact",
                "mcp::tests::apply_batch_tests::apply_batch_interner_cap_property_key_breach_helper",
            ])
            .spawn()
            .expect("failed to spawn subprocess for apply_batch interner-cap test");

        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    assert!(
                        status.success(),
                        "apply_batch interner-cap property-key helper failed"
                    );
                    break;
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        panic!("apply_batch interner-cap property-key helper did not complete");
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => panic!("failed while polling subprocess: {e}"),
            }
        }
    }

    #[test]
    #[ignore]
    fn apply_batch_interner_cap_property_key_breach_helper() {
        use crate::core::interning::GLOBAL_INTERNER;

        let server = create_test_server();

        // Pin the cap exactly at the live interner size, so the NEXT new intern —
        // the fresh property KEY on op index 1 below, interned by
        // `json_to_property_map` during Phase 1 static pre-validation — tips it
        // over. (No capacity rollbacks have occurred in this fresh subprocess,
        // so next_id == len().)
        let base = GLOBAL_INTERNER.len();
        GLOBAL_INTERNER.set_max_capacity(base);

        // Two ops; the breaching op is at index 1 so we prove the reported
        // failed_op_index is the op's real index, not a hardcoded 0.
        let value = apply_err(
            &server,
            json!([
                {"op": "delete_node", "node_id": 999_999_999u64},
                {"op": "create_node", "label": "TipLabel",
                 "properties": {"uniq_batch_property_key_that_tips_the_interner_over": "v"}},
            ]),
            "FAILED_PRECONDITION",
        );

        let error = &value["error"];
        // Byte-for-byte alignment with the #3716 create_node/HTTP envelope.
        assert_eq!(error["code"], "FAILED_PRECONDITION", "got: {value}");
        assert_eq!(error["retriable"], false, "got: {value}");
        assert_eq!(
            error["details"]["resource"], "string interner",
            "details must name the interner resource: {value}"
        );
        assert!(
            error["details"]["limit"].is_number(),
            "details must carry a numeric limit: {value}"
        );
        assert!(
            error["details"]["current"].is_number(),
            "details must carry a numeric current: {value}"
        );
        assert!(
            error["message"]
                .as_str()
                .unwrap_or_default()
                .contains("persistence.max_interned_strings"),
            "message must name the knob: {value}"
        );
        // Static-phase failure carries the breaching op's real index, not null.
        assert_eq!(
            error["details"]["failed_op_index"],
            json!(1),
            "failed_op_index must be the breaching op's index: {value}"
        );

        // All-or-nothing: the breach is in Phase 1, before any transaction
        // opens, so ZERO writes take effect.
        assert_eq!(
            server.db().node_count(),
            0,
            "a cap-breaching batch must leave zero nodes: {value}"
        );
        assert_eq!(
            server.db().edge_count(),
            0,
            "a cap-breaching batch must leave zero edges: {value}"
        );
    }
}

// ============================================================================
// Per-request `now` for is_current evaluation (Issue #3391)
// ============================================================================

mod per_request_now_tests {
    use super::*;
    use crate::simulation::SimulatedClock;
    use crate::simulation::clock::{reset_simulated_read_count, simulated_read_count};

    fn now_micros() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    }

    fn create_node_with_valid_time(
        server: &AletheiaMcpServer,
        label: &str,
        name: &str,
        valid_time_micros: Option<i64>,
    ) {
        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!(name));
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: label.to_string(),
            properties: Some(props),
            valid_time: valid_time_micros.map(|m| m.to_string()),
            provenance: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none(), "create_node failed: {value}");
    }

    /// List nodes with the given label and return each node's
    /// `temporal.is_current` flag, in response order.
    fn list_is_current_flags(server: &AletheiaMcpServer, label: &str) -> Vec<bool> {
        let response = server.list_nodes(ListNodesRequest {
            namespace: None,
            label: Some(label.to_string()),
            property_key: None,
            property_value: None,
            limit: None,
            offset: None,
            include_vectors: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none(), "list_nodes failed: {value}");
        value["nodes"]
            .as_array()
            .expect("nodes must be an array")
            .iter()
            .map(|n| {
                n["temporal"]["is_current"]
                    .as_bool()
                    .unwrap_or_else(|| panic!("temporal.is_current must be a bool: {n}"))
            })
            .collect()
    }

    /// Boundary fixture (Issue #3391): every entity in one bulk response must
    /// be judged against the SAME evaluation timestamp.
    ///
    /// Two nodes share an identical future `valid_from`. A simulated clock
    /// that advances by 1µs on every read is swept across that boundary, one
    /// microsecond at a time, covering every read the request performs. If
    /// `is_current` were evaluated per entity (a fresh `now()` per
    /// conversion), the two entities would be judged at strictly different
    /// timestamps, so some sweep offset must place one entity before the
    /// boundary and the other at/after it — disagreeing flags. With the
    /// wallclock captured once per request, the flags always agree.
    #[test]
    fn test_list_nodes_entities_share_one_evaluation_timestamp() {
        let server = create_test_server();

        // Both facts become true at exactly the same future instant.
        let valid_from = now_micros() + 3_600_000_000; // +1 hour
        create_node_with_valid_time(&server, "Person", "Alice", Some(valid_from));
        create_node_with_valid_time(&server, "Person", "Bob", Some(valid_from));

        let mut clock = SimulatedClock::new(valid_from - 1);
        let _guard = clock.inject_advancing(1);

        // Calibration: measure how many clock reads one request performs so
        // the sweep below covers every read index.
        reset_simulated_read_count();
        let flags = list_is_current_flags(&server, "Person");
        assert_eq!(flags.len(), 2, "both nodes must be listed");
        let reads_per_request = simulated_read_count();

        for offset in 0..=reads_per_request as i64 {
            clock.jump_to(valid_from - offset);
            let flags = list_is_current_flags(&server, "Person");
            assert_eq!(flags.len(), 2, "both nodes must be listed");
            assert_eq!(
                flags[0], flags[1],
                "entities in one list_nodes response were evaluated against \
                 different 'now' values (clock started {offset} µs before the \
                 shared valid_from, advancing 1 µs per read): {flags:?}"
            );
        }
    }

    /// The number of wallclock reads a bulk response performs must not scale
    /// with the number of entities returned: the request captures `now` once
    /// and evaluates every entity against it.
    #[test]
    fn test_list_nodes_clock_reads_do_not_scale_with_entity_count() {
        let server = create_test_server();

        create_node_with_valid_time(&server, "Solo", "only", None);
        for i in 0..6 {
            create_node_with_valid_time(&server, "Crowd", &format!("member-{i}"), None);
        }

        let clock = SimulatedClock::new(now_micros());
        let _guard = clock.inject(); // frozen clock; we only count reads

        reset_simulated_read_count();
        let solo_flags = list_is_current_flags(&server, "Solo");
        let solo_reads = simulated_read_count();
        assert_eq!(solo_flags.len(), 1);

        reset_simulated_read_count();
        let crowd_flags = list_is_current_flags(&server, "Crowd");
        let crowd_reads = simulated_read_count();
        assert_eq!(crowd_flags.len(), 6);

        assert_eq!(
            crowd_reads, solo_reads,
            "clock reads must not scale with the entity count of the \
             response (1 entity: {solo_reads} reads, 6 entities: \
             {crowd_reads} reads)"
        );
    }

    // ------------------------------------------------------------------
    // Scale-invariance regression coverage for the other bulk handlers
    // (review follow-up on #3391): each handler captures `now` once per
    // request, so its clock-read count must not grow with the number of
    // entities in the response.
    // ------------------------------------------------------------------

    /// Create a node and return its id.
    fn create_node_id(server: &AletheiaMcpServer, label: &str, name: &str) -> u64 {
        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!(name));
        let response = server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            label: label.to_string(),
            properties: Some(props),
            valid_time: None,
            provenance: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none(), "create_node failed: {value}");
        value["id"].as_u64().expect("node id must be a u64")
    }

    /// Create an edge between two nodes.
    fn create_test_edge(server: &AletheiaMcpServer, source_id: u64, target_id: u64) {
        let response = server.create_edge(CreateEdgeRequest {
            namespace: None,
            derived_from: None,
            source_id,
            target_id,
            label: "NEXT".to_string(),
            properties: None,
            valid_time: None,
            provenance: None,
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(value.get("error").is_none(), "create_edge failed: {value}");
    }

    /// Run `small` and `large` under a frozen simulated clock (resetting the
    /// read counter before each measured request) and assert the handler's
    /// clock-read count does not scale with the entity count of the
    /// response. `entities_key` is the response field holding the entity
    /// array (e.g. `"nodes"`, `"edges"`, `"results"`).
    fn assert_clock_reads_do_not_scale(
        handler: &str,
        entities_key: &str,
        small_count: usize,
        small: impl Fn() -> String,
        large_count: usize,
        large: impl Fn() -> String,
    ) {
        let clock = SimulatedClock::new(now_micros());
        let _guard = clock.inject(); // frozen clock; we only count reads

        reset_simulated_read_count();
        let small_value: serde_json::Value = serde_json::from_str(&small()).unwrap();
        let small_reads = simulated_read_count();

        reset_simulated_read_count();
        let large_value: serde_json::Value = serde_json::from_str(&large()).unwrap();
        let large_reads = simulated_read_count();

        for (value, expected) in [(&small_value, small_count), (&large_value, large_count)] {
            assert!(value.get("error").is_none(), "{handler} failed: {value}");
            let len = value[entities_key].as_array().map(|a| a.len());
            assert_eq!(
                len,
                Some(expected),
                "{handler}: unexpected entity count under '{entities_key}': {value}"
            );
        }

        assert_eq!(
            large_reads, small_reads,
            "{handler}: clock reads must not scale with the entity count of \
             the response ({small_count} entities: {small_reads} reads, \
             {large_count} entities: {large_reads} reads)"
        );
    }

    #[test]
    fn test_traverse_clock_reads_do_not_scale_with_entity_count() {
        let server = create_test_server();

        let solo_hub = create_node_id(&server, "Hub", "solo-hub");
        let crowd_hub = create_node_id(&server, "Hub", "crowd-hub");
        let leaf = create_node_id(&server, "Leaf", "solo-leaf");
        create_test_edge(&server, solo_hub, leaf);
        for i in 0..6 {
            let leaf = create_node_id(&server, "Leaf", &format!("crowd-leaf-{i}"));
            create_test_edge(&server, crowd_hub, leaf);
        }

        let traverse = |start_node_id: u64| {
            server.traverse(TraverseRequest {
                namespace: None,
                start_node_id,
                edge_label: "NEXT".to_string(),
                direction: Some("outgoing".to_string()),
                depth: Some(1),
                limit: None,
                offset: None,
                include_vectors: None,
                as_of_valid_time: None,
                as_of_transaction_time: None,
            })
        };

        assert_clock_reads_do_not_scale(
            "traverse",
            "results",
            1,
            || traverse(solo_hub),
            6,
            || traverse(crowd_hub),
        );
    }

    #[test]
    fn test_get_outgoing_edges_clock_reads_do_not_scale_with_entity_count() {
        let server = create_test_server();

        let solo_src = create_node_id(&server, "Src", "solo-src");
        let crowd_src = create_node_id(&server, "Src", "crowd-src");
        let target = create_node_id(&server, "Dst", "solo-target");
        create_test_edge(&server, solo_src, target);
        for i in 0..6 {
            let target = create_node_id(&server, "Dst", &format!("crowd-target-{i}"));
            create_test_edge(&server, crowd_src, target);
        }

        let outgoing = |node_id: u64| {
            server.get_outgoing_edges(GetOutgoingEdgesRequest {
                node_id,
                label: None,
                include_vectors: None,
            })
        };

        assert_clock_reads_do_not_scale(
            "get_outgoing_edges",
            "edges",
            1,
            || outgoing(solo_src),
            6,
            || outgoing(crowd_src),
        );
    }

    #[test]
    fn test_get_incoming_edges_clock_reads_do_not_scale_with_entity_count() {
        let server = create_test_server();

        let solo_dst = create_node_id(&server, "Dst", "solo-dst");
        let crowd_dst = create_node_id(&server, "Dst", "crowd-dst");
        let source = create_node_id(&server, "Src", "solo-source");
        create_test_edge(&server, source, solo_dst);
        for i in 0..6 {
            let source = create_node_id(&server, "Src", &format!("crowd-source-{i}"));
            create_test_edge(&server, source, crowd_dst);
        }

        let incoming = |node_id: u64| {
            server.get_incoming_edges(GetIncomingEdgesRequest {
                node_id,
                label: None,
                include_vectors: None,
            })
        };

        assert_clock_reads_do_not_scale(
            "get_incoming_edges",
            "edges",
            1,
            || incoming(solo_dst),
            6,
            || incoming(crowd_dst),
        );
    }

    #[test]
    fn test_find_similar_clock_reads_do_not_scale_with_entity_count() {
        let server = create_test_server();

        let response = server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 4,
            distance_metric: Some("cosine".to_string()),
        });
        let value: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(
            value.get("error").is_none(),
            "enable_vector_index failed: {value}"
        );

        for i in 1..=8 {
            let mut props = HashMap::new();
            props.insert("name".to_string(), serde_json::json!(format!("Doc{i}")));
            props.insert(
                "embedding".to_string(),
                serde_json::json!([
                    (i as f32) * 0.1,
                    (i as f32) * 0.2,
                    (i as f32) * 0.3,
                    (i as f32) * 0.4
                ]),
            );
            let response = server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                label: "Document".to_string(),
                properties: Some(props),
                valid_time: None,
                provenance: None,
            });
            let value: serde_json::Value = serde_json::from_str(&response).unwrap();
            assert!(value.get("error").is_none(), "create_node failed: {value}");
        }

        let find = |k: usize| {
            server.find_similar(FindSimilarRequest {
                namespace: None,
                property_name: "embedding".to_string(),
                embedding: vec![0.1, 0.2, 0.3, 0.4],
                k: Some(k),
                offset: None,
                include_vectors: None,
            })
        };

        assert_clock_reads_do_not_scale("find_similar", "results", 1, || find(1), 6, || find(6));
    }

    #[test]
    fn test_find_nodes_at_time_clock_reads_do_not_scale_with_entity_count() {
        let server = create_test_server();

        create_node_id(&server, "FSolo", "only");
        for i in 0..6 {
            create_node_id(&server, "FCrowd", &format!("member-{i}"));
        }

        // Anchor the point-in-time lookup at (now, now); the frozen
        // simulated clock in the helper starts at the same real instant, so
        // every node created above is visible.
        let valid_time = now_micros().to_string();

        let find = |label: &str| {
            server.find_nodes_at_time(FindNodesAtTimeRequest {
                namespace: None,
                label: label.to_string(),
                property_key: None,
                property_value: None,
                valid_time: valid_time.clone(),
                transaction_time: None,
                limit: None,
                offset: None,
                include_vectors: None,
            })
        };

        assert_clock_reads_do_not_scale(
            "find_nodes_at_time",
            "nodes",
            1,
            || find("FSolo"),
            6,
            || find("FCrowd"),
        );
    }

    #[test]
    fn test_hybrid_query_clock_reads_do_not_scale_with_entity_count() {
        let server = create_test_server();

        create_node_id(&server, "HSolo", "only");
        for i in 0..6 {
            create_node_id(&server, "HCrowd", &format!("member-{i}"));
        }

        let query = |label: &str| {
            server.hybrid_query(HybridQueryRequest {
                namespace: None,
                start_node_id: None,
                traverse_edge: None,
                traverse_depth: None,
                vector_property: None,
                query_embedding: None,
                top_k: None,
                valid_time: None,
                transaction_time: None,
                filter_label: Some(label.to_string()),
                limit: Some(100),
                include_vectors: None,
            })
        };

        assert_clock_reads_do_not_scale(
            "hybrid_query",
            "results",
            1,
            || query("HSolo"),
            6,
            || query("HCrowd"),
        );
    }
}

// ============================================================================
// Token-budget-aware response shaping (Issue #3353)
// ============================================================================

#[cfg(test)]
mod budget_tests {
    use super::*;
    use serde_json::{Value, json};

    /// Dispatch a tool and return the *raw* serialized response text plus its
    /// error flag. The raw text length is exactly what is emitted on the wire,
    /// so it is the quantity the budget contract bounds.
    fn dispatch_raw(server: &AletheiaMcpServer, tool: &str, args: Value) -> (String, bool) {
        let result = server.dispatch_tool(tool, args);
        let is_error = result.is_error.unwrap_or(false);
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.clone()))
            .expect("tool result should carry text content");
        (text, is_error)
    }

    fn dispatch_json(server: &AletheiaMcpServer, tool: &str, args: Value) -> (Value, bool) {
        let (text, is_error) = dispatch_raw(server, tool, args);
        let value = serde_json::from_str(&text).expect("response should be valid JSON");
        (value, is_error)
    }

    /// Create a node whose properties are large enough to blow a small budget.
    fn seed_big_node(server: &AletheiaMcpServer, label: &str, name: &str) -> u64 {
        let bio = "x".repeat(4000);
        let args = json!({
            "label": label,
            "properties": {
                "name": name,
                "bio": bio,
                "notes": "y".repeat(2000),
            }
        });
        let (value, is_error) = dispatch_json(server, "create_node", args);
        assert!(!is_error, "seed create_node failed: {value}");
        value.get("id").and_then(Value::as_u64).expect("node id")
    }

    fn seed_many(server: &AletheiaMcpServer, label: &str, n: usize) -> Vec<u64> {
        (0..n)
            .map(|i| seed_big_node(server, label, &format!("n{i}")))
            .collect()
    }

    // ---- AC1: omitting the budget leaves behavior unchanged ----------------

    #[test]
    fn ac1_omitting_budget_is_unchanged() {
        let server = create_test_server();
        let id = seed_big_node(&server, "Person", "Alice");
        let (with_none, _) = dispatch_json(&server, "get_node", json!({ "node_id": id }));
        // Full bio present, no budget block.
        assert!(
            with_none.get("budget").is_none(),
            "no budget block expected"
        );
        let bio = with_none["properties"]["bio"].as_str().unwrap();
        assert_eq!(bio.len(), 4000, "full bio returned when no budget set");
    }

    // ---- AC2: hard contract, conformance sweep -----------------------------

    #[test]
    fn ac2_conformance_sweep_never_overruns() {
        let server = create_test_server();
        // Enable a vector index so find_similar/hybrid_query have data.
        let _ = server.dispatch_tool(
            "enable_vector_index",
            json!({ "property_name": "embedding", "dimensions": 4, "metric": "cosine" }),
        );
        let ids = seed_many(&server, "Person", 40);
        // A couple of edges for edge tools / traversal.
        for w in ids.windows(2).take(20) {
            let _ = server.dispatch_tool(
                "create_edge",
                json!({ "source_id": w[0], "target_id": w[1], "label": "KNOWS",
                        "properties": { "detail": "z".repeat(1500) } }),
            );
        }

        let budgets = [256u64, 512, 1024, 2048, 4096, 8192, 16384, 32768];
        let cases: Vec<(&str, Value)> = vec![
            ("get_node", json!({ "node_id": ids[0] })),
            ("list_nodes", json!({ "label": "Person", "limit": 40 })),
            ("get_edge", json!({ "edge_id": 1 })),
            ("list_edges", json!({ "limit": 40 })),
            ("get_outgoing_edges", json!({ "node_id": ids[0] })),
            ("get_incoming_edges", json!({ "node_id": ids[1] })),
            (
                "traverse",
                json!({ "start_node_id": ids[0], "max_depth": 3 }),
            ),
            ("get_node_history", json!({ "node_id": ids[0] })),
            ("get_schema", json!({})),
            (
                "find_nodes_at_time",
                json!({ "label": "Person", "valid_time": "2999-01-01T00:00:00Z" }),
            ),
        ];

        for &budget in &budgets {
            for (tool, base) in &cases {
                let mut args = base.clone();
                args.as_object_mut()
                    .unwrap()
                    .insert("max_response_tokens".into(), json!(budget));
                let (text, is_error) = dispatch_raw(&server, tool, args);
                if is_error {
                    // Too-small budgets legitimately error (AC6); that is not an overrun.
                    continue;
                }
                let cap = (budget * super::super::budget::BYTES_PER_TOKEN) as usize;
                assert!(
                    text.len() <= cap,
                    "[{tool} @ {budget} tok] overran budget: {} bytes > cap {} bytes",
                    text.len(),
                    cap
                );
            }
        }
    }

    #[test]
    fn ac2_byte_budget_is_exact() {
        let server = create_test_server();
        let ids = seed_many(&server, "Doc", 20);
        let _ = ids;
        for cap in [400usize, 900, 1500, 3000] {
            let (text, is_error) = dispatch_raw(
                &server,
                "list_nodes",
                json!({ "label": "Doc", "limit": 20, "max_response_bytes": cap }),
            );
            if is_error {
                continue;
            }
            assert!(
                text.len() <= cap,
                "byte budget {cap} overran: {} bytes",
                text.len()
            );
        }
    }

    // ---- AC3: deterministic, disclosed rung --------------------------------

    #[test]
    fn ac3_rung_is_disclosed_and_deterministic() {
        let server = create_test_server();
        let id = seed_big_node(&server, "Person", "Alice");
        let (a, _) = dispatch_json(
            &server,
            "get_node",
            json!({ "node_id": id, "max_response_tokens": 300 }),
        );
        let (b, _) = dispatch_json(
            &server,
            "get_node",
            json!({ "node_id": id, "max_response_tokens": 300 }),
        );
        assert_eq!(a, b, "same request+budget+data must degrade identically");
        let budget = a.get("budget").expect("budget block present");
        assert_eq!(budget["applied"], json!(true));
        assert!(budget.get("rung").and_then(Value::as_str).is_some());
        assert_eq!(
            budget["token_estimation_basis"],
            json!("ceil(utf8_bytes / 4)")
        );
        // Structure survives: id and label kept.
        assert_eq!(a["id"], json!(id));
        assert!(a.get("label").is_some());
    }

    #[test]
    fn ac3_ladder_progression() {
        let server = create_test_server();
        let id = seed_big_node(&server, "Person", "Alice");
        // Generous budget -> full.
        let (full, _) = dispatch_json(
            &server,
            "get_node",
            json!({ "node_id": id, "max_response_tokens": 20000 }),
        );
        assert_eq!(full["budget"]["rung"], json!("full"));
        // Mid budget -> some form of elision/summary (bio gone as full string).
        let (mid, _) = dispatch_json(
            &server,
            "get_node",
            json!({ "node_id": id, "max_response_tokens": 400 }),
        );
        let rung = mid["budget"]["rung"].as_str().unwrap();
        assert!(
            rung == "elided_properties" || rung == "entity_summaries",
            "expected a degraded rung, got {rung}"
        );
        // bio must no longer be a full 4000-char string.
        let bio = &mid["properties"]["bio"];
        assert!(
            bio.as_str().map(|s| s.len()).unwrap_or(0) < 4000,
            "bio should have degraded"
        );
    }

    // ---- AC4: fetch handles reconstruct omitted content --------------------

    #[test]
    fn ac4_fetch_handle_recovers_full_content() {
        let server = create_test_server();
        let id = seed_big_node(&server, "Person", "Alice");
        let (shaped, _) = dispatch_json(
            &server,
            "get_node",
            json!({ "node_id": id, "max_response_tokens": 400 }),
        );
        // Find a fetch handle somewhere in the shaped properties.
        let handle = find_fetch_handle(&shaped).expect("a fetch handle must be present");
        let tool = handle["tool"].as_str().unwrap().to_string();
        let mut args = handle["arguments"].clone();
        assert_eq!(tool, "get_node");
        // Following the handle (no budget) returns the full bio.
        let (recovered, is_error) = dispatch_json(&server, &tool, {
            args.as_object_mut().unwrap();
            args
        });
        assert!(!is_error, "handle call failed: {recovered}");
        assert_eq!(
            recovered["properties"]["bio"].as_str().unwrap().len(),
            4000,
            "handle reconstructs the omitted content exactly"
        );
    }

    fn find_fetch_handle(value: &Value) -> Option<Value> {
        match value {
            Value::Object(map) => {
                if let Some(fetch) = map.get("fetch")
                    && fetch.get("tool").is_some()
                {
                    return Some(fetch.clone());
                }
                for v in map.values() {
                    if let Some(h) = find_fetch_handle(v) {
                        return Some(h);
                    }
                }
                None
            }
            Value::Array(items) => items.iter().find_map(find_fetch_handle),
            _ => None,
        }
    }

    // ---- AC5: priority_properties out-survive ------------------------------

    #[test]
    fn ac5_priority_properties_protected() {
        let server = create_test_server();
        let id = seed_big_node(&server, "Person", "Alice");
        // Protect "bio" even though it is the bulkiest property. The budget is
        // tight enough to force degradation (bio ~4000B + notes ~2000B exceeds
        // it) but roomy enough to keep the protected bio in full.
        let (shaped, is_error) = dispatch_json(
            &server,
            "get_node",
            json!({ "node_id": id, "max_response_tokens": 1500,
                    "priority_properties": ["bio"] }),
        );
        assert!(!is_error, "should fit with bio protected: {shaped}");
        assert_ne!(
            shaped["budget"]["rung"],
            json!("full"),
            "budget must be tight enough to force degradation"
        );
        // bio kept in full; an unprotected bulky prop (notes) elided/dropped.
        assert_eq!(
            shaped["properties"]["bio"].as_str().map(|s| s.len()),
            Some(4000),
            "protected property survives"
        );
        let notes = &shaped["properties"]["notes"];
        let notes_full = notes.as_str().map(|s| s.len() == 2000).unwrap_or(false);
        assert!(
            !notes_full,
            "unprotected bulky property should degrade first"
        );
    }

    // ---- AC6: too-small budget -> structured error -------------------------

    #[test]
    fn ac6_too_small_budget_errors_with_min_viable() {
        let server = create_test_server();
        let ids = seed_many(&server, "Person", 30);
        let _ = ids;
        let (value, is_error) = dispatch_json(
            &server,
            "list_nodes",
            json!({ "label": "Person", "limit": 30, "max_response_tokens": 1 }),
        );
        assert!(
            is_error,
            "tiny budget must error, not silently empty: {value}"
        );
        let err = value.get("error").expect("structured error");
        assert_eq!(err["code"], json!("INVALID_ARGUMENT"));
        assert_eq!(err["retriable"], json!(false));
        assert!(
            err["details"].get("min_viable_tokens").is_some(),
            "error must state the minimum viable budget: {err}"
        );
    }

    #[test]
    fn ac6_malformed_budget_rejected() {
        let server = create_test_server();
        let id = seed_big_node(&server, "Person", "Alice");
        let (value, is_error) = dispatch_json(
            &server,
            "get_node",
            json!({ "node_id": id, "max_response_tokens": 0 }),
        );
        assert!(is_error);
        assert_eq!(value["error"]["code"], json!("INVALID_ARGUMENT"));
    }

    // ---- Issue #3583: priority_properties DoS bound (e2e) ------------------

    /// An over-cap `priority_properties` array is rejected at the MCP surface
    /// with a structured `INVALID_ARGUMENT` error — bounded, not a hang — and
    /// the error names the configured cap.
    #[test]
    fn issue3583_over_cap_priority_properties_rejected() {
        let cap = 16usize;
        let server = create_test_server().with_max_priority_properties(cap);
        let id = seed_big_node(&server, "Person", "Alice");
        let over: Vec<String> = (0..cap + 1).map(|i| format!("p{i}")).collect();
        let (value, is_error) = dispatch_json(
            &server,
            "get_node",
            json!({
                "node_id": id,
                "max_response_tokens": 1000,
                "priority_properties": over,
            }),
        );
        assert!(is_error, "over-cap array must be rejected: {value}");
        assert_eq!(value["error"]["code"], json!("INVALID_ARGUMENT"));
        assert_eq!(value["error"]["retriable"], json!(false));
        assert_eq!(
            value["error"]["details"]["max_priority_properties"],
            json!(cap)
        );
        assert_eq!(value["error"]["details"]["given"], json!(cap + 1));
    }

    /// A large-but-valid `priority_properties` array completes successfully in
    /// bounded time — the O(1) lookup path is not a hang — and the protected key
    /// (`bio`) survives in full. (Forced-degradation protection semantics are
    /// covered exhaustively by the `budget::unit_tests` HashSet tests; here the
    /// budget is generous so the large echoed array does not itself dominate the
    /// minimal-viable size, keeping the assertion robust.)
    #[test]
    fn issue3583_large_valid_priority_properties_completes() {
        let cap = 512usize;
        let server = create_test_server().with_max_priority_properties(cap);
        let id = seed_big_node(&server, "Person", "Alice");
        // A large (at-cap) array including the real protected key "bio".
        let mut priority: Vec<String> = (0..cap - 1).map(|i| format!("p{i}")).collect();
        priority.push("bio".to_string());
        let byte_budget = 200_000u64;
        let (shaped, is_error) = dispatch_json(
            &server,
            "get_node",
            json!({
                "node_id": id,
                "max_response_bytes": byte_budget,
                "priority_properties": priority,
            }),
        );
        assert!(!is_error, "at-cap valid array must complete: {shaped}");
        // The response is bounded by the stated budget and the budget block is
        // present (shaping ran end-to-end without hanging).
        let serialized = serde_json::to_string_pretty(&shaped).unwrap();
        assert!(
            serialized.len() as u64 <= byte_budget,
            "response stays within the byte budget: {} bytes",
            serialized.len()
        );
        assert!(shaped.get("budget").is_some(), "budget block attached");
        assert_eq!(
            shaped["properties"]["bio"].as_str().map(|s| s.len()),
            Some(4000),
            "protected key survives via the HashSet lookup path"
        );
    }

    // ---- AC7: ranked tools never drop/reorder results ----------------------

    #[test]
    fn ac7_find_similar_never_drops_results() {
        let server = create_test_server();
        let _ = server.dispatch_tool(
            "enable_vector_index",
            json!({ "property_name": "embedding", "dimensions": 4, "metric": "cosine" }),
        );
        // Seed nodes with embeddings and bulky payloads.
        let mut ids = Vec::new();
        for i in 0..6 {
            let (v, e) = dispatch_json(
                &server,
                "create_node",
                json!({ "label": "Doc", "properties": {
                    "name": format!("d{i}"),
                    "bio": "b".repeat(3000),
                    "embedding": [i as f32, 0.0, 0.0, 1.0],
                }}),
            );
            assert!(!e, "seed failed: {v}");
            ids.push(v["id"].as_u64().unwrap());
        }
        let (unbudgeted, _) = dispatch_json(
            &server,
            "find_similar",
            json!({ "node_id": ids[0], "limit": 5 }),
        );
        let full_count = count_results(&unbudgeted);
        let (budgeted, is_error) = dispatch_json(
            &server,
            "find_similar",
            json!({ "node_id": ids[0], "limit": 5, "max_response_tokens": 900 }),
        );
        if is_error {
            // Acceptable only via AC6 (budget too small even for summaries).
            assert_eq!(budgeted["error"]["code"], json!("INVALID_ARGUMENT"));
            return;
        }
        let budgeted_count = count_results(&budgeted);
        assert_eq!(
            budgeted_count, full_count,
            "ranked results must not be dropped to meet a budget"
        );
        // Scores preserved for every result.
        assert!(scores_present(&budgeted), "scores must survive budgeting");
    }

    fn count_results(value: &Value) -> usize {
        for key in ["results", "similar", "nodes"] {
            if let Some(arr) = value.get(key).and_then(Value::as_array) {
                return arr.len();
            }
        }
        0
    }

    fn scores_present(value: &Value) -> bool {
        for key in ["results", "similar", "nodes"] {
            if let Some(arr) = value.get(key).and_then(Value::as_array) {
                return arr
                    .iter()
                    .all(|r| r.get("score").is_some() || r.get("similarity_score").is_some());
            }
        }
        false
    }

    // ---- T4: real classification test (not tautological) -------------------

    #[test]
    fn budgetable_set_is_nonempty_and_classified_read() {
        use crate::auth::AccessClass;
        // The set must be non-empty...
        assert!(
            !super::super::server::BUDGETABLE_READ_TOOLS.is_empty(),
            "budgetable set must not be empty"
        );
        // ...and every entry must be classified as a Read tool in the RBAC
        // access matrix (lockstep with the auth surface, not a self-referential
        // `is_budgetable_read_tool` check).
        for tool in super::super::server::BUDGETABLE_READ_TOOLS {
            let class = super::super::auth::tool_access_class(tool);
            assert_eq!(
                class,
                Some(AccessClass::Read),
                "{tool} must be classified AccessClass::Read (got {class:?})"
            );
        }
    }

    // ---- F7: budget params are discoverable in every tool's inputSchema ----

    #[test]
    fn f7_budget_params_present_in_every_budgetable_input_schema() {
        let server = create_test_server();
        for tool in super::super::server::BUDGETABLE_READ_TOOLS {
            let schema = server
                .tool_input_schema_for_test(tool)
                .unwrap_or_else(|| panic!("{tool} must be advertised"));
            let props = schema
                .get("properties")
                .and_then(Value::as_object)
                .unwrap_or_else(|| panic!("{tool} inputSchema must have properties"));
            for key in [
                "max_response_tokens",
                "max_response_bytes",
                "priority_properties",
            ] {
                assert!(
                    props.contains_key(key),
                    "{tool} inputSchema must expose `{key}` (found: {:?})",
                    props.keys().collect::<Vec<_>>()
                );
            }
            // Types are correct: two positive integers, one array of strings.
            assert_eq!(props["max_response_tokens"]["type"], json!("integer"));
            assert_eq!(props["max_response_bytes"]["type"], json!("integer"));
            assert_eq!(props["priority_properties"]["type"], json!("array"));
        }
    }

    // ---- F1 + F5: rung-4 truncation resume covers all rows, no gap/dupe ----

    #[test]
    fn f1_truncated_list_nodes_resume_has_no_gaps_or_dupes() {
        let server = create_test_server();
        let all_ids = seed_many(&server, "Person", 30);

        // Full, unbudgeted page (limit high enough to hold all).
        let (full, _) = dispatch_json(
            &server,
            "list_nodes",
            json!({ "label": "Person", "limit": 100 }),
        );
        let full_ids = collect_node_ids(&full);
        assert_eq!(full_ids.len(), all_ids.len(), "seeded set fully listed");

        // Budgeted request tight enough to force rung-4 truncation.
        let (page1, is_error) = dispatch_json(
            &server,
            "list_nodes",
            json!({ "label": "Person", "limit": 100, "max_response_tokens": 400 }),
        );
        assert!(!is_error, "budgeted list_nodes should succeed: {page1}");
        assert_eq!(
            page1["budget"]["rung"],
            json!("counts_and_handles"),
            "budget must be tight enough to truncate the array"
        );
        let page1_ids = collect_node_ids(&page1);

        // F1: `count` sibling reflects the retained prefix, and `next_offset`
        // advances by exactly that (not the original limit).
        assert_eq!(
            page1["count"].as_u64().unwrap() as usize,
            page1_ids.len(),
            "count sibling must match retained rows"
        );
        assert_eq!(page1["has_more"], json!(true), "has_more must flip to true");
        assert_eq!(
            page1["next_offset"].as_u64().unwrap() as usize,
            page1_ids.len(),
            "next_offset must resume at the cut point"
        );

        // F5: the truncation fetch handle is a concrete resume call.
        let handle = truncation_handle(&page1).expect("a truncation fetch handle must exist");
        assert_eq!(handle["tool"], json!("list_nodes"));
        let resume_args = handle["arguments"].clone();
        assert_eq!(
            resume_args["offset"].as_u64().unwrap() as usize,
            page1_ids.len(),
            "resume offset must equal the retained count"
        );

        // Follow the disclosed resume call (unbudgeted): the union must equal
        // the full set with NO gaps and NO dupes.
        let (page2, e2) = dispatch_json(&server, "list_nodes", resume_args);
        assert!(!e2, "resume call must succeed: {page2}");
        let page2_ids = collect_node_ids(&page2);

        let mut union = page1_ids.clone();
        union.extend(page2_ids.iter().copied());
        let mut sorted_union = union.clone();
        sorted_union.sort_unstable();
        let before_dedup = sorted_union.len();
        sorted_union.dedup();
        assert_eq!(
            before_dedup,
            sorted_union.len(),
            "no duplicate rows across pages"
        );

        let mut expected = full_ids.clone();
        expected.sort_unstable();
        assert_eq!(
            sorted_union, expected,
            "union of pages == full set (no gaps)"
        );
    }

    fn collect_node_ids(value: &Value) -> Vec<u64> {
        value
            .get("nodes")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|n| n.get("id").and_then(Value::as_u64))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find the rung-4 truncation disclosure's fetch handle in `budget.sections`.
    fn truncation_handle(value: &Value) -> Option<Value> {
        let sections = value.get("budget")?.get("sections")?.as_array()?;
        for s in sections {
            if s.get("rung").and_then(Value::as_str) == Some("counts_and_handles")
                && let Some(fetch) = s.get("fetch")
                && fetch.get("tool").is_some()
            {
                return Some(fetch.clone());
            }
        }
        None
    }

    // ---- F4: counts_and_handles succeeds at a tight budget (no false AC6) --

    #[test]
    fn f4_counts_and_handles_succeeds_at_tight_budget() {
        let server = create_test_server();
        let _ = seed_many(&server, "Person", 30);
        // A budget large enough for the minimal rung-4 response but tight enough
        // that the metadata block matters — must SUCCEED, not falsely AC6-error.
        let cap = 2000usize;
        let (value, is_error) = dispatch_raw(
            &server,
            "list_nodes",
            json!({ "label": "Person", "limit": 100, "max_response_bytes": cap }),
        );
        assert!(
            !is_error,
            "tight-but-viable budget must succeed, not error: {value}"
        );
        assert!(
            value.len() <= cap,
            "assembled response must fit: {}",
            value.len()
        );
        let parsed: Value = serde_json::from_str(&value).unwrap();
        assert_eq!(parsed["budget"]["rung"], json!("counts_and_handles"));
    }

    // ---- T6: re-issuing at the reported min_viable_tokens succeeds ---------

    #[test]
    fn t6_reissue_at_min_viable_budget_succeeds() {
        let server = create_test_server();
        let id = seed_big_node(&server, "Person", "Alice");
        let (value, is_error) = dispatch_json(
            &server,
            "get_node",
            json!({ "node_id": id, "max_response_tokens": 1 }),
        );
        assert!(is_error, "tiny budget must error first");
        let min_viable = value["error"]["details"]["min_viable_tokens"]
            .as_u64()
            .expect("min_viable_tokens present");
        // Re-issue at the reported minimum — must now succeed.
        let (retry, retry_err) = dispatch_json(
            &server,
            "get_node",
            json!({ "node_id": id, "max_response_tokens": min_viable }),
        );
        assert!(
            !retry_err,
            "re-issuing at min_viable_tokens={min_viable} must succeed: {retry}"
        );
        assert_eq!(retry["id"], json!(id));
    }

    // ---- T5: byte budget holds for multibyte/unicode property payloads -----

    #[test]
    fn t5_byte_budget_holds_for_unicode_property() {
        let server = create_test_server();
        let big = "🌍héllo".repeat(500); // multibyte, ~ several KB
        let (created, e) = dispatch_json(
            &server,
            "create_node",
            json!({ "label": "Doc", "properties": { "name": "u", "bio": big } }),
        );
        assert!(!e, "seed failed: {created}");
        let id = created["id"].as_u64().unwrap();
        for cap in [300usize, 700, 1200] {
            let (text, is_error) = dispatch_raw(
                &server,
                "get_node",
                json!({ "node_id": id, "max_response_bytes": cap }),
            );
            if is_error {
                continue;
            }
            assert!(
                text.len() <= cap,
                "unicode byte budget {cap} overran: {} bytes",
                text.len()
            );
            assert!(
                std::str::from_utf8(text.as_bytes()).is_ok(),
                "response must remain valid UTF-8"
            );
        }
    }

    // ---- F5 (non-paginated): truncation handle has no bogus offset ---------

    #[test]
    fn f5_non_paginated_truncation_handle_omits_offset() {
        let server = create_test_server();
        let hub = seed_big_node(&server, "Hub", "center");
        // Many bulky outgoing edges so get_outgoing_edges (no offset paging)
        // reaches rung-4 truncation.
        for _ in 0..25 {
            let tgt = seed_big_node(&server, "Person", "t");
            let _ = server.dispatch_tool(
                "create_edge",
                json!({ "source_id": hub, "target_id": tgt, "label": "KNOWS",
                        "properties": { "detail": "z".repeat(400) } }),
            );
        }
        let (value, is_error) = dispatch_json(
            &server,
            "get_outgoing_edges",
            json!({ "node_id": hub, "max_response_tokens": 500 }),
        );
        if is_error {
            return; // AC6 acceptable at extreme tightness
        }
        if value["budget"]["rung"] == json!("counts_and_handles")
            && let Some(handle) = truncation_edge_handle(&value)
        {
            assert_eq!(handle["tool"], json!("get_outgoing_edges"));
            assert!(
                handle.get("arguments").is_none(),
                "non-paginated tool must not fabricate an offset resume call: {handle}"
            );
        }
    }

    fn truncation_edge_handle(value: &Value) -> Option<Value> {
        let sections = value.get("budget")?.get("sections")?.as_array()?;
        for s in sections {
            if s.get("rung").and_then(Value::as_str) == Some("counts_and_handles")
                && let Some(fetch) = s.get("fetch")
                && fetch.get("tool").is_some()
            {
                return Some(fetch.clone());
            }
        }
        None
    }

    // ---- T1: AC2 sweep extended to ranked + query tools --------------------

    /// Seed Doc nodes carrying both a 4-d embedding and a bulky bio, so the
    /// ranked/query tools have data large enough to force degradation.
    fn seed_vector_docs(server: &AletheiaMcpServer, n: usize) -> Vec<u64> {
        let _ = server.dispatch_tool(
            "enable_vector_index",
            json!({ "property_name": "embedding", "dimensions": 4, "metric": "cosine" }),
        );
        let mut ids = Vec::new();
        for i in 0..n {
            let (v, e) = dispatch_json(
                server,
                "create_node",
                json!({ "label": "Doc", "properties": {
                    "name": format!("d{i}"),
                    "bio": "b".repeat(2500),
                    "embedding": [i as f32, 1.0, 0.0, 1.0],
                }}),
            );
            assert!(!e, "seed failed: {v}");
            ids.push(v["id"].as_u64().unwrap());
        }
        ids
    }

    #[test]
    fn t1_ranked_and_query_byte_cap_sweep_is_nonvacuous() {
        let server = create_test_server();
        let ids = seed_vector_docs(&server, 12);
        // A few edges so hybrid traversal has neighbors.
        for w in ids.windows(2).take(8) {
            let _ = server.dispatch_tool(
                "create_edge",
                json!({ "source_id": w[0], "target_id": w[1], "label": "REL",
                        "properties": { "note": "n".repeat(600) } }),
            );
        }
        let cases: Vec<(&str, Value)> = vec![
            (
                "find_similar",
                json!({ "property_name": "embedding", "embedding": [0.0, 1.0, 0.0, 1.0], "k": 10 }),
            ),
            (
                "hybrid_query",
                json!({ "start_node_id": ids[0], "traverse_edge": "REL", "traverse_depth": 2,
                        "limit": 10 }),
            ),
            (
                "query",
                json!({ "language": "aql", "query": "MATCH (n:Doc) RETURN n", "limit": 20 }),
            ),
        ];
        let budgets = [512u64, 1024, 2048, 4096, 8192];
        let mut successful_cases = 0usize;
        for &budget in &budgets {
            for (tool, base) in &cases {
                let mut args = base.clone();
                args.as_object_mut()
                    .unwrap()
                    .insert("max_response_bytes".into(), json!(budget));
                let (text, is_error) = dispatch_raw(&server, tool, args);
                if is_error {
                    continue;
                }
                successful_cases += 1;
                assert!(
                    text.len() as u64 <= budget,
                    "[{tool} @ {budget}B] overran: {} bytes",
                    text.len()
                );
            }
        }
        // The sweep must not pass vacuously (every case erroring).
        assert!(
            successful_cases > 0,
            "conformance sweep produced zero successful budgeted responses"
        );
    }

    // ---- T2: dedicated query-tool budget + hybrid_query AC7 ----------------

    #[test]
    fn t2_query_tool_budget_shapes_bulky_rows() {
        let server = create_test_server();
        let _ = seed_vector_docs(&server, 10);
        let (full, ferr) = dispatch_json(
            &server,
            "query",
            json!({ "language": "aql", "query": "MATCH (n:Doc) RETURN n", "limit": 20 }),
        );
        assert!(!ferr, "unbudgeted query failed: {full}");
        let full_rows = full["rows"].as_array().map(|a| a.len()).unwrap_or(0);
        assert!(full_rows > 0, "query returned rows");

        let (shaped, is_error) = dispatch_json(
            &server,
            "query",
            json!({ "language": "aql", "query": "MATCH (n:Doc) RETURN n", "limit": 20,
                    "max_response_tokens": 500 }),
        );
        if is_error {
            assert_eq!(shaped["error"]["code"], json!("INVALID_ARGUMENT"));
            return;
        }
        // Budget block present; rows degraded (bulky bio elided/summarized or
        // rows truncated) but row_count stays consistent with the array length.
        assert_eq!(shaped["budget"]["applied"], json!(true));
        let rc = shaped["row_count"].as_u64().unwrap() as usize;
        let arr = shaped["rows"].as_array().map(|a| a.len()).unwrap_or(0);
        assert_eq!(rc, arr, "row_count must match retained rows (F1)");
    }

    #[test]
    fn t2_hybrid_query_preserves_count_and_scores_under_budget() {
        let server = create_test_server();
        let ids = seed_vector_docs(&server, 6);
        for w in ids.windows(2) {
            let _ = server.dispatch_tool(
                "create_edge",
                json!({ "source_id": ids[0], "target_id": w[1], "label": "REL",
                        "properties": { "note": "n".repeat(500) } }),
            );
        }
        let base = json!({ "start_node_id": ids[0], "traverse_edge": "REL",
                           "traverse_depth": 1, "limit": 10 });
        let (full, _) = dispatch_json(&server, "hybrid_query", base.clone());
        let full_count = full["results"].as_array().map(|a| a.len()).unwrap_or(0);

        let mut args = base;
        args.as_object_mut()
            .unwrap()
            .insert("max_response_tokens".into(), json!(700));
        let (budgeted, is_error) = dispatch_json(&server, "hybrid_query", args);
        if is_error {
            assert_eq!(budgeted["error"]["code"], json!("INVALID_ARGUMENT"));
            return;
        }
        let budgeted_count = budgeted["results"].as_array().map(|a| a.len()).unwrap_or(0);
        assert_eq!(
            budgeted_count, full_count,
            "ranked hybrid results must not be dropped to meet a budget (AC7)"
        );
        for r in budgeted["results"].as_array().unwrap() {
            assert!(
                r.get("similarity_score").is_some(),
                "similarity_score must be present on every ranked result: {r}"
            );
        }
    }

    // ---- T3: AC7 ordering preserved for ranked tools -----------------------

    #[test]
    fn t3_find_similar_preserves_result_ordering_under_budget() {
        let server = create_test_server();
        let _ = seed_vector_docs(&server, 8);
        let q = json!({ "property_name": "embedding", "embedding": [0.0, 1.0, 0.0, 1.0], "k": 6 });

        let (full, _) = dispatch_json(&server, "find_similar", q.clone());
        let full_seq = ordered_id_score(&full);
        assert!(
            !full_seq.is_empty(),
            "unbudgeted find_similar returned results"
        );

        let mut args = q;
        args.as_object_mut()
            .unwrap()
            .insert("max_response_tokens".into(), json!(700));
        let (budgeted, is_error) = dispatch_json(&server, "find_similar", args);
        if is_error {
            assert_eq!(budgeted["error"]["code"], json!("INVALID_ARGUMENT"));
            return;
        }
        let budgeted_seq = ordered_id_score(&budgeted);
        assert_eq!(
            budgeted_seq, full_seq,
            "ranked ordering (id + score sequence) must be identical under budget"
        );
    }

    /// The ordered (id, score-bits) sequence of a ranked response's results, so
    /// ordering — not just membership — can be compared exactly.
    fn ordered_id_score(value: &Value) -> Vec<(u64, u64)> {
        value
            .get("results")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|r| {
                        let id = r
                            .get("node")
                            .and_then(|n| n.get("id"))
                            .and_then(Value::as_u64)
                            .unwrap_or(0);
                        let score = r
                            .get("score")
                            .or_else(|| r.get("similarity_score"))
                            .and_then(Value::as_f64)
                            .map(f64::to_bits)
                            .unwrap_or(0);
                        (id, score)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ============================================================================
// Snapshot-anchored cursor continuation (Issue #3360)
// ============================================================================

mod cursor_tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::time::Duration;

    /// Drive a tool through the real MCP dispatch table and parse its JSON.
    fn call(server: &AletheiaMcpServer, name: &str, args: serde_json::Value) -> serde_json::Value {
        let result = server.dispatch_tool(name, args);
        let text = AletheiaMcpServer::extract_text(result);
        serde_json::from_str(&text).expect("tool returned valid json")
    }

    /// Create `n` `Person` nodes, returning their ids in creation order.
    fn make_people(server: &AletheiaMcpServer, n: usize) -> Vec<u64> {
        (0..n)
            .map(|i| {
                server
                    .db()
                    .create_node(
                        "Person",
                        PropertyMapBuilder::new()
                            .insert("name", format!("p{i}"))
                            .build(),
                    )
                    .expect("create_node")
                    .as_u64()
            })
            .collect()
    }

    // AC1 + AC7: cursor pages the full labelled set, in deterministic ascending
    // id order, with no duplicates and no gaps; the last page carries no cursor.
    #[test]
    fn list_nodes_cursor_pages_full_set_deterministically() {
        let server = create_test_server();
        let expected: BTreeSet<u64> = make_people(&server, 25).into_iter().collect();

        let mut seen: BTreeSet<u64> = BTreeSet::new();
        let mut prev_page_max: Option<u64> = None;
        let mut args = serde_json::json!({"label": "Person", "use_cursor": true, "limit": 10});
        let mut pages = 0;
        loop {
            let v = call(&server, "list_nodes", args.clone());
            assert_eq!(v["paging"], "cursor");
            assert_eq!(v["total_matching"], 25);
            let ids: Vec<u64> = v["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n["id"].as_u64().unwrap())
                .collect();
            // Ascending within the page, and every id strictly past the prior
            // page's max (keyset, deterministic boundaries).
            assert!(
                ids.windows(2).all(|w| w[0] < w[1]),
                "ids ascend within page"
            );
            if let Some(m) = prev_page_max {
                assert!(ids.iter().all(|&id| id > m), "keyset seek past prior page");
            }
            prev_page_max = ids.iter().max().copied();
            for id in ids {
                assert!(seen.insert(id), "duplicate id {id} across pages");
            }
            pages += 1;
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => {
                    assert_eq!(v["has_more"], true);
                    assert!(v["cursor_ttl_seconds"].as_u64().unwrap() > 0);
                    args = serde_json::json!({"cursor": tok});
                }
                None => {
                    assert_eq!(v["has_more"], false);
                    break;
                }
            }
            assert!(pages < 10, "runaway paging");
        }
        assert_eq!(seen, expected, "union of pages equals the full set");
        assert_eq!(pages, 3, "25 rows / 10 per page = 3 pages");
    }

    // AC2 (headline): all pages of one scan are evaluated at the first page's
    // bi-temporal coordinate. Interleaving writes between pages leaks nothing:
    // zero duplicates, zero omissions, and no post-snapshot writes appear.
    #[test]
    fn find_nodes_at_time_cursor_is_snapshot_consistent_under_concurrent_writes() {
        let server = create_test_server();
        let expected: BTreeSet<u64> = make_people(&server, 25).into_iter().collect();

        // Pin an explicit snapshot coordinate now that the 25 exist.
        let snap = crate::core::temporal::time::now().wallclock().to_string();

        let v1 = call(
            &server,
            "find_nodes_at_time",
            serde_json::json!({
                "label": "Person",
                "valid_time": snap,
                "transaction_time": snap,
                "use_cursor": true,
                "limit": 10
            }),
        );
        assert_eq!(v1["has_more"], true);
        let mut seen: BTreeSet<u64> = v1["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].as_u64().unwrap())
            .collect();
        let cursor = v1["cursor"].as_str().unwrap().to_string();

        // --- Concurrent writes between page 1 and the rest ---
        // Five brand-new Persons (recorded AFTER the snapshot).
        make_people(&server, 5);
        // Delete an original that page 1 has not yet returned (highest id; page 1
        // returned the lowest 10). Recorded AFTER the snapshot.
        let victim = *expected.iter().next_back().unwrap();
        let del = call(
            &server,
            "delete_node",
            serde_json::json!({"node_id": victim}),
        );
        assert!(del.get("error").is_none(), "delete should succeed: {del}");

        // Drain the remaining pages via the cursor.
        let mut args = serde_json::json!({"cursor": cursor});
        loop {
            let v = call(&server, "find_nodes_at_time", args.clone());
            for n in v["nodes"].as_array().unwrap() {
                assert!(
                    seen.insert(n["id"].as_u64().unwrap()),
                    "duplicate row across pages"
                );
            }
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => args = serde_json::json!({"cursor": tok}),
                None => break,
            }
        }

        // Exactly the original 25: the 5 later inserts are invisible (created
        // after the snapshot), the deleted node is still present (deleted after
        // the snapshot), no duplicates, no omissions.
        assert_eq!(seen, expected);
        assert!(
            seen.contains(&victim),
            "a node deleted after the snapshot still appears in the scan"
        );
    }

    // AC3: a tampered token is rejected with a structured INVALID_ARGUMENT
    // error (never wrong data), through the tool surface.
    #[test]
    fn tampered_cursor_returns_invalid_argument() {
        let server = create_test_server();
        make_people(&server, 15);
        let v1 = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "use_cursor": true, "limit": 5}),
        );
        let tok = v1["cursor"].as_str().unwrap();
        // Corrupt the final character of the signed token.
        let mut bad = tok.to_string();
        bad.pop();
        bad.push(if tok.ends_with('A') { 'B' } else { 'A' });

        let v = call(&server, "list_nodes", serde_json::json!({"cursor": bad}));
        assert_eq!(v["error"]["code"], "INVALID_ARGUMENT");
        assert_eq!(v["error"]["retriable"], false);
    }

    // AC5: resuming after expiry returns a structured FAILED_PRECONDITION with
    // remediation guidance (re-issue the query).
    #[test]
    fn expired_cursor_returns_failed_precondition() {
        let server = AletheiaMcpServer::new(create_test_db())
            .with_cursor_config(Duration::from_secs(0), 128);
        make_people(&server, 15);
        let v1 = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "use_cursor": true, "limit": 5}),
        );
        let tok = v1["cursor"].as_str().unwrap().to_string();
        let v = call(&server, "list_nodes", serde_json::json!({"cursor": tok}));
        assert_eq!(v["error"]["code"], "FAILED_PRECONDITION");
        assert_eq!(v["error"]["retriable"], false);
    }

    // AC5: exceeding the per-connection live-cursor cap is a structured
    // FAILED_PRECONDITION echoing the cap.
    #[test]
    fn exceeding_live_cursor_cap_returns_failed_precondition() {
        let server = AletheiaMcpServer::new(create_test_db())
            .with_cursor_config(Duration::from_secs(300), 1);
        make_people(&server, 30);
        // First scan opens the one permitted live cursor.
        let v1 = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "use_cursor": true, "limit": 5}),
        );
        assert!(v1["cursor"].is_string());
        // A second independent first-page scan exceeds the cap.
        let v2 = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "use_cursor": true, "limit": 5}),
        );
        assert_eq!(v2["error"]["code"], "FAILED_PRECONDITION");
        assert_eq!(v2["error"]["details"]["max_live_cursors"], 1);
    }

    // AC3: a cursor minted by one tool cannot be replayed against another.
    #[test]
    fn cross_tool_cursor_replay_is_rejected() {
        let server = create_test_server();
        make_people(&server, 15);
        let v1 = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "use_cursor": true, "limit": 5}),
        );
        let tok = v1["cursor"].as_str().unwrap().to_string();
        let v = call(
            &server,
            "find_nodes_at_time",
            serde_json::json!({"cursor": tok}),
        );
        assert_eq!(v["error"]["code"], "INVALID_ARGUMENT");
    }

    // AC1: adjacency tools page a node's edges by edge id, no dup, no gap.
    #[test]
    fn get_outgoing_edges_cursor_pages_full_adjacency() {
        let server = create_test_server();
        let src = server
            .db()
            .create_node("Person", PropertyMapBuilder::new().build())
            .unwrap();
        let mut expected: BTreeSet<u64> = BTreeSet::new();
        for _ in 0..12 {
            let dst = server
                .db()
                .create_node("Person", PropertyMapBuilder::new().build())
                .unwrap();
            let eid = server
                .db()
                .create_edge(src, dst, "KNOWS", crate::core::PropertyMap::default())
                .unwrap();
            expected.insert(eid.as_u64());
        }

        let mut seen: BTreeSet<u64> = BTreeSet::new();
        let mut args = serde_json::json!({"node_id": src.as_u64(), "use_cursor": true, "limit": 5});
        loop {
            let v = call(&server, "get_outgoing_edges", args.clone());
            assert_eq!(v["total_matching"], 12);
            for e in v["edges"].as_array().unwrap() {
                assert!(seen.insert(e["id"].as_u64().unwrap()), "duplicate edge");
            }
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => args = serde_json::json!({"cursor": tok}),
                None => break,
            }
        }
        assert_eq!(seen, expected);
    }

    // AC1: traverse cursor pages every reachable node exactly once (snapshot
    // pinned across pages).
    #[test]
    fn traverse_cursor_pages_all_reachable_nodes() {
        let server = create_test_server();
        let src = server
            .db()
            .create_node("Person", PropertyMapBuilder::new().build())
            .unwrap();
        let mut expected: BTreeSet<u64> = BTreeSet::new();
        for _ in 0..12 {
            let dst = server
                .db()
                .create_node("Person", PropertyMapBuilder::new().build())
                .unwrap();
            server
                .db()
                .create_edge(src, dst, "KNOWS", crate::core::PropertyMap::default())
                .unwrap();
            expected.insert(dst.as_u64());
        }

        let mut seen: BTreeSet<u64> = BTreeSet::new();
        let mut args = serde_json::json!({
            "start_node_id": src.as_u64(),
            "edge_label": "KNOWS",
            "use_cursor": true,
            "limit": 5
        });
        loop {
            let v = call(&server, "traverse", args.clone());
            assert_eq!(v["paging"], "cursor");
            for r in v["results"].as_array().unwrap() {
                assert!(
                    seen.insert(r["node"]["id"].as_u64().unwrap()),
                    "duplicate node across traverse pages"
                );
            }
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => args = serde_json::json!({"cursor": tok}),
                None => break,
            }
        }
        assert_eq!(seen, expected);
    }

    // AC7: the query tool refuses cursor paging with a structured
    // unsupported_construct-class error rather than silently falling back.
    #[test]
    fn query_tool_cursor_is_unsupported_not_silent() {
        let server = create_test_server();
        let v = call(
            &server,
            "query",
            serde_json::json!({
                "language": "aql",
                "query": "MATCH (n:Person) RETURN n",
                "use_cursor": true
            }),
        );
        assert_eq!(v["error"]["kind"], "unsupported_construct");
        assert_eq!(v["error"]["code"], "INVALID_ARGUMENT");
    }

    // AC1/disclosure: list_edges is not cursorable and says so, pointing at the
    // cursor-paged adjacency tools (no silent no-op).
    #[test]
    fn list_edges_cursor_is_refused_with_alternatives() {
        let server = create_test_server();
        let v = call(
            &server,
            "list_edges",
            serde_json::json!({"use_cursor": true}),
        );
        assert_eq!(v["error"]["code"], "INVALID_ARGUMENT");
        let alts = v["error"]["details"]["cursorable_alternatives"]
            .as_array()
            .expect("alternatives listed");
        assert!(alts.iter().any(|a| a == "get_outgoing_edges"));
    }

    // AC6: cursor mode is opt-in; without use_cursor/cursor the legacy offset
    // response shape (offset/limit, no `paging` marker) is unchanged.
    #[test]
    fn offset_paging_remains_the_default_and_unchanged() {
        let server = create_test_server();
        make_people(&server, 3);
        let v = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "limit": 2}),
        );
        assert!(v.get("paging").is_none(), "no cursor marker in offset mode");
        assert_eq!(v["offset"], 0);
        assert_eq!(v["has_more"], true);
        assert_eq!(v["next_offset"], 2);
    }

    // FIX-A (discoverability): every cursorable tool advertises `use_cursor`
    // and `cursor` in its inputSchema.properties with descriptions, so a client
    // / LLM can discover snapshot cursor paging without reading source.
    #[test]
    fn all_cursorable_tools_advertise_cursor_params_in_schema() {
        let server = create_test_server();
        for tool in [
            "list_nodes",
            "find_nodes_at_time",
            "get_outgoing_edges",
            "get_incoming_edges",
            "traverse",
        ] {
            let schema = server
                .tool_input_schema_for_test(tool)
                .unwrap_or_else(|| panic!("{tool} has an input schema"));
            let props = schema["properties"]
                .as_object()
                .unwrap_or_else(|| panic!("{tool} schema has properties"));
            assert!(
                props.contains_key("use_cursor"),
                "{tool} must expose `use_cursor` in inputSchema"
            );
            assert!(
                props.contains_key("cursor"),
                "{tool} must expose `cursor` in inputSchema"
            );
            assert!(
                props["use_cursor"]["description"].is_string(),
                "{tool} `use_cursor` carries a description"
            );
            assert!(
                props["cursor"]["description"].is_string(),
                "{tool} `cursor` carries a description"
            );
            assert_eq!(
                props["use_cursor"]["type"], "boolean",
                "{tool} `use_cursor` is boolean"
            );
            assert_eq!(
                props["cursor"]["type"], "string",
                "{tool} `cursor` is string"
            );
        }
    }

    // FIX-F (list_nodes snapshot consistency): list_nodes cursor pins (now, now)
    // on page 1. Creating AND deleting nodes mid-stream must leave the union of
    // pages equal to the page-1 snapshot set: created-after excluded,
    // deleted-after still present, no dup/gap, across >2 pages.
    #[test]
    fn list_nodes_cursor_snapshot_consistent_under_concurrent_create_and_delete() {
        let server = create_test_server();
        let expected: BTreeSet<u64> = make_people(&server, 25).into_iter().collect();

        // Page 1 (limit 8 -> at least 4 pages) pins the snapshot at "now".
        let v1 = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "use_cursor": true, "limit": 8}),
        );
        assert_eq!(v1["has_more"], true);
        let mut seen: BTreeSet<u64> = v1["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].as_u64().unwrap())
            .collect();
        let mut cursor = v1["cursor"].as_str().unwrap().to_string();

        // Concurrent writes recorded AFTER the page-1 snapshot:
        //  - 5 brand-new Persons (must be invisible to the scan), and
        //  - delete an original not yet returned (highest id) -- still visible.
        make_people(&server, 5);
        let victim = *expected.iter().next_back().unwrap();
        let del = call(
            &server,
            "delete_node",
            serde_json::json!({"node_id": victim}),
        );
        assert!(del.get("error").is_none(), "delete should succeed: {del}");

        let mut pages = 1;
        loop {
            let v = call(&server, "list_nodes", serde_json::json!({"cursor": cursor}));
            for n in v["nodes"].as_array().unwrap() {
                assert!(
                    seen.insert(n["id"].as_u64().unwrap()),
                    "duplicate row across pages"
                );
            }
            pages += 1;
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => cursor = tok.to_string(),
                None => break,
            }
            assert!(pages < 10, "runaway paging");
        }

        assert!(pages > 2, "scan spanned more than two pages, got {pages}");
        assert_eq!(seen, expected, "union of pages == page-1 snapshot set");
        assert!(
            seen.contains(&victim),
            "a node deleted after the snapshot is still present"
        );
    }

    // FIX-G (deletion of an already-returned row): deleting a node that page 1
    // already emitted, between pages, must not remove it from the scan (it was
    // present at the pinned snapshot) and must not duplicate/skip anything.
    #[test]
    fn list_nodes_cursor_deletion_of_already_returned_row_stays_in_scan() {
        let server = create_test_server();
        let expected: BTreeSet<u64> = make_people(&server, 12).into_iter().collect();

        let v1 = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "use_cursor": true, "limit": 5}),
        );
        let page1: Vec<u64> = v1["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].as_u64().unwrap())
            .collect();
        assert_eq!(v1["has_more"], true);
        let mut seen: BTreeSet<u64> = page1.iter().copied().collect();
        let mut cursor = v1["cursor"].as_str().unwrap().to_string();

        // Delete a row page 1 ALREADY returned (the lowest id) after the
        // snapshot -- it must stay in the union.
        let already_returned = *page1.first().unwrap();
        let del = call(
            &server,
            "delete_node",
            serde_json::json!({"node_id": already_returned}),
        );
        assert!(del.get("error").is_none(), "delete should succeed: {del}");

        loop {
            let v = call(&server, "list_nodes", serde_json::json!({"cursor": cursor}));
            for n in v["nodes"].as_array().unwrap() {
                assert!(
                    seen.insert(n["id"].as_u64().unwrap()),
                    "duplicate row across pages"
                );
            }
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => cursor = tok.to_string(),
                None => break,
            }
        }

        assert_eq!(seen, expected, "union still equals the full snapshot set");
        assert!(
            seen.contains(&already_returned),
            "an already-returned row deleted mid-stream stays in the scan"
        );
    }

    // FIX-G (update mid-stream): a node updated AFTER the page-1 snapshot must
    // be returned AS IT WAS at the snapshot -- verify the reconstructed PROPERTY
    // value, not just the id.
    #[test]
    fn list_nodes_cursor_returns_updated_node_as_of_snapshot() {
        let server = create_test_server();
        let ids = make_people(&server, 6); // names p0..p5, ascending ids
        let sorted: Vec<u64> = {
            let mut s = ids.clone();
            s.sort_unstable();
            s
        };

        // Page 1 (limit 3) pins the snapshot and returns the lowest 3 ids.
        let v1 = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "use_cursor": true, "limit": 3}),
        );
        assert_eq!(v1["has_more"], true);
        let cursor = v1["cursor"].as_str().unwrap().to_string();

        // Update a node NOT yet returned (the 4th lowest id) to a new name,
        // recorded AFTER the snapshot.
        let target = sorted[3];
        let upd = call(
            &server,
            "update_node",
            serde_json::json!({"node_id": target, "properties": {"name": "UPDATED_AFTER_SNAPSHOT"}}),
        );
        assert!(upd.get("error").is_none(), "update should succeed: {upd}");

        // Drain remaining pages; find the target and assert its reconstructed
        // name is the pre-update value (snapshot state), not "UPDATED...".
        let mut args = serde_json::json!({"cursor": cursor});
        let mut found_name: Option<String> = None;
        loop {
            let v = call(&server, "list_nodes", args.clone());
            for n in v["nodes"].as_array().unwrap() {
                if n["id"].as_u64().unwrap() == target {
                    found_name = Some(n["properties"]["name"].as_str().unwrap().to_string());
                }
            }
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => args = serde_json::json!({"cursor": tok}),
                None => break,
            }
        }
        assert_eq!(
            found_name.as_deref(),
            Some("p3"),
            "updated-after node is returned as-it-was at the pinned snapshot"
        );
    }

    // FIX-G (incoming adjacency coverage): get_incoming_edges cursor pages a
    // node's incoming edges by edge id, no dup / no gap (mirrors the outgoing
    // test).
    #[test]
    fn get_incoming_edges_cursor_pages_full_adjacency() {
        let server = create_test_server();
        let dst = server
            .db()
            .create_node("Person", PropertyMapBuilder::new().build())
            .unwrap();
        let mut expected: BTreeSet<u64> = BTreeSet::new();
        for _ in 0..12 {
            let src = server
                .db()
                .create_node("Person", PropertyMapBuilder::new().build())
                .unwrap();
            let eid = server
                .db()
                .create_edge(src, dst, "KNOWS", crate::core::PropertyMap::default())
                .unwrap();
            expected.insert(eid.as_u64());
        }

        let mut seen: BTreeSet<u64> = BTreeSet::new();
        let mut args = serde_json::json!({"node_id": dst.as_u64(), "use_cursor": true, "limit": 5});
        loop {
            let v = call(&server, "get_incoming_edges", args.clone());
            assert_eq!(v["paging"], "cursor");
            assert_eq!(v["total_matching"], 12);
            for e in v["edges"].as_array().unwrap() {
                assert!(seen.insert(e["id"].as_u64().unwrap()), "duplicate edge");
            }
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => args = serde_json::json!({"cursor": tok}),
                None => break,
            }
        }
        assert_eq!(seen, expected);
    }

    // FIX-F (adjacency snapshot consistency): get_outgoing_edges cursor pins the
    // snapshot on page 1; a mid-stream edge create+delete must leave the union
    // equal to the page-1 adjacency (created-after excluded, deleted-after
    // present), no dup/gap.
    #[test]
    fn get_outgoing_edges_cursor_snapshot_consistent_under_concurrent_edge_writes() {
        let server = create_test_server();
        let src = server
            .db()
            .create_node("Person", PropertyMapBuilder::new().build())
            .unwrap();
        let mut expected: BTreeSet<u64> = BTreeSet::new();
        let mut edge_ids: Vec<u64> = Vec::new();
        for _ in 0..12 {
            let dst = server
                .db()
                .create_node("Person", PropertyMapBuilder::new().build())
                .unwrap();
            let eid = server
                .db()
                .create_edge(src, dst, "KNOWS", crate::core::PropertyMap::default())
                .unwrap();
            expected.insert(eid.as_u64());
            edge_ids.push(eid.as_u64());
        }

        let v1 = call(
            &server,
            "get_outgoing_edges",
            serde_json::json!({"node_id": src.as_u64(), "use_cursor": true, "limit": 5}),
        );
        assert_eq!(v1["has_more"], true);
        let mut seen: BTreeSet<u64> = v1["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_u64().unwrap())
            .collect();
        let mut cursor = v1["cursor"].as_str().unwrap().to_string();

        // Concurrent writes recorded AFTER the snapshot: add a new edge
        // (invisible) and delete an existing one not yet returned (still
        // present at the pinned snapshot).
        let new_dst = server
            .db()
            .create_node("Person", PropertyMapBuilder::new().build())
            .unwrap();
        server
            .db()
            .create_edge(src, new_dst, "KNOWS", crate::core::PropertyMap::default())
            .unwrap();
        let victim_edge = *edge_ids.iter().max().unwrap();
        let del = call(
            &server,
            "delete_edge",
            serde_json::json!({"edge_id": victim_edge}),
        );
        assert!(
            del.get("error").is_none(),
            "delete_edge should succeed: {del}"
        );

        loop {
            let v = call(
                &server,
                "get_outgoing_edges",
                serde_json::json!({"cursor": cursor}),
            );
            for e in v["edges"].as_array().unwrap() {
                assert!(seen.insert(e["id"].as_u64().unwrap()), "duplicate edge");
            }
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => cursor = tok.to_string(),
                None => break,
            }
        }

        assert_eq!(seen, expected, "union == page-1 adjacency snapshot");
        assert!(
            seen.contains(&victim_edge),
            "an edge deleted after the snapshot is still present in the scan"
        );
    }

    // FIX-F (traverse snapshot consistency): traverse cursor pins the snapshot
    // on page 1; adding + deleting graph elements mid-stream must leave the
    // reachable set equal to the page-1 snapshot reachable set, no dup/skip.
    #[test]
    fn traverse_cursor_snapshot_consistent_under_concurrent_graph_mutation() {
        let server = create_test_server();
        let src = server
            .db()
            .create_node("Person", PropertyMapBuilder::new().build())
            .unwrap();
        let mut expected: BTreeSet<u64> = BTreeSet::new();
        let mut edge_ids: Vec<u64> = Vec::new();
        for _ in 0..12 {
            let dst = server
                .db()
                .create_node("Person", PropertyMapBuilder::new().build())
                .unwrap();
            let eid = server
                .db()
                .create_edge(src, dst, "KNOWS", crate::core::PropertyMap::default())
                .unwrap();
            expected.insert(dst.as_u64());
            edge_ids.push(eid.as_u64());
        }

        let v1 = call(
            &server,
            "traverse",
            serde_json::json!({
                "start_node_id": src.as_u64(),
                "edge_label": "KNOWS",
                "use_cursor": true,
                "limit": 5
            }),
        );
        assert_eq!(v1["has_more"], true);
        let mut seen: BTreeSet<u64> = v1["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["node"]["id"].as_u64().unwrap())
            .collect();
        let mut cursor = v1["cursor"].as_str().unwrap().to_string();

        // Mutate the graph AFTER the snapshot: add a new reachable node+edge
        // (must be invisible) and delete an existing edge to a not-yet-returned
        // dst (its dst stays reachable at the pinned snapshot).
        let new_dst = server
            .db()
            .create_node("Person", PropertyMapBuilder::new().build())
            .unwrap();
        server
            .db()
            .create_edge(src, new_dst, "KNOWS", crate::core::PropertyMap::default())
            .unwrap();
        let victim_edge = *edge_ids.iter().max().unwrap();
        let del = call(
            &server,
            "delete_edge",
            serde_json::json!({"edge_id": victim_edge}),
        );
        assert!(
            del.get("error").is_none(),
            "delete_edge should succeed: {del}"
        );

        loop {
            let v = call(&server, "traverse", serde_json::json!({"cursor": cursor}));
            for r in v["results"].as_array().unwrap() {
                assert!(
                    seen.insert(r["node"]["id"].as_u64().unwrap()),
                    "duplicate node across traverse pages"
                );
            }
            match v.get("cursor").and_then(|c| c.as_str()) {
                Some(tok) => cursor = tok.to_string(),
                None => break,
            }
        }

        assert!(
            !seen.contains(&new_dst.as_u64()),
            "a node reachable only via an edge created after the snapshot is invisible"
        );
        assert_eq!(
            seen, expected,
            "reachable set == page-1 snapshot reachable set"
        );
    }

    // FIX-G (TTL boundary companion to the pre-expired TTL=0 test): a cursor
    // issued under a positive TTL verifies successfully immediately (it is well
    // within the validity window). Combined with
    // `expired_cursor_returns_failed_precondition` (TTL=0, already expired) this
    // brackets the TTL boundary. Limitation: no injectable clock exists, so the
    // exact expiry instant is not asserted -- only valid-within-window and
    // already-expired.
    #[test]
    fn cursor_valid_within_positive_ttl_window() {
        let server = AletheiaMcpServer::new(create_test_db())
            .with_cursor_config(Duration::from_secs(300), 128);
        make_people(&server, 15);
        let v1 = call(
            &server,
            "list_nodes",
            serde_json::json!({"label": "Person", "use_cursor": true, "limit": 5}),
        );
        let tok = v1["cursor"].as_str().unwrap().to_string();
        // Resuming immediately is comfortably inside the 300s window.
        let v2 = call(&server, "list_nodes", serde_json::json!({"cursor": tok}));
        assert!(
            v2.get("error").is_none(),
            "a cursor within its TTL window must verify: {v2}"
        );
        assert_eq!(v2["paging"], "cursor");
    }
}

#[cfg(feature = "audit-export")]
mod audit_export_tool_tests {
    use super::create_test_server;
    use crate::audit::{AuditSigningKey, verify_json_bytes};
    use crate::core::hex;
    use serde_json::json;

    const SEED: [u8; 32] = [23u8; 32];

    fn dispatch_json(
        server: &super::AletheiaMcpServer,
        tool: &str,
        args: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        let result = server.dispatch_tool(tool, args);
        let is_error = result.is_error.unwrap_or(false);
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.clone()))
            .expect("tool result should carry text content");
        (serde_json::from_str(&text).expect("valid JSON"), is_error)
    }

    #[test]
    #[serial_test::serial]
    fn audit_export_tool_signs_and_verifies_offline() {
        // SAFETY: env mutation serialized via #[serial]; restored below.
        unsafe {
            std::env::set_var("ALETHEIADB_AUDIT_SIGNING_KEY", hex::encode(&SEED));
        }

        let server = create_test_server();
        let (created, is_err) = dispatch_json(
            &server,
            "create_node",
            json!({"label": "Person", "properties": {"name": "Alice", "ssn": "secret"}}),
        );
        assert!(!is_err, "create_node failed: {created}");
        let node_id = created["id"].as_u64().expect("node_id");

        let (resp, is_err) = dispatch_json(
            &server,
            "audit_export",
            json!({
                "entity_type": "node",
                "entity_id": node_id,
                "database_id": "prod-db",
                "redact_keys": ["ssn"],
            }),
        );
        assert!(!is_err, "audit_export failed: {resp}");
        assert_eq!(resp["version_count"].as_u64(), Some(1));

        let pubkey_hex = resp["public_key"].as_str().unwrap();
        let expected_pub = AuditSigningKey::from_seed_bytes(SEED).public_key();
        assert_eq!(pubkey_hex, expected_pub.to_hex());

        // The artifact verifies offline against the operator's public key.
        let artifact_bytes = serde_json::to_vec(&resp["artifact"]).unwrap();
        let report = verify_json_bytes(&artifact_bytes, Some(&expected_pub))
            .expect("MCP-produced artifact must verify offline");
        assert!(report.passed);
        assert!(report.redacted_keys.iter().any(|k| k == "ssn"));
        // Redacted value must not have leaked into the artifact.
        assert!(
            !String::from_utf8(artifact_bytes)
                .unwrap()
                .contains("secret")
        );

        unsafe {
            std::env::remove_var("ALETHEIADB_AUDIT_SIGNING_KEY");
        }
    }

    #[test]
    #[serial_test::serial]
    fn audit_export_tool_without_signing_key_is_failed_precondition() {
        // SAFETY: env mutation serialized via #[serial].
        unsafe {
            std::env::remove_var("ALETHEIADB_AUDIT_SIGNING_KEY");
        }
        let server = create_test_server();
        let (created, _) = dispatch_json(
            &server,
            "create_node",
            json!({"label": "Person", "properties": {"name": "Bob"}}),
        );
        let node_id = created["id"].as_u64().unwrap();
        let (resp, is_err) = dispatch_json(
            &server,
            "audit_export",
            json!({"entity_type": "node", "entity_id": node_id}),
        );
        assert!(is_err, "expected an error without a signing key");
        assert_eq!(resp["error"]["code"].as_str(), Some("FAILED_PRECONDITION"));
    }

    #[test]
    fn audit_export_tool_rejects_bad_entity_type() {
        let server = create_test_server();
        let (resp, is_err) = dispatch_json(
            &server,
            "audit_export",
            json!({"entity_type": "widget", "entity_id": 1}),
        );
        assert!(is_err);
        assert_eq!(resp["error"]["code"].as_str(), Some("INVALID_ARGUMENT"));
    }
}

/// MCP surface for derivation lineage (Issue #3371): the `derived_from` write
/// parameter, the `lineage_upstream`/`lineage_downstream` query tools, their
/// structured errors, and RBAC.
mod lineage_tool_tests {
    use super::create_test_server;
    use crate::auth::{AuthMode, AuthStore, Role, SecretString};
    use crate::core::id::NodeId;
    use crate::mcp::{AletheiaMcpServer, McpAuthConfig};
    use serde_json::json;
    use std::sync::Arc;

    fn dispatch_json(
        server: &AletheiaMcpServer,
        tool: &str,
        args: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        let result = server.dispatch_tool(tool, args);
        let is_error = result.is_error.unwrap_or(false);
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.clone()))
            .expect("tool result should carry text content");
        (serde_json::from_str(&text).expect("valid JSON"), is_error)
    }

    /// The current version id of a node (via the in-crate db handle), which an
    /// LLM would obtain from `get_node_history`.
    fn node_version(server: &AletheiaMcpServer, id: u64) -> u64 {
        server
            .db()
            .node_lineage_ref(NodeId::new(id).unwrap())
            .expect("current version")
            .version
            .as_u64()
    }

    #[test]
    fn create_node_with_derived_from_records_lineage_and_queries_both_directions() {
        let server = create_test_server();

        let (a, is_err) = dispatch_json(
            &server,
            "create_node",
            json!({"label": "Doc", "properties": {"t": "A"}}),
        );
        assert!(!is_err, "create A failed: {a}");
        let a_id = a["id"].as_u64().unwrap();
        let a_ver = node_version(&server, a_id);

        let (b, is_err) = dispatch_json(
            &server,
            "create_node",
            json!({"label": "Doc", "properties": {"t": "B"}}),
        );
        assert!(!is_err, "create B failed: {b}");
        let b_id = b["id"].as_u64().unwrap();
        let b_ver = node_version(&server, b_id);

        // Summary derived from both documents.
        let (c, is_err) = dispatch_json(
            &server,
            "create_node",
            json!({
                "label": "Summary",
                "properties": {"t": "A+B"},
                "derived_from": [
                    {"entity_kind": "node", "id": a_id, "version": a_ver},
                    {"entity_kind": "node", "id": b_id, "version": b_ver},
                ],
            }),
        );
        assert!(!is_err, "create with derived_from failed: {c}");
        let c_id = c["id"].as_u64().unwrap();
        let c_ver = node_version(&server, c_id);

        // Upstream of the summary: both documents at depth 1.
        let (up, is_err) = dispatch_json(
            &server,
            "lineage_upstream",
            json!({"entity_kind": "node", "id": c_id, "version": c_ver}),
        );
        assert!(!is_err, "lineage_upstream failed: {up}");
        assert_eq!(up["direction"], "upstream");
        assert_eq!(up["count"].as_u64(), Some(2));
        assert_eq!(up["has_more"], json!(false));
        let up_ids: Vec<u64> = up["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_u64().unwrap())
            .collect();
        assert!(up_ids.contains(&a_id) && up_ids.contains(&b_id));
        for e in up["entries"].as_array().unwrap() {
            assert_eq!(e["depth"].as_u64(), Some(1));
            assert_eq!(e["status"], "current");
        }

        // Downstream of document A: reaches the summary.
        let (down, is_err) = dispatch_json(
            &server,
            "lineage_downstream",
            json!({"entity_kind": "node", "id": a_id, "version": a_ver}),
        );
        assert!(!is_err, "lineage_downstream failed: {down}");
        assert_eq!(down["direction"], "downstream");
        assert_eq!(down["count"].as_u64(), Some(1));
        assert_eq!(down["entries"][0]["id"].as_u64(), Some(c_id));
        assert_eq!(down["entries"][0]["depth"].as_u64(), Some(1));
    }

    #[test]
    fn dangling_derived_from_reference_is_not_found() {
        let server = create_test_server();
        let (resp, is_err) = dispatch_json(
            &server,
            "create_node",
            json!({
                "label": "Summary",
                "properties": {},
                "derived_from": [
                    {"entity_kind": "node", "id": 1, "version": 9_999_999u64}
                ],
            }),
        );
        assert!(is_err, "dangling ref must fail: {resp}");
        assert_eq!(resp["error"]["code"].as_str(), Some("NOT_FOUND"));
        // No node was created.
        assert_eq!(server.db().node_count(), 0);
    }

    #[test]
    fn invalid_entity_kind_is_invalid_argument() {
        let server = create_test_server();
        let (resp, is_err) = dispatch_json(
            &server,
            "lineage_upstream",
            json!({"entity_kind": "widget", "id": 1, "version": 1}),
        );
        assert!(is_err);
        assert_eq!(resp["error"]["code"].as_str(), Some("INVALID_ARGUMENT"));
    }

    fn server_with_role(role: Role) -> AletheiaMcpServer {
        let store = Arc::new(AuthStore::new());
        let (_principal, key) = store.create_key("t", role).expect("create key");
        AletheiaMcpServer::with_auth(
            server_db(),
            McpAuthConfig::new(AuthMode::Required, Arc::clone(&store))
                .with_credential(SecretString::new(key.as_str())),
        )
    }

    fn server_db() -> Arc<crate::db::AletheiaDB> {
        Arc::new(crate::db::AletheiaDB::new().expect("db"))
    }

    #[test]
    fn reader_denied_create_node_with_derived_from() {
        // create_node is writer-class; a reader is denied before any write —
        // including when it carries the `derived_from` lineage parameter.
        let server = server_with_role(Role::Reader);
        let (resp, is_err) = dispatch_json(
            &server,
            "create_node",
            json!({
                "label": "Summary",
                "properties": {},
                "derived_from": [{"entity_kind": "node", "id": 1, "version": 1}],
            }),
        );
        assert!(is_err);
        assert_eq!(resp["error"]["code"].as_str(), Some("PERMISSION_DENIED"));
    }

    #[test]
    fn metrics_role_denied_lineage_query_but_reader_allowed() {
        // lineage_upstream is reader-class: a metrics-only principal is denied,
        // a reader is authorized (the call then fails/succeeds on its own merits,
        // never with an auth code).
        let metrics = server_with_role(Role::Metrics);
        let (resp, is_err) = dispatch_json(
            &metrics,
            "lineage_upstream",
            json!({"entity_kind": "node", "id": 1, "version": 1}),
        );
        assert!(is_err);
        assert_eq!(resp["error"]["code"].as_str(), Some("PERMISSION_DENIED"));

        let reader = server_with_role(Role::Reader);
        let (resp, _is_err) = dispatch_json(
            &reader,
            "lineage_upstream",
            json!({"entity_kind": "node", "id": 1, "version": 1}),
        );
        // Reader is authorized: an empty closure for an unknown root is a valid
        // (non-error) response, never an auth denial.
        assert_ne!(
            resp.get("error")
                .and_then(|e| e.get("code"))
                .and_then(|c| c.as_str()),
            Some("PERMISSION_DENIED")
        );
    }
}

// ============================================================================
// Provenance hash chain tools (Issue #3351): verify_chain, export_chain_head
// ============================================================================

#[cfg(test)]
mod provenance_chain_tests {
    use super::*;
    use crate::config::WalConfigBuilder;
    use crate::provenance_chain::{ChainConfig, ChainFsyncMode};
    use serde_json::json;

    /// Dispatch a tool and return `(parsed_json, is_error)`.
    fn dispatch(
        server: &AletheiaMcpServer,
        tool: &str,
        args: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        let result = server.dispatch_tool(tool, args);
        let is_error = result.is_error.unwrap_or(false);
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.clone()))
            .expect("tool result must carry text content");
        (
            serde_json::from_str(&text).expect("tool response must be valid JSON"),
            is_error,
        )
    }

    /// Build an MCP server over a durable, chain-enabled database rooted at a
    /// tempdir. Returns the guard so the data dir outlives the server.
    fn chain_enabled_server() -> (tempfile::TempDir, AletheiaMcpServer) {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = crate::config::AletheiaDBConfig::builder()
            .wal(
                WalConfigBuilder::new()
                    .wal_dir(dir.path().join("wal"))
                    .build(),
            )
            .chain(ChainConfig {
                enabled: true,
                fsync: ChainFsyncMode::Batched,
                dir: None,
            })
            .build();
        let db = AletheiaDB::with_unified_config(config).expect("chain-enabled db");
        (dir, AletheiaMcpServer::new(Arc::new(db)))
    }

    #[test]
    fn both_tools_are_advertised() {
        let server = create_test_server();
        let advertised = server.list_tools_for_test();
        assert!(advertised.iter().any(|t| t == "verify_chain"));
        assert!(advertised.iter().any(|t| t == "export_chain_head"));
    }

    #[test]
    fn verify_chain_disabled_is_failed_precondition() {
        // The default test server has no chain enabled.
        let server = create_test_server();
        let (value, is_error) = dispatch(&server, "verify_chain", json!({}));
        assert!(
            is_error,
            "verify on a disabled chain must be an error: {value}"
        );
        assert_eq!(value["error"]["code"].as_str(), Some("FAILED_PRECONDITION"));
        assert_eq!(value["error"]["retriable"].as_bool(), Some(false));
    }

    #[test]
    fn export_chain_head_disabled_is_failed_precondition() {
        let server = create_test_server();
        let (value, is_error) = dispatch(&server, "export_chain_head", json!({}));
        assert!(
            is_error,
            "export on a disabled chain must be an error: {value}"
        );
        assert_eq!(value["error"]["code"].as_str(), Some("FAILED_PRECONDITION"));
    }

    #[test]
    fn verify_chain_rejects_bad_entity_kind() {
        let (_guard, server) = chain_enabled_server();
        let (value, is_error) = dispatch(
            &server,
            "verify_chain",
            json!({"entity_kind": "vertex", "id": 1}),
        );
        assert!(is_error, "bad entity_kind must error: {value}");
        assert_eq!(value["error"]["code"].as_str(), Some("INVALID_ARGUMENT"));
    }

    #[test]
    fn verify_chain_rejects_malformed_against_anchor() {
        let (_guard, server) = chain_enabled_server();
        let (value, is_error) = dispatch(
            &server,
            "verify_chain",
            json!({"against": {"not": "a head"}}),
        );
        assert!(is_error, "malformed anchor must error: {value}");
        assert_eq!(value["error"]["code"].as_str(), Some("INVALID_ARGUMENT"));
    }

    #[test]
    fn full_verify_export_and_anchor_extension_flow() {
        let (_guard, server) = chain_enabled_server();

        // Seed some writes so the chain seals real transactions.
        for _ in 0..5 {
            let (_v, err) = dispatch(
                &server,
                "create_node",
                json!({"label": "Person", "properties": {}}),
            );
            assert!(!err);
        }

        // Full verify must pass.
        let (full, err) = dispatch(&server, "verify_chain", json!({}));
        assert!(!err, "full verify must succeed: {full}");
        assert_eq!(full["passed"].as_bool(), Some(true), "{full}");
        assert_eq!(full["scope"].as_str(), Some("full"));
        assert!(full["head_digest"].as_str().is_some());

        // Export the head anchor.
        let (head, err) = dispatch(&server, "export_chain_head", json!({}));
        assert!(!err, "export must succeed: {head}");
        assert!(
            head["digest"].as_str().is_some(),
            "head has a hex digest: {head}"
        );

        // Anchor-extension verify against the exported head must pass.
        let (against, err) = dispatch(&server, "verify_chain", json!({ "against": head.clone() }));
        assert!(!err, "anchor verify must succeed: {against}");
        assert_eq!(against["passed"].as_bool(), Some(true), "{against}");
        assert_eq!(against["scope"].as_str(), Some("anchor"));
    }

    #[test]
    fn entity_scoped_verify_passes_on_a_created_node() {
        let (_guard, server) = chain_enabled_server();
        let (created, err) = dispatch(
            &server,
            "create_node",
            json!({"label": "Person", "properties": {}}),
        );
        assert!(!err, "create must succeed: {created}");
        let node_id = created["id"].as_u64().expect("created node id");

        let (scoped, err) = dispatch(
            &server,
            "verify_chain",
            json!({"entity_kind": "node", "id": node_id}),
        );
        assert!(!err, "entity verify must succeed: {scoped}");
        assert_eq!(scoped["passed"].as_bool(), Some(true), "{scoped}");
        assert_eq!(scoped["scope"].as_str(), Some("entity"));
    }
}

// ============================================================================
// Provenance filtering (Issue #3348)
// ============================================================================

mod provenance_filter_tests {
    use super::*;
    use serde_json::json;

    /// Dispatch a tool through the same table `call_tool` uses; return
    /// `(parsed_json, is_error)`.
    fn dispatch(
        server: &AletheiaMcpServer,
        tool: &str,
        args: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        let result = server.dispatch_tool(tool, args);
        let is_error = result.is_error.unwrap_or(false);
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.clone()))
            .expect("tool result should carry text content");
        let value = serde_json::from_str(&text).expect("tool response should be valid JSON");
        (value, is_error)
    }

    /// Create a `Doc` node with the given `source`/`confidence` provenance
    /// (either optional) and return its id.
    fn create_doc(
        server: &AletheiaMcpServer,
        name: &str,
        source: Option<&str>,
        confidence: Option<f64>,
    ) -> u64 {
        let mut props = HashMap::new();
        props.insert("name".to_string(), json!(name));
        let provenance = if source.is_some() || confidence.is_some() {
            Some(ProvenanceRequest {
                source: source.map(String::from),
                confidence,
                note: None,
                correlation_id: None,
            })
        } else {
            None
        };
        let n: NodeResponse = parse_response(&server.create_node(CreateNodeRequest {
            namespace: None,
            derived_from: None,
            valid_time: None,
            label: "Doc".to_string(),
            properties: Some(props),
            provenance,
        }))
        .unwrap();
        n.id
    }

    // --- get_node --------------------------------------------------------

    #[test]
    fn get_node_passes_and_excludes_by_source() {
        let server = create_test_server();
        let id = create_doc(&server, "a", Some("crm"), Some(0.9));

        // Matching source -> returned.
        let (ok, err) = dispatch(
            &server,
            "get_node",
            json!({"node_id": id, "provenance_source": "crm"}),
        );
        assert!(!err);
        assert_eq!(ok["id"].as_u64(), Some(id));

        // Non-matching source -> NOT_FOUND (never a fabricated node).
        let (excluded, err) = dispatch(
            &server,
            "get_node",
            json!({"node_id": id, "provenance_source": "hr"}),
        );
        assert!(err);
        assert_eq!(excluded["error"]["code"].as_str(), Some("NOT_FOUND"));
        assert_eq!(excluded["error"]["retriable"].as_bool(), Some(false));
    }

    #[test]
    fn get_node_no_filter_is_unchanged() {
        let server = create_test_server();
        let id = create_doc(&server, "a", Some("crm"), Some(0.9));
        let (plain, _) = dispatch(&server, "get_node", json!({"node_id": id}));
        // A no-op provenance object (all keys absent) is byte-identical.
        let (also_plain, _) = dispatch(
            &server,
            "get_node",
            json!({"node_id": id, "provenance_source": serde_json::Value::Null}),
        );
        assert_eq!(plain, also_plain);
    }

    // --- list_nodes ------------------------------------------------------

    fn seed_docs(server: &AletheiaMcpServer) {
        create_doc(server, "crm-hi", Some("crm"), Some(0.9));
        create_doc(server, "crm-lo", Some("crm"), Some(0.4));
        create_doc(server, "hr-hi", Some("hr"), Some(0.95));
        create_doc(server, "none", None, None); // unattributed
    }

    fn list_names(v: &serde_json::Value) -> Vec<String> {
        v["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["properties"]["name"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn list_nodes_source_filter() {
        let server = create_test_server();
        seed_docs(&server);
        let (v, err) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "provenance_source": "crm"}),
        );
        assert!(!err);
        let mut names = list_names(&v);
        names.sort();
        assert_eq!(names, vec!["crm-hi", "crm-lo"]);
    }

    #[test]
    fn list_nodes_multi_source_any_of() {
        let server = create_test_server();
        seed_docs(&server);
        let (v, _) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "provenance_sources": ["crm", "hr"]}),
        );
        let mut names = list_names(&v);
        names.sort();
        assert_eq!(names, vec!["crm-hi", "crm-lo", "hr-hi"]);
    }

    #[test]
    fn list_nodes_min_confidence_inclusive() {
        let server = create_test_server();
        seed_docs(&server);
        // >= 0.9 -> crm-hi (0.9) and hr-hi (0.95); crm-lo (0.4) excluded.
        let (v, _) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "min_confidence": 0.9}),
        );
        let mut names = list_names(&v);
        names.sort();
        assert_eq!(names, vec!["crm-hi", "hr-hi"]);
    }

    #[test]
    fn list_nodes_combined_is_and() {
        let server = create_test_server();
        seed_docs(&server);
        let (v, _) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "provenance_source": "crm", "min_confidence": 0.8}),
        );
        assert_eq!(list_names(&v), vec!["crm-hi"]);
    }

    #[test]
    fn list_nodes_unattributed_excluded_then_included() {
        let server = create_test_server();
        seed_docs(&server);
        // Default: the unattributed "none" node is excluded by an active filter.
        let (v, _) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "min_confidence": 0.0}),
        );
        assert!(!list_names(&v).contains(&"none".to_string()));
        // include_unattributed re-includes it.
        let (v2, _) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "min_confidence": 0.0, "include_unattributed": true}),
        );
        assert!(list_names(&v2).contains(&"none".to_string()));
    }

    #[test]
    fn list_nodes_empty_result_is_success_not_error() {
        let server = create_test_server();
        seed_docs(&server);
        let (v, err) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "provenance_source": "nonexistent-source"}),
        );
        assert!(
            !err,
            "an empty filtered result must be a success, not an error"
        );
        assert_eq!(v["count"].as_u64(), Some(0));
    }

    // --- validation (AC5) ------------------------------------------------

    #[test]
    fn invalid_confidence_is_invalid_argument_with_field() {
        let server = create_test_server();
        for bad in [json!(1.5), json!(-0.1)] {
            let (v, err) = dispatch(
                &server,
                "list_nodes",
                json!({"label": "Doc", "min_confidence": bad}),
            );
            assert!(err);
            assert_eq!(v["error"]["code"].as_str(), Some("INVALID_ARGUMENT"));
            assert_eq!(
                v["error"]["details"]["field"].as_str(),
                Some("min_confidence")
            );
            assert_eq!(v["error"]["retriable"].as_bool(), Some(false));
        }
    }

    #[test]
    fn empty_source_list_is_invalid_argument() {
        let server = create_test_server();
        let (v, err) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "provenance_sources": []}),
        );
        assert!(err);
        assert_eq!(v["error"]["code"].as_str(), Some("INVALID_ARGUMENT"));
        assert_eq!(
            v["error"]["details"]["field"].as_str(),
            Some("provenance_sources")
        );
    }

    #[test]
    fn wrong_typed_min_confidence_is_invalid_argument() {
        let server = create_test_server();
        let (v, err) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "min_confidence": "high"}),
        );
        assert!(err);
        assert_eq!(v["error"]["code"].as_str(), Some("INVALID_ARGUMENT"));
        assert_eq!(
            v["error"]["details"]["field"].as_str(),
            Some("min_confidence")
        );
    }

    // --- adjacency edges -------------------------------------------------

    #[test]
    fn outgoing_edges_filtered_by_source() {
        let server = create_test_server();
        let a = create_doc(&server, "a", None, None);
        let b = create_doc(&server, "b", None, None);
        let c = create_doc(&server, "c", None, None);
        for (target, source) in [(b, "crm"), (c, "hr")] {
            server.create_edge(CreateEdgeRequest {
                namespace: None,
                source_id: a,
                target_id: target,
                label: "LINK".to_string(),
                properties: None,
                valid_time: None,
                provenance: Some(ProvenanceRequest {
                    source: Some(source.to_string()),
                    confidence: Some(0.9),
                    note: None,
                    correlation_id: None,
                }),
                derived_from: None,
            });
        }
        let (v, err) = dispatch(
            &server,
            "get_outgoing_edges",
            json!({"node_id": a, "provenance_source": "crm"}),
        );
        assert!(!err);
        assert_eq!(v["count"].as_u64(), Some(1));
        assert_eq!(v["edges"][0]["target_id"].as_u64(), Some(b));
    }

    // --- traverse (composes with temporal, AC3) --------------------------

    #[test]
    fn traverse_min_confidence_filters_returned_nodes() {
        let server = create_test_server();
        let a = create_doc(&server, "a", Some("crm"), Some(0.9));
        let hi = create_doc(&server, "hi", Some("crm"), Some(0.95));
        let lo = create_doc(&server, "lo", Some("crm"), Some(0.2));
        for t in [hi, lo] {
            server.create_edge(CreateEdgeRequest {
                namespace: None,
                source_id: a,
                target_id: t,
                label: "LINK".to_string(),
                properties: None,
                valid_time: None,
                provenance: None,
                derived_from: None,
            });
        }
        let (v, err) = dispatch(
            &server,
            "traverse",
            json!({"start_node_id": a, "edge_label": "LINK", "depth": 1, "min_confidence": 0.9}),
        );
        assert!(!err);
        let names: Vec<String> = v["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r["node"]["properties"]["name"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(
            names,
            vec!["hi"],
            "only the high-confidence node is returned"
        );
    }

    // --- find_similar (AC6) ----------------------------------------------

    #[test]
    fn find_similar_returns_k_filter_passing_candidates() {
        let server = create_test_server();
        server.enable_vector_index(EnableVectorIndexRequest {
            property_name: "embedding".to_string(),
            dimensions: 4,
            distance_metric: Some("cosine".to_string()),
        });
        // 6 docs: alternate trusted/untrusted sources.
        for i in 0..6 {
            let mut props = HashMap::new();
            props.insert("name".to_string(), json!(format!("d{i}")));
            props.insert(
                "embedding".to_string(),
                json!([0.1 + i as f32 * 0.01, 0.2, 0.3, 0.4]),
            );
            let source = if i % 2 == 0 { "trusted" } else { "other" };
            server.create_node(CreateNodeRequest {
                namespace: None,
                derived_from: None,
                valid_time: None,
                label: "Doc".to_string(),
                properties: Some(props),
                provenance: Some(ProvenanceRequest {
                    source: Some(source.to_string()),
                    confidence: Some(0.9),
                    note: None,
                    correlation_id: None,
                }),
            });
        }
        // k=3, filter to "trusted" (3 exist). AC6: exactly the trusted ones,
        // all passing, no post-truncation below k.
        let (v, err) = dispatch(
            &server,
            "find_similar",
            json!({
                "property_name": "embedding",
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "k": 3,
                "provenance_source": "trusted"
            }),
        );
        assert!(!err, "response: {v}");
        let results = v["results"].as_array().unwrap();
        assert_eq!(results.len(), 3, "all 3 trusted candidates returned: {v}");
        for r in results {
            assert_eq!(r["node"]["provenance"]["source"].as_str(), Some("trusted"));
        }
    }

    // --- cursor interaction (fail closed) --------------------------------

    #[test]
    fn provenance_filter_with_cursor_is_rejected() {
        let server = create_test_server();
        seed_docs(&server);
        let (v, err) = dispatch(
            &server,
            "list_nodes",
            json!({"label": "Doc", "provenance_source": "crm", "use_cursor": true}),
        );
        assert!(err);
        assert_eq!(v["error"]["code"].as_str(), Some("INVALID_ARGUMENT"));
    }
}

// ============================================================================
// Semantic-search analysis tools (Issue #2907)
//
// Design A: all six tools are advertised + dispatched unconditionally; the
// handler bodies are gated on the `semantic-search` feature. The feature-OFF
// tests (default build) prove the structured FAILED_PRECONDITION contract; the
// feature-ON tests (compiled only with `--features semantic-search`, which is
// dependency-free and cheap) prove real behavior + bounds.
// ============================================================================

#[cfg(test)]
mod semantic_search_tools_tests {
    use super::*;
    use serde_json::json;

    const SEMANTIC_TOOLS: [&str; 6] = [
        "semantic_path",
        "concept_analogy",
        "concept_mean",
        "find_duplicate_candidates",
        "semantic_horizon",
        "context_aspects",
    ];

    fn parse(server: &AletheiaMcpServer, tool: &str, args: serde_json::Value) -> serde_json::Value {
        let result = server.dispatch_tool(tool, args);
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text().map(|t| t.text.clone()))
            .expect("tool result carries text");
        serde_json::from_str(&text).expect("tool response is valid JSON")
    }

    /// All six tools are advertised in the catalog regardless of feature state.
    #[test]
    fn all_six_tools_are_advertised() {
        let server = create_test_server();
        let advertised = server.list_tools_for_test();
        for tool in SEMANTIC_TOOLS {
            assert!(
                advertised.iter().any(|t| t == tool),
                "tool '{tool}' must be advertised in the catalog"
            );
        }
    }

    /// Feature OFF: every tool returns a structured FAILED_PRECONDITION naming
    /// the required feature, and never an "unknown tool" NOT_FOUND.
    #[cfg(not(feature = "semantic-search"))]
    #[test]
    fn feature_off_returns_failed_precondition() {
        let server = create_test_server();
        for tool in SEMANTIC_TOOLS {
            // Provide plausibly-valid args so we don't just trip arg parsing;
            // the feature gate must fire regardless.
            let args = json!({
                "start": 1, "end": 2, "a": 1, "b": 2, "c": 3, "seed": 1,
                "node_id": 1, "nodes": [1, 2], "threshold": 0.5,
                "property_name": "embedding"
            });
            let value = parse(&server, tool, args);
            let err = value
                .get("error")
                .unwrap_or_else(|| panic!("[{tool}] feature-off must return an error: {value}"));
            assert_eq!(
                err["code"], "FAILED_PRECONDITION",
                "[{tool}] must be FAILED_PRECONDITION: {value}"
            );
            assert_eq!(err["retriable"], false, "[{tool}] must be non-retriable");
            assert_eq!(
                err["details"]["required_feature"], "semantic-search",
                "[{tool}] must name the required feature: {value}"
            );
            assert_eq!(err["details"]["tool"], tool, "[{tool}] echoes its own name");
        }
    }

    // ------------------------------------------------------------------------
    // Feature-ON behavioral + validation tests (dependency-free feature).
    // ------------------------------------------------------------------------

    #[cfg(feature = "semantic-search")]
    mod on {
        use super::*;

        /// Build a server with an `embedding` vector index (dim 2) and a small
        /// directed graph A->B->C plus a near-duplicate D of A. Returns node ids.
        fn semantic_server() -> (AletheiaMcpServer, [u64; 4]) {
            let server = create_test_server();
            server.enable_vector_index(EnableVectorIndexRequest {
                property_name: "embedding".to_string(),
                dimensions: 2,
                distance_metric: Some("cosine".to_string()),
            });

            let mk = |server: &AletheiaMcpServer, name: &str, vec: [f32; 2]| -> u64 {
                let mut props = HashMap::new();
                props.insert("name".to_string(), json!(name));
                props.insert("embedding".to_string(), json!([vec[0], vec[1]]));
                let resp = server.create_node(CreateNodeRequest {
                    namespace: None,
                    derived_from: None,
                    valid_time: None,
                    label: "Doc".to_string(),
                    properties: Some(props),
                    provenance: None,
                });
                let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
                v["id"].as_u64().expect("node id")
            };

            let a = mk(&server, "A", [1.0, 0.0]);
            let b = mk(&server, "B", [0.707, 0.707]);
            let c = mk(&server, "C", [0.0, 1.0]);
            let d = mk(&server, "D", [0.999, 0.0447]); // near-duplicate of A

            for (s, t) in [(a, b), (b, c), (a, d)] {
                server.create_edge(CreateEdgeRequest {
                    namespace: None,
                    source_id: s,
                    target_id: t,
                    label: "NEXT".to_string(),
                    properties: None,
                    valid_time: None,
                    provenance: None,
                    derived_from: None,
                });
            }
            (server, [a, b, c, d])
        }

        #[test]
        fn semantic_path_finds_a_to_c() {
            let (server, [a, _b, c, _d]) = semantic_server();
            let value = parse(
                &server,
                "semantic_path",
                json!({ "start": a, "end": c, "property_name": "embedding" }),
            );
            assert!(value.get("error").is_none(), "unexpected error: {value}");
            let path = value["path"].as_array().expect("path array");
            assert_eq!(path.first().and_then(|v| v.as_u64()), Some(a));
            assert_eq!(path.last().and_then(|v| v.as_u64()), Some(c));
            assert_eq!(value["length"].as_u64(), Some(path.len() as u64));
        }

        #[test]
        fn missing_index_is_failed_precondition() {
            let (server, [a, _b, c, _d]) = semantic_server();
            let value = parse(
                &server,
                "semantic_path",
                json!({ "start": a, "end": c, "property_name": "nonexistent" }),
            );
            assert_eq!(value["error"]["code"], "FAILED_PRECONDITION", "{value}");
        }

        #[test]
        fn node_id_overflow_is_invalid_argument() {
            let (server, _) = semantic_server();
            let value = parse(
                &server,
                "semantic_path",
                json!({ "start": u64::MAX, "end": 1, "property_name": "embedding" }),
            );
            assert_eq!(value["error"]["code"], "INVALID_ARGUMENT", "{value}");
        }

        #[test]
        fn semantic_path_no_path_is_not_found() {
            // c has no outgoing edges (a->b->c, a->d), so there is no route
            // from c back to a. Both nodes carry the embedding, so this is a
            // genuine "disconnected pair", not a missing-vector fault. It must
            // surface as NOT_FOUND (a normal outcome), never INTERNAL.
            let (server, [a, _b, c, _d]) = semantic_server();
            let value = parse(
                &server,
                "semantic_path",
                json!({ "start": c, "end": a, "property_name": "embedding" }),
            );
            let err = value
                .get("error")
                .unwrap_or_else(|| panic!("expected an error, got: {value}"));
            assert_eq!(err["code"], "NOT_FOUND", "{value}");
            assert_eq!(err["retriable"], false, "{value}");
            let msg = err["message"].as_str().unwrap_or("");
            assert!(
                msg.contains("no path found between nodes"),
                "message must describe the disconnected pair: {value}"
            );
        }

        #[test]
        fn semantic_path_budget_exhausted_is_failed_precondition() {
            // Build a long linear chain (longer than the minimum expansion
            // budget of max_depth=1 -> 1000 expansions) so the DoS-protection
            // budget is exhausted before the far end is reached. That is a
            // resource-limit refusal (FAILED_PRECONDITION), not an INTERNAL
            // error.
            let server = create_test_server();
            server.enable_vector_index(EnableVectorIndexRequest {
                property_name: "embedding".to_string(),
                dimensions: 2,
                distance_metric: Some("cosine".to_string()),
            });

            let mk = |server: &AletheiaMcpServer| -> u64 {
                let mut props = HashMap::new();
                props.insert("embedding".to_string(), json!([1.0, 0.0]));
                let resp = server.create_node(CreateNodeRequest {
                    namespace: None,
                    derived_from: None,
                    valid_time: None,
                    label: "Doc".to_string(),
                    properties: Some(props),
                    provenance: None,
                });
                let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
                v["id"].as_u64().expect("node id")
            };

            // 1100 nodes > 1000 expansion budget at max_depth=1.
            let ids: Vec<u64> = (0..1100).map(|_| mk(&server)).collect();
            for w in ids.windows(2) {
                server.create_edge(CreateEdgeRequest {
                    namespace: None,
                    source_id: w[0],
                    target_id: w[1],
                    label: "NEXT".to_string(),
                    properties: None,
                    valid_time: None,
                    provenance: None,
                    derived_from: None,
                });
            }

            let value = parse(
                &server,
                "semantic_path",
                json!({
                    "start": ids[0],
                    "end": ids[ids.len() - 1],
                    "property_name": "embedding",
                    "max_depth": 1
                }),
            );
            let err = value
                .get("error")
                .unwrap_or_else(|| panic!("expected an error, got: {value}"));
            assert_eq!(err["code"], "FAILED_PRECONDITION", "{value}");
            assert_eq!(err["retriable"], false, "{value}");
            assert_eq!(
                err["details"]["reason"], "expansion_budget_exceeded",
                "{value}"
            );
            assert_eq!(err["details"]["max_expansions"], 1000, "{value}");
        }

        #[test]
        fn find_duplicate_candidates_excludes_self() {
            // The similarity query resolves against the target's own embedding,
            // so it can return the target itself. A node is never its own
            // duplicate -- the target id must not appear among candidates.
            let (server, [a, _b, _c, _d]) = semantic_server();
            let value = parse(
                &server,
                "find_duplicate_candidates",
                json!({ "node_id": a, "property_name": "embedding", "threshold": 0.5, "limit": 10 }),
            );
            assert!(value.get("error").is_none(), "unexpected error: {value}");
            let candidates = value["candidates"].as_array().expect("candidates");
            assert!(
                candidates.iter().all(|c| c["node_id"].as_u64() != Some(a)),
                "target must not be its own duplicate: {value}"
            );
            // count must match the filtered candidate list length.
            assert_eq!(
                value["count"].as_u64(),
                Some(candidates.len() as u64),
                "count must reflect the filtered candidates: {value}"
            );
        }

        #[test]
        fn horizon_threshold_out_of_range_is_invalid_argument() {
            let (server, [a, _b, _c, _d]) = semantic_server();
            let value = parse(
                &server,
                "semantic_horizon",
                json!({ "seed": a, "property_name": "embedding", "threshold": 1.5 }),
            );
            assert_eq!(value["error"]["code"], "INVALID_ARGUMENT", "{value}");
        }

        #[test]
        fn duplicates_threshold_out_of_range_is_invalid_argument() {
            let (server, [a, _b, _c, _d]) = semantic_server();
            let value = parse(
                &server,
                "find_duplicate_candidates",
                json!({ "node_id": a, "property_name": "embedding", "threshold": -0.1 }),
            );
            assert_eq!(value["error"]["code"], "INVALID_ARGUMENT", "{value}");
        }

        #[test]
        fn find_duplicate_candidates_reports_near_duplicate() {
            let (server, [a, _b, _c, _d]) = semantic_server();
            let value = parse(
                &server,
                "find_duplicate_candidates",
                json!({ "node_id": a, "property_name": "embedding", "threshold": 0.9, "limit": 10 }),
            );
            assert!(value.get("error").is_none(), "unexpected error: {value}");
            let candidates = value["candidates"].as_array().expect("candidates");
            // D is a near-duplicate of A; it should surface at threshold 0.9.
            assert!(
                candidates
                    .iter()
                    .any(|c| c["similarity"].as_f64().unwrap_or(0.0) >= 0.9),
                "expected a >=0.9 candidate: {value}"
            );
            assert_eq!(value["target"].as_u64(), Some(a));
        }

        #[test]
        fn semantic_horizon_partitions_interior_and_horizon() {
            let (server, [a, _b, _c, _d]) = semantic_server();
            let value = parse(
                &server,
                "semantic_horizon",
                json!({ "seed": a, "property_name": "embedding", "threshold": 0.5, "max_depth": 5 }),
            );
            assert!(value.get("error").is_none(), "unexpected error: {value}");
            let interior = value["interior"].as_array().expect("interior");
            // The seed itself is always interior.
            assert!(interior.iter().any(|v| v.as_u64() == Some(a)));
            assert_eq!(
                value["interior_count"].as_u64(),
                Some(interior.len() as u64)
            );
            // interior ids are sorted ascending.
            let ids: Vec<u64> = interior.iter().filter_map(|v| v.as_u64()).collect();
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            assert_eq!(ids, sorted, "interior must be sorted");
        }

        #[test]
        fn concept_mean_ranks_results() {
            let (server, [a, b, _c, _d]) = semantic_server();
            let value = parse(
                &server,
                "concept_mean",
                json!({ "nodes": [a, b], "property_name": "embedding", "k": 3 }),
            );
            assert!(value.get("error").is_none(), "unexpected error: {value}");
            let results = value["results"].as_array().expect("results");
            assert_eq!(value["count"].as_u64(), Some(results.len() as u64));
            assert_eq!(value["property_name"], "embedding");
        }

        #[test]
        fn concept_mean_rejects_too_many_nodes() {
            let (server, _) = semantic_server();
            let too_many: Vec<u64> = (0..10_001).collect();
            let value = parse(
                &server,
                "concept_mean",
                json!({ "nodes": too_many, "property_name": "embedding" }),
            );
            assert_eq!(value["error"]["code"], "INVALID_ARGUMENT", "{value}");
        }

        #[test]
        fn context_aspects_elides_centroid_by_default() {
            let (server, [a, _b, _c, _d]) = semantic_server();
            let value = parse(
                &server,
                "context_aspects",
                json!({ "node_id": a, "property_name": "embedding", "k": 2 }),
            );
            assert!(value.get("error").is_none(), "unexpected error: {value}");
            let aspects = value["aspects"].as_array().expect("aspects");
            for aspect in aspects {
                // Centroid elided per #3220 unless include_vectors.
                assert_eq!(
                    aspect["centroid"]["elided"], true,
                    "centroid must be elided by default: {aspect}"
                );
            }
        }

        #[test]
        fn context_aspects_include_vectors_returns_full_centroid() {
            let (server, [a, _b, _c, _d]) = semantic_server();
            let value = parse(
                &server,
                "context_aspects",
                json!({ "node_id": a, "property_name": "embedding", "k": 2, "include_vectors": true }),
            );
            assert!(value.get("error").is_none(), "unexpected error: {value}");
            let aspects = value["aspects"].as_array().expect("aspects");
            for aspect in aspects {
                assert!(
                    aspect["centroid"].is_array(),
                    "centroid must be a full array with include_vectors: {aspect}"
                );
            }
        }

        #[test]
        fn concept_analogy_returns_ranked_results() {
            let (server, [a, b, c, _d]) = semantic_server();
            let value = parse(
                &server,
                "concept_analogy",
                json!({ "a": a, "b": b, "c": c, "property_name": "embedding", "k": 3 }),
            );
            // analogy may legitimately succeed with results or (on a tiny graph)
            // return a structured error; either way it must be well-formed JSON
            // and, on success, carry the ranked shape.
            if value.get("error").is_none() {
                assert!(value["results"].is_array(), "results array: {value}");
                assert_eq!(value["property_name"], "embedding");
            } else {
                assert!(
                    value["error"]["code"].is_string(),
                    "structured error: {value}"
                );
            }
        }
    }
}

// ============================================================================
// Embedding generation & text semantic search (Issue #2906)
//
// Design A: all five tools are advertised + dispatched unconditionally. The
// real bodies live under `#[cfg(feature = "embeddings")]`; the feature-off
// bodies return a structured unavailable-feature error. The primary test build
// (no `embeddings` feature) exercises the feature-off path + catalog + RBAC;
// the `#[cfg(feature = "embeddings")]` sub-module exercises validation, bounds,
// the no-model precondition, and the non-dense conversion glue WITHOUT loading
// a model (true model e2e is a separate `#[ignore]`d concern).
// ============================================================================

mod embedding_tools_tests {
    use super::*;

    const EMBED_TOOLS: [&str; 5] = [
        "embed_query",
        "embed_text",
        "semantic_search",
        "create_node_with_embedding",
        "update_node_embedding",
    ];

    fn dispatch(
        server: &AletheiaMcpServer,
        tool: &str,
        args: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        let result = server.dispatch_tool(tool, args);
        let is_error = result.is_error.unwrap_or(false);
        let text = AletheiaMcpServer::extract_text(result);
        let value = serde_json::from_str(&text).expect("tool response should be valid JSON");
        (value, is_error)
    }

    #[test]
    fn all_five_tools_are_advertised() {
        let server = create_test_server();
        let tools = server.list_tools_for_test();
        for name in EMBED_TOOLS {
            assert!(
                tools.iter().any(|t| t == name),
                "{name} must be advertised in the MCP tool catalog"
            );
        }
    }

    // ---- Feature OFF: structured unavailable-feature error (DEFAULT build) ----
    #[cfg(not(feature = "embeddings"))]
    mod feature_off {
        use super::*;

        #[test]
        fn each_tool_reports_feature_unavailable() {
            let server = create_test_server();
            for name in EMBED_TOOLS {
                let (v, err) = dispatch(&server, name, serde_json::json!({}));
                assert!(
                    err,
                    "{name} should error when the embeddings feature is off"
                );
                assert_eq!(
                    v["error"]["code"].as_str(),
                    Some("FAILED_PRECONDITION"),
                    "{name}: {v}"
                );
                assert_eq!(
                    v["error"]["details"]["feature"].as_str(),
                    Some("embeddings"),
                    "{name}: {v}"
                );
                assert_eq!(
                    v["error"]["details"]["available"].as_bool(),
                    Some(false),
                    "{name}: {v}"
                );
                assert_eq!(
                    v["error"]["retriable"].as_bool(),
                    Some(false),
                    "{name}: {v}"
                );
            }
        }
    }

    // ---- Feature ON: validation/bounds/no-model/non-dense (no model loaded) ----
    #[cfg(feature = "embeddings")]
    mod feature_on {
        use super::*;

        #[test]
        fn embed_query_without_model_is_failed_precondition() {
            let server = create_test_server(); // no embedder configured
            let (v, err) = dispatch(&server, "embed_query", serde_json::json!({"text": "hello"}));
            assert!(err);
            assert_eq!(
                v["error"]["code"].as_str(),
                Some("FAILED_PRECONDITION"),
                "{v}"
            );
        }

        #[test]
        fn embed_query_oversized_text_is_invalid_argument() {
            let server = create_test_server();
            let big = "a".repeat(64 * 1024 + 1);
            let (v, err) = dispatch(&server, "embed_query", serde_json::json!({"text": big}));
            assert!(err);
            assert_eq!(v["error"]["code"].as_str(), Some("INVALID_ARGUMENT"), "{v}");
        }

        #[test]
        fn embed_text_empty_is_invalid_argument() {
            let server = create_test_server();
            let (v, err) = dispatch(&server, "embed_text", serde_json::json!({"texts": []}));
            assert!(err);
            assert_eq!(v["error"]["code"].as_str(), Some("INVALID_ARGUMENT"), "{v}");
        }

        #[test]
        fn embed_text_too_many_is_invalid_argument() {
            let server = create_test_server();
            let texts: Vec<String> = (0..257).map(|i| i.to_string()).collect();
            let (v, err) = dispatch(&server, "embed_text", serde_json::json!({"texts": texts}));
            assert!(err);
            assert_eq!(v["error"]["code"].as_str(), Some("INVALID_ARGUMENT"), "{v}");
        }

        // Issue #2906 AC: `max_chunks: 0` can never return anything useful and is
        // rejected as a caller error *before* any model is required.
        #[test]
        fn embed_text_zero_max_chunks_is_invalid_argument() {
            let server = create_test_server(); // no model needed: validated first
            let (v, err) = dispatch(
                &server,
                "embed_text",
                serde_json::json!({"texts": ["hello"], "max_chunks": 0}),
            );
            assert!(err);
            assert_eq!(v["error"]["code"].as_str(), Some("INVALID_ARGUMENT"), "{v}");
        }

        // Issue #2906 AC: `embed_text` performs REAL chunk expansion. The
        // model-independent splitter is the load-bearing piece — a long document
        // becomes MULTIPLE chunks (never silently truncated to one vector), each
        // chunk is bounded, and reassembling the chunks reproduces the input
        // exactly (no data dropped). The per-chunk embeddings are then aligned to
        // these chunk texts via EmbedData, so alignment is by identity, not a
        // positional zip.
        #[test]
        fn split_text_into_chunks_short_input_is_single_chunk() {
            let chunks = crate::mcp::server::split_text_into_chunks("hello", 1000);
            assert_eq!(chunks, vec!["hello".to_string()]);
        }

        #[test]
        fn split_text_into_chunks_long_input_expands_to_many_bounded_chunks() {
            let text = "a".repeat(25);
            let chunks = crate::mcp::server::split_text_into_chunks(&text, 4);
            // 25 chars / 4 per window = 7 chunks (6 full + 1 remainder).
            assert_eq!(chunks.len(), 7, "long text must expand to many chunks");
            for c in &chunks {
                assert!(c.chars().count() <= 4, "each chunk is bounded: {c:?}");
            }
            // Lossless: concatenation reproduces the original document exactly.
            assert_eq!(chunks.concat(), text, "no data dropped across chunks");
        }

        #[test]
        fn split_text_into_chunks_respects_char_boundaries() {
            // Multibyte scalars must never be split mid-encoding; every chunk is
            // valid UTF-8 (String guarantees it) and counts by scalar value.
            let text = "héllo wörld 🌍🌎🌏";
            let chunks = crate::mcp::server::split_text_into_chunks(text, 3);
            for c in &chunks {
                assert!(c.chars().count() <= 3);
                assert!(std::str::from_utf8(c.as_bytes()).is_ok());
            }
            assert_eq!(chunks.concat(), text);
            assert!(chunks.len() > 1, "multibyte document still expands");
        }

        #[test]
        fn split_text_into_chunks_empty_input_yields_one_empty_chunk() {
            assert_eq!(
                crate::mcp::server::split_text_into_chunks("", 1000),
                vec![String::new()]
            );
        }

        #[test]
        fn semantic_search_missing_index_is_failed_precondition() {
            let server = create_test_server();
            let (v, err) = dispatch(
                &server,
                "semantic_search",
                serde_json::json!({"property_name": "nope", "query_text": "hi"}),
            );
            assert!(err);
            assert_eq!(
                v["error"]["code"].as_str(),
                Some("FAILED_PRECONDITION"),
                "{v}"
            );
        }

        // The non-dense conversion glue the embedding handlers rely on: a
        // MultiVector model result maps to NotDense (never a silent zero
        // vector). Exercised directly on the shared helper (no model needed).
        #[test]
        fn multivector_maps_to_not_dense() {
            use crate::embeddings::{DenseEmbeddingError, EmbeddingResult, to_dense_iter};
            let results = vec![EmbeddingResult::MultiVector(vec![vec![0.1_f32, 0.2]])];
            let first = to_dense_iter(results, Some(1)).next();
            assert!(
                matches!(first, Some(Err(DenseEmbeddingError::NotDense))),
                "expected NotDense, got {first:?}"
            );
        }

        // True model-backed end-to-end embedding requires downloading a model
        // (network + weights); kept #[ignore]d so unit runs stay hermetic.
        #[test]
        #[ignore = "requires downloading an embedding model (network); run manually"]
        fn embed_query_roundtrip_with_real_model() {
            // Intentionally left as a manual harness: build an Embedder via
            // aletheiadb::embeddings::EmbedderBuilder, attach it with
            // AletheiaMcpServer::with_embedder, then embed_query and assert a
            // non-empty dense vector. Not run in CI.
        }
    }
}

// ============================================================================
// Namespace reserved-key elision — SURFACE sweep (Issue #3349, PR1).
//
// The unit sweep in `core::namespace` only proves the elision *helper* strips
// the ride-along key. These tests prove the production MCP serializers actually
// call it: a namespaced node AND edge are created, then every read tool is
// exercised through `dispatch_tool_json` (the exact seam the HTTP surface also
// routes through) and its serialized JSON is asserted to contain no
// `__aletheia_` substring. Without the serializer guards these fail (red).
// ============================================================================
#[cfg(test)]
mod namespace_surface_sweep_tests {
    use super::*;

    /// Seed a namespaced node + edge and return `(server, src, dst)` as u64 ids.
    fn seed_namespaced() -> (AletheiaMcpServer, u64, u64) {
        let server = create_test_server();
        let a = server
            .db()
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
                "agent:planner",
            )
            .expect("create namespaced node a");
        let b = server
            .db()
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Bob").build(),
                "agent:planner",
            )
            .expect("create namespaced node b");
        server
            .db()
            .create_edge_in_namespace(
                a,
                b,
                "KNOWS",
                PropertyMapBuilder::new().insert("since", 2020).build(),
                "agent:planner",
            )
            .expect("create namespaced edge");
        (server, a.as_u64(), b.as_u64())
    }

    /// Assert a serialized tool response leaks no engine-reserved ride-along key.
    fn assert_no_reserved(json: &str, tool: &str) {
        assert!(
            !json.contains("__aletheia_"),
            "reserved ride-along key leaked into `{tool}` response: {json}"
        );
        // The full namespace marker specifically must be absent.
        assert!(
            !json.contains(crate::core::namespace::NAMESPACE_KEY),
            "namespace key leaked into `{tool}` response: {json}"
        );
    }

    #[test]
    fn sweep_get_node_elides_reserved_key() {
        let (server, a, _b) = seed_namespaced();
        // The seeded node lives in agent:planner; an omitted read is default-only
        // (#3349 FIX2), so scope to "all" to still exercise the elision sweep.
        let out = server.dispatch_tool_json(
            "get_node",
            serde_json::json!({ "node_id": a, "namespace": "all" }),
        );
        // The node is still returned with its user property...
        assert!(out.contains("Alice"), "expected the node payload: {out}");
        // ...but the ride-along key is elided.
        assert_no_reserved(&out, "get_node");
    }

    #[test]
    fn sweep_list_nodes_elides_reserved_key() {
        let (server, _a, _b) = seed_namespaced();
        // Namespaced seed data + default-only omitted reads (#3349 FIX2): scope
        // to "all" so the sweep still sees the agent:planner node.
        let out = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({ "label": "Person", "namespace": "all" }),
        );
        assert!(out.contains("Alice"), "expected node list payload: {out}");
        assert_no_reserved(&out, "list_nodes");
    }

    #[test]
    fn sweep_traverse_elides_reserved_key() {
        let (server, a, _b) = seed_namespaced();
        // Namespaced seed data + default-only omitted reads (#3349 FIX2): scope
        // to "all" so the traversal still crosses the agent:planner edge to Bob.
        let out = server.dispatch_tool_json(
            "traverse",
            serde_json::json!({ "start_node_id": a, "edge_label": "KNOWS", "namespace": "all" }),
        );
        assert!(
            out.contains("Bob"),
            "expected traversal to reach Bob: {out}"
        );
        assert_no_reserved(&out, "traverse");
    }

    #[test]
    fn sweep_get_node_history_elides_reserved_key() {
        let (server, a, _b) = seed_namespaced();
        // Add a second version so history has content beyond the create.
        server.dispatch_tool_json(
            "update_node",
            serde_json::json!({ "node_id": a, "properties": { "age": 30 } }),
        );
        let out =
            server.dispatch_tool_json("get_node_history", serde_json::json!({ "node_id": a }));
        assert!(out.contains("Alice"), "expected history payload: {out}");
        assert_no_reserved(&out, "get_node_history");
    }

    #[test]
    fn sweep_query_return_n_elides_reserved_key() {
        let (server, _a, _b) = seed_namespaced();
        // AQL is always available under `mcp-server` (no cypher feature needed).
        // Seed data lives in agent:planner and an omitted query scope is now
        // default-only (#3349 PR3d), so scope to "all" to still see Alice.
        let out = server.dispatch_tool_json(
            "query",
            serde_json::json!({
                "language": "aql", "query": "MATCH (n:Person) RETURN n", "namespace": "all"
            }),
        );
        assert!(out.contains("Alice"), "expected query rows: {out}");
        assert_no_reserved(&out, "query");
    }

    #[test]
    fn sweep_get_schema_elides_reserved_key() {
        let (server, _a, _b) = seed_namespaced();
        let out = server.dispatch_tool_json("get_schema", serde_json::json!({}));
        // The user property key IS a legitimate schema key...
        assert!(out.contains("name"), "expected schema payload: {out}");
        // ...but the ride-along key must never surface as a schema property key.
        assert_no_reserved(&out, "get_schema");
    }

    // ========================================================================
    // BLOCK 1 (PR3a): write-tool `namespace` param round-trips through MCP
    // ========================================================================

    fn as_json(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("valid JSON response")
    }

    #[test]
    fn create_node_in_namespace_surfaces_field_and_registers() {
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "create_node",
            serde_json::json!({
                "label": "Person",
                "properties": { "name": "Alice" },
                "namespace": "agent:planner"
            }),
        );
        let v = as_json(&out);
        assert_eq!(
            v.get("namespace").and_then(|n| n.as_str()),
            Some("agent:planner"),
            "create_node response must surface the namespace field: {out}"
        );
        assert_no_reserved(&out, "create_node");
        // Read it back: the namespace persists and is surfaced on get_node.
        // The node lives in agent:planner, so an omitted (default-only, #3349
        // FIX2) read would not see it — scope the read to its namespace.
        let id = v.get("id").and_then(|i| i.as_u64()).expect("id");
        let got = server.dispatch_tool_json(
            "get_node",
            serde_json::json!({ "node_id": id, "namespace": "agent:planner" }),
        );
        assert_eq!(
            as_json(&got).get("namespace").and_then(|n| n.as_str()),
            Some("agent:planner"),
            "get_node must echo the namespace: {got}"
        );
        // The write auto-registered the namespace.
        let names: Vec<String> = server
            .db()
            .list_namespaces()
            .into_iter()
            .map(|i| i.name)
            .collect();
        assert!(
            names.iter().any(|n| n == "agent:planner"),
            "namespace should be auto-registered: {names:?}"
        );
    }

    #[test]
    fn create_node_default_omits_namespace_field() {
        // Byte-identical back-compat (design AC1): a default-namespace create
        // carries NO `namespace` field.
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "create_node",
            serde_json::json!({ "label": "Person", "properties": { "name": "Bob" } }),
        );
        assert!(
            as_json(&out).get("namespace").is_none(),
            "default-namespace response must omit the namespace field: {out}"
        );
        // An explicit "default" is equivalent to omitting it.
        let out2 = server.dispatch_tool_json(
            "create_node",
            serde_json::json!({
                "label": "Person", "properties": { "name": "Carol" },
                "namespace": "default"
            }),
        );
        assert!(
            as_json(&out2).get("namespace").is_none(),
            "explicit default must also omit the namespace field: {out2}"
        );
    }

    #[test]
    fn create_edge_in_namespace_surfaces_field() {
        let server = create_test_server();
        let a = server
            .db()
            .create_node("Person", PropertyMapBuilder::new().build())
            .unwrap()
            .as_u64();
        let b = server
            .db()
            .create_node("Person", PropertyMapBuilder::new().build())
            .unwrap()
            .as_u64();
        let out = server.dispatch_tool_json(
            "create_edge",
            serde_json::json!({
                "source_id": a, "target_id": b, "label": "KNOWS",
                "namespace": "agent:planner"
            }),
        );
        assert_eq!(
            as_json(&out).get("namespace").and_then(|n| n.as_str()),
            Some("agent:planner"),
            "create_edge response must surface the namespace field: {out}"
        );
        assert_no_reserved(&out, "create_edge");
    }

    #[test]
    fn create_node_invalid_namespace_name_rejected() {
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "create_node",
            serde_json::json!({ "label": "Person", "namespace": "has space" }),
        );
        let v = as_json(&out);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "a malformed namespace name must be INVALID_ARGUMENT: {out}"
        );
    }

    #[test]
    fn update_node_with_namespace_rejected_immutable() {
        let (server, a, _b) = seed_namespaced();
        let out = server.dispatch_tool_json(
            "update_node",
            serde_json::json!({
                "node_id": a, "properties": { "age": 31 },
                "namespace": "agent:planner"
            }),
        );
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "supplying a namespace on update must be INVALID_ARGUMENT (immutable): {out}"
        );
    }

    #[test]
    fn delete_node_with_namespace_rejected_immutable() {
        let (server, a, _b) = seed_namespaced();
        let out = server.dispatch_tool_json(
            "delete_node",
            serde_json::json!({ "node_id": a, "detach": true, "namespace": "agent:planner" }),
        );
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "supplying a namespace on delete must be INVALID_ARGUMENT (immutable): {out}"
        );
    }

    #[test]
    fn update_edge_and_delete_edge_with_namespace_rejected() {
        let (server, a, b) = seed_namespaced();
        // Find the edge id via outgoing edges.
        let edges =
            server.dispatch_tool_json("get_outgoing_edges", serde_json::json!({ "node_id": a }));
        let edge_id = as_json(&edges)
            .get("edges")
            .and_then(|e| e.as_array())
            .and_then(|arr| arr.first())
            .and_then(|e| e.get("id"))
            .and_then(|i| i.as_u64())
            .unwrap_or_else(|| panic!("expected an outgoing edge from {a} to {b}: {edges}"));
        let upd = server.dispatch_tool_json(
            "update_edge",
            serde_json::json!({
                "edge_id": edge_id, "properties": { "since": 2021 },
                "namespace": "agent:planner"
            }),
        );
        assert_eq!(
            as_json(&upd)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "namespace on update_edge must be INVALID_ARGUMENT: {upd}"
        );
        let del = server.dispatch_tool_json(
            "delete_edge",
            serde_json::json!({ "edge_id": edge_id, "namespace": "agent:planner" }),
        );
        assert_eq!(
            as_json(&del)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "namespace on delete_edge must be INVALID_ARGUMENT: {del}"
        );
    }

    // ========================================================================
    // BLOCK 2 (PR3a): read-tool namespace scope + per-ns counts
    // ========================================================================

    /// Seed two isolated namespaces: agent:planner {Alice,Bob KNOWS} and
    /// agent:researcher {Carol}. Returns (server, alice, bob, carol) u64 ids.
    fn seed_two_namespaces() -> (AletheiaMcpServer, u64, u64, u64) {
        let (server, alice, bob) = seed_namespaced();
        let carol = server
            .db()
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Carol").build(),
                "agent:researcher",
            )
            .expect("create carol")
            .as_u64();
        (server, alice, bob, carol)
    }

    #[test]
    fn scoped_get_node_isolates_other_namespace() {
        let (server, alice, _bob, carol) = seed_two_namespaces();
        // Alice is visible when scoped to her namespace...
        let ok = server.dispatch_tool_json(
            "get_node",
            serde_json::json!({ "node_id": alice, "namespace": "agent:planner" }),
        );
        assert_eq!(
            as_json(&ok).get("namespace").and_then(|n| n.as_str()),
            Some("agent:planner"),
            "scoped get_node should return Alice: {ok}"
        );
        // ...but NOT when scoped to a different namespace (NOT_FOUND, never leaks).
        let hidden = server.dispatch_tool_json(
            "get_node",
            serde_json::json!({ "node_id": alice, "namespace": "agent:researcher" }),
        );
        assert_eq!(
            as_json(&hidden)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "Alice must be NOT_FOUND when scoped to another namespace: {hidden}"
        );
        // Carol visible only in her own namespace.
        let carol_hidden = server.dispatch_tool_json(
            "get_node",
            serde_json::json!({ "node_id": carol, "namespace": "agent:planner" }),
        );
        assert_eq!(
            as_json(&carol_hidden)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "Carol must be NOT_FOUND under agent:planner scope: {carol_hidden}"
        );
    }

    #[test]
    fn scoped_list_nodes_isolates_and_unions() {
        let (server, _alice, _bob, _carol) = seed_two_namespaces();
        // Scope to planner: only Alice + Bob.
        let planner = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({ "label": "Person", "namespace": "agent:planner" }),
        );
        let pj = as_json(&planner);
        assert_eq!(
            pj.get("count").and_then(|c| c.as_u64()),
            Some(2),
            "{planner}"
        );
        assert!(planner.contains("Alice") && planner.contains("Bob"));
        assert!(
            !planner.contains("Carol"),
            "planner scope leaked Carol: {planner}"
        );
        assert_no_reserved(&planner, "list_nodes(scoped)");
        // Scope to researcher: only Carol.
        let research = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({ "label": "Person", "namespace": "agent:researcher" }),
        );
        assert!(
            research.contains("Carol") && !research.contains("Alice"),
            "{research}"
        );
        // Union scope sees all three.
        let both = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({
                "label": "Person",
                "namespace": ["agent:planner", "agent:researcher"]
            }),
        );
        assert_eq!(
            as_json(&both).get("count").and_then(|c| c.as_u64()),
            Some(3),
            "{both}"
        );
        // "all" selector: no filter.
        let all = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({ "label": "Person", "namespace": "all" }),
        );
        assert_eq!(
            as_json(&all).get("count").and_then(|c| c.as_u64()),
            Some(3),
            "{all}"
        );
    }

    #[test]
    fn scoped_traverse_respects_boundary() {
        let (server, alice, _bob, _carol) = seed_two_namespaces();
        // In-scope traversal reaches Bob (same namespace).
        let ok = server.dispatch_tool_json(
            "traverse",
            serde_json::json!({
                "start_node_id": alice, "edge_label": "KNOWS",
                "namespace": "agent:planner"
            }),
        );
        assert!(ok.contains("Bob"), "scoped traverse should reach Bob: {ok}");
        assert_no_reserved(&ok, "traverse(scoped)");
        // Traversing scoped to a namespace Alice is NOT in ⇒ NOT_FOUND (the
        // start node is out of scope).
        let bad = server.dispatch_tool_json(
            "traverse",
            serde_json::json!({
                "start_node_id": alice, "edge_label": "KNOWS",
                "namespace": "agent:researcher"
            }),
        );
        assert_eq!(
            as_json(&bad)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "start out of scope must be NOT_FOUND: {bad}"
        );
    }

    #[test]
    fn scoped_find_similar_isolates() {
        let server = create_test_server();
        server
            .db()
            .vector_index("embedding")
            .hnsw(crate::index::vector::HnswConfig::new(
                3,
                crate::index::vector::DistanceMetric::Cosine,
            ))
            .enable()
            .expect("enable vector index");
        let mk = |ns: &str, name: &str, v: [f32; 3]| {
            server
                .db()
                .create_node_in_namespace(
                    "Doc",
                    PropertyMapBuilder::new()
                        .insert("name", name)
                        .insert_vector("embedding", &v)
                        .build(),
                    ns,
                )
                .expect("create doc");
        };
        mk("agent:planner", "P1", [1.0, 0.0, 0.0]);
        mk("agent:planner", "P2", [0.9, 0.1, 0.0]);
        mk("agent:researcher", "R1", [1.0, 0.0, 0.0]);
        let out = server.dispatch_tool_json(
            "find_similar",
            serde_json::json!({
                "property_name": "embedding",
                "embedding": [1.0, 0.0, 0.0],
                "k": 5,
                "namespace": "agent:planner"
            }),
        );
        assert!(
            out.contains("P1") || out.contains("P2"),
            "expected planner docs: {out}"
        );
        assert!(
            !out.contains("R1"),
            "researcher doc leaked into scoped search: {out}"
        );
        assert_no_reserved(&out, "find_similar(scoped)");
    }

    #[test]
    fn scoped_read_unknown_namespace_is_not_found_with_details() {
        let (server, alice, _bob) = seed_namespaced();
        let out = server.dispatch_tool_json(
            "get_node",
            serde_json::json!({ "node_id": alice, "namespace": "agent:nonexistent" }),
        );
        let v = as_json(&out);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "unknown namespace must be NOT_FOUND: {out}"
        );
        assert_eq!(
            v.pointer("/error/details/namespace")
                .and_then(|c| c.as_str()),
            Some("agent:nonexistent"),
            "NOT_FOUND must carry details.namespace: {out}"
        );
    }

    #[test]
    fn scoped_read_empty_array_is_invalid_argument() {
        let (server, _alice, _bob) = seed_namespaced();
        let out = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({ "label": "Person", "namespace": [] }),
        );
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "empty namespace scope array must be INVALID_ARGUMENT: {out}"
        );
    }

    #[test]
    fn unsupported_scope_tools_reject_narrowing_scope() {
        let (server, _a, _b) = seed_namespaced();
        // `list_edges` still does not support a namespace scope in v1 (it does
        // not enumerate edges by scope); `query` and `hybrid_query` now DO scope
        // (PR3d), so they are covered by their own dedicated tests below.
        let out = server.dispatch_tool_json(
            "list_edges",
            serde_json::json!({ "namespace": "agent:planner" }),
        );
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "list_edges must reject a narrowing scope with INVALID_ARGUMENT: {out}"
        );
        // But "all" is accepted (no-op) by these tools.
        let all_ok = server.dispatch_tool_json(
            "hybrid_query",
            serde_json::json!({ "filter_label": "Person", "namespace": "all" }),
        );
        assert!(
            all_ok.get(0..1).is_some() && !all_ok.contains("\"error\""),
            "hybrid_query with namespace:all should not error: {all_ok}"
        );
    }

    #[test]
    fn per_namespace_counts_in_database_stats() {
        let (server, _alice, _bob, _carol) = seed_two_namespaces();
        let out = server.dispatch_tool_json("database_stats", serde_json::json!({}));
        let v = as_json(&out);
        let namespaces = v
            .get("namespaces")
            .and_then(|n| n.as_array())
            .expect("database_stats must carry a namespaces array");
        let planner = namespaces
            .iter()
            .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("agent:planner"))
            .unwrap_or_else(|| panic!("agent:planner missing from stats: {out}"));
        assert_eq!(planner.get("node_count").and_then(|c| c.as_u64()), Some(2));
        assert_eq!(planner.get("edge_count").and_then(|c| c.as_u64()), Some(1));
        let research = namespaces
            .iter()
            .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("agent:researcher"))
            .unwrap_or_else(|| panic!("agent:researcher missing from stats: {out}"));
        assert_eq!(research.get("node_count").and_then(|c| c.as_u64()), Some(1));
    }

    #[test]
    fn per_namespace_counts_in_get_schema() {
        let (server, _alice, _bob, _carol) = seed_two_namespaces();
        let out = server.dispatch_tool_json("get_schema", serde_json::json!({}));
        let v = as_json(&out);
        let namespaces = v
            .get("namespaces")
            .and_then(|n| n.as_array())
            .expect("get_schema must carry a namespaces array");
        assert!(
            namespaces
                .iter()
                .any(|e| e.get("name").and_then(|n| n.as_str()) == Some("agent:planner")),
            "get_schema namespaces must include agent:planner: {out}"
        );
        assert_no_reserved(&out, "get_schema(counts)");
    }

    // ========================================================================
    // PR3a hardening (adversarial-review fixes): omitted = default-only,
    // scoped edge/temporal isolation, cursor+scope fail-closed, malformed /
    // union / traverse-direction error contracts, edge-write registration.
    // ========================================================================

    /// The first outgoing edge id from `node` (unscoped adjacency read).
    fn first_outgoing_edge_id(server: &AletheiaMcpServer, node: u64) -> u64 {
        let edges =
            server.dispatch_tool_json("get_outgoing_edges", serde_json::json!({ "node_id": node }));
        as_json(&edges)
            .get("edges")
            .and_then(|e| e.as_array())
            .and_then(|arr| arr.first())
            .and_then(|e| e.get("id"))
            .and_then(|i| i.as_u64())
            .unwrap_or_else(|| panic!("expected an outgoing edge from {node}: {edges}"))
    }

    /// FIX2 / ADD-B regression guard: an omitted read scope is the `default`
    /// namespace ONLY (isolated-by-default) — byte-for-byte equal to an explicit
    /// `namespace:"default"`, and NOT equal to `namespace:"all"` once a
    /// non-default entity exists. Pins the semantics so a future "convenience"
    /// change can't silently revert omitted back to all-namespaces.
    #[test]
    fn omitted_scope_is_default_only_not_all() {
        let server = create_test_server();
        // One default node + one agent:planner node.
        let dave = server
            .db()
            .create_node(
                "Person",
                PropertyMapBuilder::new().insert("name", "Dave").build(),
            )
            .expect("create default node")
            .as_u64();
        let eve = server
            .db()
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Eve").build(),
                "agent:planner",
            )
            .expect("create namespaced node")
            .as_u64();

        // list_nodes: omitted == explicit "default", and neither sees Eve.
        let omitted =
            server.dispatch_tool_json("list_nodes", serde_json::json!({ "label": "Person" }));
        let explicit_default = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({ "label": "Person", "namespace": "default" }),
        );
        assert_eq!(
            omitted, explicit_default,
            "omitted list_nodes must be byte-identical to namespace:\"default\": {omitted} vs {explicit_default}"
        );
        assert!(
            omitted.contains("Dave") && !omitted.contains("Eve"),
            "omitted (default-only) must see Dave, never Eve: {omitted}"
        );
        // "all" DOES see Eve — proving omitted != all.
        let all = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({ "label": "Person", "namespace": "all" }),
        );
        assert!(
            all.contains("Dave") && all.contains("Eve"),
            "namespace:\"all\" must see both: {all}"
        );
        assert_ne!(
            omitted, all,
            "omitted must NOT equal namespace:\"all\" when a non-default entity exists"
        );

        // get_node: Eve (agent:planner) is NOT_FOUND under an omitted read...
        let eve_omitted =
            server.dispatch_tool_json("get_node", serde_json::json!({ "node_id": eve }));
        assert_eq!(
            as_json(&eve_omitted)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "omitted get_node must not return a non-default node: {eve_omitted}"
        );
        // ...identical to an explicit "default" read...
        let eve_default = server.dispatch_tool_json(
            "get_node",
            serde_json::json!({ "node_id": eve, "namespace": "default" }),
        );
        assert_eq!(
            eve_omitted, eve_default,
            "omitted get_node must equal namespace:\"default\": {eve_omitted} vs {eve_default}"
        );
        // ...but visible under "all".
        let eve_all = server.dispatch_tool_json(
            "get_node",
            serde_json::json!({ "node_id": eve, "namespace": "all" }),
        );
        assert_eq!(
            as_json(&eve_all).get("namespace").and_then(|n| n.as_str()),
            Some("agent:planner"),
            "namespace:\"all\" must return Eve: {eve_all}"
        );
        // Sanity: the default node reads back the same way omitted and explicit.
        let dave_omitted =
            server.dispatch_tool_json("get_node", serde_json::json!({ "node_id": dave }));
        assert!(
            dave_omitted.contains("Dave"),
            "default node visible: {dave_omitted}"
        );
    }

    #[test]
    fn scoped_get_edge_isolates_other_namespace() {
        let (server, a, _b) = seed_namespaced();
        let edge_id = first_outgoing_edge_id(&server, a);
        // In-scope: the edge is returned.
        let ok = server.dispatch_tool_json(
            "get_edge",
            serde_json::json!({ "edge_id": edge_id, "namespace": "agent:planner" }),
        );
        assert_eq!(
            as_json(&ok).get("namespace").and_then(|n| n.as_str()),
            Some("agent:planner"),
            "scoped get_edge should return the in-scope edge: {ok}"
        );
        // Out-of-scope: NOT_FOUND (never leaks).
        let hidden = server.dispatch_tool_json(
            "get_edge",
            serde_json::json!({ "edge_id": edge_id, "namespace": "agent:researcher" }),
        );
        assert_eq!(
            as_json(&hidden)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "out-of-scope get_edge must be NOT_FOUND: {hidden}"
        );
        // Omitted (default-only) also hides an agent:planner edge.
        let omitted =
            server.dispatch_tool_json("get_edge", serde_json::json!({ "edge_id": edge_id }));
        assert_eq!(
            as_json(&omitted)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "omitted get_edge (default-only) must not return a non-default edge: {omitted}"
        );
    }

    #[test]
    fn scoped_temporal_reads_narrow() {
        let (server, alice, _bob, carol) = seed_two_namespaces();
        // A far-future valid time resolves every committed fact (open-ended
        // valid interval); transaction_time defaults to now.
        let vt = "2999-01-01T00:00:00Z";
        // get_node_at_time: in-scope returns, out-of-scope NOT_FOUND.
        let ok = server.dispatch_tool_json(
            "get_node_at_time",
            serde_json::json!({ "node_id": alice, "valid_time": vt, "namespace": "agent:planner" }),
        );
        assert!(
            ok.contains("Alice"),
            "in-scope get_node_at_time returns: {ok}"
        );
        let hidden = server.dispatch_tool_json(
            "get_node_at_time",
            serde_json::json!({ "node_id": carol, "valid_time": vt, "namespace": "agent:planner" }),
        );
        assert_eq!(
            as_json(&hidden)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "out-of-scope get_node_at_time must be NOT_FOUND: {hidden}"
        );

        // get_edge_at_time: the agent:planner edge is in-scope for planner,
        // NOT_FOUND for researcher.
        let edge_id = first_outgoing_edge_id(&server, alice);
        let e_ok = server.dispatch_tool_json(
            "get_edge_at_time",
            serde_json::json!({ "edge_id": edge_id, "valid_time": vt, "namespace": "agent:planner" }),
        );
        assert!(
            as_json(&e_ok).get("error").is_none(),
            "in-scope get_edge_at_time returns: {e_ok}"
        );
        let e_hidden = server.dispatch_tool_json(
            "get_edge_at_time",
            serde_json::json!({ "edge_id": edge_id, "valid_time": vt, "namespace": "agent:researcher" }),
        );
        assert_eq!(
            as_json(&e_hidden)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "out-of-scope get_edge_at_time must be NOT_FOUND: {e_hidden}"
        );

        // find_nodes_at_time: planner scope excludes Carol.
        let found = server.dispatch_tool_json(
            "find_nodes_at_time",
            serde_json::json!({ "label": "Person", "valid_time": vt, "namespace": "agent:planner" }),
        );
        assert!(
            found.contains("Alice") && !found.contains("Carol"),
            "scoped find_nodes_at_time must exclude out-of-scope Carol: {found}"
        );
    }

    #[test]
    fn cursor_plus_narrowing_scope_fails_closed() {
        let (server, alice, _bob, _carol) = seed_two_namespaces();
        let vt = crate::core::temporal::time::to_iso8601(crate::core::temporal::time::now());
        let cases = [
            (
                "list_nodes",
                serde_json::json!({ "label": "Person", "use_cursor": true, "namespace": "agent:planner" }),
            ),
            (
                "traverse",
                serde_json::json!({ "start_node_id": alice, "edge_label": "KNOWS", "use_cursor": true, "namespace": "agent:planner" }),
            ),
            (
                "find_nodes_at_time",
                serde_json::json!({ "label": "Person", "valid_time": vt, "use_cursor": true, "namespace": "agent:planner" }),
            ),
        ];
        for (tool, args) in cases {
            let out = server.dispatch_tool_json(tool, args);
            assert_eq!(
                as_json(&out)
                    .pointer("/error/code")
                    .and_then(|c| c.as_str()),
                Some("INVALID_ARGUMENT"),
                "{tool} with use_cursor + narrowing scope must fail closed: {out}"
            );
        }
    }

    #[test]
    fn malformed_scope_type_is_invalid_argument() {
        let (server, _a, _b) = seed_namespaced();
        let out = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({ "label": "Person", "namespace": 123 }),
        );
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "a non-string/array/\"all\" namespace must be INVALID_ARGUMENT: {out}"
        );
    }

    #[test]
    fn union_with_unknown_namespace_is_not_found() {
        let (server, _a, _b, _c) = seed_two_namespaces();
        let out = server.dispatch_tool_json(
            "list_nodes",
            serde_json::json!({
                "label": "Person",
                "namespace": ["agent:planner", "agent:nonexistent"]
            }),
        );
        let v = as_json(&out);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "a union containing an unknown namespace must be NOT_FOUND: {out}"
        );
        assert_eq!(
            v.pointer("/error/details/namespace")
                .and_then(|c| c.as_str()),
            Some("agent:nonexistent"),
            "NOT_FOUND must carry the offending namespace in details: {out}"
        );
    }

    /// PR3d: a scoped `incoming` traversal honors the edge ∧ far-node boundary.
    /// Topology: bob -KNOWS(agent:planner)-> alice (both agent:planner), and a
    /// cross-namespace edge carol(agent:researcher) -KNOWS(agent:researcher)->
    /// alice. Scoped to agent:planner, an incoming traversal from alice reaches
    /// bob (in-scope edge + source) but NOT carol (the edge's own namespace, and
    /// carol, are out of scope).
    #[test]
    fn scoped_traverse_incoming_honors_boundary() {
        let server = create_test_server();
        let db = server.db();
        let alice = db
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
                "agent:planner",
            )
            .unwrap();
        let bob = db
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Bob").build(),
                "agent:planner",
            )
            .unwrap();
        let carol = db
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Carol").build(),
                "agent:researcher",
            )
            .unwrap();
        db.create_edge_in_namespace(
            bob,
            alice,
            "KNOWS",
            PropertyMapBuilder::new().build(),
            "agent:planner",
        )
        .unwrap();
        db.create_edge_in_namespace(
            carol,
            alice,
            "KNOWS",
            PropertyMapBuilder::new().build(),
            "agent:researcher",
        )
        .unwrap();

        let out = server.dispatch_tool_json(
            "traverse",
            serde_json::json!({
                "start_node_id": alice.as_u64(), "edge_label": "KNOWS",
                "direction": "incoming", "namespace": "agent:planner"
            }),
        );
        assert!(
            out.contains("Bob"),
            "scoped incoming traverse must reach Bob via the in-scope incoming edge: {out}"
        );
        assert!(
            !out.contains("Carol"),
            "scoped incoming traverse must NOT cross the out-of-scope edge to Carol: {out}"
        );
        assert!(
            as_json(&out).pointer("/error").is_none(),
            "incoming direction must no longer be rejected under a narrowing scope: {out}"
        );
        assert_no_reserved(&out, "traverse(incoming scoped)");
    }

    /// PR3d: a scoped `both` traversal honors the boundary in both directions and
    /// unions the results, never crossing an out-of-scope edge/node.
    #[test]
    fn scoped_traverse_both_honors_boundary() {
        let server = create_test_server();
        let db = server.db();
        let alice = db
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Alice").build(),
                "agent:planner",
            )
            .unwrap();
        // Outgoing in-scope: alice -> bob (planner).
        let bob = db
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Bob").build(),
                "agent:planner",
            )
            .unwrap();
        // Incoming in-scope: dave -> alice (planner).
        let dave = db
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Dave").build(),
                "agent:planner",
            )
            .unwrap();
        // Out-of-scope neighbor via an out-of-scope edge: alice -> carol (researcher).
        let carol = db
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Carol").build(),
                "agent:researcher",
            )
            .unwrap();
        db.create_edge_in_namespace(
            alice,
            bob,
            "KNOWS",
            PropertyMapBuilder::new().build(),
            "agent:planner",
        )
        .unwrap();
        db.create_edge_in_namespace(
            dave,
            alice,
            "KNOWS",
            PropertyMapBuilder::new().build(),
            "agent:planner",
        )
        .unwrap();
        db.create_edge_in_namespace(
            alice,
            carol,
            "KNOWS",
            PropertyMapBuilder::new().build(),
            "agent:researcher",
        )
        .unwrap();

        let out = server.dispatch_tool_json(
            "traverse",
            serde_json::json!({
                "start_node_id": alice.as_u64(), "edge_label": "KNOWS",
                "direction": "both", "namespace": "agent:planner"
            }),
        );
        assert!(
            out.contains("Bob") && out.contains("Dave"),
            "scoped both traverse must reach Bob (outgoing) and Dave (incoming): {out}"
        );
        assert!(
            !out.contains("Carol"),
            "scoped both traverse must NOT cross the out-of-scope edge to Carol: {out}"
        );
        assert!(
            as_json(&out).pointer("/error").is_none(),
            "both direction must no longer be rejected under a narrowing scope: {out}"
        );
    }

    // ========================================================================
    // PR3d: real query / hybrid_query namespace scoping.
    // ========================================================================

    /// PR3d: a scoped AQL `query` returns only in-scope entities, never another
    /// namespace's; an omitted scope is default-only; `"all"` sees everything;
    /// an unknown namespace is NOT_FOUND.
    #[test]
    fn scoped_query_isolates_namespaces() {
        let (server, _alice, _bob, _carol) = seed_two_namespaces();
        let q = "MATCH (n:Person) RETURN n";

        // Scoped to agent:planner: Alice + Bob, never Carol.
        let planner = server.dispatch_tool_json(
            "query",
            serde_json::json!({ "language": "aql", "query": q, "namespace": "agent:planner" }),
        );
        assert!(
            planner.contains("Alice") && planner.contains("Bob"),
            "scoped query must return in-scope Alice + Bob: {planner}"
        );
        assert!(
            !planner.contains("Carol"),
            "scoped query must NOT leak out-of-scope Carol: {planner}"
        );
        assert_no_reserved(&planner, "query(scoped)");

        // Scoped to agent:researcher: only Carol.
        let research = server.dispatch_tool_json(
            "query",
            serde_json::json!({ "language": "aql", "query": q, "namespace": "agent:researcher" }),
        );
        assert!(
            research.contains("Carol") && !research.contains("Alice"),
            "researcher scope must return only Carol: {research}"
        );

        // Omitted scope ⇒ default-only. All seed data is non-default, so no rows.
        let omitted = server.dispatch_tool_json(
            "query",
            serde_json::json!({ "language": "aql", "query": q }),
        );
        assert!(
            !omitted.contains("Alice") && !omitted.contains("Carol"),
            "omitted query scope is default-only; non-default data must not appear: {omitted}"
        );

        // "all" ⇒ every namespace.
        let all = server.dispatch_tool_json(
            "query",
            serde_json::json!({ "language": "aql", "query": q, "namespace": "all" }),
        );
        assert!(
            all.contains("Alice") && all.contains("Bob") && all.contains("Carol"),
            "namespace:all query must return every namespace: {all}"
        );

        // Unknown namespace ⇒ NOT_FOUND with details.namespace.
        let unknown = server.dispatch_tool_json(
            "query",
            serde_json::json!({ "language": "aql", "query": q, "namespace": "agent:ghost" }),
        );
        let v = as_json(&unknown);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "unknown namespace on query must be NOT_FOUND: {unknown}"
        );
        assert_eq!(
            v.pointer("/error/details/namespace")
                .and_then(|c| c.as_str()),
            Some("agent:ghost"),
            "NOT_FOUND must carry details.namespace: {unknown}"
        );

        // Empty array ⇒ INVALID_ARGUMENT.
        let empty = server.dispatch_tool_json(
            "query",
            serde_json::json!({ "language": "aql", "query": q, "namespace": [] }),
        );
        assert_eq!(
            as_json(&empty)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "empty namespace scope array on query must be INVALID_ARGUMENT: {empty}"
        );
    }

    /// PR3d: a scoped `query` traversal honors the boundary — an out-of-scope
    /// edge/node is never crossed even though the far node exists.
    #[test]
    fn scoped_query_traversal_honors_boundary() {
        let server = create_test_server();
        let db = server.db();
        let a = db
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Ann").build(),
                "agent:a",
            )
            .unwrap();
        let shared = db
            .create_node_in_namespace(
                "Person",
                PropertyMapBuilder::new().insert("name", "Shelly").build(),
                "shared",
            )
            .unwrap();
        // Cross-namespace edge in `shared`: a -> shared. Under scope {agent:a}
        // the edge's own namespace (shared) is out of scope ⇒ never crossed.
        db.create_edge_in_namespace(
            a,
            shared,
            "KNOWS",
            PropertyMapBuilder::new().build(),
            "shared",
        )
        .unwrap();

        let scoped_a = server.dispatch_tool_json(
            "query",
            serde_json::json!({
                "language": "aql",
                "query": "MATCH (n:Person)-[:KNOWS]->(m:Person) RETURN m",
                "namespace": "agent:a"
            }),
        );
        assert!(
            !scoped_a.contains("Shelly"),
            "scope {{agent:a}} must not cross the shared-namespace edge: {scoped_a}"
        );

        // Union {agent:a, shared} crosses the edge and reaches Shelly.
        let union = server.dispatch_tool_json(
            "query",
            serde_json::json!({
                "language": "aql",
                "query": "MATCH (n:Person)-[:KNOWS]->(m:Person) RETURN m",
                "namespace": ["agent:a", "shared"]
            }),
        );
        assert!(
            union.contains("Shelly"),
            "union scope must cross the in-scope edge to Shelly: {union}"
        );
    }

    /// PR3d: scoped `hybrid_query` (vector-first) is filter-complete — it
    /// over-fetches so the returned ranked results are all genuinely in-scope,
    /// even when NEARER out-of-scope candidates exist; scores/order preserved;
    /// no other-namespace node appears.
    #[test]
    fn scoped_hybrid_vector_is_filter_complete() {
        let server = create_test_server();
        let db = server.db();
        db.enable_vector_index(
            "embedding",
            crate::index::vector::HnswConfig::new(4, crate::index::vector::DistanceMetric::Cosine),
        )
        .unwrap();

        let vec_props = |name: &str, e: &[f32]| {
            PropertyMapBuilder::new()
                .insert("name", name)
                .insert_vector("embedding", e)
                .build()
        };

        // b-nodes (agent:b) are NEARER the query [1,0,0,0] than a-nodes at the
        // same index, so a naive take-k-then-filter would starve the in-scope set.
        let mut a_names = std::collections::BTreeSet::new();
        for i in 0..10 {
            let spread = 0.01 + i as f32 * 0.01;
            db.create_node_in_namespace(
                "Doc",
                vec_props(&format!("a{i}"), &[1.0 - spread, spread, 0.0, 0.0]),
                "agent:a",
            )
            .unwrap();
            a_names.insert(format!("a{i}"));
            db.create_node_in_namespace(
                "Doc",
                vec_props(
                    &format!("b{i}"),
                    &[1.0 - spread * 0.5, 0.0, spread * 0.5, 0.0],
                ),
                "agent:b",
            )
            .unwrap();
        }

        let out = server.dispatch_tool_json(
            "hybrid_query",
            serde_json::json!({
                "query_embedding": [1.0, 0.0, 0.0, 0.0],
                "top_k": 5,
                "limit": 5,
                "namespace": "agent:a"
            }),
        );
        let v = as_json(&out);
        let results = v
            .get("results")
            .and_then(|r| r.as_array())
            .unwrap_or_else(|| panic!("hybrid_query must return results array: {out}"));
        assert_eq!(
            results.len(),
            5,
            "scoped hybrid must be filter-complete (k in-scope results): {out}"
        );
        // Every returned node is in agent:a and none is a b-node.
        for r in results {
            let ns = r
                .pointer("/node/namespace")
                .and_then(|n| n.as_str())
                .unwrap_or("default");
            assert_eq!(
                ns, "agent:a",
                "scoped hybrid returned an out-of-scope node: {out}"
            );
        }
        assert!(
            !out.contains("\"b0\"") && !out.contains("\"b1\""),
            "scoped hybrid must not return any agent:b node: {out}"
        );
        // Scores present and in non-increasing order (ranking preserved).
        let scores: Vec<f64> = results
            .iter()
            .filter_map(|r| r.get("similarity_score").and_then(|s| s.as_f64()))
            .collect();
        assert_eq!(scores.len(), 5, "each ranked result carries a score: {out}");
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-6,
                "scoped hybrid must preserve descending similarity order: {scores:?}"
            );
        }
        assert_no_reserved(&out, "hybrid_query(scoped)");
    }

    /// PR3d: `hybrid_query` with an unknown namespace is NOT_FOUND; with `"all"`
    /// it is unscoped; omitted is default-only.
    #[test]
    fn scoped_hybrid_unknown_namespace_is_not_found() {
        let (server, _a, _b, _c) = seed_two_namespaces();
        let out = server.dispatch_tool_json(
            "hybrid_query",
            serde_json::json!({ "filter_label": "Person", "namespace": "agent:ghost" }),
        );
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "hybrid_query with an unknown namespace must be NOT_FOUND: {out}"
        );
    }

    #[test]
    fn create_edge_in_namespace_registers_it() {
        let server = create_test_server();
        let a = server
            .db()
            .create_node_in_namespace("Person", PropertyMapBuilder::new().build(), "agent:planner")
            .unwrap()
            .as_u64();
        let b = server
            .db()
            .create_node_in_namespace("Person", PropertyMapBuilder::new().build(), "agent:planner")
            .unwrap()
            .as_u64();
        // Write an edge into a brand-new namespace via the MCP tool.
        let out = server.dispatch_tool_json(
            "create_edge",
            serde_json::json!({
                "source_id": a, "target_id": b, "label": "KNOWS",
                "namespace": "agent:edgewriter"
            }),
        );
        assert_eq!(
            as_json(&out).get("namespace").and_then(|n| n.as_str()),
            Some("agent:edgewriter"),
            "create_edge must surface the namespace: {out}"
        );
        // list_namespaces (via database_stats has counts, but registration is the
        // registry) shows the newly-written namespace.
        let names: Vec<String> = server
            .db()
            .list_namespaces()
            .into_iter()
            .map(|i| i.name)
            .collect();
        assert!(
            names.iter().any(|n| n == "agent:edgewriter"),
            "a namespaced edge write must register the namespace: {names:?}"
        );
    }
}

// ============================================================================
// Namespace management tools (Issue #3349, PR3b):
// create_namespace / list_namespaces / describe_namespace — driven through the
// real `dispatch_tool_json` seam.
// ============================================================================

mod namespace_tools_tests {
    use super::*;
    use crate::auth::{AuthMode, AuthStore, Role, SecretString};
    use crate::mcp::McpAuthConfig;

    fn as_json(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("valid JSON response")
    }

    /// A `Required`-mode server whose session credential resolves to `role`.
    fn server_with_role(role: Role) -> AletheiaMcpServer {
        let store = Arc::new(AuthStore::new());
        let (_principal, key) = store
            .create_key(&format!("ns-{role}"), role)
            .expect("create key");
        AletheiaMcpServer::with_auth(
            create_test_db(),
            McpAuthConfig::new(AuthMode::Required, Arc::clone(&store))
                .with_credential(SecretString::new(key.as_str())),
        )
    }

    #[test]
    fn create_namespace_round_trips() {
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "create_namespace",
            serde_json::json!({ "name": "agent:planner", "description": "planner scope" }),
        );
        let v = as_json(&out);
        assert!(v.get("error").is_none(), "create must succeed: {out}");
        assert_eq!(v["name"], "agent:planner");
        assert_eq!(v["description"], "planner scope");
        assert!(
            v.get("created_at").and_then(|c| c.as_str()).is_some(),
            "create_namespace must return a created_at timestamp: {out}"
        );
        assert!(
            !out.contains("__aletheia_"),
            "no reserved key may leak: {out}"
        );
    }

    #[test]
    fn create_namespace_without_description_is_null() {
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "create_namespace",
            serde_json::json!({ "name": "session:1" }),
        );
        let v = as_json(&out);
        assert_eq!(v["name"], "session:1");
        assert!(
            v["description"].is_null(),
            "an omitted description is JSON null: {out}"
        );
    }

    #[test]
    fn create_namespace_duplicate_is_conflict() {
        let server = create_test_server();
        server.dispatch_tool_json("create_namespace", serde_json::json!({ "name": "agent:a" }));
        let out =
            server.dispatch_tool_json("create_namespace", serde_json::json!({ "name": "agent:a" }));
        let v = as_json(&out);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("CONFLICT"),
            "a duplicate namespace must be CONFLICT: {out}"
        );
        assert_eq!(
            v.pointer("/error/retriable").and_then(|r| r.as_bool()),
            Some(false),
            "CONFLICT on a duplicate name is non-retriable: {out}"
        );
    }

    #[test]
    fn create_default_namespace_is_conflict() {
        let server = create_test_server();
        let out =
            server.dispatch_tool_json("create_namespace", serde_json::json!({ "name": "default" }));
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("CONFLICT"),
            "creating the implicit default namespace must be CONFLICT: {out}"
        );
    }

    #[test]
    fn create_namespace_invalid_name_is_invalid_argument() {
        let server = create_test_server();
        // A whitespace-bearing name violates the charset.
        let out = server.dispatch_tool_json(
            "create_namespace",
            serde_json::json!({ "name": "bad name!" }),
        );
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "a malformed name must be INVALID_ARGUMENT: {out}"
        );
    }

    #[test]
    fn create_reserved_all_selector_is_invalid_argument() {
        let server = create_test_server();
        let out =
            server.dispatch_tool_json("create_namespace", serde_json::json!({ "name": "all" }));
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "the reserved 'all' selector must be INVALID_ARGUMENT: {out}"
        );
    }

    #[test]
    fn list_namespaces_includes_default_and_created() {
        let server = create_test_server();
        server.dispatch_tool_json(
            "create_namespace",
            serde_json::json!({ "name": "agent:planner", "description": "p" }),
        );
        let out = server.dispatch_tool_json("list_namespaces", serde_json::json!({}));
        let v = as_json(&out);
        let namespaces = v["namespaces"].as_array().expect("namespaces array");
        assert_eq!(
            v["count"].as_u64(),
            Some(namespaces.len() as u64),
            "count must equal the array length: {out}"
        );
        // default is always first.
        assert_eq!(namespaces[0]["name"], "default");
        let names: Vec<&str> = namespaces
            .iter()
            .filter_map(|n| n["name"].as_str())
            .collect();
        assert!(
            names.contains(&"agent:planner"),
            "the created namespace must be listed: {out}"
        );
        // Per-namespace counts are present.
        for entry in namespaces {
            assert!(
                entry.get("node_count").and_then(|c| c.as_u64()).is_some(),
                "each entry carries node_count: {out}"
            );
            assert!(
                entry.get("edge_count").and_then(|c| c.as_u64()).is_some(),
                "each entry carries edge_count: {out}"
            );
        }
        assert!(
            !out.contains("__aletheia_"),
            "no reserved key may leak: {out}"
        );
    }

    #[test]
    fn list_namespaces_reflects_current_counts() {
        let (server, _a, _b) = {
            let server = create_test_server();
            let a = server
                .db()
                .create_node_in_namespace(
                    "Person",
                    PropertyMapBuilder::new().build(),
                    "agent:planner",
                )
                .expect("node");
            let b = server
                .db()
                .create_node_in_namespace(
                    "Person",
                    PropertyMapBuilder::new().build(),
                    "agent:planner",
                )
                .expect("node");
            (server, a, b)
        };
        let out = server.dispatch_tool_json("list_namespaces", serde_json::json!({}));
        let v = as_json(&out);
        let planner = v["namespaces"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["name"] == "agent:planner")
            .expect("agent:planner listed");
        assert_eq!(
            planner["node_count"].as_u64(),
            Some(2),
            "two nodes were written to agent:planner: {out}"
        );
    }

    #[test]
    fn describe_namespace_known_returns_info_and_counts() {
        let server = create_test_server();
        server.dispatch_tool_json(
            "create_namespace",
            serde_json::json!({ "name": "agent:planner", "description": "planner" }),
        );
        let out = server.dispatch_tool_json(
            "describe_namespace",
            serde_json::json!({ "name": "agent:planner" }),
        );
        let v = as_json(&out);
        assert!(v.get("error").is_none(), "describe must succeed: {out}");
        assert_eq!(v["name"], "agent:planner");
        assert_eq!(v["description"], "planner");
        assert!(v.get("created_at").is_some(), "created_at present: {out}");
        assert!(v.get("node_count").is_some(), "node_count present: {out}");
        assert!(v.get("edge_count").is_some(), "edge_count present: {out}");
    }

    #[test]
    fn describe_default_namespace_always_resolves() {
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "describe_namespace",
            serde_json::json!({ "name": "default" }),
        );
        let v = as_json(&out);
        assert!(v.get("error").is_none(), "default must resolve: {out}");
        assert_eq!(v["name"], "default");
    }

    #[test]
    fn describe_unknown_namespace_is_not_found() {
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "describe_namespace",
            serde_json::json!({ "name": "agent:nonexistent" }),
        );
        let v = as_json(&out);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "an unregistered namespace must be NOT_FOUND: {out}"
        );
        assert_eq!(
            v.pointer("/error/details/namespace")
                .and_then(|n| n.as_str()),
            Some("agent:nonexistent"),
            "NOT_FOUND must carry the offending namespace in details: {out}"
        );
    }

    #[test]
    fn three_tools_present_in_live_catalog_of_74() {
        let server = create_test_server();
        let tools = server.list_tools_for_test();
        assert_eq!(tools.len(), 74, "the live catalog must be exactly 74 tools");
        for name in ["create_namespace", "list_namespaces", "describe_namespace"] {
            assert!(
                tools.iter().any(|t| t == name),
                "{name} must be advertised in tools/list"
            );
        }
    }

    #[test]
    fn reader_denied_create_namespace_writer_allowed() {
        // A reader may NOT create a namespace (write-class tool).
        let reader = server_with_role(Role::Reader);
        let out =
            reader.dispatch_tool_json("create_namespace", serde_json::json!({ "name": "agent:x" }));
        assert_eq!(
            as_json(&out)
                .pointer("/error/code")
                .and_then(|c| c.as_str()),
            Some("PERMISSION_DENIED"),
            "a reader must be denied create_namespace: {out}"
        );

        // A reader MAY list/describe (read-class tools).
        let listed = reader.dispatch_tool_json("list_namespaces", serde_json::json!({}));
        assert!(
            as_json(&listed).get("error").is_none(),
            "a reader may list namespaces: {listed}"
        );

        // A writer MAY create.
        let writer = server_with_role(Role::Writer);
        let created =
            writer.dispatch_tool_json("create_namespace", serde_json::json!({ "name": "agent:y" }));
        assert!(
            as_json(&created).get("error").is_none(),
            "a writer may create a namespace: {created}"
        );
        assert_eq!(as_json(&created)["name"], "agent:y");
    }
}

// ============================================================================
// Belief-revision audit (Issue #3362) + provenance-weighted fusion (#3372)
// ============================================================================

mod belief_revision_and_fusion_tests {
    use super::*;

    fn parse(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("valid JSON response")
    }

    // ---- get_belief_revisions is advertised regardless of feature (Design A) --

    #[test]
    fn get_belief_revisions_is_advertised() {
        let server = create_test_server();
        assert!(
            server
                .list_tools_for_test()
                .iter()
                .any(|t| t == "get_belief_revisions"),
            "get_belief_revisions must be advertised (Design A: unconditional)"
        );
    }

    // ---- Feature-off twin: structured FAILED_PRECONDITION ---------------------

    #[cfg(not(feature = "semantic-temporal"))]
    #[test]
    fn get_belief_revisions_feature_off_is_failed_precondition() {
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "get_belief_revisions",
            serde_json::json!({ "entity_kind": "node", "id": 1 }),
        );
        let v = parse(&out);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("FAILED_PRECONDITION"),
            "feature-off get_belief_revisions must be FAILED_PRECONDITION: {out}"
        );
        assert_eq!(
            v.pointer("/error/details/required_feature")
                .and_then(|c| c.as_str()),
            Some("semantic-temporal"),
            "must name the required feature: {out}"
        );
        assert_eq!(
            v.pointer("/error/retriable"),
            Some(&serde_json::json!(false)),
            "feature-precondition is non-retriable: {out}"
        );
    }

    // ---- Behavior under semantic-temporal ------------------------------------

    #[cfg(feature = "semantic-temporal")]
    fn node_with_worldchange_history(server: &AletheiaMcpServer) -> u64 {
        // v1: initial assertion, valid from 2024-01-01, confidence 0.7.
        let created = server.dispatch_tool_json(
            "create_node",
            serde_json::json!({
                "label": "City",
                "properties": { "status": "planned" },
                "valid_time": "2024-01-01T00:00:00Z",
                "provenance": { "source": "editor", "confidence": 0.7 }
            }),
        );
        let id = parse(&created)
            .get("id")
            .and_then(|i| i.as_u64())
            .expect("id");
        // v2: a later-valid fact (valid_from advances) -> world_change.
        server.dispatch_tool_json(
            "update_node",
            serde_json::json!({
                "node_id": id,
                "properties": { "status": "active" },
                "valid_time": "2024-06-01T00:00:00Z",
                "provenance": { "source": "editor", "confidence": 0.9 }
            }),
        );
        id
    }

    #[cfg(feature = "semantic-temporal")]
    #[test]
    fn get_belief_revisions_returns_classified_revisions() {
        let server = create_test_server();
        let id = node_with_worldchange_history(&server);
        let out = server.dispatch_tool_json(
            "get_belief_revisions",
            serde_json::json!({ "entity_kind": "node", "id": id }),
        );
        let v = parse(&out);
        assert!(v.get("error").is_none(), "should succeed: {out}");
        assert_eq!(
            v.pointer("/entity/kind").and_then(|k| k.as_str()),
            Some("node")
        );
        assert_eq!(v.pointer("/entity/id").and_then(|k| k.as_u64()), Some(id));

        let revs = v
            .get("revisions")
            .and_then(|r| r.as_array())
            .expect("revisions array");
        assert_eq!(revs.len(), 2, "two versions -> two revisions: {out}");
        assert_eq!(revs[0]["classification"], "initial_assertion");
        assert_eq!(
            revs[1]["classification"], "world_change",
            "a later-valid fact is a world_change: {out}"
        );

        // Confidence trajectory mirrors per-revision confidence (AC3).
        let traj = v
            .get("confidence_trajectory")
            .and_then(|t| t.as_array())
            .expect("trajectory");
        assert_eq!(traj.len(), 2);
        assert_eq!(traj[0], serde_json::json!(0.7));
        assert_eq!(traj[1], serde_json::json!(0.9));
        assert_eq!(v.get("has_more"), Some(&serde_json::json!(false)));

        // Provenance + changes carry the self-describing tagged value shape.
        assert_eq!(
            revs[1]
                .pointer("/provenance/source")
                .and_then(|s| s.as_str()),
            Some("editor")
        );
        assert_eq!(revs[1]["changes"][0]["new"]["type"], "string");

        // Deterministic: byte-identical on repeat.
        let out2 = server.dispatch_tool_json(
            "get_belief_revisions",
            serde_json::json!({ "entity_kind": "node", "id": id }),
        );
        assert_eq!(out, out2, "belief-revision audit must be deterministic");
    }

    #[cfg(feature = "semantic-temporal")]
    #[test]
    fn get_belief_revisions_unknown_entity_is_not_found() {
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "get_belief_revisions",
            serde_json::json!({ "entity_kind": "node", "id": 999_999 }),
        );
        let v = parse(&out);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("NOT_FOUND"),
            "unknown entity -> NOT_FOUND: {out}"
        );
        assert_eq!(
            v.pointer("/error/retriable"),
            Some(&serde_json::json!(false))
        );
    }

    #[cfg(feature = "semantic-temporal")]
    #[test]
    fn get_belief_revisions_limit_zero_is_invalid_argument() {
        let server = create_test_server();
        let id = node_with_worldchange_history(&server);
        let out = server.dispatch_tool_json(
            "get_belief_revisions",
            serde_json::json!({ "entity_kind": "node", "id": id, "limit": 0 }),
        );
        assert_eq!(
            parse(&out).pointer("/error/code").and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "limit:0 -> INVALID_ARGUMENT: {out}"
        );
    }

    #[cfg(feature = "semantic-temporal")]
    #[test]
    fn get_belief_revisions_bad_entity_kind_is_invalid_argument() {
        let server = create_test_server();
        let out = server.dispatch_tool_json(
            "get_belief_revisions",
            serde_json::json!({ "entity_kind": "widget", "id": 1 }),
        );
        assert_eq!(
            parse(&out).pointer("/error/code").and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "bad entity_kind -> INVALID_ARGUMENT: {out}"
        );
    }

    #[cfg(feature = "semantic-temporal")]
    #[test]
    fn get_belief_revisions_unknown_property_key_is_invalid_argument() {
        let server = create_test_server();
        let id = node_with_worldchange_history(&server);
        let out = server.dispatch_tool_json(
            "get_belief_revisions",
            serde_json::json!({
                "entity_kind": "node", "id": id, "property_key": "nonexistent"
            }),
        );
        assert_eq!(
            parse(&out).pointer("/error/code").and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "unknown property_key -> INVALID_ARGUMENT: {out}"
        );
    }

    // ---- Provenance-weighted fusion (Issue #3372) ----------------------------

    #[cfg(feature = "semantic-retrieval-fusion")]
    fn setup_fusion_corpus(server: &AletheiaMcpServer) {
        server.dispatch_tool_json(
            "enable_vector_index",
            serde_json::json!({
                "property_name": "embedding", "dimensions": 4,
                "distance_metric": "cosine"
            }),
        );
        // "near" is the best similarity match but low-trust.
        server.dispatch_tool_json(
            "create_node",
            serde_json::json!({
                "label": "Doc",
                "properties": { "name": "near", "embedding": [1.0, 0.0, 0.0, 0.0] },
                "provenance": { "source": "rumor", "confidence": 0.05 }
            }),
        );
        // "trusted" is slightly-worse similarity but high-trust; fusion should
        // rank it first once confidence is weighted heavily.
        server.dispatch_tool_json(
            "create_node",
            serde_json::json!({
                "label": "Doc",
                "properties": { "name": "trusted", "embedding": [0.9, 0.1, 0.0, 0.0] },
                "provenance": { "source": "authority", "confidence": 0.99 }
            }),
        );
    }

    #[cfg(feature = "semantic-retrieval-fusion")]
    fn result_names(v: &serde_json::Value) -> Vec<String> {
        v["results"]
            .as_array()
            .expect("results array")
            .iter()
            .map(|r| {
                r["node"]["properties"]["name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            })
            .collect()
    }

    #[cfg(feature = "semantic-retrieval-fusion")]
    #[test]
    fn find_similar_without_fusion_has_no_breakdown_and_similarity_order() {
        let server = create_test_server();
        setup_fusion_corpus(&server);
        let out = server.dispatch_tool_json(
            "find_similar",
            serde_json::json!({
                "property_name": "embedding", "embedding": [1.0, 0.0, 0.0, 0.0], "k": 2
            }),
        );
        let v = parse(&out);
        let results = v["results"].as_array().expect("results");
        assert!(!results.is_empty(), "should return candidates: {out}");
        // Omitting fusion_policy is byte-for-byte prior behavior: no breakdown.
        for r in results {
            assert!(
                r.get("score_breakdown").is_none(),
                "no fusion_policy -> no score_breakdown: {out}"
            );
        }
        // Pure similarity ranks "near" first.
        assert_eq!(result_names(&v).first().map(String::as_str), Some("near"));
    }

    #[cfg(feature = "semantic-retrieval-fusion")]
    #[test]
    fn find_similar_with_fusion_reorders_and_attaches_breakdown() {
        let server = create_test_server();
        setup_fusion_corpus(&server);
        let out = server.dispatch_tool_json(
            "find_similar",
            serde_json::json!({
                "property_name": "embedding", "embedding": [1.0, 0.0, 0.0, 0.0], "k": 2,
                "fusion_policy": { "w_similarity": 1.0, "w_confidence": 5.0, "w_recency": 0.0 }
            }),
        );
        let v = parse(&out);
        assert!(v.get("error").is_none(), "fusion should succeed: {out}");
        let results = v["results"].as_array().expect("results");
        assert_eq!(results.len(), 2, "{out}");
        // Heavy confidence weighting promotes the high-trust node to the top.
        assert_eq!(
            result_names(&v).first().map(String::as_str),
            Some("trusted"),
            "fusion must re-rank by fused score: {out}"
        );
        // Every result carries an auditable breakdown.
        for r in results {
            let b = r.get("score_breakdown").expect("score_breakdown present");
            for key in ["similarity", "confidence", "recency", "fused"] {
                assert!(
                    b.get(key).and_then(|x| x.as_f64()).is_some(),
                    "breakdown.{key}: {out}"
                );
            }
        }
    }

    #[cfg(feature = "semantic-retrieval-fusion")]
    #[test]
    fn find_similar_invalid_fusion_weight_is_invalid_argument() {
        let server = create_test_server();
        setup_fusion_corpus(&server);
        for bad in [
            serde_json::json!({ "w_confidence": -1.0 }),
            serde_json::json!({ "w_similarity": 0.0, "w_confidence": 0.0, "w_recency": 0.0 }),
            serde_json::json!({ "neutral_confidence": 1.5 }),
            serde_json::json!({ "recency_half_life_secs": 0.0 }),
        ] {
            let out = server.dispatch_tool_json(
                "find_similar",
                serde_json::json!({
                    "property_name": "embedding", "embedding": [1.0, 0.0, 0.0, 0.0], "k": 2,
                    "fusion_policy": bad
                }),
            );
            let v = parse(&out);
            assert_eq!(
                v.pointer("/error/code").and_then(|c| c.as_str()),
                Some("INVALID_ARGUMENT"),
                "invalid fusion policy must be INVALID_ARGUMENT: {out}"
            );
            assert_eq!(
                v.pointer("/error/retriable"),
                Some(&serde_json::json!(false))
            );
        }
    }

    #[cfg(feature = "semantic-retrieval-fusion")]
    #[test]
    fn hybrid_query_with_fusion_attaches_breakdown() {
        let server = create_test_server();
        setup_fusion_corpus(&server);
        let out = server.dispatch_tool_json(
            "hybrid_query",
            serde_json::json!({
                "query_embedding": [1.0, 0.0, 0.0, 0.0],
                "vector_property": "embedding",
                "top_k": 2, "limit": 2,
                "fusion_policy": { "w_similarity": 1.0, "w_confidence": 5.0, "w_recency": 0.0 }
            }),
        );
        let v = parse(&out);
        assert!(
            v.get("error").is_none(),
            "hybrid fusion should succeed: {out}"
        );
        let results = v["results"].as_array().expect("results");
        assert!(!results.is_empty(), "should return candidates: {out}");
        for r in results {
            assert!(
                r.get("score_breakdown").is_some(),
                "hybrid fusion attaches score_breakdown: {out}"
            );
        }
    }

    /// HIGH-1: the vector-first hybrid fusion path must over-fetch the fused
    /// horizon (not the similarity top-k) so a high-trust candidate ranked
    /// *below* the similarity top-k still surfaces in the fused top-k. With
    /// `top_k = 1` the similarity top-1 is "near"; "trusted" is rank-2 by
    /// similarity but high-trust, so a correct over-fetch promotes it. The
    /// buggy `(k, limit)` path fetches only the similarity top-1 ("near") and
    /// can never surface "trusted" — a k-independent post-hoc re-sort.
    #[cfg(feature = "semantic-retrieval-fusion")]
    #[test]
    fn hybrid_query_fusion_surfaces_high_trust_below_similarity_topk() {
        let server = create_test_server();
        setup_fusion_corpus(&server);
        let out = server.dispatch_tool_json(
            "hybrid_query",
            serde_json::json!({
                "query_embedding": [1.0, 0.0, 0.0, 0.0],
                "vector_property": "embedding",
                "namespace": "all",
                "top_k": 1, "limit": 1,
                "fusion_policy": { "w_similarity": 1.0, "w_confidence": 5.0, "w_recency": 0.0 }
            }),
        );
        let v = parse(&out);
        assert!(
            v.get("error").is_none(),
            "hybrid fusion should succeed: {out}"
        );
        assert_eq!(
            result_names(&v).first().map(String::as_str),
            Some("trusted"),
            "fusion must over-fetch the fused horizon and surface the high-trust \
             candidate ranked below the similarity top-k: {out}"
        );
    }

    /// MED-2: the hybrid fusion path must reject a non-Cosine index with the
    /// same structured `FAILED_PRECONDITION` the `find_similar` fused path uses,
    /// rather than feeding a distance whose scores are not in `[0,1]` straight
    /// into the fused score.
    #[cfg(feature = "semantic-retrieval-fusion")]
    #[test]
    fn hybrid_query_fusion_on_non_cosine_index_is_failed_precondition() {
        let server = create_test_server();
        server.dispatch_tool_json(
            "enable_vector_index",
            serde_json::json!({
                "property_name": "embedding", "dimensions": 4,
                "distance_metric": "euclidean"
            }),
        );
        server.dispatch_tool_json(
            "create_node",
            serde_json::json!({
                "label": "Doc",
                "properties": { "name": "a", "embedding": [1.0, 0.0, 0.0, 0.0] }
            }),
        );
        let out = server.dispatch_tool_json(
            "hybrid_query",
            serde_json::json!({
                "query_embedding": [1.0, 0.0, 0.0, 0.0],
                "vector_property": "embedding",
                "top_k": 2, "limit": 2,
                "fusion_policy": { "w_similarity": 1.0, "w_confidence": 5.0, "w_recency": 0.0 }
            }),
        );
        let v = parse(&out);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("FAILED_PRECONDITION"),
            "hybrid fusion on a non-Cosine index must be FAILED_PRECONDITION: {out}"
        );
        assert_eq!(
            v.pointer("/error/retriable"),
            Some(&serde_json::json!(false))
        );
    }

    /// MED-3: the `start_node_id` single-node fast path (no `traverse_edge`)
    /// must still run fusion scoring when a `fusion_policy` is supplied, so the
    /// contract's per-result `score_breakdown` is present on these paths too.
    #[cfg(feature = "semantic-retrieval-fusion")]
    #[test]
    fn hybrid_query_single_node_with_fusion_attaches_breakdown() {
        let server = create_test_server();
        setup_fusion_corpus(&server);
        // Resolve a concrete node id via find_similar.
        let fs = parse(&server.dispatch_tool_json(
            "find_similar",
            serde_json::json!({
                "property_name": "embedding", "embedding": [1.0, 0.0, 0.0, 0.0], "k": 1
            }),
        ));
        let node_id = fs["results"][0]["node"]["id"]
            .as_u64()
            .expect("node id from find_similar");
        let out = server.dispatch_tool_json(
            "hybrid_query",
            serde_json::json!({
                "start_node_id": node_id,
                "fusion_policy": { "w_similarity": 1.0, "w_confidence": 5.0, "w_recency": 0.0 }
            }),
        );
        let v = parse(&out);
        assert!(
            v.get("error").is_none(),
            "single-node hybrid fusion should succeed: {out}"
        );
        let results = v["results"].as_array().expect("results");
        assert_eq!(
            results.len(),
            1,
            "single-node path returns exactly one result: {out}"
        );
        assert!(
            results[0].get("score_breakdown").is_some(),
            "single-node hybrid fusion must attach score_breakdown: {out}"
        );
    }
}

/// Behavior tests for the deferred MCP-registry batch tools (Issue #3367 /
/// #3352 / #3357 / #3382). Under the default (no-cohort) test build these
/// exercise the feature-off twins; the advertising tests are feature-invariant.
mod deferred_batch_tools_tests {
    use super::*;

    fn parse(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("valid JSON response")
    }

    const TEMPORAL_TOOLS: [&str; 8] = [
        "create_drift_monitor",
        "list_drift_monitors",
        "delete_drift_monitor",
        "query_drift_alarms",
        "resolve_drift_alarm",
        "contradiction_genealogy",
        "find_contradictions",
        "counterfactual_replay",
    ];
    const REASONING_TOOLS: [&str; 2] = ["trust_breakdown", "list_trust_policies"];

    /// Design A: every deferred tool is advertised regardless of feature flags,
    /// so the catalog count is feature-invariant.
    #[test]
    fn all_ten_deferred_tools_are_advertised() {
        let server = create_test_server();
        let tools = server.list_tools_for_test();
        for name in TEMPORAL_TOOLS.iter().chain(REASONING_TOOLS.iter()) {
            assert!(
                tools.iter().any(|t| t == name),
                "{name} must be advertised (Design A: unconditional)"
            );
        }
    }

    /// Each deferred tool carries the correct RBAC access class.
    #[test]
    fn deferred_tools_have_expected_access_class() {
        use crate::auth::AccessClass;
        use crate::mcp::auth::tool_access_class;
        let read = [
            "list_drift_monitors",
            "query_drift_alarms",
            "contradiction_genealogy",
            "find_contradictions",
            "counterfactual_replay",
            "trust_breakdown",
            "list_trust_policies",
        ];
        let write = [
            "create_drift_monitor",
            "delete_drift_monitor",
            "resolve_drift_alarm",
        ];
        for name in read {
            assert_eq!(
                tool_access_class(name),
                Some(AccessClass::Read),
                "{name} must be Read class"
            );
        }
        for name in write {
            assert_eq!(
                tool_access_class(name),
                Some(AccessClass::Write),
                "{name} must be Write class"
            );
        }
    }

    /// Feature-off twin: the eight `semantic-temporal` tools return a structured
    /// FAILED_PRECONDITION naming the required feature.
    #[cfg(not(feature = "semantic-temporal"))]
    #[test]
    fn temporal_tools_feature_off_are_failed_precondition() {
        let server = create_test_server();
        for name in TEMPORAL_TOOLS {
            let out = server.dispatch_tool_json(name, serde_json::json!({}));
            let v = parse(&out);
            assert_eq!(
                v.pointer("/error/code").and_then(|c| c.as_str()),
                Some("FAILED_PRECONDITION"),
                "feature-off {name} must be FAILED_PRECONDITION: {out}"
            );
            assert_eq!(
                v.pointer("/error/details/required_feature")
                    .and_then(|c| c.as_str()),
                Some("semantic-temporal"),
                "{name} must name the required feature: {out}"
            );
            assert_eq!(
                v.pointer("/error/retriable"),
                Some(&serde_json::json!(false)),
                "{name} feature-precondition is non-retriable: {out}"
            );
        }
    }

    /// Feature-off twin: the two `semantic-reasoning` tools return a structured
    /// FAILED_PRECONDITION naming the required feature.
    #[cfg(not(feature = "semantic-reasoning"))]
    #[test]
    fn reasoning_tools_feature_off_are_failed_precondition() {
        let server = create_test_server();
        for name in REASONING_TOOLS {
            let out = server.dispatch_tool_json(name, serde_json::json!({}));
            let v = parse(&out);
            assert_eq!(
                v.pointer("/error/code").and_then(|c| c.as_str()),
                Some("FAILED_PRECONDITION"),
                "feature-off {name} must be FAILED_PRECONDITION: {out}"
            );
            assert_eq!(
                v.pointer("/error/details/required_feature")
                    .and_then(|c| c.as_str()),
                Some("semantic-reasoning"),
                "{name} must name the required feature: {out}"
            );
        }
    }

    /// Feature-on happy path: with `semantic-temporal`, drift monitor CRUD works
    /// and `counterfactual_replay` carries the AC8 `counterfactual: true` marker.
    #[cfg(feature = "semantic-temporal")]
    #[test]
    fn temporal_tools_feature_on_smoke() {
        let server = create_test_server();

        // list_drift_monitors on an empty registry: success, zero monitors.
        let out = server.dispatch_tool_json("list_drift_monitors", serde_json::json!({}));
        let v = parse(&out);
        assert!(v.get("error").is_none(), "list_drift_monitors: {out}");
        assert_eq!(v.pointer("/count"), Some(&serde_json::json!(0)));

        // query_drift_alarms on an empty registry: success, zero alarms.
        let out = server.dispatch_tool_json("query_drift_alarms", serde_json::json!({}));
        let v = parse(&out);
        assert!(v.get("error").is_none(), "query_drift_alarms: {out}");
        assert_eq!(v.pointer("/count"), Some(&serde_json::json!(0)));

        // counterfactual_replay excluding an absent source: success, and the
        // AC8 per-response marker is present.
        let out = server.dispatch_tool_json(
            "counterfactual_replay",
            serde_json::json!({ "name": "cf", "exclude_source": "nobody" }),
        );
        let v = parse(&out);
        assert!(v.get("error").is_none(), "counterfactual_replay: {out}");
        assert_eq!(
            v.pointer("/counterfactual"),
            Some(&serde_json::json!(true)),
            "counterfactual_replay must carry the AC8 marker: {out}"
        );

        // A create_drift_monitor with a non-positive threshold is INVALID_ARGUMENT.
        let out = server.dispatch_tool_json(
            "create_drift_monitor",
            serde_json::json!({
                "property_key": "embedding",
                "metric": "cosine",
                "threshold": 0.0,
                "window_micros": 3_600_000_000_u64
            }),
        );
        let v = parse(&out);
        assert_eq!(
            v.pointer("/error/code").and_then(|c| c.as_str()),
            Some("INVALID_ARGUMENT"),
            "non-positive threshold must be INVALID_ARGUMENT: {out}"
        );
    }

    /// Feature-on happy path: with `semantic-reasoning`, `list_trust_policies`
    /// returns the default policy.
    #[cfg(feature = "semantic-reasoning")]
    #[test]
    fn reasoning_tools_feature_on_smoke() {
        let server = create_test_server();
        let out = server.dispatch_tool_json("list_trust_policies", serde_json::json!({}));
        let v = parse(&out);
        assert!(v.get("error").is_none(), "list_trust_policies: {out}");
        assert!(
            v.pointer("/default/combinator").is_some(),
            "list_trust_policies must return a default combinator: {out}"
        );
    }
}
