use honcho_mcp_rs::config::parse_config_from_pairs;
use honcho_mcp_rs::honcho_client::HonchoClient;
use honcho_mcp_rs::mcp_server::text_result;
use honcho_mcp_rs::tools;
use honcho_mcp_rs::tools::sessions;
use honcho_mcp_rs::tools::workspace;
use serde_json::json;

#[test]
fn parse_headers_matches_worker_defaults() {
    let config = parse_config_from_pairs(
        &[
            ("authorization", "Bearer test-key"),
            ("x-honcho-user-name", "samuel"),
        ],
        "http://api:8000/",
    )
    .expect("headers should parse");

    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.authorization, "Bearer test-key");
    assert_eq!(config.user_name, "samuel");
    assert_eq!(config.assistant_name, "Assistant");
    assert_eq!(config.workspace_id, "default");
    assert_eq!(config.base_url, "http://api:8000");
}

#[test]
fn parse_headers_rejects_missing_authorization() {
    let error = parse_config_from_pairs(&[("x-honcho-user-name", "samuel")], "http://api:8000")
        .expect_err("authorization should be required");

    assert!(error.to_string().contains("Authorization"));
}

#[test]
fn tool_names_match_typescript_contract() {
    let expected = vec![
        "inspect_workspace",
        "list_workspaces",
        "search",
        "get_metadata",
        "set_metadata",
        "create_peer",
        "list_peers",
        "chat",
        "get_peer_card",
        "set_peer_card",
        "get_peer_context",
        "get_representation",
        "create_session",
        "list_sessions",
        "delete_session",
        "clone_session",
        "add_peers_to_session",
        "remove_peers_from_session",
        "get_session_peers",
        "inspect_session",
        "add_messages_to_session",
        "get_session_messages",
        "get_session_message",
        "get_session_context",
        "list_conclusions",
        "query_conclusions",
        "create_conclusions",
        "delete_conclusion",
        "schedule_dream",
        "get_queue_status",
    ];

    assert_eq!(tools::tool_names(), expected);
}

#[test]
fn create_peer_schema_preserves_sdk_configuration_casing() {
    let tool = tools::get_tool("create_peer").expect("create_peer tool should exist");
    let schema = tool.schema_as_json_value();

    assert!(
        schema
            .pointer("/properties/configuration/properties/observeMe")
            .is_some()
    );
    assert_eq!(schema.pointer("/required/0"), Some(&json!("peer_id")));
}

#[test]
fn text_result_matches_typescript_text_wrapping() {
    let result = text_result(json!({ "a": 1 }));

    assert_eq!(result.content.len(), 1);
    assert_eq!(result.content[0].as_text().unwrap().text, "{\"a\":1}");
    assert!(result.structured_content.is_none());
}

#[test]
fn session_peer_inputs_become_api_peer_map() {
    let peers = json!([
        "alice",
        { "peer_id": "bot", "observe_me": false, "observe_others": true }
    ]);

    let map = sessions::session_peer_map_for_test(peers.as_array().unwrap())
        .expect("peer input should be valid");

    assert_eq!(map["alice"], json!({}));
    assert_eq!(
        map["bot"],
        json!({ "observe_me": false, "observe_others": true })
    );
}

#[test]
fn honcho_client_builds_v3_workspace_urls() {
    let client = HonchoClient::new_for_test("http://api:8000/", "Bearer test-key");

    assert_eq!(
        client.endpoint_for_test(&["workspaces", "default", "peers"]),
        "http://api:8000/v3/workspaces/default/peers"
    );
    assert_eq!(
        client.endpoint_for_test(&["workspaces", "needs/slash"]),
        "http://api:8000/v3/workspaces/needs%2Fslash"
    );
}

#[test]
fn workspace_pages_match_typescript_mcp_shapes() {
    let page = json!({
        "items": [
            { "id": "codex", "metadata": { "ignored": true } },
            { "name": "default" }
        ],
        "total": 2,
        "page": 1,
        "pages": 1
    });

    assert_eq!(
        workspace::format_workspaces_page_for_test(&page),
        json!({
            "workspaces": [{ "id": "codex" }, { "id": "default" }],
            "total": 2,
            "page": 1,
            "pages": 1
        })
    );
}

#[test]
fn inspect_workspace_matches_typescript_mcp_shape() {
    let workspace_value = json!({
        "id": "codex",
        "metadata": { "a": 1 },
        "configuration": { "b": 2 }
    });
    let peers = json!({
        "items": [{ "id": "alice" }, { "name": "bot" }],
        "total": 2
    });
    let sessions = json!({
        "items": [{ "id": "s1" }],
        "total": 1
    });

    assert_eq!(
        workspace::format_inspect_workspace_for_test(&workspace_value, &peers, &sessions, "codex"),
        json!({
            "workspace_id": "codex",
            "metadata": { "a": 1 },
            "configuration": { "b": 2 },
            "peer_count": 2,
            "peers": [{ "id": "alice" }, { "id": "bot" }],
            "session_count": 1,
            "sessions": [{ "id": "s1" }]
        })
    );
}

#[test]
fn search_results_drop_extra_api_fields() {
    let messages = json!([
        {
            "id": "m1",
            "content": "hello",
            "peer_id": "alice",
            "session_id": "s1",
            "metadata": { "kind": "smoke" },
            "created_at": "2026-06-11T00:00:00Z",
            "token_count": 99,
            "workspace_id": "codex"
        }
    ]);

    assert_eq!(
        workspace::format_search_messages_for_test(&messages),
        json!([
            {
                "id": "m1",
                "content": "hello",
                "peer_id": "alice",
                "session_id": "s1",
                "metadata": { "kind": "smoke" },
                "created_at": "2026-06-11T00:00:00Z"
            }
        ])
    );
}
