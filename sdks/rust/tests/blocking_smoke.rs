//! Blocking smoke tests — wiremock-backed tests for every blocking method.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_pass_by_value,
    clippy::uninlined_format_args,
    clippy::manual_range_contains,
    missing_docs
)]

use honcho_ai::blocking::Honcho;
use honcho_ai::types::message::MessageSearchOptions;
use honcho_ai::types::workspace::WorkspaceConfiguration;
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn ws_json() -> serde_json::Value {
    serde_json::json!({
        "id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn ws_json_with_config() -> serde_json::Value {
    serde_json::json!({
        "id": "ws1",
        "metadata": {},
        "configuration": {
            "reasoning": {"enabled": true}
        },
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn peer_json() -> serde_json::Value {
    serde_json::json!({
        "id": "alice",
        "workspace_id": "ws1",
        "created_at": "2025-01-15T10:30:00Z",
        "metadata": {},
        "configuration": {}
    })
}

fn session_json() -> serde_json::Value {
    serde_json::json!({
        "id": "sess1",
        "workspace_id": "ws1",
        "is_active": true,
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn msg_json(id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "content": "hello world",
        "peer_id": "alice",
        "session_id": "sess1",
        "metadata": {},
        "created_at": "2025-01-15T10:30:00Z",
        "workspace_id": "ws1",
        "token_count": 2
    })
}

fn context_json() -> serde_json::Value {
    serde_json::json!({
        "id": "sess1",
        "messages": [msg_json("m1")],
        "summary": {
            "content": "a summary",
            "message_id": "msg0",
            "summary_type": "short",
            "created_at": "2025-01-15T10:30:00Z",
            "token_count": 5
        },
        "peer_representation": "some rep",
        "peer_card": ["fact1"]
    })
}

fn page_json(
    items: Vec<serde_json::Value>,
    total: u64,
    page: u64,
    size: u64,
    pages: u64,
) -> serde_json::Value {
    serde_json::json!({
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages
    })
}

fn queue_status_json() -> serde_json::Value {
    serde_json::json!({
        "total_work_units": 5,
        "completed_work_units": 3,
        "in_progress_work_units": 1,
        "pending_work_units": 1
    })
}

async fn mount_ensure_ws(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(serde_json::json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(ws_json()))
        .up_to_n_times(1)
        .mount(server)
        .await;
}

async fn mount_create_session(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions"))
        .and(body_json(serde_json::json!({"id": "sess1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(session_json()))
        .up_to_n_times(1)
        .mount(server)
        .await;
}

async fn mount_create_peer(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers"))
        .and(body_json(serde_json::json!({"id": "alice"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
        .up_to_n_times(1)
        .mount(server)
        .await;
}

fn blocking<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    std::thread::scope(|s| s.spawn(f).join().unwrap())
}

// ─── Session: context ────────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_session_context() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/context"))
        .and(query_param("summary", "true"))
        .and(query_param("limit_to_session", "false"))
        .respond_with(ResponseTemplate::new(200).set_body_json(context_json()))
        .mount(&server)
        .await;

    let uri = server.uri();
    let ctx = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let session = client.session("sess1", None, None, None).unwrap();
        session.context().unwrap()
    });
    assert_eq!(ctx.id, "sess1");
    assert_eq!(ctx.messages.len(), 1);
}

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_session_context_with_options() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/context"))
        .and(query_param("summary", "false"))
        .and(query_param("limit_to_session", "true"))
        .respond_with(ResponseTemplate::new(200).set_body_json(context_json()))
        .mount(&server)
        .await;

    let uri = server.uri();
    let ctx = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let session = client.session("sess1", None, None, None).unwrap();
        let opts = honcho_ai::types::session::SessionContextOptions::builder()
            .summary(false)
            .limit_to_session(true)
            .build();
        session.context_with_options(&opts).unwrap()
    });
    assert_eq!(ctx.id, "sess1");
}

// ─── Session: summaries ──────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_session_summaries() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/summaries"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "sess1",
            "short_summary": {
                "content": "short",
                "message_id": "m0",
                "summary_type": "short",
                "created_at": "2025-01-15T10:30:00Z",
                "token_count": 3
            }
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let summaries = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let session = client.session("sess1", None, None, None).unwrap();
        session.summaries().unwrap()
    });
    assert_eq!(summaries.id, "sess1");
    assert!(summaries.short_summary.is_some());
    assert_eq!(summaries.short_summary.unwrap().content, "short");
}

// ─── Session: search ─────────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_session_search() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/search"))
        .and(body_json(serde_json::json!({
            "query": "hello",
            "limit": 10
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(vec![msg_json("m1")]))
        .mount(&server)
        .await;

    let uri = server.uri();
    let results = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let session = client.session("sess1", None, None, None).unwrap();
        session.search("hello").unwrap()
    });
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id(), "m1");
}

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_session_search_with_options() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/search"))
        .and(body_json(serde_json::json!({
            "query": "hello",
            "limit": 20
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(vec![msg_json("m1")]))
        .mount(&server)
        .await;

    let uri = server.uri();
    let results = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let session = client.session("sess1", None, None, None).unwrap();
        session
            .search_with_options(&MessageSearchOptions {
                query: "hello".into(),
                filters: None,
                limit: 20,
            })
            .unwrap()
    });
    assert_eq!(results.len(), 1);
}

// ─── Session: representation ─────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_session_representation() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/representation"))
        .and(body_json(serde_json::json!({"session_id": "sess1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "representation": "Alice likes Rust"
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let rep = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let session = client.session("sess1", None, None, None).unwrap();
        session.representation("alice").unwrap()
    });
    assert_eq!(rep, "Alice likes Rust");
}

// ─── Session: queue_status ───────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_session_queue_status() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/queue/status"))
        .and(query_param("session_id", "sess1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(queue_status_json()))
        .mount(&server)
        .await;

    let uri = server.uri();
    let status = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let session = client.session("sess1", None, None, None).unwrap();
        session.queue_status(None, None).unwrap()
    });
    assert_eq!(status.total_work_units, 5);
    assert_eq!(status.completed_work_units, 3);
}

// ─── Session: messages ───────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_session_messages() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages/list"))
        .and(query_param("page", "1"))
        .and(query_param("size", "50"))
        .respond_with(ResponseTemplate::new(200).set_body_json(page_json(
            vec![msg_json("m1")],
            1,
            1,
            50,
            1,
        )))
        .mount(&server)
        .await;

    let uri = server.uri();
    let msgs = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let session = client.session("sess1", None, None, None).unwrap();
        session.messages().unwrap()
    });
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].id(), "m1");
}

// ─── Peer: search ────────────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_peer_search() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/search"))
        .and(body_json(serde_json::json!({
            "query": "hello",
            "limit": 10
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(vec![msg_json("m1")]))
        .mount(&server)
        .await;

    let uri = server.uri();
    let results = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let peer = client.peer("alice", None, None).unwrap();
        peer.search("hello").unwrap()
    });
    assert_eq!(results.len(), 1);
}

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_peer_search_with_options() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/search"))
        .and(body_json(serde_json::json!({
            "query": "hello",
            "limit": 25
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(vec![msg_json("m1")]))
        .mount(&server)
        .await;

    let uri = server.uri();
    let results = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let peer = client.peer("alice", None, None).unwrap();
        peer.search_with_options(&MessageSearchOptions {
            query: "hello".into(),
            filters: None,
            limit: 25,
        })
        .unwrap()
    });
    assert_eq!(results.len(), 1);
}

// ─── Peer: context ───────────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_peer_context() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_peer(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/peers/alice/context"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "peer_id": "alice",
            "target_id": "alice",
            "representation": "curious mind",
            "peer_card": ["friendly"]
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let ctx = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let peer = client.peer("alice", None, None).unwrap();
        peer.context().unwrap()
    });
    assert_eq!(ctx.peer_id, "alice");
    assert_eq!(ctx.representation.as_deref(), Some("curious mind"));
}

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_peer_context_with_target() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_peer(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/peers/alice/context"))
        .and(query_param("target", "bob"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "peer_id": "alice",
            "target_id": "bob",
            "representation": "Bob helps"
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let ctx = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let peer = client.peer("alice", None, None).unwrap();
        peer.context_builder().target("bob").send().unwrap()
    });
    assert_eq!(ctx.target_id, "bob");
}

// ─── Peer: sessions ──────────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_peer_sessions() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/sessions"))
        .and(query_param("page", "1"))
        .and(query_param("size", "50"))
        .respond_with(ResponseTemplate::new(200).set_body_json(page_json(
            vec![session_json()],
            1,
            1,
            50,
            1,
        )))
        .mount(&server)
        .await;

    let uri = server.uri();
    let sessions = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let peer = client.peer("alice", None, None).unwrap();
        peer.sessions().unwrap()
    });
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, "sess1");
}

// ─── Peer: representation ────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_peer_representation() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/representation"))
        .and(body_json(serde_json::json!({})))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "representation": "Alice likes cats"
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let rep = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let peer = client.peer("alice", None, None).unwrap();
        peer.representation().unwrap()
    });
    assert_eq!(rep, "Alice likes cats");
}

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_peer_representation_builder_with_options() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/representation"))
        .and(body_json(serde_json::json!({
            "search_query": "hobbies",
            "search_top_k": 10,
            "search_max_distance": 0.5,
            "include_most_frequent": true,
            "max_conclusions": 25
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "representation": "curated hobbies"
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let rep = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let peer = client.peer("alice", None, None).unwrap();
        peer.representation_builder()
            .search_query("hobbies")
            .search_top_k(10)
            .search_max_distance(0.5)
            .include_most_frequent(true)
            .max_conclusions(25)
            .send()
            .unwrap()
    });
    assert_eq!(rep, "curated hobbies");
}

// ─── Client: get_configuration ───────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_get_configuration() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(serde_json::json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(ws_json_with_config()))
        .mount(&server)
        .await;

    let uri = server.uri();
    let config = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client.get_configuration().unwrap()
    });
    assert!(config.reasoning.is_some());
    assert_eq!(config.reasoning.unwrap().enabled, Some(true));
}

// ─── Client: set_configuration ───────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_set_configuration() {
    let server = MockServer::start().await;

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ws_json_with_config()))
        .mount(&server)
        .await;

    let uri = server.uri();
    blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let config: WorkspaceConfiguration = serde_json::from_value(serde_json::json!({
            "reasoning": {"enabled": true}
        }))
        .unwrap();
        client.set_configuration(&config).unwrap();
    });
}

// ─── Client: search ──────────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_search() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/search"))
        .and(body_json(serde_json::json!({
            "query": "hello",
            "limit": 10
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(vec![msg_json("m1")]))
        .mount(&server)
        .await;

    let uri = server.uri();
    let results = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client.search("hello", None, None).unwrap()
    });
    assert_eq!(results.len(), 1);
}

// ─── Client: queue_status ────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_queue_status() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/queue/status"))
        .respond_with(ResponseTemplate::new(200).set_body_json(queue_status_json()))
        .mount(&server)
        .await;

    let uri = server.uri();
    let status = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client.queue_status(None, None, None).unwrap()
    });
    assert_eq!(status.total_work_units, 5);
}

// ─── Client: schedule_dream ──────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_schedule_dream() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/schedule_dream"))
        .and(body_json(serde_json::json!({
            "observer": "alice",
            "observed": "alice",
            "dream_type": "omni"
        })))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let uri = server.uri();
    blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client.schedule_dream("alice", None, None).unwrap();
    });
}

// ─── Client: peers_with_filters ──────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_peers_with_filters() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/list"))
        .and(query_param("page", "1"))
        .and(query_param("size", "10"))
        .respond_with(ResponseTemplate::new(200).set_body_json(page_json(
            vec![peer_json()],
            1,
            1,
            10,
            1,
        )))
        .mount(&server)
        .await;

    let uri = server.uri();
    let peers = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client
            .peers_with_filters(std::collections::HashMap::new(), 1, 10, false)
            .unwrap()
    });
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].id, "alice");
}

// ─── Client: sessions_with_filters ───────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_sessions_with_filters() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/list"))
        .and(query_param("page", "1"))
        .and(query_param("size", "10"))
        .respond_with(ResponseTemplate::new(200).set_body_json(page_json(
            vec![session_json()],
            1,
            1,
            10,
            1,
        )))
        .mount(&server)
        .await;

    let uri = server.uri();
    let sessions = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client
            .sessions_with_filters(std::collections::HashMap::new(), 1, 10, false)
            .unwrap()
    });
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, "sess1");
}

// ─── Client: workspaces ──────────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_workspaces() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/list"))
        .and(query_param("page", "1"))
        .and(query_param("size", "50"))
        .respond_with(ResponseTemplate::new(200).set_body_json(page_json(
            vec![
                serde_json::json!({"id": "ws_a", "metadata": {}, "configuration": {}, "created_at": "2025-01-15T10:30:00Z"}),
                serde_json::json!({"id": "ws_b", "metadata": {}, "configuration": {}, "created_at": "2025-01-15T10:30:00Z"}),
            ],
            2, 1, 50, 1,
        )))
        .mount(&server)
        .await;

    let uri = server.uri();
    let ws = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client.workspaces().unwrap()
    });
    assert_eq!(ws.len(), 2);
    assert_eq!(ws[0], "ws_a");
    assert_eq!(ws[1], "ws_b");
}

// ─── Session: search validates empty query ───────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_session_search_validates_empty() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_session(&server).await;

    let uri = server.uri();
    let err = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let session = client.session("sess1", None, None, None).unwrap();
        session.search("").unwrap_err()
    });
    assert_eq!(err.code(), "validation_error");
}

// ─── Peer: search validates empty query ──────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_peer_search_validates_empty() {
    let server = MockServer::start().await;
    mount_ensure_ws(&server).await;
    mount_create_peer(&server).await;

    let uri = server.uri();
    let err = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let peer = client.peer("alice", None, None).unwrap();
        peer.search("").unwrap_err()
    });
    assert_eq!(err.code(), "validation_error");
}

// ─── Client: delete_workspace ────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_delete_workspace() {
    let server = MockServer::start().await;

    Mock::given(method("DELETE"))
        .and(path("/v3/workspaces/old-ws"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    let uri = server.uri();
    blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client.delete_workspace("old-ws").unwrap();
    });
}

// ─── Client: get/set metadata ────────────────────────────────────────

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_get_metadata() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(serde_json::json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "ws1",
            "metadata": {"env": "test"},
            "configuration": {},
            "created_at": "2025-01-15T10:30:00Z"
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let meta = blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client.get_metadata().unwrap()
    });
    assert_eq!(meta.get("env").unwrap(), "test");
}

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_set_metadata() {
    let server = MockServer::start().await;

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ws_json()))
        .mount(&server)
        .await;

    let uri = server.uri();
    blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        let mut meta = std::collections::HashMap::new();
        meta.insert("key".into(), serde_json::json!("value"));
        client.set_metadata(meta).unwrap();
    });
}

#[cfg(feature = "blocking")]
#[tokio::test]
async fn blocking_client_refresh() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(serde_json::json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "ws1",
            "metadata": {"env": "test"},
            "configuration": {"reasoning": {"enabled": true}},
            "created_at": "2025-01-15T10:30:00Z"
        })))
        .up_to_n_times(3)
        .mount(&server)
        .await;

    let uri = server.uri();
    blocking(move || {
        let client = Honcho::new(&uri, "ws1").unwrap();
        client.refresh().unwrap();
    });
}
