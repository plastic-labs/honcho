//! Integration tests for Session context, summaries, search, representation, and `queue_status`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_borrows_for_generic_args,
    missing_docs
)]

use honcho_ai::Honcho;
use honcho_ai::session::Session;
use honcho_ai::types::session::SessionContextOptions;
use serde_json::{Value, json};
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn workspace_response_json() -> Value {
    json!({
        "id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn session_response_json() -> Value {
    json!({
        "id": "sess1",
        "workspace_id": "ws1",
        "is_active": true,
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

async fn make_session(server: &MockServer) -> Session {
    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_response_json()))
        .up_to_n_times(1)
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions"))
        .and(body_json(json!({"id": "sess1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(session_response_json()))
        .up_to_n_times(1)
        .mount(server)
        .await;

    let honcho = Honcho::new(&server.uri(), "ws1").unwrap();
    honcho.session("sess1").await.unwrap()
}

fn context_response_json() -> Value {
    json!({
        "id": "sess1",
        "messages": [
            {
                "id": "m1",
                "content": "hello",
                "peer_id": "user1",
                "session_id": "sess1",
                "metadata": {},
                "created_at": "2025-01-15T10:30:00Z",
                "workspace_id": "ws1",
                "token_count": 1
            }
        ],
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

// ── F6.6: Context ────────────────────────────────────────────────────

#[tokio::test]
async fn session_context_returns_session_context() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/context"))
        .and(query_param("summary", "true"))
        .and(query_param("limit_to_session", "false"))
        .respond_with(ResponseTemplate::new(200).set_body_json(context_response_json()))
        .mount(&server)
        .await;

    let ctx = session.context().await.unwrap();
    assert_eq!(ctx.id, "sess1");
    assert_eq!(ctx.messages.len(), 1);
    assert_eq!(ctx.messages[0].content, "hello");
    assert!(ctx.summary.is_some());
    assert_eq!(ctx.summary.unwrap().content, "a summary");
    assert_eq!(ctx.peer_representation, Some("some rep".to_string()));
    assert_eq!(ctx.peer_card, Some(vec!["fact1".to_string()]));
}

// ── F6.8: Summaries ──────────────────────────────────────────────────

#[tokio::test]
async fn session_summaries_returns_both() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/summaries"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "sess1",
            "short_summary": {
                "content": "short one",
                "message_id": "msg1",
                "summary_type": "short",
                "created_at": "2025-01-15T10:30:00Z",
                "token_count": 3
            },
            "long_summary": {
                "content": "long one",
                "message_id": "msg2",
                "summary_type": "long",
                "created_at": "2025-01-15T10:30:00Z",
                "token_count": 10
            }
        })))
        .mount(&server)
        .await;

    let summaries = session.summaries().await.unwrap();
    assert_eq!(summaries.id, "sess1");
    assert!(summaries.short_summary.is_some());
    assert_eq!(summaries.short_summary.unwrap().content, "short one");
    assert!(summaries.long_summary.is_some());
    assert_eq!(summaries.long_summary.unwrap().content, "long one");
}

#[tokio::test]
async fn session_summaries_none_when_not_available() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/summaries"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "sess1"
        })))
        .mount(&server)
        .await;

    let summaries = session.summaries().await.unwrap();
    assert_eq!(summaries.id, "sess1");
    assert!(summaries.short_summary.is_none());
    assert!(summaries.long_summary.is_none());
}

// ── F6.9: Search ─────────────────────────────────────────────────────

#[tokio::test]
async fn session_search_returns_messages() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/search"))
        .and(body_json(json!({
            "query": "hello",
            "limit": 10
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {
                "id": "m1",
                "content": "hello world",
                "peer_id": "user1",
                "session_id": "sess1",
                "metadata": {},
                "created_at": "2025-01-15T10:30:00Z",
                "workspace_id": "ws1",
                "token_count": 2
            }
        ])))
        .mount(&server)
        .await;

    let results = session.search("hello").await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content(), "hello world");
}

#[tokio::test]
async fn session_search_validates_empty_query() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let err = session.search("").await.unwrap_err();
    assert_eq!(err.code(), "validation_error");
}

// ── F6.9: Representation ──────────────────────────────────────────────

#[tokio::test]
async fn session_representation_posts_to_peer_representation() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/representation"))
        .and(body_json(json!({"session_id": "sess1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "representation": "Alice likes Rust"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rep = session.representation("alice").await.unwrap();
    assert_eq!(rep, "Alice likes Rust");
}

// ── F6.9: Queue Status ────────────────────────────────────────────────

#[tokio::test]
async fn session_queue_status_gets_with_session_id() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/queue/status"))
        .and(query_param("session_id", "sess1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "total_work_units": 5,
            "completed_work_units": 3,
            "in_progress_work_units": 1,
            "pending_work_units": 1
        })))
        .expect(1)
        .mount(&server)
        .await;

    let status = session.queue_status(None, None).await.unwrap();
    assert_eq!(status.total_work_units, 5);
    assert_eq!(status.completed_work_units, 3);
    assert_eq!(status.in_progress_work_units, 1);
    assert_eq!(status.pending_work_units, 1);
}

// ── Context with options ───────────────────────────────────────────

#[tokio::test]
async fn session_context_with_options_sends_all_query_params() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/context"))
        .and(query_param("summary", "false"))
        .and(query_param("limit_to_session", "true"))
        .and(query_param("tokens", "4096"))
        .and(query_param("peer_target", "bob"))
        .and(query_param("peer_perspective", "alice"))
        .and(query_param("search_query", "preferences"))
        .and(query_param("search_top_k", "10"))
        .and(query_param("search_max_distance", "0.5"))
        .and(query_param("include_most_frequent", "true"))
        .and(query_param("max_conclusions", "20"))
        .respond_with(ResponseTemplate::new(200).set_body_json(context_response_json()))
        .expect(1)
        .mount(&server)
        .await;

    let opts = SessionContextOptions::builder()
        .summary(false)
        .limit_to_session(true)
        .tokens(4096)
        .peer_target("bob")
        .peer_perspective("alice")
        .search_query("preferences")
        .search_top_k(10)
        .search_max_distance(0.5)
        .include_most_frequent(true)
        .max_conclusions(20)
        .build();

    let ctx = session.context_with_options(&opts).await.unwrap();
    assert_eq!(ctx.id, "sess1");
}

#[tokio::test]
async fn session_context_with_options_sends_only_set_params() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/context"))
        .and(query_param("summary", "true"))
        .and(query_param("limit_to_session", "false"))
        .respond_with(ResponseTemplate::new(200).set_body_json(context_response_json()))
        .expect(1)
        .mount(&server)
        .await;

    let opts = SessionContextOptions::builder()
        .summary(true)
        .limit_to_session(false)
        .build();

    let ctx = session.context_with_options(&opts).await.unwrap();
    assert_eq!(ctx.id, "sess1");
}
