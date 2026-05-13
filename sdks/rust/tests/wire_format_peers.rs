//! Wire-format regression tests for session peer management.
//!
//! Verifies that `add_peers`, `set_peers`, and `remove_peers` send the correct
//! JSON payloads to the server (flat map, no double-wrap; list for DELETE).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_borrows_for_generic_args,
    missing_docs
)]

use honcho_ai::Honcho;
use honcho_ai::session::Session;
use serde_json::{Value, json};
use wiremock::matchers::{body_json, method, path};
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

// ── add_peers ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn add_peers_sends_flat_map_without_peers_wrapper() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let expected_body = json!({
        "alice": {},
        "bob": {}
    });

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(&expected_body))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    session.add_peers(["alice", "bob"]).await.unwrap();
}

#[tokio::test]
async fn add_peer_single_sends_flat_map() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(json!({"alice": {}})))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    session.add_peer("alice").await.unwrap();
}

// ── set_peers ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn set_peers_sends_flat_map_without_peers_wrapper() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let expected_body = json!({
        "alice": {},
        "bob": {}
    });

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(&expected_body))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    session.set_peers(["alice", "bob"]).await.unwrap();
}

// ── remove_peers ───────────────────────────────────────────────────────────

#[tokio::test]
async fn remove_peers_sends_list_of_ids() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("DELETE"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(json!(["alice", "bob"])))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    session.remove_peers(["alice", "bob"]).await.unwrap();
}
