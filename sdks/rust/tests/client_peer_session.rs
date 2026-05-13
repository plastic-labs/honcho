//! Tests for `Honcho::peer()` and `Honcho::session()` (F4.4).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_borrows_for_generic_args,
    clippy::unused_async,
    missing_docs
)]

use honcho_ai::Honcho;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn peer_response() -> serde_json::Value {
    serde_json::json!({
        "id": "alice",
        "workspace_id": "ws1",
        "created_at": "2025-01-15T10:30:00Z",
        "metadata": {"role": "admin"},
        "configuration": {}
    })
}

fn session_response() -> serde_json::Value {
    serde_json::json!({
        "id": "sess1",
        "is_active": true,
        "workspace_id": "ws1",
        "metadata": {"env": "test"},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn workspace_response() -> serde_json::Value {
    serde_json::json!({
        "id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn make_honcho(server: &MockServer) -> Honcho {
    Honcho::from_params(
        Honcho::builder()
            .base_url(server.uri())
            .workspace_id("ws1")
            .build(),
    )
    .unwrap()
}

async fn mount_workspace_ensure(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(serde_json::json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_response()))
        .mount(server)
        .await;
}

// ── F4.4.1: peer makes get-or-create POST, returns Peer ───────────

#[tokio::test]
async fn peer_makes_get_or_create_post_returns_peer() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server);

    let expected_body = serde_json::json!({"id": "alice"});

    mount_workspace_ensure(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers"))
        .and(body_json(&expected_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(peer_response()))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    assert_eq!(peer.id(), "alice");
    let meta = peer.metadata().unwrap();
    assert_eq!(meta["role"], "admin");
}

// ── F4.4.3: peer calls ensure_workspace first ──────────────────────

#[tokio::test]
async fn peer_calls_ensure_workspace_first() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server);

    assert_eq!(honcho.workspace_id(), "ws1");

    mount_workspace_ensure(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers"))
        .respond_with(ResponseTemplate::new(200).set_body_json(peer_response()))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    assert_eq!(peer.id(), "alice");
}

// ── F4.4.5: session makes get-or-create POST ───────────────────────

#[tokio::test]
async fn session_makes_get_or_create_post() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server);

    let expected_body = serde_json::json!({"id": "sess1"});

    mount_workspace_ensure(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions"))
        .and(body_json(&expected_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(session_response()))
        .mount(&server)
        .await;

    let session = honcho.session("sess1").await.unwrap();
    assert_eq!(session.id(), "sess1");
    assert!(session.is_active());
    let meta = session.metadata().unwrap();
    assert_eq!(meta["env"], "test");
}
