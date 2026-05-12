//! Wiremock tests for Peer methods (F5.5–F5.7).

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use honcho_ai::Honcho;
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn peer_response() -> serde_json::Value {
    serde_json::json!({
        "id": "alice",
        "workspace_id": "ws1",
        "created_at": "2025-01-15T10:30:00Z",
        "metadata": {},
        "configuration": {}
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

async fn make_honcho(server: &MockServer) -> Honcho {
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
        .and(body_json(&serde_json::json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_response()))
        .mount(server)
        .await;
}

async fn mount_peer_create(server: &MockServer) {
    mount_workspace_ensure(server).await;
    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers"))
        .and(body_json(&serde_json::json!({"id": "alice"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(peer_response()))
        .mount(server)
        .await;
}

// ── F5.5 Representation ────────────────────────────────────────────

#[tokio::test]
async fn peer_representation_basic() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server).await;
    mount_peer_create(&server).await;

    let repr_body = serde_json::json!({});
    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/representation"))
        .and(body_json(&repr_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "representation": "Alice likes cats and Rust."
        })))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    let repr = peer.representation().await.unwrap();
    assert_eq!(repr, "Alice likes cats and Rust.");
}

#[tokio::test]
async fn peer_representation_with_options() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server).await;
    mount_peer_create(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/representation"))
        .and(body_json(&serde_json::json!({
            "session_id": "sess1",
            "target": "bob",
            "search_query": "preferences",
            "search_top_k": 5,
            "search_max_distance": 0.8,
            "include_most_frequent": true,
            "max_conclusions": 20
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "representation": "curated result"
        })))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    let repr = peer
        .representation_builder()
        .session_id("sess1")
        .target("bob")
        .search_query("preferences")
        .search_top_k(5)
        .search_max_distance(0.8)
        .include_most_frequent(true)
        .max_conclusions(20)
        .send()
        .await
        .unwrap();
    assert_eq!(repr, "curated result");
}

#[tokio::test]
async fn peer_representation_validates_search_top_k() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server).await;
    mount_peer_create(&server).await;

    let peer = honcho.peer("alice").await.unwrap();

    let err = peer
        .representation_builder()
        .search_top_k(0)
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, honcho_ai::error::HonchoError::Configuration(ref msg)
            if msg.contains("search_top_k")),
        "expected Configuration error for search_top_k, got {err:?}"
    );

    let err = peer
        .representation_builder()
        .search_top_k(101)
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, honcho_ai::error::HonchoError::Configuration(ref msg)
            if msg.contains("search_top_k")),
        "expected Configuration error for search_top_k=101, got {err:?}"
    );
}

#[tokio::test]
async fn peer_representation_validates_search_max_distance() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server).await;
    mount_peer_create(&server).await;

    let peer = honcho.peer("alice").await.unwrap();

    let err = peer
        .representation_builder()
        .search_max_distance(1.5)
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, honcho_ai::error::HonchoError::Configuration(ref msg)
            if msg.contains("search_max_distance")),
        "expected Configuration error for search_max_distance, got {err:?}"
    );

    let err = peer
        .representation_builder()
        .search_max_distance(-0.1)
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, honcho_ai::error::HonchoError::Configuration(ref msg)
            if msg.contains("search_max_distance")),
        "expected Configuration error for negative search_max_distance, got {err:?}"
    );
}

#[tokio::test]
async fn peer_representation_validates_max_conclusions() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server).await;
    mount_peer_create(&server).await;

    let peer = honcho.peer("alice").await.unwrap();

    let err = peer
        .representation_builder()
        .max_conclusions(0)
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, honcho_ai::error::HonchoError::Configuration(ref msg)
            if msg.contains("max_conclusions")),
        "expected Configuration error for max_conclusions=0, got {err:?}"
    );

    let err = peer
        .representation_builder()
        .max_conclusions(101)
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, honcho_ai::error::HonchoError::Configuration(ref msg)
            if msg.contains("max_conclusions")),
        "expected Configuration error for max_conclusions=101, got {err:?}"
    );
}

// ── F5.6 Context ───────────────────────────────────────────────────

#[tokio::test]
async fn peer_context_returns_peer_context() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server).await;
    mount_peer_create(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/peers/alice/context"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "peer_id": "alice",
            "target_id": "alice",
            "representation": "Alice is curious.",
            "peer_card": ["friendly", "inquisitive"]
        })))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    let ctx = peer.context().await.unwrap();
    assert_eq!(ctx.peer_id, "alice");
    assert_eq!(ctx.target_id, "alice");
    assert_eq!(ctx.representation.as_deref(), Some("Alice is curious."));
    assert_eq!(
        ctx.peer_card.as_deref(),
        Some(&["friendly".to_owned(), "inquisitive".to_owned()][..])
    );
}

#[tokio::test]
async fn peer_context_with_target_sends_query() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server).await;
    mount_peer_create(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/peers/alice/context"))
        .and(query_param("target", "bob"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "peer_id": "alice",
            "target_id": "bob",
            "representation": "Bob is helpful.",
            "peer_card": null
        })))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    let ctx = peer.context_with_target("bob").await.unwrap();
    assert_eq!(ctx.peer_id, "alice");
    assert_eq!(ctx.target_id, "bob");
    assert_eq!(ctx.representation.as_deref(), Some("Bob is helpful."));
    assert!(ctx.peer_card.is_none());
}

// ── F5.7 Sessions ──────────────────────────────────────────────────

#[tokio::test]
async fn peer_sessions_returns_paginated() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server).await;
    mount_peer_create(&server).await;

    let session1 = serde_json::json!({
        "id": "s1",
        "is_active": true,
        "workspace_id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    });
    let session2 = serde_json::json!({
        "id": "s2",
        "is_active": false,
        "workspace_id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-16T10:30:00Z"
    });

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/sessions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "items": [session1, session2],
            "total": 2,
            "page": 1,
            "size": 50,
            "pages": 1
        })))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    let page = peer.sessions().await.unwrap();
    assert_eq!(page.total(), 2);
    assert_eq!(page.pages(), 1);
    let items = page.items();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].id, "s1");
    assert!(items[0].is_active);
    assert_eq!(items[1].id, "s2");
    assert!(!items[1].is_active);
}
