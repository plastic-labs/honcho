//! Integration tests for Session core: F6.1–F6.3.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_pass_by_value,
    clippy::needless_borrows_for_generic_args,
    clippy::unused_async,
    missing_docs
)]

use std::collections::HashMap;

use honcho_ai::Honcho;
use honcho_ai::Session;
use honcho_ai::session::PeerSpec;
use honcho_ai::types::session::SessionPeerConfig;
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
        "metadata": {"topic": "test"},
        "configuration": {"model": "gpt-4"},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn session_response_with(metadata: Value, configuration: Value) -> Value {
    json!({
        "id": "sess1",
        "workspace_id": "ws1",
        "is_active": true,
        "metadata": metadata,
        "configuration": configuration,
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn peer_config(observe_me: Option<bool>, observe_others: Option<bool>) -> SessionPeerConfig {
    serde_json::from_value(json!({
        "observe_me": observe_me,
        "observe_others": observe_others
    }))
    .unwrap()
}

fn make_honcho(server: &MockServer) -> Honcho {
    Honcho::new(&server.uri(), "ws1").unwrap()
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

    let honcho = make_honcho(server);
    honcho.session("sess1").await.unwrap()
}

// ── F6.1: Construction + Metadata/Config CRUD ────────────────────────

#[tokio::test]
async fn session_id_returns_id() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;
    assert_eq!(session.id(), "sess1");
}

#[tokio::test]
async fn session_is_active_returns_true() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;
    assert!(session.is_active());
}

#[tokio::test]
async fn session_metadata_returns_cached() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;
    let meta = session.metadata().unwrap();
    assert_eq!(meta.get("topic").unwrap(), "test");
}

#[tokio::test]
async fn session_configuration_returns_cached() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;
    let config = session.configuration().unwrap();
    assert_eq!(config.get("model").unwrap(), "gpt-4");
}

#[tokio::test]
async fn session_refresh_updates_caches() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let updated = session_response_with(
        json!({"topic": "updated", "priority": 1}),
        json!({"model": "claude"}),
    );

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&updated))
        .mount(&server)
        .await;

    session.refresh().await.unwrap();

    let meta = session.metadata().unwrap();
    assert_eq!(meta.get("topic").unwrap(), "updated");
    assert_eq!(meta.get("priority").unwrap(), 1);

    let config = session.configuration().unwrap();
    assert_eq!(config.get("model").unwrap(), "claude");
}

#[tokio::test]
async fn session_get_metadata_returns_from_cache() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let updated = session_response_with(json!({"k": "v"}), json!({}));

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&updated))
        .mount(&server)
        .await;

    let meta = session.get_metadata().await.unwrap();
    assert_eq!(meta.get("k").unwrap(), "v");
}

#[tokio::test]
async fn session_set_metadata_puts_to_session_endpoint() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let mut new_meta = HashMap::new();
    new_meta.insert("updated".to_owned(), json!(true));

    let resp = session_response_with(json!({"updated": true}), json!({"model": "gpt-4"}));

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/sessions/sess1"))
        .and(body_json(json!({"metadata": {"updated": true}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(&resp))
        .mount(&server)
        .await;

    session.set_metadata(new_meta).await.unwrap();

    let cached = session.metadata().unwrap();
    assert_eq!(cached.get("updated").unwrap(), true);
}

#[tokio::test]
async fn session_get_configuration_returns_from_cache() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let updated = session_response_with(json!({}), json!({"theme": "dark"}));

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&updated))
        .mount(&server)
        .await;

    let config = session.get_configuration().await.unwrap();
    assert_eq!(config.get("theme").unwrap(), "dark");
}

#[tokio::test]
async fn session_set_configuration_puts_to_session_endpoint() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let mut new_config = HashMap::new();
    new_config.insert("mode".to_owned(), json!("fast"));

    let resp = session_response_with(json!({"topic": "test"}), json!({"mode": "fast"}));

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/sessions/sess1"))
        .and(body_json(json!({"configuration": {"mode": "fast"}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(&resp))
        .mount(&server)
        .await;

    session.set_configuration(new_config).await.unwrap();

    let cached = session.configuration().unwrap();
    assert_eq!(cached.get("mode").unwrap(), "fast");
}

// ── F6.2: Peer Management ────────────────────────────────────────────

#[tokio::test]
async fn session_add_peer_posts_to_session_peers() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(json!({"alice": {}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;

    session.add_peer("alice").await.unwrap();
}

#[tokio::test]
async fn session_add_peers_with_config() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let cfg = peer_config(Some(true), Some(false));

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(json!({
            "alice": {"observe_me": true, "observe_others": false}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;

    session
        .add_peers([PeerSpec::WithConfig("alice".to_owned(), cfg)])
        .await
        .unwrap();
}

#[tokio::test]
async fn session_set_peers_puts_to_session_peers() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(json!({"bob": {}, "carol": {}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;

    session.set_peers(["bob", "carol"]).await.unwrap();
}

#[tokio::test]
async fn session_remove_peers_deletes_with_json_array_body() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("DELETE"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(json!(["alice", "bob"])))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    session.remove_peers(["alice", "bob"]).await.unwrap();
}

#[tokio::test]
async fn session_peers_flattens_paginated_response() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "items": [
                {
                    "id": "alice",
                    "workspace_id": "ws1",
                    "created_at": "2025-01-15T10:30:00Z",
                    "metadata": {},
                    "configuration": {}
                },
                {
                    "id": "bob",
                    "workspace_id": "ws1",
                    "created_at": "2025-01-15T10:30:00Z",
                    "metadata": {},
                    "configuration": {}
                }
            ],
            "total": 2,
            "page": 1,
            "size": 50,
            "pages": 1
        })))
        .mount(&server)
        .await;

    let peers = session.peers().await.unwrap();
    assert_eq!(peers.len(), 2);
    assert_eq!(peers[0].id(), "alice");
    assert_eq!(peers[1].id(), "bob");
}

// ── PeerSpec From impls ──────────────────────────────────────────────

#[tokio::test]
async fn peer_spec_from_str() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(json!({"alice": {}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;

    session.add_peers(["alice"]).await.unwrap();
}

#[tokio::test]
async fn peer_spec_from_string() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(json!({"alice": {}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;

    session
        .add_peers([PeerSpec::from(String::from("alice"))])
        .await
        .unwrap();
}

#[tokio::test]
async fn peer_spec_from_tuple_str_config() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let cfg = peer_config(Some(true), None);

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers"))
        .and(body_json(json!({
            "alice": {"observe_me": true}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;

    session
        .add_peers([PeerSpec::from(("alice", cfg))])
        .await
        .unwrap();
}

// ── F6.3: Per-peer configuration ─────────────────────────────────────

#[tokio::test]
async fn session_get_peer_configuration_gets_config() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers/alice/config"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "observe_me": true,
            "observe_others": false
        })))
        .mount(&server)
        .await;

    let cfg = session.get_peer_configuration("alice").await.unwrap();
    assert_eq!(cfg.observe_me, Some(true));
    assert_eq!(cfg.observe_others, Some(false));
}

#[tokio::test]
async fn session_set_peer_configuration_puts_config() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let cfg = peer_config(Some(true), Some(false));

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/peers/alice/config"))
        .and(body_json(json!({
            "observe_me": true,
            "observe_others": false
        })))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    session.set_peer_configuration("alice", &cfg).await.unwrap();
}
