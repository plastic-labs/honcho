//! Integration tests for Peer core, chat, search, and card methods (Phase 5).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_pass_by_value,
    clippy::needless_borrows_for_generic_args,
    clippy::unused_async,
    clippy::items_after_statements,
    missing_docs
)]

use std::collections::HashMap;

use honcho_ai::Honcho;
use honcho_ai::Peer;
use honcho_ai::PeerConfig;
use serde_json::{Value, json};
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn peer_response_json() -> Value {
    json!({
        "id": "alice",
        "workspace_id": "ws1",
        "created_at": "2025-01-15T10:30:00Z",
        "metadata": {"role": "admin"},
        "configuration": {"observe_me": true}
    })
}

fn peer_response_with(metadata: Value, configuration: Value) -> Value {
    json!({
        "id": "alice",
        "workspace_id": "ws1",
        "created_at": "2025-01-15T10:30:00Z",
        "metadata": metadata,
        "configuration": configuration
    })
}

fn workspace_response_json() -> Value {
    json!({
        "id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn make_honcho(server: &MockServer) -> Honcho {
    Honcho::new(&server.uri(), "ws1").unwrap()
}

async fn make_peer(server: &MockServer) -> Peer {
    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(&json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_response_json()))
        .up_to_n_times(1)
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers"))
        .and(body_json(&json!({"id": "alice"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(peer_response_json()))
        .up_to_n_times(1)
        .mount(server)
        .await;

    let honcho = make_honcho(server);
    honcho.peer("alice").await.unwrap()
}

// ── F5.1: Construction + Metadata ──────────────────────────────────────

#[tokio::test]
async fn peer_refresh_updates_caches() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let updated = peer_response_with(
        json!({"role": "user", "level": 5}),
        json!({"observe_me": false, "observe_others": true}),
    );

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&updated))
        .mount(&server)
        .await;

    peer.refresh().await.unwrap();

    let meta = peer.metadata().unwrap();
    assert_eq!(meta.get("role").unwrap(), "user");
    assert_eq!(meta.get("level").unwrap(), 5);

    let config = peer.configuration().unwrap();
    assert_eq!(config.observe_me, Some(false));
    assert_eq!(config.observe_others, Some(true));
}

#[tokio::test]
async fn peer_get_metadata_returns_from_cache() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let updated = peer_response_with(json!({"k": "v"}), json!({}));

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&updated))
        .mount(&server)
        .await;

    let meta = peer.get_metadata().await.unwrap();
    assert_eq!(meta.get("k").unwrap(), "v");
}

#[tokio::test]
async fn peer_set_metadata_puts_to_peer_endpoint() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let mut new_meta = HashMap::new();
    new_meta.insert("updated".to_owned(), json!(true));

    let resp = peer_response_with(json!({"updated": true}), json!({"language": "en"}));

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/peers/alice"))
        .and(body_json(&json!({"metadata": {"updated": true}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(&resp))
        .mount(&server)
        .await;

    peer.set_metadata(new_meta).await.unwrap();

    let cached = peer.metadata().unwrap();
    assert_eq!(cached.get("updated").unwrap(), true);
}

#[tokio::test]
async fn peer_set_configuration_puts_to_peer_endpoint() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let new_config: PeerConfig = serde_json::from_value(json!({"observe_me": true})).unwrap();

    let resp = peer_response_with(json!({"role": "admin"}), json!({"observe_me": true}));

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/peers/alice"))
        .and(body_json(&json!({"configuration": {"observe_me": true}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(&resp))
        .mount(&server)
        .await;

    peer.set_configuration(&new_config).await.unwrap();

    let cached = peer.configuration().unwrap();
    assert_eq!(cached.observe_me, Some(true));
}

#[tokio::test]
async fn peer_get_configuration_returns_from_cache() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let updated = peer_response_with(
        json!({}),
        json!({"observe_me": true, "observe_others": false}),
    );

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&updated))
        .mount(&server)
        .await;

    let config = peer.get_configuration().await.unwrap();
    assert_eq!(config.observe_me, Some(true));
    assert_eq!(config.observe_others, Some(false));
}

// ── F5.2: Chat (non-streaming) ────────────────────────────────────────

#[tokio::test]
async fn peer_chat_basic_query() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/chat"))
        .and(body_json(&json!({"query": "hello"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": "Hi there!"
        })))
        .mount(&server)
        .await;

    let result = peer.chat("hello").await.unwrap();
    assert_eq!(result, Some("Hi there!".to_owned()));
}

#[tokio::test]
async fn peer_chat_empty_content_returns_none() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": null
        })))
        .mount(&server)
        .await;

    let result = peer.chat("hello").await.unwrap();
    assert_eq!(result, None);
}

#[tokio::test]
async fn peer_chat_empty_string_content_returns_none() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": ""
        })))
        .mount(&server)
        .await;

    let result = peer.chat("hello").await.unwrap();
    assert_eq!(result, None);
}

#[tokio::test]
async fn peer_chat_validates_empty_query() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let err = peer.chat("").await.unwrap_err();
    assert_eq!(err.code(), "validation_error");
}

// ── F5.3: Search ──────────────────────────────────────────────────────

#[tokio::test]
async fn peer_search_returns_messages() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/search"))
        .and(body_json(&serde_json::json!({
            "query": "hello",
            "limit": 10
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {
                "id": "msg1",
                "content": "hello world",
                "peer_id": "alice",
                "session_id": "sess1",
                "metadata": {},
                "created_at": "2025-01-15T10:30:00Z",
                "workspace_id": "ws1",
                "token_count": 2
            }
        ])))
        .mount(&server)
        .await;

    let results = peer.search("hello").await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id(), "msg1");
    assert_eq!(results[0].content(), "hello world");
}

#[tokio::test]
async fn peer_search_returns_empty_vec() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
        .mount(&server)
        .await;

    let results = peer.search("test").await.unwrap();
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn peer_chat_with_session_and_target() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": "Bob likes Rust"
        })))
        .mount(&server)
        .await;

    use honcho_ai::types::dialectic::DialecticOptions;
    let options = DialecticOptions::builder()
        .query("what do you know?")
        .stream(false)
        .session_id("sess1")
        .target("bob")
        .build();

    let result = peer.chat_with_options(&options).await.unwrap();
    assert_eq!(result, Some("Bob likes Rust".to_owned()));
}

// ── F5.4: Card ────────────────────────────────────────────────────────

#[tokio::test]
async fn peer_get_card_returns_vec() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/peers/alice/card"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "peer_card": ["fact1", "fact2"]
        })))
        .mount(&server)
        .await;

    let card = peer.get_card().await.unwrap();
    assert_eq!(card, Some(vec!["fact1".to_string(), "fact2".to_string()]));
}

#[tokio::test]
async fn peer_get_card_with_target_sends_query() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/peers/alice/card"))
        .and(query_param("target", "bob"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "peer_card": ["knows bob"]
        })))
        .mount(&server)
        .await;

    let card = peer.get_card_with_target("bob").await.unwrap();
    assert_eq!(card, Some(vec!["knows bob".to_string()]));
}

#[tokio::test]
async fn peer_get_card_none_when_empty() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/peers/alice/card"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "peer_card": null
        })))
        .mount(&server)
        .await;

    let card = peer.get_card().await.unwrap();
    assert_eq!(card, None);
}

#[tokio::test]
async fn peer_set_card_puts_card() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/peers/alice/card"))
        .and(body_json(&json!({
            "peer_card": ["new fact"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "peer_card": ["new fact"]
        })))
        .mount(&server)
        .await;

    let card = peer.set_card(vec!["new fact".to_string()]).await.unwrap();
    assert_eq!(card, Some(vec!["new fact".to_string()]));
}

#[tokio::test]
async fn peer_set_card_with_target_sends_query() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/peers/alice/card"))
        .and(query_param("target", "bob"))
        .and(body_json(&json!({
            "peer_card": ["fact about bob"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "peer_card": ["fact about bob"]
        })))
        .mount(&server)
        .await;

    let card = peer
        .set_card_with_target(vec!["fact about bob".to_string()], "bob")
        .await
        .unwrap();
    assert_eq!(card, Some(vec!["fact about bob".to_string()]));
}

// ── F5.8: Message builder tests ───────────────────────────────────────

#[tokio::test]
async fn peer_message_builder_does_not_call_api() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let _msg = peer.message("hello").build().unwrap();
}

#[tokio::test]
async fn peer_message_builder_fields() {
    use honcho_ai::types::message::MessageConfiguration;

    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let msg = peer
        .message("hello")
        .metadata(HashMap::from([("k".to_string(), json!("v"))]))
        .configuration(MessageConfiguration::default())
        .build()
        .unwrap();

    assert_eq!(msg.peer_id, "alice");
    assert_eq!(msg.content, "hello");
    assert_eq!(msg.metadata.as_ref().unwrap().get("k").unwrap(), "v");
}

#[tokio::test]
async fn peer_message_whitespace_only_is_ok() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let msg = peer.message("   ").build().unwrap();
    assert_eq!(msg.content, "   ");
}

#[tokio::test]
async fn peer_message_empty_string_is_ok() {
    let server = MockServer::start().await;
    let peer = make_peer(&server).await;

    let msg = peer.message("").build().unwrap();
    assert_eq!(msg.content, "");
}
