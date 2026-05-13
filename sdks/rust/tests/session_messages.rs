//! Integration tests for Session messages, delete, clone, and message ops (F6.4–F6.5).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_borrows_for_generic_args,
    clippy::unused_async,
    missing_docs
)]

use std::collections::HashMap;

use honcho_ai::Honcho;
use honcho_ai::session::Session;
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
        "is_active": true,
        "workspace_id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn cloned_session_response_json() -> Value {
    json!({
        "id": "sess2",
        "is_active": true,
        "workspace_id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T11:00:00Z"
    })
}

fn message_json(id: &str, content: &str, peer_id: &str) -> Value {
    json!({
        "id": id,
        "content": content,
        "peer_id": peer_id,
        "session_id": "sess1",
        "metadata": {},
        "created_at": "2025-01-15T10:30:00Z",
        "workspace_id": "ws1",
        "token_count": 2
    })
}

fn make_honcho(server: &MockServer) -> Honcho {
    Honcho::new(&server.uri(), "ws1").unwrap()
}

async fn make_session(server: &MockServer) -> Session {
    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(&json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_response_json()))
        .up_to_n_times(1)
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions"))
        .and(body_json(&json!({"id": "sess1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(session_response_json()))
        .up_to_n_times(1)
        .mount(server)
        .await;

    let honcho = make_honcho(server);
    honcho.session("sess1").await.unwrap()
}

// ── F6.4: add_messages ─────────────────────────────────────────────────

#[tokio::test]
async fn add_messages_single_message() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let msg_body = json!({
        "messages": [{
            "content": "hello",
            "peer_id": "alice"
        }]
    });

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages"))
        .and(body_json(&msg_body))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!([message_json("msg1", "hello", "alice")])),
        )
        .mount(&server)
        .await;

    let messages = vec![honcho_ai::types::message::MessageCreate {
        content: "hello".to_string(),
        peer_id: "alice".to_string(),
        metadata: None,
        configuration: None,
        created_at: None,
    }];

    let result = session.add_messages(messages).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id(), "msg1");
    assert_eq!(result[0].content(), "hello");
}

#[tokio::test]
async fn add_messages_batch_under_100_one_request() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let mut msgs = Vec::new();
    let mut expected_response = Vec::new();
    for i in 0..3 {
        msgs.push(honcho_ai::types::message::MessageCreate {
            content: format!("msg{i}"),
            peer_id: "alice".to_string(),
            metadata: None,
            configuration: None,
            created_at: None,
        });
        expected_response.push(message_json(&format!("m{i}"), &format!("msg{i}"), "alice"));
    }

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(expected_response))
        .expect(1)
        .mount(&server)
        .await;

    let result = session.add_messages(msgs).await.unwrap();
    assert_eq!(result.len(), 3);
}

#[tokio::test]
async fn add_messages_exactly_100_is_one_request() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let mut msgs = Vec::new();
    let mut expected_response = Vec::new();
    for i in 0..100 {
        msgs.push(honcho_ai::types::message::MessageCreate {
            content: format!("msg{i}"),
            peer_id: "alice".to_string(),
            metadata: None,
            configuration: None,
            created_at: None,
        });
        expected_response.push(message_json(&format!("m{i}"), &format!("msg{i}"), "alice"));
    }

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(expected_response))
        .expect(1)
        .mount(&server)
        .await;

    let result = session.add_messages(msgs).await.unwrap();
    assert_eq!(result.len(), 100);
    assert_eq!(result[0].id(), "m0");
    assert_eq!(result[99].id(), "m99");
}

#[tokio::test]
async fn add_messages_101_is_two_requests() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let mut msgs = Vec::new();
    let mut response_chunk1 = Vec::new();
    let mut response_chunk2 = Vec::new();

    for i in 0..101 {
        msgs.push(honcho_ai::types::message::MessageCreate {
            content: format!("msg{i}"),
            peer_id: "alice".to_string(),
            metadata: None,
            configuration: None,
            created_at: None,
        });
        if i < 100 {
            response_chunk1.push(message_json(&format!("m{i}"), &format!("msg{i}"), "alice"));
        } else {
            response_chunk2.push(message_json(&format!("m{i}"), &format!("msg{i}"), "alice"));
        }
    }

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_chunk1))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_chunk2))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    let result = session.add_messages(msgs).await.unwrap();
    assert_eq!(result.len(), 101);
    assert_eq!(result[0].id(), "m0");
    assert_eq!(result[100].id(), "m100");
}

#[tokio::test]
async fn add_messages_batch_over_100_chunks() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let mut msgs = Vec::new();
    let mut response_chunk1 = Vec::new();
    let mut response_chunk2 = Vec::new();

    for i in 0..150 {
        msgs.push(honcho_ai::types::message::MessageCreate {
            content: format!("msg{i}"),
            peer_id: "alice".to_string(),
            metadata: None,
            configuration: None,
            created_at: None,
        });
        if i < 100 {
            response_chunk1.push(message_json(&format!("m{i}"), &format!("msg{i}"), "alice"));
        } else {
            response_chunk2.push(message_json(&format!("m{i}"), &format!("msg{i}"), "alice"));
        }
    }

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_chunk1))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_chunk2))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    let result = session.add_messages(msgs).await.unwrap();
    assert_eq!(result.len(), 150);
    assert_eq!(result[0].id(), "m0");
    assert_eq!(result[149].id(), "m149");
}

// ── F6.4: messages (paginated) ─────────────────────────────────────────

#[tokio::test]
async fn messages_paginated() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let page_response = json!({
        "items": [
            message_json("msg1", "hello", "alice"),
            message_json("msg2", "world", "bob")
        ],
        "total": 2,
        "page": 1,
        "size": 50,
        "pages": 1
    });

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages/list"))
        .respond_with(ResponseTemplate::new(200).set_body_json(page_response))
        .mount(&server)
        .await;

    let page = session.messages().await.unwrap();
    assert_eq!(page.total(), 2);
    let items = page.items();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].id(), "msg1");
    assert_eq!(items[1].id(), "msg2");
}

// ── F6.5: delete ───────────────────────────────────────────────────────

#[tokio::test]
async fn delete_calls_delete_endpoint() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("DELETE"))
        .and(path("/v3/workspaces/ws1/sessions/sess1"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    session.delete().await.unwrap();
}

// ── F6.5: clone_session ───────────────────────────────────────────────

#[tokio::test]
async fn clone_session_returns_new_session() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/clone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(cloned_session_response_json()))
        .mount(&server)
        .await;

    let cloned = session.clone_session().await.unwrap();
    assert_eq!(cloned.id(), "sess2");
    assert!(cloned.is_active());
}

// ── F6.5: clone_session_with_message ───────────────────────────────────

#[tokio::test]
async fn clone_session_with_message_id_sends_query() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/clone"))
        .and(query_param("message_id", "msg42"))
        .respond_with(ResponseTemplate::new(200).set_body_json(cloned_session_response_json()))
        .mount(&server)
        .await;

    let cloned = session.clone_session_with_message("msg42").await.unwrap();
    assert_eq!(cloned.id(), "sess2");
}

// ── F6.5: get_message ──────────────────────────────────────────────────

#[tokio::test]
async fn get_message_returns_message() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages/msg99"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(message_json("msg99", "found it", "alice")),
        )
        .mount(&server)
        .await;

    let msg = session.get_message("msg99").await.unwrap();
    assert_eq!(msg.id(), "msg99");
    assert_eq!(msg.content(), "found it");
}

// ── F6.5: update_message ───────────────────────────────────────────────

#[tokio::test]
async fn update_message_puts_metadata() {
    let server = MockServer::start().await;
    let session = make_session(&server).await;

    let updated_msg = json!({
        "id": "msg1",
        "content": "hello",
        "peer_id": "alice",
        "session_id": "sess1",
        "metadata": {"tagged": true},
        "created_at": "2025-01-15T10:30:00Z",
        "workspace_id": "ws1",
        "token_count": 2
    });

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/ws1/sessions/sess1/messages/msg1"))
        .and(body_json(&json!({"metadata": {"tagged": true}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(updated_msg))
        .mount(&server)
        .await;

    let mut meta = HashMap::new();
    meta.insert("tagged".to_string(), json!(true));

    let msg = session.update_message("msg1", meta).await.unwrap();
    assert_eq!(msg.id(), "msg1");
    assert_eq!(msg.metadata().get("tagged").unwrap(), true);
}
