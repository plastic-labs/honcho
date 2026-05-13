#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use honcho_ai::client::Honcho;
use honcho_ai::types::dream::QueueStatus;
use honcho_ai::types::message::MessageResponse;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn workspace_json() -> serde_json::Value {
    serde_json::json!({
        "id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

async fn make_honcho(server: &MockServer, workspace_id: &str) -> Honcho {
    Honcho::from_params(
        Honcho::builder()
            .base_url(server.uri())
            .workspace_id(workspace_id)
            .build(),
    )
    .unwrap()
}

fn message_json(id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "content": "hello world",
        "peer_id": "alice",
        "session_id": "sess1",
        "created_at": "2025-01-15T10:30:00Z",
        "metadata": {},
        "workspace_id": "ws1",
        "token_count": 2
    })
}

async fn mount_ensure_workspace(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(&serde_json::json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json()))
        .mount(server)
        .await;
}

#[tokio::test]
async fn search_returns_messages() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server, "ws1").await;

    mount_ensure_workspace(&server).await;

    let search_body = serde_json::json!({
        "query": "hello",
        "filters": null,
        "limit": 10
    });

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/search"))
        .and(body_json(&search_body))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!([message_json("m1"), message_json("m2")])),
        )
        .mount(&server)
        .await;

    let results: Vec<MessageResponse> = honcho.search("hello").await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "m1");
    assert_eq!(results[1].id, "m2");
}

#[tokio::test]
async fn queue_status_returns_status() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server, "ws1").await;

    mount_ensure_workspace(&server).await;

    let response_body = serde_json::json!({
        "total_work_units": 10,
        "completed_work_units": 8,
        "in_progress_work_units": 1,
        "pending_work_units": 1,
        "sessions": {
            "sess1": {
                "session_id": "sess1",
                "total_work_units": 5,
                "completed_work_units": 4,
                "in_progress_work_units": 1,
                "pending_work_units": 0
            }
        }
    });

    Mock::given(method("GET"))
        .and(path("/v3/workspaces/ws1/queue/status"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&server)
        .await;

    let status: QueueStatus = honcho.queue_status().await.unwrap();
    assert_eq!(status.total_work_units, 10);
    assert_eq!(status.completed_work_units, 8);
    assert_eq!(status.in_progress_work_units, 1);
    assert_eq!(status.pending_work_units, 1);
    assert!(status.sessions.is_some());
}

#[tokio::test]
async fn schedule_dream_posts_correct_body() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server, "ws1").await;

    mount_ensure_workspace(&server).await;

    let expected_body = serde_json::json!({
        "observer": "alice",
        "observed": "alice",
        "session_id": null,
        "dream_type": "omni"
    });

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/schedule_dream"))
        .and(body_json(&expected_body))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    honcho.schedule_dream("alice").await.unwrap();
}

#[tokio::test]
async fn delete_workspace_calls_delete() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server, "ws1").await;

    Mock::given(method("DELETE"))
        .and(path("/v3/workspaces/ws_to_delete"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    honcho.delete_workspace("ws_to_delete").await.unwrap();
}
