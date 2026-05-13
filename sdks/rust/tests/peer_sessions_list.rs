//! Wire tests for `Peer::sessions` and `Peer::sessions_with_options`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_borrows_for_generic_args,
    missing_docs
)]

use honcho_ai::Honcho;
use honcho_ai::types::pagination::Page;
use honcho_ai::types::session::{Session, SessionListOptions};
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn workspace_response() -> serde_json::Value {
    serde_json::json!({
        "id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn peer_response() -> serde_json::Value {
    serde_json::json!({
        "id": "alice",
        "workspace_id": "ws1",
        "created_at": "2025-01-15T10:30:00Z",
        "metadata": {},
        "configuration": {}
    })
}

fn session_json(id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "is_active": true,
        "workspace_id": "ws1",
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

#[allow(clippy::needless_pass_by_value)]
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

fn make_honcho(server: &MockServer) -> Honcho {
    Honcho::from_params(
        Honcho::builder()
            .base_url(server.uri())
            .workspace_id("ws1")
            .build(),
    )
    .unwrap()
}

async fn mount_workspace_and_peer(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(serde_json::json!({"id": "ws1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_response()))
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers"))
        .and(body_json(serde_json::json!({"id": "alice"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(peer_response()))
        .mount(server)
        .await;
}

#[tokio::test]
async fn peer_sessions_defaults_no_body() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server);
    mount_workspace_and_peer(&server).await;

    let body = page_json(vec![session_json("s1")], 1, 1, 50, 1);

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/sessions"))
        .and(query_param("page", "1"))
        .and(query_param("size", "50"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    let page: Page<Session> = peer.sessions().await.unwrap();
    assert_eq!(page.items().len(), 1);
    assert_eq!(page.items()[0].id, "s1");
}

#[tokio::test]
async fn peer_sessions_with_options_sends_filters_and_pagination() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server);
    mount_workspace_and_peer(&server).await;

    let body = page_json(vec![session_json("s2")], 1, 2, 10, 1);

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/sessions"))
        .and(body_json(
            serde_json::json!({"filters": {"is_active": true}}),
        ))
        .and(query_param("page", "2"))
        .and(query_param("size", "10"))
        .and(query_param("reverse", "true"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    let opts = SessionListOptions::builder()
        .filters(std::collections::HashMap::from([(
            "is_active".to_string(),
            serde_json::json!(true),
        )]))
        .page(2)
        .size(10)
        .reverse(true)
        .build();
    let page: Page<Session> = peer.sessions_with_options(&opts).await.unwrap();
    assert_eq!(page.items().len(), 1);
    assert_eq!(page.items()[0].id, "s2");
}

#[tokio::test]
async fn peer_sessions_with_options_minimal_body() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server);
    mount_workspace_and_peer(&server).await;

    let body = page_json(vec![], 0, 1, 50, 0);

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/alice/sessions"))
        .and(query_param("page", "1"))
        .and(query_param("size", "50"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let peer = honcho.peer("alice").await.unwrap();
    let opts = SessionListOptions::builder().build();
    let page: Page<Session> = peer.sessions_with_options(&opts).await.unwrap();
    assert_eq!(page.items().len(), 0);
}
