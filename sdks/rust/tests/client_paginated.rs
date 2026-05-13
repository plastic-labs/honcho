#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value,
    missing_docs
)]

use honcho_ai::client::Honcho;
use honcho_ai::types::pagination::Page;
use honcho_ai::types::peer::Peer;
use honcho_ai::types::session::Session;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn peer_json(id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
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

fn workspace_json(id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
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

fn make_honcho(server: &MockServer, workspace_id: &str) -> Honcho {
    Honcho::from_params(
        Honcho::builder()
            .base_url(server.uri())
            .workspace_id(workspace_id)
            .build(),
    )
    .unwrap()
}

fn ws_json(id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

async fn mock_ensure_workspace(server: &MockServer, workspace_id: &str) {
    use wiremock::matchers::{body_json, method, path};
    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(serde_json::json!({"id": workspace_id})))
        .respond_with(ResponseTemplate::new(200).set_body_json(ws_json(workspace_id)))
        .mount(server)
        .await;
}

#[tokio::test]
async fn peers_returns_paginated_typed() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server, "ws1");
    mock_ensure_workspace(&server, "ws1").await;

    let body = page_json(vec![peer_json("alice"), peer_json("bob")], 2, 1, 50, 1);

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/list"))
        .and(query_param("page", "1"))
        .and(query_param("size", "50"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let page: Page<Peer> = honcho.peers().await.unwrap();
    assert_eq!(page.items().len(), 2);
    assert_eq!(page.items()[0].id, "alice");
    assert_eq!(page.items()[1].id, "bob");
    assert_eq!(page.total(), 2);
    assert_eq!(page.page(), 1);
    assert!(!page.has_next());
}

#[tokio::test]
async fn peers_default_page_and_size() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server, "ws1");
    mock_ensure_workspace(&server, "ws1").await;

    let body = page_json(vec![peer_json("alice")], 1, 1, 50, 1);

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/peers/list"))
        .and(query_param("page", "1"))
        .and(query_param("size", "50"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let page = honcho.peers().await.unwrap();
    assert_eq!(page.page(), 1);
    assert_eq!(page.size(), 50);
}

#[tokio::test]
async fn sessions_returns_paginated() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server, "ws1");
    mock_ensure_workspace(&server, "ws1").await;

    let body = page_json(vec![session_json("s1"), session_json("s2")], 2, 1, 50, 1);

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/ws1/sessions/list"))
        .and(query_param("page", "1"))
        .and(query_param("size", "50"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let page: Page<Session> = honcho.sessions().await.unwrap();
    assert_eq!(page.items().len(), 2);
    assert_eq!(page.items()[0].id, "s1");
    assert_eq!(page.items()[1].id, "s2");
    assert_eq!(page.total(), 2);
}

#[tokio::test]
async fn workspaces_returns_ids() {
    let server = MockServer::start().await;
    let honcho = make_honcho(&server, "ws1");

    let body = page_json(
        vec![workspace_json("ws_abc"), workspace_json("ws_def")],
        2,
        1,
        50,
        1,
    );

    Mock::given(method("POST"))
        .and(path("/v3/workspaces/list"))
        .and(query_param("page", "1"))
        .and(query_param("size", "50"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let page = honcho.workspaces().await.unwrap();
    let ids = page.items();
    assert_eq!(ids.len(), 2);
    assert_eq!(ids[0], "ws_abc");
    assert_eq!(ids[1], "ws_def");
}
