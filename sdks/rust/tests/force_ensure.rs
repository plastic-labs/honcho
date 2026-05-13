#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use honcho_ai::Honcho;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn workspace_json(id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "metadata": {},
        "configuration": {},
        "created_at": "2025-01-15T10:30:00Z"
    })
}

fn honcho(server: &MockServer, workspace_id: &str) -> Honcho {
    Honcho::from_params(
        Honcho::builder()
            .base_url(server.uri())
            .workspace_id(workspace_id)
            .build(),
    )
    .unwrap()
}

#[tokio::test]
async fn force_ensure_idempotent_same_workspace_id() {
    let server = MockServer::start().await;
    let create_body = serde_json::json!({"id": "ws-shared"});

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(&create_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json("ws-shared")))
        .expect(2)
        .mount(&server)
        .await;

    let a = honcho(&server, "ws-shared");
    let b = honcho(&server, "ws-shared");

    a.force_ensure().await.unwrap();
    b.force_ensure().await.unwrap();

    server.verify().await;
}

#[tokio::test]
async fn force_ensure_separate_instances_both_succeed() {
    let server = MockServer::start().await;
    let call_count = Arc::new(AtomicU32::new(0));
    let check = call_count.clone();
    let ws_json = workspace_json("ws-x");

    Mock::given(method("POST"))
        .respond_with(move |_: &Request| {
            call_count.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(&ws_json)
        })
        .mount(&server)
        .await;

    let a = honcho(&server, "ws-x");
    let b = honcho(&server, "ws-x");

    a.force_ensure().await.unwrap();
    b.force_ensure().await.unwrap();

    assert_eq!(check.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn force_ensure_repeated_on_same_instance_hits_server_once() {
    let server = MockServer::start().await;
    let create_body = serde_json::json!({"id": "ws-once"});

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(&create_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json("ws-once")))
        .expect(1)
        .mount(&server)
        .await;

    let honcho = honcho(&server, "ws-once");

    for _ in 0..5 {
        honcho.force_ensure().await.unwrap();
    }

    server.verify().await;
}

#[tokio::test]
async fn force_ensure_server_returns_existing_workspace_still_ok() {
    let server = MockServer::start().await;
    let create_body = serde_json::json!({"id": "ws-exist"});

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(&create_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json("ws-exist")))
        .expect(1)
        .mount(&server)
        .await;

    let honcho = honcho(&server, "ws-exist");

    honcho.force_ensure().await.unwrap();
    honcho.force_ensure().await.unwrap();

    server.verify().await;
}

#[cfg(feature = "blocking")]
#[test]
fn blocking_force_ensure_idempotent() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = rt.block_on(MockServer::start());
    let create_body = serde_json::json!({"id": "ws-blk"});

    rt.block_on(async {
        Mock::given(method("POST"))
            .and(path("/v3/workspaces"))
            .and(body_json(&create_body))
            .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json("ws-blk")))
            .expect(1)
            .mount(&server)
            .await;
    });

    let blocking = honcho_ai::blocking::Honcho::from_params(
        honcho_ai::Honcho::builder()
            .base_url(server.uri())
            .workspace_id("ws-blk")
            .build(),
    )
    .unwrap();

    blocking.force_ensure().unwrap();
    blocking.force_ensure().unwrap();

    rt.block_on(server.verify());
}

#[cfg(feature = "blocking")]
#[test]
fn blocking_force_ensure_separate_instances_both_succeed() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = rt.block_on(MockServer::start());
    let call_count = Arc::new(AtomicU32::new(0));
    let check = call_count.clone();
    let ws_json = workspace_json("ws-blk2");

    rt.block_on(async {
        Mock::given(method("POST"))
            .respond_with(move |_: &Request| {
                call_count.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(200).set_body_json(&ws_json)
            })
            .mount(&server)
            .await;
    });

    let a = honcho_ai::blocking::Honcho::from_params(
        honcho_ai::Honcho::builder()
            .base_url(server.uri())
            .workspace_id("ws-blk2")
            .build(),
    )
    .unwrap();
    let b = honcho_ai::blocking::Honcho::from_params(
        honcho_ai::Honcho::builder()
            .base_url(server.uri())
            .workspace_id("ws-blk2")
            .build(),
    )
    .unwrap();

    a.force_ensure().unwrap();
    b.force_ensure().unwrap();

    assert_eq!(check.load(Ordering::SeqCst), 2);
}
