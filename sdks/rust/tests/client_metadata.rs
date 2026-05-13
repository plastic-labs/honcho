#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value,
    missing_docs
)]

use std::collections::HashMap;

use honcho_ai::client::Honcho;
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn workspace_response(
    metadata: serde_json::Value,
    configuration: serde_json::Value,
) -> serde_json::Value {
    json!({
        "id": "test-ws",
        "metadata": metadata,
        "configuration": configuration,
        "created_at": "2025-01-15T10:30:00Z"
    })
}

#[tokio::test]
async fn get_metadata_posts_to_workspaces_with_id() {
    let server = MockServer::start().await;

    let metadata = json!({"env": "production", "team": "core"});
    let response = workspace_response(metadata, json!({}));

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(json!({"id": "test-ws"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let honcho = Honcho::new(&server.uri(), "test-ws").unwrap();
    let result = honcho.get_metadata().await.unwrap();

    assert_eq!(result.get("env").unwrap().as_str(), Some("production"));
    assert_eq!(result.get("team").unwrap().as_str(), Some("core"));
}

#[tokio::test]
async fn get_metadata_empty_when_no_metadata() {
    let server = MockServer::start().await;

    let response = workspace_response(json!({}), json!({}));

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let honcho = Honcho::new(&server.uri(), "test-ws").unwrap();
    let result = honcho.get_metadata().await.unwrap();

    assert!(result.is_empty());
}

#[tokio::test]
async fn set_metadata_puts_to_workspace_id() {
    let server = MockServer::start().await;

    let metadata = json!({"env": "staging"});
    let response = workspace_response(metadata.clone(), json!({}));

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/test-ws"))
        .and(body_json(json!({"metadata": metadata})))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let honcho = Honcho::new(&server.uri(), "test-ws").unwrap();

    let mut meta = HashMap::new();
    meta.insert("env".to_string(), json!("staging"));
    honcho.set_metadata(meta).await.unwrap();
}

#[tokio::test]
async fn set_metadata_server_error_returns_error() {
    let server = MockServer::start().await;

    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let honcho = Honcho::new(&server.uri(), "test-ws").unwrap();

    let mut meta = HashMap::new();
    meta.insert("env".to_string(), json!("staging"));
    let err = honcho.set_metadata(meta).await.unwrap_err();
    assert!(
        matches!(
            err,
            honcho_ai::error::HonchoError::Server { status: 503, .. }
        ),
        "expected Server(503), got {err:?}"
    );
}

#[tokio::test]
async fn get_configuration_posts_to_workspaces_with_id() {
    let server = MockServer::start().await;

    let config = json!({"reasoning": {"enabled": true}});
    let response = workspace_response(json!({}), config);

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .and(body_json(json!({"id": "test-ws"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let honcho = Honcho::new(&server.uri(), "test-ws").unwrap();
    let result = honcho.get_configuration().await.unwrap();

    assert_eq!(
        result
            .get("reasoning")
            .unwrap()
            .get("enabled")
            .unwrap()
            .as_bool(),
        Some(true)
    );
}

#[tokio::test]
async fn get_configuration_empty_when_no_configuration() {
    let server = MockServer::start().await;

    let response = workspace_response(json!({}), json!({}));

    Mock::given(method("POST"))
        .and(path("/v3/workspaces"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let honcho = Honcho::new(&server.uri(), "test-ws").unwrap();
    let result = honcho.get_configuration().await.unwrap();

    assert!(result.is_empty());
}

#[tokio::test]
async fn set_configuration_puts_to_workspace_id() {
    let server = MockServer::start().await;

    let config = json!({"reasoning": {"enabled": false}});
    let response = workspace_response(json!({}), config.clone());

    Mock::given(method("PUT"))
        .and(path("/v3/workspaces/test-ws"))
        .and(body_json(json!({"configuration": config})))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let honcho = Honcho::new(&server.uri(), "test-ws").unwrap();

    let mut cfg = HashMap::new();
    cfg.insert("reasoning".to_string(), json!({"enabled": false}));
    honcho.set_configuration(cfg).await.unwrap();
}

#[tokio::test]
async fn workspace_id_accessor() {
    let server = MockServer::start().await;
    let honcho = Honcho::new(&server.uri(), "my-workspace").unwrap();
    assert_eq!(honcho.workspace_id(), "my-workspace");
}

#[tokio::test]
async fn honcho_constructor_rejects_invalid_url() {
    let result = Honcho::new("not a url", "test-ws");
    assert!(result.is_err());
}
