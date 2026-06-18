use axum::body::Body;
use honcho_api_rs::app::{AppState, build_router};
use honcho_api_rs::auth::{AuthConfig, create_hs256_token_for_test};
use http::{Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

#[tokio::test]
async fn health_route_returns_ok() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn create_key_returns_disabled_when_auth_is_off() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/keys?workspace_id=workspace-a")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Feature is disabled"})
    );
}

#[tokio::test]
async fn create_key_requires_admin_token_before_scope_validation() {
    let token = create_hs256_token_for_test(&json!({"t": "", "w": "workspace-a"}), "secret");
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/keys")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Resource requires admin privileges"})
    );
}

#[tokio::test]
async fn create_key_requires_at_least_one_scope() {
    let token = create_hs256_token_for_test(&json!({"t": "", "ad": true}), "secret");
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/keys")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "At least one of workspace_id, peer_id, or session_id must be provided"})
    );

    let token = create_hs256_token_for_test(&json!({"t": "", "ad": true}), "secret");
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/keys?workspace_id=&peer_id=&session_id=")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "At least one of workspace_id, peer_id, or session_id must be provided"})
    );
}

#[tokio::test]
async fn create_key_returns_verifiable_scoped_key() {
    let admin_token = create_hs256_token_for_test(&json!({"t": "", "ad": true}), "secret");
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post(
                "/v3/keys?workspace_id=workspace-a&peer_id=peer-a&expires_at=2030-06-15T10%3A20%3A30.456%2B03%3A00",
            )
            .header("authorization", format!("Bearer {admin_token}"))
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    let token = body["key"].as_str().expect("key should be a string");
    assert!(
        honcho_api_rs::auth::authorize(
            &AuthConfig {
                use_auth: true,
                jwt_secret: Some("secret".to_string()),
            },
            Some(&format!("Bearer {token}")),
            false,
            Some("workspace-a"),
            Some("peer-a"),
            None,
        )
        .is_ok()
    );
}

#[tokio::test]
async fn workspace_write_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"workspace-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn workspace_update_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"metadata":{"key":"value"}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn workspace_update_route_rejects_invalid_path_name() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/bad%20name")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"metadata":{"key":"value"}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "string_pattern_mismatch",
                "loc": ["body", "workspace_id"],
                "msg": "String should match pattern '^[a-zA-Z0-9_-]+$'",
                "input": "bad name",
                "ctx": {"pattern": "^[a-zA-Z0-9_-]+$"}
            }]
        })
    );
}

#[tokio::test]
async fn workspace_write_route_requires_name_or_id() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces")
                .header("content-type", "application/json")
                .body(Body::from(r#"{}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "missing",
                "loc": ["body", "id"],
                "msg": "Field required",
                "input": {}
            }]
        })
    );
}

#[tokio::test]
async fn workspace_write_route_rejects_invalid_summary_configuration() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"workspace-a","configuration":{"summary":{"messages_per_short_summary":1}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "greater_than_equal",
                "loc": ["body", "configuration", "summary", "messages_per_short_summary"],
                "msg": "Input should be greater than or equal to 10",
                "input": 1,
                "ctx": {"ge": 10}
            }]
        })
    );
}

#[tokio::test]
async fn workspace_write_route_rejects_invalid_name_with_fastapi_shape() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"bad name"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "string_pattern_mismatch",
                "loc": ["body", "name"],
                "msg": "String should match pattern '^[a-zA-Z0-9_-]+$'",
                "input": "bad name",
                "ctx": {"pattern": "^[a-zA-Z0-9_-]+$"}
            }]
        })
    );
}

#[tokio::test]
async fn workspace_write_route_rejects_mismatched_workspace_scope() {
    let token = create_hs256_token_for_test(&json!({"t": "", "w": "workspace-b"}), "secret");
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces")
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"workspace-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Unauthorized access to resource"})
    );
}

#[tokio::test]
async fn workspace_write_route_rejects_non_admin_without_workspace_scope() {
    let token = create_hs256_token_for_test(&json!({"t": "", "p": "peer-a"}), "secret");
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces")
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"workspace-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Unauthorized access to resource"})
    );
}

#[tokio::test]
async fn workspace_update_route_rejects_mismatched_workspace_scope() {
    let token = create_hs256_token_for_test(&json!({"t": "", "w": "workspace-b"}), "secret");
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a")
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"metadata":{"key":"value"}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "JWT not permissioned for this resource"})
    );
}

#[tokio::test]
async fn peer_write_routes_are_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let app = build_router(state);

    let create_response = app
        .clone()
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/peers")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"peer-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(create_response).await,
        json!({"detail": "Rust write routes are disabled"})
    );

    let update_response = app
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/peers/peer-a")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"metadata":{"key":"value"}}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(update_response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn peer_create_rejects_mismatched_workspace_or_peer_scope() {
    let workspace_token =
        create_hs256_token_for_test(&json!({"t": "", "w": "workspace-b"}), "secret");
    let peer_token = create_hs256_token_for_test(&json!({"t": "", "p": "peer-b"}), "secret");
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let app = build_router(state);

    let workspace_response = app
        .clone()
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/peers")
                .header("authorization", format!("Bearer {workspace_token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"peer-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(workspace_response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(workspace_response).await,
        json!({"detail": "Unauthorized access to resource"})
    );

    let peer_response = app
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/peers")
                .header("authorization", format!("Bearer {peer_token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"peer-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(peer_response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(peer_response).await,
        json!({"detail": "Unauthorized access to resource"})
    );
}

#[tokio::test]
async fn session_write_routes_are_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let app = build_router(state);

    let create_response = app
        .clone()
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"id":"session-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create_response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(create_response).await,
        json!({"detail": "Rust write routes are disabled"})
    );

    let update_response = app
        .clone()
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/sessions/session-a")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"metadata":{"key":"value"}}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(update_response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(update_response).await,
        json!({"detail": "Rust write routes are disabled"})
    );

    let delete_response = app
        .oneshot(
            Request::delete("/v3/workspaces/workspace-a/sessions/session-a")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(delete_response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn session_clone_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let app = build_router(state);

    let clone_response = app
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions/session-a/clone")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(clone_response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(clone_response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn session_clone_rejects_mismatched_session_scope() {
    let session_token = create_hs256_token_for_test(&json!({"t": "", "s": "session-b"}), "secret");
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions/session-a/clone")
                .header("authorization", format!("Bearer {session_token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "JWT not permissioned for this resource"})
    );
}

#[tokio::test]
async fn session_create_allows_peer_only_token_until_write_guard() {
    let token = create_hs256_token_for_test(&json!({"t": "", "p": "peer-a"}), "secret");
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions")
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"id":"session-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn session_create_rejects_mismatched_workspace_or_session_scope() {
    let workspace_token =
        create_hs256_token_for_test(&json!({"t": "", "w": "workspace-b"}), "secret");
    let session_token = create_hs256_token_for_test(&json!({"t": "", "s": "session-b"}), "secret");
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let app = build_router(state);

    let workspace_response = app
        .clone()
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions")
                .header("authorization", format!("Bearer {workspace_token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"id":"session-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(workspace_response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(workspace_response).await,
        json!({"detail": "Unauthorized access to resource"})
    );

    let session_response = app
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions")
                .header("authorization", format!("Bearer {session_token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"id":"session-a"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(session_response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(session_response).await,
        json!({"detail": "Unauthorized access to resource"})
    );
}

#[tokio::test]
async fn session_update_rejects_mismatched_workspace_or_session_scope() {
    let workspace_token = create_hs256_token_for_test(
        &json!({"t": "", "w": "workspace-b", "s": "session-a"}),
        "secret",
    );
    let session_token = create_hs256_token_for_test(&json!({"t": "", "s": "session-b"}), "secret");
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let app = build_router(state);

    let workspace_response = app
        .clone()
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/sessions/session-a")
                .header("authorization", format!("Bearer {workspace_token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"metadata":{"key":"value"}}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(workspace_response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(workspace_response).await,
        json!({"detail": "JWT not permissioned for this resource"})
    );

    let session_response = app
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/sessions/session-a")
                .header("authorization", format!("Bearer {session_token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"metadata":{"key":"value"}}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(session_response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(session_response).await,
        json!({"detail": "JWT not permissioned for this resource"})
    );
}

#[tokio::test]
async fn session_update_rejects_peer_membership_payload_until_implemented() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/sessions/session-a")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"metadata":{"key":"value"},"peers":{"peer-a":{}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "value_error",
                "loc": ["body", "peers"],
                "msg": "Value error, session peer membership is not implemented in the Rust write shadow yet",
                "input": {"peer-a": {}},
                "ctx": {"error": {}}
            }]
        })
    );
}

#[tokio::test]
async fn session_add_peers_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions/session-a/peers")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"peer-a":{}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn session_set_peers_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/sessions/session-a/peers")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"peer-a":{}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn session_remove_peers_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::delete("/v3/workspaces/workspace-a/sessions/session-a/peers")
                .header("content-type", "application/json")
                .body(Body::from(r#"["peer-a"]"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn session_peer_config_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/sessions/session-a/peers/peer-a/config")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"observe_me":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn message_create_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions/session-a/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"messages":[{"content":"hi","peer_id":"peer-a"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn message_create_rejects_empty_batch() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions/session-a/messages")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "too_short",
                "loc": ["body", "messages"],
                "msg": "List should have at least 1 item after validation, not 0",
                "input": [],
                "ctx": {"field_type": "List", "min_length": 1, "actual_length": 0}
            }]
        })
    );
}

#[tokio::test]
async fn message_create_rejects_missing_content_with_integer_index_loc() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions/session-a/messages")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"messages":[{"peer_id":"peer-a"}]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "missing",
                "loc": ["body", "messages", 0, "content"],
                "msg": "Field required",
                "input": {"peer_id": "peer-a"}
            }]
        })
    );
}

#[tokio::test]
async fn message_create_rejects_missing_peer_id() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions/session-a/messages")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"messages":[{"content":"hello"}]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "missing",
                "loc": ["body", "messages", 0, "peer_id"],
                "msg": "Field required",
                "input": {"content": "hello"}
            }]
        })
    );
}

#[tokio::test]
async fn message_create_rejects_missing_messages_field() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions/session-a/messages")
                .header("content-type", "application/json")
                .body(Body::from(r#"{}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "missing",
                "loc": ["body", "messages"],
                "msg": "Field required",
                "input": {}
            }]
        })
    );
}

#[tokio::test]
async fn message_update_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/sessions/session-a/messages/msg-a")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"metadata":{"k":"v"}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn message_update_rejects_non_dict_metadata() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/sessions/session-a/messages/msg-a")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"metadata":"not-a-dict"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "dict_type",
                "loc": ["body", "metadata"],
                "msg": "Input should be a valid dictionary",
                "input": "not-a-dict"
            }]
        })
    );
}

#[tokio::test]
async fn webhook_create_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/webhooks")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"url":"https://example.com/hook"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn webhook_delete_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::delete("/v3/workspaces/workspace-a/webhooks/endpoint-a")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn webhook_create_rejects_private_ip_url() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/webhooks")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"url":"http://127.0.0.1/hook"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "value_error",
                "loc": ["body", "url"],
                "msg": "Value error, Private/internal IP addresses are not allowed",
                "input": "http://127.0.0.1/hook",
                "ctx": {"error": {}}
            }]
        })
    );
}

#[tokio::test]
async fn webhook_create_requires_url() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/webhooks")
                .header("content-type", "application/json")
                .body(Body::from(r#"{}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "missing",
                "loc": ["body", "url"],
                "msg": "Field required",
                "input": {}
            }]
        })
    );
}

#[tokio::test]
async fn conclusion_delete_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::delete("/v3/workspaces/workspace-a/conclusions/conc-a")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn conclusion_delete_rejects_mismatched_workspace_scope() {
    let token = create_hs256_token_for_test(&json!({"t": "", "w": "workspace-b"}), "secret");
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::delete("/v3/workspaces/workspace-a/conclusions/conc-a")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn set_peer_card_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/peers/peer-a/card")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"peer_card":["likes cats"]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn set_peer_card_requires_peer_card_field() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/peers/peer-a/card")
                .header("content-type", "application/json")
                .body(Body::from(r#"{}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "missing",
                "loc": ["body", "peer_card"],
                "msg": "Field required",
                "input": {}
            }]
        })
    );
}

#[tokio::test]
async fn set_peer_card_rejects_non_string_item() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::put("/v3/workspaces/workspace-a/peers/peer-a/card")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"peer_card":["ok",42]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "string_type",
                "loc": ["body", "peer_card", 1],
                "msg": "Input should be a valid string",
                "input": 42
            }]
        })
    );
}

async fn response_json(response: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should be readable");
    serde_json::from_slice(&bytes).expect("body should be JSON")
}

fn no_auth_state() -> AppState {
    AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    })
}

async fn get_context(query: &str) -> axum::response::Response {
    let uri = format!("/v3/workspaces/workspace-a/sessions/session-a/context{query}");
    build_router(no_auth_state())
        .oneshot(Request::get(&uri).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn get_context_rejects_tokens_above_max_with_fastapi_shape() {
    let response = get_context("?tokens=150000").await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "less_than_equal",
            "loc": ["query", "tokens"],
            "msg": "Input should be less than or equal to 100000",
            "input": "150000",
            "ctx": {"le": 100000}
        }]})
    );
}

#[tokio::test]
async fn get_context_rejects_non_integer_tokens_with_fastapi_shape() {
    let response = get_context("?tokens=abc").await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "int_parsing",
            "loc": ["query", "tokens"],
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "abc"
        }]})
    );
}

#[tokio::test]
async fn get_context_rejects_invalid_summary_bool_with_fastapi_shape() {
    let response = get_context("?summary=maybe").await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "bool_parsing",
            "loc": ["query", "summary"],
            "msg": "Input should be a valid boolean, unable to interpret input",
            "input": "maybe"
        }]})
    );
}

#[tokio::test]
async fn get_context_requires_peer_target_when_perspective_given() {
    let response = get_context("?peer_perspective=alice").await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "peer_target must be provided if peer_perspective is provided"})
    );
}

#[tokio::test]
async fn get_context_perspective_path_is_not_implemented() {
    let response = get_context("?peer_target=alice").await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Perspective-scoped session context (peer_target) is not yet supported by the Rust API"})
    );
}
