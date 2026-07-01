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
async fn get_representation_requires_auth() {
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/peers/peer-a/representation")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn get_representation_validates_search_top_k_bounds() {
    // A peer-scoped token passes auth, so validation (422) fires before any DB
    // access — proving the route is wired (no 404/405) and bounds are enforced.
    let token =
        create_hs256_token_for_test(&json!({"t": "", "w": "workspace-a", "p": "peer-a"}), "secret");
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/peers/peer-a/representation")
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"search_top_k": 0}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "search_top_k must be between 1 and 100"})
    );
}

#[tokio::test]
async fn test_webhook_requires_auth() {
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::get("/v3/workspaces/workspace-a/webhooks/test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_webhook_rejects_mismatched_workspace_scope() {
    // A non-admin token scoped to a different workspace is rejected before any
    // enqueue, proving the route is wired and the scope check fires.
    let token = create_hs256_token_for_test(&json!({"t": "", "w": "workspace-b"}), "secret");
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::get("/v3/workspaces/workspace-a/webhooks/test")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    // The webhook routes are now workspace-scoped via `authorize` (port of #679),
    // so a mismatched-workspace token is denied with the standard message.
    assert_eq!(
        response_json(response).await,
        json!({"detail": "JWT not permissioned for this resource"})
    );
}

#[tokio::test]
async fn get_peer_context_requires_auth() {
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::get("/v3/workspaces/workspace-a/peers/peer-a/context")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn get_peer_context_validates_search_top_k_bounds() {
    let token =
        create_hs256_token_for_test(&json!({"t": "", "w": "workspace-a", "p": "peer-a"}), "secret");
    let state = AppState::for_test(AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    });
    let response = build_router(state)
        .oneshot(
            Request::get("/v3/workspaces/workspace-a/peers/peer-a/context?search_top_k=0")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
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
async fn workspace_delete_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::delete("/v3/workspaces/workspace-a")
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
async fn schedule_dream_route_is_disabled_by_default() {
    let state = AppState::for_test(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/schedule_dream")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"observer":"alice","dream_type":"omni"}"#))
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
async fn schedule_dream_requires_observer() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/schedule_dream")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"dream_type":"omni"}"#))
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
                "loc": ["body", "observer"],
                "msg": "Field required",
                "input": {"dream_type": "omni"}
            }]
        })
    );
}

#[tokio::test]
async fn schedule_dream_rejects_invalid_dream_type() {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/schedule_dream")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"observer":"alice","dream_type":"lucid"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({
            "detail": [{
                "type": "enum",
                "loc": ["body", "dream_type"],
                "msg": "Input should be 'omni'",
                "input": "lucid",
                "ctx": {"expected": "'omni'"}
            }]
        })
    );
}

#[tokio::test]
async fn schedule_dream_returns_400_when_dreams_disabled() {
    // Body validation (422) precedes the DREAM.ENABLED check, so a valid body
    // is required to reach the 400 branch — which fires before any DB access.
    let mut state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    state.dream_enabled = false;
    let response = build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/schedule_dream")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"observer":"alice","dream_type":"omni"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Dreams are not enabled in the system configuration"})
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
    // A peer-scoped token without its parent workspace is now malformed and
    // rejected at verify time by the token-shape invariant (port of #679).
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Invalid JWT scope: peer/session token missing workspace"})
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
    // `{p: peer-b}` carries no workspace, so the token-shape invariant rejects it
    // at verify time (port of #679).
    assert_eq!(
        response_json(peer_response).await,
        json!({"detail": "Invalid JWT scope: peer/session token missing workspace"})
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
    // `{s: session-b}` carries no workspace, so the token-shape invariant rejects
    // it at verify time (port of #679).
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Invalid JWT scope: peer/session token missing workspace"})
    );
}

#[tokio::test]
async fn session_create_rejects_peer_only_token_without_workspace() {
    // A peer-scoped token with no workspace is malformed under the token-shape
    // invariant (#679) and rejected at verify, before the write guard.
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

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Invalid JWT scope: peer/session token missing workspace"})
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
    // `{s: session-b}` carries no workspace → rejected by the token-shape
    // invariant at verify time (port of #679).
    assert_eq!(
        response_json(session_response).await,
        json!({"detail": "Invalid JWT scope: peer/session token missing workspace"})
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
    // `{s: session-b}` carries no workspace → rejected by the token-shape
    // invariant at verify time (port of #679).
    assert_eq!(
        response_json(session_response).await,
        json!({"detail": "Invalid JWT scope: peer/session token missing workspace"})
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

#[tokio::test]
async fn chat_streaming_is_implemented() {
    // `stream: true` is now wired (SSE); it is no longer a 501. Without a
    // configured pool in this harness it reaches DB access and surfaces 500.
    let response = build_router(no_auth_state())
        .oneshot(
            Request::post("/v3/workspaces/ws/peers/alice/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"query": "hi", "stream": true}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(response.status(), StatusCode::NOT_IMPLEMENTED);
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn chat_rejects_empty_query_with_422() {
    let response = build_router(no_auth_state())
        .oneshot(
            Request::post("/v3/workspaces/ws/peers/alice/chat")
                .header("content-type", "application/json")
                .body(Body::from(json!({"query": "   "}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

fn no_auth_writes_state() -> AppState {
    AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    })
}

async fn post_create_conclusions(state: AppState, body: &str) -> axum::response::Response {
    build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/conclusions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn create_conclusions_route_is_disabled_by_default() {
    let response =
        post_create_conclusions(no_auth_state(), r#"{"conclusions": []}"#).await;
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Rust write routes are disabled"})
    );
}

#[tokio::test]
async fn create_conclusions_requires_conclusions_field() {
    let response = post_create_conclusions(no_auth_writes_state(), "{}").await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "missing",
            "loc": ["body", "conclusions"],
            "msg": "Field required",
            "input": {}
        }]})
    );
}

#[tokio::test]
async fn create_conclusions_rejects_empty_list() {
    let response = post_create_conclusions(no_auth_writes_state(), r#"{"conclusions": []}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "too_short",
            "loc": ["body", "conclusions"],
            "msg": "List should have at least 1 item after validation, not 0",
            "input": [],
            "ctx": {"field_type": "List", "min_length": 1, "actual_length": 0}
        }]})
    );
}

#[tokio::test]
async fn create_conclusions_reports_item_field_with_index_loc() {
    let response = post_create_conclusions(
        no_auth_writes_state(),
        r#"{"conclusions": [{"observer_id": "a", "observed_id": "b"}]}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "missing",
            "loc": ["body", "conclusions", 0, "content"],
            "msg": "Field required",
            "input": {"observer_id": "a", "observed_id": "b"}
        }]})
    );
}

#[tokio::test]
async fn create_conclusions_honors_observations_alias_in_loc() {
    let response =
        post_create_conclusions(no_auth_writes_state(), r#"{"observations": [{}]}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "missing",
            "loc": ["body", "observations", 0, "content"],
            "msg": "Field required",
            "input": {}
        }]})
    );
}

async fn get_context(query: &str) -> axum::response::Response {
    let uri = format!("/v3/workspaces/workspace-a/sessions/session-a/context{query}");
    build_router(no_auth_state())
        .oneshot(Request::get(&uri).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

/// Build a minimal `multipart/form-data` body with optional `peer_id` field and
/// one file part, returning `(content_type_header, body)`.
fn multipart_upload_body(
    peer_id: Option<&str>,
    filename: &str,
    file_content_type: &str,
    file_bytes: &[u8],
) -> (String, Vec<u8>) {
    let boundary = "----rusttestboundary";
    let mut body: Vec<u8> = Vec::new();
    if let Some(peer_id) = peer_id {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"peer_id\"\r\n\r\n{peer_id}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {file_content_type}\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn post_upload(content_type: String, body: Vec<u8>) -> axum::response::Response {
    let state = AppState::for_test_with_writes(AuthConfig {
        use_auth: false,
        jwt_secret: None,
    });
    build_router(state)
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/sessions/session-a/messages/upload")
                .header("content-type", content_type)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn upload_rejects_unsupported_content_type() {
    let (content_type, body) =
        multipart_upload_body(Some("alice"), "doc.pdf", "application/pdf", b"%PDF-1.4 fake");
    let response = post_upload(content_type, body).await;
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "Unsupported file type: application/pdf"})
    );
}

#[tokio::test]
async fn upload_requires_peer_id_field() {
    let (content_type, body) =
        multipart_upload_body(None, "notes.txt", "text/plain", b"hello world");
    let response = post_upload(content_type, body).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "peer_id form field is required"})
    );
}

async fn post_search(body: &str) -> axum::response::Response {
    build_router(no_auth_state())
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/search")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn post_conclusion_query(body: &str) -> axum::response::Response {
    build_router(no_auth_state())
        .oneshot(
            Request::post("/v3/workspaces/workspace-a/conclusions/query")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn conclusion_query_requires_query_field() {
    let response = post_conclusion_query("{}").await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "missing",
            "loc": ["body", "query"],
            "msg": "Field required",
            "input": {}
        }]})
    );
}

#[tokio::test]
async fn conclusion_query_rejects_top_k_above_maximum() {
    let response = post_conclusion_query(r#"{"query": "q", "top_k": 101}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "less_than_equal",
            "loc": ["body", "top_k"],
            "msg": "Input should be less than or equal to 100",
            "input": 101,
            "ctx": {"le": 100}
        }]})
    );
}

#[tokio::test]
async fn conclusion_query_rejects_distance_above_one() {
    let response = post_conclusion_query(r#"{"query": "q", "distance": 2.0}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "less_than_equal",
            "loc": ["body", "distance"],
            "msg": "Input should be less than or equal to 1",
            "input": 2.0,
            "ctx": {"le": 1.0}
        }]})
    );
}

#[tokio::test]
async fn conclusion_query_requires_observer_and_observed() {
    // Valid schema, but no observer/observed in filters -> ValidationException.
    let response = post_conclusion_query(r#"{"query": "q"}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": "observer and observed must be specified for semantic search"})
    );
}

#[tokio::test]
async fn search_requires_query_with_fastapi_shape() {
    let response = post_search("{}").await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "missing",
            "loc": ["body", "query"],
            "msg": "Field required",
            "input": {}
        }]})
    );
}

#[tokio::test]
async fn search_rejects_non_string_query() {
    let response = post_search(r#"{"query": 123}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "string_type",
            "loc": ["body", "query"],
            "msg": "Input should be a valid string",
            "input": 123
        }]})
    );
}

#[tokio::test]
async fn search_rejects_limit_below_minimum() {
    let response = post_search(r#"{"query": "hi", "limit": 0}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "greater_than_equal",
            "loc": ["body", "limit"],
            "msg": "Input should be greater than or equal to 1",
            "input": 0,
            "ctx": {"ge": 1}
        }]})
    );
}

#[tokio::test]
async fn search_rejects_limit_above_maximum() {
    let response = post_search(r#"{"query": "hi", "limit": 101}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "less_than_equal",
            "loc": ["body", "limit"],
            "msg": "Input should be less than or equal to 100",
            "input": 101,
            "ctx": {"le": 100}
        }]})
    );
}

#[tokio::test]
async fn search_rejects_fractional_limit() {
    let response = post_search(r#"{"query": "hi", "limit": 1.5}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "int_from_float",
            "loc": ["body", "limit"],
            "msg": "Input should be a valid integer, got a number with a fractional part",
            "input": 1.5
        }]})
    );
}

#[tokio::test]
async fn search_rejects_non_dict_filters() {
    let response = post_search(r#"{"query": "hi", "filters": 5}"#).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response_json(response).await,
        json!({"detail": [{
            "type": "dict_type",
            "loc": ["body", "filters"],
            "msg": "Input should be a valid dictionary",
            "input": 5
        }]})
    );
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
async fn get_context_perspective_path_is_implemented() {
    // The perspective path is now ported (no longer a 501); without a configured
    // pool in this harness it reaches DB access and surfaces 500. The real
    // behavior is covered by the gated `perspective_session_context_*` DB test.
    let response = get_context("?peer_target=alice").await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_ne!(response.status(), StatusCode::NOT_IMPLEMENTED);
}
