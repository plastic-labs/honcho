use honcho_api_rs::auth::{AuthConfig, JwtParams, authorize, create_scoped_key};
use honcho_api_rs::cache::{
    CacheConfig, PeerCache, peer_cache_key, redis_client_url, session_cache_key,
};
use honcho_api_rs::config::AppConfig;
use honcho_api_rs::db::{peer_card_json, session_summaries_json, webhook_endpoint_json};
use honcho_api_rs::filters::{FilterTarget, build_filter_clause};
use honcho_api_rs::pagination::{Pagination, page_response};
use honcho_api_rs::queue_status::{QueueStatusCounts, build_queue_status};
use serde_json::json;

#[test]
fn config_defaults_and_normalizes_python_postgres_url() {
    let env = [
        (
            "DB_CONNECTION_URI",
            "postgresql+psycopg://postgres:postgres@database:5432/postgres",
        ),
        ("AUTH_USE_AUTH", "true"),
        ("AUTH_JWT_SECRET", "secret"),
    ];

    let config = AppConfig::from_pairs(env).expect("config should parse");

    assert_eq!(config.bind_address.to_string(), "0.0.0.0:8001");
    assert_eq!(
        config.database_url,
        "postgresql://postgres:postgres@database:5432/postgres"
    );
    assert_eq!(config.db_schema, "public");
    assert!(config.auth.use_auth);
    assert_eq!(config.auth.jwt_secret.as_deref(), Some("secret"));
    assert!(!config.write_enabled);
    assert_eq!(
        config.cache,
        CacheConfig {
            enabled: false,
            url: "redis://localhost:6379/0?suppress=true".to_string(),
            namespace: "honcho".to_string(),
        }
    );
}

#[test]
fn config_embedding_defaults_and_gemini_cap() {
    // Defaults: EMBED_MESSAGES on, OpenAI provider uses MAX_INPUT_TOKENS as-is.
    let config = AppConfig::from_pairs([(
        "DB_CONNECTION_URI",
        "postgresql://postgres:postgres@database/postgres",
    )])
    .expect("config should parse");
    assert!(config.embed_messages);
    assert_eq!(config.embedding_max_tokens, 8192);

    // Gemini caps the per-chunk budget at 2048; EMBED_MESSAGES can be disabled.
    let config = AppConfig::from_pairs([
        (
            "DB_CONNECTION_URI",
            "postgresql://postgres:postgres@database/postgres",
        ),
        ("EMBED_MESSAGES", "false"),
        ("EMBEDDING_PROVIDER", "gemini"),
        ("EMBEDDING_MAX_INPUT_TOKENS", "8192"),
    ])
    .expect("config should parse");
    assert!(!config.embed_messages);
    assert_eq!(config.embedding_max_tokens, 2048);

    // OpenAI honors a custom MAX_INPUT_TOKENS without the 2048 cap.
    let config = AppConfig::from_pairs([
        (
            "DB_CONNECTION_URI",
            "postgresql://postgres:postgres@database/postgres",
        ),
        ("EMBEDDING_MAX_INPUT_TOKENS", "4096"),
    ])
    .expect("config should parse");
    assert_eq!(config.embedding_max_tokens, 4096);
}

#[test]
fn config_parses_write_guard() {
    let config = AppConfig::from_pairs([
        (
            "DB_CONNECTION_URI",
            "postgresql://postgres:postgres@database/postgres",
        ),
        ("RUST_API_ENABLE_WRITES", "true"),
        ("AUTH_USE_AUTH", "false"),
    ])
    .expect("config should parse");

    assert!(config.write_enabled);
}

#[test]
fn config_parses_python_cache_settings() {
    let config = AppConfig::from_pairs([
        (
            "DB_CONNECTION_URI",
            "postgresql://postgres:postgres@database/postgres",
        ),
        ("CACHE_ENABLED", "true"),
        ("CACHE_URL", "redis://redis:6379/2?suppress=true"),
        ("CACHE_NAMESPACE", "custom-cache"),
    ])
    .expect("config should parse");

    assert_eq!(
        config.cache,
        CacheConfig {
            enabled: true,
            url: "redis://redis:6379/2?suppress=true".to_string(),
            namespace: "custom-cache".to_string(),
        }
    );
}

#[test]
fn config_uses_top_level_namespace_for_cache_namespace() {
    let config = AppConfig::from_pairs([
        (
            "DB_CONNECTION_URI",
            "postgresql://postgres:postgres@database/postgres",
        ),
        ("NAMESPACE", "tenant-a"),
    ])
    .expect("config should parse");

    assert_eq!(config.cache.namespace, "tenant-a");
}

#[test]
fn config_uses_top_level_namespace_when_cache_namespace_is_blank() {
    let config = AppConfig::from_pairs([
        (
            "DB_CONNECTION_URI",
            "postgresql://postgres:postgres@database/postgres",
        ),
        ("NAMESPACE", "tenant-a"),
        ("CACHE_NAMESPACE", ""),
    ])
    .expect("config should parse");

    assert_eq!(config.cache.namespace, "tenant-a");
}

#[test]
fn config_rejects_invalid_cache_enabled_bool() {
    let error = AppConfig::from_pairs([
        (
            "DB_CONNECTION_URI",
            "postgresql://postgres:postgres@database/postgres",
        ),
        ("CACHE_ENABLED", "treu"),
    ])
    .expect_err("invalid cache bool should fail");

    assert!(error.to_string().contains("CACHE_ENABLED"));
}

#[test]
fn peer_cache_key_matches_python_key() {
    assert_eq!(
        peer_cache_key("tenant-a", "workspace-a", "peer-a"),
        "tenant-a:v2:workspace:workspace-a:peer:peer-a"
    );
}

#[test]
fn session_cache_key_matches_python_key() {
    assert_eq!(
        session_cache_key("tenant-a", "workspace-a", "session-a"),
        "tenant-a:v2:workspace:workspace-a:session:session-a"
    );
}

#[test]
fn redis_client_url_drops_python_suppress_query_but_keeps_database() {
    assert_eq!(
        redis_client_url("redis://redis:6379/2?suppress=true"),
        "redis://redis:6379/2"
    );
}

#[tokio::test]
async fn peer_cache_invalidation_deletes_real_redis_value_key_when_configured() {
    let Ok(redis_url) = std::env::var("HONCHO_API_RS_REDIS_TEST_URL") else {
        eprintln!("skipping Redis integration test: HONCHO_API_RS_REDIS_TEST_URL is unset");
        return;
    };
    let namespace = format!(
        "rust-cache-test-{}",
        chrono::Utc::now()
            .timestamp_nanos_opt()
            .expect("current timestamp should be representable")
    );
    let workspace_name = "workspace-a";
    let peer_name = "peer-a";
    let cache_key = peer_cache_key(&namespace, workspace_name, peer_name);
    let client = redis::Client::open(redis_client_url(&redis_url)).expect("Redis URL should parse");
    let mut connection = client
        .get_multiplexed_async_connection()
        .await
        .expect("Redis should be reachable for integration test");
    let _: () = redis::cmd("SET")
        .arg(&cache_key)
        .arg("cached-peer")
        .query_async(&mut connection)
        .await
        .expect("test key should be seeded");

    let cache = PeerCache::new(CacheConfig {
        enabled: true,
        url: redis_url,
        namespace,
    });
    cache.invalidate_peer(workspace_name, peer_name).await;

    let exists_count: i64 = redis::cmd("EXISTS")
        .arg(&cache_key)
        .query_async(&mut connection)
        .await
        .expect("test key existence should be readable");
    assert_eq!(exists_count, 0);
}

#[test]
fn config_requires_database_url() {
    let error = AppConfig::from_pairs([("AUTH_USE_AUTH", "false")])
        .expect_err("missing database URL should fail");

    assert!(error.to_string().contains("DB_CONNECTION_URI"));
}

#[test]
fn auth_allows_everything_when_disabled() {
    let config = AuthConfig {
        use_auth: false,
        jwt_secret: None,
    };

    let params = authorize(&config, None, false, Some("workspace"), None, None)
        .expect("auth disabled should allow request");

    assert_eq!(params, JwtParams::admin());
}

#[test]
fn auth_rejects_non_admin_for_admin_route() {
    let config = AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    };
    let token = honcho_api_rs::auth::create_hs256_token_for_test(
        &json!({ "t": "", "w": "workspace" }),
        "secret",
    );

    let error = authorize(
        &config,
        Some(&format!("Bearer {token}")),
        true,
        None,
        None,
        None,
    )
    .expect_err("workspace token is not admin");

    assert!(error.to_string().contains("admin"));
}

#[test]
fn auth_allows_workspace_scoped_token_for_workspace_route() {
    let config = AuthConfig {
        use_auth: true,
        jwt_secret: Some("secret".to_string()),
    };
    let token = honcho_api_rs::auth::create_hs256_token_for_test(
        &json!({ "t": "", "w": "workspace" }),
        "secret",
    );

    let params = authorize(
        &config,
        Some(&format!("Bearer {token}")),
        false,
        Some("workspace"),
        None,
        None,
    )
    .expect("workspace token should authorize matching workspace");

    assert_eq!(params.workspace.as_deref(), Some("workspace"));
}

#[test]
fn key_creation_includes_scoped_claims_and_zulu_expiration() {
    let expires_at = chrono::DateTime::parse_from_rfc3339("2030-06-15T10:20:30.456+03:00")
        .expect("valid timestamp")
        .with_timezone(&chrono::Utc);
    let token = create_scoped_key(
        JwtParams {
            workspace: Some("workspace-a".to_string()),
            peer: Some("peer-a".to_string()),
            session: Some("session-a".to_string()),
            exp: Some(expires_at.to_rfc3339()),
            ..JwtParams::default()
        },
        "secret",
    )
    .expect("scoped key should be created");

    let params = authorize(
        &AuthConfig {
            use_auth: true,
            jwt_secret: Some("secret".to_string()),
        },
        Some(&format!("Bearer {token}")),
        false,
        Some("workspace-a"),
        Some("peer-a"),
        Some("session-a"),
    )
    .expect("created token should verify");

    assert!(!params.timestamp.is_empty());
    assert_eq!(params.exp.as_deref(), Some("2030-06-15T07:20:30Z"));
    assert_eq!(params.admin, None);
    assert_eq!(params.workspace.as_deref(), Some("workspace-a"));
    assert_eq!(params.peer.as_deref(), Some("peer-a"));
    assert_eq!(params.session.as_deref(), Some("session-a"));
}

#[test]
fn pagination_uses_fastapi_page_shape() {
    let response = page_response(
        vec![json!({ "id": "a" }), json!({ "id": "b" })],
        5,
        Pagination { page: 2, size: 2 },
    );

    assert_eq!(
        response,
        json!({
            "items": [{ "id": "a" }, { "id": "b" }],
            "total": 5,
            "page": 2,
            "size": 2,
            "pages": 3
        })
    );
}

#[test]
fn webhook_endpoint_shape_uses_workspace_id_alias() {
    let created_at = chrono::DateTime::parse_from_rfc3339("2026-06-15T10:00:00+00:00")
        .expect("valid timestamp")
        .with_timezone(&chrono::Utc);

    let response = webhook_endpoint_json(
        "endpoint-a".to_string(),
        "workspace-a".to_string(),
        "https://example.com/hook".to_string(),
        created_at,
    );

    assert_eq!(
        response,
        json!({
            "id": "endpoint-a",
            "workspace_id": "workspace-a",
            "url": "https://example.com/hook",
            "created_at": "2026-06-15T10:00:00Z"
        })
    );
    assert!(response.get("workspace_name").is_none());
}

#[test]
fn session_summaries_shape_aliases_public_message_id() {
    let response = session_summaries_json(
        "session-a",
        &json!({
            "summaries": {
                "honcho_chat_summary_short": {
                    "content": "Short summary",
                    "message_id": 123,
                    "message_public_id": "msg-public-short",
                    "summary_type": "short",
                    "created_at": "2026-06-15T10:00:00+00:00",
                    "token_count": 12
                },
                "honcho_chat_summary_long": {
                    "content": "Long summary",
                    "message_id": 456,
                    "message_public_id": "msg-public-long",
                    "summary_type": "long",
                    "created_at": "2026-06-15T11:00:00+00:00",
                    "token_count": 34
                }
            }
        }),
    )
    .expect("summary should shape");

    assert_eq!(
        response,
        json!({
            "id": "session-a",
            "short_summary": {
                "content": "Short summary",
                "message_id": "msg-public-short",
                "summary_type": "short",
                "created_at": "2026-06-15T10:00:00+00:00",
                "token_count": 12
            },
            "long_summary": {
                "content": "Long summary",
                "message_id": "msg-public-long",
                "summary_type": "long",
                "created_at": "2026-06-15T11:00:00+00:00",
                "token_count": 34
            }
        })
    );
}

#[test]
fn session_summaries_use_empty_public_message_id_fallback() {
    let response = session_summaries_json(
        "session-a",
        &json!({
            "summaries": {
                "honcho_chat_summary_short": {
                    "content": "Short summary",
                    "message_id": 123,
                    "summary_type": "short",
                    "created_at": "2026-06-15T10:00:00+00:00",
                    "token_count": 12
                }
            }
        }),
    )
    .expect("summary should shape with fallback");

    assert_eq!(response["short_summary"]["message_id"], json!(""));
    assert_eq!(response["long_summary"], json!(null));
}

#[test]
fn session_summaries_reject_malformed_stored_summary() {
    let error = session_summaries_json(
        "session-a",
        &json!({
            "summaries": {
                "honcho_chat_summary_short": {
                    "message_id": 123,
                    "message_public_id": "msg-public-short",
                    "summary_type": "short",
                    "created_at": "2026-06-15T10:00:00+00:00",
                    "token_count": 12
                }
            }
        }),
    )
    .expect_err("missing required content should fail like Python schema shaping");

    assert!(error.to_string().contains("summary.content"));
}

#[test]
fn peer_card_selects_self_card_key() {
    let response = peer_card_json(
        "alice",
        "alice",
        &json!({
            "peer_card": ["self fact", "self preference"],
            "bob_peer_card": ["bob fact"]
        }),
    );

    assert_eq!(
        response,
        json!({ "peer_card": ["self fact", "self preference"] })
    );
}

#[test]
fn peer_card_selects_target_card_key() {
    let response = peer_card_json(
        "alice",
        "bob",
        &json!({
            "peer_card": ["self fact"],
            "bob_peer_card": ["bob fact"]
        }),
    );

    assert_eq!(response, json!({ "peer_card": ["bob fact"] }));
}

#[test]
fn peer_card_returns_null_for_missing_or_malformed_card() {
    assert_eq!(
        peer_card_json("alice", "alice", &json!({})),
        json!({ "peer_card": null })
    );
    assert_eq!(
        peer_card_json("alice", "alice", &json!({ "peer_card": "not a list" })),
        json!({ "peer_card": null })
    );
    assert_eq!(
        peer_card_json("alice", "alice", &json!({ "peer_card": ["ok", 1] })),
        json!({ "peer_card": null })
    );
}

#[test]
fn filters_build_simple_id_and_metadata_conditions() {
    let clause = build_filter_clause(
        FilterTarget::Peer,
        Some(&json!({
            "id": "alice",
            "metadata": { "kind": "user" }
        })),
    )
    .expect("filter should build");

    assert_eq!(clause.sql, " AND name = $1 AND metadata @> $2::jsonb");
    assert_eq!(
        clause.bindings,
        vec![json!("alice"), json!({"kind": "user"})]
    );
}

#[test]
fn filters_ignore_wildcard_values() {
    let clause = build_filter_clause(
        FilterTarget::Session,
        Some(&json!({
            "id": "*",
            "metadata": { "kind": "*" }
        })),
    )
    .expect("filter should build");

    assert_eq!(clause.sql, "");
    assert!(clause.bindings.is_empty());
}

#[test]
fn filters_build_message_alias_conditions() {
    let clause = build_filter_clause(
        FilterTarget::Message,
        Some(&json!({
            "peer_id": "alice",
            "session_id": "session-a",
            "workspace_id": "workspace-a",
            "token_count": { "gte": 3 },
            "metadata": { "kind": "contract" }
        })),
    )
    .expect("message filter should build");

    assert_eq!(
        clause.sql,
        " AND metadata @> $1::jsonb AND peer_name = $2 AND session_name = $3 AND token_count >= $4 AND workspace_name = $5"
    );
    assert_eq!(
        clause.bindings,
        vec![
            json!({"kind": "contract"}),
            json!("alice"),
            json!("session-a"),
            json!(3),
            json!("workspace-a")
        ]
    );
}

#[test]
fn filters_build_conclusion_alias_and_internal_metadata_conditions() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "session_id": "session-a",
            "observer_id": "alice",
            "observed_id": "bob",
            "metadata": { "kind": "contract" }
        })),
    )
    .expect("conclusion filter should build");

    assert_eq!(
        clause.sql,
        " AND internal_metadata @> $1::jsonb AND observed = $2 AND observer = $3 AND session_name = $4"
    );
    assert_eq!(
        clause.bindings,
        vec![
            json!({"kind": "contract"}),
            json!("bob"),
            json!("alice"),
            json!("session-a")
        ]
    );
}

#[test]
fn filters_cast_created_at_comparisons_to_timestamptz() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "created_at": {
                "gte": "2026-01-02T03:04:05Z",
                "lt": "2026-01-03"
            }
        })),
    )
    .expect("datetime filter should build");

    assert_eq!(
        clause.sql,
        " AND created_at >= $1::timestamptz AND created_at < $2::timestamptz"
    );
    assert_eq!(
        clause.bindings,
        vec![
            json!("2026-01-02T03:04:05+00:00"),
            json!("2026-01-03T00:00:00+00:00")
        ]
    );
}

#[test]
fn filters_cast_created_at_in_values_to_timestamptz() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "created_at": {
                "in": [
                    "2026-01-02T03:04:05Z",
                    "2026-01-03"
                ]
            }
        })),
    )
    .expect("datetime in filter should build");

    assert_eq!(
        clause.sql,
        " AND created_at IN ($1::timestamptz, $2::timestamptz)"
    );
    assert_eq!(
        clause.bindings,
        vec![
            json!("2026-01-02T03:04:05+00:00"),
            json!("2026-01-03T00:00:00+00:00")
        ]
    );
}

#[test]
fn filters_empty_created_at_in_skips_condition() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "created_at": { "in": [] }
        })),
    )
    .expect("empty created_at in filter should build");

    assert_eq!(clause.sql, "");
    assert!(clause.bindings.is_empty());
}

#[test]
fn filters_empty_non_datetime_in_builds_false_predicate() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "observer_id": { "in": [] }
        })),
    )
    .expect("empty non-datetime in filter should build");

    assert_eq!(clause.sql, " AND FALSE");
    assert!(clause.bindings.is_empty());
}

#[test]
fn filters_build_conclusion_metadata_type_aware_comparisons() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "metadata": {
                "enabled": { "ne": true },
                "score": { "gte": "10.5" },
                "tier": { "lt": "z" }
            }
        })),
    )
    .expect("metadata comparison filter should build");

    assert_eq!(
        clause.sql,
        " AND internal_metadata->>'enabled' != $1 AND CASE WHEN internal_metadata->>'score' = '' THEN NULL WHEN internal_metadata->>'score' IS NULL THEN NULL WHEN internal_metadata->>'score' ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN (internal_metadata->>'score')::numeric ELSE NULL END >= $2::numeric AND internal_metadata->>'tier' < $3"
    );
    assert_eq!(
        clause.bindings,
        vec![json!("true"), json!("10.5"), json!("z")]
    );
}

#[test]
fn filters_build_conclusion_metadata_in_conditions() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "metadata": {
                "order": { "in": [2, 3] },
                "enabled": { "in": [true, false] }
            }
        })),
    )
    .expect("metadata in filter should build");

    assert_eq!(
        clause.sql,
        " AND internal_metadata->>'enabled' IN ($1, $2) AND internal_metadata->>'order' IN ($3, $4)"
    );
    assert_eq!(
        clause.bindings,
        vec![json!("True"), json!("False"), json!("2"), json!("3")]
    );
}

#[test]
fn filters_build_conclusion_metadata_empty_in_as_false() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "metadata": {
                "order": { "in": [] }
            }
        })),
    )
    .expect("empty metadata in filter should build");

    assert_eq!(clause.sql, " AND FALSE");
    assert!(clause.bindings.is_empty());
}

#[test]
fn filters_build_conclusion_metadata_in_python_scalar_strings() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "metadata": {
                "nullable": { "in": [null] },
                "enabled": { "in": [true, false] }
            }
        })),
    )
    .expect("metadata scalar in filter should build");

    assert_eq!(
        clause.sql,
        " AND internal_metadata->>'enabled' IN ($1, $2) AND internal_metadata->>'nullable' IN ($3)"
    );
    assert_eq!(
        clause.bindings,
        vec![json!("True"), json!("False"), json!("None")]
    );
}

#[test]
fn filters_escape_contains_patterns() {
    let clause = build_filter_clause(
        FilterTarget::Conclusion,
        Some(&json!({
            "metadata": {
                "label": { "icontains": "100%_path\\segment" }
            }
        })),
    )
    .expect("contains filter should build");

    assert_eq!(
        clause.sql,
        " AND internal_metadata->>'label' ILIKE '%' || $1 || '%' ESCAPE '\\'"
    );
    assert_eq!(clause.bindings, vec![json!("100\\%\\_path\\\\segment")]);
}

#[test]
fn filters_reject_python_incompatible_message_columns() {
    for field in [
        "id",
        "public_id",
        "content",
        "peer_name",
        "session_name",
        "workspace_name",
    ] {
        let error = build_filter_clause(FilterTarget::Message, Some(&json!({ field: "value" })))
            .expect_err("message filter should reject unsupported column");

        assert_eq!(
            error.to_string(),
            format!(
                "Column '{field}' is not allowed to be filtered on or does not exist on Message"
            )
        );
    }
}

#[test]
fn queue_status_shape_matches_python_for_workspace_scope() {
    let status = build_queue_status(
        None,
        QueueStatusCounts {
            total: 3,
            completed: 1,
            in_progress: 1,
            pending: 1,
            sessions: vec![("session-id".to_string(), 1, 1, 0)],
        },
    );

    assert_eq!(
        status,
        json!({
            "sessions": {
                "session-id": {
                    "session_id": "session-id",
                    "total_work_units": 2,
                    "completed_work_units": 1,
                    "in_progress_work_units": 1,
                    "pending_work_units": 0
                }
            },
            "total_work_units": 3,
            "completed_work_units": 1,
            "in_progress_work_units": 1,
            "pending_work_units": 1
        })
    );
}

#[test]
fn queue_status_omits_sessions_for_session_scope() {
    let status = build_queue_status(
        Some("session-a"),
        QueueStatusCounts {
            total: 2,
            completed: 1,
            in_progress: 0,
            pending: 1,
            sessions: vec![("session-id".to_string(), 1, 0, 1)],
        },
    );

    assert_eq!(
        status,
        json!({
            "sessions": null,
            "total_work_units": 2,
            "completed_work_units": 1,
            "in_progress_work_units": 0,
            "pending_work_units": 1
        })
    );
}
