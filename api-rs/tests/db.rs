//! DB-integration smoke tests for the ported `db.rs` layer.
//!
//! These run only when `TEST_DATABASE_URL` points at a Postgres server with the
//! `vector` extension available (e.g. the pgvector test container). Each test
//! provisions an isolated ephemeral database, loads the captured honcho schema
//! (`fixtures/schema.sql`), and exercises the real `honcho_api_rs::db` functions
//! against a live Postgres. With the env var unset (the default), every test
//! early-returns so `cargo test` stays green in DB-less environments.
//!
//! Run against the pgvector test container with:
//!   TEST_DATABASE_URL=postgres://postgres@localhost:5432/postgres \
//!     cargo test --test db

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{TimeZone, Utc};
use honcho_api_rs::db;
use honcho_api_rs::db::{ConclusionWriteError, MessageInsert, MessageSnippet, NewConclusion};
use honcho_api_rs::dialectic;
use honcho_api_rs::embedding;
use honcho_api_rs::filters::{FilterClause, FilterTarget, build_filter_clause};
use honcho_api_rs::llm::http::{Credentials, ReqwestHttp};
use honcho_api_rs::search;
use honcho_api_rs::producer::QueueRecord;
use honcho_api_rs::search::{HybridSearchParams, MessageRef};
use serde_json::json;
use serde_json::Value;
use sqlx::{Connection, Executor, PgConnection, PgPool, Row};

/// The embedding column is `vector(1536)`; build a one-hot literal for seeding.
const EMBED_DIM: usize = 1536;

fn one_hot_literal(dim: usize) -> String {
    let mut s = String::from("[");
    for i in 0..EMBED_DIM {
        if i > 0 {
            s.push(',');
        }
        s.push_str(if i == dim { "1" } else { "0" });
    }
    s.push(']');
    s
}

/// A 1536-dim query vector with the given (index, weight) entries set.
fn query_vector(weights: &[(usize, f32)]) -> Vec<f32> {
    let mut v = vec![0.0_f32; EMBED_DIM];
    for &(i, w) in weights {
        v[i] = w;
    }
    v
}

/// The honcho schema captured from the app database (psql meta-commands
/// stripped so sqlx's SQL executor can run it as a single batch).
const SCHEMA: &str = include_str!("fixtures/schema.sql");

/// A throwaway database holding the honcho schema for one test.
struct TestDb {
    admin_url: String,
    db_name: String,
    pool: PgPool,
}

impl TestDb {
    /// Provision an isolated database, or `None` when `TEST_DATABASE_URL` is
    /// unset so callers can skip cleanly.
    async fn setup() -> Option<TestDb> {
        let admin_url = std::env::var("TEST_DATABASE_URL").ok()?;
        let db_name = unique_db_name();

        let mut admin = PgConnection::connect(&admin_url)
            .await
            .expect("connect to admin database");
        admin
            .execute(format!("CREATE DATABASE \"{db_name}\"").as_str())
            .await
            .expect("create test database");
        admin.close().await.ok();

        let pool = PgPool::connect(&swap_db_name(&admin_url, &db_name))
            .await
            .expect("connect to test database");
        pool.execute(SCHEMA).await.expect("load honcho schema");

        Some(TestDb {
            admin_url,
            db_name,
            pool,
        })
    }

    /// Drop the ephemeral database. Called explicitly at the end of each test; a
    /// panicking test leaks its database (acceptable on a throwaway server).
    async fn teardown(self) {
        self.pool.close().await;
        let mut admin = PgConnection::connect(&self.admin_url)
            .await
            .expect("reconnect to admin database");
        admin
            .execute(
                format!("DROP DATABASE IF EXISTS \"{}\" WITH (FORCE)", self.db_name).as_str(),
            )
            .await
            .expect("drop test database");
        admin.close().await.ok();
    }
}

fn unique_db_name() -> String {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("apirs_test_{nanos}_{seq}")
}

/// Replace the database-name path segment of a postgres URL.
fn swap_db_name(url: &str, db_name: &str) -> String {
    match url.rfind('/') {
        Some(idx) => format!("{}/{}", &url[..idx], db_name),
        None => format!("{url}/{db_name}"),
    }
}

#[tokio::test]
async fn get_or_create_workspace_inserts_then_reuses() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // First call inserts and reports created=true.
    let (created, was_created) =
        db::get_or_create_workspace(&test_db.pool, "ws-1", json!({"k": "v"}), json!({}))
            .await
            .expect("create workspace");
    assert!(was_created);
    assert_eq!(created["id"], json!("ws-1"));
    assert_eq!(created["metadata"], json!({"k": "v"}));

    // Second call returns the existing row (created=false) and ignores the new
    // metadata — get-or-create, not upsert.
    let (existing, was_created_again) =
        db::get_or_create_workspace(&test_db.pool, "ws-1", json!({"other": 1}), json!({}))
            .await
            .expect("reuse workspace");
    assert!(!was_created_again);
    assert_eq!(existing["metadata"], json!({"k": "v"}));

    test_db.teardown().await;
}

#[tokio::test]
async fn update_workspace_merges_configuration_and_replaces_metadata() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws-2", json!({"a": 1}), json!({"x": 1}))
        .await
        .expect("create workspace");

    // metadata is replaced wholesale; configuration is merged (||).
    let updated = db::update_workspace(
        &test_db.pool,
        "ws-2",
        Some(json!({"b": 2})),
        Some(json!({"y": 2})),
    )
    .await
    .expect("update workspace");

    assert_eq!(updated["metadata"], json!({"b": 2}));
    assert_eq!(updated["configuration"], json!({"x": 1, "y": 2}));

    test_db.teardown().await;
}

#[tokio::test]
async fn get_or_create_peer_updates_metadata_on_conflict() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // First call creates (and auto-creates the workspace).
    let (created, was_created) = db::get_or_create_peer(
        &test_db.pool,
        "ws",
        "alice",
        Some(json!({"role": "user"})),
        None,
    )
    .await
    .expect("create peer");
    assert!(was_created);
    assert_eq!(created["metadata"], json!({"role": "user"}));

    // Unlike workspaces, a second call with Some(metadata) UPDATEs via COALESCE.
    let (updated, was_created_again) = db::get_or_create_peer(
        &test_db.pool,
        "ws",
        "alice",
        Some(json!({"role": "admin"})),
        None,
    )
    .await
    .expect("update peer");
    assert!(!was_created_again);
    assert_eq!(updated["metadata"], json!({"role": "admin"}));

    // A None metadata preserves the existing value (COALESCE keeps it).
    let (preserved, _) = db::get_or_create_peer(&test_db.pool, "ws", "alice", None, None)
        .await
        .expect("preserve peer");
    assert_eq!(preserved["metadata"], json!({"role": "admin"}));

    test_db.teardown().await;
}

/// Build the message `FilterClause` the search route passes to the legs:
/// `{"workspace_id": name, ...extra}` (mirrors the route forcing `workspace_id`
/// into the filter dict).
fn message_filter(value: Value) -> FilterClause {
    build_filter_clause(FilterTarget::Message, Some(&value)).expect("valid filter")
}

fn ws_filter(name: &str) -> FilterClause {
    message_filter(json!({ "workspace_id": name }))
}

fn message(peer: &str, content: &str) -> MessageInsert {
    MessageInsert {
        peer_name: peer.to_string(),
        content: content.to_string(),
        metadata: json!({}),
        created_at: None,
        token_count: 1,
    }
}

#[tokio::test]
async fn create_messages_assigns_gap_free_sequence_numbers() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // create_messages auto-creates the session and joins sender peers.
    let first = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("alice", "hello"), message("bob", "hi")],
        false,
        8192,
    )
    .await
    .expect("create first batch");
    assert_eq!(first.len(), 2);
    assert_eq!(first[0].seq_in_session, 1);
    assert_eq!(first[1].seq_in_session, 2);
    assert_eq!(first[0].content, "hello");

    // A second batch continues the sequence without gaps.
    let second = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("alice", "again")],
        false,
        8192,
    )
    .await
    .expect("create second batch");
    assert_eq!(second[0].seq_in_session, 3);

    test_db.teardown().await;
}

#[tokio::test]
async fn create_messages_inserts_pending_embeddings_when_enabled() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("alice", "embed me please")],
        true,
        8192,
    )
    .await
    .expect("create with embeddings");

    // One pending embedding row per chunk; short content -> a single chunk.
    let row = sqlx::query(
        "SELECT count(*) AS n, bool_and(sync_state = 'pending') AS all_pending \
         FROM message_embeddings WHERE message_id = $1",
    )
    .bind(&created[0].public_id)
    .fetch_one(&test_db.pool)
    .await
    .expect("count embeddings");
    let count: i64 = row.get("n");
    let all_pending: bool = row.get("all_pending");
    assert_eq!(count, 1);
    assert!(all_pending);

    // With embedding disabled, no rows are written.
    let none = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("alice", "no embed")],
        false,
        8192,
    )
    .await
    .expect("create without embeddings");
    let n: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM message_embeddings WHERE message_id = $1",
    )
    .bind(&none[0].public_id)
    .fetch_one(&test_db.pool)
    .await
    .expect("count none");
    assert_eq!(n, 0);

    test_db.teardown().await;
}

/// Seed an embedding row pointing at `message_public_id` with a one-hot vector.
async fn seed_embedding(pool: &PgPool, message_public_id: &str, one_hot_dim: usize) {
    sqlx::query(
        "INSERT INTO message_embeddings \
         (content, embedding, message_id, workspace_name, session_name, peer_name, sync_state) \
         VALUES ('chunk', $1::vector, $2, 'ws', 'sess', 'alice', 'synced')",
    )
    .bind(one_hot_literal(one_hot_dim))
    .bind(message_public_id)
    .execute(pool)
    .await
    .expect("seed embedding");
}

#[tokio::test]
async fn semantic_search_orders_by_cosine_distance_and_dedups() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("alice", "apple"), message("alice", "banana")],
        false,
        8192,
    )
    .await
    .expect("create messages");
    let apple_id = created[0].public_id.clone();
    let banana_id = created[1].public_id.clone();

    // apple -> dim 0, banana -> dim 5. Two chunks for apple to exercise dedup.
    seed_embedding(&test_db.pool, &apple_id, 0).await;
    seed_embedding(&test_db.pool, &apple_id, 0).await;
    seed_embedding(&test_db.pool, &banana_id, 5).await;

    // A query leaning on dim 0 is closest to apple.
    let toward_apple = query_vector(&[(0, 1.0), (5, 0.2)]);
    let ranked = search::semantic_search_pgvector(&test_db.pool, "ws", &ws_filter("ws"), &toward_apple, 10)
        .await
        .expect("semantic search");
    assert_eq!(ranked, vec![apple_id.clone(), banana_id.clone()]);
    // Despite apple having two embedding chunks, it appears once.
    assert_eq!(ranked.iter().filter(|id| **id == apple_id).count(), 1);

    // Flip the query toward dim 5 and banana ranks first.
    let toward_banana = query_vector(&[(5, 1.0), (0, 0.2)]);
    let ranked = search::semantic_search_pgvector(&test_db.pool, "ws", &ws_filter("ws"), &toward_banana, 10)
        .await
        .expect("semantic search");
    assert_eq!(ranked.first(), Some(&banana_id));

    test_db.teardown().await;
}

#[tokio::test]
async fn fulltext_search_matches_fts_and_special_char_ilike() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "the quick brown fox"),
            message("alice", "lazy dogs sleeping"),
            message("alice", "c++ template metaprogramming"),
        ],
        false,
        8192,
    )
    .await
    .expect("create messages");

    // Natural-language query -> FTS leg; only the fox message matches "fox".
    let fox = search::fulltext_search(&test_db.pool, &ws_filter("ws"), "fox", 10)
        .await
        .expect("fts fox");
    assert_eq!(fox.len(), 1);

    // "dog" stems to match "dogs" via the english FTS config.
    let dog = search::fulltext_search(&test_db.pool, &ws_filter("ws"), "dog", 10)
        .await
        .expect("fts dog");
    assert_eq!(dog.len(), 1);

    // A query with FTS-hostile chars ("c++") takes the literal ILIKE path.
    let cpp = search::fulltext_search(&test_db.pool, &ws_filter("ws"), "c++", 10)
        .await
        .expect("ilike c++");
    assert_eq!(cpp.len(), 1);

    test_db.teardown().await;
}

#[tokio::test]
async fn peer_perspective_filter_keeps_only_in_window_messages() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    db::get_or_create_peer(&test_db.pool, "ws", "alice", None, None)
        .await
        .expect("peer");
    db::get_or_create_session(&test_db.pool, "ws", "sess", None, None)
        .await
        .expect("session");

    // alice was a member of "sess" only between 2020-01-01 and 2020-06-01.
    let joined = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let left = Utc.with_ymd_and_hms(2020, 6, 1, 0, 0, 0).unwrap();
    sqlx::query(
        "INSERT INTO session_peers (workspace_name, session_name, peer_name, joined_at, left_at) \
         VALUES ('ws', 'sess', 'alice', $1, $2)",
    )
    .bind(joined)
    .bind(left)
    .execute(&test_db.pool)
    .await
    .expect("seed membership");

    let in_window = MessageRef {
        public_id: "m_in".to_string(),
        session_name: "sess".to_string(),
        created_at: Utc.with_ymd_and_hms(2020, 3, 1, 0, 0, 0).unwrap(),
    };
    let before_join = MessageRef {
        public_id: "m_before".to_string(),
        session_name: "sess".to_string(),
        created_at: Utc.with_ymd_and_hms(2019, 12, 1, 0, 0, 0).unwrap(),
    };
    let after_leave = MessageRef {
        public_id: "m_after".to_string(),
        session_name: "sess".to_string(),
        created_at: Utc.with_ymd_and_hms(2020, 9, 1, 0, 0, 0).unwrap(),
    };
    let other_session = MessageRef {
        public_id: "m_other".to_string(),
        session_name: "other".to_string(),
        created_at: Utc.with_ymd_and_hms(2020, 3, 1, 0, 0, 0).unwrap(),
    };

    let kept = search::filter_by_peer_perspective(
        &test_db.pool,
        "ws",
        "alice",
        &[
            in_window.clone(),
            before_join,
            after_leave,
            other_session,
        ],
    )
    .await
    .expect("perspective filter");

    // Only the in-window message in a joined session survives.
    assert_eq!(kept, vec![in_window]);

    test_db.teardown().await;
}

#[tokio::test]
async fn fetch_messages_by_ids_preserves_order_and_drops_missing() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "first"),
            message("alice", "second"),
            message("alice", "third"),
        ],
        false,
        8192,
    )
    .await
    .expect("create messages");

    // Request in a deliberately scrambled order, with one unknown id mixed in.
    let requested = vec![
        created[2].public_id.clone(),
        "does-not-exist".to_string(),
        created[0].public_id.clone(),
        created[1].public_id.clone(),
    ];
    let hydrated = db::fetch_messages_by_ids(&test_db.pool, &requested)
        .await
        .expect("hydrate");

    // Output follows the requested order; the unknown id is dropped.
    let contents: Vec<&str> = hydrated
        .iter()
        .filter_map(|m| m["content"].as_str())
        .collect();
    assert_eq!(contents, vec!["third", "first", "second"]);

    // Empty input -> empty output, no query.
    let empty = db::fetch_messages_by_ids(&test_db.pool, &[])
        .await
        .expect("empty");
    assert!(empty.is_empty());

    test_db.teardown().await;
}

#[tokio::test]
async fn hybrid_search_fuses_semantic_and_fulltext_legs() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "alpha apple"),
            message("alice", "beta banana"),
            message("alice", "gamma cherry"),
        ],
        false,
        8192,
    )
    .await
    .expect("create messages");
    seed_embedding(&test_db.pool, &created[0].public_id, 0).await;
    seed_embedding(&test_db.pool, &created[1].public_id, 1).await;
    seed_embedding(&test_db.pool, &created[2].public_id, 2).await;

    // FTS "apple" matches only alpha; the embedding leans toward alpha (dim 0).
    // alpha appears top of BOTH legs, so RRF ranks it first.
    let embedding = query_vector(&[(0, 1.0), (1, 0.3), (2, 0.1)]);
    let filter = ws_filter("ws");
    let params = search::HybridSearchParams {
        workspace_name: "ws",
        query: "apple",
        filter: &filter,
        query_embedding: Some(&embedding),
        peer_perspective: None,
        limit: 10,
    };
    let results = search::hybrid_search(&test_db.pool, &params)
        .await
        .expect("hybrid search");

    let contents: Vec<&str> = results
        .iter()
        .filter_map(|m| m["content"].as_str())
        .collect();
    assert_eq!(contents.first(), Some(&"alpha apple"));
    // The semantic leg contributes all three; fusion keeps them.
    assert_eq!(contents.len(), 3);

    test_db.teardown().await;
}

#[tokio::test]
async fn hybrid_search_without_embedding_uses_fulltext_only() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "alpha apple"),
            message("alice", "beta banana"),
        ],
        false,
        8192,
    )
    .await
    .expect("create messages");

    // No embedding -> only the full-text leg runs; "banana" matches beta alone.
    let filter = ws_filter("ws");
    let params = search::HybridSearchParams {
        workspace_name: "ws",
        query: "banana",
        filter: &filter,
        query_embedding: None,
        peer_perspective: None,
        limit: 10,
    };
    let results = search::hybrid_search(&test_db.pool, &params)
        .await
        .expect("hybrid search");
    let contents: Vec<&str> = results
        .iter()
        .filter_map(|m| m["content"].as_str())
        .collect();
    assert_eq!(contents, vec!["beta banana"]);

    test_db.teardown().await;
}

#[tokio::test]
async fn enqueue_legs_observer_query_and_queue_insert() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("alice", "hi"), message("bob", "yo")],
        false,
        8192,
    )
    .await
    .expect("create messages");

    // get_session_peer_configuration returns both joined senders, active.
    let observers = db::get_session_peer_configuration(&test_db.pool, "ws", "sess")
        .await
        .expect("observer query");
    assert!(observers.contains_key("alice"));
    assert!(observers.contains_key("bob"));
    assert!(observers["alice"].is_active);

    // The queue FK references sessions.id (the internal nanoid), which the route
    // gets from fetch_session_for_enqueue — not the session name.
    let (session_id, _config) = db::fetch_session_for_enqueue(&test_db.pool, "ws", "sess")
        .await
        .expect("fetch session")
        .expect("session exists");

    // insert_queue_records writes the batched rows referencing a real message id.
    let records = vec![
        QueueRecord {
            work_unit_key: "wu1".to_string(),
            payload: json!({"task": "representation"}),
            session_id: session_id.clone(),
            task_type: "representation".to_string(),
            workspace_name: "ws".to_string(),
            message_id: created[0].id,
        },
        QueueRecord {
            work_unit_key: "wu2".to_string(),
            payload: json!({"task": "summary"}),
            session_id: session_id.clone(),
            task_type: "summary".to_string(),
            workspace_name: "ws".to_string(),
            message_id: created[0].id,
        },
    ];
    db::insert_queue_records(&test_db.pool, &records)
        .await
        .expect("insert queue records");

    let rows = sqlx::query(
        "SELECT task_type, payload FROM queue WHERE workspace_name = 'ws' ORDER BY task_type",
    )
    .fetch_all(&test_db.pool)
    .await
    .expect("query queue");
    assert_eq!(rows.len(), 2);
    let first_task: String = rows[0].get("task_type");
    assert_eq!(first_task, "representation"); // 'representation' < 'summary'
    let first_payload: Value = rows[0].get("payload");
    assert_eq!(first_payload, json!({"task": "representation"}));

    // Empty input is a no-op (no panic, no rows added).
    db::insert_queue_records(&test_db.pool, &[])
        .await
        .expect("empty insert");

    test_db.teardown().await;
}

#[tokio::test]
async fn delete_conclusion_soft_deletes_once() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    db::get_or_create_peer(&test_db.pool, "ws", "alice", None, None)
        .await
        .expect("peer");

    // Seed the (observer, observed) collection and a conclusion document.
    let collection_id = "a".repeat(21);
    let document_id = "b".repeat(21);
    sqlx::query(
        "INSERT INTO collections (id, workspace_name, observer, observed) \
         VALUES ($1, 'ws', 'alice', 'alice')",
    )
    .bind(&collection_id)
    .execute(&test_db.pool)
    .await
    .expect("seed collection");
    sqlx::query(
        "INSERT INTO documents (id, content, workspace_name, observer, observed) \
         VALUES ($1, 'a fact', 'ws', 'alice', 'alice')",
    )
    .bind(&document_id)
    .execute(&test_db.pool)
    .await
    .expect("seed document");

    // First delete soft-deletes (sets deleted_at) and reports success.
    let deleted = db::delete_conclusion(&test_db.pool, "ws", &document_id)
        .await
        .expect("delete");
    assert!(deleted);
    let deleted_at: Option<chrono::DateTime<Utc>> =
        sqlx::query_scalar("SELECT deleted_at FROM documents WHERE id = $1")
            .bind(&document_id)
            .fetch_one(&test_db.pool)
            .await
            .expect("read deleted_at");
    assert!(deleted_at.is_some());

    // A second delete is a no-op (WHERE deleted_at IS NULL) -> false.
    let again = db::delete_conclusion(&test_db.pool, "ws", &document_id)
        .await
        .expect("delete again");
    assert!(!again);

    // An unknown id -> false.
    let unknown = db::delete_conclusion(&test_db.pool, "ws", &"c".repeat(21))
        .await
        .expect("delete unknown");
    assert!(!unknown);

    test_db.teardown().await;
}

#[tokio::test]
async fn update_message_replaces_metadata_and_none_is_unchanged() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[MessageInsert {
            peer_name: "alice".to_string(),
            content: "hi".to_string(),
            metadata: json!({"a": 1}),
            created_at: None,
            token_count: 1,
        }],
        false,
        8192,
    )
    .await
    .expect("create message");
    let id = created[0].public_id.clone();

    // Some(metadata) overwrites.
    let updated = db::update_message(&test_db.pool, "ws", "sess", &id, Some(json!({"b": 2})))
        .await
        .expect("update")
        .expect("message exists");
    assert_eq!(updated["metadata"], json!({"b": 2}));

    // None returns the row unchanged (metadata still {"b":2}).
    let unchanged = db::update_message(&test_db.pool, "ws", "sess", &id, None)
        .await
        .expect("update none")
        .expect("message exists");
    assert_eq!(unchanged["metadata"], json!({"b": 2}));

    // get_message round-trips; unknown id -> None.
    let fetched = db::get_message(&test_db.pool, "ws", "sess", &id)
        .await
        .expect("get message");
    assert_eq!(fetched.expect("exists")["id"], json!(id));
    let missing = db::get_message(&test_db.pool, "ws", "sess", "nope")
        .await
        .expect("get missing");
    assert_eq!(missing, None);

    test_db.teardown().await;
}

#[tokio::test]
async fn peer_card_self_and_other_labels_coexist() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");

    // Self card -> stored under "peer_card".
    db::set_peer_card(&test_db.pool, "ws", "alice", "alice", &["i am alice".to_string()])
        .await
        .expect("set self card");
    let self_card = db::get_peer_card(&test_db.pool, "ws", "alice", "alice")
        .await
        .expect("get self card");
    assert_eq!(self_card, Some(json!({"peer_card": ["i am alice"]})));

    // Card about another peer -> stored under "{observed}_peer_card", merged
    // alongside the self card rather than clobbering it.
    db::set_peer_card(&test_db.pool, "ws", "alice", "bob", &["bob is helpful".to_string()])
        .await
        .expect("set other card");
    let bob_card = db::get_peer_card(&test_db.pool, "ws", "alice", "bob")
        .await
        .expect("get other card");
    assert_eq!(bob_card, Some(json!({"peer_card": ["bob is helpful"]})));

    // The self card is still intact after the other-card merge.
    let still_self = db::get_peer_card(&test_db.pool, "ws", "alice", "alice")
        .await
        .expect("get self card again");
    assert_eq!(still_self, Some(json!({"peer_card": ["i am alice"]})));

    // A peer with no card row at all -> None.
    let missing = db::get_peer_card(&test_db.pool, "ws", "ghost", "ghost")
        .await
        .expect("get missing card");
    assert_eq!(missing, None);

    test_db.teardown().await;
}

async fn message_count(pool: &PgPool, session_name: &str) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM messages WHERE session_name = $1 AND workspace_name = 'ws'")
        .bind(session_name)
        .fetch_one(pool)
        .await
        .expect("count messages")
}

#[tokio::test]
async fn clone_session_copies_messages_up_to_cutoff() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "one"),
            message("alice", "two"),
            message("alice", "three"),
        ],
        false,
        8192,
    )
    .await
    .expect("create messages");

    // Clone up to (and including) the second message.
    let cloned = db::clone_session(&test_db.pool, "ws", "sess", Some(&created[1].public_id))
        .await
        .expect("clone with cutoff");
    let cloned_name = cloned["id"].as_str().expect("clone session id").to_string();
    assert_ne!(cloned_name, "sess");
    assert_eq!(message_count(&test_db.pool, &cloned_name).await, 2);

    // Cloning with no cutoff copies the whole session.
    let full = db::clone_session(&test_db.pool, "ws", "sess", None)
        .await
        .expect("clone all");
    let full_name = full["id"].as_str().expect("full session id").to_string();
    assert_eq!(message_count(&test_db.pool, &full_name).await, 3);

    // Cloning a nonexistent session is an error, not a panic.
    let missing = db::clone_session(&test_db.pool, "ws", "nope", None).await;
    assert!(missing.is_err());

    test_db.teardown().await;
}

/// A message with an explicit token_count, for exercising the running-token-sum
/// window in `get_session_context` / `get_messages_id_range`.
fn message_tok(peer: &str, content: &str, token_count: i32) -> MessageInsert {
    MessageInsert {
        peer_name: peer.to_string(),
        content: content.to_string(),
        metadata: json!({}),
        created_at: None,
        token_count,
    }
}

async fn set_session_summaries(pool: &PgPool, session_name: &str, summaries: Value) {
    sqlx::query("UPDATE sessions SET internal_metadata = $1 WHERE name = $2")
        .bind(json!({ "summaries": summaries }))
        .bind(session_name)
        .execute(pool)
        .await
        .expect("set session summaries");
}

#[tokio::test]
async fn get_session_context_selects_long_summary_and_windows_messages() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message_tok("alice", "m1", 10),
            message_tok("alice", "m2", 10),
            message_tok("alice", "m3", 10),
            message_tok("alice", "m4", 10),
        ],
        false,
        8192,
    )
    .await
    .expect("create messages");

    // Long summary (token_count 30) covers up to m2; short (10) covers up to m1.
    // With token_limit 100 -> summary budget 40; long fits (30<=40) and is larger
    // than short (10), so it is chosen. message budget becomes 70, start at m2.id.
    set_session_summaries(
        &test_db.pool,
        "sess",
        json!({
            "honcho_chat_summary_long": {
                "content": "LONG",
                "message_id": created[1].id,
                "summary_type": "long",
                "created_at": "2026-01-01T00:00:00Z",
                "token_count": 30,
                "message_public_id": created[1].public_id,
            },
            "honcho_chat_summary_short": {
                "content": "SHORT",
                "message_id": created[0].id,
                "summary_type": "short",
                "created_at": "2026-01-01T00:00:00Z",
                "token_count": 10,
                "message_public_id": created[0].public_id,
            },
        }),
    )
    .await;

    let ctx = db::get_session_context(&test_db.pool, "ws", "sess", 100, true)
        .await
        .expect("get context");

    assert_eq!(ctx["id"], json!("sess"));
    assert_eq!(ctx["peer_representation"], Value::Null);
    assert_eq!(ctx["peer_card"], Value::Null);
    assert_eq!(
        ctx["summary"],
        json!({
            "content": "LONG",
            "message_id": created[1].public_id,
            "summary_type": "long",
            "created_at": "2026-01-01T00:00:00Z",
            "token_count": 30
        })
    );
    // Messages from m2 onward (id >= start), ascending; all fit in the 70 budget.
    let ids: Vec<&str> = ctx["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        vec![
            created[1].public_id.as_str(),
            created[2].public_id.as_str(),
            created[3].public_id.as_str(),
        ]
    );

    test_db.teardown().await;
}

#[tokio::test]
async fn get_session_context_token_budget_drops_oldest_messages() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message_tok("alice", "m1", 10),
            message_tok("alice", "m2", 10),
            message_tok("alice", "m3", 10),
            message_tok("alice", "m4", 10),
        ],
        false,
        8192,
    )
    .await
    .expect("create messages");

    // No summaries set: full token_limit goes to messages from the start. The
    // descending running sum keeps only the newest messages within the budget.
    // limit 25 -> m4 (sum 10), m3 (sum 20); m2 would be 30 (> 25) -> excluded.
    let ctx = db::get_session_context(&test_db.pool, "ws", "sess", 25, true)
        .await
        .expect("get context");

    assert_eq!(ctx["summary"], Value::Null);
    let ids: Vec<&str> = ctx["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        vec![created[2].public_id.as_str(), created[3].public_id.as_str()]
    );

    // A non-positive token limit short-circuits to an empty context.
    let empty = db::get_session_context(&test_db.pool, "ws", "sess", 0, true)
        .await
        .expect("get empty context");
    assert_eq!(empty["messages"], json!([]));
    assert_eq!(empty["summary"], Value::Null);

    test_db.teardown().await;
}

#[tokio::test]
async fn enqueue_workspace_deletion_inserts_queue_item_and_guards() {
    use honcho_api_rs::db::WorkspaceDeleteError;
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // Missing workspace -> NotFound (404), no queue row.
    let missing = db::enqueue_workspace_deletion(&test_db.pool, "nope").await;
    assert!(matches!(missing, Err(WorkspaceDeleteError::NotFound)));

    db::get_or_create_workspace(&test_db.pool, "ws-del", json!({}), json!({}))
        .await
        .expect("workspace");

    // Workspace exists, no sessions -> accepted; one deletion queue item written.
    db::enqueue_workspace_deletion(&test_db.pool, "ws-del")
        .await
        .expect("enqueue deletion");

    let row = sqlx::query(
        "SELECT work_unit_key, payload, task_type, workspace_name, session_id, message_id \
         FROM queue WHERE workspace_name = $1 AND task_type = 'deletion'",
    )
    .bind("ws-del")
    .fetch_one(&test_db.pool)
    .await
    .expect("deletion queue row");
    let work_unit_key: String = row.get("work_unit_key");
    let payload: Value = row.get("payload");
    let session_id: Option<String> = row.get("session_id");
    let message_id: Option<i64> = row.get("message_id");
    assert_eq!(work_unit_key, "deletion:ws-del:workspace:ws-del");
    assert_eq!(
        payload,
        json!({"task_type": "deletion", "deletion_type": "workspace", "resource_id": "ws-del"})
    );
    assert_eq!(session_id, None);
    assert_eq!(message_id, None);

    // Create an active session -> deletion is refused with ActiveSessions (409).
    db::create_messages(
        &test_db.pool,
        "ws-del",
        "sess-active",
        &[message("alice", "hi")],
        false,
        8192,
    )
    .await
    .expect("create message in session");
    let conflict = db::enqueue_workspace_deletion(&test_db.pool, "ws-del").await;
    assert!(matches!(conflict, Err(WorkspaceDeleteError::ActiveSessions)));

    test_db.teardown().await;
}

#[tokio::test]
async fn enqueue_dream_inserts_queue_item_and_dedups() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws-dream", json!({}), json!({}))
        .await
        .expect("workspace");

    // First enqueue writes a single `dream` queue item.
    db::enqueue_dream(&test_db.pool, "ws-dream", "alice", "alice", "omni", None)
        .await
        .expect("enqueue dream");

    let row = sqlx::query(
        "SELECT work_unit_key, payload, task_type, workspace_name, session_id, message_id \
         FROM queue WHERE workspace_name = $1 AND task_type = 'dream'",
    )
    .bind("ws-dream")
    .fetch_one(&test_db.pool)
    .await
    .expect("dream queue row");
    let work_unit_key: String = row.get("work_unit_key");
    let payload: Value = row.get("payload");
    let session_id: Option<String> = row.get("session_id");
    let message_id: Option<i64> = row.get("message_id");
    assert_eq!(work_unit_key, "dream:omni:ws-dream:alice:alice");
    assert_eq!(
        payload,
        json!({
            "task_type": "dream",
            "dream_type": "omni",
            "observer": "alice",
            "observed": "alice",
            "trigger_reason": "manual",
            "delay_reason": "immediate"
        })
    );
    assert_eq!(session_id, None);
    assert_eq!(message_id, None);

    // A second enqueue with the same work_unit_key is skipped while the first is
    // still pending (processed = false) -> still exactly one row.
    db::enqueue_dream(&test_db.pool, "ws-dream", "alice", "alice", "omni", None)
        .await
        .expect("enqueue dream (dedup)");
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM queue WHERE workspace_name = $1 AND task_type = 'dream'",
    )
    .bind("ws-dream")
    .fetch_one(&test_db.pool)
    .await
    .expect("count dream rows");
    assert_eq!(count, 1);

    test_db.teardown().await;
}

#[tokio::test]
async fn grep_messages_builds_and_merges_context_snippets() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // seqs 1..6 assigned in insertion order.
    db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "alpha"),
            message("bob", "find me here"),
            message("alice", "beta"),
            message("bob", "gamma"),
            message("alice", "delta"),
            message("bob", "find me too"),
        ],
        false,
        8192,
    )
    .await
    .expect("seed messages");

    // context_window = 1: the two matches (seq 2 and seq 6) are far enough apart
    // (3 + 1 < 5) to stay separate -> two snippets, each re-sorted by sequence.
    let snippets: Vec<MessageSnippet> =
        db::grep_messages(&test_db.pool, "ws", Some("sess"), "find me", 10, 1, None)
            .await
            .expect("grep cw=1");
    assert_eq!(snippets.len(), 2);
    // First snippet: match seq 2, context seq 1..=3.
    assert_eq!(snippets[0].matched.len(), 1);
    assert_eq!(snippets[0].matched[0]["content"], json!("find me here"));
    assert_eq!(snippets[0].context.len(), 3);
    assert_eq!(snippets[0].context[0]["content"], json!("alpha"));
    assert_eq!(snippets[0].context[2]["content"], json!("beta"));
    // Second snippet: match seq 6, context seq 5..=6 (no seq 7 exists).
    assert_eq!(snippets[1].matched.len(), 1);
    assert_eq!(snippets[1].matched[0]["content"], json!("find me too"));
    assert_eq!(snippets[1].context.len(), 2);
    assert_eq!(snippets[1].context[0]["content"], json!("delta"));

    // context_window = 2: seq-2 window [0,4] and seq-6 window [4,8] are adjacent
    // (4 <= 4 + 1) -> merged into one snippet covering all six messages.
    let merged: Vec<MessageSnippet> =
        db::grep_messages(&test_db.pool, "ws", Some("sess"), "find me", 10, 2, None)
            .await
            .expect("grep cw=2");
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].matched.len(), 2);
    assert_eq!(merged[0].context.len(), 6);

    // observer with no session memberships -> empty, before any message scan.
    let none = db::grep_messages(&test_db.pool, "ws", None, "find me", 10, 1, Some("ghost"))
        .await
        .expect("grep ghost observer");
    assert!(none.is_empty());

    test_db.teardown().await;
}

#[tokio::test]
async fn search_messages_semantic_ranks_and_windows() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // seqs 1..3; one-hot embeddings on distinct dimensions.
    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "red"),
            message("bob", "green"),
            message("alice", "blue"),
        ],
        false,
        8192,
    )
    .await
    .expect("seed messages");
    seed_embedding(&test_db.pool, &created[0].public_id, 0).await;
    seed_embedding(&test_db.pool, &created[1].public_id, 1).await;
    seed_embedding(&test_db.pool, &created[2].public_id, 2).await;

    // Query closest to dim 1 -> the "green" message (seq 2) is the top match.
    let embedding = query_vector(&[(1, 1.0), (0, 0.1), (2, 0.1)]);
    let snippets = db::search_messages_semantic(
        &test_db.pool,
        "ws",
        Some("sess"),
        None,
        &embedding,
        None,
        None,
        1,
        1,
    )
    .await
    .expect("semantic search");

    assert_eq!(snippets.len(), 1);
    assert_eq!(snippets[0].matched.len(), 1);
    assert_eq!(snippets[0].matched[0]["content"], json!("green"));
    // context_window 1 around seq 2 -> seqs 1..=3 (all three messages).
    assert_eq!(snippets[0].context.len(), 3);
    assert_eq!(snippets[0].context[0]["content"], json!("red"));
    assert_eq!(snippets[0].context[2]["content"], json!("blue"));

    // A far-future after-date filters everything out.
    let future = chrono::Utc::now() + chrono::Duration::days(3650);
    let none = db::search_messages_semantic(
        &test_db.pool,
        "ws",
        Some("sess"),
        None,
        &embedding,
        Some(future),
        None,
        10,
        1,
    )
    .await
    .expect("semantic search after-date");
    assert!(none.is_empty());

    test_db.teardown().await;
}

#[tokio::test]
async fn get_observation_context_expands_around_targets() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // seqs 1..5.
    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "one"),
            message("bob", "two"),
            message("alice", "three"),
            message("bob", "four"),
            message("alice", "five"),
        ],
        false,
        8192,
    )
    .await
    .expect("seed messages");

    // Empty ids -> empty, no query.
    let empty = db::get_observation_context(&test_db.pool, "ws", Some("sess"), &[], None)
        .await
        .expect("empty ids");
    assert!(empty.is_empty());

    // Single target at seq 3 -> seqs 2..=4 in ascending order.
    let single = db::get_observation_context(
        &test_db.pool,
        "ws",
        Some("sess"),
        &[created[2].public_id.clone()],
        None,
    )
    .await
    .expect("single target");
    let contents: Vec<&str> = single.iter().filter_map(|m| m["content"].as_str()).collect();
    assert_eq!(contents, vec!["two", "three", "four"]);

    // Two targets at seq 1 and seq 5 -> the two ±1 windows {1,2} and {4,5}
    // (seq 0 and 6 don't exist), still globally ordered ascending.
    let two = db::get_observation_context(
        &test_db.pool,
        "ws",
        Some("sess"),
        &[created[0].public_id.clone(), created[4].public_id.clone()],
        None,
    )
    .await
    .expect("two targets");
    let contents: Vec<&str> = two.iter().filter_map(|m| m["content"].as_str()).collect();
    assert_eq!(contents, vec!["one", "two", "four", "five"]);

    test_db.teardown().await;
}

#[tokio::test]
async fn dialectic_read_tool_handlers_format_results() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "the quarterly report is ready"),
            message("bob", "thanks for the report"),
            message("alice", "see you tomorrow"),
        ],
        false,
        8192,
    )
    .await
    .expect("seed messages");

    let ctx = dialectic::ToolContext {
        workspace_name: "ws".to_string(),
        observer: "alice".to_string(),
        observed: "alice".to_string(),
        session_name: Some("sess".to_string()),
    };

    // grep: missing text -> ERROR; a match -> formatted snippet header.
    let err = dialectic::handle_grep_messages(&test_db.pool, &ctx, &json!({}))
        .await
        .expect("grep no text");
    assert_eq!(err, "ERROR: 'text' parameter is required");

    let grep = dialectic::handle_grep_messages(
        &test_db.pool,
        &ctx,
        &json!({"text": "report", "context_window": 0}),
    )
    .await
    .expect("grep report");
    // seq 1 & 2 both match and are adjacent (start <= prev_end + 1) -> one snippet.
    assert!(grep.starts_with("Found 2 messages containing 'report' in 1 conversation snippets:"));

    let grep_none = dialectic::handle_grep_messages(&test_db.pool, &ctx, &json!({"text": "zzz"}))
        .await
        .expect("grep none");
    assert_eq!(grep_none, "No messages found containing 'zzz'");

    // date_range: bad date -> ERROR; default -> "all time, newest first".
    let bad = dialectic::handle_get_messages_by_date_range(
        &test_db.pool,
        &ctx,
        &json!({"after_date": "nonsense"}),
    )
    .await
    .expect("bad date");
    assert_eq!(
        bad,
        "ERROR: Invalid after_date format 'nonsense'. Use ISO format (e.g., '2024-01-15')"
    );

    let range = dialectic::handle_get_messages_by_date_range(&test_db.pool, &ctx, &json!({}))
        .await
        .expect("date range");
    assert!(range.starts_with("Found 3 messages (all time, newest first):"));

    // observation_context: empty ids -> Python-repr sentinel; a real id -> output.
    let empty = dialectic::handle_get_observation_context(&test_db.pool, &ctx, &json!({"message_ids": []}))
        .await
        .expect("obs empty");
    assert_eq!(empty, "No messages found for IDs []");

    let obs = dialectic::handle_get_observation_context(
        &test_db.pool,
        &ctx,
        &json!({"message_ids": [created[1].public_id]}),
    )
    .await
    .expect("obs context");
    // seq 2 target, ±1 -> 3 messages with context.
    assert!(obs.starts_with("Retrieved 3 messages with context:"));

    test_db.teardown().await;
}

#[tokio::test]
async fn dialectic_search_tool_handlers_format_results() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "red apple"),
            message("bob", "green pear"),
            message("alice", "blue plum"),
        ],
        false,
        8192,
    )
    .await
    .expect("seed messages");
    seed_embedding(&test_db.pool, &created[0].public_id, 0).await;
    seed_embedding(&test_db.pool, &created[1].public_id, 1).await;
    seed_embedding(&test_db.pool, &created[2].public_id, 2).await;

    let ctx = dialectic::ToolContext {
        workspace_name: "ws".to_string(),
        observer: "alice".to_string(),
        observed: "alice".to_string(),
        session_name: Some("sess".to_string()),
    };
    let embedding = query_vector(&[(1, 1.0), (0, 0.1), (2, 0.1)]); // closest to dim 1

    // query required.
    let err = dialectic::handle_search_messages(&test_db.pool, &ctx, &json!({}), &embedding)
        .await
        .expect("search no query");
    assert_eq!(err, "ERROR: 'query' parameter is required");

    let search = dialectic::handle_search_messages(
        &test_db.pool,
        &ctx,
        &json!({"query": "fruit"}),
        &embedding,
    )
    .await
    .expect("search messages");
    // All three messages have embeddings -> all returned as matches, merged into
    // one snippet (seqs 1..3, context_window 2).
    assert!(search.starts_with("Found 3 matching messages in 1 conversation snippets for query 'fruit':"));
    assert!(search.contains("green pear"));

    // Temporal with a far-future after-date -> no results, with date suffix.
    let temporal = dialectic::handle_search_messages_temporal(
        &test_db.pool,
        &ctx,
        &json!({"query": "fruit", "after_date": "2999-01-01"}),
        &embedding,
    )
    .await
    .expect("search temporal");
    assert_eq!(
        temporal,
        "No messages found for query 'fruit' (after 2999-01-01)"
    );

    // Bad ISO date -> ERROR sentinel.
    let bad = dialectic::handle_search_messages_temporal(
        &test_db.pool,
        &ctx,
        &json!({"query": "fruit", "before_date": "nope"}),
        &embedding,
    )
    .await
    .expect("bad temporal date");
    assert_eq!(
        bad,
        "ERROR: Invalid before_date format 'nope'. Use ISO format (e.g., '2024-01-15')"
    );

    test_db.teardown().await;
}

struct FixedEmbedder(Vec<f32>);
impl dialectic::Embedder for FixedEmbedder {
    async fn embed(&self, _query: &str) -> Result<Vec<f32>, String> {
        Ok(self.0.clone())
    }
    async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts.iter().map(|_| self.0.clone()).collect())
    }
}

#[tokio::test]
async fn dialectic_tool_executor_dispatches_by_name() {
    use honcho_api_rs::llm::tool_loop::ToolExecutor;
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "the report is here"),
            message("bob", "ok thanks"),
        ],
        false,
        8192,
    )
    .await
    .expect("seed messages");
    seed_embedding(&test_db.pool, &created[0].public_id, 0).await;
    seed_embedding(&test_db.pool, &created[1].public_id, 1).await;

    let executor = dialectic::DialecticToolExecutor {
        pool: &test_db.pool,
        ctx: dialectic::ToolContext {
            workspace_name: "ws".to_string(),
            observer: "alice".to_string(),
            observed: "alice".to_string(),
            session_name: Some("sess".to_string()),
        },
        embedder: FixedEmbedder(query_vector(&[(0, 1.0)])),
    };

    // DB-only tool dispatched to its handler.
    let grep = executor
        .execute("grep_messages", &json!({"text": "report"}))
        .await
        .expect("grep ok");
    assert!(grep.starts_with("Found 1 messages containing 'report'"));

    // Semantic tool: embedder seam supplies the vector.
    let search = executor
        .execute("search_messages", &json!({"query": "report"}))
        .await
        .expect("search ok");
    assert!(search.starts_with("Found 2 matching messages"));

    // Unknown tool -> Err (the loop folds this into an is_error result).
    let unknown = executor.execute("create_observations", &json!({})).await;
    assert!(unknown.is_err());

    test_db.teardown().await;
}

/// A real (or arbitrary) 1536-d vector as a pgvector text literal `[v1,v2,...]`.
fn vector_literal(values: &[f32]) -> String {
    let mut s = String::from("[");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&v.to_string());
    }
    s.push(']');
    s
}

async fn seed_embedding_vector(
    pool: &PgPool,
    message_public_id: &str,
    peer: &str,
    content: &str,
    vector: &[f32],
) {
    sqlx::query(
        "INSERT INTO message_embeddings \
         (content, embedding, message_id, workspace_name, session_name, peer_name, sync_state) \
         VALUES ($1, $2::vector, $3, 'ws', 'sess', $4, 'synced')",
    )
    .bind(content)
    .bind(vector_literal(vector))
    .bind(message_public_id)
    .bind(peer)
    .execute(pool)
    .await
    .expect("seed embedding vector");
}

#[tokio::test]
async fn search_filter_scopes_fulltext_to_peer() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // Both peers send a message matching "report"; the peer_id filter must keep
    // only alice's, exercising the FilterClause threaded into the FTS leg.
    db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "alice quarterly report"),
            message("bob", "bob annual report"),
        ],
        false,
        8192,
    )
    .await
    .expect("create messages");

    let unfiltered = search::fulltext_search(&test_db.pool, &ws_filter("ws"), "report", 10)
        .await
        .expect("fts unfiltered");
    assert_eq!(unfiltered.len(), 2);

    let alice_filter = message_filter(json!({"workspace_id": "ws", "peer_id": "alice"}));
    let alice_only = search::fulltext_search(&test_db.pool, &alice_filter, "report", 10)
        .await
        .expect("fts filtered");
    assert_eq!(alice_only.len(), 1);

    test_db.teardown().await;
}

/// End-to-end semantic search against the live OpenAI embeddings API. Gated on
/// BOTH `TEST_DATABASE_URL` and `OPENAI_TEST_TOKEN`; skips cleanly otherwise.
/// Embeds two message bodies and a query for real, stores the message vectors as
/// pgvector rows, and asserts the semantically-relevant message tops the fused
/// hybrid-search ranking.
#[tokio::test]
async fn search_with_live_openai_embedding_ranks_semantically() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    let Some(api_key) = openai_test_token() else {
        test_db.teardown().await;
        return;
    };

    let http = ReqwestHttp::default();
    let credentials = Credentials::new(api_key);
    let model = "text-embedding-3-small";
    let dims = 1536;

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message("alice", "I really enjoy eating fresh apples, mangoes and oranges"),
            message("alice", "The quarterly financial report shows rising interest rates"),
        ],
        false,
        8192,
    )
    .await
    .expect("create messages");

    for msg in &created {
        let vector =
            embedding::embed_openai(&http, &credentials, model, &msg.content, dims, false, 8192)
                .await
                .expect("embed message content");
        assert_eq!(vector.len(), dims);
        seed_embedding_vector(&test_db.pool, &msg.public_id, "alice", &msg.content, &vector).await;
    }

    let query_embedding =
        embedding::embed_openai(&http, &credentials, model, "tropical fruit", dims, false, 8192)
            .await
            .expect("embed query");

    let filter = ws_filter("ws");
    let params = HybridSearchParams {
        workspace_name: "ws",
        query: "tropical fruit",
        filter: &filter,
        query_embedding: Some(&query_embedding),
        peer_perspective: None,
        limit: 10,
    };
    let results = search::hybrid_search(&test_db.pool, &params)
        .await
        .expect("hybrid search");

    // The fruit message is semantically closest to "tropical fruit".
    assert_eq!(
        results.first().and_then(|m| m["content"].as_str()),
        Some(created[0].content.as_str())
    );

    test_db.teardown().await;
}

/// The live OpenAI key, treating an unset *or empty* `OPENAI_TEST_TOKEN` as
/// "skip" so the gated tests don't false-fail with a blank key.
fn openai_test_token() -> Option<String> {
    std::env::var("OPENAI_TEST_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
}

/// A valid 21-char document id (nanoid charset) for seeding.
fn doc_id(n: usize) -> String {
    format!("doc{n:018}")
}

#[allow(clippy::too_many_arguments)]
async fn seed_document(
    pool: &PgPool,
    id: &str,
    observer: &str,
    observed: &str,
    content: &str,
    vector: &[f32],
) {
    sqlx::query(
        "INSERT INTO documents \
         (id, content, embedding, workspace_name, observer, observed, level, sync_state) \
         VALUES ($1, $2, $3::vector, 'ws', $4, $5, 'explicit', 'synced')",
    )
    .bind(id)
    .bind(content)
    .bind(vector_literal(vector))
    .bind(observer)
    .bind(observed)
    .execute(pool)
    .await
    .expect("seed document");
}

/// End-to-end semantic conclusion query against live OpenAI embeddings. Gated on
/// `TEST_DATABASE_URL` + `OPENAI_TEST_TOKEN`. Seeds two conclusions for an
/// (observer, observed) pair and asserts the semantically relevant one tops the
/// cosine-ordered result.
#[tokio::test]
async fn query_documents_with_live_openai_ranks_semantically() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    let Some(api_key) = openai_test_token() else {
        test_db.teardown().await;
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["observer_peer", "observed_peer"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    // documents FK the (observer, observed, workspace_name) collection.
    sqlx::query(
        "INSERT INTO collections (id, workspace_name, observer, observed) \
         VALUES ($1, 'ws', 'observer_peer', 'observed_peer')",
    )
    .bind(doc_id(99))
    .execute(&test_db.pool)
    .await
    .expect("seed collection");

    let http = ReqwestHttp::default();
    let credentials = Credentials::new(api_key);
    let model = "text-embedding-3-small";

    let conclusions = [
        "The user loves eating apples, mangoes and other tropical fruit",
        "The user is worried about interest rates and the stock market",
    ];
    for (i, content) in conclusions.iter().enumerate() {
        let vector =
            embedding::embed_openai(&http, &credentials, model, content, 1536, false, 8192)
                .await
                .expect("embed document");
        seed_document(
            &test_db.pool,
            &doc_id(i + 1),
            "observer_peer",
            "observed_peer",
            content,
            &vector,
        )
        .await;
    }

    let query_embedding =
        embedding::embed_openai(&http, &credentials, model, "fruit", 1536, false, 8192)
            .await
            .expect("embed query");

    let filter_value = json!({
        "workspace_id": "ws",
        "observer_id": "observer_peer",
        "observed_id": "observed_peer"
    });
    let filter = build_filter_clause(FilterTarget::Conclusion, Some(&filter_value))
        .expect("valid conclusion filter");

    let results = db::query_documents_pgvector(
        &test_db.pool,
        "ws",
        "observer_peer",
        "observed_peer",
        &query_embedding,
        &filter,
        None,
        10,
    )
    .await
    .expect("query documents");

    assert_eq!(results.len(), 2);
    assert_eq!(
        results.first().and_then(|d| d["content"].as_str()),
        Some(conclusions[0])
    );

    test_db.teardown().await;
}

#[tokio::test]
async fn create_conclusions_validates_inserts_and_marks_synced() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    let Some(api_key) = openai_test_token() else {
        test_db.teardown().await;
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["observer_peer", "observed_peer"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }

    let new = |content: &str| NewConclusion {
        content: content.to_string(),
        observer_id: "observer_peer".to_string(),
        observed_id: "observed_peer".to_string(),
        session_id: None,
    };

    // A missing peer is a 404 before any embedding.
    let bad = vec![NewConclusion {
        observer_id: "ghost_peer".to_string(),
        ..new("x")
    }];
    assert!(matches!(
        db::prepare_conclusions(&test_db.pool, "ws", &bad).await,
        Err(ConclusionWriteError::PeerNotFound(name)) if name == "ghost_peer"
    ));

    let conclusions = vec![new("The user enjoys long-distance hiking on weekends")];
    db::prepare_conclusions(&test_db.pool, "ws", &conclusions)
        .await
        .expect("prepare");

    // The collection was get-or-created.
    let collection_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM collections \
         WHERE workspace_name = 'ws' AND observer = 'observer_peer' AND observed = 'observed_peer'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("count collections");
    assert_eq!(collection_count, 1);

    let http = ReqwestHttp::default();
    let credentials = Credentials::new(api_key);
    let mut embeddings = Vec::new();
    for conclusion in &conclusions {
        embeddings.push(
            embedding::embed_openai(
                &http,
                &credentials,
                "text-embedding-3-small",
                &conclusion.content,
                1536,
                false,
                8192,
            )
            .await
            .expect("embed"),
        );
    }

    let created = db::insert_conclusions(&test_db.pool, "ws", &conclusions, &embeddings)
        .await
        .expect("insert conclusions");
    assert_eq!(created.len(), 1);
    assert_eq!(created[0]["content"], json!("The user enjoys long-distance hiking on weekends"));
    assert_eq!(created[0]["observer_id"], json!("observer_peer"));
    assert_eq!(created[0]["observed_id"], json!("observed_peer"));

    // The stored document is explicit, synced, and carries its embedding.
    let id = created[0]["id"].as_str().expect("doc id");
    let row = sqlx::query(
        "SELECT level, sync_state, embedding IS NOT NULL AS has_embedding \
         FROM documents WHERE id = $1",
    )
    .bind(id)
    .fetch_one(&test_db.pool)
    .await
    .expect("fetch document");
    let level: String = row.get("level");
    let sync_state: String = row.get("sync_state");
    let has_embedding: bool = row.get("has_embedding");
    assert_eq!(level, "explicit");
    assert_eq!(sync_state, "synced");
    assert!(has_embedding);

    test_db.teardown().await;
}

fn message_at(peer: &str, content: &str, at: chrono::DateTime<Utc>) -> MessageInsert {
    MessageInsert {
        peer_name: peer.to_string(),
        content: content.to_string(),
        metadata: json!({}),
        created_at: Some(at),
        token_count: 1,
    }
}

#[tokio::test]
async fn get_messages_by_date_range_filters_orders_and_scopes() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap();
    let t3 = Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap();

    db::create_messages(
        &test_db.pool,
        "ws",
        "s1",
        &[
            message_at("alice", "jan", t1),
            message_at("alice", "feb", t2),
            message_at("alice", "mar", t3),
        ],
        false,
        8192,
    )
    .await
    .expect("messages s1");
    db::create_messages(
        &test_db.pool,
        "ws",
        "s2",
        &[message_at("bob", "bob-feb", t2)],
        false,
        8192,
    )
    .await
    .expect("messages s2");

    // after=t2, session s1, desc -> feb & mar, newest first.
    let after = db::get_messages_by_date_range(
        &test_db.pool,
        "ws",
        Some("s1"),
        None,
        Some(t2),
        None,
        20,
        true,
    )
    .await
    .expect("after range");
    let contents: Vec<&str> = after.iter().filter_map(|m| m["content"].as_str()).collect();
    assert_eq!(contents, vec!["mar", "feb"]);

    // asc order flips it.
    let asc = db::get_messages_by_date_range(
        &test_db.pool,
        "ws",
        Some("s1"),
        None,
        Some(t2),
        None,
        20,
        false,
    )
    .await
    .expect("asc range");
    let asc_contents: Vec<&str> = asc.iter().filter_map(|m| m["content"].as_str()).collect();
    assert_eq!(asc_contents, vec!["feb", "mar"]);

    // before=t1 -> only jan.
    let before = db::get_messages_by_date_range(
        &test_db.pool,
        "ws",
        Some("s1"),
        None,
        None,
        Some(t1),
        20,
        true,
    )
    .await
    .expect("before range");
    let before_contents: Vec<&str> = before
        .iter()
        .filter_map(|m| m["content"].as_str())
        .collect();
    assert_eq!(before_contents, vec!["jan"]);

    // observer "bob" (no session arg) is scoped to s2 only.
    let bob = db::get_messages_by_date_range(
        &test_db.pool,
        "ws",
        None,
        Some("bob"),
        None,
        None,
        20,
        true,
    )
    .await
    .expect("observer scope");
    let bob_contents: Vec<&str> = bob.iter().filter_map(|m| m["content"].as_str()).collect();
    assert_eq!(bob_contents, vec!["bob-feb"]);

    // An observer with no memberships sees nothing.
    let ghost = db::get_messages_by_date_range(
        &test_db.pool,
        "ws",
        None,
        Some("ghost"),
        None,
        None,
        20,
        true,
    )
    .await
    .expect("ghost scope");
    assert!(ghost.is_empty());

    test_db.teardown().await;
}

#[allow(clippy::too_many_arguments)]
async fn seed_document_full(
    pool: &PgPool,
    id: &str,
    content: &str,
    times_derived: i32,
    created_at: chrono::DateTime<Utc>,
) {
    sqlx::query(
        "INSERT INTO documents \
         (id, content, workspace_name, observer, observed, level, times_derived, created_at, sync_state) \
         VALUES ($1, $2, 'ws', 'obs', 'obsd', 'explicit', $3, $4, 'synced')",
    )
    .bind(id)
    .bind(content)
    .bind(times_derived)
    .bind(created_at)
    .execute(pool)
    .await
    .expect("seed document full");
}

#[tokio::test]
async fn query_documents_recent_and_most_derived_order_correctly() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    sqlx::query(
        "INSERT INTO collections (id, workspace_name, observer, observed) \
         VALUES ($1, 'ws', 'obs', 'obsd')",
    )
    .bind(doc_id(99))
    .execute(&test_db.pool)
    .await
    .expect("collection");

    let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap();
    let t3 = Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap();
    // (content, times_derived, created_at)
    seed_document_full(&test_db.pool, &doc_id(1), "oldest_low", 1, t1).await;
    seed_document_full(&test_db.pool, &doc_id(2), "middle_high", 5, t2).await;
    seed_document_full(&test_db.pool, &doc_id(3), "newest_mid", 3, t3).await;

    // Recent: newest created_at first.
    let recent = db::query_documents_recent(&test_db.pool, "ws", "obs", "obsd", None, 10)
        .await
        .expect("recent");
    let recent_contents: Vec<&str> = recent.iter().filter_map(|d| d["content"].as_str()).collect();
    assert_eq!(recent_contents, vec!["newest_mid", "middle_high", "oldest_low"]);

    // Most derived: highest times_derived first (5, 3, 1).
    let derived = db::query_documents_most_derived(&test_db.pool, "ws", "obs", "obsd", 10)
        .await
        .expect("most derived");
    let derived_contents: Vec<&str> = derived.iter().filter_map(|d| d["content"].as_str()).collect();
    assert_eq!(derived_contents, vec!["middle_high", "newest_mid", "oldest_low"]);

    test_db.teardown().await;
}

/// `delete_documents` soft-deletes the matching live rows in one statement,
/// returns `(id, level)` for the rows actually deleted, skips ids outside the
/// `(observer, observed)` scope, and skips ids already soft-deleted.
#[tokio::test]
async fn delete_documents_soft_deletes_scoped_rows_and_returns_levels() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd", "other"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    for (idx, observed) in [(50, "obsd"), (51, "other")] {
        sqlx::query(
            "INSERT INTO collections (id, workspace_name, observer, observed) \
             VALUES ($1, 'ws', 'obs', $2)",
        )
        .bind(doc_id(idx))
        .bind(observed)
        .execute(&test_db.pool)
        .await
        .expect("collection");
    }

    let insert = |id: String, level: &'static str, observed: &'static str| {
        let pool = test_db.pool.clone();
        async move {
            sqlx::query(
                "INSERT INTO documents \
                 (id, content, workspace_name, observer, observed, level, sync_state) \
                 VALUES ($1, 'c', 'ws', 'obs', $2, $3, 'synced')",
            )
            .bind(&id)
            .bind(observed)
            .bind(level)
            .execute(&pool)
            .await
            .expect("seed doc");
        }
    };

    insert(doc_id(1), "explicit", "obsd").await;
    insert(doc_id(2), "deductive", "obsd").await;
    // Out-of-scope: different observed peer — must NOT be deleted.
    insert(doc_id(3), "explicit", "other").await;

    // Pre-soft-delete doc 4 so it's skipped (already deleted).
    insert(doc_id(4), "inductive", "obsd").await;
    db::mark_document_deleted(&test_db.pool, "ws", &doc_id(4))
        .await
        .expect("pre-delete");

    let ids = vec![doc_id(1), doc_id(2), doc_id(3), doc_id(4), "missing".to_string()];
    let mut deleted = db::delete_documents(&test_db.pool, "ws", &ids, "obs", "obsd", None)
        .await
        .expect("delete_documents");
    deleted.sort();

    // Only docs 1 and 2 (in scope, live) are deleted, with their levels.
    assert_eq!(
        deleted,
        vec![
            (doc_id(1), Some("explicit".to_string())),
            (doc_id(2), Some("deductive".to_string())),
        ]
    );

    // Doc 3 (wrong observed) remains live.
    let live: i64 =
        sqlx::query_scalar("SELECT count(*) FROM documents WHERE id = $1 AND deleted_at IS NULL")
            .bind(doc_id(3))
            .fetch_one(&test_db.pool)
            .await
            .expect("count");
    assert_eq!(live, 1);

    // Empty id list is a no-op.
    let empty = db::delete_documents(&test_db.pool, "ws", &[], "obs", "obsd", None)
        .await
        .expect("empty");
    assert!(empty.is_empty());

    test_db.teardown().await;
}

/// `create_documents_returning_levels` inserts the accepted docs and reports each
/// accepted level in order; `query_documents_recent_full` returns them as
/// `representation::Document`s ordered newest-first.
#[tokio::test]
async fn create_documents_returning_levels_and_recent_full() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    db::get_or_create_collection(&test_db.pool, "ws", "obs", "obsd")
        .await
        .expect("collection");

    let mk = |content: &str, level: &str| db::DocumentToCreate {
        content: content.to_string(),
        session_name: None,
        level: level.to_string(),
        internal_metadata: json!({"message_ids": [], "message_created_at": "2026-06-22T00:00:00Z"}),
        embedding: query_vector(&[(0, 1.0)]),
        times_derived: 1,
        source_ids: None,
    };

    // deduplicate=false so both accepted; one-hot[0] twice would dedupe otherwise.
    let levels = db::create_documents_returning_levels(
        &test_db.pool,
        vec![mk("first deductive", "deductive"), mk("second inductive", "inductive")],
        "ws",
        "obs",
        "obsd",
        false,
    )
    .await
    .expect("create returning levels");
    assert_eq!(levels, vec!["deductive".to_string(), "inductive".to_string()]);

    let recent = db::query_documents_recent_full(&test_db.pool, "ws", "obs", "obsd", None, 10)
        .await
        .expect("recent full");
    assert_eq!(recent.len(), 2);
    // Both documents carry their level through the Document shape.
    let recent_levels: std::collections::HashSet<&str> =
        recent.iter().map(|d| d.level.as_str()).collect();
    assert!(recent_levels.contains("deductive"));
    assert!(recent_levels.contains("inductive"));

    test_db.teardown().await;
}

/// The perspective session-context path: `query_working_representation` blends
/// the `(observer, observed)` conclusions under the observation budget, and
/// `get_perspective_session_context` injects the representation markdown + peer
/// card alongside the token-budgeted session messages.
#[tokio::test]
async fn perspective_session_context_injects_representation_and_card() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    db::get_or_create_session(&test_db.pool, "ws", "sess", None, None)
        .await
        .expect("session");
    db::get_or_create_collection(&test_db.pool, "ws", "obs", "obsd")
        .await
        .expect("collection");

    db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("obs", "hello"), message("obsd", "hi there")],
        false,
        8192,
    )
    .await
    .expect("messages");

    let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap();
    seed_document_full(&test_db.pool, &doc_id(1), "the user likes coffee", 1, t1).await;
    seed_document_full(&test_db.pool, &doc_id(2), "the user is a runner", 2, t2).await;

    // No search query and no most-frequent → the whole budget is recent
    // explicit observations.
    let representation = db::query_working_representation(
        &test_db.pool,
        "ws",
        "obs",
        "obsd",
        None,
        None,
        false,
        None,
        None,
        false,
        db::WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    )
    .await
    .expect("working representation");
    let contents: std::collections::HashSet<&str> = representation
        .explicit
        .iter()
        .map(|obs| obs.content.as_str())
        .collect();
    assert!(contents.contains("the user likes coffee"));
    assert!(contents.contains("the user is a runner"));

    let markdown = representation.format_as_markdown(false);
    let card = Some(vec!["IDENTITY: marathon runner".to_string()]);
    let value = db::get_perspective_session_context(
        &test_db.pool,
        "ws",
        "sess",
        100_000,
        true,
        markdown.clone(),
        card.clone(),
    )
    .await
    .expect("perspective context");

    assert_eq!(value["id"], json!("sess"));
    assert_eq!(value["peer_representation"], json!(markdown));
    assert_eq!(value["peer_card"], json!(["IDENTITY: marathon runner"]));
    assert!(value["summary"].is_null());
    // No summary covers the session, so both messages fit the generous budget.
    assert_eq!(value["messages"].as_array().map(|items| items.len()), Some(2));
    assert!(markdown.contains("the user likes coffee"));

    test_db.teardown().await;
}

/// `merge_message_internal_metadata` shallow-merges a patch into a message's
/// `internal_metadata` (the file-upload post-create update), scoped by id, and
/// reports whether a row matched.
#[tokio::test]
async fn merge_message_internal_metadata_shallow_merges_by_id() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("alice", "chunk one")],
        false,
        8192,
    )
    .await
    .expect("create message");
    let message_id = created[0].id;

    // First merge sets file metadata.
    let updated = db::merge_message_internal_metadata(
        &test_db.pool,
        "ws",
        "sess",
        message_id,
        &json!({"file_id": "f1", "chunk_index": 0}),
    )
    .await
    .expect("merge 1");
    assert!(updated);

    // Second merge adds a key and overrides one, preserving the rest.
    db::merge_message_internal_metadata(
        &test_db.pool,
        "ws",
        "sess",
        message_id,
        &json!({"chunk_index": 1, "total_chunks": 3}),
    )
    .await
    .expect("merge 2");

    let stored: Value = sqlx::query_scalar("SELECT internal_metadata FROM messages WHERE id = $1")
        .bind(message_id)
        .fetch_one(&test_db.pool)
        .await
        .expect("read internal_metadata");
    assert_eq!(stored["file_id"], json!("f1"));
    assert_eq!(stored["chunk_index"], json!(1));
    assert_eq!(stored["total_chunks"], json!(3));

    // A non-matching id reports no update.
    let missing = db::merge_message_internal_metadata(
        &test_db.pool,
        "ws",
        "sess",
        999_999,
        &json!({"x": 1}),
    )
    .await
    .expect("merge missing");
    assert!(!missing);

    test_db.teardown().await;
}

/// `query_documents_full` returns the full document shape (level, source_ids,
/// internal_metadata) and `Representation::from_documents` reconstructs the
/// observations from it. Uses deterministic one-hot embeddings (no OpenAI), so
/// it runs whenever `TEST_DATABASE_URL` is set.
#[tokio::test]
async fn query_documents_full_feeds_from_documents() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    sqlx::query(
        "INSERT INTO collections (id, workspace_name, observer, observed) \
         VALUES ($1, 'ws', 'obs', 'obsd')",
    )
    .bind(doc_id(99))
    .execute(&test_db.pool)
    .await
    .expect("collection");

    // An explicit doc (one-hot at dim 5) with metadata-sourced created_at, and a
    // deductive doc (one-hot at dim 10) whose source_ids column overrides the
    // metadata premise_ids.
    sqlx::query(
        "INSERT INTO documents \
         (id, content, embedding, workspace_name, observer, observed, level, internal_metadata, sync_state) \
         VALUES ($1, $2, $3::vector, 'ws', 'obs', 'obsd', 'explicit', $4, 'synced')",
    )
    .bind(doc_id(1))
    .bind("the user has a dog")
    .bind(one_hot_literal(5))
    .bind(json!({"message_ids": [[2, 2], 1], "message_created_at": "2025-01-01T09:00:00Z"}))
    .execute(&test_db.pool)
    .await
    .expect("seed explicit");

    sqlx::query(
        "INSERT INTO documents \
         (id, content, embedding, workspace_name, observer, observed, level, source_ids, internal_metadata, sync_state) \
         VALUES ($1, $2, $3::vector, 'ws', 'obs', 'obsd', 'deductive', $4, $5, 'synced')",
    )
    .bind(doc_id(2))
    .bind("the dog is old")
    .bind(one_hot_literal(10))
    .bind(json!([doc_id(1)]))
    .bind(json!({"premises": ["the user has a dog"], "premise_ids": ["ignored"]}))
    .execute(&test_db.pool)
    .await
    .expect("seed deductive");

    let filter = build_filter_clause(FilterTarget::Conclusion, None).expect("empty filter");
    let documents = db::query_documents_full(
        &test_db.pool,
        "ws",
        "obs",
        "obsd",
        &query_vector(&[(5, 1.0)]),
        &filter,
        None,
        10,
    )
    .await
    .expect("query documents full");

    assert_eq!(documents.len(), 2);
    // Cosine-ordered: the explicit doc (one-hot at the query's dim) ranks first.
    assert_eq!(documents[0].level, "explicit");

    let rep = honcho_api_rs::representation::Representation::from_documents(&documents);
    assert_eq!(rep.explicit.len(), 1);
    assert_eq!(rep.deductive.len(), 1);

    // Explicit: message_ids flattened+sorted, created_at from metadata.
    let e = &rep.explicit[0];
    assert_eq!(e.message_ids, vec![1, 2]);
    assert_eq!(
        e.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
        "2025-01-01 09:00:00"
    );

    // Deductive: source_ids column wins over metadata premise_ids.
    let d = &rep.deductive[0];
    assert_eq!(d.source_ids, vec![doc_id(1)]);
    assert_eq!(d.premises, vec!["the user has a dog".to_string()]);

    test_db.teardown().await;
}

/// `query_documents_by_levels` restricts results to the requested reasoning
/// levels (the dialectic prefetch path). Deterministic one-hot embeddings.
/// A fixed-vector embedder for the dialectic agent test (no network).
struct StubEmbedder;
impl honcho_api_rs::dialectic::Embedder for StubEmbedder {
    async fn embed(&self, _query: &str) -> Result<Vec<f32>, String> {
        Ok(vec![0.0_f32; 1536])
    }
    async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts.iter().map(|_| vec![0.0_f32; 1536]).collect())
    }
}

/// An LlmHttp stub returning a fixed OpenAI completion (no tool calls), so the
/// dialectic tool loop returns on the first call.
struct StubLlmHttp;
impl honcho_api_rs::llm::http::LlmHttp for StubLlmHttp {
    async fn post_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &Value,
    ) -> Result<Value, honcho_api_rs::llm::http::LlmHttpError> {
        Ok(json!({
            "choices": [{"message": {"content": "the synthesized answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}
        }))
    }
}

/// End-to-end DialecticAgent::answer over a live DB: empty memory (prefetch
/// finds nothing), a seeded session for history, and a stubbed LLM that returns
/// a direct answer with no tool calls.
#[tokio::test]
async fn dialectic_agent_answer_end_to_end() {
    use honcho_api_rs::dialectic_config::{DialecticSettings, ReasoningLevel};
    use honcho_api_rs::llm::credentials::TransportApiKeys;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["alice", "bob"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    db::create_messages(
        &test_db.pool,
        "ws",
        "s1",
        &[message_at("alice", "hi bob", Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap())],
        false,
        8192,
    )
    .await
    .expect("messages");

    let settings = DialecticSettings::default();
    let keys = TransportApiKeys {
        anthropic: None,
        openai: Some("sk-test".to_string()),
        gemini: None,
    };

    let answer = honcho_api_rs::dialectic_agent::answer(
        &test_db.pool,
        &StubLlmHttp,
        keys,
        StubEmbedder,
        &settings,
        "ws",
        Some("s1"),
        "alice",
        "bob",
        None,
        None,
        "what does bob like?",
        ReasoningLevel::Low,
    )
    .await
    .expect("dialectic answer");

    assert_eq!(answer, "the synthesized answer");

    test_db.teardown().await;
}

/// `get_session_messages_within_token_limit` returns the most-recent messages
/// within the token budget (each seeded message has token_count=1) in
/// chronological order.
#[tokio::test]
async fn session_messages_respect_token_limit_and_order() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap();
    let t3 = Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap();
    db::create_messages(
        &test_db.pool,
        "ws",
        "s1",
        &[
            message_at("alice", "jan", t1),
            message_at("alice", "feb", t2),
            message_at("alice", "mar", t3),
        ],
        false,
        8192,
    )
    .await
    .expect("messages");

    // token_limit=2 -> the 2 most recent, chronological.
    let recent = db::get_session_messages_within_token_limit(&test_db.pool, "ws", "s1", 2)
        .await
        .expect("token limit 2");
    let contents: Vec<&str> = recent.iter().filter_map(|m| m["content"].as_str()).collect();
    assert_eq!(contents, vec!["feb", "mar"]);

    // High limit -> all, chronological.
    let all = db::get_session_messages_within_token_limit(&test_db.pool, "ws", "s1", 100)
        .await
        .expect("token limit 100");
    let all_contents: Vec<&str> = all.iter().filter_map(|m| m["content"].as_str()).collect();
    assert_eq!(all_contents, vec!["jan", "feb", "mar"]);

    test_db.teardown().await;
}

#[tokio::test]
async fn query_documents_by_levels_filters_by_level() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    sqlx::query(
        "INSERT INTO collections (id, workspace_name, observer, observed) \
         VALUES ($1, 'ws', 'obs', 'obsd')",
    )
    .bind(doc_id(99))
    .execute(&test_db.pool)
    .await
    .expect("collection");

    for (id, content, level) in [
        (doc_id(1), "explicit fact", "explicit"),
        (doc_id(2), "deductive conclusion", "deductive"),
        (doc_id(3), "inductive pattern", "inductive"),
    ] {
        sqlx::query(
            "INSERT INTO documents \
             (id, content, embedding, workspace_name, observer, observed, level, sync_state) \
             VALUES ($1, $2, $3::vector, 'ws', 'obs', 'obsd', $4, 'synced')",
        )
        .bind(&id)
        .bind(content)
        .bind(one_hot_literal(5))
        .bind(level)
        .execute(&test_db.pool)
        .await
        .expect("seed document");
    }

    // Only explicit.
    let explicit = db::query_documents_by_levels(
        &test_db.pool,
        "ws",
        "obs",
        "obsd",
        &query_vector(&[(5, 1.0)]),
        &["explicit".to_string()],
        10,
    )
    .await
    .expect("explicit query");
    assert_eq!(explicit.len(), 1);
    assert_eq!(explicit[0].level, "explicit");

    // Derived levels (deductive + inductive).
    let derived = db::query_documents_by_levels(
        &test_db.pool,
        "ws",
        "obs",
        "obsd",
        &query_vector(&[(5, 1.0)]),
        &["deductive".to_string(), "inductive".to_string()],
        10,
    )
    .await
    .expect("derived query");
    assert_eq!(derived.len(), 2);
    assert!(derived.iter().all(|d| d.level != "explicit"));

    test_db.teardown().await;
}

#[tokio::test]
async fn reasoning_chain_documents_and_children() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    sqlx::query(
        "INSERT INTO collections (id, workspace_name, observer, observed) \
         VALUES ($1, 'ws', 'obs', 'obsd')",
    )
    .bind(doc_id(99))
    .execute(&test_db.pool)
    .await
    .expect("collection");

    // Two explicit premises (no source_ids) and a deductive child deriving from
    // both. The child's source_ids JSONB array references the two premises.
    for (id, content, level, sources) in [
        (doc_id(1), "premise one", "explicit", None::<Value>),
        (doc_id(2), "premise two", "explicit", None),
        (
            doc_id(3),
            "derived conclusion",
            "deductive",
            Some(json!([doc_id(1), doc_id(2)])),
        ),
    ] {
        sqlx::query(
            "INSERT INTO documents \
             (id, content, workspace_name, observer, observed, level, source_ids, sync_state) \
             VALUES ($1, $2, 'ws', 'obs', 'obsd', $3, $4, 'synced')",
        )
        .bind(&id)
        .bind(content)
        .bind(level)
        .bind(sources)
        .execute(&test_db.pool)
        .await
        .expect("seed document");
    }

    // Empty ids -> empty, no query.
    let empty = db::get_documents_by_ids(&test_db.pool, "ws", &[])
        .await
        .expect("empty ids");
    assert!(empty.is_empty());

    // The deductive doc carries level + both source ids.
    let child = db::get_documents_by_ids(&test_db.pool, "ws", &[doc_id(3)])
        .await
        .expect("get child");
    assert_eq!(child.len(), 1);
    assert_eq!(child[0].content, "derived conclusion");
    assert_eq!(child[0].level, "deductive");
    let mut sources = child[0].source_ids.clone();
    sources.sort();
    assert_eq!(sources, vec![doc_id(1), doc_id(2)]);

    // Explicit premise carries an empty source_ids (NULL JSONB).
    let premise = db::get_documents_by_ids(&test_db.pool, "ws", &[doc_id(1)])
        .await
        .expect("get premise");
    assert_eq!(premise[0].level, "explicit");
    assert!(premise[0].source_ids.is_empty());

    // get_child_observations: premise one's only child is the deductive doc.
    let children =
        db::get_child_observations(&test_db.pool, "ws", &doc_id(1), Some("obs"), Some("obsd"))
            .await
            .expect("children");
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].id, doc_id(3));

    // A mismatched observer filters it out.
    let none = db::get_child_observations(&test_db.pool, "ws", &doc_id(1), Some("ghost"), None)
        .await
        .expect("children mismatched observer");
    assert!(none.is_empty());

    test_db.teardown().await;
}

#[tokio::test]
async fn format_reasoning_chain_renders_markdown() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    sqlx::query(
        "INSERT INTO collections (id, workspace_name, observer, observed) \
         VALUES ($1, 'ws', 'obs', 'obsd')",
    )
    .bind(doc_id(99))
    .execute(&test_db.pool)
    .await
    .expect("collection");

    for (id, content, level, sources) in [
        (doc_id(1), "premise one", "explicit", None::<Value>),
        (doc_id(2), "premise two", "explicit", None),
        (
            doc_id(3),
            "the conclusion",
            "deductive",
            Some(json!([doc_id(1), doc_id(2)])),
        ),
    ] {
        sqlx::query(
            "INSERT INTO documents \
             (id, content, workspace_name, observer, observed, level, source_ids, sync_state) \
             VALUES ($1, $2, 'ws', 'obs', 'obsd', $3, $4, 'synced')",
        )
        .bind(&id)
        .bind(content)
        .bind(level)
        .bind(sources)
        .execute(&test_db.pool)
        .await
        .expect("seed document");
    }

    // Deductive doc, direction "both": header + Premises (2) + no children.
    let chain = dialectic::format_reasoning_chain(
        &test_db.pool,
        "ws",
        &doc_id(3),
        "both",
        Some("obs"),
        Some("obsd"),
    )
    .await
    .expect("reasoning chain deductive");
    assert!(chain.starts_with(&format!(
        "**Observation [id:{}] (deductive):**\nthe conclusion",
        doc_id(3)
    )));
    assert!(chain.contains("\n**Premises (2):**\n"));
    assert!(chain.contains(&format!(" - [id:{}] (explicit): premise one", doc_id(1))));
    assert!(chain.contains(&format!(" - [id:{}] (explicit): premise two", doc_id(2))));
    assert!(chain.ends_with("\n**Derived Conclusions:** None found"));

    // Explicit premise: N/A premises, and one derived conclusion (the deductive).
    let premise_chain = dialectic::format_reasoning_chain(
        &test_db.pool,
        "ws",
        &doc_id(1),
        "both",
        Some("obs"),
        Some("obsd"),
    )
    .await
    .expect("reasoning chain explicit");
    assert!(
        premise_chain.contains("\n**Premises/Sources:** N/A (explicit observations have no premises)")
    );
    assert!(premise_chain.contains(&format!(
        "\n**Derived Conclusions (1):**\n - [id:{}] (deductive): the conclusion",
        doc_id(3)
    )));

    // Not found + invalid direction sentinels.
    let missing =
        dialectic::format_reasoning_chain(&test_db.pool, "ws", "nope", "both", None, None)
            .await
            .expect("missing");
    assert_eq!(missing, "ERROR: Observation 'nope' not found");

    let bad_dir =
        dialectic::format_reasoning_chain(&test_db.pool, "ws", &doc_id(1), "sideways", None, None)
            .await
            .expect("bad direction");
    assert_eq!(
        bad_dir,
        "ERROR: Invalid direction 'sideways'. Must be 'premises', 'conclusions', or 'both'"
    );

    test_db.teardown().await;
}

// --- deriver queue claim + lifecycle (QueueManager DB ops) ---

/// Insert a raw `queue` row, returning its bigint id.
async fn insert_queue_row(
    pool: &PgPool,
    work_unit_key: &str,
    task_type: &str,
    message_id: Option<i64>,
) -> i64 {
    sqlx::query_scalar(
        "INSERT INTO queue (work_unit_key, payload, task_type, workspace_name, message_id, processed) \
         VALUES ($1, '{}'::jsonb, $2, 'ws', $3, false) RETURNING id",
    )
    .bind(work_unit_key)
    .bind(task_type)
    .bind(message_id)
    .fetch_one(pool)
    .await
    .expect("insert queue row")
}

fn message_with_tokens(peer: &str, content: &str, token_count: i32) -> MessageInsert {
    MessageInsert {
        peer_name: peer.to_string(),
        content: content.to_string(),
        metadata: json!({}),
        created_at: None,
        token_count,
    }
}

#[tokio::test]
async fn claim_work_units_inserts_and_dedups() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let keys = vec![
        "representation:ws:sess:a".to_string(),
        "representation:ws:sess:b".to_string(),
    ];
    let claimed = db::claim_work_units(&test_db.pool, &keys)
        .await
        .expect("claim");
    assert_eq!(claimed.len(), 2);
    assert!(claimed.contains_key("representation:ws:sess:a"));
    assert!(claimed.contains_key("representation:ws:sess:b"));
    // Each aqs id is a 21-char nanoid.
    assert!(claimed.values().all(|id| id.len() == 21));

    // Re-claiming the same keys conflicts on the work_unit_key unique constraint
    // and returns nothing new.
    let again = db::claim_work_units(&test_db.pool, &keys)
        .await
        .expect("claim again");
    assert!(again.is_empty());

    let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM active_queue_sessions")
        .fetch_one(&test_db.pool)
        .await
        .expect("count aqs");
    assert_eq!(total, 2);

    test_db.teardown().await;
}

#[tokio::test]
async fn get_and_claim_respects_ownership_threshold_and_flush() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // Seed messages with known token counts (auto-creates ws/sess/peers).
    let created = db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[
            message_with_tokens("bob", "small", 5),
            message_with_tokens("bob", "big", 20),
        ],
        false,
        8192,
    )
    .await
    .expect("seed messages");
    let small_id = created[0].id;
    let big_id = created[1].id;

    // R_small: representation work unit under the 10-token cap (excluded).
    insert_queue_row(&test_db.pool, "representation:ws:sess:small", "representation", Some(small_id)).await;
    // R_big: representation work unit over the cap (eligible).
    insert_queue_row(&test_db.pool, "representation:ws:sess:big", "representation", Some(big_id)).await;
    // A summary work unit: non-representation, always eligible.
    insert_queue_row(&test_db.pool, "summary:ws:sess:a:bob", "summary", None).await;
    // An already-owned work unit: excluded by the NOT EXISTS guard.
    insert_queue_row(&test_db.pool, "representation:ws:sess:owned", "representation", Some(big_id)).await;
    db::claim_work_units(&test_db.pool, &["representation:ws:sess:owned".to_string()])
        .await
        .expect("pre-claim owned");

    // Threshold active (flush off, cap 10): big + summary eligible, small excluded.
    let claimed = db::get_and_claim_work_units(&test_db.pool, 10, 1, 10, false)
        .await
        .expect("claim with threshold");
    let mut keys: Vec<String> = claimed.keys().cloned().collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "representation:ws:sess:big".to_string(),
            "summary:ws:sess:a:bob".to_string(),
        ]
    );

    // Clean the freshly-claimed rows so the next call can re-evaluate them.
    for id in claimed.values() {
        sqlx::query("DELETE FROM active_queue_sessions WHERE id = $1")
            .bind(id)
            .execute(&test_db.pool)
            .await
            .expect("delete claimed");
    }

    // Flush enabled: the small representation unit is now eligible too.
    let flushed = db::get_and_claim_work_units(&test_db.pool, 10, 1, 10, true)
        .await
        .expect("claim with flush");
    assert!(flushed.contains_key("representation:ws:sess:small"));
    assert!(flushed.contains_key("representation:ws:sess:big"));

    test_db.teardown().await;
}

#[tokio::test]
async fn get_and_claim_limit_zero_returns_empty() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    insert_queue_row(&test_db.pool, "summary:ws:sess:a:bob", "summary", None).await;
    // owned_count == workers -> no capacity.
    let claimed = db::get_and_claim_work_units(&test_db.pool, 2, 2, 0, false)
        .await
        .expect("claim");
    assert!(claimed.is_empty());
    test_db.teardown().await;
}

#[tokio::test]
async fn mark_processed_errored_and_cleanup_work_unit() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");

    let key = "summary:ws:sess:a:bob";
    let id1 = insert_queue_row(&test_db.pool, key, "summary", None).await;
    let id2 = insert_queue_row(&test_db.pool, key, "summary", None).await;

    // Own the work unit with an intentionally old last_updated.
    let old = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    sqlx::query("INSERT INTO active_queue_sessions (id, work_unit_key, last_updated) VALUES ($1, $2, $3)")
        .bind("aqs-1")
        .bind(key)
        .bind(old)
        .execute(&test_db.pool)
        .await
        .expect("insert aqs");

    db::mark_queue_items_as_processed(&test_db.pool, &[id1, id2], key)
        .await
        .expect("mark processed");
    let processed: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM queue WHERE work_unit_key = $1 AND processed")
            .bind(key)
            .fetch_one(&test_db.pool)
            .await
            .expect("count processed");
    assert_eq!(processed, 2);
    let bumped: bool = sqlx::query_scalar(
        "SELECT last_updated > $2 FROM active_queue_sessions WHERE work_unit_key = $1",
    )
    .bind(key)
    .bind(old)
    .fetch_one(&test_db.pool)
    .await
    .expect("check last_updated");
    assert!(bumped, "last_updated should be bumped to now()");

    // Errored item: a third row gets marked processed with error text.
    let id3 = insert_queue_row(&test_db.pool, key, "summary", None).await;
    db::mark_queue_item_as_errored(&test_db.pool, id3, key, "boom")
        .await
        .expect("mark errored");
    let error: Option<String> = sqlx::query_scalar("SELECT error FROM queue WHERE id = $1")
        .bind(id3)
        .fetch_one(&test_db.pool)
        .await
        .expect("get error");
    assert_eq!(error.as_deref(), Some("boom"));

    // Cleanup removes the aqs row once, then reports nothing to remove.
    assert!(db::cleanup_work_unit(&test_db.pool, "aqs-1", key)
        .await
        .expect("cleanup"));
    assert!(!db::cleanup_work_unit(&test_db.pool, "aqs-1", key)
        .await
        .expect("cleanup again"));

    test_db.teardown().await;
}

#[tokio::test]
async fn cleanup_stale_work_units_removes_only_old_rows() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    let old = Utc::now() - chrono::Duration::minutes(30);
    let fresh = Utc::now();
    sqlx::query("INSERT INTO active_queue_sessions (id, work_unit_key, last_updated) VALUES ($1, $2, $3)")
        .bind("stale-1")
        .bind("summary:ws:sess:a:old")
        .bind(old)
        .execute(&test_db.pool)
        .await
        .expect("insert stale");
    sqlx::query("INSERT INTO active_queue_sessions (id, work_unit_key, last_updated) VALUES ($1, $2, $3)")
        .bind("fresh-1")
        .bind("summary:ws:sess:a:new")
        .bind(fresh)
        .execute(&test_db.pool)
        .await
        .expect("insert fresh");

    // 5-minute staleness window: the 30-minute-old row is removed, fresh kept.
    let deleted = db::cleanup_stale_work_units(&test_db.pool, 5)
        .await
        .expect("cleanup stale");
    assert_eq!(deleted, 1);
    let remaining: Vec<String> = sqlx::query_scalar("SELECT id FROM active_queue_sessions")
        .fetch_all(&test_db.pool)
        .await
        .expect("remaining");
    assert_eq!(remaining, vec!["fresh-1".to_string()]);

    test_db.teardown().await;
}

#[tokio::test]
async fn cleanup_queue_items_deletes_processed_respecting_error_retention() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");

    let now = Utc::now();
    let old = now - chrono::Duration::hours(2);
    // (key, processed, error, created_at)
    let rows: [(&str, bool, Option<&str>, chrono::DateTime<Utc>); 4] = [
        ("summary:ws:s:a:ok", true, None, now),         // processed, no error → delete
        ("summary:ws:s:a:err_fresh", true, Some("boom"), now), // errored, fresh → keep
        ("summary:ws:s:a:err_old", true, Some("boom"), old), // errored, old → delete
        ("summary:ws:s:a:pending", false, None, now),   // not processed → keep
    ];
    for (key, processed, error, created_at) in rows {
        sqlx::query(
            "INSERT INTO queue (work_unit_key, payload, task_type, workspace_name, processed, error, created_at) \
             VALUES ($1, '{}'::jsonb, 'summary', 'ws', $2, $3, $4)",
        )
        .bind(key)
        .bind(processed)
        .bind(error)
        .bind(created_at)
        .execute(&test_db.pool)
        .await
        .expect("insert queue row");
    }

    // Retention 1h: the fresh errored row is kept, the 2h-old errored row deleted.
    let deleted = db::cleanup_queue_items(&test_db.pool, 3600)
        .await
        .expect("cleanup queue items");
    assert_eq!(deleted, 2);

    let mut remaining: Vec<String> =
        sqlx::query_scalar("SELECT work_unit_key FROM queue ORDER BY work_unit_key")
            .fetch_all(&test_db.pool)
            .await
            .expect("remaining");
    remaining.sort();
    assert_eq!(
        remaining,
        vec![
            "summary:ws:s:a:err_fresh".to_string(),
            "summary:ws:s:a:pending".to_string(),
        ]
    );

    test_db.teardown().await;
}

#[tokio::test]
async fn cleanup_soft_deleted_documents_pgvector_removes_only_old_soft_deletes() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    db::get_or_create_peer(&test_db.pool, "ws", "alice", None, None)
        .await
        .expect("peer");
    sqlx::query(
        "INSERT INTO collections (id, workspace_name, observer, observed) \
         VALUES ($1, 'ws', 'alice', 'alice')",
    )
    .bind("c".repeat(21))
    .execute(&test_db.pool)
    .await
    .expect("seed collection");

    let now = Utc::now();
    let old = now - chrono::Duration::minutes(10);
    // (id, deleted_at)
    let docs: [(&str, Option<chrono::DateTime<Utc>>); 3] = [
        ("doc_old_softdel____01", Some(old)), // old soft-delete → cleaned
        ("doc_fresh_softdel__02", Some(now)), // fresh soft-delete → kept
        ("doc_live___________03", None),      // not soft-deleted → kept
    ];
    for (id, deleted_at) in docs {
        sqlx::query(
            "INSERT INTO documents (id, content, workspace_name, observer, observed, deleted_at) \
             VALUES ($1, 'a fact', 'ws', 'alice', 'alice', $2)",
        )
        .bind(id)
        .bind(deleted_at)
        .execute(&test_db.pool)
        .await
        .expect("seed document");
    }

    let deleted = db::cleanup_soft_deleted_documents_pgvector(&test_db.pool, 50, 5)
        .await
        .expect("cleanup docs");
    assert_eq!(deleted, 1);

    let mut remaining: Vec<String> =
        sqlx::query_scalar("SELECT id FROM documents ORDER BY id")
            .fetch_all(&test_db.pool)
            .await
            .expect("remaining");
    remaining.sort();
    assert_eq!(
        remaining,
        vec![
            "doc_fresh_softdel__02".to_string(),
            "doc_live___________03".to_string(),
        ]
    );

    test_db.teardown().await;
}

#[tokio::test]
async fn try_enqueue_reconciler_task_dedups_against_pending_and_in_progress() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };

    // First enqueue inserts a reconciler queue item.
    let enqueued = db::try_enqueue_reconciler_task(
        &test_db.pool,
        "reconciler:sync_vectors",
        "sync_vectors",
    )
    .await
    .expect("first enqueue");
    assert!(enqueued);

    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM queue WHERE work_unit_key = 'reconciler:sync_vectors'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("count");
    assert_eq!(count, 1);
    // Payload is exactly {"reconciler_type": ...} (no task_type), task_type column set.
    let payload: serde_json::Value = sqlx::query_scalar(
        "SELECT payload FROM queue WHERE work_unit_key = 'reconciler:sync_vectors'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("payload");
    assert_eq!(payload, json!({"reconciler_type": "sync_vectors"}));

    // Second enqueue is skipped: a pending item already exists.
    let again = db::try_enqueue_reconciler_task(
        &test_db.pool,
        "reconciler:sync_vectors",
        "sync_vectors",
    )
    .await
    .expect("second enqueue");
    assert!(!again);

    // Mark the pending item processed, then claim it (in-progress AQS row).
    sqlx::query("UPDATE queue SET processed = true WHERE work_unit_key = 'reconciler:sync_vectors'")
        .execute(&test_db.pool)
        .await
        .expect("mark processed");
    db::claim_work_units(&test_db.pool, &["reconciler:sync_vectors".to_string()])
        .await
        .expect("claim");

    // Still skipped while in-progress, even though no pending item remains.
    let blocked = db::try_enqueue_reconciler_task(
        &test_db.pool,
        "reconciler:sync_vectors",
        "sync_vectors",
    )
    .await
    .expect("blocked enqueue");
    assert!(!blocked);
    let total: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM queue WHERE work_unit_key = 'reconciler:sync_vectors'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("total");
    assert_eq!(total, 1);

    test_db.teardown().await;
}

#[tokio::test]
async fn get_next_queue_item_orders_and_checks_ownership() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");

    let key = "summary:ws:sess:a:bob";
    let first = insert_queue_row(&test_db.pool, key, "summary", None).await;
    let second = insert_queue_row(&test_db.pool, key, "summary", None).await;

    let claimed = db::claim_work_units(&test_db.pool, &[key.to_string()])
        .await
        .expect("claim");
    let aqs_id = claimed.get(key).expect("owned").clone();

    // Earliest unprocessed item first.
    let item = db::get_next_queue_item(&test_db.pool, key, &aqs_id)
        .await
        .expect("get next")
        .expect("an item");
    assert_eq!(item.id, first);
    assert_eq!(item.task_type, "summary");

    // After processing the first, the next item surfaces.
    db::mark_queue_items_as_processed(&test_db.pool, &[first], key)
        .await
        .expect("mark processed");
    let item = db::get_next_queue_item(&test_db.pool, key, &aqs_id)
        .await
        .expect("get next 2")
        .expect("second item");
    assert_eq!(item.id, second);

    // A wrong aqs_id (ownership lost) returns nothing.
    let none = db::get_next_queue_item(&test_db.pool, key, "not-the-owner")
        .await
        .expect("get next wrong owner");
    assert!(none.is_none());

    // Drained work unit returns None.
    db::mark_queue_items_as_processed(&test_db.pool, &[second], key)
        .await
        .expect("mark processed 2");
    let drained = db::get_next_queue_item(&test_db.pool, key, &aqs_id)
        .await
        .expect("get next drained");
    assert!(drained.is_none());

    test_db.teardown().await;
}

// --- get_queue_item_batch (representation context window) ---

async fn insert_queue_row_payload(
    pool: &PgPool,
    work_unit_key: &str,
    message_id: i64,
    payload: Value,
) -> i64 {
    sqlx::query_scalar(
        "INSERT INTO queue (work_unit_key, payload, task_type, workspace_name, message_id, processed) \
         VALUES ($1, $2, 'representation', 'ws', $3, false) RETURNING id",
    )
    .bind(work_unit_key)
    .bind(payload)
    .bind(message_id)
    .fetch_one(pool)
    .await
    .expect("insert queue row payload")
}

/// Create one message from `peer` with the given token count, returning its id.
async fn seed_one(pool: &PgPool, peer: &str, content: &str, tokens: i32) -> i64 {
    db::create_messages(
        pool,
        "ws",
        "sess",
        &[message_with_tokens(peer, content, tokens)],
        false,
        8192,
    )
    .await
    .expect("seed message")[0]
        .id
}

async fn claim_one(pool: &PgPool, key: &str) -> String {
    db::claim_work_units(pool, &[key.to_string()])
        .await
        .expect("claim")
        .get(key)
        .expect("owned")
        .clone()
}

#[tokio::test]
async fn batch_collects_messages_and_items_under_cap() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    let key = "representation:ws:sess:bob";

    let m1 = seed_one(&test_db.pool, "bob", "m1", 3).await;
    let m2 = seed_one(&test_db.pool, "bob", "m2", 4).await;
    insert_queue_row_payload(&test_db.pool, key, m1, json!({})).await;
    insert_queue_row_payload(&test_db.pool, key, m2, json!({})).await;
    let aqs = claim_one(&test_db.pool, key).await;

    let result = db::get_queue_item_batch(&test_db.pool, key, &aqs, 100, false)
        .await
        .expect("batch");
    let ctx_ids: Vec<i64> = result.messages_context.iter().map(|m| m.id).collect();
    assert_eq!(ctx_ids, vec![m1, m2]);
    assert_eq!(result.items_to_process.len(), 2);
    assert!(!result.hit_batch_token_cap);
    assert_eq!(result.batch_max_tokens, 100);
    assert!(result.configuration.is_none());

    test_db.teardown().await;
}

#[tokio::test]
async fn batch_clamps_at_token_cap_and_flags_it() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    let key = "representation:ws:sess:bob";

    let m1 = seed_one(&test_db.pool, "bob", "m1", 5).await;
    let m2 = seed_one(&test_db.pool, "bob", "m2", 10).await;
    let m3 = seed_one(&test_db.pool, "bob", "m3", 10).await;
    insert_queue_row_payload(&test_db.pool, key, m1, json!({})).await;
    insert_queue_row_payload(&test_db.pool, key, m2, json!({})).await;
    insert_queue_row_payload(&test_db.pool, key, m3, json!({})).await;
    let aqs = claim_one(&test_db.pool, key).await;

    // cap 12: m1 (cum 5) fits; m2 (cum 15) exceeds and is excluded; only m1 stays
    // (always-include-first keeps it). The CTE still flags the cap as exceeded.
    let result = db::get_queue_item_batch(&test_db.pool, key, &aqs, 12, false)
        .await
        .expect("batch");
    let ctx_ids: Vec<i64> = result.messages_context.iter().map(|m| m.id).collect();
    assert_eq!(ctx_ids, vec![m1]);
    assert_eq!(result.items_to_process.len(), 1);
    assert!(result.hit_batch_token_cap);

    test_db.teardown().await;
}

#[tokio::test]
async fn batch_prepends_preceding_message_from_other_peer() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    let key = "representation:ws:sess:bob";

    // alice's question precedes bob's reply; only bob's message is queued.
    let q1 = seed_one(&test_db.pool, "alice", "question", 2).await;
    let m1 = seed_one(&test_db.pool, "bob", "answer", 3).await;
    insert_queue_row_payload(&test_db.pool, key, m1, json!({})).await;
    let aqs = claim_one(&test_db.pool, key).await;

    let result = db::get_queue_item_batch(&test_db.pool, key, &aqs, 100, false)
        .await
        .expect("batch");
    let ctx_ids: Vec<i64> = result.messages_context.iter().map(|m| m.id).collect();
    // Preceding alice message is included as context; only bob's item is queued.
    assert_eq!(ctx_ids, vec![q1, m1]);
    assert_eq!(result.items_to_process.len(), 1);
    assert_eq!(result.items_to_process[0].message_id, Some(m1));

    test_db.teardown().await;
}

#[tokio::test]
async fn batch_excludes_preceding_message_from_same_peer() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    let key = "representation:ws:sess:bob";

    let _m0 = seed_one(&test_db.pool, "bob", "earlier", 2).await;
    let m1 = seed_one(&test_db.pool, "bob", "queued", 3).await;
    insert_queue_row_payload(&test_db.pool, key, m1, json!({})).await;
    let aqs = claim_one(&test_db.pool, key).await;

    let result = db::get_queue_item_batch(&test_db.pool, key, &aqs, 100, false)
        .await
        .expect("batch");
    let ctx_ids: Vec<i64> = result.messages_context.iter().map(|m| m.id).collect();
    // Same-peer preceding message is NOT prepended as context.
    assert_eq!(ctx_ids, vec![m1]);

    test_db.teardown().await;
}

#[tokio::test]
async fn batch_trims_to_homogeneous_configuration_prefix() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    let key = "representation:ws:sess:bob";

    // Two different configurations: the second breaks the homogeneous prefix.
    let config_a = json!({
        "reasoning": {"enabled": true},
        "peer_card": {"use": true, "create": true},
        "summary": {"enabled": true, "messages_per_short_summary": 20, "messages_per_long_summary": 60},
        "dream": {"enabled": true},
    });
    let config_b = json!({
        "reasoning": {"enabled": false},
        "peer_card": {"use": true, "create": true},
        "summary": {"enabled": true, "messages_per_short_summary": 20, "messages_per_long_summary": 60},
        "dream": {"enabled": true},
    });

    let m1 = seed_one(&test_db.pool, "bob", "m1", 1).await;
    let m2 = seed_one(&test_db.pool, "bob", "m2", 1).await;
    insert_queue_row_payload(&test_db.pool, key, m1, json!({"configuration": config_a})).await;
    insert_queue_row_payload(&test_db.pool, key, m2, json!({"configuration": config_b})).await;
    let aqs = claim_one(&test_db.pool, key).await;

    let result = db::get_queue_item_batch(&test_db.pool, key, &aqs, 100, false)
        .await
        .expect("batch");
    // Only the first item survives; context clipped to its message.
    assert_eq!(result.items_to_process.len(), 1);
    assert_eq!(result.items_to_process[0].message_id, Some(m1));
    let ctx_ids: Vec<i64> = result.messages_context.iter().map(|m| m.id).collect();
    assert_eq!(ctx_ids, vec![m1]);
    assert!(result.configuration.is_some());
    assert!(result.configuration.unwrap().reasoning_enabled);

    test_db.teardown().await;
}

#[tokio::test]
async fn batch_returns_empty_when_ownership_lost() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    let key = "representation:ws:sess:bob";
    let m1 = seed_one(&test_db.pool, "bob", "m1", 1).await;
    insert_queue_row_payload(&test_db.pool, key, m1, json!({})).await;

    let result = db::get_queue_item_batch(&test_db.pool, key, "not-owner", 100, false)
        .await
        .expect("batch");
    assert!(result.messages_context.is_empty());
    assert!(result.items_to_process.is_empty());
    assert_eq!(result.batch_max_tokens, 100);

    test_db.teardown().await;
}

#[tokio::test]
async fn process_representation_work_unit_once_drives_batch_and_marks_processed() {
    use honcho_api_rs::deriver::consumer::process_representation_work_unit_once;
    use honcho_api_rs::deriver::deriver::{DeriverBatchContext, DeriverModelSettings};
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::llm::{ModelConfig, Provider};
    use honcho_api_rs::producer::parse_work_unit_key;
    use honcho_api_rs::telemetry::NoopEmitter;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    db::get_or_create_peer(&test_db.pool, "ws", "alice", None, None)
        .await
        .expect("peer alice");

    let key = "representation:ws:sess:bob";
    let m1 = seed_one(&test_db.pool, "bob", "I like coffee", 4).await;
    let m2 = seed_one(&test_db.pool, "bob", "I live in Berlin", 4).await;
    // New-format payload carries the observer list.
    insert_queue_row_payload(&test_db.pool, key, m1, json!({"observers": ["alice"]})).await;
    insert_queue_row_payload(&test_db.pool, key, m2, json!({"observers": ["alice"]})).await;
    let aqs = claim_one(&test_db.pool, key).await;

    let http = CannedLlmHttp(json!({
        "content": [{"type": "text", "text":
            "{\"explicit\":[{\"content\":\"bob likes coffee\"},{\"content\":\"bob lives in Berlin\"}]}"}],
        "usage": {"input_tokens": 40, "output_tokens": 6},
        "stop_reason": "end_turn"
    }));
    let settings = DeriverModelSettings {
        model_config: ModelConfig::new("claude-x", Provider::Anthropic),
        ..DeriverModelSettings::default()
    };
    let emitter = NoopEmitter;
    let ctx = DeriverBatchContext {
        pool: &test_db.pool,
        http: &http,
        keys: TransportApiKeys {
            anthropic: Some("k".to_string()),
            openai: None,
            gemini: None,
        },
        embedder: &IndexedEmbedder,
        settings,
        emitter: &emitter,
        dream_schedule_settings:
            honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::default(),
    };

    let work_unit = parse_work_unit_key(key).expect("parse key");

    // First iteration: both queued items processed.
    let processed = process_representation_work_unit_once(&ctx, &work_unit, key, &aqs, 100, false)
        .await
        .expect("work unit ok");
    assert_eq!(processed, Some(2));

    // Documents were written to alice→bob.
    let doc_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM documents \
         WHERE observer = 'alice' AND observed = 'bob' AND deleted_at IS NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("doc count");
    assert_eq!(doc_count, 2);

    // Both queue items are marked processed.
    let unprocessed: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM queue WHERE work_unit_key = $1 AND processed = false")
            .bind(key)
            .fetch_one(&test_db.pool)
            .await
            .expect("unprocessed count");
    assert_eq!(unprocessed, 0);

    // Second iteration: nothing left → None (the loop would break).
    let again = process_representation_work_unit_once(&ctx, &work_unit, key, &aqs, 100, false)
        .await
        .expect("work unit ok");
    assert_eq!(again, None);

    test_db.teardown().await;
}

#[tokio::test]
async fn deriver_worker_poll_once_claims_processes_and_releases() {
    use honcho_api_rs::deriver::deriver::DeriverModelSettings;
    use honcho_api_rs::deriver::queue_manager::DeriverWorker;
    use honcho_api_rs::webhooks::ReqwestWebhookSender;
    use honcho_api_rs::deriver::settings::DeriverSettings;
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::llm::{ModelConfig, Provider};
    use honcho_api_rs::summarizer::SummaryGlobalSettings;
    use honcho_api_rs::telemetry::NoopEmitter;
    use std::sync::Arc;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    db::get_or_create_peer(&test_db.pool, "ws", "alice", None, None)
        .await
        .expect("peer alice");

    let key = "representation:ws:sess:bob";
    let m1 = seed_one(&test_db.pool, "bob", "I like coffee", 4).await;
    let m2 = seed_one(&test_db.pool, "bob", "I live in Berlin", 4).await;
    insert_queue_row_payload(&test_db.pool, key, m1, json!({"observers": ["alice"]})).await;
    insert_queue_row_payload(&test_db.pool, key, m2, json!({"observers": ["alice"]})).await;

    let http = Arc::new(CannedLlmHttp(json!({
        "content": [{"type": "text", "text":
            "{\"explicit\":[{\"content\":\"bob likes coffee\"},{\"content\":\"bob lives in Berlin\"}]}"}],
        "usage": {"input_tokens": 40, "output_tokens": 6},
        "stop_reason": "end_turn"
    })));
    let model_settings = DeriverModelSettings {
        model_config: ModelConfig::new("claude-x", Provider::Anthropic),
        ..DeriverModelSettings::default()
    };
    // flush_enabled bypasses the batch-token threshold so the small batch claims.
    let poll_settings = DeriverSettings {
        workers: 1,
        flush_enabled: true,
        ..DeriverSettings::default()
    };
    let worker = Arc::new(DeriverWorker::new(
        test_db.pool.clone(),
        http,
        Arc::new(IndexedEmbedder),
        Arc::new(NoopEmitter),
        TransportApiKeys {
            anthropic: Some("k".to_string()),
            openai: None,
            gemini: None,
        },
        model_settings,
        SummaryGlobalSettings::default(),
        honcho_api_rs::dreamer::orchestrator::DreamModelSettings::default(),
        honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::default(),
        poll_settings,
        Arc::new(ReqwestWebhookSender::new()),
        None,
    ));

    let processed = worker.poll_once().await.expect("poll once");
    assert_eq!(processed, 2);

    // Documents written, queue drained, and the active-queue-session claim released.
    let docs: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM documents WHERE observer = 'alice' AND observed = 'bob' AND deleted_at IS NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("doc count");
    assert_eq!(docs, 2);

    // The two representation items are processed. Draining them enqueues a
    // `queue.empty` webhook item (Python's process_work_unit finally), which is
    // the only thing left unprocessed.
    let unprocessed_repr: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM queue WHERE processed = false AND task_type = 'representation'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("unprocessed repr");
    assert_eq!(unprocessed_repr, 0);

    let queue_empty: (i64, serde_json::Value) = sqlx::query_as(
        "SELECT COUNT(*), COALESCE(MIN(payload::text)::jsonb, 'null'::jsonb) \
         FROM queue WHERE task_type = 'webhook' AND processed = false",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("queue.empty row");
    assert_eq!(queue_empty.0, 1);
    assert_eq!(queue_empty.1["event_type"], json!("queue.empty"));
    assert_eq!(queue_empty.1["data"]["queue_type"], json!("representation"));
    assert_eq!(queue_empty.1["data"]["observed"], json!("bob"));

    let claims: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM active_queue_sessions")
        .fetch_one(&test_db.pool)
        .await
        .expect("claims");
    assert_eq!(claims, 0); // released by cleanup_work_unit

    // A second poll drains the queue.empty webhook (no endpoints registered, so
    // delivery is a best-effort no-op) and finds nothing further afterward.
    let again = worker.poll_once().await.expect("poll again");
    assert_eq!(again, 1);
    let unprocessed: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM queue WHERE processed = false")
            .fetch_one(&test_db.pool)
            .await
            .expect("unprocessed");
    assert_eq!(unprocessed, 0);

    test_db.teardown().await;
}

// --- worker hard-delete crud ---

/// Seed a workspace with one session, two bob messages, observer alice, and two
/// alice→bob documents. Returns nothing — callers query counts directly.
async fn seed_deletion_fixture(pool: &PgPool) {
    use honcho_api_rs::representation::{ExplicitObservation, Representation};
    use honcho_api_rs::representation_manager::save_representation;

    db::create_messages(pool, "ws", "sess", &[message("bob", "hi"), message("bob", "yo")], false, 8192)
        .await
        .expect("seed messages");
    db::get_or_create_peer(pool, "ws", "alice", None, None)
        .await
        .expect("peer alice");

    let created = Utc.with_ymd_and_hms(2025, 3, 4, 12, 0, 0).unwrap();
    let rep = Representation {
        explicit: vec![
            ExplicitObservation {
                id: String::new(),
                created_at: created,
                message_ids: vec![1],
                session_name: Some("sess".to_string()),
                content: "bob likes coffee".to_string(),
            },
            ExplicitObservation {
                id: String::new(),
                created_at: created,
                message_ids: vec![1],
                session_name: Some("sess".to_string()),
                content: "bob lives in Berlin".to_string(),
            },
        ],
        ..Representation::default()
    };
    save_representation(
        pool,
        &IndexedEmbedder,
        "ws",
        "alice",
        "bob",
        &rep,
        &[1],
        "sess",
        created,
        true,
        None,
    )
    .await
    .expect("save docs");
}

#[tokio::test]
async fn hard_delete_session_removes_data_and_counts() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    seed_deletion_fixture(&test_db.pool).await;

    let counts = db::hard_delete_session(&test_db.pool, "ws", "sess")
        .await
        .expect("hard delete session");
    assert_eq!(counts.messages_deleted, 2);
    assert_eq!(counts.conclusions_deleted, 2);

    // Session, messages, and documents are gone; peers + workspace remain.
    let sessions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM sessions WHERE workspace_name = 'ws'")
        .fetch_one(&test_db.pool)
        .await
        .expect("sessions");
    assert_eq!(sessions, 0);
    let messages: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM messages WHERE workspace_name = 'ws'")
        .fetch_one(&test_db.pool)
        .await
        .expect("messages");
    assert_eq!(messages, 0);
    let peers: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM peers WHERE workspace_name = 'ws'")
        .fetch_one(&test_db.pool)
        .await
        .expect("peers");
    assert_eq!(peers, 2); // bob + alice survive a session delete

    // Missing session → NotFound.
    let missing = db::hard_delete_session(&test_db.pool, "ws", "nope").await;
    assert!(matches!(missing, Err(db::HardDeleteError::NotFound)));

    test_db.teardown().await;
}

#[tokio::test]
async fn hard_delete_workspace_cascades_everything() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    seed_deletion_fixture(&test_db.pool).await;

    let counts = db::hard_delete_workspace(&test_db.pool, "ws")
        .await
        .expect("hard delete workspace");
    assert_eq!(counts.peers_deleted, 2);
    assert_eq!(counts.sessions_deleted, 1);
    assert_eq!(counts.messages_deleted, 2);
    assert_eq!(counts.conclusions_deleted, 2);

    let workspaces: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM workspaces WHERE name = 'ws'")
        .fetch_one(&test_db.pool)
        .await
        .expect("workspaces");
    assert_eq!(workspaces, 0);
    let docs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM documents WHERE workspace_name = 'ws'")
        .fetch_one(&test_db.pool)
        .await
        .expect("docs");
    assert_eq!(docs, 0);

    // Missing workspace → NotFound.
    let missing = db::hard_delete_workspace(&test_db.pool, "nope").await;
    assert!(matches!(missing, Err(db::HardDeleteError::NotFound)));

    test_db.teardown().await;
}

#[tokio::test]
async fn mark_document_deleted_is_idempotent() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    seed_deletion_fixture(&test_db.pool).await;

    let doc_id: String = sqlx::query_scalar(
        "SELECT id FROM documents WHERE workspace_name = 'ws' AND content = 'bob likes coffee'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("doc id");

    // First mark soft-deletes it.
    let first = db::mark_document_deleted(&test_db.pool, "ws", &doc_id)
        .await
        .expect("mark");
    assert!(first);
    let deleted_at: Option<chrono::DateTime<Utc>> =
        sqlx::query_scalar("SELECT deleted_at FROM documents WHERE id = $1")
            .bind(&doc_id)
            .fetch_one(&test_db.pool)
            .await
            .expect("deleted_at");
    assert!(deleted_at.is_some());

    // Second mark is a no-op (already deleted) → false.
    let second = db::mark_document_deleted(&test_db.pool, "ws", &doc_id)
        .await
        .expect("mark again");
    assert!(!second);

    // Unknown id → false.
    let unknown = db::mark_document_deleted(&test_db.pool, "ws", "missing")
        .await
        .expect("mark unknown");
    assert!(!unknown);

    test_db.teardown().await;
}

#[tokio::test]
async fn save_and_get_summary_round_trips_and_merges() {
    use honcho_api_rs::summarizer::{Summary, SummaryType};

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    // create_messages auto-creates ws + session "sess".
    db::create_messages(&test_db.pool, "ws", "sess", &[message("bob", "hi")], false, 8192)
        .await
        .expect("seed session");

    // No summary yet.
    let none = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("get none");
    assert!(none.is_none());

    let short = Summary {
        content: "short summary".to_string(),
        message_id: 1,
        summary_type: SummaryType::Short.as_str().to_string(),
        created_at: "2025-03-04T12:00:00+00:00".to_string(),
        token_count: 2,
        message_public_id: "msg_pub_1".to_string(),
    };
    let updated = db::save_summary(
        &test_db.pool,
        "ws",
        "sess",
        SummaryType::Short.as_str(),
        &serde_json::to_value(&short).unwrap(),
    )
    .await
    .expect("save short");
    assert!(updated);

    // Round-trips back to the same struct.
    let fetched = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("get short")
        .expect("present");
    let parsed: Summary = serde_json::from_value(fetched).unwrap();
    assert_eq!(parsed, short);

    // Saving a long summary preserves the short one (shallow merge).
    let long = Summary {
        content: "long summary".to_string(),
        message_id: 1,
        summary_type: SummaryType::Long.as_str().to_string(),
        created_at: "2025-03-04T12:05:00+00:00".to_string(),
        token_count: 2,
        message_public_id: "msg_pub_1".to_string(),
    };
    db::save_summary(
        &test_db.pool,
        "ws",
        "sess",
        SummaryType::Long.as_str(),
        &serde_json::to_value(&long).unwrap(),
    )
    .await
    .expect("save long");

    let short_still = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("get short again")
        .expect("present");
    assert_eq!(short_still["content"], "short summary");
    let long_now = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Long.as_str())
        .await
        .expect("get long")
        .expect("present");
    assert_eq!(long_now["content"], "long summary");

    // Overwriting the short summary replaces just that key.
    let short2 = serde_json::json!({
        "content": "newer short", "message_id": 5, "summary_type": SummaryType::Short.as_str(),
        "created_at": "2025-03-04T12:10:00+00:00", "token_count": 2, "message_public_id": "msg_pub_5"
    });
    db::save_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str(), &short2)
        .await
        .expect("overwrite short");
    let short_new = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("get short3")
        .expect("present");
    assert_eq!(short_new["content"], "newer short");
    assert_eq!(short_new["message_id"], 5);

    // Missing session → None (no row updated).
    let missing = db::save_summary(&test_db.pool, "ws", "nope", SummaryType::Short.as_str(), &short2)
        .await
        .expect("save missing");
    assert!(!missing);

    test_db.teardown().await;
}

#[tokio::test]
async fn create_and_save_summary_persists_and_skips_when_covered() {
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::llm::{ModelConfig, Provider};
    use honcho_api_rs::summarizer::{
        SummaryModelSettings, SummaryType, create_and_save_summary,
    };

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("bob", "hi"), message("bob", "more"), message("bob", "even more")],
        false,
        8192,
    )
    .await
    .expect("seed messages"); // ids/seqs 1,2,3

    let http = CannedLlmHttp(json!({
        "content": [{"type": "text", "text": "a concise summary of the chat"}],
        "usage": {"input_tokens": 30, "output_tokens": 8},
        "stop_reason": "end_turn"
    }));
    let settings = SummaryModelSettings {
        model_config: ModelConfig::new("claude-x", Provider::Anthropic),
        max_tokens_short: 1000,
        max_tokens_long: 4000,
        messages_per_short_summary: 3,
        messages_per_long_summary: 6,
    };
    let keys = TransportApiKeys {
        anthropic: Some("k".to_string()),
        openai: None,
        gemini: None,
    };

    create_and_save_summary(
        &test_db.pool,
        &http,
        &keys,
        &settings,
        "ws",
        "sess",
        3, // message_id
        3, // message_seq_in_session
        "pub_3",
        SummaryType::Short,
        "2025-03-04T12:00:00+00:00",
    )
    .await
    .expect("create summary");

    let saved = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("get summary")
        .expect("present");
    assert_eq!(saved["content"], "a concise summary of the chat");
    assert_eq!(saved["message_id"], 3);
    assert_eq!(saved["token_count"], 8); // response.output_tokens
    assert_eq!(saved["created_at"], "2025-03-04T12:00:00+00:00");

    // Calling again for the same (already-covered) message is a no-op: a fresh
    // canned response with different text must NOT overwrite.
    let http2 = CannedLlmHttp(json!({
        "content": [{"type": "text", "text": "DIFFERENT TEXT"}],
        "usage": {"input_tokens": 30, "output_tokens": 8},
        "stop_reason": "end_turn"
    }));
    create_and_save_summary(
        &test_db.pool, &http2, &keys, &settings, "ws", "sess", 3, 3, "pub_3",
        SummaryType::Short, "2025-03-04T13:00:00+00:00",
    )
    .await
    .expect("skip covered");
    let unchanged = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("get summary")
        .expect("present");
    assert_eq!(unchanged["content"], "a concise summary of the chat");

    test_db.teardown().await;
}

#[tokio::test]
async fn create_and_save_summary_does_not_save_empty_fallback() {
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::llm::{ModelConfig, Provider};
    use honcho_api_rs::summarizer::{
        SummaryModelSettings, SummaryType, create_and_save_summary,
    };

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::create_messages(&test_db.pool, "ws", "sess", &[message("bob", "hi")], false, 8192)
        .await
        .expect("seed");

    // Empty LLM output → fallback → not saved.
    let http = CannedLlmHttp(json!({
        "content": [{"type": "text", "text": "   "}],
        "usage": {"input_tokens": 5, "output_tokens": 0},
        "stop_reason": "end_turn"
    }));
    let settings = SummaryModelSettings {
        model_config: ModelConfig::new("claude-x", Provider::Anthropic),
        max_tokens_short: 1000,
        max_tokens_long: 4000,
        messages_per_short_summary: 1,
        messages_per_long_summary: 6,
    };
    create_and_save_summary(
        &test_db.pool,
        &http,
        &TransportApiKeys { anthropic: Some("k".to_string()), openai: None, gemini: None },
        &settings,
        "ws",
        "sess",
        1,
        1,
        "pub_1",
        SummaryType::Short,
        "2025-03-04T12:00:00+00:00",
    )
    .await
    .expect("fallback ok");

    let none = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("get");
    assert!(none.is_none()); // fallback summaries are not persisted

    test_db.teardown().await;
}

#[tokio::test]
async fn summarize_if_needed_creates_both_tiers_at_long_boundary() {
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::llm::{ModelConfig, Provider};
    use honcho_api_rs::summarizer::{SummaryModelSettings, SummaryType, summarize_if_needed};

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    // Seed 6 messages (seqs 1..6).
    let msgs: Vec<_> = (0..6).map(|i| message("bob", &format!("m{i}"))).collect();
    db::create_messages(&test_db.pool, "ws", "sess", &msgs, false, 8192)
        .await
        .expect("seed");

    let http = CannedLlmHttp(json!({
        "content": [{"type": "text", "text": "summary text"}],
        "usage": {"input_tokens": 20, "output_tokens": 5},
        "stop_reason": "end_turn"
    }));
    let settings = SummaryModelSettings {
        model_config: ModelConfig::new("claude-x", Provider::Anthropic),
        max_tokens_short: 1000,
        max_tokens_long: 4000,
        messages_per_short_summary: 3,
        messages_per_long_summary: 6,
    };
    let keys = TransportApiKeys { anthropic: Some("k".to_string()), openai: None, gemini: None };

    // seq 6 is a multiple of both 3 and 6 → both tiers created.
    summarize_if_needed(
        &test_db.pool, &http, &keys, &settings, true, "ws", "sess", 6, 6, "pub_6",
        "2025-03-04T12:00:00+00:00",
    )
    .await
    .expect("summarize");

    let short = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("short")
        .expect("present");
    let long = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Long.as_str())
        .await
        .expect("long")
        .expect("present");
    assert_eq!(short["content"], "summary text");
    assert_eq!(long["content"], "summary text");

    // Disabled → no-op even at a boundary.
    let none_db = TestDb::setup().await.unwrap();
    db::create_messages(&none_db.pool, "ws", "sess", &msgs, false, 8192)
        .await
        .expect("seed2");
    summarize_if_needed(
        &none_db.pool, &http, &keys, &settings, false, "ws", "sess", 6, 6, "pub_6",
        "2025-03-04T12:00:00+00:00",
    )
    .await
    .expect("disabled");
    let nothing = db::get_summary(&none_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("short2");
    assert!(nothing.is_none());
    none_db.teardown().await;

    test_db.teardown().await;
}

/// Captures emitted telemetry event bodies for assertions.
struct CapturingEmitter {
    events: std::sync::Mutex<Vec<(String, serde_json::Value)>>,
}
impl honcho_api_rs::telemetry::Emitter for CapturingEmitter {
    fn emit(&self, event: &dyn honcho_api_rs::telemetry::TelemetryEvent) {
        self.events
            .lock()
            .unwrap()
            .push((event.event_type().to_string(), event.to_body()));
    }
}

#[tokio::test]
async fn process_deletion_workspace_emits_event_and_cascades() {
    use honcho_api_rs::deriver::consumer::process_deletion;
    use honcho_api_rs::deriver::payload::{DeletionPayload, DeletionType};

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    seed_deletion_fixture(&test_db.pool).await;

    let emitter = CapturingEmitter {
        events: std::sync::Mutex::new(Vec::new()),
    };
    let payload = DeletionPayload {
        deletion_type: DeletionType::Workspace,
        resource_id: "ws".to_string(),
    };
    process_deletion(&test_db.pool, &emitter, &payload, "ws")
        .await
        .expect("process deletion");

    // Workspace cascade-deleted.
    let workspaces: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM workspaces WHERE name = 'ws'")
        .fetch_one(&test_db.pool)
        .await
        .expect("ws count");
    assert_eq!(workspaces, 0);

    // Exactly one deletion.completed event with the cascade counts.
    let events = emitter.events.lock().unwrap().clone();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "deletion.completed");
    let body = &events[0].1;
    assert_eq!(body["success"], true);
    assert_eq!(body["deletion_type"], "workspace");
    assert_eq!(body["peers_deleted"], 2);
    assert_eq!(body["sessions_deleted"], 1);
    assert_eq!(body["messages_deleted"], 2);
    assert_eq!(body["conclusions_deleted"], 2);

    test_db.teardown().await;
}

#[tokio::test]
async fn process_deletion_observation_soft_deletes_and_emits() {
    use honcho_api_rs::deriver::consumer::process_deletion;
    use honcho_api_rs::deriver::payload::{DeletionPayload, DeletionType};

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    seed_deletion_fixture(&test_db.pool).await;
    let doc_id: String = sqlx::query_scalar(
        "SELECT id FROM documents WHERE workspace_name = 'ws' AND content = 'bob likes coffee'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("doc id");

    let emitter = CapturingEmitter {
        events: std::sync::Mutex::new(Vec::new()),
    };
    let payload = DeletionPayload {
        deletion_type: DeletionType::Observation,
        resource_id: doc_id.clone(),
    };
    process_deletion(&test_db.pool, &emitter, &payload, "ws")
        .await
        .expect("process deletion");

    // The document is soft-deleted; the event reports one conclusion deleted.
    let deleted_at: Option<chrono::DateTime<Utc>> =
        sqlx::query_scalar("SELECT deleted_at FROM documents WHERE id = $1")
            .bind(&doc_id)
            .fetch_one(&test_db.pool)
            .await
            .expect("deleted_at");
    assert!(deleted_at.is_some());

    let events = emitter.events.lock().unwrap().clone();
    assert_eq!(events.len(), 1);
    let body = &events[0].1;
    assert_eq!(body["success"], true);
    assert_eq!(body["deletion_type"], "observation");
    assert_eq!(body["conclusions_deleted"], 1);

    test_db.teardown().await;
}

#[tokio::test]
async fn deriver_worker_processes_a_deletion_work_unit() {
    use honcho_api_rs::deriver::deriver::DeriverModelSettings;
    use honcho_api_rs::deriver::queue_manager::DeriverWorker;
    use honcho_api_rs::webhooks::ReqwestWebhookSender;
    use honcho_api_rs::deriver::settings::DeriverSettings;
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::summarizer::SummaryGlobalSettings;
    use honcho_api_rs::telemetry::Emitter;
    use std::sync::Arc;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    seed_deletion_fixture(&test_db.pool).await;

    // Enqueue a workspace-deletion work unit (as the API accept path would).
    sqlx::query(
        "INSERT INTO queue (work_unit_key, payload, task_type, workspace_name, message_id, processed) \
         VALUES ($1, $2, 'deletion', 'ws', NULL, false)",
    )
    .bind("deletion:ws:workspace:ws")
    .bind(json!({"task_type": "deletion", "deletion_type": "workspace", "resource_id": "ws"}))
    .execute(&test_db.pool)
    .await
    .expect("enqueue deletion");

    let emitter = Arc::new(CapturingEmitter {
        events: std::sync::Mutex::new(Vec::new()),
    });
    let worker = Arc::new(DeriverWorker::new(
        test_db.pool.clone(),
        Arc::new(CannedLlmHttp(json!(null))), // not used by the deletion path
        Arc::new(IndexedEmbedder),
        Arc::clone(&emitter) as Arc<dyn Emitter>,
        TransportApiKeys::default(),
        DeriverModelSettings::default(),
        SummaryGlobalSettings::default(),
        honcho_api_rs::dreamer::orchestrator::DreamModelSettings::default(),
        honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::default(),
        DeriverSettings {
            workers: 1,
            ..DeriverSettings::default()
        },
        Arc::new(ReqwestWebhookSender::new()),
        None,
    ));

    let processed = worker.poll_once().await.expect("poll once");
    assert_eq!(processed, 1);

    // The workspace was cascade-deleted and a deletion.completed event emitted.
    let workspaces: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM workspaces WHERE name = 'ws'")
        .fetch_one(&test_db.pool)
        .await
        .expect("ws count");
    assert_eq!(workspaces, 0);

    let events = emitter.events.lock().unwrap().clone();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "deletion.completed");
    assert_eq!(events[0].1["success"], true);
    assert_eq!(events[0].1["peers_deleted"], 2);

    test_db.teardown().await;
}

#[tokio::test]
async fn deriver_worker_processes_a_summary_work_unit_with_public_id_fallback() {
    use honcho_api_rs::deriver::deriver::DeriverModelSettings;
    use honcho_api_rs::deriver::queue_manager::DeriverWorker;
    use honcho_api_rs::webhooks::ReqwestWebhookSender;
    use honcho_api_rs::deriver::settings::DeriverSettings;
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::summarizer::{SummaryGlobalSettings, SummaryType};
    use honcho_api_rs::telemetry::{Emitter, NoopEmitter};
    use std::sync::Arc;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    // Seed 3 messages (seqs 1..3); a short summary is due at seq 3.
    let msgs: Vec<_> = (0..3).map(|i| message("bob", &format!("m{i}"))).collect();
    let created = db::create_messages(&test_db.pool, "ws", "sess", &msgs, false, 8192)
        .await
        .expect("seed");
    let boundary = &created[2];
    assert_eq!(boundary.seq_in_session, 3);

    // Enqueue a summary work unit. message_public_id is intentionally OMITTED
    // from the payload to exercise the worker's DB fallback lookup.
    sqlx::query(
        "INSERT INTO queue (work_unit_key, payload, task_type, workspace_name, message_id, processed) \
         VALUES ($1, $2, 'summary', 'ws', $3, false)",
    )
    .bind("summary:ws:sess:bob:bob")
    .bind(json!({
        "task_type": "summary",
        "session_name": "sess",
        "message_seq_in_session": 3,
        "configuration": {
            "reasoning": {"enabled": true},
            "peer_card": {"use": false, "create": false},
            "summary": {
                "enabled": true,
                "messages_per_short_summary": 3,
                "messages_per_long_summary": 6
            },
            "dream": {"enabled": false}
        }
    }))
    .bind(boundary.id)
    .execute(&test_db.pool)
    .await
    .expect("enqueue summary");

    // OpenAI-shaped response: SummaryGlobalSettings::default() targets an
    // OpenAI model, so complete_single dispatches to the OpenAI backend.
    let http = CannedLlmHttp(json!({
        "choices": [{"message": {"content": "summary text"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5}
    }));
    let worker = Arc::new(DeriverWorker::new(
        test_db.pool.clone(),
        Arc::new(http),
        Arc::new(IndexedEmbedder),
        Arc::new(NoopEmitter) as Arc<dyn Emitter>,
        TransportApiKeys {
            anthropic: None,
            openai: Some("k".to_string()),
            gemini: None,
        },
        DeriverModelSettings::default(),
        SummaryGlobalSettings::default(),
        honcho_api_rs::dreamer::orchestrator::DreamModelSettings::default(),
        honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::default(),
        DeriverSettings {
            workers: 1,
            ..DeriverSettings::default()
        },
        Arc::new(ReqwestWebhookSender::new()),
        None,
    ));

    let processed = worker.poll_once().await.expect("poll once");
    assert_eq!(processed, 1);

    // The short summary was saved, carrying the public id resolved via fallback.
    let short = db::get_summary(&test_db.pool, "ws", "sess", SummaryType::Short.as_str())
        .await
        .expect("short")
        .expect("present");
    assert_eq!(short["content"], "summary text");
    assert_eq!(short["message_public_id"], boundary.public_id);

    test_db.teardown().await;
}

#[tokio::test]
async fn deriver_worker_processes_a_reconciler_sync_vectors_work_unit() {
    use honcho_api_rs::deriver::deriver::DeriverModelSettings;
    use honcho_api_rs::deriver::queue_manager::DeriverWorker;
    use honcho_api_rs::webhooks::ReqwestWebhookSender;
    use honcho_api_rs::deriver::settings::DeriverSettings;
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::summarizer::SummaryGlobalSettings;
    use honcho_api_rs::telemetry::Emitter;
    use std::sync::Arc;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    // embed_messages=true → create_messages inserts pending message_embeddings
    // rows with a NULL vector (embedding deferred to the reconciler).
    db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("bob", "hello world"), message("bob", "another chunk")],
        true,
        8192,
    )
    .await
    .expect("seed");

    let pending_before: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM message_embeddings WHERE sync_state = 'pending' AND embedding IS NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("pending count");
    assert!(pending_before >= 1, "expected pending embeddings to reconcile");

    // Enqueue a reconciler sync_vectors work unit (no workspace_name/message_id).
    sqlx::query(
        "INSERT INTO queue (work_unit_key, payload, task_type, workspace_name, message_id, processed) \
         VALUES ($1, $2, 'reconciler', NULL, NULL, false)",
    )
    .bind("reconciler:sync_vectors")
    .bind(json!({"task_type": "reconciler", "reconciler_type": "sync_vectors"}))
    .execute(&test_db.pool)
    .await
    .expect("enqueue reconciler");

    let emitter = Arc::new(CapturingEmitter {
        events: std::sync::Mutex::new(Vec::new()),
    });
    let worker = Arc::new(DeriverWorker::new(
        test_db.pool.clone(),
        Arc::new(CannedLlmHttp(json!(null))), // unused by the reconciler path
        Arc::new(IndexedEmbedder),
        Arc::clone(&emitter) as Arc<dyn Emitter>,
        TransportApiKeys::default(),
        DeriverModelSettings::default(),
        SummaryGlobalSettings::default(),
        honcho_api_rs::dreamer::orchestrator::DreamModelSettings::default(),
        honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::default(),
        DeriverSettings {
            workers: 1,
            ..DeriverSettings::default()
        },
        Arc::new(ReqwestWebhookSender::new()),
        None,
    ));

    let processed = worker.poll_once().await.expect("poll once");
    assert_eq!(processed, 1);

    // Every pending row is now synced with a persisted vector.
    let still_pending: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM message_embeddings WHERE sync_state = 'pending'")
            .fetch_one(&test_db.pool)
            .await
            .expect("still pending");
    assert_eq!(still_pending, 0);
    let synced_with_vector: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM message_embeddings WHERE sync_state = 'synced' AND embedding IS NOT NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("synced count");
    assert_eq!(synced_with_vector, pending_before);

    // A single sync_vectors.completed event reports the synced count.
    let events = emitter.events.lock().unwrap().clone();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "reconciliation.sync_vectors.completed");
    assert_eq!(events[0].1["message_embeddings_synced"], pending_before);
    assert_eq!(events[0].1["message_embeddings_failed"], 0);

    test_db.teardown().await;
}

/// A [`WebhookSender`] that records every delivery and reports HTTP 200, so the
/// worker test can assert what was POSTed without a live HTTP endpoint.
struct CapturingWebhookSender {
    #[allow(clippy::type_complexity)]
    calls: std::sync::Mutex<Vec<(String, String, Vec<(String, String)>)>>,
}

impl honcho_api_rs::webhooks::WebhookSender for CapturingWebhookSender {
    fn post<'a>(
        &'a self,
        url: &'a str,
        body: &'a str,
        headers: &'a [(String, String)],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u16, String>> + Send + 'a>>
    {
        let url = url.to_string();
        let body = body.to_string();
        let headers = headers.to_vec();
        Box::pin(async move {
            self.calls.lock().unwrap().push((url, body, headers));
            Ok(200u16)
        })
    }
}

#[tokio::test]
async fn deriver_worker_processes_a_webhook_work_unit() {
    use honcho_api_rs::deriver::deriver::DeriverModelSettings;
    use honcho_api_rs::deriver::queue_manager::DeriverWorker;
    use honcho_api_rs::deriver::settings::DeriverSettings;
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::summarizer::SummaryGlobalSettings;
    use honcho_api_rs::telemetry::{Emitter, NoopEmitter};
    use honcho_api_rs::webhooks::generate_webhook_signature;
    use std::sync::Arc;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");

    // Two registered endpoints — both must receive the same signed body.
    for url in ["https://example.com/a", "https://example.com/b"] {
        db::get_or_create_webhook_endpoint(&test_db.pool, "ws", url)
            .await
            .expect("register endpoint");
    }

    // Enqueue a webhook work unit (key `webhook:{workspace_name}`).
    sqlx::query(
        "INSERT INTO queue (work_unit_key, payload, task_type, workspace_name, message_id, processed) \
         VALUES ($1, $2, 'webhook', $3, NULL, false)",
    )
    .bind("webhook:ws")
    .bind(json!({
        "task_type": "webhook",
        "event_type": "test.event",
        "data": {"session_id": "s1", "count": 2},
    }))
    .bind("ws")
    .execute(&test_db.pool)
    .await
    .expect("enqueue webhook");

    let sender = Arc::new(CapturingWebhookSender {
        calls: std::sync::Mutex::new(Vec::new()),
    });
    let worker = Arc::new(DeriverWorker::new(
        test_db.pool.clone(),
        Arc::new(CannedLlmHttp(json!(null))), // unused by the webhook path
        Arc::new(IndexedEmbedder),
        Arc::new(NoopEmitter) as Arc<dyn Emitter>,
        TransportApiKeys::default(),
        DeriverModelSettings::default(),
        SummaryGlobalSettings::default(),
        honcho_api_rs::dreamer::orchestrator::DreamModelSettings::default(),
        honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::default(),
        DeriverSettings {
            workers: 1,
            ..DeriverSettings::default()
        },
        Arc::clone(&sender) as Arc<dyn honcho_api_rs::webhooks::WebhookSender>,
        Some("topsecret".to_string()),
    ));

    let processed = worker.poll_once().await.expect("poll once");
    assert_eq!(processed, 1);

    // Both endpoints received the delivery.
    let calls = sender.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 2);
    let mut urls: Vec<&str> = calls.iter().map(|(url, _, _)| url.as_str()).collect();
    urls.sort();
    assert_eq!(urls, vec!["https://example.com/a", "https://example.com/b"]);

    // The signed body is identical across endpoints and signs to the sent secret.
    let (_, body, headers) = &calls[0];
    assert_eq!(&calls[1].1, body, "all endpoints get the same body");
    let signature = headers
        .iter()
        .find(|(name, _)| name == "X-Honcho-Signature")
        .map(|(_, value)| value.as_str())
        .expect("signature header present");
    assert_eq!(signature, generate_webhook_signature("topsecret", body));
    assert!(
        headers
            .iter()
            .any(|(name, value)| name == "Content-Type" && value == "application/json")
    );

    // Body is the sorted-key envelope carrying the event type + data.
    let envelope: serde_json::Value = serde_json::from_str(body).expect("valid json body");
    assert_eq!(envelope["type"], "test.event");
    assert_eq!(envelope["data"]["count"], 2);

    // The queue item is marked processed.
    let unprocessed: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM queue WHERE task_type = 'webhook' AND processed = false")
            .fetch_one(&test_db.pool)
            .await
            .expect("unprocessed count");
    assert_eq!(unprocessed, 0);

    test_db.teardown().await;
}

#[tokio::test]
async fn deriver_worker_processes_a_dream_work_unit() {
    use honcho_api_rs::deriver::deriver::DeriverModelSettings;
    use honcho_api_rs::deriver::queue_manager::DeriverWorker;
    use honcho_api_rs::deriver::settings::DeriverSettings;
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::summarizer::SummaryGlobalSettings;
    use honcho_api_rs::telemetry::Emitter;
    use honcho_api_rs::webhooks::ReqwestWebhookSender;
    use std::sync::Arc;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    db::get_or_create_collection(&test_db.pool, "ws", "obs", "obsd")
        .await
        .expect("collection");

    // Enqueue a dream work unit (key `dream:{dream_type}:{ws}:{observer}:{observed}`).
    sqlx::query(
        "INSERT INTO queue (work_unit_key, payload, task_type, workspace_name, message_id, processed) \
         VALUES ($1, $2, 'dream', $3, NULL, false)",
    )
    .bind("dream:omni:ws:obs:obsd")
    .bind(json!({
        "task_type": "dream",
        "dream_type": "omni",
        "observer": "obs",
        "observed": "obsd",
    }))
    .bind("ws")
    .execute(&test_db.pool)
    .await
    .expect("enqueue dream");

    // Specialists answer immediately (no tool calls).
    let http = CannedLlmHttp(json!({
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1}
    }));
    let emitter = Arc::new(CapturingEmitter {
        events: std::sync::Mutex::new(Vec::new()),
    });
    let worker = Arc::new(DeriverWorker::new(
        test_db.pool.clone(),
        Arc::new(http),
        Arc::new(IndexedEmbedder),
        Arc::clone(&emitter) as Arc<dyn Emitter>,
        TransportApiKeys {
            anthropic: None,
            openai: Some("k".to_string()),
            gemini: None,
        },
        DeriverModelSettings::default(),
        SummaryGlobalSettings::default(),
        honcho_api_rs::dreamer::orchestrator::DreamModelSettings::default(),
        honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::default(),
        DeriverSettings {
            workers: 1,
            ..DeriverSettings::default()
        },
        Arc::new(ReqwestWebhookSender::new()),
        None,
    ));

    let processed = worker.poll_once().await.expect("poll once");
    assert_eq!(processed, 1);

    // The dream ran: a dream.run event was emitted.
    let run_events = {
        let events = emitter.events.lock().unwrap();
        events.iter().filter(|(t, _)| t == "dream.run").count()
    };
    assert_eq!(run_events, 1);

    // The dream queue item is marked processed.
    let unprocessed: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM queue WHERE task_type = 'dream' AND processed = false")
            .fetch_one(&test_db.pool)
            .await
            .expect("unprocessed count");
    assert_eq!(unprocessed, 0);

    test_db.teardown().await;
}

#[tokio::test]
async fn check_and_schedule_dream_enqueues_then_dedups() {
    use honcho_api_rs::dreamer::scheduler::{check_and_schedule_dream, DreamScheduleSettings};

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    db::get_or_create_collection(&test_db.pool, "ws", "obs", "obsd")
        .await
        .expect("collection");

    // Two explicit documents — meets a threshold of 2.
    for (i, content) in ["fact one", "fact two"].iter().enumerate() {
        sqlx::query(
            "INSERT INTO documents (id, content, workspace_name, observer, observed, level) \
             VALUES ($1, $2, 'ws', 'obs', 'obsd', 'explicit')",
        )
        .bind(format!("dreamdoc{i:013}"))
        .bind(content)
        .execute(&test_db.pool)
        .await
        .expect("seed document");
    }

    let settings = DreamScheduleSettings {
        document_threshold: 2,
        ..DreamScheduleSettings::default()
    };
    let now = Utc::now();

    // First call: threshold met, no pending dream → schedule one.
    let scheduled = check_and_schedule_dream(
        &test_db.pool,
        &settings,
        "ws",
        "obs",
        "obsd",
        &json!({}),
        "sess",
        now,
    )
    .await
    .expect("schedule");
    assert!(scheduled);

    // Exactly one dream queue item, carrying the scheduling-attribution payload.
    let row: (String, Value) = sqlx::query_as(
        "SELECT work_unit_key, payload FROM queue WHERE task_type = 'dream'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("dream queue item");
    assert_eq!(row.0, "dream:omni:ws:obs:obsd");
    assert_eq!(row.1["dream_type"], json!("omni"));
    assert_eq!(row.1["trigger_reason"], json!("document_threshold"));
    assert_eq!(row.1["delay_reason"], json!("idle_timeout"));
    assert_eq!(row.1["documents_since_last_dream_at_schedule"], json!(2));
    assert_eq!(row.1["document_threshold"], json!(2));
    assert_eq!(row.1["session_name"], json!("sess"));

    // Second call: an unprocessed dream is already pending → no duplicate.
    let again = check_and_schedule_dream(
        &test_db.pool,
        &settings,
        "ws",
        "obs",
        "obsd",
        &json!({}),
        "sess",
        now,
    )
    .await
    .expect("schedule again");
    assert!(!again);

    let dream_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM queue WHERE task_type = 'dream'")
            .fetch_one(&test_db.pool)
            .await
            .expect("count");
    assert_eq!(dream_count, 1);

    // Below threshold (count unchanged but baseline now equals it) → no schedule.
    let below = check_and_schedule_dream(
        &test_db.pool,
        &settings,
        "ws",
        "obs",
        "obsd",
        &json!({"dream": {"last_dream_document_count": 2}}),
        "sess",
        now,
    )
    .await
    .expect("below threshold");
    assert!(!below);

    test_db.teardown().await;
}

#[tokio::test]
async fn get_or_create_collection_creates_then_reuses() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["alice", "bob"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }

    let first = db::get_or_create_collection(&test_db.pool, "ws", "alice", "bob")
        .await
        .expect("create collection");
    assert_eq!(first.observer, "alice");
    assert_eq!(first.observed, "bob");
    assert_eq!(first.workspace_name, "ws");
    assert_eq!(first.id.len(), 21);
    assert_eq!(first.metadata, json!({}));

    // Same key returns the same row (no duplicate created).
    let second = db::get_or_create_collection(&test_db.pool, "ws", "alice", "bob")
        .await
        .expect("reuse collection");
    assert_eq!(second.id, first.id);

    // A different (observer, observed) pair makes a distinct collection.
    let other = db::get_or_create_collection(&test_db.pool, "ws", "bob", "alice")
        .await
        .expect("other collection");
    assert_ne!(other.id, first.id);

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM collections WHERE workspace_name = 'ws'")
        .fetch_one(&test_db.pool)
        .await
        .expect("count");
    assert_eq!(count, 2);

    test_db.teardown().await;
}

// --- create_documents + dedup (representation write path) ---

async fn setup_collection_fixtures(pool: &PgPool) {
    db::get_or_create_workspace(pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["alice", "bob"] {
        db::get_or_create_peer(pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    db::get_or_create_collection(pool, "ws", "alice", "bob")
        .await
        .expect("collection");
}

fn doc_to_create(content: &str, dim: usize) -> db::DocumentToCreate {
    db::DocumentToCreate {
        content: content.to_string(),
        session_name: None,
        level: "explicit".to_string(),
        internal_metadata: json!({}),
        embedding: query_vector(&[(dim, 1.0)]),
        times_derived: 1,
        source_ids: None,
    }
}

#[tokio::test]
async fn create_documents_inserts_and_marks_synced() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    setup_collection_fixtures(&test_db.pool).await;

    let count = db::create_documents(
        &test_db.pool,
        vec![doc_to_create("first", 5), doc_to_create("second", 6)],
        "ws",
        "alice",
        "bob",
        false,
    )
    .await
    .expect("create documents");
    assert_eq!(count, 2);

    let synced: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM documents WHERE observer = 'alice' AND observed = 'bob' \
         AND sync_state = 'synced' AND deleted_at IS NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("count synced");
    assert_eq!(synced, 2);

    test_db.teardown().await;
}

#[tokio::test]
async fn create_documents_dedup_keeps_more_informative_new_doc() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    setup_collection_fixtures(&test_db.pool).await;

    // Existing short observation.
    db::create_documents(&test_db.pool, vec![doc_to_create("cat", 5)], "ws", "alice", "bob", false)
        .await
        .expect("seed existing");

    // New doc with the SAME embedding (distance 0 -> duplicate) but more tokens.
    let count = db::create_documents(
        &test_db.pool,
        vec![doc_to_create("the orange cat sleeps on the warm windowsill", 5)],
        "ws",
        "alice",
        "bob",
        true,
    )
    .await
    .expect("dedup create");
    assert_eq!(count, 1);

    // Existing is soft-deleted; exactly one live doc remains, the longer one,
    // with times_derived carried forward to 2.
    let live: Vec<(String, i32)> = sqlx::query_as(
        "SELECT content, times_derived FROM documents \
         WHERE observer = 'alice' AND observed = 'bob' AND deleted_at IS NULL",
    )
    .fetch_all(&test_db.pool)
    .await
    .expect("live docs");
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].0, "the orange cat sleeps on the warm windowsill");
    assert_eq!(live[0].1, 2);

    let deleted: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM documents WHERE observer = 'alice' AND observed = 'bob' \
         AND deleted_at IS NOT NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("count deleted");
    assert_eq!(deleted, 1);

    test_db.teardown().await;
}

#[tokio::test]
async fn create_documents_dedup_rejects_less_informative_new_doc() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    setup_collection_fixtures(&test_db.pool).await;

    db::create_documents(
        &test_db.pool,
        vec![doc_to_create("the orange cat sleeps on the warm windowsill", 5)],
        "ws",
        "alice",
        "bob",
        false,
    )
    .await
    .expect("seed existing");

    // New shorter doc, same embedding -> existing wins, new rejected.
    let count = db::create_documents(
        &test_db.pool,
        vec![doc_to_create("cat", 5)],
        "ws",
        "alice",
        "bob",
        true,
    )
    .await
    .expect("dedup create");
    assert_eq!(count, 0);

    // Existing survives with times_derived incremented to 2.
    let row: (i64, i32) = sqlx::query_as(
        "SELECT COUNT(*), MAX(times_derived) FROM documents \
         WHERE observer = 'alice' AND observed = 'bob' AND deleted_at IS NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("existing");
    assert_eq!(row.0, 1);
    assert_eq!(row.1, 2);

    test_db.teardown().await;
}

#[tokio::test]
async fn create_documents_dedup_keeps_dissimilar_doc() {
    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    setup_collection_fixtures(&test_db.pool).await;

    db::create_documents(&test_db.pool, vec![doc_to_create("cat", 5)], "ws", "alice", "bob", false)
        .await
        .expect("seed existing");

    // Orthogonal embedding (distance 1 > 0.05) -> not a duplicate, kept.
    let count = db::create_documents(
        &test_db.pool,
        vec![doc_to_create("dog", 6)],
        "ws",
        "alice",
        "bob",
        true,
    )
    .await
    .expect("dedup create");
    assert_eq!(count, 1);

    let live: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM documents WHERE observer = 'alice' AND observed = 'bob' \
         AND deleted_at IS NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("count live");
    assert_eq!(live, 2);

    test_db.teardown().await;
}

// --- save_representation orchestrator ---

struct IndexedEmbedder;
impl honcho_api_rs::dialectic::Embedder for IndexedEmbedder {
    async fn embed(&self, _query: &str) -> Result<Vec<f32>, String> {
        Ok(query_vector(&[(0, 1.0)]))
    }
    async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        // Distinct one-hot per input so dedup doesn't collapse them.
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| query_vector(&[(i, 1.0)]))
            .collect())
    }
}

#[tokio::test]
async fn create_observations_writes_documents_with_levels() {
    use honcho_api_rs::dreamer::handlers::{ObservationInput, create_observations};

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }

    let obs = vec![
        ObservationInput {
            content: "  alice is a swe  ".into(),
            level: "deductive".into(),
            source_ids: Some(vec!["s1".into()]),
            premises: Some(vec!["works at google".into()]),
            sources: None,
            pattern_type: None,
            confidence: None,
        },
        // Empty-after-strip → dropped before embed/insert.
        ObservationInput {
            content: "   ".into(),
            level: "deductive".into(),
            source_ids: Some(vec!["s2".into()]),
            premises: Some(vec!["x".into()]),
            sources: None,
            pattern_type: None,
            confidence: None,
        },
    ];

    let out = create_observations(
        &test_db.pool,
        &IndexedEmbedder,
        "ws",
        "obs",
        "obsd",
        None,
        obs,
        &[],
        "2026-06-22T00:00:00Z",
        true,
    )
    .await
    .expect("create_observations");

    assert_eq!(out.created_count, 1);
    assert_eq!(out.created_levels, vec!["deductive".to_string()]);
    assert!(out.failed.is_empty());

    // The accepted document is persisted with stripped content + metadata.
    let row: (String, String, Value) = sqlx::query_as(
        "SELECT content, level, internal_metadata FROM documents \
         WHERE workspace_name = 'ws' AND observer = 'obs' AND observed = 'obsd' AND deleted_at IS NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("fetch doc");
    assert_eq!(row.0, "alice is a swe");
    assert_eq!(row.1, "deductive");
    assert_eq!(row.2["premises"], json!(["works at google"]));
    assert_eq!(row.2["source_ids"], json!(["s1"]));

    test_db.teardown().await;
}

#[tokio::test]
async fn dreamer_tool_executor_creates_updates_deletes_and_rolls_up() {
    use honcho_api_rs::dialectic::ToolContext;
    use honcho_api_rs::dreamer::executor::DreamerToolExecutor;
    use honcho_api_rs::llm::tool_loop::ToolExecutor;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }

    let emitter = CapturingEmitter {
        events: std::sync::Mutex::new(Vec::new()),
    };
    let ctx = ToolContext {
        workspace_name: "ws".to_string(),
        observer: "obs".to_string(),
        observed: "obsd".to_string(),
        session_name: None,
    };
    let exec = DreamerToolExecutor::new(
        &test_db.pool,
        ctx,
        &IndexedEmbedder,
        true,  // include_observation_ids
        true,  // peer_card_create
        "run1".to_string(),
        "deduction".to_string(),
        "dream".to_string(),
        &emitter,
        Some("9.9.9".to_string()),
        true, // deduplicate
    );

    // create_observations_deductive
    let created = exec
        .execute(
            "create_observations_deductive",
            &json!({"observations": [{"content": "alice is a swe", "source_ids": ["s1"], "premises": ["works at google"]}]}),
        )
        .await
        .expect("create");
    assert!(created.starts_with("Created 1 observations"), "{created}");

    // update_peer_card (one valid, one rejected)
    let pc = exec
        .execute(
            "update_peer_card",
            &json!({"content": ["IDENTITY: Name: Alice", "TRAIT: nope"]}),
        )
        .await
        .expect("peer card");
    assert!(pc.contains("Updated peer card for obsd by obs with 1 entries."), "{pc}");

    // get_recent_observations shows the created deductive observation with id.
    let recent = exec
        .execute("get_recent_observations", &json!({"limit": 10}))
        .await
        .expect("recent");
    assert!(recent.contains("recent observations from all sessions"), "{recent}");
    assert!(recent.contains("alice is a swe"), "{recent}");

    // Rollups so far.
    let metrics = exec.metrics_snapshot();
    assert_eq!(metrics.created_observation_count, 1);
    assert_eq!(metrics.created_counts_by_level.get("deductive"), Some(&1));
    assert!(metrics.peer_card_updated);

    // Emitted the created + peer_card_updated events.
    {
        let events = emitter.events.lock().unwrap();
        let types: Vec<&str> = events.iter().map(|(t, _)| t.as_str()).collect();
        assert!(types.contains(&"agent.tool.conclusions.created"));
        assert!(types.contains(&"agent.tool.peer_card.updated"));
    }

    // Delete the created observation by its id.
    let doc_id: String = sqlx::query_scalar(
        "SELECT id FROM documents WHERE workspace_name='ws' AND observer='obs' AND observed='obsd' AND deleted_at IS NULL",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("doc id");
    let deleted = exec
        .execute("delete_observations", &json!({"observation_ids": [doc_id]}))
        .await
        .expect("delete");
    assert_eq!(deleted, "Deleted 1 observations");
    let metrics = exec.metrics_snapshot();
    assert_eq!(metrics.deleted_observation_count, 1);
    assert_eq!(metrics.deleted_counts_by_level.get("deductive"), Some(&1));

    test_db.teardown().await;
}

#[tokio::test]
async fn run_specialist_preflights_runs_loop_and_emits_event() {
    use honcho_api_rs::dreamer::specialists::{SpecialistKind, run_specialist};
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::llm::{ModelConfig, Provider};

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");

    // Model answers immediately (no tool calls) — exercises the assembly:
    // preflight peers, prompt build, caller, loop, event emit.
    let http = CannedLlmHttp(json!({
        "choices": [{"message": {"content": "done dreaming"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3}
    }));
    let emitter = CapturingEmitter {
        events: std::sync::Mutex::new(Vec::new()),
    };

    let result = run_specialist(
        SpecialistKind::Deduction,
        &test_db.pool,
        &http,
        TransportApiKeys {
            anthropic: None,
            openai: Some("k".to_string()),
            gemini: None,
        },
        &IndexedEmbedder,
        "ws",
        "obs",
        "obsd",
        None,
        None,
        true, // peer_card_create
        ModelConfig::new("gpt-5.4-mini", Provider::Openai),
        "run-xyz",
        &emitter,
        Some("9.9.9".to_string()),
        true,
    )
    .await
    .expect("run specialist");

    assert!(result.success);
    assert_eq!(result.specialist_type, "deduction");
    assert_eq!(result.run_id, "run-xyz");
    assert_eq!(result.content, "done dreaming");
    assert_eq!(result.input_tokens, 11);
    assert_eq!(result.output_tokens, 3);

    // Preflight created both peers.
    let peers: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM peers WHERE workspace_name='ws' AND name IN ('obs','obsd')",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("peer count");
    assert_eq!(peers, 2);

    // Emitted exactly one dream.specialist event, success=true.
    let events = emitter.events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "dream.specialist");
    assert_eq!(events[0].1["success"], json!(true));
    assert_eq!(events[0].1["specialist_type"], json!("deduction"));

    test_db.teardown().await;
}

#[tokio::test]
async fn process_dream_runs_specialists_and_writes_guard_pair() {
    use honcho_api_rs::deriver::payload::{DreamPayload, DreamType};
    use honcho_api_rs::dreamer::orchestrator::{DreamModelSettings, process_dream};
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::producer::ResolvedConfiguration;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");
    for peer in ["obs", "obsd"] {
        db::get_or_create_peer(&test_db.pool, "ws", peer, None, None)
            .await
            .expect("peer");
    }
    db::get_or_create_collection(&test_db.pool, "ws", "obs", "obsd")
        .await
        .expect("collection");
    // Two explicit documents → guard count should record 2.
    for i in 0..2 {
        sqlx::query(
            "INSERT INTO documents (id, content, workspace_name, observer, observed, level, sync_state) \
             VALUES ($1, 'fact', 'ws', 'obs', 'obsd', 'explicit', 'synced')",
        )
        .bind(doc_id(i))
        .execute(&test_db.pool)
        .await
        .expect("seed explicit doc");
    }

    // Specialists answer immediately (no tool calls).
    let http = CannedLlmHttp(json!({
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2}
    }));
    let emitter = CapturingEmitter {
        events: std::sync::Mutex::new(Vec::new()),
    };
    let payload = DreamPayload {
        dream_type: DreamType::Omni,
        observer: "obs".to_string(),
        observed: "obsd".to_string(),
        session_name: None,
        trigger_reason: Some("document_threshold".to_string()),
        delay_reason: None,
        documents_since_last_dream_at_schedule: Some(2),
        document_threshold: Some(2),
    };

    process_dream(
        &test_db.pool,
        &http,
        TransportApiKeys {
            anthropic: None,
            openai: Some("k".to_string()),
            gemini: None,
        },
        &IndexedEmbedder,
        &payload,
        "ws",
        &ResolvedConfiguration::default(),
        &DreamModelSettings::default(),
        &emitter,
        Some("9.9.9".to_string()),
        "2026-06-22T00:00:00Z",
    )
    .await;

    // Two specialist events + one run event.
    let types: Vec<String> = {
        let events = emitter.events.lock().unwrap();
        events.iter().map(|(t, _)| t.clone()).collect()
    };
    assert_eq!(types.iter().filter(|t| *t == "dream.specialist").count(), 2);
    assert_eq!(types.iter().filter(|t| *t == "dream.run").count(), 1);

    // Guard-pair write recorded the explicit count + timestamp.
    let dream_meta: Value = sqlx::query_scalar(
        "SELECT internal_metadata -> 'dream' FROM collections \
         WHERE workspace_name='ws' AND observer='obs' AND observed='obsd'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("dream meta");
    assert_eq!(dream_meta["last_dream_document_count"], json!(2));
    assert_eq!(dream_meta["last_dream_at"], json!("2026-06-22T00:00:00Z"));

    test_db.teardown().await;
}

#[tokio::test]
async fn save_representation_writes_documents() {
    use honcho_api_rs::representation::{ExplicitObservation, Representation};
    use honcho_api_rs::representation_manager::save_representation;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    // create_messages auto-creates workspace+session+peer bob; add peer alice.
    db::create_messages(&test_db.pool, "ws", "sess", &[message("bob", "hi")], false, 8192)
        .await
        .expect("seed session");
    db::get_or_create_peer(&test_db.pool, "ws", "alice", None, None)
        .await
        .expect("peer alice");

    let created = Utc.with_ymd_and_hms(2025, 3, 4, 12, 0, 0).unwrap();
    let rep = Representation {
        explicit: vec![
            ExplicitObservation {
                id: String::new(),
                created_at: created,
                message_ids: vec![1],
                session_name: Some("sess".to_string()),
                content: "bob likes coffee".to_string(),
            },
            ExplicitObservation {
                id: String::new(),
                created_at: created,
                message_ids: vec![1],
                session_name: Some("sess".to_string()),
                content: "bob lives in Berlin".to_string(),
            },
        ],
        ..Representation::default()
    };

    let count = save_representation(
        &test_db.pool,
        &IndexedEmbedder,
        "ws",
        "alice",
        "bob",
        &rep,
        &[1],
        "sess",
        created,
        true,
        None,
    )
    .await
    .expect("save representation");
    assert_eq!(count, 2);

    let rows: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT content, level, sync_state FROM documents \
         WHERE observer = 'alice' AND observed = 'bob' AND deleted_at IS NULL ORDER BY content",
    )
    .fetch_all(&test_db.pool)
    .await
    .expect("docs");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, "bob likes coffee");
    assert_eq!(rows[0].1, "explicit");
    assert_eq!(rows[0].2, "synced");

    // The collection was created.
    let collections: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM collections WHERE observer = 'alice' AND observed = 'bob'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("collection count");
    assert_eq!(collections, 1);

    test_db.teardown().await;
}

#[tokio::test]
async fn save_representation_schedules_dream_when_enabled() {
    use honcho_api_rs::dreamer::scheduler::DreamScheduleSettings;
    use honcho_api_rs::representation::{ExplicitObservation, Representation};
    use honcho_api_rs::representation_manager::save_representation;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::create_messages(&test_db.pool, "ws", "sess", &[message("bob", "hi")], false, 8192)
        .await
        .expect("seed session");
    db::get_or_create_peer(&test_db.pool, "ws", "alice", None, None)
        .await
        .expect("peer alice");

    let created = Utc.with_ymd_and_hms(2025, 3, 4, 12, 0, 0).unwrap();
    let mk = |content: &str| ExplicitObservation {
        id: String::new(),
        created_at: created,
        message_ids: vec![1],
        session_name: Some("sess".to_string()),
        content: content.to_string(),
    };
    let rep = Representation {
        explicit: vec![mk("bob likes coffee"), mk("bob lives in Berlin")],
        ..Representation::default()
    };

    // threshold = 1 → the two explicit docs trip the gate.
    let dream_settings = DreamScheduleSettings {
        document_threshold: 1,
        ..DreamScheduleSettings::default()
    };
    let count = save_representation(
        &test_db.pool,
        &IndexedEmbedder,
        "ws",
        "alice",
        "bob",
        &rep,
        &[1],
        "sess",
        created,
        true,
        Some(&dream_settings),
    )
    .await
    .expect("save representation");
    assert_eq!(count, 2);

    // A dream was scheduled for (alice, bob).
    let dream_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM queue WHERE task_type = 'dream' \
         AND work_unit_key = 'dream:omni:ws:alice:bob'",
    )
    .fetch_one(&test_db.pool)
    .await
    .expect("dream count");
    assert_eq!(dream_count, 1);

    test_db.teardown().await;
}

#[tokio::test]
async fn save_representation_empty_writes_nothing() {
    use honcho_api_rs::representation::Representation;
    use honcho_api_rs::representation_manager::save_representation;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");

    let created = Utc.with_ymd_and_hms(2025, 3, 4, 12, 0, 0).unwrap();
    let count = save_representation(
        &test_db.pool,
        &IndexedEmbedder,
        "ws",
        "alice",
        "bob",
        &Representation::default(),
        &[1],
        "sess",
        created,
        true,
        None,
    )
    .await
    .expect("save empty");
    assert_eq!(count, 0);

    // No collection created (we returned before any DB work).
    let collections: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM collections")
        .fetch_one(&test_db.pool)
        .await
        .expect("collection count");
    assert_eq!(collections, 0);

    test_db.teardown().await;
}

// --- process_representation_tasks_batch orchestrator ---

/// A fixed-response `LlmHttp` for the deriver orchestrator test: every call
/// returns the same canned provider body, so the structured-output content is
/// deterministic without a real LLM.
struct CannedLlmHttp(serde_json::Value);
impl honcho_api_rs::llm::http::LlmHttp for CannedLlmHttp {
    async fn post_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &serde_json::Value,
    ) -> Result<serde_json::Value, honcho_api_rs::llm::http::LlmHttpError> {
        Ok(self.0.clone())
    }
}

#[tokio::test]
async fn process_representation_tasks_batch_saves_and_emits() {
    use honcho_api_rs::db::BatchMessage;
    use honcho_api_rs::deriver::deriver::{
        DeriverBatchContext, DeriverModelSettings, process_representation_tasks_batch,
    };
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::llm::{ModelConfig, Provider};
    use honcho_api_rs::producer::ResolvedConfiguration;
    use honcho_api_rs::telemetry::NoopEmitter;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    // Seed ws/sess/peer bob (message id 1) and observer peer alice.
    db::create_messages(
        &test_db.pool,
        "ws",
        "sess",
        &[message("bob", "I like coffee and live in Berlin")],
        false,
        8192,
    )
    .await
    .expect("seed session");
    db::get_or_create_peer(&test_db.pool, "ws", "alice", None, None)
        .await
        .expect("peer alice");

    // The LLM "returns" a representation as its text content (Anthropic shape).
    let content_json =
        "{\"explicit\":[{\"content\":\"bob likes coffee\"},{\"content\":\"bob lives in Berlin\"}]}";
    let http = CannedLlmHttp(json!({
        "content": [{"type": "text", "text": content_json}],
        "usage": {"input_tokens": 50, "output_tokens": 7},
        "stop_reason": "end_turn"
    }));

    let settings = DeriverModelSettings {
        model_config: ModelConfig::new("claude-x", Provider::Anthropic),
        ..DeriverModelSettings::default()
    };
    let keys = TransportApiKeys {
        anthropic: Some("k".to_string()),
        openai: None,
        gemini: None,
    };
    let emitter = NoopEmitter;
    let ctx = DeriverBatchContext {
        pool: &test_db.pool,
        http: &http,
        keys,
        embedder: &IndexedEmbedder,
        settings,
        emitter: &emitter,
        dream_schedule_settings:
            honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::default(),
    };

    let created = Utc.with_ymd_and_hms(2025, 3, 4, 12, 0, 0).unwrap();
    let messages = vec![BatchMessage {
        id: 1,
        public_id: "msg_1".to_string(),
        content: "I like coffee and live in Berlin".to_string(),
        created_at: created,
        peer_name: "bob".to_string(),
        token_count: 10,
        session_name: "sess".to_string(),
        workspace_name: "ws".to_string(),
    }];
    let configuration = ResolvedConfiguration::default();

    let event = process_representation_tasks_batch(
        &ctx,
        &messages,
        &configuration,
        &["alice".to_string()],
        "bob",
        &[1],
        false,
        false,
        1024,
    )
    .await
    .expect("batch ok")
    .expect("processed (Some)");

    // Telemetry accounting.
    assert_eq!(event.observer_count, 1);
    assert_eq!(event.explicit_conclusion_count, 2);
    assert_eq!(event.message_count, 1);
    assert_eq!(event.queue_items_processed, 1);
    assert_eq!(event.queued_message_count, 1);
    assert_eq!(event.input_tokens, 10); // messages_tokens (id 1 queued, token_count 10)
    assert_eq!(event.total_input_tokens, 50); // response.input_tokens
    assert_eq!(event.output_tokens, 7);
    assert_eq!(event.earliest_message_id, "msg_1");
    assert_eq!(event.latest_message_id, "msg_1");
    assert!(!event.hit_input_token_cap);

    // The two explicit observations were written to alice→bob.
    let rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT content, level FROM documents \
         WHERE observer = 'alice' AND observed = 'bob' AND deleted_at IS NULL ORDER BY content",
    )
    .fetch_all(&test_db.pool)
    .await
    .expect("docs");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, "bob likes coffee");
    assert_eq!(rows[0].1, "explicit");
    assert_eq!(rows[1].0, "bob lives in Berlin");

    test_db.teardown().await;
}

#[tokio::test]
async fn process_representation_tasks_batch_skips_when_reasoning_disabled() {
    use honcho_api_rs::db::BatchMessage;
    use honcho_api_rs::deriver::deriver::{
        DeriverBatchContext, DeriverModelSettings, process_representation_tasks_batch,
    };
    use honcho_api_rs::llm::credentials::TransportApiKeys;
    use honcho_api_rs::producer::ResolvedConfiguration;
    use honcho_api_rs::telemetry::NoopEmitter;

    let Some(test_db) = TestDb::setup().await else {
        return;
    };
    db::get_or_create_workspace(&test_db.pool, "ws", json!({}), json!({}))
        .await
        .expect("workspace");

    // An http that would panic if hit — it must not be called when disabled.
    let http = CannedLlmHttp(json!(null));
    let emitter = NoopEmitter;
    let ctx = DeriverBatchContext {
        pool: &test_db.pool,
        http: &http,
        keys: TransportApiKeys::default(),
        embedder: &IndexedEmbedder,
        settings: DeriverModelSettings::default(),
        emitter: &emitter,
        dream_schedule_settings:
            honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::default(),
    };

    let created = Utc.with_ymd_and_hms(2025, 3, 4, 12, 0, 0).unwrap();
    let messages = vec![BatchMessage {
        id: 1,
        public_id: "msg_1".to_string(),
        content: "hello".to_string(),
        created_at: created,
        peer_name: "bob".to_string(),
        token_count: 3,
        session_name: "sess".to_string(),
        workspace_name: "ws".to_string(),
    }];
    let configuration = ResolvedConfiguration {
        reasoning_enabled: false,
        ..ResolvedConfiguration::default()
    };

    let result = process_representation_tasks_batch(
        &ctx,
        &messages,
        &configuration,
        &["alice".to_string()],
        "bob",
        &[1],
        false,
        false,
        1024,
    )
    .await
    .expect("batch ok");
    assert!(result.is_none()); // reasoning disabled → early return

    // Nothing written.
    let docs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM documents")
        .fetch_one(&test_db.pool)
        .await
        .expect("doc count");
    assert_eq!(docs, 0);

    test_db.teardown().await;
}
