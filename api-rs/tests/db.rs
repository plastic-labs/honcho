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
use honcho_api_rs::db::{ConclusionWriteError, MessageInsert, NewConclusion};
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
