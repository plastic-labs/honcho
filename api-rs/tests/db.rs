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
use honcho_api_rs::db::MessageInsert;
use honcho_api_rs::search;
use honcho_api_rs::search::MessageRef;
use serde_json::json;
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
    let ranked = search::semantic_search_pgvector(&test_db.pool, "ws", &toward_apple, 10)
        .await
        .expect("semantic search");
    assert_eq!(ranked, vec![apple_id.clone(), banana_id.clone()]);
    // Despite apple having two embedding chunks, it appears once.
    assert_eq!(ranked.iter().filter(|id| **id == apple_id).count(), 1);

    // Flip the query toward dim 5 and banana ranks first.
    let toward_banana = query_vector(&[(5, 1.0), (0, 0.2)]);
    let ranked = search::semantic_search_pgvector(&test_db.pool, "ws", &toward_banana, 10)
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
    let fox = search::fulltext_search(&test_db.pool, "ws", "fox", 10)
        .await
        .expect("fts fox");
    assert_eq!(fox.len(), 1);

    // "dog" stems to match "dogs" via the english FTS config.
    let dog = search::fulltext_search(&test_db.pool, "ws", "dog", 10)
        .await
        .expect("fts dog");
    assert_eq!(dog.len(), 1);

    // A query with FTS-hostile chars ("c++") takes the literal ILIKE path.
    let cpp = search::fulltext_search(&test_db.pool, "ws", "c++", 10)
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
