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

use honcho_api_rs::db;
use serde_json::json;
use sqlx::{Connection, Executor, PgConnection, PgPool};

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
