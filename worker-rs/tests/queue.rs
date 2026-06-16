//! DB-backed state-transition tests for the queue primitives and no-op consumer.
//!
//! Gated on `HONCHO_WORKER_RS_TEST_DATABASE_URL` (a Postgres the test may create
//! schemas in). Each test runs in its own uniquely-named schema so they are
//! parallel-safe and self-cleaning. Example:
//!
//! ```bash
//! HONCHO_WORKER_RS_TEST_DATABASE_URL=postgres://postgres@127.0.0.1:5432/postgres \
//!   rtk cargo test --manifest-path worker-rs/Cargo.toml
//! ```
//!
//! Without the env var the tests no-op (pass) so the default `cargo test` run
//! stays hermetic, matching `api-rs`'s opt-in Redis integration test.

use honcho_worker_rs::config::normalize_python_postgres_url;
use honcho_worker_rs::{consumer, queue};
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};

/// Per-test database fixture: an isolated schema plus a pool whose connections
/// have `search_path` pinned to it.
struct TestDb {
    pool: PgPool,
    admin: PgPool,
    schema: String,
}

impl TestDb {
    async fn teardown(self) {
        let _ = sqlx::query(&format!("DROP SCHEMA IF EXISTS \"{}\" CASCADE", self.schema))
            .execute(&self.admin)
            .await;
        self.pool.close().await;
        self.admin.close().await;
    }

    async fn seed_item(&self, work_unit_key: &str, task_type: &str) -> i64 {
        sqlx::query_scalar::<_, i64>(
            "INSERT INTO queue (work_unit_key, task_type, payload) \
             VALUES ($1, $2, '{}'::jsonb) RETURNING id",
        )
        .bind(work_unit_key)
        .bind(task_type)
        .fetch_one(&self.pool)
        .await
        .expect("seed queue item")
    }

    async fn item_state(&self, id: i64) -> (bool, Option<String>) {
        let row = sqlx::query("SELECT processed, error FROM queue WHERE id = $1")
            .bind(id)
            .fetch_one(&self.pool)
            .await
            .expect("fetch item state");
        (row.get("processed"), row.get("error"))
    }

    async fn active_session_count(&self, work_unit_key: &str) -> i64 {
        sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM active_queue_sessions WHERE work_unit_key = $1",
        )
        .bind(work_unit_key)
        .fetch_one(&self.pool)
        .await
        .expect("count active sessions")
    }

    async fn pending_count(&self, work_unit_key: &str) -> i64 {
        sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM queue WHERE work_unit_key = $1 AND NOT processed",
        )
        .bind(work_unit_key)
        .fetch_one(&self.pool)
        .await
        .expect("count pending")
    }

    async fn backdate_active_session(&self, work_unit_key: &str, minutes: i64) {
        sqlx::query(
            "UPDATE active_queue_sessions \
             SET last_updated = now() - make_interval(mins => $2::int) \
             WHERE work_unit_key = $1",
        )
        .bind(work_unit_key)
        .bind(minutes)
        .execute(&self.pool)
        .await
        .expect("backdate active session");
    }
}

/// Returns `None` (test no-ops) when the gating env var is absent.
async fn setup() -> Option<TestDb> {
    let base_url = normalize_python_postgres_url(
        &std::env::var("HONCHO_WORKER_RS_TEST_DATABASE_URL").ok()?,
    );

    let schema = format!("wtest_{}", random_suffix());
    let admin = PgPoolOptions::new()
        .max_connections(1)
        .connect(&base_url)
        .await
        .expect("connect admin pool");
    sqlx::query(&format!("CREATE SCHEMA \"{schema}\""))
        .execute(&admin)
        .await
        .expect("create test schema");

    let search_path = format!("SET search_path TO \"{schema}\"");
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .after_connect(move |conn, _meta| {
            let search_path = search_path.clone();
            Box::pin(async move {
                sqlx::query(&search_path).execute(conn).await?;
                Ok(())
            })
        })
        .connect(&base_url)
        .await
        .expect("connect work pool");

    // Minimal DDL for the columns/constraints the worker relies on. The unique
    // constraint on active_queue_sessions.work_unit_key is the claim mutex and is
    // load-bearing; the rest mirror src/models.py.
    sqlx::query(
        "CREATE TABLE queue ( \
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, \
            session_id TEXT, \
            work_unit_key TEXT NOT NULL, \
            task_type TEXT NOT NULL, \
            payload JSONB NOT NULL, \
            processed BOOLEAN NOT NULL DEFAULT false, \
            error TEXT, \
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), \
            workspace_name TEXT, \
            message_id BIGINT \
        )",
    )
    .execute(&pool)
    .await
    .expect("create queue table");
    sqlx::query(
        "CREATE TABLE active_queue_sessions ( \
            id TEXT PRIMARY KEY, \
            work_unit_key TEXT NOT NULL UNIQUE, \
            last_updated TIMESTAMPTZ NOT NULL DEFAULT now() \
        )",
    )
    .execute(&pool)
    .await
    .expect("create active_queue_sessions table");

    Some(TestDb {
        pool,
        admin,
        schema,
    })
}

fn random_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    // Thread id adds entropy so parallel tests starting in the same nanosecond
    // still get distinct schema names.
    format!("{nanos:x}_{:?}", std::thread::current().id())
        .replace(['(', ')', ' ', '#'], "")
}

macro_rules! db_test {
    ($name:ident, $db:ident, $body:block) => {
        #[tokio::test]
        async fn $name() {
            let Some($db) = setup().await else {
                eprintln!(
                    "skipping {}: set HONCHO_WORKER_RS_TEST_DATABASE_URL to run",
                    stringify!($name)
                );
                return;
            };
            $body
            $db.teardown().await;
        }
    };
}

// pending -> processing: claiming inserts an active session and returns its token.
db_test!(claim_creates_active_session, db, {
    let key = "deletion:ws:session:s1";
    db.seed_item(key, "deletion").await;

    let claimed = queue::claim_work_units(&db.pool, 5).await.expect("claim");
    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].work_unit_key, key);
    assert!(!claimed[0].aqs_id.is_empty());
    assert_eq!(db.active_session_count(key).await, 1);
});

// The unique work_unit_key makes claiming a work unit exclusive: a second claim
// while the active session still exists returns nothing.
db_test!(claim_is_exclusive_per_work_unit, db, {
    let key = "deletion:ws:session:s2";
    db.seed_item(key, "deletion").await;

    let first = queue::claim_work_units(&db.pool, 5).await.expect("first claim");
    assert_eq!(first.len(), 1);
    let second = queue::claim_work_units(&db.pool, 5).await.expect("second claim");
    assert!(second.is_empty(), "claimed unit must not be re-claimable");
});

// processing -> completed.
db_test!(mark_processed_completes_item, db, {
    let key = "deletion:ws:session:s3";
    let id = db.seed_item(key, "deletion").await;
    let claimed = queue::claim_work_units(&db.pool, 5).await.expect("claim");
    let unit = &claimed[0];

    let item = queue::next_unprocessed_item(&db.pool, &unit.work_unit_key, &unit.aqs_id)
        .await
        .expect("next item")
        .expect("an item is pending");
    assert_eq!(item.id, id);

    queue::mark_items_processed(&db.pool, &[item.id], &unit.work_unit_key)
        .await
        .expect("mark processed");
    assert_eq!(db.item_state(id).await, (true, None));
    assert!(
        queue::next_unprocessed_item(&db.pool, &unit.work_unit_key, &unit.aqs_id)
            .await
            .expect("next after drain")
            .is_none()
    );
});

// processing -> failed (with retry): only the FIRST failing item is marked
// errored; the rest of the batch stays pending and is retried.
db_test!(error_marks_only_first_item, db, {
    let key = "summary:ws:session:None:None";
    let first = db.seed_item(key, "summary").await;
    let second = db.seed_item(key, "summary").await;
    let third = db.seed_item(key, "summary").await;
    let claimed = queue::claim_work_units(&db.pool, 5).await.expect("claim");
    let unit = &claimed[0];

    queue::mark_item_errored(&db.pool, first, &unit.work_unit_key, "boom")
        .await
        .expect("mark errored");

    assert_eq!(db.item_state(first).await, (true, Some("boom".to_string())));
    assert_eq!(db.item_state(second).await, (false, None));
    assert_eq!(db.item_state(third).await, (false, None));

    // Next pending item is the second (lowest remaining id) — retry continues.
    let next = queue::next_unprocessed_item(&db.pool, &unit.work_unit_key, &unit.aqs_id)
        .await
        .expect("next")
        .expect("second item pending");
    assert_eq!(next.id, second);
});

// next item is FIFO by id within a work unit.
db_test!(next_item_is_fifo_by_id, db, {
    let key = "deletion:ws:session:s4";
    let a = db.seed_item(key, "deletion").await;
    let b = db.seed_item(key, "deletion").await;
    let c = db.seed_item(key, "deletion").await;
    let claimed = queue::claim_work_units(&db.pool, 5).await.expect("claim");
    let unit = &claimed[0];

    for expected in [a, b, c] {
        let item = queue::next_unprocessed_item(&db.pool, &unit.work_unit_key, &unit.aqs_id)
            .await
            .expect("next")
            .expect("pending item");
        assert_eq!(item.id, expected);
        queue::mark_items_processed(&db.pool, &[item.id], &unit.work_unit_key)
            .await
            .expect("mark processed");
    }
});

// stale processing -> pending: cleanup deletes the aged active session and the
// (still unprocessed) work unit becomes claimable again.
db_test!(stale_cleanup_returns_unit_to_pending, db, {
    use std::time::Duration;
    let key = "deletion:ws:session:s5";
    db.seed_item(key, "deletion").await;
    let claimed = queue::claim_work_units(&db.pool, 5).await.expect("claim");
    assert_eq!(claimed.len(), 1);
    assert_eq!(db.active_session_count(key).await, 1);

    // Heartbeat older than the timeout.
    db.backdate_active_session(key, 10).await;
    let reclaimed = queue::cleanup_stale_work_units(&db.pool, Duration::from_secs(5 * 60))
        .await
        .expect("stale cleanup");
    assert_eq!(reclaimed, 1);
    assert_eq!(db.active_session_count(key).await, 0);

    // Fresh sessions are NOT reaped.
    let again = queue::claim_work_units(&db.pool, 5).await.expect("reclaim");
    assert_eq!(again.len(), 1, "unit returns to the pending pool");
    let fresh = queue::cleanup_stale_work_units(&db.pool, Duration::from_secs(5 * 60))
        .await
        .expect("stale cleanup of fresh session");
    assert_eq!(fresh, 0);
});

// release deletes the active session; a second release is a no-op.
db_test!(release_is_idempotent, db, {
    let key = "deletion:ws:session:s6";
    db.seed_item(key, "deletion").await;
    let claimed = queue::claim_work_units(&db.pool, 5).await.expect("claim");
    let unit = &claimed[0];

    assert!(queue::release_work_unit(&db.pool, &unit.aqs_id, &unit.work_unit_key)
        .await
        .expect("release"));
    assert_eq!(db.active_session_count(key).await, 0);
    assert!(!queue::release_work_unit(&db.pool, &unit.aqs_id, &unit.work_unit_key)
        .await
        .expect("second release"));
});

// Consumer: draining a claimed unit processes every item and releases the session.
db_test!(drain_work_unit_processes_all_and_releases, db, {
    let key = "deletion:ws:session:s7";
    for _ in 0..3 {
        db.seed_item(key, "deletion").await;
    }
    let claimed = queue::claim_work_units(&db.pool, 5).await.expect("claim");
    let processed = consumer::drain_work_unit(&db.pool, &claimed[0])
        .await
        .expect("drain");
    assert_eq!(processed, 3);
    assert_eq!(db.pending_count(key).await, 0);
    assert_eq!(db.active_session_count(key).await, 0);
});

// Consumer: run_once claims and drains multiple work units in one cycle.
db_test!(run_once_drains_multiple_units, db, {
    db.seed_item("deletion:ws:session:a", "deletion").await;
    db.seed_item("deletion:ws:session:a", "deletion").await;
    db.seed_item("summary:ws:session:None:None", "summary").await;

    let total = consumer::run_once(&db.pool, 5).await.expect("run_once");
    assert_eq!(total, 3);
    assert_eq!(db.pending_count("deletion:ws:session:a").await, 0);
    assert_eq!(db.pending_count("summary:ws:session:None:None").await, 0);
});

// Consumer: run_once never claims more work units than the worker limit.
db_test!(run_once_respects_worker_limit, db, {
    db.seed_item("deletion:ws:session:x", "deletion").await;
    db.seed_item("deletion:ws:session:y", "deletion").await;
    db.seed_item("deletion:ws:session:z", "deletion").await;

    let total = consumer::run_once(&db.pool, 1).await.expect("run_once");
    assert_eq!(total, 1, "only one work unit drained per cycle at limit 1");

    let still_pending: i64 = db.pending_count("deletion:ws:session:x").await
        + db.pending_count("deletion:ws:session:y").await
        + db.pending_count("deletion:ws:session:z").await;
    assert_eq!(still_pending, 2, "the other two units remain pending");
});
