//! `honcho-worker-rs` entry point.
//!
//! Minimal polling loop for the no-op consumer milestone: claim work units,
//! drain them as no-ops, and periodically reclaim stale sessions. The adaptive
//! backoff/jitter and per-task handlers documented in the queue-schema reference
//! are added in later phases; this loop deliberately stays simple and runnable.

use std::time::Duration;

use honcho_worker_rs::config::WorkerConfig;
use honcho_worker_rs::{consumer, queue};
use sqlx::postgres::PgPoolOptions;
use tracing_subscriber::EnvFilter;

const POLL_INTERVAL: Duration = Duration::from_secs(1);

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let config = WorkerConfig::from_env()?;
    let search_path_sql = format!("SET search_path TO \"{}\"", config.db_schema.replace('"', "\"\""));
    let pool = PgPoolOptions::new()
        .max_connections((config.workers.max(1) as u32) + 1)
        .after_connect(move |connection, _meta| {
            let search_path_sql = search_path_sql.clone();
            Box::pin(async move {
                sqlx::query(&search_path_sql).execute(connection).await?;
                Ok(())
            })
        })
        .connect(&config.database_url)
        .await?;

    tracing::info!(workers = config.workers, "honcho-worker-rs starting");

    let mut shutdown = std::pin::pin!(tokio::signal::ctrl_c());
    loop {
        if let Err(error) = queue::cleanup_stale_work_units(&pool, config.stale_session_timeout).await
        {
            tracing::error!(%error, "stale work-unit cleanup failed");
        }
        match consumer::run_once(&pool, config.workers).await {
            Ok(count) if count > 0 => tracing::debug!(processed = count, "poll cycle processed items"),
            Ok(_) => {}
            Err(error) => tracing::error!(%error, "poll cycle failed"),
        }

        tokio::select! {
            _ = tokio::time::sleep(POLL_INTERVAL) => {}
            _ = &mut shutdown => {
                tracing::info!("shutdown signal received, exiting");
                break;
            }
        }
    }

    Ok(())
}
