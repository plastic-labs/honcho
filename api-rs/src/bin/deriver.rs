//! The deriver worker binary, porting `python -m src.deriver`
//! (`QueueManager.main`): a long-running queue consumer that forms peer
//! representations off the background queue.
//!
//! Mirrors the API server's bootstrap (`src/main.rs`) for config + pool wiring,
//! then runs [`DeriverWorker::run`] until SIGINT/SIGTERM, draining in-flight work
//! units on shutdown. Representation, deletion, summary, reconciler, and webhook
//! work units are fully processed today; `dream` remains unported (see the module
//! docs in `deriver::queue_manager`). Telemetry uses a no-op emitter until the
//! CloudEvents transport lands.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use honcho_api_rs::config::AppConfig;
use honcho_api_rs::db::quote_identifier;
use honcho_api_rs::deriver::deriver::DeriverModelSettings;
use honcho_api_rs::deriver::queue_manager::DeriverWorker;
use honcho_api_rs::deriver::settings::DeriverSettings;
use honcho_api_rs::dialectic::OwnedOpenAiEmbedder;
use honcho_api_rs::llm::http::{Credentials, ReqwestHttp};
use honcho_api_rs::reconciler::scheduler::{
    DEFAULT_SYNC_VECTORS_INTERVAL_SECONDS, default_reconciler_tasks, run_reconciler_scheduler,
};
use honcho_api_rs::summarizer::SummaryGlobalSettings;
use honcho_api_rs::telemetry::NoopEmitter;
use honcho_api_rs::webhooks::ReqwestWebhookSender;
use sqlx::postgres::PgPoolOptions;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let config = AppConfig::from_env()?;
    let db_schema = config.db_schema.clone();
    let search_path_sql = format!("SET search_path TO {}", quote_identifier(&db_schema));
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .after_connect(move |connection, _meta| {
            let search_path_sql = search_path_sql.clone();
            Box::pin(async move {
                sqlx::query(&search_path_sql).execute(connection).await?;
                Ok(())
            })
        })
        .connect(&config.database_url)
        .await?;

    let embedding = config.embedding_config();
    let keys = config.llm_keys();

    // The completion transport and the embedding transport are independent
    // reqwest clients (the embedder owns its own so the worker can hold it for
    // 'static spawned tasks).
    let http = Arc::new(ReqwestHttp::default());
    let embedder = Arc::new(OwnedOpenAiEmbedder {
        http: ReqwestHttp::default(),
        credentials: Credentials::with_base_url(
            embedding.api_key.clone().unwrap_or_default(),
            embedding.base_url.clone(),
        ),
        model: embedding.model.clone(),
        vector_dimensions: embedding.vector_dimensions,
        send_dimensions: embedding.send_dimensions,
        max_tokens: embedding.max_tokens,
    });

    // The reconciler scheduler shares the pool; clone before the worker takes it.
    let scheduler_pool = pool.clone();

    let poll_settings = DeriverSettings::from_env();
    let webhook_secret = config.webhook_secret.clone();
    let worker = Arc::new(DeriverWorker::new(
        pool,
        http,
        embedder,
        Arc::new(NoopEmitter),
        keys,
        DeriverModelSettings::default(),
        SummaryGlobalSettings::default(),
        honcho_api_rs::dreamer::orchestrator::DreamModelSettings::default(),
        honcho_api_rs::dreamer::scheduler::DreamScheduleSettings::from_env(),
        poll_settings,
        Arc::new(ReqwestWebhookSender::new()),
        webhook_secret,
    ));

    // Graceful shutdown on SIGINT/SIGTERM.
    let shutdown = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&shutdown);
    tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        tracing::info!("shutdown signal received");
        signal_flag.store(true, Ordering::Relaxed);
    });

    // Host the reconciler scheduler in-process alongside the queue loop (Python
    // starts it from the deriver). It enqueues `reconciler:*` tasks on their
    // intervals; the worker then drains them. The sync_vectors interval source
    // (VECTOR_STORE.RECONCILIATION_INTERVAL_SECONDS) is not yet in AppConfig, so
    // the default (5 min) is used.
    let scheduler_tasks = default_reconciler_tasks(DEFAULT_SYNC_VECTORS_INTERVAL_SECONDS);
    let scheduler_handle = tokio::spawn(run_reconciler_scheduler(
        scheduler_pool,
        scheduler_tasks,
        Arc::clone(&shutdown),
    ));

    tracing::info!("deriver worker starting");
    Arc::clone(&worker).run(shutdown).await;
    // The worker returns once shutdown is set; the scheduler observes the same
    // flag, so just await its exit.
    let _ = scheduler_handle.await;
    tracing::info!("deriver worker stopped");
    Ok(())
}

/// Resolve when either SIGINT (Ctrl-C) or SIGTERM arrives.
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("install SIGTERM handler");
        let mut sigint =
            signal(SignalKind::interrupt()).expect("install SIGINT handler");
        tokio::select! {
            _ = sigterm.recv() => {}
            _ = sigint.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
