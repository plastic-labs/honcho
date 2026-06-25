use honcho_api_rs::app::{AppState, build_router};
use honcho_api_rs::cache::PeerCache;
use honcho_api_rs::config::AppConfig;
use honcho_api_rs::db::quote_identifier;
use sqlx::postgres::PgPoolOptions;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let config = AppConfig::from_env()?;
    let db_schema = config.db_schema.clone();
    let peer_cache = PeerCache::new(config.cache.clone());
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

    // Apply pending schema migrations before serving, mirroring the Python
    // entrypoint (`scripts/provision_db.py` -> `alembic upgrade head`). The
    // migrations are embedded at compile time from ./migrations and run inside
    // the configured search_path schema (set by the pool's after_connect hook),
    // so this honours DB_SCHEMA exactly like the Python provisioner. Opt out
    // with RUST_API_RUN_MIGRATIONS=false to manage schema out of band.
    let run_migrations = std::env::var("RUST_API_RUN_MIGRATIONS")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "false" | "0" | "no"
            )
        })
        .unwrap_or(true);
    if run_migrations {
        tracing::info!("applying database migrations");
        sqlx::migrate!("./migrations").run(&pool).await?;
        tracing::info!("database migrations up to date");
    }

    let bind_address = config.bind_address;
    let embedding = config.embedding_config();
    let dream_enabled = config.dream_enabled;
    let llm_keys = config.llm_keys();
    let app = build_router(AppState::new(
        pool,
        config.auth,
        config.db_schema,
        config.write_enabled,
        peer_cache,
        config.embed_messages,
        config.embedding_max_tokens,
        embedding,
        dream_enabled,
        honcho_api_rs::dialectic_config::DialecticSettings::default(),
        llm_keys,
    ));

    tracing::info!("honcho-api-rs listening on {bind_address}");
    let listener = TcpListener::bind(bind_address).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
