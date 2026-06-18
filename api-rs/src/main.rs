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
    let bind_address = config.bind_address;
    let app = build_router(AppState::new(
        pool,
        config.auth,
        config.db_schema,
        config.write_enabled,
        peer_cache,
        config.embed_messages,
        config.embedding_max_tokens,
    ));

    tracing::info!("honcho-api-rs listening on {bind_address}");
    let listener = TcpListener::bind(bind_address).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
