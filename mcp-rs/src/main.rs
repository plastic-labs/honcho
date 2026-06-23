use axum::Router;
use honcho_mcp_rs::mcp_server::HonchoMcp;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use std::net::SocketAddr;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

const DEFAULT_BIND_ADDRESS: &str = "0.0.0.0:8787";
const DEFAULT_HONCHO_API_URL: &str = "https://api.honcho.dev";
const DEFAULT_ALLOWED_HOSTS: &[&str] = &["localhost", "127.0.0.1", "::1"];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let bind_address = std::env::var("MCP_BIND_ADDRESS")
        .unwrap_or_else(|_| DEFAULT_BIND_ADDRESS.to_string())
        .parse::<SocketAddr>()?;
    let api_url =
        std::env::var("HONCHO_API_URL").unwrap_or_else(|_| DEFAULT_HONCHO_API_URL.to_string());
    let allowed_hosts = std::env::var("MCP_ALLOWED_HOSTS")
        .ok()
        .map(|hosts| {
            hosts
                .split(',')
                .map(str::trim)
                .filter(|host| !host.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|hosts| !hosts.is_empty())
        .unwrap_or_else(|| {
            DEFAULT_ALLOWED_HOSTS
                .iter()
                .map(|host| (*host).to_string())
                .collect()
        });
    let cancellation_token = CancellationToken::new();

    let service = StreamableHttpService::new(
        move || Ok(HonchoMcp::new(api_url.clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default()
            .with_allowed_hosts(allowed_hosts)
            .with_cancellation_token(cancellation_token.child_token()),
    );
    let app = Router::new().fallback_service(service);

    tracing::info!("honcho-mcp-rs listening on {bind_address}");
    let listener = tokio::net::TcpListener::bind(bind_address).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
