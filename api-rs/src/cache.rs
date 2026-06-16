use std::time::Duration;

const DEFAULT_CACHE_URL: &str = "redis://localhost:6379/0?suppress=true";
const DEFAULT_CACHE_NAMESPACE: &str = "honcho";
const CACHE_INVALIDATION_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheConfig {
    pub enabled: bool,
    pub url: String,
    pub namespace: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            url: DEFAULT_CACHE_URL.to_string(),
            namespace: DEFAULT_CACHE_NAMESPACE.to_string(),
        }
    }
}

#[derive(Clone)]
pub struct PeerCache {
    config: CacheConfig,
    client: Option<redis::Client>,
}

impl std::fmt::Debug for PeerCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PeerCache")
            .field("config", &self.config)
            .field("client_configured", &self.client.is_some())
            .finish()
    }
}

impl PeerCache {
    pub fn new(config: CacheConfig) -> Self {
        let client = if config.enabled {
            match redis::Client::open(redis_client_url(&config.url)) {
                Ok(client) => Some(client),
                Err(error) => {
                    tracing::warn!("failed to configure Redis cache client: {error}");
                    None
                }
            }
        } else {
            None
        };

        Self { config, client }
    }

    pub fn disabled() -> Self {
        Self::new(CacheConfig::default())
    }

    pub async fn invalidate_peer(&self, workspace_name: &str, peer_name: &str) {
        self.invalidate_key(peer_cache_key(
            &self.config.namespace,
            workspace_name,
            peer_name,
        ))
        .await;
    }

    pub async fn invalidate_session(&self, workspace_name: &str, session_name: &str) {
        self.invalidate_key(session_cache_key(
            &self.config.namespace,
            workspace_name,
            session_name,
        ))
        .await;
    }

    async fn invalidate_key(&self, cache_key: String) {
        let Some(client) = &self.client else {
            return;
        };
        let operation = async {
            let mut connection = client.get_multiplexed_async_connection().await?;
            let _: i64 = redis::cmd("DEL")
                .arg(&cache_key)
                .query_async(&mut connection)
                .await?;
            Ok::<(), redis::RedisError>(())
        };

        match tokio::time::timeout(CACHE_INVALIDATION_TIMEOUT, operation).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!("failed to delete cache key {cache_key}: {error}");
            }
            Err(_) => {
                tracing::warn!(
                    "timed out deleting cache key {cache_key} after {}ms",
                    CACHE_INVALIDATION_TIMEOUT.as_millis()
                );
            }
        }
    }
}

pub fn peer_cache_key(namespace: &str, workspace_name: &str, peer_name: &str) -> String {
    format!("{namespace}:v2:workspace:{workspace_name}:peer:{peer_name}")
}

pub fn session_cache_key(namespace: &str, workspace_name: &str, session_name: &str) -> String {
    format!("{namespace}:v2:workspace:{workspace_name}:session:{session_name}")
}

pub fn default_cache_url() -> &'static str {
    DEFAULT_CACHE_URL
}

pub fn default_cache_namespace() -> &'static str {
    DEFAULT_CACHE_NAMESPACE
}

pub fn redis_client_url(value: &str) -> &str {
    value
        .trim()
        .split_once('?')
        .map_or_else(|| value.trim(), |(url, _)| url.trim())
}
