#![allow(clippy::print_stderr)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

use honcho_ai::Honcho;

fn unique_suffix() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or_else(|_| "0".to_string(), |d| d.as_millis().to_string())
}

pub fn api_base_url() -> String {
    env::var("HONCHO_API_URL").unwrap_or_else(|_| "http://localhost:8000".to_string())
}

pub fn api_key() -> Option<String> {
    env::var("HONCHO_API_KEY").ok()
}

pub fn unique_workspace_id() -> String {
    format!("rust-int-test-{}", unique_suffix())
}

pub async fn try_client() -> Option<Honcho> {
    let base_url = api_base_url();
    let ws_id = unique_workspace_id();
    let key = api_key();

    let params = match key {
        Some(k) => Honcho::builder()
            .base_url(&base_url)
            .api_key(k)
            .workspace_id(&ws_id)
            .build(),
        None => Honcho::builder()
            .base_url(&base_url)
            .workspace_id(&ws_id)
            .build(),
    };

    let client = Honcho::from_params(params).ok()?;

    match client.force_ensure().await {
        Ok(()) => Some(client),
        Err(e) => {
            eprintln!("skipping integration test: could not connect to server: {e}");
            None
        }
    }
}
