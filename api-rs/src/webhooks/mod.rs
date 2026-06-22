//! Port of `src/webhooks/webhook_delivery.py`: deliver a webhook event to every
//! endpoint configured for a workspace, signing the body with HMAC-SHA256.
//!
//! Delivery is best-effort — every error (missing secret, transport failure,
//! non-2xx response) is logged and swallowed, matching Python's broad `except`.

use std::future::Future;
use std::pin::Pin;

use hmac::{Hmac, Mac};
use serde_json::{Map, Value};
use sha2::Sha256;
use sqlx::PgPool;

use crate::db;
use crate::deriver::payload::WebhookPayload;

type HmacSha256 = Hmac<Sha256>;

/// Generic HTTP POST used to deliver a webhook. Returns the response status code,
/// or `Err` on a transport-level failure. Object-safe (boxed future) so the
/// worker can hold an `Arc<dyn WebhookSender>` like it does for the emitter.
pub trait WebhookSender: Send + Sync {
    fn post<'a>(
        &'a self,
        url: &'a str,
        body: &'a str,
        headers: &'a [(String, String)],
    ) -> Pin<Box<dyn Future<Output = Result<u16, String>> + Send + 'a>>;
}

/// Production [`WebhookSender`] over a `reqwest::Client` with a 30s timeout
/// (matching Python's `httpx.AsyncClient(timeout=30.0)`).
pub struct ReqwestWebhookSender {
    client: reqwest::Client,
}

impl ReqwestWebhookSender {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self { client }
    }
}

impl Default for ReqwestWebhookSender {
    fn default() -> Self {
        Self::new()
    }
}

impl WebhookSender for ReqwestWebhookSender {
    fn post<'a>(
        &'a self,
        url: &'a str,
        body: &'a str,
        headers: &'a [(String, String)],
    ) -> Pin<Box<dyn Future<Output = Result<u16, String>> + Send + 'a>> {
        Box::pin(async move {
            let mut request = self.client.post(url).body(body.to_string());
            for (name, value) in headers {
                request = request.header(name, value);
            }
            let response = request.send().await.map_err(|error| error.to_string())?;
            Ok(response.status().as_u16())
        })
    }
}

/// Recursively serialize `value` with sorted object keys and no whitespace —
/// the wire format of Python `json.dumps(..., separators=(",", ":"),
/// sort_keys=True)`. Crucially the signature is computed over *these exact bytes*
/// and the same bytes are POSTed, so verification on the receiver matches.
pub fn sorted_compact_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = String::from("{");
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                // serde_json::to_string on a &str applies JSON string escaping.
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                out.push_str(&sorted_compact_json(&map[*key]));
            }
            out.push('}');
            out
        }
        Value::Array(items) => {
            let mut out = String::from("[");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&sorted_compact_json(item));
            }
            out.push(']');
            out
        }
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
    }
}

/// Build the signed event envelope `{"data", "timestamp", "type"}` (sorted keys)
/// for `payload` at `timestamp_iso`.
pub fn build_event_json(event_type: &str, data: &Value, timestamp_iso: &str) -> String {
    let mut envelope = Map::new();
    envelope.insert("type".to_string(), Value::String(event_type.to_string()));
    envelope.insert("data".to_string(), data.clone());
    envelope.insert("timestamp".to_string(), Value::String(timestamp_iso.to_string()));
    sorted_compact_json(&Value::Object(envelope))
}

/// HMAC-SHA256 hex signature of `payload` keyed by `secret` (Python
/// `_generate_webhook_signature`). HMAC accepts a key of any length.
pub fn generate_webhook_signature(secret: &str, payload: &str) -> String {
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(payload.as_bytes());
    let bytes = mac.finalize().into_bytes();
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Port of `deliver_webhook`: look up the workspace's endpoints, build + sign the
/// event envelope, and POST it to each. Best-effort: a DB lookup failure or a
/// missing secret logs and returns without delivering; per-endpoint failures are
/// logged and do not abort the rest. `now_iso` is the envelope timestamp;
/// `webhook_secret` is `settings.WEBHOOK.SECRET`.
pub async fn deliver_webhook<S: WebhookSender + ?Sized>(
    pool: &PgPool,
    sender: &S,
    payload: &WebhookPayload,
    workspace_name: &str,
    webhook_secret: Option<&str>,
    now_iso: &str,
) {
    let urls = match db::get_webhook_endpoint_urls(pool, workspace_name).await {
        Ok(urls) => urls,
        Err(error) => {
            tracing::error!(%workspace_name, "error fetching webhook endpoints: {error}");
            return;
        }
    };
    if urls.is_empty() {
        tracing::debug!(%workspace_name, "no webhook endpoints, skipping");
        return;
    }

    let event_json = build_event_json(&payload.event_type, &payload.data, now_iso);
    let signature = match webhook_secret.filter(|secret| !secret.is_empty()) {
        Some(secret) => generate_webhook_signature(secret, &event_json),
        None => {
            tracing::error!("WEBHOOK_SECRET not set - cannot sign webhook");
            return;
        }
    };

    let headers = vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("X-Honcho-Signature".to_string(), signature),
    ];

    // Python gathers concurrently; we deliver sequentially (best-effort
    // background work, small endpoint counts). Errors are logged, never raised.
    for url in &urls {
        match sender.post(url, &event_json, &headers).await {
            Ok(status) if (200..300).contains(&status) => {
                tracing::debug!(%url, event_type = %payload.event_type, "delivered webhook");
            }
            Ok(status) => {
                tracing::error!(%url, event_type = %payload.event_type, status, "webhook delivery failed");
            }
            Err(error) => {
                tracing::error!(%url, event_type = %payload.event_type, "webhook delivery error: {error}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sorted_compact_json_sorts_recursively_and_strips_space() {
        let value = json!({
            "b": 1,
            "a": {"z": [3, 2], "y": "x"},
        });
        assert_eq!(sorted_compact_json(&value), r#"{"a":{"y":"x","z":[3,2]},"b":1}"#);
    }

    #[test]
    fn build_event_json_orders_envelope_keys() {
        let json = build_event_json(
            "queue.empty",
            &json!({"session_id": "s1", "count": 0}),
            "2026-06-21T12:00:00Z",
        );
        // Envelope keys sorted: data, timestamp, type; nested data keys sorted too.
        assert_eq!(
            json,
            r#"{"data":{"count":0,"session_id":"s1"},"timestamp":"2026-06-21T12:00:00Z","type":"queue.empty"}"#
        );
    }

    #[test]
    fn signature_matches_known_hmac_sha256() {
        // Golden HMAC-SHA256("secret", "message") hex (RFC-verified value).
        let sig = generate_webhook_signature("secret", "message");
        assert_eq!(
            sig,
            "8b5f48702995c1598c573db1e21866a9b825d4a794d169d7060a03605796360b"
        );
        assert_eq!(sig.len(), 64);
    }
}
