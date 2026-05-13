use std::any::TypeId;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use reqwest::Method;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use serde::Serialize;
use serde::de::DeserializeOwned;
use url::Url;

use crate::error::{self, HonchoError, Result};
use crate::http::decode;

const DEFAULT_MAX_RETRIES: u32 = 2;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(500);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

struct Inner {
    client: reqwest::Client,
    base_url: Url,
    max_retries: u32,
    default_query: Vec<(String, String)>,
    timeout: Duration,
    default_headers: HeaderMap,
}

#[derive(Clone)]
#[doc(hidden)]
pub struct HttpClient {
    inner: Arc<Inner>,
}

#[derive(bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
#[doc(hidden)]
pub struct HttpClientParams {
    base_url: String,
    api_key: Option<String>,
    #[builder(default = DEFAULT_MAX_RETRIES)]
    max_retries: u32,
    #[builder(default)]
    default_headers: HeaderMap,
    #[builder(default)]
    default_query: Vec<(String, String)>,
    #[builder(default = DEFAULT_TIMEOUT)]
    timeout: Duration,
    http_client: Option<reqwest::Client>,
}

impl HttpClient {
    pub fn builder() -> HttpClientParamsBuilder {
        HttpClientParams::builder()
    }

    #[expect(dead_code)]
    pub(crate) fn base_url_hint(&self) -> String {
        self.inner.base_url.to_string()
    }

    pub fn from_params(params: HttpClientParams) -> Result<Self> {
        let mut base_url = Url::parse(&params.base_url)
            .map_err(|e| HonchoError::Configuration(format!("invalid base_url: {e}")))?;

        let path = base_url.path().to_owned();
        if path.ends_with('/') && path.len() > 1 {
            let trimmed = path.trim_end_matches('/');
            base_url.set_path(trimmed);
        }

        let version = env!("CARGO_PKG_VERSION");
        let mut client_headers = HeaderMap::new();
        client_headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&format!("honcho-ai/{version}"))
                .unwrap_or_else(|_| HeaderValue::from_static("honcho-ai/unknown")),
        );
        client_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        client_headers.insert(ACCEPT, HeaderValue::from_static("application/json"));

        if let Some(ref key) = params.api_key {
            let val = HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|e| HonchoError::Configuration(format!("invalid api_key: {e}")))?;
            client_headers.insert(AUTHORIZATION, val);
        }

        for (name, value) in params.default_headers {
            if let Some(n) = name {
                let _ = client_headers.insert(n, value);
            }
        }

        let (client, extra_headers) = match params.http_client {
            Some(c) => (c, client_headers),
            None => (
                reqwest::ClientBuilder::new()
                    .default_headers(client_headers)
                    .timeout(params.timeout)
                    .build()
                    .map_err(|e| {
                        HonchoError::Configuration(format!("failed to build HTTP client: {e}"))
                    })?,
                HeaderMap::new(),
            ),
        };

        Ok(Self {
            inner: Arc::new(Inner {
                client,
                base_url,
                max_retries: params.max_retries,
                default_query: params.default_query,
                timeout: params.timeout,
                default_headers: extra_headers,
            }),
        })
    }

    pub(crate) async fn request<TBody, TResp>(
        &self,
        method: Method,
        path: &str,
        body: Option<&TBody>,
        query: &[(&str, &str)],
    ) -> Result<TResp>
    where
        TBody: Serialize + ?Sized,
        TResp: DeserializeOwned + 'static,
    {
        let url = self
            .inner
            .base_url
            .join(path)
            .map_err(|e| HonchoError::Configuration(format!("failed to join URL path: {e}")))?;

        let merged_query: Vec<(&str, &str)> = self
            .inner
            .default_query
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .chain(query.iter().copied())
            .collect();

        let mut attempt = 0u32;
        loop {
            let mut req_builder = self
                .inner
                .client
                .request(method.clone(), url.clone())
                .headers(self.inner.default_headers.clone())
                .query(&merged_query)
                .timeout(self.inner.timeout);

            if let Some(b) = body {
                req_builder = req_builder.json(b);
            }

            let response = match req_builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    let error = if e.is_timeout() {
                        HonchoError::Timeout {
                            message: e.to_string(),
                        }
                    } else if e.is_connect() {
                        HonchoError::Connection {
                            message: e.to_string(),
                        }
                    } else {
                        HonchoError::Transport(e)
                    };

                    let should_retry = !matches!(error, HonchoError::Transport(_))
                        && attempt < self.inner.max_retries;

                    if should_retry {
                        attempt += 1;
                        tokio::time::sleep(delay_for_attempt(attempt)).await;
                        continue;
                    }

                    return Err(error);
                }
            };

            let status = response.status();

            if status.is_success() {
                return self.handle_success_response(response).await;
            }

            let headers = response.headers().clone();
            let Ok(body_bytes) = response.bytes().await else {
                let msg = format!(
                    "request failed with status {} (could not read response body)",
                    status.as_u16()
                );
                return Err(if status.is_server_error() {
                    HonchoError::Server {
                        status: status.as_u16(),
                        message: msg,
                    }
                } else {
                    HonchoError::Client {
                        status: status.as_u16(),
                        message: msg,
                    }
                });
            };
            let api_error = error::from_response(status, &headers, &body_bytes, Utc::now());

            let is_retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504);

            if is_retryable && attempt < self.inner.max_retries {
                attempt += 1;
                let retry_after = headers
                    .get("retry-after")
                    .and_then(|v| error::parse_retry_after(v, Utc::now()));
                let delay = retry_after.unwrap_or_else(|| delay_for_attempt(attempt));
                tokio::time::sleep(delay).await;
                continue;
            }

            return Err(api_error);
        }
    }

    async fn handle_success_response<TResp: DeserializeOwned + 'static>(
        &self,
        response: reqwest::Response,
    ) -> Result<TResp> {
        let bytes = response.bytes().await.map_err(|e| {
            if e.is_timeout() {
                HonchoError::Timeout {
                    message: e.to_string(),
                }
            } else {
                HonchoError::Transport(e)
            }
        })?;

        let is_unit = TypeId::of::<TResp>() == TypeId::of::<()>();

        if is_unit {
            return serde_json::from_value::<TResp>(serde_json::Value::Null).map_err(|e| {
                HonchoError::Decode {
                    path: String::new(),
                    source: e,
                }
            });
        }

        if bytes.is_empty() {
            return Err(HonchoError::Decode {
                path: String::new(),
                source: serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "empty response body",
                )),
            });
        }

        decode::deserialize_with_path(&bytes)
    }

    pub(crate) async fn get<TResp: DeserializeOwned + 'static>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<TResp> {
        self.request::<(), TResp>(Method::GET, path, None, query)
            .await
    }

    pub(crate) async fn post<TBody: Serialize + ?Sized, TResp: DeserializeOwned + 'static>(
        &self,
        path: &str,
        body: Option<&TBody>,
        query: &[(&str, &str)],
    ) -> Result<TResp> {
        self.request(Method::POST, path, body, query).await
    }

    pub(crate) async fn put<TBody: Serialize + ?Sized, TResp: DeserializeOwned + 'static>(
        &self,
        path: &str,
        body: Option<&TBody>,
        query: &[(&str, &str)],
    ) -> Result<TResp> {
        self.request(Method::PUT, path, body, query).await
    }

    pub(crate) async fn patch<TBody: Serialize + ?Sized, TResp: DeserializeOwned + 'static>(
        &self,
        path: &str,
        body: Option<&TBody>,
        query: &[(&str, &str)],
    ) -> Result<TResp> {
        self.request(Method::PATCH, path, body, query).await
    }

    pub(crate) async fn delete<TResp: DeserializeOwned + 'static>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<TResp> {
        self.request::<(), TResp>(Method::DELETE, path, None, query)
            .await
    }

    pub(crate) async fn request_multipart<TResp: DeserializeOwned + 'static>(
        &self,
        method: Method,
        path: &str,
        form: reqwest::multipart::Form,
        query: &[(String, String)],
    ) -> Result<TResp> {
        let url = self
            .inner
            .base_url
            .join(path)
            .map_err(|e| HonchoError::Configuration(format!("failed to join URL path: {e}")))?;

        let merged_query: Vec<(&str, &str)> = self
            .inner
            .default_query
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .chain(query.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .collect();

        let req_builder = self
            .inner
            .client
            .request(method, url)
            .headers(self.inner.default_headers.clone())
            .query(&merged_query)
            .timeout(self.inner.timeout)
            .multipart(form);

        let response = req_builder.send().await.map_err(|e| {
            if e.is_timeout() {
                HonchoError::Timeout {
                    message: e.to_string(),
                }
            } else if e.is_connect() {
                HonchoError::Connection {
                    message: e.to_string(),
                }
            } else {
                HonchoError::Transport(e)
            }
        })?;

        let status = response.status();

        if status.is_success() {
            return self.handle_success_response(response).await;
        }

        let headers = response.headers().clone();
        let body_bytes = response.bytes().await.unwrap_or_default();
        Err(error::from_response(
            status,
            &headers,
            &body_bytes,
            Utc::now(),
        ))
    }

    pub(crate) async fn request_streaming(
        &self,
        method: Method,
        path: &str,
        body: Option<&serde_json::Value>,
        query: &[(&str, &str)],
    ) -> Result<reqwest::Response> {
        let url = self
            .inner
            .base_url
            .join(path)
            .map_err(|e| HonchoError::Configuration(format!("failed to join URL path: {e}")))?;

        let merged_query: Vec<(&str, &str)> = self
            .inner
            .default_query
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .chain(query.iter().copied())
            .collect();

        let mut req_builder = self
            .inner
            .client
            .request(method, url)
            .headers(self.inner.default_headers.clone())
            .header(ACCEPT, HeaderValue::from_static("text/event-stream"))
            .query(&merged_query)
            .timeout(self.inner.timeout);

        if let Some(b) = body {
            req_builder = req_builder.json(b);
        }

        let response = match req_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                return Err(if e.is_timeout() {
                    HonchoError::Timeout {
                        message: e.to_string(),
                    }
                } else if e.is_connect() {
                    HonchoError::Connection {
                        message: e.to_string(),
                    }
                } else {
                    HonchoError::Transport(e)
                });
            }
        };

        let status = response.status();

        if status.is_success() {
            return Ok(response);
        }

        let headers = response.headers().clone();
        let body_bytes = response.bytes().await.unwrap_or_default();
        Err(error::from_response(
            status,
            &headers,
            &body_bytes,
            Utc::now(),
        ))
    }

    pub(crate) async fn post_multipart<TResp: DeserializeOwned + 'static>(
        &self,
        path: &str,
        form: reqwest::multipart::Form,
        query: &[(String, String)],
    ) -> Result<TResp> {
        self.request_multipart(Method::POST, path, form, query)
            .await
    }
}

#[doc(hidden)]
#[must_use]
pub fn delay_for_attempt(attempt: u32) -> Duration {
    let shift = attempt.min(31);
    INITIAL_RETRY_DELAY
        .saturating_mul(1u32 << shift)
        .min(MAX_RETRY_DELAY)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unnecessary_wraps,
        clippy::needless_pass_by_value,
        clippy::unused_async
    )]

    use super::*;
    use crate::error::HonchoError;
    use crate::http::routes;
    use crate::types::peer::Peer;
    use crate::types::workspace::Workspace;
    use std::time::Duration;
    use wiremock::matchers::{body_json, header, header_exists, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_client(server: &MockServer) -> HttpClient {
        HttpClient::from_params(HttpClient::builder().base_url(server.uri()).build()).unwrap()
    }

    fn make_client_with_key(server: &MockServer, api_key: &str) -> HttpClient {
        HttpClient::from_params(
            HttpClient::builder()
                .base_url(server.uri())
                .api_key(api_key.to_string())
                .build(),
        )
        .unwrap()
    }

    fn peer_json() -> serde_json::Value {
        serde_json::json!({
            "id": "p1",
            "workspace_id": "ws1",
            "created_at": "2025-01-15T10:30:00Z",
            "metadata": {},
            "configuration": {}
        })
    }

    fn workspace_json() -> serde_json::Value {
        serde_json::json!({
            "id": "ws_abc123",
            "metadata": {},
            "configuration": {},
            "created_at": "2025-01-15T10:30:00Z"
        })
    }

    // ── Builder ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn builder_creates_client_with_valid_url() {
        let server = MockServer::start().await;
        let result = HttpClient::from_params(HttpClient::builder().base_url(server.uri()).build());
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn builder_rejects_invalid_url() {
        let result = HttpClient::from_params(
            HttpClient::builder()
                .base_url("not a url".to_string())
                .build(),
        );
        let Err(err) = result else {
            panic!("expected Configuration error")
        };
        assert!(matches!(err, HonchoError::Configuration(_)));
    }

    #[tokio::test]
    async fn builder_strips_trailing_slash() {
        let server = MockServer::start().await;
        let base = format!("{}/", server.uri());
        let client = HttpClient::from_params(HttpClient::builder().base_url(base).build()).unwrap();

        Mock::given(method("GET"))
            .and(path("/v3/test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        let result: Peer = client.get("/v3/test", &[]).await.unwrap();
        assert_eq!(result.id, "p1");
    }

    #[tokio::test]
    async fn builder_default_max_retries_is_2() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .expect(3)
            .mount(&server)
            .await;

        assert!(client.get::<Peer>("/v3/test", &[]).await.is_err());
    }

    #[tokio::test]
    async fn builder_custom_http_client_preserved() {
        let server = MockServer::start().await;
        let custom = reqwest::Client::new();
        let client = HttpClient::from_params(
            HttpClient::builder()
                .base_url(server.uri())
                .http_client(custom)
                .build(),
        )
        .unwrap();

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        let result: Peer = client.get("/v3/test", &[]).await.unwrap();
        assert_eq!(result.id, "p1");
    }

    // ── Helper smoke ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn get_sends_get_request() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        let result: Peer = client.get("/v3/test", &[]).await.unwrap();
        assert_eq!(result.id, "p1");
    }

    #[tokio::test]
    async fn post_sends_post_with_body() {
        let server = MockServer::start().await;
        let client = make_client(&server);
        let body = serde_json::json!({"name": "test"});

        Mock::given(method("POST"))
            .and(body_json(&body))
            .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json()))
            .mount(&server)
            .await;

        let result: Workspace = client.post("/v3/test", Some(&body), &[]).await.unwrap();
        assert_eq!(result.id, "ws_abc123");
    }

    #[tokio::test]
    async fn put_sends_put_with_body() {
        let server = MockServer::start().await;
        let client = make_client(&server);
        let body = serde_json::json!({"name": "updated"});

        Mock::given(method("PUT"))
            .and(body_json(&body))
            .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json()))
            .mount(&server)
            .await;

        let result: Workspace = client.put("/v3/test", Some(&body), &[]).await.unwrap();
        assert_eq!(result.id, "ws_abc123");
    }

    #[tokio::test]
    async fn patch_sends_patch_with_body() {
        let server = MockServer::start().await;
        let client = make_client(&server);
        let body = serde_json::json!({"name": "patched"});

        Mock::given(method("PATCH"))
            .and(body_json(&body))
            .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json()))
            .mount(&server)
            .await;

        let result: Workspace = client.patch("/v3/test", Some(&body), &[]).await.unwrap();
        assert_eq!(result.id, "ws_abc123");
    }

    #[tokio::test]
    async fn delete_returns_unit() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("DELETE"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let _: () = client.delete("/v3/test", &[]).await.unwrap();
    }

    // ── Auth & headers ───────────────────────────────────────────────────

    #[tokio::test]
    async fn sends_bearer_auth_when_api_key_set() {
        let server = MockServer::start().await;
        let client = make_client_with_key(&server, "test-key");

        Mock::given(method("GET"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        let _: Peer = client.get("/v3/test", &[]).await.unwrap();
    }

    #[tokio::test]
    async fn sends_user_agent() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .and(header_exists("user-agent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        let _: Peer = client.get("/v3/test", &[]).await.unwrap();
    }

    #[tokio::test]
    async fn sends_content_type() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        let _: Peer = client.get("/v3/test", &[]).await.unwrap();
    }

    #[tokio::test]
    async fn sends_accept() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .and(header("accept", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        let _: Peer = client.get("/v3/test", &[]).await.unwrap();
    }

    #[tokio::test]
    async fn default_query_params_sent() {
        let server = MockServer::start().await;
        let client = HttpClient::from_params(
            HttpClient::builder()
                .base_url(server.uri())
                .default_query(vec![("foo".to_string(), "bar".to_string())])
                .build(),
        )
        .unwrap();

        Mock::given(method("GET"))
            .and(query_param("foo", "bar"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        let _: Peer = client.get("/v3/test", &[]).await.unwrap();
    }

    // ── Unit type / empty body ───────────────────────────────────────────

    #[tokio::test]
    async fn unit_type_for_204_no_content() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("DELETE"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let _: () = client.delete("/v3/test", &[]).await.unwrap();
    }

    #[tokio::test]
    async fn unit_type_for_200_empty_body() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let _: () = client.get("/v3/test", &[]).await.unwrap();
    }

    // ── Timeout ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn request_times_out() {
        let server = MockServer::start().await;
        let client = HttpClient::from_params(
            HttpClient::builder()
                .base_url(server.uri())
                .timeout(Duration::from_millis(100))
                .build(),
        )
        .unwrap();

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
            .mount(&server)
            .await;

        let err = client.get::<Peer>("/v3/test", &[]).await.unwrap_err();
        assert!(
            matches!(err, HonchoError::Timeout { .. }),
            "expected Timeout, got {err:?}"
        );
    }

    // ── Error mapping ────────────────────────────────────────────────────

    #[rstest::rstest]
    #[case(400, "bad_request")]
    #[case(401, "authentication_error")]
    #[case(403, "permission_denied")]
    #[case(404, "not_found")]
    #[case(409, "conflict")]
    #[case(422, "unprocessable_entity")]
    #[tokio::test]
    async fn non_retryable_4xx_returns_error(
        #[case] status: u16,
        #[case] expected_code: &'static str,
    ) {
        let server = MockServer::start().await;
        let client = HttpClient::from_params(
            HttpClient::builder()
                .base_url(server.uri())
                .max_retries(5)
                .build(),
        )
        .unwrap();

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(status))
            .expect(1)
            .mount(&server)
            .await;

        let err = client.get::<Peer>("/v3/test", &[]).await.unwrap_err();
        assert_eq!(err.code(), expected_code);
    }

    #[tokio::test]
    async fn server_error_without_retries() {
        let server = MockServer::start().await;
        let client = HttpClient::from_params(
            HttpClient::builder()
                .base_url(server.uri())
                .max_retries(0)
                .build(),
        )
        .unwrap();

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;

        let err = client.get::<Peer>("/v3/test", &[]).await.unwrap_err();
        assert!(
            matches!(err, HonchoError::Server { status: 503, .. }),
            "expected Server(503), got {err:?}"
        );
    }

    #[tokio::test]
    async fn malformed_json_returns_decode_error() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"bad": true})),
            )
            .mount(&server)
            .await;

        let err = client.get::<Peer>("/v3/test", &[]).await.unwrap_err();
        assert!(
            matches!(err, HonchoError::Decode { .. }),
            "expected Decode, got {err:?}"
        );
    }

    // ── Retry ────────────────────────────────────────────────────────────

    #[rstest::rstest]
    #[case(429)]
    #[case(500)]
    #[case(502)]
    #[case(503)]
    #[case(504)]
    #[tokio::test]
    async fn retries_on_retryable_status(#[case] status: u16) {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(status))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let result: Peer = client.get("/v3/test", &[]).await.unwrap();
        assert_eq!(result.id, "p1");
    }

    #[rstest::rstest]
    #[case(400)]
    #[case(401)]
    #[case(403)]
    #[case(404)]
    #[case(409)]
    #[case(422)]
    #[tokio::test]
    async fn no_retry_non_retryable_4xx(#[case] status: u16) {
        let server = MockServer::start().await;
        let client = HttpClient::from_params(
            HttpClient::builder()
                .base_url(server.uri())
                .max_retries(5)
                .build(),
        )
        .unwrap();

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(status))
            .expect(1)
            .mount(&server)
            .await;

        assert!(client.get::<Peer>("/v3/test", &[]).await.is_err());
    }

    #[tokio::test]
    async fn max_retries_zero_no_retry() {
        let server = MockServer::start().await;
        let client = HttpClient::from_params(
            HttpClient::builder()
                .base_url(server.uri())
                .max_retries(0)
                .build(),
        )
        .unwrap();

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;

        assert!(client.get::<Peer>("/v3/test", &[]).await.is_err());
    }

    #[tokio::test]
    async fn max_retries_3_means_4_total_attempts() {
        let server = MockServer::start().await;
        let client = HttpClient::from_params(
            HttpClient::builder()
                .base_url(server.uri())
                .max_retries(3)
                .build(),
        )
        .unwrap();

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .expect(4)
            .mount(&server)
            .await;

        assert!(client.get::<Peer>("/v3/test", &[]).await.is_err());
    }

    // ── Backoff ──────────────────────────────────────────────────────────

    #[test]
    fn delay_for_attempt_sequence() {
        assert_eq!(delay_for_attempt(0), Duration::from_millis(500));
        assert_eq!(delay_for_attempt(1), Duration::from_secs(1));
        assert_eq!(delay_for_attempt(2), Duration::from_secs(2));
        assert_eq!(delay_for_attempt(3), Duration::from_secs(4));
    }

    #[test]
    fn delay_for_attempt_capped_at_30s() {
        assert_eq!(delay_for_attempt(100), Duration::from_secs(30));
        assert_eq!(delay_for_attempt(31), Duration::from_secs(30));
    }

    #[tokio::test]
    async fn retry_after_zero_retries_immediately() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let result: Peer = client.get("/v3/test", &[]).await.unwrap();
        assert_eq!(result.id, "p1");
    }

    #[tokio::test]
    async fn retry_after_seconds_used() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(peer_json()))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "1"))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let result: Peer = client.get("/v3/test", &[]).await.unwrap();
        assert_eq!(result.id, "p1");
    }

    // ── E2E ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn e2e_get_peer_typed() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        let fixture = serde_json::json!({
            "id": "p1",
            "workspace_id": "ws1",
            "created_at": "2025-01-15T10:30:00Z",
            "metadata": {"role": "admin", "version": 2},
            "configuration": {"language": "en", "features": {"beta": true}}
        });

        Mock::given(method("GET"))
            .and(path("/v3/workspaces/ws1/peers/alice"))
            .respond_with(ResponseTemplate::new(200).set_body_json(fixture))
            .mount(&server)
            .await;

        let result: Peer = client
            .get(&routes::peer("ws1", "alice"), &[])
            .await
            .unwrap();
        assert_eq!(result.id, "p1");
        assert_eq!(result.workspace_id, "ws1");
    }

    #[tokio::test]
    async fn e2e_post_workspace_typed() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        let fixture = serde_json::json!({
            "id": "ws_abc123",
            "metadata": {"env": "production", "team": "core"},
            "configuration": {"reasoning": {"enabled": true}},
            "created_at": "2025-01-15T10:30:00Z"
        });

        let create_body = serde_json::json!({"id": "ws_abc123"});

        Mock::given(method("POST"))
            .and(path("/v3/workspaces"))
            .and(body_json(&create_body))
            .respond_with(ResponseTemplate::new(200).set_body_json(fixture))
            .mount(&server)
            .await;

        let result: Workspace = client
            .post(&routes::workspaces(), Some(&create_body), &[])
            .await
            .unwrap();
        assert_eq!(result.id, "ws_abc123");
    }

    // ── Multipart ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn post_multipart_sends_form_data() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("POST"))
            .and(path("/v3/upload"))
            .and(wiremock::matchers::body_string_contains("field_value"))
            .and(wiremock::matchers::body_string_contains("test upload"))
            .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json()))
            .mount(&server)
            .await;

        let form = reqwest::multipart::Form::new()
            .text("field_name", "field_value")
            .text("description", "test upload");

        let result: Workspace = client
            .post_multipart("/v3/upload", form, &[])
            .await
            .unwrap();
        assert_eq!(result.id, "ws_abc123");
    }

    #[tokio::test]
    async fn post_multipart_with_query_params() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("POST"))
            .and(path("/v3/upload"))
            .and(query_param("workspace_id", "ws1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(workspace_json()))
            .mount(&server)
            .await;

        let form = reqwest::multipart::Form::new().text("key", "value");

        let result: Workspace = client
            .post_multipart(
                "/v3/upload",
                form,
                &[("workspace_id".to_string(), "ws1".to_string())],
            )
            .await
            .unwrap();
        assert_eq!(result.id, "ws_abc123");
    }

    #[tokio::test]
    async fn post_multipart_server_error_returns_error() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let form = reqwest::multipart::Form::new().text("key", "value");

        let err = client
            .post_multipart::<Workspace>("/v3/upload", form, &[])
            .await
            .unwrap_err();
        assert!(
            matches!(err, HonchoError::Server { status: 503, .. }),
            "expected Server(503), got {err:?}"
        );
    }

    // ── Streaming ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn request_streaming_returns_response_on_200() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        let sse_body = "data: hello\n\ndata: world\n\n";

        Mock::given(method("POST"))
            .and(header("accept", "text/event-stream"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let response = client
            .request_streaming(
                Method::POST,
                "/v3/workspaces/ws1/peers/alice/chat",
                None,
                &[],
            )
            .await
            .unwrap();

        assert!(response.status().is_success());
        let bytes = response.bytes().await.unwrap();
        assert_eq!(bytes, sse_body.as_bytes());
    }

    #[tokio::test]
    async fn request_streaming_error_status_maps_to_honcho_error() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(422))
            .mount(&server)
            .await;

        let err = client
            .request_streaming(
                Method::POST,
                "/v3/workspaces/ws1/peers/alice/chat",
                None,
                &[],
            )
            .await
            .unwrap_err();

        assert!(
            matches!(err, HonchoError::UnprocessableEntity { .. }),
            "expected UnprocessableEntity, got {err:?}"
        );
    }

    #[tokio::test]
    async fn request_streaming_sends_accept_event_stream_header() {
        let server = MockServer::start().await;
        let client = make_client(&server);

        Mock::given(method("POST"))
            .and(header("accept", "text/event-stream"))
            .respond_with(ResponseTemplate::new(200).set_body_string("data: ok\n\n"))
            .expect(1)
            .mount(&server)
            .await;

        let _response = client
            .request_streaming(
                Method::POST,
                "/v3/workspaces/ws1/peers/alice/chat",
                None,
                &[],
            )
            .await
            .unwrap();
    }
}
