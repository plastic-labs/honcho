//! HTTP transport seam for the LLM backends.
//!
//! The provider backends build a request body (deterministic, tested) and parse
//! a response body (deterministic, tested); the only non-deterministic step is
//! the POST in between. [`LlmHttp`] abstracts that single step so the backends'
//! `complete()` flow is exercisable end-to-end with a mock transport — no
//! network — while [`ReqwestHttp`] provides the real implementation.

use std::pin::Pin;

use futures::{Stream, StreamExt};
use serde_json::Value;

/// A boxed stream of decoded text fragments of an SSE response body.
pub type TextStream = Pin<Box<dyn Stream<Item = Result<String, LlmHttpError>> + Send>>;

/// A transport error from an LLM HTTP call.
#[derive(Debug, Clone)]
pub enum LlmHttpError {
    /// The request configuration was invalid before any provider call was made.
    Validation(String),
    /// The request never produced a response (connection/timeout/DNS).
    Transport(String),
    /// A non-2xx HTTP status, with the raw response body for diagnostics.
    Status { status: u16, body: String },
    /// The 2xx response body was not valid JSON.
    Decode(String),
}

impl std::fmt::Display for LlmHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmHttpError::Validation(message) => write!(f, "validation error: {message}"),
            LlmHttpError::Transport(message) => write!(f, "transport error: {message}"),
            LlmHttpError::Status { status, body } => {
                write!(f, "http status {status}: {body}")
            }
            LlmHttpError::Decode(message) => write!(f, "decode error: {message}"),
        }
    }
}

impl std::error::Error for LlmHttpError {}

/// Resolved per-call credentials for a provider request, mirroring the subset of
/// `credentials::resolve_credentials` the transport needs: the API key and an
/// optional base-URL override. A `None` `base_url` means the provider's default
/// endpoint (each backend supplies its own constant).
#[derive(Debug, Clone)]
pub struct Credentials {
    pub api_key: String,
    pub base_url: Option<String>,
}

impl Credentials {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
        }
    }

    pub fn with_base_url(api_key: impl Into<String>, base_url: Option<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url,
        }
    }

    /// The effective base URL, falling back to `default` and trimming a trailing
    /// slash so backends can append a leading-slash path unconditionally.
    pub fn effective_base_url<'a>(&'a self, default: &'a str) -> &'a str {
        self.base_url
            .as_deref()
            .unwrap_or(default)
            .trim_end_matches('/')
    }
}

/// One JSON POST. Backends call this with a provider URL, headers, and the body
/// produced by their `build_request`/`build_config`, and parse the returned
/// [`Value`] with their `parse_response`. Implemented for real by
/// [`ReqwestHttp`] and by mocks in tests.
pub trait LlmHttp {
    fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
    ) -> impl std::future::Future<Output = Result<Value, LlmHttpError>> + Send;
}

/// The production [`LlmHttp`] backed by `reqwest`.
#[derive(Debug, Clone)]
pub struct ReqwestHttp {
    client: reqwest::Client,
}

impl ReqwestHttp {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

impl Default for ReqwestHttp {
    fn default() -> Self {
        Self::new(reqwest::Client::new())
    }
}

impl LlmHttp for ReqwestHttp {
    async fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
    ) -> Result<Value, LlmHttpError> {
        let mut request = self.client.post(url).json(body);
        for (name, value) in headers {
            request = request.header(name, value);
        }
        let response = request
            .send()
            .await
            .map_err(|error| LlmHttpError::Transport(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmHttpError::Status {
                status: status.as_u16(),
                body,
            });
        }
        response
            .json::<Value>()
            .await
            .map_err(|error| LlmHttpError::Decode(error.to_string()))
    }
}

/// A streaming JSON POST: the response body is returned as a [`TextStream`] of
/// UTF-8 fragments of the SSE wire format (consumed by
/// [`crate::llm::streaming::decode_stream`]). Separate from [`LlmHttp`] so the
/// non-streaming backends and their mocks stay unaffected.
pub trait LlmStreamHttp {
    fn post_json_stream(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
    ) -> impl std::future::Future<Output = Result<TextStream, LlmHttpError>> + Send;
}

impl LlmStreamHttp for ReqwestHttp {
    async fn post_json_stream(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
    ) -> Result<TextStream, LlmHttpError> {
        let mut request = self.client.post(url).json(body);
        for (name, value) in headers {
            request = request.header(name, value);
        }
        let response = request
            .send()
            .await
            .map_err(|error| LlmHttpError::Transport(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmHttpError::Status {
                status: status.as_u16(),
                body,
            });
        }

        // Decode bytes → text, carrying a partial trailing UTF-8 sequence across
        // chunk boundaries so a split multi-byte char isn't corrupted.
        let bytes = Box::pin(response.bytes_stream());
        let text = futures::stream::unfold(
            (bytes, Vec::<u8>::new()),
            |(mut bytes, mut leftover)| async move {
                loop {
                    match bytes.next().await {
                        Some(Ok(chunk)) => {
                            leftover.extend_from_slice(&chunk);
                            let valid_up_to = match std::str::from_utf8(&leftover) {
                                Ok(_) => leftover.len(),
                                Err(error) => error.valid_up_to(),
                            };
                            if valid_up_to == 0 {
                                continue; // need more bytes to complete a char
                            }
                            let text = String::from_utf8_lossy(&leftover[..valid_up_to]).into_owned();
                            leftover.drain(..valid_up_to);
                            return Some((Ok(text), (bytes, leftover)));
                        }
                        Some(Err(error)) => {
                            return Some((
                                Err(LlmHttpError::Transport(error.to_string())),
                                (bytes, leftover),
                            ));
                        }
                        None => return None,
                    }
                }
            },
        );
        Ok(Box::pin(text))
    }
}

#[cfg(test)]
pub(crate) mod mock {
    use super::{LlmHttp, LlmHttpError};
    use serde_json::Value;
    use std::sync::Mutex;

    /// A captured request: `(url, headers, body)`.
    type CapturedRequest = (String, Vec<(String, String)>, Value);

    /// A mock transport that records the last `(url, headers, body)` it received
    /// and returns a canned result, so backend `complete()` flows are testable
    /// without a network.
    pub struct MockHttp {
        result: Result<Value, LlmHttpError>,
        pub captured: Mutex<Option<CapturedRequest>>,
    }

    impl MockHttp {
        pub fn ok(response: Value) -> Self {
            Self {
                result: Ok(response),
                captured: Mutex::new(None),
            }
        }

        pub fn err(error: LlmHttpError) -> Self {
            Self {
                result: Err(error),
                captured: Mutex::new(None),
            }
        }

        pub fn last_url(&self) -> String {
            self.captured.lock().unwrap().as_ref().unwrap().0.clone()
        }

        pub fn last_body(&self) -> Value {
            self.captured.lock().unwrap().as_ref().unwrap().2.clone()
        }

        pub fn last_headers(&self) -> Vec<(String, String)> {
            self.captured.lock().unwrap().as_ref().unwrap().1.clone()
        }
    }

    impl LlmHttp for MockHttp {
        async fn post_json(
            &self,
            url: &str,
            headers: &[(String, String)],
            body: &Value,
        ) -> Result<Value, LlmHttpError> {
            *self.captured.lock().unwrap() = Some((url.to_string(), headers.to_vec(), body.clone()));
            self.result.clone()
        }
    }

    /// A mock streaming transport that records the last request and replays canned
    /// SSE text fragments, so backend `stream()` flows are testable without a
    /// network.
    pub struct MockStreamHttp {
        fragments: Vec<String>,
        pub captured: Mutex<Option<CapturedRequest>>,
    }

    impl MockStreamHttp {
        pub fn new(fragments: Vec<String>) -> Self {
            Self {
                fragments,
                captured: Mutex::new(None),
            }
        }

        pub fn last_url(&self) -> String {
            self.captured.lock().unwrap().as_ref().unwrap().0.clone()
        }

        pub fn last_body(&self) -> Value {
            self.captured.lock().unwrap().as_ref().unwrap().2.clone()
        }

        pub fn last_headers(&self) -> Vec<(String, String)> {
            self.captured.lock().unwrap().as_ref().unwrap().1.clone()
        }
    }

    impl super::LlmStreamHttp for MockStreamHttp {
        async fn post_json_stream(
            &self,
            url: &str,
            headers: &[(String, String)],
            body: &Value,
        ) -> Result<super::TextStream, LlmHttpError> {
            *self.captured.lock().unwrap() = Some((url.to_string(), headers.to_vec(), body.clone()));
            let fragments: Vec<Result<String, LlmHttpError>> =
                self.fragments.iter().cloned().map(Ok).collect();
            Ok(Box::pin(futures::stream::iter(fragments)))
        }
    }
}
