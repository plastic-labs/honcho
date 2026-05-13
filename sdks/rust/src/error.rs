//! Error types for the Honcho SDK.

use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use httpdate::parse_http_date;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::StatusCode;

/// Error type for all Honcho SDK operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HonchoError {
    /// 400 Bad Request
    #[error("Honcho API error: HTTP 400 {message}")]
    BadRequest {
        /// Error message from the API.
        message: String,
        /// Full response body if available.
        body: Option<serde_json::Value>,
    },
    /// 401 Authentication Error
    #[error("Honcho API error: HTTP 401 {message}")]
    Authentication {
        /// Error message.
        message: String,
    },
    /// 403 Permission Denied
    #[error("Honcho API error: HTTP 403 {message}")]
    PermissionDenied {
        /// Error message.
        message: String,
    },
    /// 404 Not Found
    #[error("Honcho API error: HTTP 404 {message}")]
    NotFound {
        /// Error message.
        message: String,
    },
    /// 409 Conflict
    #[error("Honcho API error: HTTP 409 {message}")]
    Conflict {
        /// Error message.
        message: String,
        /// Full response body if available.
        body: Option<serde_json::Value>,
    },
    /// 422 Unprocessable Entity
    #[error("Honcho API error: HTTP 422 {message}")]
    UnprocessableEntity {
        /// Error message.
        message: String,
        /// Full response body if available.
        body: Option<serde_json::Value>,
    },
    /// 429 Rate Limit Exceeded
    #[error("Honcho API error: HTTP 429 {message}")]
    RateLimit {
        /// Error message.
        message: String,
        /// Suggested wait time from Retry-After header.
        retry_after: Option<Duration>,
    },
    /// 4xx Client Error (unmapped status codes like 405, 408, 413, etc.)
    #[error("Honcho API error: HTTP {status} {message}")]
    Client {
        /// HTTP status code.
        status: u16,
        /// Error message.
        message: String,
    },
    /// 5xx Server Error
    #[error("Honcho API error: HTTP {status} {message}")]
    Server {
        /// HTTP status code.
        status: u16,
        /// Error message.
        message: String,
    },
    /// Request timed out.
    #[error("Request timed out: {message}")]
    Timeout {
        /// Error message.
        message: String,
    },
    /// Connection error.
    #[error("Connection error: {message}")]
    Connection {
        /// Error message.
        message: String,
    },
    /// HTTP transport error from reqwest.
    #[error(transparent)]
    Transport(#[from] reqwest::Error),
    /// Failed to decode response body.
    #[error("Failed to decode response at {path}: {source}")]
    Decode {
        /// JSON path where decoding failed.
        path: String,
        /// The underlying serde error.
        #[source]
        source: serde_json::Error,
    },
    /// IO error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Configuration(String),
    /// Validation error (e.g. duplicate inputs, invalid arguments).
    #[error("Validation error: {0}")]
    Validation(String),
}

impl HonchoError {
    /// Returns a stable error code string for pattern matching.
    ///
    /// Parity with Python SDK's `error.code` field.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::BadRequest { .. } => "bad_request",
            Self::Authentication { .. } => "authentication_error",
            Self::PermissionDenied { .. } => "permission_denied",
            Self::NotFound { .. } => "not_found",
            Self::Conflict { .. } => "conflict",
            Self::UnprocessableEntity { .. } => "unprocessable_entity",
            Self::RateLimit { .. } => "rate_limit_exceeded",
            Self::Client { .. } => "client_error",
            Self::Server { .. } => "server_error",
            Self::Timeout { .. } => "timeout",
            Self::Connection { .. } => "connection_error",
            Self::Transport(_) => "transport_error",
            Self::Decode { .. } => "decode_error",
            Self::Io(_) => "io_error",
            Self::Configuration(_) => "configuration_error",
            Self::Validation(_) => "validation_error",
        }
    }
}

/// Alias for `Result<T, HonchoError>`.
pub type Result<T> = std::result::Result<T, HonchoError>;

/// Parse an error response body, extracting message and body.
///
/// Tries to extract `detail`, `message`, or `error` fields in order (`FastAPI` convention).
#[must_use]
pub fn parse_error_body(body: &[u8]) -> (String, Option<serde_json::Value>) {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
        let msg = String::from_utf8_lossy(body).to_string();
        return (msg, None);
    };

    let full_body = Some(value.clone());

    if let Some(obj) = value.as_object() {
        if let Some(detail) = obj.get("detail").and_then(|v| v.as_str()) {
            return (detail.to_string(), full_body);
        }
        if let Some(message) = obj.get("message").and_then(|v| v.as_str()) {
            return (message.to_string(), full_body);
        }
        if let Some(error) = obj.get("error").and_then(|v| v.as_str()) {
            return (error.to_string(), full_body);
        }
        return (value.to_string(), full_body);
    }

    if let Some(s) = value.as_str() {
        return (s.to_string(), full_body);
    }

    (value.to_string(), full_body)
}

/// Parse a Retry-After header value.
///
/// Accepts either seconds (f64) or HTTP-date format.
/// Returns `None` if the value cannot be parsed.
/// Clamps negative durations to zero (parity with Python's `max(0.0, ...)`).
pub fn parse_retry_after(value: &HeaderValue, now: DateTime<Utc>) -> Option<Duration> {
    let s = value.to_str().ok()?;

    if let Ok(secs) = s.parse::<f64>() {
        return Some(Duration::from_secs_f64(secs.max(0.0)));
    }

    let target = parse_http_date(s).ok()?;
    let now_systime: SystemTime = now.into();
    let diff = target.duration_since(now_systime).ok()?;
    Some(diff)
}

/// Construct a `HonchoError` from an HTTP response.
pub fn from_response(
    status: StatusCode,
    headers: &HeaderMap,
    body: &bytes::Bytes,
    now: DateTime<Utc>,
) -> HonchoError {
    let (message, body_value) = parse_error_body(body);

    match status.as_u16() {
        400 => HonchoError::BadRequest {
            message,
            body: body_value,
        },
        401 => HonchoError::Authentication { message },
        403 => HonchoError::PermissionDenied { message },
        404 => HonchoError::NotFound { message },
        409 => HonchoError::Conflict {
            message,
            body: body_value,
        },
        422 => HonchoError::UnprocessableEntity {
            message,
            body: body_value,
        },
        429 => {
            let retry_after = headers
                .get("retry-after")
                .and_then(|v| parse_retry_after(v, now));
            HonchoError::RateLimit {
                message,
                retry_after,
            }
        }
        s if s >= 500 => HonchoError::Server { status: s, message },
        s if (400..500).contains(&s) => HonchoError::Client { status: s, message },
        _ => HonchoError::Decode {
            path: String::new(),
            source: serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("from_response called with non-error status {status}"),
            )),
        },
    }
}
