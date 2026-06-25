use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use thiserror::Error;

use crate::auth::AuthError;
use crate::filters::FilterError;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("{0}")]
    Auth(#[from] AuthError),
    #[error("{0}")]
    Filter(#[from] FilterError),
    #[error("database pool is not configured")]
    MissingPool,
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("queue producer error: {0}")]
    Producer(#[from] crate::producer::ProducerError),
    #[error("Feature is disabled")]
    Disabled,
    #[error("Rust write routes are disabled")]
    WriteDisabled,
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Authentication(String),
    #[error("{0}")]
    Validation(String),
    #[error("request validation failed")]
    RequestValidation(Value),
    /// A route path that exists in the Python API but is not yet ported to the
    /// Rust sidecar (e.g. perspective-scoped session context, which needs the
    /// representation/embedding subsystem). Surfaces as 501 so the boundary is
    /// explicit rather than silently returning a wrong-shaped 200.
    #[error("{0}")]
    NotImplemented(String),
    /// A non-`ValueError` embedding-call failure (transport/auth/no-vector).
    /// Mirrors Python letting these propagate out of `search()` as a 500
    /// (only token-limit / dimension `ValueError`s become 422 validation errors).
    #[error("{0}")]
    Embedding(String),
    /// A dialectic completion failure (the agentic tool loop's LLM call failed
    /// after retries). Surfaces as a 500.
    #[error("{0}")]
    Llm(String),
    /// An uploaded file exceeds `MAX_FILE_SIZE` (`FileTooLargeError`, 413).
    #[error("{0}")]
    FileTooLarge(String),
    /// An uploaded file's content type has no extractor (`UnsupportedFileTypeError`,
    /// 415).
    #[error("{0}")]
    UnsupportedFileType(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::Auth(_) => StatusCode::UNAUTHORIZED,
            Self::Authentication(_) => StatusCode::UNAUTHORIZED,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Filter(_) | Self::Validation(_) | Self::RequestValidation(_) => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
            Self::MissingPool
            | Self::Database(_)
            | Self::Producer(_)
            | Self::Embedding(_)
            | Self::Llm(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Disabled | Self::WriteDisabled => StatusCode::METHOD_NOT_ALLOWED,
            Self::NotImplemented(_) => StatusCode::NOT_IMPLEMENTED,
            Self::FileTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UnsupportedFileType(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
        };
        let detail = match self {
            Self::RequestValidation(detail) => detail,
            other => Value::String(other.to_string()),
        };
        (status, Json(json!({ "detail": detail }))).into_response()
    }
}
