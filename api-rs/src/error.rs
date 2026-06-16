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
    #[error("Feature is disabled")]
    Disabled,
    #[error("Rust write routes are disabled")]
    WriteDisabled,
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Authentication(String),
    #[error("{0}")]
    Validation(String),
    #[error("request validation failed")]
    RequestValidation(Value),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::Auth(_) => StatusCode::UNAUTHORIZED,
            Self::Authentication(_) => StatusCode::UNAUTHORIZED,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Filter(_) | Self::Validation(_) | Self::RequestValidation(_) => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
            Self::MissingPool | Self::Database(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Disabled | Self::WriteDisabled => StatusCode::METHOD_NOT_ALLOWED,
        };
        let detail = match self {
            Self::RequestValidation(detail) => detail,
            other => Value::String(other.to_string()),
        };
        (status, Json(json!({ "detail": detail }))).into_response()
    }
}
