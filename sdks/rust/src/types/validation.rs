//! `FastAPI` validation error types.
//!
//! These types model the [`HTTPValidationError`] and [`ValidationError`] schemas
//! that `FastAPI` returns for request validation failures (HTTP 422). They are
//! standalone types usable alongside [`crate::error::HonchoError`] for
//! inspecting validation details from the `body` field of
//! [`crate::error::HonchoError::UnprocessableEntity`].

use serde::{Deserialize, Serialize};

/// A single validation error from `FastAPI`.
///
/// Corresponds to the `ValidationError` schema in the `OpenAPI` spec.
///
/// Required fields: `loc`, `msg`, `type`.
/// Optional fields: `input`, `ctx`.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationError {
    /// Location of the error as a path through the request structure.
    ///
    /// Elements are either strings (field/object names) or integers (array indices).
    pub loc: Vec<LocationSegment>,
    /// Human-readable error message.
    pub msg: String,
    /// Machine-readable error type identifier (e.g. `"value_error.missing"`).
    #[serde(rename = "type")]
    pub error_type: String,
    /// The input value that failed validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    /// Optional context providing additional details about the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx: Option<serde_json::Map<String, serde_json::Value>>,
}

/// A segment within a validation error location path.
///
/// `FastAPI` location paths contain either string keys (field names) or integer
/// indices (array positions). This enum captures both.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LocationSegment {
    /// A string field or key name.
    String(String),
    /// An integer array index.
    Integer(i64),
}

/// HTTP-level wrapper for `FastAPI` validation errors.
///
/// Corresponds to the `HTTPValidationError` schema in the `OpenAPI` spec.
/// Returned as the response body for HTTP 422 responses.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HTTPValidationError {
    /// List of individual validation errors.
    pub detail: Vec<ValidationError>,
}
