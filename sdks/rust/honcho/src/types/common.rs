//! Common types shared across domain modules.

/// JSON value alias for metadata and configuration fields.
pub type JsonValue = serde_json::Value;

/// Metadata map type.
pub type Metadata = serde_json::Map<String, serde_json::Value>;
