//! Response deserialization with path tracking.

use serde::de::DeserializeOwned;

use crate::error::{HonchoError, Result};

/// Deserialize bytes into T with `serde_path_to_error` tracking.
///
/// Returns `HonchoError::Decode` on failure, including the JSON path
/// where the error occurred.
pub fn deserialize_with_path<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let mut de = serde_json::Deserializer::from_slice(bytes);
    serde_path_to_error::deserialize(&mut de).map_err(|err| HonchoError::Decode {
        path: err.path().to_string(),
        source: err.into_inner(),
    })
}
