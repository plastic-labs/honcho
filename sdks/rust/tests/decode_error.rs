#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use honcho_ai::error::HonchoError;
use honcho_ai::http::decode::deserialize_with_path;

use serde::Deserialize;

#[derive(Deserialize)]
struct RequiresString {
    #[allow(dead_code)] // Field intentionally unused — tests decoder path
    id: String,
}

#[test]
fn decode_malformed_json_returns_decode_with_path() {
    let json = r#"{"id":null}"#;
    let result: Result<RequiresString, _> = deserialize_with_path(json.as_bytes());

    match result {
        Err(HonchoError::Decode { path, .. }) => {
            assert!(
                path.contains("id") || path.contains("root"),
                "path should contain field name, got: {path}"
            );
        }
        Err(other) => panic!("expected Decode error, got {other:?}"),
        Ok(_) => panic!("expected error, got success"),
    }
}
