#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::time::Duration;

use chrono::{TimeZone, Utc};
use honcho_ai::error::{from_response, parse_error_body, parse_retry_after, HonchoError};
use pretty_assertions::assert_eq;
use reqwest::header::{HeaderMap, HeaderValue};
use rstest::rstest;
use static_assertions::assert_impl_all;

#[rstest]
#[case(400, "bad_request")]
#[case(401, "authentication_error")]
#[case(403, "permission_denied")]
#[case(404, "not_found")]
#[case(409, "conflict")]
#[case(422, "unprocessable_entity")]
fn status_maps_to_variant(#[case] status: u16, #[case] expected_code: &str) {
    let status = reqwest::StatusCode::from_u16(status).unwrap();
    let headers = HeaderMap::new();
    let body = bytes::Bytes::from(r#"{"message":"test error"}"#);
    let now = Utc::now();

    let err = from_response(status, &headers, &body, now);

    assert_eq!(err.code(), expected_code);
    match status.as_u16() {
        400 => assert!(matches!(err, HonchoError::BadRequest { .. })),
        401 => assert!(matches!(err, HonchoError::Authentication { .. })),
        403 => assert!(matches!(err, HonchoError::PermissionDenied { .. })),
        404 => assert!(matches!(err, HonchoError::NotFound { .. })),
        409 => assert!(matches!(err, HonchoError::Conflict { .. })),
        422 => assert!(matches!(err, HonchoError::UnprocessableEntity { .. })),
        _ => panic!("unexpected status"),
    }
}

#[rstest]
#[case(500)]
#[case(502)]
#[case(503)]
#[case(504)]
fn server_5xx_maps_to_server_with_status(#[case] status: u16) {
    let status = reqwest::StatusCode::from_u16(status).unwrap();
    let headers = HeaderMap::new();
    let body = bytes::Bytes::from("internal server error");
    let now = Utc::now();

    let err = from_response(status, &headers, &body, now);

    assert!(matches!(
        err,
        HonchoError::Server {
            status: s,
            ..
        } if s == status.as_u16()
    ));
    assert_eq!(err.code(), "server_error");
}

#[test]
fn rate_limit_429_parses_retry_after_seconds() {
    let status = reqwest::StatusCode::TOO_MANY_REQUESTS;
    let mut headers = HeaderMap::new();
    headers.insert("retry-after", HeaderValue::from_static("7"));
    let body = bytes::Bytes::from(r#"{"message":"rate limited"}"#);
    let now = Utc::now();

    let err = from_response(status, &headers, &body, now);

    match err {
        HonchoError::RateLimit { retry_after, .. } => {
            assert_eq!(retry_after, Some(Duration::from_secs(7)));
        }
        _ => panic!("expected RateLimit, got {err:?}"),
    }
}

#[test]
fn rate_limit_429_parses_retry_after_http_date() {
    let status = reqwest::StatusCode::TOO_MANY_REQUESTS;
    let mut headers = HeaderMap::new();
    let now = Utc.with_ymd_and_hms(2026, 10, 21, 7, 27, 55).unwrap();
    headers.insert(
        "retry-after",
        HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
    );
    let body = bytes::Bytes::from(r#"{"message":"rate limited"}"#);

    let err = from_response(status, &headers, &body, now);

    match err {
        HonchoError::RateLimit {
            retry_after: Some(dur),
            ..
        } => {
            let secs = dur.as_secs_f64();
            assert!(secs >= 4.9 && secs <= 5.1, "expected ~5s, got {secs}s");
        }
        _ => panic!("expected RateLimit with retry_after, got {err:?}"),
    }
}

#[test]
fn rate_limit_429_without_retry_after_is_none() {
    let status = reqwest::StatusCode::TOO_MANY_REQUESTS;
    let headers = HeaderMap::new();
    let body = bytes::Bytes::from(r#"{"message":"rate limited"}"#);
    let now = Utc::now();

    let err = from_response(status, &headers, &body, now);

    match err {
        HonchoError::RateLimit {
            retry_after: None, ..
        } => {}
        _ => panic!("expected RateLimit with None retry_after, got {err:?}"),
    }
}

#[test]
fn retry_after_with_garbage_returns_none() {
    let mut headers = HeaderMap::new();
    headers.insert("retry-after", HeaderValue::from_static("not-a-valid-value"));
    let now = Utc::now();

    let result = parse_retry_after(headers.get("retry-after").unwrap(), now);
    assert!(result.is_none());
}

#[test]
fn error_body_extracts_message_field_priority() {
    let (msg, _) = parse_error_body(r#"{"detail":"d","message":"m","error":"e"}"#.as_bytes());
    assert_eq!(msg, "d");

    let (msg, _) = parse_error_body(r#"{"message":"m","error":"e"}"#.as_bytes());
    assert_eq!(msg, "m");

    let (msg, _) = parse_error_body(r#"{"error":"e"}"#.as_bytes());
    assert_eq!(msg, "e");

    let (msg, _) = parse_error_body(r#""plain string""#.as_bytes());
    assert_eq!(msg, "plain string");
}

#[rstest]
#[case(400)]
#[case(401)]
#[case(404)]
#[case(500)]
fn display_includes_status_and_message(#[case] status: u16) {
    let status = reqwest::StatusCode::from_u16(status).unwrap();
    let headers = HeaderMap::new();
    let body = bytes::Bytes::from(r#"{"message":"something went wrong"}"#);
    let now = Utc::now();

    let err = from_response(status, &headers, &body, now);
    let display = format!("{err}");

    assert!(
        display.contains(&status.as_u16().to_string()),
        "display should contain status code"
    );
    assert!(
        display.contains("something went wrong"),
        "display should contain message"
    );
}

#[test]
fn error_code_is_stable_string() {
    let codes = [
        (
            "bad_request",
            HonchoError::BadRequest {
                message: String::new(),
                body: None,
            },
        ),
        (
            "authentication_error",
            HonchoError::Authentication {
                message: String::new(),
            },
        ),
        (
            "permission_denied",
            HonchoError::PermissionDenied {
                message: String::new(),
            },
        ),
        (
            "not_found",
            HonchoError::NotFound {
                message: String::new(),
            },
        ),
        (
            "conflict",
            HonchoError::Conflict {
                message: String::new(),
                body: None,
            },
        ),
        (
            "unprocessable_entity",
            HonchoError::UnprocessableEntity {
                message: String::new(),
                body: None,
            },
        ),
        (
            "rate_limit_exceeded",
            HonchoError::RateLimit {
                message: String::new(),
                retry_after: None,
            },
        ),
        (
            "server_error",
            HonchoError::Server {
                status: 500,
                message: String::new(),
            },
        ),
        (
            "timeout",
            HonchoError::Timeout {
                message: String::new(),
            },
        ),
        (
            "connection_error",
            HonchoError::Connection {
                message: String::new(),
            },
        ),
    ];

    for (expected_code, err) in codes {
        assert_eq!(err.code(), expected_code, "mismatch for {expected_code}");
    }
}

use std::error::Error;

#[tokio::test]
async fn source_chain_for_transport_and_io_and_decode() {
    let transport_err: HonchoError = reqwest::Client::new()
        .get("http://0.0.0.0:1")
        .send()
        .await
        .unwrap_err()
        .into();
    assert!(transport_err.source().is_some());

    let json_err = serde_json::from_str::<Vec<i32>>("{}").unwrap_err();
    let decode_err = HonchoError::Decode {
        path: "root".to_string(),
        source: json_err,
    };
    assert!(decode_err.source().is_some());

    let bad_request = HonchoError::BadRequest {
        message: String::new(),
        body: None,
    };
    assert!(bad_request.source().is_none());
}

#[test]
fn error_bounds() {
    assert_impl_all!(HonchoError: Send, Sync, std::error::Error);
}
