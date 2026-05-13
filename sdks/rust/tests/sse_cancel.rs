//! A.3 — SSE cancel-safety and malformed-data tests.
//!
//! Test 1: Dropping the stream mid-read causes a TCP disconnect visible to
//!         the wiremock mockserver.
//! Test 2: Malformed JSON mid-stream never panics; valid content around it is
//!         still yielded.

#![allow(clippy::unwrap_used)]

use std::pin::Pin;
use std::time::Duration;

use futures_util::StreamExt;
use honcho_ai::error::HonchoError;
use honcho_ai::http::sse::parse_sse_stream;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn sse_chunk(data: &str) -> String {
    format!("data: {data}\n\n")
}

// ═══════════════════════════════════════════════════════════════════════
// Test 1: cancel-safety — drop stream, assert TCP disconnect
// ═══════════════════════════════════════════════════════════════════════
#[tokio::test]
async fn drop_stream_causes_tcp_disconnect() {
    let server = MockServer::start().await;

    let first_chunk = sse_chunk(r#"{"delta":{"content":"chunk1"}}"#);
    let _second_chunk = sse_chunk(r#"{"delta":{"content":"chunk2"}}"#);

    let body = format!("{first_chunk}{{slow}}");

    Mock::given(method("POST"))
        .and(path("/v1/test/sse"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body.as_bytes(), "text/event-stream"))
        .expect(1)
        .mount(&server)
        .await;

    let resp = reqwest::Client::new()
        .post(format!("{}{}", server.uri(), "/v1/test/sse"))
        .header("accept", "text/event-stream")
        .send()
        .await
        .unwrap();

    let byte_stream = resp.bytes_stream();
    let mut s: Pin<Box<dyn futures_util::Stream<Item = Result<String, HonchoError>> + Send>> =
        Box::pin(parse_sse_stream(byte_stream));

    let first = s.next().await.expect("should yield first chunk").unwrap();
    assert_eq!(first, "chunk1");

    drop(s);

    tokio::time::sleep(Duration::from_millis(200)).await;

    server.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: malformed JSON mid-stream does not panic
// ═══════════════════════════════════════════════════════════════════════
#[tokio::test]
async fn malformed_json_mid_stream_no_panic() {
    let server = MockServer::start().await;

    let chunk1 = sse_chunk(r#"{"delta":{"content":"good_before"}}"#);
    let bad = "data: {not valid json!!!\n\n".to_string();
    let chunk3 = sse_chunk(r#"{"delta":{"content":"good_after"}}"#);
    let body = format!("{chunk1}{bad}{chunk3}");

    Mock::given(method("POST"))
        .and(path("/v1/test/sse"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body.as_bytes(), "text/event-stream"))
        .mount(&server)
        .await;

    let resp = reqwest::Client::new()
        .post(format!("{}{}", server.uri(), "/v1/test/sse"))
        .header("accept", "text/event-stream")
        .send()
        .await
        .unwrap();

    let s = parse_sse_stream(resp.bytes_stream());
    let results: Vec<Result<String, HonchoError>> = s.collect().await;

    let ok_results: Vec<String> = results
        .into_iter()
        .filter_map(std::result::Result::ok)
        .collect();

    assert!(
        ok_results.contains(&"good_before".to_string()),
        "should yield content before malformed JSON"
    );
    assert!(
        ok_results.contains(&"good_after".to_string()),
        "should yield content after malformed JSON"
    );
    assert_eq!(ok_results.len(), 2, "only valid chunks should appear");
}
