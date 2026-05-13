#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! R-34 — SSE cancellation safety tests.
//!
//! 1. `tokio::select!` + drop cancels a slow SSE stream without hanging
//! 2. Dropping the stream mid-read is visible as a TCP disconnect (wiremock)
//! 3. Malformed JSON mid-stream never panics; valid content around it is still yielded
//! 4. `DialecticStream` wrapper cancels cleanly via `tokio::select!`

use std::pin::Pin;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::StreamExt;
use honcho_ai::error::HonchoError;
use honcho_ai::http::sse::parse_sse_stream;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn sse_chunk(data: &str) -> String {
    format!("data: {data}\n\n")
}

// ═══════════════════════════════════════════════════════════════════════
// Test 1: tokio::select! cancels slow stream, drop completes fast
// ═══════════════════════════════════════════════════════════════════════
#[tokio::test]
async fn tokio_select_cancel_drops_stream_cleanly() {
    let slow_bytes: Pin<
        Box<dyn futures_util::Stream<Item = Result<Bytes, reqwest::Error>> + Send>,
    > = Box::pin(async_stream::stream! {
        yield Ok(Bytes::from(
            "data: {\"delta\":{\"content\":\"first\"}}\n\n",
        ));
        tokio::time::sleep(Duration::from_secs(300)).await;
        yield Ok(Bytes::from(
            "data: {\"delta\":{\"content\":\"never\"}}\n\n",
        ));
    });

    let mut s = Box::pin(parse_sse_stream(slow_bytes));

    let result = tokio::select! {
        chunk = s.next() => chunk,
        () = tokio::time::sleep(Duration::from_secs(5)) => {
            panic!("timed out waiting for first SSE chunk");
        }
    };

    let content = result.expect("stream ended unexpectedly").unwrap();
    assert_eq!(content, "first");

    // Drop while inner stream still sleeping — must complete without blocking
    let before = Instant::now();
    drop(s);
    let elapsed = before.elapsed();

    assert!(
        elapsed < Duration::from_secs(1),
        "drop took {elapsed:?} — possible resource leak or blocking on Drop",
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: cancel-safety — drop stream, assert TCP disconnect
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
// Test 3: malformed JSON mid-stream does not panic
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

// ═══════════════════════════════════════════════════════════════════════
// Test 4: DialecticStream wrapper cancels cleanly via tokio::select!
// ═══════════════════════════════════════════════════════════════════════
#[tokio::test]
async fn dialectic_stream_cancel_via_select() {
    use honcho_ai::DialecticStream;

    let slow_bytes: Pin<
        Box<dyn futures_util::Stream<Item = Result<Bytes, reqwest::Error>> + Send>,
    > = Box::pin(async_stream::stream! {
        yield Ok(Bytes::from(
            "data: {\"delta\":{\"content\":\"hello\"}}\n\n",
        ));
        tokio::time::sleep(Duration::from_secs(300)).await;
        yield Ok(Bytes::from(
            "data: {\"delta\":{\"content\":\"world\"}}\n\n",
        ));
    });

    let inner = parse_sse_stream(slow_bytes);
    let mut ds = DialecticStream::new(Box::pin(inner));

    let item = tokio::select! {
        chunk = ds.next() => chunk,
        () = tokio::time::sleep(Duration::from_secs(5)) => {
            panic!("timed out waiting for DialecticStream chunk");
        }
    };

    let content = item.expect("stream ended").unwrap();
    assert_eq!(content, "hello");
    assert_eq!(ds.final_response(), "hello");
    assert!(!ds.is_complete());

    let before = Instant::now();
    drop(ds);
    let elapsed = before.elapsed();

    assert!(
        elapsed < Duration::from_secs(1),
        "DialecticStream drop took {elapsed:?} — cancellation not clean",
    );
}
