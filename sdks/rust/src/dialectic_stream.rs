//! Stream adapter that accumulates dialectic content for `final_response()`.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;

use crate::error::Result;

/// A single delta in a streaming dialectic response.
///
/// Corresponds to `DialecticStreamDelta` in the Python SDK.
#[allow(dead_code)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct DialecticStreamDelta {
    pub(crate) content: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct DialecticStreamChunk {
    pub(crate) delta: DialecticStreamDelta,
    #[serde(default)]
    pub(crate) done: bool,
}

/// The fully-accumulated content of a completed dialectic stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalResponse {
    content: String,
}

impl FinalResponse {
    /// Create a new `FinalResponse` from the accumulated content.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    /// Access the accumulated response text.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }
}

impl std::fmt::Display for FinalResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.content)
    }
}

/// Stream adapter that accumulates content from a dialectic SSE stream.
///
/// Wraps any `Stream<Item = Result<String>>` and builds a [`FinalResponse`]
/// from all yielded chunks. The stream is pass-through — callers still
/// receive each chunk individually while the adapter silently accumulates.
pub struct DialecticStream<S> {
    inner: S,
    final_response: FinalResponse,
    complete: bool,
}

#[allow(clippy::missing_fields_in_debug)]
impl<S> std::fmt::Debug for DialecticStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DialecticStream")
            .field("is_complete", &self.complete)
            .field("content_len", &self.final_response.content.len())
            .finish()
    }
}

impl<S> DialecticStream<S>
where
    S: Stream<Item = Result<String>> + Unpin,
{
    /// Wrap a stream, accumulating all successful content chunks.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    pub fn new(stream: S) -> Self {
        Self {
            inner: stream,
            final_response: FinalResponse {
                content: String::with_capacity(1024),
            },
            complete: false,
        }
    }

    /// Returns the accumulated response so far (partial if stream is still in progress).
    #[must_use]
    pub fn final_response(&self) -> &FinalResponse {
        &self.final_response
    }

    /// `true` once the inner stream has returned `None` (end-of-stream).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.complete
    }
}

impl<S> Stream for DialecticStream<S>
where
    S: Stream<Item = Result<String>> + Unpin,
{
    type Item = Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(content))) => {
                self.final_response.content.push_str(&content);
                Poll::Ready(Some(Ok(content)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                self.complete = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unnecessary_wraps,
    clippy::needless_pass_by_value,
    clippy::unused_async
)]
mod tests {
    use futures_util::StreamExt;

    use super::*;
    use crate::error::HonchoError;

    fn ok_chunk(s: &str) -> Result<String> {
        Ok(s.to_string())
    }

    #[tokio::test]
    async fn dialectic_stream_accumulates_during_iteration() {
        let chunks = vec![ok_chunk("hello"), ok_chunk(" "), ok_chunk("world")];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);

        let mut collected = Vec::new();
        while let Some(item) = stream.next().await {
            collected.push(item.unwrap());
        }

        assert_eq!(collected, vec!["hello", " ", "world"]);
        assert_eq!(stream.final_response().content(), "hello world");
    }

    #[tokio::test]
    async fn dialectic_stream_is_complete_after_done() {
        let chunks = vec![ok_chunk("a"), ok_chunk("b")];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);

        assert!(!stream.is_complete());

        while stream.next().await.is_some() {}

        assert!(stream.is_complete());
        assert_eq!(stream.final_response().content(), "ab");
    }

    #[tokio::test]
    async fn dialectic_stream_final_response_before_completion_returns_partial() {
        let chunks = vec![ok_chunk("first"), ok_chunk(" second")];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, "first");
        assert_eq!(stream.final_response().content(), "first");
        assert!(!stream.is_complete());
    }

    #[tokio::test]
    async fn dialectic_stream_propagates_errors() {
        let chunks: Vec<Result<String>> = vec![
            ok_chunk("ok"),
            Err(HonchoError::Connection {
                message: "boom".to_string(),
            }),
        ];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, "ok");

        let err = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(err, HonchoError::Connection { .. }));
        assert_eq!(stream.final_response().content(), "ok");
    }

    #[tokio::test]
    async fn dialectic_stream_empty_input() {
        let chunks: Vec<Result<String>> = vec![];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);

        assert!(stream.next().await.is_none());
        assert!(stream.is_complete());
        assert_eq!(stream.final_response().content(), "");
    }

    #[tokio::test]
    async fn dialectic_stream_final_response_display() {
        let chunks = vec![ok_chunk("hello")];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);
        while stream.next().await.is_some() {}

        assert_eq!(stream.final_response().to_string(), "hello");
    }

    #[tokio::test]
    async fn dialectic_stream_final_response_eq() {
        let chunks = vec![ok_chunk("abc")];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);
        while stream.next().await.is_some() {}

        let expected = FinalResponse {
            content: "abc".to_string(),
        };
        assert_eq!(stream.final_response(), &expected);
    }
}
