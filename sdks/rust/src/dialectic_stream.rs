//! Stream adapter that accumulates dialectic content for `final_response()`.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;

use crate::error::Result;

/// Stream adapter that accumulates content from a dialectic SSE stream.
///
/// Wraps any `Stream<Item = Result<String>>` and builds a `final_response()`
/// string from all yielded chunks. The stream is pass-through — callers still
/// receive each chunk individually while the adapter silently accumulates.
pub struct DialecticStream<S> {
    inner: S,
    accumulated: String,
    complete: bool,
}

impl<S> DialecticStream<S>
where
    S: Stream<Item = Result<String>> + Unpin,
{
    /// Wrap a stream, accumulating all successful content chunks.
    pub fn new(stream: S) -> Self {
        Self {
            inner: stream,
            accumulated: String::new(),
            complete: false,
        }
    }

    /// Returns all content accumulated so far (partial if stream is still in progress).
    #[must_use]
    pub fn final_response(&self) -> &str {
        &self.accumulated
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
                self.accumulated.push_str(&content);
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
mod tests {
    #![allow(clippy::unwrap_used)]

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
        assert_eq!(stream.final_response(), "hello world");
    }

    #[tokio::test]
    async fn dialectic_stream_is_complete_after_done() {
        let chunks = vec![ok_chunk("a"), ok_chunk("b")];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);

        assert!(!stream.is_complete());

        while stream.next().await.is_some() {}

        assert!(stream.is_complete());
        assert_eq!(stream.final_response(), "ab");
    }

    #[tokio::test]
    async fn dialectic_stream_final_response_before_completion_returns_partial() {
        let chunks = vec![ok_chunk("first"), ok_chunk(" second")];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, "first");
        assert_eq!(stream.final_response(), "first");
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
        assert_eq!(stream.final_response(), "ok");
    }

    #[tokio::test]
    async fn dialectic_stream_empty_input() {
        let chunks: Vec<Result<String>> = vec![];
        let inner = futures_util::stream::iter(chunks);
        let mut stream = DialecticStream::new(inner);

        assert!(stream.next().await.is_none());
        assert!(stream.is_complete());
        assert_eq!(stream.final_response(), "");
    }
}
