//! Message wrapper — construction, getters, custom Debug/Display.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::http::client::HttpClient;
use crate::types::message::MessageResponse;

pub(crate) struct MessageInner {
    #[expect(dead_code)]
    http: HttpClient,
    workspace_id: String,
    id: String,
    content: String,
    peer_id: String,
    session_id: String,
    metadata: HashMap<String, Value>,
    created_at: DateTime<Utc>,
    token_count: u64,
}

/// An enriched message in a Honcho workspace.
///
/// Wraps the raw [`MessageResponse`] with workspace context and provides
/// convenient field accessors.
#[derive(Clone)]
pub struct Message {
    inner: Arc<MessageInner>,
}

impl Message {
    pub(crate) fn from_raw(http: HttpClient, workspace_id: String, resp: MessageResponse) -> Self {
        Self {
            inner: Arc::new(MessageInner {
                http,
                workspace_id,
                id: resp.id,
                content: resp.content,
                peer_id: resp.peer_id,
                session_id: resp.session_id,
                metadata: resp.metadata,
                created_at: resp.created_at,
                token_count: resp.token_count,
            }),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn from_response(honcho: &crate::Honcho, resp: MessageResponse) -> Self {
        Self {
            inner: Arc::new(MessageInner {
                http: honcho.http().clone(),
                workspace_id: honcho.workspace_id().to_owned(),
                id: resp.id,
                content: resp.content,
                peer_id: resp.peer_id,
                session_id: resp.session_id,
                metadata: resp.metadata,
                created_at: resp.created_at,
                token_count: resp.token_count,
            }),
        }
    }

    /// The message's unique identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.inner.id
    }

    /// The message content text.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.inner.content
    }

    /// ID of the peer that authored this message.
    #[must_use]
    pub fn peer_id(&self) -> &str {
        &self.inner.peer_id
    }

    /// ID of the session this message belongs to.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    /// Arbitrary key-value metadata attached to the message.
    #[must_use]
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.inner.metadata
    }

    /// When this message was created.
    #[must_use]
    pub fn created_at(&self) -> &DateTime<Utc> {
        &self.inner.created_at
    }

    /// Token count for the message content.
    #[must_use]
    pub fn token_count(&self) -> u64 {
        self.inner.token_count
    }

    /// The workspace this message belongs to.
    #[must_use]
    pub fn workspace_id(&self) -> &str {
        &self.inner.workspace_id
    }
}

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let truncated = if self.inner.content.len() > 50 {
            let end = self
                .inner
                .content
                .char_indices()
                .nth(50)
                .map_or(self.inner.content.len(), |(i, _)| i);
            format!("{}...", &self.inner.content[..end])
        } else {
            self.inner.content.clone()
        };
        f.debug_struct("Message")
            .field("id", &self.inner.id)
            .field("content", &truncated)
            .field("peer_id", &self.inner.peer_id)
            .field("session_id", &self.inner.session_id)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.content)
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
    use static_assertions::assert_impl_all;

    use super::*;

    assert_impl_all!(Message: Send, Sync, Clone, fmt::Debug, fmt::Display);

    fn fake_response() -> MessageResponse {
        MessageResponse {
            id: "msg_1".to_owned(),
            content: "hello world".to_owned(),
            peer_id: "peer_a".to_owned(),
            session_id: "sess_x".to_owned(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            workspace_id: "ws_1".to_owned(),
            token_count: 3,
        }
    }

    #[test]
    fn from_response_maps_fields() {
        let resp = fake_response();
        let honcho = crate::Honcho::new("http://localhost:9999", "ws_1").unwrap();
        let msg = Message::from_response(&honcho, resp);
        assert_eq!(msg.id(), "msg_1");
        assert_eq!(msg.content(), "hello world");
        assert_eq!(msg.peer_id(), "peer_a");
        assert_eq!(msg.session_id(), "sess_x");
        assert_eq!(msg.workspace_id(), "ws_1");
        assert_eq!(msg.token_count(), 3);
        assert!(msg.metadata().is_empty());
    }

    #[test]
    fn debug_truncates_long_content() {
        let mut resp = fake_response();
        resp.content = "a".repeat(80);
        let honcho = crate::Honcho::new("http://localhost:9999", "ws_1").unwrap();
        let msg = Message::from_response(&honcho, resp);
        let dbg = format!("{msg:?}");
        assert!(dbg.contains("aaa..."));
        assert!(!dbg.contains(&"a".repeat(80)));
    }

    #[test]
    fn debug_short_content_not_truncated() {
        let resp = fake_response();
        let honcho = crate::Honcho::new("http://localhost:9999", "ws_1").unwrap();
        let msg = Message::from_response(&honcho, resp);
        let dbg = format!("{msg:?}");
        assert!(dbg.contains("hello world"));
        assert!(!dbg.contains("..."));
    }

    #[test]
    fn display_returns_full_content() {
        let mut resp = fake_response();
        resp.content = "a".repeat(80);
        let honcho = crate::Honcho::new("http://localhost:9999", "ws_1").unwrap();
        let msg = Message::from_response(&honcho, resp);
        assert_eq!(format!("{msg}"), "a".repeat(80));
    }

    #[test]
    fn debug_truncation_multibyte_utf8() {
        let mut resp = fake_response();
        resp.content = "\u{4e00}".repeat(60);
        let honcho = crate::Honcho::new("http://localhost:9999", "ws_1").unwrap();
        let msg = Message::from_response(&honcho, resp);
        let dbg = format!("{msg:?}");
        assert!(dbg.contains("..."));
    }
}
