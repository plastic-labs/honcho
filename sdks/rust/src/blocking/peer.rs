use std::collections::HashMap;
use std::pin::Pin;

use futures_util::Stream;
use serde_json::Value;

use crate::dialectic_stream::DialecticStream;
use crate::error::Result;
use crate::types::dialectic::{DialecticOptions, ReasoningLevel};
use crate::types::message::MessageSearchOptions;
use crate::types::pagination::Page;
use crate::types::peer::{PeerConfig, PeerContext};
use crate::types::session::{Session, SessionListOptions};

use super::conclusion::ConclusionScope;
use super::iter::{BlockingIter, collect_all_pages};
use super::runtime::block_on;

/// Synchronous wrapper around [`crate::Peer`].
#[derive(Clone)]
pub struct Peer {
    inner: crate::Peer,
}

impl Peer {
    pub(crate) fn new(inner: crate::Peer) -> Self {
        Self { inner }
    }

    /// Peer's unique identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        self.inner.id()
    }

    /// Cached metadata.
    #[must_use]
    pub fn metadata(&self) -> Option<HashMap<String, Value>> {
        self.inner.metadata()
    }

    /// Cached configuration.
    #[must_use]
    pub fn configuration(&self) -> Option<PeerConfig> {
        self.inner.configuration()
    }

    /// Refresh cached state from the server.
    pub fn refresh(&self) -> Result<()> {
        block_on(self.inner.refresh())
    }

    /// Fetch and return metadata, updating the cache.
    pub fn get_metadata(&self) -> Result<HashMap<String, Value>> {
        block_on(self.inner.get_metadata())
    }

    /// Set metadata on the server and update the cache.
    pub fn set_metadata(&self, metadata: HashMap<String, Value>) -> Result<()> {
        block_on(self.inner.set_metadata(metadata))
    }

    /// Fetch and return configuration, updating the cache.
    pub fn get_configuration(&self) -> Result<PeerConfig> {
        block_on(self.inner.get_configuration())
    }

    /// Set configuration on the server and update the cache.
    pub fn set_configuration(&self, config: &PeerConfig) -> Result<()> {
        block_on(self.inner.set_configuration(config))
    }

    /// Fetch configuration as a raw JSON map.
    pub fn get_configuration_raw(&self) -> Result<HashMap<String, Value>> {
        block_on(self.inner.get_configuration_raw())
    }

    /// Set configuration from a raw JSON map.
    pub fn set_configuration_raw(&self, config: HashMap<String, Value>) -> Result<()> {
        block_on(self.inner.set_configuration_raw(config))
    }

    /// Patch-update metadata.
    pub fn update(&self, metadata: HashMap<String, Value>) -> Result<()> {
        block_on(self.inner.update(metadata))
    }

    /// Non-streaming dialectic chat.
    pub fn chat(&self, query: &str) -> Result<Option<String>> {
        block_on(self.inner.chat(query))
    }

    /// Non-streaming dialectic chat with full options.
    pub fn chat_with_options(&self, options: &DialecticOptions) -> Result<Option<String>> {
        block_on(self.inner.chat_with_options(options))
    }

    /// Start a streaming dialectic chat. Returns a builder; call `.send()` for an iterator.
    #[must_use]
    pub fn chat_stream(&self, query: impl Into<String>) -> BlockingChatStreamBuilder {
        BlockingChatStreamBuilder {
            inner: self.inner.chat_stream(query),
        }
    }

    /// Get the peer's representation.
    pub fn representation(&self) -> Result<String> {
        block_on(self.inner.representation())
    }

    /// Get a builder for fine-grained representation parameters.
    #[must_use]
    pub fn representation_builder(&self) -> BlockingRepresentationBuilder {
        BlockingRepresentationBuilder {
            inner: self.inner.representation_builder(),
        }
    }

    /// Get a builder for fine-grained context parameters.
    #[must_use]
    pub fn context_builder(&self) -> BlockingContextBuilder {
        BlockingContextBuilder {
            inner: self.inner.context_builder(),
        }
    }

    /// Get the peer's context.
    pub fn context(&self) -> Result<PeerContext> {
        block_on(self.inner.context())
    }

    /// Get the peer's context scoped to a target.
    #[deprecated(since = "0.1.1", note = "use `Peer::context_builder()` instead")]
    #[allow(deprecated)]
    pub fn context_with_target(&self, target: &str) -> Result<PeerContext> {
        block_on(self.inner.context_with_target(target))
    }

    /// Get the peer's context with custom options.
    #[deprecated(since = "0.1.1", note = "use `Peer::context_builder()` instead")]
    #[allow(deprecated)]
    pub fn context_with_options(
        &self,
        options: &crate::types::peer::PeerContextOptions,
    ) -> Result<PeerContext> {
        block_on(self.inner.context_with_options(options))
    }

    /// List sessions for this peer, collecting across pages.
    pub fn sessions(&self) -> Result<Vec<Session>> {
        block_on(async {
            let page = self.inner.sessions().await?;
            collect_all_pages(page).await
        })
    }

    /// List sessions with filters and pagination options. Returns a [`Page`].
    pub fn sessions_with_options(&self, options: &SessionListOptions) -> Result<Page<Session>> {
        block_on(self.inner.sessions_with_options(options))
    }

    /// Search messages for this peer.
    pub fn search(&self, query: &str) -> Result<Vec<crate::Message>> {
        block_on(self.inner.search(query))
    }

    /// Search messages for this peer with custom options.
    pub fn search_with_options(
        &self,
        options: &MessageSearchOptions,
    ) -> Result<Vec<crate::Message>> {
        block_on(self.inner.search_with_options(options))
    }

    /// Get this peer's card.
    pub fn get_card(&self) -> Result<Option<Vec<String>>> {
        block_on(self.inner.get_card())
    }

    /// Get this peer's card scoped to a target.
    pub fn get_card_with_target(&self, target: &str) -> Result<Option<Vec<String>>> {
        block_on(self.inner.get_card_with_target(target))
    }

    /// Set this peer's card.
    pub fn set_card(&self, card: Vec<String>) -> Result<Option<Vec<String>>> {
        block_on(self.inner.set_card(card))
    }

    /// Set this peer's card scoped to a target.
    pub fn set_card_with_target(
        &self,
        card: Vec<String>,
        target: &str,
    ) -> Result<Option<Vec<String>>> {
        block_on(self.inner.set_card_with_target(card, target))
    }

    /// Self-scoped conclusion handle.
    #[must_use]
    pub fn conclusions(&self) -> ConclusionScope {
        ConclusionScope::new(self.inner.conclusions())
    }

    /// Cross-peer conclusion handle.
    #[must_use]
    pub fn conclusions_of(&self, target: impl Into<String>) -> ConclusionScope {
        ConclusionScope::new(self.inner.conclusions_of(target))
    }

    /// Create a message builder (sync, no API call).
    #[must_use]
    pub fn message(&self, content: impl Into<String>) -> crate::peer::MessageBuilder {
        self.inner.message(content)
    }
}

/// Blocking streaming dialectic chat builder.
pub struct BlockingChatStreamBuilder {
    inner: crate::peer::ChatStreamBuilder,
}

impl BlockingChatStreamBuilder {
    /// Scope the chat to a target peer.
    #[must_use]
    pub fn target(self, target: impl Into<String>) -> Self {
        Self {
            inner: self.inner.target(target),
        }
    }

    /// Scope the chat to a session.
    #[must_use]
    pub fn session(self, session_id: impl Into<String>) -> Self {
        Self {
            inner: self.inner.session(session_id),
        }
    }

    /// Set the reasoning level.
    #[must_use]
    pub fn reasoning_level(self, level: ReasoningLevel) -> Self {
        Self {
            inner: self.inner.reasoning_level(level),
        }
    }

    /// Send and return an iterator over SSE chunks.
    pub fn send(self) -> Result<ChatStreamIterator> {
        let stream = block_on(self.inner.send())?;
        Ok(ChatStreamIterator {
            inner: BlockingIter::new(stream),
        })
    }
}

/// Iterator over streaming dialectic chat chunks.
///
/// Wraps a [`DialecticStream`], so [`final_response`](DialecticStream::final_response)
/// and [`is_complete`](DialecticStream::is_complete) are available after iteration.
#[allow(clippy::type_complexity)]
pub struct ChatStreamIterator {
    inner: BlockingIter<DialecticStream<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>>,
}

impl ChatStreamIterator {
    /// Access the accumulated [`FinalResponse`](crate::FinalResponse).
    #[must_use]
    pub fn final_response(&self) -> &crate::FinalResponse {
        self.inner.stream().final_response()
    }

    /// Whether the underlying stream has ended.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.inner.stream().is_complete()
    }
}

impl Iterator for ChatStreamIterator {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Blocking builder for fine-grained representation requests.
pub struct BlockingRepresentationBuilder {
    inner: crate::peer::RepresentationBuilder,
}

impl BlockingRepresentationBuilder {
    /// Scope to a session.
    #[must_use]
    pub fn session_id(self, val: impl Into<String>) -> Self {
        Self {
            inner: self.inner.session_id(val),
        }
    }

    /// Target peer.
    #[must_use]
    pub fn target(self, val: impl Into<String>) -> Self {
        Self {
            inner: self.inner.target(val),
        }
    }

    /// Semantic search query.
    #[must_use]
    pub fn search_query(self, val: impl Into<String>) -> Self {
        Self {
            inner: self.inner.search_query(val),
        }
    }

    /// Top-K for semantic search (1–100).
    #[must_use]
    pub fn search_top_k(self, val: u32) -> Self {
        Self {
            inner: self.inner.search_top_k(val),
        }
    }

    /// Max cosine distance (0.0–1.0).
    #[must_use]
    pub fn search_max_distance(self, val: f64) -> Self {
        Self {
            inner: self.inner.search_max_distance(val),
        }
    }

    /// Include most frequent conclusions.
    #[must_use]
    pub fn include_most_frequent(self, val: bool) -> Self {
        Self {
            inner: self.inner.include_most_frequent(val),
        }
    }

    /// Max conclusions (1–100).
    #[must_use]
    pub fn max_conclusions(self, val: u32) -> Self {
        Self {
            inner: self.inner.max_conclusions(val),
        }
    }

    /// Send the representation request.
    pub fn send(self) -> Result<String> {
        block_on(self.inner.send())
    }
}

/// Blocking builder for fine-grained context requests.
pub struct BlockingContextBuilder {
    inner: crate::peer::ContextBuilder,
}

impl BlockingContextBuilder {
    /// Scope to a target peer.
    #[must_use]
    pub fn target(self, val: impl Into<String>) -> Self {
        Self {
            inner: self.inner.target(val),
        }
    }

    /// Whether to include the peer card summary.
    #[must_use]
    pub fn summary(self, val: bool) -> Self {
        Self {
            inner: self.inner.summary(val),
        }
    }

    /// Limit to a specific session.
    #[must_use]
    pub fn limit_to_session(self, val: bool) -> Self {
        Self {
            inner: self.inner.limit_to_session(val),
        }
    }

    /// Max conclusions (1–100).
    #[must_use]
    pub fn max_conclusions(self, val: u32) -> Self {
        Self {
            inner: self.inner.max_conclusions(val),
        }
    }

    /// Semantic search query.
    #[must_use]
    pub fn search_query(self, val: impl Into<String>) -> Self {
        Self {
            inner: self.inner.search_query(val),
        }
    }

    /// Top-K for semantic search (1–100).
    #[must_use]
    pub fn search_top_k(self, val: u32) -> Self {
        Self {
            inner: self.inner.search_top_k(val),
        }
    }

    /// Max cosine distance (0.0–1.0).
    #[must_use]
    pub fn search_max_distance(self, val: f64) -> Self {
        Self {
            inner: self.inner.search_max_distance(val),
        }
    }

    /// Include most frequent conclusions.
    #[must_use]
    pub fn include_most_frequent(self, val: bool) -> Self {
        Self {
            inner: self.inner.include_most_frequent(val),
        }
    }

    /// Send the context request.
    pub fn send(self) -> crate::error::Result<crate::types::peer::PeerContext> {
        block_on(self.inner.send())
    }
}
