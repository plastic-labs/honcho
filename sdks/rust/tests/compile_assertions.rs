#![allow(clippy::used_underscore_items)]

use std::fmt::Display;

use honcho_ai::http::client::HttpClient;
use honcho_ai::{Conclusion, ConclusionScope, Honcho, Message, Peer, Session};
use static_assertions::assert_impl_all;

assert_impl_all!(HttpClient: Send, Sync, Clone);
assert_impl_all!(Conclusion: Send, Sync, Clone, std::fmt::Debug, Display);
assert_impl_all!(ConclusionScope: Send, Sync, Clone);
assert_impl_all!(honcho_ai::Peer: std::fmt::Debug);
assert_impl_all!(honcho_ai::DialecticStream<futures_util::stream::Empty<honcho_ai::error::Result<String>>>: std::fmt::Debug);
assert_impl_all!(Honcho: Send, Sync, Clone);
assert_impl_all!(Peer: Send, Sync, Clone);
assert_impl_all!(Session: Send, Sync, Clone);
assert_impl_all!(Message: Send, Sync, Clone, std::fmt::Debug, Display);
assert_impl_all!(honcho_ai::error::HonchoError: Send, Sync, std::error::Error);

fn _assert_future_send<F: std::future::Future + Send>(_: F) {}

fn _honcho_peers_future_is_send(h: &Honcho) {
    _assert_future_send(h.peers());
}

fn _honcho_search_future_is_send(h: &Honcho) {
    _assert_future_send(h.search("q", None, None));
}

fn _peer_chat_stream_future_is_send(p: &Peer) {
    _assert_future_send(async move {
        let _ = p.chat_stream("q").send().await;
    });
}

fn _session_messages_future_is_send(s: &Session) {
    _assert_future_send(s.messages());
}
