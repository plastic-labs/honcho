//! Port of the deterministic pieces of `src/deriver/deriver.py`
//! (`process_representation_tasks_batch`). The full orchestrator (LLM call +
//! save + telemetry emission) is wired separately once the telemetry layer is
//! ported; the token-accounting arithmetic that feeds the
//! `RepresentationCompletedEvent` is pure and lands here first.

use std::collections::HashSet;

use crate::db::BatchMessage;

/// The per-batch token breakdown computed in `process_representation_tasks_batch`.
///
/// `messages_tokens` counts only the messages actually being processed (those in
/// `queue_item_message_ids`); the rest are interleaving context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBreakdown {
    /// Tokens from the queued messages only (Python `messages_tokens`).
    pub messages_tokens: i64,
    /// Number of queued message ids (Python `queued_message_count`).
    pub queued_message_count: usize,
    /// Number of messages in the prompt incl. context (Python `prompt_message_count`).
    pub prompt_message_count: usize,
    /// Tokens across all prompt messages (Python `prompt_message_tokens`).
    pub prompt_message_tokens: i64,
    /// Context-only message count, clamped at 0 (Python `extra_context_message_count`).
    pub extra_context_message_count: usize,
    /// Context-only tokens, clamped at 0 (Python `extra_context_tokens`).
    pub extra_context_tokens: i64,
}

/// Port of the token-breakdown arithmetic in `process_representation_tasks_batch`.
pub fn compute_token_breakdown(
    messages: &[BatchMessage],
    queue_item_message_ids: &[i64],
) -> TokenBreakdown {
    let queued_ids: HashSet<i64> = queue_item_message_ids.iter().copied().collect();

    // messages_tokens: only messages whose id is in the queue-item set.
    let messages_tokens: i64 = messages
        .iter()
        .filter(|m| queued_ids.contains(&m.id))
        .map(|m| m.token_count as i64)
        .sum();

    let queued_message_count = queue_item_message_ids.len();
    let prompt_message_count = messages.len();
    let prompt_message_tokens: i64 = messages.iter().map(|m| m.token_count as i64).sum();
    let extra_context_message_count = prompt_message_count.saturating_sub(queued_message_count);
    let extra_context_tokens = (prompt_message_tokens - messages_tokens).max(0);

    TokenBreakdown {
        messages_tokens,
        queued_message_count,
        prompt_message_count,
        prompt_message_tokens,
        extra_context_message_count,
        extra_context_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    fn msg(id: i64, token_count: i32) -> BatchMessage {
        BatchMessage {
            id,
            public_id: format!("pub_{id}"),
            content: String::new(),
            created_at: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
            peer_name: "p".to_string(),
            token_count,
            session_name: "s".to_string(),
            workspace_name: "w".to_string(),
        }
    }

    #[test]
    fn breakdown_separates_queued_from_context() {
        // ids 1,2 queued; 3,4 are interleaving context.
        let messages = vec![msg(1, 10), msg(2, 20), msg(3, 5), msg(4, 7)];
        let queued = vec![1, 2];
        let b = compute_token_breakdown(&messages, &queued);
        assert_eq!(b.messages_tokens, 30);
        assert_eq!(b.queued_message_count, 2);
        assert_eq!(b.prompt_message_count, 4);
        assert_eq!(b.prompt_message_tokens, 42);
        assert_eq!(b.extra_context_message_count, 2);
        assert_eq!(b.extra_context_tokens, 12);
    }

    #[test]
    fn breakdown_all_queued_has_no_extra_context() {
        let messages = vec![msg(1, 10), msg(2, 20)];
        let queued = vec![1, 2];
        let b = compute_token_breakdown(&messages, &queued);
        assert_eq!(b.messages_tokens, 30);
        assert_eq!(b.extra_context_message_count, 0);
        assert_eq!(b.extra_context_tokens, 0);
    }

    #[test]
    fn breakdown_clamps_when_queued_ids_exceed_messages() {
        // A queued id with no matching message (already-processed context window):
        // counts toward queued_message_count but contributes no tokens, and the
        // clamps keep extras non-negative.
        let messages = vec![msg(1, 10)];
        let queued = vec![1, 99, 100];
        let b = compute_token_breakdown(&messages, &queued);
        assert_eq!(b.messages_tokens, 10);
        assert_eq!(b.queued_message_count, 3);
        assert_eq!(b.prompt_message_count, 1);
        assert_eq!(b.extra_context_message_count, 0); // saturating_sub
        assert_eq!(b.extra_context_tokens, 0); // max(10 - 10, 0)
    }
}
