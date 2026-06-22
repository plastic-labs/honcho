//! Two-tier session summarization, porting the deterministic core of
//! `src/utils/summarizer.py`: the short/long prompt builders and their token
//! estimators. The LLM calls (`create_short_summary`/`create_long_summary`) and
//! the `summarize_if_needed` orchestrator (DB get/save of summaries, tier
//! gating) follow once the summaries crud is ported.

use serde::{Deserialize, Serialize};

use crate::tokens::estimate_tokens;

/// The summary kinds stored in session metadata (`SummaryType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummaryType {
    Short,
    Long,
}

impl SummaryType {
    /// The metadata key value for this summary type (also the JSON map key under
    /// `internal_metadata["summaries"]`).
    pub fn as_str(self) -> &'static str {
        match self {
            SummaryType::Short => "honcho_chat_summary_short",
            SummaryType::Long => "honcho_chat_summary_long",
        }
    }
}

/// Port of the `Summary` TypedDict: a stored session summary. Serializes to the
/// exact JSON shape persisted under `internal_metadata["summaries"][type]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Summary {
    pub content: String,
    /// The primary-key id of the message this summary covers up to.
    pub message_id: i64,
    pub summary_type: String,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    pub token_count: i64,
    #[serde(default)]
    pub message_public_id: String,
}

/// Port of `short_summary_prompt`. The Python `c(...)` (`inspect.cleandoc`) only
/// strips the leading/trailing blank line here — the body is flush-left — so the
/// trimmed literal below is byte-identical.
pub fn short_summary_prompt(
    formatted_messages: &str,
    output_words: i64,
    previous_summary_text: &str,
) -> String {
    format!(
        "You are a system that summarizes parts of a conversation to create a concise and accurate summary. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

Provide a concise, factual summary that captures the essence of the conversation. Your summary should be detailed enough to serve as context for future messages, but brief enough to be helpful. Prefer a thorough chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{formatted_messages}
</conversation>

Hard limit: {output_words} words maximum. If needed, drop lower-priority detail to stay within the limit."
    )
}

/// Port of `long_summary_prompt`.
pub fn long_summary_prompt(
    formatted_messages: &str,
    output_words: i64,
    previous_summary_text: &str,
) -> String {
    format!(
        "You are a system that creates thorough, comprehensive summaries of conversations. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed in detail
5. User's apparent emotional state and personality traits
6. Important themes and patterns across the conversation

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

Provide a thorough and detailed summary that captures the essence of the conversation. Your summary should serve as a comprehensive record of the important information in this conversation. Prefer an exhaustive chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{formatted_messages}
</conversation>

Hard limit: {output_words} words maximum. If needed, drop lower-priority detail to stay within the limit."
    )
}

/// Port of `estimate_short_summary_prompt_tokens`: the scaffold token count with
/// no messages/previous-summary/word-limit. (Python memoizes with `@cache`; this
/// recomputes — cheap.)
pub fn estimate_short_summary_prompt_tokens() -> usize {
    estimate_tokens(&short_summary_prompt("", 0, ""))
}

/// Port of `estimate_long_summary_prompt_tokens`.
pub fn estimate_long_summary_prompt_tokens() -> usize {
    estimate_tokens(&long_summary_prompt("", 0, ""))
}

/// The `previous_summary` placeholder used when there is no prior summary
/// (`create_short_summary`/`create_long_summary`).
pub const NO_PREVIOUS_SUMMARY: &str =
    "There is no previous summary -- the messages are the beginning of the conversation.";

/// The text fed into the prompt's `<previous_summary>` slot: the prior summary
/// when present and non-empty (Python's truthy check), else [`NO_PREVIOUS_SUMMARY`].
pub fn previous_summary_text(previous_summary: Option<&str>) -> &str {
    match previous_summary {
        Some(text) if !text.is_empty() => text,
        _ => NO_PREVIOUS_SUMMARY,
    }
}

/// Port of `create_short_summary`'s word-budget: `int(min(input_tokens,
/// max_tokens_short) * 0.75)`. The 4:3 word/token ratio makes the short summary
/// smaller than the content it condenses.
pub fn short_summary_output_words(input_tokens: i64, max_tokens_short: i64) -> i64 {
    (input_tokens.min(max_tokens_short) as f64 * 0.75) as i64
}

/// Port of `create_long_summary`'s word-budget: `int(max_tokens_long * 0.75)`.
pub fn long_summary_output_words(max_tokens_long: i64) -> i64 {
    (max_tokens_long as f64 * 0.75) as i64
}

/// Port of the start-sequence computation in `_create_and_save_summary`: the
/// summary covers the last `messages_per_summary` messages up to the trigger,
/// clamped so the range starts at sequence 1.
pub fn summary_start_seq(message_seq_in_session: i64, messages_per_summary: i64) -> i64 {
    (message_seq_in_session - messages_per_summary + 1).max(1)
}

/// Which summaries `summarize_if_needed` should create for a message at
/// `message_seq_in_session`, given the per-tier intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SummaryDecision {
    pub create_short: bool,
    pub create_long: bool,
}

impl SummaryDecision {
    /// True when at least one summary should be created.
    pub fn any(self) -> bool {
        self.create_short || self.create_long
    }
}

/// Port of the gating in `summarize_if_needed`: when summaries are enabled, a
/// tier fires when the message sequence is an exact multiple of its interval.
/// Returns "create nothing" when disabled.
pub fn decide_summaries(
    enabled: bool,
    message_seq_in_session: i64,
    messages_per_short_summary: i64,
    messages_per_long_summary: i64,
) -> SummaryDecision {
    if !enabled {
        return SummaryDecision {
            create_short: false,
            create_long: false,
        };
    }
    SummaryDecision {
        create_short: message_seq_in_session % messages_per_short_summary == 0,
        create_long: message_seq_in_session % messages_per_long_summary == 0,
    }
}

/// Port of `_format_messages`: each message rendered as `peer_name: content`,
/// joined by newlines (empty string for no messages).
pub fn format_messages<I, S>(messages: I) -> String
where
    I: IntoIterator<Item = (S, S)>,
    S: AsRef<str>,
{
    messages
        .into_iter()
        .map(|(peer, content)| format!("{}: {}", peer.as_ref(), content.as_ref()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_type_metadata_keys() {
        assert_eq!(SummaryType::Short.as_str(), "honcho_chat_summary_short");
        assert_eq!(SummaryType::Long.as_str(), "honcho_chat_summary_long");
    }

    #[test]
    fn short_prompt_matches_python_golden() {
        let prompt = short_summary_prompt("MSGS", 100, "PREV");
        // cleandoc(short_summary_prompt(...)) captured from Python.
        let expected = "You are a system that summarizes parts of a conversation to create a concise and accurate summary. Focus on capturing:\n\n1. Key facts and information shared (**Capture as many explicit facts as possible**)\n2. User preferences, opinions, and questions\n3. Important context and requests\n4. Core topics discussed\n\nIf there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.\n\nProvide a concise, factual summary that captures the essence of the conversation. Your summary should be detailed enough to serve as context for future messages, but brief enough to be helpful. Prefer a thorough chronological narrative over a list of bullet points.\n\nReturn only the summary without any explanation or meta-commentary.\n\n<previous_summary>\nPREV\n</previous_summary>\n\n<conversation>\nMSGS\n</conversation>\n\nHard limit: 100 words maximum. If needed, drop lower-priority detail to stay within the limit.";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn long_prompt_has_six_capture_points_and_slots() {
        let prompt = long_summary_prompt("CONV", 250, "OLD");
        assert!(prompt.starts_with(
            "You are a system that creates thorough, comprehensive summaries of conversations."
        ));
        assert!(prompt.contains("6. Important themes and patterns across the conversation"));
        assert!(prompt.contains("<previous_summary>\nOLD\n</previous_summary>"));
        assert!(prompt.contains("<conversation>\nCONV\n</conversation>"));
        assert!(prompt.ends_with("Hard limit: 250 words maximum. If needed, drop lower-priority detail to stay within the limit."));
    }

    #[test]
    fn estimators_match_python_o200k_counts() {
        // Golden from estimate_tokens(short/long_summary_prompt("", 0, "")).
        assert_eq!(estimate_short_summary_prompt_tokens(), 194);
        assert_eq!(estimate_long_summary_prompt_tokens(), 207);
    }

    #[test]
    fn previous_summary_text_falls_back_when_absent_or_empty() {
        assert_eq!(previous_summary_text(Some("prior")), "prior");
        assert_eq!(previous_summary_text(Some("")), NO_PREVIOUS_SUMMARY);
        assert_eq!(previous_summary_text(None), NO_PREVIOUS_SUMMARY);
    }

    #[test]
    fn output_words_match_python_arithmetic() {
        // short: int(min(input, max) * 0.75)
        assert_eq!(short_summary_output_words(1000, 4000), 750); // min=1000 -> 750
        assert_eq!(short_summary_output_words(8000, 4000), 3000); // min=4000 -> 3000
        assert_eq!(short_summary_output_words(101, 4000), 75); // 101*0.75=75.75 -> 75 (trunc)
        // long: int(max * 0.75)
        assert_eq!(long_summary_output_words(6000), 4500);
        assert_eq!(long_summary_output_words(101), 75);
    }

    #[test]
    fn start_seq_clamps_at_one() {
        assert_eq!(summary_start_seq(20, 20), 1); // 20-20+1 = 1
        assert_eq!(summary_start_seq(60, 20), 41); // 60-20+1 = 41
        assert_eq!(summary_start_seq(5, 20), 1); // -14 clamped to 1
    }

    #[test]
    fn decide_summaries_gates_on_intervals_and_enabled() {
        // disabled → nothing.
        let off = decide_summaries(false, 60, 20, 60);
        assert_eq!(off, SummaryDecision { create_short: false, create_long: false });
        assert!(!off.any());

        // seq 60 is a multiple of both 20 and 60.
        let both = decide_summaries(true, 60, 20, 60);
        assert_eq!(both, SummaryDecision { create_short: true, create_long: true });

        // seq 20: short only.
        let short_only = decide_summaries(true, 20, 20, 60);
        assert_eq!(short_only, SummaryDecision { create_short: true, create_long: false });

        // seq 25: neither.
        assert!(!decide_summaries(true, 25, 20, 60).any());
    }

    #[test]
    fn format_messages_prefixes_peer_and_joins() {
        let msgs = vec![("bob", "hi"), ("alice", "hello there")];
        assert_eq!(format_messages(msgs), "bob: hi\nalice: hello there");
        let empty: Vec<(&str, &str)> = vec![];
        assert_eq!(format_messages(empty), "");
    }
}
