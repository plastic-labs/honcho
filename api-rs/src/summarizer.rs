//! Two-tier session summarization, porting the deterministic core of
//! `src/utils/summarizer.py`: the short/long prompt builders and their token
//! estimators. The LLM calls (`create_short_summary`/`create_long_summary`) and
//! the `summarize_if_needed` orchestrator (DB get/save of summaries, tier
//! gating) follow once the summaries crud is ported.

use crate::tokens::estimate_tokens;

/// The summary kinds stored in session metadata (`SummaryType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummaryType {
    Short,
    Long,
}

impl SummaryType {
    /// The metadata key value for this summary type.
    pub fn as_str(self) -> &'static str {
        match self {
            SummaryType::Short => "honcho_chat_summary_short",
            SummaryType::Long => "honcho_chat_summary_long",
        }
    }
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
}
