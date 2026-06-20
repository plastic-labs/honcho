//! Minimal deriver prompt, ported from `src/deriver/prompts.py`.
//!
//! Produces byte-identical output to the Python `minimal_deriver_prompt` and
//! the token-estimate helpers. The Python prompts wrap their bodies in
//! `inspect.cleandoc` (imported as `c`); since the main prompt's body sits at
//! column 0 with the custom-instructions placeholder also at column 0, its
//! `cleandoc` only strips the leading/trailing blank line — so the final body
//! is written here already-stripped, and only the indented
//! custom-instructions section needs an actual [`cleandoc`].

use crate::text::cleandoc;
use crate::tokens::estimate_tokens;

/// Return stripped custom instructions, or `None` when absent/blank
/// (`_normalized_custom_instructions`).
fn normalized_custom_instructions(custom_instructions: Option<&str>) -> Option<String> {
    let normalized = custom_instructions?.trim();
    if normalized.is_empty() {
        None
    } else {
        Some(normalized.to_string())
    }
}

/// Render the optional custom-instructions section (`_custom_instructions_section`).
/// Empty string when no instructions; otherwise the cleandoc'd block.
fn custom_instructions_section(custom_instructions: Option<&str>) -> String {
    match normalized_custom_instructions(custom_instructions) {
        None => String::new(),
        Some(normalized) => cleandoc(&format!(
            "\n        CUSTOM INSTRUCTIONS:\n        {normalized}\n        "
        )),
    }
}

/// Generate the minimal prompt for fast observation extraction
/// (`minimal_deriver_prompt`).
pub fn minimal_deriver_prompt(
    peer_id: &str,
    messages: &str,
    custom_instructions: Option<&str>,
) -> String {
    let section = custom_instructions_section(custom_instructions);
    format!(
        r#"Analyze messages from {peer_id} to extract **explicit atomic facts** about them.

[EXPLICIT] DEFINITION: Facts about {peer_id} that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Properly attribute observations to the correct subject: if it is about {peer_id}, say so. If {peer_id} is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand {peer_id}.
- Extract ALL observations from {peer_id} messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES:
- EXPLICIT: "I just had my 25th birthday last Saturday" → "{peer_id} is 25 years old", "{peer_id}'s birthday is June 21st"
- EXPLICIT: "I took my dog for a walk in NYC" → "{peer_id} has a dog", "{peer_id} lives in NYC"
- EXPLICIT: "{peer_id} attended college" + general knowledge → "{peer_id} completed high school or equivalent"

{section}

Messages to analyze:
<messages>
{messages}
</messages>"#
    )
}

/// Estimate the static minimal prompt (no custom instructions). Python memoizes
/// this with `@cache`; the count is cheap enough here to recompute.
pub fn estimate_minimal_deriver_prompt_tokens() -> usize {
    estimate_tokens(&minimal_deriver_prompt("", "", None))
}

/// Estimate prompt tokens including custom instructions when present
/// (`estimate_deriver_prompt_tokens`).
pub fn estimate_deriver_prompt_tokens(custom_instructions: Option<&str>) -> usize {
    match normalized_custom_instructions(custom_instructions) {
        None => estimate_minimal_deriver_prompt_tokens(),
        Some(normalized) => estimate_tokens(&minimal_deriver_prompt("", "", Some(&normalized))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden strings captured from Python `minimal_deriver_prompt(...)`.
    const GOLDEN_NO_CI: &str = "Analyze messages from alice to extract **explicit atomic facts** about them.\n\n[EXPLICIT] DEFINITION: Facts about alice that can be derived directly from their messages.\n   - Transform statements into one or multiple conclusions\n   - Each conclusion must be self-contained with enough context\n   - Use absolute dates/times when possible (e.g. \"June 26, 2025\" not \"yesterday\")\n\nRULES:\n- Properly attribute observations to the correct subject: if it is about alice, say so. If alice is referencing someone or something else, make that clear.\n- Observations should make sense on their own. Each observation will be used in the future to better understand alice.\n- Extract ALL observations from alice messages, using others as context.\n- Contextualize each observation sufficiently (e.g. \"Ann is nervous about the job interview at the pharmacy\" not just \"Ann is nervous\")\n\nEXAMPLES:\n- EXPLICIT: \"I just had my 25th birthday last Saturday\" → \"alice is 25 years old\", \"alice's birthday is June 21st\"\n- EXPLICIT: \"I took my dog for a walk in NYC\" → \"alice has a dog\", \"alice lives in NYC\"\n- EXPLICIT: \"alice attended college\" + general knowledge → \"alice completed high school or equivalent\"\n\n\n\nMessages to analyze:\n<messages>\nhello world\n</messages>";

    const GOLDEN_CI: &str = "Analyze messages from alice to extract **explicit atomic facts** about them.\n\n[EXPLICIT] DEFINITION: Facts about alice that can be derived directly from their messages.\n   - Transform statements into one or multiple conclusions\n   - Each conclusion must be self-contained with enough context\n   - Use absolute dates/times when possible (e.g. \"June 26, 2025\" not \"yesterday\")\n\nRULES:\n- Properly attribute observations to the correct subject: if it is about alice, say so. If alice is referencing someone or something else, make that clear.\n- Observations should make sense on their own. Each observation will be used in the future to better understand alice.\n- Extract ALL observations from alice messages, using others as context.\n- Contextualize each observation sufficiently (e.g. \"Ann is nervous about the job interview at the pharmacy\" not just \"Ann is nervous\")\n\nEXAMPLES:\n- EXPLICIT: \"I just had my 25th birthday last Saturday\" → \"alice is 25 years old\", \"alice's birthday is June 21st\"\n- EXPLICIT: \"I took my dog for a walk in NYC\" → \"alice has a dog\", \"alice lives in NYC\"\n- EXPLICIT: \"alice attended college\" + general knowledge → \"alice completed high school or equivalent\"\n\nCUSTOM INSTRUCTIONS:\nbe terse\n\nMessages to analyze:\n<messages>\nhello world\n</messages>";

    const GOLDEN_CI_ML: &str = "Analyze messages from alice to extract **explicit atomic facts** about them.\n\n[EXPLICIT] DEFINITION: Facts about alice that can be derived directly from their messages.\n   - Transform statements into one or multiple conclusions\n   - Each conclusion must be self-contained with enough context\n   - Use absolute dates/times when possible (e.g. \"June 26, 2025\" not \"yesterday\")\n\nRULES:\n- Properly attribute observations to the correct subject: if it is about alice, say so. If alice is referencing someone or something else, make that clear.\n- Observations should make sense on their own. Each observation will be used in the future to better understand alice.\n- Extract ALL observations from alice messages, using others as context.\n- Contextualize each observation sufficiently (e.g. \"Ann is nervous about the job interview at the pharmacy\" not just \"Ann is nervous\")\n\nEXAMPLES:\n- EXPLICIT: \"I just had my 25th birthday last Saturday\" → \"alice is 25 years old\", \"alice's birthday is June 21st\"\n- EXPLICIT: \"I took my dog for a walk in NYC\" → \"alice has a dog\", \"alice lives in NYC\"\n- EXPLICIT: \"alice attended college\" + general knowledge → \"alice completed high school or equivalent\"\n\n        CUSTOM INSTRUCTIONS:\n        line1\nline2\n        \n\nMessages to analyze:\n<messages>\nmsgs\n</messages>";

    #[test]
    fn prompt_without_custom_instructions_matches_golden() {
        assert_eq!(
            minimal_deriver_prompt("alice", "hello world", None),
            GOLDEN_NO_CI
        );
    }

    #[test]
    fn prompt_with_custom_instructions_matches_golden() {
        // Surrounding whitespace is stripped by _normalized_custom_instructions.
        assert_eq!(
            minimal_deriver_prompt("alice", "hello world", Some("  be terse  ")),
            GOLDEN_CI
        );
    }

    #[test]
    fn prompt_with_multiline_custom_instructions_matches_golden() {
        assert_eq!(
            minimal_deriver_prompt("alice", "msgs", Some("line1\nline2")),
            GOLDEN_CI_ML
        );
    }

    #[test]
    fn blank_custom_instructions_render_as_no_section() {
        assert_eq!(
            minimal_deriver_prompt("alice", "hello world", Some("   ")),
            GOLDEN_NO_CI
        );
    }

    #[test]
    fn estimate_with_blank_equals_minimal() {
        assert_eq!(
            estimate_deriver_prompt_tokens(Some("   ")),
            estimate_minimal_deriver_prompt_tokens()
        );
        assert_eq!(
            estimate_deriver_prompt_tokens(None),
            estimate_minimal_deriver_prompt_tokens()
        );
    }

    #[test]
    fn estimate_with_instructions_is_larger() {
        assert!(
            estimate_deriver_prompt_tokens(Some("be very terse and specific"))
                > estimate_minimal_deriver_prompt_tokens()
        );
    }
}
