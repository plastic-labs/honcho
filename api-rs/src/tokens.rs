//! o200k_base token counting with exact parity to Python's
//! `src.utils.tokens.estimate_tokens`.
//!
//! Python computes `len(tiktoken.get_encoding("o200k_base").encode(text))`
//! (falling back to `len(text) // 4` only if encoding raises, which does not
//! happen for ordinary text). We mirror that with `tiktoken-rs`' bundled
//! o200k_base ranks, using ordinary encoding (no special-token handling) so the
//! counts match Python's default `encode` for non-special content.

use std::sync::OnceLock;

use tiktoken_rs::{CoreBPE, o200k_base};

fn tokenizer() -> &'static CoreBPE {
    static TOKENIZER: OnceLock<CoreBPE> = OnceLock::new();
    TOKENIZER.get_or_init(|| o200k_base().expect("failed to load bundled o200k_base tokenizer"))
}

/// Count o200k_base tokens for `text`, matching Python's `estimate_tokens`.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    tokenizer().encode_ordinary(text).len()
}

#[cfg(test)]
mod tests {
    use super::estimate_tokens;

    /// Expected counts captured from Python's
    /// `tiktoken.get_encoding("o200k_base").encode(...)`.
    #[test]
    fn matches_python_o200k_base_counts() {
        let cases: &[(&str, usize)] = &[
            ("", 0),
            (" ", 1),
            ("hello", 1),
            ("hello world", 2),
            ("Hello, world!", 4),
            ("The quick brown fox jumps over the lazy dog.", 10),
            (&"a".repeat(100), 13),
            ("café résumé naïve", 5),
            ("日本語のテキストです", 7),
            ("🚀🔥😀 emoji test", 6),
            ("line1\nline2\nline3", 8),
            ("tokens   with   extra   spaces", 7),
            ("Mixed 123 numbers and CAPS and punctuation!!! ??? ...", 11),
            ("supercalifragilisticexpialidocious", 10),
        ];
        for (text, expected) in cases {
            assert_eq!(
                estimate_tokens(text),
                *expected,
                "token count mismatch for {text:?}"
            );
        }
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(estimate_tokens(""), 0);
    }
}
