//! File-upload helpers ported from `src/utils/files.py`.
//!
//! Currently the deterministic, dependency-free core: the text chunk splitter.
//! The per-content-type text extraction (text/JSON/PDF) and the multipart route
//! wiring are not yet ported — PDF extraction in particular cannot be made
//! byte-exact against Python's PDF library.

/// Default per-message chunk size, matching Python's `split_text_into_chunks`
/// default (`max_chars=49500`).
pub const DEFAULT_MAX_CHARS: usize = 49500;

/// Port of `split_text_into_chunks`. Splits `text` into chunks of at most
/// `max_chars` **Unicode code points** (Python `str` length/slicing semantics),
/// preferring to break at the last `"\n\n"`, `"\n"`, `". "`, or `" "` delimiter
/// within each window; falls back to a hard split at `max_chars` when no
/// delimiter is found past the chunk start.
pub fn split_text_into_chunks(text: &str, max_chars: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current_pos = 0usize;
    while current_pos < len {
        let end_pos = current_pos + max_chars;
        if end_pos >= len {
            chunks.push(chars[current_pos..].iter().collect());
            break;
        }

        // Try delimiters in priority order; use the first whose last occurrence
        // lies strictly past the chunk start (Python's `> current_pos` guard).
        let mut break_pos = end_pos;
        for delimiter in ["\n\n", "\n", ". ", " "] {
            if let Some(idx) = rfind_chars(&chars, delimiter, current_pos, end_pos)
                && idx > current_pos
            {
                break_pos = idx + delimiter.chars().count();
                break;
            }
        }

        chunks.push(chars[current_pos..break_pos].iter().collect());
        current_pos = break_pos;
    }

    chunks
}

/// `str.rfind(sub, start, end)` over a code-point slice: the highest index `i`
/// in `[start, end - sub_len]` with `chars[i..i + sub_len] == sub`, else `None`.
fn rfind_chars(chars: &[char], sub: &str, start: usize, end: usize) -> Option<usize> {
    let sub: Vec<char> = sub.chars().collect();
    let sub_len = sub.len();
    if sub_len == 0 || end < sub_len {
        return None;
    }
    let mut i = end - sub_len;
    loop {
        if i < start {
            return None;
        }
        if chars[i..i + sub_len] == sub[..] {
            return Some(i);
        }
        if i == 0 {
            return None;
        }
        i -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden values captured from Python `split_text_into_chunks`.
    #[test]
    fn matches_python_golden() {
        assert_eq!(split_text_into_chunks("hello world", 100), vec!["hello world"]);
        assert_eq!(
            split_text_into_chunks("aaaa bbbb cccc dddd", 10),
            vec!["aaaa bbbb ", "cccc dddd"]
        );
        assert_eq!(
            split_text_into_chunks("para one.\n\npara two is here.\n\npara three", 20),
            vec!["para one.\n\n", "para two is here.\n\n", "para three"]
        );
        assert_eq!(
            split_text_into_chunks("one. two. three. four.", 12),
            vec!["one. two. ", "three. four."]
        );
        // No delimiter past the start -> hard split at max_chars.
        assert_eq!(
            split_text_into_chunks("nodelimiterssolongtextwithoutbreaks", 10),
            vec!["nodelimite", "rssolongte", "xtwithoutb", "reaks"]
        );
        // Multibyte chars: splitting is by code point, not byte.
        assert_eq!(
            split_text_into_chunks("café déjà señor naïve façade", 12),
            vec!["café déjà ", "señor naïve ", "façade"]
        );
    }

    #[test]
    fn short_text_is_single_chunk() {
        assert_eq!(split_text_into_chunks("", 10), vec![""]);
        assert_eq!(split_text_into_chunks("exact", 5), vec!["exact"]);
    }
}
