//! File-upload helpers ported from `src/utils/files.py`.
//!
//! Ported here: the chunk splitter, the `text/*` text extraction, and the
//! per-chunk message-data builder (`process_file_uploads_for_messages`'s
//! deterministic core). Still TODO: `application/json` extraction (Python
//! `json.dumps(loads(...), ensure_ascii=False)`, which needs insertion-ordered
//! re-serialization with `", "`/`": "` separators — not reproducible via
//! `serde_json::Value`'s `BTreeMap` without the global `preserve_order` feature),
//! `application/pdf` extraction (pdfplumber — not byte-exact portable), and the
//! multipart `/messages/upload` route wiring.

use serde_json::{Value, json};

/// Default per-message chunk size, matching Python's `split_text_into_chunks`
/// default (`max_chars=49500`).
pub const DEFAULT_MAX_CHARS: usize = 49500;

/// Errors from file text extraction, mirroring the Python upload exceptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileError {
    /// No processor supports the content type (`UnsupportedFileTypeError`). Also
    /// returned for `application/pdf` and `application/json` until those
    /// extractors are ported.
    UnsupportedType(String),
    /// A `text/*` file that decoded as none of UTF-8/UTF-16/Latin-1
    /// (`ValueError("Could not decode text file")`). Latin-1 always succeeds, so
    /// this is effectively unreachable for `text/*`, but kept for fidelity.
    Decode,
}

/// One chunk's worth of message data, mirroring an entry of
/// `process_file_uploads_for_messages`'s return list: the message `content` plus
/// the `file_metadata` to merge into the message's `internal_metadata`.
#[derive(Debug, Clone, PartialEq)]
pub struct FileMessageData {
    pub content: String,
    pub file_metadata: Value,
}

/// Extract text from `text/*` content, porting `TextProcessor.extract_text`: try
/// UTF-8, then UTF-16 (BOM-aware), then Latin-1 (which never fails). Returns
/// [`FileError::Decode`] only if every attempt fails (unreachable in practice).
pub fn decode_text(content: &[u8]) -> Result<String, FileError> {
    if let Ok(text) = std::str::from_utf8(content) {
        return Ok(text.to_string());
    }
    if let Some(text) = decode_utf16(content) {
        return Ok(text);
    }
    // Latin-1: every byte maps to the code point of the same value.
    Ok(content.iter().map(|&byte| byte as char).collect())
}

/// Decode UTF-16 with optional BOM (Python's `bytes.decode("utf-16")`): a
/// leading `FF FE` selects little-endian, `FE FF` big-endian; absent a BOM,
/// Python defaults to the platform's native order, which is little-endian on the
/// supported targets. Returns `None` when the byte length is odd or the units do
/// not form valid UTF-16.
fn decode_utf16(content: &[u8]) -> Option<String> {
    let (bytes, big_endian) = match content {
        [0xFF, 0xFE, rest @ ..] => (rest, false),
        [0xFE, 0xFF, rest @ ..] => (rest, true),
        _ => (content, false),
    };
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| {
            if big_endian {
                u16::from_be_bytes([pair[0], pair[1]])
            } else {
                u16::from_le_bytes([pair[0], pair[1]])
            }
        })
        .collect();
    String::from_utf16(&units).ok()
}

/// Dispatch text extraction by content type, porting
/// `FileProcessingService.extract_text_from_upload`. Only `text/*` is ported;
/// `application/json` and `application/pdf` return [`FileError::UnsupportedType`]
/// until their extractors land (see the module docs).
pub fn extract_text(content: &[u8], content_type: &str) -> Result<String, FileError> {
    if content_type.starts_with("text/") {
        decode_text(content)
    } else {
        Err(FileError::UnsupportedType(content_type.to_string()))
    }
}

/// Port of `process_file_uploads_for_messages`'s post-extraction core: split the
/// already-extracted `text` into chunks and build one [`FileMessageData`] per
/// chunk, each carrying the per-file metadata (a shared `file_id`, the chunk
/// index/total, and the approximate `chunk_character_range` Python records —
/// `[i*max_chars, min((i+1)*max_chars, len)]` over code points, independent of
/// where the delimiter split actually fell).
pub fn build_file_message_data(
    text: &str,
    file_id: &str,
    filename: Option<&str>,
    content_type: Option<&str>,
    original_file_size: Option<i64>,
    max_chars: usize,
) -> Vec<FileMessageData> {
    let chunks = split_text_into_chunks(text, max_chars);
    let total_chunks = chunks.len();
    let text_len = text.chars().count();

    chunks
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| {
            let range_start = index * max_chars;
            let range_end = ((index + 1) * max_chars).min(text_len);
            FileMessageData {
                content: chunk,
                file_metadata: json!({
                    "file_id": file_id,
                    "filename": filename,
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                    "original_file_size": original_file_size,
                    "content_type": content_type,
                    "chunk_character_range": [range_start, range_end],
                }),
            }
        })
        .collect()
}

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

    #[test]
    fn decode_text_prefers_utf8_then_falls_back() {
        // Plain ASCII / valid UTF-8.
        assert_eq!(decode_text(b"hello \xC3\xA9").unwrap(), "hello é");
        // UTF-16 LE with BOM.
        let utf16 = [0xFF, 0xFE, b'h', 0x00, b'i', 0x00];
        assert_eq!(decode_text(&utf16).unwrap(), "hi");
        // UTF-16 BE with BOM.
        let utf16_be = [0xFE, 0xFF, 0x00, b'h', 0x00, b'i'];
        assert_eq!(decode_text(&utf16_be).unwrap(), "hi");
        // Invalid UTF-8, odd length (so not UTF-16) → Latin-1 (0xE9 → 'é').
        assert_eq!(decode_text(&[b'a', 0xE9, b'b']).unwrap(), "aéb");
    }

    #[test]
    fn extract_text_dispatches_by_content_type() {
        assert_eq!(extract_text(b"plain", "text/plain").unwrap(), "plain");
        assert_eq!(
            extract_text(b"{}", "application/json"),
            Err(FileError::UnsupportedType("application/json".to_string()))
        );
        assert_eq!(
            extract_text(b"%PDF", "application/pdf"),
            Err(FileError::UnsupportedType("application/pdf".to_string()))
        );
    }

    #[test]
    fn build_file_message_data_carries_per_chunk_metadata() {
        // 19 chars, max 10 → two chunks split at the space after "bbbb ".
        let data = build_file_message_data(
            "aaaa bbbb cccc dddd",
            "file123",
            Some("notes.txt"),
            Some("text/plain"),
            Some(19),
            10,
        );
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].content, "aaaa bbbb ");
        assert_eq!(data[1].content, "cccc dddd");

        assert_eq!(data[0].file_metadata["file_id"], json!("file123"));
        assert_eq!(data[0].file_metadata["chunk_index"], json!(0));
        assert_eq!(data[0].file_metadata["total_chunks"], json!(2));
        assert_eq!(data[0].file_metadata["filename"], json!("notes.txt"));
        assert_eq!(data[0].file_metadata["content_type"], json!("text/plain"));
        assert_eq!(data[0].file_metadata["original_file_size"], json!(19));
        // The recorded range is the nominal window, clipped to text length.
        assert_eq!(data[0].file_metadata["chunk_character_range"], json!([0, 10]));
        assert_eq!(data[1].file_metadata["chunk_character_range"], json!([10, 19]));
    }

    #[test]
    fn build_file_message_data_handles_empty_text() {
        let data = build_file_message_data("", "fid", None, None, Some(0), 10);
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].content, "");
        assert_eq!(data[0].file_metadata["total_chunks"], json!(1));
        assert_eq!(data[0].file_metadata["filename"], Value::Null);
        assert_eq!(data[0].file_metadata["chunk_character_range"], json!([0, 0]));
    }
}
