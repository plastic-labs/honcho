//! Embedding chunking, ported from `src/embedding_client.py`
//! (`_prepare_chunks` / `_chunk_text_with_tokens` / `prepare_chunks`).
//!
//! This is the deterministic, network-free part of the embedding pipeline: it
//! splits text into token-bounded chunks (with 20% overlap) so the message-write
//! path can persist one pending `MessageEmbedding` row per chunk. The actual
//! embedding vectors are generated later by the reconciler, so only the chunk
//! *text* needs byte-exact parity here.
//!
//! Embedding models (OpenAI `text-embedding-3-*`, etc.) tokenize with
//! `cl100k_base`, so chunk boundaries are computed against that vocabulary —
//! distinct from the `o200k_base` budget used for message `token_count`
//! (`crate::tokens`). Decoding mirrors tiktoken's Python `decode`, which is
//! `bytes.decode("utf-8", errors="replace")`: a multi-byte character split
//! across a chunk boundary becomes U+FFFD rather than an error.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use tiktoken_rs::{CoreBPE, Rank, cl100k_base};

fn tokenizer() -> &'static CoreBPE {
    static TOKENIZER: OnceLock<CoreBPE> = OnceLock::new();
    TOKENIZER.get_or_init(|| cl100k_base().expect("failed to load bundled cl100k_base tokenizer"))
}

/// Tokenize and chunk each input text, returning ordered chunk texts per id.
/// Mirrors `EmbeddingClient.prepare_chunks`.
pub fn prepare_chunks(
    id_resource_dict: &BTreeMap<String, String>,
    max_tokens: usize,
) -> BTreeMap<String, Vec<String>> {
    id_resource_dict
        .iter()
        .map(|(id, text)| {
            let chunks = chunk_text(text, max_tokens)
                .into_iter()
                .map(|(chunk, _)| chunk)
                .collect();
            (id.clone(), chunks)
        })
        .collect()
}

/// Split `text` into `(chunk_text, token_count)` pairs, porting
/// `_prepare_chunks` + `_chunk_text_with_tokens`. Texts within the limit return a
/// single whole-text chunk; longer texts are sliced into `max_tokens`-wide
/// windows that advance by `max_tokens - floor(max_tokens * 0.2)` (20% overlap).
pub fn chunk_text(text: &str, max_tokens: usize) -> Vec<(String, usize)> {
    let tokenizer = tokenizer();
    let tokens = tokenizer.encode_ordinary(text);
    if tokens.len() <= max_tokens {
        return vec![(text.to_string(), tokens.len())];
    }

    // `int(max_tokens * 0.2)` truncates toward zero; `step` is therefore always
    // >= 1 for max_tokens >= 1, but guard anyway so we never loop forever.
    let overlap = (max_tokens as f64 * 0.2) as usize;
    let step = (max_tokens - overlap).max(1);

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < tokens.len() {
        let end = (start + max_tokens).min(tokens.len());
        let chunk = decode_lossy(tokenizer, &tokens[start..end]);
        let count = max_tokens.min(tokens.len() - start);
        chunks.push((chunk, count));
        start += step;
    }
    chunks
}

/// Decode a token slice the way tiktoken's Python `decode` does: concatenate the
/// per-token byte sequences and UTF-8 decode with replacement, so boundaries
/// that split a character yield U+FFFD instead of failing.
fn decode_lossy(tokenizer: &CoreBPE, tokens: &[Rank]) -> String {
    let bytes: Vec<u8> = tokenizer
        ._decode_native_and_split(tokens.to_vec())
        .flatten()
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden values captured from Python's `cl100k_base` chunking
    /// (`enc.decode(toks[i:i+max])` with 20% overlap).
    #[test]
    fn chunk_text_matches_python_cl100k_base() {
        // Within the limit -> single whole-text chunk, count = total tokens.
        assert_eq!(
            chunk_text("hello world", 5),
            vec![("hello world".to_string(), 2)]
        );
        // Exactly at the boundary -> still a single chunk.
        assert_eq!(
            chunk_text("one two three four five six", 6),
            vec![("one two three four five six".to_string(), 6)]
        );
        // 12 tokens, max 6, overlap 1, step 5 -> windows at 0, 5, 10.
        assert_eq!(
            chunk_text(
                "The quick brown fox jumps over the lazy dog repeatedly today friends",
                6
            ),
            vec![
                ("The quick brown fox jumps over".to_string(), 6),
                (" over the lazy dog repeatedly today".to_string(), 6),
                (" today friends".to_string(), 2),
            ]
        );
        // 12 tokens, max 4, overlap 0, step 4 -> contiguous windows at 0, 4, 8.
        assert_eq!(
            chunk_text("café résumé naïve façade jalapeño", 4),
            vec![
                ("café résum".to_string(), 4),
                ("é naïve faç".to_string(), 4),
                ("ade jalapeño".to_string(), 4),
            ]
        );
    }

    #[test]
    fn prepare_chunks_returns_ordered_texts_per_id() {
        let mut input = BTreeMap::new();
        input.insert("a".to_string(), "hello world".to_string());
        input.insert(
            "b".to_string(),
            "The quick brown fox jumps over the lazy dog repeatedly today friends".to_string(),
        );
        let chunks = prepare_chunks(&input, 6);
        assert_eq!(chunks["a"], vec!["hello world".to_string()]);
        assert_eq!(
            chunks["b"],
            vec![
                "The quick brown fox jumps over".to_string(),
                " over the lazy dog repeatedly today".to_string(),
                " today friends".to_string(),
            ]
        );
    }

    #[test]
    fn empty_text_is_a_single_empty_chunk() {
        // No tokens (0 <= max) -> single chunk preserving the original text.
        assert_eq!(chunk_text("", 6), vec![(String::new(), 0)]);
    }
}
