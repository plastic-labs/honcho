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

// ---------------------------------------------------------------------------
// Embedding generation (OpenAI `/embeddings`), ported from
// `_EmbeddingClient.embed` (OpenAI branch).
//
// Like the LLM backends, this reuses the `LlmHttp` transport seam: the request
// body and response parse are deterministic and mock-testable; only the POST is
// non-deterministic. The Gemini branch (genai SDK `embed_content`) is deferred
// for the same reason its chat backend is — the SDK remaps the request into a
// REST `embedContent` body we can't reproduce faithfully without fixtures.
// ---------------------------------------------------------------------------

use serde_json::{Value, json};

use crate::llm::backends::gemini::DEFAULT_BASE_URL as GEMINI_DEFAULT_BASE_URL;
use crate::llm::http::{Credentials, LlmHttp, LlmHttpError};

/// The OpenAI SDK's default API base (already version-suffixed), used when no
/// `base_url` override is configured.
pub const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// An embedding-generation failure, mirroring the `ValueError`s raised in
/// `_EmbeddingClient.embed` plus the transport error from the POST.
#[derive(Debug, Clone)]
pub enum EmbedError {
    /// Query exceeds `max_embedding_tokens` (the cl100k token-count guard).
    TokenLimit { limit: usize, got: usize },
    /// The response carried no embedding vector.
    NoEmbedding,
    /// Vector length differed from the configured `vector_dimensions`.
    DimensionMismatch { expected: usize, got: usize },
    /// The underlying HTTP POST failed.
    Http(LlmHttpError),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::TokenLimit { limit, got } => write!(
                f,
                "Query exceeds maximum token limit of {limit} tokens (got {got} tokens)"
            ),
            EmbedError::NoEmbedding => write!(f, "No embedding returned from API"),
            EmbedError::DimensionMismatch { expected, got } => {
                write!(f, "Embedding dimension mismatch. Expected {expected}, got {got}.")
            }
            EmbedError::Http(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for EmbedError {}

/// cl100k token count of `query`, the same count `embed()` checks against
/// `max_embedding_tokens` before calling the provider.
pub fn embedding_token_count(query: &str) -> usize {
    tokenizer().encode_ordinary(query).len()
}

/// Build the OpenAI `/embeddings` request body, porting the `openai_kwargs`
/// assembly: `input` is always a one-element list; `dimensions` is sent only
/// when `dimensions` is `Some` (the `send_dimensions` flag).
pub fn build_openai_embedding_request(
    model: &str,
    query: &str,
    dimensions: Option<usize>,
) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), json!(model));
    body.insert("input".to_string(), json!([query]));
    if let Some(dimensions) = dimensions {
        body.insert("dimensions".to_string(), json!(dimensions));
    }
    Value::Object(body)
}

/// Extract `data[0].embedding` from an OpenAI embeddings response as `f32`s,
/// validating the length against `vector_dimensions` (the
/// `_validate_embedding_dimensions` check). An absent/empty vector is
/// [`EmbedError::NoEmbedding`].
pub fn parse_openai_embedding_response(
    response: &Value,
    vector_dimensions: usize,
) -> Result<Vec<f32>, EmbedError> {
    let values = response
        .get("data")
        .and_then(Value::as_array)
        .and_then(|data| data.first())
        .and_then(|first| first.get("embedding"))
        .and_then(Value::as_array)
        .ok_or(EmbedError::NoEmbedding)?;
    if values.is_empty() {
        return Err(EmbedError::NoEmbedding);
    }
    let embedding: Vec<f32> = values
        .iter()
        .filter_map(|value| value.as_f64().map(|f| f as f32))
        .collect();
    if embedding.len() != vector_dimensions {
        return Err(EmbedError::DimensionMismatch {
            expected: vector_dimensions,
            got: embedding.len(),
        });
    }
    Ok(embedding)
}

/// Embed a single query via OpenAI, porting `embed()`'s OpenAI branch: enforce
/// the token limit, POST `{base}/embeddings` with Bearer auth, then extract and
/// dimension-validate the vector. `send_dimensions` mirrors the config flag.
#[allow(clippy::too_many_arguments)]
pub async fn embed_openai<H: LlmHttp>(
    http: &H,
    credentials: &Credentials,
    model: &str,
    query: &str,
    vector_dimensions: usize,
    send_dimensions: bool,
    max_embedding_tokens: usize,
) -> Result<Vec<f32>, EmbedError> {
    let token_count = embedding_token_count(query);
    if token_count > max_embedding_tokens {
        return Err(EmbedError::TokenLimit {
            limit: max_embedding_tokens,
            got: token_count,
        });
    }
    let dimensions = send_dimensions.then_some(vector_dimensions);
    let body = build_openai_embedding_request(model, query, dimensions);
    let url = format!(
        "{}/embeddings",
        credentials.effective_base_url(OPENAI_DEFAULT_BASE_URL)
    );
    let headers = [(
        "Authorization".to_string(),
        format!("Bearer {}", credentials.api_key),
    )];
    let response = http
        .post_json(&url, &headers, &body)
        .await
        .map_err(EmbedError::Http)?;
    parse_openai_embedding_response(&response, vector_dimensions)
}

/// Build the OpenAI `/embeddings` request body for a batch of inputs (the
/// `input` field is the full list, embedded in one request).
pub fn build_openai_embedding_request_batch(
    model: &str,
    inputs: &[String],
    dimensions: Option<usize>,
) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), json!(model));
    body.insert(
        "input".to_string(),
        Value::Array(inputs.iter().map(|s| Value::String(s.clone())).collect()),
    );
    if let Some(dimensions) = dimensions {
        body.insert("dimensions".to_string(), json!(dimensions));
    }
    Value::Object(body)
}

/// Parse a batch embedding response, reassembling vectors into input order by
/// each item's `index` (the OpenAI API may return them out of order). Validates
/// the count and per-vector dimensionality.
pub fn parse_openai_embedding_response_batch(
    response: &Value,
    expected_count: usize,
    vector_dimensions: usize,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let data = response
        .get("data")
        .and_then(Value::as_array)
        .ok_or(EmbedError::NoEmbedding)?;
    if data.len() != expected_count {
        return Err(EmbedError::NoEmbedding);
    }
    let mut out: Vec<Option<Vec<f32>>> = vec![None; expected_count];
    for item in data {
        let index = item
            .get("index")
            .and_then(Value::as_u64)
            .ok_or(EmbedError::NoEmbedding)? as usize;
        if index >= expected_count {
            return Err(EmbedError::NoEmbedding);
        }
        let values = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or(EmbedError::NoEmbedding)?;
        let embedding: Vec<f32> = values
            .iter()
            .filter_map(|value| value.as_f64().map(|f| f as f32))
            .collect();
        if embedding.len() != vector_dimensions {
            return Err(EmbedError::DimensionMismatch {
                expected: vector_dimensions,
                got: embedding.len(),
            });
        }
        out[index] = Some(embedding);
    }
    out.into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(EmbedError::NoEmbedding)
}

/// Batch-embed `texts` in a single OpenAI request, mirroring
/// `embedding_client.simple_batch_embed`: validate each input's cl100k token
/// count against `max_embedding_tokens` up front (erroring on the first
/// violation, as Python raises `ValueError`), then embed and return vectors in
/// input order. An empty input yields an empty result.
///
/// Deviation: Python sub-batches across multiple requests to respect a
/// per-request token cap; this sends all inputs in one request. That is faithful
/// for the deriver/dreamer batch sizes (bounded well under the OpenAI
/// per-request limit); the per-input cap — the only correctness-relevant
/// validation — is preserved.
pub async fn embed_openai_batch<H: LlmHttp>(
    http: &H,
    credentials: &Credentials,
    model: &str,
    texts: &[String],
    vector_dimensions: usize,
    send_dimensions: bool,
    max_embedding_tokens: usize,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    for text in texts {
        let token_count = embedding_token_count(text);
        if token_count > max_embedding_tokens {
            return Err(EmbedError::TokenLimit {
                limit: max_embedding_tokens,
                got: token_count,
            });
        }
    }
    let dimensions = send_dimensions.then_some(vector_dimensions);
    let body = build_openai_embedding_request_batch(model, texts, dimensions);
    let url = format!(
        "{}/embeddings",
        credentials.effective_base_url(OPENAI_DEFAULT_BASE_URL)
    );
    let headers = [(
        "Authorization".to_string(),
        format!("Bearer {}", credentials.api_key),
    )];
    let response = http
        .post_json(&url, &headers, &body)
        .await
        .map_err(EmbedError::Http)?;
    parse_openai_embedding_response_batch(&response, texts.len(), vector_dimensions)
}

// ---------------------------------------------------------------------------
// Gemini embeddings (`:embedContent` / `:batchEmbedContents`), ported from
// `_EmbeddingClient.embed`'s Gemini branch. Like the Gemini chat backend, this
// targets the documented REST API directly rather than the genai SDK: the SDK's
// `embed_content(contents=query|[..], config={output_dimensionality})` maps to
// these endpoints. `output_dimensionality` is always sent (Python passes it
// unconditionally; there is no `send_dimensions` flag for Gemini). Parity is
// against the documented REST shape, not a live/SDK oracle.
// ---------------------------------------------------------------------------

/// Address the model as `models/{model}` (idempotent if already prefixed), the
/// same convention as the Gemini chat backend.
fn gemini_model_path(model: &str) -> String {
    if model.starts_with("models/") {
        model.to_string()
    } else {
        format!("models/{model}")
    }
}

/// Validate one Gemini `values` array against `vector_dimensions`, mirroring
/// `_validate_embedding_dimensions`. Empty → [`EmbedError::NoEmbedding`].
fn gemini_vector(values: &[Value], vector_dimensions: usize) -> Result<Vec<f32>, EmbedError> {
    if values.is_empty() {
        return Err(EmbedError::NoEmbedding);
    }
    let embedding: Vec<f32> = values
        .iter()
        .filter_map(|value| value.as_f64().map(|f| f as f32))
        .collect();
    if embedding.len() != vector_dimensions {
        return Err(EmbedError::DimensionMismatch {
            expected: vector_dimensions,
            got: embedding.len(),
        });
    }
    Ok(embedding)
}

/// Build a Gemini `:embedContent` request body.
pub fn build_gemini_embedding_request(
    model: &str,
    query: &str,
    vector_dimensions: usize,
) -> Value {
    json!({
        "model": gemini_model_path(model),
        "content": {"parts": [{"text": query}]},
        "output_dimensionality": vector_dimensions,
    })
}

/// Extract `embedding.values` from a Gemini `:embedContent` response.
pub fn parse_gemini_embedding_response(
    response: &Value,
    vector_dimensions: usize,
) -> Result<Vec<f32>, EmbedError> {
    let values = response
        .get("embedding")
        .and_then(|embedding| embedding.get("values"))
        .and_then(Value::as_array)
        .ok_or(EmbedError::NoEmbedding)?;
    gemini_vector(values, vector_dimensions)
}

/// Embed a single query via Gemini, porting `embed()`'s Gemini branch: enforce
/// the token limit, POST `{base}/v1beta/models/{model}:embedContent` with
/// `x-goog-api-key` auth, then extract + dimension-validate the vector.
pub async fn embed_gemini<H: LlmHttp>(
    http: &H,
    credentials: &Credentials,
    model: &str,
    query: &str,
    vector_dimensions: usize,
    max_embedding_tokens: usize,
) -> Result<Vec<f32>, EmbedError> {
    let token_count = embedding_token_count(query);
    if token_count > max_embedding_tokens {
        return Err(EmbedError::TokenLimit {
            limit: max_embedding_tokens,
            got: token_count,
        });
    }
    let body = build_gemini_embedding_request(model, query, vector_dimensions);
    let url = format!(
        "{}/v1beta/{}:embedContent",
        credentials.effective_base_url(GEMINI_DEFAULT_BASE_URL),
        gemini_model_path(model)
    );
    let headers = [("x-goog-api-key".to_string(), credentials.api_key.clone())];
    let response = http
        .post_json(&url, &headers, &body)
        .await
        .map_err(EmbedError::Http)?;
    parse_gemini_embedding_response(&response, vector_dimensions)
}

/// Build a Gemini `:batchEmbedContents` request body (one `requests` entry per
/// input, in order).
pub fn build_gemini_embedding_request_batch(
    model: &str,
    inputs: &[String],
    vector_dimensions: usize,
) -> Value {
    let requests: Vec<Value> = inputs
        .iter()
        .map(|text| {
            json!({
                "model": gemini_model_path(model),
                "content": {"parts": [{"text": text}]},
                "output_dimensionality": vector_dimensions,
            })
        })
        .collect();
    json!({ "requests": requests })
}

/// Parse a Gemini `:batchEmbedContents` response: `embeddings` is returned in
/// request order (no `index` field), one `values` array per input.
pub fn parse_gemini_embedding_response_batch(
    response: &Value,
    expected_count: usize,
    vector_dimensions: usize,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let embeddings = response
        .get("embeddings")
        .and_then(Value::as_array)
        .ok_or(EmbedError::NoEmbedding)?;
    if embeddings.len() != expected_count {
        return Err(EmbedError::NoEmbedding);
    }
    embeddings
        .iter()
        .map(|item| {
            let values = item
                .get("values")
                .and_then(Value::as_array)
                .ok_or(EmbedError::NoEmbedding)?;
            gemini_vector(values, vector_dimensions)
        })
        .collect()
}

/// Batch-embed `texts` via Gemini `:batchEmbedContents`, mirroring
/// `simple_batch_embed`'s Gemini branch (per-input token cap, one request).
pub async fn embed_gemini_batch<H: LlmHttp>(
    http: &H,
    credentials: &Credentials,
    model: &str,
    texts: &[String],
    vector_dimensions: usize,
    max_embedding_tokens: usize,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    for text in texts {
        let token_count = embedding_token_count(text);
        if token_count > max_embedding_tokens {
            return Err(EmbedError::TokenLimit {
                limit: max_embedding_tokens,
                got: token_count,
            });
        }
    }
    let body = build_gemini_embedding_request_batch(model, texts, vector_dimensions);
    let url = format!(
        "{}/v1beta/{}:batchEmbedContents",
        credentials.effective_base_url(GEMINI_DEFAULT_BASE_URL),
        gemini_model_path(model)
    );
    let headers = [("x-goog-api-key".to_string(), credentials.api_key.clone())];
    let response = http
        .post_json(&url, &headers, &body)
        .await
        .map_err(EmbedError::Http)?;
    parse_gemini_embedding_response_batch(&response, texts.len(), vector_dimensions)
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

    #[test]
    fn openai_embedding_request_sends_dimensions_only_when_requested() {
        assert_eq!(
            build_openai_embedding_request("text-embedding-3-small", "hi", Some(1536)),
            json!({"model": "text-embedding-3-small", "input": ["hi"], "dimensions": 1536})
        );
        assert_eq!(
            build_openai_embedding_request("text-embedding-3-small", "hi", None),
            json!({"model": "text-embedding-3-small", "input": ["hi"]})
        );
    }

    #[test]
    fn batch_request_includes_all_inputs() {
        let inputs = vec!["a".to_string(), "b".to_string()];
        assert_eq!(
            build_openai_embedding_request_batch("text-embedding-3-small", &inputs, Some(1536)),
            json!({"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 1536})
        );
    }

    #[test]
    fn gemini_embedding_request_and_parse() {
        // Single: model is prefixed and output_dimensionality is always sent.
        assert_eq!(
            build_gemini_embedding_request("gemini-embedding-001", "hi", 768),
            json!({
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "hi"}]},
                "output_dimensionality": 768
            })
        );
        // Already-prefixed model is left as-is.
        assert_eq!(
            build_gemini_embedding_request("models/text-embedding-004", "x", 3)["model"],
            json!("models/text-embedding-004")
        );
        // Single parse pulls embedding.values and validates dimensionality.
        let response = json!({"embedding": {"values": [0.1, 0.2, 0.3]}});
        assert_eq!(
            parse_gemini_embedding_response(&response, 3).unwrap(),
            vec![0.1_f32, 0.2, 0.3]
        );
        assert!(matches!(
            parse_gemini_embedding_response(&response, 2),
            Err(EmbedError::DimensionMismatch { expected: 2, got: 3 })
        ));
    }

    #[test]
    fn gemini_batch_request_and_parse_preserve_order() {
        let inputs = vec!["a".to_string(), "b".to_string()];
        let body = build_gemini_embedding_request_batch("gemini-embedding-001", &inputs, 2);
        assert_eq!(body["requests"].as_array().map(Vec::len), Some(2));
        assert_eq!(body["requests"][1]["content"]["parts"][0]["text"], json!("b"));

        // Batch responses are in request order (no index field).
        let response = json!({"embeddings": [
            {"values": [0.1, 0.2]},
            {"values": [0.3, 0.4]},
        ]});
        assert_eq!(
            parse_gemini_embedding_response_batch(&response, 2, 2).unwrap(),
            vec![vec![0.1_f32, 0.2], vec![0.3_f32, 0.4]]
        );
        // A count mismatch is a NoEmbedding error.
        assert!(matches!(
            parse_gemini_embedding_response_batch(&response, 3, 2),
            Err(EmbedError::NoEmbedding)
        ));
    }

    #[test]
    fn batch_parse_reorders_by_index() {
        // Returned out of order; reassembled into input order.
        let response = json!({"data": [
            {"embedding": [0.3, 0.4], "index": 1},
            {"embedding": [0.1, 0.2], "index": 0},
        ]});
        assert_eq!(
            parse_openai_embedding_response_batch(&response, 2, 2).unwrap(),
            vec![vec![0.1_f32, 0.2], vec![0.3_f32, 0.4]]
        );
    }

    #[test]
    fn batch_parse_rejects_count_and_dimension_mismatch() {
        let response = json!({"data": [{"embedding": [0.1, 0.2], "index": 0}]});
        assert!(matches!(
            parse_openai_embedding_response_batch(&response, 2, 2),
            Err(EmbedError::NoEmbedding)
        ));
        assert!(matches!(
            parse_openai_embedding_response_batch(&response, 1, 3),
            Err(EmbedError::DimensionMismatch { expected: 3, got: 2 })
        ));
    }

    #[test]
    fn parse_embedding_extracts_first_vector_and_validates_length() {
        let response = json!({"data": [{"embedding": [0.1, 0.2, 0.3]}]});
        assert_eq!(
            parse_openai_embedding_response(&response, 3).unwrap(),
            vec![0.1_f32, 0.2, 0.3]
        );
        // Wrong length -> mismatch.
        assert!(matches!(
            parse_openai_embedding_response(&response, 4),
            Err(EmbedError::DimensionMismatch { expected: 4, got: 3 })
        ));
        // No data -> NoEmbedding.
        assert!(matches!(
            parse_openai_embedding_response(&json!({"data": []}), 3),
            Err(EmbedError::NoEmbedding)
        ));
    }

    #[tokio::test]
    async fn embed_openai_posts_to_embeddings_endpoint() {
        use crate::llm::http::mock::MockHttp;

        let http = MockHttp::ok(json!({"data": [{"embedding": [1.0, 2.0]}]}));
        let vector = embed_openai(
            &http,
            &Credentials::new("sk-e"),
            "text-embedding-3-small",
            "hello",
            2,
            true,
            8192,
        )
        .await
        .unwrap();

        assert_eq!(http.last_url(), "https://api.openai.com/v1/embeddings");
        assert_eq!(
            http.last_headers(),
            vec![("Authorization".to_string(), "Bearer sk-e".to_string())]
        );
        assert_eq!(
            http.last_body(),
            json!({"model": "text-embedding-3-small", "input": ["hello"], "dimensions": 2})
        );
        assert_eq!(vector, vec![1.0_f32, 2.0]);
    }

    #[tokio::test]
    async fn embed_openai_enforces_token_limit_before_posting() {
        use crate::llm::http::mock::MockHttp;

        // A 1-token cap with a multi-token query trips the guard without a POST.
        let http = MockHttp::ok(json!({"data": [{"embedding": [1.0]}]}));
        let err = embed_openai(
            &http,
            &Credentials::new("sk-e"),
            "text-embedding-3-small",
            "several words exceed the cap",
            1,
            false,
            1,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, EmbedError::TokenLimit { limit: 1, .. }));
        // No request was made.
        assert!(http.captured.lock().unwrap().is_none());
    }
}
