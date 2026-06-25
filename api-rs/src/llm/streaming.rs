//! The deterministic core of SSE streaming, ported from the `stream()` methods of
//! `src/llm/backends/{anthropic,openai}.py`.
//!
//! Python drives streaming through the provider SDKs, which frame the raw
//! Server-Sent-Events wire format and hand back typed chunk objects. This port
//! speaks the raw SSE wire format directly (like the non-streaming backends), so
//! the reusable pieces are:
//!
//! - [`SseBuffer`] — frames a byte/text stream into complete events and yields
//!   each event's `data:` payload (handling cross-chunk boundaries + the OpenAI
//!   `[DONE]` sentinel).
//! - [`AnthropicStreamDecoder`] / [`OpenAiStreamDecoder`] — turn one decoded
//!   `data:` JSON value into zero or more [`StreamChunk`]s, accumulating the
//!   per-stream `finish_reason`/`output_tokens` state the SDKs surface at the end.
//!
//! These mirror the SDK chunk handling exactly (Anthropic `content_block_delta`
//! → `delta.text`; OpenAI `choices[0].delta.content` + the trailing usage chunk).
//! The live `reqwest` streaming transport, the `stream_final` tool-loop path
//! (re-issue the settled final turn as a streaming call — `tool_loop.py`
//! `stream_final_response`), and the axum SSE route are the remaining wiring;
//! Gemini streaming is not ported (non-default provider).

use std::collections::VecDeque;

use futures::Stream;
use serde_json::Value;

use super::Provider;
use super::http::LlmHttpError;

/// One streamed delta, mirroring `backend.StreamChunk`. A content chunk carries
/// text; the terminal chunk has `is_done = true` and the final reason/usage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamChunk {
    pub content: String,
    pub is_done: bool,
    pub finish_reason: Option<String>,
    pub output_tokens: Option<i64>,
}

impl StreamChunk {
    /// A text delta.
    pub fn content(text: impl Into<String>) -> Self {
        Self {
            content: text.into(),
            ..Self::default()
        }
    }

    /// The terminal chunk.
    pub fn done(finish_reason: Option<String>, output_tokens: Option<i64>) -> Self {
        Self {
            content: String::new(),
            is_done: true,
            finish_reason,
            output_tokens,
        }
    }
}

/// The `data:` payload of one SSE event, or the OpenAI end-of-stream sentinel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsePayload {
    /// A `data:` payload (the concatenated data lines of one event).
    Data(String),
    /// The `[DONE]` sentinel that OpenAI sends to close the stream.
    Done,
}

/// Frames a Server-Sent-Events stream. Bytes/text arrive in arbitrary chunks
/// (network framing has no relation to event boundaries), so [`push`](Self::push)
/// buffers and emits a payload only once a full event (terminated by a blank
/// line) has been seen.
#[derive(Debug, Default)]
pub struct SseBuffer {
    buffer: String,
}

impl SseBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next text chunk; return the payloads of any events that became
    /// complete. Events are separated by a blank line; within an event, `data:`
    /// lines are concatenated with `\n` (the SSE spec). A lone `[DONE]` data
    /// payload becomes [`SsePayload::Done`].
    pub fn push(&mut self, chunk: &str) -> Vec<SsePayload> {
        // Normalize CRLF so the blank-line split is uniform.
        self.buffer.push_str(&chunk.replace("\r\n", "\n"));
        let mut payloads = Vec::new();
        while let Some(index) = self.buffer.find("\n\n") {
            let event = self.buffer[..index].to_string();
            self.buffer.drain(..index + 2);
            if let Some(payload) = Self::event_data(&event) {
                payloads.push(payload);
            }
        }
        payloads
    }

    /// Extract the `data:` payload of one event block, joining multiple `data:`
    /// lines with `\n` and mapping `[DONE]` to the sentinel. Returns `None` when
    /// the event carries no data (e.g. a bare `event:`/comment line).
    fn event_data(event: &str) -> Option<SsePayload> {
        let mut data = String::new();
        let mut saw_data = false;
        for line in event.lines() {
            if let Some(rest) = line.strip_prefix("data:") {
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                if saw_data {
                    data.push('\n');
                }
                data.push_str(rest);
                saw_data = true;
            }
        }
        if !saw_data {
            return None;
        }
        if data.trim() == "[DONE]" {
            Some(SsePayload::Done)
        } else {
            Some(SsePayload::Data(data))
        }
    }
}

/// Decodes Anthropic Messages-API stream events, porting the `stream()` handler:
/// `content_block_delta` events with a `delta.text` yield a content chunk;
/// `message_delta` accumulates the final `stop_reason` + `usage.output_tokens`;
/// `message_stop` emits the terminal chunk.
#[derive(Debug, Default)]
pub struct AnthropicStreamDecoder {
    finish_reason: Option<String>,
    output_tokens: Option<i64>,
}

impl AnthropicStreamDecoder {
    pub fn push(&mut self, data: &Value) -> Vec<StreamChunk> {
        match data.get("type").and_then(Value::as_str) {
            Some("content_block_delta") => {
                // Only text_delta blocks have `delta.text`; thinking_delta carries
                // `delta.thinking`, which the Python handler also ignores here.
                if let Some(text) = data
                    .get("delta")
                    .and_then(|delta| delta.get("text"))
                    .and_then(Value::as_str)
                {
                    return vec![StreamChunk::content(text)];
                }
            }
            Some("message_delta") => {
                if let Some(reason) = data
                    .get("delta")
                    .and_then(|delta| delta.get("stop_reason"))
                    .and_then(Value::as_str)
                {
                    self.finish_reason = Some(reason.to_string());
                }
                if let Some(tokens) = data
                    .get("usage")
                    .and_then(|usage| usage.get("output_tokens"))
                    .and_then(Value::as_i64)
                {
                    self.output_tokens = Some(tokens);
                }
            }
            Some("message_stop") => {
                return vec![StreamChunk::done(
                    self.finish_reason.clone(),
                    self.output_tokens,
                )];
            }
            _ => {}
        }
        Vec::new()
    }

    /// Anthropic emits its terminal chunk on `message_stop`, so end-of-stream
    /// adds nothing.
    pub fn finalize(&mut self) -> Vec<StreamChunk> {
        Vec::new()
    }
}

/// Decodes OpenAI Chat-Completions stream chunks, porting the `stream()` handler:
/// `choices[0].delta.content` yields a content chunk; `choices[0].finish_reason`
/// is accumulated; the trailing usage-bearing chunk emits the terminal chunk with
/// `usage.completion_tokens` (requested via `stream_options.include_usage`).
#[derive(Debug, Default)]
pub struct OpenAiStreamDecoder {
    finish_reason: Option<String>,
}

impl OpenAiStreamDecoder {
    pub fn push(&mut self, data: &Value) -> Vec<StreamChunk> {
        let mut out = Vec::new();
        let choice = data
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first());

        if let Some(content) = choice
            .and_then(|choice| choice.get("delta"))
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        {
            out.push(StreamChunk::content(content));
        }
        if let Some(reason) = choice
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(Value::as_str)
        {
            self.finish_reason = Some(reason.to_string());
        }
        if let Some(usage) = data.get("usage").filter(|usage| !usage.is_null()) {
            out.push(StreamChunk::done(
                self.finish_reason.clone(),
                usage.get("completion_tokens").and_then(Value::as_i64),
            ));
        }
        out
    }

    /// OpenAI emits its terminal chunk on the trailing usage chunk, so
    /// end-of-stream adds nothing.
    pub fn finalize(&mut self) -> Vec<StreamChunk> {
        Vec::new()
    }
}

/// Decodes Gemini `streamGenerateContent` chunks, porting the `stream()` handler:
/// each chunk yields its joined candidate text; the terminal `finish_reason` +
/// `candidates_token_count` come from the *last* chunk and are emitted at
/// end-of-stream (Gemini has no explicit done event), via [`finalize`](Self::finalize).
#[derive(Debug, Default)]
pub struct GeminiStreamDecoder {
    finish_reason: Option<String>,
    output_tokens: Option<i64>,
}

impl GeminiStreamDecoder {
    pub fn push(&mut self, data: &Value) -> Vec<StreamChunk> {
        let candidate = data
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first());

        // Join this chunk's text parts (the SDK's `chunk.text` convenience).
        let mut text = String::new();
        if let Some(parts) = candidate
            .and_then(|candidate| candidate.get("content"))
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(piece) = part.get("text").and_then(Value::as_str) {
                    text.push_str(piece);
                }
            }
        }

        // Accumulate the terminal info (last writer wins).
        if let Some(reason) = candidate
            .and_then(|candidate| dual_field(candidate, "finish_reason", "finishReason"))
            .and_then(Value::as_str)
        {
            self.finish_reason = Some(reason.to_string());
        }
        if let Some(tokens) = dual_field(data, "usage_metadata", "usageMetadata")
            .and_then(|usage| dual_field(usage, "candidates_token_count", "candidatesTokenCount"))
            .and_then(Value::as_i64)
        {
            self.output_tokens = Some(tokens);
        }

        if text.is_empty() {
            Vec::new()
        } else {
            vec![StreamChunk::content(text)]
        }
    }

    /// Emit the terminal chunk from the accumulated last-chunk metadata.
    pub fn finalize(&mut self) -> Vec<StreamChunk> {
        vec![StreamChunk::done(
            self.finish_reason.clone().or(Some("stop".to_string())),
            self.output_tokens,
        )]
    }
}

/// Read a field under its proto snake_case or camelCase name (Gemini REST
/// responses are camelCase; SDK-shaped fixtures are snake_case).
fn dual_field<'a>(value: &'a Value, snake: &str, camel: &str) -> Option<&'a Value> {
    value.get(snake).or_else(|| value.get(camel))
}

/// Provider-dispatched stream decoder, selected by transport.
enum StreamDecoder {
    Anthropic(AnthropicStreamDecoder),
    OpenAi(OpenAiStreamDecoder),
    Gemini(GeminiStreamDecoder),
}

impl StreamDecoder {
    fn for_provider(provider: Provider) -> Self {
        match provider {
            Provider::Anthropic => StreamDecoder::Anthropic(AnthropicStreamDecoder::default()),
            Provider::Openai => StreamDecoder::OpenAi(OpenAiStreamDecoder::default()),
            Provider::Gemini => StreamDecoder::Gemini(GeminiStreamDecoder::default()),
        }
    }

    fn push(&mut self, data: &Value) -> Vec<StreamChunk> {
        match self {
            StreamDecoder::Anthropic(decoder) => decoder.push(data),
            StreamDecoder::OpenAi(decoder) => decoder.push(data),
            StreamDecoder::Gemini(decoder) => decoder.push(data),
        }
    }

    fn finalize(&mut self) -> Vec<StreamChunk> {
        match self {
            StreamDecoder::Anthropic(decoder) => decoder.finalize(),
            StreamDecoder::OpenAi(decoder) => decoder.finalize(),
            StreamDecoder::Gemini(decoder) => decoder.finalize(),
        }
    }
}

/// Turn a raw text-chunk transport stream (the SSE body, arriving in arbitrary
/// network chunks) into a stream of [`StreamChunk`]s for `provider`: frame events
/// with [`SseBuffer`], parse each `data:` payload as JSON, and run the matching
/// decoder. A transport error ends the stream after surfacing it; a `[DONE]`
/// sentinel or end-of-stream triggers the decoder's [`finalize`](StreamDecoder::finalize).
pub fn decode_stream<S>(
    provider: Provider,
    text_stream: S,
) -> impl Stream<Item = Result<StreamChunk, LlmHttpError>>
where
    S: Stream<Item = Result<String, LlmHttpError>> + Unpin + Send,
{
    use futures::StreamExt;

    struct State<S> {
        text_stream: S,
        buffer: SseBuffer,
        decoder: StreamDecoder,
        pending: VecDeque<StreamChunk>,
        ended: bool,
    }

    let state = State {
        text_stream,
        buffer: SseBuffer::new(),
        decoder: StreamDecoder::for_provider(provider),
        pending: VecDeque::new(),
        ended: false,
    };

    futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(chunk) = state.pending.pop_front() {
                return Some((Ok(chunk), state));
            }
            if state.ended {
                return None;
            }
            match state.text_stream.next().await {
                Some(Ok(text)) => {
                    for payload in state.buffer.push(&text) {
                        match payload {
                            SsePayload::Data(data) => {
                                if let Ok(value) = serde_json::from_str::<Value>(&data) {
                                    state.pending.extend(state.decoder.push(&value));
                                }
                            }
                            SsePayload::Done => {
                                state.ended = true;
                                state.pending.extend(state.decoder.finalize());
                            }
                        }
                    }
                }
                Some(Err(error)) => {
                    state.ended = true;
                    return Some((Err(error), state));
                }
                None => {
                    // End-of-stream without a sentinel (e.g. Anthropic/Gemini):
                    // let the decoder emit any terminal chunk it accumulated.
                    state.ended = true;
                    state.pending.extend(state.decoder.finalize());
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sse_buffer_frames_events_across_chunk_boundaries() {
        let mut buffer = SseBuffer::new();
        // First network chunk: one complete event + the start of a second.
        assert_eq!(
            buffer.push("data: {\"a\":1}\n\ndata: {\"b"),
            vec![SsePayload::Data("{\"a\":1}".to_string())]
        );
        // Second chunk completes the second event and adds the [DONE] sentinel.
        let payloads = buffer.push("\":2}\n\ndata: [DONE]\n\n");
        assert_eq!(
            payloads,
            vec![
                SsePayload::Data("{\"b\":2}".to_string()),
                SsePayload::Done,
            ]
        );
    }

    #[test]
    fn sse_buffer_ignores_event_and_comment_lines_and_joins_data() {
        let mut buffer = SseBuffer::new();
        let payloads = buffer.push("event: message\ndata: line1\ndata: line2\n\n");
        assert_eq!(
            payloads,
            vec![SsePayload::Data("line1\nline2".to_string())]
        );
    }

    #[test]
    fn anthropic_decoder_yields_text_then_terminal_chunk() {
        let mut decoder = AnthropicStreamDecoder::default();
        assert_eq!(
            decoder.push(&json!({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"}
            })),
            vec![StreamChunk::content("Hello")]
        );
        // A thinking delta carries no `delta.text` → no chunk.
        assert!(
            decoder
                .push(&json!({
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "hmm"}
                }))
                .is_empty()
        );
        // message_delta accumulates stop_reason + output tokens (no chunk yet).
        assert!(
            decoder
                .push(&json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 7}
                }))
                .is_empty()
        );
        assert_eq!(
            decoder.push(&json!({"type": "message_stop"})),
            vec![StreamChunk::done(Some("end_turn".to_string()), Some(7))]
        );
    }

    #[test]
    fn gemini_decoder_emits_terminal_on_finalize() {
        let mut decoder = GeminiStreamDecoder::default();
        assert_eq!(
            decoder.push(&json!({"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]})),
            vec![StreamChunk::content("Hi")]
        );
        // Last chunk carries finishReason + usage; no content.
        assert!(
            decoder
                .push(&json!({
                    "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
                    "usageMetadata": {"candidatesTokenCount": 9}
                }))
                .is_empty()
        );
        assert_eq!(
            decoder.finalize(),
            vec![StreamChunk::done(Some("STOP".to_string()), Some(9))]
        );
    }

    #[tokio::test]
    async fn decode_stream_frames_openai_sse_into_chunks() {
        use futures::StreamExt;

        // Two network chunks; the event boundary falls mid-frame.
        let text_chunks = vec![
            Ok("data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\ndata: {\"choi".to_string()),
            Ok(
                "ces\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":4}}\n\ndata: [DONE]\n\n"
                    .to_string(),
            ),
        ];
        let chunks: Vec<_> = decode_stream(Provider::Openai, futures::stream::iter(text_chunks))
            .map(|result| result.unwrap())
            .collect()
            .await;
        assert_eq!(
            chunks,
            vec![
                StreamChunk::content("Hel"),
                StreamChunk::content("lo"),
                StreamChunk::done(Some("stop".to_string()), Some(4)),
            ]
        );
    }

    #[test]
    fn openai_decoder_yields_content_and_trailing_usage() {
        let mut decoder = OpenAiStreamDecoder::default();
        assert_eq!(
            decoder.push(&json!({"choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]})),
            vec![StreamChunk::content("Hi")]
        );
        // finish_reason arrives on a content-less chunk; accumulated, not emitted.
        assert!(
            decoder
                .push(&json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
                .is_empty()
        );
        // The trailing usage chunk (empty choices) emits the terminal chunk.
        assert_eq!(
            decoder.push(&json!({"choices": [], "usage": {"completion_tokens": 5}})),
            vec![StreamChunk::done(Some("stop".to_string()), Some(5))]
        );
    }
}
