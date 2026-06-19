//! Deterministic formatting helpers for the dialectic tool layer, ported from
//! `src/utils/agent_tools.py` + `src/utils/formatting.py`.
//!
//! These are the network-free, byte-exact output side of the dialectic tools
//! (the agent loop, tool dispatch, and LLM executor that *call* them are the
//! larger unported phase). All length math is on **Unicode code points** to
//! match Python `str` `len`/slicing semantics.
//!
//! Char-cap defaults are hardcoded constants (Python reads env-overridable
//! `settings.LLM.*`; same simplification as other ported defaults).

use std::future::Future;

use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use serde_json::Value;
use sqlx::PgPool;

use crate::db::{self, MessageSnippet, ReasoningDocument};
use crate::llm::tool_loop::ToolExecutor;

/// The dialectic-relevant subset of Python's `ToolContext` — the peer/session
/// scope the read-only tool handlers thread into their data-layer calls.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub workspace_name: String,
    pub observer: String,
    pub observed: String,
    pub session_name: Option<String>,
}

/// `settings.LLM.MAX_TOOL_OUTPUT_CHARS` default.
pub const MAX_TOOL_OUTPUT_CHARS: usize = 10_000;
/// `settings.LLM.MAX_MESSAGE_CONTENT_CHARS` default.
pub const MAX_MESSAGE_CONTENT_CHARS: usize = 2_000;

/// Port of `_safe_int`: coerce a tool-input JSON value to an integer, returning
/// `default` on failure (LLMs sometimes pass non-numeric strings). Mirrors
/// Python `int(value)`: integers as-is, floats truncated toward zero, booleans
/// to 0/1, plain base-10 integer strings parsed (whitespace-trimmed), everything
/// else (null/array/object/malformed string) → `default`.
pub fn safe_int(value: &Value, default: i64) -> i64 {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|float| float.trunc() as i64))
            .unwrap_or(default),
        Value::Bool(flag) => i64::from(*flag),
        Value::String(text) => text.trim().parse::<i64>().unwrap_or(default),
        _ => default,
    }
}

/// Python `f"{n:,}"` — group digits in threes with commas.
fn comma_grouped(n: usize) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len + len / 3);
    for (index, byte) in bytes.iter().enumerate() {
        if index > 0 && (len - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*byte as char);
    }
    out
}

/// Port of `_truncate_tool_output`: clamp `output` to `max_chars` code points,
/// appending a comma-grouped truncation notice. Returns
/// `(text, original_chars, was_truncated)` — the truncation signal the caller
/// threads into the tool-result metadata.
pub fn truncate_tool_output(output: &str, max_chars: usize) -> (String, usize, bool) {
    let chars: Vec<char> = output.chars().collect();
    let original = chars.len();
    if original <= max_chars {
        return (output.to_string(), original, false);
    }
    let head: String = chars[..max_chars].iter().collect();
    let truncated = format!(
        "{head}\n\n[OUTPUT TRUNCATED - showing {} of {} characters]",
        comma_grouped(max_chars),
        comma_grouped(original)
    );
    (truncated, original, true)
}

/// Port of `_truncate_message_content`: simple head truncation of a single
/// message's content to `max_chars` code points (appending `...` when clamped).
pub fn truncate_message_content(content: &str, max_chars: usize) -> String {
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= max_chars {
        return content.to_string();
    }
    let head: String = chars[..max_chars].iter().collect();
    format!("{head}...")
}

/// First char of a char's lowercase expansion, keeping a 1:1 code-point mapping
/// so match indices stay aligned. Exact for ASCII (the realistic grep case);
/// approximates Python `re.IGNORECASE` only for the rare chars whose case fold
/// expands to multiple code points.
fn lower1(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

/// Case-insensitive literal search over a code-point slice. Returns the match's
/// `(start_index, char_len)`; a literal pattern matches `pattern.chars().count()`
/// code points.
fn find_case_insensitive(haystack: &[char], pattern: &str) -> Option<(usize, usize)> {
    let needle: Vec<char> = pattern.chars().map(lower1).collect();
    if needle.is_empty() {
        return Some((0, 0));
    }
    if needle.len() > haystack.len() {
        return None;
    }
    let hay: Vec<char> = haystack.iter().map(|&c| lower1(c)).collect();
    (0..=hay.len() - needle.len()).find_map(|start| {
        if hay[start..start + needle.len()] == needle[..] {
            Some((start, needle.len()))
        } else {
            None
        }
    })
}

/// Port of `_extract_pattern_snippet`: for grep/exact-text results, extract a
/// `max_chars`-code-point window centered on the first case-insensitive match
/// of `pattern` (literal), with `...` prefix/suffix when the window doesn't
/// reach the content edges. Content within the cap is returned unchanged; no
/// match → the head of the content. Window math mirrors Python's floor
/// division and boundary adjustment exactly.
pub fn extract_pattern_snippet(content: &str, pattern: &str, max_chars: usize) -> String {
    let chars: Vec<char> = content.chars().collect();
    let n = chars.len() as i64;
    if chars.len() <= max_chars {
        return content.to_string();
    }

    let Some((match_start, match_len)) = find_case_insensitive(&chars, pattern) else {
        let head: String = chars[..max_chars].iter().collect();
        return format!("{head}...");
    };

    let max = max_chars as i64;
    let match_start = match_start as i64;
    let match_end = match_start + match_len as i64;
    let remaining = max - match_len as i64;
    let before = remaining.div_euclid(2);
    let after = remaining - before;

    let mut start = (match_start - before).max(0);
    let mut end = (match_end + after).min(n);
    if start == 0 {
        end = n.min(max);
    } else if end == n {
        start = (n - max).max(0);
    }

    let snippet: String = chars[start as usize..end as usize].iter().collect();
    let prefix = if start > 0 { "..." } else { "" };
    let suffix = if end < n { "..." } else { "" };
    format!("{prefix}{snippet}{suffix}")
}

/// Port of `format_new_turn_with_timestamp`: `"YYYY-MM-DD HH:MM:SS speaker: text"`.
/// The timestamp is rendered in UTC (matching the timezone-aware `created_at`
/// values stored in the DB; Python's `strftime` prints the datetime's own
/// components without conversion).
pub fn format_new_turn_with_timestamp(
    new_turn: &str,
    current_time: DateTime<Utc>,
    speaker: &str,
) -> String {
    format!(
        "{} {speaker}: {new_turn}",
        current_time.format("%Y-%m-%d %H:%M:%S")
    )
}

/// A string field from a message JSON `Value` (the shape produced by the
/// data-layer `message`/`snippet` serializers), empty if absent.
fn message_str<'a>(message: &'a Value, key: &str) -> &'a str {
    message.get(key).and_then(Value::as_str).unwrap_or("")
}

/// The `created_at` of a message JSON `Value`, parsed back from its RFC3339
/// serialization (lossless at the second precision the formatter renders).
fn message_created_at(message: &Value) -> DateTime<Utc> {
    message
        .get("created_at")
        .and_then(Value::as_str)
        .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).expect("epoch is valid"))
}

/// One timestamped turn line for `rendered` content (already truncated/extracted)
/// of a message JSON `Value`.
fn turn_line(message: &Value, rendered: String) -> String {
    format_new_turn_with_timestamp(
        &rendered,
        message_created_at(message),
        message_str(message, "peer_id"),
    )
}

/// `"--- Snippet i (session: X, N match(es)) ---\n" + lines`, where each line is
/// a timestamped turn produced by `render` over the snippet's context messages.
fn format_snippet_block(
    index: usize,
    snippet: &MessageSnippet,
    render: impl Fn(&Value) -> String,
) -> String {
    let lines: Vec<String> = snippet
        .context
        .iter()
        .map(|message| turn_line(message, render(message)))
        .collect();
    let session = snippet
        .context
        .first()
        .map(|message| message_str(message, "session_id"))
        .unwrap_or("unknown");
    format!(
        "--- Snippet {index} (session: {session}, {} match(es)) ---\n{}",
        snippet.matched.len(),
        lines.join("\n")
    )
}

fn total_matches(snippets: &[MessageSnippet]) -> usize {
    snippets.iter().map(|snippet| snippet.matched.len()).sum()
}

/// Port of `_format_message_snippets` (the dialectic `search_messages` /
/// `search_messages_temporal` result formatter): each context message is head-
/// truncated via [`truncate_message_content`]. Returns the
/// [`truncate_tool_output`]-clamped string (matching Python's
/// `_truncate_tool_output(output)[0]`). `desc` is the trailing qualifier, e.g.
/// `"for query 'foo'"`.
pub fn format_message_snippets(snippets: &[MessageSnippet], desc: &str) -> String {
    let blocks: Vec<String> = snippets
        .iter()
        .enumerate()
        .map(|(i, snippet)| {
            format_snippet_block(i + 1, snippet, |message| {
                truncate_message_content(message_str(message, "content"), MAX_MESSAGE_CONTENT_CHARS)
            })
        })
        .collect();
    let output = format!(
        "Found {} matching messages in {} conversation snippets {desc}:\n\n{}",
        total_matches(snippets),
        snippets.len(),
        blocks.join("\n\n")
    );
    truncate_tool_output(&output, MAX_TOOL_OUTPUT_CHARS).0
}

/// Port of the `grep_messages` result formatter: each context message is shown
/// via [`extract_pattern_snippet`] around `text` (not simple head truncation),
/// and the header counts matches "containing 'text'". Returns the
/// [`truncate_tool_output`] triple (matching `_maybe_truncated_result`).
pub fn format_grep_result(text: &str, snippets: &[MessageSnippet]) -> (String, usize, bool) {
    let blocks: Vec<String> = snippets
        .iter()
        .enumerate()
        .map(|(i, snippet)| {
            format_snippet_block(i + 1, snippet, |message| {
                extract_pattern_snippet(
                    message_str(message, "content"),
                    text,
                    MAX_MESSAGE_CONTENT_CHARS,
                )
            })
        })
        .collect();
    let output = format!(
        "Found {} messages containing '{text}' in {} conversation snippets:\n\n{}",
        total_matches(snippets),
        snippets.len(),
        blocks.join("\n\n")
    );
    truncate_tool_output(&output, MAX_TOOL_OUTPUT_CHARS)
}

/// Port of the shared message-list join used by the `get_messages_by_date_range`
/// and `get_observation_context` result formatters: one head-truncated,
/// timestamped turn per message (newline-joined). Empty input → empty string.
pub fn format_message_list(messages: &[Value]) -> String {
    messages
        .iter()
        .map(|message| {
            turn_line(
                message,
                truncate_message_content(message_str(message, "content"), MAX_MESSAGE_CONTENT_CHARS),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `" - [id:X] (level): content"` — one premise/source/child line.
fn reasoning_doc_line(doc: &ReasoningDocument) -> String {
    format!(" - [id:{}] ({}): {}", doc.id, doc.level, doc.content)
}

/// Port of the `get_reasoning_chain` tool handler: fetch the observation, then
/// (per `direction` — `"premises"`, `"conclusions"`, or `"both"`) render its
/// premises/sources (for deductive/inductive levels with `source_ids`) and/or
/// its derived conclusions as a markdown chain. Returns the handler's exact
/// strings — including the `ERROR:`/not-found/`None`-case sentinels — so the
/// tool result is byte-identical. The DB calls reuse the ported reasoning-tree
/// primitives; only the tool-input extraction + ToolContext wrapping (the
/// dispatch layer) sit above this.
pub async fn format_reasoning_chain(
    pool: &PgPool,
    workspace_name: &str,
    observation_id: &str,
    direction: &str,
    observer: Option<&str>,
    observed: Option<&str>,
) -> Result<String, sqlx::Error> {
    if observation_id.is_empty() {
        return Ok("ERROR: 'observation_id' is required".to_string());
    }
    if !matches!(direction, "premises" | "conclusions" | "both") {
        return Ok(format!(
            "ERROR: Invalid direction '{direction}'. Must be 'premises', 'conclusions', or 'both'"
        ));
    }

    let docs = db::get_documents_by_ids(pool, workspace_name, &[observation_id.to_string()]).await?;
    let Some(doc) = docs.into_iter().next() else {
        return Ok(format!("ERROR: Observation '{observation_id}' not found"));
    };

    let mut parts: Vec<String> =
        vec![format!("**Observation [id:{}] ({}):**\n{}", doc.id, doc.level, doc.content)];

    let want_premises = matches!(direction, "premises" | "both");
    let want_conclusions = matches!(direction, "conclusions" | "both");

    if want_premises {
        // "deductive" -> Premises, "inductive" -> Sources; both fetch source_ids.
        let label = match doc.level.as_str() {
            "deductive" if !doc.source_ids.is_empty() => Some("Premises"),
            "inductive" if !doc.source_ids.is_empty() => Some("Sources"),
            _ => None,
        };
        match label {
            Some(label) => {
                let referenced = doc.source_ids.len();
                let sources = db::get_documents_by_ids(pool, workspace_name, &doc.source_ids).await?;
                if sources.is_empty() {
                    parts.push(format!(
                        "\n**{label}:** Referenced {referenced} {} IDs but none found in database",
                        label.to_lowercase().trim_end_matches('s')
                    ));
                } else {
                    let lines: Vec<String> = sources.iter().map(reasoning_doc_line).collect();
                    parts.push(format!(
                        "\n**{label} ({}):**\n{}",
                        sources.len(),
                        lines.join("\n")
                    ));
                }
            }
            None if doc.level == "explicit" => parts.push(
                "\n**Premises/Sources:** N/A (explicit observations have no premises)".to_string(),
            ),
            None => parts.push("\n**Premises/Sources:** None recorded".to_string()),
        }
    }

    if want_conclusions {
        let children =
            db::get_child_observations(pool, workspace_name, &doc.id, observer, observed).await?;
        if children.is_empty() {
            parts.push("\n**Derived Conclusions:** None found".to_string());
        } else {
            let lines: Vec<String> = children.iter().map(reasoning_doc_line).collect();
            parts.push(format!(
                "\n**Derived Conclusions ({}):**\n{}",
                children.len(),
                lines.join("\n")
            ));
        }
    }

    Ok(parts.join("\n"))
}

/// A string field of a tool-input JSON object, or `None` if absent/non-string.
fn input_str<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(Value::as_str)
}

/// `_safe_int(tool_input.get(key), default)` — coerce a (possibly absent) field.
fn input_int(input: &Value, key: &str, default: i64) -> i64 {
    safe_int(input.get(key).unwrap_or(&Value::Null), default)
}

/// Python `repr` of a list of strings: `['a', 'b']` (single-quoted, `, `-joined;
/// empty → `[]`). Used by the `get_observation_context` no-results message.
/// Faithful for nanoid-style ids; embedded quotes/specials are not escaped
/// (Python switches quoting then) — out of scope for message ids.
fn python_str_list_repr(items: &[String]) -> String {
    let inner = items
        .iter()
        .map(|item| format!("'{item}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

/// Port of `_parse_date`: empty/absent → `Ok(None)`; otherwise parse an ISO
/// string (`Z`→`+00:00` first, like Python), returning the handler's exact
/// `ERROR: Invalid {param} format '{s}'. Use ISO format (e.g., '2024-01-15')`
/// on failure. Covers `datetime.fromisoformat`'s common shapes (date-only,
/// datetime with/without offset, fractional seconds, space or `T` separator);
/// naive values are read as UTC. Rare ISO forms (week/ordinal dates) are not
/// reproduced — they don't arrive from the agent.
fn parse_date(value: Option<&str>, param: &str) -> Result<Option<DateTime<Utc>>, String> {
    let Some(text) = value.filter(|candidate| !candidate.is_empty()) else {
        return Ok(None);
    };
    let normalized = text.replace('Z', "+00:00");
    let invalid =
        || format!("ERROR: Invalid {param} format '{text}'. Use ISO format (e.g., '2024-01-15')");

    if let Ok(dt) = DateTime::parse_from_rfc3339(&normalized) {
        return Ok(Some(dt.with_timezone(&Utc)));
    }
    for separator in ['T', ' '] {
        for format in ["%Y-%m-%d%H:%M:%S%.f", "%Y-%m-%d%H:%M:%S"] {
            let pattern = format.replacen("%Y-%m-%d", &format!("%Y-%m-%d{separator}"), 1);
            if let Ok(naive) = NaiveDateTime::parse_from_str(&normalized, &pattern) {
                return Ok(Some(Utc.from_utc_datetime(&naive)));
            }
        }
    }
    if let Ok(date) = NaiveDate::parse_from_str(&normalized, "%Y-%m-%d") {
        return Ok(Some(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).expect("midnight"))));
    }
    Err(invalid())
}

/// Port of the `grep_messages` tool handler: parse/cap input (`text` required;
/// `limit` capped at 30, `context_window` at 2), run the data layer, and either
/// the no-results sentinel or the formatted snippets.
pub async fn handle_grep_messages(
    pool: &PgPool,
    ctx: &ToolContext,
    input: &Value,
) -> Result<String, sqlx::Error> {
    let text = input_str(input, "text").unwrap_or("");
    if text.is_empty() {
        return Ok("ERROR: 'text' parameter is required".to_string());
    }
    let limit = input_int(input, "limit", 10).min(30);
    let context_window = input_int(input, "context_window", 2).min(2);

    let snippets = db::grep_messages(
        pool,
        &ctx.workspace_name,
        ctx.session_name.as_deref(),
        text,
        limit,
        context_window,
        Some(&ctx.observer),
    )
    .await?;
    if snippets.is_empty() {
        return Ok(format!("No messages found containing '{text}'"));
    }
    Ok(format_grep_result(text, &snippets).0)
}

/// Port of the `get_messages_by_date_range` tool handler: parse `after_date` /
/// `before_date` (ERROR string on bad ISO), `limit` (capped at 20), `order`
/// (`asc`→oldest-first else newest-first), run the data layer, and assemble the
/// "Found N messages (range, order):" output (or the no-results sentinel).
pub async fn handle_get_messages_by_date_range(
    pool: &PgPool,
    ctx: &ToolContext,
    input: &Value,
) -> Result<String, sqlx::Error> {
    let after_str = input_str(input, "after_date");
    let before_str = input_str(input, "before_date");
    let limit = input_int(input, "limit", 20).min(20);
    let order = input_str(input, "order").unwrap_or("desc");

    let after = match parse_date(after_str, "after_date") {
        Ok(value) => value,
        Err(message) => return Ok(message),
    };
    let before = match parse_date(before_str, "before_date") {
        Ok(value) => value,
        Err(message) => return Ok(message),
    };

    let messages = db::get_messages_by_date_range(
        pool,
        &ctx.workspace_name,
        ctx.session_name.as_deref(),
        Some(&ctx.observer),
        after,
        before,
        limit,
        order != "asc",
    )
    .await?;

    // The range description uses the raw input strings' truthiness.
    let mut range_parts: Vec<String> = Vec::new();
    if let Some(after) = after_str.filter(|value| !value.is_empty()) {
        range_parts.push(format!("after {after}"));
    }
    if let Some(before) = before_str.filter(|value| !value.is_empty()) {
        range_parts.push(format!("before {before}"));
    }

    if messages.is_empty() {
        let range_desc = if range_parts.is_empty() {
            "specified range".to_string()
        } else {
            range_parts.join(" and ")
        };
        return Ok(format!("No messages found {range_desc}"));
    }

    let range_desc = if range_parts.is_empty() {
        "all time".to_string()
    } else {
        range_parts.join(" and ")
    };
    let order_desc = if order == "asc" {
        "oldest first"
    } else {
        "newest first"
    };
    let output = format!(
        "Found {} messages ({range_desc}, {order_desc}):\n\n{}",
        messages.len(),
        format_message_list(&messages)
    );
    Ok(truncate_tool_output(&output, MAX_TOOL_OUTPUT_CHARS).0)
}

/// Port of the `get_observation_context` tool handler: pull the `message_ids`
/// list, run the ±1-context data layer, and either the Python-repr no-results
/// sentinel or the "Retrieved N messages with context:" output.
pub async fn handle_get_observation_context(
    pool: &PgPool,
    ctx: &ToolContext,
    input: &Value,
) -> Result<String, sqlx::Error> {
    let message_ids: Vec<String> = input
        .get("message_ids")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let messages = db::get_observation_context(
        pool,
        &ctx.workspace_name,
        ctx.session_name.as_deref(),
        &message_ids,
        Some(&ctx.observer),
    )
    .await?;
    if messages.is_empty() {
        return Ok(format!(
            "No messages found for IDs {}",
            python_str_list_repr(&message_ids)
        ));
    }
    let output = format!(
        "Retrieved {} messages with context:\n{}",
        messages.len(),
        format_message_list(&messages)
    );
    Ok(truncate_tool_output(&output, MAX_TOOL_OUTPUT_CHARS).0)
}

/// Port of the `get_reasoning_chain` tool handler input wrapping:
/// extract `observation_id` / `direction` (default `"both"`) and delegate to
/// [`format_reasoning_chain`] (which owns the ERROR/not-found sentinels).
pub async fn handle_get_reasoning_chain(
    pool: &PgPool,
    ctx: &ToolContext,
    input: &Value,
) -> Result<String, sqlx::Error> {
    let observation_id = input_str(input, "observation_id").unwrap_or("");
    let direction = input_str(input, "direction").unwrap_or("both");
    format_reasoning_chain(
        pool,
        &ctx.workspace_name,
        observation_id,
        direction,
        Some(&ctx.observer),
        Some(&ctx.observed),
    )
    .await
}

/// Port of the `search_messages` tool handler (semantic message search). Takes
/// a **pre-computed** `query_embedding` — the embedding call is external and is
/// the caller's responsibility (same boundary as the data layer). `query`
/// required (else ERROR), `limit` capped at 20, context window fixed at 2.
pub async fn handle_search_messages(
    pool: &PgPool,
    ctx: &ToolContext,
    input: &Value,
    query_embedding: &[f32],
) -> Result<String, sqlx::Error> {
    let query = input_str(input, "query").unwrap_or("");
    if query.is_empty() {
        return Ok("ERROR: 'query' parameter is required".to_string());
    }
    let limit = input_int(input, "limit", 10).min(20);

    let snippets = db::search_messages_semantic(
        pool,
        &ctx.workspace_name,
        ctx.session_name.as_deref(),
        Some(&ctx.observer),
        query_embedding,
        None,
        None,
        limit,
        2,
    )
    .await?;
    if snippets.is_empty() {
        return Ok(format!("No messages found for query '{query}'"));
    }
    Ok(format_message_snippets(&snippets, &format!("for query '{query}'")))
}

/// Port of the `search_messages_temporal` tool handler (semantic search + date
/// window). Takes a pre-computed `query_embedding`. `query` required, `limit`
/// capped at 10, `context_window` at 2; `after_date`/`before_date` parsed (ERROR
/// on bad ISO). The result desc carries the date-filter suffix.
pub async fn handle_search_messages_temporal(
    pool: &PgPool,
    ctx: &ToolContext,
    input: &Value,
    query_embedding: &[f32],
) -> Result<String, sqlx::Error> {
    let query = input_str(input, "query").unwrap_or("");
    if query.is_empty() {
        return Ok("ERROR: 'query' parameter is required".to_string());
    }
    let after_str = input_str(input, "after_date");
    let before_str = input_str(input, "before_date");
    let limit = input_int(input, "limit", 10).min(10);
    let context_window = input_int(input, "context_window", 2).min(2);

    let after = match parse_date(after_str, "after_date") {
        Ok(value) => value,
        Err(message) => return Ok(message),
    };
    let before = match parse_date(before_str, "before_date") {
        Ok(value) => value,
        Err(message) => return Ok(message),
    };

    let snippets = db::search_messages_semantic(
        pool,
        &ctx.workspace_name,
        ctx.session_name.as_deref(),
        Some(&ctx.observer),
        query_embedding,
        after,
        before,
        limit,
        context_window,
    )
    .await?;

    let mut date_filter: Vec<String> = Vec::new();
    if let Some(after) = after_str.filter(|value| !value.is_empty()) {
        date_filter.push(format!("after {after}"));
    }
    if let Some(before) = before_str.filter(|value| !value.is_empty()) {
        date_filter.push(format!("before {before}"));
    }
    let filter_desc = if date_filter.is_empty() {
        String::new()
    } else {
        format!(" ({})", date_filter.join(" and "))
    };

    if snippets.is_empty() {
        return Ok(format!("No messages found for query '{query}'{filter_desc}"));
    }
    Ok(format_message_snippets(
        &snippets,
        &format!("for query '{query}'{filter_desc}"),
    ))
}

/// The query-embedding seam the dialectic tool executor needs for the
/// semantic-search tools. Production wraps the OpenAI embedding client
/// (`embedding::embed_openai`); tests use a fixed-vector mock. Mirrors how the
/// LLM tool loop abstracts its completion/transport seams.
pub trait Embedder {
    fn embed(&self, query: &str) -> impl Future<Output = Result<Vec<f32>, String>> + Send;
}

/// Bridges the ported dialectic tool handlers to the generic
/// [`crate::llm::tool_loop::execute_tool_loop`] via its [`ToolExecutor`] seam:
/// dispatches a tool call by name to the matching handler, embedding the query
/// first for the semantic-search tools. DB errors and embed failures map to
/// `Err(String)` (the loop folds those into an `is_error` tool result, like
/// Python's `except` branch); the handlers' own `ERROR:` strings are normal
/// `Ok` results. Unhandled tools (writes / `search_memory` / dreamer tools)
/// return an `Err`.
pub struct DialecticToolExecutor<'a, E: Embedder> {
    pub pool: &'a PgPool,
    pub ctx: ToolContext,
    pub embedder: E,
}

impl<E: Embedder + Sync> ToolExecutor for DialecticToolExecutor<'_, E> {
    async fn execute(&self, name: &str, input: &Value) -> Result<String, String> {
        let to_err = |error: sqlx::Error| error.to_string();
        match name {
            "grep_messages" => handle_grep_messages(self.pool, &self.ctx, input)
                .await
                .map_err(to_err),
            "get_messages_by_date_range" => {
                handle_get_messages_by_date_range(self.pool, &self.ctx, input)
                    .await
                    .map_err(to_err)
            }
            "get_observation_context" => {
                handle_get_observation_context(self.pool, &self.ctx, input)
                    .await
                    .map_err(to_err)
            }
            "get_reasoning_chain" => handle_get_reasoning_chain(self.pool, &self.ctx, input)
                .await
                .map_err(to_err),
            "search_messages" => {
                // Embed only when the query is present; an empty query short-
                // circuits to the handler's ERROR before the embedding is used.
                let query = input_str(input, "query").unwrap_or("");
                let embedding = if query.is_empty() {
                    Vec::new()
                } else {
                    self.embedder.embed(query).await?
                };
                handle_search_messages(self.pool, &self.ctx, input, &embedding)
                    .await
                    .map_err(to_err)
            }
            "search_messages_temporal" => {
                let query = input_str(input, "query").unwrap_or("");
                let embedding = if query.is_empty() {
                    Vec::new()
                } else {
                    self.embedder.embed(query).await?
                };
                handle_search_messages_temporal(self.pool, &self.ctx, input, &embedding)
                    .await
                    .map_err(to_err)
            }
            other => Err(format!("Unknown or unsupported dialectic tool: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    fn msg(content: &str, peer: &str, session: &str, created_at: &str) -> Value {
        json!({
            "id": "m",
            "content": content,
            "peer_id": peer,
            "session_id": session,
            "metadata": {},
            "created_at": created_at,
            "workspace_id": "ws",
            "token_count": 1
        })
    }

    fn snippet(matched: usize, context: Vec<Value>) -> MessageSnippet {
        MessageSnippet {
            matched: (0..matched).map(|_| json!({})).collect(),
            context,
        }
    }

    #[test]
    fn safe_int_coerces_like_python() {
        assert_eq!(safe_int(&json!(7), 10), 7);
        assert_eq!(safe_int(&json!(3.9), 10), 3); // truncate toward zero
        assert_eq!(safe_int(&json!(-3.9), 10), -3);
        assert_eq!(safe_int(&json!("5"), 10), 5);
        assert_eq!(safe_int(&json!("  12 "), 10), 12);
        assert_eq!(safe_int(&json!("Infinity"), 10), 10); // ValueError -> default
        assert_eq!(safe_int(&json!("5.0"), 10), 10); // int("5.0") raises -> default
        assert_eq!(safe_int(&json!(true), 10), 1);
        assert_eq!(safe_int(&json!(false), 10), 0);
        assert_eq!(safe_int(&Value::Null, 10), 10);
        assert_eq!(safe_int(&json!([1]), 10), 10);
    }

    #[test]
    fn comma_grouping_matches_python_format() {
        assert_eq!(comma_grouped(0), "0");
        assert_eq!(comma_grouped(999), "999");
        assert_eq!(comma_grouped(1_000), "1,000");
        assert_eq!(comma_grouped(10_000), "10,000");
        assert_eq!(comma_grouped(1_234_567), "1,234,567");
    }

    #[test]
    fn truncate_tool_output_clamps_with_notice() {
        let (text, original, was) = truncate_tool_output("hello", 100);
        assert_eq!(text, "hello");
        assert_eq!(original, 5);
        assert!(!was);

        let long = "x".repeat(12);
        let (text, original, was) = truncate_tool_output(&long, 10);
        assert!(was);
        assert_eq!(original, 12);
        assert_eq!(
            text,
            "xxxxxxxxxx\n\n[OUTPUT TRUNCATED - showing 10 of 12 characters]"
        );
    }

    #[test]
    fn truncate_tool_output_counts_code_points() {
        // 6 multibyte chars, cap 4 -> head of 4 code points.
        let (text, original, was) = truncate_tool_output("☃☃☃☃☃☃", 4);
        assert!(was);
        assert_eq!(original, 6);
        assert!(text.starts_with("☃☃☃☃\n\n[OUTPUT TRUNCATED"));
    }

    #[test]
    fn truncate_message_content_appends_ellipsis() {
        assert_eq!(truncate_message_content("short", 100), "short");
        assert_eq!(truncate_message_content("abcdef", 3), "abc...");
        assert_eq!(truncate_message_content("café", 4), "café");
    }

    #[test]
    fn extract_pattern_snippet_returns_short_content_unchanged() {
        assert_eq!(extract_pattern_snippet("hello world", "world", 100), "hello world");
    }

    #[test]
    fn extract_pattern_snippet_windows_around_match() {
        // 40-char content, pattern in the middle, cap 10 -> window with both ...
        let content = "0123456789ABCDEFGHIJfindKLMNOPQRSTUVWXYZ"; // len 39
        let snippet = extract_pattern_snippet(content, "find", 10);
        assert!(snippet.starts_with("..."));
        assert!(snippet.ends_with("..."));
        assert!(snippet.contains("find"));
        // 10-char window + both "..." markers.
        assert_eq!(snippet.chars().count(), 10 + 6);
    }

    #[test]
    fn extract_pattern_snippet_no_match_returns_head() {
        let content = "a".repeat(50);
        let snippet = extract_pattern_snippet(&content, "zzz", 10);
        assert_eq!(snippet, format!("{}...", "a".repeat(10)));
    }

    #[test]
    fn extract_pattern_snippet_match_at_start_no_prefix() {
        let content = format!("find{}", "x".repeat(50));
        let snippet = extract_pattern_snippet(&content, "FIND", 10);
        assert!(!snippet.starts_with("..."));
        assert!(snippet.ends_with("..."));
        assert!(snippet.to_lowercase().starts_with("find"));
    }

    #[test]
    fn format_new_turn_with_timestamp_golden() {
        let time = Utc.with_ymd_and_hms(2023, 5, 8, 13, 56, 0).unwrap();
        assert_eq!(
            format_new_turn_with_timestamp("hello", time, "alice"),
            "2023-05-08 13:56:00 alice: hello"
        );
    }

    #[test]
    fn format_message_snippets_golden() {
        let snippets = vec![
            snippet(
                1,
                vec![
                    msg("hello there", "alice", "s1", "2023-05-08T13:56:00Z"),
                    msg("general reply", "bob", "s1", "2023-05-08T13:57:00Z"),
                ],
            ),
            snippet(2, vec![msg("later", "alice", "s2", "2023-05-08T14:00:00Z")]),
        ];
        let output = format_message_snippets(&snippets, "for query 'hi'");
        assert_eq!(
            output,
            "Found 3 matching messages in 2 conversation snippets for query 'hi':\n\n\
             --- Snippet 1 (session: s1, 1 match(es)) ---\n\
             2023-05-08 13:56:00 alice: hello there\n\
             2023-05-08 13:57:00 bob: general reply\n\n\
             --- Snippet 2 (session: s2, 2 match(es)) ---\n\
             2023-05-08 14:00:00 alice: later"
        );
    }

    #[test]
    fn format_grep_result_golden() {
        let snippets = vec![snippet(
            1,
            vec![msg("look here now", "alice", "s1", "2023-05-08T13:56:00Z")],
        )];
        let (output, original, was) = format_grep_result("here", &snippets);
        assert!(!was);
        assert_eq!(original, output.chars().count());
        assert_eq!(
            output,
            "Found 1 messages containing 'here' in 1 conversation snippets:\n\n\
             --- Snippet 1 (session: s1, 1 match(es)) ---\n\
             2023-05-08 13:56:00 alice: look here now"
        );
    }

    #[test]
    fn python_str_list_repr_matches_python() {
        assert_eq!(python_str_list_repr(&[]), "[]");
        assert_eq!(python_str_list_repr(&["a".to_string()]), "['a']");
        assert_eq!(
            python_str_list_repr(&["id1".to_string(), "id2".to_string()]),
            "['id1', 'id2']"
        );
    }

    #[test]
    fn parse_date_handles_iso_shapes() {
        assert_eq!(parse_date(None, "after_date"), Ok(None));
        assert_eq!(parse_date(Some(""), "after_date"), Ok(None));
        assert_eq!(
            parse_date(Some("2024-01-15"), "after_date"),
            Ok(Some(Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap()))
        );
        assert_eq!(
            parse_date(Some("2024-01-15T10:30:00Z"), "after_date"),
            Ok(Some(Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap()))
        );
        assert_eq!(
            parse_date(Some("2024-01-15T10:30:00+00:00"), "after_date"),
            Ok(Some(Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap()))
        );
        assert_eq!(
            parse_date(Some("2024-01-15 10:30:00"), "after_date"),
            Ok(Some(Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap()))
        );
        assert_eq!(
            parse_date(Some("not-a-date"), "before_date"),
            Err("ERROR: Invalid before_date format 'not-a-date'. Use ISO format (e.g., '2024-01-15')".to_string())
        );
    }

    #[test]
    fn format_message_list_golden() {
        let messages = vec![
            msg("first", "alice", "s1", "2023-05-08T13:56:00Z"),
            msg("second", "bob", "s1", "2023-05-08T13:57:00Z"),
        ];
        assert_eq!(
            format_message_list(&messages),
            "2023-05-08 13:56:00 alice: first\n2023-05-08 13:57:00 bob: second"
        );
        assert_eq!(format_message_list(&[]), "");
    }
}
