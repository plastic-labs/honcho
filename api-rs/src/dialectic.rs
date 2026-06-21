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

/// Port of the `search_memory` tool handler. Takes a **pre-computed**
/// `query_embedding` (the embedding call is the caller's responsibility, like
/// the sibling semantic-search handlers). `top_k` defaults to 20, capped at 40.
///
/// The resulting `Representation` is rendered with its `Display` impl: the
/// dialectic agent runs with `include_observation_ids = false`, so the
/// `str_with_ids` variant is not used here. On empty memory it falls back to a
/// message search (the Python `agent_type == "dialectic"` branch — the only
/// agent this read-only sidecar hosts), since an empty representation means the
/// workspace/peer is early in its lifecycle and a message hit is more useful
/// than a bare "nothing found".
pub async fn handle_search_memory(
    pool: &PgPool,
    ctx: &ToolContext,
    input: &Value,
    query_embedding: &[f32],
) -> Result<String, sqlx::Error> {
    let query = input_str(input, "query").unwrap_or("");
    let top_k = input_int(input, "top_k", 20).min(40);

    // search_memory passes no level filter (all reasoning levels) and no
    // distance bound — an empty clause leaves the query's fixed WHERE intact.
    let filter = crate::filters::FilterClause {
        sql: String::new(),
        bindings: Vec::new(),
    };
    let documents = db::query_documents_full(
        pool,
        &ctx.workspace_name,
        &ctx.observer,
        &ctx.observed,
        query_embedding,
        &filter,
        None,
        top_k,
    )
    .await?;
    let representation = crate::representation::Representation::from_documents(&documents);
    let total = representation.count();

    if total == 0 {
        // Empty-memory fallback: re-use the same embedding for a message search
        // (limit re-derived from the raw top_k, capped at 20; context window 0).
        let limit = input_int(input, "top_k", 20).min(20);
        let snippets = db::search_messages_semantic(
            pool,
            &ctx.workspace_name,
            ctx.session_name.as_deref(),
            Some(&ctx.observer),
            query_embedding,
            None,
            None,
            limit,
            0,
        )
        .await?;
        if !snippets.is_empty() {
            let message_output =
                format_message_snippets(&snippets, &format!("for query '{query}'"));
            return Ok(format!(
                "No observations yet. Message search results:\n\n{message_output}"
            ));
        }
        return Ok(format!(
            "No observations found for query '{query}', and no messages found in history. \
             Try a different phrasing or use grep_messages for exact text."
        ));
    }

    let mem_str = representation.to_string();
    Ok(format!(
        "Found {total} observations for query '{query}':\n\n{mem_str}"
    ))
}

/// Format the two prefetched representations into the observations block the
/// dialectic injects into the user message, porting the formatting tail of
/// `DialecticAgent._prefetch_relevant_observations`. Explicit observations render
/// without ids (lowest reasoning level); derived observations render *with* ids so
/// the agent can follow up via `get_reasoning_chain`. Returns `None` when both are
/// empty (no prefetch block is added).
pub fn format_prefetched_observations(
    explicit: &crate::representation::Representation,
    derived: &crate::representation::Representation,
) -> Option<String> {
    if explicit.is_empty() && derived.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if !explicit.is_empty() {
        parts.push(explicit.format_as_markdown(false));
    }
    if !derived.is_empty() {
        parts.push(derived.format_as_markdown(true));
    }
    Some(parts.join("\n"))
}

/// Prefetch semantically relevant observations for `query`, porting
/// `DialecticAgent._prefetch_relevant_observations`. Runs two level-scoped
/// searches (explicit; derived = deductive/inductive/contradiction) over the
/// pre-computed `query_embedding` so the agent gets immediate context without a
/// tool call. `prefetch_limit` is the per-search cap (Python: 10 at the `minimal`
/// reasoning level, 25 otherwise). Returns the formatted block, or `None` when no
/// observations are found.
pub async fn prefetch_observations(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    query_embedding: &[f32],
    prefetch_limit: i64,
) -> Result<Option<String>, sqlx::Error> {
    let explicit_docs = db::query_documents_by_levels(
        pool,
        workspace_name,
        observer,
        observed,
        query_embedding,
        &["explicit".to_string()],
        prefetch_limit,
    )
    .await?;
    let derived_docs = db::query_documents_by_levels(
        pool,
        workspace_name,
        observer,
        observed,
        query_embedding,
        &[
            "deductive".to_string(),
            "inductive".to_string(),
            "contradiction".to_string(),
        ],
        prefetch_limit,
    )
    .await?;

    let explicit = crate::representation::Representation::from_documents(&explicit_docs);
    let derived = crate::representation::Representation::from_documents(&derived_docs);
    Ok(format_prefetched_observations(&explicit, &derived))
}

/// The query-embedding seam the dialectic tool executor needs for the
/// semantic-search tools. Production wraps the OpenAI embedding client
/// (`embedding::embed_openai`); tests use a fixed-vector mock. Mirrors how the
/// LLM tool loop abstracts its completion/transport seams.
pub trait Embedder {
    fn embed(&self, query: &str) -> impl Future<Output = Result<Vec<f32>, String>> + Send;

    /// Batch-embed `texts`, returning vectors in input order. The production
    /// [`OpenAiEmbedder`] issues one batched request matching
    /// `simple_batch_embed`; test mocks typically map [`Embedder::embed`].
    fn batch_embed(
        &self,
        texts: &[String],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, String>> + Send;
}

/// The production [`Embedder`]: wraps [`crate::embedding::embed_openai`] with the
/// configured model/dimensions/credentials. Embedding errors (token-limit,
/// transport) map to `Err(String)` so the tool loop folds them into an
/// `is_error` result rather than aborting the whole call.
pub struct OpenAiEmbedder<'a, H: crate::llm::http::LlmHttp> {
    pub http: &'a H,
    pub credentials: crate::llm::http::Credentials,
    pub model: String,
    pub vector_dimensions: usize,
    pub send_dimensions: bool,
    pub max_tokens: usize,
}

impl<H: crate::llm::http::LlmHttp + Sync> Embedder for OpenAiEmbedder<'_, H> {
    async fn embed(&self, query: &str) -> Result<Vec<f32>, String> {
        crate::embedding::embed_openai(
            self.http,
            &self.credentials,
            &self.model,
            query,
            self.vector_dimensions,
            self.send_dimensions,
            self.max_tokens,
        )
        .await
        .map_err(|error| error.to_string())
    }

    async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        crate::embedding::embed_openai_batch(
            self.http,
            &self.credentials,
            &self.model,
            texts,
            self.vector_dimensions,
            self.send_dimensions,
            self.max_tokens,
        )
        .await
        .map_err(|error| error.to_string())
    }
}

/// An owned-transport variant of [`OpenAiEmbedder`] for contexts that must hold
/// the embedder for a `'static` lifetime (e.g. the deriver worker's spawned
/// tasks, where the borrow-based [`OpenAiEmbedder`] can't satisfy `Arc<E>`).
/// Owns its `H` transport and delegates to the same embedding calls.
pub struct OwnedOpenAiEmbedder<H: crate::llm::http::LlmHttp> {
    pub http: H,
    pub credentials: crate::llm::http::Credentials,
    pub model: String,
    pub vector_dimensions: usize,
    pub send_dimensions: bool,
    pub max_tokens: usize,
}

impl<H: crate::llm::http::LlmHttp + Sync> Embedder for OwnedOpenAiEmbedder<H> {
    async fn embed(&self, query: &str) -> Result<Vec<f32>, String> {
        crate::embedding::embed_openai(
            &self.http,
            &self.credentials,
            &self.model,
            query,
            self.vector_dimensions,
            self.send_dimensions,
            self.max_tokens,
        )
        .await
        .map_err(|error| error.to_string())
    }

    async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        crate::embedding::embed_openai_batch(
            &self.http,
            &self.credentials,
            &self.model,
            texts,
            self.vector_dimensions,
            self.send_dimensions,
            self.max_tokens,
        )
        .await
        .map_err(|error| error.to_string())
    }
}

/// Bridges the ported dialectic tool handlers to the generic
/// [`crate::llm::tool_loop::execute_tool_loop`] via its [`ToolExecutor`] seam:
/// dispatches a tool call by name to the matching handler, embedding the query
/// first for the semantic-search tools. DB errors and embed failures map to
/// `Err(String)` (the loop folds those into an `is_error` tool result, like
/// Python's `except` branch); the handlers' own `ERROR:` strings are normal
/// `Ok` results. Unhandled tools (writes / dreamer tools) return an `Err`.
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
            "search_memory" => {
                // search_memory always embeds (the query drives both the vector
                // memory search and the empty-memory message fallback).
                let query = input_str(input, "query").unwrap_or("");
                let embedding = self.embedder.embed(query).await?;
                handle_search_memory(self.pool, &self.ctx, input, &embedding)
                    .await
                    .map_err(to_err)
            }
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

/// The dialectic agent's tool schemas (`DIALECTIC_TOOLS`), verbatim from Python's
/// `TOOLS`, in Anthropic-native `{name, description, input_schema}` form — the
/// provider request builders (`backends::{anthropic,openai}::build_request`)
/// convert these to each backend's wire shape.
pub fn dialectic_tools() -> Vec<serde_json::Value> {
    vec![
        search_memory_tool(),
        search_messages_tool(),
        get_observation_context_tool(),
        grep_messages_tool(),
        get_messages_by_date_range_tool(),
        search_messages_temporal_tool(),
        get_reasoning_chain_tool(),
    ]
}

/// The reduced toolset used at the `minimal` reasoning level
/// (`DIALECTIC_TOOLS_MINIMAL`): just `search_memory` + `search_messages`.
pub fn dialectic_tools_minimal() -> Vec<serde_json::Value> {
    vec![search_memory_tool(), search_messages_tool()]
}

/// The fully-static body of the dialectic system prompt (from
/// `## AVAILABLE TOOLS` onward), extracted verbatim from
/// `dialectic/prompts.py::agent_system_prompt`. Kept as an included file rather
/// than an inline literal to avoid transcription drift in ~150 lines of prompt.
const SYSTEM_PROMPT_BODY: &str = include_str!("dialectic_prompt_body.txt");

/// Port of `dialectic/prompts.py::agent_system_prompt`. Builds the templated
/// prefix (directional vs global perspective by `observer == observed`, the
/// observer/observed peer-card sections, and the peer-card explanation gate)
/// and appends the static body. Byte-identical to the Python f-string, including
/// its blank-line spacing. Note the empty-list-vs-`None` distinction: the
/// peer-card *explanation* is gated on `card is not None` (an empty list still
/// enables it) while each card *section* is gated on the list being non-empty.
pub fn agent_system_prompt(
    observer: &str,
    observed: &str,
    observer_peer_card: Option<&[String]>,
    observed_peer_card: Option<&[String]>,
) -> String {
    let peer_cards_enabled = observer_peer_card.is_some() || observed_peer_card.is_some();
    fn non_empty(card: Option<&[String]>) -> Option<&[String]> {
        card.filter(|items| !items.is_empty())
    }

    let perspective_section = if observer != observed {
        let observer_card_section = match non_empty(observer_peer_card) {
            Some(card) => format!(
                "\nKnown biographical information about {observer} (the one asking):\n<observer_peer_card>\n{}\n</observer_peer_card>\n",
                card.join("\n")
            ),
            None => String::new(),
        };
        let observed_card_section = match non_empty(observed_peer_card) {
            Some(card) => format!(
                "\nKnown biographical information about {observed} (the subject):\n<observed_peer_card>\n{}\n</observed_peer_card>\n",
                card.join("\n")
            ),
            None => String::new(),
        };
        format!(
            "\nYou are answering queries from the perspective of {observer}'s understanding of {observed}.\nThis is a directional query - {observer} wants to know about {observed}.\n\n{observer_card_section}\n{observed_card_section}\n"
        )
    } else {
        let peer_card_section = match non_empty(observer_peer_card) {
            Some(card) => format!(
                "\nKnown biographical information about {observed}:\n<peer_card>\n{}\n</peer_card>\n",
                card.join("\n")
            ),
            None => String::new(),
        };
        format!("\nYou are answering queries about '{observed}'.\n\n{peer_card_section}\n")
    };

    let peer_card_explanation = if peer_cards_enabled {
        "\nPeer cards are **constructed summaries** - they are synthesized from the same observations stored in memory. This means:\n- Information in a peer card originates from observations you can also find via `search_memory`\n- The peer card is a convenience summary, not a separate source of truth\n"
    } else {
        ""
    };

    let intro = "\nYou are a helpful and concise context synthesis agent that answers questions about users by gathering relevant information from a memory system.\n\nAlways give users the answer *they expect* based on the message history -- the goal is to help recall and *reason through* insights that the memory system has already gathered. You have many tools for gathering context. Search wisely.\n";

    format!("{intro}\n{perspective_section}\n{peer_card_explanation}\n{SYSTEM_PROMPT_BODY}")
}

/// Port of the session-history block built in
/// `DialecticAgent._initialize_session_history`. `formatted_messages` are the
/// already-rendered turn lines (each from [`format_new_turn_with_timestamp`]),
/// joined with newlines into a `<session_history>` section that the caller
/// appends to the system prompt. Returns `None` for an empty list, mirroring the
/// Python early-return that skips the append entirely when no messages are found.
pub fn session_history_section(formatted_messages: &[String]) -> Option<String> {
    if formatted_messages.is_empty() {
        return None;
    }
    Some(format!(
        "\n\n## SESSION HISTORY\n\n\
         The following is the recent conversation history from this session. \
         Use this as immediate context when answering the query.\n\n\
         <session_history>\n{}\n</session_history>",
        formatted_messages.join("\n")
    ))
}

/// Port of the user-message construction in `DialecticAgent._prepare_query`.
/// When `prefetched_observations` is present and non-empty (Python's truthiness
/// test treats both `None` and `""` as falsy), the query is wrapped with the
/// prefetched-observations preamble; otherwise it is the bare `Query: {query}`.
pub fn build_user_content(query: &str, prefetched_observations: Option<&str>) -> String {
    match prefetched_observations {
        Some(observations) if !observations.is_empty() => format!(
            "Query: {query}\n\n\
             ## Relevant Observations (prefetched)\n\
             The following observations were found to be semantically relevant to your query. \
             Use these as primary context. You may still use tools to find additional information if needed.\n\n\
             {observations}"
        ),
        _ => format!("Query: {query}"),
    }
}

fn search_memory_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "search_memory",
        "description": "Search for observations in memory using semantic similarity. Use this to find relevant information about the peer when you need to recall specific details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {"type": "integer", "description": "(Optional) number of results to return (default: 20, max: 40)", "default": 20}
            },
            "required": ["query"]
        }
    })
}

fn search_messages_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "search_messages",
        "description": "Search for messages using semantic similarity and retrieve conversation snippets. Returns matching messages with surrounding context (2 messages before and after). Nearby matches within the same session are merged into a single snippet to avoid repetition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text to find relevant messages"},
                "limit": {"type": "integer", "description": "Maximum number of matching messages to return (default: 10, max: 20)", "default": 10}
            },
            "required": ["query"]
        }
    })
}

fn get_observation_context_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "get_observation_context",
        "description": "Retrieve messages for given message IDs along with surrounding context. Takes message IDs (from an observation's message_ids field) and retrieves those messages plus the messages immediately before and after each one to provide conversation context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_ids": {"type": "array", "items": {"type": "string"}, "description": "List of message IDs to retrieve (get these from observation.message_ids in search results)"}
            },
            "required": ["message_ids"]
        }
    })
}

fn grep_messages_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "grep_messages",
        "description": "Search for messages containing specific text (case-insensitive). Unlike semantic search, this finds EXACT text matches. Use for finding specific names, dates, phrases, or keywords mentioned in conversations. Returns messages with surrounding context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to search for (case-insensitive substring match)"},
                "limit": {"type": "integer", "description": "Maximum messages to return (default: 10, max: 30)", "default": 10},
                "context_window": {"type": "integer", "description": "Number of messages before/after each match to include (default: 2)", "default": 2}
            },
            "required": ["text"]
        }
    })
}

fn get_messages_by_date_range_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "get_messages_by_date_range",
        "description": "Get messages from a specific date range. Use this to find what was discussed during a particular time period, or to compare information before vs after a date. Essential for knowledge update questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "after_date": {"type": "string", "description": "Start date (ISO format, e.g., '2024-01-15'). Returns messages after this date."},
                "before_date": {"type": "string", "description": "End date (ISO format). Returns messages before this date."},
                "limit": {"type": "integer", "description": "Maximum messages to return (default: 20, max: 50)", "default": 20},
                "order": {"type": "string", "enum": ["asc", "desc"], "description": "Sort order: 'asc' for oldest first, 'desc' for newest first (default: desc)", "default": "desc"}
            }
        }
    })
}

fn search_messages_temporal_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "search_messages_temporal",
        "description": "Semantic search for messages with optional date filtering. Combines the power of semantic search with time constraints. Use after_date to find recent mentions of a topic, or before_date to find what was said about something before a certain point. Best for knowledge update questions where you need to find the MOST RECENT discussion of a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Semantic search query"},
                "after_date": {"type": "string", "description": "Only return messages after this date (ISO format, e.g., '2024-01-15')"},
                "before_date": {"type": "string", "description": "Only return messages before this date (ISO format)"},
                "limit": {"type": "integer", "description": "Maximum messages to return (default: 10, max: 20)", "default": 10},
                "context_window": {"type": "integer", "description": "Messages before/after each match (default: 2)", "default": 2}
            },
            "required": ["query"]
        }
    })
}

fn get_reasoning_chain_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "get_reasoning_chain",
        "description": "Get the reasoning chain for an observation - traverse the tree to find premises (for deductive) or sources (for inductive), and/or find conclusions derived from this observation. Use this to understand how an observation was derived or what conclusions depend on it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observation_id": {"type": "string", "description": "The document ID of the observation to get the reasoning chain for"},
                "direction": {"type": "string", "enum": ["premises", "conclusions", "both"], "description": "'premises' to get what this observation is based on, 'conclusions' to get what depends on it, 'both' for full context", "default": "both"}
            },
            "required": ["observation_id"]
        }
    })
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
    fn agent_system_prompt_matches_python_fixtures() {
        let fixtures: std::collections::HashMap<String, String> =
            serde_json::from_str(include_str!("../tests/fixtures/dialectic_prompts.json")).unwrap();

        assert_eq!(
            agent_system_prompt("alice", "alice", None, None),
            fixtures["global_no_cards"]
        );
        assert_eq!(
            agent_system_prompt(
                "alice",
                "alice",
                Some(&["IDENTITY: Alice".to_string(), "ATTRIBUTE: likes tea".to_string()]),
                None
            ),
            fixtures["global_with_card"]
        );
        assert_eq!(
            agent_system_prompt(
                "alice",
                "bob",
                Some(&["IDENTITY: Alice".to_string()]),
                Some(&["IDENTITY: Bob".to_string()])
            ),
            fixtures["directional_both_cards"]
        );
    }

    #[tokio::test]
    async fn openai_embedder_returns_vector() {
        use crate::llm::http::Credentials;
        use crate::llm::http::mock::MockHttp;

        let http = MockHttp::ok(serde_json::json!({"data": [{"embedding": [1.0, 2.0]}]}));
        let embedder = OpenAiEmbedder {
            http: &http,
            credentials: Credentials::new("sk-e"),
            model: "text-embedding-3-small".to_string(),
            vector_dimensions: 2,
            send_dimensions: true,
            max_tokens: 8192,
        };
        assert_eq!(embedder.embed("hello").await.unwrap(), vec![1.0_f32, 2.0]);
    }

    #[test]
    fn format_prefetched_observations_empty_is_none() {
        use crate::representation::Representation;
        let empty = Representation::default();
        assert_eq!(format_prefetched_observations(&empty, &empty), None);
    }

    #[test]
    fn format_prefetched_observations_joins_present_sections() {
        use crate::representation::{ExplicitObservation, Representation};
        let explicit = Representation {
            explicit: vec![ExplicitObservation {
                id: "e1".to_string(),
                created_at: DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
                message_ids: vec![],
                session_name: None,
                content: "a fact".to_string(),
            }],
            ..Representation::default()
        };
        let derived = Representation::default();
        // Only explicit present -> just the explicit section, no leading/trailing join newline.
        let formatted = format_prefetched_observations(&explicit, &derived).unwrap();
        assert!(formatted.starts_with("## Explicit Observations"));
        assert!(formatted.contains("a fact"));
        assert!(!formatted.contains("## Deductive"));
    }

    #[test]
    fn session_history_section_empty_is_none() {
        assert_eq!(session_history_section(&[]), None);
    }

    #[test]
    fn session_history_section_matches_python() {
        let messages = vec![
            "2023-05-08 13:56:00 alice: hi".to_string(),
            "2023-05-08 13:57:00 bob: hello".to_string(),
        ];
        assert_eq!(
            session_history_section(&messages).unwrap(),
            "\n\n## SESSION HISTORY\n\n\
             The following is the recent conversation history from this session. \
             Use this as immediate context when answering the query.\n\n\
             <session_history>\n\
             2023-05-08 13:56:00 alice: hi\n\
             2023-05-08 13:57:00 bob: hello\n\
             </session_history>"
        );
    }

    #[test]
    fn build_user_content_without_observations() {
        assert_eq!(build_user_content("who is bob?", None), "Query: who is bob?");
        // Empty string is falsy in Python, so it also yields the bare query.
        assert_eq!(
            build_user_content("who is bob?", Some("")),
            "Query: who is bob?"
        );
    }

    #[test]
    fn build_user_content_with_observations() {
        assert_eq!(
            build_user_content("who is bob?", Some("- bob likes tea")),
            "Query: who is bob?\n\n\
             ## Relevant Observations (prefetched)\n\
             The following observations were found to be semantically relevant to your query. \
             Use these as primary context. You may still use tools to find additional information if needed.\n\n\
             - bob likes tea"
        );
    }

    #[test]
    fn dialectic_tool_schemas_match_python() {
        let tools = dialectic_tools();
        let names: Vec<&str> = tools
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec![
                "search_memory",
                "search_messages",
                "get_observation_context",
                "grep_messages",
                "get_messages_by_date_range",
                "search_messages_temporal",
                "get_reasoning_chain",
            ]
        );
        // Spot-check a schema's shape (Anthropic-native input_schema).
        let grep = &tools[3];
        assert_eq!(grep["input_schema"]["required"], json!(["text"]));
        assert_eq!(grep["input_schema"]["properties"]["limit"]["default"], json!(10));
        assert_eq!(
            grep["input_schema"]["properties"]["context_window"]["default"],
            json!(2)
        );

        let minimal = dialectic_tools_minimal();
        let minimal_names: Vec<&str> = minimal
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(minimal_names, vec!["search_memory", "search_messages"]);
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
