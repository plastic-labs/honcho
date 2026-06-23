//! The DialecticAgent assembly, porting `DialecticAgent.answer` (dialectic/core.py).
//!
//! Ties together the already-ported pieces: the system prompt
//! ([`agent_system_prompt`] + an optional [`session_history_section`]), the
//! prefetched-observations block injected into the user message
//! ([`prefetch_observations`] + [`build_user_content`]), and the agentic tool
//! loop driven by the provider-backed [`HonchoCaller`] over the
//! [`DialecticToolExecutor`].

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use sqlx::PgPool;

use crate::db;
use crate::dialectic::{
    DialecticToolExecutor, Embedder, ToolContext, agent_system_prompt, build_user_content,
    dialectic_tools, dialectic_tools_minimal, format_new_turn_with_timestamp,
    prefetch_observations, session_history_section,
};
use crate::dialectic_config::{DialecticSettings, ReasoningLevel};
use crate::llm::credentials::{TransportApiKeys, resolve_credentials};
use crate::llm::executor::{HonchoCaller, execute_stream};
use crate::llm::http::{LlmHttp, LlmStreamHttp};
use crate::llm::tool_loop::{ToolLoopError, execute_tool_loop};
use futures::{Stream, StreamExt};

/// An error answering a dialectic query.
#[derive(Debug)]
pub enum DialecticError {
    /// A database error fetching session history (prefetch errors degrade to "no
    /// prefetch" instead, matching Python's `except`).
    Db(sqlx::Error),
    /// The agentic tool loop failed (e.g. the completion call after retries).
    ToolLoop(ToolLoopError),
}

impl std::fmt::Display for DialecticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DialecticError::Db(error) => write!(f, "database error: {error}"),
            DialecticError::ToolLoop(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for DialecticError {}

/// The `created_at` of a message JSON value (RFC3339 from `message_json`),
/// defaulting to the epoch if absent/unparseable.
fn message_created_at(message: &Value) -> DateTime<Utc> {
    message
        .get("created_at")
        .and_then(Value::as_str)
        .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).expect("epoch is valid"))
}

/// Answer a query about `observed` from `observer`'s perspective, porting
/// `DialecticAgent.answer` (non-streaming). Builds the system prompt (+ session
/// history when a session is given and history injection is enabled), prefetches
/// observations into the user message, then runs the agentic tool loop at the
/// requested reasoning `level`. Returns the synthesized answer text.
///
/// `embedder` is consumed: it embeds the query for the prefetch, then moves into
/// the tool executor for the semantic-search tools. `llm_http` is the transport
/// for the completion calls; `keys` resolves per-provider credentials.
#[allow(clippy::too_many_arguments)]
pub async fn answer<H, E>(
    pool: &PgPool,
    llm_http: &H,
    keys: TransportApiKeys,
    embedder: E,
    settings: &DialecticSettings,
    workspace_name: &str,
    session_name: Option<&str>,
    observer: &str,
    observed: &str,
    observer_card: Option<&[String]>,
    observed_card: Option<&[String]>,
    query: &str,
    level: ReasoningLevel,
) -> Result<String, DialecticError>
where
    H: LlmHttp + Sync,
    E: Embedder + Sync,
{
    let level_settings = settings.level(level);

    // 1. System prompt, with session history appended when enabled + present.
    let mut system = agent_system_prompt(observer, observed, observer_card, observed_card);
    if settings.session_history_max_tokens > 0 {
        if let Some(session) = session_name {
            let messages = db::get_session_messages_within_token_limit(
                pool,
                workspace_name,
                session,
                settings.session_history_max_tokens,
            )
            .await
            .map_err(DialecticError::Db)?;
            let formatted: Vec<String> = messages
                .iter()
                .map(|message| {
                    format_new_turn_with_timestamp(
                        message.get("content").and_then(Value::as_str).unwrap_or(""),
                        message_created_at(message),
                        message.get("peer_id").and_then(Value::as_str).unwrap_or(""),
                    )
                })
                .collect();
            if let Some(section) = session_history_section(&formatted) {
                system.push_str(&section);
            }
        }
    }

    // 2. Prefetch observations for the user message. Any failure (embed or DB)
    //    degrades to "no prefetch", matching Python's broad except.
    let prefetch_limit = if matches!(level, ReasoningLevel::Minimal) {
        10
    } else {
        25
    };
    let prefetch: Option<String> = match embedder.embed(query).await {
        Ok(embedding) => {
            prefetch_observations(pool, workspace_name, observer, observed, &embedding, prefetch_limit)
                .await
                .unwrap_or(None)
        }
        Err(_) => None,
    };

    let user_content = build_user_content(query, prefetch.as_deref());

    // 3. Seed the conversation with system + user messages (provider-native
    //    shaping happens in the backends).
    let messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user_content}),
    ];

    // 4. Provider-backed caller (per-call knobs left unset so the level's own
    //    ModelConfig values apply) + the dialectic tool executor.
    let caller = HonchoCaller::new(
        llm_http,
        keys,
        level_settings.model_config.clone(),
        settings.effective_max_output_tokens(level),
    );
    let tool_choice = level_settings
        .tool_choice
        .as_ref()
        .map(|choice| Value::String(choice.clone()));
    let tools = if matches!(level, ReasoningLevel::Minimal) {
        dialectic_tools_minimal()
    } else {
        dialectic_tools()
    };
    let executor = DialecticToolExecutor {
        pool,
        ctx: ToolContext {
            workspace_name: workspace_name.to_string(),
            observer: observer.to_string(),
            observed: observed.to_string(),
            session_name: session_name.map(str::to_string),
        },
        embedder,
    };

    // 5. Run the agentic loop and return the synthesized answer text.
    let response = execute_tool_loop(
        &caller,
        &executor,
        &user_content,
        Some(&messages),
        &tools,
        tool_choice.as_ref(),
        level_settings.max_tool_iterations as usize,
        Some(settings.max_input_tokens as usize),
    )
    .await
    .map_err(DialecticError::ToolLoop)?;

    Ok(match response.content {
        Value::String(text) => text,
        other => other.to_string(),
    })
}

/// A boxed `'static` stream of answer-text chunks (boxing erases the transport +
/// embedder type params so the stream — which owns the response body — is
/// independent of the borrows used during setup).
pub type AnswerStream = std::pin::Pin<Box<dyn Stream<Item = String> + Send>>;

/// Streaming counterpart of [`answer`], porting `DialecticAgent.answer_stream` +
/// `tool_loop.py` `stream_final`: the agentic tool loop runs non-streaming to
/// settle the conversation, then the final no-tool-call turn is re-issued as a
/// streaming call and its text deltas are yielded.
///
/// Deviation: Python pins the streaming retry to the tool loop's "winning" attempt
/// plan (so a fallback that settled the loop also serves the stream). Here the
/// streaming call uses the level's primary `ModelConfig` and has no retry/fallback
/// (the non-streaming loop already succeeded); a connect failure ends the stream.
#[allow(clippy::too_many_arguments)]
pub async fn answer_stream<H, E>(
    pool: &PgPool,
    llm_http: &H,
    keys: TransportApiKeys,
    embedder: E,
    settings: &DialecticSettings,
    workspace_name: &str,
    session_name: Option<&str>,
    observer: &str,
    observed: &str,
    observer_card: Option<&[String]>,
    observed_card: Option<&[String]>,
    query: &str,
    level: ReasoningLevel,
) -> Result<AnswerStream, DialecticError>
where
    H: LlmHttp + LlmStreamHttp + Sync,
    E: Embedder + Sync,
{
    let level_settings = settings.level(level);

    // 1. System prompt (+ session history), identical to `answer`.
    let mut system = agent_system_prompt(observer, observed, observer_card, observed_card);
    if settings.session_history_max_tokens > 0
        && let Some(session) = session_name
    {
        let messages = db::get_session_messages_within_token_limit(
            pool,
            workspace_name,
            session,
            settings.session_history_max_tokens,
        )
        .await
        .map_err(DialecticError::Db)?;
        let formatted: Vec<String> = messages
            .iter()
            .map(|message| {
                format_new_turn_with_timestamp(
                    message.get("content").and_then(Value::as_str).unwrap_or(""),
                    message_created_at(message),
                    message.get("peer_id").and_then(Value::as_str).unwrap_or(""),
                )
            })
            .collect();
        if let Some(section) = session_history_section(&formatted) {
            system.push_str(&section);
        }
    }

    // 2. Prefetch observations into the user message (degrades to none on error).
    let prefetch_limit = if matches!(level, ReasoningLevel::Minimal) {
        10
    } else {
        25
    };
    let prefetch: Option<String> = match embedder.embed(query).await {
        Ok(embedding) => {
            prefetch_observations(pool, workspace_name, observer, observed, &embedding, prefetch_limit)
                .await
                .unwrap_or(None)
        }
        Err(_) => None,
    };
    let user_content = build_user_content(query, prefetch.as_deref());
    let messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user_content}),
    ];

    // 3. Resolve the streaming credentials/config up front (before `keys` moves
    //    into the non-streaming caller).
    let stream_config = level_settings.model_config.clone();
    let credentials = resolve_credentials(&stream_config, &keys);
    let max_tokens = settings.effective_max_output_tokens(level);

    let caller = HonchoCaller::new(llm_http, keys, level_settings.model_config.clone(), max_tokens);
    let tool_choice = level_settings
        .tool_choice
        .as_ref()
        .map(|choice| Value::String(choice.clone()));
    let tools = if matches!(level, ReasoningLevel::Minimal) {
        dialectic_tools_minimal()
    } else {
        dialectic_tools()
    };
    let executor = DialecticToolExecutor {
        pool,
        ctx: ToolContext {
            workspace_name: workspace_name.to_string(),
            observer: observer.to_string(),
            observed: observed.to_string(),
            session_name: session_name.map(str::to_string),
        },
        embedder,
    };

    // 4. Run the non-streaming loop to settle the conversation.
    let response = execute_tool_loop(
        &caller,
        &executor,
        &user_content,
        Some(&messages),
        &tools,
        tool_choice.as_ref(),
        level_settings.max_tool_iterations as usize,
        Some(settings.max_input_tokens as usize),
    )
    .await
    .map_err(DialecticError::ToolLoop)?;

    // 5. Re-issue the settled conversation as a streaming, tool-free call.
    let stream = execute_stream(
        llm_http,
        &credentials,
        &stream_config,
        &response.final_conversation,
        max_tokens,
        None,
        None,
        None,
        None,
    )
    .await
    .map_err(|error| DialecticError::ToolLoop(ToolLoopError::Caller(error.to_string())))?;

    // Yield only the text content of each chunk; transport errors mid-stream end
    // it (the content gathered so far is preserved client-side).
    Ok(Box::pin(stream.filter_map(|chunk| async move {
        match chunk {
            Ok(chunk) if !chunk.content.is_empty() => Some(chunk.content),
            _ => None,
        }
    })))
}
