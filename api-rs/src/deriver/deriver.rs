//! Port of `src/deriver/deriver.py` (`process_representation_tasks_batch`): the
//! minimal deriver's single-LLM-call batch processor. The pure token-accounting
//! arithmetic ([`compute_token_breakdown`]) landed first; the full async
//! orchestrator (prompt → structured-output LLM call → save → telemetry) follows
//! here now that the LLM, save, and telemetry layers are ported.

use std::collections::HashSet;
use std::time::Instant;

use chrono::Utc;
use serde_json::json;

use crate::db::BatchMessage;
use crate::dialectic::{Embedder, format_new_turn_with_timestamp};
use crate::llm::conversation::{count_message_tokens, truncate_messages_to_fit};
use crate::llm::credentials::TransportApiKeys;
use crate::llm::executor::HonchoCaller;
use crate::llm::http::LlmHttp;
use crate::llm::{ModelConfig, StructuredOutputMode};
use crate::producer::ResolvedConfiguration;
use crate::representation::Representation;
use crate::representation_manager::save_representation;
use crate::structured_output::finalize_structured_output;
use crate::telemetry::Emitter;
use crate::telemetry::events::RepresentationCompletedEvent;

use super::prompts::{estimate_deriver_prompt_tokens, minimal_deriver_prompt};

/// The deriver model + write knobs read by [`process_representation_tasks_batch`],
/// porting the subset of `settings.DERIVER` / `settings.LLM` that fixes the LLM
/// call shape. (The polling/batching subset lives in [`super::settings`].)
#[derive(Debug, Clone)]
pub struct DeriverModelSettings {
    /// `settings.DERIVER.MODEL_CONFIG` (default openai / gpt-5.4-mini).
    pub model_config: ModelConfig,
    /// `settings.LLM.DEFAULT_MAX_TOKENS` (2500) — used when the model config
    /// pins no `max_output_tokens`.
    pub default_max_tokens: i64,
    /// `settings.DERIVER.MAX_INPUT_TOKENS` (25000) — input truncation cap that
    /// also drives the `hit_input_token_cap` telemetry signal.
    pub max_input_tokens: i64,
    /// `settings.DERIVER.DEDUPLICATE` (true) — document dedup on write.
    pub deduplicate: bool,
}

impl Default for DeriverModelSettings {
    fn default() -> Self {
        Self {
            model_config: ModelConfig::new("gpt-5.4-mini", crate::llm::Provider::Openai),
            default_max_tokens: 2500,
            max_input_tokens: 25000,
            deduplicate: true,
        }
    }
}

impl DeriverModelSettings {
    /// Read from the process environment (Python `DERIVER_*` / `LLM_*`).
    pub fn from_env() -> Self {
        Self::from_pairs(std::env::vars())
    }

    /// Read from an arbitrary key/value source (testable). `model_config` honors
    /// nested `DERIVER_MODEL_CONFIG__*` overrides; missing/unparseable scalars
    /// fall back to the Python defaults.
    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let values = super::settings::collect_env(pairs);
        let defaults = Self::default();
        Self {
            model_config: defaults
                .model_config
                .with_env_overrides(&values, "DERIVER_MODEL_CONFIG"),
            default_max_tokens: super::settings::parse_or(
                &values,
                "LLM_DEFAULT_MAX_TOKENS",
                defaults.default_max_tokens,
            ),
            max_input_tokens: super::settings::parse_or(
                &values,
                "DERIVER_MAX_INPUT_TOKENS",
                defaults.max_input_tokens,
            ),
            deduplicate: super::settings::parse_bool_or(
                &values,
                "DERIVER_DEDUPLICATE",
                defaults.deduplicate,
            ),
        }
    }
}

/// The collaborators [`process_representation_tasks_batch`] needs: a DB pool, the
/// LLM transport + per-transport API keys, an embedder for the write path, the
/// model settings, and the telemetry emitter. Bundled so the orchestrator's
/// signature stays legible.
pub struct DeriverBatchContext<'a, H: LlmHttp + Sync, E: Embedder + Sync> {
    pub pool: &'a sqlx::PgPool,
    pub http: &'a H,
    pub keys: TransportApiKeys,
    pub embedder: &'a E,
    pub settings: DeriverModelSettings,
    pub emitter: &'a dyn Emitter,
    /// Global `DREAM.*` scheduling knobs for `check_and_schedule_dream` (run from
    /// `save_representation` when the resolved configuration enables dreams).
    pub dream_schedule_settings: crate::dreamer::scheduler::DreamScheduleSettings,
}

/// Port of `process_representation_tasks_batch`: format the batch into the
/// prompt, run one structured-output completion, lift the result into a
/// [`Representation`], and save it to every observer collection — then build
/// (and emit) the [`RepresentationCompletedEvent`].
///
/// Returns `Ok(None)` for the two early-exit cases (no messages; reasoning
/// disabled) and `Ok(Some(event))` once processed (the same event is also passed
/// to the emitter). Returning the event is a testability affordance — Python
/// returns `None` and only emits.
///
/// Deviations from Python, all documented and observability-only:
/// - The `message_level_configuration is None` DB-fallback path is not ported;
///   the worker always supplies the resolved configuration via the queue payload.
/// - `accumulate_metric` / `log_performance_metrics` / Prometheus token tracking
///   / `LOG_OBSERVATIONS` blob logging are skipped (pure observability).
/// - `check_and_schedule_dream` runs inside [`save_representation`] when the
///   resolved configuration enables dreams (no idle-timeout debounce — see
///   [`crate::dreamer::scheduler`]).
#[allow(clippy::too_many_arguments)]
pub async fn process_representation_tasks_batch<H, E>(
    ctx: &DeriverBatchContext<'_, H, E>,
    messages: &[BatchMessage],
    configuration: &ResolvedConfiguration,
    observers: &[String],
    observed: &str,
    queue_item_message_ids: &[i64],
    hit_batch_token_cap: bool,
    was_flush_enabled: bool,
    batch_max_tokens: i64,
) -> Result<Option<RepresentationCompletedEvent>, sqlx::Error>
where
    H: LlmHttp + Sync,
    E: Embedder + Sync,
{
    if messages.is_empty() {
        return Ok(None);
    }

    let overall_start = Instant::now();

    // Sort by id (Python sorts the list in place); operate on references so the
    // caller's slice is untouched.
    let mut sorted: Vec<&BatchMessage> = messages.iter().collect();
    sorted.sort_by_key(|m| m.id);
    let earliest = sorted[0];
    let latest = sorted[sorted.len() - 1];

    // Skip if disabled.
    if !configuration.reasoning_enabled {
        return Ok(None);
    }
    let custom_instructions = configuration.reasoning_custom_instructions.as_deref();

    // Format messages with timestamps.
    let formatted_messages = sorted
        .iter()
        .map(|m| format_new_turn_with_timestamp(&m.content, m.created_at, &m.peer_name))
        .collect::<Vec<_>>()
        .join("\n");

    // Token accounting (queued vs. interleaved context).
    let prompt_tokens = estimate_deriver_prompt_tokens(custom_instructions);
    let breakdown = compute_token_breakdown(messages, queue_item_message_ids);

    // Build prompt.
    let prompt = minimal_deriver_prompt(observed, &formatted_messages, custom_instructions);

    let context_prep_duration = overall_start.elapsed().as_secs_f64() * 1000.0;

    // Validation on settings means max_tokens is always > 0.
    let base_model_config = &ctx.settings.model_config;
    let max_tokens = base_model_config
        .max_output_tokens
        .unwrap_or(ctx.settings.default_max_tokens);

    // Input-token cap signal + truncation, mirroring the toolless branch of
    // `honcho_llm_call` (count the single user message against MAX_INPUT_TOKENS).
    let base_messages = vec![json!({"role": "user", "content": prompt})];
    let max_input = ctx.settings.max_input_tokens.max(0) as usize;
    let hit_input_token_cap = count_message_tokens(&base_messages) > max_input;
    let call_messages = truncate_messages_to_fit(&base_messages, max_input, true);

    // Single LLM call (structured output): complete, then validate+repair the
    // returned JSON content into a PromptRepresentation.
    tracing::info!(observed = %observed, msgs = sorted.len(), "deriver: calling LLM");
    let llm_start = Instant::now();
    let mut caller = HonchoCaller::new(
        ctx.http,
        ctx.keys.clone(),
        base_model_config.clone(),
        max_tokens,
    );
    // Force the PromptRepresentation shape via structured output. The default
    // remains strict json_schema; json_object mode swaps to prompt-injected schema
    // for OpenAI-compatible providers that reject Structured Outputs.
    caller.json_mode = true;
    caller.response_format = Some(match base_model_config.structured_output_mode {
        Some(StructuredOutputMode::JsonObject) => {
            crate::structured_output::prompt_representation_json_object_response_format()
        }
        _ => crate::structured_output::prompt_representation_response_format(),
    });
    let response = caller
        .complete_single(&call_messages)
        .await
        .map_err(|e| sqlx::Error::Protocol(format!("deriver llm call failed: {e}")))?;
    let llm_duration = llm_start.elapsed().as_secs_f64() * 1000.0;

    tracing::info!(
        content_len = response.content.to_string().len(),
        "deriver: parsing structured output"
    );
    let prompt_repr = finalize_structured_output(&response.content);
    tracing::info!("deriver: structured output parsed; saving representation");

    // Only the observed peer's own messages anchor the observations.
    let message_ids: Vec<i64> = sorted
        .iter()
        .filter(|m| m.peer_name == observed)
        .map(|m| m.id)
        .collect();

    let observations = Representation::from_prompt_representation(
        &prompt_repr,
        &message_ids,
        &latest.session_name,
        latest.created_at,
    );

    let mut successful_observer_count: i64 = 0;
    if observations.is_empty() || message_ids.is_empty() {
        tracing::warn!(
            earliest = earliest.id,
            latest = latest.id,
            workspace = %latest.workspace_name,
            session = %latest.session_name,
            "Deriver generated zero observations"
        );
    } else {
        for observer in observers {
            match save_representation(
                ctx.pool,
                ctx.embedder,
                &latest.workspace_name,
                observer,
                observed,
                &observations,
                &message_ids,
                &latest.session_name,
                latest.created_at,
                ctx.settings.deduplicate,
                if configuration.dream_enabled {
                    Some(&ctx.dream_schedule_settings)
                } else {
                    None
                },
            )
            .await
            {
                Ok(_) => successful_observer_count += 1,
                Err(e) => {
                    tracing::error!(observer = %observer, "Failed to save representation: {e:?}");
                }
            }
        }
    }

    let overall_duration = overall_start.elapsed().as_secs_f64() * 1000.0;

    // Data-quality invariants — best-effort, telemetry never bleeds into the
    // deriver path, but log loudly so analytics alerting catches silent
    // estimator failures at the source.
    if response.input_tokens < breakdown.messages_tokens {
        tracing::warn!(
            response_input_tokens = response.input_tokens,
            messages_tokens = breakdown.messages_tokens,
            observed = %observed,
            latest = %latest.public_id,
            "token-breakdown invariant violated: response.input_tokens < messages_tokens"
        );
    }
    if prompt_tokens == 0 {
        tracing::warn!(
            observed = %observed,
            latest = %latest.public_id,
            "prompt_scaffold_tokens estimated as 0 — estimate_deriver_prompt_tokens may have failed silently"
        );
    }

    let event = RepresentationCompletedEvent {
        timestamp: Utc::now(),
        workspace_name: latest.workspace_name.clone(),
        session_name: latest.session_name.clone(),
        observed: observed.to_string(),
        queue_items_processed: queue_item_message_ids.len() as i64,
        earliest_message_id: earliest.public_id.clone(),
        latest_message_id: latest.public_id.clone(),
        message_count: sorted.len() as i64,
        explicit_conclusion_count: observations.explicit.len() as i64,
        context_preparation_ms: context_prep_duration,
        llm_call_ms: llm_duration,
        total_duration_ms: overall_duration,
        input_tokens: breakdown.messages_tokens,
        total_input_tokens: response.input_tokens,
        output_tokens: response.output_tokens,
        queued_message_count: breakdown.queued_message_count as i64,
        prompt_message_count: breakdown.prompt_message_count as i64,
        prompt_message_tokens: breakdown.prompt_message_tokens,
        extra_context_message_count: breakdown.extra_context_message_count as i64,
        extra_context_tokens: breakdown.extra_context_tokens,
        prompt_scaffold_tokens: prompt_tokens as i64,
        batch_max_tokens,
        max_input_tokens: ctx.settings.max_input_tokens,
        was_flush_enabled,
        hit_batch_token_cap,
        hit_input_token_cap,
        observer_count: successful_observer_count,
    };

    ctx.emitter.emit(&event);

    Ok(Some(event))
}

/// The per-batch token breakdown computed in `process_representation_tasks_batch`.
///
/// `messages_tokens` counts only the messages actually being processed (those in
/// `queue_item_message_ids`); the rest are interleaving context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBreakdown {
    /// Tokens from the queued messages only (Python `messages_tokens`).
    pub messages_tokens: i64,
    /// Number of queued message ids (Python `queued_message_count`).
    pub queued_message_count: usize,
    /// Number of messages in the prompt incl. context (Python `prompt_message_count`).
    pub prompt_message_count: usize,
    /// Tokens across all prompt messages (Python `prompt_message_tokens`).
    pub prompt_message_tokens: i64,
    /// Context-only message count, clamped at 0 (Python `extra_context_message_count`).
    pub extra_context_message_count: usize,
    /// Context-only tokens, clamped at 0 (Python `extra_context_tokens`).
    pub extra_context_tokens: i64,
}

/// Port of the token-breakdown arithmetic in `process_representation_tasks_batch`.
pub fn compute_token_breakdown(
    messages: &[BatchMessage],
    queue_item_message_ids: &[i64],
) -> TokenBreakdown {
    let queued_ids: HashSet<i64> = queue_item_message_ids.iter().copied().collect();

    // messages_tokens: only messages whose id is in the queue-item set.
    let messages_tokens: i64 = messages
        .iter()
        .filter(|m| queued_ids.contains(&m.id))
        .map(|m| m.token_count as i64)
        .sum();

    let queued_message_count = queue_item_message_ids.len();
    let prompt_message_count = messages.len();
    let prompt_message_tokens: i64 = messages.iter().map(|m| m.token_count as i64).sum();
    let extra_context_message_count = prompt_message_count.saturating_sub(queued_message_count);
    let extra_context_tokens = (prompt_message_tokens - messages_tokens).max(0);

    TokenBreakdown {
        messages_tokens,
        queued_message_count,
        prompt_message_count,
        prompt_message_tokens,
        extra_context_message_count,
        extra_context_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    #[test]
    fn model_settings_from_pairs_defaults_and_overrides() {
        let s = DeriverModelSettings::from_pairs(Vec::<(String, String)>::new());
        assert_eq!(s.default_max_tokens, 2500);
        assert_eq!(s.max_input_tokens, 25000);
        assert!(s.deduplicate);
        assert_eq!(s.model_config.model, "gpt-5.4-mini");

        let s = DeriverModelSettings::from_pairs([
            ("DERIVER_MODEL_CONFIG__MODEL", "gpt-5.4"),
            ("DERIVER_MAX_INPUT_TOKENS", "12000"),
            ("LLM_DEFAULT_MAX_TOKENS", "3000"),
            ("DERIVER_DEDUPLICATE", "false"),
            (
                "DERIVER_MODEL_CONFIG__STRUCTURED_OUTPUT_MODE",
                "json_object",
            ),
        ]);
        assert_eq!(s.model_config.model, "gpt-5.4");
        assert_eq!(
            s.model_config.structured_output_mode,
            Some(StructuredOutputMode::JsonObject)
        );
        assert_eq!(s.max_input_tokens, 12000);
        assert_eq!(s.default_max_tokens, 3000);
        assert!(!s.deduplicate);
    }

    fn msg(id: i64, token_count: i32) -> BatchMessage {
        BatchMessage {
            id,
            public_id: format!("pub_{id}"),
            content: String::new(),
            created_at: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
            peer_name: "p".to_string(),
            token_count,
            session_name: "s".to_string(),
            workspace_name: "w".to_string(),
        }
    }

    #[test]
    fn breakdown_separates_queued_from_context() {
        // ids 1,2 queued; 3,4 are interleaving context.
        let messages = vec![msg(1, 10), msg(2, 20), msg(3, 5), msg(4, 7)];
        let queued = vec![1, 2];
        let b = compute_token_breakdown(&messages, &queued);
        assert_eq!(b.messages_tokens, 30);
        assert_eq!(b.queued_message_count, 2);
        assert_eq!(b.prompt_message_count, 4);
        assert_eq!(b.prompt_message_tokens, 42);
        assert_eq!(b.extra_context_message_count, 2);
        assert_eq!(b.extra_context_tokens, 12);
    }

    #[test]
    fn breakdown_all_queued_has_no_extra_context() {
        let messages = vec![msg(1, 10), msg(2, 20)];
        let queued = vec![1, 2];
        let b = compute_token_breakdown(&messages, &queued);
        assert_eq!(b.messages_tokens, 30);
        assert_eq!(b.extra_context_message_count, 0);
        assert_eq!(b.extra_context_tokens, 0);
    }

    #[test]
    fn breakdown_clamps_when_queued_ids_exceed_messages() {
        // A queued id with no matching message (already-processed context window):
        // counts toward queued_message_count but contributes no tokens, and the
        // clamps keep extras non-negative.
        let messages = vec![msg(1, 10)];
        let queued = vec![1, 99, 100];
        let b = compute_token_breakdown(&messages, &queued);
        assert_eq!(b.messages_tokens, 10);
        assert_eq!(b.queued_message_count, 3);
        assert_eq!(b.prompt_message_count, 1);
        assert_eq!(b.extra_context_message_count, 0); // saturating_sub
        assert_eq!(b.extra_context_tokens, 0); // max(10 - 10, 0)
    }
}
