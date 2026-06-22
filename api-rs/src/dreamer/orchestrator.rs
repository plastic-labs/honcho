//! Dream orchestrator — port of `run_dream` / `process_dream`
//! (`src/dreamer/orchestrator.py`). Runs the deduction then induction specialist,
//! aggregates their metrics, and emits a `DreamRunEvent` (always). `process_dream`
//! adds the OMNI guard-pair collection write.
//!
//! Surprisal pre-sampling is omitted (see [`super`] module docs): it is disabled
//! by default, so `exploration_hints` is always `None` and `surprisal_enabled` is
//! reported `false`. The resolved [`ResolvedConfiguration`] is supplied by the
//! caller (the worker resolves it, as it does for the other task types) rather
//! than refetched here.

use sqlx::PgPool;

use super::specialists::{SpecialistKind, SpecialistResult, run_specialist};
use crate::db;
use crate::deriver::payload::{DreamPayload, DreamType};
use crate::dialectic::Embedder;
use crate::llm::http::LlmHttp;
use crate::llm::{ModelConfig, Provider, credentials::TransportApiKeys};
use crate::producer::ResolvedConfiguration;
use crate::telemetry::Emitter;
use crate::telemetry::events::DreamRunEvent;
use chrono::Utc;

/// Per-deploy dream settings the worker threads in (port of the `DREAM.*` config
/// the orchestrator reads). The full per-level surprisal config is omitted
/// (surprisal disabled/deferred).
#[derive(Debug, Clone)]
pub struct DreamModelSettings {
    pub deduction_model_config: ModelConfig,
    pub induction_model_config: ModelConfig,
    /// `settings.DREAM.ENABLED` — the deploy-global gate.
    pub enabled: bool,
    /// `len(settings.DREAM.ENABLED_TYPES)` — reported on the event.
    pub enabled_types_count: i64,
    /// `settings.DERIVER.DEDUPLICATE` (observation dedup on write).
    pub deduplicate: bool,
}

impl Default for DreamModelSettings {
    fn default() -> Self {
        Self {
            deduction_model_config: ModelConfig::new("gpt-5.4-mini", Provider::Openai),
            induction_model_config: ModelConfig::new("gpt-5.4-mini", Provider::Openai),
            enabled: true,
            enabled_types_count: 1,
            deduplicate: true,
        }
    }
}

/// Aggregate outcome of a dream cycle (port of `DreamResult`).
#[derive(Debug, Clone)]
pub struct DreamRunOutcome {
    pub run_id: String,
    pub deduction_success: bool,
    pub induction_success: bool,
    pub total_iterations: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_duration_ms: f64,
}

/// Scheduling context threaded onto the `DreamRunEvent` (from the queue payload).
#[derive(Debug, Clone, Default)]
pub struct DreamScheduleContext {
    pub dream_type: Option<String>,
    pub trigger_reason: Option<String>,
    pub delay_reason: Option<String>,
    pub documents_since_last_dream_at_schedule: Option<i64>,
    pub document_threshold: Option<i64>,
}

/// Port of `run_dream`: run deduction then induction, aggregate, and emit a
/// `DreamRunEvent`. Returns `None` when dreams are disabled (deploy-global or the
/// resolved configuration). A specialist whose preflight DB call fails counts as
/// a failed (zero-contribution) specialist; the run still emits its event.
#[allow(clippy::too_many_arguments)]
pub async fn run_dream<H, E>(
    pool: &PgPool,
    http: &H,
    keys: TransportApiKeys,
    embedder: &E,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    session_name: Option<&str>,
    configuration: &ResolvedConfiguration,
    settings: &DreamModelSettings,
    emitter: &dyn Emitter,
    honcho_version: Option<String>,
    schedule: &DreamScheduleContext,
) -> Option<DreamRunOutcome>
where
    H: LlmHttp + Sync,
    E: Embedder + Sync,
{
    if !settings.enabled {
        return None;
    }
    let run_id = db::generate_nanoid();
    if !configuration.dream_enabled {
        return None;
    }

    let start = std::time::Instant::now();

    // Deduction first, then induction (so induction sees fresh deductive obs).
    let deduction = run_specialist(
        SpecialistKind::Deduction,
        pool,
        http,
        keys.clone(),
        embedder,
        workspace_name,
        observer,
        observed,
        session_name,
        None, // exploration_hints (surprisal disabled)
        configuration.peer_card_create,
        settings.deduction_model_config.clone(),
        &run_id,
        emitter,
        honcho_version.clone(),
        settings.deduplicate,
    )
    .await
    .ok();

    let induction = run_specialist(
        SpecialistKind::Induction,
        pool,
        http,
        keys,
        embedder,
        workspace_name,
        observer,
        observed,
        session_name,
        None,
        configuration.peer_card_create,
        settings.induction_model_config.clone(),
        &run_id,
        emitter,
        honcho_version.clone(),
        settings.deduplicate,
    )
    .await
    .ok();

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    let success = |r: &Option<SpecialistResult>| r.as_ref().map(|s| s.success).unwrap_or(false);
    let sum_i64 = |r: &Option<SpecialistResult>, f: fn(&SpecialistResult) -> i64| {
        r.as_ref().map(f).unwrap_or(0)
    };
    let deduction_success = success(&deduction);
    let induction_success = success(&induction);
    let total_iterations =
        sum_i64(&deduction, |s| s.iterations) + sum_i64(&induction, |s| s.iterations);
    let total_input_tokens =
        sum_i64(&deduction, |s| s.input_tokens) + sum_i64(&induction, |s| s.input_tokens);
    let total_output_tokens =
        sum_i64(&deduction, |s| s.output_tokens) + sum_i64(&induction, |s| s.output_tokens);

    emitter.emit(&DreamRunEvent {
        timestamp: Utc::now(),
        run_id: run_id.clone(),
        workspace_name: workspace_name.to_string(),
        session_name: session_name.map(str::to_string),
        observer: observer.to_string(),
        observed: observed.to_string(),
        specialists_run: vec!["deduction".to_string(), "induction".to_string()],
        deduction_success,
        induction_success,
        surprisal_enabled: false,
        surprisal_conclusion_count: 0,
        total_iterations,
        total_input_tokens,
        total_output_tokens,
        total_duration_ms: duration_ms,
        dream_type: schedule.dream_type.clone(),
        enabled_types_count: settings.enabled_types_count,
        trigger_reason: schedule.trigger_reason.clone(),
        delay_reason: schedule.delay_reason.clone(),
        documents_since_last_dream_at_schedule: schedule.documents_since_last_dream_at_schedule,
        document_threshold: schedule.document_threshold,
    });

    Some(DreamRunOutcome {
        run_id,
        deduction_success,
        induction_success,
        total_iterations,
        total_input_tokens,
        total_output_tokens,
        total_duration_ms: duration_ms,
    })
}

/// Port of `process_dream`: run the OMNI dream cycle, then (on a real run) the
/// guard-pair collection write that advances `dream.last_dream_at` /
/// `last_dream_document_count` together. `now_iso` is injected (Python uses
/// `datetime.now(UTC).isoformat()`). Errors are logged and swallowed so the queue
/// item is still marked processed (matching Python's non-re-raising `except`).
#[allow(clippy::too_many_arguments)]
pub async fn process_dream<H, E>(
    pool: &PgPool,
    http: &H,
    keys: TransportApiKeys,
    embedder: &E,
    payload: &DreamPayload,
    workspace_name: &str,
    configuration: &ResolvedConfiguration,
    settings: &DreamModelSettings,
    emitter: &dyn Emitter,
    honcho_version: Option<String>,
    now_iso: &str,
) where
    H: LlmHttp + Sync,
    E: Embedder + Sync,
{
    match payload.dream_type {
        DreamType::Omni => {
            let schedule = DreamScheduleContext {
                dream_type: Some(payload.dream_type.as_str().to_string()),
                trigger_reason: payload.trigger_reason.clone(),
                delay_reason: payload.delay_reason.clone(),
                documents_since_last_dream_at_schedule: payload
                    .documents_since_last_dream_at_schedule,
                document_threshold: payload.document_threshold,
            };
            let outcome = run_dream(
                pool,
                http,
                keys,
                embedder,
                workspace_name,
                &payload.observer,
                &payload.observed,
                payload.session_name.as_deref(),
                configuration,
                settings,
                emitter,
                honcho_version,
                &schedule,
            )
            .await;

            // Guard-pair write only on a real run (Python: `if result is not None`).
            if outcome.is_some()
                && let Err(error) = db::record_dream_guard(
                    pool,
                    workspace_name,
                    &payload.observer,
                    &payload.observed,
                    now_iso,
                )
                .await
            {
                tracing::error!(?error, "dream guard-pair write failed");
            }
        }
    }
}
