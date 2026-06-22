//! `DreamerToolExecutor` — the [`ToolExecutor`] the dream specialists drive,
//! porting the dreamer slice of `create_tool_executor` + the write/read tool
//! handlers (`_handle_create_observations_*`, `_handle_delete_observations`,
//! `_handle_update_peer_card`, `_handle_get_recent_observations`) in
//! `src/utils/agent_tools.py`.
//!
//! Python threads per-tool metadata back through `all_tool_calls` and the
//! specialist rolls it up after the loop. The Rust `execute_tool_loop` records
//! only the tool-result string, so this executor accumulates the same rollups
//! itself behind a `Mutex` ([`DreamerToolMetrics`]); the specialist reads them
//! after the loop. Telemetry events fire best-effort with `iteration = 0` (the
//! Rust loop does not expose a per-call iteration counter — these events are
//! observability-only, documented deviation).

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde_json::Value;
use sqlx::PgPool;

use super::handlers::{
    ObservationFailure, create_observations, parse_observation, render_recent_observations,
};
use super::tools::{MAX_PEER_CARD_FACTS, validate_peer_card_entry};
use crate::db;
use crate::dialectic::{Embedder, ToolContext, handle_search_memory, handle_search_messages};
use crate::telemetry::Emitter;
use crate::telemetry::events::{
    AgentToolConclusionsCreatedEvent, AgentToolConclusionsDeletedEvent,
    AgentToolPeerCardUpdatedEvent,
};
use chrono::Utc;

/// Specialist-run rollups accumulated across tool calls (the fields the
/// `DreamSpecialistEvent` reports). Mirrors the per-`all_tool_calls` aggregation
/// in `BaseSpecialist.run`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DreamerToolMetrics {
    pub created_observation_count: i64,
    pub deleted_observation_count: i64,
    pub created_counts_by_level: BTreeMap<String, i64>,
    pub deleted_counts_by_level: BTreeMap<String, i64>,
    pub peer_card_updated: bool,
    pub search_tool_calls_count: i64,
}

const REJECTED_SAMPLE_CAP: usize = 3;
const REJECTED_SAMPLE_LINE_LIMIT: usize = 120;

/// Outcome of [`normalize_peer_card`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerCardNormalization {
    pub normalized: Vec<String>,
    pub rejected_count: usize,
    pub rejected_samples: Vec<String>,
}

/// Python `str(item)` for a JSON scalar (used when a peer-card entry isn't a
/// plain string). Faithful for the common cases; arrays/objects fall back to
/// compact JSON (vanishingly rare given the tool schema is `array<string>`).
fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Port of the normalize/validate/dedupe loop in `_handle_update_peer_card`:
/// trim, drop empties, structurally validate (allowed prefix + body + length),
/// case-insensitive whitespace-normalized dedupe; track rejects + a capped
/// sample (each truncated to 120 chars).
pub fn normalize_peer_card(items: &[String]) -> PeerCardNormalization {
    let mut normalized: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut rejected_count = 0usize;
    let mut rejected_samples: Vec<String> = Vec::new();

    for item in items {
        let line = item.trim();
        if line.is_empty() {
            continue;
        }
        if !validate_peer_card_entry(line) {
            rejected_count += 1;
            if rejected_samples.len() < REJECTED_SAMPLE_CAP {
                rejected_samples.push(line.chars().take(REJECTED_SAMPLE_LINE_LIMIT).collect());
            }
            continue;
        }
        let key = line.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        normalized.push(line.to_string());
    }

    PeerCardNormalization {
        normalized,
        rejected_count,
        rejected_samples,
    }
}

/// Port of `_format_rejection_feedback`: a self-correction hint for the model.
/// `scope` is grammar glue ("all 3" or "3 of 12").
fn format_rejection_feedback(
    scope: &str,
    rejected_count: usize,
    rejected_samples: &[String],
) -> String {
    let samples_block = if rejected_samples.is_empty() {
        String::new()
    } else {
        let sample_lines = rejected_samples
            .iter()
            .map(|s| format!("  - {s:?}"))
            .collect::<Vec<_>>()
            .join("\n");
        let extra = if rejected_count > rejected_samples.len() {
            format!(" (+{} more)", rejected_count - rejected_samples.len())
        } else {
            String::new()
        };
        format!(" Examples of rejected entries{extra}:\n{sample_lines}")
    };
    format!(
        "Rejected {scope} entries for failing structural validation. \
         Each entry must start with one of `IDENTITY: `, `ATTRIBUTE: `, \
         `RELATIONSHIP: `, or `INSTRUCTION: ` and stay under the per-entry \
         length cap.{samples_block}"
    )
}

/// The dreamer's [`ToolExecutor`] (port of the dreamer slice of
/// `create_tool_executor`). `ctx` carries `(workspace, observer, observed,
/// session)`; the extra fields cover peer-card gating, telemetry attribution,
/// the (injected) `message_created_at`, and dedup.
pub struct DreamerToolExecutor<'a, E: Embedder> {
    pub pool: &'a PgPool,
    pub ctx: ToolContext,
    pub embedder: &'a E,
    pub include_observation_ids: bool,
    pub peer_card_create: bool,
    pub run_id: String,
    pub agent_type: String,
    pub parent_category: String,
    pub emitter: &'a dyn Emitter,
    pub honcho_version: Option<String>,
    pub message_created_at: String,
    pub deduplicate: bool,
    pub metrics: Mutex<DreamerToolMetrics>,
}

impl<'a, E: Embedder + Sync> DreamerToolExecutor<'a, E> {
    /// Construct with a fresh metrics accumulator and `message_created_at` set to
    /// `utc_now_iso()` (Python's value when there are no current messages).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pool: &'a PgPool,
        ctx: ToolContext,
        embedder: &'a E,
        include_observation_ids: bool,
        peer_card_create: bool,
        run_id: String,
        agent_type: String,
        parent_category: String,
        emitter: &'a dyn Emitter,
        honcho_version: Option<String>,
        deduplicate: bool,
    ) -> Self {
        Self {
            pool,
            ctx,
            embedder,
            include_observation_ids,
            peer_card_create,
            run_id,
            agent_type,
            parent_category,
            emitter,
            honcho_version,
            message_created_at: crate::representation::format_datetime_utc(Utc::now()),
            deduplicate,
            metrics: Mutex::new(DreamerToolMetrics::default()),
        }
    }

    /// Snapshot the accumulated rollups (called by the specialist after the loop).
    pub fn metrics_snapshot(&self) -> DreamerToolMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn telemetry_ready(&self) -> bool {
        !self.run_id.is_empty() && !self.agent_type.is_empty() && !self.parent_category.is_empty()
    }

    async fn handle_create(&self, input: &Value, forced_level: &str) -> Result<String, String> {
        let raw = match input.get("observations").and_then(Value::as_array) {
            Some(arr) if !arr.is_empty() => arr,
            _ => return Ok("ERROR: observations list is empty".to_string()),
        };

        let mut observations = Vec::new();
        let mut validation_failures: Vec<ObservationFailure> = Vec::new();
        for obs in raw {
            match parse_observation(obs, forced_level) {
                Ok(parsed) => observations.push(parsed),
                Err(failure) => validation_failures.push(failure),
            }
        }
        if observations.is_empty() {
            return Ok(format!(
                "ERROR: All observations failed validation: {}",
                join_failures(&validation_failures)
            ));
        }

        let out = create_observations(
            self.pool,
            self.embedder,
            &self.ctx.workspace_name,
            &self.ctx.observer,
            &self.ctx.observed,
            self.ctx.session_name.as_deref(),
            observations,
            &[],
            &self.message_created_at,
            self.deduplicate,
        )
        .await
        .map_err(|e| e.to_string())?;

        let mut all_failures = validation_failures;
        all_failures.extend(out.failed.iter().cloned());

        let level_count = |level: &str| out.created_levels.iter().filter(|l| *l == level).count();
        let explicit = level_count("explicit");
        let deductive = level_count("deductive");
        let inductive = level_count("inductive");
        let contradiction = level_count("contradiction");

        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.created_observation_count += out.created_count as i64;
            for level in &out.created_levels {
                *metrics.created_counts_by_level.entry(level.clone()).or_insert(0) += 1;
            }
        }

        if self.telemetry_ready() {
            self.emitter.emit(&AgentToolConclusionsCreatedEvent {
                timestamp: Utc::now(),
                run_id: self.run_id.clone(),
                iteration: 0,
                parent_category: self.parent_category.clone(),
                agent_type: self.agent_type.clone(),
                workspace_name: self.ctx.workspace_name.clone(),
                observer: self.ctx.observer.clone(),
                observed: self.ctx.observed.clone(),
                conclusion_count: out.created_count as i64,
                levels: out.created_levels.clone(),
            });
        }

        let mut response = format!(
            "Created {} observations for {} by {} ({explicit} explicit, {deductive} deductive, {inductive} inductive, {contradiction} contradiction)",
            out.created_count, self.ctx.observed, self.ctx.observer
        );
        if !all_failures.is_empty() {
            response.push_str(&format!(
                "\nFailed {}: {}",
                all_failures.len(),
                join_failures(&all_failures)
            ));
        }
        Ok(response)
    }

    async fn handle_delete(&self, input: &Value) -> Result<String, String> {
        let ids: Vec<String> = match input.get("observation_ids").and_then(Value::as_array) {
            Some(arr) if !arr.is_empty() => {
                arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
            }
            _ => return Ok("ERROR: observation_ids list is empty".to_string()),
        };

        let deleted = db::delete_documents(
            self.pool,
            &self.ctx.workspace_name,
            &ids,
            &self.ctx.observer,
            &self.ctx.observed,
            None,
        )
        .await
        .map_err(|e| e.to_string())?;

        let deleted_count = deleted.len();
        let levels: Vec<String> = deleted.iter().filter_map(|(_, level)| level.clone()).collect();
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.deleted_observation_count += deleted_count as i64;
            for level in &levels {
                *metrics.deleted_counts_by_level.entry(level.clone()).or_insert(0) += 1;
            }
        }

        if deleted_count > 0 && self.telemetry_ready() {
            self.emitter.emit(&AgentToolConclusionsDeletedEvent {
                timestamp: Utc::now(),
                run_id: self.run_id.clone(),
                iteration: 0,
                parent_category: self.parent_category.clone(),
                agent_type: self.agent_type.clone(),
                workspace_name: self.ctx.workspace_name.clone(),
                observer: self.ctx.observer.clone(),
                observed: self.ctx.observed.clone(),
                conclusion_count: deleted_count as i64,
                levels,
            });
        }

        Ok(format!("Deleted {deleted_count} observations"))
    }

    async fn handle_update_peer_card(&self, input: &Value) -> Result<String, String> {
        if !self.peer_card_create {
            return Ok(
                "Peer card creation is disabled for this workspace/session configuration."
                    .to_string(),
            );
        }
        let raw = match input.get("content") {
            None | Some(Value::Null) => {
                return Ok("Peer card content was empty, no update performed.".to_string());
            }
            Some(value) => value,
        };
        let items: Vec<String> = match raw {
            Value::Array(arr) => arr.iter().map(py_str).collect(),
            other => vec![py_str(other)],
        };

        let norm = normalize_peer_card(&items);

        if norm.normalized.is_empty() {
            if norm.rejected_count > 0 {
                return Ok(format_rejection_feedback(
                    &format!("all {}", norm.rejected_count),
                    norm.rejected_count,
                    &norm.rejected_samples,
                ));
            }
            return Ok(
                "Peer card content was empty after normalization, no update performed.".to_string(),
            );
        }

        let mut normalized = norm.normalized;
        if normalized.len() > MAX_PEER_CARD_FACTS {
            normalized.truncate(MAX_PEER_CARD_FACTS);
        }

        db::set_peer_card(
            self.pool,
            &self.ctx.workspace_name,
            &self.ctx.observer,
            &self.ctx.observed,
            &normalized,
        )
        .await
        .map_err(|e| e.to_string())?;

        if self.telemetry_ready() {
            self.emitter.emit(&AgentToolPeerCardUpdatedEvent {
                timestamp: Utc::now(),
                run_id: self.run_id.clone(),
                iteration: 0,
                parent_category: self.parent_category.clone(),
                agent_type: self.agent_type.clone(),
                workspace_name: self.ctx.workspace_name.clone(),
                observer: self.ctx.observer.clone(),
                observed: self.ctx.observed.clone(),
                facts_count: normalized.len() as i64,
            });
        }

        {
            self.metrics.lock().unwrap().peer_card_updated = true;
        }

        let mut content = format!(
            "Updated peer card for {} by {} with {} entries.",
            self.ctx.observed,
            self.ctx.observer,
            normalized.len()
        );
        if norm.rejected_count > 0 {
            let total = normalized.len() + norm.rejected_count;
            content.push(' ');
            content.push_str(&format_rejection_feedback(
                &format!("{} of {}", norm.rejected_count, total),
                norm.rejected_count,
                &norm.rejected_samples,
            ));
        }
        Ok(content)
    }

    async fn handle_get_recent(&self, input: &Value) -> Result<String, String> {
        let session_only = input.get("session_only").and_then(Value::as_bool).unwrap_or(false);
        let null = Value::Null;
        let limit = crate::dialectic::safe_int(input.get("limit").unwrap_or(&null), 10).min(100);
        let session = if session_only {
            self.ctx.session_name.as_deref()
        } else {
            None
        };
        let documents = db::query_documents_recent_full(
            self.pool,
            &self.ctx.workspace_name,
            &self.ctx.observer,
            &self.ctx.observed,
            session,
            limit,
        )
        .await
        .map_err(|e| e.to_string())?;
        Ok(render_recent_observations(
            &documents,
            session_only,
            self.include_observation_ids,
        ))
    }
}

/// Port of `"; ".join(f"'{preview}': {error}" ...)`.
fn join_failures(failures: &[ObservationFailure]) -> String {
    failures
        .iter()
        .map(|f| format!("'{}': {}", f.content_preview, f.error))
        .collect::<Vec<_>>()
        .join("; ")
}

impl<E: Embedder + Sync> crate::llm::tool_loop::ToolExecutor for DreamerToolExecutor<'_, E> {
    async fn execute(&self, name: &str, input: &Value) -> Result<String, String> {
        match name {
            "search_memory" => {
                {
                    self.metrics.lock().unwrap().search_tool_calls_count += 1;
                }
                let query = input.get("query").and_then(Value::as_str).unwrap_or("");
                let embedding = self.embedder.embed(query).await?;
                handle_search_memory(self.pool, &self.ctx, input, &embedding)
                    .await
                    .map_err(|e| e.to_string())
            }
            "search_messages" => {
                {
                    self.metrics.lock().unwrap().search_tool_calls_count += 1;
                }
                let query = input.get("query").and_then(Value::as_str).unwrap_or("");
                let embedding = if query.is_empty() {
                    Vec::new()
                } else {
                    self.embedder.embed(query).await?
                };
                handle_search_messages(self.pool, &self.ctx, input, &embedding)
                    .await
                    .map_err(|e| e.to_string())
            }
            "get_recent_observations" => self.handle_get_recent(input).await,
            "create_observations_deductive" => self.handle_create(input, "deductive").await,
            "create_observations_inductive" => self.handle_create(input, "inductive").await,
            "delete_observations" => self.handle_delete(input).await,
            "update_peer_card" => self.handle_update_peer_card(input).await,
            other => Err(format!("Unknown or unsupported dreamer tool: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_dedupes_and_rejects() {
        let items = vec![
            "IDENTITY: Name: Alice".to_string(),
            "  IDENTITY:  name:  alice  ".to_string(), // dupe (case/space-normalized)
            "TRAIT: Analytical".to_string(),           // rejected (bad prefix)
            "".to_string(),                            // skipped (empty)
            "ATTRIBUTE: Location: NYC".to_string(),
        ];
        let norm = normalize_peer_card(&items);
        assert_eq!(
            norm.normalized,
            vec!["IDENTITY: Name: Alice".to_string(), "ATTRIBUTE: Location: NYC".to_string()]
        );
        assert_eq!(norm.rejected_count, 1);
        assert_eq!(norm.rejected_samples, vec!["TRAIT: Analytical".to_string()]);
    }

    #[test]
    fn rejection_feedback_lists_samples() {
        let feedback =
            format_rejection_feedback("all 2", 2, &["TRAIT: x".to_string(), "foo".to_string()]);
        assert!(feedback.starts_with("Rejected all 2 entries"));
        assert!(feedback.contains("Examples of rejected entries:"));
        assert!(feedback.contains("\"TRAIT: x\""));
    }

    #[test]
    fn rejection_feedback_reports_extra_count() {
        let samples = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let feedback = format_rejection_feedback("5 of 10", 5, &samples);
        assert!(feedback.contains("(+2 more)"));
    }

    #[test]
    fn py_str_scalars() {
        assert_eq!(py_str(&Value::String("x".into())), "x");
        assert_eq!(py_str(&Value::Bool(true)), "True");
        assert_eq!(py_str(&Value::Null), "None");
        assert_eq!(py_str(&serde_json::json!(5)), "5");
    }
}
