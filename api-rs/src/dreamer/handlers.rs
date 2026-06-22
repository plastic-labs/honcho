//! Dreamer write-tool data layer — port of `create_observations`, the
//! `ObservationInput` validation in `_handle_create_observations_impl`, and the
//! `get_recent_observations` rendering (`src/utils/agent_tools.py`).
//!
//! The validation error *messages* are not byte-identical to pydantic's verbose
//! `ValidationError` text (which would be impractical to reproduce); the
//! *behavior* — which observations pass, the per-level counts, the document
//! metadata shape, and the response template — is faithful. The non-deterministic
//! `message_created_at` (`utc_now_iso()`) is injected by the caller.

use serde_json::{Map, Value};
use sqlx::PgPool;

use crate::db::{self, DocumentToCreate};
use crate::dialectic::Embedder;
use crate::representation::Representation;

/// Levels that carry `source_ids` (and a `source_ids` column value).
const SOURCED_LEVELS: [&str; 3] = ["deductive", "inductive", "contradiction"];
const PATTERN_TYPES: [&str; 5] =
    ["preference", "behavior", "personality", "tendency", "correlation"];
const CONFIDENCE_LEVELS: [&str; 3] = ["high", "medium", "low"];

/// Validated observation input (port of `schemas.ObservationInput`).
#[derive(Debug, Clone)]
pub struct ObservationInput {
    pub content: String,
    pub level: String,
    pub source_ids: Option<Vec<String>>,
    pub premises: Option<Vec<String>>,
    pub sources: Option<Vec<String>>,
    pub pattern_type: Option<String>,
    pub confidence: Option<String>,
}

/// A per-observation failure (port of `ObservationFailure`): a 50-char content
/// preview plus the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationFailure {
    pub content_preview: String,
    pub error: String,
}

/// First 50 Unicode code points of the raw content (Python `str(...)[:50]`).
fn content_preview(raw: &Value) -> String {
    let s = raw
        .get("content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_default();
    s.chars().take(50).collect()
}

fn opt_str_list(value: &Value, key: &str) -> Result<Option<Vec<String>>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item.as_str() {
                    Some(s) => out.push(s.to_string()),
                    None => return Err(format!("'{key}' must be a list of strings")),
                }
            }
            Ok(Some(out))
        }
        Some(_) => Err(format!("'{key}' must be a list of strings")),
    }
}

fn opt_enum(value: &Value, key: &str, allowed: &[&str]) -> Result<Option<String>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if allowed.contains(&s.as_str()) => Ok(Some(s.clone())),
        Some(_) => Err(format!("'{key}' must be one of {allowed:?}")),
    }
}

/// Port of `schemas.ObservationInput.model_validate` + the `level` injection in
/// `_handle_create_observations_impl`: validate one raw observation at the given
/// (already-resolved) `level`. Content must be non-empty (NUL stripped after);
/// deductive/inductive require non-empty `source_ids`; contradiction requires
/// ≥2. Returns the validation failure (with content preview) on rejection.
pub fn parse_observation(raw: &Value, level: &str) -> Result<ObservationInput, ObservationFailure> {
    let fail = |error: String| ObservationFailure {
        content_preview: content_preview(raw),
        error: format!("Validation failed: {error}"),
    };

    // content: required, min_length 1; NUL chars stripped (sanitize_content).
    let content_raw = match raw.get("content") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(Value::String(_)) | None => {
            return Err(fail("'content' must be a non-empty string".to_string()));
        }
        Some(_) => return Err(fail("'content' must be a string".to_string())),
    };
    let content = content_raw.replace('\u{0}', "");

    let source_ids = opt_str_list(raw, "source_ids").map_err(&fail)?;
    let premises = opt_str_list(raw, "premises").map_err(&fail)?;
    let sources = opt_str_list(raw, "sources").map_err(&fail)?;
    let pattern_type = opt_enum(raw, "pattern_type", &PATTERN_TYPES).map_err(&fail)?;
    let confidence = opt_enum(raw, "confidence", &CONFIDENCE_LEVELS).map_err(&fail)?;

    // model_validator: level-specific source_ids requirements.
    let has_sources = source_ids.as_ref().is_some_and(|ids| !ids.is_empty());
    match level {
        "deductive" if !has_sources => {
            return Err(fail(
                "deductive observations require 'source_ids' field with document IDs of premises"
                    .to_string(),
            ));
        }
        "inductive" if !has_sources => {
            return Err(fail(
                "inductive observations require 'source_ids' field with document IDs of sources"
                    .to_string(),
            ));
        }
        "contradiction" if source_ids.as_ref().map(|ids| ids.len()).unwrap_or(0) < 2 => {
            return Err(fail(
                "contradiction observations require 'source_ids' field with at least 2 IDs of contradicting observations"
                    .to_string(),
            ));
        }
        _ => {}
    }

    Ok(ObservationInput {
        content,
        level: level.to_string(),
        source_ids,
        premises,
        sources,
        pattern_type,
        confidence,
    })
}

/// Result of [`create_observations`] (port of `ObservationsCreatedResult`).
#[derive(Debug, Clone, Default)]
pub struct CreateObservationsOutput {
    pub created_count: usize,
    pub created_levels: Vec<String>,
    pub failed: Vec<ObservationFailure>,
}

fn str_list_value(items: &[String]) -> Value {
    Value::Array(items.iter().map(|s| Value::String(s.clone())).collect())
}

/// Build the `internal_metadata` JSON for one observation (port of
/// `DocumentMetadata(...).model_dump(exclude_none=True)`): always
/// `message_ids` + `message_created_at`; level-specific `source_ids` / `premises`
/// / `sources` / `pattern_type` / `confidence` only when applicable and non-null.
fn build_internal_metadata(obs: &ObservationInput, message_ids: &[i64], message_created_at: &str) -> Value {
    let mut meta = Map::new();
    meta.insert(
        "message_ids".to_string(),
        Value::Array(message_ids.iter().map(|&id| Value::from(id)).collect()),
    );
    meta.insert(
        "message_created_at".to_string(),
        Value::String(message_created_at.to_string()),
    );
    let sourced = SOURCED_LEVELS.contains(&obs.level.as_str());
    if sourced && let Some(ids) = &obs.source_ids {
        meta.insert("source_ids".to_string(), str_list_value(ids));
    }
    if obs.level == "deductive" && let Some(premises) = &obs.premises {
        meta.insert("premises".to_string(), str_list_value(premises));
    }
    if (obs.level == "inductive" || obs.level == "contradiction")
        && let Some(sources) = &obs.sources
    {
        meta.insert("sources".to_string(), str_list_value(sources));
    }
    if obs.level == "inductive" {
        if let Some(pt) = &obs.pattern_type {
            meta.insert("pattern_type".to_string(), Value::String(pt.clone()));
        }
        // confidence defaults to "medium" for inductive when unset.
        let confidence = obs.confidence.clone().unwrap_or_else(|| "medium".to_string());
        meta.insert("confidence".to_string(), Value::String(confidence));
    }
    Value::Object(meta)
}

/// Port of `create_observations` (agent_tools.py:849): embed the (stripped, non-
/// empty) observation contents, ensure the collection exists, then bulk-create
/// the documents with dedup. Returns the accepted count, the per-document levels
/// (after dedup), and any per-observation failures. Embeds in a single batch and,
/// on batch failure, falls back to per-observation embedding (collecting embed
/// failures) — matching the Python fallback.
#[allow(clippy::too_many_arguments)]
pub async fn create_observations<E: Embedder + Sync>(
    pool: &PgPool,
    embedder: &E,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    session_name: Option<&str>,
    observations: Vec<ObservationInput>,
    message_ids: &[i64],
    message_created_at: &str,
    deduplicate: bool,
) -> Result<CreateObservationsOutput, sqlx::Error> {
    if observations.is_empty() {
        return Ok(CreateObservationsOutput::default());
    }

    // Strip content; drop observations empty after strip.
    let normalized: Vec<ObservationInput> = observations
        .into_iter()
        .filter_map(|mut obs| {
            let stripped = obs.content.trim().to_string();
            if stripped.is_empty() {
                None
            } else {
                obs.content = stripped;
                Some(obs)
            }
        })
        .collect();
    if normalized.is_empty() {
        return Ok(CreateObservationsOutput::default());
    }

    // Ensure the collection exists (short DB scope).
    db::get_or_create_collection(pool, workspace_name, observer, observed).await?;

    // Embed (external call, no DB session held). Batch first; per-obs fallback.
    let contents: Vec<String> = normalized.iter().map(|o| o.content.clone()).collect();
    let mut failed: Vec<ObservationFailure> = Vec::new();
    let mut embedded: Vec<(ObservationInput, Vec<f32>)> = Vec::new();

    match embedder.batch_embed(&contents).await {
        Ok(embeddings) => {
            for (obs, embedding) in normalized.into_iter().zip(embeddings) {
                embedded.push((obs, embedding));
            }
        }
        Err(_) => {
            for obs in normalized {
                match embedder.embed(&obs.content).await {
                    Ok(embedding) => embedded.push((obs, embedding)),
                    Err(e) => failed.push(ObservationFailure {
                        content_preview: obs.content.chars().take(50).collect(),
                        error: format!("Embedding failed: {e}"),
                    }),
                }
            }
        }
    }

    // Build documents with level-specific metadata + source_ids column.
    let documents: Vec<DocumentToCreate> = embedded
        .into_iter()
        .map(|(obs, embedding)| {
            let internal_metadata = build_internal_metadata(&obs, message_ids, message_created_at);
            let source_ids_column = if SOURCED_LEVELS.contains(&obs.level.as_str()) {
                obs.source_ids.as_ref().map(|ids| str_list_value(ids))
            } else {
                None
            };
            DocumentToCreate {
                content: obs.content,
                session_name: session_name.map(str::to_string),
                level: obs.level,
                internal_metadata,
                embedding,
                times_derived: 1,
                source_ids: source_ids_column,
            }
        })
        .collect();

    let created_levels = if documents.is_empty() {
        Vec::new()
    } else {
        db::create_documents_returning_levels(
            pool,
            documents,
            workspace_name,
            observer,
            observed,
            deduplicate,
        )
        .await?
    };

    Ok(CreateObservationsOutput {
        created_count: created_levels.len(),
        created_levels,
        failed,
    })
}

/// Port of `_handle_get_recent_observations`'s rendering: build a representation
/// from the recent documents and format it (with ids when `include_ids`).
pub fn render_recent_observations(
    documents: &[crate::representation::Document],
    session_only: bool,
    include_ids: bool,
) -> String {
    let representation = Representation::from_documents(documents);
    let total = representation.count();
    if total == 0 {
        return "No recent observations found".to_string();
    }
    let scope = if session_only { "this session" } else { "all sessions" };
    let repr_str = if include_ids {
        representation.str_with_ids()
    } else {
        representation.to_string()
    };
    format!("Found {total} recent observations from {scope}:\n\n{repr_str}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_rejects_empty_content() {
        let raw = json!({"content": "", "source_ids": ["a"]});
        let err = parse_observation(&raw, "deductive").unwrap_err();
        assert_eq!(err.content_preview, "");
        assert!(err.error.contains("content"));
    }

    #[test]
    fn parse_requires_source_ids_for_deductive_and_inductive() {
        let raw = json!({"content": "x"});
        assert!(parse_observation(&raw, "deductive").is_err());
        assert!(parse_observation(&raw, "inductive").is_err());
        // explicit needs no source_ids.
        assert!(parse_observation(&raw, "explicit").is_ok());
    }

    #[test]
    fn parse_contradiction_needs_two_source_ids() {
        let one = json!({"content": "x", "source_ids": ["a"]});
        assert!(parse_observation(&one, "contradiction").is_err());
        let two = json!({"content": "x", "source_ids": ["a", "b"]});
        assert!(parse_observation(&two, "contradiction").is_ok());
    }

    #[test]
    fn parse_validates_enums_and_strips_nul() {
        let bad = json!({"content": "x", "source_ids": ["a"], "pattern_type": "nope"});
        assert!(parse_observation(&bad, "inductive").is_err());
        let ok = json!({"content": "a\u{0}b", "source_ids": ["a", "b"], "pattern_type": "tendency", "confidence": "high"});
        let parsed = parse_observation(&ok, "inductive").unwrap();
        assert_eq!(parsed.content, "ab");
        assert_eq!(parsed.pattern_type.as_deref(), Some("tendency"));
    }

    #[test]
    fn content_preview_truncates_to_50_chars() {
        let raw = json!({"content": "x".repeat(60)});
        assert_eq!(content_preview(&raw).chars().count(), 50);
    }

    #[test]
    fn internal_metadata_includes_level_specific_fields() {
        let ded = ObservationInput {
            content: "c".into(),
            level: "deductive".into(),
            source_ids: Some(vec!["s1".into()]),
            premises: Some(vec!["p1".into()]),
            sources: None,
            pattern_type: None,
            confidence: None,
        };
        let meta = build_internal_metadata(&ded, &[1, 2], "2026-06-22T00:00:00Z");
        assert_eq!(meta["message_ids"], json!([1, 2]));
        assert_eq!(meta["message_created_at"], "2026-06-22T00:00:00Z");
        assert_eq!(meta["source_ids"], json!(["s1"]));
        assert_eq!(meta["premises"], json!(["p1"]));
        assert!(meta.get("sources").is_none());
        assert!(meta.get("confidence").is_none());

        let ind = ObservationInput {
            content: "c".into(),
            level: "inductive".into(),
            source_ids: Some(vec!["s1".into(), "s2".into()]),
            premises: None,
            sources: Some(vec!["e1".into()]),
            pattern_type: Some("tendency".into()),
            confidence: None,
        };
        let meta = build_internal_metadata(&ind, &[], "2026-06-22T00:00:00Z");
        // confidence defaults to "medium" for inductive.
        assert_eq!(meta["confidence"], "medium");
        assert_eq!(meta["pattern_type"], "tendency");
        assert_eq!(meta["sources"], json!(["e1"]));
        assert!(meta.get("premises").is_none());
    }
}
