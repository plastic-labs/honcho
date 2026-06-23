//! Port of the read-path surface of `src/utils/representation.py`.
//!
//! A [`Representation`] is the traversable map of observations about a peer, in
//! four reasoning levels: explicit (facts), deductive (logical necessities),
//! inductive (patterns), and contradiction (conflicting statements). The
//! dialectic prefetch path turns the documents returned by `search_memory` into
//! a `Representation` and renders it with [`Representation::format_as_markdown`].
//!
//! Only the read-path methods (`is_empty`, `len`, `format_as_markdown`, plus the
//! explicit observation's `Display`/`__str__`) are ported here; the write-path
//! helpers (`diff`/`merge`, `from_documents`, the other `str_*` variants) land
//! with the data layer that produces them.

use std::fmt;

use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use serde_json::Value;

/// Port of `_strip_microseconds_and_timezone` + `str(datetime)`: renders the
/// timestamp's UTC components as `YYYY-MM-DD HH:MM:SS`, dropping sub-second
/// precision and timezone suffix (matching Python's `datetime.replace(
/// microsecond=0, tzinfo=None)` followed by `str()`).
fn strip_timestamp(timestamp: &DateTime<Utc>) -> String {
    timestamp.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Port of `format_datetime_utc`: ISO 8601 with a `Z` suffix and second-level
/// precision (sub-second always dropped). The input is already UTC, so this is
/// `dt` rendered as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn format_datetime_utc(dt: DateTime<Utc>) -> String {
    dt.format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// One observation ready to persist (`_save_representation_internal`'s per-doc
/// shaping). `premises` is `Some` only for deductive observations (carried into
/// the document metadata); explicit observations leave it `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveObservation {
    pub level: String,
    pub content: String,
    pub premises: Option<Vec<String>>,
}

/// Port of the per-document construction in `_save_representation_internal`:
/// zip each [`SaveObservation`] (from [`Representation::observations_for_save`])
/// with its embedding into a [`crate::db::DocumentToCreate`]. The shared
/// `message_ids` / `session_name` / `message_created_at` are stamped into each
/// document's `internal_metadata` (mirroring `DocumentMetadata` with
/// `exclude_none`: `premises` is present only for deductive observations).
/// `times_derived` starts at 1; `source_ids` is unset on this (deriver) path.
///
/// `observations` and `embeddings` must be equal length and aligned; extra
/// elements of either are ignored (the Python `zip(..., strict=True)` guarantees
/// equal length upstream).
pub fn build_documents_to_create(
    observations: &[SaveObservation],
    embeddings: &[Vec<f32>],
    message_ids: &[i64],
    session_name: &str,
    message_created_at: DateTime<Utc>,
) -> Vec<crate::db::DocumentToCreate> {
    let created_at = format_datetime_utc(message_created_at);
    observations
        .iter()
        .zip(embeddings.iter())
        .map(|(obs, embedding)| {
            let mut metadata = serde_json::Map::new();
            metadata.insert(
                "message_ids".to_string(),
                Value::Array(message_ids.iter().map(|&id| Value::from(id)).collect()),
            );
            metadata.insert(
                "message_created_at".to_string(),
                Value::String(created_at.clone()),
            );
            if let Some(premises) = &obs.premises {
                metadata.insert(
                    "premises".to_string(),
                    Value::Array(premises.iter().map(|p| Value::String(p.clone())).collect()),
                );
            }
            crate::db::DocumentToCreate {
                content: obs.content.clone(),
                session_name: Some(session_name.to_string()),
                level: obs.level.clone(),
                internal_metadata: Value::Object(metadata),
                embedding: embedding.clone(),
                times_derived: 1,
                source_ids: None,
            }
        })
        .collect()
}

/// `"[id:{id}] "` when `include_ids` is set and the observation has an id, else
/// empty — the prefix used by `format_as_markdown` for derived observations.
fn id_prefix(id: &str, include_ids: bool) -> String {
    if include_ids && !id.is_empty() {
        format!("[id:{id}] ")
    } else {
        String::new()
    }
}

/// `"    - {item}"` per item, joined by newlines (the 4-space bullet indent used
/// by the observation `__str__`/`str_with_id` renderers — distinct from the
/// 3-space indent `format_as_markdown` uses).
fn indented_bullets(items: &[String]) -> String {
    items
        .iter()
        .map(|item| format!("    - {item}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The leading-newline bullet block for inductive/contradiction sources: empty
/// when there are no sources, else `"\n"` + the indented bullets (porting the
/// `sources_text` construction shared by those observations).
fn leading_bullets(items: &[String]) -> String {
    if items.is_empty() {
        String::new()
    } else {
        format!("\n{}", indented_bullets(items))
    }
}

/// A present, non-null value at `key` in a JSONB metadata object. Mirrors
/// Python's `dict.get(key)` returning `None` for a missing key — except we also
/// fold an explicit JSON `null` to `None`, since the call sites that matter
/// (`message_created_at`, the str-list lookups) treat null and absent alike.
fn meta_get<'a>(meta: &'a Value, key: &str) -> Option<&'a Value> {
    meta.as_object()
        .and_then(|map| map.get(key))
        .filter(|value| !value.is_null())
}

/// Port of `internal_metadata.get(key, [])` for a list of strings: the string
/// elements of the array at `key`, or empty when the key is absent/null/non-array.
fn meta_str_list(meta: &Value, key: &str) -> Vec<String> {
    meta_get(meta, key)
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(|value| value.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Port of `internal_metadata.get(key, [])` for inductive/contradiction
/// `message_ids`, which (unlike explicit/deductive) are stored without passing
/// through `flatten_message_ids`. Takes the top-level integer elements as-is.
fn meta_int_list(meta: &Value, key: &str) -> Vec<i64> {
    meta_get(meta, key)
        .and_then(Value::as_array)
        .map(|array| array.iter().filter_map(Value::as_i64).collect())
        .unwrap_or_default()
}

/// Port of `internal_metadata.get(key, default)` for a string field.
fn meta_string(meta: &Value, key: &str, default: &str) -> String {
    meta_get(meta, key)
        .and_then(Value::as_str)
        .map(String::from)
        .unwrap_or_else(|| default.to_string())
}

/// Port of `flatten_message_ids`: flattens nested int arrays / `[start, end]`
/// tuple ranges into a deduped, sorted flat list. The metadata value is a JSON
/// array whose elements are either integers or arrays of integers.
fn flatten_message_ids(value: Option<&Value>) -> Vec<i64> {
    let array = match value.and_then(Value::as_array) {
        Some(array) => array,
        None => return Vec::new(),
    };
    let mut out: Vec<i64> = Vec::new();
    for item in array {
        match item {
            Value::Array(inner) => out.extend(inner.iter().filter_map(Value::as_i64)),
            other => {
                if let Some(n) = other.as_i64() {
                    out.push(n);
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Port of `doc.source_ids or internal_metadata.get(key, [])`: the document's
/// own `source_ids` column when non-empty (Python truthiness), else the
/// metadata fallback (for backward compatibility with older rows).
fn source_ids_or(doc_source_ids: &[String], meta: &Value, meta_key: &str) -> Vec<String> {
    if doc_source_ids.is_empty() {
        meta_str_list(meta, meta_key)
    } else {
        doc_source_ids.to_vec()
    }
}

/// Port of `parse_datetime_iso` reduced to the wall-clock components Python
/// keeps after `_strip_microseconds_and_timezone` (which drops the tz label
/// *without converting*). A `Z`/`z` suffix is normalized to `+00:00`; a parsed
/// offset's local wall clock is returned verbatim (so `12:00:00+05:00` yields
/// `12:00:00`, matching Python — and unlike `filters::parse_datetime`, which
/// converts to UTC). Naive inputs are taken as-is. Returns `None` on the inputs
/// Python rejects (empty, control chars) or fails to parse, which the caller
/// folds into the fallback timestamp.
fn parse_iso_wall_clock(raw: &str) -> Option<NaiveDateTime> {
    if raw.is_empty()
        || raw.contains('\0')
        || raw.contains('\r')
        || raw.contains('\n')
        || raw.chars().any(|c| (c as u32) < 32 && c != '\t')
    {
        return None;
    }
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let normalized = if let Some(stripped) = trimmed
        .strip_suffix('Z')
        .or_else(|| trimmed.strip_suffix('z'))
    {
        format!("{stripped}+00:00")
    } else {
        trimmed.to_string()
    };

    if let Ok(parsed) = DateTime::parse_from_rfc3339(&normalized) {
        // naive_local() keeps the wall clock in the parsed offset, matching
        // Python's tzinfo-drop (no UTC conversion).
        return Some(parsed.naive_local());
    }
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Some(naive);
        }
    }
    // Date-only inputs default to midnight (fromisoformat behavior).
    NaiveDate::parse_from_str(trimmed, "%Y-%m-%d")
        .ok()
        .and_then(|date| date.and_hms_opt(0, 0, 0))
}

/// Port of `_safe_datetime_from_metadata`: prefer the `message_created_at`
/// timestamp recorded in the document's metadata, falling back to the document's
/// own `created_at` when it is absent, null, or unparseable. The returned
/// `DateTime<Utc>` carries the wall-clock numbers Python would render (the naive
/// parse result relabeled as UTC), so downstream formatting is byte-identical.
fn safe_datetime_from_metadata(meta: &Value, fallback: DateTime<Utc>) -> DateTime<Utc> {
    match meta_get(meta, "message_created_at") {
        Some(Value::String(text)) => parse_iso_wall_clock(text)
            .map(|naive| Utc.from_utc_datetime(&naive))
            .unwrap_or(fallback),
        _ => fallback,
    }
}

/// Mirror of the `models.Document` columns consumed by
/// [`Representation::from_documents`]. The vector/recency document queries
/// produce these rows; `internal_metadata` is the raw JSONB object.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub level: String,
    pub created_at: DateTime<Utc>,
    pub session_name: Option<String>,
    pub source_ids: Vec<String>,
    pub internal_metadata: Value,
}

/// An explicit observation: a fact literally stated by the peer. Fields flatten
/// `ObservationMetadata` + `ExplicitObservationBase` (Python uses multiple
/// inheritance; Rust inlines the shared metadata into each observation type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplicitObservation {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub message_ids: Vec<i64>,
    pub session_name: Option<String>,
    pub content: String,
}

impl fmt::Display for ExplicitObservation {
    /// Port of `ExplicitObservation.__str__`: `"[{timestamp}] {content}"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", strip_timestamp(&self.created_at), self.content)
    }
}

impl ExplicitObservation {
    /// Port of `ExplicitObservation.str_with_id`: the `__str__` rendering with an
    /// `[id:...]` prefix when this observation has an id.
    pub fn str_with_id(&self) -> String {
        format!(
            "{}[{}] {}",
            id_prefix(&self.id, true),
            strip_timestamp(&self.created_at),
            self.content
        )
    }
}

/// A deductive observation: a conclusion that must hold given its premises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeductiveObservation {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub message_ids: Vec<i64>,
    pub session_name: Option<String>,
    pub source_ids: Vec<String>,
    pub premises: Vec<String>,
    pub conclusion: String,
}

impl fmt::Display for DeductiveObservation {
    /// Port of `DeductiveObservation.__str__`: the conclusion line followed by a
    /// newline and the 4-space-indented premises (an empty premise list still
    /// leaves the trailing newline, matching the Python f-string).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}\n{}",
            strip_timestamp(&self.created_at),
            self.conclusion,
            indented_bullets(&self.premises)
        )
    }
}

impl DeductiveObservation {
    /// Port of `DeductiveObservation.str_with_id`.
    pub fn str_with_id(&self) -> String {
        format!(
            "{}[{}] {}\n{}",
            id_prefix(&self.id, true),
            strip_timestamp(&self.created_at),
            self.conclusion,
            indented_bullets(&self.premises)
        )
    }
}

/// An inductive observation: a pattern/generalization inferred over many
/// observations, with a confidence and pattern type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InductiveObservation {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub message_ids: Vec<i64>,
    pub session_name: Option<String>,
    pub source_ids: Vec<String>,
    pub sources: Vec<String>,
    pub pattern_type: String,
    pub conclusion: String,
    pub confidence: String,
}

impl fmt::Display for InductiveObservation {
    /// Port of `InductiveObservation.__str__`:
    /// `"[{timestamp}] [{confidence}] {conclusion}{sources}"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] [{}] {}{}",
            strip_timestamp(&self.created_at),
            self.confidence,
            self.conclusion,
            leading_bullets(&self.sources)
        )
    }
}

impl InductiveObservation {
    /// Port of `InductiveObservation.str_with_id`.
    pub fn str_with_id(&self) -> String {
        format!(
            "{}[{}] [{}] {}{}",
            id_prefix(&self.id, true),
            strip_timestamp(&self.created_at),
            self.confidence,
            self.conclusion,
            leading_bullets(&self.sources)
        )
    }
}

/// A contradiction observation: conflicting statements the peer has made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContradictionObservation {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub message_ids: Vec<i64>,
    pub session_name: Option<String>,
    pub source_ids: Vec<String>,
    pub sources: Vec<String>,
    pub content: String,
}

impl fmt::Display for ContradictionObservation {
    /// Port of `ContradictionObservation.__str__`:
    /// `"[{timestamp}] CONTRADICTION: {content}{sources}"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] CONTRADICTION: {}{}",
            strip_timestamp(&self.created_at),
            self.content,
            leading_bullets(&self.sources)
        )
    }
}

impl ContradictionObservation {
    /// Port of `ContradictionObservation.str_with_id`.
    pub fn str_with_id(&self) -> String {
        format!(
            "{}[{}] CONTRADICTION: {}{}",
            id_prefix(&self.id, true),
            strip_timestamp(&self.created_at),
            self.content,
            leading_bullets(&self.sources)
        )
    }
}

/// The LLM structured-output shape (`PromptRepresentation`). The minimal
/// deriver only asks for `explicit`; each entry is an `ExplicitObservationBase`
/// (`{"content": "..."}`). Stored here as the bare content strings since that is
/// all [`Representation::from_prompt_representation`] consumes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptRepresentation {
    pub explicit: Vec<String>,
}

impl PromptRepresentation {
    /// Parse the LLM's JSON object, mirroring the pydantic model: a missing or
    /// `null` `explicit` resolves to an empty list (`default_factory` +
    /// `convert_none_to_empty_list`), and each element must be an object with a
    /// string `content` (the required `ExplicitObservationBase` field) — a
    /// missing/non-string `content` or a non-list `explicit` is a validation
    /// error, just as `model_validate` would raise. Extra keys are ignored.
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let explicit = match value.get("explicit") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    let content = item.get("content").and_then(Value::as_str).ok_or_else(
                        || "explicit observation requires a string `content`".to_string(),
                    )?;
                    out.push(content.to_string());
                }
                out
            }
            Some(_) => return Err("`explicit` must be a list".to_string()),
        };
        Ok(Self { explicit })
    }
}

/// A traversable map of observations across the four reasoning levels.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Representation {
    pub explicit: Vec<ExplicitObservation>,
    pub deductive: Vec<DeductiveObservation>,
    pub inductive: Vec<InductiveObservation>,
    pub contradiction: Vec<ContradictionObservation>,
}

impl Representation {
    /// Port of `Representation.from_documents`: partition documents by `level`
    /// into the four observation lists. Per-level metadata extraction mirrors the
    /// Python exactly, including the asymmetries: explicit/deductive flatten
    /// `message_ids` while inductive/contradiction take them as-is, and derived
    /// levels fall back from the `source_ids` column to a metadata key (deductive
    /// reads `premise_ids`; inductive/contradiction read `source_ids`).
    pub fn from_documents(documents: &[Document]) -> Self {
        let mut representation = Self::default();
        for doc in documents {
            let meta = &doc.internal_metadata;
            let created_at = safe_datetime_from_metadata(meta, doc.created_at);
            match doc.level.as_str() {
                "explicit" => representation.explicit.push(ExplicitObservation {
                    id: doc.id.clone(),
                    created_at,
                    message_ids: flatten_message_ids(meta_get(meta, "message_ids")),
                    session_name: doc.session_name.clone(),
                    content: doc.content.clone(),
                }),
                "deductive" => representation.deductive.push(DeductiveObservation {
                    id: doc.id.clone(),
                    created_at,
                    message_ids: flatten_message_ids(meta_get(meta, "message_ids")),
                    session_name: doc.session_name.clone(),
                    source_ids: source_ids_or(&doc.source_ids, meta, "premise_ids"),
                    premises: meta_str_list(meta, "premises"),
                    conclusion: doc.content.clone(),
                }),
                "inductive" => representation.inductive.push(InductiveObservation {
                    id: doc.id.clone(),
                    created_at,
                    message_ids: meta_int_list(meta, "message_ids"),
                    session_name: doc.session_name.clone(),
                    source_ids: source_ids_or(&doc.source_ids, meta, "source_ids"),
                    sources: meta_str_list(meta, "sources"),
                    pattern_type: meta_string(meta, "pattern_type", "pattern"),
                    conclusion: doc.content.clone(),
                    confidence: meta_string(meta, "confidence", "medium"),
                }),
                "contradiction" => representation.contradiction.push(ContradictionObservation {
                    id: doc.id.clone(),
                    created_at,
                    message_ids: meta_int_list(meta, "message_ids"),
                    session_name: doc.session_name.clone(),
                    source_ids: source_ids_or(&doc.source_ids, meta, "source_ids"),
                    sources: meta_str_list(meta, "sources"),
                    content: doc.content.clone(),
                }),
                _ => {}
            }
        }
        representation
    }

    /// Port of `Representation.is_empty`: true when every level is empty.
    pub fn is_empty(&self) -> bool {
        self.explicit.is_empty()
            && self.deductive.is_empty()
            && self.inductive.is_empty()
            && self.contradiction.is_empty()
    }

    /// Port of `Representation.merge_representation` (the `max_observations=None`
    /// path used by `get_working_representation`): fold `other` into `self`,
    /// deduplicate each level, and re-sort by `created_at`. Python dedups via
    /// `set()` over the whole observation; document ids are unique, so deduping
    /// by id is equivalent. The sort is stable, so observations sharing a
    /// timestamp keep insertion (FIFO) order.
    pub fn merge(&mut self, other: Representation) {
        fn merge_level<T>(
            base: &mut Vec<T>,
            other: Vec<T>,
            id_of: impl Fn(&T) -> &str,
            created_at_of: impl Fn(&T) -> DateTime<Utc>,
        ) {
            base.extend(other);
            let mut seen = std::collections::HashSet::new();
            base.retain(|item| seen.insert(id_of(item).to_string()));
            base.sort_by_key(|item| created_at_of(item));
        }
        merge_level(
            &mut self.explicit,
            other.explicit,
            |o| &o.id,
            |o| o.created_at,
        );
        merge_level(
            &mut self.deductive,
            other.deductive,
            |o| &o.id,
            |o| o.created_at,
        );
        merge_level(
            &mut self.inductive,
            other.inductive,
            |o| &o.id,
            |o| o.created_at,
        );
        merge_level(
            &mut self.contradiction,
            other.contradiction,
            |o| &o.id,
            |o| o.created_at,
        );
    }

    /// Port of `Representation.len`: total observation count across all levels.
    /// (Named `count` to avoid colliding with the `len`/`is_empty` clippy lint;
    /// Python's `len()` is a method, not the dunder.)
    #[allow(clippy::len_without_is_empty)]
    pub fn count(&self) -> usize {
        self.explicit.len()
            + self.deductive.len()
            + self.inductive.len()
            + self.contradiction.len()
    }

    /// Port of `Representation.from_prompt_representation`: lift the LLM's
    /// explicit content strings into `ExplicitObservation`s, stamping each with
    /// the shared `created_at`, `message_ids`, and `session_name` and an empty
    /// id (the `ObservationMetadata.id` default). Deductive/inductive/
    /// contradiction levels stay empty — the minimal deriver only emits explicit.
    pub fn from_prompt_representation(
        prompt: &PromptRepresentation,
        message_ids: &[i64],
        session_name: &str,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            explicit: prompt
                .explicit
                .iter()
                .map(|content| ExplicitObservation {
                    id: String::new(),
                    created_at,
                    message_ids: message_ids.to_vec(),
                    session_name: Some(session_name.to_string()),
                    content: content.clone(),
                })
                .collect(),
            deductive: Vec::new(),
            inductive: Vec::new(),
            contradiction: Vec::new(),
        }
    }

    /// Port of the ordering/filtering at the top of `save_representation` +
    /// `_normalized_observation`: the observations to persist, **deductive first
    /// then explicit**, each with its text stripped and any whose text is empty
    /// after stripping dropped. Embeddings are computed over this exact ordered
    /// list, so callers must keep it aligned with [`build_documents_to_create`].
    pub fn observations_for_save(&self) -> Vec<SaveObservation> {
        let mut out: Vec<SaveObservation> = Vec::new();
        for deductive in &self.deductive {
            let content = deductive.conclusion.trim();
            if content.is_empty() {
                continue;
            }
            out.push(SaveObservation {
                level: "deductive".to_string(),
                content: content.to_string(),
                premises: Some(deductive.premises.clone()),
            });
        }
        for explicit in &self.explicit {
            let content = explicit.content.trim();
            if content.is_empty() {
                continue;
            }
            out.push(SaveObservation {
                level: "explicit".to_string(),
                content: content.to_string(),
                premises: None,
            });
        }
        out
    }

    /// Port of `Representation.format_as_markdown`. Each non-empty level becomes
    /// a `##` section; sub-second precision is always stripped from timestamps.
    /// `include_ids` adds `[id:...]` prefixes to derived observations so the
    /// agent can reference them with `get_reasoning_chain` (explicit
    /// observations never get an id prefix, matching the Python comment).
    pub fn format_as_markdown(&self, include_ids: bool) -> String {
        let mut parts: Vec<String> = Vec::new();

        if !self.explicit.is_empty() {
            parts.push("## Explicit Observations\n".to_string());
            for obs in &self.explicit {
                parts.push(obs.to_string());
            }
            parts.push(String::new());
        }

        if !self.deductive.is_empty() {
            parts.push("## Deductive Observations\n".to_string());
            for obs in &self.deductive {
                let prefix = id_prefix(&obs.id, include_ids);
                let timestamp = strip_timestamp(&obs.created_at);
                parts.push(format!("{prefix}[{timestamp}] {}", obs.conclusion));
                if !obs.premises.is_empty() {
                    parts.push("   Premises:".to_string());
                    for premise in &obs.premises {
                        parts.push(format!("   - {premise}"));
                    }
                }
                parts.push(String::new());
            }
            parts.push(String::new());
        }

        if !self.inductive.is_empty() {
            parts.push("## Inductive Observations\n".to_string());
            for obs in &self.inductive {
                let prefix = id_prefix(&obs.id, include_ids);
                parts.push(format!(
                    "{prefix} **Pattern** [{}]: {}",
                    obs.confidence, obs.conclusion
                ));
                if !obs.pattern_type.is_empty() {
                    parts.push(format!("   **Type**: {}", obs.pattern_type));
                }
                if !obs.sources.is_empty() {
                    parts.push("   **Sources**:".to_string());
                    for source in obs.sources.iter().take(5) {
                        parts.push(format!("   - {source}"));
                    }
                    if obs.sources.len() > 5 {
                        parts.push(format!("   - ... and {} more", obs.sources.len() - 5));
                    }
                }
                parts.push(String::new());
            }
            parts.push(String::new());
        }

        if !self.contradiction.is_empty() {
            parts.push("## Contradictions\n".to_string());
            for obs in &self.contradiction {
                let prefix = id_prefix(&obs.id, include_ids);
                parts.push(format!("{prefix} **CONTRADICTION**: {}", obs.content));
                if !obs.sources.is_empty() {
                    parts.push("   **Conflicting statements**:".to_string());
                    for source in &obs.sources {
                        parts.push(format!("   - {source}"));
                    }
                }
                parts.push(String::new());
            }
            parts.push(String::new());
        }

        parts.join("\n")
    }

    /// Port of `Representation.str_with_ids`: the same four `EXPLICIT:`/
    /// `DEDUCTIVE:`/`INDUCTIVE:`/`CONTRADICTION:` numbered sections as the
    /// `Display`/`__str__` rendering, but with `[id:...]` prefixes so agents can
    /// reference observations for `get_reasoning_chain` / deletion.
    pub fn str_with_ids(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        parts.extend(numbered_section("EXPLICIT:\n", &self.explicit, |o| o.str_with_id()));
        parts.extend(numbered_section("DEDUCTIVE:\n", &self.deductive, |o| {
            o.str_with_id()
        }));
        parts.extend(numbered_section("INDUCTIVE:\n", &self.inductive, |o| {
            o.str_with_id()
        }));
        parts.extend(numbered_section("CONTRADICTION:\n", &self.contradiction, |o| {
            o.str_with_id()
        }));
        parts.join("\n")
    }
}

/// One `HEADER:\n` block followed by `"{i}. {rendered}"` per item (1-indexed)
/// and a trailing empty string, mirroring the `parts.append` sequence shared by
/// `Representation.__str__` and `str_with_ids`.
fn numbered_section<T>(header: &str, items: &[T], render: impl Fn(&T) -> String) -> Vec<String> {
    let mut parts = vec![header.to_string()];
    for (index, item) in items.iter().enumerate() {
        parts.push(format!("{}. {}", index + 1, render(item)));
    }
    parts.push(String::new());
    parts
}

impl fmt::Display for Representation {
    /// Port of `Representation.__str__`: all four numbered sections are always
    /// emitted (unlike `format_as_markdown`, which skips empty levels).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        parts.extend(numbered_section("EXPLICIT:\n", &self.explicit, ToString::to_string));
        parts.extend(numbered_section("DEDUCTIVE:\n", &self.deductive, ToString::to_string));
        parts.extend(numbered_section("INDUCTIVE:\n", &self.inductive, ToString::to_string));
        parts.extend(numbered_section(
            "CONTRADICTION:\n",
            &self.contradiction,
            ToString::to_string,
        ));
        write!(f, "{}", parts.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ts(rfc3339: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(rfc3339)
            .unwrap()
            .with_timezone(&Utc)
    }

    fn explicit(content: &str, at: &str) -> ExplicitObservation {
        ExplicitObservation {
            id: String::new(),
            created_at: ts(at),
            message_ids: vec![],
            session_name: None,
            content: content.to_string(),
        }
    }

    #[test]
    fn observation_display_and_str_with_id() {
        let e = ExplicitObservation {
            id: "abc".to_string(),
            ..explicit("fact one", "2025-01-01T12:00:00Z")
        };
        assert_eq!(e.to_string(), "[2025-01-01 12:00:00] fact one");
        assert_eq!(e.str_with_id(), "[id:abc] [2025-01-01 12:00:00] fact one");

        let d = DeductiveObservation {
            id: "ghi".to_string(),
            created_at: ts("2025-01-01T12:01:00Z"),
            message_ids: vec![],
            session_name: None,
            source_ids: vec![],
            premises: vec!["p1".to_string(), "p2".to_string()],
            conclusion: "concl".to_string(),
        };
        assert_eq!(d.to_string(), "[2025-01-01 12:01:00] concl\n    - p1\n    - p2");
        assert_eq!(
            d.str_with_id(),
            "[id:ghi] [2025-01-01 12:01:00] concl\n    - p1\n    - p2"
        );

        // Empty premises still leave the trailing newline (Python f-string).
        let d_empty = DeductiveObservation {
            premises: vec![],
            ..d.clone()
        };
        assert_eq!(d_empty.to_string(), "[2025-01-01 12:01:00] concl\n");

        let i = InductiveObservation {
            id: "jkl".to_string(),
            created_at: ts("2025-01-01T12:05:00Z"),
            message_ids: vec![],
            session_name: None,
            source_ids: vec![],
            sources: vec!["s1".to_string()],
            pattern_type: "behavior".to_string(),
            conclusion: "pattern".to_string(),
            confidence: "high".to_string(),
        };
        assert_eq!(i.to_string(), "[2025-01-01 12:05:00] [high] pattern\n    - s1");
        assert_eq!(
            i.str_with_id(),
            "[id:jkl] [2025-01-01 12:05:00] [high] pattern\n    - s1"
        );
        // No sources -> no trailing block.
        let i_empty = InductiveObservation {
            sources: vec![],
            ..i.clone()
        };
        assert_eq!(i_empty.to_string(), "[2025-01-01 12:05:00] [high] pattern");

        let c = ContradictionObservation {
            id: "xyz".to_string(),
            created_at: ts("2025-01-01T12:06:00Z"),
            message_ids: vec![],
            session_name: None,
            source_ids: vec![],
            sources: vec!["c1".to_string(), "c2".to_string()],
            content: "conflict".to_string(),
        };
        assert_eq!(
            c.to_string(),
            "[2025-01-01 12:06:00] CONTRADICTION: conflict\n    - c1\n    - c2"
        );
        assert_eq!(
            c.str_with_id(),
            "[id:xyz] [2025-01-01 12:06:00] CONTRADICTION: conflict\n    - c1\n    - c2"
        );
    }

    #[test]
    fn representation_display_empty_emits_all_headers() {
        assert_eq!(
            Representation::default().to_string(),
            "EXPLICIT:\n\n\nDEDUCTIVE:\n\n\nINDUCTIVE:\n\n\nCONTRADICTION:\n\n"
        );
    }

    #[test]
    fn representation_display_and_str_with_ids() {
        let rep = Representation {
            explicit: vec![ExplicitObservation {
                id: "abc".to_string(),
                ..explicit("fact one", "2025-01-01T12:00:00Z")
            }],
            deductive: vec![DeductiveObservation {
                id: "ghi".to_string(),
                created_at: ts("2025-01-01T12:01:00Z"),
                message_ids: vec![],
                session_name: None,
                source_ids: vec![],
                premises: vec!["p1".to_string(), "p2".to_string()],
                conclusion: "concl".to_string(),
            }],
            inductive: vec![InductiveObservation {
                id: "jkl".to_string(),
                created_at: ts("2025-01-01T12:05:00Z"),
                message_ids: vec![],
                session_name: None,
                source_ids: vec![],
                sources: vec!["s1".to_string()],
                pattern_type: "behavior".to_string(),
                conclusion: "pattern".to_string(),
                confidence: "high".to_string(),
            }],
            contradiction: vec![ContradictionObservation {
                id: "xyz".to_string(),
                created_at: ts("2025-01-01T12:06:00Z"),
                message_ids: vec![],
                session_name: None,
                source_ids: vec![],
                sources: vec!["c1".to_string(), "c2".to_string()],
                content: "conflict".to_string(),
            }],
        };

        assert_eq!(
            rep.to_string(),
            "EXPLICIT:\n\n1. [2025-01-01 12:00:00] fact one\n\n\
             DEDUCTIVE:\n\n1. [2025-01-01 12:01:00] concl\n    - p1\n    - p2\n\n\
             INDUCTIVE:\n\n1. [2025-01-01 12:05:00] [high] pattern\n    - s1\n\n\
             CONTRADICTION:\n\n1. [2025-01-01 12:06:00] CONTRADICTION: conflict\n    - c1\n    - c2\n"
        );

        // str_with_ids matches __str__ but with [id:...] prefixes on every item.
        assert_eq!(
            rep.str_with_ids(),
            "EXPLICIT:\n\n1. [id:abc] [2025-01-01 12:00:00] fact one\n\n\
             DEDUCTIVE:\n\n1. [id:ghi] [2025-01-01 12:01:00] concl\n    - p1\n    - p2\n\n\
             INDUCTIVE:\n\n1. [id:jkl] [2025-01-01 12:05:00] [high] pattern\n    - s1\n\n\
             CONTRADICTION:\n\n1. [id:xyz] [2025-01-01 12:06:00] CONTRADICTION: conflict\n    - c1\n    - c2\n"
        );
    }

    #[test]
    fn empty_and_count() {
        let rep = Representation::default();
        assert!(rep.is_empty());
        assert_eq!(rep.count(), 0);

        let rep = Representation {
            explicit: vec![explicit("a", "2025-01-01T12:00:00Z")],
            deductive: vec![],
            inductive: vec![],
            contradiction: vec![],
        };
        assert!(!rep.is_empty());
        assert_eq!(rep.count(), 1);
    }

    #[test]
    fn explicit_display_strips_subseconds() {
        let obs = explicit("The user has a dog named Rover", "2025-01-01T12:00:00.123456Z");
        assert_eq!(obs.to_string(), "[2025-01-01 12:00:00] The user has a dog named Rover");
    }

    #[test]
    fn format_as_markdown_all_levels_without_ids() {
        let rep = Representation {
            explicit: vec![explicit("The user has a dog named Rover", "2025-01-01T12:00:00Z")],
            deductive: vec![DeductiveObservation {
                id: "ghi".to_string(),
                created_at: ts("2025-01-01T12:01:00Z"),
                message_ids: vec![],
                session_name: None,
                source_ids: vec![],
                premises: vec![
                    "The user has a dog named Rover".to_string(),
                    "The user's dog is 5 years old".to_string(),
                ],
                conclusion: "Rover is 5 years old".to_string(),
            }],
            inductive: vec![InductiveObservation {
                id: "jkl".to_string(),
                created_at: ts("2025-01-01T12:05:00Z"),
                message_ids: vec![],
                session_name: None,
                source_ids: vec![],
                sources: vec!["s1".to_string(), "s2".to_string()],
                pattern_type: "behavior".to_string(),
                conclusion: "User tends to be methodical".to_string(),
                confidence: "high".to_string(),
            }],
            contradiction: vec![ContradictionObservation {
                id: "xyz".to_string(),
                created_at: ts("2025-01-01T12:06:00Z"),
                message_ids: vec![],
                session_name: None,
                source_ids: vec![],
                sources: vec!["said 25".to_string(), "said 30".to_string()],
                content: "conflicting ages".to_string(),
            }],
        };

        let expected = "## Explicit Observations\n\n\
            [2025-01-01 12:00:00] The user has a dog named Rover\n\
            \n\
            ## Deductive Observations\n\n\
            [2025-01-01 12:01:00] Rover is 5 years old\n\
            \u{20}  Premises:\n\
            \u{20}  - The user has a dog named Rover\n\
            \u{20}  - The user's dog is 5 years old\n\
            \n\
            \n\
            ## Inductive Observations\n\n\
            \u{20}**Pattern** [high]: User tends to be methodical\n\
            \u{20}  **Type**: behavior\n\
            \u{20}  **Sources**:\n\
            \u{20}  - s1\n\
            \u{20}  - s2\n\
            \n\
            \n\
            ## Contradictions\n\n\
            \u{20}**CONTRADICTION**: conflicting ages\n\
            \u{20}  **Conflicting statements**:\n\
            \u{20}  - said 25\n\
            \u{20}  - said 30\n\
            \n";

        assert_eq!(rep.format_as_markdown(false), expected);
    }

    #[test]
    fn format_as_markdown_with_ids_prefixes_derived_only() {
        let rep = Representation {
            explicit: vec![ExplicitObservation {
                id: "abc".to_string(),
                ..explicit("a fact", "2025-01-01T12:00:00Z")
            }],
            deductive: vec![DeductiveObservation {
                id: "ghi".to_string(),
                created_at: ts("2025-01-01T12:01:00Z"),
                message_ids: vec![],
                session_name: None,
                source_ids: vec![],
                premises: vec![],
                conclusion: "a conclusion".to_string(),
            }],
            inductive: vec![],
            contradiction: vec![],
        };

        let md = rep.format_as_markdown(true);
        // Explicit observations never carry an id prefix.
        assert!(md.contains("[2025-01-01 12:00:00] a fact"));
        assert!(!md.contains("[id:abc]"));
        // Derived observations do.
        assert!(md.contains("[id:ghi] [2025-01-01 12:01:00] a conclusion"));
    }

    fn doc(level: &str, content: &str, meta: Value) -> Document {
        Document {
            id: format!("{level}-id"),
            content: content.to_string(),
            level: level.to_string(),
            created_at: ts("2024-12-31T00:00:00Z"),
            session_name: Some("s1".to_string()),
            source_ids: vec![],
            internal_metadata: meta,
        }
    }

    #[test]
    fn flatten_message_ids_handles_flat_nested_and_dupes() {
        assert_eq!(flatten_message_ids(Some(&json!([3, 1, 2]))), vec![1, 2, 3]);
        assert_eq!(flatten_message_ids(Some(&json!([[1, 2], [3, 4]]))), vec![1, 2, 3, 4]);
        assert_eq!(flatten_message_ids(Some(&json!([[105, 105]]))), vec![105]);
        assert_eq!(flatten_message_ids(None), Vec::<i64>::new());
    }

    #[test]
    fn safe_datetime_prefers_metadata_then_falls_back() {
        use serde_json::json;
        let fallback = ts("2024-12-31T00:00:00Z");

        // Absent / null / unparseable -> fallback.
        assert_eq!(safe_datetime_from_metadata(&json!({}), fallback), fallback);
        assert_eq!(
            safe_datetime_from_metadata(&json!({"message_created_at": null}), fallback),
            fallback
        );
        assert_eq!(
            safe_datetime_from_metadata(&json!({"message_created_at": "not-a-date"}), fallback),
            fallback
        );

        // Offset is kept as wall-clock (NOT converted to UTC), matching Python's
        // tzinfo-drop: 12:00:00+05:00 renders as 12:00:00.
        let meta = json!({"message_created_at": "2025-03-04T12:00:00+05:00"});
        let parsed = safe_datetime_from_metadata(&meta, fallback);
        assert_eq!(parsed.format("%Y-%m-%d %H:%M:%S").to_string(), "2025-03-04 12:00:00");
    }

    #[test]
    fn from_documents_partitions_by_level_with_metadata() {
        use serde_json::json;

        let documents = vec![
            doc(
                "explicit",
                "The user has a dog",
                json!({"message_ids": [[2, 2], 1], "message_created_at": "2025-01-01T09:00:00Z"}),
            ),
            doc(
                "deductive",
                "Rover is old",
                json!({"message_ids": [5], "premises": ["p1", "p2"], "premise_ids": ["d1"]}),
            ),
            doc(
                "inductive",
                "User is methodical",
                json!({"sources": ["s1"], "pattern_type": "behavior", "confidence": "high"}),
            ),
            doc(
                "contradiction",
                "conflicting ages",
                json!({"sources": ["said 25", "said 30"]}),
            ),
            doc("unknown_level", "ignored", json!({})),
        ];

        let rep = Representation::from_documents(&documents);

        assert_eq!(rep.explicit.len(), 1);
        assert_eq!(rep.deductive.len(), 1);
        assert_eq!(rep.inductive.len(), 1);
        assert_eq!(rep.contradiction.len(), 1);

        // Explicit: message_ids flattened + sorted; created_at from metadata.
        let e = &rep.explicit[0];
        assert_eq!(e.message_ids, vec![1, 2]);
        assert_eq!(e.created_at.format("%Y-%m-%d %H:%M:%S").to_string(), "2025-01-01 09:00:00");

        // Deductive: source_ids fall back to metadata premise_ids; created_at
        // falls back to the document column (no message_created_at).
        let d = &rep.deductive[0];
        assert_eq!(d.source_ids, vec!["d1".to_string()]);
        assert_eq!(d.premises, vec!["p1".to_string(), "p2".to_string()]);
        assert_eq!(d.created_at.format("%Y-%m-%d %H:%M:%S").to_string(), "2024-12-31 00:00:00");

        // Inductive: defaults applied when present, source_ids from metadata.
        let i = &rep.inductive[0];
        assert_eq!(i.pattern_type, "behavior");
        assert_eq!(i.confidence, "high");
        assert_eq!(i.sources, vec!["s1".to_string()]);

        // Contradiction: defaults (pattern/medium n/a), sources captured.
        let c = &rep.contradiction[0];
        assert_eq!(c.sources, vec!["said 25".to_string(), "said 30".to_string()]);
    }

    #[test]
    fn from_documents_prefers_column_source_ids_over_metadata() {
        use serde_json::json;
        let mut document = doc("deductive", "concl", json!({"premise_ids": ["meta-id"]}));
        document.source_ids = vec!["col-id".to_string()];
        let rep = Representation::from_documents(&[document]);
        assert_eq!(rep.deductive[0].source_ids, vec!["col-id".to_string()]);
    }

    #[test]
    fn from_documents_applies_inductive_defaults() {
        use serde_json::json;
        let rep = Representation::from_documents(&[doc("inductive", "pat", json!({}))]);
        assert_eq!(rep.inductive[0].pattern_type, "pattern");
        assert_eq!(rep.inductive[0].confidence, "medium");
        assert!(rep.inductive[0].sources.is_empty());
    }

    #[test]
    fn format_as_markdown_truncates_inductive_sources_past_five() {
        let rep = Representation {
            explicit: vec![],
            deductive: vec![],
            inductive: vec![InductiveObservation {
                id: String::new(),
                created_at: ts("2025-01-01T12:00:00Z"),
                message_ids: vec![],
                session_name: None,
                source_ids: vec![],
                sources: (1..=7).map(|i| format!("source {i}")).collect(),
                pattern_type: "pattern".to_string(),
                conclusion: "a pattern".to_string(),
                confidence: "medium".to_string(),
            }],
            contradiction: vec![],
        };

        let md = rep.format_as_markdown(false);
        assert!(md.contains("   - source 5"));
        assert!(!md.contains("   - source 6"));
        assert!(md.contains("   - ... and 2 more"));
    }

    // --- PromptRepresentation / from_prompt_representation ---

    #[test]
    fn prompt_representation_parses_explicit_objects() {
        let value = json!({
            "explicit": [{"content": "a is 25"}, {"content": "a has a dog"}],
        });
        let parsed = PromptRepresentation::from_value(&value).unwrap();
        assert_eq!(parsed.explicit, vec!["a is 25", "a has a dog"]);
    }

    #[test]
    fn prompt_representation_null_or_missing_explicit_is_empty() {
        assert_eq!(
            PromptRepresentation::from_value(&json!({"explicit": Value::Null})).unwrap(),
            PromptRepresentation::default()
        );
        assert_eq!(
            PromptRepresentation::from_value(&json!({})).unwrap(),
            PromptRepresentation::default()
        );
    }

    #[test]
    fn prompt_representation_ignores_extra_keys() {
        let value = json!({"explicit": [{"content": "x", "extra": 1}], "junk": true});
        assert_eq!(
            PromptRepresentation::from_value(&value).unwrap().explicit,
            vec!["x"]
        );
    }

    #[test]
    fn prompt_representation_errors_on_bad_shape() {
        assert!(PromptRepresentation::from_value(&json!({"explicit": "nope"})).is_err());
        assert!(PromptRepresentation::from_value(&json!({"explicit": [{"no_content": 1}]})).is_err());
        assert!(PromptRepresentation::from_value(&json!({"explicit": [{"content": 5}]})).is_err());
    }

    #[test]
    fn from_prompt_representation_stamps_metadata() {
        let prompt = PromptRepresentation {
            explicit: vec!["fact one".to_string(), "fact two".to_string()],
        };
        let created = ts("2025-03-04T12:00:00Z");
        let rep = Representation::from_prompt_representation(&prompt, &[10, 20], "sess1", created);

        assert_eq!(rep.explicit.len(), 2);
        assert!(rep.deductive.is_empty());
        assert!(rep.inductive.is_empty());
        assert!(rep.contradiction.is_empty());

        let first = &rep.explicit[0];
        assert_eq!(first.content, "fact one");
        assert_eq!(first.id, "");
        assert_eq!(first.created_at, created);
        assert_eq!(first.message_ids, vec![10, 20]);
        assert_eq!(first.session_name.as_deref(), Some("sess1"));
    }

    // --- observations_for_save / build_documents_to_create ---

    fn deductive(conclusion: &str, premises: &[&str]) -> DeductiveObservation {
        DeductiveObservation {
            id: String::new(),
            created_at: ts("2025-01-01T00:00:00Z"),
            message_ids: vec![],
            session_name: None,
            source_ids: vec![],
            premises: premises.iter().map(|p| p.to_string()).collect(),
            conclusion: conclusion.to_string(),
        }
    }

    #[test]
    fn observations_for_save_orders_deductive_first_and_strips() {
        let rep = Representation {
            explicit: vec![
                explicit("  exp one  ", "2025-01-01T00:00:00Z"),
                explicit("   ", "2025-01-01T00:00:00Z"), // blank -> dropped
            ],
            deductive: vec![deductive("ded one", &["p1", "p2"])],
            ..Representation::default()
        };
        let saved = rep.observations_for_save();
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[0].level, "deductive");
        assert_eq!(saved[0].content, "ded one");
        assert_eq!(saved[0].premises, Some(vec!["p1".to_string(), "p2".to_string()]));
        assert_eq!(saved[1].level, "explicit");
        assert_eq!(saved[1].content, "exp one"); // stripped
        assert_eq!(saved[1].premises, None);
    }

    #[test]
    fn format_datetime_utc_second_precision_with_z() {
        let dt = DateTime::parse_from_rfc3339("2023-01-01T12:00:00.500Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(format_datetime_utc(dt), "2023-01-01T12:00:00Z");
    }

    #[test]
    fn build_documents_to_create_stamps_metadata_and_levels() {
        let observations = vec![
            SaveObservation {
                level: "deductive".to_string(),
                content: "ded".to_string(),
                premises: Some(vec!["p1".to_string()]),
            },
            SaveObservation {
                level: "explicit".to_string(),
                content: "exp".to_string(),
                premises: None,
            },
        ];
        let embeddings = vec![vec![0.1_f32, 0.2], vec![0.3_f32, 0.4]];
        let docs = build_documents_to_create(
            &observations,
            &embeddings,
            &[10, 20],
            "sess1",
            ts("2025-03-04T12:00:00Z"),
        );
        assert_eq!(docs.len(), 2);

        let ded = &docs[0];
        assert_eq!(ded.level, "deductive");
        assert_eq!(ded.content, "ded");
        assert_eq!(ded.session_name.as_deref(), Some("sess1"));
        assert_eq!(ded.times_derived, 1);
        assert_eq!(ded.source_ids, None);
        assert_eq!(ded.embedding, vec![0.1_f32, 0.2]);
        assert_eq!(
            ded.internal_metadata,
            json!({
                "message_ids": [10, 20],
                "message_created_at": "2025-03-04T12:00:00Z",
                "premises": ["p1"],
            })
        );

        // Explicit: premises omitted (exclude_none).
        let exp = &docs[1];
        assert_eq!(
            exp.internal_metadata,
            json!({
                "message_ids": [10, 20],
                "message_created_at": "2025-03-04T12:00:00Z",
            })
        );
    }

    #[test]
    fn from_prompt_representation_empty_is_empty() {
        let rep = Representation::from_prompt_representation(
            &PromptRepresentation::default(),
            &[1],
            "s",
            ts("2025-01-01T00:00:00Z"),
        );
        assert!(rep.is_empty());
    }
}
