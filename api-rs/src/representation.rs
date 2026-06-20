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

/// `"[id:{id}] "` when `include_ids` is set and the observation has an id, else
/// empty — the prefix used by `format_as_markdown` for derived observations.
fn id_prefix(id: &str, include_ids: bool) -> String {
    if include_ids && !id.is_empty() {
        format!("[id:{id}] ")
    } else {
        String::new()
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        use serde_json::json;
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
}
