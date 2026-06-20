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

use chrono::{DateTime, Utc};

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
