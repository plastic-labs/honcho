//! Port of `src/llm/structured_output.py`, specialized to the only structured
//! output honcho actually requests: [`PromptRepresentation`] (the minimal
//! deriver's `response_model`).
//!
//! Because the Rust `PromptRepresentation` models only `explicit` (the minimal
//! deriver shape), the Python `repair_response_model_json` deductive-patching is
//! a no-op here and is omitted; everything else is faithful.

use serde_json::Value;

use crate::json_parser::validate_and_repair_json;
use crate::representation::PromptRepresentation;

/// Port of `StructuredOutputFailurePolicy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailurePolicy {
    Raise,
    RepairThenRaise,
    RepairThenEmpty,
}

/// Port of `validate_structured_output` for `PromptRepresentation`.
///
/// A string payload is parsed as JSON then validated (`model_validate_json`); an
/// object payload is validated directly (`model_validate`); any other payload is
/// an error (`StructuredOutputError` "Unsupported payload").
pub fn validate_structured_output(content: &Value) -> Result<PromptRepresentation, String> {
    match content {
        Value::String(s) => {
            let value: Value =
                serde_json::from_str(s).map_err(|e| format!("invalid JSON string: {e}"))?;
            PromptRepresentation::from_value(&value)
        }
        Value::Object(_) => PromptRepresentation::from_value(content),
        other => Err(format!(
            "Unsupported structured output payload: {}",
            match other {
                Value::Array(_) => "array",
                Value::Number(_) => "number",
                Value::Bool(_) => "bool",
                Value::Null => "null",
                _ => "value",
            }
        )),
    }
}

/// Port of `repair_response_model_json` for `PromptRepresentation`.
///
/// Repairs malformed JSON, validates it, and falls back to an empty
/// representation on any failure (the Python `PromptRepresentation(explicit=[])`
/// branch). Never fails for this model — matching Python, where the
/// PromptRepresentation `ValidationError` is swallowed into an empty value.
pub fn repair_response_model_json(raw_content: &str) -> PromptRepresentation {
    let final_json = validate_and_repair_json(raw_content).unwrap_or_default();
    // The deductive-patch step is a no-op for the explicit-only Rust shape.
    serde_json::from_str::<Value>(&final_json)
        .ok()
        .and_then(|v| PromptRepresentation::from_value(&v).ok())
        .unwrap_or_default()
}

/// Port of `empty_structured_output` for `PromptRepresentation`.
pub fn empty_structured_output() -> PromptRepresentation {
    PromptRepresentation::default()
}

/// Port of `attempt_structured_output_repair`: only string payloads are
/// repairable (and for `PromptRepresentation` repair always yields a value).
fn attempt_structured_output_repair(content: &Value) -> Option<PromptRepresentation> {
    match content {
        Value::String(s) => Some(repair_response_model_json(s)),
        _ => None,
    }
}

/// Port of the post-call logic in `execute_structured_output_call`: validate the
/// content, then (unless the policy is `Raise`) attempt repair, then apply the
/// failure policy.
pub fn finalize_structured_output(
    content: &Value,
    failure_policy: FailurePolicy,
) -> Result<PromptRepresentation, String> {
    if let Ok(validated) = validate_structured_output(content) {
        return Ok(validated);
    }
    if failure_policy == FailurePolicy::Raise {
        return Err("structured output validation failed".to_string());
    }

    if let Some(repaired) = attempt_structured_output_repair(content) {
        return Ok(repaired);
    }

    if failure_policy == FailurePolicy::RepairThenEmpty {
        return Ok(empty_structured_output());
    }

    Err("Failed to produce valid structured output for PromptRepresentation".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validate_accepts_object_and_string() {
        let obj = json!({"explicit": [{"content": "a"}, {"content": "b"}]});
        assert_eq!(
            validate_structured_output(&obj).unwrap().explicit,
            vec!["a".to_string(), "b".to_string()]
        );

        let s = Value::String("{\"explicit\": [{\"content\": \"x\"}]}".to_string());
        assert_eq!(
            validate_structured_output(&s).unwrap().explicit,
            vec!["x".to_string()]
        );

        // Missing/null explicit -> empty list (pydantic default_factory).
        assert!(
            validate_structured_output(&json!({}))
                .unwrap()
                .explicit
                .is_empty()
        );
    }

    #[test]
    fn validate_rejects_non_object_non_string() {
        assert!(validate_structured_output(&json!([1, 2])).is_err());
        assert!(validate_structured_output(&json!(42)).is_err());
        // An object whose explicit items lack string content is invalid.
        assert!(validate_structured_output(&json!({"explicit": ["bare"]})).is_err());
    }

    #[test]
    fn repair_recovers_truncated_json() {
        // Truncated but valid-shaped explicit objects.
        let pr = repair_response_model_json("{\"explicit\": [{\"content\": \"a\"}, {\"content\": \"b\"");
        assert_eq!(pr.explicit, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn repair_falls_back_to_empty_on_garbage() {
        assert!(repair_response_model_json("garbage not json at all").explicit.is_empty());
        // Wrong-shaped explicit (bare strings) -> validation fails -> empty.
        assert!(repair_response_model_json("{\"explicit\": [\"a\", \"b\"]}").explicit.is_empty());
    }

    #[test]
    fn finalize_applies_failure_policy() {
        // Valid passes straight through regardless of policy.
        let valid = json!({"explicit": [{"content": "ok"}]});
        assert_eq!(
            finalize_structured_output(&valid, FailurePolicy::Raise).unwrap().explicit,
            vec!["ok".to_string()]
        );

        // Invalid object: not a string, so no repair path.
        let bad_obj = json!({"explicit": ["bare"]});
        assert!(finalize_structured_output(&bad_obj, FailurePolicy::Raise).is_err());
        assert!(finalize_structured_output(&bad_obj, FailurePolicy::RepairThenRaise).is_err());
        assert!(
            finalize_structured_output(&bad_obj, FailurePolicy::RepairThenEmpty)
                .unwrap()
                .explicit
                .is_empty()
        );

        // Invalid string: repaired (to empty here) for any non-Raise policy.
        let bad_str = Value::String("not json".to_string());
        assert!(
            finalize_structured_output(&bad_str, FailurePolicy::RepairThenRaise)
                .unwrap()
                .explicit
                .is_empty()
        );
        assert!(finalize_structured_output(&bad_str, FailurePolicy::Raise).is_err());
    }
}
