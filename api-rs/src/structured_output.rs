//! Port of `src/llm/structured_output.py`, specialized to the only structured
//! output honcho actually requests: [`PromptRepresentation`] (the minimal
//! deriver's `response_model`).
//!
//! Because the Rust `PromptRepresentation` models only `explicit` (the minimal
//! deriver shape), the Python `repair_response_model_json` deductive-patching is
//! a no-op here and is omitted; everything else is faithful.

use serde_json::{Value, json};

use crate::json_parser::validate_and_repair_json;
use crate::representation::PromptRepresentation;

/// The OpenAI `response_format` payload that pins the deriver's structured output
/// to the [`PromptRepresentation`] shape (`{"explicit": [{"content": str}]}`).
///
/// Ported from the Python deriver passing `response_model=PromptRepresentation`,
/// which the OpenAI SDK lowers to a strict `json_schema`. Emitting this is what
/// forces the model to return JSON instead of prose — bare `json_mode`
/// (`{"type": "json_object"}`) is insufficient because the deriver prompt never
/// names "JSON" (OpenAI rejects `json_object` without it) and conveys no shape.
/// The field descriptions mirror Python's `PromptRepresentation` /
/// `ExplicitObservationBase` so the model sees the same guidance.
pub fn prompt_representation_response_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "PromptRepresentation",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "explicit": {
                        "type": "array",
                        "description": "Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference. Example: ['The user is 25 years old', 'The user has a dog named Rover']",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The explicit observation"
                                }
                            },
                            "required": ["content"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["explicit"],
                "additionalProperties": false
            }
        }
    })
}

/// The non-strict schema shape Pydantic emits for `PromptRepresentation`.
///
/// Used only for `structured_output_mode=json_object`, where the schema is
/// injected into the prompt instead of sent as OpenAI Structured Outputs.
pub fn prompt_representation_json_object_response_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "PromptRepresentation",
            "schema": {
                "$defs": {
                    "ExplicitObservationBase": {
                        "properties": {
                            "content": {
                                "description": "The explicit observation",
                                "title": "Content",
                                "type": "string"
                            }
                        },
                        "required": ["content"],
                        "title": "ExplicitObservationBase",
                        "type": "object"
                    }
                },
                "description": "The representation format that is used when getting structured output from an LLM.",
                "properties": {
                    "explicit": {
                        "description": "Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference. Example: ['The user is 25 years old', 'The user has a dog named Rover']",
                        "items": {
                            "$ref": "#/$defs/ExplicitObservationBase"
                        },
                        "title": "Explicit",
                        "type": "array"
                    }
                },
                "title": "PromptRepresentation",
                "type": "object"
            }
        }
    })
}

pub fn json_object_instruction_for_response_format(response_format: &Value) -> String {
    let schema = response_format
        .get("json_schema")
        .and_then(|json_schema| json_schema.get("schema"))
        .unwrap_or(response_format);
    format!(
        "You must respond with a single JSON object that conforms exactly to \
the following JSON schema. Do not include any text, markdown, or code \
fences outside the JSON object.\n\nJSON schema:\n{}",
        json_dumps_python_default(schema)
    )
}

fn json_dumps_python_default(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
        }
        Value::Array(items) => {
            let items = items
                .iter()
                .map(json_dumps_python_default)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{items}]")
        }
        Value::Object(object) => {
            let items = object
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string()),
                        json_dumps_python_default(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{items}}}")
        }
    }
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

/// Validate the structured content, then repair strings, then fall back to empty.
pub fn finalize_structured_output(content: &Value) -> PromptRepresentation {
    if let Ok(validated) = validate_structured_output(content) {
        return validated;
    }
    match content {
        Value::String(s) => repair_response_model_json(s),
        _ => empty_structured_output(),
    }
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
        let pr =
            repair_response_model_json("{\"explicit\": [{\"content\": \"a\"}, {\"content\": \"b\"");
        assert_eq!(pr.explicit, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn repair_falls_back_to_empty_on_garbage() {
        assert!(
            repair_response_model_json("garbage not json at all")
                .explicit
                .is_empty()
        );
        // Wrong-shaped explicit (bare strings) -> validation fails -> empty.
        assert!(
            repair_response_model_json("{\"explicit\": [\"a\", \"b\"]}")
                .explicit
                .is_empty()
        );
    }

    #[test]
    fn json_object_instruction_matches_python_golden() {
        assert_eq!(
            json_object_instruction_for_response_format(
                &prompt_representation_json_object_response_format()
            ),
            "You must respond with a single JSON object that conforms exactly to the following JSON schema. Do not include any text, markdown, or code fences outside the JSON object.\n\nJSON schema:\n{\"$defs\": {\"ExplicitObservationBase\": {\"properties\": {\"content\": {\"description\": \"The explicit observation\", \"title\": \"Content\", \"type\": \"string\"}}, \"required\": [\"content\"], \"title\": \"ExplicitObservationBase\", \"type\": \"object\"}}, \"description\": \"The representation format that is used when getting structured output from an LLM.\", \"properties\": {\"explicit\": {\"description\": \"Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference. Example: ['The user is 25 years old', 'The user has a dog named Rover']\", \"items\": {\"$ref\": \"#/$defs/ExplicitObservationBase\"}, \"title\": \"Explicit\", \"type\": \"array\"}}, \"title\": \"PromptRepresentation\", \"type\": \"object\"}"
        );
    }

    #[test]
    fn finalize_validates_repairs_then_empty() {
        let valid = json!({"explicit": [{"content": "ok"}]});
        assert_eq!(
            finalize_structured_output(&valid).explicit,
            vec!["ok".to_string()]
        );

        let bad_obj = json!({"explicit": ["bare"]});
        assert!(finalize_structured_output(&bad_obj).explicit.is_empty());

        let bad_str = Value::String("not json".to_string());
        assert!(finalize_structured_output(&bad_str).explicit.is_empty());
    }
}
