//! Dreamer agent-tool schemas, peer-card validation, and the per-specialist tool
//! lists — port of the relevant pieces of `src/utils/agent_tools.py` (the `TOOLS`
//! dict entries the specialists use, `DEDUCTION_SPECIALIST_TOOLS` /
//! `INDUCTION_SPECIALIST_TOOLS`, the observation item schemas, and
//! `_validate_peer_card_entry` + its constants). Pure/deterministic; the handlers
//! and `create_tool_executor` are ported separately.

use serde_json::{Value, json};

/// Hard cap on peer-card entries (`MAX_PEER_CARD_FACTS`).
pub const MAX_PEER_CARD_FACTS: usize = 40;

/// Per-entry character cap (`MAX_PEER_CARD_ENTRY_LENGTH`).
pub const MAX_PEER_CARD_ENTRY_LENGTH: usize = 200;

/// Identity-marker prefixes allowed on the peer card (`PEER_CARD_ALLOWED_PREFIXES`).
pub const PEER_CARD_ALLOWED_PREFIXES: [&str; 4] =
    ["IDENTITY:", "ATTRIBUTE:", "RELATIONSHIP:", "INSTRUCTION:"];

/// Port of `_validate_peer_card_entry` (agent_tools.py:49): structural (form-only)
/// validation — non-empty, within the length cap, starts with an allowed prefix
/// followed by a space, and has a non-empty body after the prefix.
///
/// The length check uses Python `len(str)` semantics (Unicode code points), so we
/// count `chars()`, not bytes.
pub fn validate_peer_card_entry(line: &str) -> bool {
    if line.is_empty() || line.chars().count() > MAX_PEER_CARD_ENTRY_LENGTH {
        return false;
    }
    for prefix in PEER_CARD_ALLOWED_PREFIXES {
        let prefix_with_space = format!("{prefix} ");
        if let Some(body) = line.strip_prefix(&prefix_with_space) {
            return !body.trim().is_empty();
        }
    }
    false
}

/// Port of `_deductive_observation_item_schema` (agent_tools.py:205).
fn deductive_observation_item_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The deductive conclusion as a self-contained statement"
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Required non-empty list of source observation IDs supporting the deduction"
            },
            "premises": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Required human-readable premise text matching the source observations"
            }
        },
        "required": ["content", "source_ids", "premises"],
        "additionalProperties": false
    })
}

/// Port of `_inductive_observation_item_schema` (agent_tools.py:231).
fn inductive_observation_item_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The inductive pattern or generalization as a self-contained statement"
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Required list of at least two source observation IDs supporting the pattern"
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Required human-readable evidence text matching the source observations"
            },
            "pattern_type": {
                "type": "string",
                "enum": ["preference", "behavior", "personality", "tendency", "correlation"],
                "description": "Required pattern category"
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Required confidence level based on evidence count"
            }
        },
        "required": ["content", "source_ids", "sources", "pattern_type", "confidence"],
        "additionalProperties": false
    })
}

/// `TOOLS["search_memory"]`.
pub fn search_memory_tool() -> Value {
    json!({
        "name": "search_memory",
        "description": "Search for observations in memory using semantic similarity. Use this to find relevant information about the peer when you need to recall specific details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {
                    "type": "integer",
                    "description": "(Optional) number of results to return (default: 20, max: 40)",
                    "default": 20
                }
            },
            "required": ["query"]
        }
    })
}

/// `TOOLS["search_messages"]`.
pub fn search_messages_tool() -> Value {
    json!({
        "name": "search_messages",
        "description": "Search for messages using semantic similarity and retrieve conversation snippets. Returns matching messages with surrounding context (2 messages before and after). Nearby matches within the same session are merged into a single snippet to avoid repetition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text to find relevant messages"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matching messages to return (default: 10, max: 20)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    })
}

/// `TOOLS["get_recent_observations"]`.
pub fn get_recent_observations_tool() -> Value {
    json!({
        "name": "get_recent_observations",
        "description": "Get the most recent observations about the peer. Useful for understanding what's been learned recently.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of observations to return (default: 10)",
                    "default": 10
                },
                "session_only": {
                    "type": "boolean",
                    "description": "If true, only return observations from the current session (default: false)",
                    "default": false
                }
            }
        }
    })
}

/// `TOOLS["create_observations_deductive"]`.
pub fn create_observations_deductive_tool() -> Value {
    json!({
        "name": "create_observations_deductive",
        "description": "Create new deductive observations discovered while answering the query. Every observation must include non-empty source_ids and premise text. Use this only for novel deductions grounded in existing observations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "List of new deductive observations to create",
                    "items": deductive_observation_item_schema()
                }
            },
            "required": ["observations"]
        }
    })
}

/// `TOOLS["create_observations_inductive"]`.
pub fn create_observations_inductive_tool() -> Value {
    json!({
        "name": "create_observations_inductive",
        "description": "Create new inductive observations discovered while answering the query. Every observation must include source_ids, source text, pattern_type, and confidence. Use this only for patterns supported by multiple observations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "List of new inductive observations to create",
                    "items": inductive_observation_item_schema()
                }
            },
            "required": ["observations"]
        }
    })
}

/// `TOOLS["delete_observations"]`.
pub fn delete_observations_tool() -> Value {
    json!({
        "name": "delete_observations",
        "description": "Delete observations by their IDs. Use the exact ID shown in [id:xxx] format from search results. Example: if observation shows '[id:abc123XYZ]', pass 'abc123XYZ' to delete it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of observation IDs to delete (use the exact ID from [id:xxx] in search results)"
                }
            },
            "required": ["observation_ids"]
        }
    })
}

/// `TOOLS["update_peer_card"]`.
pub fn update_peer_card_tool() -> Value {
    json!({
        "name": "update_peer_card",
        "description": "Update the peer card with stable identity markers about the observed peer. An identity marker distinguishes the peer from others of its kind and persists across interactions. The peer may be any entity with identity that changes over time (human, agent, codebase, team, organization) — do not assume the peer is human. Each entry must start with one of four prefixes: `IDENTITY:` (canonical name, kind, aliases, IDs), `ATTRIBUTE:` (stable durable property, including explicitly stated standing preferences), `RELATIONSHIP:` (durable link to another entity), or `INSTRUCTION:` (standing rule of engagement the peer has explicitly stated). Do not write `TRAIT:` or behavioral `PREFERENCE:` entries, one-off observations, transient state, inferred facts not directly supported by evidence, evidence bundles / `e.g.` clauses, or entries about co-occurring peers. Entries without an allowed prefix or that exceed the per-entry length cap are rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "array",
                    "description": "Complete deduplicated peer card list (max 40 entries). Each entry must start with one of the allowed prefixes (`IDENTITY: `, `ATTRIBUTE: `, `RELATIONSHIP: `, `INSTRUCTION: `) followed by one concise identity marker. Entries without an allowed prefix are rejected.",
                    "items": {"type": "string"}
                }
            },
            "required": ["content"]
        }
    })
}

/// Port of `DEDUCTION_SPECIALIST_TOOLS` (agent_tools.py:824).
pub fn deduction_specialist_tools() -> Vec<Value> {
    vec![
        get_recent_observations_tool(),
        search_memory_tool(),
        search_messages_tool(),
        create_observations_deductive_tool(),
        delete_observations_tool(),
        update_peer_card_tool(),
    ]
}

/// Port of `INDUCTION_SPECIALIST_TOOLS` (agent_tools.py:839).
pub fn induction_specialist_tools() -> Vec<Value> {
    vec![
        get_recent_observations_tool(),
        search_memory_tool(),
        search_messages_tool(),
        create_observations_inductive_tool(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_card_validation() {
        // Valid: allowed prefix + space + non-empty body.
        assert!(validate_peer_card_entry("IDENTITY: Name: Alice"));
        assert!(validate_peer_card_entry("ATTRIBUTE: Location: NYC"));
        assert!(validate_peer_card_entry("RELATIONSHIP: Spouse: Bob"));
        assert!(validate_peer_card_entry("INSTRUCTION: Call me Vee"));
        // No allowed prefix.
        assert!(!validate_peer_card_entry("Name: Alice"));
        assert!(!validate_peer_card_entry("TRAIT: Analytical"));
        // Empty / whitespace-only body after prefix.
        assert!(!validate_peer_card_entry("IDENTITY: "));
        assert!(!validate_peer_card_entry("IDENTITY:    "));
        // Empty line.
        assert!(!validate_peer_card_entry(""));
        // Prefix without the required trailing space.
        assert!(!validate_peer_card_entry("IDENTITY:Name"));
        // Over the length cap (200 code points).
        let long = format!("IDENTITY: {}", "x".repeat(200));
        assert!(!validate_peer_card_entry(&long));
        // Exactly at the cap (200 code points total) is allowed.
        let at_cap = format!("IDENTITY: {}", "x".repeat(MAX_PEER_CARD_ENTRY_LENGTH - 10));
        assert_eq!(at_cap.chars().count(), MAX_PEER_CARD_ENTRY_LENGTH);
        assert!(validate_peer_card_entry(&at_cap));
    }

    #[test]
    fn constants_match_python() {
        assert_eq!(MAX_PEER_CARD_FACTS, 40);
        assert_eq!(MAX_PEER_CARD_ENTRY_LENGTH, 200);
        assert_eq!(
            PEER_CARD_ALLOWED_PREFIXES,
            ["IDENTITY:", "ATTRIBUTE:", "RELATIONSHIP:", "INSTRUCTION:"]
        );
    }

    #[test]
    fn deduction_tools_match_python_golden() {
        let expected: Value =
            serde_json::from_str(include_str!("fixtures/deduction_tools.json")).unwrap();
        let actual = Value::Array(deduction_specialist_tools());
        assert_eq!(actual, expected);
    }

    #[test]
    fn induction_tools_match_python_golden() {
        let expected: Value =
            serde_json::from_str(include_str!("fixtures/induction_tools.json")).unwrap();
        let actual = Value::Array(induction_specialist_tools());
        assert_eq!(actual, expected);
    }

    #[test]
    fn specialist_tool_names_in_order() {
        let ded_tools = deduction_specialist_tools();
        let ded: Vec<&str> = ded_tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            ded,
            vec![
                "get_recent_observations",
                "search_memory",
                "search_messages",
                "create_observations_deductive",
                "delete_observations",
                "update_peer_card"
            ]
        );
        let ind_tools = induction_specialist_tools();
        let ind: Vec<&str> = ind_tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            ind,
            vec![
                "get_recent_observations",
                "search_memory",
                "search_messages",
                "create_observations_inductive"
            ]
        );
    }
}
