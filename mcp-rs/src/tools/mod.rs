use crate::config::HonchoConfig;
use crate::honcho_client::{HonchoApiError, HonchoClient};
use rmcp::model::{JsonObject, Tool};
use serde_json::{Map, Value, json};
use std::fmt;

pub mod conclusions;
pub mod peers;
pub mod sessions;
pub mod system;
pub mod workspace;

#[derive(Debug)]
pub enum ToolError {
    InvalidInput(String),
    Api(HonchoApiError),
}

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::Api(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ToolError {}

impl From<HonchoApiError> for ToolError {
    fn from(value: HonchoApiError) -> Self {
        Self::Api(value)
    }
}

struct ToolDefinition {
    name: &'static str,
    description: &'static str,
    schema: fn() -> JsonObject,
}

static TOOL_DEFINITIONS: &[ToolDefinition] = &[
    ToolDefinition {
        name: "inspect_workspace",
        description: "Inspect the current workspace at a glance.",
        schema: empty_schema,
    },
    ToolDefinition {
        name: "list_workspaces",
        description: "List accessible workspaces.",
        schema: empty_schema,
    },
    ToolDefinition {
        name: "search",
        description: "Search messages across workspace, peer, or session scope.",
        schema: workspace::search_schema,
    },
    ToolDefinition {
        name: "get_metadata",
        description: "Get workspace, peer, or session metadata.",
        schema: workspace::metadata_scope_schema,
    },
    ToolDefinition {
        name: "set_metadata",
        description: "Set workspace, peer, or session metadata.",
        schema: workspace::set_metadata_schema,
    },
    ToolDefinition {
        name: "create_peer",
        description: "Create or get a peer.",
        schema: peers::create_peer_schema,
    },
    ToolDefinition {
        name: "list_peers",
        description: "List peers in the current workspace.",
        schema: empty_schema,
    },
    ToolDefinition {
        name: "chat",
        description: "Ask a natural language question about a peer.",
        schema: peers::chat_schema,
    },
    ToolDefinition {
        name: "get_peer_card",
        description: "Get a peer card.",
        schema: peers::peer_target_schema,
    },
    ToolDefinition {
        name: "set_peer_card",
        description: "Set or update a peer card.",
        schema: peers::set_peer_card_schema,
    },
    ToolDefinition {
        name: "get_peer_context",
        description: "Get comprehensive context for a peer.",
        schema: peers::get_peer_context_schema,
    },
    ToolDefinition {
        name: "get_representation",
        description: "Get the formatted representation for a peer.",
        schema: peers::get_representation_schema,
    },
    ToolDefinition {
        name: "create_session",
        description: "Create or get a session.",
        schema: sessions::session_id_schema,
    },
    ToolDefinition {
        name: "list_sessions",
        description: "List sessions in the current workspace.",
        schema: empty_schema,
    },
    ToolDefinition {
        name: "delete_session",
        description: "Delete a session and its messages.",
        schema: sessions::session_id_schema,
    },
    ToolDefinition {
        name: "clone_session",
        description: "Clone a session.",
        schema: sessions::clone_session_schema,
    },
    ToolDefinition {
        name: "add_peers_to_session",
        description: "Add peers to a session.",
        schema: sessions::add_peers_schema,
    },
    ToolDefinition {
        name: "remove_peers_from_session",
        description: "Remove peers from a session.",
        schema: sessions::remove_peers_schema,
    },
    ToolDefinition {
        name: "get_session_peers",
        description: "Get all peers participating in a session.",
        schema: sessions::session_id_schema,
    },
    ToolDefinition {
        name: "inspect_session",
        description: "Inspect a session at a glance.",
        schema: sessions::session_id_schema,
    },
    ToolDefinition {
        name: "add_messages_to_session",
        description: "Add messages to a session.",
        schema: sessions::add_messages_schema,
    },
    ToolDefinition {
        name: "get_session_messages",
        description: "Get messages from a session.",
        schema: sessions::get_messages_schema,
    },
    ToolDefinition {
        name: "get_session_message",
        description: "Get one message from a session.",
        schema: sessions::get_message_schema,
    },
    ToolDefinition {
        name: "get_session_context",
        description: "Get optimized context for a session.",
        schema: sessions::get_context_schema,
    },
    ToolDefinition {
        name: "list_conclusions",
        description: "List conclusions about a peer.",
        schema: conclusions::peer_target_schema,
    },
    ToolDefinition {
        name: "query_conclusions",
        description: "Semantic search across a peer's conclusions.",
        schema: conclusions::query_schema,
    },
    ToolDefinition {
        name: "create_conclusions",
        description: "Manually create conclusions about a peer.",
        schema: conclusions::create_schema,
    },
    ToolDefinition {
        name: "delete_conclusion",
        description: "Delete a conclusion.",
        schema: conclusions::delete_schema,
    },
    ToolDefinition {
        name: "schedule_dream",
        description: "Schedule a memory consolidation task.",
        schema: system::schedule_dream_schema,
    },
    ToolDefinition {
        name: "get_queue_status",
        description: "Get current processing queue status.",
        schema: empty_schema,
    },
];

pub fn tool_names() -> Vec<&'static str> {
    TOOL_DEFINITIONS
        .iter()
        .map(|definition| definition.name)
        .collect()
}

pub fn all_tools() -> Vec<Tool> {
    TOOL_DEFINITIONS
        .iter()
        .map(|definition| {
            Tool::new(
                definition.name,
                definition.description,
                (definition.schema)(),
            )
        })
        .collect()
}

pub fn get_tool(name: &str) -> Option<Tool> {
    TOOL_DEFINITIONS
        .iter()
        .find(|definition| definition.name == name)
        .map(|definition| {
            Tool::new(
                definition.name,
                definition.description,
                (definition.schema)(),
            )
        })
}

pub async fn dispatch(
    client: &HonchoClient,
    config: &HonchoConfig,
    name: &str,
    args: Map<String, Value>,
) -> Result<Value, ToolError> {
    let _workspace = workspace::ensure_workspace(client, config).await?;

    match name {
        "inspect_workspace" => workspace::inspect_workspace(client, config).await,
        "list_workspaces" => workspace::list_workspaces(client).await,
        "search" => workspace::search(client, config, &args).await,
        "get_metadata" => workspace::get_metadata(client, config, &args).await,
        "set_metadata" => workspace::set_metadata(client, config, &args).await,
        "create_peer" => peers::create_peer(client, config, &args).await,
        "list_peers" => peers::list_peers(client, config).await,
        "chat" => peers::chat(client, config, &args).await,
        "get_peer_card" => peers::get_peer_card(client, config, &args).await,
        "set_peer_card" => peers::set_peer_card(client, config, &args).await,
        "get_peer_context" => peers::get_peer_context(client, config, &args).await,
        "get_representation" => peers::get_representation(client, config, &args).await,
        "create_session" => sessions::create_session(client, config, &args).await,
        "list_sessions" => sessions::list_sessions(client, config).await,
        "delete_session" => sessions::delete_session(client, config, &args).await,
        "clone_session" => sessions::clone_session(client, config, &args).await,
        "add_peers_to_session" => sessions::add_peers_to_session(client, config, &args).await,
        "remove_peers_from_session" => {
            sessions::remove_peers_from_session(client, config, &args).await
        }
        "get_session_peers" => sessions::get_session_peers(client, config, &args).await,
        "inspect_session" => sessions::inspect_session(client, config, &args).await,
        "add_messages_to_session" => sessions::add_messages_to_session(client, config, &args).await,
        "get_session_messages" => sessions::get_session_messages(client, config, &args).await,
        "get_session_message" => sessions::get_session_message(client, config, &args).await,
        "get_session_context" => sessions::get_session_context(client, config, &args).await,
        "list_conclusions" => conclusions::list_conclusions(client, config, &args).await,
        "query_conclusions" => conclusions::query_conclusions(client, config, &args).await,
        "create_conclusions" => conclusions::create_conclusions(client, config, &args).await,
        "delete_conclusion" => conclusions::delete_conclusion(client, config, &args).await,
        "schedule_dream" => system::schedule_dream(client, config, &args).await,
        "get_queue_status" => system::get_queue_status(client, config).await,
        other => Err(ToolError::InvalidInput(format!("Unknown tool: {other}"))),
    }
}

pub(crate) fn empty_schema() -> JsonObject {
    object_schema(vec![], &[])
}

pub(crate) fn object_schema(properties: Vec<(&str, Value)>, required: &[&str]) -> JsonObject {
    let mut property_map = Map::new();
    for (name, value) in properties {
        property_map.insert(name.to_string(), value);
    }

    object_from_value(json!({
        "type": "object",
        "properties": property_map,
        "required": required,
    }))
}

pub(crate) fn string_prop(description: &str) -> Value {
    json!({ "type": "string", "description": description })
}

pub(crate) fn number_prop(description: &str) -> Value {
    json!({ "type": "number", "description": description })
}

pub(crate) fn boolean_prop(description: &str) -> Value {
    json!({ "type": "boolean", "description": description })
}

pub(crate) fn string_array_prop(description: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "string" },
        "description": description
    })
}

pub(crate) fn object_prop(description: &str) -> Value {
    json!({
        "type": "object",
        "additionalProperties": true,
        "description": description
    })
}

pub(crate) fn enum_prop(description: &str, values: &[&str]) -> Value {
    json!({
        "type": "string",
        "enum": values,
        "description": description
    })
}

pub(crate) fn object_from_value(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        _ => unreachable!("schema root must be a JSON object"),
    }
}

pub(crate) fn required_string(args: &Map<String, Value>, name: &str) -> Result<String, ToolError> {
    optional_string(args, name)?
        .ok_or_else(|| ToolError::InvalidInput(format!("Missing required string argument: {name}")))
}

pub(crate) fn optional_string(
    args: &Map<String, Value>,
    name: &str,
) -> Result<Option<String>, ToolError> {
    match args.get(name) {
        Some(Value::String(value)) if !value.is_empty() => Ok(Some(value.clone())),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ToolError::InvalidInput(format!(
            "Argument {name} must be a string"
        ))),
    }
}

pub(crate) fn optional_bool(
    args: &Map<String, Value>,
    name: &str,
) -> Result<Option<bool>, ToolError> {
    match args.get(name) {
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ToolError::InvalidInput(format!(
            "Argument {name} must be a boolean"
        ))),
    }
}

pub(crate) fn optional_u64(
    args: &Map<String, Value>,
    name: &str,
) -> Result<Option<u64>, ToolError> {
    match args.get(name) {
        Some(Value::Number(value)) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| ToolError::InvalidInput(format!("Argument {name} must be a number"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ToolError::InvalidInput(format!(
            "Argument {name} must be a number"
        ))),
    }
}

pub(crate) fn required_array<'a>(
    args: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a Vec<Value>, ToolError> {
    match args.get(name) {
        Some(Value::Array(values)) => Ok(values),
        Some(_) => Err(ToolError::InvalidInput(format!(
            "Argument {name} must be an array"
        ))),
        None => Err(ToolError::InvalidInput(format!(
            "Missing required array argument: {name}"
        ))),
    }
}

pub(crate) fn required_object<'a>(
    args: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a Map<String, Value>, ToolError> {
    match args.get(name) {
        Some(Value::Object(value)) => Ok(value),
        Some(_) => Err(ToolError::InvalidInput(format!(
            "Argument {name} must be an object"
        ))),
        None => Err(ToolError::InvalidInput(format!(
            "Missing required object argument: {name}"
        ))),
    }
}

pub(crate) fn required_string_array(
    args: &Map<String, Value>,
    name: &str,
) -> Result<Vec<String>, ToolError> {
    required_array(args, name)?
        .iter()
        .map(|value| match value {
            Value::String(value) if !value.is_empty() => Ok(value.clone()),
            _ => Err(ToolError::InvalidInput(format!(
                "Argument {name} must contain only strings"
            ))),
        })
        .collect()
}

pub(crate) fn query_push_optional(
    query: &mut Vec<(&'static str, String)>,
    key: &'static str,
    value: Option<String>,
) {
    if let Some(value) = value {
        query.push((key, value));
    }
}

pub(crate) fn response_field_or_value(value: Value, field: &str) -> Value {
    value.get(field).cloned().unwrap_or(value)
}

pub(crate) fn resource_id(value: &Value) -> Value {
    value
        .get("id")
        .or_else(|| value.get("name"))
        .cloned()
        .unwrap_or(Value::Null)
}

pub(crate) fn page_id_list(page: &Value, key: &str) -> Value {
    let items = page
        .get("items")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| json!({ "id": resource_id(item) }))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    json!({
        key: items,
        "total": page.get("total").cloned().unwrap_or(Value::Null),
        "page": page.get("page").cloned().unwrap_or(Value::Null),
        "pages": page.get("pages").cloned().unwrap_or(Value::Null),
    })
}

pub(crate) fn format_message(message: &Value) -> Value {
    json!({
        "id": message.get("id").or_else(|| message.get("public_id")).cloned().unwrap_or(Value::Null),
        "content": message.get("content").cloned().unwrap_or(Value::Null),
        "peer_id": message.get("peer_id").or_else(|| message.get("peer_name")).cloned().unwrap_or(Value::Null),
        "session_id": message.get("session_id").or_else(|| message.get("session_name")).cloned().unwrap_or(Value::Null),
        "metadata": message.get("metadata").cloned().unwrap_or_else(|| json!({})),
        "created_at": message.get("created_at").cloned().unwrap_or(Value::Null),
    })
}

pub(crate) fn format_messages(messages: &Value) -> Value {
    Value::Array(
        messages
            .as_array()
            .map(|items| items.iter().map(format_message).collect())
            .unwrap_or_default(),
    )
}

pub(crate) fn format_summary(summary: &Value) -> Value {
    json!({
        "content": summary.get("content").cloned().unwrap_or(Value::Null),
        "message_id": summary.get("message_id").cloned().unwrap_or(Value::Null),
        "summary_type": summary.get("summary_type").cloned().unwrap_or(Value::Null),
        "created_at": summary.get("created_at").cloned().unwrap_or(Value::Null),
        "token_count": summary.get("token_count").cloned().unwrap_or(Value::Null),
    })
}
