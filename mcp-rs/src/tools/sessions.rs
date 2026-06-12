use crate::config::HonchoConfig;
use crate::honcho_client::HonchoClient;
use crate::tools::{
    ToolError, boolean_prop, format_message, format_messages, format_summary, object_prop,
    object_schema, optional_bool, optional_string, optional_u64, page_id_list, required_array,
    required_string, required_string_array, resource_id, string_array_prop, string_prop,
};
use rmcp::model::JsonObject;
use serde_json::{Map, Value, json};

pub(crate) fn session_id_schema() -> JsonObject {
    object_schema(
        vec![("session_id", string_prop("The session identifier"))],
        &["session_id"],
    )
}

pub(crate) fn clone_session_schema() -> JsonObject {
    object_schema(
        vec![
            ("session_id", string_prop("The session to clone")),
            ("message_id", string_prop("Optional cutoff message ID")),
        ],
        &["session_id"],
    )
}

pub(crate) fn add_peers_schema() -> JsonObject {
    object_schema(
        vec![
            ("session_id", string_prop("The session to add peers to")),
            (
                "peers",
                json!({
                    "type": "array",
                    "items": {
                        "oneOf": [
                            { "type": "string" },
                            {
                                "type": "object",
                                "properties": {
                                    "peer_id": { "type": "string" },
                                    "observe_me": { "type": ["boolean", "null"] },
                                    "observe_others": { "type": ["boolean", "null"] }
                                },
                                "required": ["peer_id"]
                            }
                        ]
                    }
                }),
            ),
        ],
        &["session_id", "peers"],
    )
}

pub(crate) fn remove_peers_schema() -> JsonObject {
    object_schema(
        vec![
            (
                "session_id",
                string_prop("The session to remove peers from"),
            ),
            ("peer_ids", string_array_prop("Peer IDs to remove")),
        ],
        &["session_id", "peer_ids"],
    )
}

pub(crate) fn add_messages_schema() -> JsonObject {
    object_schema(
        vec![
            ("session_id", string_prop("The session to add messages to")),
            (
                "messages",
                json!({
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "peer_id": { "type": "string" },
                            "content": { "type": "string" },
                            "metadata": { "type": "object", "additionalProperties": true }
                        },
                        "required": ["peer_id", "content"]
                    }
                }),
            ),
        ],
        &["session_id", "messages"],
    )
}

pub(crate) fn get_messages_schema() -> JsonObject {
    object_schema(
        vec![
            (
                "session_id",
                string_prop("The session to get messages from"),
            ),
            ("filters", object_prop("Optional metadata filter criteria")),
        ],
        &["session_id"],
    )
}

pub(crate) fn get_message_schema() -> JsonObject {
    object_schema(
        vec![
            (
                "session_id",
                string_prop("The session the message belongs to"),
            ),
            ("message_id", string_prop("The message ID to fetch")),
        ],
        &["session_id", "message_id"],
    )
}

pub(crate) fn get_context_schema() -> JsonObject {
    object_schema(
        vec![
            ("session_id", string_prop("The session to get context for")),
            (
                "summary",
                boolean_prop("Include a summary of older messages"),
            ),
            ("tokens", crate::tools::number_prop("Target token budget")),
        ],
        &["session_id"],
    )
}

pub(crate) async fn create_session(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let session: Value = client
        .post_json(
            &["workspaces", &config.workspace_id, "sessions"],
            json!({ "id": session_id }),
        )
        .await?;
    Ok(json!({ "session_id": resource_id(&session) }))
}

pub(crate) async fn list_sessions(
    client: &HonchoClient,
    config: &HonchoConfig,
) -> Result<Value, ToolError> {
    let page: Value = client
        .post_json(
            &["workspaces", &config.workspace_id, "sessions", "list"],
            json!({}),
        )
        .await?;
    Ok(page_id_list(&page, "sessions"))
}

pub(crate) async fn delete_session(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let _response: Value = client
        .delete_json(
            &["workspaces", &config.workspace_id, "sessions", &session_id],
            &[],
            None,
        )
        .await?;
    Ok(json!("Session deleted successfully"))
}

pub(crate) async fn clone_session(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let mut query = Vec::new();
    if let Some(message_id) = optional_string(args, "message_id")? {
        query.push(("message_id", message_id));
    }

    let cloned: Value = client
        .post_json_with_query(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "clone",
            ],
            &query,
            json!({}),
        )
        .await?;
    Ok(json!({ "session_id": resource_id(&cloned) }))
}

pub(crate) async fn add_peers_to_session(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let peers = session_peer_map(required_array(args, "peers")?)?;
    let _session: Value = client
        .post_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "peers",
            ],
            Value::Object(peers),
        )
        .await?;
    Ok(json!("Peers added to session successfully"))
}

pub(crate) async fn remove_peers_from_session(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let peer_ids = required_string_array(args, "peer_ids")?;
    let _session: Value = client
        .delete_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "peers",
            ],
            &[],
            Some(json!(peer_ids)),
        )
        .await?;
    Ok(json!("Peers removed from session successfully"))
}

pub(crate) async fn get_session_peers(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let page: Value = client
        .get_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "peers",
            ],
            &[],
        )
        .await?;
    let ids = page
        .get("items")
        .and_then(Value::as_array)
        .map(|items| items.iter().map(resource_id).collect::<Vec<_>>())
        .unwrap_or_default();
    Ok(Value::Array(ids))
}

pub(crate) async fn inspect_session(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let peers: Value = client
        .get_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "peers",
            ],
            &[],
        )
        .await?;
    let messages: Value = client
        .post_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "messages",
                "list",
            ],
            json!({}),
        )
        .await?;
    let summaries: Value = client
        .get_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "summaries",
            ],
            &[],
        )
        .await?;

    let peer_items = peers
        .get("items")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|peer| json!({ "id": resource_id(peer) }))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(json!({
        "session_id": session_id,
        "peers": peer_items,
        "message_count": messages.get("total").cloned().unwrap_or(Value::Null),
        "summaries": format_session_summaries(&summaries, &session_id),
    }))
}

pub(crate) async fn add_messages_to_session(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let messages = Value::Array(required_array(args, "messages")?.clone());
    let _created: Value = client
        .post_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "messages",
            ],
            json!({ "messages": messages }),
        )
        .await?;
    Ok(json!("Messages added to session successfully"))
}

pub(crate) async fn get_session_messages(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let mut body = json!({});
    if let Some(filters) = args.get("filters").filter(|value| !value.is_null()) {
        body["filters"] = filters.clone();
    }

    let page: Value = client
        .post_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "messages",
                "list",
            ],
            body,
        )
        .await?;

    Ok(json!({
        "messages": format_messages(page.get("items").unwrap_or(&Value::Null)),
        "total": page.get("total").cloned().unwrap_or(Value::Null),
        "page": page.get("page").cloned().unwrap_or(Value::Null),
        "pages": page.get("pages").cloned().unwrap_or(Value::Null),
    }))
}

pub(crate) async fn get_session_message(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let message_id = required_string(args, "message_id")?;
    let message: Value = client
        .get_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "messages",
                &message_id,
            ],
            &[],
        )
        .await?;
    Ok(format_message(&message))
}

pub(crate) async fn get_session_context(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let session_id = required_string(args, "session_id")?;
    let mut query = Vec::new();
    if let Some(summary) = optional_bool(args, "summary")? {
        query.push(("summary", summary.to_string()));
    }
    if let Some(tokens) = optional_u64(args, "tokens")? {
        query.push(("tokens", tokens.to_string()));
    }

    let context: Value = client
        .get_json(
            &[
                "workspaces",
                &config.workspace_id,
                "sessions",
                &session_id,
                "context",
            ],
            &query,
        )
        .await?;

    Ok(json!({
        "session_id": context.get("id").or_else(|| context.get("name")).cloned().unwrap_or_else(|| json!(session_id)),
        "summary": context.get("summary").cloned().unwrap_or(Value::Null),
        "messages": format_messages(context.get("messages").unwrap_or(&Value::Null)),
    }))
}

pub fn session_peer_map_for_test(peers: &[Value]) -> Result<Map<String, Value>, ToolError> {
    session_peer_map(peers)
}

fn session_peer_map(peers: &[Value]) -> Result<Map<String, Value>, ToolError> {
    let mut map = Map::new();

    for peer in peers {
        match peer {
            Value::String(peer_id) if !peer_id.is_empty() => {
                map.insert(peer_id.clone(), json!({}));
            }
            Value::Object(peer) => {
                let peer_id = peer
                    .get("peer_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        ToolError::InvalidInput(
                            "Peer objects must include a non-empty peer_id".to_string(),
                        )
                    })?;
                let mut config = Map::new();
                if let Some(value) = peer.get("observe_me") {
                    config.insert("observe_me".to_string(), value.clone());
                }
                if let Some(value) = peer.get("observe_others") {
                    config.insert("observe_others".to_string(), value.clone());
                }
                map.insert(peer_id.to_string(), Value::Object(config));
            }
            _ => {
                return Err(ToolError::InvalidInput(
                    "Peers must be strings or objects with peer_id".to_string(),
                ));
            }
        }
    }

    Ok(map)
}

fn format_session_summaries(summaries: &Value, session_id: &str) -> Value {
    json!({
        "session_id": summaries.get("id").or_else(|| summaries.get("name")).cloned().unwrap_or_else(|| json!(session_id)),
        "short_summary": summaries
            .get("short_summary")
            .filter(|value| !value.is_null())
            .map(format_summary)
            .unwrap_or(Value::Null),
        "long_summary": summaries
            .get("long_summary")
            .filter(|value| !value.is_null())
            .map(format_summary)
            .unwrap_or(Value::Null),
    })
}
