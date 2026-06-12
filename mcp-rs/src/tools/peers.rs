use crate::config::HonchoConfig;
use crate::honcho_client::HonchoClient;
use crate::tools::{
    ToolError, enum_prop, number_prop, object_schema, optional_string, optional_u64, page_id_list,
    query_push_optional, required_array, required_string, resource_id, response_field_or_value,
    string_array_prop, string_prop,
};
use rmcp::model::JsonObject;
use serde_json::{Map, Value, json};

pub(crate) fn create_peer_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("Unique identifier for the peer")),
            (
                "configuration",
                json!({
                    "type": "object",
                    "properties": {
                        "observeMe": {
                            "type": ["boolean", "null"],
                            "description": "Whether derivation tasks should be created for this peer's messages."
                        }
                    },
                    "additionalProperties": true
                }),
            ),
        ],
        &["peer_id"],
    )
}

pub(crate) fn chat_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The peer to query about")),
            ("query", string_prop("Natural-language question")),
            ("target_peer_id", string_prop("Optional target peer")),
            ("session_id", string_prop("Optional session scope")),
            (
                "reasoning_level",
                enum_prop(
                    "Reasoning effort",
                    &["minimal", "low", "medium", "high", "max"],
                ),
            ),
        ],
        &["peer_id", "query"],
    )
}

pub(crate) fn peer_target_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The observer peer")),
            ("target_peer_id", string_prop("Optional target peer")),
        ],
        &["peer_id"],
    )
}

pub(crate) fn set_peer_card_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The observer peer")),
            ("peer_card", string_array_prop("Array of fact strings")),
            ("target_peer_id", string_prop("Optional target peer")),
        ],
        &["peer_id", "peer_card"],
    )
}

pub(crate) fn get_peer_context_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The observer peer")),
            ("target_peer_id", string_prop("Optional target peer")),
            (
                "search_query",
                string_prop("Optional semantic search query"),
            ),
            (
                "max_conclusions",
                number_prop("Optional max number of conclusions"),
            ),
        ],
        &["peer_id"],
    )
}

pub(crate) fn get_representation_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The observer peer")),
            ("target_peer_id", string_prop("Optional target peer")),
            ("session_id", string_prop("Optional session scope")),
            (
                "search_query",
                string_prop("Optional semantic search query"),
            ),
            (
                "max_conclusions",
                number_prop("Optional max number of conclusions"),
            ),
        ],
        &["peer_id"],
    )
}

pub(crate) async fn create_peer(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let mut body = json!({ "id": peer_id });
    if let Some(configuration) = args.get("configuration") {
        body["configuration"] = configuration.clone();
    }

    let peer: Value = client
        .post_json(&["workspaces", &config.workspace_id, "peers"], body)
        .await?;
    Ok(json!({
        "peer_id": resource_id(&peer),
        "configuration": peer.get("configuration").cloned().unwrap_or_else(|| json!({})),
    }))
}

pub(crate) async fn list_peers(
    client: &HonchoClient,
    config: &HonchoConfig,
) -> Result<Value, ToolError> {
    let page: Value = client
        .post_json(
            &["workspaces", &config.workspace_id, "peers", "list"],
            json!({}),
        )
        .await?;
    Ok(page_id_list(&page, "peers"))
}

pub(crate) async fn chat(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let query = required_string(args, "query")?;
    let mut body = json!({
        "query": query,
        "stream": false,
    });
    if let Some(target) = optional_string(args, "target_peer_id")? {
        body["target"] = json!(target);
    }
    if let Some(session_id) = optional_string(args, "session_id")? {
        body["session_id"] = json!(session_id);
    }
    if let Some(reasoning_level) = optional_string(args, "reasoning_level")? {
        body["reasoning_level"] = json!(reasoning_level);
    }

    let response: Value = client
        .post_json(
            &[
                "workspaces",
                &config.workspace_id,
                "peers",
                &peer_id,
                "chat",
            ],
            body,
        )
        .await?;
    Ok(response
        .get("content")
        .cloned()
        .filter(|value| !value.is_null())
        .unwrap_or_else(|| json!("None")))
}

pub(crate) async fn get_peer_card(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let mut query = Vec::new();
    query_push_optional(
        &mut query,
        "target",
        optional_string(args, "target_peer_id")?,
    );
    let response: Value = client
        .get_json(
            &[
                "workspaces",
                &config.workspace_id,
                "peers",
                &peer_id,
                "card",
            ],
            &query,
        )
        .await?;
    Ok(response
        .get("peer_card")
        .cloned()
        .filter(|value| !value.is_null())
        .unwrap_or_else(|| json!("No peer card found.")))
}

pub(crate) async fn set_peer_card(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let peer_card = Value::Array(required_array(args, "peer_card")?.clone());
    let mut query = Vec::new();
    query_push_optional(
        &mut query,
        "target",
        optional_string(args, "target_peer_id")?,
    );

    let response: Value = client
        .put_json_with_query(
            &[
                "workspaces",
                &config.workspace_id,
                "peers",
                &peer_id,
                "card",
            ],
            &query,
            json!({ "peer_card": peer_card }),
        )
        .await?;
    Ok(response
        .get("peer_card")
        .cloned()
        .unwrap_or_else(|| json!("Peer card set successfully")))
}

pub(crate) async fn get_peer_context(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let mut query = Vec::new();
    query_push_optional(
        &mut query,
        "target",
        optional_string(args, "target_peer_id")?,
    );
    query_push_optional(
        &mut query,
        "search_query",
        optional_string(args, "search_query")?,
    );
    if let Some(max_conclusions) = optional_u64(args, "max_conclusions")? {
        query.push(("max_conclusions", max_conclusions.to_string()));
    }

    let response: Value = client
        .get_json(
            &[
                "workspaces",
                &config.workspace_id,
                "peers",
                &peer_id,
                "context",
            ],
            &query,
        )
        .await?;
    Ok(json!({
        "peer_id": response.get("peer_id").cloned().unwrap_or_else(|| json!(peer_id)),
        "target_id": response.get("target_id").cloned().unwrap_or(Value::Null),
        "representation": response.get("representation").cloned().unwrap_or(Value::Null),
        "peer_card": response.get("peer_card").cloned().unwrap_or(Value::Null),
    }))
}

pub(crate) async fn get_representation(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let mut body = json!({});
    if let Some(target) = optional_string(args, "target_peer_id")? {
        body["target"] = json!(target);
    }
    if let Some(session_id) = optional_string(args, "session_id")? {
        body["session_id"] = json!(session_id);
    }
    if let Some(search_query) = optional_string(args, "search_query")? {
        body["search_query"] = json!(search_query);
    }
    if let Some(max_conclusions) = optional_u64(args, "max_conclusions")? {
        body["max_conclusions"] = json!(max_conclusions);
    }

    let response: Value = client
        .post_json(
            &[
                "workspaces",
                &config.workspace_id,
                "peers",
                &peer_id,
                "representation",
            ],
            body,
        )
        .await?;
    Ok(response_field_or_value(response, "representation"))
}
