use crate::config::HonchoConfig;
use crate::honcho_client::HonchoClient;
use crate::tools::{
    ToolError, format_messages, object_prop, object_schema, optional_string, page_id_list,
    required_object, required_string, resource_id, response_field_or_value, string_prop,
};
use rmcp::model::JsonObject;
use serde_json::{Map, Value, json};

pub(crate) fn search_schema() -> JsonObject {
    object_schema(
        vec![
            ("query", string_prop("Search query")),
            ("peer_id", string_prop("Optional peer scope")),
            ("session_id", string_prop("Optional session scope")),
        ],
        &["query"],
    )
}

pub(crate) fn metadata_scope_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("Optional peer scope")),
            ("session_id", string_prop("Optional session scope")),
        ],
        &[],
    )
}

pub(crate) fn set_metadata_schema() -> JsonObject {
    object_schema(
        vec![
            ("metadata", object_prop("Metadata object")),
            ("peer_id", string_prop("Optional peer scope")),
            ("session_id", string_prop("Optional session scope")),
        ],
        &["metadata"],
    )
}

pub(crate) async fn ensure_workspace(
    client: &HonchoClient,
    config: &HonchoConfig,
) -> Result<Value, ToolError> {
    Ok(client
        .post_json(&["workspaces"], json!({ "id": config.workspace_id }))
        .await?)
}

pub(crate) async fn list_workspaces(client: &HonchoClient) -> Result<Value, ToolError> {
    let page = client.post_json(&["workspaces", "list"], json!({})).await?;
    Ok(format_workspaces_page(&page))
}

pub(crate) async fn inspect_workspace(
    client: &HonchoClient,
    config: &HonchoConfig,
) -> Result<Value, ToolError> {
    let workspace = ensure_workspace(client, config).await?;
    let peers: Value = client
        .post_json(
            &["workspaces", &config.workspace_id, "peers", "list"],
            json!({}),
        )
        .await?;
    let sessions: Value = client
        .post_json(
            &["workspaces", &config.workspace_id, "sessions", "list"],
            json!({}),
        )
        .await?;

    Ok(format_inspect_workspace(
        &workspace,
        &peers,
        &sessions,
        &config.workspace_id,
    ))
}

pub(crate) async fn search(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let query = required_string(args, "query")?;
    let body = json!({ "query": query });

    if let Some(session_id) = optional_string(args, "session_id")? {
        let messages = client
            .post_json(
                &[
                    "workspaces",
                    &config.workspace_id,
                    "sessions",
                    &session_id,
                    "search",
                ],
                body,
            )
            .await?;
        return Ok(format_search_messages(&messages));
    }

    if let Some(peer_id) = optional_string(args, "peer_id")? {
        let messages = client
            .post_json(
                &[
                    "workspaces",
                    &config.workspace_id,
                    "peers",
                    &peer_id,
                    "search",
                ],
                body,
            )
            .await?;
        return Ok(format_search_messages(&messages));
    }

    let messages = client
        .post_json(&["workspaces", &config.workspace_id, "search"], body)
        .await?;
    Ok(format_search_messages(&messages))
}

pub(crate) async fn get_metadata(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    if let Some(peer_id) = optional_string(args, "peer_id")? {
        let peer = client
            .post_json(
                &["workspaces", &config.workspace_id, "peers"],
                json!({ "id": peer_id }),
            )
            .await?;
        return Ok(response_field_or_value(peer, "metadata"));
    }

    if let Some(session_id) = optional_string(args, "session_id")? {
        let session = client
            .post_json(
                &["workspaces", &config.workspace_id, "sessions"],
                json!({ "id": session_id }),
            )
            .await?;
        return Ok(response_field_or_value(session, "metadata"));
    }

    let workspace = ensure_workspace(client, config).await?;
    Ok(response_field_or_value(workspace, "metadata"))
}

pub(crate) async fn set_metadata(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let metadata = Value::Object(required_object(args, "metadata")?.clone());

    if let Some(peer_id) = optional_string(args, "peer_id")? {
        let _peer: Value = client
            .put_json(
                &["workspaces", &config.workspace_id, "peers", &peer_id],
                json!({ "metadata": metadata }),
            )
            .await?;
        return Ok(json!("Peer metadata set successfully"));
    }

    if let Some(session_id) = optional_string(args, "session_id")? {
        let _session: Value = client
            .put_json(
                &["workspaces", &config.workspace_id, "sessions", &session_id],
                json!({ "metadata": metadata }),
            )
            .await?;
        return Ok(json!("Session metadata set successfully"));
    }

    let _workspace: Value = client
        .put_json(
            &["workspaces", &config.workspace_id],
            json!({ "metadata": metadata }),
        )
        .await?;
    Ok(json!("Workspace metadata set successfully"))
}

pub fn format_workspaces_page_for_test(page: &Value) -> Value {
    format_workspaces_page(page)
}

pub fn format_inspect_workspace_for_test(
    workspace: &Value,
    peers: &Value,
    sessions: &Value,
    workspace_id: &str,
) -> Value {
    format_inspect_workspace(workspace, peers, sessions, workspace_id)
}

pub fn format_search_messages_for_test(messages: &Value) -> Value {
    format_search_messages(messages)
}

fn format_workspaces_page(page: &Value) -> Value {
    page_id_list(page, "workspaces")
}

fn format_inspect_workspace(
    workspace: &Value,
    peers: &Value,
    sessions: &Value,
    workspace_id: &str,
) -> Value {
    json!({
        "workspace_id": workspace.get("id").or_else(|| workspace.get("name")).cloned().unwrap_or_else(|| json!(workspace_id)),
        "metadata": workspace.get("metadata").cloned().unwrap_or_else(|| json!({})),
        "configuration": workspace.get("configuration").cloned().unwrap_or_else(|| json!({})),
        "peer_count": peers.get("total").cloned().unwrap_or(Value::Null),
        "peers": page_items_as_ids(peers),
        "session_count": sessions.get("total").cloned().unwrap_or(Value::Null),
        "sessions": page_items_as_ids(sessions),
    })
}

fn format_search_messages(messages: &Value) -> Value {
    format_messages(messages)
}

fn page_items_as_ids(page: &Value) -> Value {
    Value::Array(
        page.get("items")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| json!({ "id": resource_id(item) }))
                    .collect()
            })
            .unwrap_or_default(),
    )
}
