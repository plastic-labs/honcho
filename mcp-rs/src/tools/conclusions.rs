use crate::config::HonchoConfig;
use crate::honcho_client::HonchoClient;
use crate::tools::{
    ToolError, number_prop, object_schema, optional_string, optional_u64, required_string,
    required_string_array, string_array_prop, string_prop,
};
use rmcp::model::JsonObject;
use serde_json::{Map, Value, json};

pub(crate) fn peer_target_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The observer peer")),
            ("target_peer_id", string_prop("Optional observed peer")),
        ],
        &["peer_id"],
    )
}

pub(crate) fn query_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The observer peer")),
            ("query", string_prop("Semantic search query")),
            ("target_peer_id", string_prop("Optional observed peer")),
            ("top_k", number_prop("Max results to return")),
        ],
        &["peer_id", "query"],
    )
}

pub(crate) fn create_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The observer peer")),
            ("target_peer_id", string_prop("The observed peer")),
            (
                "conclusions",
                string_array_prop("Conclusion content strings"),
            ),
            ("session_id", string_prop("Optional session scope")),
        ],
        &["peer_id", "target_peer_id", "conclusions"],
    )
}

pub(crate) fn delete_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The observer peer")),
            ("target_peer_id", string_prop("The observed peer")),
            ("conclusion_id", string_prop("The conclusion ID")),
        ],
        &["peer_id", "target_peer_id", "conclusion_id"],
    )
}

pub(crate) async fn list_conclusions(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let target = optional_string(args, "target_peer_id")?.unwrap_or_else(|| peer_id.clone());
    let page: Value = client
        .post_json(
            &["workspaces", &config.workspace_id, "conclusions", "list"],
            json!({
                "filters": {
                    "observer": peer_id,
                    "observed": target,
                }
            }),
        )
        .await?;

    Ok(json!({
        "conclusions": format_conclusions(page.get("items").unwrap_or(&Value::Null)),
        "total": page.get("total").cloned().unwrap_or(Value::Null),
        "page": page.get("page").cloned().unwrap_or(Value::Null),
        "pages": page.get("pages").cloned().unwrap_or(Value::Null),
    }))
}

pub(crate) async fn query_conclusions(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let query = required_string(args, "query")?;
    let target = optional_string(args, "target_peer_id")?.unwrap_or_else(|| peer_id.clone());
    let top_k = optional_u64(args, "top_k")?.unwrap_or(10);

    let conclusions: Value = client
        .post_json(
            &["workspaces", &config.workspace_id, "conclusions", "query"],
            json!({
                "query": query,
                "top_k": top_k,
                "filters": {
                    "observer_id": peer_id,
                    "observed_id": target,
                }
            }),
        )
        .await?;
    Ok(format_conclusions(&conclusions))
}

pub(crate) async fn create_conclusions(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let target_peer_id = required_string(args, "target_peer_id")?;
    let session_id = optional_string(args, "session_id")?;
    let conclusions = required_string_array(args, "conclusions")?;
    let body = json!({
        "conclusions": conclusions
            .iter()
            .map(|content| {
                json!({
                    "content": content,
                    "observer_id": peer_id,
                    "observed_id": target_peer_id,
                    "session_id": session_id,
                })
            })
            .collect::<Vec<_>>()
    });

    let _created: Value = client
        .post_json(&["workspaces", &config.workspace_id, "conclusions"], body)
        .await?;
    Ok(json!(format!(
        "Created {} conclusion{} successfully",
        conclusions.len(),
        if conclusions.len() == 1 { "" } else { "s" }
    )))
}

pub(crate) async fn delete_conclusion(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let conclusion_id = required_string(args, "conclusion_id")?;
    let _deleted: Value = client
        .delete_json(
            &[
                "workspaces",
                &config.workspace_id,
                "conclusions",
                &conclusion_id,
            ],
            &[],
            None,
        )
        .await?;
    Ok(json!("Conclusion deleted successfully"))
}

fn format_conclusions(value: &Value) -> Value {
    Value::Array(
        value
            .as_array()
            .map(|items| items.iter().map(format_conclusion).collect())
            .unwrap_or_default(),
    )
}

fn format_conclusion(conclusion: &Value) -> Value {
    json!({
        "id": conclusion.get("id").cloned().unwrap_or(Value::Null),
        "content": conclusion.get("content").cloned().unwrap_or(Value::Null),
        "observer_id": conclusion.get("observer_id").or_else(|| conclusion.get("observer")).cloned().unwrap_or(Value::Null),
        "observed_id": conclusion.get("observed_id").or_else(|| conclusion.get("observed")).cloned().unwrap_or(Value::Null),
        "session_id": conclusion.get("session_id").or_else(|| conclusion.get("session_name")).cloned().unwrap_or(Value::Null),
        "created_at": conclusion.get("created_at").cloned().unwrap_or(Value::Null),
    })
}
