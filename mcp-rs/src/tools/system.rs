use crate::config::HonchoConfig;
use crate::honcho_client::HonchoClient;
use crate::tools::{ToolError, object_schema, optional_string, required_string, string_prop};
use rmcp::model::JsonObject;
use serde_json::{Map, Value, json};

pub(crate) fn schedule_dream_schema() -> JsonObject {
    object_schema(
        vec![
            ("peer_id", string_prop("The observer peer to dream for")),
            ("target_peer_id", string_prop("Optional target peer")),
            ("session_id", string_prop("Optional session scope")),
        ],
        &["peer_id"],
    )
}

pub(crate) async fn schedule_dream(
    client: &HonchoClient,
    config: &HonchoConfig,
    args: &Map<String, Value>,
) -> Result<Value, ToolError> {
    let peer_id = required_string(args, "peer_id")?;
    let observed = optional_string(args, "target_peer_id")?;
    let session_id = optional_string(args, "session_id")?;
    let _response: Value = client
        .post_json(
            &["workspaces", &config.workspace_id, "schedule_dream"],
            json!({
                "observer": peer_id,
                "observed": observed,
                "dream_type": "omni",
                "session_id": session_id,
            }),
        )
        .await?;
    Ok(json!("Dream scheduled successfully"))
}

pub(crate) async fn get_queue_status(
    client: &HonchoClient,
    config: &HonchoConfig,
) -> Result<Value, ToolError> {
    Ok(client
        .get_json(
            &["workspaces", &config.workspace_id, "queue", "status"],
            &[],
        )
        .await?)
}
