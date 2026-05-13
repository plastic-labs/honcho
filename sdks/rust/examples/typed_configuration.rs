#![allow(clippy::print_stdout, clippy::expect_used)]
//! Typed workspace configuration: get and set structured config.
//!
//! Demonstrates reading and writing workspace-level configuration
//! using both the typed `WorkspaceConfiguration` API and raw JSON.
//!
//! Run with `cargo run --example typed_configuration`

use std::collections::HashMap;

use honcho_ai::Honcho;

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("config-demo")
            .build(),
    )?;

    let config = honcho.get_configuration().await?;
    println!("Current config: {config:#?}");

    if let Some(ref reasoning) = config.reasoning {
        println!("Reasoning enabled: {:?}", reasoning.enabled);
    }

    let new_config = serde_json::json!({
        "reasoning": {
            "enabled": true,
            "custom_instructions": "Focus on user preferences"
        },
        "summary": {
            "enabled": true,
            "messages_per_short_summary": 20,
            "messages_per_long_summary": 60
        },
        "dream": {
            "enabled": true
        }
    });

    let config_map: HashMap<String, serde_json::Value> =
        serde_json::from_value(new_config).expect("valid JSON");

    honcho.set_configuration_raw(config_map).await?;
    println!("Configuration updated via raw JSON");

    let updated = honcho.get_configuration().await?;
    println!(
        "Reasoning custom instructions: {:?}",
        updated
            .reasoning
            .as_ref()
            .and_then(|r| r.custom_instructions.as_ref())
    );

    let raw = honcho.get_configuration_raw().await?;
    println!("Raw config keys: {:?}", raw.keys().collect::<Vec<_>>());

    Ok(())
}
