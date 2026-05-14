#![allow(clippy::print_stdout)]
//! Session context with custom options: summary + session-scoped representation.
//!
//! Demonstrates `context_with_options` for controlling what context is returned.
//!
//! Run with `cargo run --example context_with_options`

use honcho_ai::Honcho;
use honcho_ai::types::session::SessionContextOptions;

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("context-demo")
            .build(),
    )?;

    let peer = honcho.peer("user-1", None, None).await?;
    let session = honcho.session("sess-1", None, None, None).await?;

    session
        .add_messages(vec![peer.message("Hello from context example!").build()?])
        .await?;

    let ctx = session.context().await?;
    println!("Default context: {} messages", ctx.messages.len());

    let ctx_with_summary = session
        .context_with_options(&SessionContextOptions::builder().summary(true).build())
        .await?;
    println!(
        "With summary: {} messages, summary: {:?}",
        ctx_with_summary.messages.len(),
        ctx_with_summary.summary.as_ref().map(|s| &s.content)
    );

    let ctx_session_only = session
        .context_with_options(
            &SessionContextOptions::builder()
                .summary(true)
                .limit_to_session(true)
                .build(),
        )
        .await?;
    println!(
        "Session-scoped: {} messages",
        ctx_session_only.messages.len()
    );

    let openai = ctx.to_openai("user-1");
    println!("OpenAI format: {} turns", openai.len());

    Ok(())
}
