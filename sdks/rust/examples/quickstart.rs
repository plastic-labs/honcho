#![allow(clippy::print_stdout)]
//! Basic usage: create a peer, start a session, send messages, and get a response.
//!
//! Run with `cargo run --example quickstart`

use honcho_ai::Honcho;

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("quickstart-demo")
            .build(),
    )?;

    let peer = honcho.peer("user-1").await?;
    let session = honcho.session("sess-1").await?;

    session
        .add_messages(vec![peer.message("Hello, Honcho!").build()?])
        .await?;

    let response = peer.chat("What do you know about me?").await?;
    if let Some(text) = response {
        println!("Response: {text}");
    }

    Ok(())
}
