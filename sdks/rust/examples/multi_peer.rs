#![allow(clippy::print_stdout)]
//! Multi-peer session with peer management.
//!
//! Demonstrates adding multiple peers to a session, exchanging messages,
//! and using per-peer dialectic chat.
//!
//! Run with `cargo run --example multi_peer`

use honcho_ai::Honcho;
use honcho_ai::types::dialectic::DialecticOptions;

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("multi-peer-demo")
            .build(),
    )?;

    let alice = honcho.peer("alice").await?;
    let bob = honcho.peer("bob").await?;
    let carol = honcho.peer("carol").await?;

    let session = honcho.session("group-chat").await?;

    session.set_peers([&alice, &bob, &carol]).await?;

    session
        .add_messages(vec![alice.message("Hi everyone!").build()?])
        .await?;
    session
        .add_messages(vec![bob.message("Hey Alice!").build()?])
        .await?;

    let response = alice
        .chat_with_options(
            &DialecticOptions::builder()
                .query("Summarize the conversation")
                .session_id("group-chat")
                .build(),
        )
        .await?;
    if let Some(text) = response {
        println!("Alice's response: {text}");
    }

    let peers = session.peers().await?;
    println!("Session has {} peer(s)", peers.len());

    Ok(())
}
