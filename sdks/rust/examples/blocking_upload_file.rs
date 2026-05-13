#![allow(clippy::print_stdout)]
//! Blocking (synchronous) API with session messages and configuration.
//!
//! Demonstrates the blocking facade for creating peers, sessions,
//! sending messages, and reading context without async/await.
//!
//! Run with `cargo run --example blocking_upload_file --features blocking`

use honcho_ai::blocking::Honcho;

fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::new("http://localhost:8000", "blocking-file-demo")?;
    honcho.force_ensure()?;

    let peer = honcho.peer("user-1")?;
    let session = honcho.session("sess-1")?;

    session.add_messages(vec![
        peer.message("Uploading a file synchronously").build()?,
    ])?;

    if let Some(response) = peer.chat("What files have I uploaded?")? {
        println!("Response: {response}");
    }

    let opts = honcho_ai::types::session::SessionContextOptions::builder()
        .summary(true)
        .build();
    let ctx = session.context_with_options(&opts)?;
    println!("Context has {} messages", ctx.messages.len());

    let config = session.get_configuration()?;
    println!("Session config: {config:?}");

    Ok(())
}
