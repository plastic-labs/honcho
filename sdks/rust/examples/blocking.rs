#![allow(clippy::print_stdout)]
//! Blocking (synchronous) API usage.
//!
//! Run with `cargo run --example blocking --features blocking`

use honcho_ai::blocking::Honcho;

fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::new("http://localhost:8000", "blocking-demo")?;
    honcho.force_ensure()?;

    let peer = honcho.peer("user-1", None, None)?;
    let session = honcho.session("sess-1", None, None, None)?;

    session.add_messages(vec![peer.message("Hello from blocking!").build()?])?;

    if let Some(response) = peer.chat("What do you know about me?")? {
        println!("Response: {response}");
    }

    Ok(())
}
