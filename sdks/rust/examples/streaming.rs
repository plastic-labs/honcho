#![allow(clippy::print_stdout)]
//! Streaming dialectic: `chat_stream` with stdout drain.
//!
//! Run with `cargo run --example streaming`

use futures_util::StreamExt;
use honcho_ai::Honcho;

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("streaming-demo")
            .build(),
    )?;

    let peer = honcho.peer("user-1").await?;

    let mut stream = peer.chat_stream("Tell me a story").send().await?;

    while let Some(chunk) = stream.next().await {
        let text = chunk?;
        print!("{text}");
    }
    println!();

    Ok(())
}
