#![allow(clippy::print_stdout)]
//! Upload a file to a session.
//!
//! Run with `cargo run --example upload`

use honcho_ai::{FileSource, Honcho};

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("upload-demo")
            .build(),
    )?;

    let peer = honcho.peer("user-1").await?;
    let session = honcho.session("sess-1").await?;

    let source = FileSource::bytes("hello.txt", b"Hello from a file!".as_slice(), "text/plain");
    let messages = session.upload_file(source).peer(peer.id()).send().await?;

    println!("Uploaded {} message(s)", messages.len());

    Ok(())
}
