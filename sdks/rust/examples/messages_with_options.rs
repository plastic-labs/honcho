#![allow(clippy::print_stdout)]
//! Messages with filters: pagination, metadata, and search.
//!
//! Demonstrates adding messages with metadata, listing, searching,
//! and updating message metadata.
//!
//! Run with `cargo run --example messages_with_options`

use std::collections::HashMap;

use honcho_ai::Honcho;

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("messages-demo")
            .build(),
    )?;

    let peer = honcho.peer("user-1", None, None).await?;
    let session = honcho.session("sess-1", None, None, None).await?;

    let mut meta = HashMap::new();
    meta.insert("tag".into(), "important".into());

    let msg = peer
        .message("Important announcement")
        .metadata(meta.clone())
        .build()?;

    let created = session.add_messages(vec![msg]).await?;
    println!("Created {} message(s)", created.len());

    let page = session.messages().await?;
    println!(
        "Session has {} messages (page {})",
        page.items().len(),
        page.page()
    );

    let results = session.search("announcement").await?;
    println!("Search returned {} result(s)", results.len());

    let search_opts = honcho_ai::types::message::MessageSearchOptions {
        query: "important".into(),
        filters: None,
        limit: 5,
    };
    let filtered = session.search_with_options(&search_opts).await?;
    println!("Filtered search returned {} result(s)", filtered.len());

    if let Some(first) = created.first() {
        let mut update_meta = HashMap::new();
        update_meta.insert("reviewed".into(), true.into());
        let updated = session.update_message(first.id(), update_meta).await?;
        println!("Updated message {}", updated.id());
    }

    Ok(())
}
