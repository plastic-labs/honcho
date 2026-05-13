//! Conclusion lifecycle example: create → query → list → delete.
//!
//! This file demonstrates the conclusion API surface. It compiles but does
//! not run (requires a live Honcho server).

use honcho_ai::{ConclusionCreateParams, Honcho};

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("demo-ws")
            .build(),
    )?;

    let peer = honcho.peer("alice").await?;

    // Self-scoped conclusions (observer = observed = alice)
    let scope = peer.conclusions();

    // Create one conclusion
    let created = scope
        .create([ConclusionCreateParams::new("Alice likes dark mode")])
        .await?;
    println!("created: {:?}", created);

    // Create with session scope
    let _ = scope
        .create([ConclusionCreateParams::builder()
            .content("Alice prefers async/await".to_owned())
            .session_id("sess-1".to_owned())
            .build()])
        .await?;

    // Cross-peer conclusions (observer = alice, observed = bob)
    let cross = peer.conclusions_of("bob");
    let _ = cross
        .create([ConclusionCreateParams::new("Bob is a morning person")])
        .await?;

    // List conclusions (paginated)
    let page = scope.list().page(1).size(10).send().await?;
    println!("list: total={}, page={}", page.total(), page.page());

    // Semantic query
    let results = scope
        .query("programming preferences")
        .top_k(5)
        .send()
        .await?;
    println!("query returned {} results", results.len());

    // Scoped representation
    let rep = cross
        .representation()
        .search_query("personality")
        .max_conclusions(20)
        .send()
        .await?;
    println!("representation: {rep}");

    // Delete
    if let Some(first) = created.first() {
        scope.delete(first.id()).await?;
        println!("deleted {}", first.id());
    }

    Ok(())
}
