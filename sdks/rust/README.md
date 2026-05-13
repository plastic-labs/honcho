# honcho-ai

![rust-sdk](https://github.com/plastic-labs/honcho/actions/workflows/rust-sdk.yml/badge.svg)

> **Status:** Alpha — do not use in production. This SDK is under active development.

Rust SDK for [Honcho](https://github.com/plastic-labs/honcho) — AI agent memory and social cognition infrastructure.

## Installation

```bash
cargo add honcho-ai
```

## Quickstart

```rust
use honcho_ai::Honcho;

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("my-app")
            .build(),
    )?;

    let peer = honcho.peer("alice").await?;
    let session = honcho.session("sess-1").await?;

    session
        .add_messages(vec![peer.message("Hello from the Rust SDK!").build()?])
        .await?;

    let reply = peer.chat("What do you know about me?").await?;
    if let Some(text) = reply {
        println!("{text}");
    }

    Ok(())
}
```

## API Overview

### Client

```rust
let client = Honcho::new("http://localhost:8000", "my-workspace")?;
// or with builder:
let client = Honcho::from_params(
    Honcho::builder()
        .api_key("...")
        .base_url("http://localhost:8000")
        .workspace_id("my-workspace")
        .build(),
)?;

// Workspace operations
client.force_ensure().await?;
let meta = client.get_metadata().await?;
client.set_metadata(meta).await?;
let config = client.get_configuration().await?;
let page = client.peers().await?;
let page = client.sessions().await?;
let results = client.search("query").await?;
client.schedule_dream("alice").await?;
client.delete_workspace("old-ws").await?;
```

### Peer

```rust
let peer = client.peer("alice").await?;

// Chat (dialectic)
let reply = peer.chat("What does Alice like?").await?;
let reply = peer.chat_with_options(&opts).await?;

// Streaming chat
let stream = peer.chat_stream("Hello").target("bob").send().await?;

// Representation
let rep = peer.representation().await?;
let rep = peer.representation_builder()
    .search_query("hobbies")
    .search_top_k(10)
    .send().await?;

// Context & Card
let ctx = peer.context().await?;
let ctx = peer.context_with_target("bob").await?;
let card = peer.get_card().await?;
peer.set_card(vec!["friendly".into()]).await?;

// Conclusions
let scope = peer.conclusions();
let conclusions = scope.list().await?;
let cross = peer.conclusions_of("bob");

// Sessions
let page = peer.sessions().await?;

// Search
let results = peer.search("topic").await?;

// Metadata CRUD
peer.refresh().await?;
let meta = peer.get_metadata().await?;
peer.set_metadata(meta).await?;
peer.update(patch).await?;
```

### Session

```rust
let session = client.session("sess-1").await?;

// Messages (batch up to 100)
let msgs = session.add_messages(vec![msg]).await?;
let page = session.messages().await?;
let msg = session.get_message("msg-1").await?;
let msg = session.update_message("msg-1", metadata).await?;

// File upload (builder pattern)
let msgs = session.upload_file(FileSource::bytes("doc.pdf", data, "application/pdf"))
    .peer("alice")
    .metadata(json!({"source": "upload"}))
    .send().await?;

// Streaming upload
let msgs = session.upload_file_streamed("large.bin", reader, "application/octet-stream")
    .peer("alice")
    .send().await?;

// Peer management
session.add_peer("alice").await?;
session.add_peers([("alice", config)]).await?;
session.set_peers(["alice", "bob"]).await?;
session.remove_peers(["bob"]).await?;
let peers = session.peers().await?;
let cfg = session.get_peer_configuration("alice").await?;
session.set_peer_configuration("alice", &cfg).await?;

// Context & Search
let ctx = session.context().await?;
let results = session.search("topic").await?;
let summaries = session.summaries().await?;
let rep = session.representation("alice").await?;
let status = session.queue_status().await?;

// Clone & Delete
let cloned = session.clone_session().await?;
let cloned = session.clone_session_with_message("msg-42").await?;
session.delete().await?;

// Metadata CRUD
session.refresh().await?;
let meta = session.get_metadata().await?;
session.set_metadata(meta).await?;
let config = session.get_configuration().await?;
session.set_configuration(config).await?;
```

### Pagination

```rust
let page = client.peers().await?;
for peer in page.items() {
    println!("{}", peer.id);
}

// Auto-fetch all pages
let stream = page.into_stream();
pin_mut!(stream);
while let Some(item) = stream.next().await {
    let peer = item?;
}

// Manual pagination
while page.has_next() {
    page = page.next_page().await?;
}
```

### Blocking API

Enable the `blocking` feature for synchronous access:

```toml
[dependencies]
honcho-ai = { version = "0.2", features = ["blocking"] }
```

```rust
use honcho_ai::blocking::Honcho;

let client = Honcho::new("http://localhost:8000", "my-workspace")?;
let peer = client.peer("alice")?;
let reply = peer.chat("Hello")?;
```

### Rust-Only APIs

These APIs have no equivalent in the Python/TypeScript SDKs:

- `Honcho::force_ensure()` — explicitly trigger workspace creation (normally lazy)
- `Peer::conclusions()` / `Peer::conclusions_of()` — scoped conclusion handles with lazy evaluation
- `FileSource` enum — typed file upload sources (`bytes`, `path`, `stream`)
- `DialecticStream::is_complete()` — check stream termination without consuming

## Features

| Feature       | Default | Description                |
|---------------|---------|----------------------------|
| `rustls-tls`  | yes     | TLS via rustls             |
| `native-tls`  |         | TLS via native backend     |
| `blocking`    |         | Synchronous API wrapper    |
| `tracing`     |         | Emit `tracing` spans       |

## MSRV

1.88

## Parity with Python SDK

| Feature                    | Python | Rust |
|----------------------------|--------|------|
| Peer CRUD                  | ✓      | ✓    |
| Session CRUD               | ✓      | ✓    |
| Messages (batch)           | ✓      | ✓    |
| Dialectic chat             | ✓      | ✓    |
| Streaming chat             | ✓      | ✓    |
| Representation             | ✓      | ✓    |
| Peer card                  | ✓      | ✓    |
| Conclusions                | ✓      | ✓    |
| File upload                | ✓      | ✓    |
| Context (OpenAI/Anthropic) | ✓      | ✓    |
| Blocking API               | ✗      | ✓    |
| Webhooks                   | ✓      | ✗    |
| API keys                   | ✓      | ✗    |

## License

[Apache-2.0](LICENSE-APACHE)

## Links

- [Documentation](https://docs.rs/honcho-ai)
- [Repository](https://github.com/plastic-labs/honcho)
- [OpenAPI Spec](https://github.com/plastic-labs/honcho/tree/main/docs)
- [Migration Guide](./MIGRATION.md)
- [Changelog](./CHANGELOG.md)
