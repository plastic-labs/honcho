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

## Features

| Feature       | Default | Description                |
|---------------|---------|----------------------------|
| `rustls-tls`  | yes     | TLS via rustls             |
| `native-tls`  |         | TLS via native backend     |
| `blocking`    |         | Synchronous API wrapper    |
| `tracing`     |         | Emit `tracing` spans       |

## MSRV

1.80

## License

[Apache-2.0](LICENSE-APACHE)

## Links

- [Documentation](https://docs.rs/honcho-ai)
- [Repository](https://github.com/plastic-labs/honcho)
- [OpenAPI Spec](https://github.com/plastic-labs/honcho/tree/main/docs)
