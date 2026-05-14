# Migration Guide

## 0.1.0 → 0.1.1

### R-27: `DialecticStream::final_response` returns `FinalResponse`

**Before:**
```rust
let text: &str = stream.final_response();
```

**After:**
```rust
let resp: &FinalResponse = stream.final_response();
println!("{}", resp.content);
```

`final_response()` now returns a typed [`FinalResponse`](crate::types::dialectic::FinalResponse) struct with a `content: String` field instead of a raw `&str`. Access the text via `.content`. This matches the Python SDK's `get_final_response()` which returns `{"content": "..."}`.

### R-20: Session message methods return `Message` instead of `MessageResponse`

**Before:**
```rust
let msgs: Vec<MessageResponse> = session.add_messages(vec![msg]).await?;
let msg: MessageResponse = session.get_message("id").await?;
let msg: MessageResponse = session.update_message("id", meta).await?;
let results: Vec<MessageResponse> = session.search("query").await?;
// Field access: msg.id, msg.content, msg.metadata
```

**After:**
```rust
let msgs: Vec<Message> = session.add_messages(vec![msg]).await?;
let msg: Message = session.get_message("id").await?;
let msg: Message = session.update_message("id", meta).await?;
let results: Vec<Message> = session.search("query").await?;
// Accessor methods: msg.id(), msg.content(), msg.metadata()
```

`Session::add_messages`, `get_message`, `update_message`, `search`, and `search_with_options` now return the enriched [`Message`](crate::Message) wrapper instead of the raw [`MessageResponse`](crate::types::message::MessageResponse). Access fields via methods (`msg.id()`, `msg.content()`, etc.) instead of direct field access. The blocking equivalents in `blocking::Session` are also updated.

### R-03: `Session::context_with_options`

**Before:**
```rust
session.context_with_options(true, false).await?;
```

**After:**
```rust
use honcho_ai::types::session::SessionContextOptions;
let opts = SessionContextOptions::builder()
    .summary(true)
    .build();
session.context_with_options(&opts).await?;
```

`context_with_options` now takes `&SessionContextOptions` instead of `(bool, bool)`. The old `(summary, limit_to_session)` tuple is replaced by the corresponding fields on the options struct. `context()` still works the same (delegates with `summary(true)`).

> **Important:** When setting `peer_perspective` or `peer_target`, call `.validate()?` after `.build()`:
> ```rust
> let opts = SessionContextOptions::builder()
>     .peer_perspective("alice")
>     .peer_target("bob")
>     .build();
> opts.validate()?;
> session.context_with_options(&opts).await?;
> ```

### R-07: `Page::next_page`

**Before:**
```rust
if let Some(next) = page.next_page().await {
    // ...
}
```

**After:**
```rust
if let Some(next) = page.next_page().await? {
    // ...
}
```

`next_page()` now returns `Result<Option<Page<T>>>` instead of `Option<Page<T>>`. HTTP errors propagate as `Err` instead of being silently swallowed as `None`.

### R-08: `collect_all_pages`

**Before:**
```rust
let all: Vec<T> = collect_all_pages(page).await;
```

**After:**
```rust
let all: Vec<T> = collect_all_pages(page).await?;
```

`collect_all_pages` now returns `Result<Vec<T>>` to propagate pagination errors.

### R-22: Builder `finish_fn` consistency

All `bon::Builder` structs now use `#[builder(finish_fn = build)]`. This was already the case in 0.1.0 but is now enforced as a convention. No migration needed — existing code calling `.build()` continues to work.

### SessionConfiguration: typed session config replaces `HashMap`

**Before (0.1.x):**
```rust
let config: HashMap<String, Value> = session.get_configuration().await?;
let model = config.get("model").and_then(|v| v.as_str());
```

**After (0.1.1):**
```rust
use honcho_ai::types::session::SessionConfiguration;
let config = session.get_configuration().await?;
let enabled = config.reasoning.map(|r| r.enabled).flatten();
```

`get_configuration()` returns `SessionConfiguration` (typed struct) instead of `HashMap<String, Value>`. Access fields directly: `config.reasoning`, `config.summary`, `config.peer_card`, `config.dream`. Use `get_configuration_raw()` for untyped access.

`set_configuration()` takes `&SessionConfiguration` instead of `HashMap<String, Value>`:
```rust
let config = SessionConfiguration {
    summary: Some(SummaryConfiguration {
        enabled: Some(false),
        ..Default::default()
    }),
    ..Default::default()
};
session.set_configuration(&config).await?;
```

### PeerConfig: typed peer config replaces `HashMap`

**Before (0.1.x):**
```rust
let mut config = HashMap::new();
config.insert("observe_me".to_owned(), json!(true));
peer.set_configuration(config).await?;
```

**After (0.1.1):**
```rust
use honcho_ai::PeerConfig;
let config = PeerConfig {
    observe_me: Some(true),
    observe_others: None,
    ..Default::default()
};
peer.set_configuration(&config).await?;
```

`get_configuration()` returns `PeerConfig`, `set_configuration()` takes `&PeerConfig`. Use `get_configuration_raw()` / `set_configuration_raw()` for untyped access (escape hatch).

### Removed: `Peer::card()`

`Peer::card()` has been removed (deprecated since 0.1.0). Use `Peer::get_card()` instead:
```rust
// Before: peer.card().await
// After:
let card = peer.get_card().await?;
```
