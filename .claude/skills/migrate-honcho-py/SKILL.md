---
name: migrate-honcho
description: Migrates Honcho Python SDK code from v1.6.0 to v2.1.1. Use when upgrading honcho package, fixing breaking changes after upgrade, or when errors mention AsyncHoncho, observations, Representation class, .core property, or get_config methods.
---

# Honcho Python SDK Migration (v1.6.0 → v2.1.1)

## Overview

This skill migrates code from `honcho` Python SDK v1.6.0 to v2.1.1 (required for Honcho 3.0.0+).

**Key breaking changes:**

- `AsyncHoncho`/`AsyncPeer`/`AsyncSession` removed → use `.aio` accessor
- "Observation" → "Conclusion" terminology
- `Representation` class removed (returns `str` now)
- `get_config`/`set_config` → `get_configuration`/`set_configuration`
- Streaming via `chat_stream()` instead of `chat(stream=True)`
- `poll_deriver_status()` removed
- `.core` property removed

## Quick Migration

### 1. Update async architecture

```python
# Before
from honcho import AsyncHoncho, AsyncPeer, AsyncSession

async_client = AsyncHoncho()
peer = await async_client.peer("user-123")
response = await peer.chat("query")

# After
from honcho import Honcho

client = Honcho()
peer = await client.aio.peer("user-123")
response = await peer.aio.chat("query")

# Async iteration
async for p in client.aio.peers():
    print(p.id)
```

### 2. Replace observations with conclusions

```python
# Before
from honcho import Observation, ObservationScope, AsyncObservationScope

scope = peer.observations
scope = peer.observations_of("other-peer")
rep = scope.get_representation()

# After
from honcho import Conclusion, ConclusionScope, ConclusionScopeAio

scope = peer.conclusions
scope = peer.conclusions_of("other-peer")
rep = scope.representation()  # Returns str
```

### 3. Update representation handling

```python
# Before
from honcho import Representation, ExplicitObservation, DeductiveObservation

rep: Representation = peer.working_rep()
print(rep.explicit)
print(rep.deductive)
if rep.is_empty():
    print("No observations")

# After
rep: str = peer.representation()
print(rep)  # Just a string now
if not rep:
    print("No conclusions")
```

### 4. Rename configuration methods

```python
# Before
config = peer.get_config()
peer.set_config({"observe_me": False})
session.get_config()
client.get_config()

# After
from honcho.api_types import PeerConfig, SessionConfiguration, WorkspaceConfiguration

config = peer.get_configuration()
peer.set_configuration(PeerConfig(observe_me=False))
session.get_configuration()
client.get_configuration()
```

### 5. Update method names

```python
# Before
peer.working_rep()
peer.get_context()
peer.get_sessions()
session.get_context()
session.get_summaries()
session.get_messages()
session.get_peers()
session.get_peer_config()
client.get_peers()
client.get_sessions()
client.get_workspaces()

# After
peer.representation()
peer.context()
peer.sessions()
session.context()
session.summaries()
session.messages()
session.peers()
session.get_peer_configuration()
client.peers()
client.sessions()
client.workspaces()
```

### 6. Update streaming

```python
# Before
response = peer.chat("query", stream=True)
for chunk in response:
    print(chunk, end="")

# After
stream = peer.chat_stream("query")
for chunk in stream:
    print(chunk, end="")
```

### 7. Update queue status (formerly deriver)

```python
# Before
from honcho_core.types import DeriverStatus

status = client.get_deriver_status()
status = client.poll_deriver_status(timeout=300.0)  # Removed!

# After
from honcho.api_types import QueueStatusResponse

status = client.queue_status()
# poll_deriver_status removed - implement polling manually if needed
```

### 8. Update representation parameters

```python
# Before
rep = peer.working_rep(
    include_most_derived=True,
    max_observations=50
)

# After
rep = peer.representation(
    include_most_frequent=True,
    max_conclusions=50
)
```

### 9. Move update_message to session

```python
# Before
updated = client.update_message(message=msg, metadata={"key": "value"}, session="sess-id")

# After
updated = session.update_message(message=msg, metadata={"key": "value"})
```

### 10. Update card() return type and method name

```python
# Before
card: str = peer.card()  # Returns str

# After (v2.0.0+)
card: list[str] | None = peer.get_card()  # Returns list[str] | None
if card:
    print("\n".join(card))

# peer.card() still works but is deprecated — use get_card()

# New in v2.0.1: set_card()
peer.set_card(["Prefers dark mode", "Located in US"])
```

### 11. Strict input validation (v2.0.2+)

All input models now reject unknown fields via `extra="forbid"` Pydantic validation. Previously, misspelled or extraneous fields were silently ignored.

```python
# Before (v2.0.1 and earlier) — silently ignored
peer = client.peer("user-1", configuration=PeerConfig(observe_mee=True))  # typo silently ignored

# After (v2.0.2+) — raises ValidationError
peer = client.peer("user-1", configuration=PeerConfig(observe_mee=True))  # ValidationError!
```

### 12. peer() and session() always make API calls (v2.1.0+)

**Breaking**: `peer()` and `session()` now always make a get-or-create API call. Previously, calling without metadata/configuration returned a lazy object with no API call.

```python
# Before (v2.0.x) — no API call without options
peer = client.peer("user-123")  # Lazy, no network request

# After (v2.1.0+) — always hits the API
peer = client.peer("user-123")  # Makes POST to /peers (get-or-create)

# Async
peer = await client.aio.peer("user-123")  # Also always hits API
```

### 13. New properties and methods (v2.1.0+)

```python
# created_at on Peer and Session
peer = client.peer("user-123")
print(peer.created_at)  # datetime | None

session = client.session("sess-1")
print(session.created_at)  # datetime | None

# is_active on Session
print(session.is_active)  # bool | None

# get_message() on Session
msg = session.get_message("msg-id")
# Async: msg = await session.aio.get_message("msg-id")
```

### 14. Pagination parameters on list methods (v2.1.0+)

All list methods now accept `page`, `size`, and `reverse` parameters:

```python
# Before (v2.0.x) — only filters
peers_page = client.peers(filters={"metadata": {"role": "admin"}})

# After (v2.1.0+) — pagination controls
peers_page = client.peers(
    filters={"metadata": {"role": "admin"}},
    page=2,
    size=25,
    reverse=True
)

# Works on: client.peers(), client.sessions(), peer.sessions(),
# session.messages(), scope.list()
```

### 15. Broader HTTP retry logic (v2.1.1+)

The SDK now retries on `httpx.NetworkError` and `httpx.RemoteProtocolError` in addition to `TimeoutException` and `ConnectError`. No code changes needed — this is transparent.

## Quick Reference Table

| v1.6.0 | v2.0.0 |
|--------|--------|
| `AsyncHoncho()` | `Honcho()` + `.aio` accessor |
| `AsyncPeer` | `Peer` + `.aio` accessor |
| `AsyncSession` | `Session` + `.aio` accessor |
| `Observation` | `Conclusion` |
| `ObservationScope` | `ConclusionScope` |
| `AsyncObservationScope` | `ConclusionScopeAio` |
| `Representation` | `str` |
| `.observations` | `.conclusions` |
| `.observations_of()` | `.conclusions_of()` |
| `.get_config()` | `.get_configuration()` |
| `.set_config()` | `.set_configuration()` |
| `.working_rep()` | `.representation()` |
| `.get_context()` | `.context()` |
| `.get_sessions()` | `.sessions()` |
| `.get_peers()` | `.peers()` |
| `.get_messages()` | `.messages()` |
| `.get_summaries()` | `.summaries()` |
| `.get_deriver_status()` | `.queue_status()` |
| `.poll_deriver_status()` | *(removed)* |
| `.get_peer_config()` | `.get_peer_configuration()` |
| `.set_peer_config()` | `.set_peer_configuration()` |
| `client.update_message()` | `session.update_message()` |
| `peer.card()` | `peer.get_card()` *(card() deprecated)* |
| *(new)* | `peer.set_card(list[str])` |
| `chat(stream=True)` | `chat_stream()` |
| `include_most_derived=` | `include_most_frequent=` |
| `max_observations=` | `max_conclusions=` |
| `last_user_message=` | `search_query=` |
| `config=` | `configuration=` |
| `PeerContext` | `PeerContextResponse` |
| `DeriverStatus` | `QueueStatusResponse` |
| `client.core` | *(removed)* |
| *(new v2.1.0)* | `peer.created_at` / `session.created_at` |
| *(new v2.1.0)* | `session.is_active` |
| *(new v2.1.0)* | `session.get_message(id)` |
| *(new v2.1.0)* | `page=`, `size=`, `reverse=` on list methods |

## Detailed Reference

For comprehensive details on each change, see:

- [DETAILED-CHANGES.md](DETAILED-CHANGES.md) - Full API change documentation
- [MIGRATION-CHECKLIST.md](MIGRATION-CHECKLIST.md) - Step-by-step checklist

## New Exception Types

```python
from honcho import (
    HonchoError,
    APIError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ConnectionError,
)
```

## New Import Locations

```python
# Configuration types
from honcho.api_types import (
    PeerConfig,
    SessionConfiguration,
    WorkspaceConfiguration,
    SessionPeerConfig,
    QueueStatusResponse,
    PeerContextResponse,
)

# Async type hints
from honcho import HonchoAio, PeerAio, SessionAio

# Message types (note: Params is plural now)
from honcho import Message, MessageCreateParams
```
