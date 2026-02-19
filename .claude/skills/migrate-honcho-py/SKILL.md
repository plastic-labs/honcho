---
name: migrate-honcho
description: Migrates Honcho Python SDK code from v1.6.0 to v2.0.0. Use when upgrading honcho package, fixing breaking changes after upgrade, or when errors mention AsyncHoncho, observations, Representation class, .core property, or get_config methods.
---

# Honcho Python SDK Migration (v1.6.0 → v2.0.0)

## Overview

This skill migrates code from `honcho` Python SDK v1.6.0 to v2.0.0 (required for Honcho 3.0.0+).

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

### 10. Update card() return type

```python
# Before
card: str = peer.card()  # Returns str

# After
card: list[str] | None = peer.card()  # Returns list[str] | None
if card:
    print("\n".join(card))
```

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
| `chat(stream=True)` | `chat_stream()` |
| `include_most_derived=` | `include_most_frequent=` |
| `max_observations=` | `max_conclusions=` |
| `last_user_message=` | `search_query=` |
| `config=` | `configuration=` |
| `PeerContext` | `PeerContextResponse` |
| `DeriverStatus` | `QueueStatusResponse` |
| `client.core` | *(removed)* |

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
