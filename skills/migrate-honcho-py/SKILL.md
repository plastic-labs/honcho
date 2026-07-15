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

The four changes below are the structural breaks you hit first — apply them inline. The remaining renames and smaller changes are one-line lookups in the [Quick Reference Table](#quick-reference-table); the full before/after for every change lives in [DETAILED-CHANGES.md](DETAILED-CHANGES.md).

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

### Changes 5–15 (at a glance)

The remaining breaks are one-line mappings in the [Quick Reference Table](#quick-reference-table) below, with full before/after in [DETAILED-CHANGES.md](DETAILED-CHANGES.md):

- **5. Method renames** — `get_context()`→`context()`, `get_sessions()`→`sessions()`, `get_messages()`→`messages()`, etc. (drop the `get_` prefix)
- **6. Streaming** — `chat("q", stream=True)` → `chat_stream("q")`
- **7. Queue status** — `get_deriver_status()`→`queue_status()`; `poll_deriver_status()` removed (poll manually)
- **8. Representation params** — `include_most_derived=`→`include_most_frequent=`, `max_observations=`→`max_conclusions=`
- **9. `update_message`** — moved from `client.update_message(..., session=)` to `session.update_message(...)`
- **10. Card** — `card(): str` → `get_card(): list[str] | None`; new `set_card(list[str])` (v2.0.1); `card()` deprecated
- **11. Strict validation** (v2.0.2) — unknown config fields now raise `ValidationError` instead of being ignored
- **12. `peer()`/`session()`** (v2.1.0) — now always make a get-or-create API call (previously lazy)
- **13. New properties** (v2.1.0) — `created_at`, `session.is_active`, `session.get_message(id)`
- **14. Pagination** (v2.1.0) — `page=`, `size=`, `reverse=` on all list methods
- **15. Retries** (v2.1.1) — broader HTTP retry coverage; transparent, no code changes

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
