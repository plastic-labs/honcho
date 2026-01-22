# Detailed API Changes

## 1. Async Client Architecture (Major Change)

The separate `AsyncHoncho`, `AsyncPeer`, and `AsyncSession` classes have been removed. Use the `.aio` accessor instead.

### Before (v1.6.0)

```python
from honcho import Honcho, AsyncHoncho, AsyncPeer, AsyncSession

# Sync client
client = Honcho()

# Async client - separate class
async_client = AsyncHoncho()
peer = await async_client.peer("user-123")
response = await peer.chat("query")
```

### After (v2.0.0)

```python
from honcho import Honcho

# Single client with .aio accessor for async operations
client = Honcho()

# Sync operations
peer = client.peer("user-123")
response = peer.chat("query")

# Async operations via .aio accessor
peer = await client.aio.peer("user-123")
response = await peer.aio.chat("query")

# Async iteration
async for p in client.aio.peers():
    print(p.id)
```

**Migration steps:**

1. Remove all `AsyncHoncho`, `AsyncPeer`, `AsyncSession` imports
2. Replace `AsyncHoncho()` with `Honcho()` and use `.aio` accessor
3. Replace `AsyncPeer` type hints with `Peer`
4. Replace `AsyncSession` type hints with `Session`
5. Access async methods via `.aio` property on instances

---

## 2. Observations → Conclusions (Terminology Change)

### Before (v1.6.0)

```python
from honcho import Observation, ObservationScope, AsyncObservationScope

# Access observations
scope = peer.observations
scope = peer.observations_of("other-peer")

# List observations
obs_list = scope.list()

# Query observations
results = scope.query("preferences")

# Create observations
scope.create([{"content": "User likes dark mode", "session_id": "sess-1"}])

# Get representation from observations
rep = scope.get_representation()
```

### After (v2.0.0)

```python
from honcho import Conclusion, ConclusionScope, ConclusionScopeAio

# Access conclusions
scope = peer.conclusions
scope = peer.conclusions_of("other-peer")

# List conclusions (now returns SyncPage, not list)
conclusions_page = scope.list()
for conclusion in conclusions_page:
    print(conclusion.content)

# Query conclusions
results = scope.query("preferences")

# Create conclusions
scope.create([{"content": "User likes dark mode", "session_id": "sess-1"}])

# Get representation from conclusions
rep = scope.representation()  # Returns str, not Representation object
```

---

## 3. Representation Type Change (Major Change)

The `Representation` class has been removed. Representations are now simple strings.

### Before (v1.6.0)

```python
from honcho import Representation, ExplicitObservation, DeductiveObservation

# Get working representation
rep: Representation = peer.working_rep()

# Access explicit and deductive observations
for obs in rep.explicit:
    print(obs.content, obs.created_at)

for obs in rep.deductive:
    print(obs.conclusion, obs.premises)

# Check if empty
if rep.is_empty():
    print("No observations")

# Merge representations
rep.merge_representation(other_rep)

# Diff representations
diff = rep.diff_representation(other_rep)

# String formatting
print(str(rep))
print(rep.str_no_timestamps())
print(rep.format_as_markdown())
```

### After (v2.0.0)

```python
# Get representation - now returns str directly
rep: str = peer.representation()

# It's just a string now
print(rep)

# Check if empty
if not rep:
    print("No conclusions")
```

**Removed methods:**

- `.explicit` property
- `.deductive` property
- `.is_empty()`
- `.merge_representation()`
- `.diff_representation()`
- `.str_no_timestamps()`
- `.format_as_markdown()`

---

## 4. Configuration Parameter Rename

All `config` parameters have been renamed to `configuration`, and configuration types are now strongly typed.

### Before (v1.6.0)

```python
# Creating resources with config
peer = client.peer("user-1", config={"observe_me": True})
session = client.session("sess-1", config={"some_setting": True})

# Getting/setting config
config = peer.get_config()
peer.set_config({"observe_me": False})

config = session.get_config()
session.set_config({"some_setting": False})

config = client.get_config()
client.set_config({"workspace_setting": True})

# Message config parameter
msg = peer.message("Hello", config={"reasoning": {"enabled": True}})
```

### After (v2.0.0)

```python
from honcho.api_types import PeerConfig, SessionConfiguration, WorkspaceConfiguration

# Creating resources with configuration (typed)
peer = client.peer("user-1", configuration=PeerConfig(observe_me=True))
session = client.session("sess-1", configuration=SessionConfiguration())

# Getting/setting configuration (returns typed objects)
config: PeerConfig = peer.get_configuration()
peer.set_configuration(PeerConfig(observe_me=False))

config: SessionConfiguration = session.get_configuration()
session.set_configuration(SessionConfiguration())

config: WorkspaceConfiguration = client.get_configuration()
client.set_configuration(WorkspaceConfiguration())

# Message configuration parameter
msg = peer.message("Hello", configuration={"reasoning": {"enabled": True}})
```

---

## 5. Streaming Chat API Change

### Before (v1.6.0)

```python
# Streaming via parameter
response = peer.chat("query", stream=True)
for chunk in response:
    print(chunk, end="")

final = response.get_final_response()
```

### After (v2.0.0)

```python
# Streaming via separate method
stream = peer.chat_stream("query")
for chunk in stream:
    print(chunk, end="")

final = stream.get_final_response()

# Non-streaming (no stream parameter needed)
response = peer.chat("query")
```

---

## 6. Deriver Status → Queue Status

### Before (v1.6.0)

```python
from honcho_core.types import DeriverStatus

# Get status
status: DeriverStatus = client.get_deriver_status()
status = session.get_deriver_status()

# Poll until complete
status = client.poll_deriver_status(timeout=300.0)
status = session.poll_deriver_status(timeout=300.0)

# Access fields
print(status.pending_work_units)
print(status.in_progress_work_units)
```

### After (v2.0.0)

```python
from honcho.api_types import QueueStatusResponse

# Get status
status: QueueStatusResponse = client.queue_status()
status = session.queue_status()

# Access fields (same as before)
print(status.pending_work_units)
print(status.in_progress_work_units)

# poll_deriver_status has been removed - implement polling manually if needed:
import time

def poll_until_complete(client, timeout=300.0):
    start = time.time()
    while time.time() - start < timeout:
        status = client.queue_status()
        if status.pending_work_units == 0 and status.in_progress_work_units == 0:
            return status
        time.sleep(1)
    raise TimeoutError("Queue processing did not complete in time")
```

---

## 7. PeerContext Changes

### Before (v1.6.0)

```python
from honcho import PeerContext

context: PeerContext = peer.get_context()

# Access representation (was Representation object)
rep: Representation = context.representation
if rep:
    print(rep.explicit)
    print(rep.deductive)
```

### After (v2.0.0)

```python
from honcho.api_types import PeerContextResponse

context: PeerContextResponse = peer.context()

# Access representation (now str)
rep: str | None = context.representation
if rep:
    print(rep)
```

---

## 8. Card Method Return Type Change

### Before (v1.6.0)

```python
# card() returned str (joined with newlines)
card: str = peer.card()
print(card)  # "line1\nline2\nline3"
```

### After (v2.0.0)

```python
# card() returns list[str] | None
card: list[str] | None = peer.card()
if card:
    print("\n".join(card))  # Join manually if needed
```

---

## 9. Message Update Location Change

### Before (v1.6.0)

```python
# Update message via client
updated = client.update_message(
    message=msg,
    metadata={"key": "value"},
    session="session-id"  # Required if message is string ID
)
```

### After (v2.0.0)

```python
# Update message via session
updated = session.update_message(
    message=msg,
    metadata={"key": "value"}
)
```

---

## 10. Removed: `core` Property

### Before (v1.6.0)

```python
# Access underlying Stainless-generated client
core_client = client.core
workspace = client.core.workspaces.get_or_create(id="custom-workspace")
```

### After (v2.0.0)

```python
# The `core` property has been removed
# The SDK no longer uses a Stainless-generated client internally
# Use the SDK's public API directly
```

---

## 11. Environment Changes

### Before (v1.6.0)

```python
# Three environments available
client = Honcho(environment="local")
client = Honcho(environment="production")
client = Honcho(environment="demo")
```

### After (v2.0.0)

```python
# Only two environments
client = Honcho(environment="local")
client = Honcho(environment="production")
# "demo" environment has been removed
```

---

## 12. Reasoning Level Parameter (New Feature)

The chat method now supports a `reasoning_level` parameter:

```python
# New in v2.0.0
response = peer.chat(
    "complex query",
    reasoning_level="high"  # "minimal", "low", "medium", "high", "max"
)

stream = peer.chat_stream(
    "complex query",
    reasoning_level="max"
)
```

---

## 13. Import Changes Summary

### Removed Imports

```python
# These no longer exist in v2.0.0
from honcho import AsyncHoncho          # Use Honcho with .aio accessor
from honcho import AsyncPeer            # Use Peer with .aio accessor
from honcho import AsyncSession         # Use Session with .aio accessor
from honcho import Observation          # Renamed to Conclusion
from honcho import ObservationScope     # Renamed to ConclusionScope
from honcho import AsyncObservationScope # Renamed to ConclusionScopeAio
from honcho import Representation       # Removed (now str)
from honcho import ExplicitObservation  # Removed
from honcho import DeductiveObservation # Removed
from honcho import PeerContext          # Use PeerContextResponse from api_types
```

### New Imports

```python
from honcho import Conclusion, ConclusionScope
from honcho import ConclusionScopeAio
from honcho import HonchoAio, PeerAio, SessionAio  # For type hints
from honcho import MessageCreateParams, Message

# Typed configuration classes
from honcho.api_types import (
    PeerConfig,
    SessionConfiguration,
    WorkspaceConfiguration,
    SessionPeerConfig,
    QueueStatusResponse,
    PeerContextResponse,
)
```

### Message Type Import Changes

```python
# Before
from honcho_core.types.workspaces.sessions import MessageCreateParam
from honcho_core.types.workspaces.sessions.message import Message
from honcho.session import SessionPeerConfig

# After
from honcho import Message, MessageCreateParams  # Note: plural "Params"
from honcho.api_types import SessionPeerConfig
```

**Note:** `MessageCreateParam` (singular) is now `MessageCreateParams` (plural).
