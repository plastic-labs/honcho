---
name: bot-integrations
description: Integrate Honcho long-term memory into bot frameworks (nanobot, openclaw, picoclaw, etc). Adaptive skill that explores the codebase, identifies the framework's architectural pattern, and implements Honcho wiring using framework-specific references when available.
---

# Honcho Integration for Bot Frameworks

Add [Honcho](https://honcho.dev) long-term memory to a bot framework. This skill targets the common architectural pattern shared by conversational bot frameworks: agent loop, session manager, tool registry, message bus.

## Supported Frameworks

When a known framework is detected, concrete reference implementations are used from `{baseDir}/references/<framework>/`.

| Framework | Status | Reference Dir |
|-----------|--------|---------------|
| [nanobot](https://github.com/HKUDS/nanobot) | concrete references | `references/nanobot/` |
| openclaw | planned | -- |
| picoclaw | planned | -- |

For unknown frameworks, the skill adapts the general pattern below to the codebase's architecture.

## Prerequisites

- A working bot installation with an agent loop, session manager, and tool registry
- A Honcho API key from [app.honcho.dev](https://app.honcho.dev)
- `HONCHO_API_KEY` set in environment or the bot's env config

## Phase 1: Explore

Before implementing anything, explore the codebase to identify:

1. **Agent loop**: Where messages are processed (look for `while` loops calling an LLM)
2. **Session manager**: How conversation history is stored (JSONL files, database, in-memory)
3. **Tool registry**: How tools/functions are registered for the LLM to call
4. **Message bus**: How inbound/outbound messages are routed between channels and the agent
5. **Config system**: How the bot loads configuration (JSON, YAML, env vars, pydantic models)
6. **CLI entry points**: How the bot is started (commands, gateway, agent modes)

If the framework matches a known one (e.g., nanobot), pull the concrete references from `{baseDir}/references/<framework>/` and use them as the implementation target.

## Phase 2: Interview

Ask the user about their deployment:

- **Peer model**: Who are the participants? (typically: one user peer per channel:chat_id, one shared assistant peer)
- **Session granularity**: One session per chat? Per user? Per channel?
- **Observation settings**: Should Honcho observe the user's messages? The assistant's? Both?
- **Workspace ID**: What namespace for this bot's Honcho data?
- **Feature flag**: Should Honcho be opt-in (default `false`) or opt-out (default `true`)?

## Phase 3: Implement

### Step 1: Add `honcho-ai` dependency

Add `honcho-ai>=2.0.1` as a dependency. If the framework supports optional dependencies, make it optional:

```toml
[project.optional-dependencies]
honcho = ["honcho-ai>=2.0.1"]
```

### Step 2: Add config schema

Add a Honcho config section to the bot's configuration system:

```python
class HonchoConfig(BaseModel):
    """Honcho AI-native memory integration (optional feature flag)."""
    enabled: bool = False  # or True for Honcho-first deployments
    workspace_id: str = "default"
    prefetch: bool = True  # inject user context into system prompts
    context_tokens: int | None = None
    environment: str = "production"
```

### Step 3: Create the honcho package

Create a honcho integration package with:

- **Client singleton** (`client.py`): Lazy initialization, deferred imports, `get_honcho_client()` factory
- **Session manager** (`session.py`): Maps bot sessions to Honcho sessions with peer configuration
- **Agent tool** (`honcho_tool.py`): Tool the agent can call to query user context via `peer.chat()`

Key patterns:
- `from __future__ import annotations` + `TYPE_CHECKING` for all honcho imports
- Runtime imports inside functions (never top-level) so the bot doesn't crash without `honcho-ai`
- IDs sanitized to `^[a-zA-Z0-9_-]+` (Honcho requirement)
- User peer: `observe_me=True, observe_others=True`
- Assistant peer: `observe_me=False, observe_others=True`

If references exist for this framework, use them directly from `{baseDir}/references/<framework>/`.

### Step 4: Wire into the agent loop

Add these integration points to the agent loop:

1. **Tool registration** (at startup): If `honcho.enabled` and `HONCHO_API_KEY` set, initialize client + register Honcho tools. Wrap in `try/except ImportError` for graceful degradation.

2. **Context setup** (per message): Set session context on Honcho tools, ensure Honcho session exists.

3. **Prefetch** (per message): Call `session.context()` to get user representation and inject into system prompt before the LLM call.

4. **Sync** (after response): After saving to local session, sync the user+assistant message pair to Honcho.

5. **Migration** (on first activation): If Honcho session is empty but local session has history, upload prior messages as a file via `session.upload_file()`. Also upload `MEMORY.md` and `HISTORY.md` if they exist (from frameworks with local memory consolidation). Archive originals after successful upload.

### Step 5: Pass config through CLI

Pass `honcho_config` to every agent loop instantiation in the CLI commands.

### Step 6: Migration support

When Honcho activates on an instance with existing local data, migrate automatically:

- **Session messages** (JSONL files): Format as XML transcript, upload via `session.upload_file()`
- **Consolidated memory** (MEMORY.md, HISTORY.md): Upload as tagged files with context annotations
- **Archive originals**: Move to `migrated/` subdirectory after successful upload
- **Idempotent**: Skip if Honcho session already has messages

## Phase 4: Verify

After integration, verify:

- [ ] Bot starts normally without `honcho-ai` installed (no import errors)
- [ ] Bot starts normally with `honcho-ai` but without `HONCHO_API_KEY` (graceful skip)
- [ ] With both present and `enabled=true`, logs show "Honcho tools registered"
- [ ] User context is prefetched and visible in system prompts
- [ ] Messages sync to Honcho after each exchange
- [ ] Local session migration works on first Honcho activation
- [ ] Memory file migration works for MEMORY.md/HISTORY.md (if applicable)

## SDK Reference

The integration uses `honcho-ai>=2.0.1`. Key API surface:

| Method | Purpose |
|--------|---------|
| `honcho.peer(name)` | Get or create a peer (lazy, no API call) |
| `honcho.session(name)` | Get or create a session (lazy) |
| `session.add_peers([(peer, config)])` | Configure observation per session |
| `session.add_messages([peer.message(content)])` | Save messages |
| `session.context(summary, tokens)` | Fetch messages + user representation |
| `session.upload_file(file, peer, metadata)` | Upload file as prior context |
| `peer.chat(query, session)` | Dialectic reasoning about the user |
| `peer.conclusions.create(content, sessionId)` | Save explicit facts |

## Patterns

- **Lazy imports everywhere**: `from __future__ import annotations` + `TYPE_CHECKING` for type hints, runtime imports inside functions
- **Feature flag gating**: Always check `config.enabled` AND `HONCHO_API_KEY` before touching Honcho
- **Graceful degradation**: `ImportError` and generic `Exception` catches with logger warnings, never crash the bot
- **Sanitize IDs**: Honcho requires `^[a-zA-Z0-9_-]+` -- replace colons, dots, spaces with dashes
- **Sync after success**: Only mark messages as synced after the API call succeeds, not before
- **Cache consistency**: When creating aliased sessions, store under both original and derived keys
