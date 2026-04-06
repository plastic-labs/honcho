# Honcho Memory Skill for Zo Computer

Give your AI persistent memory across conversations using [Honcho](https://honcho.dev).

## Features

- **Auto-Memory**: Save user and assistant messages to Honcho with one call
- **Query Memory**: Ask natural language questions about what Honcho remembers ("What are my hobbies?")
- **Context Injection**: Retrieve conversation context formatted for direct LLM use
- **Multi-Workspace Support**: Manage separate memory spaces via `HONCHO_WORKSPACE_ID`

## Installation

```bash
pip install honcho-ai python-dotenv
```

Or with uv:

```bash
uv add honcho-ai python-dotenv
```

## Environment Variables

Create a `.env` file:

```env
HONCHO_API_KEY=your-api-key-here
HONCHO_WORKSPACE_ID=default
```

Get your API key at [honcho.dev](https://honcho.dev).

## Quick Start

```python
from tools.save_memory import save_memory
from tools.query_memory import query_memory
from tools.get_context import get_context

# Save a conversation turn
save_memory("alice", "I love hiking in the mountains", "user", "session-1")
save_memory("alice", "That sounds wonderful!", "assistant", "session-1")

# Query what Honcho remembers
answer = query_memory("alice", "What are my hobbies?", "session-1")
print(answer)  # "Alice enjoys hiking in the mountains."

# Get context ready for an LLM call
messages = get_context("alice", "session-1", "assistant", tokens=4000)
# messages is a list of {"role": ..., "content": ...} dicts
```

## Tool Reference

### `save_memory(user_id, content, role, session_id, assistant_id="assistant")`

Saves a message to Honcho memory.

| Param | Type | Description |
|---|---|---|
| `user_id` | `str` | Unique user identifier |
| `content` | `str` | Message text |
| `role` | `str` | `"user"` or `"assistant"` |
| `session_id` | `str` | Session/conversation identifier |
| `assistant_id` | `str` | Peer ID for the assistant. Defaults to `"assistant"` |

Returns a confirmation string.

---

### `query_memory(user_id, query, session_id=None)`

Queries stored memory using Honcho's Dialectic API.

| Param | Type | Description |
|---|---|---|
| `user_id` | `str` | Unique user identifier |
| `query` | `str` | Natural language question |
| `session_id` | `str \| None` | Optional: scope to a specific session. Defaults to `None` (global memory) |

Returns a natural language answer.

> **Note:** In shared workspaces, `query_memory` may return data from other peers if the queried user has no stored memory yet. The Dialectic API draws from workspace-level context as a fallback. Use unique `HONCHO_WORKSPACE_ID` values per user group in production to prevent cross-peer data leakage.

---

### `get_context(user_id, session_id, assistant_id, tokens=4000)`

Retrieves conversation context in OpenAI message format.

| Param | Type | Description |
|---|---|---|
| `user_id` | `str` | Unique user identifier |
| `session_id` | `str` | Session/conversation identifier |
| `assistant_id` | `str` | Peer ID for the assistant |
| `tokens` | `int` | Max tokens to include (default: 4000) |

Returns a list of `{"role": ..., "content": ...}` dicts.

## Concept Mapping

| Zo Computer | Honcho |
|---|---|
| Account | Workspace |
| User | Peer |
| Conversation | Session |
| Message | Message |

## Running Tests

Requires a running Honcho server. See the [main repo](../../README.md) for setup instructions.

```bash
uv run pytest tests/ -v
```

## Submitting to the Zo Skill Marketplace

To publish this skill to the [Zo Skills Registry](https://github.com/zocomputer/skills):

1. **Fork** the `zocomputer/skills` repository.
2. **Copy** this directory into the `/Community` folder of your fork, naming it `honcho-memory`:

   ```
   Community/
   └── honcho-memory/
       ├── SKILL.md
       ├── README.md
       ├── client.py
       ├── pyproject.toml
       └── tools/
   ```

3. **Validate** your skill:

   ```bash
   bun validate
   ```

4. **Submit a pull request** to the upstream registry repository.

Once merged, the skill will be automatically added to the Zo marketplace `manifest.json`.

## License

AGPL-3.0-or-later
