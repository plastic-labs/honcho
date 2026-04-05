---
name: honcho-memory
description: Gives AI agents persistent memory across conversations using Honcho. Automatically saves and retrieves user context so the AI remembers preferences, history, and facts between sessions. Use when you need the AI to remember past conversations, recall what a user has told it, inject relevant context into prompts, or manage separate memory spaces for different topics.
license: AGPL-3.0
compatibility: Requires Python 3.9+, honcho-ai>=2.1.1, and a Honcho API key from honcho.dev. Set HONCHO_API_KEY and optionally HONCHO_WORKSPACE_ID in your environment.
metadata:
  author: plastic-labs
  version: "0.1.0"
  honcho-sdk: "2.1.0"
---

# Honcho Memory Skill

This skill provides three tools for storing and retrieving AI memory using [Honcho](https://honcho.dev).

## Setup

1. Get a Honcho API key at [honcho.dev](https://honcho.dev).
2. Set environment variables:
   ```
   HONCHO_API_KEY=your-api-key
   HONCHO_WORKSPACE_ID=default   # optional, defaults to "default"
   ```
3. Install dependencies:
   ```
   pip install honcho-ai python-dotenv
   ```

## Tools

### `save_memory`

Saves a conversation turn (user or assistant message) to Honcho.

**When to use:** After every message exchange to build up the user's memory.

```python
from tools.save_memory import save_memory

save_memory(
    user_id="alice",           # unique user identifier
    content="I love hiking",   # message text
    role="user",               # "user" or "assistant"
    session_id="chat-1"        # conversation session ID
)
```

### `query_memory`

Asks a natural language question against stored memory using Honcho's Dialectic API.

**When to use:** When the user asks "do you remember...?", or when you need to recall facts about the user before responding.

```python
from tools.query_memory import query_memory

answer = query_memory(
    user_id="alice",
    query="What are Alice's hobbies?",
    session_id="chat-1"   # optional: scope to a session
)
# Returns: "Alice enjoys hiking."
```

### `get_context`

Retrieves recent conversation history formatted for direct use in an LLM API call.

**When to use:** At the start of each LLM call to inject relevant context from past conversations.

```python
from tools.get_context import get_context

messages = get_context(
    user_id="alice",
    session_id="chat-1",
    assistant_id="assistant",
    tokens=4000              # max tokens to include
)
# Returns: [{"role": "user", "content": "..."}, ...]
```

## Concept Mapping

| Zo Computer | Honcho |
|---|---|
| Account | Workspace |
| User | Peer |
| Conversation | Session |
| Message | Message |

## Example: Full Conversation Flow

```python
from tools.save_memory import save_memory
from tools.query_memory import query_memory
from tools.get_context import get_context

user_id = "alice"
session_id = "session-1"

# 1. Save user message
save_memory(user_id, "I'm learning Rust and love rock climbing", "user", session_id)

# 2. Save assistant reply
save_memory(user_id, "That's great! Both require patience.", "assistant", session_id)

# 3. In a later session, recall what you know
print(query_memory(user_id, "What does Alice do in her free time?"))
# → "Alice is learning Rust and enjoys rock climbing."

# 4. Get context window for next LLM call
messages = get_context(user_id, session_id, "assistant", tokens=4000)
```
