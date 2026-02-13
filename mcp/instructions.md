# Honcho MCP Server — Instructions

## Quick Start: Bespoke Flow

The simplest way to use Honcho is the **bespoke flow** — three tools that handle everything for a standard user/assistant conversation.

### 1. Start a conversation (once per conversation)

```
start_conversation
```

Returns a `session_id`. Store it for the rest of this conversation.

### 2. Get personalization insights (before responding, when helpful)

```
get_personalization_insights
  session_id: "<session_id>"
  query: "What communication style does this user prefer?"
```

This calls Honcho's reasoning system to answer your question about the user, grounded in everything Honcho has learned across all their conversations. It takes a few seconds, so use it when personalization would genuinely improve your response.

**Good queries:**

- "What does this message reveal about the user's communication preferences?"
- "How formal or casual should I be?"
- "What is the user really asking for beyond their explicit question?"
- "What emotional state might the user be in right now?"

### 3. Record the turn (after every exchange)

```
add_turn
  session_id: "<session_id>"
  messages:
    - role: "user"
      content: "<exact user message>"
    - role: "assistant"
      content: "<your exact response>"
```

**Always** call this after responding so Honcho can learn from the conversation.

---

## General Tools

Beyond the bespoke flow, Honcho exposes the full API for advanced use cases.

### Workspace Tools

| Tool | When to use |
| --- | --- |
| `search_workspace` | Find messages across all sessions and peers |
| `get_workspace_metadata` | Read workspace-level settings |
| `set_workspace_metadata` | Store workspace-level settings |

### Peer Tools

| Tool | When to use |
| --- | --- |
| `create_peer` | Register a new participant (user or agent) |
| `list_peers` | See all participants in the workspace |
| `chat` | Ask Honcho what it knows about any peer. Accepts optional `reasoning_level` (`minimal`–`max`) to control depth vs. speed. |
| `get_peer_card` | Get compact biographical facts about a peer |
| `set_peer_card` | Manually set/correct facts about a peer |
| `get_peer_context` | Get full context (representation + peer card) |
| `get_representation` | Get the textual representation from conclusions |
| `get_peer_metadata` / `set_peer_metadata` | Custom attributes on a peer |
| `search_peer_messages` | Find messages by a specific peer |

### Session Tools

| Tool | When to use |
| --- | --- |
| `create_session` | Create a raw session (use `start_conversation` for the simple flow) |
| `list_sessions` | Discover existing conversations |
| `delete_session` | Permanently remove a session |
| `clone_session` | Fork a conversation (optionally up to a specific message) |
| `add_peers_to_session` / `remove_peers_from_session` | Manage session participants |
| `get_session_peers` | See who is in a session |
| `add_messages_to_session` | Add messages from specific peers |
| `get_session_messages` | Read conversation history |
| `search_session_messages` | Semantic search within a session |
| `get_session_context` | Get LLM-ready context (messages + summary) |
| `get_session_representation` | Get a peer's session-scoped representation |
| `get_session_metadata` / `set_session_metadata` | Custom attributes on a session |

### Conclusion Tools

| Tool | When to use |
| --- | --- |
| `list_conclusions` | See what Honcho has derived about a peer |
| `query_conclusions` | Semantic search across derived facts |
| `create_conclusions` | Inject facts manually |
| `delete_conclusion` | Remove incorrect or outdated facts |

### System Tools

| Tool | When to use |
| --- | --- |
| `schedule_dream` | Trigger memory consolidation for better insights |
| `get_queue_status` | Check if background processing is complete |

---

## Key Concepts

### Peers

A **peer** is any participant — human or AI. Each peer has a unique ID within the workspace.

### Sessions

A **session** is a conversation context. Sessions track message history, manage which peers participate, and provide context retrieval for LLMs.

### Conclusions

**Conclusions** are facts and observations that Honcho derives from conversations. They power the representation — Honcho's understanding of a peer.

### Representations

A **representation** is a formatted text summary built from a peer's conclusions. Query it with `get_representation` or `chat`.

### Peer Cards

A **peer card** is a compact list of biographical facts about a peer, automatically maintained by Honcho (or manually via `set_peer_card`).

### Reasoning Level

Several tools accept an optional `reasoning_level` parameter (`minimal`, `low`, `medium`, `high`, `max`). Higher levels produce more thorough answers but take longer and cost more. Default is `low`. Use `minimal` for the fastest lookups; use `high` or `max` when depth matters.

### Dreams

A **dream** is a background memory-consolidation process. It reviews conclusions, merges redundancies, and generates higher-level insights. Schedule one with `schedule_dream` after long conversations.
