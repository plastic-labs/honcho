# Honcho MCP Server — Instructions

## Quick Start: Recommended Flow

The simplest way to use Honcho for a standard user/assistant conversation. Three steps using the general tools.

### 1. Start a conversation (once per conversation)

Create a session and set up the user and assistant peers:

```
create_session
  session_id: "<unique-id>"
```

Then add peers to the session:

```
create_peer
  peer_id: "<user-name>"

create_peer
  peer_id: "Assistant"

add_peers_to_session
  session_id: "<session_id>"
  peers:
    - peer_id: "<user-name>"
      observe_me: true
      observe_others: true
    - peer_id: "Assistant"
      observe_me: false
      observe_others: true
```

Store the `session_id` for the rest of this conversation.

### 2. Get personalization insights (before responding, when helpful)

```
chat
  peer_id: "Assistant"
  query: "What communication style does this user prefer?"
  target_peer_id: "<user-name>"
  session_id: "<session_id>"
```

This calls Honcho's reasoning system to answer your question about the user, grounded in everything Honcho has learned across all their conversations. It takes a few seconds, so use it when personalization would genuinely improve your response.

**Good queries:**

- "What does this message reveal about the user's communication preferences?"
- "How formal or casual should I be?"
- "What is the user really asking for beyond their explicit question?"
- "What emotional state might the user be in right now?"

### 3. Record the turn (after every exchange)

```
add_messages_to_session
  session_id: "<session_id>"
  messages:
    - peer_id: "<user-name>"
      content: "<exact user message>"
    - peer_id: "Assistant"
      content: "<your exact response>"
```

**Always** call this after responding so Honcho can learn from the conversation.

---

## General Tools

The full API for advanced use cases.

### Workspace Tools

| Tool | When to use |
| --- | --- |
| `inspect_workspace` | Inspect a single workspace's details |
| `list_workspaces` | Enumerate available workspaces |
| `search` | Semantic search across messages — scope with optional `peer_id` or `session_id` params |
| `get_metadata` | Read metadata for workspace, peer, or session (scope with optional `peer_id` or `session_id`) |
| `set_metadata` | Store metadata for workspace, peer, or session (scope with optional `peer_id` or `session_id`) |

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

### Session Tools

| Tool | When to use |
| --- | --- |
| `create_session` | Create or get a session with the given ID |
| `list_sessions` | Discover existing conversations |
| `delete_session` | Permanently remove a session |
| `clone_session` | Fork a conversation (optionally up to a specific message) |
| `add_peers_to_session` | Add peers to a session with optional per-session config |
| `remove_peers_from_session` | Remove peers from a session |
| `get_session_peers` | See who is in a session |
| `inspect_session` | Inspect detailed session structure/metadata |
| `add_messages_to_session` | Add messages from specific peers |
| `get_session_messages` | Read conversation history (paginated, with optional metadata filters) |
| `get_session_message` | Get a single message from a session by ID |
| `get_session_context` | Get LLM-ready context (messages + summary) |

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
