# Honcho MCP Server — Instructions

Honcho keeps a model of the people and agents in a conversation ("peers") and answers questions about them. Two modes use this connection, and they need different tools.

- **Recall** — read what Honcho already knows: `chat`, `workspace_chat`, `search`, `get_representation`, `get_peer_context`, `list_conclusions`. Nothing is written. This is the usual case when the user's own application writes to Honcho and this conversation only reads from it.
- **Memory store** — this conversation is itself what Honcho learns from, because the user asked for it to be remembered or set Honcho up as this assistant's memory. Adds `create_session`, `create_peer`, `add_peers_to_session`, and `add_messages_to_session`.

If the user hasn't asked for this conversation to be recorded, stay in recall mode.

## Setting up a session

In memory-store mode, a session holds the conversation and the peers in it:

```
create_session
  workspace_id: "<workspace-id>"
  session_id: "<unique-id>"
```

Then add peers to the session:

```
create_peer
  workspace_id: "<workspace-id>"
  peer_id: "<user-name>"

create_peer
  workspace_id: "<workspace-id>"
  peer_id: "Assistant"

add_peers_to_session
  workspace_id: "<workspace-id>"
  session_id: "<session_id>"
  peers:
    - peer_id: "<user-name>"
      observe_me: true
      observe_others: true
    - peer_id: "Assistant"
      observe_me: false
      observe_others: true
```

Reuse that `session_id` for the rest of the conversation.

## Recording messages

`add_messages_to_session` appends messages to a tracked session:

```
add_messages_to_session
  workspace_id: "<workspace-id>"
  session_id: "<session_id>"
  messages:
    - peer_id: "<user-name>"
      content: "<the user's message>"
    - peer_id: "Assistant"
      content: "<your response>"
```

Record both sides of each exchange as it happens. Honcho can only learn from
messages it has, and gaps in a session produce a weaker representation than a
complete one.

## Asking what Honcho knows

`chat` answers a question about one peer, grounded in everything Honcho has learned across that peer's conversations:

```
chat
  workspace_id: "<workspace-id>"
  peer_id: "Assistant"
  query: "What communication style does this user prefer?"
  target_peer_id: "<user-name>"
  session_id: "<session_id>"
```

It runs live reasoning and takes a few seconds, so reach for it when the answer would change your response. `workspace_chat` does the same across all peers.

Questions it answers well:

- "What does this message reveal about the user's communication preferences?"
- "How formal or casual should I be?"
- "What is the user really asking for beyond their explicit question?"
- "What emotional state might the user be in right now?"

---

## Best Practices

- **Group messages into coherent context buckets** — give each distinct context its own `session_id` (a chat thread, a project, a channel) and reuse that same `session_id` for every turn within it, rather than minting a new one per turn. Honcho reasons over the messages in a session together, so keeping a context's messages in one bucket produces a coherent representation; scattering them across sessions fragments it.
- **Use one stable `peer_id` per real person**, reused across every session and channel. A fresh or per-channel ID (`user-web` vs. `user-discord`) builds separate, weaker representations instead of one.
- **`observe_me: false` skips building a model of a peer** — reserve it for deterministic bots (nothing meaningful to model). For a real AI assistant it's fine to leave observation on.
- **Reasoning is asynchronous** — don't poll or wait for it to finish before responding. A brand-new or low-volume peer legitimately has little to show yet.
- **Reach for reads before `chat`** — `get_session_context` / `get_peer_context` / `get_representation` / `search` are near-instant; `chat` runs live reasoning and takes a few seconds. Use `chat` only when you need a reasoned answer.

---

## Key Concepts

### Peers

A **peer** is any participant — human or AI. Each peer has a unique ID within the workspace.

### Sessions

A **session** is a conversation context. Sessions track message history, manage which peers participate, and provide context retrieval for LLMs.

### Conclusions

**Conclusions** are facts and observations that Honcho derives from conversations. They power the representation — Honcho's understanding of a peer.

Every conclusion carries its own attribution. `level` says how it was reached: `explicit` conclusions are extracted straight from messages, while `deductive`, `inductive` and `contradiction` conclusions are derived while dreaming. `source_ids` names the conclusions a derived one was built from, and is null for explicit ones. `times_derived` counts how many times Honcho independently reached the same conclusion — a rough confidence signal.

Those fields make the reasoning tree walkable in both directions: `get_conclusions` on a conclusion's `source_ids` steps down toward the explicit facts it rests on, and `get_derived_conclusions` steps up to whatever was built on top of it. Walk down before correcting a fact, and up before deleting one.

### Evidence

`chat` and `workspace_chat` accept `include_evidence`. With it, the answer arrives alongside the conclusions and messages the agent read and the tools it called. This is collated from what the agent actually accessed rather than reported by the model, so it over-reports — a listed conclusion was read, which is not proof the answer leaned on it. It costs no extra model tokens, so reach for it whenever an answer needs auditing.

### Representations

A **representation** is a formatted text summary built from a peer's conclusions. Query it with `get_representation` or `chat`.

### Peer Cards

A **peer card** is a compact list of biographical facts about a peer, automatically maintained by Honcho (or manually via `set_peer_card`).

### Reasoning Level

Several tools accept an optional `reasoning_level` parameter (`minimal`, `low`, `medium`, `high`, `max`). Higher levels produce more thorough answers but take longer and cost more. Default is `low`. Use `minimal` for the fastest lookups; use `high` or `max` when depth matters.

### Dreams

A **dream** is a background memory-consolidation process. It reviews conclusions, merges redundancies, and generates higher-level insights. Schedule one with `schedule_dream` after long conversations.
