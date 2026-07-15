---
name: honcho-mcp
description: Use Honcho as memory through its MCP server — recall what's known about the user and record turns so it keeps learning. Use when you have Honcho MCP tools available (create_session, add_messages_to_session, chat, search, conclusions, etc.) and want to remember a user across conversations. For the CLI access path use honcho-cli; for the concepts behind the loop see honcho-memory.
---

# Using Honcho via MCP

You're connected to Honcho through its MCP server. This skill is the mechanics of the **recall → respond → record** loop using the MCP tools. For the mental model (peers, sessions, conclusions, why token-batching matters), see the `honcho-memory` skill; for the `honcho` CLI access path, see `honcho-cli`.

> Tool names can vary slightly by deployment — check your actual tool list. This is the canonical set from `mcp.honcho.dev`.

## Recommended flow

```
# 1. Once per conversation: session + peers
create_session            session_id: "<unique-id>"
create_peer               peer_id: "<user-name>"
create_peer               peer_id: "Assistant"
add_peers_to_session      session_id: "<session-id>"
                          peers:
                            - peer_id: "<user-name>"   observe_me: true   observe_others: true
                            - peer_id: "Assistant"     observe_me: false  observe_others: true

# 2. Before responding (when personalization helps): ask what Honcho knows
chat                      peer_id: "Assistant"
                          target_peer_id: "<user-name>"
                          session_id: "<session-id>"
                          query: "What communication style does this user prefer?"

# 3. After every exchange: record the turn
add_messages_to_session   session_id: "<session-id>"
                          messages:
                            - peer_id: "<user-name>"   content: "<exact user message>"
                            - peer_id: "Assistant"     content: "<your exact reply>"
```

**Reuse the same `session_id` for the whole continuous conversation** (don't mint a new one per turn) — that's what lets the user's messages accumulate past the ~1,000-token reasoning threshold so Honcho actually reasons over them. And use **one stable `peer_id` per real person**, reused across every session and channel; a fresh or per-channel ID (`user-web` vs `user-discord`) builds separate, weaker representations instead of one. Set `observe_me: false` on the assistant peer — you want a model of the user, not of yourself. Reasoning is asynchronous; don't poll or wait for it.

## Other useful tools

| Tool | When to use |
| --- | --- |
| `search` | Semantic search across past messages (scope by peer or session). |
| `get_representation` | A contextualized snapshot of the peer's representation, as text you can drop straight into a system prompt to ground the model in who the user is. Fast read. |
| `get_peer_context` / `get_session_context` | Fuller context. `get_session_context` returns a blend of session summary + recent messages covering the whole session; pass a peer target to also fold in that peer's representation (otherwise it's session-local). `get_peer_context` returns the peer's representation + peer card. |
| `get_peer_card` / `set_peer_card` | Read or correct compact biographical facts. |
| `create_conclusions` | Store a fact directly instead of waiting for background reasoning. |
| `list_conclusions` / `query_conclusions` | Review what's known (check before storing duplicates) or find one to delete. |
| `delete_conclusion` | Remove an incorrect or outdated fact. |

## Speed: reads vs. reasoning

- **`chat` is the slow one** — it runs the dialectic (live reasoning over the user's memory), so it takes a few seconds. Use it when you need a reasoned answer, not for every turn.
- **`get_context` / `get_peer_context` / `get_representation` / `search` are reads** — near-instantaneous. Reach for these first when you just need the current representation or history; only call `chat` when you actually need reasoning.

## Reasoning levels

`chat` takes an optional `reasoning_level` (defaults to `low`) that trades speed for depth. Pick by task; higher = slower and costs more:

| Level | Good for |
|-------|----------|
| `minimal` | fast factual lookups |
| `low` | default balance |
| `medium` | multi-step or ambiguous questions |
| `high` | complex synthesis across sources |
| `max` | deep research, the hardest queries |

Full mechanics (model routing, thinking budgets, tool counts) are in the [chat docs](https://honcho.dev/docs/v3/documentation/features/chat.md).

## Verifying it's working

If memory seems off (the user isn't being remembered, `chat` comes back empty), the fastest way to check whether messages landed and the representation is building is the `honcho` CLI — see the **`honcho-cli`** skill (`honcho peer inspect`, `honcho conclusion list`, `honcho session context`). Reasoning is asynchronous and token-batched, so a brand-new or low-volume peer legitimately has little to show yet.

## Connecting (if not connected yet)

Get a free API key at <https://app.honcho.dev> (starts with `hch-`). Point your client at `https://mcp.honcho.dev` with two headers:

- `Authorization: Bearer hch-your-key-here`
- `X-Honcho-User-Name: YourName` (what Honcho should call the user)

Optional: `X-Honcho-Assistant-Name` (default `Assistant`) and `X-Honcho-Workspace-ID` (default `default`; set it to isolate memory per project). Client-specific config snippets (Claude Desktop, Claude Code, Cursor, Codex, Windsurf, VS Code, Cline, Zed) are in the [MCP integration guide](https://honcho.dev/docs/v3/guides/integrations/mcp.md). Restart the client fully after adding config.

> Before hand-rolling, check whether your environment has a first-class Honcho integration that handles this for you: <https://honcho.dev/docs/v3/guides/overview.md>.

## Resources

- Full MCP usage walkthrough: <https://raw.githubusercontent.com/plastic-labs/honcho/refs/heads/main/mcp/instructions.md>
- MCP server & client setup: <https://honcho.dev/docs/v3/guides/integrations/mcp.md>
- Full docs index (for agents): <https://honcho.dev/docs/llms.txt>
