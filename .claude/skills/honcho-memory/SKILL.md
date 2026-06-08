---
name: honcho-memory
description: Use a connected Honcho to give yourself persistent memory of the user — recall what you've learned about them and record new turns so you keep learning. Use when Honcho memory tools are available (via MCP or the honcho CLI) and you want to remember a user across conversations, recall their preferences, or store what was said. This is the runtime "how do I USE Honcho" skill — not for adding the SDK to a codebase (see honcho-integration for that).
---

# Using Honcho as Memory

Honcho is a memory layer for agents. You feed it the messages from your conversations; in the background it reasons over them and builds a **representation** of each participant. At any point you can ask it natural-language questions about the user ("How technical are they?", "What are they trying to do?") and get grounded, reasoned answers.

This skill is for when Honcho is **already connected** to you and you want to use it. If you're instead adding Honcho to a codebase from scratch, use the `honcho-integration` skill.

> **What's durable vs. what to look up.** The concepts and the recall/record loop below change rarely — rely on them. Specifics that change often — the exact set of integrations, tool names, install commands, headers, and defaults — are illustrative here; treat the linked docs (and your own live tool list) as authoritative and fetch them when the details matter.

## The mental model

- **Peer** — any participant, human or AI. You and the user are both peers. Honcho builds a representation of peers it observes (typically the user, not you).
- **Session** — one conversation thread; messages live in sessions. Reasoning runs on **token-batched** input: Honcho only reasons over a peer once that peer accumulates **~1,000 tokens within a single session** (it queues short turns like "yes"/"ok" until the batch is meaningful — nothing is lost, it just waits). Scope sessions so each observed peer clears that bar; don't fragment a continuous conversation across many thin sessions, or each one stalls below the threshold. For low-volume or trickle inputs, append to one ongoing session rather than spinning up a new one each time. See [design patterns](https://honcho.dev/docs/v3/documentation/core-concepts/design-patterns.md) and [token batching](https://honcho.dev/docs/v3/documentation/core-concepts/reasoning.md).
- **Message** — the raw turns you feed in. No messages → no reasoning → no memory.
- **Conclusion** — a fact Honcho derived (or you stored) about a peer. Conclusions power the representation.
- **Representation / peer card** — the synthesized understanding of a peer, queryable via `chat`. A peer's representation **accumulates across every session** it appears in — that's the cross-conversation memory. Session-scoped data (recent messages, summaries) stays local to one session.

Reasoning happens **asynchronously**. After you record a turn, don't poll or wait — the representation updates in the background and is richer next time you ask.

## The loop: recall → respond → record

Do this every conversation. It's the whole skill.

1. **Once per conversation** — make sure there's a session with you and the user as peers (observe the user, don't observe yourself).
2. **Before responding, when personalization helps** — ask Honcho what it knows about the user (`chat`), or pull relevant past context (`search`). Skip it for trivial turns; it costs a few seconds.
3. **After every exchange** — record both the user's message and your reply. This is what makes Honcho learn. Don't skip it.

Optionally, when you learn a durable fact you don't want to wait for background reasoning to surface, **store a conclusion** directly.

---

## How you're connected

Figure out which access path you have, then use the matching commands below. If you're unsure, list your available tools and look for Honcho memory tools (an MCP connection) before falling back to the CLI.

### Path A — MCP tools

The Honcho MCP server exposes these tools. Names can vary slightly by deployment, so check your actual tool list; this is the canonical set from `mcp.honcho.dev`.

**Recommended flow:**

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

**Reuse the same `session_id` for the whole continuous conversation** (don't mint a new one per turn) — that's what lets the user's messages accumulate past the ~1,000-token reasoning threshold. And use **one stable `peer_id` per real person**, reused across every session and channel; a fresh or per-channel ID (`user-web` vs `user-discord`) builds separate, weaker representations instead of one.

**Other useful tools:**

| Tool | When to use |
| --- | --- |
| `search` | Semantic search across past messages (scope by peer or session). |
| `get_representation` | The user's representation as text — lightweight. |
| `get_peer_context` / `get_session_context` | Fuller context (representation + peer card, or LLM-ready history). Note: `get_session_context` is session-local — for cross-session memory, pass a peer target so it folds in that peer's representation. |
| `get_peer_card` / `set_peer_card` | Read or correct compact biographical facts. |
| `create_conclusions` | Store a fact directly instead of waiting for background reasoning. |
| `list_conclusions` / `query_conclusions` | Review what's known (check before storing duplicates) or find one to delete. |
| `delete_conclusion` | Remove an incorrect or outdated fact. |

`chat` accepts an optional `reasoning_level` (`minimal` → `max`). Use `minimal`/`low` for quick lookups, `high`/`max` when depth genuinely matters — higher is slower and costs more.

### Path B — `honcho` CLI

If you have a terminal and the `honcho` CLI (`uv tool install honcho-cli`, then `honcho init`), the same loop maps to commands. Pass `--json` whenever you process output programmatically.

```bash
# Recall — what does Honcho know about the user?
honcho peer chat <user-id> "What communication style does this user prefer?" --json
honcho peer card <user-id> --json
honcho peer search <user-id> "topic" --json

# Inspect what's stored
honcho conclusion list --observer <user-id> --json
honcho conclusion search "topic" --observer <user-id> --json
honcho session context <session-id> --json

# Record / correct
honcho conclusion create --observer <user-id> "<durable fact>"
honcho conclusion delete <conclusion-id>
```

The CLI is strongest for **inspection and debugging** an existing deployment (peer memory, session context, queue status, dialectic quality). For deep debugging, see the `honcho-cli` skill. Note: recording live conversation turns is the MCP server's job — the CLI doesn't have a one-shot "add this exchange" command the way `add_messages_to_session` does.

---

## Setup (if not connected yet)

You need a Honcho API key — get one free at <https://app.honcho.dev>. It starts with `hch-`.

### First, check for a purpose-built integration

Before hand-rolling raw MCP or CLI, see whether your environment already has a first-class Honcho integration or plugin — it'll handle session mapping, peer setup, and the record loop for you.

**Fetch the integrations overview for the current list:** <https://honcho.dev/docs/v3/guides/overview.md>. It's the authoritative, always-current index — the specific integrations and install commands change often, so read it there rather than trusting any list baked into this skill. Broadly, it spans:

- **Coding agents** (e.g. Claude Code, OpenCode plugins)
- **MCP clients** (Claude Desktop, Cursor, Windsurf, Cline, VS Code, and any MCP client)
- **Agent frameworks** (e.g. LangGraph, CrewAI, Vercel AI SDK, n8n)
- **Platform connectors** (e.g. Discord, Telegram, chat/email/meeting ingestion)

If a purpose-built integration fits your environment, follow its guide from that page and skip the manual MCP/CLI setup below. If you're embedding Honcho into your own Python/TypeScript codebase, use the `honcho-integration` skill instead. Otherwise, wire it up directly:

### Manual connection

**MCP** — point your client at `https://mcp.honcho.dev` with two headers:

- `Authorization: Bearer hch-your-key-here`
- `X-Honcho-User-Name: YourName` (what Honcho should call the user)

Optional headers: `X-Honcho-Assistant-Name` (default `Assistant`) and `X-Honcho-Workspace-ID` (default `default`; set it to isolate memory per project). Client-specific config snippets (Claude Desktop, Claude Code, Cursor, Codex, Windsurf, VS Code, Cline, Zed) are in the [MCP integration guide](https://honcho.dev/docs/v3/guides/integrations/mcp.md). Restart the client fully after adding config.

**CLI** — `uv tool install honcho-cli`, then `honcho init` (stores `apiKey` + `environmentUrl` in `~/.honcho/config.json`), then `honcho doctor` to verify connectivity.

---

## Rules of thumb

- **Always record turns.** Memory only grows from messages you feed in. Recording is the one non-optional step.
- **Don't observe yourself.** The assistant peer should be `observe_me: false` — you want a model of the user, not of the agent.
- **One stable peer ID per entity.** Reuse the same `peer_id` for a person across every session and channel; splitting them (`user`, `user-web`, `user-discord`) builds separate representations and fragments memory.
- **Scope sessions so reasoning actually fires.** Reasoning is token-batched per peer (~1,000 tokens within a session). Keep a continuous interaction in one session and let trickle inputs accumulate there; many tiny sessions each stall below the threshold and never get reasoned over.
- **Don't block on reasoning.** It's asynchronous. Respond now; the representation will be richer next time.
- **Recall when it pays off.** Querying takes a few seconds and costs tokens — use it when personalization improves the response, skip it for trivial turns.
- **Check before you store.** Background reasoning derives most conclusions automatically. Store a conclusion manually only for a durable fact you want available immediately; `list`/`query` first to avoid duplicates.
- **One workspace per app/user-context.** Don't scatter the same user's memory across multiple workspaces.

## Resources

These are the LLM-friendly Markdown versions (append `.md` to any Honcho docs URL to get the raw Markdown; the full machine-readable index is at <https://honcho.dev/docs/llms.txt>).

- Full docs index (for agents): <https://honcho.dev/docs/llms.txt>
- All integrations & plugins: <https://honcho.dev/docs/v3/guides/overview.md>
- MCP server & client setup: <https://honcho.dev/docs/v3/guides/integrations/mcp.md>
- Full MCP usage walkthrough: <https://raw.githubusercontent.com/plastic-labs/honcho/refs/heads/main/mcp/instructions.md>
- Agent development overview: <https://honcho.dev/docs/v3/documentation/introduction/vibecoding.md>
- CLI reference: <https://honcho.dev/docs/v3/documentation/reference/cli.md>
