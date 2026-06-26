---
name: honcho-memory
description: Concepts and strategy for using a connected Honcho as persistent memory of the user — the recall/record loop, session and peer design, and token-batching so reasoning actually fires. Start here to understand how Honcho memory works, then use the path skill for your connection: honcho-mcp (MCP tools) or honcho-cli (CLI). For embedding the SDK into a codebase, use honcho-integration.
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
2. **Before responding, when personalization helps** — pull the user's current context (`get_context` / `get_representation`) or search past messages (`search`) — these are fast reads. For a reasoned answer to a specific question, ask the dialectic (`chat`) — that one takes a few seconds, so use it when it earns its keep.
3. **After every exchange** — record both the user's message and your reply. This is what makes Honcho learn. Don't skip it.

Optionally, when you learn a durable fact you don't want to wait for background reasoning to surface, **store a conclusion** directly.

---

## Pick your access path

The loop is the same; the mechanics depend on how you reach Honcho. **Prefer a purpose-built integration over wiring up raw MCP yourself** — they handle sessions, peers, and the record loop for you, stay current, and are tuned per environment.

1. **A first-class integration exists for your environment? Use it.** In Claude Code, install the [Claude Code plugin](https://honcho.dev/docs/v3/guides/integrations/claude-code.md) (`/plugin marketplace add plastic-labs/claude-honcho`) for persistent memory out of the box; there are also plugins/integrations for [OpenCode](https://honcho.dev/docs/v3/guides/integrations/opencode.md), LangGraph, CrewAI, Discord, and more. Browse the always-current list: <https://honcho.dev/docs/v3/guides/overview.md>.
2. **No integration, but you have MCP tools** (`create_session`, `add_messages_to_session`, `chat`, …) → use the **`honcho-mcp`** skill. The fallback for connected agents.
3. **`honcho` CLI available** in a terminal → use the **`honcho-cli`** skill — for the recall/record loop, and for verifying that memory is actually building (did messages land? is the representation growing? why doesn't it remember me?).
4. **Embedding Honcho into your own codebase** (not just using a connected instance) → use the **`honcho-integration`** skill.

If you're unsure, list your available tools and look for Honcho memory tools (an MCP connection) before falling back to the CLI. Even on the MCP path, the `honcho-cli` skill is the best way to **verify the loop is working** if memory seems off.

---

## Setup (if not connected yet)

You need a Honcho API key — get one free at <https://app.honcho.dev> (starts with `hch-`). Then connect via the path you picked above — a purpose-built integration (recommended), then `honcho-mcp` or `honcho-cli` for the raw connection.

---

## Rules of thumb

- **Always record turns.** Memory only grows from messages you feed in. Recording is the one non-optional step.
- **Don't observe yourself.** The assistant peer should be `observe_me: false` — you want a model of the user, not of the agent.
- **One stable peer ID per entity.** Reuse the same `peer_id` for a person across every session and channel; splitting them (`user`, `user-web`, `user-discord`) builds separate representations and fragments memory.
- **Scope sessions so reasoning actually fires.** Reasoning is token-batched per peer (~1,000 tokens within a session). Scope a session to one active interaction (per-conversation, per-channel, per-task, per-project); create a new one when context genuinely resets (new topic, new day), reuse it while context should keep accumulating. Many tiny sessions each stall below the threshold and never get reasoned over.
- **Don't block on reasoning.** It's asynchronous. Respond now; the representation will be richer next time.
- **Reads are cheap; reasoning isn't.** Fetching the representation/context (`get_context`, `get_representation`, `search`) is a near-instant read — use it freely. The dialectic (`chat`) runs live reasoning and takes a few seconds, so save it for when you genuinely need a reasoned answer, not every turn.
- **Check before you store.** Background reasoning derives most conclusions automatically. Store a conclusion manually only for a durable fact you want available immediately; `list`/`query` first to avoid duplicates.
- **One workspace per app/user-context.** Don't scatter the same user's memory across multiple workspaces.
- **Unify memory across tools with a shared workspace + peer ID.** To give one user continuous memory across several apps or agents (e.g. Claude Code, Cursor, your own app), point them at the same workspace and reuse the same peer ID — that shared ID is what links the representation. See [Unified Memory Setup](https://honcho.dev/docs/v3/guides/recipes/unified-memory-setup.md).

## Resources

These are the LLM-friendly Markdown versions (append `.md` to any Honcho docs URL to get the raw Markdown; the full machine-readable index is at <https://honcho.dev/docs/llms.txt>).

- Full docs index (for agents): <https://honcho.dev/docs/llms.txt>
- All integrations & plugins: <https://honcho.dev/docs/v3/guides/overview.md>
- MCP server & client setup: <https://honcho.dev/docs/v3/guides/integrations/mcp.md>
- Full MCP usage walkthrough: <https://raw.githubusercontent.com/plastic-labs/honcho/refs/heads/main/mcp/instructions.md>
- Agent development overview: <https://honcho.dev/docs/v3/documentation/introduction/vibecoding.md>
- CLI reference: <https://honcho.dev/docs/v3/documentation/reference/cli.md>
