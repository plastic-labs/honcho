---
name: honcho-memory
description: Concepts and strategy for using a connected Honcho as persistent memory of the user — the recall/record loop and session and peer design. Start here to understand how Honcho memory works, then connect — via a first-class integration for your environment if one exists (preferred), or raw MCP tools (covered here) or the honcho-cli skill (CLI). For embedding the SDK into a codebase, use honcho-integration.
---

# Using Honcho as Memory

Honcho is a memory layer for agents. You feed it the messages from your conversations; in the background it reasons over them and builds a **representation** of each participant. At any point you can ask it natural-language questions about the user ("How technical are they?", "What are they trying to do?") and get grounded, reasoned answers.

This skill is for when Honcho is **already connected** to you and you want to use it. If you're instead adding Honcho to a codebase from scratch, use the `honcho-integration` skill.

> **What's durable vs. what to look up.** The concepts and the recall/record loop below change rarely — rely on them. Specifics that change often — the exact set of integrations, tool names, install commands, headers, and defaults — are illustrative here; treat the linked docs (and your own live tool list) as authoritative and fetch them when the details matter.

## The mental model

- **Peer** — any participant, human or AI. You and the user are both peers. Honcho builds a representation of peers it observes (typically the user, not you).
- **Session** — one conversation thread; messages live in sessions. Honcho reasons over the messages in a session together, so scope each session to one coherent context (a conversation, channel, task, or project) and keep that context's turns in the same session rather than fragmenting them across many thin ones. For low-volume or trickle inputs, append to one ongoing session rather than spinning up a new one each time. See [design patterns](https://honcho.dev/docs/v3/documentation/core-concepts/design-patterns.md) and [reasoning](https://honcho.dev/docs/v3/documentation/core-concepts/reasoning.md).
- **Message** — the raw turns you feed in. No messages → no reasoning → no memory.
- **Conclusion** — a fact Honcho derived (or you stored) about a peer. Conclusions power the representation.
- **Representation / peer card** — the synthesized understanding of a peer, queryable via `chat`. A peer's representation **accumulates across every session** it appears in — that's the cross-conversation memory. Session-scoped data (recent messages, summaries) stays local to one session.

Reasoning happens **asynchronously**. After you record a turn, don't poll or wait — the representation updates in the background and is richer next time you ask.

## The loop: recall → respond → record

Do this every conversation. It's the whole skill.

1. **Once per conversation** — make sure there's a session with you and the user as peers (observe the user, don't observe yourself).
2. **Before responding, when personalization helps** — pull the user's current context (`get_session_context` / `get_representation`) or search past messages (`search`) — these are fast reads. For a reasoned answer to a specific question, ask the dialectic (`chat`) — that one takes a few seconds, so use it when it earns its keep.
3. **After every exchange** — record both the user's message and your reply. This is what makes Honcho learn. Don't skip it.

Optionally, when you learn a durable fact you don't want to wait for background reasoning to surface, **store a conclusion** directly.

## What you get back when you recall

Three ways to pull memory, cheapest first:

- **Representation** (`get_representation`) — Honcho's synthesized understanding of the user as text, ready to drop straight into a system prompt. Near-instant read.
- **Context** (`get_session_context`) — the fuller session view: a session summary + recent messages covering the conversation, and — *only if you target a peer* — that peer's representation folded in. Without a peer target it's session-local (recent turns + summary) and carries no cross-conversation memory. Near-instant read.
- **Dialectic** (`chat`) — a *reasoned* natural-language answer to a specific question ("How does this user like to receive feedback?"). Runs live reasoning, so it takes a few seconds. Use it when a plain read won't answer the question.

The dialectic (`chat`) also takes a **reasoning level** that trades speed for depth — from `minimal` (fast factual lookup) through `low` (the default balance) to `max` (deep synthesis for the hardest questions). Pick the lowest level that answers the question; higher levels are slower and cost more. The full level-by-level table and model routing are in the [chat docs](https://honcho.dev/docs/v3/documentation/features/chat.md).

---

## Pick your access path

The loop is the same; the mechanics depend on how you reach Honcho. **Prefer a purpose-built integration over wiring up raw MCP yourself** — they handle sessions, peers, and the record loop for you, stay current, and are tuned per environment.

1. **A first-class integration exists for your environment? Use it.** In Claude Code, install the [Claude Code plugin](https://honcho.dev/docs/v3/guides/integrations/claude-code.md) (`/plugin marketplace add plastic-labs/claude-honcho`) for persistent memory out of the box; there are also plugins/integrations for [OpenCode](https://honcho.dev/docs/v3/guides/integrations/opencode.md), LangGraph, CrewAI, Discord, and more. Browse the always-current list: <https://honcho.dev/docs/v3/guides/overview.md>.
2. **No integration, but you have MCP tools** (`create_session`, `add_messages_to_session`, `chat`, …) → drive them with the loop above. The MCP server injects its own usage guide on connect, so there's nothing extra to load; to connect a client yourself, see [Setup](#setup-if-not-connected-yet) below. This is the fallback for connected agents.
3. **`honcho` CLI available** in a terminal → use the **`honcho-cli`** skill — for the recall/record loop, and for verifying that memory is actually building (did messages land? is the representation growing? why doesn't it remember me?).
4. **Embedding Honcho into your own codebase** (not just using a connected instance) → use the **`honcho-integration`** skill.

If you're unsure, list your available tools and look for Honcho memory tools (an MCP connection) before falling back to the CLI. Even on the MCP path, the `honcho-cli` skill is the best way to **verify the loop is working** if memory seems off.

---

## Setup (if not connected yet)

You need a Honcho API key — get one free at <https://app.honcho.dev> (starts with `hch-`). Then connect via the path you picked above — a purpose-built integration (recommended), or a raw connection:

- **MCP** — point your client at `https://mcp.honcho.dev` with two headers: `Authorization: Bearer hch-your-key-here` and `X-Honcho-User-Name: YourName` (what Honcho should call the user). Optional: `X-Honcho-Assistant-Name` (default `Assistant`) and `X-Honcho-Workspace-ID` (default `default`; set it to isolate memory per project). Restart the client fully after adding config. Per-client config snippets (Claude Desktop, Cursor, Codex, Windsurf, VS Code, Cline, Zed) are in the [MCP integration guide](https://honcho.dev/docs/v3/guides/integrations/mcp.md). Once connected, the server tells your assistant how to use the tools automatically.
- **CLI** — use the `honcho-cli` skill.

---

## Rules of thumb

- **Always record turns.** Memory only grows from messages you feed in. Recording is the one non-optional step.
- **Modeling the assistant is optional.** Setting `observe_me: false` on the assistant peer skips building a model of it — required only for deterministic bots (scripted output, nothing meaningful to model). For an AI assistant it's fine to leave observation on if you also want a model of the agent.
- **One stable peer ID per entity.** Reuse the same `peer_id` for a person across every session and channel; splitting them (`user`, `user-web`, `user-discord`) builds separate representations and fragments memory.
- **Scope sessions to coherent context buckets.** Honcho reasons over a session's messages together. Scope a session to one active interaction (per-conversation, per-channel, per-task, per-project); create a new one when context genuinely resets (new topic, new day), reuse it while context should keep accumulating. Keeping a context's turns in one session produces a coherent representation; scattering them fragments it.
- **Don't block on reasoning.** It's asynchronous. Respond now; the representation will be richer next time.
- **Reads are cheap; reasoning isn't.** Fetching the representation/context (`get_session_context`, `get_representation`, `search`) is a near-instant read — use it freely. The dialectic (`chat`) runs live reasoning and takes a few seconds, so save it for when you genuinely need a reasoned answer, not every turn.
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
