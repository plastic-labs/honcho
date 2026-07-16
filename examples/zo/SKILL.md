---
name: honcho-memory
description: Long-term memory for AI agents via the official Honcho CLI. Save conversation turns, ask natural-language memory questions, and pull context for prompts.
license: AGPL-3.0
compatibility: Requires uv, honcho-cli, and HONCHO_API_KEY. The wrapper auto-installs honcho-cli with uv if missing.
metadata:
  author: plastic-labs
  version: "0.4.0"
  honcho-cli: "0.1.0+"
---
# Honcho Memory

Use this when an agent needs persistent memory across conversations.

This skill is intentionally thin: `scripts/memory.py` is a small Zo-friendly wrapper around the official `honcho` CLI.

## Setup

1. Get a Honcho API key from [app.honcho.dev](https://app.honcho.dev/api-keys).
2. Add `HONCHO_API_KEY` in [Settings > Advanced](/?t=settings&s=advanced).
3. Optional: add `HONCHO_WORKSPACE_ID`, `HONCHO_USER_ID`, `HONCHO_ASSISTANT_ID`, or `HONCHO_SESSION_ID`.
4. Verify:

```bash
python3 Skills/honcho-memory/scripts/memory.py test
```

## Agent workflow

Before responding, get context when memory would help:

```bash
python3 Skills/honcho-memory/scripts/memory.py context --session chat-1
```

After meaningful exchanges, save the turn:

```bash
python3 Skills/honcho-memory/scripts/memory.py save "User prefers concise tools" --session chat-1
```

Ask Honcho directly:

```bash
python3 Skills/honcho-memory/scripts/memory.py ask "What should I know about the user's preferences?"
```

## Commands

```bash
python3 Skills/honcho-memory/scripts/memory.py save "memory" --session chat-1
python3 Skills/honcho-memory/scripts/memory.py ask "question"
python3 Skills/honcho-memory/scripts/memory.py context --session chat-1 --tokens 2000
python3 Skills/honcho-memory/scripts/memory.py search "topic"
python3 Skills/honcho-memory/scripts/memory.py messages --session chat-1
python3 Skills/honcho-memory/scripts/memory.py doctor
python3 Skills/honcho-memory/scripts/memory.py test
```

For a full exchange, pipe JSON:

```bash
cat messages.json | python3 Skills/honcho-memory/scripts/memory.py save --session chat-1
```

```json
[
  {"role": "user", "content": "I'm learning Rust"},
  {"role": "assistant", "content": "Nice — Rust rewards careful thinking."}
]
```

## Direct CLI

Use the official CLI directly when you need lower-level control:

```bash
honcho doctor --json
honcho message create "memory" -p "$ZO_USER" -s chat-1 --json
honcho session context chat-1 --json
honcho peer chat "what does the user prefer?" -p "$ZO_USER" --json
```
