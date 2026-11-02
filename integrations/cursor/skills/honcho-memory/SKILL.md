---
name: honcho-memory
description: Use Honcho as your persistent memory of the user. Apply at the start of every conversation (recall what Honcho knows), whenever personalization or past context would help, when the user asks what you remember or know about them, and after every exchange (record the turn). Works the same in Cursor and in Grok Bot.
---

# Honcho memory

Honcho is connected to you as an MCP server (`honcho`). You feed it the turns of
your conversations; in the background it reasons over them and builds a
representation of the user. You can read that representation at any time and ask
it questions in natural language.

Everything below works in both Cursor and Grok Bot. Nothing here depends on
hooks, rules or local files: the skill plus the MCP tools are the whole system.
If you are a Grok Bot, also apply the `honcho-grok-bot` skill, which sets the
peer and session names for Bots.

## The loop: recall, respond, record

1. **Resolve where memory lives** (once per conversation, see below).
2. **Recall before you respond.** Call `get_peer_context` for the user peer at
   the start of a conversation and read it as if it were a note from a
   colleague who knows the user. When a specific question would help
   ("how does this user like tests written?"), ask `chat`.
3. **Record after every exchange.** Append the user's message and your reply
   with `add_messages_to_session`. This is the only non-optional step: Honcho
   learns from nothing else. **Exception:** if the context injected at session
   start says automatic capture is on, the plugin's hooks are already recording
   prompts and replies for this conversation; skip this step and only store
   conclusions.

Reasoning happens asynchronously. Never wait or poll after recording.

## Where memory lives

Workspace, peers and sessions are explicit arguments on every tool call, so
resolve them once at the start of a conversation and reuse them. Every id may
contain only letters, digits, `_` and `-`; lowercase everything else and
replace other characters with `-`.

**Workspace.** Call `get_metadata` with `workspace_id: "cursor"` and no other
scope. If the result has a `default_workspace` key, that is the workspace to
use; otherwise use `cursor`. The `honcho-setup` skill writes that pointer when
the user chooses to share memory with another tool.

**User peer.** Call `get_metadata` on the chosen workspace. Use its `user_peer`
value. If there is none, ask the user once what Honcho should call them (a
short handle, no spaces), then store it on the workspace so every future
conversation, and every other Bot or editor on this account, finds the same
answer. `set_metadata` replaces the whole metadata object, so read the current
metadata first and write it back with `user_peer` added, never just the one
key. One stable peer id per
person: never invent a new one per conversation.

**Your peer.** In Cursor, `cursor`. In Grok Bot, your Bot's name in kebab-case
(the `honcho-grok-bot` skill has the details).

**Session.** In Cursor, one session per project. If the context injected at
session start names a session, use that name. Otherwise derive it: when the
project has a git `origin` remote, `cursor-<owner>-<repo>` from that URL (so
two clones of one repo share a session and two projects that happen to share
a folder name do not); when it has no remote, `cursor-<folder-name>` from the
workspace root. Lowercase and replace other characters with `-` (`My Project`
gives `cursor-my-project`). Keep using it across days; a project is one
coherent context. In Grok Bot, follow the `honcho-grok-bot` skill.

**First use of a session.** `create_session`, then `add_peers_to_session` with
two entries: the user peer with `observe_me: true, observe_others: false`, and
your peer with `observe_me: false, observe_others: false`. Honcho builds one
model, the user's. Creating a session or peer that already exists is harmless,
so you can do this whenever you are unsure.

## Recording well

- Record the user's words verbatim. Record your reply as prose; drop code
  blocks longer than a dozen lines and say what they contained.
- Replace anything that looks like a secret (API keys, tokens, passwords,
  connection strings) with `[redacted]` before sending.
- Do not record raw tool output or file contents the user did not write.
  Summarize tool activity in one line if it matters to the story.
- Batch when you can: one `add_messages_to_session` call with several
  messages beats several calls.

## Recalling well

Three reads, cheapest first:

- `get_peer_context` for the user peer: the synthesized picture of the user
  plus matching conclusions. Near-instant. Use at the start of every
  conversation and whenever the topic shifts.
- `search`: past messages matching a query, across the workspace or one
  session. Near-instant. Use for "what did we decide about X".
- `chat` with the user peer: a reasoned answer to a specific question. Takes
  seconds. Pass `reasoning_level: "low"` by default and raise it only for
  hard questions.

When the user states a durable fact about themselves ("I always use uv",
"never auto-commit") or asks you to remember something, store it right away
with `create_conclusions` so it is available before background reasoning
catches up. Set both `peer_id` and `target_peer_id` to the user peer: that pair
is the user's own memory, and with your peer as `peer_id` the fact lands where
recall never reads it. Pass the current `session_id`. Check with `search`
first to avoid duplicates, and confirm what you saved in one line.

## Rules of thumb

- The user should never have to repeat themselves. Check what you know before
  asking a preference question.
- Reference past context naturally ("last time you preferred …"), and say
  when you are unsure whether memory is current.
- Never store API keys or secrets in Honcho, in any field.
- If a tool call fails with an authentication error, stop calling tools and
  run the `honcho-setup` skill instead of retrying.
