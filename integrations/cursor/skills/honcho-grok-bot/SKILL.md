---
name: honcho-grok-bot
description: Conventions for using Honcho memory as a Grok Bot. Apply whenever you are a Grok Bot (an xAI or Cursor Bot with its own cloud computer) and Honcho is connected, together with the honcho-memory skill. Covers how each Bot names its peer and sessions, how several Bots on one account share the user's memory, group chats, routines, and per-turn logging without hooks.
---

# Honcho for Grok Bots

Grok Bot loads only two parts of this plugin: the Honcho MCP server and the
skills. There are no lifecycle hooks, so nothing is recorded unless you record
it. Apply the `honcho-memory` skill for the loop and use the conventions below
so that every Bot on the account writes to Honcho the same way.

## Identity

- **The user** is the account owner. Use the `user_peer` stored in the
  workspace metadata, exactly as the `honcho-memory` skill describes. Every Bot
  on the account shares that one peer; that is what makes the user's memory
  continuous across Bots.
- **You** are one peer per Bot: your Bot name in kebab-case (`nori`,
  `research-bot`). Do not create a peer per conversation and do not reuse
  another Bot's peer. If your name changes, keep the old peer id and store the
  display name with `set_metadata` on your peer.
- **Other Bots** in a group chat are their own peers, named the same way.

## Sessions

Pick the first that applies:

1. **A conversation with a stable id you can read**: `<bot>:conv:<id>`.
2. **A direct chat without an id**: one session per Bot per day,
   `<bot>:chat:<YYYY-MM-DD>` in the user's time zone.
3. **A group chat**: `group:<group-name>:<YYYY-MM-DD>`, with the user and
   every participating Bot added as peers. Record each Bot's messages under
   that Bot's own peer.
4. **A routine run** (scheduled or event-triggered work with no user in the
   loop): `<bot>:routine:<routine-name>:<YYYY-MM-DD>`. Record what you did as
   your own messages. Do not record user messages you did not receive.

Keep a context's turns in one session. Do not start a new session mid-day
because the topic changed; Honcho reasons over a session as a whole and
summaries follow sessions.

## Logging each turn

At the end of every turn, before you yield, append the user's message and your
reply with `add_messages_to_session`. If a turn produced several of your
messages, send them in one call. Follow the recording rules in
`honcho-memory`: verbatim user text, prose replies, secrets replaced with
`[redacted]`, tool activity summarized in one line at most.

If a conversation produced no new turns since the last record, record nothing.

## Setup across Bots

The plugin is installed once for the account and is available to every Bot.
The first Bot the user sets up runs the `honcho-setup` skill and writes the
workspace metadata (`user_peer`, and `default_workspace` on the `cursor`
workspace if the user shares memory with other tools). Any later Bot reads
that metadata on its first conversation and needs nothing from the user. If
the metadata is missing, run `honcho-setup` rather than guessing.

Your own memory is a fine place to cache the resolved workspace and peer ids
so you do not re-read metadata every turn. Honcho's metadata remains the
source of truth.

## Signing in

Honcho uses OAuth. If a tool call returns an authentication error, tell the
user "I need to sign in to Honcho" and follow the connect card; do not ask for
an API key. Never store a Honcho key in your memory, in a file, or in Honcho.

## What not to do

- Do not log raw tool output, downloaded pages, or file contents.
- Do not create a workspace per Bot. Bots share the user's workspace.
- Do not record a turn twice. If you are unsure whether the last turn was
  recorded, `search` the session for its opening words first.
