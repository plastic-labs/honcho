---
name: honcho-setup
description: First-run setup and health check for Honcho memory in Cursor or Grok Bot. Use when the user asks to set up, connect, sign in to, or check the status of Honcho; when Honcho tool calls return authentication errors; or when the workspace metadata has no user_peer yet.
user-invocable: true
---

# Honcho setup

Honcho connects as an MCP server over HTTP with OAuth. There is no API key to
paste and no local process to start. Setup is: sign in, pick where memory
lives, record that choice in Honcho so every tool and Bot on the account finds
it.

## Step 1: Sign in

- **Cursor:** open Customize, find the Honcho MCP server, and choose
  Authenticate (or Reconnect). A browser window completes the sign-in.
- **Grok Bot:** tell the user "I need to sign in to Honcho" and follow the
  connect card the Bot shows. Ask the user to complete it in their browser.

Verify by calling `list_workspaces`. An authentication error means the sign-in
did not complete; do not retry other tools until it does.

## Step 2: Choose a workspace

Memory is scoped to a workspace. Ask the user one question:

> Keep Cursor's memory on its own, or share it with another Honcho tool?

- **On its own:** use the workspace `cursor`. Create it implicitly by calling
  `get_metadata` with `workspace_id: "cursor"`.
- **Shared:** show the user the result of `list_workspaces` and let them pick
  the workspace their other tool uses (for Claude Code the default is
  `claude_code`). Then store the pointer so future conversations find it:
  `set_metadata` with `workspace_id: "cursor"` and
  `{"default_workspace": "<chosen>"}`.

## Step 3: Choose the user's peer

Call `get_metadata` on the chosen workspace. If it already has `user_peer`,
confirm it with the user and stop. Otherwise ask what Honcho should call them
(a short handle, no spaces; in a shared workspace it must match the peer their
other tool uses), then `set_metadata` on that workspace with
`{"user_peer": "<handle>"}`.

## Step 4: Confirm

Report, in a few lines:

- the workspace in use, and whether it is shared;
- the user peer, and your own peer (`cursor`, or your Bot name);
- what `get_peer_context` returns for the user peer right now (a sentence or
  "nothing yet, memory starts with this conversation").

Then record this conversation as the first entry, following the
`honcho-memory` skill.

## Status check

When asked for Honcho's status, run Step 4 without changing anything, and add
how many sessions `search` finds for the current session name. If the
workspace metadata has no `user_peer`, say setup is incomplete and offer to
finish it.
