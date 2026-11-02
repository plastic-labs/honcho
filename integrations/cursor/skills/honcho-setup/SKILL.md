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

First make sure the `cursor` workspace exists: call `create_workspace` with
`workspace_id: "cursor"`. It is get-or-create, so this is safe on every run.

`set_metadata` replaces a resource's whole metadata object. Every write below
is therefore read, merge, write: call `get_metadata`, change only the key in
question, and send the full object back. Never send a single key on its own.

- **On its own:** use the workspace `cursor`. If the `cursor` workspace's
  metadata has a `default_workspace` key from an earlier choice, remove that
  key and write the rest back, or `honcho-memory` will keep using the old
  shared workspace.
- **Shared:** show the user the result of `list_workspaces` and let them pick
  the workspace their other tool uses (for Claude Code the default is
  `claude_code`). Then store the pointer so future conversations find it:
  merge `{"default_workspace": "<chosen>"}` into the `cursor` workspace's
  metadata.

## Step 3: Choose the user's peer

Call `get_metadata` on the chosen workspace. If it already has `user_peer`,
confirm it with the user and stop. Otherwise ask what Honcho should call them
(a short handle, no spaces; in a shared workspace it must match the peer their
other tool uses), then merge `{"user_peer": "<handle>"}` into that
workspace's existing metadata and write it back.

## Step 4 (Cursor only): Line up automatic capture

In Cursor the plugin also ships hooks that record every turn automatically.
They need `~/.honcho/config.json`, the file `honcho init` and the other Honcho
plugins write, with an API key in it. Check whether it exists:

- **It exists:** make sure `peerName` matches the handle from Step 3 and that
  `hosts.cursor.workspace` matches the workspace from Step 2, editing the file
  if needed. Otherwise the hooks and the skills write to different places.
- **It does not exist:** tell the user that memory works now through the MCP
  tools, and that running `honcho init` turns on automatic capture so they
  never depend on the agent remembering to record. Do not ask for an API key
  in chat.

Grok Bot never runs hooks, so skip this step there.

## Step 5: Confirm

Report, in a few lines:

- the workspace in use, and whether it is shared;
- the user peer, and your own peer (`cursor`, or your Bot name);
- what `get_peer_context` returns for the user peer right now (a sentence or
  "nothing yet, memory starts with this conversation").

Then record this conversation as the first entry, following the
`honcho-memory` skill.

## Status check

When asked for Honcho's status, run Step 5 without changing anything, and add
how many sessions `search` finds for the current session name. If the
workspace metadata has no `user_peer`, say setup is incomplete and offer to
finish it.
