# Honcho Plugin for Cursor and Grok Bot

> Persistent memory for Cursor and Grok Bot, powered by Honcho

One plugin, one Cursor Marketplace listing, two hosts. Cursor installs it from
the marketplace; Grok Bot installs the same listing from its plugin page. The
plugin connects your agent to the hosted Honcho MCP server at
`https://mcp.honcho.dev` and ships the skills that drive the memory loop, so
the agent remembers who you are, what you prefer, and what you were working on
across sessions, projects, and Bots.

There is nothing to run locally: no API key to paste, no local server, no
`bun`. Sign in with OAuth and start a conversation.

## Install

### Cursor

1. Open **Customize** in the sidebar, find **Honcho** under Plugins, and choose
   **Install** (project or user scope).
2. In the same panel, open the **honcho** MCP server and choose
   **Authenticate**. A browser window completes the sign-in.
3. Start a new chat and run `/honcho-setup` once.

### Grok Bot

1. Open the Honcho plugin page (search **Honcho** under **Connect apps**, or
   use the plugin link from [honcho.dev](https://honcho.dev/docs/v3/guides/integrations/grok-bot))
   and choose **Add to Grok Bot**.
2. Tell your Bot: "Set up Honcho." It signs in through a connect card and runs
   the `honcho-setup` skill.

The install is account-wide: every Bot on the account can use it. Each Bot gets
its own peer in Honcho; you stay one peer across all of them.

### From a local checkout (development)

Copy or symlink this directory to `~/.cursor/plugins/local/honcho` and reload
Cursor. A marketplace install with the same name takes precedence over the
local copy.

## What you get

| Component | Cursor | Grok Bot | Purpose |
| --- | --- | --- | --- |
| MCP server `honcho` | yes | yes | The hosted Honcho server: `get_peer_context`, `chat`, `search`, `add_messages_to_session`, `create_conclusions`, and the rest of the Honcho API as tools |
| Skill `honcho-memory` | yes | yes | The recall, respond, record loop and the naming conventions |
| Skill `honcho-grok-bot` | yes | yes | Peer and session conventions for Bots, group chats and routines |
| Skill `honcho-setup` | yes | yes | Sign-in, workspace choice, first-run bootstrap, status |
| Rule `honcho-memory` | yes | no | Always-on reminder to recall and record |
| Commands `/recall`, `/remember` | yes | no | Quick memory search and quick save |

Grok Bot loads only MCP servers and skills from a plugin, so the plugin is
designed to work from those two alone. The rule and commands are conveniences
for the Cursor IDE.

## How memory is organized

| Honcho concept | Cursor | Grok Bot |
| --- | --- | --- |
| Workspace | `cursor`, or a shared workspace you pick during setup | same |
| User peer | one handle you choose once, stored in workspace metadata | same handle, shared by every Bot |
| Agent peer | `cursor` | the Bot's name, one peer per Bot |
| Session | one per project folder: `cursor:<folder>` | per conversation or per day per Bot; group chats and routines get their own |

The workspace and user peer are stored as workspace metadata in Honcho, so
every agent on the account resolves the same answer with two cheap reads. The
exact rules are in [`skills/honcho-memory/SKILL.md`](skills/honcho-memory/SKILL.md)
and [`skills/honcho-grok-bot/SKILL.md`](skills/honcho-grok-bot/SKILL.md).

To share memory with another Honcho integration (for example Claude Code),
pick that tool's workspace during `/honcho-setup` and use the same user
handle. See [Unified Memory Setup](https://honcho.dev/docs/v3/guides/recipes/unified-memory-setup).

## What leaves your machine

- Every request goes to `https://mcp.honcho.dev`, which calls
  `https://api.honcho.dev` on your behalf with your OAuth token.
- Recorded content: your messages, the agent's replies as prose, and
  conclusions about you. The skills tell the agent to replace anything that
  looks like a secret with `[redacted]` and never to record raw tool output or
  file contents.
- The MCP config sends an `X-Honcho-Plugin: cursor-honcho/<version>` header so
  usage is attributed to this plugin. No other identifying header is sent.

## Development

```bash
python3 cursor-honcho/scripts/validate.py
```

The validator mirrors the Cursor Marketplace submission checklist: manifest
fields, relative paths, an `https://` MCP server with no stdio command, skill
and command frontmatter, the logo, and the root
`.cursor-plugin/marketplace.json`. CI runs it on every change under this
directory.

To release: bump `version` in `.cursor-plugin/plugin.json` and the
`X-Honcho-Plugin` header in `mcp.json` together (the validator checks they
match), add a changelog entry, and merge. The marketplace serves the repo at
`main`.

## Links

- [Cursor guide](https://honcho.dev/docs/v3/guides/integrations/cursor)
- [Grok Bot guide](https://honcho.dev/docs/v3/guides/integrations/grok-bot)
- [MCP server reference](https://honcho.dev/docs/v3/guides/integrations/mcp)
- [Honcho docs](https://honcho.dev/docs)
