# Honcho Plugin for Cursor and Grok Bot

> Persistent memory for Cursor and Grok Bot, powered by Honcho

One plugin, one Cursor Marketplace listing, two hosts. Cursor installs it from
the marketplace; Grok Bot installs the same listing from its plugin page. The
plugin connects your agent to the hosted Honcho MCP server at
`https://mcp.honcho.dev` and ships the skills that drive the memory loop, so
the agent remembers who you are, what you prefer, and what you were working on
across sessions, projects, and Bots.

There is nothing to run locally for the core: no API key to paste and no local
server. Sign in with OAuth and start a conversation. Cursor users can add
automatic capture on top, which needs `bun` and a Honcho key in
`~/.honcho/config.json`; see below.

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
| Hooks `sessionStart`, `beforeSubmitPrompt`, `afterAgentResponse` | yes, with `honcho init` | no | Inject memory before the first prompt and record every turn without the agent's involvement |
| Command `/recall` | yes | no | Ask Honcho what it remembers about a topic |

Grok Bot loads only MCP servers and skills from a plugin, so the plugin is
designed to work from those two alone. The rule and command are conveniences
for the Cursor IDE.

## How memory is organized

| Honcho concept | Cursor | Grok Bot |
| --- | --- | --- |
| Workspace | `cursor`, or a shared workspace you pick during setup | same |
| User peer | one handle you choose once, stored in workspace metadata | same handle, shared by every Bot |
| Agent peer | `cursor` | the Bot's name, one peer per Bot |
| Session | one per project: `cursor-<owner>-<repo>` from the git remote, else `cursor-<folder>` | per conversation or per day per Bot; group chats and routines get their own |

The workspace and user peer are stored as workspace metadata in Honcho, so
every agent on the account resolves the same answer with two cheap reads. The
exact rules are in [`skills/honcho-memory/SKILL.md`](skills/honcho-memory/SKILL.md)
and [`skills/honcho-grok-bot/SKILL.md`](skills/honcho-grok-bot/SKILL.md).

To share memory with another Honcho integration (for example Claude Code),
pick that tool's workspace during `/honcho-setup` and use the same user
handle. See [Unified Memory Setup](https://honcho.dev/docs/v3/guides/recipes/unified-memory-setup).

## Automatic capture in Cursor

Grok Bot runs only MCP servers and skills, so the skills carry the memory
loop there. Cursor also runs lifecycle hooks, and the plugin ships three:

| Hook | What it does |
| --- | --- |
| `sessionStart` | Creates the project's session and peers, then injects the peer card, representation and last session summary into the conversation's initial context |
| `beforeSubmitPrompt` | Records the user's prompt verbatim, secrets redacted. Always lets the prompt through |
| `afterAgentResponse` | Records the agent's reply, with code blocks over twelve lines collapsed and secrets redacted |

A hook is a separate process and cannot see the OAuth token Cursor holds for
the MCP server, so the hooks read `~/.honcho/config.json` instead: the file
`honcho init`, claude-honcho and kilo-honcho already share. Without that
file, or with `enabled: false` in it, every hook exits silently and nothing
changes; the skills keep working through the MCP tools. With it, the
session-start context tells the agent that capture is automatic, so turns are
not recorded twice. The hooks run with `bun`, like Cursor's own plugins.

The host block for this plugin is `hosts.cursor` (`workspace`, `peerName`,
`auth.apiKey`); `HONCHO_*` environment variables override it. Set
`HONCHO_HOOK_DEBUG=1` to get a one-line log per hook on stderr.

## What leaves your machine

- Every request goes to `https://mcp.honcho.dev`, which calls
  `https://api.honcho.dev` on your behalf with your OAuth token.
- Recorded content: your messages, the agent's replies as prose, and
  conclusions about you. The skills tell the agent to replace anything that
  looks like a secret with `[redacted]` and never to record raw tool output or
  file contents.
- The MCP config sends an `X-Honcho-Plugin: cursor-honcho/<version>` header so
  usage is attributed to this plugin, and the hooks send the same plugin
  header plus `X-Honcho-Host: cursor/<version>`. No other identifying header
  is sent.

## Development

The hooks are TypeScript under `src/`, bundled into the committed
`dist/hook.js` so a marketplace clone needs no install step:

```bash
bun install
bun run check   # typecheck
bun run build   # rebuild dist/hook.js after editing src/
bun run test    # builds, then runs the hooks against an in-process mock API
```

The tests spawn `dist/hook.js` the way Cursor does, with JSON on stdin and a
scrubbed environment, so a real `HONCHO_API_KEY` in your shell is never used.

Before a release, walk the Cursor Marketplace
[submission checklist](https://cursor.com/docs/reference/plugins#submission-checklist):
manifest fields present, every path relative with no `..`, the MCP server an
`https://` URL with no stdio command, frontmatter on every skill and command,
the logo committed, and the root `.cursor-plugin/marketplace.json` pointing at
this directory.

To release: bump `version` in `.cursor-plugin/plugin.json` and the
`X-Honcho-Plugin` header in `mcp.json` together, run `bun run build` so the
hooks pick up the new version, add a changelog entry, and merge. The
marketplace serves the repo at `main`.

## Links

- [Cursor guide](https://honcho.dev/docs/v3/guides/integrations/cursor)
- [Grok Bot guide](https://honcho.dev/docs/v3/guides/integrations/grok-bot)
- [MCP server reference](https://honcho.dev/docs/v3/guides/integrations/mcp)
- [Honcho docs](https://honcho.dev/docs)
