```
██╗  ██╗ ██████╗ ███╗   ██╗ ██████╗██╗  ██╗ ██████╗
██║  ██║██╔═══██╗████╗  ██║██╔════╝██║  ██║██╔═══██╗
███████║██║   ██║██╔██╗ ██║██║     ███████║██║   ██║
██╔══██║██║   ██║██║╚██╗██║██║     ██╔══██║██║   ██║
██║  ██║╚██████╔╝██║ ╚████║╚██████╗██║  ██║╚██████╔╝
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝
```

# honcho-cli

A terminal for [Honcho](https://honcho.dev) — memory that reasons.

## Install

As a standalone tool (recommended):

```bash
uv tool install honcho-cli
```

## Quick Start

```bash
honcho init        # confirm/set apiKey + Honcho URL in ~/.honcho/config.json
honcho doctor      # verify your config + connectivity
honcho             # show banner + command list
```

`honcho init` reads `apiKey` and `environmentUrl` from the top-level of `~/.honcho/config.json` (the same file other Honcho tools — plugins, host integrations — share). If both are present, it confirms them with you; if either is missing (or you decline), it prompts for the missing value(s) and writes them back. Host-specific entries under `hosts` are left untouched.

Per-command scoping (workspace / peer / session) is handled via `-w` / `-p` / `-s` flags or `HONCHO_*` env vars — not persisted as CLI defaults.

## Commands

### Onboarding

| Command | Description |
|---------|-------------|
| `honcho init` | Confirm/set `apiKey` + `environmentUrl` in `~/.honcho/config.json` |
| `honcho doctor` | Health check: config, connectivity, workspace, peer, queue |

### Workspaces

| Command | Description |
|---------|-------------|
| `honcho workspace list` | List accessible workspaces |
| `honcho workspace create <id>` | Create or get a workspace |
| `honcho workspace inspect` | Peers, sessions, config for a workspace |
| `honcho workspace search <query>` | Search messages across workspace |
| `honcho workspace queue-status` | Deriver queue status (filter with `--observer` / `--sender`) |
| `honcho workspace delete <id>` | Delete a workspace. Use `--dry-run` to preview, `--cascade` to also delete sessions, `--yes` to skip the confirm prompt |

### Peers

| Command | Description |
|---------|-------------|
| `honcho peer list` | List peers in the workspace |
| `honcho peer create <id>` | Create or get a peer |
| `honcho peer inspect <id>` | Card, session count, recent conclusions |
| `honcho peer card <id>` | Raw peer card content |
| `honcho peer chat <query>` | Query the dialectic about a peer (peer via `-p` / `HONCHO_PEER_ID`) |
| `honcho peer representation <id>` | Formatted representation |
| `honcho peer search <query>` | Search a peer's messages (peer via `-p` / `HONCHO_PEER_ID`) |
| `honcho peer get-metadata <id>` / `set-metadata` | Metadata operations |

### Sessions

| Command | Description |
|---------|-------------|
| `honcho session list` | List sessions in the workspace (filter with `--peer/-p`) |
| `honcho session create <id>` | Create or get a session (optionally `--peers` to add peers, `--metadata`) |
| `honcho session inspect <id>` | Peers, message count, summaries, config |
| `honcho session context <id>` | What an agent would see |
| `honcho session summaries <id>` | Short + long summaries |
| `honcho session peers <id>` / `add-peers` / `remove-peers` | Peer management |
| `honcho session search <id> <query>` | Search messages in a session |
| `honcho session representation <id>` | Peer representation in a session |
| `honcho session get-metadata <id>` / `set-metadata` | Metadata operations |
| `honcho session delete <id>` | Destructive; requires `--yes` |

### Messages

| Command | Description |
|---------|-------------|
| `honcho message list` | List messages in a session (session via `-s` / `HONCHO_SESSION_ID`) |
| `honcho message create <content>` | Create a message (requires `--peer/-p`, session via `-s`) |
| `honcho message get <id>` | Get a single message (session via `-s` / `HONCHO_SESSION_ID`) |

### Conclusions (observations)

| Command | Description |
|---------|-------------|
| `honcho conclusion list` | List conclusions (filter with `--observer` / `--observed`) |
| `honcho conclusion search <query>` | Semantic search (filter with `--observer` / `--observed`) |
| `honcho conclusion create` | Create a conclusion |
| `honcho conclusion delete <id>` | Delete a conclusion |

### Config

| Command | Description |
|---------|-------------|
| `honcho config` | Show current config (API key redacted) |

## Agent Usage

All commands output JSON when stdout isn't a TTY, or when `--json` is forced.
Collection commands emit JSON arrays, and single-resource commands emit JSON objects:

```bash
honcho peer list --json
honcho workspace inspect --json | jq '.peers'
honcho doctor --json              # machine-parseable health checklist
```

Errors are structured:

```json
{
  "error": {
    "code": "PEER_NOT_FOUND",
    "message": "Peer 'abc' not found in workspace 'my-ws'",
    "details": {"workspace_id": "my-ws", "peer_id": "abc"}
  }
}
```

Non-interactive onboarding:

```bash
# Pre-seed via flags / env vars; init still prompts for anything missing
HONCHO_API_KEY=hch-v3-xxx honcho init --base-url https://api.honcho.dev
```

## Agent skill

`honcho-cli` ships with a skill that teaches agents the right commands and conventions for inspecting and debugging a Honcho deployment. Install it anywhere skills are accepted:

```bash
npx skills install honcho-cli
```

## Environment Variables

All `HONCHO_*` env vars work at runtime — no config file required.

Precedence (highest first): **flag → env var → config file → default**.

| Variable | Flag | Description |
|----------|------|-------------|
| `HONCHO_API_KEY` | `--api-key` (init) | Admin JWT |
| `HONCHO_BASE_URL` | `--base-url` (init) | API URL |
| `HONCHO_WORKSPACE_ID` | `-w` / `--workspace` | Workspace scope |
| `HONCHO_PEER_ID` | `-p` / `--peer` | Peer scope |
| `HONCHO_SESSION_ID` | `-s` / `--session` | Session scope |
| `HONCHO_JSON` | `--json` | Force JSON output (`1` / `true`) |

```bash
# Per-command flags
honcho peer card -w prod -p user

# Or export once per shell
export HONCHO_WORKSPACE_ID=prod
export HONCHO_PEER_ID=user
honcho peer card

# One-off against a different server
HONCHO_BASE_URL=http://localhost:8000 honcho workspace list

# CI/CD — env vars only, no config file needed
export HONCHO_API_KEY=hch-v3-xxx
export HONCHO_BASE_URL=https://api.honcho.dev
honcho workspace list
```

## Configuration

The CLI shares `~/.honcho/config.json` with sibling Honcho tools. It owns two
top-level keys: `apiKey` and `environmentUrl` (the full Honcho API URL, e.g.
`https://api.honcho.dev` or `http://localhost:8000`). Everything else at the
top level — `hosts`, `sessions`, `saveMessages`, `sessionStrategy`, etc. —
is left untouched.

```json
{
  "apiKey": "hch-v3-...",
  "environmentUrl": "https://api.honcho.dev",
  "hosts": { "claude_code": { "...": "..." } }
}
```

`workspace_id` / `peer_id` / `session_id` are per-command only — never
persisted to the config file.

## Development

Install from source in editable mode so changes are picked up live:

```bash
git clone https://github.com/plastic-labs/honcho
cd honcho
uv tool install --force --editable --from ./honcho-cli honcho-cli
```

Re-run any time — changes to `honcho-cli/src/` are reflected immediately without reinstalling.

## License

MIT
