```
██╗  ██╗ ██████╗ ███╗   ██╗ ██████╗██╗  ██╗ ██████╗
██║  ██║██╔═══██╗████╗  ██║██╔════╝██║  ██║██╔═══██╗
███████║██║   ██║██╔██╗ ██║██║     ███████║██║   ██║
██╔══██║██║   ██║██║╚██╗██║██║     ██╔══██║██║   ██║
██║  ██║╚██████╔╝██║ ╚████║╚██████╗██║  ██║╚██████╔╝
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝
```

# honcho-ai-cli

A terminal for [Honcho](https://honcho.dev) — memory that reasons.

## Install

As a standalone tool (recommended):

```bash
uv tool install honcho-ai-cli
```

As an extra on the Honcho SDK (if you want both the SDK and the CLI in one project):

```bash
uv add honcho-ai[cli]
# or
pip install honcho-ai[cli]
```

Either way, you'll get the `honcho` command on your PATH.

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
| `honcho config show` | Show current config (API key redacted) |

## Agent Usage

All commands output JSON when stdout isn't a TTY, or when `--json` is forced:

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
# Pre-seed via flags / env vars; `honcho init` still prompts for anything missing
HONCHO_API_KEY=xxx honcho init --base-url local
```

## Context Threading

Workspace / peer / session come from flags or env vars — not persisted defaults:

```bash
# Per-command flags
honcho --workspace prod --peer ajspig peer card

# Or export once per shell
export HONCHO_WORKSPACE_ID=prod
export HONCHO_PEER_ID=ajspig
honcho peer card
honcho peer inspect other_id     # positional arg still takes precedence
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HONCHO_BASE_URL` | API base URL pre-fill for `honcho init` only — ignored at runtime |
| `HONCHO_API_KEY` | Admin JWT pre-fill for `honcho init` only — ignored at runtime |
| `HONCHO_WORKSPACE_ID` | Default workspace |
| `HONCHO_PEER_ID` | Default peer |
| `HONCHO_SESSION_ID` | Default session |
| `HONCHO_JSON` | Force JSON output (`1` / `true`) |

## Global Flags

| Flag | Description |
|------|-------------|
| `--json` | Force JSON output |
| `--quiet` / `-q` | Suppress status messages |
| `--workspace` / `-w` | Override workspace ID |
| `--peer` / `-p` | Override peer ID |
| `--session` / `-s` | Override session ID |
| `--version` / `-V` | Show version |

## Configuration

The CLI shares `~/.honcho/config.json` with sibling Honcho tools. It owns two
top-level keys: `apiKey` and `environmentUrl` (the full Honcho API URL, e.g.
`https://api.honcho.dev` or `http://localhost:8000`). Everything else at the
top level — `hosts`, `sessions`, `saveMessages`, `sessionStrategy`, etc. —
is left untouched.

Example:

```json
{
  "apiKey": "hch-v3-...",
  "environmentUrl": "https://api.honcho.dev",
  "hosts": { "claude_code": { "...": "..." } }
}
```

Precedence (highest first):

- **`apiKey`** and **`base_url`**: read only from `~/.honcho/config.json` at
  runtime. No env-var or flag fallback — a missing config file is a hard
  error. (`honcho init` still accepts `--api-key` / `HONCHO_API_KEY` and
  `--base-url` / `HONCHO_BASE_URL` as one-time pre-fills for the
  write-to-file prompts.) This keeps a single, inspectable source of truth
  for where you're connecting and with what credentials.
- **`workspace_id` / `peer_id` / `session_id`**: flag (`-w` / `-p` / `-s`)
  → env var (`HONCHO_WORKSPACE_ID` etc.). Not persisted to the config file.

## Development

Install from source in editable mode so changes are picked up live:

```bash
git clone https://github.com/plastic-labs/honcho
cd honcho
uv tool install --force --editable --from ./honcho-cli honcho-ai-cli
```

Re-run any time — changes to `honcho-cli/src/` are reflected immediately without reinstalling.

## License

MIT
