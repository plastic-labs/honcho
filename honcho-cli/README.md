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
honcho init        # interactive wizard: API key, workspace, default peer
honcho doctor      # verify your config + connectivity
honcho             # show banner + command list
```

`honcho init` walks you through picking a workspace and default peer from your available choices, tests the connection, and writes config to `~/.honcho/config.toml`.

## Commands

### Onboarding

| Command | Description |
|---------|-------------|
| `honcho init` | Interactive setup wizard (or `--yes` for non-interactive) |
| `honcho doctor` | Health check: config, connectivity, workspace, peer, queue |

### Workspaces

| Command | Description |
|---------|-------------|
| `honcho workspace list` | List accessible workspaces |
| `honcho workspace inspect` | Peers, sessions, config for a workspace |
| `honcho workspace search <query>` | Search messages across workspace |
| `honcho workspace queue-status` | Deriver queue processing status |
| `honcho workspace delete <id>` | Delete a workspace (`--dry-run` first) |

### Peers

| Command | Description |
|---------|-------------|
| `honcho peer list` | List peers in the workspace |
| `honcho peer create <id>` | Create or get a peer |
| `honcho peer inspect <id>` | Card, session count, recent conclusions |
| `honcho peer card <id>` | Raw peer card content |
| `honcho peer chat <id> <query>` | Query the dialectic about a peer |
| `honcho peer representation <id>` | Formatted representation |
| `honcho peer search <id> <query>` | Search a peer's messages |
| `honcho peer get-metadata <id>` / `set-metadata` | Metadata operations |

### Sessions

| Command | Description |
|---------|-------------|
| `honcho session list` | List sessions in the workspace |
| `honcho session inspect <id>` | Peers, message count, summaries, config |
| `honcho session messages <id>` | Recent messages |
| `honcho session context <id>` | What an agent would see |
| `honcho session summaries <id>` | Short + long summaries |
| `honcho session peers <id>` / `add-peers` / `remove-peers` | Peer management |
| `honcho session search <id> <query>` | Search messages in a session |
| `honcho session representation <id>` | Peer representation in a session |
| `honcho session delete <id>` | Destructive; requires `--yes` |

### Messages

| Command | Description |
|---------|-------------|
| `honcho message list` | List messages in a session |
| `honcho message get <id>` | Get a single message |

### Conclusions (observations)

| Command | Description |
|---------|-------------|
| `honcho conclusion list` | List conclusions |
| `honcho conclusion search <query>` | Semantic search |
| `honcho conclusion create` | Create a conclusion |
| `honcho conclusion delete <id>` | Delete a conclusion |

### Keys

| Command | Description |
|---------|-------------|
| `honcho key generate` | Generate a scoped JWT (workspace/peer/session) |

### Config & schema

| Command | Description |
|---------|-------------|
| `honcho config show` | Show current config (API key redacted) |
| `honcho config set <key> <value>` | Set a single config value |
| `honcho describe resource <name>` | Schema introspection from live server |

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
# Full flags
honcho init --yes --api-key $HONCHO_API_KEY --workspace my-ws --peer my-peer

# Or rely on existing config / env vars to fill in missing values
HONCHO_API_KEY=xxx honcho init --yes --workspace my-ws

# If config already exists, this just validates and exits 0
honcho init --yes
```

Schema discovery for agents that need to build requests dynamically:

```bash
honcho describe resource peer --json
honcho describe resource session --json
```

## Context Threading

Set defaults once, then skip IDs on subsequent commands:

```bash
honcho config set peer_id peer_abc123
honcho peer inspect              # uses default
honcho peer card                 # uses default
honcho peer inspect other_id     # positional arg overrides
```

Or override per invocation:

```bash
honcho --workspace prod --peer ajspig peer card
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HONCHO_BASE_URL` | API base URL (default `https://api.honcho.dev`) |
| `HONCHO_API_KEY` | Admin JWT |
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

Config lives at `~/.honcho/config.toml` with this precedence (highest first):

1. CLI flags (`--workspace`, `--peer`, ...)
2. Environment variables (`HONCHO_*`)
3. Config file
4. Defaults

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
