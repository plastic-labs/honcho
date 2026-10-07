# Honcho Plugin for Kilo Code

> Add AI-native memory to Kilo Code

This plugin gives Kilo Code long-term memory that survives context wipes, session restarts and fresh chats. Honcho remembers what you work on, your durable preferences and prior context across projects. The plugin runs inside `kilo serve`, so it works in the Kilo CLI, the VS Code extension and the JetBrains plugin.

## Quick Start

### Step 1: Get Your Honcho API Key

1. Go to **[app.honcho.dev](https://app.honcho.dev)**
2. Sign up or log in
3. Copy your API key

### Step 2: Install the Plugin

```bash
kilo plugin @honcho-ai/kilo-honcho --global
```

`kilo plugin` installs the package and adds it to the `plugin` list in your global Kilo config. To update an existing install, add `--force`. You can also install it from the Kilo Marketplace in VS Code.

If you edit the config by hand, add `"@honcho-ai/kilo-honcho"` to the `plugin` array in `~/.config/kilo/kilo.json`.

### Step 3: Run Setup in Kilo

1. Start Kilo
2. Run `/honcho:setup`
3. Keep the default `Honcho Cloud` option unless you want a self-hosted or local endpoint
4. Enter your Honcho API key
5. Enter your `peerName`
6. Run `/honcho:status` to check the runtime

## What You Get

- **Persistent memory.** Kilo keeps durable context across sessions.
- **Hook-driven recall.** Hooks add memory to the prompt and record significant tool activity, so recall works with any model.
- **Honcho memory skill.** A `honcho-memory` skill is installed into Kilo's skills directory, so the agent knows when to pull and save memory on its own.
- **Cloud or local.** Use Honcho Cloud, or point at a self-hosted or local Honcho instance.
- **Session mapping.** Sessions can be scoped per directory, repo, branch, chat instance or globally.
- **Peer modeling.** User and agent observation are configurable (`observationMode`, `agentObserveMe`).

## Configuration

Configuration lives in the shared file `~/.honcho/config.json`, which other Honcho plugins also read. Kilo-specific settings live under `hosts.kilo`.

```jsonc
{
  "apiKey": "hch-...",
  "peerName": "user",
  "baseUrl": "https://api.honcho.dev",
  "hosts": {
    "kilo": {
      "workspace": "kilo",
      "aiPeer": "kilo",
      "recallMode": "hybrid",
      "observationMode": "unified",
      "agentObserveMe": false,
      "sessionStrategy": "per-directory",
      "removeUserPrefix": true,
      "apiKey": "hch-..." // optional; overrides the root apiKey for Kilo
    }
  }
}
```

Set `KILO_HONCHO_CONFIG_PATH` to read and write a different file.

### Cloud vs Local

For Honcho Cloud, `apiKey` is required and `baseUrl` stays `https://api.honcho.dev`.

For self-hosted or local Honcho, point `baseUrl` at your deployment, for example `http://127.0.0.1:8000`. `apiKey` is needed only if that deployment requires authentication. If Kilo runs in Docker or on another machine, `localhost` may not be your machine; the `baseUrl` must be reachable from where `kilo serve` runs.

### Session Strategies

| Strategy | Behavior | Best for |
| --- | --- | --- |
| `per-directory` | One session per working directory | Default project memory |
| `per-repo` | One session per repository | Repos with several entry directories |
| `git-branch` | Session changes with the current branch | Branch-specific work |
| `per-session` | New session for each Kilo session id | Short-lived isolated work |
| `chat-instance` | Session follows the current chat instance | Very short-lived use |
| `global` | One session for everything | Shared memory across all work |

### Observation Mode

This decides which Honcho collection `honcho_chat`, `honcho_create_conclusion` and per-prompt recall use for the user.

| Mode | Collection | Best for |
| --- | --- | --- |
| `unified` (default for new configs) | The user's own collection (`observer=user`, `observed=user`) | Several agents sharing what they learn about you |
| `directional` | This agent's view of the user (`observer=aiPeer`, `observed=user`) | Memory kept separate per agent |

A config file that existed before the plugin first ran, for example one written by another Honcho plugin, keeps `directional` and the legacy `user-<peerName>` peer until you choose otherwise with `/honcho:setup` or `/honcho:config`.

### Agent self-observation

The agent peer is created with `observeMe: false`, so Honcho models you, not the assistant. Set `agentObserveMe: true` to give the agent its own representation.

## Operator Commands

| Command | Description |
| --- | --- |
| `/honcho:setup` | First-time setup for cloud or local Honcho |
| `/honcho:status` | Effective Honcho status for the current project |
| `/honcho:settings` | Effective config values and config paths |
| `/honcho:config` | Edit shared Honcho fields in `~/.honcho/config.json` |
| `/honcho:import` | Preview or import local Kilo session history into Honcho |

`/honcho:import` reads history through the Kilo SDK client, maps sessions with the same `sessionStrategy` as live capture, and uploads user and assistant text with the original timestamps. The first run is a dry run. Imported sessions are recorded in `~/.honcho/kilo-import-state.json` and skipped next time.

## Agent Tools

| Tool | Description |
| --- | --- |
| `honcho_setup` | Validate setup and persist shared credentials or endpoint settings |
| `honcho_status` | Show effective runtime status |
| `honcho_get_config` | Read effective and persisted settings |
| `honcho_set_config` | Update a persisted shared setting |
| `honcho_search` | Search Honcho messages in the current session |
| `honcho_chat` | Ask Honcho a question answered by reasoning over memory |
| `honcho_create_conclusion` | Save a durable fact about the user |

## Plugin Surfaces

| Purpose | Kilo hook |
| --- | --- |
| Record your prompt, fetch prompt-specific recall | `chat.message` |
| Memory instructions and a stable snapshot | `experimental.chat.system.transform` |
| Continuity block during compaction | `experimental.session.compacting` |
| Record significant tool activity | `tool.execute.after` |
| `HONCHO_*` variables for shell tools | `shell.env` |
| `honcho_*` tools | `tool` |
| Session start, assistant capture, cleanup | `event` |

The packaged `honcho-memory` skill is copied to `~/.config/kilo/skills/honcho-memory`, or `$KILO_CONFIG_DIR/skills/honcho-memory` when set.

## Development

```bash
bun install
bun run check
bun run test
```

To load a local build in Kilo, build it and point the `plugin` array at the server entry by file path. The exported plugin id makes a file-path entry valid:

```jsonc
{
  "plugin": ["file:///absolute/path/to/kilo-honcho/dist/server.js"]
}
```

Plugin logs appear in Kilo's log under `service=kilo-honcho`; run `kilo run --print-logs` to see them on stderr.
