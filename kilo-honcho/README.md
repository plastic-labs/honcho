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

`kilo plugin` installs the package and adds it to the `plugin` list of Kilo's global config (the server half) and `~/.config/kilo/tui.json` (the `/honcho:*` commands). It still uses Kilo's older `opencode.json` name for the server entry; Kilo reads `kilo.jsonc`, `kilo.json` and `opencode.json` alike. To update an existing install, add `--force`. To add it by hand, put `"@honcho-ai/kilo-honcho"` in the `plugin` array of `~/.config/kilo/kilo.jsonc` and `~/.config/kilo/tui.json`.

For the Kilo desktop app or the VS Code and JetBrains extensions, run this in any terminal instead:

```bash
npx @honcho-ai/kilo-honcho setup
```

It asks for your API key without echoing it, the name Honcho should call you, and a workspace. It saves them to `~/.honcho/config.json` with mode 600 and adds the plugin to `~/.config/kilo/kilo.jsonc` (or `kilo.json`) and `~/.config/kilo/tui.json`, keeping any comments. Restart Kilo to load it.

If you edit the config by hand, add `"@honcho-ai/kilo-honcho"` to the `plugin` array in both files.

### Step 3: Run Setup in Kilo

1. Start Kilo. The first time the plugin loads without an API key, it offers to set up Honcho.
2. Choose **Set up now**, or run `/honcho:setup`
3. Keep the default `Honcho Cloud` option unless you want a self-hosted or local endpoint
4. Enter your Honcho API key
5. Confirm the name Honcho should call you. Setup fills in the `peerName` your other Honcho tools use, or your OS user name.
6. Choose a workspace: Kilo's own, or one another Honcho tool already uses
7. Run `/honcho:status` to check the runtime

## What You Get

- **Persistent memory.** Kilo keeps durable context across sessions.
- **Hook-driven recall.** Hooks add memory to the prompt and record significant tool activity, so recall works with any model.
- **Honcho memory skill.** A `honcho-memory` skill is installed into Kilo's skills directory, so the agent knows when to pull and save memory on its own.
- **Cloud or local.** Use Honcho Cloud, or point at a self-hosted or local Honcho instance.
- **Session mapping.** Sessions can be scoped per directory, repo, branch, chat instance or globally.
- **Peer modeling.** User and agent observation are configurable (`observationMode`, `agentObserveMe`).
- **Visible recall.** A Honcho section in Kilo's sidebar shows what Honcho added to the session, and `/honcho:recall` shows the exact text.

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
      "autoConclusions": false,
      "sessionStrategy": "per-directory",
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
| `unified` (default) | The user's own collection (`observer=user`, `observed=user`) | Several agents sharing what they learn about you |
| `directional` | This agent's view of the user (`observer=aiPeer`, `observed=user`) | Memory kept separate per agent |

In `unified` mode the agent peer does not observe the user, so Honcho derives each message once. If you switch to `directional`, the agent's view of you starts from the messages saved after the switch.

### Peer name

Your peer is your `peerName`, unchanged, so Kilo uses the same peer as your other Honcho tools. Kilo reads `HONCHO_PEER_NAME`, then `peerName` in `~/.honcho/config.json`, then your OS user name (`$USER`). Characters Honcho does not allow in ids become `-`. The plugin writes `~/.honcho/config.json` only when you run setup or change a setting.

A peer belongs to one workspace. Kilo uses its own `kilo` workspace by default, so it starts with no memory of you. To share memory with another Honcho tool, choose that tool's workspace in `/honcho:setup`, or set `hosts.kilo.workspace` to it. This changes nothing for the other tool.

`peerName` and `aiPeer` must differ. If they match, Honcho calls fail with an error in the sidebar until you change one.

### Agent self-observation

The agent peer is created with `observeMe: false`, so Honcho models you, not the assistant. Set `agentObserveMe: true` to give the agent its own representation.

### Auto conclusions

Honcho derives conclusions from every saved message. With `autoConclusions: true`, the plugin also saves a prompt verbatim as a conclusion when it contains phrases such as "I prefer", "always", "never" or "remember that". The conclusion is available on the next prompt, before Honcho's deriver runs. It is off by default because it also saves prompts like "never mind, revert that".

## Kilo CLI, Desktop App and IDE Extensions

Every Kilo client starts its own local Kilo server, and the plugin runs inside it. Recall, saving and the `honcho_*` tools work in all of them. The sidebar section and the `/honcho:*` commands come from the terminal UI half, which only the Kilo CLI loads.

| | Kilo CLI | Desktop app, VS Code, JetBrains |
| --- | --- | --- |
| Setup | `/honcho:setup`, or **Set up now** on first load | Ask the agent ("Set up Honcho memory for me"), or `npx @honcho-ai/kilo-honcho setup` |
| Honcho sidebar section and `/honcho:*` commands | Yes | No |
| Check that Honcho is working | Sidebar, `/honcho:status`, `/honcho:recall` | Ask "Show my Honcho status" |

Outside the CLI:

- The desktop app and editors opened from the Dock or an app launcher do not see shell-profile variables such as `HONCHO_API_KEY`. Save the key with setup.
- Each desktop chat runs in its own folder, so each chat is its own Honcho session. Memory of you carries across chats; `honcho_search` only searches the current session.
- The desktop app and the extensions bundle their own Kilo version, and read the same `~/.config/kilo` and `~/.honcho/config.json` as the CLI.

## Seeing What Honcho Did

Kilo's sidebar shows `Memory • Disabled` when Kilo's own memory feature is off. The plugin adds a Honcho section directly below it:

```
Honcho
• Active
Peer: alice
Session: View in Honcho ↗
/honcho:recall
```

On Honcho Cloud, clicking `View in Honcho` opens the session in the Honcho dashboard, where you can read the saved messages and the conclusions. Any other Honcho server has no known dashboard address, so the sidebar shows the session name, shortened in the middle to fit. Clicking `/honcho:recall` opens the recall view. The section shows `Not set up` before `/honcho:setup`, and `Error` with the reason when a Honcho request fails.

`/honcho:recall` opens the exact text the model received. Kilo does not save this text in its session history.

The server half writes this record to `~/.honcho/kilo/sessions/<session id>.json` with mode 600, and the TUI reads it. The sidebar and the `/honcho:*` commands belong to the Kilo CLI's terminal UI. The desktop app and the VS Code and JetBrains extensions do not load TUI plugins, so they show neither. When Honcho has no API key, the agent tells the user once per session to run `npx @honcho-ai/kilo-honcho setup`.

## Operator Commands

| Command | Description |
| --- | --- |
| `/honcho:setup` | First-time setup for cloud or local Honcho |
| `/honcho:status` | Effective Honcho status for the current project |
| `/honcho:recall` | The exact memory text Honcho added to the current session |
| `/honcho:settings` | Effective config values and config paths |
| `/honcho:config` | Edit shared Honcho fields in `~/.honcho/config.json` |
| `/honcho:import` | Preview or import local Kilo session history into Honcho |

`/honcho:import` reads history through the Kilo SDK client, maps sessions with the same `sessionStrategy` as live capture, and uploads user and assistant text with the original timestamps. The first run is a dry run. Imported sessions are recorded in `~/.honcho/kilo-import-state.json` and skipped next time.

## Agent Tools

| Tool | Description |
| --- | --- |
| `honcho_setup` | Save the peer name and workspace, and ask for the API key in a window on the user's screen. A key never passes through the chat, and the saved key is never sent to a new server. |
| `honcho_status` | Show effective runtime status |
| `honcho_get_config` | Read effective and persisted settings |
| `honcho_set_config` | Update a persisted shared setting. `apiKey` and `baseUrl` can only be changed by the user. |
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
| `HONCHO_URL` and `HONCHO_WORKSPACE_ID` for shell tools (never the API key) | `shell.env` |
| `honcho_*` tools | `tool` |
| Session start, assistant capture, cleanup | `event` |
| Honcho sidebar section (TUI) | `sidebar_content` slot |

The packaged `honcho-memory` skill is copied to `~/.config/kilo/skills/honcho-memory`, or `$KILO_CONFIG_DIR/skills/honcho-memory` when set.

## Remote and Headless Use

The plugin runs wherever Kilo's server runs, and reads `~/.honcho/config.json` on that machine.

- **VS Code Remote SSH, dev containers, Codespaces.** Kilo's server runs on the remote machine, so it reads the remote's `~/.honcho/config.json`, not your laptop's. Run `npx @honcho-ai/kilo-honcho setup` in the remote terminal.
- **Key window over SSH.** On macOS and Windows, `honcho_setup` does not open the key window in an SSH session, because it would appear on that machine's own screen. It points the user at the setup command instead. On Linux it opens only when a display is available, including through X forwarding. An unanswered window closes after 2 minutes.
- **CI and scripts.** When the `CI` variable is set, the agent does not offer setup. Set `HONCHO_API_KEY` to use Honcho in CI, or `KILO_PURE=1` to turn off every plugin.

## Development

```bash
bun install
bun run check
bun run test
```

`bun run build` compiles the sidebar's JSX with `@opentui/solid`'s Bun plugin. `solid-js` and `@opentui/*` stay external: Kilo rewrites those imports to its own copies when it loads the plugin, so the sidebar shares Kilo's Solid runtime.

To load a local build in Kilo, build it and point the `plugin` array at the server entry by file path. The exported plugin id makes a file-path entry valid:

```jsonc
{
  "plugin": ["file:///absolute/path/to/kilo-honcho/dist/server.js"]
}
```

Plugin logs appear in Kilo's log under `service=kilo-honcho`; run `kilo run --print-logs` to see them on stderr.
