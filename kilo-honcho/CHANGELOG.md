# Changelog

All notable changes to this package are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package uses [Semantic Versioning](https://semver.org/).

This package versions independently of the Honcho API, `@honcho-ai/sdk`, and other host plugins.

## [Unreleased]

## [0.1.0] - Unreleased

### Added

- First release of the Kilo Code plugin, forked from `@honcho-ai/opencode-honcho` 0.2.1. Kilo's plugin API matches OpenCode 1.x, so the hooks, tools, TUI commands and transcript import carry over.
- Kilo defaults: the `hosts.kilo` config block, workspace and agent peer `kilo`, telemetry headers `X-Honcho-Host: kilo` and `X-Honcho-Plugin: kilo-honcho`, and the `KILO_HONCHO_CONFIG_PATH` override.
- The `honcho-memory` skill installs to `~/.config/kilo/skills`, or `$KILO_CONFIG_DIR/skills`, where Kilo looks for skills.
- `.kilo` and `.kilocode` count as project-root markers alongside `.git`.
- A Honcho section in Kilo's session sidebar, below Kilo's own Memory section. It shows whether Honcho is active, the user peer, and the Honcho session. On Honcho Cloud the session is a link to the dashboard; elsewhere it is the session name, shortened in the middle.
- `/honcho:recall` shows the exact memory text Honcho added to the current session's prompt and system prompt.
- The server writes this per-session record to `~/.honcho/kilo/sessions/<session id>.json` (mode 600) for the TUI to read. Files older than 14 days are removed at startup, and a file is removed when Kilo deletes its session.
- `hosts.kilo.autoConclusions` turns keyword auto-conclusions back on.
- `/honcho:setup` fills in the peer name other Honcho tools already use, and offers to share a workspace another Honcho tool uses. `honcho_setup` takes a `workspace` argument for the same choice.
- `npx @honcho-ai/kilo-honcho setup` sets up Honcho from any terminal for the desktop app and IDE extensions, which have no `/honcho:setup`. It adds the plugin to `~/.config/kilo/kilo.jsonc` (or `kilo.json`) and `tui.json`, and skips a config that already loads it.
- The TUI offers setup the first time the plugin loads without an API key. Without a key, the system prompt asks the agent to point the user at the setup command once per session.
- The sidebar ends with `/honcho:recall`, which opens the recall view when clicked.
- `environmentUrl`, which `honcho init` writes, is used when `baseUrl` is missing.
- `honcho_setup` asks for the API key in the operating system's own password window, so the agent can finish setup without the key passing through the chat. The window does not open over SSH on macOS or Windows, and closes after 2 minutes.
- The setup command takes `--peer-name`, `--workspace`, `--url` and `--cloud`, and opens the same window when it has no terminal.
- The agent does not offer setup when the `CI` variable is set.

### Changed

- The user peer is `peerName` unchanged, as in Claude Code and Hermes. opencode-honcho's `user-` prefix and the `removeUserPrefix` setting are gone. When `peerName` is unset, Kilo uses `HONCHO_PEER_NAME`, then the OS user name, instead of `user`. A `peerName` equal to `aiPeer` is an error instead of a renamed user.
- The plugin no longer writes `~/.honcho/config.json` on first run. It used to write `peerName: "user"`, which every other Honcho tool then read.
- `honcho_setup` no longer takes an `apiKey`, so a key never passes through the chat, Kilo's history or the model provider.
- `~/.honcho/config.json` is written with mode 600, since it can hold the API key.
- `observationMode` defaults to `unified` for every config. The "observation mode is unset" prompt, an opencode-honcho upgrade step, is gone.

- Keyword auto-conclusions are off by default. They saved prompts containing words like "always" or "never" verbatim, including prompts such as "never mind, revert that", and duplicated what Honcho's deriver already extracts.
- In `unified` observation mode the agent peer no longer observes the user. Nothing in unified mode read the agent's collection, so each message was derived twice. The session-start summary now reads the user's own collection in unified mode. The plugin sets the agent's session config once per session, because adding a peer to a session keeps its stored config.

### Fixed

- Prompt-specific recall is attached in `experimental.chat.messages.transform`, so it reaches the model without being saved to Kilo's session history. Kilo persists parts added in `chat.message`.
- `dispose` waits up to 2s for event work in flight, so a one-shot `kilo run` no longer exits before the assistant reply reaches Honcho.
- The system transform skips Kilo's title-generation calls and calls with no session id.
- Session keys start with the user peer, so teammates sharing a workspace get separate sessions.
- Config writes keep an `apiKey` written as `${VAR}` instead of saving the expanded secret.
- `/honcho:status` shows `HONCHO_*` environment values the server uses, and falls back to the `kilo` workspace instead of the folder name.
- TUI dialogs return the user's answer. Kilo's `clear()` runs each dialog's close callback, which settled them as cancelled, so `/honcho:setup`, `/honcho:config` and import uploads stopped after the first choice.
- The model cannot move the API key: `honcho_setup` sends the saved key only to the saved server, and a new server needs a key typed into a window that names it. `honcho_set_config` refuses `apiKey`, `baseUrl` and `${...}` values.
- `shell.env` no longer gives every agent shell `HONCHO_API_KEY`.
- Text Kilo adds itself, such as `@`-mentioned file contents, is no longer saved as the user's words. The importer also skips compaction summaries.
- `/honcho:setup` with a blank key keeps the saved key instead of writing an empty one.
- An unreadable `~/.honcho/config.json` pauses Honcho with an error in the sidebar instead of breaking every hook. Config writes are atomic.
- Each Kilo chat has its own recall. Chats in one folder shared state, so the second chat could get no recall.
- Compaction no longer copies recalled memory into Kilo's saved summary.

### Removed

- The OpenCode 2.x adapter. Kilo uses the 1.x plugin API.
