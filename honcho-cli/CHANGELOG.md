# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.6.0] - 2026-11-10

### Added

- `honcho workspace replay --from <id> --to <id>`: copy a workspace by replaying it. Workspace, peer and session metadata and configuration, session membership and scopes are copied, then every message is re-sent in order with its original `created_at`, so the destination derives its own conclusions, summaries and peer cards; nothing derived is copied. `--from-url` / `--to-url` (and `HONCHO_FROM_API_KEY` / `HONCHO_TO_API_KEY`) put either side on another Honcho instance, and the configured credentials are never sent to an instance they were not issued for. `--num-sessions N` replays only the N sessions with the most recent message, `--dry-run` reads both sides without writing and says, for each session, whether a replay will create it, resume it, find it up to date, or skip it as a conflict, and a rerun resumes by skipping messages the destination already has. A destination session whose messages differ from the source is left untouched and reported. In a terminal the command prints a short report and the `queue-status` command that shows derivation on the destination; `--json` prints the same facts as one object. Both reports count the conclusions in the source (`not_copied`), because conclusions and peer cards written through the API have no source message and are not recreated. The destination session has the source's current members from its first message, so a peer that joined late observes the whole session. With a workspace-scoped key, a source that has no sessions and no peers is reported as `NOTHING_TO_REPLAY` and is not created. Per-message configuration is not stored by Honcho and cannot be replayed (#1573)

### Fixed

- `honcho --version`, the update check and the `X-Honcho-Host` header report the installed version. `__version__` was a hardcoded string that the 0.5.0 release did not bump, so a 0.5.0 install reported itself as 0.4.0, and the update check told 0.5.0 users that 0.5.0 was available. The version is now read from the installed package metadata (#1571)

## [0.5.0] - 2026-11-05

### Added

- `honcho dream list` and `honcho dream inspect <id>` (Representation History): list the dreams in a workspace, newest first, or the dreams about one peer with `-p`, and see what a dream changed. `inspect` shows the conclusions it created, each with the conclusions it was derived from and where those stand now (current, removed and by which dream, or unavailable once deleted or removed longer ago than history is kept); the conclusions it removed, as they read when removed; and the peer card before and after. A dream past the retention window shows when it was pruned and keeps only the conclusions it created that still exist. Needs a workspace or admin key, and Honcho v3.6.0+ (#1542)

### Changed

- Requires `honcho-ai` 2.9.0 or newer, the first Python SDK with `dreams()` and `get_dream()`

### Fixed

- `HONCHO_API_KEY` wins over a saved OAuth grant, as the documented flag > env > file order says. Before, `HONCHO_API_KEY=<key> honcho workspace list` quietly ran as the saved OAuth identity. A run with an env key no longer refreshes or writes `~/.honcho/config.json`, and an OAuth refresh under `HONCHO_API_KEY` or `HONCHO_BASE_URL` no longer writes those values into it, which had switched every other tool sharing the file to that key's org. A refreshed grant records the host it was refreshed against, and `honcho doctor` names the env key as the credential in use (#1518)

## [0.4.0] - 2026-10-29

### Added

- `source_message_ids` on conclusion output: the messages an explicit conclusion cites as evidence (Honcho v3.5.0+; a count in table mode, the full list in JSON). `conclusion list --cites <message_id>` narrows to the conclusions that cite a message. Null in JSON on servers or SDKs that predate the field (#935)

### Changed

- Requires `honcho-ai` 2.8.0 or newer, the first Python SDK that carries `source_message_ids`

## [0.3.0] - 2026-10-23

### Added

- `honcho install` finds the AI agents on this machine, shows which already have Honcho, and offers a checkbox picker. Claude Code, Hermes Agent, OpenClaw and Pi get their Honcho plugin and Codex the Honcho MCP server and skill, each through the agent's own commands and the API key in `~/.honcho/config.json`. If that file has no `peerName` yet, it asks what agents should call you (default: your OS username) and saves it there for every agent to share. Other agents it finds link to their setup guide, and `honcho install grok-bot` copies a setup prompt for Grok Bot. Run from inside Claude Code or Codex, it sets up that agent without prompting. `honcho install all` or `honcho install <id>...` skips the picker; `--list` and `--json` only report

### Security

- Requires `certifi` 2026.10.19 or newer, the first CA bundle without the emSign root behind the compromised eMudhra intermediates. Upgrading the CLI now upgrades the bundle it verifies TLS with, including under `pip install -U`, which otherwise leaves dependencies alone
- A local stack keeps the Honcho image it pinned on first `honcho start`, so a stack set up before this release can still be running an image with the old bundle. Move to the patched image with `honcho stop`, then `honcho start --image ghcr.io/plastic-labs/honcho:v3.4.1`; data is kept

## [0.2.1] - 2026-10-20

### Fixed

- `honcho session summaries`, `honcho session inspect`, `honcho session context` and `honcho workspace queue-status` emit nested SDK objects (summaries, context messages, per-session queue status) as JSON objects instead of Python repr strings like `"content='...' message_id='...'"` (#1324)

## [0.2.1] - 2026-10-20

### Fixed

- `honcho session summaries`, `honcho session inspect`, `honcho session context` and `honcho workspace queue-status` emit nested SDK objects (summaries, context messages, per-session queue status) as JSON objects instead of Python repr strings like `"content='...' message_id='...'"` (#1324)

## [0.2.0] - 2026-09-21

### Added

- CLI requests carry `X-Honcho-Host: honcho-cli/<version> (<platform>)` so telemetry counts them as CLI traffic rather than direct SDK use (#1181)
- Conclusion attribution on `honcho conclusion list` and `honcho conclusion search` (Honcho v3.2.0+): every row now carries `level`, `source_ids` and `times_derived`, so an extracted fact is distinguishable from a dreamed one. Table output shows a premise count; `--json` keeps the full id list, ready to pipe into `honcho conclusion get` (#952)
- `honcho conclusion get <id>...` fetches conclusions by ID from anywhere in the workspace, with no observer/observed pair required. Feed it a conclusion's `source_ids` to walk a reasoning chain down to the explicit facts underneath it; IDs that no longer exist are reported rather than failing the command (#952)
- `honcho conclusion derived <id>` lists what was built on top of a conclusion — the same edge walked upward. Worth checking before deleting or correcting a fact (#952)
- `--level` on `honcho conclusion list` and `honcho conclusion search`, and `--derived-from` on `honcho conclusion list` (#952)
- `--evidence` on `honcho peer chat` and `honcho workspace chat` (Honcho v3.2.0+): reports the conclusions and messages the dialectic read and the tools it called. Evidence is collated from what the agent accessed rather than reported by the model, so it over-reports and costs no extra model tokens. Messages are summarised per session, since evidence carries their identity but not their text (#1129)

### Changed

- Requires `honcho-ai` 2.5.0 (was 2.4.0), which carries the attribution fields and the evidence-bearing chat response

## [0.1.5] - 2026-09-09

### Added

- `honcho scope` command group: `list`, `create`, `inspect`, `sessions`, `add-sessions`, `remove-session`, `status` (Honcho v3.1.0+) (#1162)
- `honcho workspace chat` for reasoned questions across all peers (Honcho v3.1.0+) (#1150)
- `--scope` on `honcho workspace chat` and `honcho peer chat`, and `--sessions` on `honcho peer chat`, to confine recall (Honcho v3.1.0+) (#1151)

## [0.1.4] - 2026-08-26

### Added

- A TTY notice when a newer `honcho-cli` is on PyPI (`uv tool upgrade honcho-cli`). Skipped in JSON mode; disable with `HONCHO_NO_UPDATE_CHECK`

### Fixed

- `--setup` for openai-compatible writes `EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL` into the profile `.env` alongside `LLM_OPENAI_BASE_URL` (#1068)
- `--setup` API key prompts echo `*` per character so a paste is visibly received instead of a blank getpass field

## [0.1.3] - 2026-08-25

### Added

- `honcho start`, `honcho stop`, and `honcho status` — run a personal Honcho stack in Docker (API, deriver, Postgres, Redis). Profiles live under `~/.honcho/profiles/`. First start pins `ghcr.io/plastic-labs/honcho:latest` by digest and copies the image `config.toml`. Optional `--setup basic` / `--setup advanced` wizard writes LLM overrides to `.env` (#1029)
- `honcho session view` — session transcript table (`--last N`, `--page N --size M`, `--all`, `--reverse`, `--ids`, peer filter via `-p`). Content is shown verbatim, timestamps are normalized to UTC, and the command is read-only: unlike the other session commands it never get-or-creates the session (#1006)

### Fixed

- `honcho message list --last N` no longer stops at the first page of 50 — it walks pages to fill the requested window (#1006)

## [0.1.2] - 2026-07-20

### Added

- Device-code OAuth login for managed Honcho servers. `honcho init` now offers browser-based login (RFC 8628 device authorization grant) when the host advertises the device grant in its OAuth authorization-server metadata; tokens are persisted to `~/.honcho/config.json` and auto-refreshed (#891)
- `HONCHO_CONFIG_DIR` environment variable for pointing the CLI at an alternate config directory (#891)

### Changed

- An OAuth grant now records the host it was minted against and is ignored — neither used nor refreshed — when `base_url` points elsewhere, so a staging grant is never sent to production. A live OAuth token takes precedence over a stored `apiKey`, and a dead grant degrades to the saved key with a warning instead of aborting. Device login no longer deletes the shared `apiKey`, which sibling tools read from the same config file (#891)

## [0.1.1] - 2026-06-15

### Fixed

- Declare `click` as an explicit dependency. The CLI imported `click` directly but relied on it being pulled in transitively, so installs without it on the path could fail at runtime (#787)

## [0.1.0] - 2026-04-20

### Added

- Initial release of `honcho-cli` — a terminal for inspecting and managing a Honcho deployment (#424)
- `workspace`, `peer`, `session`, `message`, `conclusion`, and `config` command groups for managing resources against any Honcho server
- `init` onboarding flow that prompts for and persists connection settings, with flag/env-var pre-seeding for non-interactive use
- Per-command flags, environment variables, and a config file for pointing the CLI at different servers (local, self-hosted, or hosted)
- Rich terminal output and an agent-usage mode for scripting against the CLI
- Documentation and an agent skill for the CLI (#589)
