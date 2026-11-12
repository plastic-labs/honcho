# Honcho Memory Provider

AI-native cross-session user modeling: one Honcho session per chat, the user's peer card at the start of each session, Honcho's answer about the user on each turn, and tools to ask, search and save.

> **Honcho docs:** <https://docs.honcho.dev/v3/guides/integrations/hermes>

## Requirements

- `honcho-ai` 2.5 or later (installed with the plugin)
- A Honcho server with scopes and workspace chat (Honcho Cloud, or self-hosted v3.1.0+)
- A Honcho Cloud account — connect via OAuth sign-in or an API key from
  [app.honcho.dev](https://app.honcho.dev) — or a self-hosted instance

## Setup

```bash
hermes memory setup honcho   # configure Honcho directly (works on a fresh install)
hermes memory setup          # generic picker, choose Honcho from the list
```

For cloud, the wizard asks **OAuth, device code, or API key**. OAuth opens a
browser sign-in and stores the grant itself — nothing to copy; tokens refresh
automatically. On SSH/headless machines choose **device**: the CLI prints a
short code and a link you open in a browser on any other machine; setup
completes once you approve there. The desktop app offers the browser flow as
a **Connect** link next to the memory-provider dropdown.

Or manually:

```bash
hermes config set memory.provider honcho
echo "HONCHO_API_KEY=***" >> ~/.hermes/.env
```

> `hermes honcho setup` also works, but only **after** Honcho is the active
> memory provider — the `honcho` subcommand is registered for the active
> provider only. On a fresh install, use `hermes memory setup honcho`.

## How It Works

The behaviour below is fixed: none of it has a setting.

### One Session per Chat, with Scopes

Each Hermes chat is one Honcho session. A chat the gateway routes (a Telegram or Discord chat or thread, and a Desktop or TUI chat it carries) uses the gateway's session key, which stays put across `/new`, compression and restarts. Any other chat uses the Hermes session id: `/new`, `/resume` and `/branch` start another Honcho session, while compression and `/undo` continue the current one. Ids longer than Honcho's 100-character limit are cut with a hash suffix.

A new session joins two Honcho scopes: its git repository and this machine's short hostname. The repository comes from the chat's working directory (`git rev-parse --git-common-dir`, so every worktree of a repo shares one scope; the directory name outside git). A messaging chat runs wherever the gateway started, so it joins the machine scope only. Scopes are sent only to a session that has no messages yet: sending them to one that has queues a copy of its memory into each scope again.

### What Gets Saved

Every user and assistant message, through a background writer. A message longer than Honcho's 25,000-character limit is split at paragraph, sentence or word boundaries, each later part marked `[continued]`; nothing is truncated. Skipped: cron and flush runs, heartbeat prompts, and the gateway's own notifications and harness turns (delegation results, compaction summaries, group-chat and discussion harnesses). Hermes itself handles slash commands without a turn.

### What Gets Injected

- **At session start**, the system prompt gets the memory directives (recall with `honcho_chat`, which scopes this chat is in, save insights with `honcho_conclude`) and the user's peer card. Hermes builds the prompt once per session and reuses it for a continuing one.
- **On each turn**, Honcho's chat answer about the user, asked in the background after the previous turn and added to the user message in a `<memory-context>` fence. The first turn waits up to 3 seconds for the session to start and 2 more for that answer; later turns never wait. Until an answer lands the question is who the user is, then what about them matters to this conversation. The reasoning level is `low`, raised one step for a message of 120+ characters and two for 400+, at most `high`; the answer is cut at 600 characters. Short or trivial messages (`ok`, `thanks`, `/help`) start nothing, and a run of empty answers widens the gap between questions, up to 8 turns.

No other context is injected: no `context()` call, session summary or representation.

### Perspective

Every peer in a session observes itself and nobody else, so each peer's memory is its own view. Reads (the injection, `honcho_chat`, `honcho_profile`, conclusions) use the user's own view of themselves. A session that existed before this release gets these flags set when the plugin first opens it.

## Tools

| Tool | LLM call? | Description |
|------|-----------|-------------|
| `honcho_chat` | Yes | Ask Honcho about the user. `scope` limits the answer to one or more scopes' sessions; `workspace: true` asks across everyone in the workspace; `reasoning_level` defaults to `medium` |
| `honcho_search` | No | Cross-session message search (hybrid semantic + keyword, ranked excerpts; 800 tok default, 2000 max) |
| `honcho_profile` | No | Read or replace a peer card |
| `honcho_conclude` | No | Write, list/search, or delete conclusions. Listing returns what Honcho derived, not the facts stated outright |

`honcho_search`, `honcho_profile` and `honcho_conclude` take an optional `peer` (`"user"` by default, `"ai"`, or a peer id). During a bot-authored turn the write tools refuse, because conclusions and cards describe the human.

## Config Resolution

Each Hermes profile keeps its Honcho settings in its own file, `$HERMES_HOME/honcho.json`
(`~/.hermes/honcho.json` for the default profile, `~/.hermes/profiles/<name>/honcho.json` for a named
one), and in that file in one block: `hosts.<host>`. The host key is derived from the active profile:
`hermes` (default) or `hermes_<profile>`.

The block is all the plugin reads: never the file's top level, another host's block or another file.
For every key, resolution order is: **host block > env var > default**.

**Migrating from 1.1.0.** Earlier releases read the first of `$HERMES_HOME/honcho.json`,
`~/.hermes/honcho.json` and `~/.honcho/config.json` that existed, and in it each key from the host
block if set there, else from the file's top level. The first time a profile starts (or saves a setting)
on this release, the plugin writes exactly what that profile resolved into its own host block, top-level
values and settings from another file included, and marks the block `"schemaVersion": 2`. Until then it
reads the same view, so the profile's connection and identity don't change. After it, edits to the top
level, to another profile's block or to `~/.honcho/config.json` no longer reach the profile. Three
spellings fold into `baseUrl` on the way: `base_url`, a top-level `endpoint.baseUrl`, and
`"environment": "local"` (`http://localhost:8000`). The settings this release fixes (sessions, recall,
injection, observation, capture, timeouts: `sessionStrategy`, `recallMode`, `dialectic*`,
`observationMode`, `saveMessages`, `writeFrequency`, `timeout` and the rest) are removed from the block.

**Credentials.** A key belongs to the server it was saved for: changing `baseUrl` without entering a new
key drops the old one. Signing in (OAuth or device code) makes the grant the credential and Honcho cloud
the server, so any `baseUrl` goes. While signed in, the access token holds the `apiKey` slot, and an API
key entered in the settings panel is ignored until you switch to one in `hermes honcho setup`.

Writes take a lock beside the file (`honcho.json.lock`, the same lock `@honcho-ai/harness-plugin-core`
takes) and replace the file atomically, mode `0600`.

## Full Configuration Reference

### Identity & Connection

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `apiKey` | string | — | API key. Falls back to `HONCHO_API_KEY` env var. When connected via OAuth, holds the auto-refreshing access token instead |
| `oauth` | object | — | OAuth grant (refresh token, expiry, client, token endpoint). Written by the Connect/sign-in flows and rotated automatically — not hand-edited. Optional: an API key alone works without it |
| `baseUrl` | string | — | Base URL for self-hosted Honcho; unset means Honcho cloud. Local URLs auto-skip API key auth |
| `enabled` | bool | auto | Master toggle. Auto-enables when `apiKey` or `baseUrl` present |
| `workspace` | string | host key | Honcho workspace ID. Shared environment — all profiles in the same workspace can see the same user identity and related memories |
| `peerName` | string | — | User peer identity |
| `aiPeer` | string | host key | AI peer identity |

### Identity Mapping (Gateway Multi-User)

In gateway deployments (Telegram, Discord, Slack, etc.) each user arrives with a platform-native runtime ID (Telegram UID, Discord snowflake, Slack user). These three keys control how those runtime IDs map to Honcho peers. The resolver is config-driven and deterministic — no automatic merging or runtime inference.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pinUserPeer` | bool | `false` | When `true`, every gateway runtime user collapses to `peerName`. Single-operator deployments where you want all your platforms (and any other users) to share one peer |
| `userPeerAliases` | object | `{}` | Map of runtime IDs to peer IDs (`{"7654321": "alice"}`). Many-to-one is the intended pattern — alias all your runtime IDs to one peer name. One-to-many is not supported; one runtime ID resolves to exactly one peer |
| `runtimePeerPrefix` | string | `""` | Prepended to unknown runtime IDs to namespace them (e.g. `"telegram_"` → `telegram_7654321`). Used only when no alias matches. Prevents collisions between platforms whose runtime IDs share the same shape |

**The machine owner is `peerName`.** Hermes treats every signed-in surface (Desktop, the web app, a phone signed in through it, the TUI, one-shot runs) as the machine's own OS account and passes it as the runtime user `uid:<n>`. When that id is this machine's own account, the session resolves to `peerName`, so your history stays on one peer. Other ids are unaffected.

Other people:

- **separate installs, one Honcho workspace** (two laptops, one household): set each install's `peerName` to its person.
- **one computer, separate OS accounts:** only the account Hermes runs under is the owner. Anyone else keeps their own `uid-<n>` peer; `userPeerAliases: {"uid:1001": "alice"}` names it.
- **an install that belongs to someone else, or a shared login:** set `peerName` to that person, or add an alias for the owner's id (`{"uid:501": "household"}`); an alias beats the owner rule.
- **several people on one Hermes server:** Hermes treats every web or phone login as the owner, so aliases can't tell them apart. Give each person their own profile (each with its own `peerName`), or let others reach it over Telegram/Discord, where they keep their own ids.

On Windows the owner id is a SID and isn't matched yet; add a `userPeerAliases` entry for it to land on `peerName`.

> **Deprecated:** `pinPeerName` is a legacy alias for `pinUserPeer`, still read for back-compat (`pinUserPeer` wins where both are set). `hermes honcho setup` migrates it onto `pinUserPeer` on touch and never writes it.

**Resolver ladder** (first match wins):

```
1. pinUserPeer / pinPeerName=true → return peerName (ignore runtime ID)
2. userPeerAliases[runtime_id]   → return aliased peer
3. userPeerAliases[runtime_id_alt] → check alt-ID too (Telegram UID + username, etc.)
4. runtime_id is this machine's owner (uid:<n>) → peerName, when set
5. runtimePeerPrefix + runtime_id → namespaced peer, with sha256 collision escalation
6. raw sanitized runtime_id      → fallback peer
7. peerName                      → no runtime ID at all (CLI/TUI)
8. neither                       → session init fails with a one-time notice; no peer is minted
```

Step 8 used to derive a peer from the session key (`user-default-<dir>`). That put a desktop or CLI session with no `peerName` on a phantom peer per directory, so its turns and memory never reached the operator's own peer (#93326). Set `peerName` (`hermes honcho peer --user <name>`) or run under a gateway that supplies a user ID.

**Why no `pinAiPeer`?** The AI peer is already pinned by construction — `aiPeer` is the only AI-side identity setting and the resolver never overrides it. Only the user-side peer has the runtime-vs-config tension that `pinUserPeer` resolves.

**Per profile.** All three keys live in the profile's host block, like every other setting; a value 1.1.0 read from the top level was copied into each profile's block once (see [Config Resolution](#config-resolution)).

**Setup — gateway identity tree.** `hermes honcho setup` only asks about identity mapping when it detects a connected gateway platform (it inspects the gateway config; off-gateway the step is skipped because these keys do nothing without a runtime user ID). When it runs, it asks *who talks to this gateway?* and derives the keys:

- **just me** → `pinUserPeer: true`. Every non-agent gateway user collapses to `peerName`; the pin overrides all aliases, so pick this only when no user-side identity needs its own peer. Personal use where you connect Hermes to your own Telegram/Discord/etc. If separate agents reach the gateway and each needs a distinct peer, do **not** pin — leave `pinUserPeer: false` and map them via `userPeerAliases` (the `[e]` editor).
- **me + other people, pooled** → `pinUserPeer: false` + `userPeerAliases` mapping your runtime IDs to `peerName`. You stay on the shared history; everyone else gets their own peer.
- **me + other people / only other people** → `pinUserPeer: false`, optional `runtimePeerPrefix`. Each runtime user → own peer. For bots serving many humans.

Pick **[e]** at the prompt to set the three keys directly instead of going through the tree.

**Interactive mapping — `hermes honcho peers map`.** The setup tree covers the common shapes; `hermes honcho peers map` is the full identity view for a gateway with many users and agents. It joins two sources: the workspace's peers fetched from the Honcho API (labeled from local config — your peer, each profile's AI peer, alias targets, runtime peers of seen accounts, `user-*` fallback peers, honestly `unrecognized` otherwise), and the gateway accounts recorded in the local session store (platform, runtime ID, name, and what each currently resolves to, with `✓` when that peer already exists and `○ new` when it would be created on first message).

Mapping targets are picked from the workspace list (`p3`) rather than typed, so a typo cannot silently create a fresh peer; typing a name stays available for the deliberate new-peer case, and `-` clears an alias. Every assignment states its consequence: aliases move future messages only, and a runtime peer left behind keeps its history. `p<N>` alone peeks at a peer's card. `w` lists every workspace the key can reach — the wrong-workspace fallback when the peers shown aren't yours — and lets you browse one and, on explicit confirmation, repoint the profile's `workspace` at it. For a standalone workspace browser beyond mapping, [honcho-cli](https://pypi.org/project/honcho-cli/) is an optional companion (`uv tool install honcho-cli`).

With multiple profiles, the accounts table marks siblings that resolve the same account to a different peer (`≠ dreamer→bob`); a save changes this profile's block only. Offline, the command degrades to typed targets over the local account list. `hermes honcho peers` without `map` stays a read-only view.

**Un-pinning (single → per-user).** Flipping `pinUserPeer` from `true` to `false` does not migrate data. Memory accumulated under `peerName` while pinned stays there; runtime users now resolve to fresh, empty peers. To preserve your own continuity, choose the **pooled** path — alias your runtime IDs back to `peerName` so your turns keep landing on the pooled history while other users get their own peers. The wizard offers this steer automatically when it detects you're un-pinning a previously pinned profile.

### Bot DMs (`a2aSessions`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `a2aSessions` | bool | `true` | Write DMs from other bots into their own session per sender bot. `false` skips bot-authored turns |

In bot mode another Hermes profile can DM this agent. The relay marks that turn with author `bot:<profile>`. A gateway platform marks a bot sender with its platform user id and a bot flag. Either way the turn never reaches the human's session. With `a2aSessions: true` (default) the turn is written into `<session>:a2a:<this agent's aiPeer>:<sanitized sender id>-<8-char digest>`: the sender's message under the sender's peer, the reply under this agent's `aiPeer`. The `aiPeer` segment keeps two profiles that share a `workspace` and a session key from writing one sender's DMs into one session. A `bot:` sender is identified by its full id, `bot:<profile>` for a profile on this machine or `bot:<connection>/<profile>` for one relayed through a Desktop connection. Its peer is the `userPeerAliases` entry for that full id if one exists, else the id after `bot:` sanitized, with no `runtimePeerPrefix`. When sanitizing changed the id, or the result equals `peerName` or an alias target, a digest suffix is added the same way `runtimePeerPrefix` users get one, so a bot never lands on the operator's peer and two connections' `coder` stay apart. A platform bot resolves like any other runtime user: alias, then prefix. `pinUserPeer` never collapses a bot onto `peerName`. A bot whose peer would equal this agent's `aiPeer` is skipped, and so is every bot turn when `a2aSessions: false`. During a bot-authored turn `honcho_conclude` and `honcho_profile` refuse writes, because conclusions and cards describe the human. Recall still reads the human's session only.

### Multi-Profile Pattern

Multiple Hermes profiles can share one workspace while maintaining separate AI identities. Each profile has its own `honcho.json` with its own host block, so a shared setting (the key, workspace, peer name) is set in each block. `~/.hermes/honcho.json` (default profile):

```json
{
  "hosts": {
    "hermes": {
      "apiKey": "***",
      "workspace": "hermes",
      "peerName": "yourname",
      "aiPeer": "hermes",
      "schemaVersion": 2
    }
  }
}
```

`~/.hermes/profiles/coder/honcho.json`:

```json
{
  "hosts": {
    "hermes_coder": {
      "apiKey": "***",
      "workspace": "hermes",
      "peerName": "yourname",
      "aiPeer": "coder",
      "schemaVersion": 2
    }
  }
}
```

Both profiles see the same user (`yourname`) in the same shared environment (`hermes`), but each AI peer builds its own memory of itself.

Host key is derived from the active Hermes profile: `hermes` (default) or `hermes_<profile>` (e.g. `hermes -p coder` -> host key `hermes_coder`). Before this release the shared values usually sat once at the top level of `~/.hermes/honcho.json`, with every profile's block beside them; the migration copies what each profile used, top-level values included, into its own file's block once (an older `hermes.<profile>` block is renamed `hermes_<profile>` on the way). The default profile's file keeps its other blocks and its top level, which no migrated profile reads.

The plugin keeps host blocks in step with Hermes profiles:

- **Create:** a new profile whose config selects Honcho (for example `hermes profile create coder --clone`) gets a `hermes_coder` block in its own `honcho.json`, cloned from the default profile's block, with `aiPeer: coder`. The default host's own `apiKey` and grant are not copied (a key 1.1.0 kept at the top level of the default profile's file is, since the new profile already read it); otherwise sign in or set a key for the new profile.
- **Rename:** `hermes profile rename coder writer` re-keys `hermes_coder` to `hermes_writer` in the profile's file and keeps `aiPeer: coder`, so the peer and its memory carry over.
- **Imported profiles**, and profiles created before Honcho was set up, get no block automatically. Run `hermes honcho sync` to create one in each profile's file.

On current Hermes the plugin does this itself and also backs the Honcho settings panel in Desktop and the dashboard. On older Hermes, Hermes core does the same work.

### Injection Audit Log

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `logging` | bool | `false` | Append one JSON record per turn to `~/.honcho/injection.log` saying what was injected and why. The file is owner-only and holds what Honcho said about the user verbatim. `HONCHO_LOGGING=1` also switches it on and `HONCHO_INJECTION_LOG=<path>` moves it |

## Environment Variables

| Variable | Fallback for |
|----------|-------------|
| `HONCHO_API_KEY` | `apiKey` |
| `HONCHO_BASE_URL` | `baseUrl` |
| `HONCHO_TIMEOUT` | Request timeout in seconds (else `honcho.timeout` in config.yaml, else 30) |
| `HONCHO_ENVIRONMENT` | SDK environment, only when the profile has no `honcho.json` |
| `HERMES_HONCHO_HOST` | Host key override |
| `HONCHO_OAUTH_DASHBOARD` | OAuth authorize origin (default: cloud dashboard; local-dev `localhost:3000`) |
| `HONCHO_OAUTH_AUTHORIZE_URL` | Full authorize URL (overrides the dashboard origin) |
| `HONCHO_OAUTH_TOKEN_URL` | Token endpoint (default: cloud API; local-dev `localhost:8000`) |
| `HONCHO_OAUTH_DEVICE_AUTH_URL` | Device-authorization endpoint (default: derived from the token URL) |
| `HONCHO_OAUTH_CLIENT_ID` | OAuth client (default `hermes-agent`) |
| `HONCHO_OAUTH_SCOPE` | Requested scope (default `write`) |

## Client Identity Headers

Every request the plugin sends to Honcho carries these headers, so Honcho can tell which integration and model the traffic comes from. None of them hold message content, user, peer, session or workspace names.

| Header | Example | Source |
|--------|---------|--------|
| `X-Honcho-Host` | `hermes/0.18.2 (darwin)` | Hermes version and OS |
| `X-Honcho-Plugin` | `hermes-plugin-honcho/1.2.0` | This plugin's version |
| `X-Honcho-Agent-Model` | `anthropic/claude-sonnet-4.5` | The model Hermes reports for the current turn (the mixture-of-agents preset name under MoA); sent only on Hermes versions that report the model to memory providers |

Links the plugin prints to the Honcho dashboard or docs carry `utm_source=hermes` parameters for the same purpose.

## CLI Commands

| Command | Description |
|---------|-------------|
| `hermes memory setup honcho` | Configure Honcho directly — works on a fresh install |
| `hermes honcho setup` | Interactive setup wizard (only registered once Honcho is the active provider; redirects to `hermes memory setup`) |
| `hermes honcho status` | Show resolved config for active profile |
| `hermes honcho enable` / `disable` | Toggle Honcho for active profile |
| `hermes honcho peer --user <name>` | Update user peer name |
| `hermes honcho peer --ai <name>` | Update AI peer name |
| `hermes honcho peers` / `peers map` | Show peer identities across profiles / map gateway accounts to peers |
| `hermes honcho identity <file>` | Seed the AI peer's identity (e.g. from SOUL.md); `--show` prints both peers |
| `hermes honcho migrate` | Walk through moving OpenClaw's local memory files into Honcho |
| `hermes honcho sync` | Create host blocks for all Hermes profiles |

## Example Config

`~/.hermes/honcho.json` for the default profile; a named profile's file has the same shape with its own host key (see [Multi-Profile Pattern](#multi-profile-pattern)).

```json
{
  "hosts": {
    "hermes": {
      "apiKey": "***",
      "workspace": "hermes",
      "peerName": "username",
      "enabled": true,
      "aiPeer": "hermes",
      "pinUserPeer": true,
      "schemaVersion": 2
    }
  }
}
```

## Development

The unit tests run against a pinned Hermes install (the plugin imports Hermes core
modules) and make no network calls:

```bash
cd hermes-plugin-honcho/tests
uv run pytest
```

The Hermes pin is the `hermes-agent` entry in `tests/pyproject.toml`; bump it together
with `uv lock` in `tests/`. Keep test dependencies out of the plugin's own
`pyproject.toml`: Hermes locks that file, and a `hermes-agent` entry there breaks every
install.
