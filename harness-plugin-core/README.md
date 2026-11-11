# @honcho-ai/harness-plugin-core

Shared runtime for Honcho harness plugins.

```ts
import { configPath, loadConfig, saveHost } from '@honcho-ai/harness-plugin-core'

const cfg = loadConfig({ host: 'harness', defaults: { aiPeer: 'harness' } })
for (const w of cfg.warnings) console.warn(w)
await saveHost(configPath(), 'harness', { workspace: 'my-workspace' })
```

Install: `bun add @honcho-ai/harness-plugin-core` or `npm install @honcho-ai/harness-plugin-core`. The package ships compiled ESM plus type declarations in `dist/`, so it loads under plain Node and `tsc` as well as bun. Locally: `"@honcho-ai/harness-plugin-core": "file:../harness-plugin-core"`, after running `bun run build` here so `dist/` exists.

## File shape

Schema version 2 (`config.schema.json`, also at `https://cdn.jsdelivr.net/npm/@honcho-ai/harness-plugin-core@0/config.schema.json`; set it as the file's `$schema` for editor validation). Each plugin keeps its settings in its own block under `hosts`; nothing is read from the top level.

```json
{
  "schemaVersion": 2,
  "hosts": {
    "harness": {
      "oauth": {
        "accessToken": "...",
        "refreshToken": "...",
        "accessExpiresAt": 1794263473,
        "clientId": "harness-honcho",
        "scope": "write"
      },
      "peerName": "alice",
      "workspace": "harness",
      "aiPeer": "harness"
    },
    "other-harness": {
      "apiKey": "hch-v3-...",
      "peerName": "alice",
      "workspace": "harness",
      "baseUrl": "http://localhost:8000"
    }
  }
}
```

A block signs in one of two ways:

- `oauth`: the plugin's own sign-in to Honcho cloud, written by `saveGrant` after the device flow and refreshed by `ensureFreshCredential`. Each plugin has its own; none reads another's.
- `apiKey`: a key from app.honcho.dev, or for a self-hosted Honcho (OAuth is cloud-only).

`HONCHO_API_KEY` overrides both. With both in a block the grant wins, even expired: a dead sign-in asks the user to sign in again rather than falling back to a key. Signing in (`saveGrant`) removes the block's `apiKey` and `baseUrl`, since it means Honcho cloud with this grant. Omit `baseUrl` for Honcho cloud. A plugin may keep keys of its own in its block.

Resolution, highest wins: `HONCHO_*` env (`HONCHO_PEER_NAME`, `HONCHO_WORKSPACE`, `HONCHO_AI_PEER`, `HONCHO_BASE_URL`) → `hosts.<host>` → `defaults` → built-in. `peerName` falls back to `$USER`; a plugin's setup should ask for it, offering `suggestPeerName(file)` (the name the other plugins agree on, else the old top-level one, else `$USER`). `workspace` and `aiPeer` fall back to the host name. The SDK pins `/v3`; config stores the origin.

`saveHost(path, host, patch, legacy?)` merges into that one block under the config lock; `undefined` removes a key. A block not migrated yet is migrated first (below). A key belongs to the server it was saved for: a patch that moves the block to another server drops the key unless it sets one.

Versions: `saveHost` and `migrateHost` mark the block `"schemaVersion": 2`, which records that its plugin reads only that block and has already taken what it needed from the top level. A file whose top-level `schemaVersion` is newer than this package knows is read but never written: `saveHost` throws and `migrateHost` does nothing, so an old plugin cannot rewrite a shape it does not understand.

## Moving an older file

Before schema 2, each plugin read this file its own way, so a plugin says what it was using with a `LegacyReader`: given the whole file, the `apiKey`, `baseUrl`, `peerName`, `workspace` and `aiPeer` its last release would have used. Each lives in its own file in `src/legacy/`, and its test compares it with fixtures recorded by running that plugin's latest published release:

| Reader | For |
| --- | --- |
| `v01Reader` | plugins that read the file through harness-plugin-core 0.1.x (recorded from 0.1.1) |
| `claudeHonchoReader` | claude-honcho up to 0.3.4 (`claude_code`, `cursor`, `obsidian`) |
| `opencodeHonchoReader` | opencode-honcho up to 0.2.1 |
| `openclawHonchoReader` | openclaw-honcho up to 1.7.0, which read nothing from this file; keeps its `owner` peer |
| `piHonchoReader` | pi-honcho up to 1.0.1; a user on the honcho CLI's grant is asked to sign in |

A plugin whose last release read the file another way adds its reader there, with fixtures from that release.

At startup, before anything reads the config or shows setup, a plugin calls `migrateHost(configPath(), host, deadKeys, legacy)`. Once, while the block is unmarked, it writes exactly what `legacy` says into `hosts.<host>`, so the plugin keeps the same key, server, peer and workspace, and marks the block. After that only the block counts: whatever an older honcho CLI writes to the top level later never reaches it. Every time, it also moves an old `auth.oauth` grant to `oauth` and drops old key names (`auth`, `endpoint`, `environmentUrl`, `workspaceId`) and `deadKeys`. A block already signed in keeps its grant on Honcho cloud rather than the key and server it read before. It returns what it changed, writes nothing when the block is current, and never touches the top level or another host's block.

Until it runs, `resolveConfig`, `pickCredential` and `saveHost` see an unmarked block as `migrateHost` will leave it, so reading first never gives a different answer. Pass the same `legacy` to them where the plugin can run before its migration. A plugin that layers its own settings over the block (OpenClaw's plugin config, say) reads it with `effectiveBlock(file, host, legacy)`.

## Checking a config file

`checkConfig(file)` lists what a plugin should tell the user, read-only: a file newer than this package, and hosts that share a workspace under different `peerName`s (the user is split across peers; only a person can pick the name). `loadConfig` returns the messages as `warnings`. There are no fixes here: each plugin fixes its own block with `migrateHost`, and the top level is the honcho CLI's.

## The top level and the honcho CLI

The top level belongs to the honcho CLI. Today the CLI still keeps its sign-in there (`apiKey`, `environmentUrl`, `oauth`), and nothing removes the old top-level keys. Once it moves its sign-in to `hosts.honcho-cli`, `honcho doctor` will remove them after every installed integration reads only its own block. Writers of this file share these rules:

1. `hosts.<host>.schemaVersion` of 2 or more means that host's plugin reads only its own block.
2. A top-level `schemaVersion` above 2 means the file is read, never written.
3. This package's writes hold `<config>.lock` (created exclusively, holding a per-writer token, broken when its holder's process is gone or after 15s, waited on for up to 20s, removed only by its holder) and replace the file by temp file + rename, mode 0600. The CLI does not take the lock yet.

## Telemetry headers

Pass `telemetryHeaders()` as the SDK's `defaultHeaders`. Arbitrary headers are accepted by both the SDK and the Honcho API; missing identity fields are omitted.

| Header | Meaning | Example |
|---|---|---|
| `X-Honcho-Host` | Host harness, `name/version (platform)` | `harness/2.1.3 (darwin)` |
| `X-Honcho-Plugin` | Honcho integration, `name/version` | `harness-honcho/0.2.11` |
| `X-Honcho-Agent-Model` | The agent's completion model, not a Honcho model | `claude-sonnet-4-5` |

Omit `hostVersion` when the harness does not expose it; `platform` defaults to `process.platform`.

```ts
import { Honcho } from '@honcho-ai/sdk'
import { configPath, ensureFreshCredential, loadConfig, setTelemetryHeaders, telemetryHeaders } from '@honcho-ai/harness-plugin-core'

const cfg = loadConfig({ host: 'harness' })
const cred = await ensureFreshCredential({ path: configPath(), host: 'harness', baseUrl: cfg.baseUrl })
const honcho = new Honcho({
  apiKey: cred?.token,
  baseURL: cfg.baseUrl,
  workspaceId: cfg.workspace,
  defaultHeaders: telemetryHeaders({
    host: 'harness',
    hostVersion: '1.3.13',
    plugin: 'harness-honcho',
    pluginVersion: '0.1.3',
    model: 'claude-sonnet-4-5',
  }),
})

setTelemetryHeaders(honcho.http.defaultHeaders, { model: 'claude-opus-4' })
```

## OAuth (Honcho cloud)

Device sign-in. Each plugin passes its own `clientId`, registered as a first-party client in Honcho cloud, so the approval screen names the tool asking. Self-hosted Honcho has no OAuth server; there the API key is used and OAuth is skipped.

```ts
import { configPath, ensureFreshCredential, pollForToken, requestDeviceCode, saveGrant } from '@honcho-ai/harness-plugin-core'

// sign in once
const device = await requestDeviceCode({ clientId: 'harness-honcho', source: 'harness' })
console.log(`Open ${device.verificationUriComplete} and confirm ${device.userCode}`)
const tokens = await pollForToken(device, { clientId: 'harness-honcho' })
await saveGrant(configPath(), 'harness', tokens)

// before each batch of requests
const cred = await ensureFreshCredential({ path: configPath(), host: 'harness', baseUrl })
if (cred?.reauthRequired) console.warn('Honcho sign-in expired; sign in again')
```

Resolution, highest wins: `HONCHO_API_KEY` env → `hosts.<host>.oauth` → `hosts.<host>.apiKey` → that grant expired. Each plugin has one grant, in its own host block, and never uses another plugin's, so plugins signed in to different tenants stay separate.

Every refresh supersedes the grant's previous access token (the API gateway may keep accepting it for up to 5 minutes from its cache, then rejects it), so a long-running process must call `ensureFreshCredential` before each batch of requests and update its SDK client's key in place rather than cache a token or build a new client.
