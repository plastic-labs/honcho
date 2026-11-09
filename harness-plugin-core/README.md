# @honcho-ai/harness-plugin-core

Shared runtime for Honcho harness plugins.

```ts
import { configPath, loadConfig, saveHost } from '@honcho-ai/harness-plugin-core'

const cfg = loadConfig({ host: 'harness', defaults: { aiPeer: 'harness' } })
for (const w of cfg.warnings) console.warn(`${w} Run \`honcho doctor\` to fix it.`)
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

`HONCHO_API_KEY` overrides both; with both in a block, a live `oauth` grant wins. Omit `baseUrl` for Honcho cloud. A plugin may keep keys of its own in its block.

Resolution, highest wins: `HONCHO_*` env (`HONCHO_PEER_NAME`, `HONCHO_WORKSPACE`, `HONCHO_AI_PEER`, `HONCHO_BASE_URL`) → `hosts.<host>` → `defaults` → built-in. `peerName` falls back to `$USER`; a plugin's setup should ask for it, offering `suggestPeerName(file)` (the name the other plugins agree on, else the old top-level one, else `$USER`). `workspace` and `aiPeer` fall back to the host name. The SDK pins `/v3`; config stores the origin.

`saveHost(path, host, patch)` merges into that one block under the config lock; `undefined` removes a key.

Versions: `saveHost` and `migrateHost` mark the block `"schemaVersion": 2`, which records that its plugin reads only that block. A file whose top-level `schemaVersion` is newer than this package knows is read but never written: `saveHost` throws and `migrateHost` does nothing, so an old plugin cannot rewrite a shape it does not understand.

## Moving an older file

A plugin fixes its own block at startup: `migrateHost(configPath(), host, deadKeys)` copies top-level `apiKey`, `peerName` and `baseUrl` into `hosts.<host>` (its `workspace` too, under `globalOverride` or when it has no block yet), renames old keys there, and drops the `deadKeys` it no longer uses. It returns what it changed, and writes nothing when the block is current. It never touches the top level or another host's block, since plugins on older versions may still read them; `honcho doctor` removes those once every plugin reads only its own block.

## Checking a config file

`checkConfig(file)` lists problems with an older file, each with the fixes `applyFixes(file, issues)` applies: top-level settings moved into every host block that lacks them, old key names (`auth.apiKey`, `endpoint`, `environmentUrl`, `workspaceId`) renamed, `globalOverride` turned into each host's `workspace`, the honcho CLI's top-level `oauth` moved to `hosts.honcho-cli`, and leftover top-level keys removed. Copies are always offered; deletions (top-level keys, a block's old key names) only once the plugins that might still read them have marked their blocks, so an old plugin never loses its settings. Hosts that share a workspace under different `peerName`s are reported without a fix. `loadConfig` returns the messages as `warnings`.

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
import { loadConfig, setTelemetryHeaders, telemetryHeaders } from '@honcho-ai/harness-plugin-core'

const cfg = loadConfig({ host: 'harness' })
const honcho = new Honcho({
  apiKey: cfg.apiKey,
  baseURL: cfg.baseUrl,
  workspaceId: cfg.workspace,
  timeout: cfg.timeoutMs,
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
