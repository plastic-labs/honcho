# @honcho-ai/harness-plugin-core

Shared runtime for Honcho harness plugins.

```ts
import { loadConfig, resolveConfig } from '@honcho-ai/harness-plugin-core'

const cfg = loadConfig({ host: 'harness' })
// a harness can pass its plugin config as an overlay of the same six keys:
const cfg = resolveConfig(file, { host: 'harness', overlay: { workspace: 'harness', auth: { apiKey } } })
```

Locally: `"@honcho-ai/harness-plugin-core": "file:../harness-plugin-core"` (bun imports the TypeScript source).

## File shape

```json
{
  "schemaVersion": 1,
  "peerName": "user",
  "workspace": "honcho",
  "baseUrl": "https://api.honcho.dev",
  "timeoutMs": 30000,
  "auth": { "apiKey": "${HONCHO_API_KEY}" },
  "enabled": true,
  "hosts": {
    "test": { "workspace": "test" }
  }
}
```

Missing `schemaVersion` is 0. On read, v0 keys (`environmentUrl`, `workspaceId`, top-level `apiKey`) are remapped in memory; the file is not rewritten.

Resolution, highest wins: `HONCHO_*` env → overlay → `hosts.<host>` → root → built-in.

A host block may override the same six fields.

Built-ins: `baseUrl = https://api.honcho.dev`, `timeoutMs = 30000`, `enabled = true`, `peerName = $USER`, `workspace` falls back to the host name. The SDK pins `/v3`; config stores the origin.

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
