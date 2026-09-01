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
| `X-Honcho-Host` | Agent host name, or `name/version` | `harness/1.3.13` |
| `X-Honcho-Plugin` | Honcho plugin version | `0.1.3` |
| `X-Honcho-Runtime` | This package's version (always sent) | `0.1.0` |
| `X-Honcho-Agent-Model` | The agent's completion model, not a Honcho model | `claude-sonnet-4-5` |

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
    pluginVersion: '0.1.3',
    model: 'claude-sonnet-4-5',
  }),
})

setTelemetryHeaders(honcho.http.defaultHeaders, { model: 'claude-opus-4' })
```
