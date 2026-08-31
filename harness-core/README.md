# @honcho-ai/harness-core

Shared runtime for Honcho harnesses.

```ts
import { loadConfig, resolveConfig } from '@honcho-ai/harness-core'

const cfg = loadConfig({ host: 'harness' })
// a harness can pass its plugin config as an overlay of the same six keys:
const cfg = resolveConfig(file, { host: 'harness', overlay: { workspace: 'harness', auth: { apiKey } } })
```

Locally: `"@honcho-ai/harness-core": "file:../honcho/harness-core"` (bun imports the TypeScript source).

## File shape

```json
{
  "peerName": "user",
  "workspace": "honcho",
  "baseUrl": "https://api.honcho.dev",
  "timeoutMs": 30000,
  "auth": { "apiKey": "${HONCHO_API_KEY}" },
  "enabled": true,
  "hosts": {
    "test": { "workspace": "test" },
  }
}
```

Resolution, highest wins: `HONCHO_*` env → overlay → `hosts.<host>` → root → built-in.

A host block may override the same six fields.

Built-ins: `baseUrl = https://api.honcho.dev`, `timeoutMs = 30000`, `enabled = true`, `peerName = $USER`, `workspace` falls back to the host name. The SDK pins `/v3`; config stores the origin.

## Telemetry headers

Pass `telemetryHeaders()` as the SDK's `defaultHeaders`. Arbitrary headers are accepted by both the SDK and the Honcho API; missing identity fields are omitted.

| Header | Meaning | Example |
|---|---|---|
| `X-Honcho-Host` | Agent host name, or `name/version` | `opencode/1.3.13` |
| `X-Honcho-Plugin` | Honcho plugin version | `0.1.3` |
| `X-Honcho-Runtime` | This package's version (always sent) | `0.1.0` |
| `X-Honcho-Agent-Model` | The agent's completion model, not a Honcho model | `claude-sonnet-4-5` |
| `X-Honcho-Api` | Where Honcho is running: `cloud` / `custom` | `cloud` |

`X-Honcho-Api` is classified from the resolved `baseUrl` (`api.honcho.dev` → `cloud`, anything else → `custom`). The raw URL is never sent.

```ts
import { Honcho } from '@honcho-ai/sdk'
import { loadConfig, telemetryHeaders } from '@honcho-ai/harness-core'

const cfg = loadConfig({ host: 'opencode' })
const honcho = new Honcho({
  apiKey: cfg.apiKey,
  baseURL: cfg.baseUrl,
  workspaceId: cfg.workspace,
  timeout: cfg.timeoutMs,
  defaultHeaders: telemetryHeaders({
    host: 'opencode',
    hostVersion: '1.3.13',
    pluginVersion: '0.1.3',
    model: 'claude-sonnet-4-5', // omit when the host does not know it
    baseUrl: cfg.baseUrl, // localhost or a self-hosted origin → X-Honcho-Api: custom
  }),
})
```

What each host can actually supply today:

| Host | Harness version | Plugin version | Model |
|---|---|---|---|
| openclaw | `api.runtime.version` | `api.version` | not at client construct |
| claude-honcho | not in hook stdin | `getPluginVersion()` | not in hook stdin |
| cursor-honcho | `hookInput.cursor_version` | plugin.json | `hookInput.model` (per hook; client is constructed per request) |
| opencode-honcho | `client.global.health().version` | package.json | `input.model.modelID` on chat hooks |
| codex-honcho | not in hook payload | package.json | not in hooks (config.toml default only) |
| hermes | `hermes_cli.__version__` | bundled, same as harness | not passed into the memory provider |
