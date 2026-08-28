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
