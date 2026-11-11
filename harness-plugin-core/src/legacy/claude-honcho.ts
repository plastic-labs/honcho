import type { LegacySettings } from '../config.js'

const isObj = (v: unknown): v is Record<string, any> => v !== null && typeof v === 'object' && !Array.isArray(v)
const str = (v: unknown) => (typeof v === 'string' && v ? v : undefined)

/**
 * How claude-honcho read the file up to 0.3.4, before it moved to this package
 * (its own `resolveConfig`), for its `claude_code`, `cursor` and `obsidian` hosts:
 * - Its block is `hosts.<host>`, also spelled with `-` for `_`.
 * - The block's `apiKey`, else the top level's. `auth.apiKey` and `environmentUrl` were never read.
 * - The server is the block's `endpoint`, else the top level's: its `baseUrl`
 *   (with or without `/v3`), else `localhost:8000` for `environment: "local"`, else cloud.
 * - `peerName` only from the top level.
 * - Under `globalOverride` the top-level `workspace` and `aiPeer` win. Otherwise the
 *   block's; only with no block at all, the top-level `workspace` and `claudePeer`
 *   (`cursorPeer` for Cursor).
 */
export function claudeHonchoReader(file: Record<string, any>, host: string): LegacySettings {
  const hosts = isObj(file.hosts) ? file.hosts : {}
  const found = hosts[host] ?? hosts[host.replace(/_/g, '-')] ?? hosts[host.replace(/-/g, '_')]
  const block = isObj(found) ? found : undefined
  const endpoint = block?.endpoint ?? file.endpoint
  const override = file.globalOverride === true
  return {
    apiKey: str(block?.apiKey) ?? str(file.apiKey),
    baseUrl: str(endpoint?.baseUrl)?.replace(/\/v3\/?$/, '') ?? (endpoint?.environment === 'local' ? 'http://localhost:8000' : undefined),
    peerName: str(file.peerName),
    workspace: str(override || !block ? file.workspace : block.workspace),
    aiPeer: str(override ? (file.aiPeer ?? block?.aiPeer) : block ? block.aiPeer : file[host === 'cursor' ? 'cursorPeer' : 'claudePeer']),
  }
}
