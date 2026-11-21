import type { LegacySettings } from '../config.js'

const isObj = (v: unknown): v is Record<string, any> => v !== null && typeof v === 'object' && !Array.isArray(v)
// `${VAR}` values were expanded at read time; they are left out rather than written into the file expanded.
const text = (v: unknown) => (typeof v === 'string' && !v.includes('${') ? v.trim() : '')
const id = (v: string) => v.toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') || 'default'

/**
 * How opencode-honcho 0.2.1 read the file, returned as the ids it actually sent:
 * - The block's `apiKey`, else the top level's. `auth.apiKey` and `environmentUrl` were never read.
 * - The server: top-level `baseUrl`, then `endpoint.baseUrl` in the block, then at the
 *   top level. The block's own `baseUrl` was ignored.
 * - `workspace` and `aiPeer` only from the block (default `opencode`), lowercased into ids.
 * - The user peer was `user-<peerName>` (top-level `peerName`, default `user`) unless
 *   the block set `removeUserPrefix`.
 */
export function opencodeHonchoReader(file: Record<string, any>, host: string): LegacySettings {
  const block: Record<string, any> = isObj(file.hosts) && isObj(file.hosts[host]) ? file.hosts[host] : {}
  const endpointUrl = (o: Record<string, any>) => (isObj(o.endpoint) ? text(o.endpoint.baseUrl) : '')
  const name = text(file.peerName) || 'user'
  const aiPeer = id(text(block.aiPeer) || 'opencode')
  const bare = block.removeUserPrefix === true || text(block.removeUserPrefix).toLowerCase() === 'true'
  return {
    apiKey: text(block.apiKey) || text(file.apiKey) || undefined,
    baseUrl: text(file.baseUrl) || endpointUrl(block) || endpointUrl(file) || undefined,
    peerName: bare && name !== aiPeer ? name : id(`user:${name}`),
    workspace: id(text(block.workspace) || 'opencode'),
    aiPeer,
  }
}
