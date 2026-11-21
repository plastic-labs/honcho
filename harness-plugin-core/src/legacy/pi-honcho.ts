import { DEFAULT_BASE_URL, normalizeBaseUrl, type LegacySettings } from '../config.js'
import { v01Reader } from './v01.js'

const isObj = (v: unknown): v is Record<string, any> => v !== null && typeof v === 'object' && !Array.isArray(v)
const str = (v: unknown) => (typeof v === 'string' && v.trim() ? v.trim() : undefined)
// `${VAR}` keys were expanded when read; they are left out rather than written into the file expanded.
const key = (v: unknown) => (str(v)?.includes('${') ? undefined : str(v))
const ID = /^[A-Za-z0-9_-]+$/
const toId = (v: string | undefined) =>
  v === undefined || ID.test(v) ? v : v.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 512) || 'user'
// pi's own older `endpoint` shapes: a string, `endpoint.baseUrl`, or `environment: "local"`.
const endpointUrl = (o: Record<string, any>) => {
  const e = o.endpoint
  if (str(e)) return str(e)
  if (!isObj(e)) return undefined
  return str(e.baseUrl) ?? (e.environment === 'local' ? 'http://localhost:8000' : undefined)
}
const hasUrl = (o: Record<string, any>) => Boolean(str(o.baseUrl) ?? str(o.environmentUrl) ?? endpointUrl(o))

/**
 * How pi-honcho up to 1.0.1 read the file:
 * - `workspace`, `peerName` and the server through 0.1.x of this package
 *   (`v01Reader`), except that its own older `endpoint` shapes win: the block's,
 *   else the top level's when the block names no server.
 * - `aiPeer` only from `hosts.pi` (default `pi`). Ids that aren't valid Honcho ids
 *   were rewritten (`My WS` became `My-WS`).
 * - The key: `hosts.pi.apiKey` or `hosts.pi.auth.apiKey`, then the honcho CLI's
 *   top-level grant when it was issued for that server, then the top-level
 *   `apiKey`, then `auth.apiKey`. A schema-2 plugin keeps only its own grant, so a
 *   user who was on the CLI's grant gets no key and is asked to sign in, rather
 *   than being moved to the top-level key.
 */
export function piHonchoReader(file: Record<string, any>, host: string): LegacySettings {
  const block: Record<string, any> = isObj(file.hosts) && isObj(file.hosts[host]) ? file.hosts[host] : {}
  const core = v01Reader(file, host)
  const blockEndpoint = str(block.baseUrl) || str(block.environmentUrl) ? undefined : endpointUrl(block)
  const topEndpoint = str(file.baseUrl) || str(file.environmentUrl) ? undefined : endpointUrl(file)
  const baseUrl = blockEndpoint ?? (topEndpoint && !hasUrl(block) ? topEndpoint : undefined) ?? core.baseUrl
  const grant = isObj(file.oauth) ? file.oauth : isObj(file.auth) && isObj(file.auth.oauth) ? file.auth.oauth : undefined
  const onCliGrant =
    grant !== undefined && typeof grant.accessToken === 'string' && typeof grant.refreshToken === 'string' &&
    (!str(grant.host) || normalizeBaseUrl(String(grant.host)) === normalizeBaseUrl(baseUrl ?? DEFAULT_BASE_URL))
  const blockKey = key(block.apiKey) ?? (isObj(block.auth) ? key(block.auth.apiKey) : undefined)
  return {
    apiKey: blockKey ?? (onCliGrant ? undefined : (key(file.apiKey) ?? (isObj(file.auth) ? key(file.auth.apiKey) : undefined))),
    baseUrl,
    workspace: toId(core.workspace),
    peerName: toId(core.peerName),
    aiPeer: toId(str(block.aiPeer)),
  }
}
