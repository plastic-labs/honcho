import type { LegacySettings } from '../config.js'

const isObj = (v: unknown): v is Record<string, any> => v !== null && typeof v === 'object' && !Array.isArray(v)
const str = (v: unknown) => (typeof v === 'string' && v ? v : undefined)

/**
 * How 0.1.x of this package (last: 0.1.1) read a plugin's settings, for plugins
 * whose last release read the file through it: the block, then the top level, field by
 * field. A file with a top-level `schemaVersion` of 1 or more was read as written:
 * `auth.apiKey`, `baseUrl`, `peerName`, `workspace`. Any other file was read with the
 * old spellings too: `auth.apiKey` before `apiKey`; `baseUrl`, then `environmentUrl`,
 * then `endpoint.baseUrl`; `workspaceId`. `aiPeer` only from the block.
 */
export function v01Reader(file: Record<string, any>, host: string): LegacySettings {
  const hosts = isObj(file.hosts) ? file.hosts : {}
  const v1 = typeof file.schemaVersion === 'number' && file.schemaVersion >= 1
  const pick = (b: unknown) => {
    const o: Record<string, any> = isObj(b) ? b : {}
    if (v1) return { apiKey: isObj(o.auth) ? str(o.auth.apiKey) : undefined, peerName: str(o.peerName), workspace: str(o.workspace), baseUrl: str(o.baseUrl) }
    return {
      apiKey: (isObj(o.auth) ? str(o.auth.apiKey) : undefined) ?? str(o.apiKey),
      peerName: str(o.peerName),
      workspace: str(o.workspace) ?? str(o.workspaceId),
      baseUrl: str(o.baseUrl) ?? str(o.environmentUrl) ?? (isObj(o.endpoint) ? str(o.endpoint.baseUrl) : undefined),
    }
  }
  const own = pick(hosts[host])
  const top = pick(file)
  return {
    apiKey: own.apiKey ?? top.apiKey,
    peerName: own.peerName ?? top.peerName,
    workspace: own.workspace ?? top.workspace,
    aiPeer: isObj(hosts[host]) ? str(hosts[host].aiPeer) : undefined,
    baseUrl: own.baseUrl ?? top.baseUrl,
  }
}
