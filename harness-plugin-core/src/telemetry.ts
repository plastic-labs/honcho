import { version } from './index.ts'

/** Optional identity a host plugin knows at Honcho-client construction time. */
export interface TelemetryIdentity {
  /** Host app name, e.g. `cursor`, `opencode`. */
  host?: string
  /** Host app version, e.g. `2026.8.1`. */
  hostVersion?: string
  /** Honcho plugin version, e.g. `0.1.2`. */
  pluginVersion?: string
  /** Agent completion model, e.g. `claude-sonnet-4-5`. Not a Honcho deriver/dialectic model. */
  model?: string
}

export const HEADER_HOST = 'X-Honcho-Host'
export const HEADER_PLUGIN = 'X-Honcho-Plugin'
export const HEADER_RUNTIME = 'X-Honcho-Runtime'
export const HEADER_AGENT_MODEL = 'X-Honcho-Agent-Model'

function sanitize(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const s = value.replace(/[\r\n]+/g, ' ').trim()
  return s || undefined
}

function hostValue(id: TelemetryIdentity): string | undefined {
  const name = sanitize(id.host)
  const ver = sanitize(id.hostVersion)
  if (name && ver) return `${name}/${ver}`
  return name || ver
}

/**
 * Headers to pass as the SDK's `defaultHeaders`. Missing fields are omitted.
 * `X-Honcho-Runtime` is always this package's version.
 */
export function telemetryHeaders(
  id: TelemetryIdentity = {},
  extra?: Record<string, string>
): Record<string, string> {
  const headers: Record<string, string> = { [HEADER_RUNTIME]: version }
  const host = hostValue(id)
  const plugin = sanitize(id.pluginVersion)
  const model = sanitize(id.model)
  if (host) headers[HEADER_HOST] = host
  if (plugin) headers[HEADER_PLUGIN] = plugin
  if (model) headers[HEADER_AGENT_MODEL] = model
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      const value = sanitize(v)
      if (value) headers[k] = value
    }
  }
  return headers
}

/** Merge identity onto a live header map (e.g. `honcho.http.defaultHeaders`). */
export function setTelemetryHeaders(
  headers: Record<string, string>,
  id: TelemetryIdentity = {},
  extra?: Record<string, string>
): Record<string, string> {
  return Object.assign(headers, telemetryHeaders(id, extra))
}
