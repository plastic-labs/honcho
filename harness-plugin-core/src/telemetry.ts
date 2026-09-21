/** Optional identity a host plugin knows at Honcho-client construction time. */
export interface TelemetryIdentity {
  /** Host harness name, e.g. `harness`. */
  host?: string
  /** Host harness version, e.g. `2.1.3`. Omit when the harness does not expose it. */
  hostVersion?: string
  /** OS platform. Defaults to `process.platform`. */
  platform?: string
  /** Integration (plugin) name, e.g. `harness-honcho`. */
  plugin?: string
  /** Integration version, e.g. `0.2.11`. */
  pluginVersion?: string
  /** Agent completion model, e.g. `claude-sonnet-4-5`. Not a Honcho deriver/dialectic model. */
  model?: string
}

export const HEADER_HOST = 'X-Honcho-Host'
export const HEADER_PLUGIN = 'X-Honcho-Plugin'
export const HEADER_AGENT_MODEL = 'X-Honcho-Agent-Model'

function sanitize(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const s = value.replace(/[\r\n]+/g, ' ').trim()
  return s || undefined
}

/** A `name/version` product token. Characters that would break parsing become `-`. */
function token(name: unknown, ver: unknown): string | undefined {
  const clean = (v: unknown) => sanitize(v)?.replace(/[\s()/;]+/g, '-')
  const n = clean(name)
  const v = clean(ver)
  if (n && v) return `${n}/${v}`
  return n || v
}

/** `X-Honcho-Host` value: `harness/2.1.3 (darwin)`. Undefined when the host is unknown. */
export function hostHeaderValue(id: TelemetryIdentity = {}): string | undefined {
  const host = token(id.host, id.hostVersion)
  if (!host) return undefined
  const platform = token(id.platform ?? process.platform, undefined)
  return platform ? `${host} (${platform})` : host
}

/** `X-Honcho-Plugin` value: `harness-honcho/0.2.11`. Undefined when the plugin is unknown. */
export function pluginHeaderValue(id: TelemetryIdentity = {}): string | undefined {
  return token(id.plugin, id.pluginVersion)
}

/**
 * Headers to pass as the SDK's `defaultHeaders`. Fields are omitted when unknown, so a
 * partial identity (e.g. just `model`) only touches the headers it names.
 */
export function telemetryHeaders(
  id: TelemetryIdentity = {},
  extra?: Record<string, string>
): Record<string, string> {
  const headers: Record<string, string> = {}
  const host = hostHeaderValue(id)
  const plugin = pluginHeaderValue(id)
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
