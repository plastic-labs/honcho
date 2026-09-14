import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

export interface AuthConfig {
  apiKey?: string
  oauth?: { accessToken?: string; refreshToken?: string; expiresAt?: string }
}

/** Identity + connection + kill switch. Valid at root and as a host override. */
export interface RootConfig {
  peerName?: string
  workspace?: string
  baseUrl?: string
  timeoutMs?: number
  auth?: AuthConfig
  enabled?: boolean
}

export type HostBlock = RootConfig

export interface FileConfig extends RootConfig {
  schemaVersion?: number
  hosts?: Record<string, HostBlock>
}

export interface ResolvedConfig {
  host: string
  peerName: string
  workspace: string
  baseUrl: string
  timeoutMs: number
  auth: AuthConfig
  apiKey?: string
  enabled: boolean
  warnings: string[]
}

export const DEFAULT_BASE_URL = 'https://api.honcho.dev'
export const DEFAULT_TIMEOUT_MS = 30_000
export const CONFIG_SCHEMA_VERSION = 1

function isObj(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
}

/** Pre-schema files (no schemaVersion) → v1 keys. Host blocks included. */
function migrate(file: unknown): Record<string, unknown> {
  if (!isObj(file)) return {}
  const v = file.schemaVersion
  if (typeof v === 'number' && v >= CONFIG_SCHEMA_VERSION) return { ...file }
  const out: Record<string, unknown> = { ...file }
  const blocks: Record<string, unknown>[] = [out]
  if (isObj(out.hosts)) {
    out.hosts = Object.fromEntries(
      Object.entries(out.hosts).map(([k, block]) => {
        if (!isObj(block)) return [k, block]
        const next = { ...block }
        blocks.push(next)
        return [k, next]
      })
    )
  }
  for (const b of blocks) {
    if (typeof b.baseUrl !== 'string') {
      if (typeof b.environmentUrl === 'string') b.baseUrl = b.environmentUrl
      else if (isObj(b.endpoint) && typeof b.endpoint.baseUrl === 'string') {
        b.baseUrl = b.endpoint.baseUrl
      }
    }
    if (typeof b.workspace !== 'string' && typeof b.workspaceId === 'string') {
      b.workspace = b.workspaceId
    }
    const auth: Record<string, unknown> = isObj(b.auth) ? { ...b.auth } : {}
    if (typeof auth.apiKey !== 'string' && typeof b.apiKey === 'string') auth.apiKey = b.apiKey
    if (!isObj(auth.oauth) && isObj(b.oauth)) auth.oauth = b.oauth
    if (Object.keys(auth).length) b.auth = auth
    delete b.environmentUrl
    delete b.endpoint
    delete b.workspaceId
    delete b.apiKey
    delete b.oauth
  }
  out.schemaVersion = 1
  return out
}

function merge<T>(base: T, over: unknown): T {
  if (over === undefined || over === null) return base
  if (Array.isArray(over) || !isObj(over)) return over as T
  const out: Record<string, unknown> = { ...(isObj(base) ? base : {}) }
  for (const [k, v] of Object.entries(over)) {
    if (v !== undefined) out[k] = k in out ? merge(out[k], v) : v
  }
  return out as T
}

/** Make a value safe to pass to the SDK as `baseURL`. */
export function normalizeBaseUrl(input: string): string {
  let s = input.trim()
  if (!s) return s
  if (!s.startsWith('http://') && !s.startsWith('https://')) {
    const host = s.split('/')[0].split(':')[0].toLowerCase()
    const local = host === 'localhost' || host === '127.0.0.1' || host === '::1'
    s = `${local ? 'http' : 'https'}://${s}`
  }
  try {
    const u = new URL(s)
    u.hostname = u.hostname.toLowerCase()
    const path = u.pathname === '/' ? '' : u.pathname.replace(/\/+$/, '')
    return `${u.protocol}//${u.host}${path}`
  } catch {
    return s
  }
}

function interpolate(value: string, env: NodeJS.Dict<string>, warnings: string[]): string {
  return value.replace(/\$\{([^}]+)\}/g, (m, name: string) => {
    const v = env[name]
    if (!v) {
      warnings.push(`${m} is not set`)
      return m
    }
    return v
  })
}

function walkStrings<T>(value: T, fn: (s: string) => string): T {
  if (typeof value === 'string') return fn(value) as T
  if (Array.isArray(value)) return value.map((x) => walkStrings(x, fn)) as T
  if (isObj(value)) {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value)) out[k] = walkStrings(v, fn)
    return out as T
  }
  return value
}

/** Pull only the six root fields. Extra host keys (injection, observation, …) are ignored. */
function pickRoot(block: unknown): RootConfig {
  if (!isObj(block)) return {}
  const auth: AuthConfig = isObj(block.auth) ? { ...(block.auth as AuthConfig) } : {}
  const out: RootConfig = {}
  if (typeof block.peerName === 'string') out.peerName = block.peerName
  if (typeof block.workspace === 'string') out.workspace = block.workspace
  if (typeof block.baseUrl === 'string') out.baseUrl = block.baseUrl
  if (typeof block.timeoutMs === 'number') out.timeoutMs = block.timeoutMs
  if (Object.keys(auth).length) out.auth = auth
  if (typeof block.enabled === 'boolean') out.enabled = block.enabled
  return out
}

function pickHost(hosts: Record<string, unknown> | undefined, name: string): RootConfig {
  if (!hosts || !isObj(hosts[name])) return {}
  return pickRoot(hosts[name])
}

/**
 * Highest wins: HONCHO_* env → overlay → hosts.<host> → root → built-in.
 */
export function resolveConfig(
  file: unknown,
  opts: { host: string; env?: NodeJS.Dict<string>; overlay?: RootConfig }
): ResolvedConfig {
  const warnings: string[] = []
  const env = opts.env ?? process.env
  const host = opts.host
  const raw = migrate(file)
  if (typeof raw.schemaVersion === 'number' && raw.schemaVersion > CONFIG_SCHEMA_VERSION) {
    warnings.push(`config schemaVersion ${raw.schemaVersion} is newer than ${CONFIG_SCHEMA_VERSION}`)
  }
  const hosts = isObj(raw.hosts) ? raw.hosts : undefined

  let acc: RootConfig = {
    baseUrl: DEFAULT_BASE_URL,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    enabled: true,
    workspace: host,
  }
  acc = merge(acc, pickRoot(raw))
  acc = merge(acc, pickHost(hosts, host))
  acc = merge(acc, pickRoot(opts.overlay))

  if (env.HONCHO_API_KEY) {
    if (acc.auth?.apiKey) warnings.push('HONCHO_API_KEY shadows auth.apiKey')
    acc = merge(acc, { auth: { apiKey: env.HONCHO_API_KEY } })
  }
  if (env.HONCHO_BASE_URL || env.HONCHO_URL || env.HONCHO_ENDPOINT) {
    const token = env.HONCHO_BASE_URL || env.HONCHO_URL || env.HONCHO_ENDPOINT || ''
    acc.baseUrl = token === 'local' ? 'http://localhost:8000' : token
  }
  if (env.HONCHO_WORKSPACE || env.HONCHO_WORKSPACE_ID) {
    acc.workspace = env.HONCHO_WORKSPACE || env.HONCHO_WORKSPACE_ID
  }
  if (env.HONCHO_PEER_NAME) acc.peerName = env.HONCHO_PEER_NAME
  if (env.HONCHO_TIMEOUT_MS) {
    const n = Number(env.HONCHO_TIMEOUT_MS)
    if (Number.isFinite(n) && n > 0) acc.timeoutMs = n
  }
  if (env.HONCHO_ENABLED === 'false') acc.enabled = false

  acc = walkStrings(acc, (s) => interpolate(s, env, warnings))
  if (acc.baseUrl) acc.baseUrl = normalizeBaseUrl(acc.baseUrl)

  const auth = acc.auth ?? {}
  return {
    host,
    peerName: acc.peerName || env.USER || env.USERNAME || 'user',
    workspace: acc.workspace || host,
    baseUrl: acc.baseUrl || DEFAULT_BASE_URL,
    timeoutMs: acc.timeoutMs && acc.timeoutMs > 0 ? acc.timeoutMs : DEFAULT_TIMEOUT_MS,
    auth,
    apiKey: auth.apiKey,
    enabled: acc.enabled !== false,
    warnings,
  }
}

/**
 * `HONCHO_CONFIG_PATH` if set, returned verbatim. Otherwise `.honcho/config.json`
 * under `HOME`, then `USERPROFILE` (Windows), then `os.homedir()`.
 *
 * `env.HOME` is consulted before `os.homedir()` because Bun's `homedir()` ignores
 * in-process changes to `process.env.HOME`, so tests that redirect HOME would
 * otherwise read and write the real config file.
 */
export function configPath(env: NodeJS.Dict<string> = process.env): string {
  if (env.HONCHO_CONFIG_PATH) return env.HONCHO_CONFIG_PATH
  const home = env.HOME || env.USERPROFILE || homedir()
  return join(home, '.honcho', 'config.json')
}

export function loadConfig(opts: {
  host: string
  env?: NodeJS.Dict<string>
  overlay?: RootConfig
}): ResolvedConfig {
  const env = opts.env ?? process.env
  const path = configPath(env)
  let file: unknown = {}
  if (existsSync(path)) {
    try {
      file = JSON.parse(readFileSync(path, 'utf-8'))
    } catch {
      file = {}
    }
  }
  return resolveConfig(file, { ...opts, env })
}
