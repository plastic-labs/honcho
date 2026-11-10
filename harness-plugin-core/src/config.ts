import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { updateConfigFile, type OAuthTokens } from './oauth.js'

/** Environment map shape; structurally identical to `process.env`, without depending on `@types/node` in consumers. */
export type Env = Record<string, string | undefined>

/** One plugin's settings: `hosts.<host>` in `~/.honcho/config.json`. */
export interface HostConfig {
  apiKey?: string
  /** This plugin's own OAuth grant; see oauth.ts. */
  oauth?: OAuthTokens
  peerName?: string
  workspace?: string
  aiPeer?: string
  /** Honcho API origin; omitted means Honcho cloud. */
  baseUrl?: string
  /** Set by the plugin once it reads only this block; `honcho doctor` waits for it. */
  schemaVersion?: number
}

/** The file. Every setting lives in a host block; nothing at the top level. */
export interface ConfigFile {
  $schema?: string
  schemaVersion?: number
  hosts?: Record<string, HostConfig>
}

export interface ResolvedConfig {
  host: string
  peerName: string
  workspace: string
  aiPeer: string
  baseUrl: string
  /** Problems in the file the user should know about; see `checkConfig`. */
  warnings: string[]
}

export const DEFAULT_BASE_URL = 'https://api.honcho.dev'
export const CONFIG_SCHEMA_VERSION = 2

function isObj(v: unknown): v is Record<string, any> {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
}

function str(v: unknown): string | undefined {
  return typeof v === 'string' && v ? v : undefined
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

/**
 * Highest wins: `HONCHO_*` env → `hosts.<host>` → `defaults` → built-in.
 * Only this host's block is read. Credentials come from `pickCredential` /
 * `ensureFreshCredential`.
 */
export function resolveConfig(
  file: unknown,
  opts: { host: string; env?: Env; defaults?: { workspace?: string; aiPeer?: string } }
): ResolvedConfig {
  const env = opts.env ?? process.env
  const { host } = opts
  const hosts = isObj(file) && isObj(file.hosts) ? file.hosts : {}
  const block = isObj(hosts[host]) ? hosts[host] : {}
  return {
    host,
    peerName: str(env.HONCHO_PEER_NAME) ?? str(block.peerName) ?? str(env.USER) ?? str(env.USERNAME) ?? 'user',
    workspace: str(env.HONCHO_WORKSPACE) ?? str(block.workspace) ?? opts.defaults?.workspace ?? host,
    aiPeer: str(env.HONCHO_AI_PEER) ?? str(block.aiPeer) ?? opts.defaults?.aiPeer ?? host,
    baseUrl: normalizeBaseUrl(str(env.HONCHO_BASE_URL) ?? str(block.baseUrl) ?? DEFAULT_BASE_URL),
    warnings: checkConfig(file).map((i) => i.message),
  }
}

/**
 * The name a setup screen should offer: the one every other plugin's block
 * agrees on, else the old top-level `peerName`, else `$USER`. For the user to
 * confirm, not to read at runtime.
 */
export function suggestPeerName(file: unknown, env: Env = process.env): string {
  const f = isObj(file) ? file : {}
  const hosts = isObj(f.hosts) ? f.hosts : {}
  const names = new Set(Object.values(hosts).map((b) => (isObj(b) ? str(b.peerName) : undefined)).filter(Boolean))
  if (names.size === 1) return [...names][0] as string
  return str(f.peerName) ?? str(env.USER) ?? str(env.USERNAME) ?? 'user'
}

/**
 * `HONCHO_CONFIG_PATH` if set, returned verbatim. Otherwise `.honcho/config.json`
 * under `HOME`, then `USERPROFILE` (Windows), then `os.homedir()`.
 *
 * `env.HOME` is consulted before `os.homedir()` because Bun's `homedir()` ignores
 * in-process changes to `process.env.HOME`, so tests that redirect HOME would
 * otherwise read and write the real config file.
 */
export function configPath(env: Env = process.env): string {
  if (env.HONCHO_CONFIG_PATH) return env.HONCHO_CONFIG_PATH
  const home = env.HOME || env.USERPROFILE || homedir()
  return join(home, '.honcho', 'config.json')
}

/** The parsed file, or `{}` when it is missing or unreadable. */
export function readConfigFile(env: Env = process.env): Record<string, any> {
  const path = configPath(env)
  if (!existsSync(path)) return {}
  try {
    const parsed = JSON.parse(readFileSync(path, 'utf-8'))
    return isObj(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

export function loadConfig(opts: {
  host: string
  env?: Env
  defaults?: { workspace?: string; aiPeer?: string }
}): ResolvedConfig {
  return resolveConfig(readConfigFile(opts.env), opts)
}

/**
 * Merge `patch` into `hosts.<host>` under the config lock; `undefined` removes a
 * key. The first write to a block also does `migrateHost`'s one-time copy, so the
 * block is never marked without it; `patch` wins for every key it names. Touches
 * nothing else in the file.
 */
export async function saveHost(path: string, host: string, patch: Partial<HostConfig>): Promise<void> {
  await updateConfigFile(path, (file) => {
    if (isNewer(file)) throw new Error(`${path} was written by a newer Honcho plugin; update this one.`)
    const { set, remove } = hostMigration(file, host)
    if (!isObj(file.hosts)) file.hosts = {}
    const block = isObj(file.hosts[host]) ? file.hosts[host] : (file.hosts[host] = {})
    Object.assign(block, set)
    for (const k of remove) delete block[k]
    for (const [k, v] of Object.entries(patch)) {
      if (v === undefined) delete block[k]
      else block[k] = v
    }
    block.schemaVersion = CONFIG_SCHEMA_VERSION
  })
}

// ── migration + checks ──────────────────────────────────────────────────────

export interface ConfigIssue {
  /** Where the problem is: `schemaVersion` or `hosts`. */
  path: string
  message: string
}

const MOVABLE = ['apiKey', 'peerName', 'workspace', 'aiPeer', 'baseUrl'] as const

/** The file was written by a newer version of this package; leave it alone. */
function isNewer(file: unknown): boolean {
  return isObj(file) && typeof file.schemaVersion === 'number' && file.schemaVersion > CONFIG_SCHEMA_VERSION
}

/** A host whose plugin reads only its own block. */
function isMigrated(block: unknown): boolean {
  return isObj(block) && typeof block.schemaVersion === 'number' && block.schemaVersion >= CONFIG_SCHEMA_VERSION
}

type Settings = Partial<Record<(typeof MOVABLE)[number], string>>

/**
 * A block's settings under their v2 names, reading old spellings too. A host's
 * explicit Honcho cloud URL is kept: it is how that host overrides a self-hosted
 * top-level URL, so it must survive until precedence is decided.
 */
function settingsOf(b: Record<string, any>): Settings {
  const endpoint = isObj(b.endpoint) ? b.endpoint : {}
  const environment = endpoint.environment === 'local' ? 'http://localhost:8000' : endpoint.environment === 'production' ? DEFAULT_BASE_URL : undefined
  const out = {
    apiKey: str(b.apiKey) ?? (isObj(b.auth) ? str(b.auth.apiKey) : undefined),
    peerName: str(b.peerName),
    workspace: str(b.workspace) ?? str(b.workspaceId),
    aiPeer: str(b.aiPeer),
    baseUrl: str(b.baseUrl) ?? str(endpoint.baseUrl) ?? environment ?? str(b.environmentUrl),
  }
  return Object.fromEntries(Object.entries(out).filter(([, v]) => v !== undefined))
}

/** Top-level settings to copy into host blocks. Honcho cloud there is the default, so it is not copied. */
function rootSettingsOf(file: Record<string, any>): Settings {
  const out = settingsOf(file)
  if (out.baseUrl && normalizeBaseUrl(out.baseUrl) === DEFAULT_BASE_URL) delete out.baseUrl
  return out
}

/** Old spellings inside a host block. */
const LEGACY_HOST_KEYS = ['auth', 'endpoint', 'environmentUrl', 'workspaceId']

/** A grant stored under the old `auth.oauth`, when the block has none under `oauth`. */
function legacyGrantOf(b: Record<string, any>): Record<string, any> | undefined {
  return b.oauth === undefined && isObj(b.auth) && isObj(b.auth.oauth) ? b.auth.oauth : undefined
}

/**
 * Problems a plugin should tell the user about. Read-only, and nothing to
 * clean: each plugin fixes its own block with `migrateHost`, and the top level
 * belongs to the honcho CLI, whose `honcho doctor` removes it once every
 * integration reads only its own block.
 */
export function checkConfig(file: unknown): ConfigIssue[] {
  if (!isObj(file)) return []
  if (isNewer(file)) {
    return [{ path: 'schemaVersion', message: `This file is schema version ${file.schemaVersion}, newer than this plugin understands; update the plugin.` }]
  }
  const issues: ConfigIssue[] = []
  const hosts: Record<string, any> = isObj(file.hosts) ? file.hosts : {}
  const names = Object.keys(hosts).filter((h) => isObj(hosts[h]))

  // Hosts sharing a workspace must agree on the user.
  const byWorkspace = new Map<string, Map<string, string[]>>()
  for (const h of names) {
    const peer = str(hosts[h].peerName) ?? str(file.peerName)
    if (!peer) continue
    const ws = (file.globalOverride === true && str(file.workspace)) || settingsOf(hosts[h]).workspace || str(file.workspace) || h
    const peers = byWorkspace.get(ws) ?? new Map<string, string[]>()
    peers.set(peer, [...(peers.get(peer) ?? []), h])
    byWorkspace.set(ws, peers)
  }
  for (const [ws, peers] of byWorkspace) {
    if (peers.size < 2) continue
    const list = [...peers].map(([p, hs]) => `${hs.join(', ')}: "${p}"`).join('; ')
    issues.push({ path: 'hosts', message: `Hosts in workspace "${ws}" use different peerNames (${list}), so the user is split across peers.` })
  }
  return issues
}

/**
 * What one plugin can fix by itself: copy top-level settings into its own block,
 * once, while the block is unmarked (its `workspace` only under `globalOverride`,
 * or when it has no block yet),
 * rename old keys in that block, and drop `deadKeys` it no longer uses. Never
 * touches another host's block or the top level, which the honcho CLI cleans.
 */
export function hostMigration(
  file: unknown,
  host: string,
  deadKeys: readonly string[] = []
): { set: Partial<HostConfig>; remove: string[] } {
  if (isNewer(file)) return { set: {}, remove: [] }
  const f = isObj(file) ? file : {}
  const hosts = isObj(f.hosts) ? f.hosts : {}
  const hasBlock = isObj(hosts[host])
  const block: Record<string, any> = hasBlock ? hosts[host] : {}
  const set: Record<string, unknown> = {}
  // The host's own value (an explicit cloud URL included) beats the top level.
  const own = settingsOf(block)
  // The top level is copied once. A marked block already took what it needed, and
  // anything written there later (an older honcho CLI's key or URL) must not reach it.
  const root: Settings = isMigrated(block) ? {} : rootSettingsOf(f)
  for (const key of MOVABLE) {
    if (block[key] !== undefined) continue
    const fromRoot = key === 'workspace' ? (hasBlock && f.globalOverride !== true ? undefined : root.workspace) : root[key]
    const value = own[key] ?? fromRoot
    if (value !== undefined) set[key] = value
  }
  if (f.globalOverride === true && root.workspace && block.workspace !== root.workspace) set.workspace = root.workspace
  const grant = legacyGrantOf(block)
  if (grant) set.oauth = grant as HostConfig['oauth']
  const remove = [...LEGACY_HOST_KEYS, ...deadKeys].filter((k) => k in block && !(k in set))
  // Mark the block as read by a current plugin, so doctor knows it may clean up.
  if ((hasBlock || Object.keys(set).length) && !isMigrated(block)) set.schemaVersion = CONFIG_SCHEMA_VERSION
  return { set: set as Partial<HostConfig>, remove }
}

/**
 * Apply `hostMigration` to the file under the config lock. Returns the settings
 * it moved or removed (not the `schemaVersion` mark); empty when there were none.
 */
export async function migrateHost(path: string, host: string, deadKeys: readonly string[] = []): Promise<string[]> {
  // Cheap check first, so a current block never takes the lock or writes.
  const peek = hostMigration(readConfigFile({ HONCHO_CONFIG_PATH: path }), host, deadKeys)
  if (!Object.keys(peek.set).length && !peek.remove.length) return []
  let changed: string[] = []
  await updateConfigFile(path, (file) => {
    const { set, remove } = hostMigration(file, host, deadKeys)
    changed = [...Object.keys(set).filter((k) => k !== 'schemaVersion'), ...remove.map((k) => `-${k}`)]
    if (!Object.keys(set).length && !remove.length) return
    if (!isObj(file.hosts)) file.hosts = {}
    const block = isObj(file.hosts[host]) ? file.hosts[host] : (file.hosts[host] = {})
    Object.assign(block, set)
    for (const k of remove) delete block[k]
  })
  return changed
}
