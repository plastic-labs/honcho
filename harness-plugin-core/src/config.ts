import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { v01Reader } from './legacy/v01.js'
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
 * Highest wins: `HONCHO_*` env → `hosts.<host>` → `defaults` → built-in. Only
 * this host's block is read; a block not migrated yet is read as `migrateHost`
 * will write it (see `effectiveBlock`), so reading first never gives a different
 * answer. Credentials come from `pickCredential` / `ensureFreshCredential`.
 */
export function resolveConfig(
  file: unknown,
  opts: { host: string; env?: Env; defaults?: { workspace?: string; aiPeer?: string }; legacy?: LegacyReader }
): ResolvedConfig {
  const env = opts.env ?? process.env
  const { host } = opts
  const block = effectiveBlock(file, host, opts.legacy)
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
  legacy?: LegacyReader
}): ResolvedConfig {
  return resolveConfig(readConfigFile(opts.env), opts)
}

/** What a plugin used before schema 2. `migrateHost` writes exactly this into its block, once. */
export type LegacySettings = Partial<Pick<HostConfig, 'apiKey' | 'peerName' | 'workspace' | 'aiPeer' | 'baseUrl'>>

/**
 * A plugin's pre-schema-2 reader: from the whole file, what its last release
 * would have used. One per plugin release line, in `src/legacy/`: `v01Reader` for
 * plugins that read the file through 0.1.x of this package, and the plugins' own.
 */
export type LegacyReader = (file: Record<string, any>, host: string) => LegacySettings

/**
 * Merge `patch` into `hosts.<host>` under the config lock; `undefined` removes a
 * key. A block not migrated yet is migrated first, with `legacy`. A key belongs to
 * the server it was saved for, so a patch that moves the block to another server
 * drops the key unless it sets one. Touches nothing else in the file.
 */
export async function saveHost(path: string, host: string, patch: Partial<HostConfig>, legacy?: LegacyReader): Promise<void> {
  await updateConfigFile(path, (file) => {
    if (isNewer(file)) throw new Error(`${path} was written by a newer Honcho plugin; update this one.`)
    const block = writeBlock(file, host, hostMigration(file, host, [], legacy))
    if ('baseUrl' in patch && !('apiKey' in patch) && !sameServer(patch.baseUrl, str(block.baseUrl))) delete block.apiKey
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

/** Two URLs name the same server; missing means Honcho cloud. */
function sameServer(a: string | undefined, b: string | undefined): boolean {
  return normalizeBaseUrl(a ?? DEFAULT_BASE_URL) === normalizeBaseUrl(b ?? DEFAULT_BASE_URL)
}

/** Old spellings inside a host block. */
const LEGACY_HOST_KEYS = ['auth', 'endpoint', 'environmentUrl', 'workspaceId']

/** A stored grant with an access token; `null` or `{}` is no grant. */
function isGrant(v: unknown): boolean {
  return isObj(v) && typeof v.accessToken === 'string' && v.accessToken !== ''
}

/** A grant stored under the old `auth.oauth`, when the block has none under `oauth`. */
function legacyGrantOf(b: Record<string, any>): Record<string, any> | undefined {
  return !isGrant(b.oauth) && isObj(b.auth) && isObj(b.auth.oauth) ? b.auth.oauth : undefined
}

/**
 * Problems a plugin should tell the user about. Read-only, and nothing to
 * clean: each plugin fixes its own block with `migrateHost`, and the top level
 * belongs to the honcho CLI.
 */
export function checkConfig(file: unknown): ConfigIssue[] {
  if (!isObj(file)) return []
  if (isNewer(file)) {
    return [{ path: 'schemaVersion', message: `This file is schema version ${file.schemaVersion}, newer than this plugin understands; update the plugin.` }]
  }
  const issues: ConfigIssue[] = []
  const hosts: Record<string, any> = isObj(file.hosts) ? file.hosts : {}

  // Hosts sharing a workspace must agree on the user.
  const byWorkspace = new Map<string, Map<string, string[]>>()
  for (const h of Object.keys(hosts).filter((k) => isObj(hosts[k]))) {
    const block = effectiveBlock(file, h)
    const peer = str(block.peerName)
    if (!peer) continue
    const ws = str(block.workspace) ?? h
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
 * What a plugin's block needs. Once, while it is unmarked: exactly what the
 * plugin used before (`legacy`, `v01Reader` by default), and the mark. A block
 * already signed in keeps its grant on Honcho cloud rather than the key and
 * server it read before. Every time: an old `auth.oauth` grant moves to `oauth`,
 * and old key names and `deadKeys` go. Never touches another host's block or the
 * top level, which the honcho CLI cleans.
 */
export function hostMigration(
  file: unknown,
  host: string,
  deadKeys: readonly string[] = [],
  legacy: LegacyReader = v01Reader
): { set: Partial<HostConfig>; remove: string[] } {
  if (isNewer(file)) return { set: {}, remove: [] }
  const f = isObj(file) ? file : {}
  const hosts = isObj(f.hosts) ? f.hosts : {}
  const block: Record<string, any> = isObj(hosts[host]) ? hosts[host] : {}
  const set: Record<string, unknown> = {}
  const drop: string[] = []
  const grant = legacyGrantOf(block)
  if (grant) set.oauth = grant
  if (!isMigrated(block)) {
    const old = { ...legacy(f, host) }
    const signedIn = isGrant(block.oauth) || grant !== undefined
    if (old.baseUrl && sameServer(old.baseUrl, undefined)) old.baseUrl = undefined
    for (const key of MOVABLE) {
      if (signedIn && (key === 'apiKey' || key === 'baseUrl')) continue
      if (old[key] !== undefined && block[key] !== old[key]) set[key] = old[key]
      else if (old[key] === undefined && block[key] !== undefined) drop.push(key)
    }
    set.schemaVersion = CONFIG_SCHEMA_VERSION
  }
  const remove = [...drop, ...LEGACY_HOST_KEYS, ...deadKeys].filter((k) => k in block && !(k in set))
  return { set: set as Partial<HostConfig>, remove }
}

/** `hosts.<host>` as its plugin reads it: as stored once marked, else as `hostMigration` will leave it. */
export function effectiveBlock(file: unknown, host: string, legacy?: LegacyReader): Record<string, any> {
  return writeBlock(structuredClone(isObj(file) ? file : {}), host, hostMigration(file, host, [], legacy))
}

/** Apply a migration to `file`, creating the block if needed, and return the block. */
/** `hosts.<host>` in `file`, created if missing. Refuses names that would reach `Object.prototype`. */
export function hostBlock(file: Record<string, any>, host: string): Record<string, any> {
  if (host === '__proto__' || host === 'constructor' || host === 'prototype') throw new Error(`invalid host name: ${host}`)
  if (!isObj(file.hosts)) file.hosts = {}
  return isObj(file.hosts[host]) ? file.hosts[host] : (file.hosts[host] = {})
}

function writeBlock(file: Record<string, any>, host: string, { set, remove }: { set: Partial<HostConfig>; remove: string[] }): Record<string, any> {
  const block = hostBlock(file, host)
  Object.assign(block, set)
  for (const k of remove) delete block[k]
  return block
}

/**
 * Apply `hostMigration` under the config lock. Call it at startup, before reading
 * the config or showing setup. Pass the plugin's own reader as `legacy` when its
 * last release did not read the file through 0.1.x of this package. Returns the
 * settings it moved or removed (not the mark); empty when there were none.
 */
export async function migrateHost(path: string, host: string, deadKeys: readonly string[] = [], legacy?: LegacyReader): Promise<string[]> {
  // Cheap check first, so a current block never takes the lock or writes.
  const peek = hostMigration(readConfigFile({ HONCHO_CONFIG_PATH: path }), host, deadKeys, legacy)
  if (!Object.keys(peek.set).length && !peek.remove.length) return []
  let changed: string[] = []
  await updateConfigFile(path, (file) => {
    const m = hostMigration(file, host, deadKeys, legacy)
    changed = [...Object.keys(m.set).filter((k) => k !== 'schemaVersion'), ...m.remove.map((k) => `-${k}`)]
    if (Object.keys(m.set).length || m.remove.length) writeBlock(file, host, m)
  })
  return changed
}
