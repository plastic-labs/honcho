import { closeSync, existsSync, mkdirSync, openSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { DEFAULT_BASE_URL, normalizeBaseUrl, type Env } from './config.js'

/**
 * OAuth device sign-in for Honcho cloud (RFC 8628).
 * Cloud only: self-hosted Honcho has no OAuth server and keeps using an API key.
 */

export const OAUTH_DEVICE_URL = 'https://api.honcho.dev/oauth/device_authorization'
export const OAUTH_TOKEN_URL = 'https://api.honcho.dev/oauth/token'
export const OAUTH_SCOPE = 'write'

const DEVICE_GRANT_TYPE = 'urn:ietf:params:oauth:grant-type:device_code'
const SLOW_DOWN_STEP_S = 5
const DEFAULT_DEVICE_EXPIRES_S = 600
const REQUEST_TIMEOUT_MS = 10_000
const REFRESH_SKEW_S = 60
const LOCK_STALE_MS = 30_000
const LOCK_WAIT_MS = 15_000
// a retry can never fix these; the grant is dead and the user must sign in again
const PERMANENT_ERRORS = new Set(['invalid_grant', 'invalid_client', 'unauthorized_client'])

/** Stored grant, in `hosts.<host>.oauth`: one per plugin. */
export interface OAuthTokens {
  accessToken: string
  refreshToken: string
  /** Epoch seconds. */
  accessExpiresAt: number
  clientId: string
  scope: string
}

export interface DeviceCode {
  deviceCode: string
  userCode: string
  verificationUri: string
  verificationUriComplete: string
  expiresIn: number
  interval: number
}

export class OAuthError extends Error {
  constructor(public readonly code: string, description?: string) {
    super(description || code)
    this.name = 'OAuthError'
  }
}

/** Test seam; production uses the cloud URLs and global fetch. */
export interface OAuthTransport {
  fetch?: typeof fetch
  deviceUrl?: string
  tokenUrl?: string
}

export function isCloud(baseUrl: string | undefined): boolean {
  return normalizeBaseUrl(baseUrl || DEFAULT_BASE_URL) === DEFAULT_BASE_URL
}

export function accessValid(t: OAuthTokens | undefined, skewS = REFRESH_SKEW_S, nowS = Date.now() / 1000): boolean {
  return !!t?.accessToken && nowS < t.accessExpiresAt - skewS
}

async function post(url: string, form: Record<string, string>, t?: OAuthTransport): Promise<{ status: number; body: any }> {
  const f = t?.fetch ?? fetch
  let res: Response
  try {
    res = await f(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json' },
      body: new URLSearchParams(form).toString(),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    })
  } catch (e) {
    throw new OAuthError('connection_error', `could not reach ${url}: ${(e as Error).message}`)
  }
  let body: any = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  return { status: res.status, body }
}

function errorOf(r: { body: any }): OAuthError {
  const b = r.body
  if (b && typeof b === 'object' && typeof b.error === 'string') return new OAuthError(b.error, b.error_description)
  return new OAuthError('invalid_response')
}

function tokensFrom(body: any, clientId: string, prev?: OAuthTokens): OAuthTokens {
  if (!body || typeof body.access_token !== 'string' || !Number.isFinite(Number(body.expires_in))) {
    throw new OAuthError('invalid_response', 'malformed token response')
  }
  return {
    accessToken: body.access_token,
    // refresh_token is optional on the refresh grant; keep the prior one if absent
    refreshToken: body.refresh_token || prev?.refreshToken || '',
    accessExpiresAt: Math.floor(Date.now() / 1000) + Number(body.expires_in),
    clientId,
    scope: body.scope || prev?.scope || OAUTH_SCOPE,
  }
}

export async function requestDeviceCode(
  opts: { clientId: string; source: string } & OAuthTransport
): Promise<DeviceCode> {
  const r = await post(
    opts.deviceUrl ?? OAUTH_DEVICE_URL,
    { client_id: opts.clientId, scope: OAUTH_SCOPE, source: opts.source },
    opts
  )
  if (r.status !== 200) throw errorOf(r)
  const b = r.body
  if (!b?.device_code || !b?.user_code || !b?.verification_uri) {
    throw new OAuthError('invalid_response', 'malformed device authorization response')
  }
  return {
    deviceCode: b.device_code,
    userCode: b.user_code,
    verificationUri: b.verification_uri,
    verificationUriComplete: b.verification_uri_complete || b.verification_uri,
    // RFC 8628 requires expires_in; without it the poll would never time out
    expiresIn: Number(b.expires_in) > 0 ? Number(b.expires_in) : DEFAULT_DEVICE_EXPIRES_S,
    interval: Number(b.interval) || 5,
  }
}

/** Polls until the user approves. Throws OAuthError `access_denied` / `expired_token` on the terminal outcomes. */
export async function pollForToken(
  device: DeviceCode,
  opts: { clientId: string; sleep?: (ms: number) => Promise<void> } & OAuthTransport
): Promise<OAuthTokens> {
  const clientId = opts.clientId
  const sleep = opts.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)))
  const deadline = Date.now() + device.expiresIn * 1000
  let interval = device.interval
  while (true) {
    if (Date.now() >= deadline) throw new OAuthError('expired_token', 'timed out waiting for approval')
    await sleep(interval * 1000)
    let r: { status: number; body: any }
    try {
      r = await post(
        opts.tokenUrl ?? OAUTH_TOKEN_URL,
        { grant_type: DEVICE_GRANT_TYPE, device_code: device.deviceCode, client_id: clientId },
        opts
      )
    } catch (e) {
      // A dropped request is transient; the device code is still good until the deadline.
      if (e instanceof OAuthError && e.code === 'connection_error') continue
      throw e
    }
    if (r.status === 200) return tokensFrom(r.body, clientId)
    if (r.status >= 500) continue
    const err = errorOf(r)
    if (err.code === 'authorization_pending') continue
    if (err.code === 'slow_down') {
      interval += SLOW_DOWN_STEP_S
      continue
    }
    throw err
  }
}

/** The server rotates the refresh token on every call; persist the result before reusing it. */
export async function refreshAccessToken(prev: OAuthTokens, opts: OAuthTransport = {}): Promise<OAuthTokens> {
  const r = await post(
    opts.tokenUrl ?? OAUTH_TOKEN_URL,
    { grant_type: 'refresh_token', refresh_token: prev.refreshToken, client_id: prev.clientId },
    opts
  )
  if (r.status !== 200) throw errorOf(r)
  return tokensFrom(r.body, prev.clientId, prev)
}

// ── config file ───────────────────────────────────────────────────────────────

function isObj(v: unknown): v is Record<string, any> {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
}

function asTokens(v: unknown): OAuthTokens | undefined {
  if (!isObj(v) || typeof v.accessToken !== 'string' || !v.accessToken) return undefined
  return {
    accessToken: v.accessToken,
    refreshToken: typeof v.refreshToken === 'string' ? v.refreshToken : '',
    accessExpiresAt: Number(v.accessExpiresAt) || 0,
    clientId: typeof v.clientId === 'string' ? v.clientId : '',
    scope: typeof v.scope === 'string' ? v.scope : '',
  }
}

function readFile(path: string): Record<string, any> {
  if (!existsSync(path)) return {}
  const parsed = JSON.parse(readFileSync(path, 'utf-8'))
  return isObj(parsed) ? parsed : {}
}

export function readGrant(file: unknown, host: string): OAuthTokens | undefined {
  if (!isObj(file)) return undefined
  return asTokens(isObj(file.hosts) ? file.hosts[host]?.oauth : undefined)
}

/** Cross-process lock: `<config>.lock` created with O_EXCL; broken when older than 30s. */
export async function withConfigLock<T>(path: string, fn: () => Promise<T>): Promise<T> {
  const lock = `${path}.lock`
  mkdirSync(dirname(path), { recursive: true })
  const start = Date.now()
  let fd: number | undefined
  while (fd === undefined) {
    try {
      fd = openSync(lock, 'wx', 0o600)
    } catch (e) {
      if ((e as { code?: string }).code !== 'EEXIST') throw e
      try {
        if (Date.now() - statSync(lock).mtimeMs > LOCK_STALE_MS) unlinkSync(lock)
      } catch {}
      if (Date.now() - start > LOCK_WAIT_MS) throw new Error(`timed out waiting for ${lock}`)
      await new Promise((r) => setTimeout(r, 50 + Math.random() * 50))
    }
  }
  try {
    writeFileSync(fd, String(process.pid))
    return await fn()
  } finally {
    closeSync(fd)
    try {
      unlinkSync(lock)
    } catch {}
  }
}

/** Atomic write (temp file + rename), mode 0600. */
function writeFileAtomic(path: string, data: Record<string, any>): void {
  const tmp = join(dirname(path), `.${Date.now()}-${process.pid}-${Math.random().toString(36).slice(2)}.tmp`)
  writeFileSync(tmp, JSON.stringify(data, null, 2) + '\n', { mode: 0o600 })
  renameSync(tmp, path)
}

/** Locked read-modify-write of the config file. Refuses to overwrite a file it cannot parse. */
export async function updateConfigFile(path: string, mutate: (file: Record<string, any>) => void): Promise<void> {
  await withConfigLock(path, async () => {
    const file = readFile(path)
    mutate(file)
    writeFileAtomic(path, file)
  })
}

function setGrant(file: Record<string, any>, host: string, tokens: OAuthTokens | null): void {
  if (!isObj(file.hosts)) file.hosts = {}
  const block = isObj(file.hosts[host]) ? file.hosts[host] : (file.hosts[host] = {})
  if (tokens) block.oauth = tokens
  else delete block.oauth
}

/** Store (or with `null`, remove) a grant. Touches only that block. */
export async function saveGrant(path: string, host: string, tokens: OAuthTokens | null): Promise<void> {
  await updateConfigFile(path, (file) => setGrant(file, host, tokens))
}

// ── resolution + refresh ──────────────────────────────────────────────────────

export type CredentialSource = 'env' | 'oauth' | 'apiKey'

export interface Credential {
  token: string
  source: CredentialSource
  /** Expiry of an OAuth access token, epoch seconds. */
  expiresAt?: number
  /** The OAuth access token is past expiry (sent anyway; the server is the judge). */
  expired?: boolean
  /** The server rejected the grant; the user must sign in again. */
  reauthRequired?: boolean
  /** OAuth error code from the last refresh attempt. */
  error?: string
}

function apiKeyOf(file: Record<string, any>, host: string): string | undefined {
  const hb = isObj(file.hosts) ? file.hosts[host] : undefined
  const k = hb?.apiKey
  return typeof k === 'string' && k ? k : undefined
}

/**
 * The token to send, highest wins: HONCHO_API_KEY env → the plugin's live OAuth
 * grant (`hosts.<host>.oauth`) → `hosts.<host>.apiKey` → its expired grant. OAuth is skipped
 * unless `baseUrl` is Honcho cloud. No network.
 */
export function pickCredential(file: unknown, opts: { host: string; baseUrl?: string; env?: Env }): Credential | null {
  const env = opts.env ?? process.env
  if (env.HONCHO_API_KEY) return { token: env.HONCHO_API_KEY, source: 'env' }
  const f = isObj(file) ? file : {}
  const g = isCloud(opts.baseUrl) ? readGrant(f, opts.host) : undefined
  if (g && accessValid(g, 0)) return { token: g.accessToken, source: 'oauth', expiresAt: g.accessExpiresAt }
  const key = apiKeyOf(f, opts.host)
  if (key) return { token: key, source: 'apiKey' }
  if (g) return { token: g.accessToken, source: 'oauth', expiresAt: g.accessExpiresAt, expired: true }
  return null
}

/**
 * Like pickCredential, but first refreshes an OAuth grant that is within 60s of
 * expiry. The refresh runs under the config lock and re-reads the file first, so
 * concurrent processes refresh once and the rest adopt the result. Never throws
 * for OAuth failures; they surface as `reauthRequired` / `error`.
 *
 * Every refresh supersedes the grant's previous access token (the gateway's
 * cache may accept it for up to 5 more minutes), so long-lived callers should
 * call this before each batch of requests rather than cache a token.
 */
export async function ensureFreshCredential(
  opts: { host: string; baseUrl?: string; env?: Env; path: string } & OAuthTransport
): Promise<Credential | null> {
  const env = opts.env ?? process.env
  if (env.HONCHO_API_KEY) return { token: env.HONCHO_API_KEY, source: 'env' }
  let file: Record<string, any>
  try {
    file = readFile(opts.path)
  } catch {
    file = {}
  }
  const g = isCloud(opts.baseUrl) ? readGrant(file, opts.host) : undefined
  if (!g || accessValid(g) || !g.refreshToken) return pickCredential(file, opts)

  let error: string | undefined
  try {
    await withConfigLock(opts.path, async () => {
      const current = readGrant(readFile(opts.path), opts.host)
      if (!current || accessValid(current)) return // another process already refreshed
      const next = await refreshAccessToken(current, opts)
      const latest = readFile(opts.path)
      setGrant(latest, opts.host, next)
      writeFileAtomic(opts.path, latest)
    })
  } catch (e) {
    error = e instanceof OAuthError ? e.code : (e as Error).message
  }
  try {
    file = readFile(opts.path)
  } catch {}
  const cred = pickCredential(file, opts)
  if (cred && error) {
    cred.error = error
    if (PERMANENT_ERRORS.has(error)) cred.reauthRequired = true
  }
  return cred
}
