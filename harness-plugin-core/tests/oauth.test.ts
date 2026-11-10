import { describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import {
  ensureFreshCredential,
  pickCredential,
  pollForToken,
  readGrant,
  requestDeviceCode,
  saveGrant,
  withConfigLock,
  OAuthError,
  type OAuthTokens,
} from '../src/oauth.js'

const now = () => Math.floor(Date.now() / 1000)
const fast = { sleep: async () => {} }
const CLIENT_ID = 'test-plugin'

/** Scripted fetch: returns the queued responses in order and records each request's form body. */
function scripted(...responses: Array<[number, unknown]>) {
  const calls: URLSearchParams[] = []
  const fetch = (async (_url: string, init?: RequestInit) => {
    calls.push(new URLSearchParams(String(init?.body ?? '')))
    const next = responses.shift()
    if (!next) throw new Error('unexpected request')
    return new Response(JSON.stringify(next[1]), { status: next[0] })
  }) as unknown as typeof globalThis.fetch
  return { fetch, calls }
}

const tokenBody = (access: string, refresh: string) => ({ access_token: access, refresh_token: refresh, expires_in: 3600, scope: 'write' })
const device = { device_code: 'dev', user_code: 'ABCD-EFGH', verification_uri: 'https://app.honcho.dev/device', expires_in: 600, interval: 5 }

function tmpConfig(data: unknown): string {
  const path = join(mkdtempSync(join(tmpdir(), 'hpc-oauth-')), 'config.json')
  writeFileSync(path, JSON.stringify(data, null, 2))
  return path
}
const read = (p: string) => JSON.parse(readFileSync(p, 'utf-8'))
const grant = (access: string, expiresAt: number, clientId = CLIENT_ID): OAuthTokens => ({
  accessToken: access,
  refreshToken: `${access}-r`,
  accessExpiresAt: expiresAt,
  clientId,
  scope: 'write',
})

describe('device sign-in', () => {
  test('sends client id, scope and source; polls through pending and slow_down', async () => {
    const s = scripted(
      [200, device],
      [400, { error: 'authorization_pending' }],
      [400, { error: 'slow_down' }],
      [200, tokenBody('hch-at-1', 'hch-rt-1')]
    )
    const d = await requestDeviceCode({ clientId: CLIENT_ID, source: 'claude-code', fetch: s.fetch })
    expect(s.calls[0].get('client_id')).toBe(CLIENT_ID)
    expect(s.calls[0].get('scope')).toBe('write')
    expect(s.calls[0].get('source')).toBe('claude-code')

    const waits: number[] = []
    const tokens = await pollForToken(d, { clientId: CLIENT_ID, fetch: s.fetch, sleep: async (ms) => void waits.push(ms) })
    expect(waits).toEqual([5000, 5000, 10000])
    expect(s.calls[1].get('device_code')).toBe('dev')
    expect(tokens).toMatchObject({ accessToken: 'hch-at-1', refreshToken: 'hch-rt-1', clientId: CLIENT_ID })
    expect(tokens.accessExpiresAt).toBeGreaterThan(now() + 3000)
  })

  test('a denied request throws access_denied', async () => {
    const s = scripted([400, { error: 'access_denied' }])
    const err = await pollForToken(
      { deviceCode: 'dev', userCode: 'X', verificationUri: '', verificationUriComplete: '', expiresIn: 600, interval: 1 },
      { clientId: CLIENT_ID, fetch: s.fetch, ...fast }
    ).catch((e) => e)
    expect(err).toBeInstanceOf(OAuthError)
    expect(err.code).toBe('access_denied')
  })
})

describe('pickCredential', () => {
  const file = (own?: OAuthTokens, apiKey?: string) => ({
    hosts: { claude_code: { ...(own ? { oauth: own } : {}), ...(apiKey ? { apiKey } : {}) } },
  })
  const pick = (f: unknown, env: Record<string, string> = {}, baseUrl?: string) =>
    pickCredential(f, { host: 'claude_code', env, baseUrl })

  test('HONCHO_API_KEY env beats everything', () => {
    expect(pick(file(grant('own', now() + 600)), { HONCHO_API_KEY: 'envkey' })).toMatchObject({ token: 'envkey', source: 'env' })
  })
  test('own live grant, then apiKey, then the expired grant', () => {
    expect(pick(file(grant('own', now() + 600), 'k'))).toMatchObject({ token: 'own', source: 'oauth' })
    expect(pick(file(grant('own', now() - 10), 'k'))).toMatchObject({ token: 'k', source: 'apiKey' })
    expect(pick(file(grant('own', now() - 10)))).toMatchObject({ token: 'own', source: 'oauth', expired: true })
    expect(pick({})).toBeNull()
  })
  test("another plugin's grant and a root grant or key are never used", () => {
    const f = { oauth: grant('root', now() + 600), hosts: { hermes: { oauth: grant('hermes', now() + 600) } } }
    expect(pick(f)).toBeNull()
    expect(pick({ apiKey: 'root', auth: { apiKey: 'root' }, hosts: { claude_code: {} } })).toBeNull()
  })
  test('OAuth is ignored off Honcho cloud', () => {
    expect(pick(file(grant('own', now() + 600), 'k'), {}, 'http://localhost:8000')).toMatchObject({ source: 'apiKey' })
    expect(pick(file(grant('own', now() + 600)), {}, 'https://api.honcho.dev/')).toMatchObject({ source: 'oauth' })
  })
})

describe('ensureFreshCredential', () => {
  test('refreshes an expired own grant and rewrites only that block', async () => {
    const old = grant('hch-at-old', now() - 5)
    const hermes = grant('hch-at-hermes', now() + 600, 'hermes-agent')
    const path = tmpConfig({ apiKey: 'shared', hosts: { claude_code: { workspace: 'w', oauth: old }, hermes: { oauth: hermes } } })
    const s = scripted([200, tokenBody('hch-at-new', 'hch-rt-new')])

    const cred = await ensureFreshCredential({ path, host: 'claude_code', env: {}, fetch: s.fetch })
    expect(cred).toMatchObject({ token: 'hch-at-new', source: 'oauth' })
    expect(s.calls[0].get('grant_type')).toBe('refresh_token')
    expect(s.calls[0].get('refresh_token')).toBe(old.refreshToken)
    expect(s.calls[0].get('client_id')).toBe(CLIENT_ID)

    const f = read(path)
    expect(f.hosts.claude_code.oauth).toMatchObject({ accessToken: 'hch-at-new', refreshToken: 'hch-rt-new' })
    expect(f.hosts.claude_code.workspace).toBe('w')
    expect(f.hosts.hermes).toEqual({ oauth: hermes })
    expect(f.apiKey).toBe('shared')
    expect(statSync(path).mode & 0o777).toBe(0o600)
  })

  test('a live grant is returned without a network call', async () => {
    const path = tmpConfig({ hosts: { claude_code: { oauth: grant('hch-at-live', now() + 600) } } })
    const s = scripted()
    expect(await ensureFreshCredential({ path, host: 'claude_code', env: {}, fetch: s.fetch })).toMatchObject({ token: 'hch-at-live' })
    expect(s.calls).toHaveLength(0)
  })

  test('a grant another process rotated while we waited for the lock is adopted, not refreshed again', async () => {
    const path = tmpConfig({ hosts: { claude_code: { oauth: grant('hch-at-old', now() - 5) } } })
    const s = scripted() // any refresh request fails the test
    let pending!: Promise<unknown>
    await withConfigLock(path, async () => {
      // We read the stale grant and queue behind the lock "another process" holds...
      pending = ensureFreshCredential({ path, host: 'claude_code', env: {}, fetch: s.fetch })
      await new Promise((r) => setTimeout(r, 100))
      // ...which rotates the grant before releasing it.
      writeFileSync(path, JSON.stringify({ hosts: { claude_code: { oauth: grant('hch-at-rotated', now() + 600) } } }))
    })
    expect(await pending).toMatchObject({ token: 'hch-at-rotated', source: 'oauth' })
    expect(s.calls).toHaveLength(0)
  })

  test('a rejected grant flags reauthRequired and falls back to apiKey', async () => {
    const path = tmpConfig({ hosts: { claude_code: { apiKey: 'k', oauth: grant('hch-at-x', now() - 5) } } })
    const s = scripted([400, { error: 'invalid_grant' }])
    const cred = await ensureFreshCredential({ path, host: 'claude_code', env: {}, fetch: s.fetch })
    expect(cred).toMatchObject({ token: 'k', source: 'apiKey', reauthRequired: true, error: 'invalid_grant' })
  })

  test('an unreachable server is not a dead grant: the expired token is sent, no reauth', async () => {
    const path = tmpConfig({ hosts: { claude_code: { oauth: grant('hch-at-x', now() - 5) } } })
    const fetch = (async () => {
      throw new Error('ECONNREFUSED')
    }) as unknown as typeof globalThis.fetch
    const cred = await ensureFreshCredential({ path, host: 'claude_code', env: {}, fetch })
    expect(cred).toMatchObject({ token: 'hch-at-x', expired: true, error: 'connection_error' })
    expect(cred?.reauthRequired).toBeUndefined()
  })
})

describe('saveGrant / readGrant', () => {
  test('stores and removes a host grant without touching other keys', async () => {
    const path = tmpConfig({ apiKey: 'k', hosts: { claude_code: { peerName: 'p' } } })
    const g = grant('a', 1)
    await saveGrant(path, 'claude_code', g)
    expect(readGrant(read(path), 'claude_code')).toEqual(g)
    await saveGrant(path, 'claude_code', null)
    expect(read(path)).toEqual({ apiKey: 'k', hosts: { claude_code: { peerName: 'p' } } })
  })

  test('refuses to overwrite a config file it cannot parse', async () => {
    const path = tmpConfig({})
    writeFileSync(path, '{ not json')
    await expect(saveGrant(path, 'h', grant('a', 1))).rejects.toThrow()
    expect(readFileSync(path, 'utf-8')).toBe('{ not json')
  })
})
