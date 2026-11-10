import { describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'
import { checkConfig, configPath, migrateHost, normalizeBaseUrl, resolveConfig, saveGrant, saveHost } from '../src/index'

const emptyEnv = {}
const baseUrlOf = (file: unknown, host: string) => resolveConfig(file, { host, env: emptyEnv }).baseUrl

describe('normalizeBaseUrl', () => {
  test('adds https and lowercases the host', () => {
    expect(normalizeBaseUrl('api.honcho.dev')).toBe('https://api.honcho.dev')
    expect(normalizeBaseUrl('API.honcho.dev')).toBe('https://api.honcho.dev')
    expect(normalizeBaseUrl('https://api.honcho.dev/')).toBe('https://api.honcho.dev')
  })

  test('leaves /v3 alone — the SDK owns the API version', () => {
    expect(normalizeBaseUrl('https://api.honcho.dev/v3')).toBe('https://api.honcho.dev/v3')
  })

  test('localhost stays http', () => {
    expect(normalizeBaseUrl('localhost:8000')).toBe('http://localhost:8000')
  })
})

describe('resolveConfig', () => {
  test('reads only its own host block; env beats it', () => {
    const file = { workspace: 'root-ws', peerName: 'root-peer', hosts: { a: { workspace: 'host-ws' } } }
    const cfg = resolveConfig(file, { host: 'a', env: { USER: 'u' } })
    expect(cfg.workspace).toBe('host-ws')
    expect(cfg.peerName).toBe('u')
    expect(resolveConfig(file, { host: 'a', env: { HONCHO_WORKSPACE: 'env-ws' } }).workspace).toBe('env-ws')
  })

  test('defaults, then the host name', () => {
    const cfg = resolveConfig({}, { host: 'my-host', env: emptyEnv, defaults: { aiPeer: 'claude' } })
    expect(cfg.baseUrl).toBe('https://api.honcho.dev')
    expect(cfg.workspace).toBe('my-host')
    expect(cfg.aiPeer).toBe('claude')
    expect(cfg.warnings).toEqual([])
  })
})

describe('saveHost', () => {
  test('merges into one block, removes undefined keys, leaves the rest alone', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    writeFileSync(path, JSON.stringify({ hosts: { a: { workspace: 'w', aiPeer: 'x' }, b: { workspace: 'other' } } }))
    await saveHost(path, 'a', { peerName: 'alice', aiPeer: undefined })
    expect(JSON.parse(readFileSync(path, 'utf-8'))).toEqual({
      hosts: { a: { workspace: 'w', peerName: 'alice', schemaVersion: 2 }, b: { workspace: 'other' } },
    })
  })

  test('a first write takes the top level once, never splitting a key from its server', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    const top = { apiKey: 'local-key', peerName: 'u', environmentUrl: 'http://localhost:8000' }
    writeFileSync(path, JSON.stringify(top))
    await saveHost(path, 'a', { peerName: 'alice', baseUrl: undefined }) // setup chose cloud: the local key stays behind
    expect(JSON.parse(readFileSync(path, 'utf-8')).hosts.a).toEqual({ peerName: 'alice', schemaVersion: 2 })
    writeFileSync(path, JSON.stringify(top))
    await saveHost(path, 'a', { apiKey: 'new-key' }) // a key entered in setup does not move to the local URL
    expect(JSON.parse(readFileSync(path, 'utf-8')).hosts.a).toEqual({ apiKey: 'new-key', peerName: 'u', schemaVersion: 2 })
  })
})

describe('migrateHost', () => {
  test("a host's explicit cloud choice beats a self-hosted top-level URL", async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    writeFileSync(path, JSON.stringify({ baseUrl: 'http://localhost:8000', hosts: { a: { endpoint: { environment: 'production' } } } }))
    await migrateHost(path, 'a')
    const file = JSON.parse(readFileSync(path, 'utf-8'))
    expect(file.hosts.a).toEqual({ baseUrl: 'https://api.honcho.dev', schemaVersion: 2 })
    expect(baseUrlOf(file, 'a')).toBe('https://api.honcho.dev')
  })

  test('a marked block never takes top-level values written later', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    const block = { oauth: { accessToken: 'at', refreshToken: 'rt', accessExpiresAt: 1, clientId: 'c', scope: 'write' }, schemaVersion: 2 }
    // e.g. an older honcho CLI's `honcho init` against another tenant and a local stack
    writeFileSync(path, JSON.stringify({ apiKey: 'other-tenant', environmentUrl: 'http://localhost:8000', hosts: { a: block } }))
    expect(await migrateHost(path, 'a')).toEqual([])
    expect(JSON.parse(readFileSync(path, 'utf-8')).hosts.a).toEqual(block)
  })

  test('a block made by sign-in takes no key or server from the top level', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    const grant = { accessToken: 'at', refreshToken: 'rt', accessExpiresAt: 1, clientId: 'c', scope: 'write' }
    await saveGrant(path, 'a', grant) // writes the block without marking it
    const file = JSON.parse(readFileSync(path, 'utf-8'))
    writeFileSync(path, JSON.stringify({ ...file, apiKey: 'cli-key', environmentUrl: 'http://localhost:8000' })) // then `honcho init`
    await migrateHost(path, 'a')
    expect(JSON.parse(readFileSync(path, 'utf-8')).hosts.a).toEqual({ oauth: grant, schemaVersion: 2 })
  })

  test("0.1.x's top-level auth.apiKey and baseUrl beat the CLI's apiKey and environmentUrl", async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    const top = { auth: { apiKey: 'plugin-key' }, baseUrl: 'https://honcho.example.com', apiKey: 'cli-key', environmentUrl: 'http://localhost:8000' }
    writeFileSync(path, JSON.stringify({ ...top, hosts: { a: {} } }))
    await migrateHost(path, 'a')
    expect(JSON.parse(readFileSync(path, 'utf-8')).hosts.a).toEqual({ apiKey: 'plugin-key', baseUrl: 'https://honcho.example.com', schemaVersion: 2 })
  })

  test('a grant under the old auth.oauth moves to oauth', async () => {
    const grant = { accessToken: 'at', refreshToken: 'rt', accessExpiresAt: 1, clientId: 'c', scope: 'write' }
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    writeFileSync(path, JSON.stringify({ hosts: { a: { auth: { oauth: grant } } } }))
    await migrateHost(path, 'a')
    expect(JSON.parse(readFileSync(path, 'utf-8')).hosts.a).toEqual({ oauth: grant, schemaVersion: 2 })
  })
})

describe('checkConfig', () => {
  test('reports a user split across peers and a newer file; leaves the top level to the CLI', () => {
    const file = { apiKey: 'k', environmentUrl: 'http://localhost:8000', hosts: { a: { workspace: 'w', peerName: 'x', endpoint: {} }, b: { workspace: 'w', peerName: 'y' } } }
    expect(checkConfig(file).map((i) => i.path)).toEqual(['hosts'])
    expect(checkConfig({ schemaVersion: 3 }).map((i) => i.path)).toEqual(['schemaVersion'])
  })
})

describe('configPath', () => {
  test('HONCHO_CONFIG_PATH, then $HOME, then os.homedir()', () => {
    expect(configPath({ HONCHO_CONFIG_PATH: '/x/cfg.json', HOME: '/h' })).toBe('/x/cfg.json')
    expect(configPath({ HOME: '/scratch' })).toBe('/scratch/.honcho/config.json')
    expect(configPath({})).toBe(join(homedir(), '.honcho', 'config.json'))
  })
})
