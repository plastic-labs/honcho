import { describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'
import { applyFixes, checkConfig, configPath, migrateHost, normalizeBaseUrl, resolveConfig, saveHost } from '../src/index'

const emptyEnv = {}
const doctor = (file: Record<string, any>) => applyFixes(file, checkConfig(file))
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
})

describe('checkConfig + applyFixes', () => {
  test("keeps the CLI's top-level sign-in until hosts.honcho-cli is marked", () => {
    const grant = { accessToken: 'at', refreshToken: 'rt', accessExpiresAt: 1, clientId: 'c', scope: 'write' }
    const file = { apiKey: 'k', environmentUrl: 'http://localhost:8000', oauth: grant, peerName: 'u', hosts: { a: { schemaVersion: 2 } } }
    const after = doctor(file)
    expect(after).toEqual({
      apiKey: 'k',
      environmentUrl: 'http://localhost:8000',
      oauth: grant,
      hosts: { a: { schemaVersion: 2, apiKey: 'k', peerName: 'u', baseUrl: 'http://localhost:8000' } },
      schemaVersion: 2,
    })
    // Once the CLI reads its own block, its settings move there and the top level goes.
    const moved = doctor({ ...file, hosts: { ...file.hosts, 'honcho-cli': { schemaVersion: 2 } } })
    expect(Object.keys(moved).sort()).toEqual(['hosts', 'schemaVersion'])
    expect(moved.hosts['honcho-cli']).toMatchObject({ apiKey: 'k', baseUrl: 'http://localhost:8000', oauth: grant })
  })

  test('an explicit cloud URL keeps a host on cloud; an empty endpoint takes the top-level URL', () => {
    const file = {
      baseUrl: 'http://localhost:8000',
      hosts: { a: { schemaVersion: 2, endpoint: { baseUrl: 'https://api.honcho.dev' } }, b: { schemaVersion: 2, endpoint: {} } },
    }
    const after = doctor(file)
    expect(after.baseUrl).toBeUndefined()
    expect(baseUrlOf(after, 'a')).toBe('https://api.honcho.dev')
    expect(baseUrlOf(after, 'b')).toBe('http://localhost:8000')
  })
})

describe('configPath', () => {
  test('HONCHO_CONFIG_PATH, then $HOME, then os.homedir()', () => {
    expect(configPath({ HONCHO_CONFIG_PATH: '/x/cfg.json', HOME: '/h' })).toBe('/x/cfg.json')
    expect(configPath({ HOME: '/scratch' })).toBe('/scratch/.honcho/config.json')
    expect(configPath({})).toBe(join(homedir(), '.honcho', 'config.json'))
  })
})
