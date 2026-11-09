import { describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'
import { configPath, normalizeBaseUrl, resolveConfig, saveHost } from '../src/index'

const emptyEnv = {}

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

describe('configPath', () => {
  test('HONCHO_CONFIG_PATH, then $HOME, then os.homedir()', () => {
    expect(configPath({ HONCHO_CONFIG_PATH: '/x/cfg.json', HOME: '/h' })).toBe('/x/cfg.json')
    expect(configPath({ HOME: '/scratch' })).toBe('/scratch/.honcho/config.json')
    expect(configPath({})).toBe(join(homedir(), '.honcho', 'config.json'))
  })
})
