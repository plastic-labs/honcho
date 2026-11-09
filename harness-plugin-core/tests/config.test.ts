import { describe, expect, test } from 'bun:test'
import { existsSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'
import { applyFixes, checkConfig, configPath, hostMigration, migrateHost, normalizeBaseUrl, resolveConfig, saveHost } from '../src/index'

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

  test('peerName falls back to another host before $USER', () => {
    const file = { hosts: { a: {}, b: { peerName: 'alice' } } }
    expect(resolveConfig(file, { host: 'a', env: { USER: 'u' } }).peerName).toBe('alice')
  })

  test('defaults, then the host name', () => {
    const cfg = resolveConfig({}, { host: 'my-host', env: emptyEnv, defaults: { aiPeer: 'claude' } })
    expect(cfg.baseUrl).toBe('https://api.honcho.dev')
    expect(cfg.workspace).toBe('my-host')
    expect(cfg.aiPeer).toBe('claude')
    expect(cfg.warnings).toEqual([])
  })

  test('warns about a file that needs honcho doctor', () => {
    expect(resolveConfig({ apiKey: 'k', hosts: { a: {} } }, { host: 'a', env: emptyEnv }).warnings).not.toEqual([])
  })
})

describe('checkConfig / applyFixes', () => {
  const dir = join(import.meta.dir, 'fixtures', 'config')
  for (const name of readdirSync(dir).filter((f) => f.endsWith('.json'))) {
    test(name, () => {
      const { input, issues, fixed } = JSON.parse(readFileSync(join(dir, name), 'utf-8'))
      const found = checkConfig(input)
      expect(found.map((i) => i.path)).toEqual(issues)
      const out = applyFixes(input, found)
      expect(out).toEqual(fixed)
      // Fixing again changes nothing.
      expect(applyFixes(out, checkConfig(out))).toEqual(out)
    })
  }
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

describe('hostMigration / migrateHost', () => {
  const file = {
    apiKey: 'root-key', peerName: 'alice', workspace: 'root-ws', sessions: { '/x': 's' },
    hosts: {
      claude_code: { workspace: 'cc', endpoint: { baseUrl: 'http://localhost:8000' }, sessionStrategy: 'chat-instance' },
      dsh: { auth: { apiKey: 'dsh-key' } },
    },
  }

  test('copies top-level settings into its own block and renames old keys there', () => {
    expect(hostMigration(file, 'claude_code', ['sessionStrategy'])).toEqual({
      set: { apiKey: 'root-key', peerName: 'alice', baseUrl: 'http://localhost:8000', schemaVersion: 2 },
      remove: ['endpoint', 'sessionStrategy'],
    })
  })

  test('a host with no block also takes the top-level workspace', () => {
    expect(hostMigration(file, 'pi').set).toEqual({ apiKey: 'root-key', peerName: 'alice', workspace: 'root-ws', schemaVersion: 2 })
  })

  test('globalOverride workspace wins over its own', () => {
    expect(hostMigration({ globalOverride: true, workspace: 'shared', hosts: { a: { workspace: 'mine' } } }, 'a').set)
      .toEqual({ workspace: 'shared', schemaVersion: 2 })
  })

  test('a current block only gets the schemaVersion mark, once', () => {
    expect(hostMigration({ hosts: { a: { apiKey: 'k' } } }, 'a')).toEqual({ set: { schemaVersion: 2 }, remove: [] })
    expect(hostMigration({ hosts: { a: { apiKey: 'k', schemaVersion: 2 } } }, 'a')).toEqual({ set: {}, remove: [] })
  })

  test('a file from a newer schema is left alone', async () => {
    expect(hostMigration({ schemaVersion: 3, apiKey: 'k' }, 'a')).toEqual({ set: {}, remove: [] })
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    writeFileSync(path, JSON.stringify({ schemaVersion: 3, hosts: {} }))
    await expect(saveHost(path, 'a', { workspace: 'w' })).rejects.toThrow('newer Honcho plugin')
    expect(JSON.parse(readFileSync(path, 'utf-8'))).toEqual({ schemaVersion: 3, hosts: {} })
  })

  test('writes only its own block, then has nothing left to do', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    writeFileSync(path, JSON.stringify(file))
    expect(await migrateHost(path, 'claude_code', ['sessionStrategy'])).toEqual(['apiKey', 'peerName', 'baseUrl', '-endpoint', '-sessionStrategy'])
    const after = JSON.parse(readFileSync(path, 'utf-8'))
    expect(after.apiKey).toBe('root-key')
    expect(after.sessions).toEqual({ '/x': 's' })
    expect(after.hosts.dsh).toEqual({ auth: { apiKey: 'dsh-key' } })
    expect(after.hosts.claude_code).toEqual({ workspace: 'cc', apiKey: 'root-key', peerName: 'alice', baseUrl: 'http://localhost:8000', schemaVersion: 2 })
    expect(await migrateHost(path, 'claude_code', ['sessionStrategy'])).toEqual([])
  })

  test('a missing file stays missing', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
    expect(await migrateHost(path, 'claude_code')).toEqual([])
    expect(existsSync(path)).toBe(false)
  })
})
