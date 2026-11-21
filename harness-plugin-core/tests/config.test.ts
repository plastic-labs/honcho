import { describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'
import {
  checkConfig,
  claudeHonchoReader,
  configPath,
  migrateHost,
  normalizeBaseUrl,
  opencodeHonchoReader,
  openclawHonchoReader,
  piHonchoReader,
  pickCredential,
  resolveConfig,
  saveGrant,
  saveHost,
  v01Reader,
  type LegacyReader,
} from '../src/index'

const CLOUD = 'https://api.honcho.dev'
const LOCAL = 'http://localhost:8000'
const SELF = 'https://self.example'
const tmp = (data: unknown) => {
  const path = join(mkdtempSync(join(tmpdir(), 'hpc-')), 'config.json')
  writeFileSync(path, JSON.stringify(data))
  return path
}
const read = (path: string) => JSON.parse(readFileSync(path, 'utf-8'))
const grant = (accessToken: string) => ({ accessToken, refreshToken: 'r', accessExpiresAt: 9999999999, clientId: 'c', scope: 'write' })

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
  test('a migrated block is all it reads; env beats it', () => {
    const file = { workspace: 'root-ws', peerName: 'root-peer', hosts: { a: { workspace: 'host-ws', schemaVersion: 2 } } }
    const cfg = resolveConfig(file, { host: 'a', env: { USER: 'u' } })
    expect(cfg.workspace).toBe('host-ws')
    expect(cfg.peerName).toBe('u')
    expect(resolveConfig(file, { host: 'a', env: { HONCHO_WORKSPACE: 'env-ws' } }).workspace).toBe('env-ws')
  })

  test('defaults, then the host name', () => {
    const cfg = resolveConfig({}, { host: 'my-host', env: {}, defaults: { aiPeer: 'claude' } })
    expect(cfg.baseUrl).toBe(CLOUD)
    expect(cfg.workspace).toBe('my-host')
    expect(cfg.aiPeer).toBe('claude')
    expect(cfg.warnings).toEqual([])
  })
})

describe('saveHost', () => {
  test('merges into one block, removes undefined keys, leaves the rest alone', async () => {
    const path = tmp({ hosts: { a: { workspace: 'w', aiPeer: 'x' }, b: { workspace: 'other' } } })
    await saveHost(path, 'a', { peerName: 'alice', aiPeer: undefined })
    expect(read(path)).toEqual({ hosts: { a: { workspace: 'w', peerName: 'alice', schemaVersion: 2 }, b: { workspace: 'other' } } })
  })

  test('refuses a host name that would reach Object.prototype', async () => {
    await expect(saveHost(tmp({}), '__proto__', { peerName: 'x' })).rejects.toThrow('invalid host name')
    expect(({} as Record<string, unknown>).peerName).toBeUndefined()
  })
})

describe('legacy readers', () => {
  // Recorded from each plugin's latest published release, run with no HONCHO_* env and USER=U.
  type Case = { name: string; host: string; file: Record<string, any>; want: Record<string, unknown> | null }
  const fixtures = (name: string): Case[] => JSON.parse(readFileSync(join(import.meta.dir, 'fixtures', name), 'utf-8'))
  // A setting the reader leaves out falls back to the plugin's schema-2 default, as it did before.
  const claudeDefaults: Record<string, Record<string, string>> = { claude_code: { workspace: 'claude_code', aiPeer: 'claude' }, cursor: { workspace: 'cursor', aiPeer: 'cursor' } }
  const releases: [string, LegacyReader, string, (host: string) => Record<string, unknown>][] = [
    ['harness-plugin-core 0.1.1', v01Reader, 'legacy-v01.json', (host) => ({ workspace: host })],
    ['claude-honcho 0.3.4', claudeHonchoReader, 'legacy-claude-honcho.json', (host) => claudeDefaults[host]],
    ['opencode-honcho 0.2.1', opencodeHonchoReader, 'legacy-opencode-honcho.json', () => ({})],
    ['openclaw-honcho 1.7.0', openclawHonchoReader, 'legacy-openclaw-honcho.json', () => ({ workspace: 'openclaw' })],
    // Where pi used the honcho CLI's grant (`usedCliGrant`), it recorded no key, and the reader gives none.
    ['pi-honcho 1.0.1', piHonchoReader, 'legacy-pi-honcho.json', () => ({ workspace: 'pi', aiPeer: 'pi' })],
  ]
  for (const [release, reader, fixture, defaults] of releases) {
    test(`reads a file as ${release} did`, () => {
      for (const c of fixtures(fixture)) {
        const s: Record<string, unknown> = reader(c.file, c.host)
        // That release had no key and ran unconfigured; only the key is comparable.
        if (c.want === null) {
          expect([c.name, s.apiKey ?? null]).toEqual([c.name, null])
          continue
        }
        const fallback: Record<string, unknown> = { apiKey: null, baseUrl: CLOUD, peerName: 'U', ...defaults(c.host) }
        const got = Object.fromEntries(Object.keys(c.want).map((k) => [k, k === 'baseUrl' ? normalizeBaseUrl(String(s[k] ?? fallback[k])) : (s[k] ?? fallback[k])]))
        expect([c.name, got]).toEqual([c.name, c.want])
      }
    })
  }
})

describe('migration', () => {
  const readers: Record<string, LegacyReader> = { v01Reader, claudeHonchoReader, opencodeHonchoReader, openclawHonchoReader, piHonchoReader }
  const tops: Record<string, Record<string, unknown>> = {
    none: {},
    cliLocal: { apiKey: 'cli', environmentUrl: LOCAL },
    selfAndCli: { auth: { apiKey: 'top' }, baseUrl: SELF, apiKey: 'cli', environmentUrl: LOCAL },
    endpointSelf: { apiKey: 'top', endpoint: { baseUrl: SELF } },
    names: { peerName: 'top-peer', workspace: 'top-ws' },
  }
  const blocks: Record<string, Record<string, any> | undefined> = {
    absent: undefined,
    empty: {},
    key: { apiKey: 'own' },
    url: { baseUrl: SELF },
    endpointLocal: { apiKey: 'own', endpoint: { baseUrl: LOCAL } },
    grant: { oauth: grant('grant') },
    oldGrant: { auth: { oauth: grant('grant') } },
    oldGrantBesideNull: { oauth: null, auth: { oauth: grant('grant') } },
    names: { workspace: 'own-ws', peerName: 'own-peer' },
    marked: { apiKey: 'own', baseUrl: LOCAL, schemaVersion: 2 },
  }
  type Run = (path: string, legacy: LegacyReader) => Promise<unknown>
  const start: Run = (path, legacy) => migrateHost(path, 'h', [], legacy)
  const then = (patch: Record<string, unknown>): Run => async (path, legacy) => {
    await start(path, legacy)
    await saveHost(path, 'h', patch, legacy)
  }
  const resolve = (file: unknown, legacy: LegacyReader) => {
    const cfg = resolveConfig(file, { host: 'h', env: { USER: 'U' }, legacy })
    const cred = pickCredential(file, { host: 'h', baseUrl: cfg.baseUrl, env: {}, legacy })
    return { server: cfg.baseUrl, token: cred?.token ?? null, workspace: cfg.workspace, peerName: cfg.peerName }
  }
  // Every order a plugin might run things in.
  const orders: Record<string, Run> = {
    start,
    'setup: name': then({ peerName: 'me' }),
    'setup: cloud': then({ baseUrl: undefined }),
    'setup: local': then({ baseUrl: LOCAL }),
    'setup: key': then({ apiKey: 'new' }),
    'setup screen saves what it read, before migrating': async (path, legacy) => {
      const r = resolveConfig(read(path), { host: 'h', env: { USER: 'U' }, legacy })
      await saveHost(path, 'h', { peerName: r.peerName, workspace: r.workspace, baseUrl: r.baseUrl === CLOUD ? undefined : r.baseUrl }, legacy)
      await start(path, legacy)
    },
    'sign in': async (path, legacy) => {
      await start(path, legacy)
      await saveGrant(path, 'h', grant('signin'))
    },
    'sign in, then sign out': async (path, legacy) => {
      await start(path, legacy)
      await saveGrant(path, 'h', grant('signin'))
      await saveGrant(path, 'h', null)
    },
  }
  // What the plugin should use afterwards: what its old reader said, then the rules.
  function expected(file: Record<string, any>, block: Record<string, any> | undefined, order: string, legacy: LegacyReader) {
    const marked = block?.schemaVersion === 2
    const g = block?.oauth?.accessToken ?? block?.auth?.oauth?.accessToken ?? null
    const old: Record<string, any> = marked ? block! : legacy(file, 'h')
    const server = g && !marked ? CLOUD : normalizeBaseUrl(old.baseUrl ?? CLOUD)
    const key = g && !marked ? null : (old.apiKey ?? null)
    const names = { workspace: old.workspace ?? 'h', peerName: old.peerName ?? 'U' }
    const at = (s: string, k: string | null) => ({ server: s, token: g && s === CLOUD ? g : k, ...names })
    if (order === 'setup: name') return { ...at(server, key), peerName: 'me' }
    if (order === 'setup: cloud') return at(CLOUD, server === CLOUD ? key : null)
    if (order === 'setup: local') return at(LOCAL, server === LOCAL ? key : null)
    if (order === 'setup: key') return at(server, 'new')
    if (order === 'sign in') return { ...at(CLOUD, null), token: 'signin' }
    if (order === 'sign in, then sign out') return { ...at(CLOUD, null), token: null }
    return at(server, key)
  }

  test('each plugin keeps what it used; reading first, later top-level writes and a second start change nothing', async () => {
    const failures: string[] = []
    for (const [rn, legacy] of Object.entries(readers)) {
      for (const [tn, top] of Object.entries(tops)) {
        for (const [bn, block] of Object.entries(blocks)) {
          for (const [on, run] of Object.entries(orders)) {
            const id = `${rn} · ${tn} · ${bn} · ${on}`
            const file = { ...structuredClone(top), ...(block ? { hosts: { h: structuredClone(block) } } : {}) }
            const path = tmp(file)
            const before = resolve(file, legacy)
            await run(path, legacy)
            const after = read(path)
            const got = resolve(after, legacy)
            const want = expected(file, block, on, legacy)
            if (JSON.stringify(got) !== JSON.stringify(want)) failures.push(`${id}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`)
            if (on === 'start' && JSON.stringify(before) !== JSON.stringify(got)) failures.push(`${id}: read before migrating gave ${JSON.stringify(before)}`)
            if ((await migrateHost(path, 'h', [], legacy)).length) failures.push(`${id}: a second start changed the block`)
            // An older honcho CLI writing the top level later.
            const later = tmp({ ...after, apiKey: 'later', environmentUrl: 'http://127.0.0.1:9999', peerName: 'later', workspace: 'later' })
            await migrateHost(later, 'h', [], legacy)
            if (JSON.stringify(resolve(read(later), legacy)) !== JSON.stringify(got)) failures.push(`${id}: a later top-level write reached the block`)
          }
        }
      }
    }
    expect(failures).toEqual([])
  })

  test('a block made by sign-in takes no key or server from the top level', async () => {
    const path = tmp({})
    await saveGrant(path, 'a', grant('at')) // before the plugin's first migration
    writeFileSync(path, JSON.stringify({ ...read(path), apiKey: 'cli-key', environmentUrl: LOCAL })) // then `honcho init`
    await migrateHost(path, 'a')
    expect(read(path).hosts.a).toEqual({ oauth: grant('at'), schemaVersion: 2 })
  })
})

describe('checkConfig', () => {
  test('reports a user split across peers and a newer file; leaves the top level to the CLI', () => {
    const file = { apiKey: 'k', environmentUrl: LOCAL, hosts: { a: { workspace: 'w', peerName: 'x', endpoint: {} }, b: { workspace: 'w', peerName: 'y' } } }
    expect(checkConfig(file).map((i) => i.path)).toEqual(['hosts'])
    expect(checkConfig({ schemaVersion: 3 }).map((i) => i.path)).toEqual(['schemaVersion'])
    // Not a guess at an unmigrated block's peer from the top level.
    expect(checkConfig({ peerName: 'top', hosts: { a: { workspace: 'w', peerName: 'x' }, b: { workspace: 'w' } } })).toEqual([])
  })
})

describe('configPath', () => {
  test('HONCHO_CONFIG_PATH, then $HOME, then os.homedir()', () => {
    expect(configPath({ HONCHO_CONFIG_PATH: '/x/cfg.json', HOME: '/h' })).toBe('/x/cfg.json')
    expect(configPath({ HOME: '/scratch' })).toBe('/scratch/.honcho/config.json')
    expect(configPath({})).toBe(join(homedir(), '.honcho', 'config.json'))
  })
})
