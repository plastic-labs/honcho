import { describe, expect, test } from 'bun:test'
import { normalizeBaseUrl, resolveConfig } from '../src/index.ts'

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
  test('host block beats root; env beats host', () => {
    const file = {
      workspace: 'root-ws',
      hosts: { claude_code: { workspace: 'claude-ws' } },
    }
    expect(resolveConfig(file, { host: 'claude_code', env: emptyEnv }).workspace).toBe('claude-ws')
    expect(
      resolveConfig(file, { host: 'claude_code', env: { HONCHO_WORKSPACE: 'env-ws' } }).workspace
    ).toBe('env-ws')
  })

  test('root apiKey / workspaceId aliases still resolve', () => {
    const cfg = resolveConfig(
      { apiKey: 'hch_x', workspaceId: 'from-id' },
      { host: 'openclaw', env: emptyEnv }
    )
    expect(cfg.apiKey).toBe('hch_x')
    expect(cfg.workspace).toBe('from-id')
  })

  test('overlay sits below env', () => {
    expect(
      resolveConfig(
        {},
        { host: 'openclaw', overlay: { workspace: 'from-openclaw' }, env: { HONCHO_WORKSPACE: 'from-env' } }
      ).workspace
    ).toBe('from-env')
    expect(
      resolveConfig({}, { host: 'openclaw', overlay: { workspace: 'from-openclaw' }, env: emptyEnv }).workspace
    ).toBe('from-openclaw')
  })

  test('empty file uses built-ins; host name is not rewritten', () => {
    const cfg = resolveConfig({}, { host: 'claude-code', env: emptyEnv })
    expect(cfg.baseUrl).toBe('https://api.honcho.dev')
    expect(cfg.timeoutMs).toBe(30_000)
    expect(cfg.enabled).toBe(true)
    expect(cfg.host).toBe('claude-code')
    expect(cfg.workspace).toBe('claude-code')
  })
})
