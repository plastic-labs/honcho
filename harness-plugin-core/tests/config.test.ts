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
      hosts: { a: { workspace: 'host-ws' } },
    }
    expect(resolveConfig(file, { host: 'a', env: emptyEnv }).workspace).toBe('host-ws')
    expect(
      resolveConfig(file, { host: 'a', env: { HONCHO_WORKSPACE: 'env-ws' } }).workspace
    ).toBe('env-ws')
  })

  test('root apiKey / workspaceId aliases still resolve', () => {
    const cfg = resolveConfig(
      { apiKey: 'hch_x', workspaceId: 'from-id' },
      { host: 'a', env: emptyEnv }
    )
    expect(cfg.apiKey).toBe('hch_x')
    expect(cfg.workspace).toBe('from-id')
  })

  test('v1 leftover environmentUrl is ignored', () => {
    const cfg = resolveConfig(
      { schemaVersion: 1, baseUrl: 'https://keep.example', environmentUrl: 'https://old.example' },
      { host: 'a', env: emptyEnv }
    )
    expect(cfg.baseUrl).toBe('https://keep.example')
  })

  test('overlay sits below env', () => {
    expect(
      resolveConfig(
        {},
        { host: 'a', overlay: { workspace: 'from-overlay' }, env: { HONCHO_WORKSPACE: 'from-env' } }
      ).workspace
    ).toBe('from-env')
    expect(
      resolveConfig({}, { host: 'a', overlay: { workspace: 'from-overlay' }, env: emptyEnv }).workspace
    ).toBe('from-overlay')
  })

  test('empty file uses built-ins; host name is not rewritten', () => {
    const cfg = resolveConfig({}, { host: 'my-host', env: emptyEnv })
    expect(cfg.baseUrl).toBe('https://api.honcho.dev')
    expect(cfg.timeoutMs).toBe(30_000)
    expect(cfg.enabled).toBe(true)
    expect(cfg.host).toBe('my-host')
    expect(cfg.workspace).toBe('my-host')
  })
})
