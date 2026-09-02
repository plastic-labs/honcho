import { describe, expect, test } from 'bun:test'
import {
  HEADER_AGENT_MODEL,
  HEADER_HOST,
  HEADER_PLUGIN,
  HEADER_RUNTIME,
  setTelemetryHeaders,
  telemetryHeaders,
  version,
} from '../src/index.ts'

describe('telemetryHeaders', () => {
  test('empty identity still sends the runtime version', () => {
    expect(telemetryHeaders()).toEqual({ [HEADER_RUNTIME]: version })
  })

  test('maps identity to headers', () => {
    expect(
      telemetryHeaders({
        host: 'opencode',
        hostVersion: '1.3.13',
        pluginVersion: '0.1.3',
        model: 'claude-sonnet-4-5',
      })
    ).toEqual({
      [HEADER_RUNTIME]: version,
      [HEADER_HOST]: 'opencode/1.3.13',
      [HEADER_PLUGIN]: '0.1.3',
      [HEADER_AGENT_MODEL]: 'claude-sonnet-4-5',
    })
  })

  test('merges extra headers last, skipping blanks', () => {
    const headers = telemetryHeaders({ host: 'codex', pluginVersion: '0.1.1' }, {
      'X-Custom': 'yes',
      [HEADER_PLUGIN]: 'override',
      'X-Empty': '  ',
    })
    expect(headers[HEADER_HOST]).toBe('codex')
    expect(headers[HEADER_PLUGIN]).toBe('override')
    expect(headers['X-Custom']).toBe('yes')
    expect(headers).not.toHaveProperty('X-Empty')
  })
})

describe('setTelemetryHeaders', () => {
  test('mutates an existing header map in place', () => {
    const headers = telemetryHeaders({ host: 'cursor', pluginVersion: '0.1.2' })
    const returned = setTelemetryHeaders(headers, { model: 'claude-opus-4' })
    expect(returned).toBe(headers)
    expect(headers[HEADER_HOST]).toBe('cursor')
    expect(headers[HEADER_PLUGIN]).toBe('0.1.2')
    expect(headers[HEADER_RUNTIME]).toBe(version)
    expect(headers[HEADER_AGENT_MODEL]).toBe('claude-opus-4')
  })
})
