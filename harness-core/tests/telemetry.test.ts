import { describe, expect, test } from 'bun:test'
import {
  HEADER_AGENT_MODEL,
  HEADER_API,
  HEADER_HOST,
  HEADER_PLUGIN,
  HEADER_RUNTIME,
  telemetryHeaders,
  version,
} from '../src/index.ts'

describe('telemetryHeaders', () => {
  test('empty identity still sends the runtime version', () => {
    expect(telemetryHeaders()).toEqual({ [HEADER_RUNTIME]: version })
  })

  test('maps identity to headers and classifies the api', () => {
    const headers = telemetryHeaders({
      host: 'opencode',
      hostVersion: '1.3.13',
      pluginVersion: '0.1.3',
      model: 'claude-sonnet-4-5',
      baseUrl: 'http://localhost:8000',
    })
    expect(headers).toEqual({
      [HEADER_RUNTIME]: version,
      [HEADER_HOST]: 'opencode/1.3.13',
      [HEADER_PLUGIN]: '0.1.3',
      [HEADER_AGENT_MODEL]: 'claude-sonnet-4-5',
      [HEADER_API]: 'custom',
    })
    expect(JSON.stringify(headers)).not.toContain('localhost')
    expect(telemetryHeaders({ baseUrl: 'https://api.honcho.dev' })[HEADER_API]).toBe('cloud')
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
