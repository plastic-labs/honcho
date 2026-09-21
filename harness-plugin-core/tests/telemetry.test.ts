import { describe, expect, test } from 'bun:test'
import {
  HEADER_AGENT_MODEL,
  HEADER_HOST,
  HEADER_PLUGIN,
  setTelemetryHeaders,
  telemetryHeaders,
} from '../src/index'

describe('telemetryHeaders', () => {
  test('maps identity to the three headers', () => {
    const headers = telemetryHeaders({
      host: 'harness',
      hostVersion: '2.1.3',
      platform: 'darwin',
      plugin: 'harness-honcho',
      pluginVersion: '0.2.11',
      model: 'claude-sonnet-4-5',
    })
    expect(headers).toEqual({
      [HEADER_HOST]: 'harness/2.1.3 (darwin)',
      [HEADER_PLUGIN]: 'harness-honcho/0.2.11',
      [HEADER_AGENT_MODEL]: 'claude-sonnet-4-5',
    })
  })

  test('omits unknown fields and defaults platform', () => {
    expect(telemetryHeaders()).toEqual({})
    expect(telemetryHeaders({ host: 'harness' })).toEqual({
      [HEADER_HOST]: `harness (${process.platform})`,
    })
  })

  test('strips separators that would break parsing', () => {
    expect(telemetryHeaders({ host: 'a b;(c)/d', hostVersion: '1\r\n2', platform: 'darwin' })).toEqual({
      [HEADER_HOST]: 'a-b-c-d/1-2 (darwin)',
    })
  })

  test('extra headers win, blanks are dropped', () => {
    const headers = telemetryHeaders({ plugin: 'harness-honcho' }, {
      [HEADER_PLUGIN]: 'override',
      'X-Empty': '  ',
    })
    expect(headers).toEqual({ [HEADER_PLUGIN]: 'override' })
  })
})

test('setTelemetryHeaders updates only the named fields in place', () => {
  const headers = telemetryHeaders({ plugin: 'harness-honcho', pluginVersion: '0.1.2' })
  expect(setTelemetryHeaders(headers, { model: 'claude-opus-4' })).toBe(headers)
  expect(headers).toEqual({
    [HEADER_PLUGIN]: 'harness-honcho/0.1.2',
    [HEADER_AGENT_MODEL]: 'claude-opus-4',
  })
})
