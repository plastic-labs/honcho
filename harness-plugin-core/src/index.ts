export {
  configPath,
  loadConfig,
  normalizeBaseUrl,
  resolveConfig,
  DEFAULT_BASE_URL,
  DEFAULT_TIMEOUT_MS,
} from './config'

export type {
  AuthConfig,
  FileConfig,
  HostBlock,
  ResolvedConfig,
  RootConfig,
} from './config'

export {
  hostHeaderValue,
  pluginHeaderValue,
  telemetryHeaders,
  setTelemetryHeaders,
  HEADER_AGENT_MODEL,
  HEADER_HOST,
  HEADER_PLUGIN,
} from './telemetry'

export type { TelemetryIdentity } from './telemetry'
