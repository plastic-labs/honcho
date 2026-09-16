export {
  configPath,
  loadConfig,
  normalizeBaseUrl,
  resolveConfig,
  DEFAULT_BASE_URL,
  DEFAULT_TIMEOUT_MS,
} from './config.js'

export type {
  AuthConfig,
  Env,
  FileConfig,
  HostBlock,
  ResolvedConfig,
  RootConfig,
} from './config.js'

export {
  hostHeaderValue,
  pluginHeaderValue,
  telemetryHeaders,
  setTelemetryHeaders,
  HEADER_AGENT_MODEL,
  HEADER_HOST,
  HEADER_PLUGIN,
} from './telemetry.js'

export type { TelemetryIdentity } from './telemetry.js'
