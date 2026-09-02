export const version = '0.1.0'

export {
  configPath,
  loadConfig,
  normalizeBaseUrl,
  resolveConfig,
  DEFAULT_BASE_URL,
  DEFAULT_TIMEOUT_MS,
} from './config.ts'

export type {
  AuthConfig,
  FileConfig,
  HostBlock,
  ResolvedConfig,
  RootConfig,
} from './config.ts'

export {
  telemetryHeaders,
  setTelemetryHeaders,
  HEADER_AGENT_MODEL,
  HEADER_HOST,
  HEADER_PLUGIN,
  HEADER_RUNTIME,
} from './telemetry.ts'

export type { TelemetryIdentity } from './telemetry.ts'
