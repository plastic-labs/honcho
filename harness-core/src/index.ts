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
