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

export {
  accessValid,
  ensureFreshCredential,
  isCloud,
  pickCredential,
  pollForToken,
  readGrant,
  refreshAccessToken,
  requestDeviceCode,
  saveGrant,
  updateConfigFile,
  withConfigLock,
  HARNESS_CLIENT_ID,
  OAuthError,
  OAUTH_DEVICE_URL,
  OAUTH_SCOPE,
  OAUTH_TOKEN_URL,
} from './oauth.js'

export type { Credential, CredentialSource, DeviceCode, OAuthTokens, OAuthTransport } from './oauth.js'
