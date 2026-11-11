export {
  checkConfig,
  hostMigration,
  migrateHost,
  configPath,
  loadConfig,
  normalizeBaseUrl,
  readConfigFile,
  resolveConfig,
  saveHost,
  suggestPeerName,
  CONFIG_SCHEMA_VERSION,
  DEFAULT_BASE_URL,
} from './config.js'

// Each plugin's pre-schema-2 reader, for `migrateHost`.
export { v01Reader } from './legacy/v01.js'
export { claudeHonchoReader } from './legacy/claude-honcho.js'
export { opencodeHonchoReader } from './legacy/opencode-honcho.js'

export type {
  ConfigFile,
  ConfigIssue,
  Env,
  HostConfig,
  LegacyReader,
  LegacySettings,
  ResolvedConfig,
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
  OAuthError,
  OAUTH_DEVICE_URL,
  OAUTH_SCOPE,
  OAUTH_TOKEN_URL,
} from './oauth.js'

export type { Credential, CredentialSource, DeviceCode, OAuthTokens, OAuthTransport } from './oauth.js'
