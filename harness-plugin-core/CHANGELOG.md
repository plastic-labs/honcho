# Changelog

All notable changes to `@honcho-ai/harness-plugin-core` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

This package versions independently of the Honcho API, `@honcho-ai/sdk`, and host plugins.

## [Unreleased]

## [0.2.0] - unreleased

### Added

- OAuth device sign-in for Honcho cloud (`requestDeviceCode`, `pollForToken`, `refreshAccessToken`). Each plugin passes its own `clientId` (a first-party client registered in Honcho cloud) and `source`. Cloud only: off `api.honcho.dev` OAuth is skipped and the API key is used.
- `pickCredential` / `ensureFreshCredential`: token resolution, highest wins: `HONCHO_API_KEY` env → the plugin's own grant (`hosts.<host>.oauth`) → `hosts.<host>.apiKey` → that grant expired. One grant per plugin; a root `oauth` block, a root `apiKey` or another host's grant is never used, so each plugin keeps the tenant it signed in to. `ensureFreshCredential` refreshes a grant within 60s of expiry under a cross-process lock and re-reads the file first, so concurrent processes refresh once.
- `saveGrant`, `readGrant`, `updateConfigFile`, `withConfigLock`: locked, atomic (temp + rename, mode 0600) writes that touch only the block they change.
- `saveHost(path, host, patch)`: locked merge into one host block. A first write to a block also does the one-time migration below; `patch` wins for the keys it names.
- `migrateHost(path, host, deadKeys)` / `hostMigration`: a plugin moves an older file's settings into its own block at startup, without touching the top level or other hosts. The block's own value, an explicit Honcho cloud URL included, beats the top level's. The top level is copied once, before the block is marked, so values written there later never reach it.
- `checkConfig(file)`: read-only problems for a plugin to show the user: a newer file, and hosts sharing a workspace under different `peerName`s. Cleaning the top level is the honcho CLI's job (`honcho doctor`), not this package's.
- `readConfigFile`, `CONFIG_SCHEMA_VERSION`.
- `suggestPeerName(file)`: the name a plugin's setup offers the user: the one the other plugins agree on, else the old top-level `peerName`, else `$USER`.
- `hosts.<host>.schemaVersion`: written by `saveHost` / `migrateHost` to mark a block whose plugin reads only that block; `honcho doctor` waits for it before removing top-level keys.
- A file with a newer `schemaVersion` is read but not written (`saveHost` throws, `migrateHost` skips), and `checkConfig` reports it.
- `config.schema.json` describes the file. Its `$id` is `https://cdn.jsdelivr.net/npm/@honcho-ai/harness-plugin-core@0/config.schema.json`, and it allows a top-level `$schema`.

### Changed

- **Config schema version 2.** Every setting lives in the plugin's own block, `hosts.<host>`: `apiKey` or `oauth` (`accessToken`, `refreshToken`, `accessExpiresAt`, `clientId`, `scope`), `peerName`, `workspace`, `aiPeer`, `baseUrl`. Nothing is read from the top level, and `auth.apiKey`, `endpoint`, `environmentUrl` and `workspaceId` are not read either. `peerName` falls back to `$USER`.
- `resolveConfig` / `loadConfig` take `{ host, env?, defaults? }` and return `{ host, peerName, workspace, aiPeer, baseUrl, warnings }`. Credentials come only from `pickCredential` / `ensureFreshCredential`.

### Removed

- `timeoutMs`, `enabled`, the `overlay` option, `${VAR}` interpolation, the v0/v1 migration on read, and the `AuthConfig`, `RootConfig`, `HostBlock`, `FileConfig` types and `DEFAULT_TIMEOUT_MS`.

## [0.1.1] - 2026-09-10

### Fixed

- Ship a compiled `dist/` (ESM JavaScript + `.d.ts`) instead of pointing `main`/`exports` at TypeScript source. 0.1.0 only loaded under runtimes that transpile TypeScript on the fly (bun, jiti, esbuild); plain Node refused to strip types under `node_modules`, and `tsc` consumers using `nodenext` resolution failed inside this package's source. Relative imports now carry `.js` extensions so the package type-checks under any module resolution.
- Public `env` parameters are typed as `Env` (`Record<string, string | undefined>`) instead of `NodeJS.Dict<string>`, so the shipped declarations do not require consumers to install `@types/node`.
