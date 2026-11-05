# Changelog

All notable changes to `@honcho-ai/harness-plugin-core` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

This package versions independently of the Honcho API, `@honcho-ai/sdk`, and host plugins.

## [Unreleased]

## [0.2.0] - 2026-11-05

### Added

- OAuth device sign-in for Honcho cloud (`requestDeviceCode`, `pollForToken`, `refreshAccessToken`). Client id `honcho-plugin` for every plugin; each passes its own `source`. Cloud only: off `api.honcho.dev` OAuth is skipped and the API key is used.
- `pickCredential` / `ensureFreshCredential`: token resolution, highest wins: `HONCHO_API_KEY` env → the plugin's own grant (`hosts.<host>.oauth`) → the shared root grant (`oauth`) → `apiKey` → an expired grant. `ensureFreshCredential` refreshes a grant within 60s of expiry under a cross-process lock and re-reads the file first, so concurrent processes refresh once.
- `saveGrant`, `readGrant`, `updateConfigFile`, `withConfigLock`: locked, atomic (temp + rename, mode 0600) writes that touch only the block they change.

### Changed

- `AuthConfig.oauth` keys are `accessToken`, `refreshToken`, `accessExpiresAt`, `clientId`, `scope`. `resolveConfig` takes a grant whole from one layer instead of merging its fields across root and host.

## [0.1.1] - 2026-09-10

### Fixed

- Ship a compiled `dist/` (ESM JavaScript + `.d.ts`) instead of pointing `main`/`exports` at TypeScript source. 0.1.0 only loaded under runtimes that transpile TypeScript on the fly (bun, jiti, esbuild); plain Node refused to strip types under `node_modules`, and `tsc` consumers using `nodenext` resolution failed inside this package's source. Relative imports now carry `.js` extensions so the package type-checks under any module resolution.
- Public `env` parameters are typed as `Env` (`Record<string, string | undefined>`) instead of `NodeJS.Dict<string>`, so the shipped declarations do not require consumers to install `@types/node`.
