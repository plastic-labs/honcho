# Changelog

All notable changes to `@honcho-ai/harness-plugin-core` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

This package versions independently of the Honcho API, `@honcho-ai/sdk`, and host plugins.

## [Unreleased]

## [0.1.1] - 2026-09-10

### Fixed

- Ship a compiled `dist/` (ESM JavaScript + `.d.ts`) instead of pointing `main`/`exports` at TypeScript source. 0.1.0 only loaded under runtimes that transpile TypeScript on the fly (bun, jiti, esbuild); plain Node refused to strip types under `node_modules`, and `tsc` consumers using `nodenext` resolution failed inside this package's source. Relative imports now carry `.js` extensions so the package type-checks under any module resolution.
- Public `env` parameters are typed as `Env` (`Record<string, string | undefined>`) instead of `NodeJS.Dict<string>`, so the shipped declarations do not require consumers to install `@types/node`.
