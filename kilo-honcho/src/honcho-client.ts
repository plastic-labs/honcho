import { readFileSync } from "node:fs"
import { Honcho } from "@honcho-ai/sdk"
import { telemetryHeaders, type TelemetryIdentity } from "@honcho-ai/harness-plugin-core"

import { HOST_ID, PACKAGE_ID, PLUGIN_ID } from "./core.js"

export { HOST_ID, PACKAGE_ID, PLUGIN_ID }

let pluginVersion: string | undefined

/**
 * Plugin version from package.json, which sits one directory above both `src/` and the bundled
 * `dist/`. Read lazily and guarded so a bad path can never stop the plugin from loading.
 */
export const getPluginVersion = (): string => {
  if (pluginVersion) return pluginVersion
  try {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf-8")) as { version?: unknown }
    if (typeof pkg.version === "string" && pkg.version) return (pluginVersion = pkg.version)
  } catch {
    // fall through to "unknown"; the next call retries
  }
  return "unknown"
}

/** The identity fields only the running host knows: its version and the agent model in use. */
export type TelemetryOverrides = Pick<TelemetryIdentity, "hostVersion" | "model">

export type HonchoClientOptions = TelemetryOverrides & {
  apiKey: string
  baseUrl: string
  workspaceId: string
}

export const telemetryIdentity = (overrides: TelemetryOverrides = {}): TelemetryIdentity => ({
  host: HOST_ID,
  plugin: PLUGIN_ID,
  pluginVersion: getPluginVersion(),
  ...(overrides.hostVersion ? { hostVersion: overrides.hostVersion } : {}),
  ...(overrides.model ? { model: overrides.model } : {}),
})

/**
 * Every Honcho client the plugin builds goes through here so the telemetry headers ride on
 * every request. The runtime builds one per hook, so the identity is always current.
 */
export const createHonchoClient = (options: HonchoClientOptions) =>
  new Honcho({
    apiKey: options.apiKey || undefined,
    baseURL: options.baseUrl || undefined,
    workspaceId: options.workspaceId,
    defaultHeaders: telemetryHeaders(telemetryIdentity(options)),
  })
