import { mkdir, readFile, writeFile } from "node:fs/promises"
import path from "node:path"
import { readActivity } from "../activity.js"
import { createHonchoClient } from "../honcho-client.js"
import { executeKiloImport, planKiloImport } from "../import.js"
import {
  DEFAULT_SETTINGS,
  SETTING_ENUMS,
  directionalKeepFollowUp,
  getNestedValue,
  isLocalBaseUrl,
  isObservationMode,
  isRecord,
  needsObservationUpgradePrompt,
  observationUpgradeNotice,
  resolveBaseUrl,
  resolveSessionPeerIds,
  sharedConfigPath,
  unifiedImportFollowUp,
  type ObservationMode,
} from "../core.js"
import { recallMessage } from "./activity-view.js"
import type { DialogOption, GlobalSettings, TuiCommandSpec, TuiSession } from "./dialogs.js"

export const resolveConfigPath = () => sharedConfigPath(process.env.KILO_HONCHO_CONFIG_PATH)

const SHARED_CONFIG_PRESETS: Record<string, readonly string[]> = Object.fromEntries(
  Object.entries(SETTING_ENUMS).map(([key, values]) => [key.toLowerCase(), values]),
)

const MODE_EDITABLE_FIELD_PATHS = [
  "apiKey",
  "baseUrl",
  "peerName",
  "hosts.kilo.workspace",
  "hosts.kilo.aiPeer",
  "hosts.kilo.recallMode",
  "hosts.kilo.observationMode",
  "hosts.kilo.agentObserveMe",
  "hosts.kilo.autoConclusions",
  "hosts.kilo.sessionStrategy",
] as const

export const readGlobalSettings = async (): Promise<GlobalSettings> => {
  const configPath = resolveConfigPath()
  try {
    const raw = await readFile(configPath, "utf-8")
    const parsed = JSON.parse(raw)
    return isRecord(parsed) ? (parsed as GlobalSettings) : {}
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return {}
    }
    throw error
  }
}

export const readSharedConfig = async (): Promise<Record<string, unknown> | null> => {
  const configPath = resolveConfigPath()
  try {
    const raw = await readFile(configPath, "utf-8")
    const parsed = JSON.parse(raw)
    if (!isRecord(parsed)) {
      throw new Error(`${configPath} must contain a JSON object at the top level.`)
    }
    return parsed
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return null
    }
    throw error
  }
}

export const writeSharedConfig = async (settings: Record<string, unknown>) => {
  const configPath = resolveConfigPath()
  await mkdir(path.dirname(configPath), { recursive: true })
  await writeFile(configPath, `${JSON.stringify(settings, null, 2)}\n`, "utf-8")
  return configPath
}

const listSharedConfigFields = (value: Record<string, unknown>, prefix = ""): string[] =>
  Object.entries(value).flatMap(([key, entryValue]) => {
    const nextKey = prefix ? `${prefix}.${key}` : key
    if (isRecord(entryValue) && Object.keys(entryValue).length > 0) {
      return listSharedConfigFields(entryValue, nextKey)
    }
    return [nextKey]
  })

const setNestedValue = (value: Record<string, unknown>, fieldPath: string, nextValue: unknown) => {
  const parts = fieldPath.split(".")
  let current: Record<string, unknown> = value
  for (const part of parts.slice(0, -1)) {
    const existing = current[part]
    if (!isRecord(existing)) {
      current[part] = {}
    }
    current = current[part] as Record<string, unknown>
  }
  current[parts.at(-1) as string] = nextValue
}

export const resolveSharedConfigField = (config: Record<string, unknown>, field: string) => {
  const canonical = listSharedConfigFields(config).find(
    (candidate) => candidate.toLowerCase() === field.trim().toLowerCase(),
  )
  if (!canonical) {
    throw new Error(`Field '${field}' does not exist in ${resolveConfigPath()}.`)
  }
  return canonical
}

export const modeEditableFieldPaths = () => [...MODE_EDITABLE_FIELD_PATHS]

export const sharedConfigPresetOptions = (fieldPath: string, currentValue: unknown) => {
  const presetKey = fieldPath.split(".").at(-1)?.toLowerCase() || fieldPath.toLowerCase()
  if (SHARED_CONFIG_PRESETS[presetKey]) {
    return [...SHARED_CONFIG_PRESETS[presetKey]]
  }
  if (typeof currentValue === "boolean") {
    return ["true", "false"]
  }
  return []
}

const parseSharedConfigValue = (currentValue: unknown, rawValue: string) => {
  const trimmed = rawValue.trim()
  if (typeof currentValue === "boolean") {
    return trimmed.toLowerCase() === "true"
  }
  if (typeof currentValue === "number") {
    const parsed = Number(trimmed)
    if (!Number.isFinite(parsed)) {
      throw new Error(`Expected a number for this field, received '${rawValue}'.`)
    }
    return parsed
  }
  return trimmed
}

const writeGlobalSettings = async (settings: GlobalSettings) => {
  const configPath = resolveConfigPath()
  await mkdir(path.dirname(configPath), { recursive: true })
  await writeFile(configPath, `${JSON.stringify(settings, null, 2)}\n`, "utf-8")
  return configPath
}

export const normalizeSettings = (settings: GlobalSettings) => ({
  baseUrl: resolveBaseUrl(settings as Record<string, unknown>) || DEFAULT_SETTINGS.baseUrl,
  apiKey: typeof settings.apiKey === "string" && settings.apiKey.trim() ? settings.apiKey.trim() : "",
  peerName: typeof settings.peerName === "string" ? settings.peerName.trim() : "",
})

export const validateCloudApiKey = (value: string) =>
  value.trim() ? null : "Honcho Cloud requires a Honcho API key. Enter a non-empty key or choose Self-hosted / local."

// The server lets HONCHO_* env values win over the file, so status reports what the server uses.
// Display only: nothing here is ever written back to the config file.
const withEnvOverrides = (normalized: ReturnType<typeof normalizeSettings>) => ({
  baseUrl: process.env.HONCHO_URL?.trim() || process.env.HONCHO_BASE_URL?.trim() || normalized.baseUrl,
  apiKey: process.env.HONCHO_API_KEY?.trim() || normalized.apiKey,
  peerName: process.env.HONCHO_PEER_NAME?.trim() || normalized.peerName,
})

export const statusMessage = (
  settings: GlobalSettings,
  liveStatus?: { workspaceName?: string; kiloSessionId?: string },
) => {
  const normalized = withEnvOverrides(normalizeSettings(settings))
  const configured = Boolean(normalized.apiKey) || isLocalBaseUrl(normalized.baseUrl)
  const deployment = isLocalBaseUrl(normalized.baseUrl)
    ? "Local / self-hosted"
    : normalized.baseUrl === DEFAULT_SETTINGS.baseUrl
      ? "Honcho Cloud"
      : "Custom endpoint"
  return [
    `Configured: ${configured ? "yes" : "no"}`,
    `Deployment: ${deployment}`,
    `Base URL: ${normalized.baseUrl}`,
    `API key: ${normalized.apiKey ? "set" : "not set"}`,
    `Peer name: ${normalized.peerName || "user"}`,
    ...(liveStatus?.workspaceName ? [`Workspace: ${liveStatus.workspaceName}`] : []),
    ...(liveStatus?.kiloSessionId ? [`Kilo session: ${liveStatus.kiloSessionId}`] : []),
    `Config path: ${resolveConfigPath()}`,
    "",
    configured ? "Honcho is ready for Kilo." : "Run /honcho:setup to finish configuration.",
    ...(configured && needsObservationUpgradePrompt(settings as Record<string, unknown>)
      ? ["", observationUpgradeNotice()]
      : []),
  ].join("\n")
}

export const settingsMessage = (settings: GlobalSettings) => {
  const host = settings.hosts?.kilo || {}
  return [
    `Config path: ${resolveConfigPath()}`,
    `API key: ${settings.apiKey?.trim() ? "set" : "not set"}`,
    `Peer name: ${settings.peerName?.trim() || "user"}`,
    `Base URL: ${resolveBaseUrl(settings as Record<string, unknown>) || DEFAULT_SETTINGS.baseUrl}`,
    `Workspace: ${host.workspace || DEFAULT_SETTINGS.workspace}`,
    `AI peer: ${host.aiPeer || DEFAULT_SETTINGS.aiPeer}`,
    `Recall mode: ${host.recallMode || DEFAULT_SETTINGS.recallMode}`,
    `Observation mode: ${host.observationMode || DEFAULT_SETTINGS.observationMode}`,
    `Agent observe me: ${host.agentObserveMe === true ? "true" : "false"}`,
    `Auto conclusions: ${host.autoConclusions === true ? "true" : "false"}`,
    `Session strategy: ${host.sessionStrategy || DEFAULT_SETTINGS.sessionStrategy}`,
    ...(!isObservationMode(settings.hosts?.kilo?.observationMode) && settings.hosts?.kilo
      ? ["", observationUpgradeNotice()]
      : []),
  ].join("\n")
}

const persistHostObservationMode = async (mode: ObservationMode) => {
  const config = (await readSharedConfig()) ?? {}
  const hosts = isRecord(config.hosts) ? { ...config.hosts } : {}
  const host = isRecord(hosts.kilo) ? hosts.kilo : {}
  hosts.kilo = { ...host, observationMode: mode }
  config.hosts = hosts
  return writeSharedConfig(config)
}

const observationUpgradeOptions = (): DialogOption<string>[] => [
  { title: "Switch to unified", value: "unified", description: "shared self-collection" },
  { title: "Keep directional", value: "directional", description: "split memory between agents" },
]

export const saveSettings = async (partial: Partial<GlobalSettings>) => {
  const current = await readGlobalSettings()
  const sharedRaw = (await readSharedConfig()) ?? {}
  const partialHost = partial.hosts?.kilo
  const nextApiKey =
    typeof partial.apiKey === "string"
      ? partial.apiKey
      : typeof current.apiKey === "string"
        ? current.apiKey
        : undefined
  const nextPeerName =
    typeof partial.peerName === "string" && partial.peerName.trim()
      ? partial.peerName.trim()
      : typeof current.peerName === "string" && current.peerName.trim()
        ? current.peerName.trim()
        : "user"
  const currentHosts = isRecord(sharedRaw.hosts) ? { ...sharedRaw.hosts } : {}
  const currentHost = isRecord(currentHosts.kilo) ? currentHosts.kilo : {}
  currentHosts.kilo = {
    ...currentHost,
    workspace: partialHost?.workspace ?? current.hosts?.kilo?.workspace ?? DEFAULT_SETTINGS.workspace,
    aiPeer: partialHost?.aiPeer ?? current.hosts?.kilo?.aiPeer ?? DEFAULT_SETTINGS.aiPeer,
    recallMode: partialHost?.recallMode ?? current.hosts?.kilo?.recallMode ?? DEFAULT_SETTINGS.recallMode,
    sessionStrategy: partialHost?.sessionStrategy ?? current.hosts?.kilo?.sessionStrategy ?? DEFAULT_SETTINGS.sessionStrategy,
  }

  const next: GlobalSettings & Record<string, unknown> = {
    ...sharedRaw,
    baseUrl:
      typeof partial.baseUrl === "string"
        ? partial.baseUrl
        : typeof current.baseUrl === "string"
          ? current.baseUrl
          : DEFAULT_SETTINGS.baseUrl,
    peerName: nextPeerName,
    hosts: currentHosts,
  }
  if (typeof nextApiKey === "string") {
    next.apiKey = nextApiKey
  } else {
    delete next.apiKey
  }
  return writeGlobalSettings(next)
}

const importConfigFromSettings = (settings: GlobalSettings) => {
  const host = settings.hosts?.kilo || {}
  const workspaceId = (host.workspace || DEFAULT_SETTINGS.workspace).trim() || DEFAULT_SETTINGS.workspace
  const aiPeer = host.aiPeer || DEFAULT_SETTINGS.aiPeer
  const { userPeerId, agentPeerId } = resolveSessionPeerIds(
    settings.peerName || "user",
    aiPeer,
    host.removeUserPrefix === true,
  )
  return {
    apiKey: settings.apiKey || "",
    baseUrl: resolveBaseUrl(settings as Record<string, unknown>) || DEFAULT_SETTINGS.baseUrl,
    workspaceId,
    userPeerId,
    agentPeerId,
    sessionStrategy: host.sessionStrategy || DEFAULT_SETTINGS.sessionStrategy,
    agentObserveMe: host.agentObserveMe === true,
    observationMode: isObservationMode(host.observationMode) ? host.observationMode : DEFAULT_SETTINGS.observationMode,
  }
}

const formatImportPreview = (plan: Awaited<ReturnType<typeof planKiloImport>>) => {
  const lines = [
    `Source: ${plan.source}`,
    `Window: last ${plan.days} days`,
    `Ready to import: ${plan.sessionCount} session(s), ${plan.messageCount} message(s)`,
    `Already imported: ${plan.alreadyImportedCount}`,
    `Skipped empty: ${plan.skippedCount}`,
    "",
  ]
  for (const session of plan.sessions.filter((item) => !item.skippedReason).slice(0, 12)) {
    lines.push(`- ${session.title} (${session.messageCount} msg)`)
  }
  if (plan.sessionCount > 12) lines.push(`- … and ${plan.sessionCount - 12} more`)
  return lines.join("\n")
}

const isConfigured = (settings: GlobalSettings) =>
  Boolean(settings.apiKey?.trim()) || isLocalBaseUrl(resolveBaseUrl(settings as Record<string, unknown>))

/** true when the user chose a mode and it was saved; false when they dismissed. */
export const runObservationUpgrade = async (session: TuiSession, followUpLines: string[]) => {
  const confirmed = await session.dialogs.confirm({
    title: "New: Honcho observation mode!",
    message:
      "Unified: one self-collection, you can share with other unified agents (new default). Directional: keeps Honcho memory specific to your Kilo agent.",
    label: { confirm: "Switch to unified", cancel: "Keep directional" },
  })
  if (confirmed === undefined) return false
  const mode: ObservationMode = confirmed ? "unified" : "directional"
  const configPath = await persistHostObservationMode(mode)
  await session.dialogs.alert({
    title: mode === "unified" ? "Switched to unified" : "Keeping directional",
    message: [
      ...followUpLines,
      `Saved observationMode=${mode} to ${configPath}`,
      mode === "unified" ? unifiedImportFollowUp() : directionalKeepFollowUp(),
    ].join("\n"),
  })
  return true
}

export const maybePromptObservationUpgrade = async (session: TuiSession) => {
  try {
    const settings = await readGlobalSettings()
    const raw = await readSharedConfig()
    if (!isConfigured(settings) || !needsObservationUpgradePrompt(raw)) return
    await runObservationUpgrade(session, [])
  } catch {
    return
  }
}

export const runStatus = async (session: TuiSession) => {
  const settings = await readGlobalSettings()
  await session.dialogs.alert({
    title: "Honcho Status",
    message: statusMessage(settings, session.liveStatus(settings)),
  })
}

export const runSettings = async (session: TuiSession) => {
  const settings = await readGlobalSettings()
  await session.dialogs.alert({ title: "Honcho Settings", message: settingsMessage(settings) })
}

export const runConfig = async (session: TuiSession) => {
  let config: Record<string, unknown> | null
  try {
    config = await readSharedConfig()
  } catch (error) {
    await session.dialogs.alert({
      title: "Honcho config invalid",
      message: error instanceof Error ? error.message : String(error),
    })
    return
  }
  if (!config) {
    await session.dialogs.alert({
      title: "Honcho config missing",
      message: `The config does not exist at ${resolveConfigPath()}.`,
    })
    return
  }

  const fieldPath = await session.dialogs.select<string>({
    title: "Which field should be modified?",
    options: modeEditableFieldPaths().map((value) => ({ title: value, value })),
  })
  if (!fieldPath) return

  const currentValue = getNestedValue(config, fieldPath)
  const presetOptions = sharedConfigPresetOptions(fieldPath, currentValue)
  const rawValue =
    presetOptions.length > 0
      ? await session.dialogs.select<string>({
          title: fieldPath.endsWith("observationMode")
            ? "Honcho observation mode"
            : `What should it be set to: ${presetOptions.join(", ")}`,
          options: fieldPath.endsWith("observationMode")
            ? observationUpgradeOptions()
            : presetOptions.map((option) => ({ title: option, value: option })),
          current: typeof currentValue === "string" ? currentValue : undefined,
        })
      : await session.dialogs.prompt({
          title: "What should it be set to:",
          value: currentValue === undefined ? "" : String(currentValue),
        })
  if (rawValue === undefined) return

  try {
    const nextConfig = structuredClone(config)
    const nextValue = parseSharedConfigValue(currentValue, rawValue)
    setNestedValue(nextConfig, fieldPath, nextValue)
    const configPath = await writeSharedConfig(nextConfig)
    const importHint =
      fieldPath.endsWith("observationMode") && nextValue === "unified" ? unifiedImportFollowUp() : null
    await session.dialogs.alert({
      title: "Honcho config updated",
      message: [`Saved settings to ${configPath}`, `Field: ${fieldPath}`, `Value: ${String(nextValue)}`, importHint]
        .filter((line): line is string => typeof line === "string")
        .join("\n"),
    })
  } catch (error) {
    await session.dialogs.alert({
      title: "Honcho config update failed",
      message: error instanceof Error ? error.message : String(error),
    })
  }
}

export const runSetup = async (session: TuiSession) => {
  const mode = await session.dialogs.select<"cloud" | "local">({
    title: "Configure Honcho",
    options: [
      { title: "Honcho Cloud", value: "cloud", description: "Use the default Honcho Cloud endpoint" },
      { title: "Self-hosted / local", value: "local", description: "Use a custom or localhost Honcho base URL" },
    ],
  })
  if (!mode) return

  let baseUrl = DEFAULT_SETTINGS.baseUrl
  let apiKey: string | undefined
  if (mode === "local") {
    const entered = await session.dialogs.prompt({
      title: "Honcho API base URL",
      placeholder: "http://127.0.0.1:8000",
      value: "http://127.0.0.1:8000",
    })
    if (entered === undefined) return
    baseUrl = entered.trim() || "http://127.0.0.1:8000"
    apiKey = await session.dialogs.prompt({
      title: "Optional Honcho API key",
      placeholder: "Leave blank for local unauthenticated mode",
    })
    if (apiKey === undefined) return
  } else {
    apiKey = await session.dialogs.prompt({ title: "Honcho API key", placeholder: "hch_..." })
    if (apiKey === undefined) return
    const validationError = validateCloudApiKey(apiKey)
    if (validationError) {
      await session.dialogs.alert({ title: "Honcho setup incomplete", message: validationError })
      return
    }
  }

  const current = await readGlobalSettings()
  const peerName = await session.dialogs.prompt({
    title: "Peer name",
    placeholder: "Your Honcho peer name",
    value: typeof current.peerName === "string" ? current.peerName : "",
  })
  if (peerName === undefined) return

  const configPath = await saveSettings({ apiKey: apiKey.trim(), baseUrl, peerName: peerName.trim() })
  const summary = [
    `Saved settings to ${configPath}`,
    `Base URL: ${baseUrl}`,
    `API key: ${apiKey.trim() ? "set" : mode === "local" ? "not required for localhost mode" : "not set"}`,
    `Peer name: ${peerName.trim() || "user"}`,
  ]
  const raw = await readSharedConfig()
  const settings = await readGlobalSettings()
  if (isConfigured(settings) && needsObservationUpgradePrompt(raw) && (await runObservationUpgrade(session, summary))) {
    return
  }
  await session.dialogs.alert({ title: "Honcho configured", message: summary.join("\n") })
}

export const runImport = async (session: TuiSession) => {
  const settings = await readGlobalSettings()
  if (!isConfigured(settings)) {
    await session.dialogs.alert({
      title: "Honcho import",
      message: "Run /honcho:setup before importing local Kilo transcripts.",
    })
    return
  }

  const config = importConfigFromSettings(settings)
  let plan: Awaited<ReturnType<typeof planKiloImport>>
  try {
    plan = await planKiloImport({
      source: session.transcripts,
      workspaceId: config.workspaceId,
      sessionStrategy: config.sessionStrategy,
      userPeerId: config.userPeerId,
      agentPeerId: config.agentPeerId,
    })
  } catch (error) {
    await session.dialogs.alert({
      title: "Honcho import",
      message: error instanceof Error ? error.message : String(error),
    })
    return
  }

  const preview = formatImportPreview(plan)
  if (plan.sessionCount === 0) {
    await session.dialogs.alert({ title: "Honcho import", message: `${preview}\n\nNothing new to import.` })
    return
  }

  const choice = await session.dialogs.select<"preview" | "upload">({
    title: "Import local Kilo transcripts into Honcho?",
    options: [
      { title: "Preview only", value: "preview", description: "Do not upload" },
      { title: "Upload now", value: "upload", description: "Sends conversation content to Honcho" },
    ],
  })
  if (choice !== "upload") {
    if (choice === "preview") await session.dialogs.alert({ title: "Honcho import preview", message: preview })
    return
  }

  try {
    const honcho = createHonchoClient({
      apiKey: config.apiKey,
      baseUrl: config.baseUrl,
      workspaceId: config.workspaceId,
    })
    const result = await executeKiloImport({
      source: session.transcripts,
      workspaceId: config.workspaceId,
      sessionStrategy: config.sessionStrategy,
      agentPeerId: config.agentPeerId,
      honcho,
      userPeerId: config.userPeerId,
      agentObserveMe: config.agentObserveMe,
      observationMode: config.observationMode,
    })
    await session.dialogs.alert({
      title: "Honcho import",
      message: [
        `Imported ${result.uploadedMessages} message(s) across ${result.uploadedSessions} session(s) into ${config.workspaceId}.`,
        result.errors.length > 0
          ? `${result.errors.length} session(s) failed.`
          : "Honcho will reason over them; no restart needed.",
      ].join("\n"),
    })
  } catch (error) {
    await session.dialogs.alert({
      title: "Honcho import failed",
      message: error instanceof Error ? error.message : String(error),
    })
  }
}

export const runRecall = async (session: TuiSession) => {
  const { kiloSessionId } = session.liveStatus(await readGlobalSettings())
  if (!kiloSessionId) {
    await session.dialogs.alert({
      title: "Honcho recall",
      message: "Open a Kilo session to see the memory Honcho added to it.",
    })
    return
  }
  const activity = await readActivity(resolveConfigPath(), kiloSessionId)
  await session.dialogs.view({ title: "Honcho recall", body: recallMessage(activity) })
}

export const COMMANDS: readonly TuiCommandSpec[] = [
  {
    id: "honcho.setup",
    title: "Honcho Setup",
    description: "Configure Honcho Cloud or local settings for Kilo",
    slash: "honcho:setup",
    run: runSetup,
  },
  {
    id: "honcho.status",
    title: "Honcho Status",
    description: "Show Honcho runtime health for the current Kilo session",
    slash: "honcho:status",
    run: runStatus,
  },
  {
    id: "honcho.recall",
    title: "Honcho Recall",
    description: "Show the memory Honcho added to this Kilo session",
    slash: "honcho:recall",
    run: runRecall,
  },
  {
    id: "honcho.settings",
    title: "Honcho Settings",
    description: "Show effective Honcho config values for Kilo",
    slash: "honcho:settings",
    run: runSettings,
  },
  {
    id: "honcho.config",
    title: "Honcho Config",
    description: "Edit shared Honcho config fields from ~/.honcho/config.json",
    slash: "honcho:config",
    run: runConfig,
  },
  {
    id: "honcho.import",
    title: "Honcho Import",
    description: "Preview or import local Kilo transcripts into Honcho",
    slash: "honcho:import",
    run: runImport,
  },
]

export const runGuarded = async (session: TuiSession, command: TuiCommandSpec) => {
  try {
    await command.run(session)
  } catch (error) {
    await session.dialogs.alert({
      title: command.title,
      message: error instanceof Error ? error.message : String(error),
    })
  }
}
