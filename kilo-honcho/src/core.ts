import { existsSync } from "node:fs"
import { homedir } from "node:os"
import { readFile } from "node:fs/promises"
import path from "node:path"

/** Kilo host id: the `hosts.kilo` config key and the `X-Honcho-Host` header name. */
export const HOST_ID = "kilo"

/** Integration name sent as `X-Honcho-Plugin: kilo-honcho/<version>`; also the log service name. */
export const PLUGIN_ID = "kilo-honcho"

/** npm package name. Kilo uses it as the plugin id and the Marketplace matches installs on it. */
export const PACKAGE_ID = "@honcho-ai/kilo-honcho"

export const SESSION_STRATEGIES = [
  "per-repo",
  "per-directory",
  "per-session",
  "global",
  "git-branch",
  "chat-instance",
] as const

export type SessionStrategy = (typeof SESSION_STRATEGIES)[number]

export const RECALL_MODES = ["hybrid", "context", "tools"] as const
export type RecallMode = (typeof RECALL_MODES)[number]

export const OBSERVATION_MODES = ["unified", "directional"] as const
export type ObservationMode = (typeof OBSERVATION_MODES)[number]

export const SETTING_ENUMS = {
  recallMode: RECALL_MODES,
  observationMode: OBSERVATION_MODES,
  sessionStrategy: SESSION_STRATEGIES,
} as const

export const isObservationMode = (value: unknown): value is ObservationMode =>
  typeof value === "string" && (OBSERVATION_MODES as readonly string[]).includes(value)

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

export const stampedHostObservationMode = (
  raw: Record<string, unknown> | null | undefined,
): ObservationMode | null => {
  if (!isRecord(raw) || !isRecord(raw.hosts) || !isRecord(raw.hosts.kilo)) {
    return null
  }
  const value = raw.hosts.kilo.observationMode
  return isObservationMode(value) ? value : null
}

export const needsObservationUpgradePrompt = (raw: Record<string, unknown> | null | undefined) => {
  if (!isRecord(raw) || !isRecord(raw.hosts) || !isRecord(raw.hosts.kilo)) {
    return false
  }
  return stampedHostObservationMode(raw) === null
}

export const observationUpgradeNotice = () =>
  `Observation mode is unset, so this install is still directional (this Kilo agent's view of you).
New installs use unified (your self-collection, shared with other unified agents).
Choose with /honcho:setup or /honcho:config (hosts.kilo.observationMode).
If you switch to unified, you can run /honcho:import to reingest local Kilo transcripts into the new collection.`

export const observationUpgradeNextSteps = () =>
  "Ask the user whether to keep directional or switch to unified. Persist the choice with honcho_set_config field=observationMode. If they choose unified, mention they can run /honcho:import to backfill local Kilo history — optional, not required."

export const unifiedImportFollowUp = () =>
  "Switched to unified. Optionally run /honcho:import to backfill local Kilo transcripts into the user self-collection. Skip if you do not want that history in the new collection."

export const directionalKeepFollowUp = () =>
  "Keeping directional. This Kilo agent continues using its own view of you. You can switch later with /honcho:config."

export type HonchoSettings = {
  apiKey: string
  baseUrl: string
  peerName: string
  aiPeer: string
  workspace: string
  recallMode: RecallMode
  observationMode: ObservationMode
  agentObserveMe: boolean
  autoConclusions: boolean
  sessionStrategy: SessionStrategy
  removeUserPrefix: boolean
}

export const DEFAULT_SETTINGS: HonchoSettings = {
  apiKey: "",
  baseUrl: "https://api.honcho.dev",
  peerName: "",
  aiPeer: HOST_ID,
  workspace: HOST_ID,
  recallMode: "hybrid",
  // Fallback when the field is omitted: keep directional so existing installs
  // do not silently switch collections. New installs stamp unified on disk.
  observationMode: "directional",
  // Default false: Honcho models the user, not the assistant. Set true to opt into agent self-observation.
  agentObserveMe: false,
  // Default false: Honcho's deriver already reasons over every message, so verbatim keyword copies only add noise.
  autoConclusions: false,
  sessionStrategy: "per-directory",
  // Default false for upgrades: keep the legacy user-<peerName> peer. New installs stamp true.
  removeUserPrefix: false,
}

export const clampText = (value: string, maxChars: number) =>
  value.length > maxChars ? `${value.slice(0, Math.max(0, maxChars - 3))}...` : value

export const timestampToIso = (value: unknown) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined
  return new Date(value).toISOString()
}

export const isLocalBaseUrl = (value: string) => {
  if (!value.trim()) return false
  try {
    const url = new URL(value)
    return ["localhost", "127.0.0.1", "::1"].includes(url.hostname)
  } catch {
    return false
  }
}

export const getNestedValue = (value: Record<string, unknown>, fieldPath: string): unknown =>
  fieldPath.split(".").reduce<unknown>((current, part) => {
    if (!isRecord(current)) return undefined
    return current[part]
  }, value)

export const SHARED_SETTINGS_DIR_NAME = ".honcho"
export const SHARED_SETTINGS_FILE_NAME = "config.json"

export const userHomeDir = () => process.env.HOME || process.env.USERPROFILE || homedir()

export const sharedGlobalSettingsPath = () =>
  path.join(userHomeDir(), SHARED_SETTINGS_DIR_NAME, SHARED_SETTINGS_FILE_NAME)

export const sharedConfigPath = (configPathOverride?: string) =>
  configPathOverride ? path.resolve(configPathOverride) : sharedGlobalSettingsPath()

const endpointBaseUrl = (scope: unknown) => {
  const endpoint = isRecord(scope) ? scope.endpoint : undefined
  return isRecord(endpoint) && typeof endpoint.baseUrl === "string" ? endpoint.baseUrl.trim() : ""
}

/** `baseUrl`, else the `endpoint.baseUrl` other Honcho integrations write (host block, then top level). */
export const resolveBaseUrl = (raw: Record<string, unknown>, hostId = HOST_ID) => {
  const baseUrl = typeof raw.baseUrl === "string" ? raw.baseUrl.trim() : ""
  if (baseUrl) return baseUrl
  return endpointBaseUrl(isRecord(raw.hosts) ? raw.hosts[hostId] : undefined) || endpointBaseUrl(raw)
}

export const normalizeId = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "default"

const hasProjectMarker = (directory: string) =>
  existsSync(path.join(directory, ".git")) ||
  existsSync(path.join(directory, ".kilo")) ||
  existsSync(path.join(directory, ".kilocode"))

export const walkToProjectRoot = (directory: string): string | null => {
  let current = path.resolve(directory)
  while (true) {
    if (hasProjectMarker(current)) return current
    const parent = path.dirname(current)
    if (parent === current) return null
    current = parent
  }
}

export const findProjectRoot = (directory: string) => walkToProjectRoot(directory) ?? path.resolve(directory)

const resolveGitDir = async (rootDir: string): Promise<string | null> => {
  const directGitDir = path.join(rootDir, ".git")
  try {
    const statTarget = await readFile(directGitDir, "utf-8")
    const prefix = "gitdir:"
    if (statTarget.trim().startsWith(prefix)) {
      return path.resolve(rootDir, statTarget.trim().slice(prefix.length).trim())
    }
  } catch {
    if (existsSync(directGitDir)) return directGitDir
  }
  return existsSync(directGitDir) ? directGitDir : null
}

export const deriveGitBranchLabel = async (rootDir: string): Promise<string | null> => {
  const gitDir = await resolveGitDir(rootDir)
  if (!gitDir) return null
  try {
    const head = (await readFile(path.join(gitDir, "HEAD"), "utf-8")).trim()
    const branchPrefix = "ref: refs/heads/"
    if (head.startsWith(branchPrefix)) return normalizeId(head.slice(branchPrefix.length))
  } catch {
    return null
  }
  return null
}

export const deriveSessionScope = async ({
  workspaceId,
  sessionStrategy,
  rootDir,
  repoName,
  currentDirectory,
  sessionId,
}: {
  workspaceId: string
  sessionStrategy: SessionStrategy
  rootDir: string
  repoName: string
  currentDirectory: string
  sessionId: string
}) => {
  if (sessionStrategy === "per-directory") {
    const relativeDirectory = path.relative(rootDir, currentDirectory)
    const directoryLabel =
      relativeDirectory && !relativeDirectory.startsWith("..") && !path.isAbsolute(relativeDirectory)
        ? normalizeId(relativeDirectory.split(path.sep).join("-"))
        : normalizeId(path.basename(currentDirectory))
    return `${workspaceId}:${directoryLabel || normalizeId(repoName)}`
  }

  if (sessionStrategy === "per-session" || sessionStrategy === "chat-instance") {
    return `${workspaceId}:${normalizeId(sessionId)}`
  }

  if (sessionStrategy === "global") {
    return `${workspaceId}:global`
  }

  if (sessionStrategy === "git-branch") {
    const branchLabel = await deriveGitBranchLabel(rootDir)
    return `${workspaceId}:${branchLabel || normalizeId(repoName)}`
  }

  return `${workspaceId}:${normalizeId(repoName)}`
}

// The user peer leads the key so teammates sharing a workspace never land in one session.
export const honchoSessionKey = (
  userPeerId: string,
  sessionStrategy: SessionStrategy,
  sessionScope: string,
  lineage: readonly string[],
) => normalizeId(`${userPeerId}:${sessionStrategy}:${sessionScope}:${lineage.join(":")}`)

export const deriveUserPeerId = (peerName: string, removeUserPrefix: boolean) => {
  const name = peerName.trim() || "user"
  // The sibling claude-honcho / hermes-honcho plugins use the peer name
  // verbatim (no case folding), so removeUserPrefix=true must skip normalizeId
  // to achieve real parity instead of silently lowercasing into a different peer.
  return removeUserPrefix ? name : normalizeId(`user:${name}`)
}

export const resolveSessionPeerIds = (peerName: string, aiPeer: string, removeUserPrefix: boolean) => {
  const agentPeerId = normalizeId(aiPeer || DEFAULT_SETTINGS.aiPeer)
  const name = peerName.trim() || "user"
  let userPeerId = deriveUserPeerId(name, removeUserPrefix)
  if (userPeerId === agentPeerId) {
    // Bare peerName colliding with aiPeer: keep them distinct so user and agent
    // memory do not share a peer. Callers on unguarded paths still assertDistinct.
    userPeerId = normalizeId(`user:${name}`)
  }
  return { userPeerId, agentPeerId }
}
