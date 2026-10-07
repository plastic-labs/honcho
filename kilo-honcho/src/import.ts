import { mkdir, readFile, writeFile } from "node:fs/promises"
import path from "node:path"
import type { TuiPlugin } from "@kilocode/plugin/tui"
import { Honcho } from "@honcho-ai/sdk"
import {
  SHARED_SETTINGS_DIR_NAME,
  clampText,
  deriveSessionScope,
  findProjectRoot,
  honchoSessionKey,
  isRecord,
  timestampToIso,
  userHomeDir,
  type SessionStrategy,
} from "./core.js"

/** Kilo SDK client, as handed to the TUI plugin. */
export type KiloClient = Parameters<TuiPlugin>[0]["client"]
type SessionListResult = Awaited<ReturnType<KiloClient["session"]["list"]>>
type MessagesResult = Awaited<ReturnType<KiloClient["session"]["messages"]>>
export type KiloSession = NonNullable<SessionListResult["data"]>[number]
export type KiloMessagePage = NonNullable<MessagesResult["data"]>
export type KiloPart = KiloMessagePage[number]["parts"][number]

/** One local Kilo session as the importer sees it. */
export type TranscriptSession = {
  id: string
  title: string
  directory: string
  timeUpdated: number
}

/** Where transcripts come from. */
export type TranscriptSource = {
  listSessions: (days: number, includeSubagents: boolean) => Promise<TranscriptSession[]>
  readTranscript: (sessionID: string) => Promise<ImportMessage[]>
}

export type ImportMessage = {
  role: "user" | "assistant"
  content: string
  createdAt?: string
}

export type PlannedImportSession = {
  id: string
  title: string
  directory: string
  honchoSessionKey: string
  timeUpdated: number
  messageCount: number
  alreadyImported: boolean
  skippedReason?: string
  messages: ImportMessage[]
}

export type ImportPlan = {
  ok: true
  source: string
  days: number
  sessionCount: number
  messageCount: number
  skippedCount: number
  alreadyImportedCount: number
  sessions: Array<Omit<PlannedImportSession, "messages"> & { messages?: ImportMessage[] }>
}

export type ImportUploadResult = ImportPlan & {
  dryRun: boolean
  uploadedSessions: number
  uploadedMessages: number
  errors: string[]
}

type ImportState = {
  imported: Record<string, number>
}

const IMPORT_STATE_FILE_NAME = "kilo-import-state.json"
const IMPORT_SOURCE = "kilo-sdk"
const MAX_IMPORT_MESSAGE_CHARS = 25_000
const ADD_MESSAGES_BATCH = 40
const SESSION_LIST_LIMIT = 10_000

export const defaultImportStatePath = () =>
  process.env.HONCHO_IMPORT_STATE_PATH ||
  path.join(userHomeDir(), SHARED_SETTINGS_DIR_NAME, IMPORT_STATE_FILE_NAME)

const importStateKey = (workspaceId: string, sessionId: string) => `${workspaceId}::${sessionId}`

const loadImportState = async (statePath: string): Promise<ImportState> => {
  try {
    const parsed = JSON.parse(await readFile(statePath, "utf-8"))
    return isRecord(parsed) && isRecord(parsed.imported)
      ? { imported: parsed.imported as Record<string, number> }
      : { imported: {} }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return { imported: {} }
    }
    throw error
  }
}

const saveImportState = async (statePath: string, state: ImportState) => {
  await mkdir(path.dirname(statePath), { recursive: true })
  await writeFile(statePath, `${JSON.stringify(state, null, 2)}\n`, "utf-8")
}

const importedSessionKey = async ({
  workspaceId,
  sessionStrategy,
  directory,
  sessionId,
  agentPeerId,
}: {
  workspaceId: string
  sessionStrategy: SessionStrategy
  directory: string
  sessionId: string
  agentPeerId: string
}) => {
  const rootDir = findProjectRoot(directory)
  const scope = await deriveSessionScope({
    workspaceId,
    sessionStrategy,
    rootDir,
    repoName: path.basename(rootDir),
    currentDirectory: directory,
    sessionId,
  })
  return honchoSessionKey(sessionStrategy, scope, [agentPeerId])
}

const unwrap = <T>(result: { data?: T; error?: unknown }, what: string): T => {
  if (result.error !== undefined || result.data === undefined) {
    const detail =
      result.error instanceof Error
        ? result.error.message
        : result.error === undefined
          ? "empty response"
          : JSON.stringify(result.error)
    throw new Error(`Kilo ${what} failed: ${detail}`)
  }
  return result.data
}

const extractTextFromParts = (parts: KiloPart[]) => {
  const texts: string[] = []
  for (const part of parts) {
    if (part.type !== "text" || part.ignored === true) continue
    const text = part.text.trim()
    if (text) texts.push(text)
  }
  return texts.join("\n").trim()
}

const extractImportMessages = (page: KiloMessagePage): ImportMessage[] => {
  const out: ImportMessage[] = []
  for (const { info, parts } of page) {
    const content = clampText(extractTextFromParts(parts), MAX_IMPORT_MESSAGE_CHARS)
    if (!content) continue
    out.push({ role: info.role, content, createdAt: timestampToIso(info.time.created) })
  }
  return out
}

const listKiloSessions = async (
  client: KiloClient,
  days: number,
  includeSubagents: boolean,
): Promise<TranscriptSession[]> => {
  // session.list is scoped to one project (chosen by `directory`), so walk every
  // known project and ask for all of its sessions regardless of sub-directory.
  const projects = unwrap(await client.project.list(), "project.list")
  const start = Date.now() - days * 24 * 60 * 60 * 1000
  const seen = new Set<string>()
  const sessions: KiloSession[] = []
  for (const project of projects) {
    const page = unwrap(
      await client.session.list({
        directory: project.worktree,
        scope: "project",
        start,
        roots: !includeSubagents,
        limit: SESSION_LIST_LIMIT,
      }),
      `session.list(${project.worktree})`,
    )
    for (const session of page) {
      if (seen.has(session.id)) continue
      seen.add(session.id)
      sessions.push(session)
    }
  }
  return sessions
    .sort((left, right) => right.time.updated - left.time.updated)
    .map((session) => ({ id: session.id, title: session.title, directory: session.directory, timeUpdated: session.time.updated }))
}

const readSessionTranscript = async (client: KiloClient, sessionID: string): Promise<ImportMessage[]> => {
  // Without `limit` the server pages through the whole transcript itself and
  // returns it oldest-first. `before` is an opaque cursor only exposed via a
  // Link header, so client-side paging is not worth it.
  const page = unwrap(await client.session.messages({ sessionID }), `session.messages(${sessionID})`)
  const ordered = [...page].sort(
    (left, right) => left.info.time.created - right.info.time.created || left.info.id.localeCompare(right.info.id),
  )
  return extractImportMessages(ordered)
}

export const transcriptSourceFromClient = (client: KiloClient): TranscriptSource => ({
  listSessions: (days, includeSubagents) => listKiloSessions(client, days, includeSubagents),
  readTranscript: (sessionID) => readSessionTranscript(client, sessionID),
})

export type PlanImportOptions = {
  /** Kilo SDK client. Ignored when `source` is given. */
  client?: KiloClient
  source?: TranscriptSource
  workspaceId: string
  sessionStrategy: SessionStrategy
  agentPeerId: string
  statePath?: string
  days?: number
  includeSubagents?: boolean
  force?: boolean
  includeMessages?: boolean
}

export const planKiloImport = async (options: PlanImportOptions): Promise<ImportPlan> => {
  const statePath = options.statePath || defaultImportStatePath()
  const days = options.days && options.days > 0 ? options.days : 30

  const source = options.source ?? (options.client ? transcriptSourceFromClient(options.client) : undefined)
  if (!source) throw new Error("Kilo import needs a transcript source")

  const state = await loadImportState(statePath)
  const rows = await source.listSessions(days, options.includeSubagents === true)
  const sessions: PlannedImportSession[] = []

  for (const row of rows) {
    const messages = await source.readTranscript(row.id)
    const alreadyImported = state.imported[importStateKey(options.workspaceId, row.id)] === row.timeUpdated
    const skippedReason =
      messages.length === 0 ? "no text messages" : alreadyImported && !options.force ? "already imported" : undefined
    sessions.push({
      id: row.id,
      title: row.title,
      directory: row.directory,
      honchoSessionKey: await importedSessionKey({
        workspaceId: options.workspaceId,
        sessionStrategy: options.sessionStrategy,
        directory: row.directory,
        sessionId: row.id,
        agentPeerId: options.agentPeerId,
      }),
      timeUpdated: row.timeUpdated,
      messageCount: messages.length,
      alreadyImported,
      skippedReason,
      messages,
    })
  }

  const uploadable = sessions.filter((session) => !session.skippedReason)
  return {
    ok: true,
    source: IMPORT_SOURCE,
    days,
    sessionCount: uploadable.length,
    messageCount: uploadable.reduce((sum, session) => sum + session.messageCount, 0),
    skippedCount: sessions.filter((session) => session.skippedReason && session.skippedReason !== "already imported").length,
    alreadyImportedCount: sessions.filter((session) => session.alreadyImported).length,
    sessions: sessions.map((session) =>
      options.includeMessages ? session : { ...session, messages: undefined },
    ),
  }
}

const addMessagesBatched = async (
  session: Awaited<ReturnType<Honcho["session"]>>,
  messages: ReturnType<Awaited<ReturnType<Honcho["peer"]>>["message"]>[],
) => {
  for (let index = 0; index < messages.length; index += ADD_MESSAGES_BATCH) {
    await session.addMessages(messages.slice(index, index + ADD_MESSAGES_BATCH))
  }
}

export type ExecuteImportOptions = PlanImportOptions & {
  honcho: Honcho
  userPeerId: string
  agentObserveMe: boolean
  statePath?: string
}

export const executeKiloImport = async (options: ExecuteImportOptions): Promise<ImportUploadResult> => {
  const plan = await planKiloImport({ ...options, includeMessages: true })
  const statePath = options.statePath || defaultImportStatePath()
  const state = await loadImportState(statePath)
  const userPeer = await options.honcho.peer(options.userPeerId, { configuration: { observeMe: true } })
  const agentPeer = await options.honcho.peer(options.agentPeerId, {
    configuration: { observeMe: options.agentObserveMe === true },
  })

  let uploadedSessions = 0
  let uploadedMessages = 0
  const errors: string[] = []

  for (const sessionPlan of plan.sessions) {
    if (sessionPlan.skippedReason || !sessionPlan.messages) continue
    try {
      const session = await options.honcho.session(sessionPlan.honchoSessionKey)
      await session.addPeers([
        [options.userPeerId, { observeMe: true, observeOthers: false }],
        [options.agentPeerId, { observeMe: options.agentObserveMe === true, observeOthers: true }],
      ] as never)
      const payload = sessionPlan.messages.map((message) => {
        const peer = message.role === "user" ? userPeer : agentPeer
        return peer.message(message.content, {
          createdAt: message.createdAt,
          metadata: {
            backfill: true,
            source: IMPORT_SOURCE,
            kiloSessionId: sessionPlan.id,
          },
        })
      })
      await addMessagesBatched(session, payload)
      state.imported[importStateKey(options.workspaceId, sessionPlan.id)] = sessionPlan.timeUpdated
      await saveImportState(statePath, state)
      uploadedSessions += 1
      uploadedMessages += sessionPlan.messageCount
    } catch (error) {
      errors.push(`${sessionPlan.id}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  return {
    ...plan,
    sessions: plan.sessions.map(({ messages: _messages, ...session }) => session),
    dryRun: false,
    uploadedSessions,
    uploadedMessages,
    errors,
  }
}
