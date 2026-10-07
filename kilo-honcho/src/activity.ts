import { randomBytes } from "node:crypto"
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises"
import path from "node:path"
import { HOST_ID, isRecord } from "./core.js"

/**
 * What Honcho did for one Kilo session. The server writes it; the TUI sidebar and
 * `/honcho:recall` read it, since recall never enters Kilo's own session history.
 */
export type HonchoActivity = {
  kiloSessionId: string
  updatedAt: string
  state: "active" | "unconfigured" | "error"
  error?: string
  workspace?: string
  userPeer?: string
  recallMode?: string
  saved: number
  profile?: { at: string; conclusions: number; text: string }
  recall?: { at: string; messageId: string; conclusions: number; text: string }
}

export type ActivityPatch = Partial<Omit<HonchoActivity, "kiloSessionId" | "updatedAt">>

const SESSION_ID_PATTERN = /^[A-Za-z0-9_-]+$/
export const ACTIVITY_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000

export const activityDir = (configPath: string) => path.join(path.dirname(configPath), HOST_ID, "sessions")

// The id becomes a file name, so anything outside Kilo's id alphabet is refused.
export const activityPath = (configPath: string, kiloSessionId: string) =>
  SESSION_ID_PATTERN.test(kiloSessionId) ? path.join(activityDir(configPath), `${kiloSessionId}.json`) : null

// Honcho prints each conclusion on a line that starts with its "[YYYY-MM-DD HH:MM:SS]" stamp; inductive ones start with "**Pattern**".
const CONCLUSION_LINE = /^(?:\[id:[^\]]+\] ?)?(?:\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\]|\s*\*\*Pattern\*\*)/gm

export const countConclusions = (text: string) => text.match(CONCLUSION_LINE)?.length ?? 0

const parseActivity = (value: unknown, kiloSessionId: string): HonchoActivity | null => {
  if (!isRecord(value) || value.kiloSessionId !== kiloSessionId) return null
  const state = value.state
  if (state !== "active" && state !== "unconfigured" && state !== "error") return null
  return { ...(value as HonchoActivity), saved: typeof value.saved === "number" ? value.saved : 0 }
}

export const readActivity = async (configPath: string, kiloSessionId: string): Promise<HonchoActivity | null> => {
  const file = activityPath(configPath, kiloSessionId)
  if (!file) return null
  try {
    return parseActivity(JSON.parse(await readFile(file, "utf-8")), kiloSessionId)
  } catch {
    return null
  }
}

const writeActivity = async (file: string, activity: HonchoActivity) => {
  await mkdir(path.dirname(file), { recursive: true, mode: 0o700 })
  // Write then rename, so a reader never sees half a file.
  const tmp = `${file}.${randomBytes(6).toString("hex")}.tmp`
  await writeFile(tmp, `${JSON.stringify(activity, null, 2)}\n`, { encoding: "utf-8", mode: 0o600 })
  await rename(tmp, file)
}

export const pruneActivity = async (configPath: string, now = Date.now()) => {
  const dir = activityDir(configPath)
  let names: string[]
  try {
    names = await readdir(dir)
  } catch {
    return 0
  }
  let removed = 0
  for (const name of names) {
    if (!name.endsWith(".json") && !name.endsWith(".tmp")) continue
    const file = path.join(dir, name)
    try {
      if (now - (await stat(file)).mtimeMs > ACTIVITY_MAX_AGE_MS) {
        await rm(file, { force: true })
        removed += 1
      }
    } catch {
      // Another process removed it first.
    }
  }
  return removed
}

const sameContent = (a: HonchoActivity, b: HonchoActivity) =>
  JSON.stringify({ ...a, updatedAt: "" }) === JSON.stringify({ ...b, updatedAt: "" })

/** Keeps one record per Kilo session in memory and writes each change to disk in order. */
export const createActivityRecorder = (configPath: string) => {
  const records = new Map<string, HonchoActivity>()
  const queues = new Map<string, Promise<void>>()

  const enqueue = (kiloSessionId: string, work: () => Promise<void>) => {
    const next = (queues.get(kiloSessionId) ?? Promise.resolve()).then(work).catch(() => undefined)
    queues.set(kiloSessionId, next)
    void next.finally(() => {
      if (queues.get(kiloSessionId) === next) queues.delete(kiloSessionId)
    })
    return next
  }

  const update = (
    kiloSessionId: string,
    change: ActivityPatch | ((current: HonchoActivity) => ActivityPatch),
  ) => {
    const file = activityPath(configPath, kiloSessionId)
    if (!file) return Promise.resolve()
    return enqueue(kiloSessionId, async () => {
      // A resumed session continues the counts its last run left on disk.
      const current =
        records.get(kiloSessionId) ??
        (await readActivity(configPath, kiloSessionId)) ?? {
          kiloSessionId,
          updatedAt: new Date().toISOString(),
          state: "active",
          saved: 0,
        }
      const patch = typeof change === "function" ? change(current) : change
      const next: HonchoActivity = { ...current, ...patch, kiloSessionId, updatedAt: new Date().toISOString() }
      if (next.state !== "error") delete next.error
      records.set(kiloSessionId, next)
      if (sameContent(current, next)) return
      await writeActivity(file, next)
    })
  }

  const remove = (kiloSessionId: string) => {
    const file = activityPath(configPath, kiloSessionId)
    records.delete(kiloSessionId)
    if (!file) return Promise.resolve()
    return enqueue(kiloSessionId, () => rm(file, { force: true }))
  }

  const flush = () => Promise.allSettled([...queues.values()])

  return { update, remove, flush }
}

export type ActivityRecorder = ReturnType<typeof createActivityRecorder>
