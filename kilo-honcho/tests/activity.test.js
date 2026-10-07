import { expect, test } from "bun:test"
import os from "node:os"
import path from "node:path"
import { mkdir, mkdtemp, readdir, utimes, writeFile } from "node:fs/promises"

import { __testing } from "../dist/index.js"

const tempConfig = async () => path.join(await mkdtemp(path.join(os.tmpdir(), "honcho-activity-")), "config.json")

test("countConclusions counts Honcho's conclusion lines, not premises or peer card entries", () => {
  const text = [
    "## User Memory Profile",
    "- Prefers tokio",
    "## Explicit Observations",
    "",
    "[2026-10-07 19:32:34] user is starting a Rust CLI called tidepool",
    "[2026-10-07 19:32:35] user runs cargo nextest",
    "## Deductive Observations",
    "",
    "[2026-10-07 19:40:00] user values reproducible tests",
    "   Premises:",
    "   - [2026-10-07 19:32:35] user runs cargo nextest",
    "## Inductive Observations",
    "",
    " **Pattern** [high]: user checks agent claims before accepting them",
  ].join("\n")
  expect(__testing.countConclusions(text)).toBe(4)
  expect(__testing.countConclusions("Session summary:\nWorked on the parser.")).toBe(0)
})

test("activityPath refuses ids that could leave the sessions directory", () => {
  expect(__testing.activityPath("/home/a/.honcho/config.json", "ses_abc-1")).toBe("/home/a/.honcho/kilo/sessions/ses_abc-1.json")
  expect(__testing.activityPath("/home/a/.honcho/config.json", "../config")).toBeNull()
  expect(__testing.activityPath("/home/a/.honcho/config.json", "a/b")).toBeNull()
})

test("concurrent updates are written in order and a new process continues the count", async () => {
  const configPath = await tempConfig()
  const first = __testing.createActivityRecorder(configPath)
  await Promise.all(Array.from({ length: 10 }, () => first.update("ses_count", (current) => ({ saved: current.saved + 1 }))))
  expect((await __testing.readActivity(configPath, "ses_count")).saved).toBe(10)

  const second = __testing.createActivityRecorder(configPath)
  await second.update("ses_count", (current) => ({ saved: current.saved + 1 }))
  expect((await __testing.readActivity(configPath, "ses_count")).saved).toBe(11)
})

test("pruneActivity removes session files older than 14 days and keeps recent ones", async () => {
  const configPath = await tempConfig()
  const dir = path.join(path.dirname(configPath), "kilo", "sessions")
  await mkdir(dir, { recursive: true })
  await writeFile(path.join(dir, "ses_old.json"), "{}")
  await writeFile(path.join(dir, "ses_new.json"), "{}")
  const old = new Date(Date.now() - 15 * 24 * 60 * 60 * 1000)
  await utimes(path.join(dir, "ses_old.json"), old, old)

  expect(await __testing.pruneActivity(configPath)).toBe(1)
  expect(await readdir(dir)).toEqual(["ses_new.json"])
})
