import { expect, test } from "bun:test"
import os from "node:os"
import path from "node:path"
import { mkdtemp } from "node:fs/promises"

import { planKiloImport } from "../dist/import.js"

test("import walks every project with scope=project and reads whole transcripts in order", async () => {
  const now = Date.now()
  const dir = await mkdtemp(path.join(os.tmpdir(), "honcho-import-"))
  const text = (id, body, extra = {}) => ({ id, sessionID: "s", messageID: id, type: "text", text: body, ...extra })
  const calls = []
  const client = {
    project: { list: async () => ({ data: [{ id: "global", worktree: "/" }, { id: "repo", worktree: dir }] }) },
    session: {
      list: async (params) => {
        calls.push(params)
        return { data: params.directory === "/" ? [{ id: "a", directory: os.homedir(), title: "A", time: { created: now, updated: now } }] : [] }
      },
      // Served newest-first with a tool-only message and an ignored part to prove filtering + sorting.
      messages: async (params) => {
        calls.push(params)
        return {
          data: [
            { info: { id: "m3", role: "assistant", time: { created: now + 2 } }, parts: [{ id: "p3", sessionID: "s", messageID: "m3", type: "tool" }] },
            { info: { id: "m2", role: "assistant", time: { created: now + 1 } }, parts: [text("p2", "hi back")] },
            { info: { id: "m1", role: "user", time: { created: now } }, parts: [text("p1", " hello "), text("p1b", "hidden", { ignored: true })] },
          ],
        }
      },
    },
  }

  const plan = await planKiloImport({
    client,
    workspaceId: "ws",
    sessionStrategy: "per-session",
    agentPeerId: "kilo",
    statePath: path.join(dir, "state.json"),
    days: 7,
    includeMessages: true,
  })

  expect(calls.map((c) => c.directory)).toEqual(["/", dir, undefined])
  expect(calls[0]).toMatchObject({ scope: "project", roots: true, limit: expect.any(Number) })
  expect(calls[0].start).toBeGreaterThan(now - 8 * 86_400_000)
  expect(calls[2]).toEqual({ sessionID: "a" })
  expect(plan.sessions[0].messages).toEqual([
    { role: "user", content: "hello", createdAt: new Date(now).toISOString() },
    { role: "assistant", content: "hi back", createdAt: new Date(now + 1).toISOString() },
  ])
})
