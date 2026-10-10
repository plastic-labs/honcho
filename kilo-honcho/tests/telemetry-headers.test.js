import { expect, test } from "bun:test"
import os from "node:os"
import path from "node:path"
import { mkdtemp, readFile } from "node:fs/promises"

import { createHonchoRuntimePlugin } from "../dist/index.js"

const pluginVersion = async () =>
  JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf-8")).version

test("every Honcho request carries host, plugin, and current agent model telemetry headers", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-telemetry-"))
  const env = { HOME: rootDir, HONCHO_API_KEY: "telemetry-key", HONCHO_CONFIG_PATH: undefined, HONCHO_URL: undefined, HONCHO_BASE_URL: undefined }
  const saved = Object.fromEntries(Object.keys(env).map((key) => [key, process.env[key]]))
  const originalFetch = globalThis.fetch
  const calls = []
  // Stand-in Honcho API: records headers, answers every create with the id it was sent.
  globalThis.fetch = async (url, init = {}) => {
    calls.push(new Headers(init.headers))
    const body = typeof init.body === "string" ? JSON.parse(init.body) : {}
    const payload = /\/messages$/.test(new URL(url).pathname)
      ? []
      : { id: body.id, metadata: {}, configuration: {}, created_at: new Date(0).toISOString(), is_active: true }
    return new Response(JSON.stringify(payload), { headers: { "content-type": "application/json" } })
  }
  const since = (start) => calls.slice(start)
  try {
    for (const [key, value] of Object.entries(env)) value === undefined ? delete process.env[key] : (process.env[key] = value)
    const hooks = await createHonchoRuntimePlugin()({
      client: { app: { log: async () => undefined } },
      project: { id: "kilo", worktree: rootDir },
      directory: rootDir,
      worktree: rootDir,
      serverUrl: new URL("http://127.0.0.1:4096"),
      $: {},
    })

    // Kilo never hands the plugin its version; Session.version on session.created names it.
    await hooks.event({ event: { type: "session.created", properties: { info: { id: "ses_test", version: "1.18.23" } } } })
    expect(calls.length).toBeGreaterThan(0)
    for (const headers of calls) {
      expect(headers.get("X-Honcho-Host")).toBe(`kilo/1.18.23 (${process.platform})`)
      expect(headers.get("X-Honcho-Plugin")).toBe(`kilo-honcho/${await pluginVersion()}`)
      expect(headers.has("X-Honcho-Agent-Model")).toBe(false)
    }

    // The resolved model rides on the user message; providerID/modelID disambiguates across providers.
    let start = calls.length
    await hooks["chat.message"](
      { sessionID: "ses_test" },
      { message: { time: { created: 0 }, model: { providerID: "anthropic", modelID: "claude-sonnet-4-5" } }, parts: [{ type: "text", text: "hi" }] },
    )
    expect(since(start).length).toBeGreaterThan(0)
    for (const headers of since(start)) expect(headers.get("X-Honcho-Agent-Model")).toBe("anthropic/claude-sonnet-4-5")

    // The completed assistant message names the model that answered, so a mid-session switch is tracked.
    start = calls.length
    await hooks.event({
      event: {
        type: "message.updated",
        properties: { info: { id: "msg_1", sessionID: "ses_test", role: "assistant", providerID: "anthropic", modelID: "claude-opus-4", time: { created: 1, completed: 2 } } },
      },
    })
    expect(since(start).length).toBeGreaterThan(0)
    for (const headers of since(start)) expect(headers.get("X-Honcho-Agent-Model")).toBe("anthropic/claude-opus-4")
  } finally {
    globalThis.fetch = originalFetch
    for (const [key, value] of Object.entries(saved)) value === undefined ? delete process.env[key] : (process.env[key] = value)
  }
})
