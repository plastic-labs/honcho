import { expect, test } from "bun:test"
import os from "node:os"
import path from "node:path"
import { mkdtemp, writeFile } from "node:fs/promises"

import { createHonchoRuntimePlugin } from "../dist/index.js"

const withEnv = async (entries, action) => {
  const previous = new Map()
  for (const [key, value] of Object.entries(entries)) {
    previous.set(key, process.env[key])
    if (value === undefined) {
      delete process.env[key]
      continue
    }
    process.env[key] = value
  }

  try {
    return await action()
  } finally {
    for (const [key, value] of previous.entries()) {
      if (value === undefined) {
        delete process.env[key]
        continue
      }
      process.env[key] = value
    }
  }
}

const withMockFetch = async (implementation, action) => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = implementation
  try {
    return await action()
  } finally {
    globalThis.fetch = originalFetch
  }
}

const jsonResponse = (value, init = {}) =>
  new Response(JSON.stringify(value), {
    status: init.status ?? 200,
    headers: { "content-type": "application/json" },
  })

const summary = (content) => ({
  content,
  message_id: "msg-summary",
  summary_type: "short",
  created_at: new Date(0).toISOString(),
  token_count: 12,
})

const createHonchoFetch = ({ failStableHydration = false } = {}) => {
  const calls = []
  const fetch = async (url, init = {}) => {
    const target = new URL(typeof url === "string" ? url : url.toString())
    const method = init.method || "GET"
    const body = typeof init.body === "string" ? JSON.parse(init.body) : null
    calls.push({ method, pathname: target.pathname, search: target.searchParams, body })

    if (method === "POST" && target.pathname === "/v3/workspaces") {
      return jsonResponse({ id: body.id, metadata: {}, configuration: {} })
    }

    if (method === "POST" && target.pathname === "/v3/workspaces/kilo/peers") {
      return jsonResponse({
        id: body.id,
        metadata: {},
        configuration: {},
        created_at: new Date(0).toISOString(),
      })
    }

    if (method === "POST" && target.pathname === "/v3/workspaces/kilo/sessions") {
      return jsonResponse({
        id: body.id,
        metadata: {},
        configuration: {},
        created_at: new Date(0).toISOString(),
        is_active: true,
      })
    }

    if (method === "POST" && /\/v3\/workspaces\/kilo\/sessions\/[^/]+\/peers$/.test(target.pathname)) {
      return new Response(null, { status: 204 })
    }
    if (method === "POST" && /\/v3\/workspaces\/kilo\/sessions\/[^/]+\/messages$/.test(target.pathname)) {
      return jsonResponse([
        {
          id: "msg-created",
          content: body.content,
          created_at: new Date().toISOString(),
        },
      ])
    }

    if (method === "GET" && /\/v3\/workspaces\/kilo\/peers\/[^/]+\/context$/.test(target.pathname)) {
      if (failStableHydration) {
        return jsonResponse({ message: "context unavailable" }, { status: 400 })
      }
      const peerId = decodeURIComponent(target.pathname.split("/").at(-2))
      return jsonResponse({
        peer_id: peerId,
        target_id: null,
        representation: peerId === "kilo"
          ? "The assistant is working on kilo-honcho."
          : "The user prefers concise engineering analysis.",
        peer_card: ["Keep changes narrowly scoped."],
      })
    }

    if (method === "GET" && /\/v3\/workspaces\/kilo\/sessions\/[^/]+\/summaries$/.test(target.pathname)) {
      if (failStableHydration) {
        return jsonResponse({ message: "summaries unavailable" }, { status: 400 })
      }
      return jsonResponse({
        id: decodeURIComponent(target.pathname.split("/").at(-2)),
        short_summary: summary("Recent work focused on Honcho memory injection."),
        long_summary: null,
      })
    }

    if (method === "POST" && /\/v3\/workspaces\/kilo\/peers\/[^/]+\/chat$/.test(target.pathname)) {
      if (failStableHydration) {
        return jsonResponse({ message: "chat unavailable" }, { status: 400 })
      }
      return jsonResponse({ content: "Durable project memory is available." })
    }

    if (method === "GET" && /\/v3\/workspaces\/kilo\/sessions\/[^/]+\/context$/.test(target.pathname)) {
      return jsonResponse({
        messages: [],
        summary: summary("Prompt-specific session summary."),
        peer_representation: `Prompt memory for ${target.searchParams.get("search_query")}`,
        peer_card: null,
      })
    }

    throw new Error(`Unexpected Honcho request in test: ${method} ${target.pathname}`)
  }
  fetch.calls = calls
  return fetch
}

const createPluginHarness = async (rootDir, configPath) => {
  const plugin = createHonchoRuntimePlugin(configPath ? { configPath } : undefined)
  return plugin({
    client: {
      app: {
        log: async () => undefined,
      },
    },
    project: {
      id: "kilo",
      worktree: rootDir,
    },
    directory: rootDir,
    worktree: rootDir,
    serverUrl: new URL("http://127.0.0.1:4096"),
    $: {},
  })
}

const runWithHarness = async (action, fetchOptions, settings) => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-context-root-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-context-home-"))
  const configPath = settings ? path.join(homeDir, "honcho.json") : undefined
  if (configPath) {
    await writeFile(configPath, `${JSON.stringify(settings, null, 2)}\n`, "utf-8")
  }
  const fetch = createHonchoFetch(fetchOptions)
  return withMockFetch(fetch, () =>
    withEnv({
      HOME: homeDir,
      USER: "test-user",
      XDG_CONFIG_HOME: undefined,
      HONCHO_API_KEY: "test-key",
      HONCHO_URL: undefined,
      HONCHO_BASE_URL: undefined,
    }, async () => {
      const hooks = await createPluginHarness(rootDir, configPath)
      return action({ hooks, fetch })
    }),
  )
}

const systemInput = (extra = {}) => ({
  sessionID: "ses-test",
  model: { providerID: "test-provider", modelID: "test-model" },
  ...extra,
})

test("system transform injects Honcho memory when Kilo provides no prompt text", async () => {
  await runWithHarness(async ({ hooks }) => {
    const output = { system: [] }

    await hooks["experimental.chat.system.transform"](systemInput(), output)

    expect(output.system).toHaveLength(2)
    expect(output.system[0]).toContain("## Honcho Memory")
    expect(output.system[1]).toContain("The user prefers concise engineering analysis.")
    expect(output.system[1]).not.toContain("Prompt memory for")
  })
})

test("tools recall mode injects instructions without hydrating stable context", async () => {
  await runWithHarness(async ({ hooks, fetch }) => {
    const output = { system: [] }

    await hooks["experimental.chat.system.transform"](systemInput(), output)

    expect(output.system).toHaveLength(1)
    expect(output.system[0]).toContain("## Honcho Memory")
    expect(output.system.join("\n")).not.toContain("The user prefers concise engineering analysis.")
    expect(fetch.calls).toHaveLength(0)
  }, undefined, { hosts: { kilo: { recallMode: "tools" } } })
})

test("system transform seals the stable context on the first turn", async () => {
  const originalNow = Date.now
  try {
    let now = 1_000_000
    Date.now = () => now

    await runWithHarness(async ({ hooks, fetch }) => {
      const firstOutput = { system: [] }
      await hooks["experimental.chat.system.transform"](systemInput(), firstOutput)
      const callsAfterFirstTurn = fetch.calls.length

      now += 301_000
      const secondOutput = { system: [] }
      await hooks["experimental.chat.system.transform"](systemInput(), secondOutput)

      expect(firstOutput.system).toHaveLength(2)
      expect(secondOutput.system).toEqual(firstOutput.system)
      expect(fetch.calls).toHaveLength(callsAfterFirstTurn)
    })
  } finally {
    Date.now = originalNow
  }
})

test("chat.message skips recall for trivial prompt text", async () => {
  await runWithHarness(async ({ hooks, fetch }) => {
    const chatOutput = {
      message: { id: "msg-trivial", role: "user", time: { created: Date.now() } },
      parts: [{ type: "text", text: "ok" }],
    }

    await hooks["chat.message"]({ sessionID: "ses-test" }, chatOutput)

    expect(chatOutput.parts).toHaveLength(1)
    const sessionContextCalls = fetch.calls.filter(
      (call) => call.method === "GET" && /\/sessions\/[^/]+\/context$/.test(call.pathname),
    )
    expect(sessionContextCalls).toHaveLength(0)
  })
})

test("chat.message appends prompt-specific memory as a synthetic part", async () => {
  await runWithHarness(async ({ hooks, fetch }) => {
    const chatOutput = {
      message: { id: "msg-prompt", role: "user", time: { created: Date.now() } },
      parts: [{ type: "text", text: "fix memory injection" }],
    }

    await hooks["chat.message"]({ sessionID: "ses-test" }, chatOutput)

    const synthetic = chatOutput.parts.find((part) => part.type === "text" && part.synthetic)
    expect(synthetic).toBeDefined()
    expect(synthetic?.messageID).toBe("msg-prompt")
    expect(synthetic?.id).toMatch(/^prt_[0-9a-f]{12}[A-Za-z0-9_-]{14}$/)
    expect(synthetic?.text).toContain("Prompt memory for memory-injection")

    const targeted = fetch.calls.find(
      (call) =>
        call.method === "GET" &&
        /\/sessions\/[^/]+\/context$/.test(call.pathname) &&
        call.search.get("search_query") === "memory-injection",
    )
    expect(targeted).toBeDefined()
    expect(targeted.search.get("peer_perspective")).toBe("user")
    expect(targeted.search.get("peer_target")).toBe("user")

    const systemOutput = { system: [] }
    await hooks["experimental.chat.system.transform"](systemInput(), systemOutput)
    expect(systemOutput.system.join("\n")).not.toContain("Prompt memory for")
  })
})

test("tool.execute.after records significant tool use to Honcho", async () => {
  await runWithHarness(async ({ hooks, fetch }) => {
    await hooks["tool.execute.after"](
      { tool: "bash", sessionID: "ses-tool", callID: "call-1", args: { command: "npm test" } },
      { title: "Bash output", output: "Tests passed", metadata: {} },
    )

    const saved = fetch.calls.find(
      (call) =>
        call.method === "POST" &&
        /\/sessions\/[^/]+\/messages$/.test(call.pathname) &&
        call.body?.messages?.some((msg) => msg.content?.includes("[Tool] Ran: npm test")),
    )
    expect(saved).toBeDefined()
  })
})
