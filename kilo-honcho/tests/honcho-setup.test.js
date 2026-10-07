import { afterEach, beforeEach, expect, test } from "bun:test"
import os from "node:os"
import path from "node:path"
import { mkdir, mkdtemp, readFile, stat, writeFile } from "node:fs/promises"

import { __testing, createHonchoRuntimePlugin } from "../dist/index.js"

// Tests set HONCHO_* themselves; values inherited from the shell would override the config under test.
const HONCHO_ENV = ["HONCHO_API_KEY", "HONCHO_URL", "HONCHO_BASE_URL", "HONCHO_PEER_NAME", "HONCHO_WORKSPACE", "HONCHO_WORKSPACE_ID", "HONCHO_AI_PEER"]
const inheritedEnv = {}
beforeEach(() => {
  for (const key of HONCHO_ENV) {
    inheritedEnv[key] = process.env[key]
    delete process.env[key]
  }
})
afterEach(() => {
  for (const key of HONCHO_ENV) {
    if (inheritedEnv[key] === undefined) delete process.env[key]
    else process.env[key] = inheritedEnv[key]
  }
})

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

const successfulValidationFetch = async (url) => {
  const target = typeof url === "string" ? url : url.toString()
  if (target.endsWith("/v3/workspaces")) {
    return new Response(JSON.stringify({ id: "kilo", metadata: {}, configuration: {} }), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  }

  if (target.includes("/v3/workspaces/kilo/sessions")) {
    return new Response(
      JSON.stringify({
        id: "setup-check-kilo",
        metadata: {},
        configuration: {},
        created_at: new Date().toISOString(),
        is_active: true,
      }),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      },
    )
  }

  throw new Error(`Unexpected validation request in test: ${target}`)
}

const createPluginHarness = async (rootDir) => {
  const plugin = createHonchoRuntimePlugin()
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

const toolContext = (rootDir) => ({
  sessionID: "ses_test",
  messageID: "msg_test",
  agent: "build",
  directory: rootDir,
  worktree: rootDir,
  abort: new AbortController().signal,
  metadata() {},
  async ask() {},
})

test("honcho_setup saves the peer name and can join another tool's workspace", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-setup-cloud-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-home-"))
  const sharedConfigPath = path.join(homeDir, ".honcho", "config.json")

  await mkdir(path.dirname(sharedConfigPath), { recursive: true })
  await writeFile(sharedConfigPath, JSON.stringify({ apiKey: "new-key" }))

  await withMockFetch(successfulValidationFetch, async () => {
    await withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
      const hooks = await createPluginHarness(rootDir)
      const result = JSON.parse(
        await hooks.tool.honcho_setup.execute({ peerName: "custom-peer", workspace: "claude-code" }, toolContext(rootDir)),
      )
      const persisted = JSON.parse(await readFile(sharedConfigPath, "utf-8"))

      expect(result.ok).toBe(true)
      expect(persisted.peerName).toBe("custom-peer")
      expect(persisted.apiKey).toBe("new-key")
      expect(persisted.hosts.kilo.workspace).toBe("claude-code")
      expect(persisted.hosts.kilo.removeUserPrefix).toBeUndefined()
      // The shared file holds the key, so only its owner may read it.
      expect((await stat(sharedConfigPath)).mode & 0o777).toBe(0o600)
    })
  })
})

test("honcho_setup takes no API key and points the user at the setup command", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-setup-nokey-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-home-nokey-"))
  await withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
    const hooks = await createPluginHarness(rootDir)
    expect(Object.keys(hooks.tool.honcho_setup.args)).not.toContain("apiKey")
    const result = JSON.parse(await hooks.tool.honcho_setup.execute({ peerName: "alice" }, toolContext(rootDir)))
    expect(result.ok).toBe(false)
    expect(result.message).toContain("npx @honcho-ai/kilo-honcho setup")
  })
})

test("honcho_setup does not persist when cloud auth validation fails", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-setup-invalid-auth-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-home-invalid-auth-"))
  const sharedConfigPath = path.join(homeDir, ".honcho", "config.json")
  await withMockFetch(
    async () =>
      new Response(JSON.stringify({ detail: "Invalid API key" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      }),
    async () => {
      await withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
        await mkdir(path.dirname(sharedConfigPath), { recursive: true })
        await writeFile(sharedConfigPath, JSON.stringify({ apiKey: "bad-key" }))
        const hooks = await createPluginHarness(rootDir)
        const result = JSON.parse(await hooks.tool.honcho_setup.execute({ peerName: "alice" }, toolContext(rootDir)))

        expect(result.ok).toBe(false)
        expect(result.error).toMatch(/Invalid API key/i)
        expect(JSON.parse(await readFile(sharedConfigPath, "utf-8")).peerName).toBeUndefined()
      })
    },
  )
})

test("honcho_status ignores a local .kilo/honcho.json", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-ignore-local-config-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-home-ignore-local-"))
  const sharedConfigPath = path.join(homeDir, ".honcho", "config.json")
  const localConfigPath = path.join(rootDir, ".kilo", "honcho.json")

  await mkdir(path.dirname(sharedConfigPath), { recursive: true })
  await mkdir(path.dirname(localConfigPath), { recursive: true })
  await writeFile(
    sharedConfigPath,
    JSON.stringify({
      peerName: "user",
      apiKey: "shared-key",
      baseUrl: "https://api.honcho.dev",
      hosts: { kilo: { aiPeer: "kilo", workspace: "kilo" } },
    }),
  )
  await writeFile(localConfigPath, JSON.stringify({ baseUrl: "http://127.0.0.1:9000", workspace: "local-workspace" }))

  await withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
    const hooks = await createPluginHarness(rootDir)
    const result = JSON.parse(await hooks.tool.honcho_status.execute({}, toolContext(rootDir)))

    expect(result.configPath).toBe(sharedConfigPath)
    expect(result.baseUrl).toBe("https://api.honcho.dev")
    expect(result.workspace).toBe("kilo")
  })
})

test("honcho_status lets HONCHO_* env values override ~/.honcho/config.json", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-env-overrides-file-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-home-env-overrides-file-"))
  const sharedConfigPath = path.join(homeDir, ".honcho", "config.json")

  await mkdir(path.dirname(sharedConfigPath), { recursive: true })
  await writeFile(
    sharedConfigPath,
    JSON.stringify({
      peerName: "user",
      apiKey: "file-key",
      baseUrl: "https://api.honcho.dev",
      hosts: { kilo: { aiPeer: "kilo", workspace: "file-workspace" } },
    }),
  )

  await withEnv(
    {
      HOME: homeDir,
      USER: "ignored-user",
      XDG_CONFIG_HOME: undefined,
      HONCHO_API_KEY: "env-key",
      HONCHO_BASE_URL: "http://127.0.0.1:8000",
      HONCHO_WORKSPACE: "env-workspace",
    },
    async () => {
      const hooks = await createPluginHarness(rootDir)
      const result = JSON.parse(await hooks.tool.honcho_status.execute({}, toolContext(rootDir)))

      expect(result.baseUrl).toBe("http://127.0.0.1:8000")
      expect(result.workspace).toBe("env-workspace")
    },
  )
})

test("without a peerName the user peer is the OS user name, and the plugin writes no config", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-fresh-root-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-fresh-home-"))
  const sharedConfigPath = path.join(homeDir, ".honcho", "config.json")

  await withEnv({ HOME: homeDir, USER: "first.last", XDG_CONFIG_HOME: undefined }, async () => {
    const hooks = await createPluginHarness(rootDir)
    const result = JSON.parse(await hooks.tool.honcho_status.execute({}, toolContext(rootDir)))

    expect(result.observationMode).toBe("unified")
    // Honcho rejects "." in ids, so only that character changes.
    expect(result.peers.userPeer.id).toBe("first-last")
    await expect(readFile(sharedConfigPath, "utf-8")).rejects.toThrow()
  })
})

test("a config from another Honcho tool gives Kilo the same peer, unprefixed and unified", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-shared-root-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-shared-home-"))
  const sharedConfigPath = path.join(homeDir, ".honcho", "config.json")

  await mkdir(path.dirname(sharedConfigPath), { recursive: true })
  const initialConfig = {
    peerName: "Alice",
    apiKey: "shared-key",
    hosts: { claude_code: { workspace: "claude-code", aiPeer: "claude" } },
  }
  await writeFile(sharedConfigPath, JSON.stringify(initialConfig, null, 2))

  await withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
    const hooks = await createPluginHarness(rootDir)
    const result = JSON.parse(await hooks.tool.honcho_status.execute({}, toolContext(rootDir)))
    const persisted = JSON.parse(await readFile(sharedConfigPath, "utf-8"))

    expect(result.observationMode).toBe("unified")
    expect(result.peers.userPeer.id).toBe("Alice")
    expect(result.workspace).toBe("kilo")
    expect(persisted).toEqual(initialConfig)
  })
})

test("a peerName equal to aiPeer fails with a clear error instead of renaming the user", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-collide-root-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-collide-home-"))
  const cfg = path.join(homeDir, ".honcho", "config.json")
  await mkdir(path.dirname(cfg), { recursive: true })

  await withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
    await writeFile(cfg, JSON.stringify({ peerName: "kilo", apiKey: "key", hosts: { kilo: { aiPeer: "kilo" } } }))
    const hooks = await createPluginHarness(rootDir)
    const result = JSON.parse(await hooks.tool.honcho_chat.execute({ query: "what do you know?" }, toolContext(rootDir)))
    expect(result.ok).toBe(false)
    expect(result.error).toMatch(/peerName and aiPeer are both 'kilo'/)
  })
})

test("hosts.kilo.apiKey is used when the root apiKey is absent", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-host-key-root-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-host-key-home-"))
  await mkdir(path.join(homeDir, ".honcho"), { recursive: true })
  const cfg = path.join(homeDir, ".honcho", "config.json")

  await withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined, HONCHO_API_KEY: undefined }, async () => {
    await writeFile(cfg, JSON.stringify({
      peerName: "alice",
      baseUrl: "http://127.0.0.1:8000",
      hosts: { kilo: { workspace: "kilo", aiPeer: "kilo", apiKey: "host-kilo-jwt" } },
    }))
    const hooks = await createPluginHarness(rootDir)
    const env = { env: {} }
    await hooks["shell.env"]({}, env)
    expect(env.env.HONCHO_API_KEY).toBe("host-kilo-jwt")
  })
})

test("config writes keep an ${VAR} apiKey reference instead of saving the secret", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-setup-envref-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-home-"))
  const sharedConfigPath = path.join(homeDir, ".honcho", "config.json")
  await mkdir(path.dirname(sharedConfigPath), { recursive: true })
  await writeFile(sharedConfigPath, JSON.stringify({ apiKey: "${HONCHO_TEST_SECRET}", peerName: "eri" }))

  await withMockFetch(successfulValidationFetch, async () => {
    await withEnv({ HOME: homeDir, HONCHO_TEST_SECRET: "hch-real-secret", XDG_CONFIG_HOME: undefined }, async () => {
      const hooks = await createPluginHarness(rootDir)
      await hooks.tool.honcho_set_config.execute({ field: "sessionStrategy", value: "per-repo" }, toolContext(rootDir))
      await hooks.tool.honcho_setup.execute({ peerName: "eri" }, toolContext(rootDir))
      const raw = await readFile(sharedConfigPath, "utf-8")

      expect(raw).not.toContain("hch-real-secret")
      expect(JSON.parse(raw).apiKey).toBe("${HONCHO_TEST_SECRET}")
    })
  })
})

test("honcho init's environmentUrl is used when there is no baseUrl", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-envurl-root-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-envurl-home-"))
  const cfg = path.join(homeDir, ".honcho", "config.json")
  await mkdir(path.dirname(cfg), { recursive: true })
  await writeFile(cfg, JSON.stringify({ apiKey: "key", environmentUrl: "https://honcho.example.com" }))

  await withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
    const result = JSON.parse(await (await createPluginHarness(rootDir)).tool.honcho_status.execute({}, toolContext(rootDir)))
    expect(result.baseUrl).toBe("https://honcho.example.com")
  })
})

test("addKiloPlugin adds the package to Kilo's server and TUI config and keeps comments", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "honcho-kilo-config-"))
  await writeFile(path.join(dir, "opencode.jsonc"), '{\n  // my models\n  "plugin": ["other-plugin"]\n}\n')

  const first = await __testing.addKiloPlugin(dir)
  expect(first.map((item) => [path.basename(item.file), item.status])).toEqual([
    ["opencode.jsonc", "added"],
    ["tui.json", "added"],
  ])
  const server = await readFile(path.join(dir, "opencode.jsonc"), "utf-8")
  expect(server).toContain("// my models")
  expect(server).toContain('"other-plugin"')
  expect(server).toContain('"@honcho-ai/kilo-honcho"')
  expect(JSON.parse(await readFile(path.join(dir, "tui.json"), "utf-8")).plugin).toEqual(["@honcho-ai/kilo-honcho"])

  const second = await __testing.addKiloPlugin(dir)
  expect(second.map((item) => item.status)).toEqual(["present", "present"])

  const broken = await mkdtemp(path.join(os.tmpdir(), "honcho-kilo-broken-"))
  await writeFile(path.join(broken, "opencode.json"), "{ not json")
  expect((await __testing.addKiloPlugin(broken))[0].status).toBe("unreadable")
  expect(await readFile(path.join(broken, "opencode.json"), "utf-8")).toBe("{ not json")
})

test("kiloConfigDir follows XDG_CONFIG_HOME like Kilo does", async () => {
  await withEnv({ XDG_CONFIG_HOME: "/tmp/xdg", HOME: "/home/a" }, async () => {
    expect(__testing.kiloConfigDir()).toBe("/tmp/xdg/kilo")
  })
  await withEnv({ XDG_CONFIG_HOME: undefined, HOME: "/home/a" }, async () => {
    expect(__testing.kiloConfigDir()).toBe("/home/a/.config/kilo")
  })
})
