import { afterEach, beforeEach, expect, test } from "bun:test"
import os from "node:os"
import path from "node:path"
import { mkdir, mkdtemp, readFile, stat, writeFile } from "node:fs/promises"

import { __testing, createHonchoRuntimePlugin } from "../dist/index.js"

// Tests set HONCHO_* themselves; values inherited from the shell would override the config under test.
// KILO_* paths too: tests must never write to a developer's real config or skills directory.
const HONCHO_ENV = ["HONCHO_API_KEY", "HONCHO_URL", "HONCHO_BASE_URL", "HONCHO_PEER_NAME", "HONCHO_WORKSPACE", "HONCHO_WORKSPACE_ID", "HONCHO_AI_PEER", "KILO_HONCHO_CONFIG_PATH", "KILO_CONFIG_DIR"]
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

// A stand-in for the platform's key window, first on PATH, so tests never open a real one.
const withFakeKeyWindow = async (output, action) => {
  const bin = await mkdtemp(path.join(os.tmpdir(), "honcho-fake-dialog-"))
  const script = output === null ? "#!/bin/sh\nexit 1\n" : `#!/bin/sh\necho '${output}'\n`
  for (const name of ["osascript", "zenity"]) await writeFile(path.join(bin, name), script, { mode: 0o755 })
  return withEnv({ PATH: `${bin}:${process.env.PATH}`, DISPLAY: ":0", SSH_CONNECTION: undefined, SSH_CLIENT: undefined, SSH_TTY: undefined }, action)
}

test("honcho_setup takes the key from a window, never from the chat", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-setup-window-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-home-window-"))
  const sharedConfigPath = path.join(homeDir, ".honcho", "config.json")
  await withMockFetch(successfulValidationFetch, () =>
    withFakeKeyWindow("hch-from-window", () =>
      withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
        const hooks = await createPluginHarness(rootDir)
        expect(Object.keys(hooks.tool.honcho_setup.args)).not.toContain("apiKey")
        const output = await hooks.tool.honcho_setup.execute({ peerName: "alice" }, toolContext(rootDir))
        const result = JSON.parse(output)
        expect(result.ok).toBe(true)
        expect(result.message).toContain("Memory starts with the next message")
        // The tool output goes back to the model, so the key must not be in it.
        expect(output).not.toContain("hch-from-window")
        const persisted = JSON.parse(await readFile(sharedConfigPath, "utf-8"))
        expect(persisted.apiKey).toBe("hch-from-window")
        expect(persisted.peerName).toBe("alice")
      }),
    ),
  )
})

test("honcho_setup points at the setup command when the key window is closed", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-setup-nokey-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-home-nokey-"))
  await withFakeKeyWindow(null, () =>
    withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
      const hooks = await createPluginHarness(rootDir)
      const result = JSON.parse(await hooks.tool.honcho_setup.execute({ peerName: "alice" }, toolContext(rootDir)))
      expect(result.ok).toBe(false)
      expect(result.message).toContain("npx @honcho-ai/kilo-honcho setup")
      await expect(readFile(path.join(homeDir, ".honcho", "config.json"), "utf-8")).rejects.toThrow()
    }),
  )
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
    expect(result.sessionUrl).toBe(`https://app.honcho.dev/explore?workspace=kilo&view=sessions&session=${encodeURIComponent(result.sessionName)}`)
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
    const fetch = recordingFetch()
    await withMockFetch(fetch, async () => {
      const hooks = await createPluginHarness(rootDir)
      await hooks.tool.honcho_chat.execute({ query: "hi" }, toolContext(rootDir))
    })
    expect(fetch.requests.length).toBeGreaterThan(0)
    expect(fetch.requests.every((request) => request.auth === "Bearer host-kilo-jwt")).toBe(true)
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

test("addKiloPlugin writes kilo.jsonc and tui.json and keeps comments", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "honcho-kilo-config-"))
  await writeFile(path.join(dir, "kilo.jsonc"), '{\n  // my models\n  "plugin": ["other-plugin"]\n}\n')

  const first = await __testing.addKiloPlugin(dir)
  expect(first.map((item) => [path.basename(item.file), item.status])).toEqual([
    ["kilo.jsonc", "added"],
    ["tui.json", "added"],
  ])
  const server = await readFile(path.join(dir, "kilo.jsonc"), "utf-8")
  expect(server).toContain("// my models")
  expect(server).toContain('"other-plugin"')
  expect(server).toContain('"@honcho-ai/kilo-honcho"')
  expect(JSON.parse(await readFile(path.join(dir, "tui.json"), "utf-8")).plugin).toEqual(["@honcho-ai/kilo-honcho"])

  const second = await __testing.addKiloPlugin(dir)
  expect(second.map((item) => item.status)).toEqual(["present", "present"])
})

test("addKiloPlugin creates kilo.jsonc when Kilo has no config yet, and counts a `kilo plugin` install as present", async () => {
  const fresh = await mkdtemp(path.join(os.tmpdir(), "honcho-kilo-fresh-"))
  expect((await __testing.addKiloPlugin(fresh))[0].file).toBe(path.join(fresh, "kilo.jsonc"))

  const viaKiloPlugin = await mkdtemp(path.join(os.tmpdir(), "honcho-kilo-cli-install-"))
  await writeFile(path.join(viaKiloPlugin, "opencode.json"), JSON.stringify({ plugin: ["@honcho-ai/kilo-honcho@0.1.0"] }))
  const [server] = await __testing.addKiloPlugin(viaKiloPlugin)
  expect(server.status).toBe("present")
  await expect(readFile(path.join(viaKiloPlugin, "kilo.jsonc"), "utf-8")).rejects.toThrow()

  const broken = await mkdtemp(path.join(os.tmpdir(), "honcho-kilo-broken-"))
  await writeFile(path.join(broken, "kilo.json"), "{ not json")
  expect((await __testing.addKiloPlugin(broken))[0].status).toBe("unreadable")
  expect(await readFile(path.join(broken, "kilo.json"), "utf-8")).toBe("{ not json")
})

test("kiloConfigDir follows XDG_CONFIG_HOME like Kilo does", async () => {
  await withEnv({ XDG_CONFIG_HOME: "/tmp/xdg", HOME: "/home/a" }, async () => {
    expect(__testing.kiloConfigDir()).toBe("/tmp/xdg/kilo")
  })
  await withEnv({ XDG_CONFIG_HOME: undefined, HOME: "/home/a" }, async () => {
    expect(__testing.kiloConfigDir()).toBe("/home/a/.config/kilo")
  })
})

test("the setup command runs end to end from piped answers", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "honcho-cli-"))
  const proc = Bun.spawn(["node", path.join(import.meta.dir, "..", "dist", "cli.js"), "setup"], {
    stdin: new TextEncoder().encode("2\nhttp://127.0.0.1:1\n\nalice\n"),
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, HOME: dir, XDG_CONFIG_HOME: path.join(dir, "xdg"), KILO_HONCHO_CONFIG_PATH: path.join(dir, "honcho.json"), HONCHO_API_KEY: "" },
  })
  const output = await new Response(proc.stdout).text()
  expect(await proc.exited).toBe(0)
  expect(output).toContain("Peer: alice")
  const saved = JSON.parse(await readFile(path.join(dir, "honcho.json"), "utf-8"))
  expect(saved).toMatchObject({ peerName: "alice", baseUrl: "http://127.0.0.1:1", hosts: { kilo: { workspace: "kilo" } } })
  expect(saved.apiKey).toBeUndefined()
  expect(await readFile(path.join(dir, "xdg", "kilo", "kilo.jsonc"), "utf-8")).toContain("@honcho-ai/kilo-honcho")
})

test("an agent can run the setup command with flags while stdin stays open", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "honcho-cli-agent-"))
  const proc = Bun.spawn(
    ["node", path.join(import.meta.dir, "..", "dist", "cli.js"), "setup", "--url", "http://127.0.0.1:1", "--peer-name", "eri", "--workspace=team"],
    {
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
      env: { ...process.env, HOME: dir, XDG_CONFIG_HOME: path.join(dir, "xdg"), KILO_HONCHO_CONFIG_PATH: path.join(dir, "honcho.json"), HONCHO_API_KEY: "" },
    },
  )
  const timeout = setTimeout(() => proc.kill(), 10_000)
  const exitCode = await proc.exited
  clearTimeout(timeout)
  expect(exitCode).toBe(0)
  const saved = JSON.parse(await readFile(path.join(dir, "honcho.json"), "utf-8"))
  expect(saved).toMatchObject({ peerName: "eri", baseUrl: "http://127.0.0.1:1", hosts: { kilo: { workspace: "team" } } })
})

test("the key window uses each platform's own password box, and none without a display", () => {
  expect(__testing.keyDialogCommand("darwin", {})[0]).toBe("osascript")
  expect(__testing.keyDialogCommand("darwin", {})[1].join(" ")).toContain("with hidden answer")
  expect(__testing.keyDialogCommand("win32", {})[0]).toBe("powershell.exe")
  expect(__testing.keyDialogCommand("linux", { DISPLAY: ":0" })?.[0]).toBe("zenity")
  expect(__testing.keyDialogCommand("linux", {})).toBeNull()
  // Over SSH the window would open on the far machine's screen, except with X forwarding on Linux.
  expect(__testing.keyDialogCommand("darwin", { SSH_CONNECTION: "1.2.3.4 22 5.6.7.8 22" })).toBeNull()
  expect(__testing.keyDialogCommand("win32", { SSH_CLIENT: "1.2.3.4 22 22" })).toBeNull()
  expect(__testing.keyDialogCommand("linux", { DISPLAY: "localhost:10.0", SSH_CONNECTION: "x" })?.[0]).toBe("zenity")
})

// Records which server each Honcho request went to and which key it carried.
const recordingFetch = () => {
  const requests = []
  const fetch = async (url, init = {}) => {
    const target = new URL(typeof url === "string" ? url : url.toString())
    const headers = new Headers(init.headers)
    requests.push({ origin: target.origin, auth: headers.get("authorization") })
    return new Response(JSON.stringify({ id: "kilo", metadata: {}, configuration: {} }), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  }
  fetch.requests = requests
  return fetch
}

const withSavedConfig = async (config, action) => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-guard-root-"))
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-guard-home-"))
  const configPath = path.join(homeDir, ".honcho", "config.json")
  await mkdir(path.dirname(configPath), { recursive: true })
  await writeFile(configPath, JSON.stringify(config))
  return withEnv({ HOME: homeDir, USER: "ignored-user", XDG_CONFIG_HOME: undefined }, async () => {
    const hooks = await createPluginHarness(rootDir)
    return action({ hooks, rootDir, configPath })
  })
}

test("honcho_setup never sends the saved key to a new server", async () => {
  const fetch = recordingFetch()
  await withMockFetch(fetch, () =>
    withFakeKeyWindow("key-for-new-server", () =>
      withSavedConfig({ apiKey: "saved-key" }, async ({ hooks, rootDir, configPath }) => {
        const result = JSON.parse(await hooks.tool.honcho_setup.execute({ baseUrl: "https://honcho.example.com" }, toolContext(rootDir)))
        expect(result.ok).toBe(true)
        const toNewServer = fetch.requests.filter((request) => request.origin === "https://honcho.example.com")
        expect(toNewServer.length).toBeGreaterThan(0)
        expect(toNewServer.every((request) => request.auth === "Bearer key-for-new-server")).toBe(true)
        expect(fetch.requests.some((request) => request.auth === "Bearer saved-key")).toBe(false)
        const saved = JSON.parse(await readFile(configPath, "utf-8"))
        expect(saved).toMatchObject({ apiKey: "key-for-new-server", baseUrl: "https://honcho.example.com" })
      }),
    ),
  )
})

test("honcho_setup sends nothing to a new server when the key window is closed, and refuses local and ${} values", async () => {
  const fetch = recordingFetch()
  await withMockFetch(fetch, () =>
    withFakeKeyWindow(null, () =>
      withSavedConfig({ apiKey: "saved-key" }, async ({ hooks, rootDir, configPath }) => {
        const closed = JSON.parse(await hooks.tool.honcho_setup.execute({ baseUrl: "https://honcho.example.com" }, toolContext(rootDir)))
        expect(closed.ok).toBe(false)
        const local = JSON.parse(await hooks.tool.honcho_setup.execute({ baseUrl: "http://127.0.0.1:8000" }, toolContext(rootDir)))
        expect(local.ok).toBe(false)
        expect(local.message).toContain("--url")
        const reference = JSON.parse(await hooks.tool.honcho_setup.execute({ peerName: "${AWS_SECRET_ACCESS_KEY}" }, toolContext(rootDir)))
        expect(reference.ok).toBe(false)
        expect(fetch.requests).toHaveLength(0)
        expect(JSON.parse(await readFile(configPath, "utf-8"))).toEqual({ apiKey: "saved-key" })
      }),
    ),
  )
})

test("honcho_set_config cannot set apiKey, baseUrl or a ${} reference", async () => {
  await withSavedConfig({ apiKey: "saved-key" }, async ({ hooks, rootDir, configPath }) => {
    for (const [field, value] of [["apiKey", "x"], ["baseUrl", "https://evil.example"], ["peerName", "${GITHUB_TOKEN}"]]) {
      const result = JSON.parse(await hooks.tool.honcho_set_config.execute({ field, value }, toolContext(rootDir)))
      expect(result.ok).toBe(false)
    }
    expect(JSON.parse(await readFile(configPath, "utf-8"))).toEqual({ apiKey: "saved-key" })
  })
})

test("honcho_setup saves only what it was given, not env values", async () => {
  await withMockFetch(recordingFetch(), () =>
    withSavedConfig({ apiKey: "saved-key", peerName: "eri" }, ({ hooks, rootDir, configPath }) =>
      withEnv({ HONCHO_URL: "https://env-only.example", HONCHO_PEER_NAME: "from-env" }, async () => {
        await hooks.tool.honcho_setup.execute({ workspace: "team" }, toolContext(rootDir))
        const saved = JSON.parse(await readFile(configPath, "utf-8"))
        expect(saved.peerName).toBe("eri")
        expect(saved.baseUrl).toBeUndefined()
        expect(saved.hosts.kilo.workspace).toBe("team")
      }),
    ),
  )
})

test("agent shells get the Honcho URL and workspace, not the API key", async () => {
  await withSavedConfig({ apiKey: "saved-key" }, async ({ hooks }) => {
    const output = { env: {} }
    await hooks["shell.env"]({ sessionID: "ses_shell", cwd: "/tmp" }, output)
    expect(output.env.HONCHO_WORKSPACE_ID).toBe("kilo")
    expect(output.env.HONCHO_API_KEY).toBeUndefined()
    expect(JSON.stringify(output.env)).not.toContain("saved-key")
  })
})

test("the key window names a non-cloud server by its origin only", () => {
  expect(__testing.keyWindowMessage()).toContain("app.honcho.dev")
  const message = __testing.keyWindowMessage('https://evil.example/"; do shell script "x')
  expect(message).toContain("https://evil.example")
  expect(message).not.toContain("do shell script")
  const [, args] = __testing.keyDialogCommand("darwin", {}, 'say "hi" \\ there')
  expect(args[1]).toContain('display dialog "say \\"hi\\" \\\\ there"')
})

test("a server that refuses requests without a key counts as not set up, so the agent offers setup", async () => {
  const unauthorized = async () =>
    new Response(JSON.stringify({ detail: "Missing API key" }), { status: 401, headers: { "content-type": "application/json" } })
  // CI suppresses the setup offer, and GitHub Actions sets it.
  await withMockFetch(unauthorized, () => withEnv({ CI: undefined }, () =>
    withSavedConfig({ baseUrl: "https://honcho.example.com" }, async ({ hooks, rootDir }) => {
      const before = JSON.parse(await hooks.tool.honcho_status.execute({}, toolContext(rootDir)))
      // Until a request is refused, a non-cloud URL may be a server without auth.
      expect(before.configured).toBe(true)
      await hooks["chat.message"](
        { sessionID: "ses_needs_key" },
        { message: { id: "msg-k", role: "user", time: { created: Date.now() } }, parts: [{ type: "text", text: "what am I building?" }] },
      )
      const system = { system: [] }
      await hooks["experimental.chat.system.transform"]({ sessionID: "ses_needs_key", model: {} }, system)
      expect(system.system.join("\n")).toContain("call honcho_setup")
      const after = JSON.parse(await hooks.tool.honcho_status.execute({}, toolContext(rootDir)))
      expect(after.configured).toBe(false)
      expect(after.nextStep).toContain("honcho_setup")
    }),
  ))
})
