import test from "node:test"
import assert from "node:assert/strict"
import os from "node:os"
import path from "node:path"
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises"

import tuiModule, { __testing } from "../dist/tui.js"

test("native TUI Honcho commands register slash aliases including import", () => {
  const commands = __testing.buildCommands({})
  assert.deepEqual(commands.map((command) => command.value), [
    "honcho.setup",
    "honcho.status",
    "honcho.settings",
    "honcho.config",
    "honcho.import",
  ])
  assert.deepEqual(
    commands.map((command) => command.slash?.name),
    ["honcho:setup", "honcho:status", "honcho:settings", "honcho:config", "honcho:import"],
  )
  assert.equal(tuiModule.id, "@honcho-ai/kilo-honcho")
})

test("tui saveSettings persists only supported root and host fields", async () => {
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-tui-clean-fields-"))
  const sharedConfigDir = path.join(homeDir, ".honcho")
  const configPath = path.join(sharedConfigDir, "config.json")
  const previousHome = process.env.HOME
  const previousUserProfile = process.env.USERPROFILE

  await mkdir(sharedConfigDir, { recursive: true })
  await writeFile(configPath, JSON.stringify({}, null, 2))
  process.env.HOME = homeDir
  process.env.USERPROFILE = homeDir

  try {
    await __testing.saveSettings({
      apiKey: "key",
      baseUrl: "https://api.honcho.dev",
      hosts: {
        kilo: {
          workspace: "kilo",
          aiPeer: "kilo",
          recallMode: "hybrid",
          sessionStrategy: "per-directory",
          writeFrequency: "session",
          peerModel: "hierarchical",
          baseUrl: "http://127.0.0.1:8000",
        },
      },
    })

    const persisted = JSON.parse(await readFile(configPath, "utf-8"))
    assert.equal(persisted.apiKey, "key")
    assert.equal(persisted.workspace, undefined)
    assert.equal(persisted.hosts.kilo.workspace, "kilo")
    assert.equal(persisted.hosts.kilo.writeFrequency, undefined)
    assert.equal(persisted.hosts.kilo.baseUrl, undefined)
  } finally {
    if (previousHome === undefined) delete process.env.HOME
    else process.env.HOME = previousHome
    if (previousUserProfile === undefined) delete process.env.USERPROFILE
    else process.env.USERPROFILE = previousUserProfile
  }
})

test("tui honors KILO_HONCHO_CONFIG_PATH override for reads, writes, and display", async () => {
  const homeDir = await mkdtemp(path.join(os.tmpdir(), "honcho-tui-override-"))
  const globalConfigPath = path.join(homeDir, ".honcho", "config.json")
  const overrideDir = path.join(homeDir, "override")
  const overrideConfigPath = path.join(overrideDir, "custom.json")
  const previousHome = process.env.HOME
  const previousUserProfile = process.env.USERPROFILE
  const previousOverride = process.env.KILO_HONCHO_CONFIG_PATH

  await mkdir(overrideDir, { recursive: true })
  await writeFile(overrideConfigPath, JSON.stringify({ peerName: "override-peer" }, null, 2))
  process.env.HOME = homeDir
  process.env.USERPROFILE = homeDir
  process.env.KILO_HONCHO_CONFIG_PATH = overrideConfigPath

  try {
    assert.equal(__testing.sharedConfigPath(), overrideConfigPath)

    const readBack = await __testing.readSharedConfig()
    assert.equal(readBack.peerName, "override-peer")

    await __testing.saveSettings({
      apiKey: "key",
      baseUrl: "https://api.honcho.dev",
      hosts: { kilo: { workspace: "kilo" } },
    })
    const persisted = JSON.parse(await readFile(overrideConfigPath, "utf-8"))
    assert.equal(persisted.apiKey, "key")
    assert.equal(persisted.peerName, "override-peer")
    assert.rejects(readFile(globalConfigPath, "utf-8"))

    assert.match(__testing.settingsMessage({}), new RegExp(`Config path: ${overrideConfigPath.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`))
  } finally {
    if (previousHome === undefined) delete process.env.HOME
    else process.env.HOME = previousHome
    if (previousUserProfile === undefined) delete process.env.USERPROFILE
    else process.env.USERPROFILE = previousUserProfile
    if (previousOverride === undefined) delete process.env.KILO_HONCHO_CONFIG_PATH
    else process.env.KILO_HONCHO_CONFIG_PATH = previousOverride
  }
})

test("tui shows endpoint.baseUrl when there is no top-level baseUrl", () => {
  const url = (settings) => __testing.settingsMessage(settings).match(/Base URL: (.*)/)[1]
  const endpoint = { baseUrl: "http://selfhosted:8000" }
  assert.equal(url({ endpoint }), "http://selfhosted:8000")
  assert.equal(url({ endpoint: { baseUrl: "http://top:1" }, hosts: { kilo: { endpoint } } }), "http://selfhosted:8000")
  assert.equal(url({ baseUrl: "http://explicit:9", endpoint }), "http://explicit:9")
  assert.equal(url({}), "https://api.honcho.dev")
})

const withHonchoEnv = async (entries, action) => {
  const keys = ["HONCHO_API_KEY", "HONCHO_URL", "HONCHO_BASE_URL", "HONCHO_PEER_NAME", "HONCHO_WORKSPACE", "HONCHO_WORKSPACE_ID"]
  const previous = Object.fromEntries(keys.map((key) => [key, process.env[key]]))
  for (const key of keys) delete process.env[key]
  Object.assign(process.env, entries)
  try {
    return await action()
  } finally {
    for (const key of keys) {
      if (previous[key] === undefined) delete process.env[key]
      else process.env[key] = previous[key]
    }
  }
}

test("tui status reports HONCHO_* env values the server will use", async () => {
  await withHonchoEnv({ HONCHO_API_KEY: "env-key", HONCHO_URL: "https://api.staging.example" }, () => {
    const message = __testing.statusMessage({ peerName: "eri" })
    assert.match(message, /Configured: yes/)
    assert.match(message, /API key: set/)
    assert.match(message, /Base URL: https:\/\/api\.staging\.example/)
    assert.doesNotMatch(message, /env-key/)
  })
})

test("tui live status falls back to the kilo workspace, not the folder name", async () => {
  await withHonchoEnv({}, () => {
    const api = { route: { current: { name: "home" } }, state: { path: { worktree: "/tmp/demo", directory: "/tmp/demo" } } }
    assert.equal(__testing.deriveLiveStatus(api, {}).workspaceName, "kilo")
    assert.equal(__testing.deriveLiveStatus(api, { hosts: { kilo: { workspace: "team" } } }).workspaceName, "team")
  })
  await withHonchoEnv({ HONCHO_WORKSPACE: "from-env" }, () => {
    const api = { route: { current: { name: "home" } }, state: { path: { worktree: "/tmp/demo" } } }
    assert.equal(__testing.deriveLiveStatus(api, { hosts: { kilo: { workspace: "team" } } }).workspaceName, "from-env")
  })
})
