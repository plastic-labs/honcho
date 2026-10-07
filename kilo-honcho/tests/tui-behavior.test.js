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
    "honcho.recall",
    "honcho.settings",
    "honcho.config",
    "honcho.import",
  ])
  assert.deepEqual(
    commands.map((command) => command.slash?.name),
    ["honcho:setup", "honcho:status", "honcho:recall", "honcho:settings", "honcho:config", "honcho:import"],
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

const activity = (extra = {}) => ({
  kiloSessionId: "ses_view",
  updatedAt: new Date(0).toISOString(),
  state: "active",
  workspace: "kilo",
  userPeer: "eri",
  saved: 0,
  ...extra,
})

test("sidebar rows show the peer and a session link, or the session name off Honcho Cloud", () => {
  assert.deepEqual(__testing.sidebarRows(activity({ state: "unconfigured" })), {
    label: "Not set up",
    tone: "muted",
    details: [{ text: "Run /honcho:setup" }],
  })
  assert.equal(__testing.sidebarRows(activity({ state: "error", error: "401 Unauthorized" })).details[0].text, "401 Unauthorized")

  const cloud = __testing.sidebarRows(activity({ session: "eri-per-directory-kilo-other-kilo", sessionUrl: "https://app.honcho.dev/explore?x" }))
  assert.deepEqual(cloud.details, [
    { text: "Peer: eri" },
    { text: "Session: View in Honcho ↗", url: "https://app.honcho.dev/explore?x" },
  ])

  const selfHosted = __testing.sidebarRows(activity({ session: "eri-per-directory-kilo-some-long-folder-name-kilo" }))
  assert.equal(selfHosted.details[1].text, "Session: eri-per-direct…der-name-kilo")
})

test("/honcho:recall shows the exact text Honcho added to the current session", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "honcho-tui-recall-"))
  const configPath = path.join(dir, "config.json")
  await mkdir(path.join(dir, "kilo", "sessions"), { recursive: true })
  await writeFile(
    path.join(dir, "kilo", "sessions", "ses_view.json"),
    JSON.stringify(
      activity({
        recall: { at: new Date(0).toISOString(), messageId: "msg", conclusions: 1, text: "[2026-10-07 19:32:34] user runs cargo nextest" },
        profile: { at: new Date(0).toISOString(), conclusions: 0, text: "## User Memory Profile\n- Prefers tokio" },
      }),
    ),
  )
  const previous = process.env.KILO_HONCHO_CONFIG_PATH
  process.env.KILO_HONCHO_CONFIG_PATH = configPath
  const shown = []
  const session = (kiloSessionId) => ({
    dialogs: { view: async (input) => shown.push(input), alert: async (input) => shown.push(input) },
    liveStatus: () => ({ kiloSessionId }),
  })
  try {
    await __testing.runRecall(session("ses_view"))
    await __testing.runRecall(session(undefined))
    await __testing.runRecall(session("ses_missing"))
  } finally {
    if (previous === undefined) delete process.env.KILO_HONCHO_CONFIG_PATH
    else process.env.KILO_HONCHO_CONFIG_PATH = previous
  }
  assert.match(shown[0].body, /Attached to your prompt at .* \(1 conclusion\):\n\n\[2026-10-07 19:32:34\] user runs cargo nextest/)
  assert.match(shown[0].body, /Added to the system prompt at .*\n\n## User Memory Profile\n- Prefers tokio/)
  assert.match(shown[1].message, /Open a Kilo session/)
  assert.match(shown[2].body, /has not added memory to this session yet/)
})

test("/honcho:setup prefills the shared peer name and can join another tool's workspace", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "honcho-tui-setup-"))
  const configPath = path.join(dir, "config.json")
  await writeFile(configPath, JSON.stringify({ peerName: "eri", hosts: { claude_code: { workspace: "claude-code" } } }))
  const previous = process.env.KILO_HONCHO_CONFIG_PATH
  process.env.KILO_HONCHO_CONFIG_PATH = configPath
  const asked = []
  const session = {
    dialogs: {
      select: async (input) => {
        asked.push(input)
        return input.title === "Configure Honcho" ? "cloud" : "claude-code"
      },
      prompt: async (input) => {
        asked.push(input)
        return input.title === "Honcho API key" ? "hch-test" : input.value
      },
      alert: async (input) => asked.push(input),
    },
    liveStatus: () => ({}),
  }
  try {
    await withHonchoEnv({}, () => __testing.runSetup(session))
  } finally {
    if (previous === undefined) delete process.env.KILO_HONCHO_CONFIG_PATH
    else process.env.KILO_HONCHO_CONFIG_PATH = previous
  }
  const namePrompt = asked.find((input) => input.title?.startsWith("Peer name"))
  assert.equal(namePrompt.value, "eri")
  assert.match(namePrompt.title, /other Honcho tools/)
  const workspaceSelect = asked.find((input) => input.title === "Which Honcho workspace should Kilo use?")
  assert.deepEqual(workspaceSelect.options.map((option) => option.value), ["kilo", "claude-code"])

  const saved = JSON.parse(await readFile(configPath, "utf-8"))
  assert.equal(saved.peerName, "eri")
  assert.equal(saved.hosts.kilo.workspace, "claude-code")
  assert.deepEqual(saved.hosts.claude_code, { workspace: "claude-code" })
})
