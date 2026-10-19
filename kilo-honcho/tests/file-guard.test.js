import { expect, test } from "bun:test"
import os from "node:os"
import path from "node:path"
import { mkdtemp } from "node:fs/promises"

import { __testing, createHonchoRuntimePlugin, HONCHO_COMMAND } from "../dist/index.js"

const { touchesHonchoFiles } = __testing
const home = os.homedir()
const defaultConfig = path.join(home, ".honcho", "config.json")
const project = path.join(home, "code", "app")

const createPluginHarness = async (rootDir, configPath) =>
  createHonchoRuntimePlugin({ configPath })({
    client: { app: { log: async () => undefined } },
    project: { id: "kilo", worktree: rootDir },
    directory: rootDir,
    worktree: rootDir,
    serverUrl: new URL("http://127.0.0.1:4096"),
    $: {},
  })

test("the guard catches the ways an agent names ~/.honcho", () => {
  const blocked = [
    { command: "cat ~/.honcho/config.json" },
    { command: "cat $HOME/.HONCHO/config.json | jq .apiKey" },
    { command: "ls -la ~/.honcho*" },
    { command: "cd ~ && cat .honcho/config.json" },
    { command: "cat config.json", workdir: path.join(home, ".honcho") },
    { filePath: defaultConfig },
    { filePath: "../../.honcho/config.json" },
    { path: "~/.honcho" },
    { pattern: "**/.honcho/**" },
    { patchText: "*** Begin Patch\n*** Update File: ~/.honcho/config.json\n@@\n-a\n+b\n*** End Patch" },
  ]
  for (const args of blocked) expect([args, touchesHonchoFiles(args, defaultConfig, project)]).toEqual([args, true])
})

test("the guard leaves ordinary project work alone", () => {
  const allowed = [
    { command: "bun test && grep -rn honcho src" },
    { command: "npx @honcho-ai/kilo-honcho --help" },
    { command: "cat kilo-honcho/README.md" },
    { filePath: path.join(project, "src", ".honchorc") },
    { filePath: path.join(project, "docs", "setup.md"), content: "Kilo saves the key to ~/.honcho/config.json." },
    { pattern: "honcho", path: project },
    { patchText: "*** Begin Patch\n*** Update File: docs/setup.md\n@@\n-old\n+See ~/.honcho/config.json\n*** End Patch" },
    "not an object",
  ]
  for (const args of allowed) expect([args, touchesHonchoFiles(args, defaultConfig, project)]).toEqual([args, false])
})

test("with KILO_HONCHO_CONFIG_PATH the guard covers that file and its activity folder", () => {
  const configPath = path.join(home, "test", "honcho-config.json")
  const activityFile = path.join(home, "test", "kilo", "sessions", "ses_1.json")
  expect(touchesHonchoFiles({ filePath: configPath }, configPath, project)).toBe(true)
  expect(touchesHonchoFiles({ filePath: "honcho-config.json" }, configPath, path.join(home, "test"))).toBe(true)
  expect(touchesHonchoFiles({ command: "cat ~/test/honcho-config.json" }, configPath, project)).toBe(true)
  expect(touchesHonchoFiles({ filePath: activityFile }, configPath, project)).toBe(true)
  expect(touchesHonchoFiles({ filePath: path.join(home, "test", "notes.md") }, configPath, project)).toBe(false)
})

test("tool.execute.before refuses a tool call that reads the Honcho config and points the agent to honcho_status", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-guard-root-"))
  const hooks = await createPluginHarness(rootDir)
  const input = { tool: "read", sessionID: "ses_guard", callID: "call_1" }
  await expect(hooks["tool.execute.before"](input, { args: { filePath: defaultConfig } })).rejects.toThrow("honcho_status")
  await expect(hooks["tool.execute.before"](input, { args: { filePath: path.join(rootDir, "a.ts") } })).resolves.toBeUndefined()
  await expect(
    hooks["tool.execute.before"]({ ...input, tool: "honcho_status" }, { args: { path: "~/.honcho" } }),
  ).resolves.toBeUndefined()
})

test("the config hook adds /honcho for the desktop app and IDE extensions without replacing a user's own", async () => {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "honcho-command-root-"))
  const hooks = await createPluginHarness(rootDir)
  const config = {}
  await hooks.config(config)
  expect(config.command.honcho).toEqual(HONCHO_COMMAND)
  expect(config.command.honcho.template).toContain("honcho_status")
  expect(config.command.honcho.template).toContain("honcho_setup")
  expect(config.command.honcho.template).toContain("$ARGUMENTS")
  expect(config.command.honcho.template).not.toContain("npx")

  const custom = { command: { honcho: { template: "my own" }, other: { template: "x" } } }
  await hooks.config(custom)
  expect(custom.command.honcho.template).toBe("my own")
  expect(custom.command.other.template).toBe("x")
})
