import { expect, test } from "bun:test"
import os from "node:os"
import path from "node:path"
import { mkdtemp, stat, utimes } from "node:fs/promises"

import { __testing } from "../dist/index.js"

const { summarizeToolExecution, redactShellCommand, ensureHonchoSkillInstalled } = __testing

test("summarizeToolExecution keeps significant tool calls and drops trivial ones", () => {
  expect(summarizeToolExecution("bash", { command: "npm test" })).toBe("Ran: npm test")
  expect(summarizeToolExecution("bash", { command: "git status" })).toBeNull()
  expect(summarizeToolExecution("bash", { command: "git status && npm test" })).toBe("Ran: npm test")
  expect(summarizeToolExecution("bash", { command: "ls | npm test" })).toBe("Ran: npm test")
  expect(summarizeToolExecution("bash", { command: "echo \"a | b\" && npm test" })).toBe("Ran: npm test")
  expect(summarizeToolExecution("edit", { path: "src/index.ts" })).toBe("Edited: src/index.ts")
  expect(summarizeToolExecution("write", { path: "README.md" })).toBe("Created: README.md")
  expect(summarizeToolExecution("task", { description: "Refactor migrations" })).toBe("Task: Refactor migrations")
  expect(summarizeToolExecution("read", { path: "src/index.ts" })).toBeNull()
  expect(summarizeToolExecution("honcho_search", { query: "test" })).toBeNull()
})

test("redactShellCommand keeps the executable and drops credential-bearing arguments", () => {
  expect(redactShellCommand("npm test")).toBe("npm test")
  expect(redactShellCommand("DEBUG=1 npm run build")).toBe("DEBUG=1 npm run build")
  expect(redactShellCommand("curl -H 'Authorization: Bearer abc123' https://api.example.com")).toBe("curl (arguments redacted)")
  expect(redactShellCommand("curl -u user:secret https://api.example.com")).toBe("curl (arguments redacted)")
  expect(redactShellCommand("curl https://user:pass@example.com")).toBe("curl (arguments redacted)")
  expect(redactShellCommand("API_KEY=abc npm test")).toBe("npm (arguments redacted)")
  expect(redactShellCommand("DATABASE_URL='postgres://user:pw@db' npm test")).toBe("npm (arguments redacted)")
  expect(redactShellCommand("export API_KEY=abc123")).toBe("export (arguments redacted)")
})

test("ensureHonchoSkillInstalled honors KILO_CONFIG_DIR and skips identical writes", async () => {
  const configDir = await mkdtemp(path.join(os.tmpdir(), "honcho-kilo-config-"))
  const previous = process.env.KILO_CONFIG_DIR
  process.env.KILO_CONFIG_DIR = configDir
  try {
    const installedPath = await ensureHonchoSkillInstalled()
    expect(installedPath).toBe(path.join(configDir, "skills", "honcho-memory", "SKILL.md"))

    const old = new Date("2000-01-01T00:00:00.000Z")
    await utimes(installedPath, old, old)
    await ensureHonchoSkillInstalled()
    expect((await stat(installedPath)).mtime.getTime()).toBe(old.getTime())
  } finally {
    if (previous === undefined) delete process.env.KILO_CONFIG_DIR
    else process.env.KILO_CONFIG_DIR = previous
  }
})
