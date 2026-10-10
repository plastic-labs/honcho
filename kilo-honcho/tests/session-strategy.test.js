import { expect, test } from "bun:test"

import { __testing } from "../dist/index.js"

test("per-directory sessions use a relative path so same-basename dirs do not collide", async () => {
  const sessionA = await __testing.deriveSessionScope({
    workspaceId: "kilo",
    sessionStrategy: "per-directory",
    rootDir: "/tmp/project",
    repoName: "project",
    currentDirectory: "/tmp/project/services/api",
    sessionId: "session-a",
  })
  const sessionB = await __testing.deriveSessionScope({
    workspaceId: "kilo",
    sessionStrategy: "per-directory",
    rootDir: "/tmp/project",
    repoName: "project",
    currentDirectory: "/tmp/project/packages/api",
    sessionId: "session-b",
  })

  expect(sessionA).toBe("kilo:services-api")
  expect(sessionB).toBe("kilo:packages-api")
  expect(__testing.defaultSettings.sessionStrategy).toBe("per-directory")
})

test("teammates in one workspace and directory get separate sessions", () => {
  const alice = __testing.honchoSessionKey("alice", "per-directory", "repo", ["kilo"])
  const bob = __testing.honchoSessionKey("bob", "per-directory", "repo", ["kilo"])
  expect(alice).not.toBe(bob)
  expect(alice).toBe("alice-per-directory-repo-kilo")
})

test("normalizeId keeps its output and stays fast on long dash runs", () => {
  expect(__testing.normalizeId("--Hello World--")).toBe("hello-world")
  expect(__testing.normalizeId("a//b")).toBe("a-b")
  expect(__testing.normalizeId("---")).toBe("default")
  const started = performance.now()
  __testing.normalizeId(`x${"-".repeat(100_000)}x`)
  __testing.normalizeId(" -".repeat(50_000))
  expect(performance.now() - started).toBeLessThan(200)
})

test("shell redaction stays fast on a token of many colons and still hides URL passwords", () => {
  const started = performance.now()
  __testing.redactShellCommand(`curl http://${":".repeat(100_000)}`)
  expect(performance.now() - started).toBeLessThan(200)
  expect(__testing.redactShellCommand("git clone https://user:hunter2@github.com/a/b")).not.toContain("hunter2")
})
