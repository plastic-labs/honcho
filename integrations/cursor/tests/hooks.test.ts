/**
 * Black-box tests for the built hooks (dist/hook.js), run the way Cursor runs
 * them: a subprocess with JSON on stdin and a scrubbed environment, against an
 * in-process mock of the Honcho API. `bun run test` builds first.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const HOOK = join(import.meta.dir, "..", "dist", "hook.js");
const now = new Date().toISOString();

interface Seen { method: string; path: string; body: unknown; headers: Record<string, string> }
const seen: Seen[] = [];
let server: ReturnType<typeof Bun.serve>;
let home: string;

beforeAll(() => {
  home = mkdtempSync(join(tmpdir(), "cursor-honcho-test-"));
  server = Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const text = req.method === "GET" ? "" : await req.text();
      const headers: Record<string, string> = {};
      req.headers.forEach((v, k) => (headers[k] = v));
      seen.push({ method: req.method, path: url.pathname, body: text ? JSON.parse(text) : null, headers });
      const p = url.pathname;
      const m = p.match(/^\/v3\/workspaces\/([^/]+)\/(sessions|peers)\/([^/]+)\/([^/?]+)/);
      let out: unknown = {};
      if (p === "/v3/workspaces") out = { id: "cursor", metadata: {}, configuration: {}, created_at: now };
      else if (p.endsWith("/sessions")) out = { id: (JSON.parse(text) as { id: string }).id, workspace_id: "cursor", is_active: true, metadata: {}, configuration: {}, created_at: now };
      else if (p.endsWith("/peers") && !m) out = { id: (JSON.parse(text) as { id: string }).id, workspace_id: "cursor", metadata: {}, configuration: {}, created_at: now };
      else if (m?.[4] === "messages") out = [];
      else if (m?.[4] === "context") out = { peer_id: m[3], target_id: m[3], representation: "MOCK REPRESENTATION", peer_card: ["MOCK CARD ITEM"] };
      else if (m?.[4] === "summaries") out = { id: m[3], short_summary: null, long_summary: { content: "MOCK SUMMARY", message_id: 1, summary_type: "long", created_at: now, token_count: 2 } };
      return Response.json(out);
    },
  });
});

afterAll(() => {
  server.stop(true);
  rmSync(home, { recursive: true, force: true });
});

async function runHook(event: string, input: Record<string, unknown>, creds: Record<string, string>): Promise<string> {
  const proc = Bun.spawn(["bun", "run", HOOK, event], {
    stdin: Buffer.from(JSON.stringify(input)),
    stdout: "pipe",
    stderr: "pipe",
    // Deliberately not process.env: a developer's real HONCHO_API_KEY must never reach these runs.
    env: { PATH: process.env.PATH ?? "", HOME: home, HONCHO_CONFIG_PATH: join(home, "missing.json"), ...creds },
  });
  const [stdout, code] = await Promise.all([new Response(proc.stdout).text(), proc.exited]);
  expect(code).toBe(0);
  return stdout;
}

const input = { session_id: "s1", conversation_id: "c1", workspace_roots: ["/tmp/My Project"], cursor_version: "3.14.0", model: "test-model" };

describe("hooks", () => {
  test("without a credential every hook is silent and never calls the API", async () => {
    seen.length = 0;
    expect(await runHook("session-start", input, {})).toBe("");
    expect(JSON.parse(await runHook("before-submit-prompt", { ...input, prompt: "hello" }, {}))).toEqual({ continue: true });
    expect(await runHook("after-agent-response", { ...input, text: "a reply long enough to record" }, {})).toBe("");
    expect(seen).toHaveLength(0);
  });

  test("with a credential: injects memory at session start, then records both sides of a turn", async () => {
    seen.length = 0;
    const creds = { HONCHO_API_KEY: "test-key", HONCHO_BASE_URL: `http://127.0.0.1:${server.port}`, HONCHO_PEER_NAME: "tester" };

    // A project with a git remote is named from the remote, not the folder.
    const repo = join(home, "My Project");
    mkdirSync(repo, { recursive: true });
    Bun.spawnSync(["git", "init", "-q"], { cwd: repo });
    Bun.spawnSync(["git", "remote", "add", "origin", "https://github.com/acme/Widgets.git"], { cwd: repo });
    const withRemote = { ...input, workspace_roots: [repo] };
    const named = JSON.parse(await runHook("session-start", withRemote, creds));
    expect(named.user_message).toContain("session cursor-acme-widgets");
    seen.length = 0;

    const start = JSON.parse(await runHook("session-start", input, creds));
    expect(start.additional_context).toContain("MOCK CARD ITEM");
    expect(start.additional_context).toContain("MOCK REPRESENTATION");
    expect(start.additional_context).toContain("MOCK SUMMARY");
    expect(start.additional_context).toContain("Automatic capture is on");
    expect(start.user_message).toContain("session cursor-my-project");
    const peers = seen.find((r) => r.path === "/v3/workspaces/cursor/sessions/cursor-my-project/peers");
    expect(peers?.body).toEqual({
      tester: { observe_me: true, observe_others: false },
      cursor: { observe_me: false, observe_others: false },
    });

    const prompt = "deploy with HONCHO_API_KEY=hch_abcdefghijklmnopqrstuvwxyz please";
    expect(JSON.parse(await runHook("before-submit-prompt", { ...input, prompt }, creds))).toEqual({ continue: true });
    const code = "```ts\n" + Array.from({ length: 14 }, (_, i) => String(i)).join("\n") + "\n```";
    expect(await runHook("after-agent-response", { ...input, text: `Done. Here is the file:\n${code}\nShip it.` }, creds)).toBe("");

    const posts = seen.filter((r) => r.path.endsWith("/messages")).map((r) => (r.body as { messages: Array<Record<string, unknown>> }).messages[0]);
    expect(posts).toHaveLength(2);
    expect(posts[0]).toMatchObject({ peer_id: "tester", content: "deploy with HONCHO_API_KEY=[redacted] please" });
    expect(posts[0].metadata).toMatchObject({ source: "cursor-honcho", hook: "beforeSubmitPrompt", conversation_id: "c1" });
    expect(posts[1]).toMatchObject({ peer_id: "cursor", content: "Done. Here is the file:\n[code block omitted, 15 lines]\nShip it." });

    const last = seen.at(-1)!;
    expect(last.headers.authorization).toBe("Bearer test-key");
    expect(last.headers["x-honcho-plugin"]).toBe("cursor-honcho/0.2.0");
    expect(last.headers["x-honcho-host"]).toMatch(/^cursor\/3\.14\.0/);
  });
});
