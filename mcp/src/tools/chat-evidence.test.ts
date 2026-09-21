import { afterEach, beforeEach, expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { honchoClients } from "../config.js";
import { register as registerPeers } from "./peers.js";
import { register as registerWorkspace } from "./workspace.js";

const config = {
  apiKey: "test-key",
  baseUrl: "https://honcho.test",
  workspaceId: "test-workspace",
};

const EVIDENCE = {
  conclusions: [
    {
      id: "c1",
      level: "explicit",
      content: "alice prefers small PRs",
      created_at: "2026-09-15T00:00:00Z",
      session_id: "s1",
      source_ids: [],
    },
  ],
  messages: [
    {
      id: "m1",
      session_id: "s1",
      peer_id: "alice",
      created_at: "2026-09-15T00:00:00Z",
    },
  ],
  tool_calls: [{ tool_name: "search_memory", tool_input: { query: "PRs" } }],
  reasoning_trace_id: null,
};

const requests: { path: string; body: Record<string, unknown> }[] = [];
const originalFetch = globalThis.fetch;
let client: Client;
let server: McpServer;

beforeEach(async () => {
  requests.length = 0;
  globalThis.fetch = async (input, init) => {
    const path = new URL(String(input)).pathname;
    const body = init?.body ? JSON.parse(String(init.body)) : {};
    requests.push({ path, body });
    if (path.endsWith("/chat")) {
      return Response.json({
        content: "She keeps them small.",
        evidence: body.include_evidence ? EVIDENCE : null,
      });
    }
    return Response.json({});
  };
  server = new McpServer({ name: "test", version: "1.0.0" });
  const ctx = { config, ...honchoClients(config) };
  registerPeers(server, ctx);
  registerWorkspace(server, ctx);
  client = new Client({ name: "test", version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  await client.connect(clientTransport);
});

afterEach(async () => {
  globalThis.fetch = originalFetch;
  await client.close();
  await server.close();
});

const text = (result: unknown) =>
  (result as { content: { text: string }[] }).content[0].text;

test.each([
  ["chat", { peer_id: "alice", query: "How does she review?" }],
  ["workspace_chat", { query: "How does the team review?" }],
])("%s returns evidence only when asked for it", async (name, args) => {
  const plain = await client.callTool({ name, arguments: args });
  expect(text(plain)).toBe("She keeps them small.");
  expect(requests.at(-1)?.body.include_evidence).toBeUndefined();

  const withEvidence = await client.callTool({
    name,
    arguments: { ...args, include_evidence: true },
  });
  expect(requests.at(-1)?.body.include_evidence).toBe(true);
  expect(JSON.parse(text(withEvidence))).toEqual({
    content: "She keeps them small.",
    evidence: EVIDENCE,
  });
});
