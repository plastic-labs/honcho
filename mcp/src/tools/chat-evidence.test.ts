import { afterEach, beforeEach, expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { Honcho } from "@honcho-ai/sdk";
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

test("workspace_chat asks the workspace route with every recall option", async () => {
  const result = await client.callTool({
    name: "workspace_chat",
    arguments: {
      workspace_id: "pi",
      query: "What are the main themes across peers?",
      session_id: "s1",
      scope: ["therapy", "work"],
      reasoning_level: "minimal",
    },
  });
  expect(result.isError).not.toBe(true);
  expect(text(result)).toBe("She keeps them small.");
  expect(requests.at(-1)).toEqual({
    path: "/v3/workspaces/pi/chat",
    body: {
      query: "What are the main themes across peers?",
      stream: false,
      session_id: "s1",
      reasoning_level: "minimal",
      scope: ["therapy", "work"],
    },
  });
});

test("chat reports an SDK that ignores include_evidence instead of answering None", async () => {
  // SDKs before 2.5.0 drop includeEvidence and resolve to the bare answer.
  const staleSdk = {
    peer: async () => ({ chat: async () => "She keeps them small." }),
  } as unknown as Honcho;
  const staleServer = new McpServer({ name: "test", version: "1.0.0" });
  registerPeers(staleServer, {
    config,
    clientFor: () => staleSdk,
    unscoped: staleSdk,
  });
  const staleClient = new Client({ name: "test", version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await staleServer.connect(serverTransport);
  await staleClient.connect(clientTransport);

  const result = await staleClient.callTool({
    name: "chat",
    arguments: { peer_id: "alice", query: "How does she review?", include_evidence: true },
  });
  expect((result as { isError?: boolean }).isError).toBe(true);
  expect(text(result)).toContain("does not support include_evidence");

  await staleClient.close();
  await staleServer.close();
});
