import { afterEach, beforeEach, expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { honchoClients } from "../config.js";
import { register as registerPeers } from "./peers.js";
import { register as registerSessions } from "./sessions.js";
import { register as registerWorkspace } from "./workspace.js";

const config = {
  apiKey: "test-key",
  baseUrl: "https://honcho.test",
  workspaceId: "test-workspace",
};
const requests: { method: string; path: string; body: unknown; query: Record<string, string> }[] = [];
const originalFetch = globalThis.fetch;
let client: Client;
let server: McpServer;

// Minimal successful bodies keyed by path suffix; get-or-create routes echo an id.
function respond(path: string, body: unknown): Response {
  if (path.endsWith("/context")) {
    return Response.json({
      id: "s1",
      messages: [],
      summary: null,
      peer_representation: "rep from scope",
      peer_card: ["card line"],
    });
  }
  if (path.endsWith("/representation")) return Response.json({ representation: "rep" });
  if (path.endsWith("/search")) return Response.json([]);
  const id = (body as { id?: string } | null)?.id ?? "x";
  return Response.json({ id, metadata: {}, configuration: {}, created_at: "2026-09-14T00:00:00Z" });
}

beforeEach(async () => {
  requests.length = 0;
  globalThis.fetch = async (input, init) => {
    const url = new URL(String(input));
    const path = url.pathname;
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    requests.push({
      method: init?.method ?? "GET",
      path,
      body,
      query: Object.fromEntries(url.searchParams),
    });
    return respond(path, body);
  };
  server = new McpServer({ name: "test", version: "1.0.0" });
  const ctx = { config, ...honchoClients(config) };
  registerPeers(server, ctx);
  registerSessions(server, ctx);
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

const bodyFor = (suffix: string) =>
  requests.find((r) => r.path.endsWith(suffix))?.body as Record<string, unknown> | undefined;

test("scope read options reach the wire on every read tool", async () => {
  const created = await client.callTool({
    name: "create_session",
    arguments: { session_id: "s1", scopes: ["therapy", "work"] },
  });
  expect(created.isError).not.toBe(true);
  expect(bodyFor("/sessions")?.scopes).toEqual(["therapy", "work"]);

  // `scope` and `sessions` are mutually exclusive in the SDK, so exercise each alone.
  const scopedRep = await client.callTool({
    name: "get_representation",
    arguments: { peer_id: "alice", scope: ["therapy", "work"] },
  });
  expect(scopedRep.isError).not.toBe(true);
  expect(bodyFor("/representation")?.scope).toEqual(["therapy", "work"]);
  requests.length = 0;
  const allowlistedRep = await client.callTool({
    name: "get_representation",
    arguments: { peer_id: "alice", sessions: ["s1"] },
  });
  expect(allowlistedRep.isError).not.toBe(true);
  // The SDK encodes the allowlist as a `session_id` filter on the wire.
  expect(bodyFor("/representation")?.filters).toEqual({ session_id: ["s1"] });

  const context = await client.callTool({
    name: "get_session_context",
    arguments: { session_id: "s1", peer_target: "alice", scope: "therapy", limit_to_session: true },
  });
  expect(context.isError).not.toBe(true);
  // Context is a GET; its options travel as query params.
  expect(requests.find((r) => r.path.endsWith("/context"))?.query).toMatchObject({
    peer_target: "alice",
    scope: "therapy",
    limit_to_session: "true",
  });
  expect(JSON.parse((context.content as { text: string }[])[0].text)).toMatchObject({
    peer_representation: "rep from scope",
    peer_card: ["card line"],
  });

  const search = await client.callTool({
    name: "search",
    arguments: { query: "q", scope: "therapy", peer_id: "alice" },
  });
  expect(search.isError).not.toBe(true);
  // A peer narrows the scope-aware workspace route via the filter rather than
  // falling back to the peer-level route, which has no scope option.
  expect(bodyFor("/search")).toMatchObject({ scope: "therapy", filters: { peer_id: "alice" } });
  expect(requests.some((r) => r.path.endsWith("/peers/alice/search"))).toBe(false);
});

test("search rejects scope combined with session_id before HTTP", async () => {
  const result = await client.callTool({
    name: "search",
    arguments: { query: "q", scope: "therapy", session_id: "s1" },
  });
  expect(result.isError).toBe(true);
  expect(requests).toEqual([]);
});
