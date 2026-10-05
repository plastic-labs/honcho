import { afterEach, beforeEach, expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { honchoClients } from "../config.js";
import { register } from "./scopes.js";

const config = {
  apiKey: "test-key",
  baseUrl: "https://honcho.test",
  workspaceId: "test-workspace",
};
const requests: { method: string; path: string }[] = [];
const originalFetch = globalThis.fetch;
let responseBody: object;
let responseStatus: number;
let client: Client;
let server: McpServer;

beforeEach(async () => {
  requests.length = 0;
  responseBody = { backfill_status: {} };
  responseStatus = 200;
  globalThis.fetch = async (input, init) => {
    requests.push({ method: init?.method ?? "GET", path: new URL(String(input)).pathname });
    return Response.json(responseBody, { status: responseStatus });
  };
  server = new McpServer({ name: "test", version: "1.0.0" });
  register(server, { config, ...honchoClients(config) });
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

test("scope status counts known states and preserves unknown session details", async () => {
  const updated_at = "2026-09-10T00:00:00Z";
  const sessions = {
    first: { state: "pending", updated_at },
    second: { state: "pending", updated_at },
    done: { state: "completed", updated_at, docs_copied: 0 },
    failed: { state: "failed", updated_at },
    unknown: { state: "paused", updated_at },
    inherited: { state: "constructor", updated_at },
    reserved: { state: "scope_id", updated_at },
  };
  responseBody = { backfill_status: sessions };
  const result = await client.callTool({
    name: "get_scope_status",
    arguments: { scope_id: "test-scope" },
  });
  expect(result.isError).not.toBe(true);
  expect(result.content).toEqual([{
    type: "text",
    text: JSON.stringify({ scope_id: "test-scope", pending: 2, completed: 1, failed: 1, sessions }),
  }]);
});

const existingScopeTools = [
  ["get_scope_status", "GET", "/status"],
  ["get_scope_sessions", "POST", "/sessions/list"],
  ["remove_session_from_scope", "DELETE", "/sessions/test-session"],
];

test.each(existingScopeTools)("%s rejects invalid scope IDs before HTTP", async (name) => {
  for (const scope_id of ["..", "%2e%2e", "../other", "valid?typo", "valid#typo", "scope.prefixed", "", "a".repeat(507)]) {
    const result = await client.callTool({
      name,
      arguments: { scope_id, session_id: "test-session" },
    });
    expect(requests).toEqual([]);
    expect(result.isError).toBe(true);
  }
});

test.each(existingScopeTools)("%s returns missing-scope errors without creating scopes", async (name, method, suffix) => {
  responseStatus = 404;
  responseBody = { detail: "Scope Test_scope-123 not found in workspace test-workspace" };
  const result = await client.callTool({
    name,
    arguments: { scope_id: "Test_scope-123", session_id: "test-session" },
  });
  expect(result.isError).toBe(true);
  expect(requests).toEqual([{
    method,
    path: `/v3/workspaces/test-workspace/scopes/Test_scope-123${suffix}`,
  }]);
});
