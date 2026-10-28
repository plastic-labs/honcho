import { afterEach, beforeEach, expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { honchoClients } from "../config.js";
import { register } from "./conclusions.js";

const config = {
  apiKey: "test-key",
  baseUrl: "https://honcho.test",
  workspaceId: "test-workspace",
};

const conclusion = (over: Record<string, unknown> = {}) => ({
  id: "c1",
  content: "alice ships small PRs",
  observer_id: "alice",
  observed_id: "alice",
  session_id: "s1",
  level: "inductive",
  source_ids: ["p1", "p2"],
  source_message_ids: null,
  times_derived: 3,
  created_at: "2026-09-15T00:00:00Z",
  ...over,
});

const requests: { method: string; path: string; body: unknown }[] = [];
const originalFetch = globalThis.fetch;
let items: object[];
let client: Client;
let server: McpServer;

beforeEach(async () => {
  requests.length = 0;
  items = [conclusion()];
  globalThis.fetch = async (input, init) => {
    const path = new URL(String(input)).pathname;
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    requests.push({ method: init?.method ?? "GET", path, body });
    if (path.endsWith("/conclusions/list")) {
      return Response.json({
        items,
        total: items.length,
        page: 1,
        size: 50,
        pages: 1,
      });
    }
    return Response.json({});
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

const payload = (result: unknown) =>
  JSON.parse((result as { content: { text: string }[] }).content[0].text);

test("list_conclusions surfaces attribution and forwards paging and filters", async () => {
  const result = await client.callTool({
    name: "list_conclusions",
    arguments: {
      peer_id: "alice",
      filters: { level: "inductive" },
      page: 2,
      size: 25,
    },
  });

  expect(payload(result).conclusions[0]).toMatchObject({
    id: "c1",
    level: "inductive",
    source_ids: ["p1", "p2"],
    source_message_ids: null,
    times_derived: 3,
  });

  const list = requests.find((r) => r.path.endsWith("/conclusions/list"));
  expect(list?.body).toMatchObject({
    filters: { observer_id: "alice", observed_id: "alice", level: "inductive" },
  });
});

test("get_conclusions reports ids the server did not return", async () => {
  const result = await client.callTool({
    name: "get_conclusions",
    arguments: { conclusion_ids: ["c1", "gone"] },
  });

  const body = payload(result);
  expect(body.conclusions.map((c: { id: string }) => c.id)).toEqual(["c1"]);
  expect(body.missing).toEqual(["gone"]);
});

test("get_derived_conclusions asks for conclusions containing the premise id", async () => {
  await client.callTool({
    name: "get_derived_conclusions",
    arguments: { conclusion_id: "p1" },
  });

  const list = requests.find((r) => r.path.endsWith("/conclusions/list"));
  expect(list?.body).toMatchObject({
    filters: { source_ids: { contains: "p1" } },
  });
});
