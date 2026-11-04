import { afterEach, beforeEach, expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { honchoClients } from "../config.js";
import { DREAM_SECTION_LIMIT, register } from "./system.js";

const config = {
  apiKey: "test-key",
  baseUrl: "https://honcho.test",
  workspaceId: "test-workspace",
};

const run = {
  id: "run1",
  observer_id: "maya",
  observed_id: "maya",
  dream_type: "omni",
  trigger_reason: "manual",
  status: "completed",
  created_count: 1,
  removed_count: 1,
  peer_card_updated: true,
  started_at: "2026-11-03T00:00:00Z",
  finished_at: "2026-11-03T00:01:00Z",
  pruned_at: null,
};

const created = (id: string) => ({
  conclusion: {
    id,
    content: "Maya relies on caffeine to work nights",
    level: "deductive",
    status: "current",
    removed_in_run: null,
  },
  sources: [
    {
      id: "c1",
      content: "Maya drinks coffee daily",
      level: "explicit",
      status: "removed",
      removed_in_run: "run1",
    },
  ],
  specialist: "deduction",
});

const removed = (id: string) => ({
  id,
  content: "Maya drinks coffee daily",
  level: "explicit",
  replaced_by: null,
  specialist: "deduction",
});

const requests: { method: string; path: string; body: unknown; query: string }[] =
  [];
const originalFetch = globalThis.fetch;
let detail: Record<string, unknown>;
let status: number;
let client: Client;
let server: McpServer;

beforeEach(async () => {
  requests.length = 0;
  status = 200;
  detail = {
    ...run,
    created: [created("c3")],
    removed: [removed("c1")],
    peer_card: { before: ["ATTRIBUTE: Drinks coffee"], after: [] },
  };
  globalThis.fetch = async (input, init) => {
    const url = new URL(String(input));
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    requests.push({
      method: init?.method ?? "GET",
      path: url.pathname,
      body,
      query: url.search,
    });
    if (status !== 200) {
      return Response.json({ detail: "Not authenticated" }, { status });
    }
    if (url.pathname.endsWith("/dreams/list")) {
      return Response.json({ items: [run], total: 1, page: 2, size: 10, pages: 2 });
    }
    if (url.pathname.includes("/dreams/")) {
      return Response.json(detail);
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

const text = (result: unknown) =>
  (result as { content: { text: string }[] }).content[0].text;
const payload = (result: unknown) => JSON.parse(text(result));

test("list_dreams forwards the peer filters and paging", async () => {
  const result = await client.callTool({
    name: "list_dreams",
    arguments: { observed_id: "maya", observer_id: "assistant", page: 2, size: 10 },
  });

  expect(payload(result)).toEqual({ dreams: [run], total: 1, page: 2, pages: 2 });
  const listCall = requests.find((r) => r.path.endsWith("/dreams/list"));
  expect(listCall?.path).toBe("/v3/workspaces/test-workspace/dreams/list");
  expect(listCall?.body).toEqual({ observed_id: "maya", observer_id: "assistant" });
  expect(listCall?.query).toContain("page=2");
  expect(listCall?.query).toContain("size=10");
});

test("list_dreams without filters sends an empty body", async () => {
  await client.callTool({ name: "list_dreams", arguments: {} });

  const listCall = requests.find((r) => r.path.endsWith("/dreams/list"));
  expect(listCall?.body).toEqual({});
});

test("get_dream returns the run and its changes", async () => {
  const result = await client.callTool({
    name: "get_dream",
    arguments: { dream_id: "run1" },
  });

  const body = payload(result);
  expect(requests.at(-1)?.path).toBe("/v3/workspaces/test-workspace/dreams/run1");
  expect(body.status).toBe("completed");
  expect(body.created[0].conclusion.id).toBe("c3");
  expect(body.created[0].sources[0].status).toBe("removed");
  expect(body.removed[0].content).toBe("Maya drinks coffee daily");
  expect(body.peer_card.before).toEqual(["ATTRIBUTE: Drinks coffee"]);
  expect([body.created_omitted, body.removed_omitted]).toEqual([0, 0]);
});

test("get_dream caps each section and counts the rest", async () => {
  const extra = 7;
  detail.created = Array.from({ length: DREAM_SECTION_LIMIT + extra }, (_, i) =>
    created(`c${i}`),
  );
  detail.removed = Array.from({ length: DREAM_SECTION_LIMIT + 1 }, (_, i) =>
    removed(`r${i}`),
  );

  const body = payload(
    await client.callTool({ name: "get_dream", arguments: { dream_id: "run1" } }),
  );

  expect(body.created).toHaveLength(DREAM_SECTION_LIMIT);
  expect(body.created_omitted).toBe(extra);
  expect(body.removed).toHaveLength(DREAM_SECTION_LIMIT);
  expect(body.removed_omitted).toBe(1);
});

test("get_dream rejects an id that is not a dream id", async () => {
  const result = await client.callTool({
    name: "get_dream",
    arguments: { dream_id: "../peers/list" },
  });

  expect((result as { isError?: boolean }).isError).toBe(true);
  expect(requests.some((r) => r.path.includes("dreams"))).toBe(false);
});

test("a 401 on a dream tool says it needs a workspace or admin key", async () => {
  status = 401;

  const result = await client.callTool({ name: "list_dreams", arguments: {} });

  expect((result as { isError?: boolean }).isError).toBe(true);
  expect(text(result)).toContain("needs a workspace or admin key");
});
