import { expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import {
  MAX_CALLS_PER_SCRIPT,
  describe,
  scriptApi,
  signature,
  unwrap,
} from "./codemode-tools.js";
import { honchoClients } from "./config.js";
import { createServer } from "./server.js";

const config = { apiKey: "test-key", baseUrl: "https://honcho.test" };
const readOnlyServer = () =>
  createServer({ config, ...honchoClients(config) }, { readOnly: true });

async function listTools(server: ReturnType<typeof createServer>) {
  const client = new Client({ name: "test", version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  await client.connect(clientTransport);
  return (await client.listTools()).tools;
}

test("a read-only server registers exactly the tools annotated read-only", async () => {
  const all = await listTools(createServer({ config, ...honchoClients(config) }));
  const readOnly = await listTools(readOnlyServer());

  expect(readOnly.map((t) => t.name)).toEqual(
    all.filter((t) => t.annotations?.readOnlyHint === true).map((t) => t.name),
  );
  expect(readOnly.length).toBeLessThan(all.length);
  expect(readOnly.map((t) => t.name)).not.toContain("create_session");
});

test("scripts get the read tools and describe, and nothing that writes", async () => {
  const api = await scriptApi(readOnlyServer());
  const fns = api.functions();

  expect(Object.keys(fns)).toContain("search");
  expect(Object.keys(fns)).toContain("describe");
  expect(Object.keys(fns)).not.toContain("add_messages_to_session");
  // One line per function, small enough to sit in a tool description.
  expect(api.listing.split("\n").length).toBe(Object.keys(fns).length);
  expect(api.listing.length).toBeLessThan(8_000);
});

test("a signature shows parameter types and marks the optional ones", async () => {
  const search = (await listTools(readOnlyServer())).find((t) => t.name === "search")!;

  const line = signature(search);
  expect(line).toStartWith("honcho.search({ workspace_id?: string, query: string,");
  expect(line).toContain(" — ");
  expect(line).not.toContain("\n");

  const detail = describe(search);
  expect(detail.description).toBe(search.description);
  expect(detail.parameters.query).toMatchObject({ type: "string", required: true });
});

test("describe names the known functions when asked for an unknown one", async () => {
  const fns = (await scriptApi(readOnlyServer())).functions();

  expect(await fns.describe({ name: "search" })).toMatchObject({ name: "search" });
  expect(fns.describe({ name: "create_session" })).rejects.toThrow(/Unknown function.*search/);
});

test("a tool result reaches the script as a value, and a tool error throws", () => {
  expect(unwrap({ content: [{ type: "text", text: '{"total":2}' }] })).toEqual({ total: 2 });
  expect(unwrap({ content: [{ type: "text", text: "plain" }] })).toBe("plain");
  expect(() =>
    unwrap({ content: [{ type: "text", text: "Search failed" }], isError: true }),
  ).toThrow("Search failed");
});

test("each script run has its own budget of Honcho calls", async () => {
  const api = await scriptApi(readOnlyServer());
  const fns = api.functions();
  // The calls fail (no such API), which still spends the budget.
  for (let i = 0; i < MAX_CALLS_PER_SCRIPT; i++) {
    await fns.get_queue_status({ workspace_id: "w" }).catch(() => undefined);
  }
  expect(fns.get_queue_status({ workspace_id: "w" })).rejects.toThrow(/at most/);
  // A new run starts from zero.
  expect(api.functions().describe({ name: "search" })).resolves.toBeDefined();
});
