import { expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { honchoClients } from "./config.js";
import { createServer } from "./server.js";

const config = {
  apiKey: "test-key",
  baseUrl: "https://honcho.test",
  workspaceId: "test-workspace",
};

/**
 * The Claude Connectors Directory rejects servers whose tools lack a title or
 * the applicable read/write hint, and caps tool names at 64 characters. These
 * also drive auto-permissions in Claude: read-only tools run without a
 * per-call prompt, destructive ones always prompt. A new tool without
 * annotations fails here rather than in review.
 */
test("every tool declares a title and a read/write hint", async () => {
  const server = createServer({ config, ...honchoClients(config) });
  const client = new Client({ name: "test", version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  await client.connect(clientTransport);

  const { tools } = await client.listTools();
  expect(tools.length).toBeGreaterThan(0);

  const untitled = tools.filter((t) => !t.annotations?.title);
  expect(untitled.map((t) => t.name)).toEqual([]);

  const unhinted = tools.filter(
    (t) =>
      t.annotations?.readOnlyHint !== true &&
      typeof t.annotations?.destructiveHint !== "boolean",
  );
  expect(unhinted.map((t) => t.name)).toEqual([]);

  // A read-only tool must not also claim to destroy anything.
  const contradictory = tools.filter(
    (t) => t.annotations?.readOnlyHint === true && t.annotations?.destructiveHint,
  );
  expect(contradictory.map((t) => t.name)).toEqual([]);

  const overlong = tools.filter((t) => t.name.length > 64);
  expect(overlong.map((t) => t.name)).toEqual([]);

  await client.close();
  await server.close();
});
