import { expect, test } from "bun:test";
import pkg from "../package.json";
import { createClient, createUnscopedClient } from "./config.ts";
import {
  HEADER_HOST,
  HEADER_PLUGIN,
  HOST_ID,
  callerFromClientInfo,
  callerFromUserAgent,
  createIdentity,
  resolveCallerIdentity,
} from "./identity.ts";

const config = { apiKey: "test-key", baseUrl: "https://api.honcho.dev" };
const hostHeader = `${HOST_ID}/${pkg.version} (${process.platform})`;

function emptyPage() {
  return { items: [], page: 1, size: 10, total: 0, pages: 0 };
}

async function capturedHeaders(run: () => Promise<unknown>): Promise<Headers[]> {
  const seen: Headers[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (_input, init) => {
    seen.push(new Headers(init?.headers as HeadersInit));
    return new Response(JSON.stringify(emptyPage()), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    await run();
    return seen;
  } finally {
    globalThis.fetch = original;
  }
}

test("every request carries host identity and no agent model", async () => {
  const headers = await capturedHeaders(() =>
    createUnscopedClient(config).workspaces(),
  );
  expect(headers.length).toBeGreaterThan(0);
  for (const h of headers) {
    expect(h.get(HEADER_HOST)).toBe(hostHeader);
    expect(h.get(HEADER_PLUGIN)).toBeNull();
    expect(h.get("X-Honcho-Agent-Model")).toBeNull();
  }
});

test("clientInfo becomes X-Honcho-Plugin", async () => {
  const identity = createIdentity(
    callerFromClientInfo({ name: "codex-mcp-client", version: "0.148.0" }),
  );
  const headers = await capturedHeaders(() =>
    createUnscopedClient(config, identity).workspaces(),
  );
  expect(headers[0].get(HEADER_HOST)).toBe(hostHeader);
  expect(headers[0].get(HEADER_PLUGIN)).toBe("codex-mcp-client/0.148.0");
});

test("self-identifying User-Agent becomes Plugin; runtime UAs do not", async () => {
  expect(callerFromUserAgent("codex-mcp-client/0.148.0-alpha.9")).toEqual({
    plugin: "codex-mcp-client",
    pluginVersion: "0.148.0-alpha.9",
  });
  expect(callerFromUserAgent("node")).toEqual({});
  expect(callerFromUserAgent("undici")).toEqual({});
  expect(callerFromUserAgent("httpx/0.27.0")).toEqual({});
  expect(
    resolveCallerIdentity({
      clientInfo: { name: "hermes", version: "1.2.3" },
      userAgent: "undici",
    }),
  ).toEqual({ plugin: "hermes", pluginVersion: "1.2.3" });

  const identity = createIdentity(
    callerFromUserAgent("codex-mcp-client/0.148.0-alpha.9"),
  );
  const headers = await capturedHeaders(() =>
    createClient(config, "sandbox", identity).workspaces(),
  );
  expect(headers[0].get(HEADER_HOST)).toBe(hostHeader);
  expect(headers[0].get(HEADER_PLUGIN)).toBe(
    "codex-mcp-client/0.148.0-alpha.9",
  );
});

test("held clients pick up Plugin after set", async () => {
  const identity = createIdentity();
  const client = createUnscopedClient(config, identity);
  const before = await capturedHeaders(() => client.workspaces());
  expect(before[0].get(HEADER_HOST)).toBe(hostHeader);
  expect(before[0].get(HEADER_PLUGIN)).toBeNull();

  identity.set({ plugin: "codex-mcp-client", pluginVersion: "0.148.0" });
  const after = await capturedHeaders(() => client.workspaces());
  expect(after[0].get(HEADER_HOST)).toBe(hostHeader);
  expect(after[0].get(HEADER_PLUGIN)).toBe("codex-mcp-client/0.148.0");
});
