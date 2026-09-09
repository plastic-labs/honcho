import { expect, test } from "bun:test";
import pkg from "../package.json";
import { createClient, createUnscopedClient } from "./config.ts";
import {
  HEADER_HOST,
  HEADER_PLUGIN,
  identityHeaders,
  pluginFromCaller,
} from "./identity.ts";

const config = { apiKey: "test-key", baseUrl: "https://api.honcho.dev" };
const hostHeader = `honcho-mcp/${pkg.version} (${process.platform})`;

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
  const plugin = pluginFromCaller({
    name: "codex-mcp-client",
    version: "0.148.0",
  });
  const headers = await capturedHeaders(() =>
    createUnscopedClient(config, identityHeaders(plugin)).workspaces(),
  );
  expect(headers[0].get(HEADER_HOST)).toBe(hostHeader);
  expect(headers[0].get(HEADER_PLUGIN)).toBe("codex-mcp-client/0.148.0");
});

test("self-identifying User-Agent becomes Plugin; runtime UAs do not", async () => {
  expect(pluginFromCaller(undefined, "codex-mcp-client/0.148.0-alpha.9")).toBe(
    "codex-mcp-client/0.148.0-alpha.9",
  );
  expect(pluginFromCaller(undefined, "node")).toBeUndefined();
  expect(pluginFromCaller(undefined, "undici")).toBeUndefined();
  expect(pluginFromCaller(undefined, "httpx/0.27.0")).toBeUndefined();
  expect(
    pluginFromCaller({ name: "hermes", version: "1.2.3" }, "undici"),
  ).toBe("hermes/1.2.3");

  const plugin = pluginFromCaller(
    undefined,
    "codex-mcp-client/0.148.0-alpha.9",
  );
  const headers = await capturedHeaders(() =>
    createClient(config, "sandbox", identityHeaders(plugin)).workspaces(),
  );
  expect(headers[0].get(HEADER_HOST)).toBe(hostHeader);
  expect(headers[0].get(HEADER_PLUGIN)).toBe(
    "codex-mcp-client/0.148.0-alpha.9",
  );
});

test("Plugin set on the map before construct is copied onto the client", async () => {
  const shared = identityHeaders();
  shared[HEADER_PLUGIN] = "codex-mcp-client/0.148.0";
  const headers = await capturedHeaders(() =>
    createUnscopedClient(config, shared).workspaces(),
  );
  expect(headers[0].get(HEADER_HOST)).toBe(hostHeader);
  expect(headers[0].get(HEADER_PLUGIN)).toBe("codex-mcp-client/0.148.0");
});
