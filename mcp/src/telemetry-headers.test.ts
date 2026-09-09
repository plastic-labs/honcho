import { expect, test } from "bun:test";
import pkg from "../package.json";
import {
  HEADER_HOST,
  HEADER_PLUGIN,
  honchoClients,
  identityHeaders,
} from "./config.ts";

const config = { apiKey: "test-key", baseUrl: "https://api.honcho.dev" };
const hostHeader = `honcho-mcp/${pkg.version}`;

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

test("caller User-Agent is sent verbatim as Plugin on scoped and unscoped clients", async () => {
  const ua = "claude-code/2.1.150 (cli)";
  const { clientFor, unscoped } = honchoClients(config, identityHeaders(ua));
  const headers = await capturedHeaders(async () => {
    await unscoped.workspaces();
    await clientFor("sandbox").workspaces();
  });
  expect(headers.length).toBe(2);
  for (const h of headers) {
    expect(h.get(HEADER_HOST)).toBe(hostHeader);
    expect(h.get(HEADER_PLUGIN)).toBe(ua);
    expect(h.get("X-Honcho-Agent-Model")).toBeNull();
  }
});

test("no User-Agent means Host only", async () => {
  const headers = await capturedHeaders(() =>
    honchoClients(config).unscoped.workspaces(),
  );
  expect(headers[0].get(HEADER_HOST)).toBe(hostHeader);
  expect(headers[0].get(HEADER_PLUGIN)).toBeNull();
  expect(identityHeaders("   ")[HEADER_PLUGIN]).toBeUndefined();
});
