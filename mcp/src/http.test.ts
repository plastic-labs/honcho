import { expect, test } from "bun:test";
import { fetch } from "./http.ts";

const origin = "http://127.0.0.1:3000";

const initializeBody = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "test", version: "0.0.0" },
  },
};

const pingBody = { jsonrpc: "2.0", id: 2, method: "ping" };

function mcpPost(headers: Record<string, string>, body: unknown) {
  return fetch(
    new Request(`${origin}/mcp`, {
      method: "POST",
      headers: {
        Accept: "application/json, text/event-stream",
        "Content-Type": "application/json",
        ...headers,
      },
      body: JSON.stringify(body),
    }),
  );
}

test("established sessions require the initialize bearer", async () => {
  const init = await mcpPost(
    { Authorization: "Bearer key-a" },
    initializeBody,
  );
  expect(init.status).toBe(200);
  const sessionId = init.headers.get("mcp-session-id");
  expect(sessionId).toBeTruthy();

  const missing = await mcpPost({ "mcp-session-id": sessionId! }, pingBody);
  expect(missing.status).toBe(401);

  const wrong = await mcpPost(
    { Authorization: "Bearer key-b", "mcp-session-id": sessionId! },
    pingBody,
  );
  expect(wrong.status).toBe(401);

  const ok = await mcpPost(
    { Authorization: "Bearer key-a", "mcp-session-id": sessionId! },
    pingBody,
  );
  expect(ok.status).toBe(200);
});

test("unknown or expired session ID returns 404 so clients can reinitialize", async () => {
  // POST carrying a stale session ID with a non-initialize body must be 404,
  // not 400: the Streamable HTTP spec says a terminated session MUST answer
  // requests containing its ID with 404, which tells the client to
  // reinitialize (modelcontextprotocol.io/specification/2025-03-26/basic/transports).
  const stalePost = await mcpPost(
    { "mcp-session-id": "00000000-0000-0000-0000-000000000000" },
    pingBody,
  );
  expect(stalePost.status).toBe(404);

  // GET carrying a stale session ID must also be 404.
  const staleGet = await fetch(
    new Request(`${origin}/mcp`, {
      method: "GET",
      headers: {
        Accept: "text/event-stream",
        "mcp-session-id": "00000000-0000-0000-0000-000000000000",
      },
    }),
  );
  expect(staleGet.status).toBe(404);

  // POST with no session ID at all and a non-initialize body stays 400.
  const noSession = await mcpPost({}, pingBody);
  expect(noSession.status).toBe(400);
});

test("MCP requests disable the idle timeout", async () => {
  const disabled: string[] = [];
  const server = {
    timeout(request: Request, seconds: number) {
      if (seconds === 0) disabled.push(new URL(request.url).pathname);
    },
  };
  await fetch(new Request(`${origin}/health`), server);
  await fetch(
    new Request(`${origin}/mcp`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pingBody),
    }),
    server,
  );
  expect(disabled).toEqual(["/mcp"]);
});

test("a session's tool calls carry the caller's plugin header", async () => {
  const identity = {
    Authorization: "Bearer key-c",
    "User-Agent": "Cursor/1.0.0",
    "X-Honcho-Plugin": "cursor-honcho/0.2.0",
  };
  const init = await mcpPost(identity, initializeBody);
  expect(init.status).toBe(200);
  const sessionId = init.headers.get("mcp-session-id")!;

  const seen: Headers[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (_input, reqInit) => {
    seen.push(new Headers(reqInit?.headers as HeadersInit));
    return new Response(
      JSON.stringify({ items: [], page: 1, size: 10, total: 0, pages: 0 }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };
  try {
    const call = await mcpPost(
      { ...identity, "mcp-session-id": sessionId },
      {
        jsonrpc: "2.0",
        id: 3,
        method: "tools/call",
        params: { name: "list_workspaces", arguments: {} },
      },
    );
    expect(call.status).toBe(200);
    await call.text();
  } finally {
    globalThis.fetch = original;
  }

  expect(seen.length).toBeGreaterThan(0);
  expect(seen[0].get("X-Honcho-Plugin")).toBe("cursor-honcho/0.2.0 (Cursor/1.0.0)");
  expect(seen[0].get("X-Honcho-Host")).toMatch(/^honcho-mcp\//);
});
