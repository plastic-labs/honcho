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

const listWorkspacesBody = {
  jsonrpc: "2.0",
  id: 4,
  method: "tools/call",
  params: { name: "list_workspaces", arguments: {} },
};

function upstream(status: number, body: unknown) {
  return async () => {
    // Long enough that calls sent together are all in flight at once.
    await new Promise((resolve) => setTimeout(resolve, 10));
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  };
}

test("a rejected OAuth token is answered with HTTP 401 invalid_token and the session is dropped", async () => {
  const headers = { Authorization: "Bearer hch-at-expired" };
  const init = await mcpPost(headers, initializeBody);
  expect(init.status).toBe(200);
  const sessionId = init.headers.get("mcp-session-id")!;
  const inSession = { ...headers, "mcp-session-id": sessionId };

  const original = globalThis.fetch;
  globalThis.fetch = upstream(401, { detail: "Invalid or expired access token" });
  try {
    // Two overlapping calls, as when an agent runs tools in parallel: both
    // get the 401, and closing the session must not strand the second one.
    const calls = await Promise.all([
      mcpPost(inSession, listWorkspacesBody),
      mcpPost(inSession, { ...listWorkspacesBody, id: 5 }),
    ]);
    for (const call of calls) {
      expect(call.status).toBe(401);
      const challenge = call.headers.get("WWW-Authenticate") ?? "";
      expect(challenge).toContain('error="invalid_token"');
      expect(challenge).toContain("resource_metadata=");
    }
  } finally {
    globalThis.fetch = original;
  }

  // The session is gone, so its id now gets 404 and the client has to start
  // a new session with its refreshed token.
  const stale = await mcpPost(inSession, pingBody);
  expect(stale.status).toBe(404);
});

test("a 401 that is not a dead OAuth token stays a tool error and keeps the session", async () => {
  const cases = [
    // A bad API key: the client has nothing to refresh.
    { token: "hch-plain-api-key", detail: "Invalid JWT" },
    // A live OAuth token on a route it may not use: refreshing won't help.
    { token: "hch-at-live", detail: "Resource requires admin privileges" },
  ];
  for (const { token, detail } of cases) {
    const headers = { Authorization: `Bearer ${token}` };
    const init = await mcpPost(headers, initializeBody);
    const sessionId = init.headers.get("mcp-session-id")!;
    const inSession = { ...headers, "mcp-session-id": sessionId };

    const original = globalThis.fetch;
    globalThis.fetch = upstream(401, { detail });
    try {
      const call = await mcpPost(inSession, listWorkspacesBody);
      expect(call.status).toBe(200);
      const body = (await call.json()) as { result: { isError?: boolean } };
      expect(body.result.isError).toBe(true);
    } finally {
      globalThis.fetch = original;
    }

    const ping = await mcpPost(inSession, pingBody);
    expect(ping.status).toBe(200);
  }
});

test("a successful write overlapping a rejected read keeps its 200", async () => {
  const headers = { Authorization: "Bearer hch-at-expiring" };
  const init = await mcpPost(headers, initializeBody);
  const inSession = {
    ...headers,
    "mcp-session-id": init.headers.get("mcp-session-id")!,
  };

  // The insertion is accepted before the token dies. Its 201 is held until
  // the read has been refused, so the rejection lands while it is in flight.
  let release!: () => void;
  const held = new Promise<void>((resolve) => (release = resolve));
  const json = (status: number, body: unknown) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  const row = { id: "x", metadata: {}, configuration: {}, created_at: "2026-10-19T00:00:00Z" };
  const original = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const path = new URL(input instanceof Request ? input.url : String(input)).pathname;
    if (path.endsWith("/workspaces/list")) {
      return json(401, { detail: "Invalid or expired access token" });
    }
    if (path.endsWith("/messages")) {
      await held;
      return json(201, [{ ...row, content: "hi", peer_id: "p", session_id: "s" }]);
    }
    return json(200, row); // workspace, session and peer get-or-create
  };
  try {
    const write = mcpPost(inSession, {
      jsonrpc: "2.0",
      id: 6,
      method: "tools/call",
      params: {
        name: "add_messages_to_session",
        arguments: {
          workspace_id: "w",
          session_id: "s",
          messages: [{ peer_id: "p", content: "hi" }],
        },
      },
    });
    const read = await mcpPost(inSession, listWorkspacesBody);
    expect(read.status).toBe(401);
    release();
    const response = await write;
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("Messages added to session successfully");
  } finally {
    globalThis.fetch = original;
  }
});
