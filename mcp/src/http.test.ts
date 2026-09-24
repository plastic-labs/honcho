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
