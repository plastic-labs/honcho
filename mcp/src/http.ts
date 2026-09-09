import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import {
  createClientFactory,
  createUnscopedClient,
  parseConfig,
  type Env,
  type HonchoConfig,
} from "./config.js";
import { createServer } from "./server.js";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

declare const process: {
  env: Record<string, string | undefined>;
};

declare const Bun: {
  serve(options: {
    hostname: string;
    port: number;
    fetch(request: Request): Response | Promise<Response>;
  }): { hostname: string; port: number };
};

const CORS_ORIGIN = "*";
const CORS_METHODS = "GET, POST, DELETE, OPTIONS";
const CORS_ALLOWED_HEADERS =
  "Content-Type, Authorization, X-Honcho-Workspace-ID, mcp-session-id, mcp-protocol-version, last-event-id";

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": CORS_ORIGIN,
  "Access-Control-Allow-Methods": CORS_METHODS,
  "Access-Control-Allow-Headers": CORS_ALLOWED_HEADERS,
  "Access-Control-Expose-Headers": "WWW-Authenticate, mcp-session-id",
};

const PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource";
const MCP_PATHS = new Set(["/", "/mcp"]);

type Session = {
  transport: WebStandardStreamableHTTPServerTransport;
  server: McpServer;
  lastSeen: number;
  apiKey: string;
};

const sessions = new Map<string, Session>();
const DEFAULT_SESSION_IDLE_MS = 30 * 60 * 1000;
const DEFAULT_SESSION_MAX = 128;

function envInt(name: string, fallback: number): number {
  const n = Number(process.env[name]);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function dropSession(id: string): void {
  const session = sessions.get(id);
  if (!session) return;
  sessions.delete(id);
  void session.transport.close();
  void session.server.close();
}

function sweepSessions(): void {
  const idleMs = envInt("MCP_SESSION_IDLE_MS", DEFAULT_SESSION_IDLE_MS);
  const now = Date.now();
  for (const [id, session] of sessions) {
    if (now - session.lastSeen > idleMs) dropSession(id);
  }
}

function envBindings(): Env {
  return { HONCHO_API_URL: process.env.HONCHO_API_URL };
}

function authorizationServer(): string {
  return process.env.HONCHO_API_URL?.trim() || "https://api.honcho.dev";
}

function withCors(response: Response): Response {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(CORS_HEADERS)) {
    headers.set(key, value);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function jsonResponse(
  body: unknown,
  status: number,
  extraHeaders?: Record<string, string>,
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...CORS_HEADERS,
      ...extraHeaders,
    },
  });
}

function configForRequest(request: Request) {
  return parseConfig(request, envBindings());
}

function configOrUnauthorized(request: Request): HonchoConfig | Response {
  try {
    return configForRequest(request);
  } catch (e) {
    const message = e instanceof Error ? e.message : "Invalid request";
    return unauthorized(request, message);
  }
}

function unauthorized(request: Request, message: string): Response {
  const resourceMetadata = `${new URL(request.url).origin}${PROTECTED_RESOURCE_PATH}`;
  return jsonResponse(
    { error: message },
    401,
    {
      "WWW-Authenticate": `Bearer resource_metadata="${resourceMetadata}"`,
    },
  );
}

async function handleMcp(request: Request): Promise<Response> {
  sweepSessions();
  const sessionId = request.headers.get("mcp-session-id");
  if (sessionId) {
    const existing = sessions.get(sessionId);
    if (existing) {
      const config = configOrUnauthorized(request);
      if (config instanceof Response) return config;
      if (config.apiKey !== existing.apiKey) {
        return unauthorized(
          request,
          "Authorization does not match this session.",
        );
      }
      existing.lastSeen = Date.now();
      return withCors(await existing.transport.handleRequest(request));
    }
  }

  if (request.method !== "POST") {
    return jsonResponse(
      {
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Bad Request: No valid session ID provided",
        },
        id: null,
      },
      400,
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      {
        jsonrpc: "2.0",
        error: { code: -32700, message: "Parse error: Invalid JSON" },
        id: null,
      },
      400,
    );
  }

  const messages = Array.isArray(body) ? body : [body];
  if (!messages.some((message) => isInitializeRequest(message))) {
    return jsonResponse(
      {
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Bad Request: No valid session ID provided",
        },
        id: null,
      },
      400,
    );
  }

  const config = configOrUnauthorized(request);
  if (config instanceof Response) return config;

  const server = createServer({
    config,
    clientFor: createClientFactory(config),
    unscoped: createUnscopedClient(config),
  });

  const maxSessions = envInt("MCP_SESSION_MAX", DEFAULT_SESSION_MAX);
  if (sessions.size >= maxSessions) {
    return jsonResponse(
      {
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Too many active sessions",
        },
        id: null,
      },
      503,
    );
  }

  const transport = new WebStandardStreamableHTTPServerTransport({
    sessionIdGenerator: () => crypto.randomUUID(),
    onsessioninitialized: (id) => {
      sessions.set(id, {
        transport,
        server,
        lastSeen: Date.now(),
        apiKey: config.apiKey,
      });
    },
  });
  transport.onclose = () => {
    const id = transport.sessionId;
    if (id) sessions.delete(id);
  };

  await server.connect(transport);
  return withCors(
    await transport.handleRequest(request, { parsedBody: body }),
  );
}

export async function fetch(request: Request): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  const pathname = new URL(request.url).pathname;

  if (pathname === "/health") {
    return jsonResponse({ status: "ok" }, 200);
  }

  if (pathname === PROTECTED_RESOURCE_PATH) {
    return jsonResponse(
      {
        resource: new URL(request.url).origin,
        authorization_servers: [authorizationServer()],
        bearer_methods_supported: ["header"],
        scopes_supported: ["read", "write"],
      },
      200,
    );
  }

  if (!MCP_PATHS.has(pathname)) {
    return jsonResponse({ error: "Not Found" }, 404);
  }

  try {
    return await handleMcp(request);
  } catch (e) {
    const message =
      e instanceof Error ? e.message : "Internal server error";
    return jsonResponse({ error: message }, 500);
  }
}

const isMain = Boolean((import.meta as { main?: boolean }).main);
if (isMain) {
  const hostname = process.env.HOST?.trim() || "0.0.0.0";
  const port = Number(process.env.PORT) || 3000;
  Bun.serve({ hostname, port, fetch });
  console.error(`honcho-mcp listening on http://${hostname}:${port}`);
}
