import { createMcpHandler } from "agents/mcp";
import {
  parseConfig,
  createClientFactory,
  createUnscopedClient,
  type Env,
} from "./config.js";
import { createServer } from "./server.js";

const CORS_ORIGIN = "*";
const CORS_METHODS = "GET, POST, DELETE, OPTIONS";
const CORS_ALLOWED_HEADERS =
  "Content-Type, Authorization, X-Honcho-Workspace-ID";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": CORS_ORIGIN,
  "Access-Control-Allow-Methods": CORS_METHODS,
  "Access-Control-Allow-Headers": CORS_ALLOWED_HEADERS,
  "Access-Control-Expose-Headers": "WWW-Authenticate",
};

const PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource";
const HEALTHCHECK_TIMEOUT_MS = 15_000;

function resourceUrl(request: Request): string {
  return new URL(request.url).origin;
}

function authorizationServer(env: Env): string {
  return env.HONCHO_API_URL?.trim() || "https://api.honcho.dev";
}

export default {
  // Probes the authorization server API to confirm it is reachable and healthy.
  async scheduled(
    _controller: ScheduledController,
    env: Env,
    _executionCtx: ExecutionContext,
  ): Promise<void> {
    const upstream = `${authorizationServer(env)}/health`;

    let failure: string | null = null;
    try {
      const response = await fetch(upstream, {
        signal: AbortSignal.timeout(HEALTHCHECK_TIMEOUT_MS),
      });
      if (!response.ok) failure = `HTTP ${response.status}`;
    } catch (e) {
      failure = e instanceof Error ? e.message : String(e);
    }
    if (!failure) return;

    console.error(`healthcheck failed: ${upstream} — ${failure}`);
    if (!env.ALERT_WEBHOOK_URL) return;

    const alertResponse = await fetch(env.ALERT_WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: `🚨 honcho-mcp cannot reach upstream API (${upstream}): ${failure}`,
      }),
      signal: AbortSignal.timeout(HEALTHCHECK_TIMEOUT_MS),
    });
    if (!alertResponse.ok) {
      throw new Error(`alert webhook failed: HTTP ${alertResponse.status}`);
    }
  },

  async fetch(
    request: Request,
    env: Env,
    executionCtx: ExecutionContext,
  ): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    // Protected Resource Metadata (RFC 9728) — served without auth so clients
    // can discover the authorization server.
    if (new URL(request.url).pathname === PROTECTED_RESOURCE_PATH) {
      return Response.json(
        {
          resource: resourceUrl(request),
          authorization_servers: [authorizationServer(env)],
          bearer_methods_supported: ["header"],
          scopes_supported: ["read", "write"],
        },
        { headers: CORS_HEADERS },
      );
    }

    let config;
    try {
      config = parseConfig(request, env);
    } catch (e) {
      const message =
        e instanceof Error ? e.message : "Invalid request";
      // WWW-Authenticate points clients at the metadata so they start the OAuth flow.
      const resourceMetadata = `${resourceUrl(request)}${PROTECTED_RESOURCE_PATH}`;
      return new Response(JSON.stringify({ error: message }), {
        status: 401,
        headers: {
          "Content-Type": "application/json",
          "WWW-Authenticate": `Bearer resource_metadata="${resourceMetadata}"`,
          ...CORS_HEADERS,
        },
      });
    }

    try {
      const server = createServer({
        config,
        clientFor: createClientFactory(config),
        unscoped: createUnscopedClient(config),
      });
      const handler = createMcpHandler(server, {
        route: "/",
        corsOptions: {
          origin: CORS_ORIGIN,
          methods: CORS_METHODS,
          headers: CORS_ALLOWED_HEADERS,
        },
      });
      return await handler(request, env, executionCtx);
    } catch (e) {
      const message =
        e instanceof Error ? e.message : "Internal server error";
      return new Response(JSON.stringify({ error: message }), {
        status: 500,
        headers: { "Content-Type": "application/json", ...CORS_HEADERS },
      });
    }
  },
};
