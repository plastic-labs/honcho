import { createMcpHandler } from "agents/mcp";
import { parseConfig, createClient, type Env } from "./config.js";
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

function resourceUrl(request: Request): string {
  return new URL(request.url).origin;
}

function authorizationServer(env: Env): string {
  return env.HONCHO_API_URL?.trim() || "https://api.honcho.dev";
}

export default {
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
      const honcho = createClient(config);
      const server = createServer({ honcho, config });
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
