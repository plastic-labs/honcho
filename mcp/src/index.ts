import { createMcpHandler } from "agents/mcp";
import { parseConfig, createClient } from "./config.js";
import { createServer } from "./server.js";

const CORS_ORIGIN = "*";
const CORS_METHODS = "GET, POST, DELETE, OPTIONS";
const CORS_ALLOWED_HEADERS =
  "Content-Type, Authorization, X-Honcho-User-Name, X-Honcho-Base-URL, X-Honcho-Workspace-ID, X-Honcho-Assistant-Name";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": CORS_ORIGIN,
  "Access-Control-Allow-Methods": CORS_METHODS,
  "Access-Control-Allow-Headers": CORS_ALLOWED_HEADERS,
};

export default {
  async fetch(
    request: Request,
    env: unknown,
    executionCtx: ExecutionContext,
  ): Promise<Response> {
    let config;
    try {
      config = parseConfig(request);
    } catch (e) {
      const message =
        e instanceof Error ? e.message : "Invalid request";
      return new Response(JSON.stringify({ error: message }), {
        status: 401,
        headers: { "Content-Type": "application/json", ...CORS_HEADERS },
      });
    }

    try {
      const honcho = createClient(config);
      const server = createServer({ honcho, config });
      const handler = createMcpHandler(server, {
        corsOptions: {
          origin: CORS_ORIGIN,
          methods: CORS_METHODS,
          headers: CORS_ALLOWED_HEADERS,
        },
      });
      return handler(request, env, executionCtx);
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
