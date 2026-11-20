// Code Mode entry point: the read-only Honcho tools behind one `execute` tool.
// The model writes a JavaScript function that chains them; it runs in a fresh
// network-less isolate (Worker Loader) and reaches Honcho only through RPC back
// to this Worker, so the caller's token never enters the script.
import {
  DynamicWorkerExecutor,
  truncateResponse,
  truncateResult,
} from "@cloudflare/codemode";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { createLegacyMcpHandler } from "agents/mcp";
import { z } from "zod";
import { MAX_CALLS_PER_SCRIPT, scriptApi } from "./codemode-tools.js";
import {
  HEADER_PLUGIN,
  honchoClients,
  identityHeaders,
  parseConfig,
  type Env,
} from "./config.js";
import { createServer } from "./server.js";
import { invalidTokenResponse } from "./types.js";
import description from "../codemode.md";
import pkg from "../package.json";

interface CodeModeEnv extends Env {
  LOADER: ConstructorParameters<typeof DynamicWorkerExecutor>[0]["loader"];
  /** Per-caller cap on script runs: each one starts a billable isolate. */
  SCRIPT_LIMITER: { limit(options: { key: string }): Promise<{ success: boolean }> };
}

/** `chat` runs a tool-using agent, so a script can far outlast the 60s default. */
const SCRIPT_TIMEOUT_MS = 300_000;

const CORS = {
  origin: "*",
  methods: "GET, POST, DELETE, OPTIONS",
  headers: "Content-Type, Authorization, X-Honcho-Workspace-ID, X-Honcho-Plugin",
};
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": CORS.origin,
  "Access-Control-Allow-Methods": CORS.methods,
  "Access-Control-Allow-Headers": CORS.headers,
  "Access-Control-Expose-Headers": "WWW-Authenticate",
};
const PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource";

export default {
  async fetch(request, env, executionCtx): Promise<Response> {
    const url = new URL(request.url);
    const resourceMetadata = `${url.origin}${PROTECTED_RESOURCE_PATH}`;
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }
    if (url.pathname === PROTECTED_RESOURCE_PATH) {
      return Response.json(
        {
          resource: url.origin,
          authorization_servers: [
            env.HONCHO_API_URL?.trim() || "https://api.honcho.dev",
          ],
          bearer_methods_supported: ["header"],
          // Not ["read"] alone: the SDK get-or-creates the workspace (and a
          // session or peer) before reading, and the Honcho gateway refuses
          // those POSTs on a read grant. Read-only here comes from registering
          // only the read tools.
          scopes_supported: ["read", "write"],
        },
        { headers: CORS_HEADERS },
      );
    }

    let config;
    try {
      config = parseConfig(request, env);
    } catch (e) {
      return new Response(
        JSON.stringify({ error: e instanceof Error ? e.message : String(e) }),
        {
          status: 401,
          headers: {
            "Content-Type": "application/json",
            "WWW-Authenticate": `Bearer resource_metadata="${resourceMetadata}"`,
            ...CORS_HEADERS,
          },
        },
      );
    }

    let rejected = false;
    const api = await scriptApi(
      createServer(
        {
          config,
          ...honchoClients(
            config,
            identityHeaders(
              request.headers.get("User-Agent"),
              request.headers.get(HEADER_PLUGIN),
            ),
          ),
          onUnauthorized: () => {
            rejected = true;
          },
        },
        { readOnly: true },
      ),
    );
    const executor = new DynamicWorkerExecutor({
      loader: env.LOADER,
      timeout: SCRIPT_TIMEOUT_MS,
    });
    // Scripts run before any Honcho call proves the token, so the cap is by
    // network address, not by token.
    const caller = request.headers.get("CF-Connecting-IP") ?? "unknown";

    const server = new McpServer({ name: "Honcho Code Mode", version: pkg.version });
    server.registerTool(
      "execute",
      {
        annotations: {
          title: "Run Honcho Script",
          readOnlyHint: true,
          destructiveHint: false,
          openWorldHint: false,
        },
        description: description
          .replace("{{functions}}", api.listing)
          .replace("{{max_calls}}", String(MAX_CALLS_PER_SCRIPT)),
        inputSchema: {
          code: z.string().describe("JavaScript async arrow function to run"),
        },
      },
      async ({ code }) => {
        const started = Date.now();
        const { success } = await env.SCRIPT_LIMITER.limit({ key: caller });
        if (!success) {
          return {
            content: [{ type: "text", text: "Error: too many scripts; wait a minute and retry." }],
            isError: true,
          };
        }
        const out = await executor.execute(code, [
          { name: "honcho", fns: api.functions() },
        ]);
        // Workers Logs: usage and failures, never the script or its results.
        console.log(
          JSON.stringify({ tool: "execute", ms: Date.now() - started, ok: !out.error }),
        );
        const logs = out.logs?.length ? `\n\nconsole:\n${out.logs.join("\n")}` : "";
        if (out.error) {
          return {
            content: [{ type: "text", text: truncateResponse(`Error: ${out.error}${logs}`) }],
            isError: true,
          };
        }
        // Compact, because truncateResult sizes the value by its compact JSON:
        // indenting it would overrun the budget and get cut mid-value below.
        const body = JSON.stringify(truncateResult(out.result)) ?? "undefined";
        return { content: [{ type: "text", text: truncateResponse(body + logs) }] };
      },
    );

    const response = await createLegacyMcpHandler(server, {
      route: "/",
      enableJsonResponse: true,
      corsOptions: CORS,
    })(request, env, executionCtx);
    return rejected
      ? invalidTokenResponse(resourceMetadata, CORS_HEADERS)
      : response;
  },
} satisfies ExportedHandler<CodeModeEnv>;
