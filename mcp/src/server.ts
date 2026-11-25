import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "./types.js";
import { register as registerWorkspaceTools } from "./tools/workspace.js";
import { register as registerPeerTools } from "./tools/peers.js";
import { register as registerSessionTools } from "./tools/sessions.js";
import { register as registerScopeTools } from "./tools/scopes.js";
import { register as registerConclusionTools } from "./tools/conclusions.js";
import { register as registerSystemTools } from "./tools/system.js";
import instructions from "../instructions.md";
import pkg from "../package.json";

export interface ServerOptions {
  /** Register only the tools annotated `readOnlyHint: true`. */
  readOnly?: boolean;
}

export function createServer(
  ctx: ToolContext,
  options: ServerOptions = {},
): McpServer {
  const server = new McpServer(
    {
      name: "Honcho MCP Server",
      version: pkg.version,
    },
    { instructions },
  );

  if (options.readOnly) {
    // Every tool declares a read/write hint (see server.test.ts), so the
    // annotation is a complete list of what may run on a read-only server.
    const register = server.registerTool.bind(server);
    server.registerTool = ((name, config, handler) =>
      config.annotations?.readOnlyHint === true
        ? register(name, config, handler)
        : undefined) as typeof server.registerTool;
  }

  registerWorkspaceTools(server, ctx);
  registerPeerTools(server, ctx);
  registerSessionTools(server, ctx);
  registerScopeTools(server, ctx);
  registerConclusionTools(server, ctx);
  registerSystemTools(server, ctx);

  return server;
}
