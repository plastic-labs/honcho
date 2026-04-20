import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "./types.js";
import { register as registerWorkspaceTools } from "./tools/workspace.js";
import { register as registerPeerTools } from "./tools/peers.js";
import { register as registerSessionTools } from "./tools/sessions.js";
import { register as registerConclusionTools } from "./tools/conclusions.js";
import { register as registerSystemTools } from "./tools/system.js";
import { register as registerAwsStatusTools } from "./tools/aws-status.js";

export function createServer(ctx: ToolContext): McpServer {
  const server = new McpServer({
    name: "Honcho MCP Server",
    version: "3.0.0",
  });

  registerWorkspaceTools(server, ctx);
  registerPeerTools(server, ctx);
  registerSessionTools(server, ctx);
  registerConclusionTools(server, ctx);
  registerSystemTools(server, ctx);
  registerAwsStatusTools(server, ctx);

  return server;
}
