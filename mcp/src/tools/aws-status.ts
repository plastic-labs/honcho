import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult } from "../types.js";

interface AwsRdsStatusResult {
  auth_method: string | null;
  rds_hostname: string | null;
  rds_port: number | null;
  aws_region: string | null;
  connection_healthy: boolean;
  error: string | null;
}

export function register(server: McpServer, ctx: ToolContext) {
  server.registerTool(
    "aws_rds_status",
    {
      description: [
        "Check the status of the Honcho API's database connection.",
        "Returns whether the connection is healthy based on the /health endpoint.",
        "Additional fields (auth_method, rds_hostname, rds_port, aws_region) are",
        "included when the backend exposes them; otherwise they default to null.",
        "Use this to diagnose connectivity issues with the Honcho backend.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const healthUrl = new URL("/health", ctx.config.baseUrl).toString();
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        let response: Response;
        try {
          response = await fetch(healthUrl, {
            headers: {
              Authorization: `Bearer ${ctx.config.apiKey}`,
            },
            signal: controller.signal,
          });
        } finally {
          clearTimeout(timeoutId);
        }

        if (!response.ok) {
          return errorResult(
            `Health check failed: HTTP ${response.status} ${response.statusText}`,
          );
        }

        const data = await response.json() as Record<string, unknown>;

        const result: AwsRdsStatusResult = {
          auth_method: typeof data.auth_method === "string" ? data.auth_method : null,
          rds_hostname: typeof data.rds_hostname === "string" ? data.rds_hostname : null,
          rds_port: typeof data.rds_port === "number" ? data.rds_port : null,
          aws_region: typeof data.aws_region === "string" ? data.aws_region : null,
          connection_healthy: data.status === "ok",
          error: null,
        };

        return textResult(result);
      } catch (e) {
        return errorResult(
          `Health check failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}
