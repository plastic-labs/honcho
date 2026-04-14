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
        "Check the status of the Honcho API's database connection and AWS RDS configuration.",
        "Returns the authentication method (password or iam), RDS hostname, port, region,",
        "and whether the database connection is healthy.",
        "Use this to diagnose connectivity issues with the Honcho backend.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const healthUrl = `${ctx.config.baseUrl}/health`;
        const response = await fetch(healthUrl, {
          headers: {
            Authorization: `Bearer ${ctx.config.apiKey}`,
          },
        });

        if (!response.ok) {
          return errorResult(
            `Health check failed: HTTP ${response.status} ${response.statusText}`,
          );
        }

        const data = await response.json() as Record<string, unknown>;

        const result: AwsRdsStatusResult = {
          auth_method: (data.auth_method as string) ?? null,
          rds_hostname: (data.rds_hostname as string) ?? null,
          rds_port: (data.rds_port as number) ?? null,
          aws_region: (data.aws_region as string) ?? null,
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
