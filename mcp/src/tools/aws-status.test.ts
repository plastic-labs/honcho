/**
 * Unit tests for the aws_rds_status MCP tool (Task 7.5).
 *
 * Tests healthy and failed health-check responses using a mock McpServer
 * and a stubbed fetch.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { register } from "./aws-status.js";
import type { ToolContext } from "../types.js";
import type { HonchoConfig } from "../config.js";

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/** Captured tool handler from registerTool */
type ToolHandler = (...args: unknown[]) => Promise<{
  content: { type: string; text: string }[];
  isError?: boolean;
}>;

function createMockServer() {
  let capturedHandler: ToolHandler | null = null;

  const server = {
    registerTool: vi.fn((_name: string, _schema: unknown, handler: ToolHandler) => {
      capturedHandler = handler;
    }),
  };

  return {
    server,
    getHandler: () => capturedHandler!,
  };
}

function createMockContext(overrides: Partial<HonchoConfig> = {}): ToolContext {
  const config: HonchoConfig = {
    apiKey: "test-api-key",
    userName: "test-user",
    assistantName: "Assistant",
    baseUrl: "https://api.honcho.dev",
    workspaceId: "default",
    ...overrides,
  };

  return {
    honcho: {} as ToolContext["honcho"],
    config,
  };
}

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

describe("aws_rds_status tool", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("registers the tool with the correct name", () => {
    const { server } = createMockServer();
    const ctx = createMockContext();
    register(server as never, ctx);

    expect(server.registerTool).toHaveBeenCalledOnce();
    expect(server.registerTool.mock.calls[0][0]).toBe("aws_rds_status");
  });

  it("returns healthy response with all required fields", async () => {
    const { server, getHandler } = createMockServer();
    const ctx = createMockContext({ baseUrl: "https://api.honcho.dev" });
    register(server as never, ctx);

    const healthPayload = {
      status: "ok",
      auth_method: "iam",
      rds_hostname: "mydb.rds.amazonaws.com",
      rds_port: 5432,
      aws_region: "us-east-1",
    };

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(healthPayload),
      }),
    );

    const result = await getHandler()();
    expect(result.isError).toBeUndefined();

    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.auth_method).toBe("iam");
    expect(parsed.rds_hostname).toBe("mydb.rds.amazonaws.com");
    expect(parsed.rds_port).toBe(5432);
    expect(parsed.aws_region).toBe("us-east-1");
    expect(parsed.connection_healthy).toBe(true);
    expect(parsed.error).toBeNull();
  });

  it("returns error result when health check HTTP fails", async () => {
    const { server, getHandler } = createMockServer();
    const ctx = createMockContext();
    register(server as never, ctx);

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        statusText: "Service Unavailable",
      }),
    );

    const result = await getHandler()();
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("503");
    expect(result.content[0].text).toContain("Service Unavailable");
  });

  it("returns error result when fetch throws (network error)", async () => {
    const { server, getHandler } = createMockServer();
    const ctx = createMockContext();
    register(server as never, ctx);

    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Network unreachable")),
    );

    const result = await getHandler()();
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("Network unreachable");
  });

  it("handles password auth_method in healthy response", async () => {
    const { server, getHandler } = createMockServer();
    const ctx = createMockContext();
    register(server as never, ctx);

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            status: "ok",
            auth_method: "password",
          }),
      }),
    );

    const result = await getHandler()();
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.auth_method).toBe("password");
    expect(parsed.connection_healthy).toBe(true);
    expect(parsed.rds_hostname).toBeNull();
    expect(parsed.rds_port).toBeNull();
    expect(parsed.aws_region).toBeNull();
  });

  it("reports connection_healthy=false when status is not ok", async () => {
    const { server, getHandler } = createMockServer();
    const ctx = createMockContext();
    register(server as never, ctx);

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            status: "degraded",
            auth_method: "iam",
            rds_hostname: "mydb.rds.amazonaws.com",
            rds_port: 5432,
            aws_region: "us-east-1",
          }),
      }),
    );

    const result = await getHandler()();
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.connection_healthy).toBe(false);
  });
});
