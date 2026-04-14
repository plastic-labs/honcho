/**
 * Property-based tests for the aws_rds_status MCP tool.
 *
 * Uses fast-check to verify correctness properties across randomized inputs.
 * Each test runs a minimum of 100 iterations.
 *
 * Feature: aws-mcp-postgres
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import fc from "fast-check";
import { register } from "./aws-status.js";
import type { ToolContext } from "../types.js";
import type { HonchoConfig } from "../config.js";

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

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
/* Arbitraries                                                         */
/* ------------------------------------------------------------------ */

/** Arbitrary for auth_method values */
const arbAuthMethod = fc.constantFrom("password", "iam");

/** Arbitrary for nullable hostname strings */
const arbHostname = fc.oneof(
  fc.constant(null),
  fc.stringMatching(/^[a-z][a-z0-9\-.]{1,40}\.rds\.amazonaws\.com$/),
);

/** Arbitrary for nullable port numbers */
const arbPort = fc.oneof(fc.constant(null), fc.integer({ min: 1, max: 65535 }));

/** Arbitrary for nullable region strings */
const arbRegion = fc.oneof(
  fc.constant(null),
  fc.stringMatching(/^[a-z]{2}-[a-z]+-[0-9]$/),
);

/** Arbitrary for health status */
const arbStatus = fc.constantFrom("ok", "degraded", "error", "unknown");

/** Arbitrary for a complete health response */
const arbHealthResponse = fc.record({
  status: arbStatus,
  auth_method: arbAuthMethod,
  rds_hostname: arbHostname,
  rds_port: arbPort,
  aws_region: arbRegion,
});

/** Arbitrary for error messages */
const arbErrorMessage = fc.string({ minLength: 1, maxLength: 100 }).filter(
  (s) => s.trim().length > 0,
);

/** Arbitrary for HTTP error status codes */
const arbHttpErrorStatus = fc.integer({ min: 400, max: 599 });

/** Arbitrary for HTTP status text */
const arbStatusText = fc.constantFrom(
  "Bad Request",
  "Unauthorized",
  "Forbidden",
  "Not Found",
  "Internal Server Error",
  "Service Unavailable",
  "Gateway Timeout",
);

/* ------------------------------------------------------------------ */
/* Property 8: MCP status tool returns all required fields             */
/* Feature: aws-mcp-postgres, Property 8: MCP status tool returns all  */
/* required fields                                                     */
/* ------------------------------------------------------------------ */

describe("Property 8: MCP status tool returns all required fields", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("should return all required fields for any health response", async () => {
    // Feature: aws-mcp-postgres, Property 8: MCP status tool returns all required fields
    // **Validates: Requirements 3.1**
    await fc.assert(
      fc.asyncProperty(arbHealthResponse, async (healthPayload) => {
        const { server, getHandler } = createMockServer();
        const ctx = createMockContext();
        register(server as never, ctx);

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

        // All required fields must be present
        expect(parsed).toHaveProperty("auth_method");
        expect(parsed).toHaveProperty("rds_hostname");
        expect(parsed).toHaveProperty("rds_port");
        expect(parsed).toHaveProperty("aws_region");
        expect(parsed).toHaveProperty("connection_healthy");
        expect(parsed).toHaveProperty("error");

        // connection_healthy should be boolean
        expect(typeof parsed.connection_healthy).toBe("boolean");
        expect(parsed.connection_healthy).toBe(healthPayload.status === "ok");

        // auth_method should match input (or null if not provided)
        expect(parsed.auth_method).toBe(healthPayload.auth_method ?? null);

        vi.unstubAllGlobals();
      }),
      { numRuns: 100 },
    );
  });
});

/* ------------------------------------------------------------------ */
/* Property 9: MCP status tool error includes failure reason           */
/* Feature: aws-mcp-postgres, Property 9: MCP status tool error        */
/* includes failure reason                                             */
/* ------------------------------------------------------------------ */

describe("Property 9: MCP status tool error includes failure reason", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("should include failure reason for HTTP errors", async () => {
    // Feature: aws-mcp-postgres, Property 9: MCP status tool error includes failure reason
    // **Validates: Requirements 3.3**
    await fc.assert(
      fc.asyncProperty(
        arbHttpErrorStatus,
        arbStatusText,
        async (statusCode, statusText) => {
          const { server, getHandler } = createMockServer();
          const ctx = createMockContext();
          register(server as never, ctx);

          vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue({
              ok: false,
              status: statusCode,
              statusText: statusText,
            }),
          );

          const result = await getHandler()();
          expect(result.isError).toBe(true);

          const errorText = result.content[0].text;
          // Error should include the HTTP status code
          expect(errorText).toContain(String(statusCode));
          // Error should include the status text
          expect(errorText).toContain(statusText);

          vi.unstubAllGlobals();
        },
      ),
      { numRuns: 100 },
    );
  });

  it("should include failure reason for network errors", async () => {
    // Feature: aws-mcp-postgres, Property 9: MCP status tool error includes failure reason
    // **Validates: Requirements 3.3**
    await fc.assert(
      fc.asyncProperty(arbErrorMessage, async (errorMsg) => {
        const { server, getHandler } = createMockServer();
        const ctx = createMockContext();
        register(server as never, ctx);

        vi.stubGlobal(
          "fetch",
          vi.fn().mockRejectedValue(new Error(errorMsg)),
        );

        const result = await getHandler()();
        expect(result.isError).toBe(true);

        const errorText = result.content[0].text;
        // Error should include the original error message
        expect(errorText).toContain(errorMsg);

        vi.unstubAllGlobals();
      }),
      { numRuns: 100 },
    );
  });
});
