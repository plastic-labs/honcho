import { z } from "zod";
import type {
  Conclusion,
  Honcho,
  Message,
  Summary,
  SessionSummaries,
} from "@honcho-ai/sdk";
import { HonchoError } from "@honcho-ai/sdk";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import type { HonchoConfig } from "./config.js";

export interface ToolContext {
  config: HonchoConfig;
  /** Return a Honcho client scoped to the given workspace (or the header default). */
  clientFor: (workspaceId?: string) => Honcho;
  /** Client used only for credential-scoped ops (list workspaces). */
  unscoped: Honcho;
  /**
   * Called when the Honcho API rejects this connection's OAuth access token.
   * The HTTP entry points answer the request with 401 instead of a tool
   * error, which is what makes an MCP client refresh its token.
   */
  onUnauthorized?: () => void;
}

/**
 * Always optional at the schema layer so a missing value reaches clientFor,
 * which returns a clear error (header or workspace_id on the next call).
 */
export function workspaceIdSchema(ctx: ToolContext) {
  const fromHeader = ctx.config.workspaceId;
  const description = fromHeader
    ? `Workspace to operate in. The connection already set X-Honcho-Workspace-ID=${fromHeader}; omit this argument unless you need a different workspace.`
    : "Workspace to operate in. Prefer the client setting X-Honcho-Workspace-ID on the connection — then you can omit this on every call. Only pass it (or use list_workspaces / create_workspace) when the header is unset.";
  const field = z.string().optional().describe(description);
  return fromHeader ? field.default(fromHeader) : field;
}

export function textResult(
  data: string | object | unknown[],
): CallToolResult {
  const text = typeof data === "string" ? data : JSON.stringify(data);
  return { content: [{ type: "text", text }] };
}

export function errorResult(msg: string): CallToolResult {
  return { content: [{ type: "text", text: msg }], isError: true };
}

/** Hosted Honcho OAuth access tokens carry this prefix; API keys do not. */
export const OAUTH_ACCESS_TOKEN_PREFIX = "hch-at-";

/**
 * The hosted gateway's 401 detail for an expired or revoked access token.
 * Honcho itself also answers 401 when a live token may not use a route, and
 * refreshing would not fix that, so the detail is what tells them apart.
 */
const GATEWAY_TOKEN_REJECTED_DETAIL = "Invalid or expired access token";

export const TOKEN_REJECTED_MESSAGE =
  "Honcho rejected the access token (expired or revoked). Re-authenticate this MCP server.";

/**
 * True when the Honcho API refused the call because the OAuth access token
 * behind this connection is no longer valid. API keys are left alone: a 401
 * on a key is a bad key, or a scoped key on an admin route, and the client
 * holding it has nothing to refresh.
 */
export function isTokenRejected(e: unknown, apiKey: string): boolean {
  return (
    e instanceof HonchoError &&
    e.status === 401 &&
    e.message === GATEWAY_TOKEN_REJECTED_DETAIL &&
    apiKey.startsWith(OAUTH_ACCESS_TOKEN_PREFIX)
  );
}

/**
 * Tool error for a failed Honcho call. A rejected OAuth token is reported in
 * plain words and flagged on the context, so the HTTP layer can turn the
 * response into a 401 (see `invalidTokenResponse`).
 */
export function failedResult(
  ctx: ToolContext,
  prefix: string,
  e: unknown,
): CallToolResult {
  if (isTokenRejected(e, ctx.config.apiKey)) {
    ctx.onUnauthorized?.();
    return errorResult(`${prefix}: ${TOKEN_REJECTED_MESSAGE}`);
  }
  return errorResult(
    `${prefix}: ${e instanceof Error ? e.message : String(e)}`,
  );
}

/**
 * HTTP 401 for a rejected access token (RFC 6750 §3, RFC 9728 §5.1). The
 * `invalid_token` error is what tells an MCP client to use its refresh token
 * rather than start a new sign-in; `resource_metadata` keeps discovery intact.
 */
export function invalidTokenResponse(
  resourceMetadataUrl: string,
  headers: Record<string, string>,
): Response {
  return new Response(
    JSON.stringify({
      error: "invalid_token",
      error_description: TOKEN_REJECTED_MESSAGE,
    }),
    {
      status: 401,
      headers: {
        "Content-Type": "application/json",
        "WWW-Authenticate": `Bearer error="invalid_token", error_description="The access token is expired or revoked", resource_metadata="${resourceMetadataUrl}"`,
        ...headers,
      },
    },
  );
}

/** Serialize a Message to a plain JSON-safe object. */
export function formatMessage(message: Message) {
  return {
    id: message.id,
    content: message.content,
    peer_id: message.peerId,
    session_id: message.sessionId,
    metadata: message.metadata,
    created_at: message.createdAt,
  };
}

/**
 * Serialize a Conclusion to a plain JSON-safe object.
 *
 * `level`, `source_ids` and `times_derived` are the attribution fields the
 * server started returning in Honcho v3.2.0. `source_ids` is empty for
 * explicit conclusions, which are extracted from messages rather than
 * derived from other conclusions. `source_message_ids` (v3.4.0) is the
 * reverse: the messages an explicit conclusion cites, null for derived ones.
 * Read structurally so this compiles against SDKs that predate the field.
 */
export function formatConclusion(conclusion: Conclusion) {
  const { sourceMessageIds = null } = conclusion as Conclusion & {
    sourceMessageIds?: string[] | null;
  };
  return {
    id: conclusion.id,
    content: conclusion.content,
    level: conclusion.level,
    source_ids: conclusion.sourceIds,
    source_message_ids: sourceMessageIds,
    times_derived: conclusion.timesDerived,
    observer_id: conclusion.observerId,
    observed_id: conclusion.observedId,
    session_id: conclusion.sessionId,
    created_at: conclusion.createdAt,
  };
}

/** Serialize a Conclusion[] to a plain JSON-safe array. */
export function formatConclusions(conclusions: Conclusion[]) {
  return conclusions.map(formatConclusion);
}

/** Serialize a Summary to a plain JSON-safe object. */
export function formatSummary(summary: Summary) {
  return {
    content: summary.content,
    message_id: summary.messageId,
    summary_type: summary.summaryType,
    created_at: summary.createdAt,
    token_count: summary.tokenCount,
  };
}

/** Serialize SessionSummaries to a plain JSON-safe object. */
export function formatSessionSummaries(summaries: SessionSummaries) {
  return {
    session_id: summaries.sessionId,
    short_summary: summaries.shortSummary
      ? formatSummary(summaries.shortSummary)
      : null,
    long_summary: summaries.longSummary
      ? formatSummary(summaries.longSummary)
      : null,
  };
}

/** Serialize a Message[] to a plain JSON-safe array. */
export function formatMessages(messages: Message[]) {
  return messages.map(formatMessage);
}
