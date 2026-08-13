import { z } from "zod";
import type { Honcho, Message, Summary, SessionSummaries } from "@honcho-ai/sdk";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import type { HonchoConfig } from "./config.js";

export interface ToolContext {
  config: HonchoConfig;
  /** Return a Honcho client scoped to the given workspace (or the header default). */
  clientFor: (workspaceId?: string) => Honcho;
  /** Client used only for credential-scoped ops (list workspaces). */
  unscoped: Honcho;
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
