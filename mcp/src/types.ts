import type { Honcho, Message, Summary, SessionSummaries } from "@honcho-ai/sdk";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import type { HonchoConfig } from "./config.js";

export interface ToolContext {
  honcho: Honcho;
  config: HonchoConfig;
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
