import type { Honcho, Message } from "@honcho-ai/sdk";
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

/** Serialize a Message[] to a plain JSON-safe array. */
export function formatMessages(messages: Message[]) {
  return messages.map((m) => ({
    id: m.id,
    content: m.content,
    peer_id: m.peerId,
    session_id: m.sessionId,
    metadata: m.metadata,
    created_at: m.createdAt,
  }));
}
