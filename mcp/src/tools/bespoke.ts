import { z } from "zod";
import { nanoid } from "nanoid";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult } from "../types.js";

export function register(server: McpServer, ctx: ToolContext) {
  // ── start_conversation ──────────────────────────────────────────────
  server.registerTool(
    "start_conversation",
    {
      description: [
        "Start a new conversation for the current user.",
        "Call this once at the beginning of every new conversation.",
        "Returns session_id, user_peer_id, and assistant_peer_id.",
        "Use the peer IDs with add_messages_to_session to record turns.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const userPeer = await ctx.honcho.peer(ctx.config.userName);
        const assistantPeer = await ctx.honcho.peer(
          ctx.config.assistantName,
          { configuration: { observeMe: false } },
        );

        const sessionId = nanoid();
        const session = await ctx.honcho.session(sessionId, { metadata: {} });

        await session.addPeers([
          userPeer,
          [assistantPeer, { observeMe: null, observeOthers: false }],
        ]);

        return textResult({
          session_id: sessionId,
          user_peer_id: userPeer.id,
          assistant_peer_id: assistantPeer.id,
        });
      } catch (e) {
        return errorResult(
          `Failed to start conversation: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}
