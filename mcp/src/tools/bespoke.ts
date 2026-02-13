import { z } from "zod";
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
        "Returns a session_id you must pass to add_turn and get_personalization_insights for the rest of this conversation.",
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

        const sessionId = crypto.randomUUID();
        const session = await ctx.honcho.session(sessionId, { metadata: {} });

        await session.addPeers([
          userPeer,
          [assistantPeer, { observeMe: null, observeOthers: false }],
        ]);

        return textResult(sessionId);
      } catch (e) {
        return errorResult(
          `Failed to start conversation: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── add_turn ────────────────────────────────────────────────────────
  server.registerTool(
    "add_turn",
    {
      description: [
        "Record a user–assistant exchange in the current conversation.",
        "Call this after every assistant response so Honcho can learn from the conversation.",
        "Pass the full messages array containing both the user's message and your response.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("Session ID from start_conversation."),
        messages: z
          .array(
            z.object({
              role: z
                .enum(["user", "assistant"])
                .describe("Who sent the message."),
              content: z.string().describe("Message text."),
              metadata: z
                .record(z.string(), z.unknown())
                .optional()
                .describe("Optional metadata."),
            }),
          )
          .describe("Ordered list of messages in this turn."),
      },
    },
    async ({ session_id, messages }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const userPeer = await ctx.honcho.peer(ctx.config.userName);
        const assistantPeer = await ctx.honcho.peer(ctx.config.assistantName);

        const sessionMessages = messages.map((msg) => {
          const peer = msg.role === "user" ? userPeer : assistantPeer;
          return msg.metadata
            ? peer.message(msg.content, { metadata: msg.metadata })
            : peer.message(msg.content);
        });

        await session.addMessages(sessionMessages);
        return textResult("Turn added successfully");
      } catch (e) {
        return errorResult(
          `Failed to add turn: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_personalization_insights ────────────────────────────────────
  server.registerTool(
    "get_personalization_insights",
    {
      description: [
        "Ask Honcho a natural-language question about the user and get a personalized answer",
        "grounded in everything Honcho has learned across all of the user's conversations.",
        "Use this before responding when personalization would genuinely improve your response — it takes a few seconds.",
        "Returns a natural-language answer.",
      ].join("\n"),
      inputSchema: {
        session_id: z
          .string()
          .describe("Session ID from start_conversation, for context."),
        query: z
          .string()
          .describe(
            "Natural-language question about the user (e.g. 'What communication style does this user prefer?').",
          ),
        reasoning_level: z
          .enum(["minimal", "low", "medium", "high", "max"])
          .optional()
          .describe(
            "How much reasoning effort to use. Higher = more detailed but slower. Default: 'low'.",
          ),
      },
    },
    async ({ session_id, query, reasoning_level }) => {
      try {
        const userPeer = await ctx.honcho.peer(ctx.config.userName);
        const result = await userPeer.chat(query, {
          session: session_id,
          reasoningLevel: reasoning_level,
        });
        return textResult(result ?? "No personalization insights found.");
      } catch (e) {
        return errorResult(
          `Failed to get insights: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}
