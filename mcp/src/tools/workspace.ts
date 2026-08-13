import { z } from "zod";
import {
  BadRequestError,
  HonchoError,
  UnprocessableEntityError,
  type PageResponse,
  type WorkspaceResponse,
} from "@honcho-ai/sdk";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { resolveWorkspaceId } from "../config.js";
import {
  textResult,
  errorResult,
  formatMessages,
  workspaceIdSchema,
} from "../types.js";

function withAdminKeyHint(prefix: string, e: unknown): string {
  const message = e instanceof Error ? e.message : String(e);
  const denied =
    e instanceof HonchoError && (e.status === 401 || e.status === 403);
  if (!denied) return `${prefix}: ${message}`;
  return `${prefix}: ${message}. This operation is only possible with an admin API key.`;
}

function formatWorkspace(workspace: WorkspaceResponse) {
  return {
    id: workspace.id,
    metadata: workspace.metadata ?? {},
    created_at: workspace.created_at,
  };
}

export function register(server: McpServer, ctx: ToolContext) {
  // ── inspect_workspace ───────────────────────────────────────────────
  server.registerTool(
    "inspect_workspace",
    {
      description: [
        "Inspect a workspace at a glance.",
        "Aggregates workspace metadata, configuration, peer IDs, and session IDs.",
        "Returns the first page of peers/sessions with total counts.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
      },
    },
    async ({ workspace_id }) => {
      try {
        const honcho = ctx.clientFor(workspace_id);
        const [metadata, configuration, peerPage, sessionPage] =
          await Promise.all([
            honcho.getMetadata(),
            honcho.getConfiguration(),
            honcho.peers(),
            honcho.sessions(),
          ]);

        return textResult({
          workspace_id: honcho.workspaceId,
          metadata,
          configuration,
          peer_count: peerPage.total,
          peers: peerPage.items.map((p) => ({ id: p.id })),
          session_count: sessionPage.total,
          sessions: sessionPage.items.map((s) => ({ id: s.id })),
        });
      } catch (e) {
        return errorResult(
          `Failed to inspect workspace: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── list_workspaces ─────────────────────────────────────────────────
  server.registerTool(
    "list_workspaces",
    {
      description: [
        "List workspaces accessible to the current credentials (paginated).",
        "Skip this if the connection already set X-Honcho-Workspace-ID — that header is the workspace; omit workspace_id on other tools.",
        "Use this only when the header is unset and you don't already know the workspace ID.",
        "Returns each workspace's id, metadata, and created_at. If none fit, call create_workspace.",
      ].join("\n"),
      inputSchema: {
        page: z
          .number()
          .int()
          .min(1)
          .optional()
          .describe("Page number (1-indexed)."),
        size: z
          .number()
          .int()
          .min(1)
          .max(100)
          .optional()
          .describe("Results per page (max 100)."),
      },
    },
    async ({ page, size }) => {
      try {
        const result = await ctx.unscoped.http.post<
          PageResponse<WorkspaceResponse>
        >("/v3/workspaces/list", {
          body: {},
          query: { page, size },
        });
        return textResult({
          workspaces: result.items.map(formatWorkspace),
          total: result.total,
          page: result.page,
          pages: result.pages,
        });
      } catch (e) {
        return errorResult(withAdminKeyHint("Failed to list workspaces", e));
      }
    },
  );

  // ── create_workspace ────────────────────────────────────────────────
  server.registerTool(
    "create_workspace",
    {
      description: [
        "Get or create a workspace with the given ID.",
        "Skip this if the connection already set X-Honcho-Workspace-ID — that header pins the workspace without a create call.",
        "Use this only when the header is unset and list_workspaces has no suitable workspace.",
        "Optional metadata helps future list_workspaces calls identify what the workspace is for.",
        "Returns the workspace id, metadata, and created_at.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        metadata: z
          .record(z.string(), z.unknown())
          .optional()
          .describe(
            "Optional key-value metadata to store on the workspace (e.g. project, purpose).",
          ),
      },
    },
    async ({ workspace_id, metadata }) => {
      try {
        const id = resolveWorkspaceId(ctx.config, workspace_id);
        const workspace = await ctx.unscoped.http.post<WorkspaceResponse>(
          "/v3/workspaces",
          {
            body: {
              id,
              metadata,
            },
          },
        );
        return textResult(formatWorkspace(workspace));
      } catch (e) {
        return errorResult(withAdminKeyHint("Failed to create workspace", e));
      }
    },
  );

  // ── search ────────────────────────────────────────────────────────
  server.registerTool(
    "search",
    {
      description: [
        "Semantic search across messages and, when peer_id is given, that peer's saved conclusions.",
        "Message scope is determined by which optional params are provided:",
        "- No scope params: search all messages in the workspace.",
        "- peer_id only: search messages authored by that peer across all sessions.",
        "- session_id only: search messages within that session.",
        "Conclusions require peer_id (self-conclusions are searched; conclusion IDs are usable with delete_conclusion).",
        "Returns {messages, conclusions}.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        query: z.string().describe("Search query."),
        peer_id: z
          .string()
          .optional()
          .describe("Optional: scope search to messages by this peer."),
        session_id: z
          .string()
          .optional()
          .describe("Optional: scope search to messages in this session."),
        message_limit: z
          .number()
          .optional()
          .describe("Optional: max message results (1-100, default 10)."),
        message_filters: z
          .record(z.string(), z.unknown())
          .optional()
          .describe(
            'Optional: filters for the message search, e.g. {"created_at": {"gte": "2026-01-01"}}. See https://honcho.dev/docs/v3/documentation/features/advanced/using-filters',
          ),
        conclusion_top_k: z
          .number()
          .optional()
          .describe("Optional: max conclusion results (default 10)."),
        conclusion_filters: z
          .record(z.string(), z.unknown())
          .optional()
          .describe(
            'Optional: filters for the conclusion search, e.g. {"level": ["deductive", "inductive"]} to only return conclusions derived during dreaming. Levels: explicit (extracted directly from messages), deductive, inductive, contradiction. The session_id param does not scope conclusions; use {"session_id": ...} here for that.',
          ),
      },
    },
    async ({
      workspace_id,
      query,
      peer_id,
      session_id,
      message_limit,
      message_filters,
      conclusion_top_k,
      conclusion_filters,
    }) => {
      try {
        const honcho = ctx.clientFor(workspace_id);
        const peer = peer_id ? await honcho.peer(peer_id) : null;
        const messageOptions = {
          filters: message_filters,
          limit: message_limit,
        };

        const searchMessages = async () => {
          if (session_id) {
            const session = await honcho.session(session_id);
            return session.search(query, messageOptions);
          }
          if (peer) {
            return peer.search(query, messageOptions);
          }
          return honcho.search(query, messageOptions);
        };

        // Conclusion search needs an (observer, observed) pair, so it only
        // runs when peer_id is given.
        const searchConclusions = async () => {
          if (!peer) {
            return [];
          }
          try {
            return await peer.conclusions.query(
              query,
              conclusion_top_k,
              undefined,
              conclusion_filters,
            );
          } catch (e) {
            if (
              conclusion_filters &&
              (e instanceof BadRequestError ||
                e instanceof UnprocessableEntityError)
            ) {
              throw e;
            }
            return [];
          }
        };

        const [messages, conclusions] = await Promise.all([
          searchMessages(),
          searchConclusions(),
        ]);
        return textResult({
          messages: formatMessages(messages),
          conclusions: conclusions.map((c) => ({
            id: c.id,
            content: c.content,
            level: c.level,
            created_at: c.createdAt,
          })),
        });
      } catch (e) {
        return errorResult(
          `Search failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_metadata ──────────────────────────────────────────────────
  server.registerTool(
    "get_metadata",
    {
      description: [
        "Get metadata for a resource. Scope is determined by which optional params are provided:",
        "- No scope params: get workspace metadata.",
        "- peer_id only: get peer metadata.",
        "- session_id only: get session metadata.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z
          .string()
          .optional()
          .describe("Optional: get metadata for this peer."),
        session_id: z
          .string()
          .optional()
          .describe("Optional: get metadata for this session."),
      },
    },
    async ({ workspace_id, peer_id, session_id }) => {
      try {
        const honcho = ctx.clientFor(workspace_id);
        let metadata;
        if (session_id) {
          const session = await honcho.session(session_id);
          metadata = await session.getMetadata();
        } else if (peer_id) {
          const peer = await honcho.peer(peer_id);
          metadata = await peer.getMetadata();
        } else {
          metadata = await honcho.getMetadata();
        }
        return textResult(metadata);
      } catch (e) {
        return errorResult(
          `Failed to get metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── set_metadata ──────────────────────────────────────────────────
  server.registerTool(
    "set_metadata",
    {
      description: [
        "Set metadata for a resource. Overwrites existing metadata.",
        "Scope is determined by which optional params are provided:",
        "- No scope params: set workspace metadata.",
        "- peer_id only: set peer metadata.",
        "- session_id only: set session metadata.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        metadata: z
          .record(z.string(), z.unknown())
          .describe("Key-value pairs to set as metadata."),
        peer_id: z
          .string()
          .optional()
          .describe("Optional: set metadata for this peer."),
        session_id: z
          .string()
          .optional()
          .describe("Optional: set metadata for this session."),
      },
    },
    async ({ workspace_id, metadata, peer_id, session_id }) => {
      try {
        const honcho = ctx.clientFor(workspace_id);
        if (session_id) {
          const session = await honcho.session(session_id);
          await session.setMetadata(metadata);
          return textResult("Session metadata set successfully");
        } else if (peer_id) {
          const peer = await honcho.peer(peer_id);
          await peer.setMetadata(metadata);
          return textResult("Peer metadata set successfully");
        } else {
          await honcho.setMetadata(metadata);
          return textResult("Workspace metadata set successfully");
        }
      } catch (e) {
        return errorResult(
          `Failed to set metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}
