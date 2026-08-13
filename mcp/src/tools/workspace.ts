import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import {
  HonchoError,
  type PageResponse,
  type WorkspaceResponse,
} from "@honcho-ai/sdk";
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
        "Semantic search across messages. Scope is determined by which optional params are provided:",
        "- No scope params: search all messages in the workspace.",
        "- peer_id only: search messages authored by that peer across all sessions.",
        "- session_id only: search messages within that session.",
        "Returns an array of matching messages with their content, peer, and session info.",
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
      },
    },
    async ({ workspace_id, query, peer_id, session_id }) => {
      try {
        const honcho = ctx.clientFor(workspace_id);
        let messages;
        if (session_id) {
          const session = await honcho.session(session_id);
          messages = await session.search(query);
        } else if (peer_id) {
          const peer = await honcho.peer(peer_id);
          messages = await peer.search(query);
        } else {
          messages = await honcho.search(query);
        }
        return textResult(formatMessages(messages));
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
