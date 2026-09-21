import { Honcho } from "@honcho-ai/sdk";
import pkg from "../package.json";

export const HEADER_HOST = "X-Honcho-Host";
export const HEADER_PLUGIN = "X-Honcho-Plugin";

/** Honcho's API caps client header values at 256 chars; match it. */
const MAX_PLUGIN_LEN = 256;

const HOST_VALUE = `honcho-mcp/${
  typeof pkg.version === "string" && pkg.version ? pkg.version : "unknown"
}`;

/**
 * Identity headers for every Honcho API request made on behalf of one MCP
 * request. Host names this server; Plugin is the caller's `User-Agent`,
 * verbatim, so the API sees whatever the harness calls itself.
 */
export function identityHeaders(
  userAgent?: string | null,
): Record<string, string> {
  const headers: Record<string, string> = { [HEADER_HOST]: HOST_VALUE };
  const plugin = userAgent?.replace(/\s+/g, " ").trim().slice(0, MAX_PLUGIN_LEN);
  if (plugin) headers[HEADER_PLUGIN] = plugin;
  return headers;
}

export interface HonchoConfig {
  apiKey: string;
  baseUrl: string;
  /** From X-Honcho-Workspace-ID (HTTP) or HONCHO_WORKSPACE_ID (stdio). */
  workspaceId?: string;
}

export interface Env {
  HONCHO_API_URL?: string;
  ALERT_WEBHOOK_URL?: string;
}

export interface EnvConfig {
  HONCHO_API_KEY?: string;
  HONCHO_API_URL?: string;
  HONCHO_WORKSPACE_ID?: string;
}

/**
 * Parse configuration from request headers and Worker env bindings.
 * Throws only when the Authorization bearer token is missing/empty.
 *
 * The Honcho API URL is read from the `HONCHO_API_URL` env var when set,
 * allowing operators to run this Worker alongside a self-hosted Honcho
 * instance (see the "Self-Hosted Honcho" section in README.md). It is
 * intentionally not exposed as a request header: routing public requests
 * to an internal URL would be a latency and security regression.
 *
 * Optional `X-Honcho-Workspace-ID` becomes the default `workspace_id` on
 * tools. If the header is omitted, each tool call must pass `workspace_id`.
 */
export function parseConfig(request: Request, env: Env = {}): HonchoConfig {
  const authHeader = request.headers.get("Authorization");
  const bearerMatch = authHeader?.trim().match(/^Bearer\s+(.*)$/i);
  if (!bearerMatch) {
    throw new Error(
      "Missing Authorization header. Provide 'Authorization: Bearer <your-honcho-key>'.",
    );
  }
  const apiKey = bearerMatch[1].trim();
  if (!apiKey) {
    throw new Error("Authorization header is empty after 'Bearer '.");
  }

  const workspaceId =
    request.headers.get("X-Honcho-Workspace-ID")?.trim() || undefined;

  return {
    apiKey,
    baseUrl: env.HONCHO_API_URL?.trim() || "https://api.honcho.dev",
    workspaceId,
  };
}

/** Parse configuration from process env. */
export function parseEnvConfig(env: EnvConfig): HonchoConfig {
  const apiKey = env.HONCHO_API_KEY?.trim();
  if (!apiKey) {
    throw new Error(
      "Missing HONCHO_API_KEY. Set HONCHO_API_KEY to your Honcho API key.",
    );
  }
  return {
    apiKey,
    baseUrl: env.HONCHO_API_URL?.trim() || "https://api.honcho.dev",
    workspaceId: env.HONCHO_WORKSPACE_ID?.trim() || undefined,
  };
}

export const MISSING_WORKSPACE_ID_MESSAGE =
  "Missing workspace_id. Pass workspace_id on the next tool call, or set X-Honcho-Workspace-ID (HTTP) / HONCHO_WORKSPACE_ID (stdio).";

export function resolveWorkspaceId(
  config: HonchoConfig,
  workspaceId?: string,
): string {
  const id = workspaceId?.trim() || config.workspaceId?.trim();
  if (!id) {
    throw new Error(MISSING_WORKSPACE_ID_MESSAGE);
  }
  return id;
}

export function createClient(
  config: HonchoConfig,
  workspaceId: string,
  headers: Record<string, string> = identityHeaders(),
): Honcho {
  return new Honcho({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
    workspaceId,
    defaultHeaders: headers,
  });
}

/** Client used only for credential-scoped ops (list workspaces). */
export function createUnscopedClient(
  config: HonchoConfig,
  headers: Record<string, string> = identityHeaders(),
): Honcho {
  return new Honcho({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
    defaultHeaders: headers,
  });
}

export function createClientFactory(
  config: HonchoConfig,
  headers: Record<string, string> = identityHeaders(),
): (workspaceId?: string) => Honcho {
  const cache = new Map<string, Honcho>();
  return (workspaceId?: string) => {
    const id = resolveWorkspaceId(config, workspaceId);
    let client = cache.get(id);
    if (!client) {
      client = createClient(config, id, headers);
      cache.set(id, client);
    }
    return client;
  };
}

/** Both Honcho clients for one MCP request, sharing one identity header set. */
export function honchoClients(
  config: HonchoConfig,
  headers: Record<string, string> = identityHeaders(),
): { clientFor: (workspaceId?: string) => Honcho; unscoped: Honcho } {
  return {
    clientFor: createClientFactory(config, headers),
    unscoped: createUnscopedClient(config, headers),
  };
}
