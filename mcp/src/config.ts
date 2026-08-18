import { Honcho } from "@honcho-ai/sdk";

export interface HonchoConfig {
  apiKey: string;
  baseUrl: string;
  /** From X-Honcho-Workspace-ID when set. */
  workspaceId?: string;
}

export interface Env {
  HONCHO_API_URL?: string;
  ALERT_WEBHOOK_URL?: string;
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

export const MISSING_WORKSPACE_ID_MESSAGE =
  "Missing workspace_id. Pass workspace_id on the next tool call, or set the X-Honcho-Workspace-ID header on the connection so it is used automatically.";

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
): Honcho {
  return new Honcho({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
    workspaceId,
  });
}

/** Client used only for credential-scoped ops (list workspaces). */
export function createUnscopedClient(config: HonchoConfig): Honcho {
  return new Honcho({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
  });
}

export function createClientFactory(
  config: HonchoConfig,
): (workspaceId?: string) => Honcho {
  const cache = new Map<string, Honcho>();
  return (workspaceId?: string) => {
    const id = resolveWorkspaceId(config, workspaceId);
    let client = cache.get(id);
    if (!client) {
      client = createClient(config, id);
      cache.set(id, client);
    }
    return client;
  };
}
