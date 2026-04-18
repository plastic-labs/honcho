import { Honcho } from "@honcho-ai/sdk";

export interface HonchoConfig {
  apiKey: string;
  userName: string;
  assistantName: string;
  baseUrl: string;
  workspaceId: string;
}

export interface Env {
  HONCHO_API_URL?: string;
}

/**
 * Parse configuration from request headers and Worker env bindings.
 * Throws on missing required fields so callers get clear errors.
 *
 * The Honcho API URL is read from the `HONCHO_API_URL` env var when set,
 * allowing operators to run this Worker alongside a self-hosted Honcho
 * instance (see the "Self-Hosted Honcho" section in README.md). It is
 * intentionally not exposed as a request header: routing public requests
 * to an internal URL would be a latency and security regression.
 */
export function parseConfig(request: Request, env: Env = {}): HonchoConfig {
  const authHeader = request.headers.get("Authorization");
  const trimmedAuthHeader = authHeader?.trim();
  if (!trimmedAuthHeader?.startsWith("Bearer ")) {
    throw new Error(
      "Missing Authorization header. Provide 'Authorization: Bearer <your-honcho-key>'.",
    );
  }
  const apiKey = trimmedAuthHeader.substring(7).trim();
  if (!apiKey) {
    throw new Error("Authorization header is empty after 'Bearer '.");
  }

  const rawUserName = request.headers.get("X-Honcho-User-Name");
  const userName = rawUserName?.trim();
  if (!userName) {
    throw new Error(
      "Missing X-Honcho-User-Name header. Provide 'X-Honcho-User-Name: <your-name>'.",
    );
  }

  return {
    apiKey,
    userName,
    assistantName: request.headers.get("X-Honcho-Assistant-Name")?.trim() || "Assistant",
    baseUrl: env.HONCHO_API_URL?.trim() || "https://api.honcho.dev",
    workspaceId: request.headers.get("X-Honcho-Workspace-ID")?.trim() || "default",
  };
}

export function createClient(config: HonchoConfig): Honcho {
  return new Honcho({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
    workspaceId: config.workspaceId,
  });
}
