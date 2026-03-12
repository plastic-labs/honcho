import { Honcho } from "@honcho-ai/sdk";

export interface HonchoConfig {
  apiKey: string;
  userName: string;
  baseUrl: string;
  workspaceId: string;
  assistantName: string;
}

/**
 * Parse configuration from request headers.
 * Throws on missing required fields so callers get clear errors.
 */
export function parseConfig(request: Request): HonchoConfig {
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
    baseUrl:
      request.headers.get("X-Honcho-Base-URL")?.trim() ||
      "https://api.honcho.dev",
    workspaceId: request.headers.get("X-Honcho-Workspace-ID") || "default",
    assistantName:
      request.headers.get("X-Honcho-Assistant-Name") || "Assistant",
  };
}

export function createClient(config: HonchoConfig): Honcho {
  return new Honcho({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
    workspaceId: config.workspaceId,
  });
}
