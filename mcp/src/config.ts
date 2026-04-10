import { Honcho } from "@honcho-ai/sdk";

export interface HonchoConfig {
  apiKey: string;
  userName: string;
  assistantName: string;
  baseUrl: string;
  workspaceId: string;
}

const DEFAULT_BASE_URL = "https://api.honcho.dev";

/**
 * Validate and return the base URL from the X-Honcho-Base-URL header.
 * Accepts only http/https URLs. Falls back to the default if the header is
 * absent or malformed — never propagates an invalid URL to the client.
 */
function parseBaseUrl(raw: string | null): string {
  const trimmed = raw?.trim();
  if (!trimmed) return DEFAULT_BASE_URL;
  try {
    const url = new URL(trimmed);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return DEFAULT_BASE_URL;
    }
    return trimmed;
  } catch {
    return DEFAULT_BASE_URL;
  }
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
    assistantName: request.headers.get("X-Honcho-Assistant-Name")?.trim() || "Assistant",
    baseUrl: parseBaseUrl(request.headers.get("X-Honcho-Base-URL")),
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
