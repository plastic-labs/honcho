import type { Honcho } from "@honcho-ai/sdk";
import pkg from "../package.json";

/** The MCP server is the host; the calling MCP client is the plugin. */
export const HOST_ID = "honcho-mcp";
export const HEADER_HOST = "X-Honcho-Host";
export const HEADER_PLUGIN = "X-Honcho-Plugin";

/** Known HTTP runtime defaults — not a caller. Omit Plugin rather than send these. */
const RUNTIME_UA = new Set([
  "node",
  "undici",
  "httpx",
  "bun",
  "cloudflare-workers",
  "wrangler",
]);

export interface CallerIdentity {
  plugin?: string;
  pluginVersion?: string;
}

export interface IdentitySlot {
  get(): CallerIdentity;
  set(caller: CallerIdentity): void;
  bind(client: Honcho): void;
}

export function hostVersion(): string {
  return typeof pkg.version === "string" && pkg.version ? pkg.version : "unknown";
}

function token(value: string): string {
  return value.replace(/[\r\n]+/g, " ").trim().replace(/[\s()/;]+/g, "-");
}

function product(name: string, version?: string): string {
  const n = token(name);
  const v = version ? token(version) : "";
  return v ? `${n}/${v}` : n;
}

function platform(): string | undefined {
  return (globalThis as { process?: { platform?: string } }).process?.platform;
}

export function defaultTelemetryHeaders(
  caller: CallerIdentity = {},
): Record<string, string> {
  const host = product(HOST_ID, hostVersion());
  const plat = platform();
  const headers: Record<string, string> = {
    [HEADER_HOST]: plat ? `${host} (${token(plat)})` : host,
  };
  if (caller.plugin) {
    headers[HEADER_PLUGIN] = product(caller.plugin, caller.pluginVersion);
  }
  return headers;
}

export function applyIdentity(client: Honcho, caller: CallerIdentity): void {
  const next = defaultTelemetryHeaders(caller);
  Object.assign(client.http.defaultHeaders, next);
  if (!next[HEADER_PLUGIN]) {
    delete client.http.defaultHeaders[HEADER_PLUGIN];
  }
}

export function createIdentity(initial: CallerIdentity = {}): IdentitySlot {
  let caller = initial;
  const clients: Honcho[] = [];
  return {
    get: () => caller,
    set: (next) => {
      caller = next;
      for (const client of clients) applyIdentity(client, caller);
    },
    bind: (client) => {
      clients.push(client);
    },
  };
}

export function callerFromClientInfo(
  info: { name?: string; version?: string } | null | undefined,
): CallerIdentity {
  const name = info?.name?.trim();
  if (!name) return {};
  const version = info?.version?.trim();
  return version ? { plugin: name, pluginVersion: version } : { plugin: name };
}

export function callerFromUserAgent(
  ua: string | null | undefined,
): CallerIdentity {
  if (!ua?.trim()) return {};
  const match = ua.trim().match(/^([^\s/]+)(?:\/(\S+))?/);
  if (!match) return {};
  const name = match[1];
  if (RUNTIME_UA.has(name.toLowerCase())) return {};
  return match[2]
    ? { plugin: name, pluginVersion: match[2] }
    : { plugin: name };
}

function clientInfoFromMcpBody(
  body: unknown,
): { name?: string; version?: string } | undefined {
  const messages = Array.isArray(body) ? body : [body];
  for (const message of messages) {
    if (!message || typeof message !== "object") continue;
    const record = message as { method?: unknown; params?: unknown };
    if (record.method !== "initialize") continue;
    const params = record.params;
    if (!params || typeof params !== "object") continue;
    const info = (params as { clientInfo?: unknown }).clientInfo;
    if (!info || typeof info !== "object") continue;
    const { name, version } = info as { name?: unknown; version?: unknown };
    return {
      name: typeof name === "string" ? name : undefined,
      version: typeof version === "string" ? version : undefined,
    };
  }
  return undefined;
}

export function resolveCallerIdentity(opts: {
  clientInfo?: { name?: string; version?: string } | null;
  userAgent?: string | null;
}): CallerIdentity {
  const fromInfo = callerFromClientInfo(opts.clientInfo);
  if (fromInfo.plugin) return fromInfo;
  return callerFromUserAgent(opts.userAgent);
}

export function identityFromRequest(
  request: Request,
  body?: unknown,
): IdentitySlot {
  return createIdentity(
    resolveCallerIdentity({
      clientInfo: body !== undefined ? clientInfoFromMcpBody(body) : undefined,
      userAgent: request.headers.get("User-Agent"),
    }),
  );
}

/** Peek at a clone so the original request body stays readable. */
export async function identityFromHttpRequest(
  request: Request,
): Promise<IdentitySlot> {
  let body: unknown;
  if (request.method === "POST") {
    try {
      body = await request.clone().json();
    } catch {
      body = undefined;
    }
  }
  return identityFromRequest(request, body);
}

/** After initialize, prefer MCP clientInfo over a User-Agent guess. */
export function bindServerClientInfo(
  server: {
    server: {
      oninitialized?: (() => void) | null;
      getClientVersion?: () => { name?: string; version?: string } | undefined;
    };
  },
  identity: IdentitySlot,
): void {
  const raw = server.server;
  const previous = raw.oninitialized;
  raw.oninitialized = () => {
    previous?.();
    const fromInfo = callerFromClientInfo(raw.getClientVersion?.());
    if (fromInfo.plugin) identity.set(fromInfo);
  };
}
