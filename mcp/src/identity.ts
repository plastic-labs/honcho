import pkg from "../package.json";

export const HEADER_HOST = "X-Honcho-Host";
export const HEADER_PLUGIN = "X-Honcho-Plugin";

const RUNTIME_UA = /^(node|undici|httpx|bun|cloudflare-workers|wrangler)(\/|$)/i;

function token(value: string): string {
  return value.replace(/[\r\n]+/g, " ").trim().replace(/[\s()/;]+/g, "-");
}

function product(name: string, version?: string): string {
  const n = token(name);
  const v = version?.trim() ? token(version) : "";
  return v ? `${n}/${v}` : n;
}

/** Shared header map. Pass the same object to every Honcho client; mutate Plugin in place. */
export function identityHeaders(plugin?: string): Record<string, string> {
  const version =
    typeof pkg.version === "string" && pkg.version ? pkg.version : "unknown";
  const plat = (globalThis as { process?: { platform?: string } }).process
    ?.platform;
  const host = product("honcho-mcp", version);
  const headers: Record<string, string> = {
    [HEADER_HOST]: plat ? `${host} (${token(plat)})` : host,
  };
  if (plugin) headers[HEADER_PLUGIN] = plugin;
  return headers;
}

/** `name/version` for X-Honcho-Plugin. Prefer MCP clientInfo; skip generic runtimes. */
export function pluginFromCaller(
  clientInfo?: { name?: string; version?: string } | null,
  userAgent?: string | null,
): string | undefined {
  const name = clientInfo?.name?.trim();
  if (name) return product(name, clientInfo?.version);
  const ua = userAgent?.trim();
  if (!ua || RUNTIME_UA.test(ua)) return undefined;
  const match = ua.match(/^([^\s/]+)(?:\/(\S+))?/);
  return match ? product(match[1], match[2]) : undefined;
}

export function pluginFromInitializeBody(body: unknown): string | undefined {
  const messages = Array.isArray(body) ? body : [body];
  for (const message of messages) {
    if (!message || typeof message !== "object") continue;
    const rec = message as {
      method?: unknown;
      params?: { clientInfo?: { name?: string; version?: string } };
    };
    if (rec.method === "initialize") {
      return pluginFromCaller(rec.params?.clientInfo);
    }
  }
}

/** After MCP initialize, write clientInfo onto the shared header map. */
export function attachClientInfo(
  server: {
    server: {
      oninitialized?: (() => void) | null;
      getClientVersion?: () =>
        | { name?: string; version?: string }
        | undefined;
    };
  },
  headers: Record<string, string>,
): void {
  const raw = server.server;
  const previous = raw.oninitialized;
  raw.oninitialized = () => {
    previous?.();
    const plugin = pluginFromCaller(raw.getClientVersion?.());
    if (plugin) headers[HEADER_PLUGIN] = plugin;
  };
}
