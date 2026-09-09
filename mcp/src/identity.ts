import pkg from "../package.json";

export const HEADER_HOST = "X-Honcho-Host";
export const HEADER_PLUGIN = "X-Honcho-Plugin";

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

/** `name/version` for X-Honcho-Plugin from MCP `initialize` clientInfo. */
export function pluginFromClientInfo(
  clientInfo?: { name?: string; version?: string } | null,
): string | undefined {
  const name = clientInfo?.name?.trim();
  return name ? product(name, clientInfo?.version) : undefined;
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
      return pluginFromClientInfo(rec.params?.clientInfo);
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
    const plugin = pluginFromClientInfo(raw.getClientVersion?.());
    if (plugin) headers[HEADER_PLUGIN] = plugin;
  };
}
