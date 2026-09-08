import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  createClientFactory,
  createUnscopedClient,
  parseEnvConfig,
} from "./config.js";
import { createServer } from "./server.js";

declare const process: {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

try {
  const config = parseEnvConfig({
    HONCHO_API_KEY: process.env.HONCHO_API_KEY,
    HONCHO_API_URL: process.env.HONCHO_API_URL,
    HONCHO_WORKSPACE_ID: process.env.HONCHO_WORKSPACE_ID,
  });
  const server = createServer({
    config,
    clientFor: createClientFactory(config),
    unscoped: createUnscopedClient(config),
  });
  await server.connect(new StdioServerTransport());
} catch (e) {
  const message = e instanceof Error ? e.message : String(e);
  console.error(message);
  process.exit(1);
}
