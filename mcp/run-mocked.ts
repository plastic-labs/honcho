// The "Bùa Ngải" Mock execution strategy requested by the User
import { plugin } from "bun";

plugin({
  name: "cloudflare-mock",
  setup(build) {
    // Intercept standard resolution for cloudflare:email
    build.onResolve({ filter: /^cloudflare:email$/ }, () => ({
      path: "mock-cloudflare-email",
      namespace: "mock",
    }));

    // Supply a dummy payload that successfully resolves but does nothing
    build.onLoad({ filter: /.*/, namespace: "mock" }, () => ({
      contents: "export default {}; export const EmailMessage = class {};",
      loader: "js",
    }));
  },
});

// We dynamically import the main index ONLY AFTER the plugin is registered
import("./src/index.ts").then((module) => {
  const port = process.env.HONCHO_MCP_PORT ? Number(process.env.HONCHO_MCP_PORT) : 8787;
  Bun.serve({
    fetch: module.default.fetch as any,
    port: port,
  });
  console.log(`[Honcho MCP] Native mock server booted up on port ${port}!`);
}).catch(err => {
  console.error("Failed to dynamically load MCP:", err);
});
