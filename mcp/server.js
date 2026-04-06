#!/usr/bin/env node
/**
 * Simple HTTP MCP server that proxies to Honcho API.
 * Handles stdio JSON-RPC from Claude Code, forwards to Honcho via HTTP+SSE.
 */

const http = require("http");
const https = require("https");

const HONCHO_API_URL = process.env.HONCHO_API_URL || "http://honcho-api:8000/";
const HONCHO_API_KEY = process.env.HONCHO_API_KEY || "not-needed";
const HONCHO_USER = process.env.HONCHO_USER;
if (!HONCHO_USER) {
  console.error("HONCHO_USER environment variable is required");
  process.exit(1);
}

const HEALTH_PORT = parseInt(process.env.HEALTH_PORT || "8788", 10);
const HEALTH_HOST = process.env.HEALTH_HOST || "0.0.0.0";
const TIMEOUT_MS = 30000;

function postToHoncho(request) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(request);
    const url = new URL(HONCHO_API_URL);
    const isHttps = url.protocol === "https:";
    const httpModule = isHttps ? https : http;
    const options = {
      hostname: url.hostname,
      port: url.port || (isHttps ? 443 : 80),
      path: url.pathname,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
        Authorization: `Bearer ${HONCHO_API_KEY}`,
        "X-Honcho-User-Name": HONCHO_USER,
      },
    };

    const req = httpModule.request(options, (res) => {
      let data = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        data += chunk;
      });
      res.on("end", () => {
        const lines = data.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              resolve(JSON.parse(line.slice(6)));
              return;
            } catch (parseErr) {
              console.error(
                "Failed to parse SSE data:",
                line.slice(0, 100),
                parseErr.message,
              );
            }
          }
        }
        reject(new Error("No valid SSE data"));
      });
    });

    req.on("error", reject);
    req.setTimeout(TIMEOUT_MS, () => {
      req.destroy();
      reject(new Error(`Honcho request timed out after ${TIMEOUT_MS}ms`));
    });

    req.write(body);
    req.end();
  });
}

function handleRequest(req) {
  const method = req.method;
  const id = req.id;

  if (method === "initialize") {
    return {
      jsonrpc: "2.0",
      id,
      result: {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "honcho", version: "1.0.0" },
      },
    };
  }

  if (method === "tools/list") {
    return postToHoncho({
      jsonrpc: "2.0",
      method: "tools/list",
      params: {},
      id: 1,
    })
      .then((r) => ({
        jsonrpc: "2.0",
        id,
        result: { tools: r.result?.tools || [] },
      }))
      .catch((e) => ({
        jsonrpc: "2.0",
        id,
        error: { code: -32603, message: e.message },
      }));
  }

  if (method === "tools/call") {
    const { name, arguments: args } = req.params || {};
    return postToHoncho({
      jsonrpc: "2.0",
      method: "tools/call",
      params: { name, arguments: args },
      id: 1,
    })
      .then((r) => ({ jsonrpc: "2.0", id, result: r.result }))
      .catch((e) => ({
        jsonrpc: "2.0",
        id,
        error: { code: -32603, message: e.message },
      }));
  }

  return {
    jsonrpc: "2.0",
    id,
    error: { code: -32601, message: "Method not found" },
  };
}

// HTTP server for healthchecks
const healthServer = http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ status: "ok" }));
});
healthServer.listen(HEALTH_PORT, HEALTH_HOST, () => {
  console.log(`Honcho MCP server listening on ${HEALTH_HOST}:${HEALTH_PORT}`);
});

// Process JSON-RPC from stdin
let buffer = "";
let processingQueue = Promise.resolve();
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  processingQueue = processingQueue.then(async () => {
    buffer += chunk;
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const req = JSON.parse(line);
        const response = await Promise.resolve(handleRequest(req));
        process.stdout.write(JSON.stringify(response) + "\n");
      } catch (e) {
        process.stdout.write(
          JSON.stringify({
            jsonrpc: "2.0",
            id: null,
            error: { code: -32700, message: e.message },
          }) + "\n",
        );
      }
    }
  });
});

process.stdin.on("end", () => process.exit(0));
