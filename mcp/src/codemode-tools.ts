// What a Code Mode script can call: the read-only Honcho tools as plain async
// functions, plus the compact listing of them that goes in the tool description.
// Kept free of Worker-only imports so it runs under `bun test`.
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult, Tool } from "@modelcontextprotocol/sdk/types.js";

/** Most Honcho calls one script may make, so a loop cannot run away. */
export const MAX_CALLS_PER_SCRIPT = 50;

type Schema = {
  type?: string;
  enum?: unknown[];
  anyOf?: Schema[];
  items?: Schema;
  description?: string;
  properties?: Record<string, Schema>;
  required?: string[];
};

function typeOf(schema: Schema | undefined): string {
  if (!schema) return "unknown";
  if (schema.enum) return schema.enum.map((v) => JSON.stringify(v)).join(" | ");
  if (schema.anyOf) return [...new Set(schema.anyOf.map(typeOf))].join(" | ");
  if (schema.type === "integer") return "number";
  if (schema.type === "array") return `${typeOf(schema.items)}[]`;
  return schema.type ?? "unknown";
}

function firstSentence(text = ""): string {
  const sentence = text.replace(/\s+/g, " ").trim().split(/(?<=[.!?])\s/)[0];
  return sentence.length > 200 ? `${sentence.slice(0, 197)}...` : sentence;
}

/** One line per tool: its call signature and the first sentence of its description. */
export function signature(tool: Tool): string {
  const schema = tool.inputSchema as Schema;
  const required = new Set(schema.required ?? []);
  const params = Object.entries(schema.properties ?? {})
    .map(([name, prop]) => `${name}${required.has(name) ? "" : "?"}: ${typeOf(prop)}`)
    .join(", ");
  return `honcho.${tool.name}({ ${params} }) — ${firstSentence(tool.description)}`;
}

/** The full description and per-parameter notes that `signature` leaves out. */
export function describe(tool: Tool) {
  const schema = tool.inputSchema as Schema;
  const required = new Set(schema.required ?? []);
  return {
    name: tool.name,
    description: tool.description,
    parameters: Object.fromEntries(
      Object.entries(schema.properties ?? {}).map(([name, prop]) => [
        name,
        { type: typeOf(prop), required: required.has(name), description: prop.description },
      ]),
    ),
  };
}

/** A tool result as the plain value a script expects; a tool error throws. */
export function unwrap(result: CallToolResult): unknown {
  const text = result.content
    .map((block) => (block.type === "text" ? block.text : ""))
    .join("\n");
  if (result.isError) throw new Error(text || "Tool call failed");
  if (result.structuredContent != null) return result.structuredContent;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export interface ScriptApi {
  /** One line per callable function, for the tool description. */
  listing: string;
  /** A fresh set of functions for one script run, with its own call budget. */
  functions: () => Record<string, (args: unknown) => Promise<unknown>>;
}

/** Expose every tool registered on `server` to scripts, by in-memory MCP calls. */
export async function scriptApi(server: McpServer): Promise<ScriptApi> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  const client = new Client({ name: "honcho-codemode", version: "1.0.0" });
  await client.connect(clientTransport);
  const { tools } = await client.listTools();
  const byName = new Map(tools.map((tool) => [tool.name, tool]));

  return {
    listing: [
      ...tools.map(signature),
      "honcho.describe({ name: string }) — Full description and parameter notes for one function above.",
    ].join("\n"),
    functions: () => {
      let calls = 0;
      const fns: Record<string, (args: unknown) => Promise<unknown>> = {
        describe: async (args) => {
          const name = typeof args === "string" ? args : (args as { name?: string })?.name;
          const tool = name ? byName.get(name) : undefined;
          if (!tool) throw new Error(`Unknown function: ${name}. Known: ${[...byName.keys()].join(", ")}`);
          return describe(tool);
        },
      };
      for (const name of byName.keys()) {
        fns[name] = async (args) => {
          if (++calls > MAX_CALLS_PER_SCRIPT) {
            throw new Error(`A script may make at most ${MAX_CALLS_PER_SCRIPT} Honcho calls.`);
          }
          const result = await client.callTool({
            name,
            arguments: (args ?? {}) as Record<string, unknown>,
          });
          return unwrap(result as CallToolResult);
        };
      }
      return fns;
    },
  };
}
