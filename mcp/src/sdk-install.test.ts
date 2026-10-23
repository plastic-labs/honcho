import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import * as sdk from "@honcho-ai/sdk";
import installed from "@honcho-ai/sdk/package.json";

// wrangler bundles whatever is in node_modules without a typecheck, so a
// checkout whose install predates a lockfile bump ships tools that call SDK
// methods the bundle lacks. A Worker built against SDK 2.2.0 failed
// workspace_chat ("client.chat is not a function"), the scope tools and
// get_conclusions, and silently dropped scope/sessions recall bounds.
test("installed @honcho-ai/sdk matches the lockfile", () => {
  const lock = readFileSync(new URL("../bun.lock", import.meta.url), "utf8");
  const pinned = lock.match(/"@honcho-ai\/sdk": \["@honcho-ai\/sdk@([^"]+)"/)?.[1];
  expect(pinned).toBeDefined();
  expect(installed.version).toBe(pinned!);
});

// The option-level gaps (scope / sessions on chat and get_representation)
// are only visible on the wire; scope-read-options.test.ts and
// chat-evidence.test.ts assert those.
test("installed @honcho-ai/sdk has the surface the tools call", () => {
  expect(typeof sdk.Honcho.prototype.chat).toBe("function"); // workspace_chat
  expect(typeof sdk.Honcho.prototype.scopes).toBe("function"); // list_scopes
  expect(typeof sdk.Scope).toBe("function"); // get_scope_status, get_scope_sessions, …

  const honcho = new sdk.Honcho({
    apiKey: "test-key",
    baseURL: "https://honcho.test",
    workspaceId: "test-workspace",
  });
  expect(typeof honcho.conclusions?.getMany).toBe("function"); // get_conclusions
  expect(typeof honcho.conclusions?.list).toBe("function"); // get_derived_conclusions
});
