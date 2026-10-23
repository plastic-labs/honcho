/**
 * Single entry point for every hook. Cursor runs
 * `bun run ${CURSOR_PLUGIN_ROOT}/dist/hook.js <event>`; the event name picks
 * the handler. Any failure exits 0 so the agent is never blocked by memory.
 */
import { afterAgentResponse } from "./after-agent-response.js";
import { beforeSubmitPrompt } from "./before-submit-prompt.js";
import { log } from "./common.js";
import { sessionStart } from "./session-start.js";

const handlers: Record<string, () => Promise<void>> = {
  "session-start": sessionStart,
  "before-submit-prompt": beforeSubmitPrompt,
  "after-agent-response": afterAgentResponse,
};

const event = process.argv[2] ?? "";
const handler = handlers[event];
if (!handler) {
  console.error(`cursor-honcho: unknown hook "${event}" (expected one of ${Object.keys(handlers).join(", ")})`);
  process.exit(0);
}

handler()
  .catch((e) => log(event, `failed: ${e instanceof Error ? e.message : String(e)}`))
  .finally(() => process.exit(0));
