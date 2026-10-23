/**
 * beforeSubmitPrompt: record the user's prompt verbatim (secrets redacted).
 * Cursor gives this hook no way to add model context, so recall stays with
 * sessionStart and the MCP tools. Always answers `continue: true`.
 */
import { emit, log, messageMetadata, readInput, record, setup, withTimeout } from "./common.js";
import { redactSecrets } from "./redact.js";

const UPLOAD_BUDGET_MS = 8_000;
const MAX_PROMPT_CHARS = 8_000;

export async function beforeSubmitPrompt(): Promise<void> {
  const input = await readInput();
  try {
    const prompt = (input.prompt ?? "").trim();
    const ctx = prompt ? setup(input, "before-submit-prompt") : null;
    if (ctx) {
      const content = redactSecrets(prompt).slice(0, MAX_PROMPT_CHARS);
      const done = await withTimeout(
        record(ctx, ctx.userPeerId, content, messageMetadata("beforeSubmitPrompt", input)).then(() => true),
        UPLOAD_BUDGET_MS,
      );
      log("before-submit-prompt", done ? `recorded ${content.length} chars` : "upload timed out or failed");
    }
  } finally {
    emit({ continue: true });
  }
}
