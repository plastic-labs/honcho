/**
 * afterAgentResponse: record the assistant's final text as the `cursor` peer,
 * with long code blocks collapsed and secrets redacted. Prints nothing.
 */
import { AI_PEER, log, messageMetadata, readInput, record, setup, withTimeout } from "./common.js";
import { collapseCodeBlocks, redactSecrets } from "./redact.js";

const UPLOAD_BUDGET_MS = 12_000;
const MIN_CHARS = 20;
const MAX_CHARS = 6_000;

export async function afterAgentResponse(): Promise<void> {
  const input = await readInput();
  const text = (input.text ?? "").trim();
  if (text.length < MIN_CHARS) return;
  const ctx = setup(input, "after-agent-response");
  if (!ctx) return;
  const content = redactSecrets(collapseCodeBlocks(text)).slice(0, MAX_CHARS);
  const done = await withTimeout(
    record(ctx, AI_PEER, content, messageMetadata("afterAgentResponse", input)).then(() => true),
    UPLOAD_BUDGET_MS,
  );
  log("after-agent-response", done ? `recorded ${content.length} chars` : "upload timed out or failed");
}
