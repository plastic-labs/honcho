/**
 * sessionStart: inject what Honcho knows about the user into the
 * conversation's initial context, and make sure the session and peers exist.
 */
import { AI_PEER, emit, ensurePeers, log, readInput, setup, withTimeout } from "./common.js";

const RECALL_BUDGET_MS = 4_000;
const MAX_REPRESENTATION_CHARS = 3_000;
const MAX_SUMMARY_CHARS = 1_500;

function clip(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max).trimEnd()}…` : text;
}

export async function sessionStart(): Promise<void> {
  const input = await readInput();
  const ctx = setup(input, "session-start");
  if (!ctx) return;

  // Peers and session first, in the background of the recall reads.
  const peersReady = withTimeout(ensurePeers(ctx), RECALL_BUDGET_MS);

  const user = await ctx.honcho.peer(ctx.userPeerId);
  const session = await ctx.honcho.session(ctx.sessionId);
  const [context, summaries] = await Promise.all([
    withTimeout(user.context({ maxConclusions: 12 }), RECALL_BUDGET_MS),
    withTimeout(session.summaries(), RECALL_BUDGET_MS),
  ]);
  await peersReady;

  const parts: string[] = [];
  parts.push(
    [
      `Honcho memory for ${ctx.userPeerId} (workspace ${ctx.config.workspace}, session ${ctx.sessionId}).`,
      "Treat the notes below as background from past sessions; verify against the live code when they conflict.",
      "Automatic capture is on: this plugin's hooks record the user's prompts and your replies for this conversation, so do not call add_messages_to_session yourself.",
      "Use get_peer_context, search or chat on the honcho MCP server when more context would help, and create_conclusions when the user states a durable preference.",
    ].join("\n"),
  );

  const card = context?.peerCard?.filter(Boolean) ?? [];
  if (card.length) parts.push(`Peer card:\n${card.map((line) => `- ${line}`).join("\n")}`);
  const representation = context?.representation?.trim();
  if (representation) parts.push(`Representation:\n${clip(representation, MAX_REPRESENTATION_CHARS)}`);
  const summary = (summaries?.longSummary?.content ?? summaries?.shortSummary?.content ?? "").trim();
  if (summary) parts.push(`Last time in this project:\n${clip(summary, MAX_SUMMARY_CHARS)}`);
  if (parts.length === 1) parts.push("Nothing recorded for this user yet; memory starts with this conversation.");

  const status = card.length || representation || summary
    ? `Honcho: memory loaded for ${ctx.userPeerId} · ${card.length} card items · session ${ctx.sessionId}`
    : `Honcho: no memory yet for ${ctx.userPeerId} · capturing session ${ctx.sessionId}`;
  log("session-start", status);
  emit({ additional_context: `[${AI_PEER}/Honcho memory]\n\n${parts.join("\n\n")}`, user_message: status });
}
