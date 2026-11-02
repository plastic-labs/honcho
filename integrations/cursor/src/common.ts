/**
 * Shared plumbing for the three Cursor hooks: config, Honcho client, naming,
 * stdin, output. Everything fails open: a hook that cannot run exits 0 with
 * nothing on stdout, so an install without `~/.honcho/config.json` behaves
 * exactly like the skills-only plugin.
 */
import { basename } from "node:path";
import { Honcho } from "@honcho-ai/sdk";
import { loadConfig, telemetryHeaders, type ResolvedConfig } from "@honcho-ai/harness-plugin-core";
import manifest from "../.cursor-plugin/plugin.json";

export const HOST = "cursor";
export const AI_PEER = "cursor";
export const PLUGIN_NAME = "cursor-honcho";
export const PLUGIN_VERSION: string = manifest.version;

/** Fields Cursor sends on stdin; every one is optional so a schema change never throws. */
export interface HookInput {
  session_id?: string;
  conversation_id?: string;
  workspace_roots?: string[];
  cwd?: string;
  prompt?: string;
  text?: string;
  model?: string;
  cursor_version?: string;
  is_background_agent?: boolean;
  hook_event_name?: string;
}

export interface HookContext {
  config: ResolvedConfig;
  honcho: Honcho;
  userPeerId: string;
  sessionId: string;
}

export async function readInput(): Promise<HookInput> {
  try {
    const raw = await Bun.stdin.text();
    return raw.trim() ? (JSON.parse(raw) as HookInput) : {};
  } catch {
    return {};
  }
}

/** Honcho ids allow only letters, digits, `_` and `-`. */
export function sanitizeId(value: string, fallback = "default"): string {
  const cleaned = value
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 100);
  return cleaned || fallback;
}

/**
 * `owner-repo` from the project's git `origin` remote, so two clones of one
 * repo share a session and two projects with the same folder name do not.
 * Undefined when there is no git, no remote, or the URL has no owner/repo.
 */
export function originSlug(root: string): string | undefined {
  try {
    const result = Bun.spawnSync(["git", "-C", root, "remote", "get-url", "origin"], {
      stdout: "pipe",
      stderr: "ignore",
      timeout: 1_500,
    });
    if (result.exitCode !== 0) return undefined;
    const match = result.stdout.toString().trim().match(/[:/]([^/:]+)\/([^/]+?)(?:\.git)?\/?$/);
    return match ? `${sanitizeId(match[1])}-${sanitizeId(match[2])}` : undefined;
  } catch {
    return undefined;
  }
}

/** One session per project, matching the honcho-memory skill: `cursor-<owner>-<repo>`, else `cursor-<folder>`. */
export function sessionIdFor(input: HookInput): string {
  const root = input.workspace_roots?.[0] || input.cwd || process.env.CURSOR_PROJECT_DIR || process.cwd();
  return `cursor-${originSlug(root) ?? sanitizeId(basename(root), "project")}`;
}

export function log(hook: string, message: string): void {
  if (process.env.HONCHO_HOOK_DEBUG) console.error(`[cursor-honcho ${hook}] ${message}`);
}

/**
 * Resolve config from `~/.honcho/config.json` (host block `cursor`) and the
 * HONCHO_* env. Returns null when the plugin is disabled or has no credential;
 * callers then exit silently so OAuth-only installs keep the skill-driven loop.
 */
export function setup(input: HookInput, hook: string): HookContext | null {
  const config = loadConfig({ host: HOST });
  const token = config.apiKey ?? config.auth.oauth?.accessToken;
  if (!config.enabled || !token) {
    log(hook, config.enabled ? "no credential in config; skipping" : "disabled in config; skipping");
    return null;
  }
  const honcho = new Honcho({
    apiKey: token,
    baseURL: config.baseUrl,
    workspaceId: config.workspace,
    timeout: config.timeoutMs,
    defaultHeaders: telemetryHeaders({
      host: HOST,
      hostVersion: input.cursor_version,
      plugin: PLUGIN_NAME,
      pluginVersion: PLUGIN_VERSION,
      model: input.model,
    }),
  });
  return {
    config,
    honcho,
    userPeerId: sanitizeId(config.peerName, "user"),
    sessionId: sessionIdFor(input),
  };
}

export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T | undefined> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(undefined), ms);
    promise.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      () => {
        clearTimeout(timer);
        resolve(undefined);
      },
    );
  });
}

export function messageMetadata(hook: string, input: HookInput): Record<string, unknown> {
  const meta: Record<string, unknown> = { source: PLUGIN_NAME, hook, host: HOST };
  if (input.conversation_id) meta.conversation_id = input.conversation_id;
  if (input.model) meta.model = input.model;
  if (input.cursor_version) meta.cursor_version = input.cursor_version;
  return meta;
}

/** Add the user and `cursor` peers to the session. Idempotent; errors are the caller's. */
export async function ensurePeers(ctx: HookContext): Promise<void> {
  const session = await ctx.honcho.session(ctx.sessionId);
  await session.addPeers([
    [ctx.userPeerId, { observeMe: true, observeOthers: false }],
    [AI_PEER, { observeMe: false, observeOthers: false }],
  ]);
}

/** Record one message; if the session or peers are missing, create them and retry once. */
export async function record(ctx: HookContext, peerId: string, content: string, metadata: Record<string, unknown>): Promise<void> {
  const session = await ctx.honcho.session(ctx.sessionId);
  const peer = await ctx.honcho.peer(peerId);
  const message = peer.message(content, { metadata });
  try {
    await session.addMessages([message]);
  } catch (first) {
    log("record", `first attempt failed, adding peers and retrying: ${String(first)}`);
    await ensurePeers(ctx);
    await session.addMessages([message]);
  }
}

export function emit(output: unknown): void {
  process.stdout.write(JSON.stringify(output) + "\n");
}
