import { existsSync } from "node:fs"
import { mkdir, readFile, writeFile } from "node:fs/promises"
import path from "node:path"
import { applyEdits, modify, parse, type ParseError } from "jsonc-parser/lib/esm/main.js"
import { PACKAGE_ID, userHomeDir } from "./core.js"
import { createHonchoClient } from "./honcho-client.js"

/** Kilo's global config directory: `$XDG_CONFIG_HOME/kilo`, else `~/.config/kilo`, on every platform. */
export const kiloConfigDir = () =>
  path.join(process.env.XDG_CONFIG_HOME?.trim() || path.join(userHomeDir(), ".config"), "kilo")

export type PluginConfigResult = { file: string; status: "added" | "present" | "unreadable" }

const listsPackage = (entries: unknown[]) =>
  entries.some((entry) => {
    const spec = Array.isArray(entry) ? entry[0] : entry
    return typeof spec === "string" && (spec === PACKAGE_ID || spec.startsWith(`${PACKAGE_ID}@`))
  })

// Same files and JSONC handling as `kilo plugin --global`, so comments in the user's config survive.
const addToConfigFile = async (dir: string, name: "opencode" | "tui"): Promise<PluginConfigResult> => {
  const file = [`${name}.json`, `${name}.jsonc`].map((base) => path.join(dir, base)).find(existsSync) ?? path.join(dir, `${name}.json`)
  const text = existsSync(file) ? await readFile(file, "utf-8") : ""
  const errors: ParseError[] = []
  const data = text.trim() ? parse(text, errors, { allowTrailingComma: true }) : {}
  if (errors.length > 0 || typeof data !== "object" || data === null || Array.isArray(data)) {
    return { file, status: "unreadable" }
  }
  const plugins: unknown[] = Array.isArray(data.plugin) ? data.plugin : []
  if (listsPackage(plugins)) return { file, status: "present" }
  const base = text.trim() ? text : "{}\n"
  const edits = modify(base, ["plugin"], [...plugins, PACKAGE_ID], { formattingOptions: { insertSpaces: true, tabSize: 2 } })
  await mkdir(dir, { recursive: true })
  await writeFile(file, applyEdits(base, edits), "utf-8")
  return { file, status: "added" }
}

/** Adds the plugin to Kilo's global server and TUI config, which the CLI, the IDE extensions and the desktop app all read. */
export const addKiloPlugin = async (dir = kiloConfigDir()) => [
  await addToConfigFile(dir, "opencode"),
  await addToConfigFile(dir, "tui"),
]

/** Throws when Honcho rejects the key. Gets or creates the workspace Kilo will write to. */
export const checkHonchoConnection = async (input: { apiKey: string; baseUrl: string; workspace: string }) => {
  await createHonchoClient({ apiKey: input.apiKey, baseUrl: input.baseUrl, workspaceId: input.workspace }).getMetadata()
}
