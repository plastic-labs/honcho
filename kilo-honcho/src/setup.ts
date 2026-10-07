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

// Kilo's own order for its global config. `kilo plugin` still writes the opencode names, so an install there counts too.
const SERVER_CONFIG_FILES = ["kilo.jsonc", "kilo.json", "opencode.jsonc", "opencode.json", "config.json"]
const TUI_CONFIG_FILES = ["tui.json", "tui.jsonc"]

const readConfig = async (file: string) => {
  const text = existsSync(file) ? await readFile(file, "utf-8") : ""
  const errors: ParseError[] = []
  const data = text.trim() ? parse(text, errors, { allowTrailingComma: true }) : {}
  const valid = errors.length === 0 && typeof data === "object" && data !== null && !Array.isArray(data)
  return { text, data: valid ? (data as Record<string, unknown>) : null }
}

const loadsPackage = (data: Record<string, unknown> | null) => Array.isArray(data?.plugin) && listsPackage(data.plugin)

// Writes the first of the two Kilo-named files that exists, else creates the first, editing as JSONC so comments survive.
const addToConfigFile = async (dir: string, names: string[]): Promise<PluginConfigResult> => {
  const files = names.map((name) => path.join(dir, name))
  for (const file of files) {
    if (loadsPackage((await readConfig(file)).data)) return { file, status: "present" }
  }
  const file = files.slice(0, 2).find(existsSync) ?? files[0]
  const { text, data } = await readConfig(file)
  if (!data) return { file, status: "unreadable" }
  const plugins: unknown[] = Array.isArray(data.plugin) ? data.plugin : []
  const base = text.trim() ? text : "{}\n"
  const edits = modify(base, ["plugin"], [...plugins, PACKAGE_ID], { formattingOptions: { insertSpaces: true, tabSize: 2 } })
  await mkdir(dir, { recursive: true })
  await writeFile(file, applyEdits(base, edits), "utf-8")
  return { file, status: "added" }
}

/** Adds the plugin to Kilo's global server and TUI config, which the CLI, the IDE extensions and the desktop app all read. */
export const addKiloPlugin = async (dir = kiloConfigDir()) => [
  await addToConfigFile(dir, SERVER_CONFIG_FILES),
  await addToConfigFile(dir, TUI_CONFIG_FILES),
]

/** Throws when Honcho rejects the key. Gets or creates the workspace Kilo will write to. */
export const checkHonchoConnection = async (input: { apiKey: string; baseUrl: string; workspace: string }) => {
  await createHonchoClient({ apiKey: input.apiKey, baseUrl: input.baseUrl, workspaceId: input.workspace }).getMetadata()
}
