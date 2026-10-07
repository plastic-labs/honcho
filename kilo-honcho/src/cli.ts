import { createInterface } from "node:readline/promises"
import { DEFAULT_SETTINGS, PACKAGE_ID, isLocalBaseUrl, peerCollisionError, resolveBaseUrl, resolvePeerName, resolveSessionPeerIds } from "./core.js"
import { addKiloPlugin, checkHonchoConnection } from "./setup.js"
import { otherToolWorkspaces, readGlobalSettings, readSharedConfig, resolveConfigPath, saveSettings } from "./tui/commands.js"

const HELP = `Usage: npx ${PACKAGE_ID} setup

Connects Kilo Code to Honcho. It asks for your Honcho API key, the name Honcho
should call you, and a workspace, then saves them to ~/.honcho/config.json and
adds the plugin to Kilo's global config. The Kilo CLI, the VS Code and JetBrains
extensions, and the Kilo desktop app all read those files.`

// Piped input is read once and answered line by line; a readline per question would drop the lines it buffered.
let pipedLines: string[] | undefined
const nextPipedLine = async () => {
  if (!pipedLines) {
    const chunks: Buffer[] = []
    for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk))
    pipedLines = Buffer.concat(chunks).toString("utf8").split(/\r?\n/)
  }
  return pipedLines.shift() ?? ""
}

const ask = async (question: string) => {
  if (!process.stdin.isTTY) {
    process.stdout.write(question)
    const line = await nextPipedLine()
    process.stdout.write("\n")
    return line.trim()
  }
  const rl = createInterface({ input: process.stdin, output: process.stdout })
  try {
    return (await rl.question(question)).trim()
  } finally {
    rl.close()
  }
}

// Reads a line without echoing it, so the key never shows on screen or in terminal scrollback.
const askHidden = (question: string) =>
  new Promise<string>((resolve, reject) => {
    const input = process.stdin
    if (!input.isTTY) {
      ask(question).then(resolve, reject)
      return
    }
    process.stdout.write(question)
    let value = ""
    const finish = (error?: Error) => {
      input.off("data", onData)
      input.setRawMode(false)
      input.pause()
      process.stdout.write("\n")
      if (error) reject(error)
      else resolve(value.trim())
    }
    const onData = (chunk: string) => {
      for (const char of chunk) {
        if (char === "\r" || char === "\n") return finish()
        if (char === "\u0003") return finish(new Error("Setup cancelled."))
        if (char === "\u007f" || char === "\b") value = value.slice(0, -1)
        else value += char
      }
    }
    input.setRawMode(true)
    input.setEncoding("utf8")
    input.resume()
    input.on("data", onData)
  })

const expandEnv = (value: string) => value.replace(/\$\{([A-Za-z0-9_]+)\}/g, (_, name: string) => process.env[name] ?? "")

const choose = async <T>(question: string, options: { label: string; value: T }[]) => {
  console.log(question)
  options.forEach((option, index) => console.log(`  ${index + 1}) ${option.label}`))
  while (true) {
    const answer = await ask("Choose [1]: ")
    const index = answer ? Number(answer) - 1 : 0
    if (Number.isInteger(index) && options[index]) return options[index].value
    console.log(`Enter a number from 1 to ${options.length}.`)
  }
}

export const runSetupCli = async () => {
  console.log("Honcho setup for Kilo Code\n")
  const current = await readGlobalSettings()
  const raw = (await readSharedConfig()) ?? {}

  const deployment = await choose("Where is your Honcho?", [
    { label: "Honcho Cloud (app.honcho.dev)", value: "cloud" as const },
    { label: "Self-hosted or local", value: "self" as const },
  ])
  const savedBaseUrl = resolveBaseUrl(raw)
  let baseUrl = DEFAULT_SETTINGS.baseUrl
  if (deployment === "self") {
    const fallback = savedBaseUrl && savedBaseUrl !== DEFAULT_SETTINGS.baseUrl ? savedBaseUrl : "http://127.0.0.1:8000"
    baseUrl = (await ask(`Honcho URL [${fallback}]: `)) || fallback
  }

  const savedKey = typeof current.apiKey === "string" ? current.apiKey.trim() : ""
  const envKey = process.env.HONCHO_API_KEY?.trim() || ""
  const keyHint = savedKey ? "Enter keeps the saved key" : envKey ? "Enter uses HONCHO_API_KEY" : isLocalBaseUrl(baseUrl) ? "optional for local" : "from app.honcho.dev"
  const enteredKey = await askHidden(`Honcho API key (hidden, ${keyHint}): `)
  const apiKey = enteredKey || (savedKey ? undefined : envKey || undefined)
  const effectiveKey = apiKey ?? expandEnv(savedKey)
  if (!effectiveKey && !isLocalBaseUrl(baseUrl)) {
    throw new Error("Honcho Cloud needs an API key. Get one at https://app.honcho.dev and run setup again.")
  }

  const sharedName = typeof current.peerName === "string" && current.peerName.trim() !== ""
  const suggestedName = resolvePeerName(current.peerName)
  const nameNote = sharedName ? " (your other Honcho tools use this name)" : ""
  const peerName = (await ask(`Name Honcho should call you${nameNote} [${suggestedName}]: `)) || suggestedName
  const ids = resolveSessionPeerIds(peerName, current.hosts?.kilo?.aiPeer || DEFAULT_SETTINGS.aiPeer)
  if (ids.userPeerId === ids.agentPeerId) throw new Error(peerCollisionError(ids.userPeerId))

  const currentWorkspace = current.hosts?.kilo?.workspace?.trim() || DEFAULT_SETTINGS.workspace
  const shared = otherToolWorkspaces(raw).filter(({ workspace }) => workspace !== DEFAULT_SETTINGS.workspace)
  let workspace = currentWorkspace
  if (shared.length > 0) {
    console.log("")
    workspace = await choose("Kilo can keep its own memory of you, or share the workspace another Honcho tool uses.", [
      { label: `Kilo's own (${DEFAULT_SETTINGS.workspace})`, value: DEFAULT_SETTINGS.workspace },
      ...shared.map(({ workspace: name, tool }) => ({ label: `Share with ${tool} (${name})`, value: name })),
    ])
  }

  if (effectiveKey || !isLocalBaseUrl(baseUrl)) {
    process.stdout.write("\nChecking the connection... ")
    await checkHonchoConnection({ apiKey: effectiveKey, baseUrl, workspace })
    console.log("ok")
  }

  const configPath = await saveSettings({ apiKey, baseUrl, peerName, hosts: { kilo: { workspace } } })
  console.log(`\nSaved ${configPath}`)
  console.log(`  Peer: ${ids.userPeerId}`)
  console.log(`  Workspace: ${workspace}`)

  for (const result of await addKiloPlugin()) {
    if (result.status === "added") console.log(`Added ${PACKAGE_ID} to ${result.file}`)
    if (result.status === "present") console.log(`${result.file} already loads ${PACKAGE_ID}`)
    if (result.status === "unreadable") console.log(`Could not read ${result.file}. Add "${PACKAGE_ID}" to its "plugin" array yourself.`)
  }
  console.log("\nRestart Kilo (the CLI, VS Code, JetBrains or the desktop app) to load Honcho.")
}

const main = async () => {
  const command = process.argv[2] ?? "setup"
  if (command === "--help" || command === "-h" || command === "help") {
    console.log(HELP)
    return
  }
  if (command !== "setup") {
    console.error(`Unknown command '${command}'.\n\n${HELP}`)
    process.exitCode = 1
    return
  }
  // The config path the plugin reads, shown so a KILO_HONCHO_CONFIG_PATH override is visible.
  if (process.env.KILO_HONCHO_CONFIG_PATH) console.log(`Using ${resolveConfigPath()}\n`)
  try {
    await runSetupCli()
  } catch (error) {
    console.error(`\n${error instanceof Error ? error.message : String(error)}`)
    process.exitCode = 1
  }
}

await main()
