import { execFile } from "node:child_process"

const MESSAGE = "Paste your Honcho API key from app.honcho.dev. Kilo saves it to ~/.honcho/config.json."

const KEY_WINDOW_TIMEOUT_MS = 2 * 60_000

// Over SSH a macOS or Windows window would open on that machine's own screen, which the remote user cannot see.
const overSsh = (env: NodeJS.ProcessEnv) => Boolean(env.SSH_CONNECTION || env.SSH_CLIENT || env.SSH_TTY)

/** The native password box for this platform, or null where none can open or the user would not see it. */
export const keyDialogCommand = (platform = process.platform, env = process.env): [string, string[]] | null => {
  if ((platform === "darwin" || platform === "win32") && overSsh(env)) return null
  if (platform === "darwin") {
    return [
      "osascript",
      [
        "-e",
        `text returned of (display dialog "${MESSAGE}" default answer "" with hidden answer with title "Honcho" buttons {"Cancel", "Save"} default button "Save" cancel button "Cancel")`,
      ],
    ]
  }
  if (platform === "win32") {
    // Windows PowerShell 5.1 shows Get-Credential as a window; PowerShell 7 would ask in a console that is not there.
    return [
      "powershell.exe",
      [
        "-NoProfile",
        "-Command",
        `$c = Get-Credential -UserName 'honcho' -Message '${MESSAGE}'; if ($c) { $c.GetNetworkCredential().Password }`,
      ],
    ]
  }
  if (!env.DISPLAY && !env.WAYLAND_DISPLAY) return null
  return ["zenity", ["--password", "--title=Honcho API key"]]
}

type Outcome = { key: string | null; missing: boolean }

const run = (command: string, args: string[]) =>
  new Promise<Outcome>((resolve) => {
    execFile(command, args, { timeout: KEY_WINDOW_TIMEOUT_MS, windowsHide: false }, (error, stdout) => {
      const missing = (error as NodeJS.ErrnoException | null)?.code === "ENOENT"
      resolve({ key: error ? null : stdout.trim() || null, missing })
    })
  })

/** Asks for the API key in a native window, so it never passes through a chat or a terminal. Null when cancelled or unavailable. */
export const promptForApiKey = async () => {
  const command = keyDialogCommand()
  if (!command) return null
  const outcome = await run(...command)
  // zenity is GNOME's; KDE desktops ship kdialog instead. A cancelled zenity box must not open a second one.
  if (outcome.missing && command[0] === "zenity") return (await run("kdialog", ["--password", MESSAGE, "--title", "Honcho"])).key
  return outcome.key
}
