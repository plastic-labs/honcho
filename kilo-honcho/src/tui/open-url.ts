import { spawn } from "node:child_process"

// rundll32 on Windows, because cmd's `start` would split the URL at each `&`.
const opener = (url: string): [string, string[]] =>
  process.platform === "darwin"
    ? ["open", [url]]
    : process.platform === "win32"
      ? ["rundll32", ["url.dll,FileProtocolHandler", url]]
      : ["xdg-open", [url]]

/** Opens a URL in the default browser, the way Kilo's own links do. Failures are ignored. */
export const openUrl = (url: string) => {
  const [command, args] = opener(url)
  try {
    spawn(command, args, { detached: true, stdio: "ignore" }).on("error", () => undefined).unref()
  } catch {
    // No browser opener on this machine.
  }
}
