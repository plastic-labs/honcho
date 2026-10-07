import { rm } from "node:fs/promises"
import solidTransform from "@opentui/solid/bun-plugin"

await rm("dist", { recursive: true, force: true })

const result = await Bun.build({
  entrypoints: ["./src/index.ts", "./src/server.ts", "./src/tui.ts", "./src/import.ts"],
  outdir: "./dist",
  target: "node",
  // Kilo rewrites these imports to its own copies at load time. A bundled second Solid would not react to Kilo's state.
  external: ["@kilocode/plugin", "solid-js", "@opentui/solid", "@opentui/core"],
  plugins: [solidTransform],
})

if (!result.success) {
  for (const log of result.logs) console.error(log)
  process.exit(1)
}
