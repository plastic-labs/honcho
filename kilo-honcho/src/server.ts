import { createHonchoRuntimePlugin } from "./index.js"
import { PACKAGE_ID } from "./honcho-client.js"

export const server = createHonchoRuntimePlugin({
  configPath: process.env.KILO_HONCHO_CONFIG_PATH,
})

// Kilo requires an exported id when the plugin loads from a file path; it matches the npm name.
const plugin = {
  id: PACKAGE_ID,
  server,
}

export default plugin
