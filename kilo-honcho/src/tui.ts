import type { TuiPluginModule } from "@kilocode/plugin/tui"
import { recallMessage, sidebarRows } from "./tui/activity-view.js"
import {
  modeEditableFieldPaths,
  normalizeSettings,
  offerSetup,
  readSharedConfig,
  resolveConfigPath,
  resolveSharedConfigField,
  runRecall,
  runSetup,
  saveSettings,
  settingsMessage,
  sharedConfigPresetOptions,
  statusMessage,
  validateCloudApiKey,
} from "./tui/commands.js"
import { buildCommands, deriveLiveStatus, dialogsFromApi, tui } from "./tui/kilo.js"
import { PACKAGE_ID } from "./honcho-client.js"


const plugin: TuiPluginModule & { id: string } = {
  id: PACKAGE_ID,
  tui,
}

export const __testing = {
  buildCommands,
  deriveLiveStatus,
  dialogsFromApi,
  normalizeSettings,
  modeEditableFieldPaths,
  offerSetup,
  readSharedConfig,
  recallMessage,
  resolveSharedConfigField,
  runRecall,
  runSetup,
  saveSettings,
  settingsMessage,
  sharedConfigPath: resolveConfigPath,
  sharedConfigPresetOptions,
  sidebarRows,
  statusMessage,
  validateCloudApiKey,
}

export default plugin
