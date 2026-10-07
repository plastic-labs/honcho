import path from "node:path"
import type { TuiPlugin, TuiPluginApi } from "@kilocode/plugin/tui"
import { HOST_ID } from "../core.js"
import { transcriptSourceFromClient } from "../import.js"
import { COMMANDS, maybePromptObservationUpgrade, runGuarded } from "./commands.js"
import type { Dialogs, GlobalSettings, TuiSession } from "./dialogs.js"

const settle = <T>(resolve: (value: T) => void) => {
  let done = false
  return (value: T) => {
    if (done) return
    done = true
    resolve(value)
  }
}

/** Lift Kilo's callback dialogs (`replace` + onSelect/onConfirm) to the promise `Dialogs` port. */
export const dialogsFromApi = (api: TuiPluginApi): Dialogs => ({
  alert: ({ title, message }) =>
    new Promise((resolve) => {
      const done = settle(resolve)
      api.ui.dialog.replace(
        () =>
          api.ui.DialogAlert({
            title,
            message,
            onConfirm: () => {
              api.ui.dialog.clear()
              done()
            },
          }),
        () => done(),
      )
    }),
  confirm: ({ title, message, label }) =>
    new Promise((resolve) => {
      const done = settle(resolve)
      const suffix =
        label?.confirm || label?.cancel
          ? `\n\nConfirm = ${label.confirm ?? "OK"}. Cancel = ${label.cancel ?? "Cancel"}.`
          : ""
      api.ui.dialog.replace(
        () =>
          api.ui.DialogConfirm({
            title,
            message: `${message}${suffix}`,
            onConfirm: () => {
              api.ui.dialog.clear()
              done(true)
            },
            onCancel: () => {
              api.ui.dialog.clear()
              done(false)
            },
          }),
        () => done(undefined),
      )
    }),
  select: ({ title, options, current }) =>
    new Promise((resolve) => {
      const done = settle(resolve)
      api.ui.dialog.replace(
        () =>
          api.ui.DialogSelect({
            title,
            flat: true,
            options,
            current,
            onSelect: (option) => {
              api.ui.dialog.clear()
              done(option.value)
            },
          }),
        () => done(undefined),
      )
    }),
  prompt: ({ title, value, placeholder }) =>
    new Promise((resolve) => {
      const done = settle(resolve)
      api.ui.dialog.replace(
        () =>
          api.ui.DialogPrompt({
            title,
            value,
            placeholder,
            onConfirm: (text) => {
              api.ui.dialog.clear()
              done(text)
            },
            onCancel: () => {
              api.ui.dialog.clear()
              done(undefined)
            },
          }),
        () => done(undefined),
      )
    }),
})

export const deriveLiveStatus = (api: TuiPluginApi, settings: GlobalSettings) => {
  const kiloSessionId =
    api.route?.current?.name === "session" && typeof api.route.current.params?.sessionID === "string"
      ? api.route.current.params.sessionID
      : undefined
  const configuredWorkspace = settings.hosts?.kilo?.workspace
  const liveWorkspace =
    typeof configuredWorkspace === "string" && configuredWorkspace.trim()
      ? configuredWorkspace.trim()
      : path.basename(api.state?.path?.worktree || api.state?.path?.directory || HOST_ID)
  return {
    workspaceName: liveWorkspace,
    kiloSessionId,
  }
}

export const sessionFromApi = (api: TuiPluginApi): TuiSession => ({
  dialogs: dialogsFromApi(api),
  liveStatus: (settings) => deriveLiveStatus(api, settings),
  transcripts: transcriptSourceFromClient(api.client),
})

export const buildCommands = (api: TuiPluginApi) =>
  COMMANDS.map((command) => ({
    title: command.title,
    value: command.id,
    description: command.description,
    category: "Honcho",
    slash: { name: command.slash },
    onSelect: () => {
      void runGuarded(sessionFromApi(api), command)
    },
  }))

export const tui: TuiPlugin = async (api) => {
  api.command?.register(() => buildCommands(api))
  void maybePromptObservationUpgrade(sessionFromApi(api))
}
