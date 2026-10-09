/** @jsxImportSource @opentui/solid */
import type { TuiPluginApi } from "@kilocode/plugin/tui"
import { createSignal, For, onCleanup, Show } from "solid-js"
import { readActivity, type HonchoActivity } from "../activity.js"
import { sidebarRows } from "./activity-view.js"
import { resolveConfigPath } from "./commands.js"
import { openUrl } from "./open-url.js"

const REFRESH_MS = 2_000

function HonchoSidebar(props: { api: TuiPluginApi; sessionID: string; onRecall: () => void }) {
  const [activity, setActivity] = createSignal<HonchoActivity | null>(null)
  const refresh = () => {
    const sessionID = props.sessionID
    void readActivity(resolveConfigPath(), sessionID).then((next) => {
      if (sessionID === props.sessionID) setActivity(next)
    })
  }
  refresh()
  const timer = setInterval(refresh, REFRESH_MS)
  timer.unref?.()
  onCleanup(() => clearInterval(timer))
  const theme = () => props.api.theme.current
  const rows = () => {
    const current = activity()
    return current ? sidebarRows(current) : null
  }

  return (
    // Kilo mounts sidebar slots once, so a conditional root would never appear.
    <box>
      <Show when={rows()}>
        {(row) => (
          <box>
            <text fg={theme().text}>
              <b>Honcho</b>
            </text>
            <box flexDirection="row" gap={1}>
              <text
                fg={row().tone === "success" ? theme().success : row().tone === "error" ? theme().error : theme().textMuted}
              >
                •
              </text>
              <text fg={theme().text}>{row().label}</text>
            </box>
            <For each={row().details}>
              {(line) => (
                <text
                  fg={line.url || line.opensRecall ? theme().primary : theme().textMuted}
                  wrapMode="word"
                  onMouseUp={() => {
                    if (line.url) openUrl(line.url)
                    if (line.opensRecall) props.onRecall()
                  }}
                >
                  {line.text}
                </text>
              )}
            </For>
          </box>
        )}
      </Show>
    </box>
  )
}

export const registerSidebar = (api: TuiPluginApi, onRecall: () => void) =>
  api.slots.register({
    // Kilo's own Memory section is order 1000, so Honcho sits directly below it.
    order: 1001,
    slots: {
      sidebar_content(_ctx, props) {
        return <HonchoSidebar api={api} sessionID={props.session_id} onRecall={onRecall} />
      },
    },
  })
