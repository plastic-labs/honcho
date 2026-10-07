/** @jsxImportSource @opentui/solid */
import type { TuiPluginApi } from "@kilocode/plugin/tui"

// A large dialog leaves about 80 columns for text.
const WRAP_COLUMNS = 80

const wrappedRows = (body: string) =>
  body.split("\n").reduce((rows, line) => rows + Math.max(1, Math.ceil(line.length / WRAP_COLUMNS)), 0)

function TextView(props: { api: TuiPluginApi; title: string; body: string }) {
  const theme = () => props.api.theme.current
  // Kilo centers the dialog, so 60% of the terminal leaves room for its frame and the title.
  const height = () => Math.max(3, Math.min(wrappedRows(props.body) + 1, Math.floor(props.api.renderer.terminalHeight * 0.6)))

  return (
    <box paddingLeft={2} paddingRight={2} gap={1}>
      <box flexDirection="row" justifyContent="space-between">
        <text fg={theme().text}>
          <b>{props.title}</b>
        </text>
        <text fg={theme().textMuted} onMouseUp={() => props.api.ui.dialog.clear()}>
          esc
        </text>
      </box>
      <scrollbox height={height()} focused paddingBottom={1}>
        <text fg={theme().textMuted} wrapMode="word">
          {props.body}
        </text>
      </scrollbox>
    </box>
  )
}

/** A read-only dialog for text too long for Kilo's alert, which cannot scroll. */
export const showTextView = (api: TuiPluginApi, input: { title: string; body: string }, onClose: () => void) => {
  api.ui.dialog.replace(() => <TextView api={api} title={input.title} body={input.body} />, onClose)
  // replace() resets the size to medium.
  api.ui.dialog.setSize("large")
}
