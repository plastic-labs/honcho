import { clampText } from "../core.js"
import type { HonchoActivity } from "../activity.js"

export type SidebarLine = { text: string; url?: string }

export type SidebarRows = {
  label: string
  tone: "success" | "muted" | "error"
  details: SidebarLine[]
}

// "Session: " plus this fits Kilo's sidebar.
const SESSION_NAME_MAX = 28

// Cut from the middle: the end of a session name is the folder, which tells sessions apart.
export const truncateMiddle = (value: string, max: number) => {
  if (value.length <= max) return value
  const keep = max - 1
  return `${value.slice(0, Math.ceil(keep / 2))}…${value.slice(value.length - Math.floor(keep / 2))}`
}

const count = (value: number, noun: string) => `${value} ${noun}${value === 1 ? "" : "s"}`

const timeOf = (iso: string) => {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleTimeString()
}

/** The Honcho section of Kilo's session sidebar. */
export const sidebarRows = (activity: HonchoActivity): SidebarRows => {
  if (activity.state === "unconfigured") {
    return { label: "Not set up", tone: "muted", details: [{ text: "Run /honcho:setup" }] }
  }
  if (activity.state === "error") {
    return {
      label: "Error",
      tone: "error",
      details: [{ text: clampText(activity.error || "A Honcho request failed.", 120) }, { text: "Run /honcho:status" }],
    }
  }
  const details: SidebarLine[] = []
  if (activity.userPeer) details.push({ text: `Peer: ${activity.userPeer}` })
  if (activity.sessionUrl) details.push({ text: "Session: View in Honcho ↗", url: activity.sessionUrl })
  else if (activity.session) details.push({ text: `Session: ${truncateMiddle(activity.session, SESSION_NAME_MAX)}` })
  return { label: "Active", tone: "success", details }
}

/** The body of `/honcho:recall`: the exact memory text this session sent to the model. */
export const recallMessage = (activity: HonchoActivity | null) => {
  if (!activity || (!activity.recall && !activity.profile)) {
    return [
      "Honcho has not added memory to this session yet.",
      "",
      "Recall starts once Honcho has conclusions about you. A new workspace needs a few messages first.",
      ...(activity?.recallMode === "tools" ? ["", "recallMode is tools, so memory reaches the model only when it calls a honcho_* tool."] : []),
    ].join("\n")
  }
  const sections: string[] = []
  const session = activity.sessionUrl ?? activity.session
  if (session) sections.push(`Session: ${session}`, "")
  if (activity.recall) {
    sections.push(
      `Attached to your prompt at ${timeOf(activity.recall.at)} (${count(activity.recall.conclusions, "conclusion")}):`,
      "",
      activity.recall.text,
      "",
    )
  }
  if (activity.profile) {
    sections.push(
      `Added to the system prompt at ${timeOf(activity.profile.at)} (${count(activity.profile.conclusions, "conclusion")}):`,
      "",
      activity.profile.text,
    )
  }
  return sections.join("\n").trimEnd()
}
