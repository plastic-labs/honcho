import { clampText } from "../core.js"
import type { HonchoActivity } from "../activity.js"

export type SidebarRows = {
  label: string
  tone: "success" | "muted" | "error"
  details: string[]
}

const count = (value: number, noun: string) => `${value} ${noun}${value === 1 ? "" : "s"}`

const timeOf = (iso: string) => {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleTimeString()
}

/** The Honcho section of Kilo's session sidebar. */
export const sidebarRows = (activity: HonchoActivity): SidebarRows => {
  if (activity.state === "unconfigured") {
    return { label: "Not set up", tone: "muted", details: ["Run /honcho:setup"] }
  }
  if (activity.state === "error") {
    return {
      label: "Error",
      tone: "error",
      details: [clampText(activity.error || "A Honcho request failed.", 120), "Run /honcho:status"],
    }
  }
  const details: string[] = []
  if (activity.userPeer && activity.workspace) details.push(`Peer: ${activity.userPeer} (${activity.workspace})`)
  if (activity.profile) {
    details.push(`Profile: ${activity.profile.conclusions > 0 ? count(activity.profile.conclusions, "conclusion") : "loaded"}`)
  }
  if (activity.recall) {
    details.push(`Recalled: ${activity.recall.conclusions > 0 ? count(activity.recall.conclusions, "conclusion") : "session summary"}`)
  }
  if (activity.recallMode === "tools") details.push("Recall: tools only")
  details.push(`Saved: ${count(activity.saved, "message")}`)
  if (activity.profile || activity.recall) details.push("/honcho:recall to view")
  const working = Boolean(activity.profile || activity.recall || activity.saved > 0)
  return { label: "Active", tone: working ? "success" : "muted", details }
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
