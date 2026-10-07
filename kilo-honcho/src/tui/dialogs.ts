import type { TranscriptSource } from "../import.js"
import type { ObservationMode, RecallMode, SessionStrategy } from "../core.js"

export type DialogOption<T = string> = {
  title: string
  value: T
  description?: string
}

/** Promise dialogs, adapted from Kilo's callback dialogs. `undefined` means the user bailed. */
export type Dialogs = {
  alert(input: { title: string; message: string }): Promise<void>
  confirm(input: {
    title: string
    message: string
    label?: { confirm?: string; cancel?: string }
  }): Promise<boolean | undefined>
  select<T>(input: { title: string; options: DialogOption<T>[]; current?: T }): Promise<T | undefined>
  prompt(input: { title: string; value?: string; placeholder?: string }): Promise<string | undefined>
  view(input: { title: string; body: string }): Promise<void>
}

export type LiveStatus = {
  workspaceName?: string
  kiloSessionId?: string
}

export type GlobalSettings = {
  apiKey?: string
  peerName?: string
  baseUrl?: string
  hosts?: {
    kilo?: {
      workspace?: string
      aiPeer?: string
      recallMode?: RecallMode
      observationMode?: ObservationMode
      agentObserveMe?: boolean
      autoConclusions?: boolean
      sessionStrategy?: SessionStrategy
      removeUserPrefix?: boolean
    }
  }
}

export type TuiSession = {
  dialogs: Dialogs
  liveStatus: (settings: GlobalSettings) => LiveStatus
  transcripts: TranscriptSource
}

export type TuiCommandSpec = {
  id: string
  title: string
  description: string
  slash: string
  run: (session: TuiSession) => Promise<void>
}
