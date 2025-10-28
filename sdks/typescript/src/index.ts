// Main entry point for the Honcho TypeScript SDK
// Exports all main classes and types

export type { Message } from '@honcho-ai/core/resources/workspaces/sessions/messages'
export { Honcho } from './client'
export { Page } from './pagination'
export { Peer } from './peer'
export { Session, SessionPeerConfig } from './session'
export {
  SessionContext,
  SessionSummaries,
  Summary,
  type SummaryData,
} from './session_context'
export {
  type DialecticStreamChunk,
  type DialecticStreamDelta,
  DialecticStreamResponse,
} from './types'

// Export validation types for advanced usage
export type {
  ChatQuery,
  ContextParams,
  DeriverStatusOptions,
  FileUpload,
  Filters,
  HonchoConfig,
  MessageAddition,
  MessageCreate,
  PeerAddition,
  PeerConfig,
  PeerMetadata,
  PeerRemoval,
  SessionConfig,
  SessionMetadata,
  WorkingRepParams,
  WorkspaceMetadata,
} from './validation'
