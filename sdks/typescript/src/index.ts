// Main entry point for the Honcho TypeScript SDK
// Exports all main classes and types

export type { Message } from '@honcho-ai/core/resources/workspaces/sessions/messages'
export { Honcho } from './client'
export { Conclusion, ConclusionScope } from './conclusions'
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
  type Conclusion as ConclusionData,
  type ConclusionQueryParams,
  type DialecticStreamChunk,
  type DialecticStreamDelta,
  DialecticStreamResponse,
} from './types'

// Export validation types for advanced usage
export type {
  ChatQuery,
  ContextParams,
  FileUpload,
  Filters,
  GetRepresentationParams,
  HonchoConfig,
  MessageAddition,
  MessageCreate,
  PeerAddition,
  PeerConfig,
  PeerGetRepresentationParams,
  PeerMetadata,
  PeerRemoval,
  QueueStatusOptions,
  SessionConfig,
  SessionMetadata,
  WorkspaceConfig,
  WorkspaceMetadata,
} from './validation'
