// Main entry point for the Honcho TypeScript SDK
// Exports all main classes and types

// Domain classes
export { Honcho } from './client'
export { Conclusion, ConclusionScope } from './conclusions'
// HTTP infrastructure
export {
  AuthenticationError,
  ConnectionError,
  HonchoError,
  NotFoundError,
  PermissionError,
  RateLimitError,
  ServerError,
  TimeoutError,
  ValidationError,
} from './http/errors'
// Streaming types
export {
  type DialecticStreamChunk,
  DialecticStreamResponse,
} from './http/streaming'
export { Page } from './pagination'
export { Peer, type PeerContextResponse } from './peer'
export { Session, SessionPeerConfig } from './session'
export {
  SessionContext,
  SessionSummaries,
  Summary,
  type SummaryData,
} from './session_context'

// Internal types
export type {
  Conclusion as ConclusionData,
  ConclusionQueryParams,
  DialecticStreamDelta,
} from './types'
// API types from our hand-written types
export type {
  ConclusionResponse,
  MessageResponse,
  PageResponse,
  PeerResponse,
  QueueStatusResponse,
  SessionContextResponse,
  SessionResponse,
  SessionSummariesResponse,
  SummaryResponse,
  WorkspaceResponse,
} from './types/api'

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
