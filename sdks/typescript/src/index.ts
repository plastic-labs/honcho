// Main entry point for the Honcho TypeScript SDK
// Exports all main classes and types

// Domain classes
export { Honcho } from './client'
export {
  Conclusion,
  type ConclusionCreateParams,
  ConclusionScope,
} from './conclusions'
// HTTP infrastructure
export {
  AuthenticationError,
  BadRequestError,
  ConflictError,
  ConnectionError,
  HonchoError,
  NotFoundError,
  PermissionDeniedError,
  RateLimitError,
  ServerError,
  TimeoutError,
  UnprocessableEntityError,
} from './http/errors'
// Streaming types
export {
  type DialecticStreamChunk,
  DialecticStreamResponse,
} from './http/streaming'
export { Message, type MessageInput } from './message'
export { Page } from './pagination'
export { Peer, PeerContext } from './peer'
export { Session } from './session'
export {
  SessionContext,
  SessionSummaries,
  Summary,
  type SummaryData,
} from './session_context'

// API types (snake_case, for advanced usage)
export type {
  ConclusionQueryParams,
  ConclusionResponse,
  MessageResponse,
  PageResponse,
  PeerContextResponse,
  PeerResponse,
  QueueStatus,
  QueueStatusResponse,
  RepresentationOptions,
  SessionContextResponse,
  SessionQueueStatus,
  SessionResponse,
  SessionSummariesResponse,
  SummaryResponse,
  WorkspaceChatResponse,
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
  PeerAddition,
  PeerConfig,
  PeerGetRepresentationParams,
  PeerMetadata,
  PeerRemoval,
  QueueStatusOptions,
  SessionConfig,
  SessionMetadata,
  SessionPeerConfig,
  WorkspaceConfig,
  WorkspaceMetadata,
} from './validation'
