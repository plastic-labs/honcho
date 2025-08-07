// Main entry point for the Honcho TypeScript SDK
// Exports all main classes and types

export { Message } from '@honcho-ai/core/src/resources/workspaces/sessions/messages'
export { Honcho } from './client'
export { Page } from './pagination'
export { Peer } from './peer'
export { Session, SessionPeerConfig } from './session'
export { SessionContext } from './session_context'

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
