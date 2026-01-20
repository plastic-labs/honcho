/**
 * API response types for the Honcho SDK.
 * These types mirror the Pydantic schemas from the backend.
 */

// =============================================================================
// Workspace Types
// =============================================================================

export interface WorkspaceResponse {
  id: string
  metadata: Record<string, unknown>
  configuration: Record<string, unknown>
  created_at: string
}

export interface WorkspaceCreateParams {
  id: string
  metadata?: Record<string, unknown>
  configuration?: Record<string, unknown>
}

export interface WorkspaceUpdateParams {
  metadata?: Record<string, unknown>
  configuration?: Record<string, unknown>
}

export interface WorkspaceListParams {
  filters?: Record<string, unknown>
  page?: number
  size?: number
}

// =============================================================================
// Peer Types
// =============================================================================

export interface PeerResponse {
  id: string
  workspace_id: string
  metadata: Record<string, unknown>
  configuration: Record<string, unknown>
  created_at: string
}

export interface PeerCreateParams {
  id: string
  metadata?: Record<string, unknown>
  configuration?: Record<string, unknown>
}

export interface PeerUpdateParams {
  metadata?: Record<string, unknown>
  configuration?: Record<string, unknown>
}

export interface PeerListParams {
  filters?: Record<string, unknown>
  page?: number
  size?: number
}

export interface PeerChatParams {
  query: string
  stream?: boolean
  session_id?: string
  target?: string
  reasoning_level?: 'minimal' | 'low' | 'medium' | 'high' | 'max'
}

export interface PeerChatResponse {
  content: string | null
}

export interface PeerRepresentationParams {
  session_id?: string
  target?: string
  search_query?: string
  search_top_k?: number
  search_max_distance?: number
  include_most_frequent?: boolean
  max_conclusions?: number
}

export interface PeerCardParams {
  target?: string
}

export interface PeerCardResponse {
  peer_card: string[] | null
}

export interface PeerContextParams {
  target?: string
  search_query?: string
  search_top_k?: number
  search_max_distance?: number
  include_most_frequent?: boolean
  max_conclusions?: number
}

export interface PeerContextResponse {
  peer_id: string
  target_id: string
  representation: string | null
  peer_card: string[] | null
}

// =============================================================================
// Session Types
// =============================================================================

export interface SessionResponse {
  id: string
  workspace_id: string
  is_active: boolean
  metadata: Record<string, unknown>
  configuration: Record<string, unknown>
  created_at: string
}

export interface SessionCreateParams {
  id: string
  metadata?: Record<string, unknown>
  configuration?: Record<string, unknown>
  peers?: Record<string, SessionPeerConfigParams>
}

export interface SessionUpdateParams {
  metadata?: Record<string, unknown>
  configuration?: Record<string, unknown>
}

export interface SessionListParams {
  filters?: Record<string, unknown>
  page?: number
  size?: number
}

export interface SessionCloneParams {
  message_id?: string
}

export interface SessionPeerConfigParams {
  observe_me?: boolean | null
  observe_others?: boolean | null
}

export interface SessionContextParams {
  tokens?: number
  summary?: boolean
  last_message?: string
  peer_target?: string
  peer_perspective?: string
  limit_to_session?: boolean
  search_top_k?: number
  search_max_distance?: number
  include_most_frequent?: boolean
  max_conclusions?: number
}

export interface SummaryResponse {
  content: string
  message_id: string
  summary_type: string
  created_at: string
  token_count: number
}

export interface SessionContextResponse {
  id: string
  messages: MessageResponse[]
  summary: SummaryResponse | null
  peer_representation: string | null
  peer_card: string[] | null
}

export interface SessionSummariesResponse {
  id: string
  short_summary: SummaryResponse | null
  long_summary: SummaryResponse | null
}

// =============================================================================
// Message Types
// =============================================================================

/**
 * Raw API response for a message (snake_case).
 * Use the Message class for SDK consumers.
 */
export interface MessageResponse {
  id: string
  content: string
  peer_id: string
  session_id: string
  workspace_id: string
  metadata: Record<string, unknown>
  created_at: string
  token_count: number
}

export interface MessageCreateParams {
  peer_id: string
  content: string
  metadata?: Record<string, unknown>
  configuration?: Record<string, unknown>
  created_at?: string
}

export interface MessageBatchCreateParams {
  messages: MessageCreateParams[]
}

export interface MessageUpdateParams {
  metadata?: Record<string, unknown>
}

export interface MessageListParams {
  filters?: Record<string, unknown>
  page?: number
  size?: number
}

export interface MessageSearchParams {
  query: string
  filters?: Record<string, unknown>
  limit?: number
}

// =============================================================================
// Conclusion Types
// =============================================================================

export interface ConclusionResponse {
  id: string
  content: string
  observer_id: string
  observed_id: string
  session_id: string
  created_at: string
}

export interface ConclusionCreateParams {
  content: string
  observer_id: string
  observed_id: string
  session_id: string
}

export interface ConclusionBatchCreateParams {
  conclusions: ConclusionCreateParams[]
}

export interface ConclusionListParams {
  filters?: Record<string, unknown>
  page?: number
  size?: number
}

export interface ConclusionQueryParams {
  query: string
  top_k?: number
  distance?: number
  filters?: Record<string, unknown>
}

// =============================================================================
// Representation Types
// =============================================================================

export interface RepresentationResponse {
  representation: string
}

/**
 * Options for representation retrieval.
 */
export interface RepresentationOptions {
  /**
   * Semantic search query to filter relevant conclusions.
   */
  searchQuery?: string

  /**
   * Number of semantically relevant conclusions to return.
   */
  searchTopK?: number

  /**
   * Maximum semantic distance for search results (0.0-1.0).
   */
  searchMaxDistance?: number

  /**
   * Whether to include the most frequent conclusions.
   */
  includeMostFrequent?: boolean

  /**
   * Maximum number of conclusions to include.
   */
  maxConclusions?: number
}

// =============================================================================
// Queue Types
// =============================================================================

export interface SessionQueueStatusResponse {
  session_id: string | null
  total_work_units: number
  completed_work_units: number
  in_progress_work_units: number
  pending_work_units: number
}

export interface QueueStatusResponse {
  total_work_units: number
  completed_work_units: number
  in_progress_work_units: number
  pending_work_units: number
  sessions?: Record<string, SessionQueueStatusResponse>
}

export interface QueueStatusParams {
  observer_id?: string
  sender_id?: string
  session_id?: string
}

/**
 * Queue status scoped to a single session.
 */
export interface SessionQueueStatus {
  sessionId: string | null
  totalWorkUnits: number
  completedWorkUnits: number
  inProgressWorkUnits: number
  pendingWorkUnits: number
}

/**
 * Queue status scoped to a workspace.
 */
export interface QueueStatus {
  totalWorkUnits: number
  completedWorkUnits: number
  inProgressWorkUnits: number
  pendingWorkUnits: number
  sessions?: Record<string, SessionQueueStatus>
}

// =============================================================================
// Pagination Types
// =============================================================================

export interface PageResponse<T> {
  items: T[]
  page: number
  size: number
  total: number
  pages: number
}
