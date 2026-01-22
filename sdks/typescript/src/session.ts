import { API_VERSION } from './api-version'
import type { HonchoHTTPClient } from './http/client'
import { Message } from './message'
import { Page } from './pagination'
import { Peer } from './peer'
import { SessionContext, SessionSummaries } from './session_context'
import type {
  MessageResponse,
  PageResponse,
  PeerResponse,
  QueueStatus,
  QueueStatusParams,
  QueueStatusResponse,
  RepresentationOptions,
  RepresentationResponse,
  SessionContextResponse,
  SessionResponse,
  SessionSummariesResponse,
} from './types/api'
import { transformQueueStatus } from './utils'
import {
  ContextParamsSchema,
  FileUploadSchema,
  FilterSchema,
  type Filters,
  GetRepresentationParamsSchema,
  LimitSchema,
  type MessageAddition,
  MessageAdditionToApiSchema,
  MessageMetadataSchema,
  type PeerAddition,
  PeerAdditionToApiSchema,
  type PeerRemoval,
  PeerRemovalSchema,
  type QueueStatusOptions,
  SearchQuerySchema,
  type SessionConfig,
  SessionConfigSchema,
  SessionMetadataSchema,
  type SessionPeerConfig,
  SessionPeerConfigSchema,
  sessionConfigFromApi,
  sessionConfigToApi,
} from './validation'

/**
 * Represents a session in the Honcho system.
 *
 * Sessions are conversation contexts that can involve multiple peers. They track
 * message history, manage peer participation with configurable observation settings,
 * and provide context retrieval for LLM interactions.
 *
 * @example
 * ```typescript
 * const session = await honcho.session('conversation-123')
 *
 * // Add peers to the session
 * await session.addPeers([user, assistant])
 *
 * // Add messages
 * await session.addMessages([
 *   user.message('Hello!'),
 *   assistant.message('Hi there!')
 * ])
 *
 * // Get context for LLM
 * const ctx = await session.context({ peerPerspective: assistant })
 * ```
 */
export class Session {
  /**
   * Unique identifier for this session.
   */
  readonly id: string
  /**
   * Workspace ID for scoping operations.
   */
  readonly workspaceId: string
  private _http: HonchoHTTPClient
  private _metadata?: Record<string, unknown>
  private _configuration?: SessionConfig
  private _ensureWorkspace: () => Promise<void>

  /**
   * Cached metadata for this session. May be stale if the session
   * was not recently fetched from the API.
   *
   * Call getMetadata() to get the latest metadata from the server,
   * which will also update this cached value.
   */
  get metadata(): Record<string, unknown> | undefined {
    return this._metadata
  }

  /**
   * Cached configuration for this session. May be stale if the session
   * was not recently fetched from the API.
   *
   * Call getConfiguration() to get the latest configuration from the server,
   * which will also update this cached value.
   */
  get configuration(): SessionConfig | undefined {
    return this._configuration
  }

  /**
   * Initialize a new Session. **Do not call this directly, use the client.session() method instead.**
   *
   * @param id - Unique identifier for this session within the workspace
   * @param workspaceId - Workspace ID for scoping operations
   * @param http - Reference to the HTTP client instance
   * @param metadata - Optional metadata to initialize the cached value
   * @param configuration - Optional configuration to initialize the cached value
   */
  constructor(
    id: string,
    workspaceId: string,
    http: HonchoHTTPClient,
    metadata?: Record<string, unknown>,
    configuration?: SessionConfig,
    ensureWorkspace: () => Promise<void> = async () => undefined
  ) {
    this.id = id
    this.workspaceId = workspaceId
    this._http = http
    this._metadata = metadata
    this._configuration = configuration
    this._ensureWorkspace = ensureWorkspace
  }

  // ===========================================================================
  // Private API Methods
  // ===========================================================================

  private async _getOrCreate(params: {
    id: string
    metadata?: Record<string, unknown>
    configuration?: SessionConfig
  }): Promise<SessionResponse> {
    await this._ensureWorkspace()
    return this._http.post<SessionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions`,
      {
        body: {
          id: params.id,
          metadata: params.metadata,
          configuration: sessionConfigToApi(params.configuration),
        },
      }
    )
  }

  private async _update(params: {
    metadata?: Record<string, unknown>
    configuration?: SessionConfig
  }): Promise<SessionResponse> {
    await this._ensureWorkspace()
    return this._http.put<SessionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}`,
      {
        body: {
          metadata: params.metadata,
          configuration: sessionConfigToApi(params.configuration),
        },
      }
    )
  }

  private async _delete(): Promise<SessionResponse> {
    await this._ensureWorkspace()
    return this._http.delete<SessionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}`
    )
  }

  private async _clone(params?: {
    message_id?: string
  }): Promise<SessionResponse> {
    await this._ensureWorkspace()
    return this._http.post<SessionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/clone`,
      { query: params }
    )
  }

  private async _getContext(params: {
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
  }): Promise<SessionContextResponse> {
    await this._ensureWorkspace()
    return this._http.get<SessionContextResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/context`,
      { query: params }
    )
  }

  private async _getSummaries(): Promise<SessionSummariesResponse> {
    await this._ensureWorkspace()
    return this._http.get<SessionSummariesResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/summaries`
    )
  }

  private async _search(params: {
    query: string
    filters?: Record<string, unknown>
    limit?: number
  }): Promise<MessageResponse[]> {
    await this._ensureWorkspace()
    return this._http.post<MessageResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/search`,
      { body: params }
    )
  }

  private async _addPeers(
    peers: Record<
      string,
      { observe_me?: boolean | null; observe_others?: boolean | null }
    >
  ): Promise<void> {
    await this._ensureWorkspace()
    await this._http.post(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers`,
      { body: peers }
    )
  }

  private async _setPeers(
    peers: Record<
      string,
      { observe_me?: boolean | null; observe_others?: boolean | null }
    >
  ): Promise<void> {
    await this._ensureWorkspace()
    await this._http.put(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers`,
      { body: peers }
    )
  }

  private async _removePeers(peerIds: string[]): Promise<void> {
    await this._ensureWorkspace()
    await this._http.delete(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers`,
      { body: peerIds }
    )
  }

  private async _listPeers(): Promise<PageResponse<PeerResponse>> {
    await this._ensureWorkspace()
    return this._http.get<PageResponse<PeerResponse>>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers`
    )
  }

  private async _getPeerConfiguration(
    peerId: string
  ): Promise<{ observe_me?: boolean | null; observe_others?: boolean | null }> {
    await this._ensureWorkspace()
    return this._http.get<{
      observe_me?: boolean | null
      observe_others?: boolean | null
    }>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers/${peerId}/config`
    )
  }

  private async _setPeerConfiguration(
    peerId: string,
    config: { observe_me?: boolean | null; observe_others?: boolean | null }
  ): Promise<void> {
    await this._ensureWorkspace()
    await this._http.put(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers/${peerId}/config`,
      { body: config }
    )
  }

  private async _createMessages(params: {
    messages: Array<{
      peer_id: string
      content: string
      metadata?: Record<string, unknown>
      configuration?: Record<string, unknown>
      created_at?: string
    }>
  }): Promise<MessageResponse[]> {
    await this._ensureWorkspace()
    return this._http.post<MessageResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/messages`,
      { body: params }
    )
  }

  private async _listMessages(params?: {
    filters?: Record<string, unknown>
    page?: number
    size?: number
  }): Promise<PageResponse<MessageResponse>> {
    await this._ensureWorkspace()
    return this._http.post<PageResponse<MessageResponse>>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/messages/list`,
      {
        body: { filters: params?.filters },
        query: { page: params?.page, size: params?.size },
      }
    )
  }

  private async _uploadFile(formData: FormData): Promise<MessageResponse[]> {
    await this._ensureWorkspace()
    return this._http.upload<MessageResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/messages/upload`,
      formData
    )
  }

  private async _getQueueStatus(
    params?: QueueStatusParams
  ): Promise<QueueStatusResponse> {
    await this._ensureWorkspace()
    const query: Record<string, string | number | boolean | undefined> = {}
    if (params?.observer_id) query.observer_id = params.observer_id
    if (params?.sender_id) query.sender_id = params.sender_id
    if (params?.session_id) query.session_id = params.session_id

    return this._http.get<QueueStatusResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/queue/status`,
      { query }
    )
  }

  private async _getRepresentation(
    peerId: string,
    params: {
      session_id?: string
      target?: string
      search_query?: string
      search_top_k?: number
      search_max_distance?: number
      include_most_frequent?: boolean
      max_conclusions?: number
    }
  ): Promise<RepresentationResponse> {
    await this._ensureWorkspace()
    return this._http.post<RepresentationResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${peerId}/representation`,
      { body: params }
    )
  }

  private async _updateMessage(
    messageId: string,
    params: { metadata: Record<string, unknown> }
  ): Promise<MessageResponse> {
    await this._ensureWorkspace()
    return this._http.put<MessageResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/messages/${messageId}`,
      { body: params }
    )
  }

  // ===========================================================================
  // Public Methods
  // ===========================================================================

  /**
   * Add peers to this session.
   *
   * Makes an API call to add one or more peers to the session. Peers can be
   * specified as IDs, Peer objects, or with observation configuration.
   *
   * @param peers - Peers to add. Can be a single peer ID, Peer object, array of either,
   *                or an object mapping peer IDs to their observation config
   *
   * @example
   * ```typescript
   * // Add by ID
   * await session.addPeers('user-123')
   *
   * // Add multiple peers
   * await session.addPeers([user, assistant])
   *
   * // Add with observation config
   * await session.addPeers({
   *   'user-123': { observeMe: true, observeOthers: true },
   *   'assistant': { observeMe: false }
   * })
   * ```
   */
  async addPeers(peers: PeerAddition): Promise<void> {
    const peerDict = PeerAdditionToApiSchema.parse(peers)
    await this._addPeers(peerDict)
  }

  /**
   * Set the peers for this session, replacing any existing peer list.
   *
   * Makes an API call to replace the session's peer list with the provided peers.
   * Any peers not included will be removed from the session.
   *
   * @param peers - Peers to set. Can be a single peer ID, Peer object, array of either,
   *                or an object mapping peer IDs to their observation config
   */
  async setPeers(peers: PeerAddition): Promise<void> {
    const peerDict = PeerAdditionToApiSchema.parse(peers)
    await this._setPeers(peerDict)
  }

  /**
   * Remove peers from this session.
   *
   * Makes an API call to remove one or more peers from the session.
   *
   * @param peers - Peers to remove. Can be a single peer ID, Peer object, or array of either
   */
  async removePeers(peers: PeerRemoval): Promise<void> {
    const validatedPeers = PeerRemovalSchema.parse(peers)
    const peerIds = Array.isArray(validatedPeers)
      ? validatedPeers.map((p) => (typeof p === 'string' ? p : p.id))
      : [
          typeof validatedPeers === 'string'
            ? validatedPeers
            : validatedPeers.id,
        ]
    await this._removePeers(peerIds)
  }

  /**
   * Get all peers in this session.
   *
   * Makes an API call to retrieve all peers that are currently part of this session.
   *
   * @returns Promise resolving to an array of Peer objects in this session
   */
  async peers(): Promise<Peer[]> {
    const peersPage = await this._listPeers()
    return peersPage.items.map(
      (peer) =>
        new Peer(
          peer.id,
          this.workspaceId,
          this._http,
          undefined,
          undefined,
          () => this._ensureWorkspace()
        )
    )
  }

  /**
   * Get the session-specific configuration for a peer.
   *
   * Makes an API call to retrieve the observation settings for a specific peer
   * within this session.
   *
   * @param peer - The peer to get configuration for (ID string or Peer object)
   * @returns Promise resolving to the peer's session configuration with observation settings
   */
  async getPeerConfiguration(peer: string | Peer): Promise<SessionPeerConfig> {
    const peerId = typeof peer === 'string' ? peer : peer.id
    const response = await this._getPeerConfiguration(peerId)
    return {
      observeMe: response.observe_me,
      observeOthers: response.observe_others,
    }
  }

  /**
   * Set the session-specific configuration for a peer.
   *
   * Makes an API call to update the observation settings for a specific peer
   * within this session.
   *
   * @param peer - The peer to configure (ID string or Peer object)
   * @param configuration - Configuration with observation settings
   * @param configuration.observeMe - Whether this peer's messages generate observations about them
   * @param configuration.observeOthers - Whether this peer observes other peers in the session
   */
  async setPeerConfiguration(
    peer: string | Peer,
    configuration: SessionPeerConfig
  ): Promise<void> {
    const peerId = typeof peer === 'string' ? peer : peer.id
    const validatedConfig = SessionPeerConfigSchema.parse(configuration)
    await this._setPeerConfiguration(peerId, {
      observe_others: validatedConfig.observeOthers,
      observe_me: validatedConfig.observeMe,
    })
  }

  /**
   * Add messages to this session.
   *
   * Makes an API call to create one or more messages in the session. Messages
   * are processed asynchronously to update peer representations.
   *
   * @param messages - Messages to add. Can be a single MessageInput or array of them.
   *                   Use `peer.message()` to create MessageInput objects.
   * @returns Promise resolving to an array of created Message objects
   *
   * @example
   * ```typescript
   * // Add a single message
   * await session.addMessages(user.message('Hello!'))
   *
   * // Add multiple messages
   * await session.addMessages([
   *   user.message('Hello!'),
   *   assistant.message('Hi there!'),
   *   user.message('How are you?')
   * ])
   * ```
   */
  async addMessages(messages: MessageAddition): Promise<Message[]> {
    const transformedMessages = MessageAdditionToApiSchema.parse(messages)
    const apiMessages = transformedMessages.map((msg) => ({
      peer_id: msg.peer_id,
      content: msg.content,
      metadata: msg.metadata,
      configuration: msg.configuration ?? undefined,
      created_at: msg.created_at ?? undefined,
    }))
    const response = await this._createMessages({ messages: apiMessages })
    return response.map(Message.fromApiResponse)
  }

  /**
   * Get all messages in this session.
   *
   * Makes an API call to retrieve messages in the session, with optional filtering.
   *
   * @param filters - Optional filter criteria for messages. See
   *                  [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
   * @returns Promise resolving to a paginated Page of Message objects
   */
  async messages(filters?: Filters): Promise<Page<Message, MessageResponse>> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const messagesPage = await this._listMessages({ filters: validatedFilter })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<MessageResponse>> => {
      return this._listMessages({ filters: validatedFilter, page, size })
    }

    return new Page(messagesPage, Message.fromApiResponse, fetchNextPage)
  }

  /**
   * Get the current metadata for this session.
   *
   * Makes an API call to retrieve metadata associated with this session.
   * This method also updates the cached metadata property.
   *
   * @returns Promise resolving to a dictionary containing the session's metadata.
   *          Returns an empty dictionary if no metadata is set
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const session = await this._getOrCreate({ id: this.id })
    this._metadata = session.metadata || {}
    return this._metadata
  }

  /**
   * Set the metadata for this session.
   *
   * Makes an API call to update the metadata associated with this session.
   * This will overwrite any existing metadata with the provided values.
   * This method also updates the cached metadata property.
   *
   * @param metadata - A dictionary of metadata to associate with this session.
   *                   Keys must be strings, values can be any JSON-serializable type
   */
  async setMetadata(metadata: Record<string, unknown>): Promise<void> {
    const validatedMetadata = SessionMetadataSchema.parse(metadata)
    await this._update({ metadata: validatedMetadata })
    this._metadata = validatedMetadata
  }

  /**
   * Get the current configuration for this session.
   *
   * Makes an API call to retrieve configuration associated with this session.
   * This method also updates the cached configuration property.
   *
   * @returns Promise resolving to the session's configuration.
   *          Returns an empty object if no configuration is set
   */
  async getConfiguration(): Promise<SessionConfig> {
    const session = await this._getOrCreate({ id: this.id })
    this._configuration = sessionConfigFromApi(session.configuration) || {}
    return this._configuration
  }

  /**
   * Set the configuration for this session.
   *
   * Makes an API call to update the configuration associated with this session.
   * This will overwrite any existing configuration with the provided values.
   * This method also updates the cached configuration property.
   *
   * @param configuration - Configuration to associate with this session.
   *                        Includes reasoning, peerCard, summary, and dream settings.
   */
  async setConfiguration(configuration: SessionConfig): Promise<void> {
    const validatedConfig = SessionConfigSchema.parse(configuration)
    await this._update({ configuration: validatedConfig })
    this._configuration = validatedConfig
  }

  /**
   * Refresh cached metadata and configuration for this session.
   *
   * Makes a single API call to retrieve the latest metadata and configuration
   * associated with this session and updates the cached properties.
   */
  async refresh(): Promise<void> {
    const session = await this._getOrCreate({ id: this.id })
    this._metadata = session.metadata || {}
    this._configuration = sessionConfigFromApi(session.configuration) || {}
  }

  /**
   * Delete this session.
   *
   * Makes an API call to permanently delete the session and all its messages.
   * This action cannot be undone.
   */
  async delete(): Promise<void> {
    await this._delete()
  }

  /**
   * Clone this session.
   *
   * Makes an API call to create a copy of the session. If a message ID is provided,
   * the clone will only include messages up to and including that message.
   *
   * @param messageId - Optional message ID to clone up to. If not provided,
   *                    clones the entire session
   * @returns Promise resolving to the new cloned Session object
   */
  async clone(messageId?: string): Promise<Session> {
    const clonedSessionData = await this._clone(
      messageId ? { message_id: messageId } : undefined
    )

    return new Session(
      clonedSessionData.id,
      this.workspaceId,
      this._http,
      clonedSessionData.metadata ?? undefined,
      sessionConfigFromApi(clonedSessionData.configuration) ?? undefined,
      () => this._ensureWorkspace()
    )
  }

  /**
   * Get context for this session, suitable for LLM prompts.
   *
   * Makes an API call to retrieve a curated context including messages, optional
   * summary, and peer representation. The context can be converted to OpenAI or
   * Anthropic message formats.
   *
   * @param options - Configuration options for context retrieval
   * @param options.summary - Whether to include a summary of earlier messages
   * @param options.tokens - Target token count for the context window
   * @param options.peerTarget - The peer to get representation for
   * @param options.lastUserMessage - Message text (string) or Message object whose content will be used for semantic search
   * @param options.peerPerspective - The peer whose perspective to use for representation
   * @param options.limitToSession - Whether to limit representation to this session only
   * @param options.representationOptions - Options for representation retrieval
   * @returns Promise resolving to a SessionContext with messages, summary, and representation
   *
   * @example
   * ```typescript
   * const ctx = await session.context({
   *   summary: true,
   *   peerPerspective: assistant,
   *   peerTarget: user
   * })
   *
   * // Convert to OpenAI format
   * const messages = ctx.toOpenAI(assistant)
   * ```
   */
  async context(options?: {
    summary?: boolean
    tokens?: number
    peerTarget?: string | Peer
    lastUserMessage?: string | Message
    peerPerspective?: string | Peer
    limitToSession?: boolean
    representationOptions?: RepresentationOptions
  }): Promise<SessionContext> {
    const opts = options || {}

    // Resolve Peer objects to their IDs
    const peerTargetId =
      typeof opts.peerTarget === 'object' ? opts.peerTarget.id : opts.peerTarget
    const peerPerspectiveId =
      typeof opts.peerPerspective === 'object'
        ? opts.peerPerspective.id
        : opts.peerPerspective
    const lastUserMessageText =
      typeof opts.lastUserMessage === 'string'
        ? opts.lastUserMessage
        : opts.lastUserMessage?.content

    const contextParams = ContextParamsSchema.parse({
      summary: opts.summary,
      tokens: opts.tokens,
      peerTarget: peerTargetId,
      lastUserMessage: lastUserMessageText,
      peerPerspective: peerPerspectiveId,
      limitToSession: opts.limitToSession,
      representationOptions: opts.representationOptions,
    })

    const lastMessageText =
      typeof contextParams.lastUserMessage === 'string'
        ? contextParams.lastUserMessage
        : contextParams.lastUserMessage?.content

    const context = await this._getContext({
      tokens: contextParams.tokens,
      summary: contextParams.summary,
      last_message: lastMessageText,
      peer_target: contextParams.peerTarget,
      peer_perspective: contextParams.peerPerspective,
      limit_to_session: contextParams.limitToSession,
      search_top_k: contextParams.representationOptions?.searchTopK,
      search_max_distance:
        contextParams.representationOptions?.searchMaxDistance,
      include_most_frequent:
        contextParams.representationOptions?.includeMostFrequent,
      max_conclusions: contextParams.representationOptions?.maxConclusions,
    })

    return SessionContext.fromApiResponse(this.id, context)
  }

  /**
   * Get the summaries for this session.
   *
   * Makes an API call to retrieve both short and long summaries for the session.
   * Summaries are generated automatically as messages accumulate.
   *
   * @returns Promise resolving to a SessionSummaries object with short and long summaries
   */
  async summaries(): Promise<SessionSummaries> {
    const data = await this._getSummaries()
    return SessionSummaries.fromApiResponse(data)
  }

  /**
   * Search for messages in this session.
   *
   * Makes an API call to perform semantic search over messages in this session.
   *
   * @param query - The search query to use
   * @param options - Search options
   * @param options.filters - Optional filters to scope the search. See
   *                          [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
   * @param options.limit - Number of results to return (1-100, default: 10)
   * @returns Promise resolving to an array of Message objects matching the query
   */
  async search(
    query: string,
    options?: { filters?: Filters; limit?: number }
  ): Promise<Message[]> {
    const validatedQuery = SearchQuerySchema.parse(query)
    const validatedFilters = options?.filters
      ? FilterSchema.parse(options.filters)
      : undefined
    const validatedLimit = options?.limit
      ? LimitSchema.parse(options.limit)
      : undefined
    const response = await this._search({
      query: validatedQuery,
      filters: validatedFilters,
      limit: validatedLimit,
    })
    return response.map(Message.fromApiResponse)
  }

  /**
   * Get the queue processing status for this session.
   *
   * Makes an API call to retrieve the current status of background processing
   * for messages in this session. The queue processes messages to update
   * peer representations.
   *
   * @param options - Configuration options for the status request
   * @param options.observer - Optional observer peer to scope the status to
   * @param options.sender - Optional sender peer to scope the status to
   * @returns Promise resolving to queue status information including work unit counts
   */
  async queueStatus(
    options?: Omit<
      QueueStatusOptions,
      'sessionId' | 'observerId' | 'senderId'
    > & {
      observer?: string | Peer
      sender?: string | Peer
    }
  ): Promise<QueueStatus> {
    const resolvedObserverId = options?.observer
      ? typeof options.observer === 'string'
        ? options.observer
        : options.observer.id
      : undefined
    const resolvedSenderId = options?.sender
      ? typeof options.sender === 'string'
        ? options.sender
        : options.sender.id
      : undefined

    const queryParams: QueueStatusParams = { session_id: this.id }
    if (resolvedObserverId) queryParams.observer_id = resolvedObserverId
    if (resolvedSenderId) queryParams.sender_id = resolvedSenderId

    const status = await this._getQueueStatus(queryParams)
    return transformQueueStatus(status)
  }

  /**
   * Upload a file to this session as a message.
   *
   * Makes an API call to upload a file, which is processed and stored as one or
   * more messages in the session.
   *
   * @param file - The file to upload. Can be a File, Blob, or an object with
   *               filename, content (Buffer/Uint8Array), and content_type
   * @param peer - The peer who is uploading the file (ID string or Peer object)
   * @param options - Upload options
   * @param options.metadata - Optional metadata to associate with the message(s)
   * @param options.configuration - Optional configuration for processing
   * @param options.createdAt - Optional timestamp for the message (string or Date)
   * @returns Promise resolving to an array of Message objects created from the file
   *
   * @example
   * ```typescript
   * // Upload a File object (browser)
   * const messages = await session.uploadFile(fileInput.files[0], user)
   *
   * // Upload from Node.js buffer
   * const messages = await session.uploadFile({
   *   filename: 'document.pdf',
   *   content: fs.readFileSync('document.pdf'),
   *   content_type: 'application/pdf'
   * }, user)
   * ```
   */
  async uploadFile(
    file:
      | File
      | Blob
      | {
          filename: string
          content: Buffer | Uint8Array
          content_type: string
        },
    peer: string | Peer,
    options?: {
      metadata?: Record<string, unknown>
      configuration?: Record<string, unknown>
      createdAt?: string | Date
    }
  ): Promise<Message[]> {
    const createdAt =
      options?.createdAt instanceof Date
        ? options.createdAt.toISOString()
        : options?.createdAt

    const resolvedPeerId = typeof peer === 'string' ? peer : peer.id

    const uploadParams = FileUploadSchema.parse({
      file,
      peer: resolvedPeerId,
      metadata: options?.metadata,
      configuration: options?.configuration,
      createdAt: createdAt,
    })

    const formData = new FormData()

    if (file instanceof File || file instanceof Blob) {
      formData.append('file', file)
    } else {
      // Convert to Uint8Array for Blob compatibility
      const content = new Uint8Array(file.content)
      const blob = new Blob([content], { type: file.content_type })
      formData.append('file', blob, file.filename)
    }

    formData.append('peer_id', resolvedPeerId)
    if (uploadParams.metadata !== undefined && uploadParams.metadata !== null) {
      formData.append('metadata', JSON.stringify(uploadParams.metadata))
    }
    if (
      uploadParams.configuration !== undefined &&
      uploadParams.configuration !== null
    ) {
      formData.append(
        'configuration',
        JSON.stringify(uploadParams.configuration)
      )
    }
    if (
      uploadParams.createdAt !== undefined &&
      uploadParams.createdAt !== null
    ) {
      formData.append('created_at', uploadParams.createdAt)
    }

    const response = await this._uploadFile(formData)
    return response.map(Message.fromApiResponse)
  }

  /**
   * Get a peer's representation scoped to this session.
   *
   * Makes an API call to retrieve the representation for a peer, limited to
   * conclusions derived from this session's messages.
   *
   * @param peer - The peer to get representation for (ID string or Peer object)
   * @param options - Representation options
   * @param options.target - Optional target peer for local representation
   * @param options.searchQuery - Optional semantic search query to filter conclusions
   * @param options.searchTopK - Number of semantically relevant conclusions to return
   * @param options.searchMaxDistance - Maximum semantic distance for search results (0.0-1.0)
   * @param options.includeMostFrequent - Whether to include the most frequent conclusions
   * @param options.maxConclusions - Maximum number of conclusions to include
   * @returns Promise resolving to a string representation containing conclusions
   */
  async representation(
    peer: string | Peer,
    options?: {
      target?: string | Peer
      searchQuery?: string
      searchTopK?: number
      searchMaxDistance?: number
      includeMostFrequent?: boolean
      maxConclusions?: number
    }
  ): Promise<string> {
    const getRepresentationParams = GetRepresentationParamsSchema.parse({
      peer,
      target: options?.target,
      options: {
        searchQuery: options?.searchQuery,
        searchTopK: options?.searchTopK,
        searchMaxDistance: options?.searchMaxDistance,
        includeMostFrequent: options?.includeMostFrequent,
        maxConclusions: options?.maxConclusions,
      },
    })
    const peerId =
      typeof getRepresentationParams.peer === 'string'
        ? getRepresentationParams.peer
        : getRepresentationParams.peer.id
    const targetId = getRepresentationParams.target
      ? typeof getRepresentationParams.target === 'string'
        ? getRepresentationParams.target
        : getRepresentationParams.target.id
      : undefined

    const response = await this._getRepresentation(peerId, {
      session_id: this.id,
      target: targetId,
      search_query: getRepresentationParams.options?.searchQuery,
      search_top_k: getRepresentationParams.options?.searchTopK,
      search_max_distance: getRepresentationParams.options?.searchMaxDistance,
      include_most_frequent:
        getRepresentationParams.options?.includeMostFrequent,
      max_conclusions: getRepresentationParams.options?.maxConclusions,
    })
    return response.representation
  }

  /**
   * Update the metadata of a message in this session.
   *
   * Makes an API call to update the metadata of a specific message.
   *
   * @param message - Either a Message object or a message ID string
   * @param metadata - The metadata to update for the message
   * @returns Promise resolving to the updated Message object
   */
  async updateMessage(
    message: Message | string,
    metadata: Record<string, unknown>
  ): Promise<Message> {
    const validatedMetadata = MessageMetadataSchema.parse(metadata)
    const messageId = typeof message === 'string' ? message : message.id

    const response = await this._updateMessage(messageId, {
      metadata: validatedMetadata ?? {},
    })
    return Message.fromApiResponse(response)
  }

  /**
   * Return a string representation of the Session.
   *
   * @returns A string representation suitable for debugging
   */
  toString(): string {
    return `Session(id='${this.id}')`
  }
}
