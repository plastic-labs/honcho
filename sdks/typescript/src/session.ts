import type { HttpClient } from './http'
import { Page } from './http/pagination'
import { Peer } from './peer'
import {
  Representation,
  type RepresentationData,
  type RepresentationOptions,
} from './representation'
import { SessionContext, SessionSummaries, Summary } from './session_context'
import type {
  DeriverStatus,
  Message,
  PageResponse,
  PeerResponse,
  SessionContextResponse,
  SessionPeerConfigResponse,
  SessionResponse,
  SessionSummariesResponse,
  Uploadable,
} from './types'
import {
  ContextParamsSchema,
  type DeriverStatusOptions,
  FileUploadSchema,
  FilterSchema,
  type Filters,
  LimitSchema,
  type MessageAddition,
  MessageAdditionSchema,
  type PeerAddition,
  PeerAdditionSchema,
  type PeerRemoval,
  PeerRemovalSchema,
  SearchQuerySchema,
  SessionPeerConfigSchema,
  WorkingRepParamsSchema,
} from './validation'

/**
 * Configuration options for a peer within a specific session.
 *
 * Controls how peers interact and observe each other within the context
 * of a particular session, allowing for fine-grained control over
 * representation building and theory-of-mind behaviors.
 */
export class SessionPeerConfig {
  /**
   * Whether other peers in this session should try to form a session-level
   * theory-of-mind representation of this peer. When false, prevents other
   * peers from building local representations of this peer within this session.
   */
  observe_me?: boolean | null

  /**
   * Whether this peer should form session-level theory-of-mind representations
   * of other peers in the session. When false, this peer will not build local
   * representations of other peers within this session.
   */
  observe_others?: boolean | null

  /**
   * Initialize SessionPeerConfig with observation settings.
   *
   * @param observe_me - Whether other peers should observe this peer in the session
   * @param observe_others - Whether this peer should observe others in the session
   */
  constructor(observe_me?: boolean | null, observe_others?: boolean | null) {
    const validatedConfig = SessionPeerConfigSchema.parse({
      observe_me,
      observe_others,
    })
    this.observe_me = validatedConfig.observe_me
    this.observe_others = validatedConfig.observe_others
  }
}

/**
 * Represents a session in the Honcho system.
 *
 * Sessions are scoped to a set of peers and contain messages/content. They create
 * bidirectional relationships between peers and provide a context for multi-party
 * conversations and interactions. Sessions serve as containers for conversations,
 * allowing peers to communicate while maintaining both global and local
 * representations of each other.
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
  /**
   * Reference to the HTTP client instance.
   */
  private _http: HttpClient
  /**
   * Private cached metadata for this session.
   */
  private _metadata?: Record<string, unknown>
  /**
   * Private cached configuration for this session.
   */
  private _configuration?: Record<string, unknown>

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
   * Call getConfig() to get the latest configuration from the server,
   * which will also update this cached value.
   */
  get configuration(): Record<string, unknown> | undefined {
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
    http: HttpClient,
    metadata?: Record<string, unknown>,
    configuration?: Record<string, unknown>
  ) {
    this.id = id
    this.workspaceId = workspaceId
    this._http = http
    this._metadata = metadata
    this._configuration = configuration
  }

  /**
   * Add peers to this session.
   *
   * Makes an API call to add one or more peers to this session. Adding peers
   * creates bidirectional relationships and allows them to participate in
   * the session's conversations.
   *
   * @param peers - Peers to add to the session
   */
  async addPeers(peers: PeerAddition): Promise<void> {
    const validatedPeers = PeerAdditionSchema.parse(peers)
    const peerDict: Record<string, SessionPeerConfig> = {}
    const peersArray = Array.isArray(validatedPeers)
      ? validatedPeers
      : [validatedPeers]

    for (const peer of peersArray) {
      if (typeof peer === 'string') {
        peerDict[peer] = {}
      } else if (Array.isArray(peer)) {
        const peerId = typeof peer[0] === 'string' ? peer[0] : peer[0].id
        peerDict[peerId] = peer[1]
      } else if (typeof peer === 'object' && 'id' in peer) {
        peerDict[peer.id] = {}
      } else {
        throw new Error(`Invalid peer type: ${typeof peer}`)
      }
    }

    await this._http.request<SessionResponse>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/peers`,
      { json: peerDict }
    )
  }

  /**
   * Set the complete peer list for this session.
   *
   * Makes an API call to replace the current peer list with the provided peers.
   *
   * @param peers - Peers to set for the session
   */
  async setPeers(peers: PeerAddition): Promise<void> {
    const validatedPeers = PeerAdditionSchema.parse(peers)
    const peerDict: Record<string, SessionPeerConfig> = {}
    const peersArray = Array.isArray(validatedPeers)
      ? validatedPeers
      : [validatedPeers]

    for (const peer of peersArray) {
      if (typeof peer === 'string') {
        peerDict[peer] = {}
      } else if (Array.isArray(peer)) {
        const peerId = typeof peer[0] === 'string' ? peer[0] : peer[0].id
        peerDict[peerId] = peer[1]
      } else if (typeof peer === 'object' && 'id' in peer) {
        peerDict[peer.id] = {}
      } else {
        throw new Error(`Invalid peer type: ${typeof peer}`)
      }
    }

    await this._http.request<SessionResponse>(
      'PUT',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/peers`,
      { json: peerDict }
    )
  }

  /**
   * Remove peers from this session.
   *
   * Makes an API call to remove one or more peers from this session.
   *
   * @param peers - Peers to remove from the session
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
    await this._http.request<SessionResponse>(
      'DELETE',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/peers`,
      { json: peerIds }
    )
  }

  /**
   * Get all peers in this session.
   *
   * Makes an API call to retrieve the list of peers that are currently
   * members of this session.
   *
   * @returns Promise resolving to a list of Peer objects that are members of this session
   */
  async getPeers(): Promise<Peer[]> {
    const data = await this._http.request<PageResponse<PeerResponse>>(
      'GET',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/peers`
    )
    return data.items.map(
      (peer) => new Peer(peer.id, this.workspaceId, this._http)
    )
  }

  /**
   * Get the configuration for a peer in this session.
   *
   * @param peer - The peer to get configuration for
   * @returns Promise resolving to SessionPeerConfig object with the peer's session settings
   */
  async getPeerConfig(peer: string | Peer): Promise<SessionPeerConfig> {
    const peerId = typeof peer === 'string' ? peer : peer.id
    const data = await this._http.request<SessionPeerConfigResponse>(
      'GET',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/peers/${peerId}/config`
    )
    return new SessionPeerConfig(data.observe_me, data.observe_others)
  }

  /**
   * Set the configuration for a peer in this session.
   *
   * @param peer - The peer to configure
   * @param config - SessionPeerConfig object specifying the observation settings
   */
  async setPeerConfig(
    peer: string | Peer,
    config: SessionPeerConfig
  ): Promise<void> {
    const peerId = typeof peer === 'string' ? peer : peer.id
    const validatedConfig = SessionPeerConfigSchema.parse(config)
    await this._http.request<void>(
      'PUT',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/peers/${peerId}/config`,
      {
        json: {
          observe_others: validatedConfig.observe_others,
          observe_me: validatedConfig.observe_me,
        },
      }
    )
  }

  /**
   * Add one or more messages to this session.
   *
   * @param messages - Messages to add to the session
   */
  async addMessages(messages: MessageAddition): Promise<Message[]> {
    const validatedMessages = MessageAdditionSchema.parse(messages)
    const messagesList = Array.isArray(validatedMessages)
      ? validatedMessages
      : [validatedMessages]
    return await this._http.request<Message[]>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/messages`,
      {
        json: { messages: messagesList },
      }
    )
  }

  /**
   * Get messages from this session with optional filtering.
   *
   * @param filters - Optional filter criteria for messages
   * @returns Promise resolving to a Page of Message objects
   */
  async getMessages(filters?: Filters): Promise<Page<Message, Message>> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const data = await this._http.request<PageResponse<Message>>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/messages/list`,
      {
        json: validatedFilter ? { filters: validatedFilter } : {},
      }
    )

    const fetchNext =
      data.page < (data.pages ?? 0)
        ? async () => {
            const nextData = await this._http.request<PageResponse<Message>>(
              'POST',
              `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/messages/list`,
              {
                json: { filters: validatedFilter, page: data.page + 1 },
              }
            )
            return new Page<Message, Message>(nextData)
          }
        : undefined

    return new Page<Message, Message>(data, undefined, fetchNext)
  }

  /**
   * Get metadata for this session.
   *
   * @returns Promise resolving to a dictionary containing the session's metadata
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const session = await this._http.request<SessionResponse>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/sessions`,
      {
        json: { id: this.id },
      }
    )
    this._metadata = session.metadata || {}
    return this._metadata
  }

  /**
   * Set metadata for this session.
   *
   * @param metadata - A dictionary of metadata to associate with this session
   */
  async setMetadata(metadata: Record<string, unknown>): Promise<void> {
    await this._http.request<SessionResponse>(
      'PUT',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}`,
      {
        json: { metadata },
      }
    )
    this._metadata = metadata
  }

  /**
   * Get configuration for this session.
   *
   * @returns Promise resolving to a dictionary containing the session's configuration
   */
  async getConfig(): Promise<Record<string, unknown>> {
    const session = await this._http.request<SessionResponse>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/sessions`,
      {
        json: { id: this.id },
      }
    )
    this._configuration = session.configuration || {}
    return this._configuration
  }

  /**
   * Set configuration for this session.
   *
   * @param configuration - A dictionary of configuration to associate with this session
   */
  async setConfig(configuration: Record<string, unknown>): Promise<void> {
    await this._http.request<SessionResponse>(
      'PUT',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}`,
      {
        json: { configuration },
      }
    )
    this._configuration = configuration
  }

  /**
   * Refresh cached metadata and configuration for this session.
   */
  async refresh(): Promise<void> {
    const session = await this._http.request<SessionResponse>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/sessions`,
      {
        json: { id: this.id },
      }
    )
    this._metadata = session.metadata || {}
    this._configuration = session.configuration || {}
  }

  /**
   * Delete this session and all associated data.
   */
  async delete(): Promise<void> {
    await this._http.request<void>(
      'DELETE',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}`
    )
  }

  /**
   * Clone this session, optionally up to a specific message.
   *
   * @param messageId - Optional message ID to cut off the clone at
   * @returns Promise resolving to a new Session object representing the cloned session
   */
  async clone(messageId?: string): Promise<Session> {
    const clonedSessionData = await this._http.request<SessionResponse>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/clone`,
      {
        params: messageId ? { message_id: messageId } : undefined,
      }
    )

    return new Session(
      clonedSessionData.id,
      this.workspaceId,
      this._http,
      clonedSessionData.metadata ?? undefined,
      clonedSessionData.configuration ?? undefined
    )
  }

  /**
   * Get optimized context for this session within a token limit.
   */
  async getContext(
    summary?: boolean,
    tokens?: number,
    peerTarget?: string | Peer,
    lastUserMessage?: string | Message,
    peerPerspective?: string | Peer,
    representationOptions?: RepresentationOptions
  ): Promise<SessionContext>
  async getContext(options?: {
    summary?: boolean
    tokens?: number
    peerTarget?: string | Peer
    lastUserMessage?: string | Message
    peerPerspective?: string | Peer
    limitToSession?: boolean
    representationOptions?: RepresentationOptions
  }): Promise<SessionContext>
  async getContext(
    summaryOrOptions?:
      | boolean
      | {
          summary?: boolean
          tokens?: number
          peerTarget?: string | Peer
          lastUserMessage?: string | Message
          peerPerspective?: string | Peer
          limitToSession?: boolean
          representationOptions?: RepresentationOptions
        },
    tokens?: number,
    peerTarget?: string | Peer,
    lastUserMessage?: string | Message,
    peerPerspective?: string | Peer,
    representationOptions?: RepresentationOptions
  ): Promise<SessionContext> {
    // Normalize positional arguments into options object
    let options: {
      summary?: boolean
      tokens?: number
      peerTarget?: string
      lastUserMessage?: string
      peerPerspective?: string
      limitToSession?: boolean
      representationOptions?: RepresentationOptions
    }

    if (
      typeof summaryOrOptions === 'boolean' ||
      // biome-ignore lint/complexity/noArguments: Need to detect which overload pattern is being used
      (summaryOrOptions === undefined && arguments.length > 1)
    ) {
      // Positional arguments pattern
      options = {
        summary: summaryOrOptions as boolean | undefined,
        tokens,
        peerTarget: typeof peerTarget === 'object' ? peerTarget.id : peerTarget,
        lastUserMessage:
          typeof lastUserMessage === 'string'
            ? lastUserMessage
            : lastUserMessage?.id,
        peerPerspective:
          typeof peerPerspective === 'object'
            ? peerPerspective.id
            : peerPerspective,
        representationOptions,
      }
    } else {
      // Options object pattern
      options = (summaryOrOptions as typeof options) || {}
    }

    const contextParams = ContextParamsSchema.parse({
      summary: options.summary,
      tokens: options.tokens,
      peerTarget: options.peerTarget,
      lastUserMessage: options.lastUserMessage,
      peerPerspective: options.peerPerspective,
      limitToSession: options.limitToSession,
      representationOptions: options.representationOptions,
    })

    // Extract message ID if lastUserMessage is a Message object
    const lastMessageId =
      typeof contextParams.lastUserMessage === 'string'
        ? contextParams.lastUserMessage
        : contextParams.lastUserMessage?.id

    const params: Record<string, string | number | boolean | undefined> = {}
    if (contextParams.tokens !== undefined) params.tokens = contextParams.tokens
    if (contextParams.summary !== undefined)
      params.summary = contextParams.summary
    if (lastMessageId) params.last_message = lastMessageId
    if (contextParams.peerTarget) params.peer_target = contextParams.peerTarget
    if (contextParams.peerPerspective)
      params.peer_perspective = contextParams.peerPerspective
    if (contextParams.limitToSession !== undefined)
      params.limit_to_session = contextParams.limitToSession
    if (contextParams.representationOptions?.searchTopK !== undefined)
      params.search_top_k = contextParams.representationOptions.searchTopK
    if (contextParams.representationOptions?.searchMaxDistance !== undefined)
      params.search_max_distance =
        contextParams.representationOptions.searchMaxDistance
    if (contextParams.representationOptions?.includeMostDerived !== undefined)
      params.include_most_derived =
        contextParams.representationOptions.includeMostDerived
    if (contextParams.representationOptions?.maxObservations !== undefined)
      params.max_observations =
        contextParams.representationOptions.maxObservations

    const context = await this._http.request<SessionContextResponse>(
      'GET',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/context`,
      { params }
    )
    // Convert the summary response to Summary object if present
    const summaryObj = context.summary ? new Summary(context.summary) : null
    return new SessionContext(
      this.id,
      context.messages,
      summaryObj,
      context.peer_representation
        ? JSON.stringify(context.peer_representation)
        : null,
      context.peer_card ?? null
    )
  }

  /**
   * Get available summaries for this session.
   *
   * @returns Promise resolving to a SessionSummaries object
   */
  async getSummaries(): Promise<SessionSummaries> {
    const data = await this._http.request<SessionSummariesResponse>(
      'GET',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/summaries`
    )
    return new SessionSummaries(data)
  }

  /**
   * Search for messages in this session.
   *
   * @param query The search query to use
   * @param options.filters - Optional filters to scope the search
   * @param options.limit Number of results to return (1-100, default: 10)
   * @returns A list of Message objects representing the search results
   */
  async search(
    query: string,
    options?: {
      filters?: Filters
      limit?: number
    }
  ): Promise<Message[]> {
    const validatedQuery = SearchQuerySchema.parse(query)
    const validatedFilters = options?.filters
      ? FilterSchema.parse(options.filters)
      : undefined
    const validatedLimit = options?.limit
      ? LimitSchema.parse(options.limit)
      : undefined
    return await this._http.request<Message[]>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/search`,
      {
        json: {
          query: validatedQuery,
          filters: validatedFilters,
          limit: validatedLimit,
        },
      }
    )
  }

  /**
   * Get the deriver processing status for this session.
   *
   * @param options - Configuration options for the status request
   * @returns Promise resolving to the deriver status information
   */
  async getDeriverStatus(
    options?: Omit<
      DeriverStatusOptions,
      'sessionId' | 'observerId' | 'senderId'
    > & {
      observer?: string | Peer
      sender?: string | Peer
    }
  ): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: DeriverStatus['sessions']
  }> {
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

    const params: Record<string, string | undefined> = {
      session_id: this.id,
    }
    if (resolvedObserverId) params.observer_id = resolvedObserverId
    if (resolvedSenderId) params.sender_id = resolvedSenderId

    const status = await this._http.request<DeriverStatus>(
      'GET',
      `/v2/workspaces/${this.workspaceId}/queue/status`,
      { params }
    )

    return {
      totalWorkUnits: status.total_work_units,
      completedWorkUnits: status.completed_work_units,
      inProgressWorkUnits: status.in_progress_work_units,
      pendingWorkUnits: status.pending_work_units,
      sessions: status.sessions || undefined,
    }
  }

  /**
   * Poll getDeriverStatus until pending_work_units and in_progress_work_units are both 0.
   *
   * @param options - Configuration options for the status request
   * @returns Promise resolving to the final deriver status when processing is complete
   * @throws Error if timeout is exceeded before processing completes
   */
  async pollDeriverStatus(
    options?: Omit<
      DeriverStatusOptions,
      'sessionId' | 'observerId' | 'senderId'
    > & {
      observer?: string | Peer
      sender?: string | Peer
    }
  ): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: DeriverStatus['sessions']
  }> {
    const timeoutMs = options?.timeoutMs ?? 300000 // Default to 5 minutes
    const startTime = Date.now()

    while (true) {
      const status = await this.getDeriverStatus(options)
      if (status.pendingWorkUnits === 0 && status.inProgressWorkUnits === 0) {
        return status
      }

      // Check if timeout has been exceeded
      const elapsedTime = Date.now() - startTime
      if (elapsedTime >= timeoutMs) {
        throw new Error(
          `Polling timeout exceeded after ${timeoutMs}ms. ` +
            `Current status: ${status.pendingWorkUnits} pending, ${status.inProgressWorkUnits} in progress work units.`
        )
      }

      // Sleep for the expected time to complete all current work units
      const totalWorkUnits =
        status.pendingWorkUnits + status.inProgressWorkUnits
      const sleepMs = Math.max(1000, totalWorkUnits * 1000)

      // Ensure we don't sleep past the timeout
      const remainingTime = timeoutMs - elapsedTime
      const actualSleepMs = Math.min(sleepMs, remainingTime)

      if (actualSleepMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, actualSleepMs))
      }
    }
  }

  /**
   * Upload a file to create messages in this session.
   *
   * @param file - File to upload
   * @param peer - The peer to attribute the created messages to
   * @param options - Optional parameters for the uploaded messages
   * @returns Promise resolving to a list of Message objects representing the created messages
   */
  async uploadFile(
    file: Uploadable,
    peer: string | Peer,
    options?: {
      metadata?: Record<string, unknown>
      configuration?: Record<string, unknown>
      created_at?: string | Date
    }
  ): Promise<Message[]> {
    const createdAt =
      options?.created_at instanceof Date
        ? options.created_at.toISOString()
        : options?.created_at

    const resolvedPeerId = typeof peer === 'string' ? peer : peer.id

    const uploadParams = FileUploadSchema.parse({
      file,
      peer: resolvedPeerId,
      metadata: options?.metadata,
      configuration: options?.configuration,
      created_at: createdAt,
    })

    // Build FormData for file upload
    const formData = new FormData()

    // Handle file based on type
    if (file instanceof File) {
      formData.append('file', file)
    } else if ('filename' in file && 'content' in file) {
      // Convert Buffer/Uint8Array to Blob
      const blob = new Blob([file.content as BlobPart], {
        type: file.content_type,
      })
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
      uploadParams.created_at !== undefined &&
      uploadParams.created_at !== null
    ) {
      formData.append('created_at', uploadParams.created_at)
    }

    return await this._http.request<Message[]>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/sessions/${this.id}/messages/upload`,
      { formData }
    )
  }

  /**
   * Get the current working representation of a peer in this session.
   *
   * @param peer - The peer to get the working representation of
   * @param target - Optional target peer for theory-of-mind representation
   * @param options - Optional representation options
   * @returns Promise resolving to a Representation object
   */
  async workingRep(
    peer: string | Peer,
    target?: string | Peer,
    options?: {
      searchQuery?: string
      searchTopK?: number
      searchMaxDistance?: number
      includeMostDerived?: boolean
      maxObservations?: number
    }
  ): Promise<Representation> {
    const workingRepParams = WorkingRepParamsSchema.parse({
      peer,
      target,
      options,
    })
    const peerId =
      typeof workingRepParams.peer === 'string'
        ? workingRepParams.peer
        : workingRepParams.peer.id
    const targetId = workingRepParams.target
      ? typeof workingRepParams.target === 'string'
        ? workingRepParams.target
        : workingRepParams.target.id
      : undefined

    const response = await this._http.request<
      RepresentationData | { representation?: RepresentationData | null }
    >(
      'POST',
      `/v2/workspaces/${this.workspaceId}/peers/${peerId}/representation`,
      {
        json: {
          session_id: this.id,
          target: targetId,
          search_query: workingRepParams.options?.searchQuery,
          search_top_k: workingRepParams.options?.searchTopK,
          search_max_distance: workingRepParams.options?.searchMaxDistance,
          include_most_derived: workingRepParams.options?.includeMostDerived,
          max_observations: workingRepParams.options?.maxObservations,
        },
      }
    )
    const maybe = response as
      | RepresentationData
      | { representation?: RepresentationData | null }
      | null
    const rep = (maybe && typeof maybe === 'object' && 'representation' in maybe
      ? (maybe as { representation?: RepresentationData | null }).representation
      : maybe) ?? { explicit: [], deductive: [] }
    return Representation.fromData(rep as RepresentationData)
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
