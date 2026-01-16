import { API_VERSION } from './api-version'
import { ConclusionScope } from './conclusions'
import type { HonchoHTTPClient } from './http/client'
import {
  createDialecticStream,
  type DialecticStreamResponse,
} from './http/streaming'
import { Message, type MessageInput } from './message'
import { Page } from './pagination'
import { Session } from './session'
import type {
  MessageResponse,
  PageResponse,
  PeerCardResponse,
  PeerChatResponse,
  PeerContextResponse,
  PeerResponse,
  RepresentationResponse,
  SessionResponse,
} from './types/api'
import {
  CardTargetSchema,
  ChatQuerySchema,
  FilterSchema,
  type Filters,
  LimitSchema,
  MessageContentSchema,
  MessageMetadataSchema,
  PeerConfigSchema,
  PeerGetRepresentationParamsSchema,
  PeerMetadataSchema,
  SearchQuerySchema,
} from './validation'

/**
 * Represents context for a peer, including representation and peer card.
 *
 * This class wraps the API response with camelCase properties for consistency
 * with the rest of the SDK.
 */
export class PeerContext {
  /**
   * The peer ID this context belongs to.
   */
  readonly peerId: string

  /**
   * The target peer ID if this is a local context.
   */
  readonly targetId: string

  /**
   * The peer's representation, if available.
   */
  readonly representation: string | null

  /**
   * The peer card, if available.
   */
  readonly peerCard: string[] | null

  constructor(
    peerId: string,
    targetId: string,
    representation: string | null,
    peerCard: string[] | null
  ) {
    this.peerId = peerId
    this.targetId = targetId
    this.representation = representation
    this.peerCard = peerCard
  }

  /**
   * Create a PeerContext from an API response.
   */
  static fromApiResponse(response: PeerContextResponse): PeerContext {
    return new PeerContext(
      response.peer_id,
      response.target_id,
      response.representation,
      response.peer_card
    )
  }

  toString(): string {
    return `PeerContext(peerId='${this.peerId}', targetId='${this.targetId}')`
  }
}

/**
 * Represents a peer in the Honcho system.
 *
 * Peers can send messages, participate in sessions, and maintain both global
 * and local representations for contextual interactions. A peer represents
 * an entity (user, assistant, etc.) that can communicate within the system.
 */
export class Peer {
  /**
   * Unique identifier for this peer.
   */
  readonly id: string
  /**
   * Workspace ID for scoping operations.
   */
  readonly workspaceId: string
  /**
   * Reference to the HTTP client instance.
   */
  private _http: HonchoHTTPClient
  /**
   * Private cached metadata for this peer.
   */
  private _metadata?: Record<string, unknown>
  /**
   * Private cached configuration for this peer.
   */
  private _configuration?: Record<string, unknown>

  /**
   * Cached metadata for this peer. May be stale if the peer
   * was not recently fetched from the API.
   *
   * Call getMetadata() to get the latest metadata from the server,
   * which will also update this cached value.
   */
  get metadata(): Record<string, unknown> | undefined {
    return this._metadata
  }

  /**
   * Cached configuration for this peer. May be stale if the peer
   * was not recently fetched from the API.
   *
   * Call getConfiguration() to get the latest configuration from the server,
   * which will also update this cached value.
   */
  get configuration(): Record<string, unknown> | undefined {
    return this._configuration
  }

  /**
   * Initialize a new Peer. **Do not call this directly, use the client.peer() method instead.**
   *
   * @param id - Unique identifier for this peer within the workspace
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
    configuration?: Record<string, unknown>
  ) {
    this.id = id
    this.workspaceId = workspaceId
    this._http = http
    this._metadata = metadata
    this._configuration = configuration
  }

  // ===========================================================================
  // Private API Methods
  // ===========================================================================

  private async _getOrCreate(params: {
    id: string
    metadata?: Record<string, unknown>
    configuration?: Record<string, unknown>
  }): Promise<PeerResponse> {
    return this._http.post<PeerResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers`,
      { body: params }
    )
  }

  private async _update(params: {
    metadata?: Record<string, unknown>
    configuration?: Record<string, unknown>
  }): Promise<PeerResponse> {
    return this._http.put<PeerResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${this.id}`,
      { body: params }
    )
  }

  private async _listSessions(params?: {
    filters?: Record<string, unknown>
    page?: number
    size?: number
  }): Promise<PageResponse<SessionResponse>> {
    return this._http.post<PageResponse<SessionResponse>>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${this.id}/sessions`,
      {
        body: { filters: params?.filters },
        query: { page: params?.page, size: params?.size },
      }
    )
  }

  private async _chat(params: {
    query: string
    stream?: boolean
    target?: string
    session_id?: string
    reasoning_level?: string
  }): Promise<PeerChatResponse> {
    return this._http.post<PeerChatResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${this.id}/chat`,
      { body: params }
    )
  }

  private async _chatStream(params: {
    query: string
    target?: string
    session_id?: string
    reasoning_level?: string
  }): Promise<Response> {
    return this._http.stream(
      'POST',
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${this.id}/chat`,
      {
        body: {
          ...params,
          stream: true,
        },
      }
    )
  }

  private async _search(params: {
    query: string
    filters?: Record<string, unknown>
    limit?: number
  }): Promise<MessageResponse[]> {
    return this._http.post<MessageResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${this.id}/search`,
      { body: params }
    )
  }

  private async _getRepresentation(params: {
    session_id?: string
    target?: string
    search_query?: string
    search_top_k?: number
    search_max_distance?: number
    include_most_frequent?: boolean
    max_conclusions?: number
  }): Promise<RepresentationResponse> {
    return this._http.post<RepresentationResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${this.id}/representation`,
      { body: params }
    )
  }

  private async _getContext(params: {
    target?: string
    search_query?: string
    search_top_k?: number
    search_max_distance?: number
    include_most_frequent?: boolean
    max_conclusions?: number
  }): Promise<PeerContextResponse> {
    return this._http.get<PeerContextResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${this.id}/context`,
      { query: params }
    )
  }

  private async _getCard(params: {
    target?: string
  }): Promise<PeerCardResponse> {
    return this._http.get<PeerCardResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${this.id}/card`,
      { query: params }
    )
  }

  // ===========================================================================
  // Public Methods
  // ===========================================================================

  /**
   * Query the peer's representation with a natural language question.
   *
   * Makes an API call to the Honcho dialectic endpoint to query either the peer's
   * global representation (all content associated with this peer) or their local
   * representation of another peer (what this peer knows about the target peer).
   *
   * @param query - The natural language question to ask
   * @param options.target - Optional target peer for local representation query. If provided,
   *                         queries what this peer knows about the target peer rather than
   *                         querying the peer's global representation. Can be a peer ID string
   *                         or a Peer object.
   * @param options.session - Optional session to scope the query to. If provided, only
   *                          information from that session is considered. Can be a session
   *                          ID string or a Session object.
   * @param options.reasoningLevel - Optional reasoning level for the query: "minimal", "low", "medium",
   *                                 "high", or "max". Defaults to "low" if not provided.
   * @returns Promise resolving to the response string, or null if no relevant information
   *
   * @example
   * ```typescript
   * // Simple query
   * const response = await peer.chat('What do you know about this user?')
   *
   * // Query with options
   * const response = await peer.chat('What does this peer think about coding?', {
   *   target: otherPeer,
   *   reasoningLevel: 'high'
   * })
   * ```
   */
  async chat(
    query: string,
    options?: {
      target?: string | Peer
      session?: string | Session
      reasoningLevel?: string
    }
  ): Promise<string | null> {
    const targetId = options?.target
      ? typeof options.target === 'string'
        ? options.target
        : options.target.id
      : undefined
    const resolvedSessionId = options?.session
      ? typeof options.session === 'string'
        ? options.session
        : options.session.id
      : undefined

    const chatParams = ChatQuerySchema.parse({
      query,
      target: targetId,
      session: resolvedSessionId,
      reasoningLevel: options?.reasoningLevel,
    })

    const response = await this._chat({
      query: chatParams.query,
      stream: false,
      target: chatParams.target,
      session_id: chatParams.session,
      reasoning_level: chatParams.reasoningLevel,
    })
    if (!response.content || response.content === 'None') {
      return null
    }
    return response.content
  }

  /**
   * Query the peer's representation with a natural language question and stream the response.
   *
   * Makes an API call to the Honcho dialectic endpoint to query either the peer's
   * global representation (all content associated with this peer) or their local
   * representation of another peer (what this peer knows about the target peer).
   * The response is streamed back as it is generated.
   *
   * @param query - The natural language question to ask
   * @param options.target - Optional target peer for local representation query. If provided,
   *                         queries what this peer knows about the target peer rather than
   *                         querying the peer's global representation. Can be a peer ID string
   *                         or a Peer object.
   * @param options.session - Optional session to scope the query to. If provided, only
   *                          information from that session is considered. Can be a session
   *                          ID string or a Session object.
   * @param options.reasoningLevel - Optional reasoning level for the query: "minimal", "low", "medium",
   *                                 "high", or "max". Defaults to "low" if not provided.
   * @returns Promise resolving to a DialecticStreamResponse that can be iterated over
   *
   * @example
   * ```typescript
   * // Stream a response
   * const stream = await peer.chatStream('What do you know about this user?')
   * for await (const chunk of stream) {
   *   process.stdout.write(chunk)
   * }
   *
   * // Stream with options
   * const stream = await peer.chatStream('What does this peer think about coding?', {
   *   target: otherPeer,
   *   reasoningLevel: 'high'
   * })
   * ```
   */
  async chatStream(
    query: string,
    options?: {
      target?: string | Peer
      session?: string | Session
      reasoningLevel?: string
    }
  ): Promise<DialecticStreamResponse> {
    const targetId = options?.target
      ? typeof options.target === 'string'
        ? options.target
        : options.target.id
      : undefined
    const resolvedSessionId = options?.session
      ? typeof options.session === 'string'
        ? options.session
        : options.session.id
      : undefined

    const chatParams = ChatQuerySchema.parse({
      query,
      target: targetId,
      session: resolvedSessionId,
      reasoningLevel: options?.reasoningLevel,
    })

    const response = await this._chatStream({
      query: chatParams.query,
      target: chatParams.target,
      session_id: chatParams.session,
      reasoning_level: chatParams.reasoningLevel,
    })

    return createDialecticStream(response)
  }

  /**
   * Get all sessions this peer is a member of.
   *
   * Makes an API call to retrieve all sessions where this peer is an active participant.
   * Sessions are created when peers are added to them or send messages to them.
   *
   * @param filters - Optional filter criteria for sessions. See [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
   * @returns Promise resolving to a paginated list of Session objects this peer belongs to.
   *          Returns an empty list if the peer is not a member of any sessions
   */
  async sessions(filters?: Filters): Promise<Page<Session, SessionResponse>> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const sessionsPage = await this._listSessions({ filters: validatedFilter })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<SessionResponse>> => {
      return this._listSessions({ filters: validatedFilter, page, size })
    }

    return new Page(
      sessionsPage,
      (session) =>
        new Session(
          session.id,
          this.workspaceId,
          this._http,
          session.metadata ?? undefined,
          session.configuration ?? undefined
        ),
      fetchNextPage
    )
  }

  /**
   * Build a message object attributed to this peer (synchronous, no API call).
   *
   * This is a convenience method for creating message objects with this peer's ID
   * already set. The returned object can then be passed to `session.addMessages()`.
   *
   * **Note:** This method is synchronous and does NOT send the message to Honcho.
   * To actually create the message on the server, pass the returned object to
   * `session.addMessages()`.
   *
   * @param content - The text content for the message
   * @param options.metadata - Optional metadata to associate with the message
   * @param options.configuration - Optional message-level configuration (e.g., reasoning settings)
   * @param options.created_at - Optional ISO 8601 timestamp for the message
   * @returns A message object ready to be passed to `session.addMessages()`
   *
   * @example
   * ```typescript
   * const msg = peer.message('Hello!')
   * await session.addMessages(msg)
   *
   * // Or batch multiple messages:
   * await session.addMessages([
   *   alice.message('Hi Bob'),
   *   bob.message('Hey Alice!')
   * ])
   * ```
   */
  message(
    content: string,
    options?: {
      metadata?: Record<string, unknown>
      configuration?: Record<string, unknown>
      createdAt?: string | Date
    }
  ): MessageInput {
    const validatedContent = MessageContentSchema.parse(content)
    const validatedMetadata = options?.metadata
      ? MessageMetadataSchema.parse(options.metadata)
      : undefined

    const createdAt =
      options?.createdAt instanceof Date
        ? options.createdAt.toISOString()
        : options?.createdAt

    return {
      peerId: this.id,
      content: validatedContent,
      metadata: validatedMetadata,
      configuration: options?.configuration,
      createdAt,
    }
  }

  /**
   * Get the current metadata for this peer.
   *
   * Makes an API call to retrieve metadata associated with this peer. Metadata
   * can include custom attributes, settings, or any other key-value data
   * associated with the peer. This method also updates the cached metadata property.
   *
   * @returns Promise resolving to a dictionary containing the peer's metadata.
   *          Returns an empty dictionary if no metadata is set
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const peer = await this._getOrCreate({ id: this.id })
    this._metadata = peer.metadata || {}
    return this._metadata
  }

  /**
   * Set the metadata for this peer.
   *
   * Makes an API call to update the metadata associated with this peer.
   * This will overwrite any existing metadata with the provided values.
   * This method also updates the cached metadata property.
   *
   * @param metadata - A dictionary of metadata to associate with this peer.
   *                   Keys must be strings, values can be any JSON-serializable type
   */
  async setMetadata(metadata: Record<string, unknown>): Promise<void> {
    const validatedMetadata = PeerMetadataSchema.parse(metadata)
    await this._update({ metadata: validatedMetadata })
    this._metadata = validatedMetadata
  }

  /**
   * Get the current workspace-level configuration for this peer.
   *
   * Makes an API call to retrieve configuration associated with this peer.
   * Configuration currently includes one optional flag, `observe_me`.
   * This method also updates the cached configuration property.
   *
   * @returns Promise resolving to a dictionary containing the peer's configuration
   */
  async getConfiguration(): Promise<Record<string, unknown>> {
    const peer = await this._getOrCreate({ id: this.id })
    this._configuration = peer.configuration || {}
    return this._configuration
  }

  /**
   * Set the configuration for this peer. Currently the only supported configuration
   * value is the `observe_me` flag, which controls whether derivation tasks
   * should be created for this peer's global representation. Default is True.
   *
   * Makes an API call to update the configuration associated with this peer.
   * This will overwrite any existing configuration with the provided values.
   * This method also updates the cached configuration property.
   *
   * @param configuration - A dictionary of configuration to associate with this peer.
   *                        Keys must be strings, values can be any JSON-serializable type
   */
  async setConfiguration(
    configuration: Record<string, unknown>
  ): Promise<void> {
    const validatedConfig = PeerConfigSchema.parse(configuration)
    await this._update({ configuration: validatedConfig })
    this._configuration = validatedConfig
  }

  /**
   * Refresh cached metadata and configuration for this peer.
   *
   * Makes a single API call to retrieve the latest metadata and configuration
   * associated with this peer and updates the cached properties.
   */
  async refresh(): Promise<void> {
    const peer = await this._getOrCreate({ id: this.id })
    this._metadata = peer.metadata || {}
    this._configuration = peer.configuration || {}
  }

  /**
   * Search for messages in the workspace with this peer as author.
   *
   * Makes an API call to search endpoint.
   *
   * @param query The search query to use
   * @param filters - Optional filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
   * @param limit - Optional limit on the number of results to return.
   * @returns Promise resolving to an array of Message objects representing the search results.
   *          Returns an empty array if no messages are found.
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
   * Get the peer card for this peer.
   *
   * Makes an API call to retrieve the peer card, which contains a representation
   * of what this peer knows. If a target is provided, returns this peer's local
   * representation of the target peer.
   *
   * @param target - Optional target peer for local card. If provided, returns this
   *                 peer's card of the target peer. Can be a Peer object or peer ID string.
   * @returns Promise resolving to an array of strings containing the peer card items,
   *          or null if no peer card exists
   */
  async card(target?: string | Peer): Promise<string[] | null> {
    const validatedTarget = CardTargetSchema.parse(target)

    const response = await this._getCard({
      target: validatedTarget,
    })

    return response.peer_card
  }

  /**
   * Get a subset of Honcho's Representation of a peer.
   *
   * Makes an API call to retrieve the representation for this peer.
   *
   * @param options.session - Optional session to scope the representation to.
   * @param options.target - Optional target peer to get the representation of. If provided,
   *                         returns the representation of the target from the perspective of this peer.
   * @param options.searchQuery - Optional semantic search query to filter relevant conclusions.
   * @param options.searchTopK - Number of semantically relevant conclusions to return.
   * @param options.searchMaxDistance - Maximum semantic distance for search results (0.0-1.0).
   * @param options.includeMostFrequent - Whether to include the most frequent conclusions.
   * @param options.maxConclusions - Maximum number of conclusions to include.
   * @returns Promise resolving to a string representation containing conclusions
   *
   * @example
   * ```typescript
   * // Get global representation
   * const globalRep = await peer.representation()
   *
   * // Get representation scoped to a session
   * const sessionRep = await peer.representation({ session: 'session-123' })
   *
   * // Get representation with semantic search
   * const searchedRep = await peer.representation({
   *   searchQuery: 'preferences',
   *   searchTopK: 10,
   *   maxConclusions: 50
   * })
   * ```
   */
  async representation(options?: {
    session?: string | Session
    target?: string | Peer
    searchQuery?: string
    searchTopK?: number
    searchMaxDistance?: number
    includeMostFrequent?: boolean
    maxConclusions?: number
  }): Promise<string> {
    const getRepresentationParams = PeerGetRepresentationParamsSchema.parse({
      session: options?.session,
      target: options?.target,
      options: {
        searchQuery: options?.searchQuery,
        searchTopK: options?.searchTopK,
        searchMaxDistance: options?.searchMaxDistance,
        includeMostFrequent: options?.includeMostFrequent,
        maxConclusions: options?.maxConclusions,
      },
    })
    const sessionId = getRepresentationParams.session
      ? typeof getRepresentationParams.session === 'string'
        ? getRepresentationParams.session
        : getRepresentationParams.session.id
      : undefined
    const targetId = getRepresentationParams.target
      ? typeof getRepresentationParams.target === 'string'
        ? getRepresentationParams.target
        : getRepresentationParams.target.id
      : undefined

    const response = await this._getRepresentation({
      session_id: sessionId,
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
   * Get context for this peer, including representation and peer card.
   *
   * This is a convenience method that retrieves both the working representation
   * and peer card in a single API call.
   *
   * @param options.target - Optional target peer to get context for. If provided, returns
   *                         the context for the target from this peer's perspective.
   * @param options.searchQuery - Optional semantic search query to filter relevant conclusions.
   * @param options.searchTopK - Number of semantically relevant conclusions to return.
   * @param options.searchMaxDistance - Maximum semantic distance for search results (0.0-1.0).
   * @param options.includeMostFrequent - Whether to include the most frequent conclusions.
   * @param options.maxConclusions - Maximum number of conclusions to include.
   * @returns Promise resolving to a PeerContext object containing representation and peer card
   *
   * @example
   * ```typescript
   * // Get own context
   * const context = await peer.context()
   * console.log(context.representation?.toString())
   * console.log(context.peerCard)
   *
   * // Get context for another peer
   * const context = await peer.context({ target: 'other-peer-id' })
   *
   * // Get context with semantic search
   * const context = await peer.context({
   *   searchQuery: 'preferences',
   *   searchTopK: 10
   * })
   * ```
   */
  async context(options?: {
    target?: string | Peer
    searchQuery?: string
    searchTopK?: number
    searchMaxDistance?: number
    includeMostFrequent?: boolean
    maxConclusions?: number
  }): Promise<PeerContext> {
    const targetId = options?.target
      ? typeof options.target === 'string'
        ? options.target
        : options.target.id
      : undefined

    const response = await this._getContext({
      target: targetId,
      search_query: options?.searchQuery,
      search_top_k: options?.searchTopK,
      search_max_distance: options?.searchMaxDistance,
      include_most_frequent: options?.includeMostFrequent,
      max_conclusions: options?.maxConclusions,
    })

    return PeerContext.fromApiResponse(response)
  }

  /**
   * Access this peer's self-conclusions (where observer == observed == self).
   *
   * This property provides a convenient way to access conclusions that this peer
   * has made about themselves. Use this for self-conclusion scenarios.
   *
   * @returns A ConclusionScope scoped to this peer's self-conclusions
   *
   * @example
   * ```typescript
   * // List self-conclusions
   * const obsList = await peer.conclusions.list()
   *
   * // Search self-conclusions
   * const results = await peer.conclusions.query('preferences')
   *
   * // Delete a self-conclusion
   * await peer.conclusions.delete('obs-123')
   * ```
   */
  get conclusions(): ConclusionScope {
    return new ConclusionScope(this._http, this.workspaceId, this.id, this.id)
  }

  /**
   * Access conclusions this peer has made about another peer.
   *
   * This method provides scoped access to conclusions where this peer is the
   * observer and the target is the observed peer.
   *
   * @param target - The target peer (either a Peer object or peer ID string)
   * @returns A ConclusionScope scoped to this peer's conclusions of the target
   *
   * @example
   * ```typescript
   * // Get conclusions about another peer
   * const bobConclusions = peer.conclusionsOf('bob')
   *
   * // List conclusions
   * const obsList = await bobConclusions.list()
   *
   * // Search conclusions
   * const results = await bobConclusions.query('work history')
   *
   * // Get the representation from these conclusions
   * const rep = await bobConclusions.getRepresentation()
   * ```
   */
  conclusionsOf(target: string | Peer): ConclusionScope {
    const targetId = typeof target === 'string' ? target : target.id
    return new ConclusionScope(this._http, this.workspaceId, this.id, targetId)
  }

  /**
   * Return a string representation of the Peer.
   *
   * @returns A string representation suitable for debugging
   */
  toString(): string {
    return `Peer(id='${this.id}')`
  }
}
