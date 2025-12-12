import type HonchoCore from '@honcho-ai/core'
import type { Message } from '@honcho-ai/core/resources/workspaces/sessions/messages'
import { ObservationScope } from './observations'
import { Page } from './pagination'
import {
  Representation,
  type RepresentationData,
  type RepresentationOptions,
} from './representation'
import { Session } from './session'
import { type DialecticStreamChunk, DialecticStreamResponse } from './types'
import {
  ChatQuerySchema,
  FilterSchema,
  type Filters,
  LimitSchema,
  MessageContentSchema,
  MessageMetadataSchema,
  PeerWorkingRepParamsSchema,
  SearchQuerySchema,
  type MessageCreate as ValidatedMessageCreate,
} from './validation'

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
   * Reference to the parent Honcho client instance.
   */
  private _client: HonchoCore
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
   * Call getConfig() to get the latest configuration from the server,
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
   * @param client - Reference to the parent Honcho client instance
   * @param metadata - Optional metadata to initialize the cached value
   * @param configuration - Optional configuration to initialize the cached value
   */
  constructor(
    id: string,
    workspaceId: string,
    client: HonchoCore,
    metadata?: Record<string, unknown>,
    configuration?: Record<string, unknown>
  ) {
    this.id = id
    this.workspaceId = workspaceId
    this._client = client
    this._metadata = metadata
    this._configuration = configuration
  }

  /**
   * Query the peer's representation with a natural language question.
   *
   * Makes an API call to the Honcho dialectic endpoint to query either the peer's
   * global representation (all content associated with this peer) or their local
   * representation of another peer (what this peer knows about the target peer).
   *
   * @param query - The natural language question to ask
   * @param stream - Whether to stream the response
   * @param target - Optional target peer for local representation query. If provided,
   *                 queries what this peer knows about the target peer rather than
   *                 querying the peer's global representation. Can be a peer ID string
   *                 or a Peer object.
   * @param session - Optional session to scope the query to. If provided, only
   *                  information from that session is considered. Can be a session
   *                  ID string or a Session object.
   * @returns Promise resolving to:
   *          - For non-streaming: response string or null if no relevant information
   *          - For streaming: DialecticStreamResponse that can be iterated over
   */
  async chat(
    query: string,
    options?: {
      stream?: boolean
      target?: string | Peer
      session?: string | Session
    }
  ): Promise<string | DialecticStreamResponse | null> {
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
      stream: options?.stream,
      target: targetId,
      session: resolvedSessionId,
    })

    if (chatParams.stream) {
      const body = {
        query: chatParams.query,
        stream: true,
        target: chatParams.target,
        session_id: chatParams.session,
      }

      const url = `${this._client.baseURL}/v2/workspaces/${this.workspaceId}/peers/${this.id}/chat`
      const apiKey = this._client.apiKey

      async function* streamResponse(): AsyncGenerator<
        string,
        void,
        undefined
      > {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            // Include auth headers if present
            ...(apiKey && {
              Authorization: `Bearer ${apiKey}`,
            }),
          },
          body: JSON.stringify(body),
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        if (!response.body) {
          throw new Error('Response body is null')
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const jsonStr = line.slice(6) // Remove "data: " prefix
                try {
                  const chunkData: DialecticStreamChunk = JSON.parse(jsonStr)
                  if (chunkData.done) {
                    return
                  }
                  const content = chunkData.delta.content
                  if (content) {
                    yield content
                  }
                } catch {}
              }
            }
          }
        } finally {
          reader.releaseLock()
        }
      }

      return new DialecticStreamResponse(streamResponse())
    }

    const response = await this._client.workspaces.peers.chat(
      this.workspaceId,
      this.id,
      {
        query: chatParams.query,
        stream: false,
        target: chatParams.target,
        session_id: chatParams.session,
      }
    )
    if (!response.content || response.content === 'None') {
      return null
    }
    return response.content
  }

  /**
   * Get all sessions this peer is a member of.
   *
   * Makes an API call to retrieve all sessions where this peer is an active participant.
   * Sessions are created when peers are added to them or send messages to them.
   *
   * @param filters - Optional filter criteria for sessions. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
   * @returns Promise resolving to a paginated list of Session objects this peer belongs to.
   *          Returns an empty list if the peer is not a member of any sessions
   */
  async getSessions(filters?: Filters | null): Promise<Page<Session>> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const sessionsPage = await this._client.workspaces.peers.sessions.list(
      this.workspaceId,
      this.id,
      {
        filters: validatedFilter,
      }
    )
    return new Page(
      sessionsPage,
      (session) => new Session(session.id, this.workspaceId, this._client)
    )
  }

  /**
   * Create a message object attributed to this peer.
   *
   * This is a convenience method for creating message objects with this peer's ID.
   * The created message object can then be added to sessions or used in other operations.
   *
   * @param content - The text content for the message
   * @param options.metadata - Optional metadata to associate with the message
   * @param options.configuration - Optional message-level configuration (e.g., deriver settings)
   * @param options.created_at - Optional ISO 8601 timestamp for the message
   * @returns A new message object with this peer's ID and the provided content
   */
  message(
    content: string,
    options?: {
      metadata?: Record<string, unknown>
      configuration?: Record<string, unknown>
      created_at?: string | Date
    }
  ): ValidatedMessageCreate {
    const validatedContent = MessageContentSchema.parse(content)
    const validatedMetadata = options?.metadata
      ? MessageMetadataSchema.parse(options.metadata)
      : undefined

    const createdAt =
      options?.created_at instanceof Date
        ? options.created_at.toISOString()
        : options?.created_at

    return {
      peer_id: this.id,
      content: validatedContent,
      metadata: validatedMetadata,
      configuration: options?.configuration,
      created_at: createdAt,
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
    const peer = await this._client.workspaces.peers.getOrCreate(
      this.workspaceId,
      { id: this.id }
    )
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
    await this._client.workspaces.peers.update(this.workspaceId, this.id, {
      metadata,
    })
    this._metadata = metadata
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
  async getConfig(): Promise<Record<string, unknown>> {
    const peer = await this._client.workspaces.peers.getOrCreate(
      this.workspaceId,
      { id: this.id }
    )
    this._configuration = peer.configuration || {}
    return this._configuration
  }

  /**
   * Set the configuration for this peer. Currently the only supported config
   * value is the `observe_me` flag, which controls whether derivation tasks
   * should be created for this peer's global representation. Default is True.
   *
   * Makes an API call to update the configuration associated with this peer.
   * This will overwrite any existing configuration with the provided values.
   * This method also updates the cached configuration property.
   *
   * @param config - A dictionary of configuration to associate with this peer.
   *                 Keys must be strings, values can be any JSON-serializable type
   */
  async setConfig(config: Record<string, unknown>): Promise<void> {
    await this._client.workspaces.peers.update(this.workspaceId, this.id, {
      configuration: config,
    })
    this._configuration = config
  }

  /**
   * Get the current workspace-level configuration for this peer.
   *
   * @deprecated Use getConfig() instead
   * @returns Promise resolving to a dictionary containing the peer's configuration
   */
  async getPeerConfig(): Promise<Record<string, unknown>> {
    return this.getConfig()
  }

  /**
   * Set the configuration for this peer.
   *
   * @deprecated Use setConfig() instead
   * @param config - A dictionary of configuration to associate with this peer
   */
  async setPeerConfig(config: Record<string, unknown>): Promise<void> {
    return this.setConfig(config)
  }

  /**
   * Refresh cached metadata and configuration for this peer.
   *
   * Makes a single API call to retrieve the latest metadata and configuration
   * associated with this peer and updates the cached properties.
   */
  async refresh(): Promise<void> {
    const peer = await this._client.workspaces.peers.getOrCreate(
      this.workspaceId,
      { id: this.id }
    )
    this._metadata = peer.metadata || {}
    this._configuration = peer.configuration || {}
  }

  /**
   * Search for messages in the workspace with this peer as author.
   *
   * Makes an API call to search endpoint.
   *
   * @param query The search query to use
   * @param filters - Optional filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
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
    return await this._client.workspaces.peers.search(
      this.workspaceId,
      this.id,
      {
        query: validatedQuery,
        filters: validatedFilters,
        limit: validatedLimit,
      }
    )
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
   * @returns Promise resolving to a string containing the peer card
   */
  async card(target?: string | Peer): Promise<string> {
    // Validate target parameter
    if (
      target !== undefined &&
      typeof target !== 'string' &&
      !(target instanceof Peer)
    ) {
      throw new TypeError(
        `target must be string, Peer, or undefined, got ${typeof target}`
      )
    }

    if (typeof target === 'string' && target.trim().length === 0) {
      throw new Error('target string cannot be empty')
    }

    const response = await this._client.workspaces.peers.card(
      this.workspaceId,
      this.id,
      {
        target: target instanceof Peer ? target.id : target,
      }
    )

    if (!response.peer_card) {
      return ''
    }

    const items: string[] = response.peer_card

    return items.join('\n')
  }

  /**
   * Get a representation for this peer.
   *
   * Makes an API call to retrieve the representation for this peer.
   *
   * @param session - Optional session to scope the representation to.
   * @param target - Optional target peer to get the representation of. If provided,
   *                 returns the representation of the target from the perspective of this peer.
   * @param options - Optional representation options to filter and configure the results
   * @returns Promise resolving to a Representation object containing explicit and deductive observations
   *
   * @example
   * ```typescript
   * // Get global representation
   * const globalRep = await peer.getRepresentation()
   * console.log(globalRep.toString())
   *
   * // Get representation scoped to a session
   * const sessionRep = await peer.getRepresentation('session-123')
   *
   * // Get representation with semantic search
   * const searchedRep = await peer.getRepresentation(undefined, undefined, {
   *   searchQuery: 'preferences',
   *   searchTopK: 10,
   *   maxObservations: 50
   * })
   * ```
   */
  async getRepresentation(
    session?: string | Session,
    target?: string | Peer,
    options?: RepresentationOptions
  ): Promise<Representation> {
    const workingRepParams = PeerWorkingRepParamsSchema.parse({
      session,
      target,
      options,
    })
    const sessionId = workingRepParams.session
      ? typeof workingRepParams.session === 'string'
        ? workingRepParams.session
        : workingRepParams.session.id
      : undefined
    const targetId = workingRepParams.target
      ? typeof workingRepParams.target === 'string'
        ? workingRepParams.target
        : workingRepParams.target.id
      : undefined

    const response = await this._client.workspaces.peers.workingRepresentation(
      this.workspaceId,
      this.id,
      {
        session_id: sessionId,
        target: targetId,
        search_query: workingRepParams.options?.searchQuery,
        search_top_k: workingRepParams.options?.searchTopK,
        search_max_distance: workingRepParams.options?.searchMaxDistance,
        include_most_derived: workingRepParams.options?.includeMostDerived,
        max_observations: workingRepParams.options?.maxObservations,
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
   * Get context for this peer, including representation and peer card.
   *
   * This is a convenience method that retrieves both the working representation
   * and peer card in a single API call.
   *
   * @param target - Optional target peer to get context for. If provided, returns
   *                 the context for the target from this peer's perspective.
   * @param options - Optional representation options to filter and configure the results
   * @returns Promise resolving to a PeerContext object containing representation and peer card
   *
   * @example
   * ```typescript
   * // Get own context
   * const context = await peer.getContext()
   * console.log(context.representation?.toString())
   * console.log(context.peerCard)
   *
   * // Get context for another peer
   * const context = await peer.getContext('other-peer-id')
   *
   * // Get context with semantic search
   * const context = await peer.getContext(undefined, {
   *   searchQuery: 'preferences',
   *   searchTopK: 10
   * })
   * ```
   */
  async getContext(
    target?: string | Peer,
    options?: RepresentationOptions
  ): Promise<PeerContext> {
    const targetId = target
      ? typeof target === 'string'
        ? target
        : target.id
      : undefined

    const response = await this._client.workspaces.peers.getContext(
      this.workspaceId,
      this.id,
      {
        target: targetId,
        search_query: options?.searchQuery,
        search_top_k: options?.searchTopK,
        search_max_distance: options?.searchMaxDistance,
        include_most_derived: options?.includeMostDerived,
        max_observations: options?.maxObservations,
      }
    )

    return PeerContext.fromApiResponse(
      response as unknown as Record<string, unknown>
    )
  }

  /**
   * Access this peer's self-observations (where observer == observed == self).
   *
   * This property provides a convenient way to access observations that this peer
   * has made about themselves. Use this for self-observation scenarios.
   *
   * @returns An ObservationScope scoped to this peer's self-observations
   *
   * @example
   * ```typescript
   * // List self-observations
   * const obsList = await peer.observations.list()
   *
   * // Search self-observations
   * const results = await peer.observations.query('preferences')
   *
   * // Delete a self-observation
   * await peer.observations.delete('obs-123')
   * ```
   */
  get observations(): ObservationScope {
    return new ObservationScope(
      this._client,
      this.workspaceId,
      this.id,
      this.id
    )
  }

  /**
   * Access observations this peer has made about another peer.
   *
   * This method provides scoped access to observations where this peer is the
   * observer and the target is the observed peer.
   *
   * @param target - The target peer (either a Peer object or peer ID string)
   * @returns An ObservationScope scoped to this peer's observations of the target
   *
   * @example
   * ```typescript
   * // Get observations about another peer
   * const bobObservations = peer.observationsOf('bob')
   *
   * // List observations
   * const obsList = await bobObservations.list()
   *
   * // Search observations
   * const results = await bobObservations.query('work history')
   *
   * // Get the representation from these observations
   * const rep = await bobObservations.getRepresentation()
   * ```
   */
  observationsOf(target: string | Peer): ObservationScope {
    const targetId = typeof target === 'string' ? target : target.id
    return new ObservationScope(
      this._client,
      this.workspaceId,
      this.id,
      targetId
    )
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

/**
 * Context for a peer, including representation and peer card.
 *
 * This class holds both the working representation and peer card for a peer,
 * typically returned from the getContext API call.
 */
export class PeerContext {
  /**
   * The ID of the observer peer.
   */
  readonly peerId: string

  /**
   * The ID of the target peer being observed.
   */
  readonly targetId: string

  /**
   * The working representation (may be null if no observations exist).
   */
  readonly representation: Representation | null

  /**
   * List of peer card strings (may be null if no card exists).
   */
  readonly peerCard: string[] | null

  constructor(
    peerId: string,
    targetId: string,
    representation: Representation | null,
    peerCard: string[] | null
  ) {
    this.peerId = peerId
    this.targetId = targetId
    this.representation = representation
    this.peerCard = peerCard
  }

  /**
   * Create a PeerContext from an API response.
   *
   * @param response - API response object with peer_id, target_id, representation, and peer_card
   * @returns A new PeerContext instance
   */
  static fromApiResponse(response: Record<string, unknown>): PeerContext {
    const peerId = (response.peer_id as string | undefined) ?? ''
    const targetId = (response.target_id as string | undefined) ?? ''

    let representation: Representation | null = null
    if (response.representation) {
      representation = Representation.fromData(
        response.representation as RepresentationData
      )
    }

    const peerCard = (response.peer_card as string[] | undefined) ?? null

    return new PeerContext(peerId, targetId, representation, peerCard)
  }

  /**
   * Return a string representation of the PeerContext.
   *
   * @returns A string representation suitable for debugging
   */
  toString(): string {
    const hasRep = this.representation !== null
    const hasCard = this.peerCard !== null && this.peerCard.length > 0
    return `PeerContext(peerId='${this.peerId}', targetId='${this.targetId}', hasRepresentation=${hasRep}, hasPeerCard=${hasCard})`
  }
}
