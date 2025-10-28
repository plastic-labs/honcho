import type HonchoCore from '@honcho-ai/core'
import type {
  DeriverStatus,
  WorkspaceDeriverStatusParams,
} from '@honcho-ai/core/resources/index'
import type { Message } from '@honcho-ai/core/resources/workspaces/sessions/messages'
import type { Uploadable } from '@honcho-ai/core/uploads'
import { Page } from './pagination'
import { Peer } from './peer'
import {
  Representation,
  type RepresentationData,
  type RepresentationOptions,
} from './representation'
import { SessionContext, SessionSummaries, Summary } from './session_context'
import {
  ContextParamsSchema,
  type DeriverStatusOptions,
  DeriverStatusOptionsSchema,
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
  observe_others?: boolean

  /**
   * Initialize SessionPeerConfig with observation settings.
   *
   * @param observe_me - Whether other peers should observe this peer in the session
   * @param observe_others - Whether this peer should observe others in the session
   */
  constructor(observe_me?: boolean | null, observe_others?: boolean) {
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
 *
 * Key features:
 * - Multi-peer conversations with configurable observation settings
 * - Message storage and retrieval with filtering capabilities
 * - Context optimization for token-limited scenarios
 * - File upload support with automatic message creation
 * - Session-scoped peer representations and theory-of-mind modeling
 * - Search functionality across session messages
 *
 * @example
 * ```typescript
 * const session = await honcho.session('conversation-123')
 *
 * // Add peers to the session
 * await session.addPeers(['user1', 'assistant1'])
 *
 * // Send messages
 * await session.addMessages([
 *   { peer_id: 'user1', content: 'Hello!' },
 *   { peer_id: 'assistant1', content: 'Hi there!' }
 * ])
 *
 * // Get optimized context
 * const context = await session.getContext(true, 4000)
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
  /**
   * Reference to the parent Honcho client instance.
   */
  private _client: HonchoCore
  /**
   * Cached metadata for this session. May be stale if the session
   * was not recently fetched from the API.
   *
   * Call getMetadata() to get the latest metadata from the server,
   * which will also update this cached value.
   */
  metadata?: Record<string, unknown> | null
  /**
   * Cached configuration for this session. May be stale if the session
   * was not recently fetched from the API.
   *
   * Call getConfig() to get the latest configuration from the server,
   * which will also update this cached value.
   */
  configuration?: Record<string, unknown> | null

  /**
   * Initialize a new Session. **Do not call this directly, use the client.session() method instead.**
   *
   * @param id - Unique identifier for this session within the workspace
   * @param workspaceId - Workspace ID for scoping operations
   * @param client - Reference to the parent Honcho client instance
   * @param metadata - Optional metadata to initialize the cached value
   * @param configuration - Optional configuration to initialize the cached value
   */
  constructor(
    id: string,
    workspaceId: string,
    client: HonchoCore,
    metadata?: Record<string, unknown> | null,
    configuration?: Record<string, unknown> | null
  ) {
    this.id = id
    this.workspaceId = workspaceId
    this._client = client
    this.metadata = metadata
    this.configuration = configuration
  }

  /**
   * Add peers to this session.
   *
   * Makes an API call to add one or more peers to this session. Adding peers
   * creates bidirectional relationships and allows them to participate in
   * the session's conversations. Peers can be added with optional session-specific
   * configuration to control observation behaviors.
   *
   * @param peers - Peers to add to the session. Can be:
   *   - string: Single peer ID
   *   - Peer: Single Peer object
   *   - Array<string | Peer>: List of peer IDs and/or Peer objects
   *   - [string | Peer, SessionPeerConfig]: Single peer with session config
   *   - Array<string | Peer | [string | Peer, SessionPeerConfig]>: Mixed list
   *     of peers and peer+config combinations
   *
   * @example
   * ```typescript
   * // Add single peer
   * await session.addPeers('user123')
   *
   * // Add multiple peers
   * await session.addPeers(['user1', 'user2', peer3])
   *
   * // Add peer with custom config
   * await session.addPeers(['user1', new SessionPeerConfig(false, true)])
   *
   * // Add mixed peers with and without configs
   * await session.addPeers([
   *   'user1',
   *   ['user2', new SessionPeerConfig(true, false)],
   *   peer3
   * ])
   * ```
   */
  async addPeers(peers: PeerAddition): Promise<void> {
    const validatedPeers = PeerAdditionSchema.parse(peers)
    const peerDict: Record<string, SessionPeerConfig> = {}
    const peersArray = Array.isArray(validatedPeers)
      ? validatedPeers
      : [validatedPeers]

    for (const peer of peersArray) {
      if (typeof peer === 'string') {
        // Handle string peer ID
        peerDict[peer] = {}
      } else if (Array.isArray(peer)) {
        // Handle tuple [string | Peer, SessionPeerConfig]
        const peerId = typeof peer[0] === 'string' ? peer[0] : peer[0].id
        peerDict[peerId] = peer[1]
      } else if (typeof peer === 'object' && 'id' in peer) {
        // Handle Peer object
        peerDict[peer.id] = {}
      } else {
        // This should never happen with proper typing, but handle gracefully
        throw new Error(`Invalid peer type: ${typeof peer}`)
      }
    }

    await this._client.workspaces.sessions.peers.add(
      this.workspaceId,
      this.id,
      peerDict
    )
  }

  /**
   * Set the complete peer list for this session.
   *
   * Makes an API call to replace the current peer list with the provided peers.
   * This will remove any peers not in the new list and add any that are missing.
   * Unlike addPeers(), this method overwrites the entire peer membership.
   *
   * @param peers - Peers to set for the session. Can be:
   *   - string: Single peer ID
   *   - Peer: Single Peer object
   *   - Array<string | Peer>: List of peer IDs and/or Peer objects
   *   - [string | Peer, SessionPeerConfig]: Single peer with session config
   *   - Array<string | Peer | [string | Peer, SessionPeerConfig]>: Mixed list
   *     of peers and peer+config combinations
   */
  async setPeers(peers: PeerAddition): Promise<void> {
    const validatedPeers = PeerAdditionSchema.parse(peers)
    const peerDict: Record<string, SessionPeerConfig> = {}
    const peersArray = Array.isArray(validatedPeers)
      ? validatedPeers
      : [validatedPeers]

    for (const peer of peersArray) {
      if (typeof peer === 'string') {
        // Handle string peer ID
        peerDict[peer] = {}
      } else if (Array.isArray(peer)) {
        // Handle tuple [string | Peer, SessionPeerConfig]
        const peerId = typeof peer[0] === 'string' ? peer[0] : peer[0].id
        peerDict[peerId] = peer[1]
      } else if (typeof peer === 'object' && 'id' in peer) {
        // Handle Peer object
        peerDict[peer.id] = {}
      } else {
        // This should never happen with proper typing, but handle gracefully
        throw new Error(`Invalid peer type: ${typeof peer}`)
      }
    }

    await this._client.workspaces.sessions.peers.set(
      this.workspaceId,
      this.id,
      peerDict
    )
  }

  /**
   * Remove peers from this session.
   *
   * Makes an API call to remove one or more peers from this session.
   * Removed peers will no longer be able to participate in the session
   * unless added back. Their existing messages remain in the session.
   *
   * @param peers - Peers to remove from the session. Can be:
   *   - string: Single peer ID
   *   - Peer: Single Peer object
   *   - Array<string | Peer>: List of peer IDs and/or Peer objects
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
    await this._client.workspaces.sessions.peers.remove(
      this.workspaceId,
      this.id,
      peerIds
    )
  }

  /**
   * Get all peers in this session.
   *
   * Makes an API call to retrieve the list of peers that are currently
   * members of this session. Automatically converts the paginated response
   * into a list for convenience -- the max number of peers in a session is usually 10.
   *
   * @returns Promise resolving to a list of Peer objects that are members of this session
   */
  async getPeers(): Promise<Peer[]> {
    const peersPage = await this._client.workspaces.sessions.peers.list(
      this.workspaceId,
      this.id
    )
    return peersPage.items.map(
      (peer) => new Peer(peer.id, this.workspaceId, this._client)
    )
  }

  /**
   * Get the configuration for a peer in this session.
   *
   * Makes an API call to retrieve the session-specific configuration for a peer.
   * This includes observation settings that control how this peer interacts
   * with other peers within this session context.
   *
   * @param peer - The peer to get configuration for. Can be peer ID string or Peer object
   * @returns Promise resolving to SessionPeerConfig object with the peer's session settings
   */
  async getPeerConfig(peer: string | Peer): Promise<SessionPeerConfig> {
    const peerId = typeof peer === 'string' ? peer : peer.id
    return await this._client.workspaces.sessions.peers.getConfig(
      this.workspaceId,
      this.id,
      peerId
    )
  }

  /**
   * Set the configuration for a peer in this session.
   *
   * Makes an API call to update the session-specific configuration for a peer.
   * This controls observation behaviors and theory-of-mind formation within
   * this session context.
   *
   * @param peer - The peer to configure. Can be peer ID string or Peer object
   * @param config - SessionPeerConfig object specifying the observation settings
   */
  async setPeerConfig(
    peer: string | Peer,
    config: SessionPeerConfig
  ): Promise<void> {
    const peerId = typeof peer === 'string' ? peer : peer.id
    const validatedConfig = SessionPeerConfigSchema.parse(config)
    await this._client.workspaces.sessions.peers.setConfig(
      this.workspaceId,
      this.id,
      peerId,
      {
        observe_others: validatedConfig.observe_others,
        observe_me: validatedConfig.observe_me,
      }
    )
  }

  /**
   * Add one or more messages to this session.
   *
   * Makes an API call to store messages in this session. Any message added
   * to a session will automatically add the creating peer to the session
   * if they are not already a member. Messages are the primary way content
   * flows through the Honcho system.
   *
   * @param messages - Messages to add to the session. Can be:
   *   - MessageCreate: Single message object with peer_id and content
   *   - MessageCreate[]: Array of message objects
   *
   * @example
   * ```typescript
   * // Add single message
   * await session.addMessages({
   *   peer_id: 'user123',
   *   content: 'Hello world!'
   * })
   *
   * // Add multiple messages
   * await session.addMessages([
   *   { peer_id: 'user1', content: 'Hello!' },
   *   { peer_id: 'assistant', content: 'Hi there!' }
   * ])
   * // Add message with custom ISO 8601 timestamp
   * await session.addMessages({
   *   peer_id: 'user123',
   *   content: 'Hello world!',
   *   created_at: '2021-01-01T00:00:00.000Z'
   * })
   * ```
   */
  async addMessages(messages: MessageAddition): Promise<Message[]> {
    const validatedMessages = MessageAdditionSchema.parse(messages)
    const messagesList = Array.isArray(validatedMessages)
      ? validatedMessages
      : [validatedMessages]
    return await this._client.workspaces.sessions.messages.create(
      this.workspaceId,
      this.id,
      {
        messages: messagesList,
      }
    )
  }

  /**
   * Get messages from this session with optional filtering.
   *
   * Makes an API call to retrieve messages from this session. Results can be
   * filtered based on various criteria and are returned in a paginated format.
   * Messages are ordered by creation time (most recent first by default).
   *
   * @param filters - Optional filter criteria for messages. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
   * @returns Promise resolving to a Page of Message objects matching the specified criteria
   */
  async getMessages(filters?: Filters): Promise<Page<Message>> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const messagesPage = await this._client.workspaces.sessions.messages.list(
      this.workspaceId,
      this.id,
      validatedFilter
    )
    return new Page(messagesPage)
  }

  /**
   * Get metadata for this session.
   *
   * Makes an API call to retrieve the current metadata associated with this session.
   * Metadata can include custom attributes, settings, or any other key-value data
   * that provides context about the session. This method also updates the cached
   * metadata property.
   *
   * @returns Promise resolving to a dictionary containing the session's metadata.
   *          Returns an empty dictionary if no metadata is set
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const session = await this._client.workspaces.sessions.getOrCreate(
      this.workspaceId,
      { id: this.id }
    )
    this.metadata = session.metadata || {}
    return this.metadata
  }

  /**
   * Set metadata for this session.
   *
   * Makes an API call to update the metadata associated with this session.
   * This will overwrite any existing metadata with the provided values.
   * Metadata is useful for storing custom attributes, configuration, or
   * contextual information about the session. This method also updates the
   * cached metadata property.
   *
   * @param metadata - A dictionary of metadata to associate with this session.
   *                   Keys must be strings, values can be any JSON-serializable type
   */
  async setMetadata(metadata: Record<string, unknown>): Promise<void> {
    await this._client.workspaces.sessions.update(this.workspaceId, this.id, {
      metadata,
    })
    this.metadata = metadata
  }

  /**
   * Get configuration for this session.
   *
   * Makes an API call to retrieve the current configuration associated with this session.
   * Configuration includes settings that control session behavior. This method also
   * updates the cached configuration property.
   *
   * @returns Promise resolving to a dictionary containing the session's configuration.
   *          Returns an empty dictionary if no configuration is set
   */
  async getConfig(): Promise<Record<string, unknown>> {
    const session = await this._client.workspaces.sessions.getOrCreate(
      this.workspaceId,
      { id: this.id }
    )
    this.configuration = session.configuration || {}
    return this.configuration
  }

  /**
   * Set configuration for this session.
   *
   * Makes an API call to update the configuration associated with this session.
   * This will overwrite any existing configuration with the provided values.
   * This method also updates the cached configuration property.
   *
   * @param configuration - A dictionary of configuration to associate with this session.
   *                        Keys must be strings, values can be any JSON-serializable type
   */
  async setConfig(configuration: Record<string, unknown>): Promise<void> {
    await this._client.workspaces.sessions.update(this.workspaceId, this.id, {
      configuration,
    })
    this.configuration = configuration
  }

  /**
   * Delete this session.
   *
   * Makes an API call to mark this session as inactive.
   */
  async delete(): Promise<void> {
    await this._client.workspaces.sessions.delete(this.workspaceId, this.id)
  }

  /**
   * Get optimized context for this session within a token limit.
   *
   * Makes an API call to retrieve a curated list of messages that provides
   * optimal context for the conversation while staying within the specified
   * token limit. Uses tiktoken for token counting, so results should be
   * compatible with OpenAI models. The context optimization balances
   * recency and relevance to provide the best conversational context.
   *
   * @param options - Configuration options for context retrieval
   * @param options.summary - Whether to include summary information in the context.
   *                          When true, includes session summary if available. Defaults to true
   * @param options.tokens - Maximum number of tokens to include in the context. If not provided,
   *                         uses the server's default configuration
   * @param options.peerTarget - The target of the perspective. If given without `peerPerspective`,
   *                             will get the Honcho-level representation and peer card for this peer.
   *                             If given with `peerPerspective`, will get the representation and card
   *                             for this peer from the perspective of that peer.
   * @param options.lastUserMessage - The most recent message, used to fetch semantically relevant
   *                                  observations and returned as part of the context object.
   *                                  Can be either a message ID string or a Message object.
   * @param options.peerPerspective - A peer to get context for. If given, response will attempt to
   *                                  include representation and card from the perspective of that peer.
   *                                  Must be provided with `peerTarget`.
   * @returns Promise resolving to a SessionContext object containing the optimized
   *          message history and summary (if available) that maximizes conversational
   *          context while respecting the token limit
   *
   * @note Token counting is performed using tiktoken. For models using different
   *       tokenizers, you may need to adjust the token limit accordingly.
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

    const context = await this._client.workspaces.sessions.getContext(
      this.workspaceId,
      this.id,
      {
        tokens: contextParams.tokens,
        summary: contextParams.summary,
        last_message: lastMessageId,
        peer_target: contextParams.peerTarget,
        peer_perspective: contextParams.peerPerspective,
        limit_to_session: contextParams.limitToSession,
        search_top_k: contextParams.representationOptions?.searchTopK,
        search_max_distance:
          contextParams.representationOptions?.searchMaxDistance,
        include_most_derived:
          contextParams.representationOptions?.includeMostDerived,
        max_observations: contextParams.representationOptions?.maxObservations,
      }
    )
    // Convert the summary response to Summary object if present
    const summary = context.summary ? new Summary(context.summary) : null
    return new SessionContext(
      this.id,
      context.messages,
      summary,
      context.peer_representation
        ? JSON.stringify(context.peer_representation)
        : null,
      context.peer_card ?? null
    )
  }

  /**
   * Get available summaries for this session.
   *
   * Makes an API call to retrieve both short and long summaries for this session,
   * if they are available. Summaries are created asynchronously by the backend
   * as messages are added to the session.
   *
   * @returns Promise resolving to a SessionSummaries object containing:
   *          - id: The session ID
   *          - shortSummary: The short summary if available, including metadata
   *          - longSummary: The long summary if available, including metadata
   *
   * @note Summaries may be null if:
   *       - Not enough messages have been added to trigger summary generation
   *       - The summary generation is still in progress
   *       - Summary generation is disabled for this session
   */
  async getSummaries(): Promise<SessionSummaries> {
    // Use the core SDK's summaries method
    const data = await this._client.workspaces.sessions.summaries(
      this.workspaceId,
      this.id
    )

    // Return a SessionSummaries instance
    return new SessionSummaries(data)
  }

  /**
   * Search for messages in this session.
   *
   * Makes an API call to search for messages in this session.
   *
   * @param query The search query to use
   * @param filters - Optional filters to scope the search: see [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
   * @param limit Number of results to return (1-100, default: 10).
   * @returns A list of Message objects representing the search results.
   *          Returns an empty list if no messages are found.
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
    return await this._client.workspaces.sessions.search(
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
   * Get the deriver processing status for this session, optionally scoped to an observer or sender.
   *
   * Makes an API call to retrieve the current status of the deriver processing queue.
   * The deriver is responsible for processing messages and updating peer representations.
   * This method automatically scopes the status to this session.
   *
   * @param options - Configuration options for the status request
   * @param options.observerId - Optional observer ID to scope the status to
   * @param options.senderId - Optional sender ID to scope the status to
   * @returns Promise resolving to the deriver status information including work unit counts
   */
  async getDeriverStatus(
    options?: Omit<DeriverStatusOptions, 'sessionId'>
  ): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<string, DeriverStatus.Sessions>
  }> {
    const validatedOptions = options
      ? DeriverStatusOptionsSchema.parse(options)
      : undefined
    const queryParams: WorkspaceDeriverStatusParams = {
      session_id: this.id, // Always use this session's ID
    }
    if (validatedOptions?.observerId)
      queryParams.observer_id = validatedOptions.observerId
    if (validatedOptions?.senderId)
      queryParams.sender_id = validatedOptions.senderId

    const status = await this._client.workspaces.deriverStatus(
      this.workspaceId,
      queryParams
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
   * This allows you to guarantee that all messages have been processed by the deriver for
   * use with the dialectic endpoint.
   *
   * The polling estimates sleep time by assuming each work unit takes 1 second.
   *
   * @param options - Configuration options for the status request
   * @param options.observerId - Optional observer ID to scope the status to
   * @param options.senderId - Optional sender ID to scope the status to
   * @param options.timeoutMs - Optional timeout in milliseconds (default: 300000 - 5 minutes)
   * @returns Promise resolving to the final deriver status when processing is complete
   * @throws Error if timeout is exceeded before processing completes
   */
  async pollDeriverStatus(
    options?: Omit<DeriverStatusOptions, 'sessionId'>
  ): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<string, DeriverStatus.Sessions>
  }> {
    const validatedOptions = options
      ? DeriverStatusOptionsSchema.parse(options)
      : undefined
    const timeoutMs = validatedOptions?.timeoutMs ?? 300000 // Default to 5 minutes
    const startTime = Date.now()

    while (true) {
      const status = await this.getDeriverStatus(validatedOptions)
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
      // Assuming each pending and in-progress work unit takes 1 second
      const totalWorkUnits =
        status.pendingWorkUnits + status.inProgressWorkUnits
      const sleepMs = Math.max(1000, totalWorkUnits * 1000) // Sleep at least 1 second

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
   * Makes an API call to upload a file and convert it into messages. The file is
   * processed to extract text content, split into appropriately sized chunks,
   * and created as messages attributed to the specified peer. The peer will be
   * automatically added to the session if not already a member.
   *
   * @param file - File to upload. Can be:
   *   - File objects (browser File API)
   *   - Buffer or Uint8Array with filename and content_type
   *   - { filename: string, content: Buffer | Uint8Array, content_type: string }
   * @param peerId - The peer ID to attribute the created messages to
   * @returns Promise resolving to a list of Message objects representing the created messages
   *
   * @note Supported file types include PDFs, text files, and JSON documents.
   *       Large files will be automatically split into multiple messages to fit
   *       within message size limits.
   *
   * @example
   * ```typescript
   * // Upload a file
   * const messages = await session.uploadFile(fileInput.files[0], 'user123')
   * console.log(`Created ${messages.length} messages from file`)
   * ```
   */
  async uploadFile(file: Uploadable, peerId: string): Promise<Message[]> {
    const uploadParams = FileUploadSchema.parse({ file, peerId })
    const response = await this._client.workspaces.sessions.messages.upload(
      this.workspaceId,
      this.id,
      {
        file: uploadParams.file,
        peer_id: uploadParams.peerId,
      }
    )

    return response
  }

  /**
   * Get the current working representation of a peer in this session.
   *
   * Makes an API call to retrieve the session-scoped representation that has been
   * built for a peer. This can be either the peer's global representation or
   * their local representation of another peer (theory-of-mind).
   *
   * @param peer - The peer to get the working representation of. Can be peer ID string or Peer object
   * @param target - Optional target peer. If provided, returns what `peer` knows about
   *                 `target` within this session context rather than `peer`'s global representation
   * @param options - Optional representation options to filter and configure the results
   * @returns Promise resolving to a Representation object containing explicit and deductive observations
   *
   * @example
   * ```typescript
   * // Get peer's global representation in this session
   * const globalRep = await session.workingRep('user123')
   * console.log(globalRep.toString())
   *
   * // Get what user123 knows about assistant in this session
   * const localRep = await session.workingRep('user123', 'assistant')
   *
   * // Get representation with semantic search
   * const searchedRep = await session.workingRep('user123', undefined, {
   *   searchQuery: 'preferences',
   *   searchTopK: 10
   * })
   * ```
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

    const response = await this._client.workspaces.peers.workingRepresentation(
      this.workspaceId,
      peerId,
      {
        session_id: this.id,
        target: targetId,
        search_query: workingRepParams.options?.searchQuery,
        search_top_k: workingRepParams.options?.searchTopK,
        search_max_distance: workingRepParams.options?.searchMaxDistance,
        include_most_derived: workingRepParams.options?.includeMostDerived,
        max_observations: workingRepParams.options?.maxObservations,
      }
    )
    const data = (response as { representation: RepresentationData })
      .representation
    return Representation.fromData(data)
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
