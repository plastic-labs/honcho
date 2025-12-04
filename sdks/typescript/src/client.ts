import HonchoCore from '@honcho-ai/core'
import type { DefaultQuery } from '@honcho-ai/core/core'
import type { Message } from '@honcho-ai/core/resources/workspaces/sessions/messages'
import type {
  DeriverStatus,
  WorkspaceDeriverStatusParams,
} from '@honcho-ai/core/resources/workspaces/workspaces'
import { Page } from './pagination'
import { Peer } from './peer'
import { Session } from './session'
import {
  type DeriverStatusOptions,
  FilterSchema,
  type Filters,
  type HonchoConfig,
  HonchoConfigSchema,
  LimitSchema,
  MessageMetadataSchema,
  type PeerConfig,
  PeerConfigSchema,
  PeerIdSchema,
  type PeerMetadata,
  PeerMetadataSchema,
  SearchQuerySchema,
  type SessionConfig,
  SessionConfigSchema,
  SessionIdSchema,
  type SessionMetadata,
  SessionMetadataSchema,
  type WorkspaceConfig,
  WorkspaceConfigSchema,
  type WorkspaceMetadata,
  WorkspaceMetadataSchema,
} from './validation'

/**
 * Main client for the Honcho TypeScript SDK.
 *
 * Provides access to peers, sessions, and workspace operations with configuration
 * from environment variables or explicit parameters. This is the primary entry
 * point for interacting with the Honcho conversational memory platform.
 *
 * For advanced usage, the underlying @honcho-ai/core client can be accessed via the
 * `core` property to use functionality not exposed through this SDK.
 *
 * @example
 * ```typescript
 * const honcho = new Honcho({
 *   apiKey: 'your-api-key',
 *   workspaceId: 'your-workspace-id'
 * })
 *
 * const peer = await honcho.peer('user123')
 * const session = await honcho.session('session456')
 * ```
 */
export class Honcho {
  /**
   * Workspace ID for scoping operations.
   */
  readonly workspaceId: string
  /**
   * Reference to the core Honcho client instance.
   */
  private _client: HonchoCore
  /**
   * Private cached metadata for this workspace.
   */
  private _metadata?: Record<string, unknown>
  /**
   * Private cached configuration for this workspace.
   */
  private _configuration?: Record<string, unknown>

  /**
   * Cached metadata for this workspace. May be stale if the workspace
   * was not recently fetched from the API.
   *
   * Call getMetadata() to get the latest metadata from the server,
   * which will also update this cached value.
   */
  get metadata(): Record<string, unknown> | undefined {
    return this._metadata
  }

  /**
   * Cached configuration for this workspace. May be stale if the workspace
   * was not recently fetched from the API.
   *
   * Call getConfig() to get the latest configuration from the server,
   * which will also update this cached value.
   */
  get configuration(): Record<string, unknown> | undefined {
    return this._configuration
  }

  /**
   * Access the underlying @honcho-ai/core client. The @honcho-ai/core client is the raw Stainless-generated client,
   * allowing users to access functionality that is not exposed through this SDK.
   *
   * @returns The underlying HonchoCore client instance
   *
   * @example
   * ```typescript
   * import { Honcho } from '@honcho-ai/sdk';
   *
   * const client = new Honcho();
   *
   * const workspace = await client.core.workspaces.getOrCreate({ id: "custom-workspace-id" });
   * ```
   */
  get core(): InstanceType<typeof HonchoCore> {
    return this._client
  }

  /**
   * Initialize the Honcho client.
   *
   * @param options - Configuration options for the client
   * @param options.apiKey - API key for authentication. If not provided, will attempt to
   *                         read from HONCHO_API_KEY environment variable
   * @param options.environment - Environment to use (local, production, or demo)
   * @param options.baseURL - Base URL for the Honcho API. If not provided, will attempt to
   *                          read from HONCHO_URL environment variable or default to the
   *                          production API URL
   * @param options.workspaceId - Workspace ID to use for operations. If not provided, will
   *                              attempt to read from HONCHO_WORKSPACE_ID environment variable
   *                              or default to "default"
   * @param options.timeout - Optional custom timeout for the HTTP client
   * @param options.maxRetries - Optional custom maximum number of retries for the HTTP client
   * @param options.defaultHeaders - Optional custom default headers for the HTTP client
   * @param options.defaultQuery - Optional custom default query parameters for the HTTP client
   */
  constructor(options: HonchoConfig) {
    const validatedOptions = HonchoConfigSchema.parse(options)
    this.workspaceId =
      validatedOptions.workspaceId ||
      process.env.HONCHO_WORKSPACE_ID ||
      'default'
    this._client = new HonchoCore({
      apiKey: validatedOptions.apiKey || process.env.HONCHO_API_KEY,
      environment: validatedOptions.environment,
      baseURL: validatedOptions.baseURL || process.env.HONCHO_URL,
      timeout: validatedOptions.timeout,
      maxRetries: validatedOptions.maxRetries,
      defaultHeaders: validatedOptions.defaultHeaders,
      defaultQuery: validatedOptions.defaultQuery as DefaultQuery,
    })
    // Note: Constructor cannot be async, so we can't await here
    // The workspace will be created on first use if it doesn't exist
    // due to the upsert behavior of the API
    this._client.workspaces.getOrCreate({ id: this.workspaceId })
  }

  /**
   * Get or create a peer with the given ID.
   *
   * Creates a Peer object that can be used to interact with the specified peer.
   * If metadata or config is provided, makes an API call to get/create the peer
   * immediately with those values.
   *
   * Provided metadata and configuration will overwrite existing data for this peer
   * if it already exists.
   *
   * @param id - Unique identifier for the peer within the workspace. Should be a
   *             stable identifier that can be used consistently across sessions.
   * @param metadata - Optional metadata dictionary to associate with this peer.
   *                   If set, will get/create peer immediately with metadata.
   * @param config - Optional configuration to set for this peer.
   *                 If set, will get/create peer immediately with flags.
   * @returns Promise resolving to a Peer object that can be used to send messages,
   *          join sessions, and query the peer's knowledge representations
   * @throws Error if the peer ID is empty or invalid
   */
  async peer(
    id: string,
    options?: {
      metadata?: PeerMetadata
      config?: PeerConfig
    }
  ): Promise<Peer> {
    const validatedId = PeerIdSchema.parse(id)
    const validatedMetadata = options?.metadata
      ? PeerMetadataSchema.parse(options.metadata)
      : undefined
    const validatedConfig = options?.config
      ? PeerConfigSchema.parse(options.config)
      : undefined

    if (validatedConfig || validatedMetadata) {
      const peerData = await this._client.workspaces.peers.getOrCreate(
        this.workspaceId,
        {
          id: validatedId,
          configuration: validatedConfig,
          metadata: validatedMetadata,
        }
      )
      return new Peer(
        validatedId,
        this.workspaceId,
        this._client,
        peerData.metadata ?? undefined,
        peerData.configuration ?? undefined
      )
    }

    return new Peer(validatedId, this.workspaceId, this._client)
  }

  /**
   * Get all peers in the current workspace.
   *
   * Makes an API call to retrieve all peers that have been created or used
   * within the current workspace. Returns a paginated result.
   *
   * @param filters - Optional filter criteria for peers. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
   * @returns Promise resolving to a Page of Peer objects representing all peers in the workspace
   */
  async getPeers(filters?: Filters): Promise<Page<Peer>> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const peersPage = await this._client.workspaces.peers.list(
      this.workspaceId,
      { filters: validatedFilter }
    )
    return new Page(
      peersPage,
      (peer) =>
        new Peer(
          peer.id,
          this.workspaceId,
          this._client,
          peer.metadata ?? undefined,
          peer.configuration ?? undefined
        )
    )
  }

  /**
   * Get or create a session with the given ID.
   *
   * Creates a Session object that can be used to manage conversations between
   * multiple peers. If metadata or config is provided, makes an API call to
   * get/create the session immediately with those values.
   *
   * Provided metadata and configuration will overwrite existing data for this session
   * if it already exists.
   *
   * @param id - Unique identifier for the session within the workspace. Should be a
   *             stable identifier that can be used consistently to reference the
   *             same conversation
   * @param metadata - Optional metadata dictionary to associate with this session.
   *                   If set, will get/create session immediately with metadata.
   * @param config - Optional configuration to set for this session.
   *                 If set, will get/create session immediately with flags.
   * @returns Promise resolving to a Session object that can be used to add peers,
   *          send messages, and manage conversation context
   * @throws Error if the session ID is empty or invalid
   */
  async session(
    id: string,
    options?: {
      metadata?: SessionMetadata
      config?: SessionConfig
    }
  ): Promise<Session> {
    const validatedId = SessionIdSchema.parse(id)
    const validatedMetadata = options?.metadata
      ? SessionMetadataSchema.parse(options.metadata)
      : undefined
    const validatedConfig = options?.config
      ? SessionConfigSchema.parse(options.config)
      : undefined

    if (validatedConfig || validatedMetadata) {
      const sessionData = await this._client.workspaces.sessions.getOrCreate(
        this.workspaceId,
        {
          id: validatedId,
          configuration: validatedConfig,
          metadata: validatedMetadata,
        }
      )
      return new Session(
        validatedId,
        this.workspaceId,
        this._client,
        sessionData.metadata ?? undefined,
        sessionData.configuration ?? undefined
      )
    }

    return new Session(validatedId, this.workspaceId, this._client)
  }

  /**
   * Get all sessions in the current workspace.
   *
   * Makes an API call to retrieve all sessions that have been created within
   * the current workspace.
   *
   * @param filters - Optional filter criteria for sessions. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
   * @returns Promise resolving to a Page of Session objects representing all sessions
   *          in the workspace. Returns an empty page if no sessions exist
   */
  async getSessions(filters?: Filters): Promise<Page<Session>> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const sessionsPage = await this._client.workspaces.sessions.list(
      this.workspaceId,
      { filters: validatedFilter }
    )
    return new Page(
      sessionsPage,
      (session) =>
        new Session(
          session.id,
          this.workspaceId,
          this._client,
          session.metadata ?? undefined,
          session.configuration ?? undefined
        )
    )
  }

  /**
   * Get metadata for the current workspace.
   *
   * Makes an API call to retrieve metadata associated with the current workspace.
   * Workspace metadata can include settings, configuration, or any other
   * key-value data associated with the workspace. This method also updates the
   * cached metadata property.
   *
   * @returns Promise resolving to a dictionary containing the workspace's metadata.
   *          Returns an empty dictionary if no metadata is set
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const workspace = await this._client.workspaces.getOrCreate({
      id: this.workspaceId,
    })
    this._metadata = workspace.metadata || {}
    return this._metadata
  }

  /**
   * Set metadata for the current workspace.
   *
   * Makes an API call to update the metadata associated with the current workspace.
   * This will overwrite any existing metadata with the provided values.
   * This method also updates the cached metadata property.
   *
   * @param metadata - A dictionary of metadata to associate with the workspace.
   *                   Keys must be strings, values can be any JSON-serializable type
   */
  async setMetadata(metadata: WorkspaceMetadata): Promise<void> {
    const validatedMetadata = WorkspaceMetadataSchema.parse(metadata)
    await this._client.workspaces.update(this.workspaceId, {
      metadata: validatedMetadata,
    })
    this._metadata = validatedMetadata
  }

  /**
   * Get configuration for the current workspace.
   *
   * Makes an API call to retrieve configuration associated with the current workspace.
   * Configuration includes settings that control workspace behavior.
   * This method also updates the cached configuration property.
   *
   * @returns Promise resolving to a dictionary containing the workspace's configuration.
   *          Returns an empty dictionary if no configuration is set
   */
  async getConfig(): Promise<Record<string, unknown>> {
    const workspace = await this._client.workspaces.getOrCreate({
      id: this.workspaceId,
    })
    this._configuration = workspace.configuration || {}
    return this._configuration
  }

  /**
   * Set configuration for the current workspace.
   *
   * Makes an API call to update the configuration associated with the current workspace.
   * This will overwrite any existing configuration with the provided values.
   * This method also updates the cached configuration property.
   *
   * @param configuration - A dictionary of configuration to associate with the workspace.
   *                        Keys must be strings, values can be any JSON-serializable type
   */
  async setConfig(configuration: WorkspaceConfig): Promise<void> {
    const validatedConfig = WorkspaceConfigSchema.parse(configuration)
    await this._client.workspaces.update(this.workspaceId, {
      configuration: validatedConfig,
    })
    this._configuration = validatedConfig
  }

  /**
   * Refresh cached metadata and configuration for the current workspace.
   *
   * Makes a single API call to retrieve the latest metadata and configuration
   * associated with the current workspace and updates the cached properties.
   */
  async refresh(): Promise<void> {
    const workspace = await this._client.workspaces.getOrCreate({
      id: this.workspaceId,
    })
    this._metadata = workspace.metadata || {}
    this._configuration = workspace.configuration || {}
  }

  /**
   * Get all workspace IDs from the Honcho instance.
   *
   * Makes an API call to retrieve all workspace IDs that the authenticated
   * user has access to.
   *
   * @param filters - Optional filter criteria for workspaces. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
   * @returns Promise resolving to a list of workspace ID strings. Returns an empty
   *          list if no workspaces are accessible or none exist
   */
  async getWorkspaces(filters?: Filters): Promise<string[]> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const workspacesPage = await this._client.workspaces.list({
      filters: validatedFilter,
    })
    const ids: string[] = []
    for await (const workspace of workspacesPage) {
      ids.push(workspace.id)
    }
    return ids
  }

  /**
   * Delete a workspace.
   *
   * Makes an API call to delete the specified workspace.
   *
   * @param workspaceId - The ID of the workspace to delete
   * @returns Promise resolving to the deleted Workspace object
   */
  async deleteWorkspace(
    workspaceId: string
  ): Promise<Awaited<ReturnType<typeof this._client.workspaces.delete>>> {
    return await this._client.workspaces.delete(workspaceId)
  }

  /**
   * Search for messages in the current workspace.
   *
   * Makes an API call to search for messages in the current workspace.
   *
   * @param query - The search query to use
   * @param filters - Optional filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
   * @param limit - Number of results to return (1-100, default: 10).
   * @returns Promise resolving to an array of Message objects representing the search results.
   *          Returns an empty array if no messages are found.
   * @throws Error if the search query is empty or invalid
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
    return await this._client.workspaces.search(this.workspaceId, {
      query: validatedQuery,
      filters: validatedFilters,
      limit: validatedLimit,
    })
  }

  /**
   * Get the deriver processing status, optionally scoped to an observer, sender, and/or session.
   *
   * Makes an API call to retrieve the current status of the deriver processing queue.
   * The deriver is responsible for processing messages and updating peer representations.
   *
   * @param options - Configuration options for the status request
   * @param options.observer - Optional observer (ID string or Peer object) to scope the status to
   * @param options.sender - Optional sender (ID string or Peer object) to scope the status to
   * @param options.session - Optional session (ID string or Session object) to scope the status to
   * @returns Promise resolving to the deriver status information including work unit counts
   */
  async getDeriverStatus(
    options?: Omit<
      DeriverStatusOptions,
      'observerId' | 'senderId' | 'sessionId'
    > & {
      observer?: string | Peer
      sender?: string | Peer
      session?: string | Session
    }
  ): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<string, DeriverStatus.Sessions>
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
    const resolvedSessionId = options?.session
      ? typeof options.session === 'string'
        ? options.session
        : options.session.id
      : undefined

    const queryParams: WorkspaceDeriverStatusParams = {}
    if (resolvedObserverId) queryParams.observer_id = resolvedObserverId
    if (resolvedSenderId) queryParams.sender_id = resolvedSenderId
    if (resolvedSessionId) queryParams.session_id = resolvedSessionId

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
   * Poll getDeriverStatus until pendingWorkUnits and inProgressWorkUnits are both 0.
   * This allows you to guarantee that all messages have been processed by the deriver for
   * use with the dialectic endpoint.
   *
   * The polling estimates sleep time by assuming each work unit takes 1 second.
   *
   * @param options - Configuration options for the status request
   * @param options.observer - Optional observer (ID string or Peer object) to scope the status to
   * @param options.sender - Optional sender (ID string or Peer object) to scope the status to
   * @param options.session - Optional session (ID string or Session object) to scope the status to
   * @param options.timeoutMs - Optional timeout in milliseconds (default: 300000 - 5 minutes)
   * @returns Promise resolving to the final deriver status when processing is complete
   * @throws Error if timeout is exceeded before processing completes
   */
  async pollDeriverStatus(
    options?: Omit<
      DeriverStatusOptions,
      'observerId' | 'senderId' | 'sessionId'
    > & {
      observer?: string | Peer
      sender?: string | Peer
      session?: string | Session
    }
  ): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<string, DeriverStatus.Sessions>
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
   * Update the metadata of a message.
   *
   * Makes an API call to update the metadata of a specific message within a session.
   *
   * @param message - Either a Message object or a message ID string
   * @param metadata - The metadata to update for the message
   * @param session - The session (ID string or Session object) - required if message is a string ID, ignored if message is a Message object
   * @returns Promise resolving to the updated Message object
   * @throws Error if message is a string ID but session is not provided
   */
  async updateMessage(
    message: Message | string,
    metadata: Record<string, unknown>,
    session?: string | Session
  ): Promise<Message> {
    const validatedMetadata = MessageMetadataSchema.parse(metadata)
    let messageId: string
    let resolvedSessionId: string

    if (typeof message === 'string') {
      messageId = message
      if (!session) {
        throw new Error('session is required when message is a string ID')
      }
      resolvedSessionId = typeof session === 'string' ? session : session.id
    } else {
      messageId = message.id
      resolvedSessionId = message.session_id
    }

    return await this._client.workspaces.sessions.messages.update(
      this.workspaceId,
      resolvedSessionId,
      messageId,
      {
        metadata: validatedMetadata,
      }
    )
  }

  /**
   * Return a string representation of the Honcho client.
   *
   * @returns A string representation suitable for debugging
   */
  toString(): string {
    return `Honcho(workspaceId='${this.workspaceId}', baseURL='${this._client.baseURL}')`
  }
}
