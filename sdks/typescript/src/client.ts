import { API_VERSION } from './api-version'
import { HonchoHTTPClient } from './http/client'
import { Message } from './message'
import { Page } from './pagination'
import { Peer } from './peer'
import { Session } from './session'
import type {
  MessageResponse,
  PageResponse,
  PeerResponse,
  QueueStatus,
  QueueStatusParams,
  QueueStatusResponse,
  SessionResponse,
  WorkspaceResponse,
} from './types/api'
import { pollUntilComplete, transformQueueStatus } from './utils'
import {
  FilterSchema,
  type Filters,
  type HonchoConfig,
  HonchoConfigSchema,
  LimitSchema,
  type PeerConfig,
  PeerConfigSchema,
  PeerIdSchema,
  type PeerMetadata,
  PeerMetadataSchema,
  peerConfigFromApi,
  peerConfigToApi,
  type QueueStatusOptions,
  SearchQuerySchema,
  type SessionConfig,
  SessionConfigSchema,
  SessionIdSchema,
  type SessionMetadata,
  SessionMetadataSchema,
  sessionConfigFromApi,
  sessionConfigToApi,
  type WorkspaceConfig,
  WorkspaceConfigSchema,
  type WorkspaceMetadata,
  WorkspaceMetadataSchema,
  workspaceConfigFromApi,
  workspaceConfigToApi,
} from './validation'

const DEFAULT_BASE_URL = 'https://api.honcho.dev'

/**
 * Main client for the Honcho TypeScript SDK.
 *
 * Provides access to peers, sessions, and workspace operations with configuration
 * from environment variables or explicit parameters. This is the primary entry
 * point for interacting with the Honcho conversational memory platform.
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
   * Reference to the HTTP client instance.
   */
  private _http: HonchoHTTPClient
  /**
   * Private cached metadata for this workspace.
   */
  private _metadata?: Record<string, unknown>
  /**
   * Private cached configuration for this workspace.
   */
  private _configuration?: WorkspaceConfig
  /**
   * Memoized workspace get-or-create call.
   */
  private _workspaceReady?: Promise<void>

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
   * Call getConfiguration() to get the latest configuration from the server,
   * which will also update this cached value.
   */
  get configuration(): WorkspaceConfig | undefined {
    return this._configuration
  }

  /**
   * Access the underlying HTTP client for advanced usage.
   *
   * @returns The HTTP client instance
   */
  get http(): HonchoHTTPClient {
    return this._http
  }

  /**
   * Get the base URL for the API.
   */
  get baseURL(): string {
    return this._http.baseURL
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
   */
  constructor(options: HonchoConfig = {}) {
    const validatedOptions = HonchoConfigSchema.parse(options)
    this.workspaceId =
      validatedOptions.workspaceId ||
      process.env.HONCHO_WORKSPACE_ID ||
      'default'

    // Resolve base URL
    let baseURL = validatedOptions.baseURL || process.env.HONCHO_URL
    if (validatedOptions.environment === 'local') {
      baseURL = 'http://localhost:8000'
    } else if (!baseURL) {
      baseURL = DEFAULT_BASE_URL
    }

    this._http = new HonchoHTTPClient({
      baseURL,
      apiKey: validatedOptions.apiKey || process.env.HONCHO_API_KEY,
      timeout: validatedOptions.timeout,
      maxRetries: validatedOptions.maxRetries,
      defaultHeaders: validatedOptions.defaultHeaders,
      defaultQuery: validatedOptions.defaultQuery,
    })
  }

  // ===========================================================================
  // Private API Methods
  // ===========================================================================

  private async _getOrCreateWorkspace(
    id: string,
    params?: {
      metadata?: Record<string, unknown>
      configuration?: WorkspaceConfig
    }
  ): Promise<WorkspaceResponse> {
    return this._http.post<WorkspaceResponse>(`/${API_VERSION}/workspaces`, {
      body: {
        id,
        metadata: params?.metadata,
        configuration: workspaceConfigToApi(params?.configuration),
      },
    })
  }

  private async _ensureWorkspace(): Promise<void> {
    /**
     * Ensure the workspace exists on the server.
     *
     * The Honcho API uses get-or-create semantics for workspaces via `POST /v3/workspaces`.
     * This SDK performs that call once per client instance (memoized) to guarantee that
     * all workspace-scoped operations run against an existing workspace.
     */
    if (!this._workspaceReady) {
      this._workspaceReady = this._getOrCreateWorkspace(this.workspaceId).then(
        () => undefined
      )
    }
    await this._workspaceReady
  }

  private async _updateWorkspace(
    workspaceId: string,
    params: {
      metadata?: Record<string, unknown>
      configuration?: WorkspaceConfig
    }
  ): Promise<WorkspaceResponse> {
    return this._http.put<WorkspaceResponse>(
      `/${API_VERSION}/workspaces/${workspaceId}`,
      {
        body: {
          metadata: params.metadata,
          configuration: workspaceConfigToApi(params.configuration),
        },
      }
    )
  }

  private async _deleteWorkspace(workspaceId: string): Promise<void> {
    await this._http.delete(`/${API_VERSION}/workspaces/${workspaceId}`)
  }

  private async _listWorkspaces(params?: {
    filters?: Record<string, unknown>
    page?: number
    size?: number
  }): Promise<PageResponse<WorkspaceResponse>> {
    return this._http.post<PageResponse<WorkspaceResponse>>(
      `/${API_VERSION}/workspaces/list`,
      {
        body: {
          filters: params?.filters,
        },
        query: {
          page: params?.page,
          size: params?.size,
        },
      }
    )
  }

  private async _searchWorkspace(
    workspaceId: string,
    params: {
      query: string
      filters?: Record<string, unknown>
      limit?: number
    }
  ): Promise<MessageResponse[]> {
    return this._http.post<MessageResponse[]>(
      `/${API_VERSION}/workspaces/${workspaceId}/search`,
      { body: params }
    )
  }

  private async _getQueueStatus(
    workspaceId: string,
    params?: QueueStatusParams
  ): Promise<QueueStatusResponse> {
    const query: Record<string, string | number | boolean | undefined> = {}
    if (params?.observer_id) query.observer_id = params.observer_id
    if (params?.sender_id) query.sender_id = params.sender_id
    if (params?.session_id) query.session_id = params.session_id

    return this._http.get<QueueStatusResponse>(
      `/${API_VERSION}/workspaces/${workspaceId}/queue/status`,
      { query }
    )
  }

  private async _listPeers(
    workspaceId: string,
    params?: {
      filters?: Record<string, unknown>
      page?: number
      size?: number
    }
  ): Promise<PageResponse<PeerResponse>> {
    return this._http.post<PageResponse<PeerResponse>>(
      `/${API_VERSION}/workspaces/${workspaceId}/peers/list`,
      {
        body: { filters: params?.filters },
        query: { page: params?.page, size: params?.size },
      }
    )
  }

  private async _getOrCreatePeer(
    workspaceId: string,
    params: {
      id: string
      metadata?: Record<string, unknown>
      configuration?: Record<string, unknown>
    }
  ): Promise<PeerResponse> {
    return this._http.post<PeerResponse>(
      `/${API_VERSION}/workspaces/${workspaceId}/peers`,
      { body: params }
    )
  }

  private async _listSessions(
    workspaceId: string,
    params?: {
      filters?: Record<string, unknown>
      page?: number
      size?: number
    }
  ): Promise<PageResponse<SessionResponse>> {
    return this._http.post<PageResponse<SessionResponse>>(
      `/${API_VERSION}/workspaces/${workspaceId}/sessions/list`,
      {
        body: { filters: params?.filters },
        query: { page: params?.page, size: params?.size },
      }
    )
  }

  private async _getOrCreateSession(
    workspaceId: string,
    params: {
      id: string
      metadata?: Record<string, unknown>
      configuration?: SessionConfig
    }
  ): Promise<SessionResponse> {
    return this._http.post<SessionResponse>(
      `/${API_VERSION}/workspaces/${workspaceId}/sessions`,
      {
        body: {
          id: params.id,
          metadata: params.metadata,
          configuration: sessionConfigToApi(params.configuration),
        },
      }
    )
  }

  // ===========================================================================
  // Public Methods
  // ===========================================================================

  /**
   * Get or create a peer with the given ID.
   *
   * Creates a Peer object that can be used to interact with the specified peer.
   * If metadata or configuration is provided, makes an API call to get/create the peer
   * immediately with those values.
   *
   * Provided metadata and configuration will overwrite existing data for this peer
   * if it already exists.
   *
   * @param id - Unique identifier for the peer within the workspace. Should be a
   *             stable identifier that can be used consistently across sessions.
   * @param metadata - Optional metadata dictionary to associate with this peer.
   *                   If set, will get/create peer immediately with metadata.
   * @param configuration - Optional configuration to set for this peer.
   *                        If set, will get/create peer immediately with flags.
   * @returns Promise resolving to a Peer object that can be used to send messages,
   *          join sessions, and query the peer's knowledge representations
   * @throws Error if the peer ID is empty or invalid
   */
  async peer(
    id: string,
    options?: {
      metadata?: PeerMetadata
      configuration?: PeerConfig
    }
  ): Promise<Peer> {
    await this._ensureWorkspace()
    const validatedId = PeerIdSchema.parse(id)
    const validatedMetadata = options?.metadata
      ? PeerMetadataSchema.parse(options.metadata)
      : undefined
    const validatedConfiguration = options?.configuration
      ? PeerConfigSchema.parse(options.configuration)
      : undefined

    if (validatedConfiguration || validatedMetadata) {
      const peerData = await this._getOrCreatePeer(this.workspaceId, {
        id: validatedId,
        configuration: peerConfigToApi(validatedConfiguration),
        metadata: validatedMetadata,
      })
      return new Peer(
        validatedId,
        this.workspaceId,
        this._http,
        peerData.metadata ?? undefined,
        peerConfigFromApi(peerData.configuration) ?? undefined,
        () => this._ensureWorkspace()
      )
    }

    return new Peer(
      validatedId,
      this.workspaceId,
      this._http,
      undefined,
      undefined,
      () => this._ensureWorkspace()
    )
  }

  /**
   * Get all peers in the current workspace.
   *
   * Makes an API call to retrieve all peers that have been created or used
   * within the current workspace. Returns a paginated result.
   *
   * @param filters - Optional filter criteria for peers. See [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
   * @returns Promise resolving to a Page of Peer objects representing all peers in the workspace
   */
  async peers(filters?: Filters): Promise<Page<Peer, PeerResponse>> {
    await this._ensureWorkspace()
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const peersPage = await this._listPeers(this.workspaceId, {
      filters: validatedFilter,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<PeerResponse>> => {
      return this._listPeers(this.workspaceId, {
        filters: validatedFilter,
        page,
        size,
      })
    }

    return new Page(
      peersPage,
      (peer) =>
        new Peer(
          peer.id,
          this.workspaceId,
          this._http,
          peer.metadata ?? undefined,
          peerConfigFromApi(peer.configuration) ?? undefined,
          () => this._ensureWorkspace()
        ),
      fetchNextPage
    )
  }

  /**
   * Get or create a session with the given ID.
   *
   * Creates a Session object that can be used to manage conversations between
   * multiple peers. If metadata or configuration is provided, makes an API call to
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
   * @param configuration - Optional configuration to set for this session.
   *                        If set, will get/create session immediately with flags.
   * @returns Promise resolving to a Session object that can be used to add peers,
   *          send messages, and manage conversation context
   * @throws Error if the session ID is empty or invalid
   */
  async session(
    id: string,
    options?: {
      metadata?: SessionMetadata
      configuration?: SessionConfig
    }
  ): Promise<Session> {
    await this._ensureWorkspace()
    const validatedId = SessionIdSchema.parse(id)
    const validatedMetadata = options?.metadata
      ? SessionMetadataSchema.parse(options.metadata)
      : undefined
    const validatedConfiguration = options?.configuration
      ? SessionConfigSchema.parse(options.configuration)
      : undefined

    if (validatedConfiguration || validatedMetadata) {
      const sessionData = await this._getOrCreateSession(this.workspaceId, {
        id: validatedId,
        configuration: validatedConfiguration,
        metadata: validatedMetadata,
      })
      return new Session(
        validatedId,
        this.workspaceId,
        this._http,
        sessionData.metadata ?? undefined,
        sessionConfigFromApi(sessionData.configuration) ?? undefined,
        () => this._ensureWorkspace()
      )
    }

    return new Session(
      validatedId,
      this.workspaceId,
      this._http,
      undefined,
      undefined,
      () => this._ensureWorkspace()
    )
  }

  /**
   * Get all sessions in the current workspace.
   *
   * Makes an API call to retrieve all sessions that have been created within
   * the current workspace.
   *
   * @param filters - Optional filter criteria for sessions. See [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
   * @returns Promise resolving to a Page of Session objects representing all sessions
   *          in the workspace. Returns an empty page if no sessions exist
   */
  async sessions(filters?: Filters): Promise<Page<Session, SessionResponse>> {
    await this._ensureWorkspace()
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const sessionsPage = await this._listSessions(this.workspaceId, {
      filters: validatedFilter,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<SessionResponse>> => {
      return this._listSessions(this.workspaceId, {
        filters: validatedFilter,
        page,
        size,
      })
    }

    return new Page(
      sessionsPage,
      (session) =>
        new Session(
          session.id,
          this.workspaceId,
          this._http,
          session.metadata ?? undefined,
          sessionConfigFromApi(session.configuration) ?? undefined,
          () => this._ensureWorkspace()
        ),
      fetchNextPage
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
    await this._ensureWorkspace()
    const workspace = await this._getOrCreateWorkspace(this.workspaceId)
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
    await this._ensureWorkspace()
    const validatedMetadata = WorkspaceMetadataSchema.parse(metadata)
    await this._updateWorkspace(this.workspaceId, {
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
   * @returns Promise resolving to the workspace's configuration.
   *          Returns an empty object if no configuration is set
   */
  async getConfiguration(): Promise<WorkspaceConfig> {
    await this._ensureWorkspace()
    const workspace = await this._getOrCreateWorkspace(this.workspaceId)
    this._configuration = workspaceConfigFromApi(workspace.configuration) || {}
    return this._configuration
  }

  /**
   * Set configuration for the current workspace.
   *
   * Makes an API call to update the configuration associated with the current workspace.
   * This will overwrite any existing configuration with the provided values.
   * This method also updates the cached configuration property.
   *
   * @param configuration - Configuration to associate with the workspace.
   *                        Includes reasoning, peerCard, summary, and dream settings.
   */
  async setConfiguration(configuration: WorkspaceConfig): Promise<void> {
    await this._ensureWorkspace()
    const validatedConfig = WorkspaceConfigSchema.parse(configuration)
    await this._updateWorkspace(this.workspaceId, {
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
    await this._ensureWorkspace()
    const workspace = await this._getOrCreateWorkspace(this.workspaceId)
    this._metadata = workspace.metadata || {}
    this._configuration = workspaceConfigFromApi(workspace.configuration) || {}
  }

  /**
   * Get all workspace IDs from the Honcho instance.
   *
   * Makes an API call to retrieve all workspace IDs that the authenticated
   * user has access to.
   *
   * @param filters - Optional filter criteria for workspaces. See [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
   * @returns Promise resolving to a Page of workspace ID strings. Returns an empty
   *          page if no workspaces are accessible or none exist
   */
  async workspaces(
    filters?: Filters
  ): Promise<Page<string, WorkspaceResponse>> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const workspacesPage = await this._listWorkspaces({
      filters: validatedFilter,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<WorkspaceResponse>> => {
      return this._listWorkspaces({
        filters: validatedFilter,
        page,
        size,
      })
    }

    return new Page(workspacesPage, (workspace) => workspace.id, fetchNextPage)
  }

  /**
   * Delete a workspace.
   *
   * Makes an API call to delete the specified workspace.
   *
   * @param workspaceId - The ID of the workspace to delete
   * @returns Promise that resolves when the workspace is deleted
   */
  async deleteWorkspace(workspaceId: string): Promise<void> {
    await this._deleteWorkspace(workspaceId)
  }

  /**
   * Search for messages in the current workspace.
   *
   * Makes an API call to search for messages in the current workspace.
   *
   * @param query - The search query to use
   * @param filters - Optional filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
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
    await this._ensureWorkspace()
    const validatedQuery = SearchQuerySchema.parse(query)
    const validatedFilters = options?.filters
      ? FilterSchema.parse(options.filters)
      : undefined
    const validatedLimit = options?.limit
      ? LimitSchema.parse(options.limit)
      : undefined
    const response = await this._searchWorkspace(this.workspaceId, {
      query: validatedQuery,
      filters: validatedFilters,
      limit: validatedLimit,
    })
    return response.map(Message.fromApiResponse)
  }

  /**
   * Get the queue processing status, optionally scoped to an observer, sender, and/or session.
   *
   * Makes an API call to retrieve the current status of the queue processing queue.
   * The queue is responsible for processing messages and updating peer representations.
   *
   * @param options - Configuration options for the status request
   * @param options.observer - Optional observer (ID string or Peer object) to scope the status to
   * @param options.sender - Optional sender (ID string or Peer object) to scope the status to
   * @param options.session - Optional session (ID string or Session object) to scope the status to
   * @returns Promise resolving to the queue status information including work unit counts
   */
  async queueStatus(
    options?: Omit<
      QueueStatusOptions,
      'observerId' | 'senderId' | 'sessionId'
    > & {
      observer?: string | Peer
      sender?: string | Peer
      session?: string | Session
    }
  ): Promise<QueueStatus> {
    await this._ensureWorkspace()
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

    const queryParams: QueueStatusParams = {}
    if (resolvedObserverId) queryParams.observer_id = resolvedObserverId
    if (resolvedSenderId) queryParams.sender_id = resolvedSenderId
    if (resolvedSessionId) queryParams.session_id = resolvedSessionId

    const status = await this._getQueueStatus(this.workspaceId, queryParams)
    return transformQueueStatus(status)
  }

  /**
   * Poll queueStatus until pendingWorkUnits and inProgressWorkUnits are both 0.
   * This allows you to guarantee that all messages have been processed by the queue for
   * use with the dialectic endpoint.
   *
   * The polling estimates sleep time by assuming each work unit takes 1 second.
   *
   * @param options - Configuration options for the status request
   * @param options.observer - Optional observer (ID string or Peer object) to scope the status to
   * @param options.sender - Optional sender (ID string or Peer object) to scope the status to
   * @param options.session - Optional session (ID string or Session object) to scope the status to
   * @param options.timeout - Optional timeout in seconds (default: 300 - 5 minutes)
   * @returns Promise resolving to the final queue status when processing is complete
   * @throws Error if timeout is exceeded before processing completes
   */
  async pollQueueStatus(
    options?: Omit<
      QueueStatusOptions,
      'observerId' | 'senderId' | 'sessionId'
    > & {
      observer?: string | Peer
      sender?: string | Peer
      session?: string | Session
      timeout?: number
    }
  ): Promise<QueueStatus> {
    await this._ensureWorkspace()
    const timeout = options?.timeout ?? 300 // Default to 5 minutes
    return pollUntilComplete(() => this.queueStatus(options), timeout)
  }

  /**
   * Return a string representation of the Honcho client.
   *
   * @returns A string representation suitable for debugging
   */
  toString(): string {
    return `Honcho(workspaceId='${this.workspaceId}', baseURL='${this._http.baseURL}')`
  }
}
