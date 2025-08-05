import HonchoCore from '@honcho-ai/core'
import type { DefaultQuery } from '@honcho-ai/core/src/core'
import type { Message } from '@honcho-ai/core/src/resources/workspaces/sessions/messages'
import type {
  DeriverStatus,
  WorkspaceDeriverStatusParams,
} from '@honcho-ai/core/src/resources/workspaces/workspaces'
import { Page } from './pagination'
import { Peer } from './peer'
import { Session } from './session'
import {
  type DeriverStatusOptions,
  DeriverStatusOptionsSchema,
  type Filter,
  FilterSchema,
  type HonchoConfig,
  HonchoConfigSchema,
  LimitSchema,
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
    this._client.workspaces.getOrCreate({ id: this.workspaceId })
  }

  /**
   * Get or create a peer with the given ID.
   *
   * Creates a Peer object that can be used to interact with the specified peer.
   * If metadata or config is provided, makes an API call to get/create the peer
   * immediately with those values.
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
    metadata?: PeerMetadata,
    config?: PeerConfig
  ): Promise<Peer> {
    const validatedId = PeerIdSchema.parse(id)
    const validatedMetadata = metadata
      ? PeerMetadataSchema.parse(metadata)
      : undefined
    const validatedConfig = config ? PeerConfigSchema.parse(config) : undefined
    const peer = new Peer(validatedId, this.workspaceId, this._client)

    if (validatedConfig || validatedMetadata) {
      await this._client.workspaces.peers.getOrCreate(this.workspaceId, {
        id: peer.id,
        configuration: validatedConfig,
        metadata: validatedMetadata,
      })
    }

    return peer
  }

  /**
   * Get all peers in the current workspace.
   *
   * Makes an API call to retrieve all peers that have been created or used
   * within the current workspace. Returns a paginated result.
   *
   * @param filter - Optional filter criteria for peers
   * @returns Promise resolving to a Page of Peer objects representing all peers in the workspace
   */
  async getPeers(filter?: Filter): Promise<Page<Peer>> {
    const validatedFilter = filter ? FilterSchema.parse(filter) : undefined
    const peersPage = await this._client.workspaces.peers.list(
      this.workspaceId,
      { filter: validatedFilter }
    )
    return new Page(
      peersPage,
      (peer) => new Peer(peer.id, this.workspaceId, this._client)
    )
  }

  /**
   * Get or create a session with the given ID.
   *
   * Creates a Session object that can be used to manage conversations between
   * multiple peers. If metadata or config is provided, makes an API call to
   * get/create the session immediately with those values.
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
    metadata?: SessionMetadata,
    config?: SessionConfig
  ): Promise<Session> {
    const validatedId = SessionIdSchema.parse(id)
    const validatedMetadata = metadata
      ? SessionMetadataSchema.parse(metadata)
      : undefined
    const validatedConfig = config
      ? SessionConfigSchema.parse(config)
      : undefined
    const session = new Session(validatedId, this.workspaceId, this._client)

    if (validatedConfig || validatedMetadata) {
      await this._client.workspaces.sessions.getOrCreate(this.workspaceId, {
        id: session.id,
        configuration: validatedConfig,
        metadata: validatedMetadata,
      })
    }

    return session
  }

  /**
   * Get all sessions in the current workspace.
   *
   * Makes an API call to retrieve all sessions that have been created within
   * the current workspace.
   *
   * @param filter - Optional filter criteria for sessions
   * @returns Promise resolving to a Page of Session objects representing all sessions
   *          in the workspace. Returns an empty page if no sessions exist
   */
  async getSessions(filter?: Filter): Promise<Page<Session>> {
    const validatedFilter = filter ? FilterSchema.parse(filter) : undefined
    const sessionsPage = await this._client.workspaces.sessions.list(
      this.workspaceId,
      { filter: validatedFilter }
    )
    return new Page(
      sessionsPage,
      (session) => new Session(session.id, this.workspaceId, this._client)
    )
  }

  /**
   * Get metadata for the current workspace.
   *
   * Makes an API call to retrieve metadata associated with the current workspace.
   * Workspace metadata can include settings, configuration, or any other
   * key-value data associated with the workspace.
   *
   * @returns Promise resolving to a dictionary containing the workspace's metadata.
   *          Returns an empty dictionary if no metadata is set
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const workspace = await this._client.workspaces.getOrCreate({
      id: this.workspaceId,
    })
    return workspace.metadata || {}
  }

  /**
   * Set metadata for the current workspace.
   *
   * Makes an API call to update the metadata associated with the current workspace.
   * This will overwrite any existing metadata with the provided values.
   *
   * @param metadata - A dictionary of metadata to associate with the workspace.
   *                   Keys must be strings, values can be any JSON-serializable type
   */
  async setMetadata(metadata: WorkspaceMetadata): Promise<void> {
    const validatedMetadata = WorkspaceMetadataSchema.parse(metadata)
    await this._client.workspaces.update(this.workspaceId, {
      metadata: validatedMetadata,
    })
  }

  /**
   * Get all workspace IDs from the Honcho instance.
   *
   * Makes an API call to retrieve all workspace IDs that the authenticated
   * user has access to.
   *
   * @param filter - Optional filter criteria for workspaces
   * @returns Promise resolving to a list of workspace ID strings. Returns an empty
   *          list if no workspaces are accessible or none exist
   */
  async getWorkspaces(filter?: Filter): Promise<string[]> {
    const validatedFilter = filter ? FilterSchema.parse(filter) : undefined
    const workspacesPage = await this._client.workspaces.list({
      filter: validatedFilter,
    })
    const ids: string[] = []
    for await (const workspace of workspacesPage) {
      ids.push(workspace.id)
    }
    return ids
  }

  /**
   * Search for messages in the current workspace.
   *
   * Makes an API call to search for messages in the current workspace.
   *
   * @param query - The search query to use
   * @param limit - Number of results to return (1-100, default: 10).
   * @returns Promise resolving to an array of Message objects representing the search results.
   *          Returns an empty array if no messages are found.
   * @throws Error if the search query is empty or invalid
   */
  async search(query: string, limit?: number): Promise<Message[]> {
    const validatedQuery = SearchQuerySchema.parse(query)
    const validatedLimit = limit ? LimitSchema.parse(limit) : undefined
    return await this._client.workspaces.search(this.workspaceId, {
      query: validatedQuery,
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
   * @param options.observerId - Optional observer ID to scope the status to
   * @param options.senderId - Optional sender ID to scope the status to
   * @param options.sessionId - Optional session ID to scope the status to
   * @returns Promise resolving to the deriver status information including work unit counts
   */
  async getDeriverStatus(options?: DeriverStatusOptions): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<string, DeriverStatus.Sessions>
  }> {
    const validatedOptions = options
      ? DeriverStatusOptionsSchema.parse(options)
      : undefined
    const queryParams: WorkspaceDeriverStatusParams = {}
    if (validatedOptions?.observerId)
      queryParams.observer_id = validatedOptions.observerId
    if (validatedOptions?.senderId)
      queryParams.sender_id = validatedOptions.senderId
    if (validatedOptions?.sessionId)
      queryParams.session_id = validatedOptions.sessionId

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
   * @param options.observerId - Optional observer ID to scope the status to
   * @param options.senderId - Optional sender ID to scope the status to
   * @param options.sessionId - Optional session ID to scope the status to
   * @param options.timeoutMs - Optional timeout in milliseconds (default: 300000 - 5 minutes)
   * @returns Promise resolving to the final deriver status when processing is complete
   * @throws Error if timeout is exceeded before processing completes
   */
  async pollDeriverStatus(options?: DeriverStatusOptions): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<string, any>
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
   * Return a string representation of the Honcho client.
   *
   * @returns A string representation suitable for debugging
   */
  toString(): string {
    return `Honcho(workspaceId='${this.workspaceId}', baseURL='${this._client.baseURL}')`
  }
}
