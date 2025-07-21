import HonchoCore from '@honcho-ai/core'
import { Page } from './pagination'
import { Peer } from './peer'
import { Session } from './session'

/**
 * Main client for the Honcho TypeScript SDK.
 * Provides access to peers, sessions, and workspace operations.
 */
export class Honcho {
  private _client: InstanceType<typeof HonchoCore>
  readonly workspaceId: string

  /**
   * Initialize the Honcho client.
   */
  constructor(options: {
    apiKey?: string
    environment?: 'local' | 'production' | 'demo'
    baseURL?: string
    workspaceId?: string
    timeout?: number
    maxRetries?: number
    defaultHeaders?: Record<string, string>
    defaultQuery?: Record<string, unknown>
  }) {
    this.workspaceId =
      options.workspaceId || process.env.HONCHO_WORKSPACE_ID || 'default'
    this._client = new HonchoCore({
      apiKey: options.apiKey || process.env.HONCHO_API_KEY,
      environment: options.environment,
      baseURL: options.baseURL || process.env.HONCHO_URL,
      timeout: options.timeout,
      maxRetries: options.maxRetries,
      defaultHeaders: options.defaultHeaders,
      defaultQuery: options.defaultQuery as any,
    }) as any
    this._client.workspaces.getOrCreate({ id: this.workspaceId })
  }

  /**
   * Get or create a peer with the given ID.
   */
  peer(id: string, options?: { config?: Record<string, unknown> }): Peer {
    if (!id || typeof id !== 'string') {
      throw new Error('Peer ID must be a non-empty string')
    }
    return new Peer(id, this, options?.config)
  }

  /**
   * Get all peers in the current workspace.
   */
  async getPeers(
    filter?: { [key: string]: unknown } | null
  ): Promise<Page<Peer>> {
    const peersPage = await this._client.workspaces.peers.list(
      this.workspaceId,
      { filter }
    )
    return new Page(peersPage, (peer: any) => new Peer(peer.id, this))
  }

  /**
   * Get or create a session with the given ID.
   */
  session(id: string, options?: { config?: Record<string, unknown> }): Session {
    if (!id || typeof id !== 'string') {
      throw new Error('Session ID must be a non-empty string')
    }
    return new Session(id, this, options?.config)
  }

  /**
   * Get all sessions in the current workspace.
   */
  async getSessions(
    filter?: { [key: string]: unknown } | null
  ): Promise<Page<Session>> {
    const sessionsPage = await this._client.workspaces.sessions.list(
      this.workspaceId,
      { filter }
    )
    return new Page(
      sessionsPage,
      (session: any) => new Session(session.id, this)
    )
  }

  /**
   * Get metadata for the current workspace.
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const workspace = await this._client.workspaces.getOrCreate({
      id: this.workspaceId,
    })
    return workspace.metadata || {}
  }

  /**
   * Set metadata for the current workspace.
   */
  async setMetadata(metadata: Record<string, unknown>): Promise<void> {
    await this._client.workspaces.update(this.workspaceId, { metadata })
  }

  /**
   * Get all workspace IDs from the Honcho instance.
   */
  async getWorkspaces(
    filter?: { [key: string]: unknown } | null
  ): Promise<string[]> {
    const workspacesPage = await this._client.workspaces.list({ filter })
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
   * @param query The search query to use
   * @returns A Page of Message objects representing the search results.
   *          Returns an empty page if no messages are found.
   */
  async search(query: string): Promise<Page<any>> {
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      throw new Error('Search query must be a non-empty string')
    }
    const messagesPage = await this._client.workspaces.search(
      this.workspaceId,
      { body: query }
    )
    return new Page(messagesPage)
  }

  /**
   * Get the deriver processing status, optionally scoped to an observer, sender, and/or session.
   *
   * @param options Configuration options for the status request
   * @param options.observerId Optional observer ID to scope the status to
   * @param options.senderId Optional sender ID to scope the status to
   * @param options.sessionId Optional session ID to scope the status to
   * @returns Promise resolving to the deriver status information
   */
  async getDeriverStatus(options?: {
    observerId?: string
    senderId?: string
    sessionId?: string
  }): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<string, any>
  }> {
    const queryParams: any = {}
    if (options?.observerId) queryParams.observer_id = options.observerId
    if (options?.senderId) queryParams.sender_id = options.senderId
    if (options?.sessionId) queryParams.session_id = options.sessionId

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
   * @param options Configuration options for the status request
   * @param options.observerId Optional observer ID to scope the status to
   * @param options.senderId Optional sender ID to scope the status to
   * @param options.sessionId Optional session ID to scope the status to
   * @param options.timeoutMs Optional timeout in milliseconds (default: 300000 - 5 minutes)
   * @returns Promise resolving to the final deriver status when processing is complete
   * @throws Error if timeout is exceeded before processing completes
   */
  async pollDeriverStatus(options?: {
    observerId?: string
    senderId?: string
    sessionId?: string
    timeoutMs?: number
  }): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<string, any>
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
}
