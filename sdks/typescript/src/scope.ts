import { API_VERSION } from './api-version'
import type { HonchoHTTPClient } from './http/client'
import { Page } from './pagination'
import { Session } from './session'
import type {
  PageResponse,
  ScopeStatusResponse,
  SessionResponse,
} from './types/api'
import { resolveId } from './utils'
import {
  ScopeSessionsSchema,
  SessionIdSchema,
  sessionConfigFromApi,
} from './validation'

/**
 * Backfill job state for one session in a scope.
 */
export interface ScopeBackfillState {
  state: 'pending' | 'completed' | 'failed'
  updatedAt: string
  /**
   * Number of documents copied into the scope. Present only once the backfill
   * for this session completes.
   */
  docsCopied?: number
}

/**
 * Backfill/reconciliation progress for a scope, keyed by session ID.
 *
 * Only sessions that have had a backfill enqueued appear. A scope whose
 * sessions were all empty when added has an empty `backfillStatus`.
 */
export interface ScopeStatus {
  backfillStatus: Record<string, ScopeBackfillState>
}

/**
 * Represents a scope in the Honcho system.
 *
 * A scope is a named set of sessions that acts as a visibility boundary. Recall
 * performed through a scope sees only what happened in that scope's sessions,
 * while the underlying peer keeps its single unified representation across
 * everything it has ever participated in.
 *
 * Membership changes are applied asynchronously: adding a session that already
 * has messages copies its existing conclusions into the scope, and removing one
 * reconciles them back out. Poll {@link Scope.status} to watch that settle.
 *
 * @example
 * ```typescript
 * const therapy = await honcho.scope('therapy')
 * await therapy.addSessions([session1, session2])
 *
 * // Ask a question answered only from the therapy sessions
 * const answer = await user.chat('What is stressing them out?', {
 *   scope: 'therapy',
 * })
 * ```
 */
export class Scope {
  /**
   * Unique identifier for this scope, without the server-side `scope.` prefix.
   */
  readonly id: string
  /**
   * Workspace ID for scoping operations.
   */
  readonly workspaceId: string
  private _http: HonchoHTTPClient
  private _metadata?: Record<string, unknown>
  private _createdAt?: string
  private _ensureWorkspace: () => Promise<void>

  /**
   * Cached metadata for this scope. May be stale if the scope was not recently
   * fetched from the API.
   */
  get metadata(): Record<string, unknown> | undefined {
    return this._metadata
  }

  /**
   * Timestamp when this scope was created. Only available if fetched from the API.
   */
  get createdAt(): string | undefined {
    return this._createdAt
  }

  /**
   * Initialize a new Scope. **Do not call this directly, use the client.scope() method instead.**
   *
   * @param id - Unprefixed scope name, unique within the workspace
   * @param workspaceId - Workspace ID for scoping operations
   * @param http - Reference to the HTTP client instance
   * @param metadata - Optional metadata to initialize the cached value
   * @param ensureWorkspace - Callback that guarantees the workspace exists
   * @param createdAt - Creation timestamp, if already fetched
   */
  constructor(
    id: string,
    workspaceId: string,
    http: HonchoHTTPClient,
    metadata?: Record<string, unknown>,
    ensureWorkspace: () => Promise<void> = async () => undefined,
    createdAt?: string
  ) {
    this.id = id
    this.workspaceId = workspaceId
    this._http = http
    this._metadata = metadata
    this._ensureWorkspace = ensureWorkspace
    this._createdAt = createdAt
  }

  // ===========================================================================
  // Private API Methods
  // ===========================================================================

  private get _basePath(): string {
    return `/${API_VERSION}/workspaces/${this.workspaceId}/scopes/${this.id}`
  }

  private async _addSessions(sessionIds: string[]): Promise<void> {
    await this._ensureWorkspace()
    await this._http.post(`${this._basePath}/sessions`, {
      body: { session_ids: sessionIds },
    })
  }

  private async _removeSession(sessionId: string): Promise<void> {
    await this._ensureWorkspace()
    await this._http.delete(`${this._basePath}/sessions/${sessionId}`)
  }

  private async _listSessions(params?: {
    page?: number
    size?: number
    reverse?: boolean
  }): Promise<PageResponse<SessionResponse>> {
    await this._ensureWorkspace()
    return this._http.post<PageResponse<SessionResponse>>(
      `${this._basePath}/sessions/list`,
      {
        query: {
          page: params?.page,
          size: params?.size,
          reverse: params?.reverse ? 'true' : undefined,
        },
      }
    )
  }

  private async _getStatus(): Promise<ScopeStatusResponse> {
    await this._ensureWorkspace()
    return this._http.get<ScopeStatusResponse>(`${this._basePath}/status`)
  }

  // ===========================================================================
  // Public API Methods
  // ===========================================================================

  /**
   * Add sessions to this scope.
   *
   * Every named session must already exist. Adding a session that is already a
   * member is a no-op.
   *
   * Sessions that already hold messages are backfilled into the scope
   * asynchronously, so recall through this scope may not reflect their history
   * immediately — poll {@link Scope.status} to watch that complete.
   *
   * @param sessions - Sessions to add, as ID strings or Session objects. At most
   *                   100 per call, matching the server's limit; split larger
   *                   membership changes into separate calls so a failure names
   *                   the batch that failed.
   */
  async addSessions(sessions: (string | Session)[]): Promise<void> {
    const sessionIds = ScopeSessionsSchema.parse(sessions.map(resolveId))
    await this._addSessions(sessionIds)
  }

  /**
   * Remove a session from this scope.
   *
   * Conclusions copied or derived while the session was a member are
   * reconciled out asynchronously, and the scope's peer card is rebuilt from
   * whatever evidence remains. Poll {@link Scope.status} to watch that settle.
   *
   * @param session - Session to remove, as an ID string or a Session object
   * @throws If the session ID is malformed
   */
  async removeSession(session: string | Session): Promise<void> {
    // Validated because this ID is interpolated into a request *path*: an
    // unvalidated value silently changes which resource the request addresses.
    // `valid-session?typo` would target `valid-session` with a stray query
    // string, removing the wrong session and reconciling against it.
    await this._removeSession(SessionIdSchema.parse(resolveId(session)))
  }

  /**
   * Get the sessions that are members of this scope.
   *
   * Ordered by how long each session has been a member — longest-standing
   * first, or most recently added first when `reverse` is true.
   *
   * @param options - Pagination options: `page`, `size`, and `reverse`
   * @returns Promise resolving to a paginated list of member Sessions
   */
  async sessions(options?: {
    page?: number
    size?: number
    reverse?: boolean
  }): Promise<Page<Session, SessionResponse>> {
    const reverse = options?.reverse
    const sessionsPage = await this._listSessions({
      page: options?.page,
      size: options?.size,
      reverse,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<SessionResponse>> => {
      return this._listSessions({ page, size, reverse })
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
          () => this._ensureWorkspace(),
          session.created_at,
          session.is_active
        ),
      fetchNextPage
    )
  }

  /**
   * Get the backfill/reconciliation progress for this scope.
   *
   * Use this after a membership change to tell "the scope knows nothing about
   * that session yet" apart from "the scope has caught up and there is genuinely
   * nothing to recall".
   *
   * @returns Promise resolving to per-session backfill state
   */
  async status(): Promise<ScopeStatus> {
    const response = await this._getStatus()
    return {
      backfillStatus: Object.fromEntries(
        Object.entries(response.backfill_status ?? {}).map(
          ([sessionId, job]) => [
            sessionId,
            {
              state: job.state,
              updatedAt: job.updated_at,
              docsCopied: job.docs_copied,
            },
          ]
        )
      ),
    }
  }

  toString(): string {
    return `Scope(id='${this.id}', workspaceId='${this.workspaceId}')`
  }
}
