import { API_VERSION } from './api-version'
import { WorkspaceConclusions } from './conclusions'
import { HonchoHTTPClient } from './http/client'
import { NotFoundError } from './http/errors'
import {
  createDialecticStream,
  type DialecticStreamResponse,
} from './http/streaming'
import { Message } from './message'
import { Page } from './pagination'
import { Peer } from './peer'
import { Scope } from './scope'
import { Session } from './session'
import type {
  ChatResponse,
  DreamRun,
  DreamRunDetail,
  IngestAccount,
  IngestAccountResponse,
  IngestDeriveResponse,
  IngestDeriveResult,
  IngestLinkResponse,
  IngestLinkResult,
  IngestPerson,
  IngestPersonResponse,
  IngestRecordResult,
  IngestResponse,
  IngestResult,
  IngestSession,
  IngestSessionResponse,
  IngestStatus,
  IngestStatusResponse,
  IngestSuggestion,
  IngestSuggestionAccepted,
  IngestSuggestionResponse,
  IngestSyncSource,
  IngestSyncSourceResponse,
  IngestSyncState,
  IngestSyncStateResponse,
  IngestUnlinkResponse,
  IngestUnlinkResult,
  MessageResponse,
  PageResponse,
  PeerResponse,
  QueueStatus,
  QueueStatusParams,
  QueueStatusResponse,
  ScopeResponse,
  SessionPeerConfigParams,
  SessionResponse,
  WorkspaceChatParams,
  WorkspaceChatResponse,
  WorkspaceResponse,
} from './types/api'
import { resolveId, transformQueueStatus } from './utils'
import {
  FilterSchema,
  type Filters,
  type HonchoConfig,
  HonchoConfigSchema,
  type IngestRecord,
  type IngestRecordDelete,
  ingestRecordToApi,
  LimitSchema,
  normalizeListOptions,
  type PeerAddition,
  PeerAdditionToApiSchema,
  type PeerConfig,
  PeerConfigSchema,
  PeerIdSchema,
  type PeerMetadata,
  PeerMetadataSchema,
  peerConfigFromApi,
  peerConfigToApi,
  type QueueStatusOptions,
  ScopeIdSchema,
  SearchQuerySchema,
  type SessionConfig,
  SessionConfigSchema,
  SessionIdSchema,
  type SessionMetadata,
  SessionMetadataSchema,
  SessionScopesSchema,
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
/** Records per ingest request: the most the ingest route takes at once. */
const INGEST_BATCH_SIZE = 100

function ingestAccountFromApi(a: IngestAccountResponse): IngestAccount {
  return {
    source: a.source,
    sourceInstance: a.source_instance,
    id: a.id ?? undefined,
    displayName: a.display_name ?? undefined,
  }
}
function ingestSyncSourceFromApi(
  s: IngestSyncSourceResponse
): IngestSyncSource {
  return {
    source: s.source,
    sourceInstance: s.source_instance,
    lastSyncedAt: s.last_synced_at,
    lastRecordAt: s.last_record_at ?? undefined,
    error: s.error ?? undefined,
  }
}
function ingestSyncStateFromApi(s: IngestSyncStateResponse): IngestSyncState {
  return {
    ...ingestSyncSourceFromApi(s),
    position: s.position,
    revision: s.revision,
  }
}
function ingestSuggestionFromApi(
  s: IngestSuggestionResponse
): IngestSuggestion {
  return {
    id: s.id,
    kind: s.kind,
    confidence: s.confidence,
    score: s.score,
    status: s.status,
    linked: s.linked,
    accounts: s.accounts.map((a) => ({
      source: a.source,
      sourceInstance: a.source_instance,
      id: a.id ?? undefined,
      displayName: a.display_name ?? undefined,
      handle: a.handle ?? undefined,
      peerId: a.peer_id ?? undefined,
      records: a.records,
    })),
    evidence: s.evidence,
    stage: s.stage,
    refreshedAt: s.refreshed_at,
  }
}
type ReasoningLevel = 'minimal' | 'low' | 'medium' | 'high' | 'max'

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
   * Private cached server-assigned identifier for this workspace.
   */
  private _uid?: string
  /**
   * Memoized workspace get-or-create call.
   */
  private _workspaceReady?: Promise<void>
  /**
   * Whether this client creates its workspace as an auto workspace.
   */
  private readonly _auto: boolean

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
   * Immutable identifier Honcho assigns to this workspace. Unlike
   * `workspaceId`, the client-chosen name, it is generated by the server. Set
   * once the client has fetched the workspace (its first API call does), and
   * undefined before that or on servers that predate it.
   */
  get uid(): string | undefined {
    return this._uid
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
   * Workspace-wide conclusions. No observer/observed pair is implied.
   *
   * Use this to list or look up conclusions across the workspace. Pair-
   * scoped create/query/delete stay on `peer.conclusions`.
   */
  get conclusions(): WorkspaceConclusions {
    return new WorkspaceConclusions(this._http, this.workspaceId, () =>
      this._ensureWorkspace()
    )
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
   * @param options.auto - Create the workspace as an auto workspace, which `ingest`
   *                       writes to. Experimental, like ingest: it may change or
   *                       stop working without notice. Only a new workspace can be
   *                       made auto: on an existing workspace that isn't auto, the
   *                       first call that creates the workspace throws
   *                       `ConflictError`. An existing auto workspace works with or
   *                       without it.
   */
  constructor(options: HonchoConfig = {}) {
    const validatedOptions = HonchoConfigSchema.parse(options)
    this._auto = validatedOptions.auto ?? false
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
    const workspace = await this._http.post<WorkspaceResponse>(
      `/${API_VERSION}/workspaces`,
      {
        body: {
          id,
          metadata: params?.metadata,
          configuration: workspaceConfigToApi(params?.configuration),
          ...(this._auto ? { auto: true } : {}),
        },
      }
    )
    this._uid = workspace.uid
    return workspace
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
    reverse?: boolean
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
          reverse: params?.reverse ? 'true' : undefined,
        },
      }
    )
  }

  private async _searchWorkspace(
    workspaceId: string,
    params: {
      query: string
      filters?: Record<string, unknown>
      scope?: string
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
      reverse?: boolean
    }
  ): Promise<PageResponse<PeerResponse>> {
    return this._http.post<PageResponse<PeerResponse>>(
      `/${API_VERSION}/workspaces/${workspaceId}/peers/list`,
      {
        body: { filters: params?.filters },
        query: {
          page: params?.page,
          size: params?.size,
          reverse: params?.reverse ? 'true' : undefined,
        },
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

  private async _getOrCreateScope(
    workspaceId: string,
    params: {
      id: string
      metadata?: Record<string, unknown>
    }
  ): Promise<ScopeResponse> {
    return this._http.post<ScopeResponse>(
      `/${API_VERSION}/workspaces/${workspaceId}/scopes`,
      { body: params }
    )
  }

  private async _getScope(
    workspaceId: string,
    scopeId: string
  ): Promise<ScopeResponse> {
    return this._http.get<ScopeResponse>(
      `/${API_VERSION}/workspaces/${workspaceId}/scopes/${scopeId}`
    )
  }

  /**
   * Fetch one page of the workspace's scopes.
   *
   * @param workspaceId - Workspace to list
   * @param params - Filters, pagination, and whether to include each scope's
   *                 `session_count`
   * @returns The raw page response
   */
  private async _listScopes(
    workspaceId: string,
    params?: {
      filters?: Record<string, unknown>
      page?: number
      size?: number
      reverse?: boolean
      includeSessionCount?: boolean
    }
  ): Promise<PageResponse<ScopeResponse>> {
    return this._http.post<PageResponse<ScopeResponse>>(
      `/${API_VERSION}/workspaces/${workspaceId}/scopes/list`,
      {
        body: { filters: params?.filters },
        query: {
          page: params?.page,
          size: params?.size,
          reverse: params?.reverse ? 'true' : undefined,
          include_session_count: params?.includeSessionCount
            ? 'true'
            : undefined,
        },
      }
    )
  }

  /**
   * Fetch one page of the workspace's sessions.
   *
   * @param workspaceId - Workspace to list
   * @param params - Filters, pagination, the `scope` / `inNoScope` membership
   *                 filters, and whether to include each session's `scopes`
   * @returns The raw page response
   */
  private async _listSessions(
    workspaceId: string,
    params?: {
      filters?: Record<string, unknown>
      page?: number
      size?: number
      reverse?: boolean
      scope?: string
      inNoScope?: boolean
      includeScopes?: boolean
    }
  ): Promise<PageResponse<SessionResponse>> {
    return this._http.post<PageResponse<SessionResponse>>(
      `/${API_VERSION}/workspaces/${workspaceId}/sessions/list`,
      {
        body: {
          filters: params?.filters,
          scope: params?.scope,
          in_no_scope: params?.inNoScope ? true : undefined,
        },
        query: {
          page: params?.page,
          size: params?.size,
          reverse: params?.reverse ? 'true' : undefined,
          include_scopes: params?.includeScopes ? 'true' : undefined,
        },
      }
    )
  }

  private async _getOrCreateSession(
    workspaceId: string,
    params: {
      id: string
      metadata?: Record<string, unknown>
      configuration?: SessionConfig
      peers?: Record<string, SessionPeerConfigParams>
      scopes?: string[]
    }
  ): Promise<SessionResponse> {
    return this._http.post<SessionResponse>(
      `/${API_VERSION}/workspaces/${workspaceId}/sessions`,
      {
        body: {
          id: params.id,
          metadata: params.metadata,
          configuration: sessionConfigToApi(params.configuration),
          peers: params.peers,
          scopes: params.scopes,
        },
      }
    )
  }

  private async _workspaceChat(
    workspaceId: string,
    params: WorkspaceChatParams
  ): Promise<WorkspaceChatResponse> {
    await this._ensureWorkspace()
    return this._http.post<WorkspaceChatResponse>(
      `/${API_VERSION}/workspaces/${workspaceId}/chat`,
      { body: params }
    )
  }

  private async _workspaceChatStream(
    workspaceId: string,
    params: Omit<WorkspaceChatParams, 'stream'>
  ): Promise<Response> {
    await this._ensureWorkspace()
    return this._http.stream(
      'POST',
      `/${API_VERSION}/workspaces/${workspaceId}/chat`,
      {
        body: {
          ...params,
          stream: true,
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
      () => this._ensureWorkspace(),
      peerData.created_at,
      peerData.uid
    )
  }

  /**
   * Get all peers in the current workspace.
   *
   * Makes an API call to retrieve all peers that have been created or used
   * within the current workspace. Returns a paginated result.
   *
   * @param options - Either a legacy raw filter object or an options object with
   *                  `filters`, `page`, `size`, and `reverse`. See
   *                  [search filters documentation](https://honcho.dev/docs/v3/documentation/core-concepts/features/using-filters).
   * @returns Promise resolving to a Page of Peer objects representing all peers in the workspace
   */
  async peers(
    options?:
      | Filters
      | {
          filters?: Filters
          page?: number
          size?: number
          reverse?: boolean
        }
  ): Promise<Page<Peer, PeerResponse>> {
    await this._ensureWorkspace()
    const normalizedOptions = normalizeListOptions(options, [
      'filters',
      'page',
      'size',
      'reverse',
    ])
    const validatedFilter = normalizedOptions.filters
      ? FilterSchema.parse(normalizedOptions.filters)
      : undefined
    const reverse = normalizedOptions.reverse
    const peersPage = await this._listPeers(this.workspaceId, {
      filters: validatedFilter,
      page: normalizedOptions.page,
      size: normalizedOptions.size,
      reverse,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<PeerResponse>> => {
      return this._listPeers(this.workspaceId, {
        filters: validatedFilter,
        page,
        size,
        reverse,
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
          () => this._ensureWorkspace(),
          peer.created_at,
          peer.uid
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
   * @param options.metadata - Optional metadata dictionary to associate with this session.
   *                   If set, will get/create session immediately with metadata.
   * @param options.configuration - Optional configuration to set for this session.
   *                        If set, will get/create session immediately with flags.
   * @param options.peers - Optional peers to attach to the session at creation.
   *                Accepts the same shape as `session.addPeers()` (peer ID strings,
   *                Peer objects, arrays of either, or a record with per-peer config).
   * @param options.scopes - Optional scopes this session should join. Each scope is
   *                created if it does not exist yet. Attaching at creation avoids the
   *                asynchronous backfill that a later `scope.addSessions()` triggers,
   *                since there is no history to copy.
   * @returns Promise resolving to a Session object that can be used to add peers,
   *          send messages, and manage conversation context
   * @throws Error if the session ID is empty or invalid
   */
  async session(
    id: string,
    options?: {
      metadata?: SessionMetadata
      configuration?: SessionConfig
      peers?: PeerAddition
      scopes?: (string | Scope)[]
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
    const validatedPeers =
      options?.peers !== undefined
        ? PeerAdditionToApiSchema.parse(options.peers)
        : undefined
    const validatedScopes =
      options?.scopes !== undefined
        ? SessionScopesSchema.parse(options.scopes.map(resolveId))
        : undefined

    const sessionData = await this._getOrCreateSession(this.workspaceId, {
      id: validatedId,
      configuration: validatedConfiguration,
      metadata: validatedMetadata,
      peers: validatedPeers,
      scopes: validatedScopes,
    })
    return new Session(
      validatedId,
      this.workspaceId,
      this._http,
      sessionData.metadata ?? undefined,
      sessionConfigFromApi(sessionData.configuration) ?? undefined,
      () => this._ensureWorkspace(),
      sessionData.created_at,
      sessionData.is_active,
      sessionData.uid
    )
  }

  /**
   * Get or create a scope with the given ID.
   *
   * A scope is a named set of sessions that acts as a visibility boundary: recall
   * performed through the scope sees only what happened in its sessions, while the
   * underlying peer keeps its single unified representation of everything.
   *
   * @param id - Unprefixed scope name, unique within the workspace
   * @param options.metadata - Optional metadata to associate with this scope
   * @returns Promise resolving to a Scope object for managing membership
   * @throws Error if the scope ID is empty or invalid, or if a peer already occupies
   *         the scope's reserved internal name
   *
   * @example
   * ```typescript
   * const therapy = await honcho.scope('therapy')
   * await therapy.addSessions([session1, session2])
   * ```
   */
  async scope(
    id: string,
    options?: {
      metadata?: Record<string, unknown>
    }
  ): Promise<Scope> {
    await this._ensureWorkspace()
    const validatedId = ScopeIdSchema.parse(id)

    const scopeData = await this._getOrCreateScope(this.workspaceId, {
      id: validatedId,
      metadata: options?.metadata,
    })
    return new Scope(
      validatedId,
      this.workspaceId,
      this._http,
      scopeData.metadata ?? undefined,
      () => this._ensureWorkspace(),
      scopeData.created_at
    )
  }

  /**
   * Get an existing scope by ID without creating it.
   *
   * Unlike {@link scope}, this never creates the scope, so it is safe for
   * lookups where a typo must not provision a new recall boundary.
   *
   * @param id - Unprefixed scope name, unique within the workspace
   * @returns Promise resolving to a Scope object for managing membership
   * @throws {NotFoundError} if no scope with this ID exists in the workspace
   * @throws Error if the scope ID is empty or invalid
   */
  async getScope(id: string): Promise<Scope> {
    await this._ensureWorkspace()
    const validatedId = ScopeIdSchema.parse(id)

    const scopeData = await this._getScope(this.workspaceId, validatedId)
    return new Scope(
      validatedId,
      this.workspaceId,
      this._http,
      scopeData.metadata ?? undefined,
      () => this._ensureWorkspace(),
      scopeData.created_at
    )
  }

  /**
   * Get all scopes in the current workspace.
   *
   * @param options - `filters` (`id` matches the scope name, e.g.
   *                  `{ id: { icontains: 'sup' } }`), `includeSessionCount` to set
   *                  each scope's `sessionCount`, and pagination: `page`, `size`,
   *                  and `reverse`
   * @returns Promise resolving to a Page of Scope objects. Returns an empty page if
   *          no scopes exist
   */
  async scopes(options?: {
    filters?: Filters
    page?: number
    size?: number
    reverse?: boolean
    includeSessionCount?: boolean
  }): Promise<Page<Scope, ScopeResponse>> {
    await this._ensureWorkspace()
    const listOptions = {
      filters: options?.filters
        ? FilterSchema.parse(options.filters)
        : undefined,
      reverse: options?.reverse,
      includeSessionCount: options?.includeSessionCount,
    }
    const scopesPage = await this._listScopes(this.workspaceId, {
      ...listOptions,
      page: options?.page,
      size: options?.size,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<ScopeResponse>> => {
      return this._listScopes(this.workspaceId, { ...listOptions, page, size })
    }

    return new Page(
      scopesPage,
      (scope) =>
        new Scope(
          scope.id,
          this.workspaceId,
          this._http,
          scope.metadata ?? undefined,
          () => this._ensureWorkspace(),
          scope.created_at,
          scope.session_count
        ),
      fetchNextPage
    )
  }

  /**
   * Get all sessions in the current workspace.
   *
   * Makes an API call to retrieve all sessions that have been created within
   * the current workspace.
   *
   * @param options - Either a legacy raw filter object or an options object with
   *                  `filters`, `page`, `size`, `reverse`, `scope`, `inNoScope`
   *                  and `includeScopes`. See
   *                  [search filters documentation](https://honcho.dev/docs/v3/documentation/core-concepts/features/using-filters).
   *                  `scope` lists only that scope's sessions (the server
   *                  returns 404 if it does not exist); `inNoScope` lists only
   *                  sessions in no scope; `includeScopes` sets each returned
   *                  session's `scopes`.
   * @returns Promise resolving to a Page of Session objects representing all sessions
   *          in the workspace. Returns an empty page if no sessions exist
   */
  async sessions(
    options?:
      | Filters
      | {
          filters?: Filters
          page?: number
          size?: number
          reverse?: boolean
          scope?: string
          inNoScope?: boolean
          includeScopes?: boolean
        }
  ): Promise<Page<Session, SessionResponse>> {
    await this._ensureWorkspace()
    const normalizedOptions = normalizeListOptions(options, [
      'filters',
      'page',
      'size',
      'reverse',
      'scope',
      'inNoScope',
      'includeScopes',
    ])
    const validatedFilter = normalizedOptions.filters
      ? FilterSchema.parse(normalizedOptions.filters)
      : undefined
    const listOptions = {
      filters: validatedFilter,
      reverse: normalizedOptions.reverse,
      scope: normalizedOptions.scope,
      inNoScope: normalizedOptions.inNoScope,
      includeScopes: normalizedOptions.includeScopes,
    }
    const sessionsPage = await this._listSessions(this.workspaceId, {
      ...listOptions,
      page: normalizedOptions.page,
      size: normalizedOptions.size,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<SessionResponse>> => {
      return this._listSessions(this.workspaceId, {
        ...listOptions,
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
          () => this._ensureWorkspace(),
          session.created_at,
          session.is_active,
          session.uid,
          session.scopes
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
   * Pass an empty object to clear explicit workspace settings and use system defaults.
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
   * @param options - Either a legacy raw filter object or an options object with
   *                  `filters`, `page`, `size`, and `reverse`. See
   *                  [search filters documentation](https://honcho.dev/docs/v3/documentation/core-concepts/features/using-filters).
   * @returns Promise resolving to a Page of workspace ID strings. Returns an empty
   *          page if no workspaces are accessible or none exist
   */
  async workspaces(
    options?:
      | Filters
      | {
          filters?: Filters
          page?: number
          size?: number
          reverse?: boolean
        }
  ): Promise<Page<string, WorkspaceResponse>> {
    const normalizedOptions = normalizeListOptions(options, [
      'filters',
      'page',
      'size',
      'reverse',
    ])
    const validatedFilter = normalizedOptions.filters
      ? FilterSchema.parse(normalizedOptions.filters)
      : undefined
    const reverse = normalizedOptions.reverse
    const workspacesPage = await this._listWorkspaces({
      filters: validatedFilter,
      page: normalizedOptions.page,
      size: normalizedOptions.size,
      reverse,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<WorkspaceResponse>> => {
      return this._listWorkspaces({
        filters: validatedFilter,
        page,
        size,
        reverse,
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
   * @param filters - Optional filters to scope the search. See [search filters documentation](https://honcho.dev/docs/v3/documentation/core-concepts/features/using-filters).
   * @param options.scope - Optional scope to restrict the search to that scope's member
   *                        sessions. Mutually exclusive with a `session_id` filter. A scope
   *                        with no member sessions matches nothing rather than everything.
   * @param limit - Number of results to return (1-100, default: 10).
   * @returns Promise resolving to an array of Message objects representing the search results.
   *          Returns an empty array if no messages are found.
   * @throws Error if the search query is empty or invalid
   */
  async search(
    query: string,
    options?: {
      filters?: Filters
      scope?: string | Scope
      limit?: number
    }
  ): Promise<Message[]> {
    await this._ensureWorkspace()
    const validatedQuery = SearchQuerySchema.parse(query)
    const validatedFilters = options?.filters
      ? FilterSchema.parse(options.filters)
      : undefined
    // Checked against undefined, not truthiness: `scope: ''` is invalid, and
    // dropping it silently would diverge from the Python SDK, which rejects it.
    const validatedScope =
      options?.scope !== undefined
        ? ScopeIdSchema.parse(resolveId(options.scope))
        : undefined
    const validatedLimit = options?.limit
      ? LimitSchema.parse(options.limit)
      : undefined
    const response = await this._searchWorkspace(this.workspaceId, {
      query: validatedQuery,
      filters: validatedFilters,
      scope: validatedScope,
      limit: validatedLimit,
    })
    return response.map(Message.fromApiResponse)
  }

  /**
   * Query the workspace's collective knowledge using natural language.
   *
   * Performs agentic search and reasoning across ALL peers and observations
   * in the workspace to synthesize a comprehensive answer. Useful for
   * cross-peer analysis, discovering common themes, and workspace-wide queries.
   *
   * @param query - The natural language question to ask
   * @param options.session - Optional session to scope message search to. Can be a session
   *                          ID string or a Session object.
   * @param options.reasoningLevel - Optional reasoning level for the query: "minimal", "low",
   *                                 "medium", "high", or "max". Defaults to "low" if not provided.
   * @param options.responseFormat - Optional JSON Schema (root type "object") the response
   *                                 must conform to. When provided, the response content is a
   *                                 JSON string matching this schema.
   * @param options.includeEvidence - When true, resolves to a ChatResponse carrying the
   *                                 answer alongside what the dialectic read to produce it.
   *                                 Evidence is collated from the agent's own reads rather
   *                                 than reported by the model, so it is broader than a
   *                                 citation list.
   * @returns Promise resolving to the response string, or null if no relevant information.
   *          With includeEvidence, a ChatResponse wrapping that content plus its evidence.
   *
   * @example
   * ```typescript
   * const response = await honcho.chat('What are common themes across all users?')
   *
   * const { content, evidence } = await honcho.chat('What are common themes?', {
   *   includeEvidence: true,
   * })
   * ```
   */
  async chat(
    query: string,
    options: {
      session?: string | Session
      reasoningLevel?: ReasoningLevel
      responseFormat?: Record<string, unknown>
      scope?: string | string[]
      includeEvidence: true
    }
  ): Promise<ChatResponse<string>>
  async chat(
    query: string,
    options?: {
      session?: string | Session
      reasoningLevel?: ReasoningLevel
      responseFormat?: Record<string, unknown>
      scope?: string | string[]
      includeEvidence?: false
    }
  ): Promise<string | null>
  async chat(
    query: string,
    options?: {
      session?: string | Session
      reasoningLevel?: ReasoningLevel
      responseFormat?: Record<string, unknown>
      scope?: string | string[]
      includeEvidence?: boolean
    }
  ): Promise<ChatResponse<string> | string | null> {
    const validatedQuery = SearchQuerySchema.parse(query)
    const resolvedSessionId = options?.session
      ? resolveId(options.session)
      : undefined

    const response = await this._workspaceChat(this.workspaceId, {
      query: validatedQuery,
      stream: false,
      session_id: resolvedSessionId,
      reasoning_level: options?.reasoningLevel,
      response_format: options?.responseFormat,
      scope: options?.scope,
      include_evidence: options?.includeEvidence ? true : undefined,
    })
    const content = response.content || null
    if (!options?.includeEvidence) {
      return content
    }
    return { content, evidence: response.evidence ?? null }
  }

  /**
   * Query the workspace's collective knowledge with streaming response.
   *
   * Performs agentic search and reasoning across ALL peers and observations
   * in the workspace to synthesize a comprehensive answer, streaming the
   * response as it is generated.
   *
   * @param query - The natural language question to ask
   * @param options.session - Optional session to scope message search to. Can be a session
   *                          ID string or a Session object.
   * @param options.reasoningLevel - Optional reasoning level for the query: "minimal", "low",
   *                                 "medium", "high", or "max". Defaults to "low" if not provided.
   * @param options.responseFormat - Optional JSON Schema (root type "object") the response
   *                                 must conform to. When provided, the response content is a
   *                                 JSON string matching this schema.
   * @param options.includeEvidence - When true, the returned stream's `evidence` is
   *                                 populated once it has been fully consumed. It cannot
   *                                 be known before then, so the server sends it on the
   *                                 terminal chunk.
   * @returns Promise resolving to a DialecticStreamResponse that can be iterated over
   *
   * @example
   * ```typescript
   * const stream = await honcho.chatStream('What do all peers have in common?')
   * for await (const chunk of stream) {
   *   process.stdout.write(chunk)
   * }
   * ```
   */
  async chatStream(
    query: string,
    options?: {
      session?: string | Session
      reasoningLevel?: ReasoningLevel
      responseFormat?: Record<string, unknown>
      scope?: string | string[]
      includeEvidence?: boolean
    }
  ): Promise<DialecticStreamResponse> {
    const validatedQuery = SearchQuerySchema.parse(query)
    const resolvedSessionId = options?.session
      ? resolveId(options.session)
      : undefined

    const response = await this._workspaceChatStream(this.workspaceId, {
      query: validatedQuery,
      session_id: resolvedSessionId,
      reasoning_level: options?.reasoningLevel,
      response_format: options?.responseFormat,
      scope: options?.scope,
      include_evidence: options?.includeEvidence ? true : undefined,
    })

    return createDialecticStream(response)
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
    const observerId = options?.observer
      ? resolveId(options.observer)
      : undefined
    const senderId = options?.sender ? resolveId(options.sender) : undefined
    const sessionId = options?.session ? resolveId(options.session) : undefined

    const queryParams: QueueStatusParams = {}
    if (observerId) queryParams.observer_id = observerId
    if (senderId) queryParams.sender_id = senderId
    if (sessionId) queryParams.session_id = sessionId

    const status = await this._getQueueStatus(this.workspaceId, queryParams)
    return transformQueueStatus(status)
  }

  /**
   * Ingest source records into this auto workspace.
   *
   * Honcho turns each record into a message, picking its session and peers
   * from the record's source context, and deduplicates on the record's source
   * identity, so sending the same records again is safe. Records are sent 100
   * per request; an async iterable is read one batch at a time.
   *
   * The workspace must be an auto workspace. A client made with `auto: true`
   * creates it on first use; otherwise it must already exist (`NotFoundError`)
   * and be auto (`ConflictError`).
   *
   * @param records - The records to ingest, and deletes (`op: 'delete'`)
   * @param options.derive - `'now'` (default) queues derivation as records
   *                         arrive, which ends the workspace's dry run;
   *                         `'later'` stores them without deriving until
   *                         `ingestDerive()`; `'auto'` is `'later'` during the
   *                         dry run and `'now'` after it
   * @returns One result per record, with `index` counted across all batches.
   *          If a request fails, the batches before it are already stored;
   *          sending all the records again is safe.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingest(
    records:
      | Iterable<IngestRecord | IngestRecordDelete>
      | AsyncIterable<IngestRecord | IngestRecordDelete>,
    options?: { derive?: 'now' | 'later' | 'auto' }
  ): Promise<IngestResult> {
    if (this._auto) {
      await this._ensureWorkspace()
    }
    const derive = options?.derive ?? 'now'
    const results: IngestRecordResult[] = []
    let batch: Record<string, unknown>[] = []
    let offset = 0
    const send = async () => {
      const response = await this._http.post<IngestResponse>(
        `/${API_VERSION}/workspaces/${this.workspaceId}/ingest`,
        { body: { derive, records: batch } }
      )
      for (const r of response.records) {
        results.push({
          index: r.index + offset,
          outcome: r.outcome,
          reason: r.reason ?? undefined,
          recordId: r.record_id ?? undefined,
          messageId: r.message_id ?? undefined,
          sessionId: r.session_id ?? undefined,
          peerId: r.peer_id ?? undefined,
          derivation: r.derivation ?? undefined,
        })
      }
      offset += batch.length
      batch = []
    }
    for await (const record of records) {
      batch.push(ingestRecordToApi(record))
      if (batch.length === INGEST_BATCH_SIZE) {
        await send()
      }
    }
    if (batch.length > 0) {
      await send()
    }
    return { records: results }
  }

  /**
   * Derive the records ingested with `derive: 'later'`.
   *
   * Releases deferred records in batches until none remain, or until a call
   * releases nothing (records still arriving with `derive: 'later'` are left
   * for the next call).
   *
   * @returns `pending`: records released by this call, in total;
   *          `remaining`: records still deferred afterwards
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestDerive(): Promise<IngestDeriveResult> {
    let released = 0
    while (true) {
      const step = await this._http.post<IngestDeriveResponse>(
        `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/derive`
      )
      released += step.pending
      if (step.remaining === 0 || step.pending === 0) {
        return { pending: released, remaining: step.remaining }
      }
    }
  }

  /**
   * Link imported people into one, from the next import on.
   *
   * Every account of `peers.slice(1)` maps to `peers[0]` from now on, so the
   * next import puts their records on one person. Nothing already imported
   * moves: the people being merged must have no imported messages yet, so
   * call `ingestReset()`, link, then import again (`ConflictError` otherwise).
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestLink(peers: Array<string | Peer>): Promise<IngestLinkResult> {
    const r = await this._http.post<IngestLinkResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/link`,
      { body: { peers: peers.map((p) => resolveId(p)) } }
    )
    return {
      peerId: r.peer_id,
      merged: r.merged,
      accounts: r.accounts.map(ingestAccountFromApi),
    }
  }

  /**
   * Take a linked account back out of `peer`, from the next import on.
   *
   * The account maps to the person it was linked from again. Its records must
   * not be imported yet: call `ingestReset()` first.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestUnlink(options: {
    peer: string | Peer
    account: { source: string; sourceInstance: string; id: string }
  }): Promise<IngestUnlinkResult> {
    const r = await this._http.post<IngestUnlinkResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/unlink`,
      {
        body: {
          peer: resolveId(options.peer),
          account: {
            source: options.account.source,
            source_instance: options.account.sourceInstance,
            id: options.account.id,
          },
        },
      }
    )
    return { peerId: r.peer_id, account: ingestAccountFromApi(r.account) }
  }

  /**
   * Delete every session holding imported records, to import again.
   *
   * The sessions are deleted in the background; `ingestStatus()` shows them
   * gone at once. People, their configuration and links stay.
   *
   * @returns The sessions whose deletion was started
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestReset(): Promise<{ sessions: string[] }> {
    return this._http.post<{ sessions: string[] }>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/reset`
    )
  }

  /**
   * What this workspace's imports produced, for review before deriving.
   *
   * Records by derivation state, the tokens still deferred, and how many
   * people and sessions the records map to, with how many of each are not
   * modeled.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestStatus(): Promise<IngestStatus> {
    const s = await this._http.get<IngestStatusResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/status`
    )
    return {
      records: s.records,
      deferredTokens: s.deferred_tokens,
      people: s.people,
      peopleNotModeled: s.people_not_modeled,
      sessions: s.sessions,
      sessionsNotModeled: s.sessions_not_modeled,
      sessionsBeingDeleted: s.sessions_being_deleted ?? 0,
      suggestions: s.suggestions ?? {
        rules: { high: 0, low: 0 },
        refreshing: false,
      },
      sources: (s.sources ?? []).map(ingestSyncSourceFromApi),
    }
  }

  /**
   * The sources connectors follow in this workspace, with each one's last sync.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestSyncStates(): Promise<IngestSyncSource[]> {
    const states = await this._http.get<IngestSyncSourceResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/sync`
    )
    return states.map(ingestSyncSourceFromApi)
  }

  /**
   * Where the connector following a source has got to; null to start afresh.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestSyncState(
    source: string,
    sourceInstance: string
  ): Promise<IngestSyncState | null> {
    try {
      return ingestSyncStateFromApi(
        await this._http.get<IngestSyncStateResponse>(
          `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/sync/${source}/${sourceInstance}`
        )
      )
    } catch (error) {
      if (error instanceof NotFoundError) {
        return null
      }
      throw error
    }
  }

  /**
   * Save where the connector following a source has got to.
   *
   * With `revision` (the one read), the save is refused with `ConflictError`
   * if another runner saved since or a reset forgot the state. Resolves to
   * the state with its new revision.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestSaveSyncState(
    source: string,
    sourceInstance: string,
    position: Record<string, unknown>,
    options?: {
      revision?: string
      lastRecordAt?: string | Date
      error?: string
    }
  ): Promise<IngestSyncState> {
    const lastRecordAt = options?.lastRecordAt
    return ingestSyncStateFromApi(
      await this._http.put<IngestSyncStateResponse>(
        `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/sync/${source}/${sourceInstance}`,
        {
          body: {
            position,
            error: options?.error ?? null,
            ...(options?.revision ? { revision: options.revision } : {}),
            ...(lastRecordAt
              ? {
                  last_record_at:
                    lastRecordAt instanceof Date
                      ? lastRecordAt.toISOString()
                      : lastRecordAt,
                }
              : {}),
          },
        }
      )
    )
  }

  /**
   * Forget a source's sync state, so its connector reads it from the start.
   * Resolves to whether there was one.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestForgetSyncState(
    source: string,
    sourceInstance: string
  ): Promise<boolean> {
    const r = await this._http.delete<{ forgotten: boolean }>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/sync/${source}/${sourceInstance}`
    )
    return r.forgotten
  }

  /**
   * Queue a refresh of the suggested links between imported accounts.
   *
   * The refresh runs in the background, reading the imports as they are when
   * it runs; `ingestStatus().suggestions.refreshing` is true until it's done.
   * Refresh after importing and before `ingestReset()`: the evidence comes from
   * the imported records. Resolves to whether a refresh was queued (one
   * already waiting covers this call).
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestRefreshSuggestions(): Promise<boolean> {
    const r = await this._http.post<{ queued: boolean }>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/suggestions/refresh`
    )
    return r.queued
  }

  /**
   * Accounts that look like one person, strongest first, with the evidence.
   *
   * Nothing is linked until a suggestion is accepted with
   * `ingestAcceptSuggestion`. `status: null` lists every state; `stage` and
   * `confidence` narrow it to one group, such as `{ stage: 'rules',
   * confidence: 'high' }`.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestSuggestions(options?: {
    status?: 'open' | 'accepted' | 'dismissed' | null
    confidence?: 'high' | 'low'
    stage?: 'rules'
    page?: number
    size?: number
  }): Promise<Page<IngestSuggestion, IngestSuggestionResponse>> {
    const status = options?.status === undefined ? 'open' : options.status
    const fetchPage = (
      page?: number,
      size?: number
    ): Promise<PageResponse<IngestSuggestionResponse>> =>
      this._http.get<PageResponse<IngestSuggestionResponse>>(
        `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/suggestions`,
        {
          query: {
            page,
            size,
            status: status ?? 'all',
            ...(options?.confidence ? { confidence: options.confidence } : {}),
            ...(options?.stage ? { stage: options.stage } : {}),
          },
        }
      )
    return new Page(
      await fetchPage(options?.page, options?.size),
      ingestSuggestionFromApi,
      fetchPage
    )
  }

  /**
   * Accept a suggestion: link its two accounts into one person.
   *
   * It is `ingestLink` for the suggestion's two people, keeping an explicit
   * peer, else the one with more records (or `keep`), and it is refused the
   * same way: call `ingestReset()` first, accept, then import again.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestAcceptSuggestion(
    suggestionId: string,
    options?: { keep?: string | Peer }
  ): Promise<IngestSuggestionAccepted> {
    const r = await this._http.post<{
      suggestion: IngestSuggestionResponse
      link: IngestLinkResponse | null
    }>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/suggestions/${suggestionId}/accept`,
      { body: options?.keep ? { keep: resolveId(options.keep) } : {} }
    )
    return {
      suggestion: ingestSuggestionFromApi(r.suggestion),
      link: r.link
        ? {
            peerId: r.link.peer_id,
            merged: r.link.merged,
            accounts: r.link.accounts.map(ingestAccountFromApi),
          }
        : undefined,
    }
  }

  /**
   * Dismiss a suggestion: the accounts stay apart, and a refresh won't bring
   * it back.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestDismissSuggestion(
    suggestionId: string
  ): Promise<IngestSuggestion> {
    return ingestSuggestionFromApi(
      await this._http.post<IngestSuggestionResponse>(
        `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/suggestions/${suggestionId}/dismiss`
      )
    )
  }

  /**
   * The peers created for imported accounts, most records first.
   *
   * Each lists its source accounts, how many records they wrote, and whether
   * the peer is modeled. Exclude one with
   * `peer.setConfiguration({ observeMe: false })` before `ingestDerive()`.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestPeople(options?: {
    page?: number
    size?: number
  }): Promise<Page<IngestPerson, IngestPersonResponse>> {
    const fetchPage = (
      page?: number,
      size?: number
    ): Promise<PageResponse<IngestPersonResponse>> =>
      this._http.get<PageResponse<IngestPersonResponse>>(
        `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/people`,
        { query: { page, size } }
      )
    return new Page(
      await fetchPage(options?.page, options?.size),
      (p) => ({
        peerId: p.peer_id,
        displayName: p.display_name ?? undefined,
        accounts: p.accounts.map(ingestAccountFromApi),
        messages: p.messages,
        modeled: p.modeled,
      }),
      fetchPage
    )
  }

  /**
   * The sessions created for imported containers, most records first.
   *
   * Each gives its container's display name and source, how many records it
   * holds, and whether reasoning is on for it.
   *
   * @experimental Ingest may change or stop working without notice.
   */
  async ingestSessions(options?: {
    page?: number
    size?: number
  }): Promise<Page<IngestSession, IngestSessionResponse>> {
    const fetchPage = (
      page?: number,
      size?: number
    ): Promise<PageResponse<IngestSessionResponse>> =>
      this._http.get<PageResponse<IngestSessionResponse>>(
        `/${API_VERSION}/workspaces/${this.workspaceId}/ingest/sessions`,
        { query: { page, size } }
      )
    return new Page(
      await fetchPage(options?.page, options?.size),
      (s) => ({
        sessionId: s.session_id,
        displayName: s.display_name ?? undefined,
        source: s.source,
        sourceInstance: s.source_instance,
        messages: s.messages,
        modeled: s.modeled,
      }),
      fetchPage
    )
  }

  /**
   * Schedule a dream task for memory consolidation.
   *
   * Dreams are background processes that consolidate observations into higher-level
   * insights and update peer cards. This method schedules a dream task for immediate
   * processing.
   *
   * @param options - Configuration options for the dream
   * @param options.observer - The observer peer (ID string or Peer object) whose perspective
   *                          to use for the dream
   * @param options.session - The session (ID string or Session object) to scope the dream to
   * @param options.observed - Optional observed peer (ID string or Peer object). If not provided,
   *                          defaults to the observer (self-reflection)
   * @returns Promise that resolves when the dream is scheduled
   */
  async scheduleDream(options: {
    observer: string | Peer
    session?: string | Session
    observed?: string | Peer
  }): Promise<void> {
    await this._ensureWorkspace()
    const observerId = resolveId(options.observer)
    const sessionId = options.session ? resolveId(options.session) : undefined
    const observedId = options.observed
      ? resolveId(options.observed)
      : observerId

    await this._http.post(
      `/${API_VERSION}/workspaces/${this.workspaceId}/schedule_dream`,
      {
        body: {
          observer: observerId,
          observed: observedId,
          session_id: sessionId,
          dream_type: 'omni',
        },
      }
    )
  }

  /**
   * List dream runs in the workspace, newest first (Representation History).
   *
   * Each run records what one dream changed in a representation. Runs are not
   * pruned unless the server sets a retention window.
   *
   * @param options.observed - Only runs about this peer (ID string or Peer object)
   * @param options.observer - Only runs from this observer's perspective. Combine
   *                           with `observed` for a single representation
   * @param options.filters - Filter object, e.g.
   *                          `{ started_at: { gte: '2026-01-01' } }` for the runs
   *                          that started in a date range. Filterable fields:
   *                          `started_at`, `finished_at`, `observer_id`,
   *                          `observed_id`. Requires a server that accepts
   *                          `filters` on the dream list
   * @param options.page - Page number (1-indexed)
   * @param options.size - Number of items per page
   * @returns Promise resolving to a Page of dream runs
   */
  async dreams(options?: {
    observed?: string | Peer
    observer?: string | Peer
    filters?: Filters
    page?: number
    size?: number
  }): Promise<Page<DreamRun, DreamRun>> {
    await this._ensureWorkspace()
    const body: Record<string, unknown> = {}
    if (options?.observed) body.observed_id = resolveId(options.observed)
    if (options?.observer) body.observer_id = resolveId(options.observer)
    if (options?.filters) body.filters = FilterSchema.parse(options.filters)

    const fetchPage = (
      page?: number,
      size?: number
    ): Promise<PageResponse<DreamRun>> =>
      this._http.post<PageResponse<DreamRun>>(
        `/${API_VERSION}/workspaces/${this.workspaceId}/dreams/list`,
        { body, query: { page, size } }
      )

    return new Page(
      await fetchPage(options?.page, options?.size),
      undefined,
      fetchPage
    )
  }

  /**
   * Get one dream run with every change it committed.
   *
   * The response lists the conclusions the dream created with the conclusions
   * each was derived from, the conclusions it removed as they read when
   * removed, and the peer card before and after.
   *
   * @param runId - The dream's run ID, from `dreams()`
   */
  async getDream(runId: string): Promise<DreamRunDetail> {
    await this._ensureWorkspace()
    return this._http.get<DreamRunDetail>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/dreams/${runId}`
    )
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
