import { API_VERSION } from './api-version'
import type { HonchoHTTPClient } from './http/client'
import { NotFoundError } from './http/errors'
import { Page } from './pagination'
import type { Session } from './session'
import type {
  ConclusionLevel,
  ConclusionResponse,
  PageResponse,
  RepresentationOptions,
  RepresentationResponse,
} from './types/api'
import { normalizeSearchQuery, RepresentationOptionsSchema } from './validation'

/**
 * Filter keys that define a conclusion scope (the observer/observed peer pair).
 * They are set from the scope itself, so a caller must not pass them in `filters`.
 */
const LIST_PAGE_CAP = 100

const SCOPE_RESERVED_KEYS = [
  'observer',
  'observed',
  'observer_id',
  'observed_id',
]

/**
 * Throw if `filters` contains keys managed by the conclusion scope.
 *
 * The observer/observed peer pair (and, on `list`, the session) is fixed by the
 * scope, so letting a user filter override it would silently return data from a
 * different scope than requested. Fail loud instead.
 */
function rejectReservedFilterKeys(
  filters: Record<string, unknown> | undefined,
  reserved: string[]
): void {
  if (!filters) return
  const clash = reserved.filter((k) => k in filters).sort()
  if (clash.length > 0) {
    let guidance =
      'Choose the peer pair via peer.conclusions / peer.conclusionsOf(target)'
    if (reserved.includes('session') || reserved.includes('session_id')) {
      guidance += '; use the session option to filter by session'
    }
    throw new Error(
      `Filter key(s) ${clash.join(', ')} are managed by this conclusion scope ` +
        `and cannot be passed in filters. ${guidance}.`
    )
  }
}

/**
 * Parameters for creating a conclusion.
 */
export interface ConclusionCreateParams {
  /** The conclusion content/text */
  content: string
  /** The session this conclusion relates to (ID string or Session object) */
  sessionId?: string | Session
}

/**
 * A conclusion from Honcho's reasoning system.
 *
 * Conclusions are facts derived from messages that help build a representation
 * of a peer.
 */
export class Conclusion {
  readonly id: string
  readonly content: string
  readonly observerId: string
  readonly observedId: string
  readonly sessionId: string | null
  /**
   * Reasoning level: 'explicit' conclusions are extracted directly from
   * messages; 'deductive'/'inductive'/'contradiction' are derived during
   * dreaming.
   */
  readonly level: ConclusionLevel
  /**
   * IDs of the conclusions this one was derived from (premises for
   * 'deductive', supporting sources for 'inductive', conflicting conclusions
   * for 'contradiction'). Null for 'explicit' conclusions.
   */
  readonly sourceIds: string[] | null
  /** Number of times this conclusion has been independently derived. */
  readonly timesDerived: number
  readonly createdAt: string

  constructor(
    id: string,
    content: string,
    observerId: string,
    observedId: string,
    sessionId: string | null,
    createdAt: string,
    level: ConclusionLevel = 'explicit',
    sourceIds: string[] | null = null,
    timesDerived: number = 1
  ) {
    this.id = id
    this.content = content
    this.observerId = observerId
    this.observedId = observedId
    this.sessionId = sessionId
    this.level = level
    this.sourceIds = sourceIds
    this.timesDerived = timesDerived
    this.createdAt = createdAt
  }

  static fromApiResponse(data: ConclusionResponse): Conclusion {
    return new Conclusion(
      data.id,
      data.content,
      data.observer_id,
      data.observed_id,
      data.session_id,
      data.created_at,
      data.level,
      data.source_ids ?? null,
      data.times_derived ?? 1
    )
  }

  toString(): string {
    const truncatedContent =
      this.content.length > 50
        ? `${this.content.slice(0, 50)}...`
        : this.content
    return `Conclusion(id='${this.id}', content='${truncatedContent}')`
  }
}

/**
 * Scoped access to conclusions for a specific observer/observed relationship.
 */
export class ConclusionScope {
  private _http: HonchoHTTPClient
  private _ensureWorkspace: () => Promise<void>
  readonly workspaceId: string
  readonly observer: string
  readonly observed: string

  constructor(
    http: HonchoHTTPClient,
    workspaceId: string,
    observer: string,
    observed: string,
    ensureWorkspace: () => Promise<void> = async () => undefined
  ) {
    this._http = http
    this.workspaceId = workspaceId
    this.observer = observer
    this.observed = observed
    this._ensureWorkspace = ensureWorkspace
  }

  // ===========================================================================
  // Private API Methods
  // ===========================================================================

  private async _list(params: {
    filters?: Record<string, unknown>
    page?: number
    size?: number
    reverse?: boolean
  }): Promise<PageResponse<ConclusionResponse>> {
    await this._ensureWorkspace()
    return this._http.post<PageResponse<ConclusionResponse>>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/list`,
      {
        body: { filters: params.filters },
        query: {
          page: params.page,
          size: params.size,
          reverse: params.reverse ? 'true' : undefined,
        },
      }
    )
  }

  private async _query(params: {
    query: string
    top_k?: number
    distance?: number
    filters?: Record<string, unknown>
  }): Promise<ConclusionResponse[]> {
    await this._ensureWorkspace()
    return this._http.post<ConclusionResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/query`,
      { body: params }
    )
  }

  private async _create(params: {
    conclusions: Array<{
      content: string
      session_id: string | null
      observer_id: string
      observed_id: string
    }>
  }): Promise<ConclusionResponse[]> {
    await this._ensureWorkspace()
    return this._http.post<ConclusionResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions`,
      { body: params }
    )
  }

  private async _get(conclusionId: string): Promise<ConclusionResponse> {
    await this._ensureWorkspace()
    const item = await this._http.get<ConclusionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/${conclusionId}`
    )
    if (
      item.observer_id !== this.observer ||
      item.observed_id !== this.observed
    ) {
      throw new NotFoundError('Conclusion not found')
    }
    return item
  }

  private async _derived(
    conclusionId: string,
    params: {
      page?: number
      size?: number
      reverse?: boolean
    }
  ): Promise<PageResponse<ConclusionResponse>> {
    // Equivalent to list with { source_ids: { contains: id } }, restricted
    // to this pair.
    return this._list({
      filters: {
        source_ids: { contains: conclusionId },
        observer_id: this.observer,
        observed_id: this.observed,
      },
      page: params.page,
      size: params.size,
      reverse: params.reverse,
    })
  }

  private async _delete(conclusionId: string): Promise<void> {
    await this._ensureWorkspace()
    await this._http.delete(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/${conclusionId}`
    )
  }

  private async _getRepresentation(
    peerId: string,
    params: {
      target?: string
      search_query?: string
      search_top_k?: number
      search_max_distance?: number
      include_most_frequent?: boolean
      max_conclusions?: number
    }
  ): Promise<RepresentationResponse> {
    await this._ensureWorkspace()
    return this._http.post<RepresentationResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${peerId}/representation`,
      { body: params }
    )
  }

  // ===========================================================================
  // Public Methods
  // ===========================================================================

  /**
   * List conclusions in this scope.
   *
   * @param options - Optional configuration for the list request
   * @param options.page - Page number (1-indexed, default: 1)
   * @param options.size - Number of items per page (default: 50)
   * @param options.session - Optional session (ID string or Session object) to filter by
   * @param options.filters - Optional additional filter criteria, merged with
   *   this scope's observer/observed (and session, if given). Supports the same
   *   operators as other list endpoints — e.g. `{ level: 'explicit' }` to get
   *   only conclusions extracted directly from messages (i.e. not derived during
   *   dreaming), or `{ source_ids: { contains: '<id>' } }` to get conclusions
   *   derived from a given conclusion (see also `derived()`). See
   *   https://honcho.dev/docs/v3/documentation/features/advanced/using-filters
   * @returns Promise resolving to a Page of Conclusion objects
   */
  async list(options?: {
    page?: number
    size?: number
    session?: string | Session
    filters?: Record<string, unknown>
    reverse?: boolean
  }): Promise<Page<Conclusion, ConclusionResponse>> {
    rejectReservedFilterKeys(options?.filters, [
      ...SCOPE_RESERVED_KEYS,
      'session',
      'session_id',
    ])
    const resolvedSessionId = options?.session
      ? typeof options.session === 'string'
        ? options.session
        : options.session.id
      : undefined
    const filters: Record<string, unknown> = {
      observer_id: this.observer,
      observed_id: this.observed,
      ...(resolvedSessionId ? { session_id: resolvedSessionId } : {}),
      ...options?.filters,
    }
    const reverse = options?.reverse

    const response = await this._list({
      filters,
      page: options?.page ?? 1,
      size: options?.size ?? 50,
      reverse,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<ConclusionResponse>> => {
      return this._list({ filters, page, size, reverse })
    }

    return new Page(
      response,
      (item) => Conclusion.fromApiResponse(item),
      fetchNextPage
    )
  }

  /**
   * Semantic search for conclusions in this scope.
   *
   * @param query - The search query string
   * @param topK - Maximum number of results to return (default: 10)
   * @param distance - Maximum cosine distance threshold (0.0-1.0)
   * @param filters - Optional additional filter criteria, merged with this
   *   scope's observer/observed. Supports the same operators as the list
   *   endpoint — e.g. `{ level: 'deductive' }` to search only conclusions
   *   derived during dreaming. See
   *   https://honcho.dev/docs/v3/documentation/features/advanced/using-filters
   */
  async query(
    query: string,
    topK: number = 10,
    distance?: number,
    filters?: Record<string, unknown>
  ): Promise<Conclusion[]> {
    rejectReservedFilterKeys(filters, SCOPE_RESERVED_KEYS)
    const response = await this._query({
      query,
      top_k: topK,
      distance,
      filters: {
        observer_id: this.observer,
        observed_id: this.observed,
        ...filters,
      },
    })

    return (response ?? []).map((item) => Conclusion.fromApiResponse(item))
  }

  /**
   * Get a single conclusion by ID.
   *
   * @param conclusionId - The ID of the conclusion to retrieve
   * @returns Promise resolving to the Conclusion object, including its
   *   attribution fields (`sourceIds`, `timesDerived`)
   */
  async get(conclusionId: string): Promise<Conclusion> {
    const response = await this._get(conclusionId)
    return Conclusion.fromApiResponse(response)
  }

  /**
   * Get multiple conclusions by ID in a single call.
   *
   * Useful for resolving a derived conclusion's premises: pass its
   * `sourceIds` to fetch all of them at once instead of one `get()` per ID.
   *
   * @param conclusionIds - The IDs of the conclusions to retrieve
   * @returns Promise resolving to the matching Conclusion objects. IDs that
   *   don't exist are omitted, so the result may be shorter than the input
   *   (order is not guaranteed to match the input either).
   */
  async getMany(conclusionIds: string[]): Promise<Conclusion[]> {
    if (conclusionIds.length === 0) return []
    const conclusions: Conclusion[] = []
    // The list endpoint caps page size at 100
    for (let start = 0; start < conclusionIds.length; start += LIST_PAGE_CAP) {
      const chunk = conclusionIds.slice(start, start + LIST_PAGE_CAP)
      const response = await this._list({
        filters: {
          id: { in: chunk },
          observer_id: this.observer,
          observed_id: this.observed,
        },
        page: 1,
        size: chunk.length,
      })
      conclusions.push(
        ...(response.items ?? []).map((item) =>
          Conclusion.fromApiResponse(item)
        )
      )
    }
    return conclusions
  }

  /**
   * Get the conclusions derived from the given conclusion — i.e. those that
   * list it in their `sourceIds`. Traverses the reasoning tree upward
   * (source -> derived).
   *
   * @param conclusionId - The ID of the source conclusion
   * @param options - Optional configuration for the request
   * @param options.page - Page number (1-indexed, default: 1)
   * @param options.size - Number of items per page (default: 50)
   * @param options.reverse - If true, reverses the default newest-first ordering
   * @returns Promise resolving to a Page of Conclusion objects
   */
  async derived(
    conclusionId: string,
    options?: {
      page?: number
      size?: number
      reverse?: boolean
    }
  ): Promise<Page<Conclusion, ConclusionResponse>> {
    const reverse = options?.reverse
    const response = await this._derived(conclusionId, {
      page: options?.page ?? 1,
      size: options?.size ?? 50,
      reverse,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<ConclusionResponse>> => {
      return this._derived(conclusionId, { page, size, reverse })
    }

    return new Page(
      response,
      (item) => Conclusion.fromApiResponse(item),
      fetchNextPage
    )
  }

  /**
   * Delete a conclusion by ID.
   */
  async delete(conclusionId: string): Promise<void> {
    await this._delete(conclusionId)
  }

  /**
   * Create conclusions in this scope.
   */
  async create(
    conclusions: ConclusionCreateParams | ConclusionCreateParams[]
  ): Promise<Conclusion[]> {
    const conclusionArray = Array.isArray(conclusions)
      ? conclusions
      : [conclusions]

    const requestConclusions = conclusionArray.map((obs) => ({
      content: obs.content,
      session_id:
        obs.sessionId === undefined
          ? null
          : typeof obs.sessionId === 'string'
            ? obs.sessionId
            : obs.sessionId.id,
      observer_id: this.observer,
      observed_id: this.observed,
    }))

    const response = await this._create({ conclusions: requestConclusions })

    return (response ?? []).map((item) => Conclusion.fromApiResponse(item))
  }

  /**
   * Get the computed representation for this scope.
   */
  async representation(options?: RepresentationOptions): Promise<string> {
    const searchQuery = normalizeSearchQuery(options?.searchQuery)
    const validatedOptions = RepresentationOptionsSchema.parse({
      searchQuery,
      searchTopK: options?.searchTopK,
      searchMaxDistance: options?.searchMaxDistance,
      includeMostFrequent: options?.includeMostFrequent,
      maxConclusions: options?.maxConclusions,
    })

    const response = await this._getRepresentation(this.observer, {
      target: this.observed,
      search_query: searchQuery,
      search_top_k: validatedOptions.searchTopK,
      search_max_distance: validatedOptions.searchMaxDistance,
      include_most_frequent: validatedOptions.includeMostFrequent,
      max_conclusions: validatedOptions.maxConclusions,
    })
    return response.representation
  }

  toString(): string {
    return `ConclusionScope(workspaceId='${this.workspaceId}', observer='${this.observer}', observed='${this.observed}')`
  }
}

/**
 * Workspace-wide conclusion access. No observer/observed pair is implied.
 *
 * Use this to list or look up conclusions across the workspace, then filter
 * down to a peer or session. Pair-scoped create/query/delete stay on
 * `peer.conclusions` / `peer.conclusionsOf(target)`.
 */
export class WorkspaceConclusions {
  private _http: HonchoHTTPClient
  private _ensureWorkspace: () => Promise<void>
  readonly workspaceId: string

  constructor(
    http: HonchoHTTPClient,
    workspaceId: string,
    ensureWorkspace: () => Promise<void> = async () => undefined
  ) {
    this._http = http
    this.workspaceId = workspaceId
    this._ensureWorkspace = ensureWorkspace
  }

  private async _list(params: {
    filters?: Record<string, unknown>
    page?: number
    size?: number
    reverse?: boolean
  }): Promise<PageResponse<ConclusionResponse>> {
    await this._ensureWorkspace()
    return this._http.post<PageResponse<ConclusionResponse>>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/list`,
      {
        body: params.filters ? { filters: params.filters } : undefined,
        query: {
          page: params.page,
          size: params.size,
          reverse: params.reverse ? 'true' : undefined,
        },
      }
    )
  }

  /**
   * List conclusions in this workspace.
   *
   * Unlike `peer.conclusions.list`, no observer/observed pair is injected.
   * Pass `filters` to narrow the view — e.g. `{ observed_id: 'alice' }` or
   * `{ session_id: '...' }`.
   */
  async list(options?: {
    page?: number
    size?: number
    filters?: Record<string, unknown>
    reverse?: boolean
  }): Promise<Page<Conclusion, ConclusionResponse>> {
    const filters = options?.filters
    const reverse = options?.reverse
    const response = await this._list({
      filters,
      page: options?.page ?? 1,
      size: options?.size ?? 50,
      reverse,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<ConclusionResponse>> => {
      return this._list({ filters, page, size, reverse })
    }

    return new Page(
      response,
      (item) => Conclusion.fromApiResponse(item),
      fetchNextPage
    )
  }

  /**
   * Get a single conclusion by ID, anywhere in the workspace.
   */
  async get(conclusionId: string): Promise<Conclusion> {
    await this._ensureWorkspace()
    const item = await this._http.get<ConclusionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/${conclusionId}`
    )
    return Conclusion.fromApiResponse(item)
  }

  /**
   * Get multiple conclusions by ID. Missing IDs are omitted.
   */
  async getMany(conclusionIds: string[]): Promise<Conclusion[]> {
    if (conclusionIds.length === 0) return []
    const conclusions: Conclusion[] = []
    for (let start = 0; start < conclusionIds.length; start += LIST_PAGE_CAP) {
      const chunk = conclusionIds.slice(start, start + LIST_PAGE_CAP)
      const response = await this._list({
        filters: { id: { in: chunk } },
        page: 1,
        size: chunk.length,
      })
      conclusions.push(
        ...(response.items ?? []).map((item) =>
          Conclusion.fromApiResponse(item)
        )
      )
    }
    return conclusions
  }

  toString(): string {
    return `WorkspaceConclusions(workspaceId='${this.workspaceId}')`
  }
}
