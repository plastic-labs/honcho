import { API_VERSION } from './api-version'
import type { HonchoHTTPClient } from './http/client'
import { Page } from './pagination'
import type { Session } from './session'
import type {
  ConclusionResponse,
  PageResponse,
  RepresentationOptions,
  RepresentationResponse,
} from './types/api'

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
  readonly createdAt: string

  constructor(
    id: string,
    content: string,
    observerId: string,
    observedId: string,
    sessionId: string | null,
    createdAt: string
  ) {
    this.id = id
    this.content = content
    this.observerId = observerId
    this.observedId = observedId
    this.sessionId = sessionId
    this.createdAt = createdAt
  }

  static fromApiResponse(data: ConclusionResponse): Conclusion {
    return new Conclusion(
      data.id,
      data.content,
      data.observer_id,
      data.observed_id,
      data.session_id,
      data.created_at
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
  }): Promise<PageResponse<ConclusionResponse>> {
    await this._ensureWorkspace()
    return this._http.post<PageResponse<ConclusionResponse>>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/list`,
      {
        body: { filters: params.filters },
        query: { page: params.page, size: params.size },
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
   * @returns Promise resolving to a Page of Conclusion objects
   */
  async list(options?: {
    page?: number
    size?: number
    session?: string | Session
  }): Promise<Page<Conclusion, ConclusionResponse>> {
    const resolvedSessionId = options?.session
      ? typeof options.session === 'string'
        ? options.session
        : options.session.id
      : undefined
    const filters: Record<string, unknown> = {
      observer_id: this.observer,
      observed_id: this.observed,
    }
    if (resolvedSessionId) {
      filters.session_id = resolvedSessionId
    }

    const response = await this._list({
      filters,
      page: options?.page ?? 1,
      size: options?.size ?? 50,
    })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<ConclusionResponse>> => {
      return this._list({ filters, page, size })
    }

    return new Page(
      response,
      (item) => Conclusion.fromApiResponse(item),
      fetchNextPage
    )
  }

  /**
   * Semantic search for conclusions in this scope.
   */
  async query(
    query: string,
    topK: number = 10,
    distance?: number
  ): Promise<Conclusion[]> {
    const filters: Record<string, unknown> = {
      observer_id: this.observer,
      observed_id: this.observed,
    }

    const response = await this._query({
      query,
      top_k: topK,
      distance,
      filters,
    })

    return (response ?? []).map((item) => Conclusion.fromApiResponse(item))
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
    const response = await this._getRepresentation(this.observer, {
      target: this.observed,
      search_query: options?.searchQuery,
      search_top_k: options?.searchTopK,
      search_max_distance: options?.searchMaxDistance,
      include_most_frequent: options?.includeMostFrequent,
      max_conclusions: options?.maxConclusions,
    })
    return response.representation
  }

  toString(): string {
    return `ConclusionScope(workspaceId='${this.workspaceId}', observer='${this.observer}', observed='${this.observed}')`
  }
}
