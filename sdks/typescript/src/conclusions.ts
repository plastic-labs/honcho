import type { HonchoHTTPClient } from './http/client'
import type { RepresentationOptions } from './representation'
import type { Session } from './session'
import type { ConclusionCreateParam } from './types'
import {
  API_VERSION,
  type ConclusionResponse,
  type PageResponse,
  type RepresentationResponse,
} from './types'

// Re-export for consumers who import from this module
export type { RepresentationOptions, ConclusionCreateParam }

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
  readonly sessionId: string
  readonly createdAt: string

  constructor(
    id: string,
    content: string,
    observerId: string,
    observedId: string,
    sessionId: string,
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
      data.id ?? '',
      data.content ?? '',
      data.observer_id ?? '',
      data.observed_id ?? '',
      data.session_id ?? '',
      data.created_at ?? ''
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
  readonly workspaceId: string
  readonly observer: string
  readonly observed: string

  constructor(
    http: HonchoHTTPClient,
    workspaceId: string,
    observer: string,
    observed: string
  ) {
    this._http = http
    this.workspaceId = workspaceId
    this.observer = observer
    this.observed = observed
  }

  // ===========================================================================
  // Private API Methods
  // ===========================================================================

  private async _list(params: {
    filters?: Record<string, unknown>
    page?: number
    size?: number
  }): Promise<PageResponse<ConclusionResponse>> {
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
    return this._http.post<ConclusionResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions/query`,
      { body: params }
    )
  }

  private async _create(params: {
    conclusions: Array<{
      content: string
      session_id: string
      observer_id: string
      observed_id: string
    }>
  }): Promise<ConclusionResponse[]> {
    return this._http.post<ConclusionResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/conclusions`,
      { body: params }
    )
  }

  private async _delete(conclusionId: string): Promise<void> {
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
   */
  async list(
    page: number = 1,
    size: number = 50,
    session?: string | Session
  ): Promise<Conclusion[]> {
    const resolvedSessionId = session
      ? typeof session === 'string'
        ? session
        : session.id
      : undefined
    const filters: Record<string, unknown> = {
      observer: this.observer,
      observed: this.observed,
    }
    if (resolvedSessionId) {
      filters.session_id = resolvedSessionId
    }

    const response = await this._list({ filters, page, size })

    return (response.items ?? []).map((item) =>
      Conclusion.fromApiResponse(item)
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
      observer: this.observer,
      observed: this.observed,
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
    conclusions: ConclusionCreateParam | ConclusionCreateParam[]
  ): Promise<Conclusion[]> {
    const conclusionArray = Array.isArray(conclusions)
      ? conclusions
      : [conclusions]

    const requestConclusions = conclusionArray.map((obs) => ({
      content: obs.content,
      session_id:
        typeof obs.sessionId === 'string' ? obs.sessionId : obs.sessionId.id,
      observer_id: this.observer,
      observed_id: this.observed,
    }))

    const response = await this._create({ conclusions: requestConclusions })

    return (response ?? []).map((item) => Conclusion.fromApiResponse(item))
  }

  /**
   * Get the computed representation for this scope.
   */
  async getRepresentation(options?: RepresentationOptions): Promise<string> {
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
