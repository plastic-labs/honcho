import type { HttpClient } from './http'
import {
  Representation,
  type RepresentationData,
  type RepresentationOptions,
} from './representation'
import type { Session } from './session'
import type { ObservationCreateParam } from './types'

// Re-export for consumers who import from this module
export type { RepresentationOptions, ObservationCreateParam }

/**
 * API response shape for an observation.
 */
interface ObservationResponse {
  id: string
  content: string
  observer_id: string
  observed_id: string
  session_id: string
  created_at: string
}

/**
 * An observation from the theory-of-mind system.
 *
 * Observations are facts derived from messages that help build a representation
 * of a peer.
 */
export class Observation {
  /**
   * Unique identifier for this observation.
   */
  readonly id: string

  /**
   * The observation content/text.
   */
  readonly content: string

  /**
   * The peer who made the observation.
   */
  readonly observerId: string

  /**
   * The peer being observed.
   */
  readonly observedId: string

  /**
   * The session where this observation was made.
   */
  readonly sessionId: string

  /**
   * When the observation was created.
   */
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

  /**
   * Create an Observation from an API response object.
   *
   * @param data - API response data
   * @returns A new Observation instance
   */
  static fromApiResponse(data: ObservationResponse): Observation {
    return new Observation(
      data.id ?? '',
      data.content ?? '',
      data.observer_id ?? '',
      data.observed_id ?? '',
      data.session_id ?? '',
      data.created_at ?? ''
    )
  }

  /**
   * Return a string representation of the Observation.
   */
  toString(): string {
    const truncatedContent =
      this.content.length > 50
        ? `${this.content.slice(0, 50)}...`
        : this.content
    return `Observation(id='${this.id}', content='${truncatedContent}')`
  }
}

/**
 * Scoped access to observations for a specific observer/observed relationship.
 *
 * This class provides convenient methods to list, query, and delete observations
 * that are automatically scoped to a specific observer/observed pair.
 *
 * Typically accessed via `peer.observations` (for self-observations) or
 * `peer.observationsOf(target)` (for observations about another peer).
 *
 * @example
 * ```typescript
 * // Get self-observations
 * const observations = peer.observations
 * const obsList = await observations.list()
 * const searchResults = await observations.query('preferences')
 *
 * // Get observations about another peer
 * const bobObservations = peer.observationsOf('bob')
 * const bobList = await bobObservations.list()
 * ```
 *
 * @note
 * This class requires the core Honcho SDK to support observation endpoints.
 * The observation endpoints are:
 * - POST /workspaces/{workspace_id}/observations/list
 * - POST /workspaces/{workspace_id}/observations/query
 * - DELETE /workspaces/{workspace_id}/observations/{observation_id}
 */
export class ObservationScope {
  private _http: HttpClient

  /**
   * The workspace ID.
   */
  readonly workspaceId: string

  /**
   * The observer peer ID.
   */
  readonly observer: string

  /**
   * The observed peer ID.
   */
  readonly observed: string

  /**
   * Initialize an ObservationScope.
   *
   * @param http - The HTTP client instance
   * @param workspaceId - The workspace ID
   * @param observer - The observer peer ID
   * @param observed - The observed peer ID
   */
  constructor(
    http: HttpClient,
    workspaceId: string,
    observer: string,
    observed: string
  ) {
    this._http = http
    this.workspaceId = workspaceId
    this.observer = observer
    this.observed = observed
  }

  /**
   * List observations in this scope.
   *
   * @param page - Page number (1-indexed)
   * @param size - Number of results per page
   * @param session - Optional session (ID string or Session object) to filter by
   * @returns Promise resolving to list of Observation objects
   */
  async list(
    page: number = 1,
    size: number = 50,
    session?: string | Session
  ): Promise<Observation[]> {
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

    const response = await this._http.request<{
      items: ObservationResponse[]
    }>('POST', `/v2/workspaces/${this.workspaceId}/observations/list`, {
      json: {
        filters,
        page,
        size,
      },
    })

    return (response.items ?? []).map((item) =>
      Observation.fromApiResponse(item)
    )
  }

  /**
   * Semantic search for observations in this scope.
   *
   * @param query - The search query string
   * @param topK - Maximum number of results to return
   * @param distance - Maximum cosine distance threshold (0.0-1.0)
   * @returns Promise resolving to list of matching Observation objects
   */
  async query(
    query: string,
    topK: number = 10,
    distance?: number
  ): Promise<Observation[]> {
    const filters: Record<string, unknown> = {
      observer: this.observer,
      observed: this.observed,
    }

    const response = await this._http.request<ObservationResponse[]>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/observations/query`,
      {
        json: {
          query,
          top_k: topK,
          distance,
          filters,
        },
      }
    )

    return (response ?? []).map((item) => Observation.fromApiResponse(item))
  }

  /**
   * Delete an observation by ID.
   *
   * @param observationId - The ID of the observation to delete
   */
  async delete(observationId: string): Promise<void> {
    await this._http.request<void>(
      'DELETE',
      `/v2/workspaces/${this.workspaceId}/observations/${observationId}`
    )
  }

  /**
   * Create observations in this scope.
   *
   * @param observations - Single observation or array of observations with content and sessionId
   * @returns Promise resolving to list of created Observation objects
   *
   * @example
   * ```typescript
   * // Create a single observation
   * const observations = await peer.observations.create(
   *   { content: 'User prefers dark mode', sessionId: 'session1' }
   * )
   *
   * // Create multiple observations
   * const observations = await peer.observations.create([
   *   { content: 'User prefers dark mode', sessionId: 'session1' },
   *   { content: 'User is interested in AI', sessionId: 'session1' },
   * ])
   * ```
   */
  async create(
    observations: ObservationCreateParam | ObservationCreateParam[]
  ): Promise<Observation[]> {
    // Normalize to array
    const observationArray = Array.isArray(observations)
      ? observations
      : [observations]

    // Build the request body with observer/observed from scope
    const requestObservations = observationArray.map((obs) => ({
      content: obs.content,
      session_id:
        typeof obs.sessionId === 'string' ? obs.sessionId : obs.sessionId.id,
      observer_id: this.observer,
      observed_id: this.observed,
    }))

    const response = await this._http.request<ObservationResponse[]>(
      'POST',
      `/v2/workspaces/${this.workspaceId}/observations`,
      {
        json: { observations: requestObservations },
      }
    )

    return (response ?? []).map((item) => Observation.fromApiResponse(item))
  }

  /**
   * Get the computed representation for this scope.
   *
   * This returns the working representation (narrative) built from the
   * observations in this scope.
   *
   * @param options - Optional options to configure the representation
   * @returns Promise resolving to a Representation object
   */
  async getRepresentation(
    options?: RepresentationOptions
  ): Promise<Representation> {
    const response = await this._http.request<
      RepresentationData | { representation?: RepresentationData | null }
    >(
      'POST',
      `/v2/workspaces/${this.workspaceId}/peers/${this.observer}/representation`,
      {
        json: {
          target: this.observed,
          search_query: options?.searchQuery,
          search_top_k: options?.searchTopK,
          search_max_distance: options?.searchMaxDistance,
          include_most_derived: options?.includeMostDerived,
          max_observations: options?.maxObservations,
        },
      }
    )

    const maybe = response as
      | RepresentationData
      | { representation?: RepresentationData | null }
      | null
    const rep = (maybe && typeof maybe === 'object' && 'representation' in maybe
      ? (maybe as { representation?: RepresentationData | null }).representation
      : maybe) ?? { explicit: [], deductive: [] }
    return Representation.fromData(rep as RepresentationData)
  }

  /**
   * Return a string representation of the ObservationScope.
   */
  toString(): string {
    return `ObservationScope(workspaceId='${this.workspaceId}', observer='${this.observer}', observed='${this.observed}')`
  }
}
