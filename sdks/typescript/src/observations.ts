import type HonchoCore from '@honcho-ai/core'
import { Representation, type RepresentationData } from './representation'

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
  static fromApiResponse(data: Record<string, any>): Observation {
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
 * Options for representation retrieval.
 */
export interface RepresentationOptions {
  /**
   * Semantic search query to filter relevant observations.
   */
  searchQuery?: string

  /**
   * Number of semantically relevant facts to return.
   */
  searchTopK?: number

  /**
   * Maximum semantic distance for search results (0.0-1.0).
   */
  searchMaxDistance?: number

  /**
   * Whether to include the most derived observations.
   */
  includeMostDerived?: boolean

  /**
   * Maximum number of observations to include.
   */
  maxObservations?: number
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
  private _client: HonchoCore

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
   * @param client - The Honcho client instance
   * @param workspaceId - The workspace ID
   * @param observer - The observer peer ID
   * @param observed - The observed peer ID
   */
  constructor(
    client: HonchoCore,
    workspaceId: string,
    observer: string,
    observed: string
  ) {
    this._client = client
    this.workspaceId = workspaceId
    this.observer = observer
    this.observed = observed
  }

  /**
   * List observations in this scope.
   *
   * @param page - Page number (1-indexed)
   * @param size - Number of results per page
   * @param sessionId - Optional session ID to filter by
   * @returns Promise resolving to list of Observation objects
   */
  async list(
    page: number = 1,
    size: number = 50,
    sessionId?: string
  ): Promise<Observation[]> {
    const filters: Record<string, any> = {
      observer: this.observer,
      observed: this.observed,
    }
    if (sessionId) {
      filters.session_id = sessionId
    }

    // Note: This requires the core SDK to support observations.list()
    const response = await (this._client.workspaces as any).observations.list(
      this.workspaceId,
      {
        filters,
        page,
        size,
      }
    )

    return (response.items ?? []).map((item: any) =>
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
    const filters: Record<string, any> = {
      observer: this.observer,
      observed: this.observed,
    }

    // Note: This requires the core SDK to support observations.query()
    const response = await (this._client.workspaces as any).observations.query(
      this.workspaceId,
      {
        query,
        top_k: topK,
        distance,
        filters,
      }
    )

    return (response ?? []).map((item: any) =>
      Observation.fromApiResponse(item)
    )
  }

  /**
   * Delete an observation by ID.
   *
   * @param observationId - The ID of the observation to delete
   */
  async delete(observationId: string): Promise<void> {
    // Note: This requires the core SDK to support observations.delete()
    await (this._client.workspaces as any).observations.delete(
      this.workspaceId,
      observationId
    )
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
    const response = await this._client.workspaces.peers.workingRepresentation(
      this.workspaceId,
      this.observer,
      {
        target: this.observed,
        search_query: options?.searchQuery,
        search_top_k: options?.searchTopK,
        search_max_distance: options?.searchMaxDistance,
        include_most_derived: options?.includeMostDerived,
        max_observations: options?.maxObservations,
      }
    )

    const maybe = response as
      | RepresentationData
      | { representation?: RepresentationData | null }
      | null
    const rep = (maybe && 'representation' in (maybe as any)
      ? (maybe as any).representation
      : (maybe as any)) ?? { explicit: [], deductive: [] }
    return Representation.fromData(rep as RepresentationData)
  }

  /**
   * Return a string representation of the ObservationScope.
   */
  toString(): string {
    return `ObservationScope(workspaceId='${this.workspaceId}', observer='${this.observer}', observed='${this.observed}')`
  }
}
