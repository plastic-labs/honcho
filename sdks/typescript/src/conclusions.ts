import type HonchoCore from '@honcho-ai/core'
import type { RepresentationOptions } from './representation'
import type { Session } from './session'
import type { ConclusionCreateParam } from './types'

// Re-export for consumers who import from this module
export type { RepresentationOptions, ConclusionCreateParam }

/**
 * An conclusion from the theory-of-mind system.
 *
 * Conclusions are facts derived from messages that help build a representation
 * of a peer.
 */
export class Conclusion {
  /**
   * Unique identifier for this conclusion.
   */
  readonly id: string

  /**
   * The conclusion content/text.
   */
  readonly content: string

  /**
   * The peer who made the conclusion.
   */
  readonly observerId: string

  /**
   * The peer being observed.
   */
  readonly observedId: string

  /**
   * The session where this conclusion was made.
   */
  readonly sessionId: string

  /**
   * When the conclusion was created.
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
   * Create an Conclusion from an API response object.
   *
   * @param data - API response data
   * @returns A new Conclusion instance
   */
  static fromApiResponse(data: Record<string, unknown>): Conclusion {
    return new Conclusion(
      (data.id as string) ?? '',
      (data.content as string) ?? '',
      (data.observer_id as string) ?? '',
      (data.observed_id as string) ?? '',
      (data.session_id as string) ?? '',
      (data.created_at as string) ?? ''
    )
  }

  /**
   * Return a string representation of the Conclusion.
   */
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
 *
 * This class provides convenient methods to list, query, and delete conclusions
 * that are automatically scoped to a specific observer/observed pair.
 *
 * Typically accessed via `peer.conclusions` (for self-conclusions) or
 * `peer.conclusionsOf(target)` (for conclusions about another peer).
 *
 * @example
 * ```typescript
 * // Get self-conclusions
 * const conclusions = peer.conclusions
 * const obsList = await conclusions.list()
 * const searchResults = await conclusions.query('preferences')
 *
 * // Get conclusions about another peer
 * const bobConclusions = peer.conclusionsOf('bob')
 * const bobList = await bobConclusions.list()
 * ```
 *
 * @note
 * This class requires the core Honcho SDK to support conclusion endpoints.
 * The conclusion endpoints are:
 * - POST /workspaces/{workspace_id}/conclusions/list
 * - POST /workspaces/{workspace_id}/conclusions/query
 * - DELETE /workspaces/{workspace_id}/conclusions/{conclusion_id}
 */
export class ConclusionScope {
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
   * Initialize an ConclusionScope.
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
   * List conclusions in this scope.
   *
   * @param page - Page number (1-indexed)
   * @param size - Number of results per page
   * @param session - Optional session (ID string or Session object) to filter by
   * @returns Promise resolving to list of Conclusion objects
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

    // biome-ignore lint/suspicious/noExplicitAny: SDK workspaces type doesn't include conclusions
    const response = await (this._client.workspaces as any).conclusions.list(
      this.workspaceId,
      {
        filters,
        page,
        size,
      }
    )

    return (response.items ?? []).map((item: unknown) =>
      Conclusion.fromApiResponse(item as Record<string, unknown>)
    )
  }

  /**
   * Semantic search for conclusions in this scope.
   *
   * @param query - The search query string
   * @param topK - Maximum number of results to return
   * @param distance - Maximum cosine distance threshold (0.0-1.0)
   * @returns Promise resolving to list of matching Conclusion objects
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

    // biome-ignore lint/suspicious/noExplicitAny: SDK workspaces type doesn't include conclusions
    const response = await (this._client.workspaces as any).conclusions.query(
      this.workspaceId,
      {
        query,
        top_k: topK,
        distance,
        filters,
      }
    )

    return (response ?? []).map((item: unknown) =>
      Conclusion.fromApiResponse(item as Record<string, unknown>)
    )
  }

  /**
   * Delete a conclusion by ID.
   *
   * @param conclusionId - The ID of the conclusion to delete
   */
  async delete(conclusionId: string): Promise<void> {
    // biome-ignore lint/suspicious/noExplicitAny: SDK workspaces type doesn't include conclusions
    await (this._client.workspaces as any).conclusions.delete(
      this.workspaceId,
      conclusionId
    )
  }

  /**
   * Create conclusions in this scope.
   *
   * @param conclusions - Single conclusion or array of conclusions with content and sessionId
   * @returns Promise resolving to list of created Conclusion objects
   *
   * @example
   * ```typescript
   * // Create a single conclusion
   * const conclusions = await peer.conclusions.create(
   *   { content: 'User prefers dark mode', sessionId: 'session1' }
   * )
   *
   * // Create multiple conclusions
   * const conclusions = await peer.conclusions.create([
   *   { content: 'User prefers dark mode', sessionId: 'session1' },
   *   { content: 'User is interested in AI', sessionId: 'session1' },
   * ])
   * ```
   */
  async create(
    conclusions: ConclusionCreateParam | ConclusionCreateParam[]
  ): Promise<Conclusion[]> {
    // Normalize to array
    const conclusionArray = Array.isArray(conclusions)
      ? conclusions
      : [conclusions]

    // Build the request body with observer/observed from scope
    const requestConclusions = conclusionArray.map((obs) => ({
      content: obs.content,
      session_id:
        typeof obs.sessionId === 'string' ? obs.sessionId : obs.sessionId.id,
      observer_id: this.observer,
      observed_id: this.observed,
    }))

    // biome-ignore lint/suspicious/noExplicitAny: SDK workspaces type doesn't include conclusions
    const response = await (this._client.workspaces as any).conclusions.create(
      this.workspaceId,
      { conclusions: requestConclusions }
    )

    return (response ?? []).map((item: unknown) =>
      Conclusion.fromApiResponse(item as Record<string, unknown>)
    )
  }

  /**
   * Get the computed representation for this scope.
   *
   * This returns the working representation (narrative) built from the
   * conclusions in this scope.
   *
   * @param options - Optional options to configure the representation
   * @returns Promise resolving to a string of the representation
   */
  async getRepresentation(options?: RepresentationOptions): Promise<string> {
    const response = await this._client.workspaces.peers.getRepresentation(
      this.workspaceId,
      this.observer,
      {
        target: this.observed,
        search_query: options?.searchQuery,
        search_top_k: options?.searchTopK,
        search_max_distance: options?.searchMaxDistance,
        include_most_frequent: options?.includeMostFrequent,
        max_conclusions: options?.maxConclusions,
      }
    )
    return response.representation
  }

  /**
   * Return a string representation of the ConclusionScope.
   */
  toString(): string {
    return `ConclusionScope(workspaceId='${this.workspaceId}', observer='${this.observer}', observed='${this.observed}')`
  }
}
