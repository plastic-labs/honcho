import { API_VERSION } from './api-version'
import type { HonchoHTTPClient } from './http/client'
import { Page } from './pagination'
// Type-only: peer.ts imports this module.
import type { Peer } from './peer'
import type {
  PageResponse,
  PeerMembershipCreateParams,
  PeerMembershipResponse,
} from './types/api'
import { resolveId } from './utils'
import {
  FilterSchema,
  type Filters,
  PeerIdSchema,
  PeerMembershipMetadataSchema,
} from './validation'

/**
 * One peer's membership in another.
 */
export interface PeerMembership {
  /** ID of the peer the member belongs to. */
  parentId: string
  /** ID of the member peer. */
  peerId: string
  workspaceId: string
  /** Application-owned metadata for the membership, such as a role. */
  metadata: Record<string, unknown>
  createdAt: string
}

function membershipFromApi(response: PeerMembershipResponse): PeerMembership {
  return {
    parentId: response.parent_id,
    peerId: response.peer_id,
    workspaceId: response.workspace_id,
    metadata: response.metadata ?? {},
    createdAt: response.created_at,
  }
}

/**
 * Fetch one page of a members or parents listing, wired to fetch the rest.
 *
 * @internal Shared by {@link PeerMembers.list} and `Peer.parents`.
 */
export async function listMemberships(
  http: HonchoHTTPClient,
  ensureWorkspace: () => Promise<void>,
  path: string,
  options?: { filters?: Filters; page?: number; size?: number }
): Promise<Page<PeerMembership, PeerMembershipResponse>> {
  const filters = options?.filters
    ? FilterSchema.parse(options.filters)
    : undefined
  const fetchPage = async (
    page?: number,
    size?: number
  ): Promise<PageResponse<PeerMembershipResponse>> => {
    await ensureWorkspace()
    return http.post<PageResponse<PeerMembershipResponse>>(path, {
      body: { filters },
      query: { page, size },
    })
  }

  return new Page(
    await fetchPage(options?.page, options?.size),
    membershipFromApi,
    fetchPage
  )
}

/**
 * The members of a peer.
 *
 * A peer's members are peers whose knowledge reads of that peer aggregate.
 * Chat, representation, context, card, and search that name the peer as their
 * target include each member's own knowledge, attributed to the member, down to
 * the requested `depth` (1 by default). No conclusions are copied: removing a
 * member simply stops reads from including it.
 *
 * Typically accessed via `peer.members`.
 *
 * @example
 * ```typescript
 * const team = await honcho.peer('engineering-team')
 * await team.members.add('alice', { metadata: { role: 'lead' } })
 * await team.members.add(bob)
 * const members = await team.members.list()
 * ```
 */
export class PeerMembers {
  readonly workspaceId: string
  /** The ID of the peer whose members this view manages. */
  readonly peerId: string
  private _http: HonchoHTTPClient
  private _ensureWorkspace: () => Promise<void>

  /**
   * Initialize a PeerMembers view. **Do not call this directly, use `peer.members` instead.**
   */
  constructor(
    http: HonchoHTTPClient,
    workspaceId: string,
    peerId: string,
    ensureWorkspace: () => Promise<void> = async () => undefined
  ) {
    this._http = http
    this.workspaceId = workspaceId
    this.peerId = peerId
    this._ensureWorkspace = ensureWorkspace
  }

  private get _basePath(): string {
    return `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${this.peerId}/members`
  }

  /**
   * Add a peer as a member of this peer.
   *
   * Both peers are created if they do not exist. Adding an existing member
   * again returns the existing membership, with its metadata replaced when
   * `metadata` is given. Requires a workspace- or admin-level key: a member's
   * knowledge becomes readable by anyone who can read this peer.
   *
   * @param peer - The peer to add, as an ID string or a Peer object
   * @param options.metadata - Optional application-owned metadata for the
   *                           membership, such as a role
   * @returns Promise resolving to the membership
   * @throws ConflictError if the membership would create a cycle
   * @throws UnprocessableEntityError if either peer is a scope, the peer names
   *         itself, or this peer already has the maximum number of members
   */
  async add(
    peer: string | Peer,
    options?: { metadata?: Record<string, unknown> }
  ): Promise<PeerMembership> {
    const body: PeerMembershipCreateParams = {
      peer_id: PeerIdSchema.parse(resolveId(peer)),
      metadata: options?.metadata
        ? PeerMembershipMetadataSchema.parse(options.metadata)
        : undefined,
    }
    await this._ensureWorkspace()
    const response = await this._http.post<PeerMembershipResponse>(
      this._basePath,
      { body }
    )
    return membershipFromApi(response)
  }

  /**
   * List this peer's members, oldest membership first.
   *
   * @param options.filters - Optional filter criteria on the member peers, in
   *                          the same shape as `honcho.peers()`
   * @param options.page - Page number (1-indexed)
   * @param options.size - Number of items per page
   * @returns Promise resolving to a paginated list of memberships
   */
  async list(options?: {
    filters?: Filters
    page?: number
    size?: number
  }): Promise<Page<PeerMembership, PeerMembershipResponse>> {
    return listMemberships(
      this._http,
      () => this._ensureWorkspace(),
      `${this._basePath}/list`,
      options
    )
  }

  /**
   * Remove a member from this peer.
   *
   * Removing a peer that is not a member is a no-op. No conclusions change: the
   * member keeps everything it knows, and reads of this peer stop including it.
   *
   * @param peer - The member to remove, as an ID string or a Peer object
   * @throws If the member ID is malformed
   */
  async remove(peer: string | Peer): Promise<void> {
    // Validated because this ID is interpolated into a request *path*:
    // `alice?typo` would otherwise remove `alice`.
    const memberId = PeerIdSchema.parse(resolveId(peer))
    await this._ensureWorkspace()
    await this._http.delete(`${this._basePath}/${memberId}`)
  }

  toString(): string {
    return `PeerMembers(peerId='${this.peerId}', workspaceId='${this.workspaceId}')`
  }
}
