import HonchoCore from '@honcho-ai/core'
import { Page } from './pagination'
import { Session } from './session'

/**
 * Represents a peer in the Honcho system.
 *
 * Peers can send messages, participate in sessions, and maintain both global
 * and local representations for contextual interactions. A peer represents
 * an entity (user, assistant, etc.) that can communicate within the system.
 */
export class Peer {
  /**
   * Unique identifier for this peer.
   */
  readonly id: string
  /**
   * Workspace ID for scoping operations.
   */
  readonly workspaceId: string
  /**
   * Reference to the parent Honcho client instance.
   */
  private _client: HonchoCore

  /**
   * Initialize a new Peer. **Do not call this directly, use the client.peer() method instead.**
   *
   * @param id - Unique identifier for this peer within the workspace
   * @param workspaceId - Workspace ID for scoping operations
   * @param client - Reference to the parent Honcho client instance
   */
  constructor(id: string, workspaceId: string, client: HonchoCore) {
    this.id = id
    this.workspaceId = workspaceId
    this._client = client
  }

  /**
   * Query the peer's representation with a natural language question.
   *
   * Makes an API call to the Honcho dialectic endpoint to query either the peer's
   * global representation (all content associated with this peer) or their local
   * representation of another peer (what this peer knows about the target peer).
   *
   * @param query - The natural language question to ask
   * @param stream - Whether to stream the response
   * @param target - Optional target peer for local representation query. If provided,
   *                 queries what this peer knows about the target peer rather than
   *                 querying the peer's global representation
   * @param sessionId - Optional session ID to scope the query to a specific session.
   *                    If provided, only information from that session is considered
   * @returns Promise resolving to response string containing the answer to the query,
   *          or null if no relevant information is available
   */
  async chat(
    query: string,
    stream?: boolean,
    target?: string | Peer,
    sessionId?: string,
  ): Promise<string | null> {
    const response = await this._client.workspaces.peers.chat(
      this.workspaceId,
      this.id,
      {
        query,
        stream: stream,
        target: target
          ? typeof target === 'string'
            ? target
            : target.id
          : undefined,
        session_id: sessionId,
      }
    )
    if (!response.content || response.content === 'None') {
      return null
    }
    return response.content
  }

  /**
   * Get all sessions this peer is a member of.
   *
   * Makes an API call to retrieve all sessions where this peer is an active participant.
   * Sessions are created when peers are added to them or send messages to them.
   *
   * @param filter - Optional filter criteria for sessions
   * @returns Promise resolving to a paginated list of Session objects this peer belongs to.
   *          Returns an empty list if the peer is not a member of any sessions
   */
  async getSessions(
    filter?: { [key: string]: unknown } | null
  ): Promise<Page<Session>> {
    const sessionsPage = await this._client.workspaces.peers.sessions.list(
      this.workspaceId,
      this.id,
      {
        filter,
      })
    return new Page(
      sessionsPage,
      (session: any) => new Session(session.id, this.workspaceId, this._client)
    )
  }

  /**
   * Create a message object attributed to this peer.
   *
   * This is a convenience method for creating message objects with this peer's ID.
   * The created message object can then be added to sessions or used in other operations.
   *
   * @param content - The text content for the message
   * @param opts - Optional parameters including metadata dictionary to associate with the message
   * @returns A new message object with this peer's ID and the provided content
   */
  message(content: string, opts?: { metadata?: Record<string, unknown> }): any {
    return {
      peerId: this.id,
      content,
      metadata: opts?.metadata,
    }
  }

  /**
   * Get the current metadata for this peer.
   *
   * Makes an API call to retrieve metadata associated with this peer. Metadata
   * can include custom attributes, settings, or any other key-value data
   * associated with the peer.
   *
   * @returns Promise resolving to a dictionary containing the peer's metadata.
   *          Returns an empty dictionary if no metadata is set
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const peer = await this._client.workspaces.peers.getOrCreate(
      this.workspaceId,
      { id: this.id }
    )
    return peer.metadata || {}
  }

  /**
   * Set the metadata for this peer.
   *
   * Makes an API call to update the metadata associated with this peer.
   * This will overwrite any existing metadata with the provided values.
   *
   * @param metadata - A dictionary of metadata to associate with this peer.
   *                   Keys must be strings, values can be any JSON-serializable type
   */
  async setMetadata(metadata: Record<string, object>): Promise<void> {
    await this._client.workspaces.peers.update(
      this.workspaceId,
      this.id,
      { metadata }
    )
  }

  /**
   * Get the current workspace-level configuration for this peer.
   *
   * Makes an API call to retrieve configuration associated with this peer.
   * Configuration currently includes one optional flag, `observe_me`.
   *
   * @returns Promise resolving to a dictionary containing the peer's configuration
   */
  async getPeerConfig(): Promise<Record<string, unknown>> {
    const peer = await this._client.workspaces.peers.getOrCreate(
      this.workspaceId,
      { id: this.id }
    )
    return peer.configuration || {}
  }

  /**
   * Set the configuration for this peer. Currently the only supported config
   * value is the `observe_me` flag, which controls whether derivation tasks
   * should be created for this peer's global representation. Default is True.
   *
   * Makes an API call to update the configuration associated with this peer.
   * This will overwrite any existing configuration with the provided values.
   *
   * @param config - A dictionary of configuration to associate with this peer.
   *                 Keys must be strings, values can be any JSON-serializable type
   */
  async setPeerConfig(config: Record<string, object>): Promise<void> {
    await this._client.workspaces.peers.update(
      this.workspaceId,
      this.id,
      { configuration: config }
    )
  }

  /**
   * Search for messages in the workspace with this peer as author.
   *
   * Makes an API call to search endpoint.
   *
   * @param query The search query to use
   * @returns A Page of Message objects representing the search results.
   *          Returns an empty page if no messages are found.
   */
  async search(query: string): Promise<Page<any>> {
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      throw new Error('Search query must be a non-empty string')
    }
    const messagesPage = await this._client.workspaces.peers.search(
      this.workspaceId,
      this.id,
      { query: query }
    )
    return new Page(messagesPage)
  }

  /**
   * Return a string representation of the Peer.
   *
   * @returns A string representation suitable for debugging
   */
  toString(): string {
    return `Peer(id='${this.id}')`
  }
}
