import { Session } from './session';
import { Page } from './pagination';
import type { Honcho } from './client';

/**
 * Represents a peer in the Honcho system.
 */
export class Peer {
  /**
   * Unique identifier for this peer.
   */
  readonly id: string;
  private _honcho: Honcho;

  /**
   * Initialize a new Peer.
   */
  constructor(id: string, honcho: Honcho, config?: Record<string, unknown>) {
    this.id = id;
    this._honcho = honcho;
    
    if (config) {
      this._honcho['_client'].workspaces.peers.getOrCreate(
        this._honcho.workspaceId,
        { id: this.id, configuration: config }
      );
    }
  }

  /**
   * Query the peer's representation with a natural language question.
   */
  async chat(queries: string | string[], opts?: {
    stream?: boolean;
    target?: string | Peer;
    sessionId?: string;
  }): Promise<string | null> {
    const response = await this._honcho['_client'].workspaces.peers.chat(
      this._honcho.workspaceId,
      this.id,
      { queries, stream: opts?.stream, target: opts?.target ? (typeof opts.target === 'string' ? opts.target : opts.target.id) : undefined, session_id: opts?.sessionId },
    );
    if (!response.content || response.content === 'None') {
      return null;
    }
    return response.content;
  }

  /**
   * Get all sessions this peer is a member of.
   */
  async getSessions(): Promise<Page<Session>> {
    const sessionsPage = await this._honcho['_client'].workspaces.peers.sessions.list(
      this.id,
      this._honcho.workspaceId,
    );
    return new Page(sessionsPage, (session: any) => new Session(session.id, this._honcho));
  }

  /**
   * Add messages or content to this peer's global representation.
   */
  async addMessages(content: string | any | any[]): Promise<void> {
    let messages: any[];
    if (typeof content === 'string') {
      messages = [{ peer_id: this.id, content, metadata: undefined }];
    } else if (Array.isArray(content)) {
      messages = content.map((msg) => ({
        peer_id: msg.peerId || this.id,
        content: msg.content,
        metadata: msg.metadata,
      }));
    } else {
      messages = [{
        peer_id: content.peerId || this.id,
        content: content.content,
        metadata: content.metadata,
      }];
    }
    await this._honcho['_client'].workspaces.peers.messages.create(
      this._honcho.workspaceId,
      this.id,
      { messages }
    );
  }

  /**
   * Get messages saved to this peer outside of a session with optional filtering.
   */
  async getMessages(opts?: { filter?: Record<string, unknown> }): Promise<Page<any>> {
    const messagesPage = await this._honcho['_client'].workspaces.peers.messages.list(
      this.id,
      this._honcho.workspaceId,
      opts?.filter,
    );
    return new Page(messagesPage);
  }

  /**
   * Create a message attributed to this peer.
   */
  message(content: string, opts?: { metadata?: Record<string, unknown> }): any {
    return {
      peerId: this.id,
      content,
      metadata: opts?.metadata,
    };
  }

  /**
   * Get the current metadata for this peer.
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const peer = await this._honcho['_client'].workspaces.peers.getOrCreate(
      this._honcho.workspaceId,
      { id: this.id }
    );
    return peer.metadata || {};
  }

  /**
   * Set the metadata for this peer.
   */
  async setMetadata(metadata: Record<string, unknown>): Promise<void> {
    await this._honcho['_client'].workspaces.peers.update(
      this._honcho.workspaceId,
      this.id,
      { metadata },
    );
  }

  /**
   * Search for messages in this peer's global representation.
   *
   * Makes an API call to search for messages in this peer's global representation.
   *
   * @param query The search query to use
   * @returns A Page of Message objects representing the search results.
   *          Returns an empty page if no messages are found.
   */
  async search(query: string): Promise<Page<any>> {
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      throw new Error('Search query must be a non-empty string');
    }
    const messagesPage = await this._honcho['_client'].workspaces.peers.search(
      this._honcho.workspaceId,
      this.id,
      { body: query }
    );
    return new Page(messagesPage);
  }
} 