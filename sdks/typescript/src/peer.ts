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
  async chat(query: string, opts?: {
    stream?: boolean;
    target?: string | Peer;
    sessionId?: string;
  }): Promise<string | null> {
    const response = await this._honcho['_client'].workspaces.peers.chat(
      this._honcho.workspaceId,
      this.id,
      { query, stream: opts?.stream, target: opts?.target ? (typeof opts.target === 'string' ? opts.target : opts.target.id) : undefined, session_id: opts?.sessionId },
    );
    if (!response.content || response.content === 'None') {
      return null;
    }
    return response.content;
  }

  /**
   * Get all sessions this peer is a member of.
   */
  async getSessions(filter?: { [key: string]: unknown } | null): Promise<Page<Session>> {
    const sessionsPage = await this._honcho['_client'].workspaces.peers.sessions.list(
      this._honcho.workspaceId,
      this.id,
      { filter }
    );
    return new Page(sessionsPage, (session: any) => new Session(session.id, this._honcho));
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
      throw new Error('Search query must be a non-empty string');
    }
    const messagesPage = await this._honcho['_client'].workspaces.peers.search(
      this._honcho.workspaceId,
      this.id,
      { query: query }
    );
    return new Page(messagesPage);
  }

  /**
   * Upload a file to create messages in this peer's global representation.
   *
   * Makes an API call to upload a file and convert it into messages. The file is
   * processed to extract text content, split into appropriately sized chunks,
   * and created as messages attributed to this peer.
   *
   * @param file File to upload. Should be an object with filename, content (as Buffer or Uint8Array), and content_type
   * @returns A list of Message objects representing the created messages
   * 
   * @note Supported file types include PDFs, text files, and JSON documents.
   *       Large files will be automatically split into multiple messages to fit
   *       within message size limits.
   */
  async uploadFile(
    file: { filename: string; content: Buffer | Uint8Array; content_type: string }
  ): Promise<any[]> {
    // Convert file to the format expected by the API
    const fileData = {
      filename: file.filename,
      content: file.content,
      content_type: file.content_type
    };

    // Call the upload endpoint
    const response = await (this._honcho['_client'] as any).workspaces.peers.messages.upload(
      this._honcho.workspaceId,
      this.id,
      {
        file: fileData
      }
    );

    return response;
  }
} 