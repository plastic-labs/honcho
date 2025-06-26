import HonchoCore from '@honcho-ai/core';
import { Page } from './pagination';
import { Peer } from './peer';
import { Session } from './session';

/**
 * Main client for the Honcho TypeScript SDK.
 * Provides access to peers, sessions, and workspace operations.
 */
export class Honcho {
  private _client: InstanceType<typeof HonchoCore>;
  readonly workspaceId: string;

  /**
   * Initialize the Honcho client.
   */
  constructor(options: {
    apiKey?: string;
    environment?: 'local' | 'production' | 'demo';
    baseURL?: string;
    workspaceId?: string;
    timeout?: number;
    maxRetries?: number;
    defaultHeaders?: Record<string, string>;
    defaultQuery?: Record<string, unknown>;
  }) {
    this.workspaceId = options.workspaceId || process.env.HONCHO_WORKSPACE_ID || 'default';
    this._client = new HonchoCore({
      apiKey: options.apiKey || process.env.HONCHO_API_KEY,
      environment: options.environment,
      baseURL: options.baseURL || process.env.HONCHO_URL,
      timeout: options.timeout,
      maxRetries: options.maxRetries,
      defaultHeaders: options.defaultHeaders,
      defaultQuery: options.defaultQuery as any,
    }) as any;
    this._client.workspaces.getOrCreate({ id: this.workspaceId })
  }

  /**
   * Get or create a peer with the given ID.
   */
  peer(id: string, options?: { config?: Record<string, unknown> }): Peer {
    if (!id || typeof id !== 'string') {
      throw new Error('Peer ID must be a non-empty string');
    }
    return new Peer(id, this, options?.config);
  }

  /**
   * Get all peers in the current workspace.
   */
  async getPeers(): Promise<Page<Peer>> {
    const peersPage = await this._client.workspaces.peers.list(this.workspaceId);
    return new Page(peersPage, (peer: any) => new Peer(peer.id, this));
  }

  /**
   * Get or create a session with the given ID.
   */
  session(id: string, options?: { config?: Record<string, unknown> }): Session {
    if (!id || typeof id !== 'string') {
      throw new Error('Session ID must be a non-empty string');
    }
    return new Session(id, this, options?.config);
  }

  /**
   * Get all sessions in the current workspace.
   */
  async getSessions(): Promise<Page<Session>> {
    const sessionsPage = await this._client.workspaces.sessions.list(this.workspaceId);
    return new Page(sessionsPage, (session: any) => new Session(session.id, this));
  }

  /**
   * Get metadata for the current workspace.
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const workspace = await this._client.workspaces.getOrCreate({ id: this.workspaceId });
    return workspace.metadata || {};
  }

  /**
   * Set metadata for the current workspace.
   */
  async setMetadata(metadata: Record<string, unknown>): Promise<void> {
    await this._client.workspaces.update(this.workspaceId, { metadata });
  }

  /**
   * Get all workspace IDs from the Honcho instance.
   */
  async getWorkspaces(): Promise<string[]> {
    const workspacesPage = await this._client.workspaces.list();
    const ids: string[] = [];
    for await (const workspace of workspacesPage) {
      ids.push(workspace.id);
    }
    return ids;
  }

  /**
   * Search for messages in the current workspace.
   *
   * Makes an API call to search for messages in the current workspace.
   *
   * @param query The search query to use
   * @returns A Page of Message objects representing the search results.
   *          Returns an empty page if no messages are found.
   */
  async search(query: string): Promise<Page<any>> {
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      throw new Error('Search query must be a non-empty string');
    }
    const messagesPage = await this._client.workspaces.search(this.workspaceId, { body: query });
    return new Page(messagesPage);
  }
} 