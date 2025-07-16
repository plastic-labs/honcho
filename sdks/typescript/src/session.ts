import { Peer } from './peer';
import { Page } from './pagination';
import { SessionContext } from './session_context';
import type { Honcho } from './client';


export class SessionPeerConfig {
  observe_others: boolean;
  observe_me: boolean;

  constructor(opts?: { observe_others?: boolean; observe_me?: boolean }) {
    this.observe_others = opts?.observe_others ?? false;
    this.observe_me = opts?.observe_me ?? true;
  }
}


/**
 * Represents a session in Honcho.
 */
export class Session {
  /**
   * Unique identifier for this session.
   */
  readonly id: string;
  private _honcho: Honcho;

  /**
   * Initialize a new Session.
   */
  constructor(id: string, honcho: Honcho, config?: Record<string, unknown>) {
    this.id = id;
    this._honcho = honcho;

    if (config) {
      this._honcho['_client'].workspaces.sessions.getOrCreate(
        this._honcho.workspaceId,
        { id: this.id, configuration: config }
      );
    }
  }

  /**
   * Add peers to this session.
   */
  async addPeers(peers: string | Peer | Array<string | Peer> | [string | Peer, SessionPeerConfig] | Array<[string | Peer, SessionPeerConfig]> | Array<string | Peer | [string | Peer, SessionPeerConfig]>): Promise<void> {
    const peerDict: Record<string, SessionPeerConfig> = {};
    if (!Array.isArray(peers)) {
      peers = [peers];
    }
    for (const peer of peers) {
      if (typeof peer === 'string') {
        peerDict[peer] = { observe_others: false, observe_me: true };
      } else if (typeof peer === 'object' && 'id' in peer) {
        peerDict[peer.id] = { observe_others: false, observe_me: true };
      } else if (Array.isArray(peer)) {
        const peerId = typeof peer[0] === 'string' ? peer[0] : peer[0].id;
        peerDict[peerId] = peer[1];
      } else if (typeof peer === 'object' && 'id' in peer && 'observe_others' in peer && 'observe_me' in peer) {
        peerDict[(peer as any).id] = { observe_others: (peer as any).observe_others, observe_me: (peer as any).observe_me };
      }
    }
    await (this._honcho['_client'] as any).workspaces.sessions.peers.add(
      this._honcho.workspaceId,
      this.id,
      peerDict
    );
  }

  /**
   * Set the complete peer list for this session.
   */
  async setPeers(peers: string | Peer | Array<string | Peer> | [string | Peer, SessionPeerConfig] | Array<[string | Peer, SessionPeerConfig]> | Array<string | Peer | [string | Peer, SessionPeerConfig]>): Promise<void> {
    const peerDict: Record<string, SessionPeerConfig> = {};
    if (!Array.isArray(peers)) {
      peers = [peers];
    }
    for (const peer of peers) {
      if (typeof peer === 'string') {
        peerDict[peer] = { observe_others: false, observe_me: true };
      } else if (typeof peer === 'object' && 'id' in peer) {
        peerDict[peer.id] = { observe_others: false, observe_me: true };
      } else if (Array.isArray(peer)) {
        const peerId = typeof peer[0] === 'string' ? peer[0] : peer[0].id;
        peerDict[peerId] = peer[1];
      } else if (typeof peer === 'object' && 'id' in peer && 'observe_others' in peer && 'observe_me' in peer) {
        peerDict[(peer as any).id] = { observe_others: (peer as any).observe_others, observe_me: (peer as any).observe_me };
      }
    }
    await (this._honcho['_client'] as any).workspaces.sessions.peers.set(
      this._honcho.workspaceId,
      this.id,
      peerDict
    );
  }

  /**
   * Remove peers from this session.
   */
  async removePeers(peers: string | Peer | Array<string | Peer>): Promise<void> {
    const peerIds = Array.isArray(peers)
      ? peers.map((p) => (typeof p === 'string' ? p : p.id))
      : [typeof peers === 'string' ? peers : peers.id];
    await (this._honcho['_client'] as any).workspaces.sessions.peers.remove(this._honcho.workspaceId, this.id, peerIds);
  }

  /**
   * Get all peers in this session. Automatically converts the paginated response
   * into a list for us -- the max number of peers in a session is usually 10.
   */
  async getPeers(): Promise<Peer[]> {
    const peersPage = await (this._honcho['_client'] as any).workspaces.sessions.peers.list(this._honcho.workspaceId, this.id);
    return peersPage.items.map((peer: any) => new Peer(peer.id, this._honcho));
  }

  /**
   * Get the configuration for a peer in this session.
   */
  async getPeerConfig(peer: string | Peer): Promise<SessionPeerConfig> {
    const peerId = typeof peer === 'string' ? peer : peer.id;
    return await (this._honcho['_client'] as any).workspaces.sessions.peers.getConfig(
      this._honcho.workspaceId,
      this.id,
      peerId
    );
  }

  /**
   * Set the configuration for a peer in this session.
   */
  async setPeerConfig(peer: string | Peer, config: SessionPeerConfig): Promise<void> {
    const peerId = typeof peer === 'string' ? peer : peer.id;
    await (this._honcho['_client'] as any).workspaces.sessions.peers.setConfig(
      this._honcho.workspaceId,
      this.id,
      peerId,
      {
        observe_others: config.observe_others,
        observe_me: config.observe_me
      }
    );
  }

  /**
   * Add one or more messages to this session.
   */
  async addMessages(messages: any | any[]): Promise<void> {
    const msgs = Array.isArray(messages) ? messages : [messages];
    await (this._honcho['_client'] as any).workspaces.sessions.messages.create(
      this._honcho.workspaceId,
      this.id,
      {
        messages: msgs.map((msg) => ({
          peer_id: msg.peerId,
          content: msg.content,
          metadata: msg.metadata,
        }))
      }
    );
  }

  /**
   * Get messages from this session with optional filtering.
   */
  async getMessages(opts?: { filter?: Record<string, unknown> }): Promise<Page<any>> {
    const messagesPage = await (this._honcho['_client'] as any).workspaces.sessions.messages.list(this._honcho.workspaceId, this.id, opts?.filter);
    return new Page(messagesPage);
  }

  /**
   * Get metadata for this session.
   */
  async getMetadata(): Promise<Record<string, unknown>> {
    const session = await (this._honcho['_client'] as any).workspaces.sessions.getOrCreate(
      this._honcho.workspaceId,
      { id: this.id }
    );
    return session.metadata || {};
  }

  /**
   * Set metadata for this session.
   */
  async setMetadata(metadata: Record<string, unknown>): Promise<void> {
    await (this._honcho['_client'] as any).workspaces.sessions.update(this._honcho.workspaceId, this.id, { metadata });
  }

  /**
   * Get optimized context for this session within a token limit.
   */
  async getContext(opts?: { summary?: boolean; tokens?: number }): Promise<SessionContext> {
    const context = await (this._honcho['_client'] as any).workspaces.sessions.getContext(
      this._honcho.workspaceId,
      this.id,
      {
        tokens: opts?.tokens,
        summary: opts?.summary
      }
    );
    return new SessionContext(this.id, context.messages, context.summary || '');
  }

  /**
   * Search for messages in this session.
   *
   * Makes an API call to search for messages in this session.
   *
   * @param query The search query to use
   * @returns A Page of Message objects representing the search results.
   *          Returns an empty page if no messages are found.
   */
  async search(query: string): Promise<Page<any>> {
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      throw new Error('Search query must be a non-empty string');
    }
    const messagesPage = await (this._honcho['_client'] as any).workspaces.sessions.search(
      this._honcho.workspaceId,
      this.id,
      { query: query }
    );
    return new Page(messagesPage);
  }

  /**
   * Upload a file to create messages in this session.
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
    file: { filename: string; content: Buffer | Uint8Array; content_type: string },
    peerId: string,
  ): Promise<any[]> {
    // Convert file to the format expected by the API
    const fileData = {
      filename: file.filename,
      content: file.content,
      content_type: file.content_type
    };

    // Call the upload endpoint
    const response = await (this._honcho['_client'] as any).workspaces.sessions.messages.upload(
      this._honcho.workspaceId,
      this.id,
      {
        files: fileData,
        peer_id: peerId,
      }
    );

    return response;
  }

  /**
   * Get the current working representation of the peer in this session.
   * 
   * @param peer The peer to get the working representation of.
   * @param target The target peer to get the representation of. If provided, queries what `peer` knows about the `target`.
   * @returns A dictionary containing information about the peer.
   */
  async workingRep(peer: string | Peer, target?: string | Peer): Promise<Record<string, unknown>> {
    const peerId = typeof peer === 'string' ? peer : peer.id;
    const targetId = target ? (typeof target === 'string' ? target : target.id) : undefined;

    return await (this._honcho['_client'] as any).workspaces.peers.workingRepresentation(
      this._honcho.workspaceId,
      peerId,
      {
        session_id: this.id,
        target: targetId
      }
    );
  }
} 