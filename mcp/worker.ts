import { Honcho } from "@honcho-ai/sdk";

interface HonchoConfig {
  apiKey: string;
  userName: string;
  baseUrl?: string;
  workspaceId?: string;
  assistantName?: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  metadata?: Record<string, any>;
}

/**
 * JSON-RPC 2.0 request interface
 */
interface JsonRpcRequest {
  jsonrpc: "2.0";
  method: string;
  params?: any;
  id?: string | number;
}

/**
 * JSON-RPC 2.0 response interface
 */
interface JsonRpcResponse {
  jsonrpc: "2.0";
  id?: string | number | null;
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
}

// MCP Tool definitions
interface Tool {
  name: string;
  description: string;
  inputSchema: {
    type: "object";
    properties: Record<string, any>;
    required?: string[];
  };
}

/**
 * Helper function to validate required arguments and create error responses
 */
function validateArguments(
  args: Record<string, any>,
  required: string[],
  requestId: string | number | null,
): Response | null {
  for (const param of required) {
    if (!args[param]) {
      return createErrorResponse(requestId, -32602, `${param} is required`);
    }
  }

  // Special validation for arrays
  if (args.messages && !Array.isArray(args.messages)) {
    return createErrorResponse(requestId, -32602, "messages must be an array");
  }
  if (args.peer_ids && !Array.isArray(args.peer_ids)) {
    return createErrorResponse(requestId, -32602, "peer_ids must be an array");
  }

  return null;
}

/**
 * Helper function to create error responses
 */
function createErrorResponse(
  id: string | number | null,
  code: number,
  message: string,
): Response {
  return new Response(
    JSON.stringify(
      createJsonRpcResponse(id, undefined, createJsonRpcError(code, message)),
    ),
    {
      status: code === -32602 ? 400 : code === -32601 ? 404 : 500,
      headers: { "Content-Type": "application/json" },
    },
  );
}

/**
 * Helper function to format messages for async iteration
 */
async function formatMessages(messagesPage: any): Promise<any[]> {
  const messages = [];
  for await (const message of messagesPage) {
    messages.push({
      id: message.id,
      content: message.content,
      peer_id: message.peer_id,
      session_id: message.session_id,
      metadata: message.metadata,
      created_at: message.created_at,
    });
  }
  return messages;
}

class HonchoWorker {
  private honcho: Honcho;
  private config: HonchoConfig;

  constructor(config: HonchoConfig) {
    this.config = {
      baseUrl: "https://api.honcho.dev",
      workspaceId: "default",
      assistantName: "Assistant",
      ...config,
    };

    this.honcho = new Honcho({
      apiKey: this.config.apiKey,
      baseURL: this.config.baseUrl,
      workspaceId: this.config.workspaceId,
    });
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///                                                                          ///
  ///  "Bespoke" tools: easy to use for user-assistant conversation paradigms  ///
  ///                                                                          ///
  ////////////////////////////////////////////////////////////////////////////////

  /**
   * Start a new conversation with a user. Call this when a user starts a new conversation.
   * @returns A session ID for the conversation
   */
  async startConversation(): Promise<string> {
    // Get/create the peers first, before session creation
    // This avoids a race condition where peer creation during session.addPeers
    // could rollback the session if there's an IntegrityError
    const userPeer = await this.honcho.peer(this.config.userName);
    const assistantPeer = await this.honcho.peer(this.config.assistantName!, {
      config: { observe_me: false },
    });

    // Create a new session - pass empty config to force API call
    const sessionId = crypto.randomUUID();
    const session = await this.honcho.session(sessionId, { config: {} });

    // Add the user and assistant peers to the session
    await session.addPeers([
      userPeer,
      [assistantPeer, { observe_me: null, observe_others: false }],
    ]);

    return sessionId;
  }

  /**
   * Add a turn to a conversation. Call this after a user has sent a message and the assistant has responded.
   * @param sessionId - The ID of the session to add the turn to
   * @param messages - A list of messages to add to the session
   */
  async addTurn(sessionId: string, messages: Message[]): Promise<void> {
    const session = await this.honcho.session(sessionId);
    const userPeer = await this.honcho.peer(this.config.userName);
    const assistantPeer = await this.honcho.peer(this.config.assistantName!);

    const sessionMessages = [];

    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];

      // Validate required fields
      if (!message || typeof message !== "object") {
        throw new Error(`Message at index ${i} must be a dictionary`);
      }

      if (!message.role) {
        throw new Error(
          `Message at index ${i} is missing required field 'role'`,
        );
      }

      if (!message.content) {
        throw new Error(
          `Message at index ${i} is missing required field 'content'`,
        );
      }

      const { role, content, metadata } = message;

      // Create message with appropriate peer
      if (role === "user") {
        if (metadata) {
          sessionMessages.push(userPeer.message(content, { metadata }));
        } else {
          sessionMessages.push(userPeer.message(content));
        }
      } else if (role === "assistant") {
        if (metadata) {
          sessionMessages.push(assistantPeer.message(content, { metadata }));
        } else {
          sessionMessages.push(assistantPeer.message(content));
        }
      } else {
        throw new Error(
          `Invalid role '${role}' at message index ${i}. Role must be one of: 'user' or 'assistant'`,
        );
      }
    }

    await session.addMessages(sessionMessages);
  }

  /**
   * Get personalization insights about the user, based on the query and the accumulated knowledge of the user across all conversations.
   * @param sessionId - The ID of the session for context
   * @param query - The question about the user's preferences, habits, etc.
   * @param reasoningLevel - Optional reasoning level: "minimal", "low", "medium", "high", or "max"
   * @returns A string with the personalization insights
   */
  async getPersonalizationInsights(
    sessionId: string,
    query: string,
    reasoningLevel?: string,
  ): Promise<string> {
    const userPeer = await this.honcho.peer(this.config.userName);

    // Get the personalization insights (non-streaming returns string | null)
    const personalizationInsights = await userPeer.chat(query, {
      session: sessionId,
      reasoningLevel,
    });

    if (!personalizationInsights || typeof personalizationInsights !== 'string') {
      return "No personalization insights found.";
    }

    return personalizationInsights;
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///                                                                          ///
  ///                      General tools for using Honcho                      ///
  ///                                                                          ///
  ////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////
  ///                                                ///
  ///              Workspace operations              ///
  ///                                                ///
  //////////////////////////////////////////////////////

  /**
   * Search for messages across the entire workspace.
   * @param query - The search query to use
   * @returns A list of message dictionaries matching the search query
   */
  async searchWorkspace(query: string): Promise<any[]> {
    const messagesPage = await this.honcho.search(query);
    return await formatMessages(messagesPage);
  }

  /**
   * Get metadata for the current workspace.
   * @returns A dictionary containing the workspace's metadata
   */
  async getWorkspaceMetadata(): Promise<Record<string, any>> {
    return await this.honcho.getMetadata();
  }

  /**
   * Set metadata for the current workspace.
   * @param metadata - A dictionary of metadata to associate with the workspace
   */
  async setWorkspaceMetadata(metadata: Record<string, any>): Promise<void> {
    await this.honcho.setMetadata(metadata);
  }

  //////////////////////////////////////////////////////
  ///                                                ///
  ///                 Peer operations                ///
  ///                                                ///
  //////////////////////////////////////////////////////

  /**
   * Create or get a peer with the specified ID and optional configuration.
   * @param peerId - Unique identifier for the peer
   * @param config - Optional configuration dictionary for the peer
   * @returns A dictionary with the peer ID and confirmation of creation
   */
  async createPeer(
    peerId: string,
    config?: Record<string, any>,
  ): Promise<{ peer_id: string; config?: Record<string, any> }> {
    const peer = await this.honcho.peer(peerId, { config });
    return {
      peer_id: peer.id,
      config,
    };
  }

  /**
   * Get metadata for a specific peer.
   * @param peerId - The ID of the peer to get metadata for
   * @returns A dictionary containing the peer's metadata
   */
  async getPeerMetadata(peerId: string): Promise<Record<string, any>> {
    const peer = await this.honcho.peer(peerId);
    return await peer.getMetadata();
  }

  /**
   * Set metadata for a specific peer.
   * @param peerId - The ID of the peer to set metadata for
   * @param metadata - A dictionary of metadata to associate with the peer
   */
  async setPeerMetadata(
    peerId: string,
    metadata: Record<string, any>,
  ): Promise<void> {
    const peer = await this.honcho.peer(peerId);
    await peer.setMetadata(metadata);
  }

  /**
   * Search for messages sent by a peer.
   * @param peerId - The ID of the peer to search messages for
   * @param query - The search query to use
   * @returns A list of message dictionaries matching the search query
   */
  async searchPeerMessages(peerId: string, query: string): Promise<any[]> {
    const peer = await this.honcho.peer(peerId);
    const messagesPage = await peer.search(query);
    return await formatMessages(messagesPage);
  }

  /**
   * Query a peer's representation with natural language questions.
   * @param peerId - The ID of the peer to query
   * @param query - The natural language question to ask
   * @param targetPeerId - Optional target peer ID for local representation queries
   * @param sessionId - Optional session ID to scope the query to a specific session
   * @param reasoningLevel - Optional reasoning level: "minimal", "low", "medium", "high", or "max"
   * @returns Response string containing the answer to the query, or "None" if no relevant information
   */
  async chat(
    peerId: string,
    query: string,
    targetPeerId?: string,
    sessionId?: string,
    reasoningLevel?: string,
  ): Promise<string> {
    const peer = await this.honcho.peer(peerId);
    let targetPeer;
    if (targetPeerId) {
      targetPeer = await this.honcho.peer(targetPeerId);
    }

    const result = await peer.chat(query, {
      target: targetPeer,
      session: sessionId,
      reasoningLevel,
    });

    if (!result || typeof result !== 'string') {
      return "None";
    }
    return result;
  }

  /**
   * Get all peers in the current workspace.
   * @returns A list of peer dictionaries with their IDs
   */
  async listPeers(): Promise<{ id: string }[]> {
    const peersPage = await this.honcho.getPeers();
    const peers = [];

    for await (const peer of peersPage) {
      peers.push({
        id: peer.id,
      });
    }

    return peers;
  }

  /**
   * Get the peer card for a peer.
   * @param peerId - The ID of the observer peer
   * @param targetPeerId - Optional target peer ID to get the card about
   * @returns The peer card content or null if not found
   */
  async getPeerCard(
    peerId: string,
    targetPeerId?: string,
  ): Promise<string | null> {
    const peer = await this.honcho.peer(peerId);
    let targetPeer;
    if (targetPeerId) {
      targetPeer = await this.honcho.peer(targetPeerId);
    }
    return await peer.card(targetPeer);
  }

  /**
   * Get the peer context (combined representation and peer card).
   * @param peerId - The ID of the observer peer
   * @param targetPeerId - Optional target peer ID
   * @param sessionId - Optional session ID to scope the context
   * @param searchQuery - Optional semantic search query
   * @param maxConclusions - Optional maximum number of conclusions to include
   * @returns Context object with representation and peer card
   */
  async getPeerContext(
    peerId: string,
    targetPeerId?: string,
    sessionId?: string,
    searchQuery?: string,
    maxConclusions?: number,
  ): Promise<Record<string, any>> {
    const peer = await this.honcho.peer(peerId);
    let targetPeer;
    if (targetPeerId) {
      targetPeer = await this.honcho.peer(targetPeerId);
    }
    const context = await peer.context({
      target: targetPeer,
      session: sessionId,
      searchQuery,
      maxConclusions,
    });
    return {
      peer_id: context.peerId,
      target_id: context.targetId,
      representation: context.representation,
      peer_card: context.peerCard,
    };
  }

  /**
   * Get the formatted representation for a peer.
   * @param peerId - The ID of the observer peer
   * @param targetPeerId - Optional target peer ID
   * @param sessionId - Optional session ID to scope the representation
   * @param searchQuery - Optional semantic search query
   * @param maxConclusions - Optional maximum number of conclusions
   * @returns Formatted representation string
   */
  async getRepresentation(
    peerId: string,
    targetPeerId?: string,
    sessionId?: string,
    searchQuery?: string,
    maxConclusions?: number,
  ): Promise<string> {
    const peer = await this.honcho.peer(peerId);
    let targetPeer;
    if (targetPeerId) {
      targetPeer = await this.honcho.peer(targetPeerId);
    }
    return await peer.representation({
      target: targetPeer,
      session: sessionId,
      searchQuery,
      maxConclusions,
    });
  }

  //////////////////////////////////////////////////////
  ///                                                ///
  ///              Conclusions operations            ///
  ///                                                ///
  //////////////////////////////////////////////////////

  /**
   * List conclusions for a peer.
   * @param peerId - The ID of the observer peer
   * @param targetPeerId - Optional target peer to get conclusions about
   * @returns List of conclusions
   */
  async listConclusions(
    peerId: string,
    targetPeerId?: string,
  ): Promise<any[]> {
    const peer = await this.honcho.peer(peerId);
    let conclusionScope;
    if (targetPeerId) {
      const targetPeer = await this.honcho.peer(targetPeerId);
      conclusionScope = peer.conclusionsOf(targetPeer);
    } else {
      conclusionScope = peer.conclusions;
    }

    const conclusionsPage = await conclusionScope.list();
    const conclusions = [];
    for await (const conclusion of conclusionsPage) {
      conclusions.push({
        id: conclusion.id,
        content: conclusion.content,
        observer_id: conclusion.observerId,
        observed_id: conclusion.observedId,
        session_id: conclusion.sessionId,
        created_at: conclusion.createdAt,
      });
    }
    return conclusions;
  }

  /**
   * Query conclusions using semantic search.
   * @param peerId - The ID of the observer peer
   * @param query - The semantic search query
   * @param targetPeerId - Optional target peer to search conclusions about
   * @param topK - Maximum number of results to return
   * @returns List of matching conclusions
   */
  async queryConclusions(
    peerId: string,
    query: string,
    targetPeerId?: string,
    topK?: number,
  ): Promise<any[]> {
    const peer = await this.honcho.peer(peerId);
    let conclusionScope;
    if (targetPeerId) {
      const targetPeer = await this.honcho.peer(targetPeerId);
      conclusionScope = peer.conclusionsOf(targetPeer);
    } else {
      conclusionScope = peer.conclusions;
    }

    // SDK query returns Conclusion[] directly, not a Page
    const conclusions = await conclusionScope.query(query, topK);
    return conclusions.map((conclusion) => ({
      id: conclusion.id,
      content: conclusion.content,
      observer_id: conclusion.observerId,
      observed_id: conclusion.observedId,
      session_id: conclusion.sessionId,
      created_at: conclusion.createdAt,
    }));
  }

  /**
   * Create conclusions manually.
   * @param peerId - The ID of the observer peer
   * @param targetPeerId - The target peer the conclusions are about
   * @param conclusions - List of conclusion content strings
   * @param sessionId - Optional session ID to associate with conclusions
   * @returns Number of conclusions created
   */
  async createConclusions(
    peerId: string,
    targetPeerId: string,
    conclusions: string[],
    sessionId?: string,
  ): Promise<number> {
    const peer = await this.honcho.peer(peerId);
    const targetPeer = await this.honcho.peer(targetPeerId);
    const conclusionScope = peer.conclusionsOf(targetPeer);

    // Convert string array to ConclusionCreateParams array
    const conclusionParams = conclusions.map((content) => ({
      content,
      sessionId,
    }));

    await conclusionScope.create(conclusionParams);
    return conclusions.length;
  }

  /**
   * Delete a conclusion.
   * @param peerId - The ID of the observer peer
   * @param targetPeerId - The target peer the conclusion is about
   * @param conclusionId - The ID of the conclusion to delete
   */
  async deleteConclusion(
    peerId: string,
    targetPeerId: string,
    conclusionId: string,
  ): Promise<void> {
    const peer = await this.honcho.peer(peerId);
    const targetPeer = await this.honcho.peer(targetPeerId);
    const conclusionScope = peer.conclusionsOf(targetPeer);

    await conclusionScope.delete(conclusionId);
  }

  //////////////////////////////////////////////////////
  ///                                                ///
  ///              System operations                 ///
  ///                                                ///
  //////////////////////////////////////////////////////

  /**
   * Schedule a dream (memory consolidation) for a peer.
   * @param peerId - The ID of the observer peer
   * @param targetPeerId - Optional target peer to dream about (defaults to observer for self-reflection)
   * @param sessionId - Optional session ID to scope the dream to
   * @returns Confirmation message
   */
  async scheduleDream(
    peerId: string,
    targetPeerId?: string,
    sessionId?: string,
  ): Promise<string> {
    const peer = await this.honcho.peer(peerId);
    const session = sessionId ? await this.honcho.session(sessionId) : undefined;
    const targetPeer = targetPeerId ? await this.honcho.peer(targetPeerId) : undefined;

    await this.honcho.scheduleDream({
      observer: peer,
      session: session,
      observed: targetPeer,
    });
    return "Dream scheduled successfully";
  }

  /**
   * Get the current queue status for the deriver.
   * @returns Queue status information
   */
  async getQueueStatus(): Promise<Record<string, any>> {
    return await this.honcho.queueStatus();
  }

  //////////////////////////////////////////////////////
  ///                                                ///
  ///                Session operations              ///
  ///                                                ///
  //////////////////////////////////////////////////////

  /**
   * Create or get a session with the specified ID and optional configuration.
   * @param sessionId - Unique identifier for the session
   * @param config - Optional configuration dictionary for the session
   * @returns A dictionary with the session ID and confirmation of creation
   */
  async createSession(
    sessionId: string,
    config?: Record<string, any>,
  ): Promise<{ session_id: string; config?: Record<string, any> }> {
    // Always pass config (even if empty) to force the API call to create the session
    // Without this, the SDK just creates a local Session object without making an API call
    const session = await this.honcho.session(sessionId, { config: config ?? {} });
    return {
      session_id: session.id,
      config,
    };
  }

  /**
   * Get metadata for a specific session.
   * @param sessionId - The ID of the session to get metadata for
   * @returns A dictionary containing the session's metadata
   */
  async getSessionMetadata(sessionId: string): Promise<Record<string, any>> {
    const session = await this.honcho.session(sessionId);
    return await session.getMetadata();
  }

  /**
   * Set metadata for a specific session.
   * @param sessionId - The ID of the session to set metadata for
   * @param metadata - A dictionary of metadata to associate with the session
   */
  async setSessionMetadata(
    sessionId: string,
    metadata: Record<string, any>,
  ): Promise<void> {
    const session = await this.honcho.session(sessionId);
    await session.setMetadata(metadata);
  }

  /**
   * Add peers to a session.
   * @param sessionId - The ID of the session to add peers to
   * @param peerIds - List of peer IDs to add to the session
   */
  async addPeersToSession(sessionId: string, peerIds: string[]): Promise<void> {
    const session = await this.honcho.session(sessionId);
    await session.addPeers(peerIds);
  }

  /**
   * Remove peers from a session.
   * @param sessionId - The ID of the session to remove peers from
   * @param peerIds - List of peer IDs to remove from the session
   */
  async removePeersFromSession(
    sessionId: string,
    peerIds: string[],
  ): Promise<void> {
    const session = await this.honcho.session(sessionId);
    await session.removePeers(peerIds);
  }

  /**
   * Get all peer IDs in a session.
   * @param sessionId - The ID of the session to get peers from
   * @returns A list of peer IDs that are members of the session
   */
  async getSessionPeers(sessionId: string): Promise<string[]> {
    const session = await this.honcho.session(sessionId);
    const peers = await session.getPeers();
    return peers.map((peer) => peer.id);
  }

  /**
   * Add messages to a session.
   * @param sessionId - The ID of the session to add messages to
   * @param messages - List of message dictionaries
   */
  async addMessagesToSession(
    sessionId: string,
    messages: {
      peer_id: string;
      content: string;
      metadata?: Record<string, any>;
    }[],
  ): Promise<void> {
    const session = await this.honcho.session(sessionId);

    const sessionMessages = [];
    for (const message of messages) {
      const peer = await this.honcho.peer(message.peer_id);
      if (message.metadata) {
        sessionMessages.push(
          peer.message(message.content, { metadata: message.metadata }),
        );
      } else {
        sessionMessages.push(peer.message(message.content));
      }
    }

    await session.addMessages(sessionMessages);
  }

  /**
   * Get messages from a session with optional filtering.
   * @param sessionId - The ID of the session to get messages from
   * @param filters - Optional dictionary of filter criteria
   * @returns A list of message dictionaries
   */
  async getSessionMessages(
    sessionId: string,
    filters?: Record<string, any>,
  ): Promise<any[]> {
    const session = await this.honcho.session(sessionId);
    const messagesPage = await session.getMessages({ filter: filters });
    return await formatMessages(messagesPage);
  }

  /**
   * Get optimized context for a session within a token limit.
   * @param sessionId - The ID of the session to get context for
   * @param summary - Whether to include summary information
   * @param tokens - Maximum number of tokens to include in the context
   * @returns A dictionary containing the session context with messages and optional summary
   */
  async getSessionContext(
    sessionId: string,
    summary: boolean = true,
    tokens?: number,
  ): Promise<any> {
    const session = await this.honcho.session(sessionId);
    const context = await session.getContext({ summary, tokens });

    return {
      session_id: context.sessionId,
      summary: context.summary,
      messages: context.messages.map((msg) => ({
        id: msg.id,
        content: msg.content,
        peer_id: msg.peer_id,
        metadata: msg.metadata,
        created_at: msg.created_at,
      })),
    };
  }

  /**
   * Search for messages in a specific session.
   * @param sessionId - The ID of the session to search messages in
   * @param query - The search query to use
   * @returns A list of message dictionaries matching the search query
   */
  async searchSessionMessages(
    sessionId: string,
    query: string,
  ): Promise<any[]> {
    const session = await this.honcho.session(sessionId);
    const messagesPage = await session.search(query);
    return await formatMessages(messagesPage);
  }

  /**
   * Get the current working representation of a peer in a session.
   * @param sessionId - The ID of the session
   * @param peerId - The ID of the peer to get the working representation of
   * @param targetPeerId - Optional target peer ID to get the representation of what peer_id knows about target_peer_id
   * @returns A dictionary containing information about the peer
   */
  async getWorkingRepresentation(
    sessionId: string,
    peerId: string,
    targetPeerId?: string,
  ): Promise<Record<string, any>> {
    const session = await this.honcho.session(sessionId);
    if (targetPeerId) {
      return await session.workingRep(peerId, targetPeerId);
    } else {
      return await session.workingRep(peerId);
    }
  }

  /**
   * Get all sessions in the current workspace.
   * @returns A list of session dictionaries with their IDs
   */
  async listSessions(): Promise<{ id: string }[]> {
    const sessionsPage = await this.honcho.getSessions();
    const sessions = [];

    for await (const session of sessionsPage) {
      sessions.push({
        id: session.id,
      });
    }

    return sessions;
  }
}

/**
 * Parse configuration from request headers
 * @param request - The incoming request
 * @returns Configuration object or null if invalid
 */
function parseConfig(request: Request): HonchoConfig | null {
  // Get API key from Authorization header
  const authHeader = request.headers.get("Authorization");
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return null;
  }
  const apiKey = authHeader.substring(7);

  if (!apiKey) {
    return null;
  }

  const userName = request.headers.get("X-Honcho-User-Name");
  if (!userName) {
    return null;
  }

  // Get configuration from headers with proper defaults
  const config: HonchoConfig = {
    apiKey,
    userName,
    baseUrl:
      request.headers.get("X-Honcho-Base-URL") || "https://api.honcho.dev",
    workspaceId: request.headers.get("X-Honcho-Workspace-ID") || "default",
    assistantName:
      request.headers.get("X-Honcho-Assistant-Name") || "Assistant",
  };

  return config;
}

/**
 * Create a JSON-RPC 2.0 response
 * @param id - Request ID
 * @param result - Response result
 * @param error - Error object if any
 * @returns JSON-RPC response object
 */
function createJsonRpcResponse(
  id: string | number | null,
  result?: any,
  error?: { code: number; message: string; data?: any },
): JsonRpcResponse {
  const response: JsonRpcResponse = {
    jsonrpc: "2.0",
    id,
  };

  if (error) {
    response.error = error;
  } else {
    response.result = result;
  }

  return response;
}

/**
 * Create a JSON-RPC 2.0 error object
 * @param code - Error code
 * @param message - Error message
 * @param data - Optional error data
 * @returns Error object
 */
function createJsonRpcError(
  code: number,
  message: string,
  data?: any,
): { code: number; message: string; data?: any } {
  return { code, message, data };
}

// Define all MCP tools based on the Python server.py functions
const tools: Tool[] = [
  // Bespoke tools
  {
    name: "start_conversation",
    description:
      "Start a new conversation with a user. Call this when a user starts a new conversation.",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },
  {
    name: "add_turn",
    description:
      "Add a turn to a conversation. Call this after a user has sent a message and the assistant has responded.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to add the turn to.",
        },
        messages: {
          type: "array",
          description: "A list of messages to add to the session.",
          items: {
            type: "object",
            properties: {
              role: {
                type: "string",
                enum: ["user", "assistant"],
                description: "The role of the message author.",
              },
              content: {
                type: "string",
                description: "The content of the message.",
              },
              metadata: {
                type: "object",
                description: "Optional metadata about the message.",
              },
            },
            required: ["role", "content"],
          },
        },
      },
      required: ["session_id", "messages"],
    },
  },
  {
    name: "get_personalization_insights",
    description:
      "Get personalization insights about the user, based on the query and the accumulated knowledge of the user across all conversations.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session for context.",
        },
        query: {
          type: "string",
          description:
            "The question about the user's preferences, habits, etc.",
        },
        reasoning_level: {
          type: "string",
          enum: ["minimal", "low", "medium", "high", "max"],
          description:
            "The reasoning level for the response. Higher levels provide more detailed analysis.",
        },
      },
      required: ["session_id", "query"],
    },
  },

  // Workspace operations
  {
    name: "search_workspace",
    description: "Search for messages across the entire workspace.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The search query to use.",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "get_workspace_metadata",
    description: "Get metadata for the current workspace.",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },
  {
    name: "set_workspace_metadata",
    description: "Set metadata for the current workspace.",
    inputSchema: {
      type: "object",
      properties: {
        metadata: {
          type: "object",
          description:
            "A dictionary of metadata to associate with the workspace.",
        },
      },
      required: ["metadata"],
    },
  },

  // Peer operations
  {
    name: "create_peer",
    description:
      "Create or get a peer with the specified ID and optional configuration.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "Unique identifier for the peer.",
        },
        config: {
          type: "object",
          description: "Optional configuration dictionary for the peer.",
        },
      },
      required: ["peer_id"],
    },
  },
  {
    name: "get_peer_metadata",
    description: "Get metadata for a specific peer.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the peer to get metadata for.",
        },
      },
      required: ["peer_id"],
    },
  },
  {
    name: "set_peer_metadata",
    description: "Set metadata for a specific peer.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the peer to set metadata for.",
        },
        metadata: {
          type: "object",
          description: "A dictionary of metadata to associate with the peer.",
        },
      },
      required: ["peer_id", "metadata"],
    },
  },
  {
    name: "search_peer_messages",
    description: "Search for messages sent by a peer.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the peer to search messages for.",
        },
        query: {
          type: "string",
          description: "The search query to use.",
        },
      },
      required: ["peer_id", "query"],
    },
  },
  {
    name: "chat",
    description:
      "Query a peer's representation with natural language questions.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the peer to query.",
        },
        query: {
          type: "string",
          description: "The natural language question to ask.",
        },
        target_peer_id: {
          type: "string",
          description:
            "Optional target peer ID for local representation queries.",
        },
        session_id: {
          type: "string",
          description:
            "Optional session ID to scope the query to a specific session.",
        },
        reasoning_level: {
          type: "string",
          enum: ["minimal", "low", "medium", "high", "max"],
          description:
            "The reasoning level for the response. Higher levels provide more detailed analysis.",
        },
      },
      required: ["peer_id", "query"],
    },
  },
  {
    name: "list_peers",
    description: "Get all peers in the current workspace.",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },
  {
    name: "get_peer_card",
    description:
      "Get the peer card for a peer. The peer card contains compact biographical facts about the peer.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the observer peer.",
        },
        target_peer_id: {
          type: "string",
          description:
            "Optional target peer ID to get the card about. If not provided, returns the peer's own card.",
        },
      },
      required: ["peer_id"],
    },
  },
  {
    name: "get_peer_context",
    description:
      "Get the peer context, which combines the representation and peer card for comprehensive peer information.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the observer peer.",
        },
        target_peer_id: {
          type: "string",
          description: "Optional target peer ID to get context about.",
        },
        session_id: {
          type: "string",
          description: "Optional session ID to scope the context.",
        },
        search_query: {
          type: "string",
          description: "Optional semantic search query to filter conclusions.",
        },
        max_conclusions: {
          type: "integer",
          description: "Maximum number of conclusions to include.",
        },
      },
      required: ["peer_id"],
    },
  },
  {
    name: "get_representation",
    description:
      "Get the formatted representation for a peer, containing their conclusions and observations.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the observer peer.",
        },
        target_peer_id: {
          type: "string",
          description: "Optional target peer ID to get representation about.",
        },
        session_id: {
          type: "string",
          description: "Optional session ID to scope the representation.",
        },
        search_query: {
          type: "string",
          description: "Optional semantic search query to filter conclusions.",
        },
        max_conclusions: {
          type: "integer",
          description: "Maximum number of conclusions to include.",
        },
      },
      required: ["peer_id"],
    },
  },

  // Conclusions operations
  {
    name: "list_conclusions",
    description:
      "List conclusions for a peer. Conclusions are facts and observations derived from conversations.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the observer peer.",
        },
        target_peer_id: {
          type: "string",
          description:
            "Optional target peer ID to get conclusions about. If not provided, returns conclusions about self.",
        },
      },
      required: ["peer_id"],
    },
  },
  {
    name: "query_conclusions",
    description: "Query conclusions using semantic search.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the observer peer.",
        },
        query: {
          type: "string",
          description: "The semantic search query.",
        },
        target_peer_id: {
          type: "string",
          description: "Optional target peer ID to search conclusions about.",
        },
        top_k: {
          type: "integer",
          description: "Maximum number of results to return.",
        },
      },
      required: ["peer_id", "query"],
    },
  },
  {
    name: "create_conclusions",
    description: "Create conclusions manually for a peer.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the observer peer.",
        },
        target_peer_id: {
          type: "string",
          description: "The target peer the conclusions are about.",
        },
        conclusions: {
          type: "array",
          items: { type: "string" },
          description: "List of conclusion content strings to create.",
        },
        session_id: {
          type: "string",
          description:
            "Optional session ID to associate with conclusions. If not provided, conclusions will be global.",
        },
      },
      required: ["peer_id", "target_peer_id", "conclusions"],
    },
  },
  {
    name: "delete_conclusion",
    description: "Delete a specific conclusion.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the observer peer.",
        },
        target_peer_id: {
          type: "string",
          description: "The target peer the conclusion is about.",
        },
        conclusion_id: {
          type: "string",
          description: "The ID of the conclusion to delete.",
        },
      },
      required: ["peer_id", "target_peer_id", "conclusion_id"],
    },
  },

  // System operations
  {
    name: "schedule_dream",
    description:
      "Schedule a dream (memory consolidation) for a peer. Dreams help consolidate and improve memory quality.",
    inputSchema: {
      type: "object",
      properties: {
        peer_id: {
          type: "string",
          description: "The ID of the observer peer.",
        },
        target_peer_id: {
          type: "string",
          description:
            "Optional target peer to dream about. If not provided, defaults to the observer peer (self-reflection).",
        },
        session_id: {
          type: "string",
          description:
            "Optional session ID to scope the dream to. If not provided, the dream will be global.",
        },
      },
      required: ["peer_id"],
    },
  },
  {
    name: "get_queue_status",
    description:
      "Get the current queue status for the deriver (background processing system).",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },

  // Session operations
  {
    name: "create_session",
    description:
      "Create or get a session with the specified ID and optional configuration.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "Unique identifier for the session.",
        },
        config: {
          type: "object",
          description: "Optional configuration dictionary for the session.",
        },
      },
      required: ["session_id"],
    },
  },
  {
    name: "get_session_metadata",
    description: "Get metadata for a specific session.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to get metadata for.",
        },
      },
      required: ["session_id"],
    },
  },
  {
    name: "set_session_metadata",
    description: "Set metadata for a specific session.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to set metadata for.",
        },
        metadata: {
          type: "object",
          description:
            "A dictionary of metadata to associate with the session.",
        },
      },
      required: ["session_id", "metadata"],
    },
  },
  {
    name: "add_peers_to_session",
    description: "Add peers to a session.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to add peers to.",
        },
        peer_ids: {
          type: "array",
          items: { type: "string" },
          description: "List of peer IDs to add to the session.",
        },
      },
      required: ["session_id", "peer_ids"],
    },
  },
  {
    name: "remove_peers_from_session",
    description: "Remove peers from a session.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to remove peers from.",
        },
        peer_ids: {
          type: "array",
          items: { type: "string" },
          description: "List of peer IDs to remove from the session.",
        },
      },
      required: ["session_id", "peer_ids"],
    },
  },
  {
    name: "get_session_peers",
    description: "Get all peer IDs in a session.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to get peers from.",
        },
      },
      required: ["session_id"],
    },
  },
  {
    name: "add_messages_to_session",
    description: "Add messages to a session.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to add messages to.",
        },
        messages: {
          type: "array",
          items: {
            type: "object",
            properties: {
              peer_id: {
                type: "string",
                description: "ID of the peer sending the message",
              },
              content: {
                type: "string",
                description: "Message content",
              },
              metadata: {
                type: "object",
                description: "Optional metadata dictionary",
              },
            },
            required: ["peer_id", "content"],
          },
          description: "List of message dictionaries.",
        },
      },
      required: ["session_id", "messages"],
    },
  },
  {
    name: "get_session_messages",
    description: "Get messages from a session with optional filtering.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to get messages from.",
        },
        filters: {
          type: "object",
          description: "Optional dictionary of filter criteria.",
        },
      },
      required: ["session_id"],
    },
  },
  {
    name: "get_session_context",
    description: "Get optimized context for a session within a token limit.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to get context for.",
        },
        summary: {
          type: "boolean",
          description: "Whether to include summary information.",
          default: true,
        },
        tokens: {
          type: "integer",
          description: "Maximum number of tokens to include in the context.",
        },
      },
      required: ["session_id"],
    },
  },
  {
    name: "search_session_messages",
    description: "Search for messages in a specific session.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session to search messages in.",
        },
        query: {
          type: "string",
          description: "The search query to use.",
        },
      },
      required: ["session_id", "query"],
    },
  },
  {
    name: "get_working_representation",
    description:
      "Get the current working representation of a peer in a session.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "The ID of the session.",
        },
        peer_id: {
          type: "string",
          description:
            "The ID of the peer to get the working representation of.",
        },
        target_peer_id: {
          type: "string",
          description:
            "Optional target peer ID to get the representation of what peer_id knows about target_peer_id.",
        },
      },
      required: ["session_id", "peer_id"],
    },
  },
  {
    name: "list_sessions",
    description: "Get all sessions in the current workspace.",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },
];

/**
 * Execute a tool with validation and consistent response handling
 */
async function executeToolCall(
  honcho: HonchoWorker,
  toolName: string,
  toolArguments: any,
  requestId: string | number | null,
): Promise<Response> {
  let result: any;

  switch (toolName) {
    // Bespoke tools
    case "start_conversation":
      result = await honcho.startConversation();
      break;

    case "add_turn": {
      const validation = validateArguments(
        toolArguments,
        ["session_id", "messages"],
        requestId,
      );
      if (validation) return validation;

      await honcho.addTurn(toolArguments.session_id, toolArguments.messages);
      result = "Turn added successfully";
      break;
    }

    case "get_personalization_insights": {
      const validation = validateArguments(
        toolArguments,
        ["session_id", "query"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getPersonalizationInsights(
        toolArguments.session_id,
        toolArguments.query,
        toolArguments.reasoning_level,
      );
      break;
    }

    // Workspace operations
    case "search_workspace": {
      const validation = validateArguments(toolArguments, ["query"], requestId);
      if (validation) return validation;

      result = await honcho.searchWorkspace(toolArguments.query);
      break;
    }

    case "get_workspace_metadata":
      result = await honcho.getWorkspaceMetadata();
      break;

    case "set_workspace_metadata": {
      const validation = validateArguments(
        toolArguments,
        ["metadata"],
        requestId,
      );
      if (validation) return validation;

      await honcho.setWorkspaceMetadata(toolArguments.metadata);
      result = "Workspace metadata set successfully";
      break;
    }

    // Peer operations
    case "create_peer": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.createPeer(
        toolArguments.peer_id,
        toolArguments.config,
      );
      break;
    }

    case "get_peer_metadata": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getPeerMetadata(toolArguments.peer_id);
      break;
    }

    case "set_peer_metadata": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id", "metadata"],
        requestId,
      );
      if (validation) return validation;

      await honcho.setPeerMetadata(
        toolArguments.peer_id,
        toolArguments.metadata,
      );
      result = "Peer metadata set successfully";
      break;
    }

    case "search_peer_messages": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id", "query"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.searchPeerMessages(
        toolArguments.peer_id,
        toolArguments.query,
      );
      break;
    }

    case "chat": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id", "query"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.chat(
        toolArguments.peer_id,
        toolArguments.query,
        toolArguments.target_peer_id,
        toolArguments.session_id,
        toolArguments.reasoning_level,
      );
      break;
    }

    case "list_peers":
      result = await honcho.listPeers();
      break;

    case "get_peer_card": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getPeerCard(
        toolArguments.peer_id,
        toolArguments.target_peer_id,
      );
      break;
    }

    case "get_peer_context": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getPeerContext(
        toolArguments.peer_id,
        toolArguments.target_peer_id,
        toolArguments.session_id,
        toolArguments.search_query,
        toolArguments.max_conclusions,
      );
      break;
    }

    case "get_representation": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getRepresentation(
        toolArguments.peer_id,
        toolArguments.target_peer_id,
        toolArguments.session_id,
        toolArguments.search_query,
        toolArguments.max_conclusions,
      );
      break;
    }

    // Conclusions operations
    case "list_conclusions": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.listConclusions(
        toolArguments.peer_id,
        toolArguments.target_peer_id,
      );
      break;
    }

    case "query_conclusions": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id", "query"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.queryConclusions(
        toolArguments.peer_id,
        toolArguments.query,
        toolArguments.target_peer_id,
        toolArguments.top_k,
      );
      break;
    }

    case "create_conclusions": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id", "target_peer_id", "conclusions"],
        requestId,
      );
      if (validation) return validation;

      const count = await honcho.createConclusions(
        toolArguments.peer_id,
        toolArguments.target_peer_id,
        toolArguments.conclusions,
        toolArguments.session_id,
      );
      result = `Created ${count} conclusions successfully`;
      break;
    }

    case "delete_conclusion": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id", "target_peer_id", "conclusion_id"],
        requestId,
      );
      if (validation) return validation;

      await honcho.deleteConclusion(
        toolArguments.peer_id,
        toolArguments.target_peer_id,
        toolArguments.conclusion_id,
      );
      result = "Conclusion deleted successfully";
      break;
    }

    // System operations
    case "schedule_dream": {
      const validation = validateArguments(
        toolArguments,
        ["peer_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.scheduleDream(
        toolArguments.peer_id,
        toolArguments.target_peer_id,
        toolArguments.session_id,
      );
      break;
    }

    case "get_queue_status":
      result = await honcho.getQueueStatus();
      break;

    // Session operations
    case "create_session": {
      const validation = validateArguments(
        toolArguments,
        ["session_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.createSession(
        toolArguments.session_id,
        toolArguments.config,
      );
      break;
    }

    case "get_session_metadata": {
      const validation = validateArguments(
        toolArguments,
        ["session_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getSessionMetadata(toolArguments.session_id);
      break;
    }

    case "set_session_metadata": {
      const validation = validateArguments(
        toolArguments,
        ["session_id", "metadata"],
        requestId,
      );
      if (validation) return validation;

      await honcho.setSessionMetadata(
        toolArguments.session_id,
        toolArguments.metadata,
      );
      result = "Session metadata set successfully";
      break;
    }

    case "add_peers_to_session": {
      const validation = validateArguments(
        toolArguments,
        ["session_id", "peer_ids"],
        requestId,
      );
      if (validation) return validation;

      await honcho.addPeersToSession(
        toolArguments.session_id,
        toolArguments.peer_ids,
      );
      result = "Peers added to session successfully";
      break;
    }

    case "remove_peers_from_session": {
      const validation = validateArguments(
        toolArguments,
        ["session_id", "peer_ids"],
        requestId,
      );
      if (validation) return validation;

      await honcho.removePeersFromSession(
        toolArguments.session_id,
        toolArguments.peer_ids,
      );
      result = "Peers removed from session successfully";
      break;
    }

    case "get_session_peers": {
      const validation = validateArguments(
        toolArguments,
        ["session_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getSessionPeers(toolArguments.session_id);
      break;
    }

    case "add_messages_to_session": {
      const validation = validateArguments(
        toolArguments,
        ["session_id", "messages"],
        requestId,
      );
      if (validation) return validation;

      await honcho.addMessagesToSession(
        toolArguments.session_id,
        toolArguments.messages,
      );
      result = "Messages added to session successfully";
      break;
    }

    case "get_session_messages": {
      const validation = validateArguments(
        toolArguments,
        ["session_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getSessionMessages(
        toolArguments.session_id,
        toolArguments.filters,
      );
      break;
    }

    case "get_session_context": {
      const validation = validateArguments(
        toolArguments,
        ["session_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getSessionContext(
        toolArguments.session_id,
        toolArguments.summary,
        toolArguments.tokens,
      );
      break;
    }

    case "search_session_messages": {
      const validation = validateArguments(
        toolArguments,
        ["session_id", "query"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.searchSessionMessages(
        toolArguments.session_id,
        toolArguments.query,
      );
      break;
    }

    case "get_working_representation": {
      const validation = validateArguments(
        toolArguments,
        ["session_id", "peer_id"],
        requestId,
      );
      if (validation) return validation;

      result = await honcho.getWorkingRepresentation(
        toolArguments.session_id,
        toolArguments.peer_id,
        toolArguments.target_peer_id,
      );
      break;
    }

    case "list_sessions":
      result = await honcho.listSessions();
      break;

    default:
      return createErrorResponse(
        requestId,
        -32601,
        `Method not found: ${toolName}`,
      );
  }

  const responseData =
    typeof result === "string" ? result : JSON.stringify(result);
  return new Response(
    JSON.stringify(
      createJsonRpcResponse(requestId, {
        content: [
          {
            type: "text",
            text: responseData,
          },
        ],
      }),
    ),
    {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    },
  );
}

/**
 * Main Cloudflare Worker export
 */
export default {
  async fetch(request: Request): Promise<Response> {
    // Handle CORS preflight requests
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 200,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers":
            "Content-Type, Authorization, X-Honcho-User-Name, X-Honcho-Base-URL, X-Honcho-Workspace-ID, X-Honcho-Assistant-Name",
        },
      });
    }

    // Only accept POST requests for JSON-RPC
    if (request.method !== "POST") {
      return createErrorResponse(null, -32600, "Invalid Request");
    }

    let requestData: JsonRpcRequest;

    try {
      requestData = (await request.json()) as JsonRpcRequest;
    } catch (error) {
      return createErrorResponse(null, -32700, "Parse error");
    }

    // Validate JSON-RPC format
    if (requestData.jsonrpc !== "2.0") {
      return createErrorResponse(
        requestData.id ?? null,
        -32600,
        "Invalid Request",
      );
    }

    if (!requestData.method) {
      return createErrorResponse(
        requestData.id ?? null,
        -32600,
        "Invalid Request",
      );
    }

    // Parse configuration
    const config = parseConfig(request);
    if (!config && requestData.method !== "initialize") {
      return createErrorResponse(
        requestData.id ?? null,
        -32602,
        "Missing or invalid API key",
      );
    }

    const honcho = config ? new HonchoWorker(config) : null;

    try {
      switch (requestData.method) {
        case "initialize":
          return new Response(
            JSON.stringify(
              createJsonRpcResponse(requestData.id ?? null, {
                protocolVersion: "2024-11-05",
                capabilities: {
                  tools: {},
                },
                serverInfo: {
                  name: "Honcho MCP Server",
                  version: "3.0.0",
                },
              }),
            ),
            {
              status: 200,
              headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
              },
            },
          );

        case "notifications/initialized":
          // MCP initialized notification - no response needed
          return new Response(null, {
            status: 204,
            headers: {
              "Access-Control-Allow-Origin": "*",
            },
          });

        case "tools/list":
          return new Response(
            JSON.stringify(
              createJsonRpcResponse(requestData.id ?? null, {
                tools: tools,
              }),
            ),
            {
              status: 200,
              headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
              },
            },
          );

        case "tools/call":
          if (!honcho) {
            return createErrorResponse(
              requestData.id ?? null,
              -32602,
              "Missing API key",
            );
          }

          const toolName = requestData.params?.name;
          const toolArguments = requestData.params?.arguments || {};

          return await executeToolCall(
            honcho,
            toolName,
            toolArguments,
            requestData.id ?? null,
          );

        default:
          return createErrorResponse(
            requestData.id ?? null,
            -32601,
            `Method not found: ${requestData.method}`,
          );
      }
    } catch (error) {
      console.error("Worker error:", error);
      const errorMessage =
        error instanceof Error ? error.message : "Internal server error";
      return createErrorResponse(requestData.id ?? null, -32603, errorMessage);
    }
  },
};
