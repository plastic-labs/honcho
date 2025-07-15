interface Env {
    // You can add environment variables here if needed
}

interface ExecutionContext {
    waitUntil(promise: Promise<any>): void;
    passThroughOnException(): void;
}

interface HonchoConfig {
    apiKey: string;
    baseUrl?: string;
    workspaceId?: string;
    userName?: string;
    assistantName?: string;
}

interface Message {
    role: 'user' | 'assistant';
    content: string;
    metadata?: Record<string, any>;
}

// JSON-RPC 2.0 interfaces
interface JsonRpcRequest {
    jsonrpc: '2.0';
    method: string;
    params?: any;
    id?: string | number;
}

interface JsonRpcResponse {
    jsonrpc: '2.0';
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
        type: 'object';
        properties: Record<string, any>;
        required?: string[];
    };
}

class HonchoWorker {
    private config: HonchoConfig;

    constructor(config: HonchoConfig) {
        this.config = {
            baseUrl: 'https://api.honcho.dev',
            workspaceId: 'default',
            userName: 'User',
            assistantName: 'Assistant',
            ...config,
        };
    }

    private async makeRequest(endpoint: string, options: RequestInit = {}) {
        const url = `${this.config.baseUrl}/v2${endpoint}`;
        const headers = {
            'Authorization': `Bearer ${this.config.apiKey}`,
            'Content-Type': 'application/json',
            ...options.headers,
        };

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000);

            const response = await fetch(url, {
                ...options,
                headers,
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`API Error: ${response.status} ${errorText}`);
                throw new Error(`Honcho API error: ${response.status} ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
                console.error(`Request timeout: ${url}`);
                throw new Error(`Request timeout: ${url}`);
            }
            console.error(`Request failed: ${url}`, error);
            throw error;
        }
    }

    async startConversation(): Promise<{ sessionId: string }> {
        const sessionId = crypto.randomUUID();

        try {
            // First, ensure the workspace exists
            await this.makeRequest(`/workspaces`, {
                method: 'POST',
                body: JSON.stringify({
                    id: this.config.workspaceId,
                }),
            });

            // Get or create the user peer
            await this.makeRequest(`/workspaces/${this.config.workspaceId}/peers`, {
                method: 'POST',
                body: JSON.stringify({
                    id: this.config.userName,
                }),
            });

            // Get or create the assistant peer
            await this.makeRequest(`/workspaces/${this.config.workspaceId}/peers`, {
                method: 'POST',
                body: JSON.stringify({
                    id: this.config.assistantName,
                }),
            });

            // Create the session
            await this.makeRequest(`/workspaces/${this.config.workspaceId}/sessions`, {
                method: 'POST',
                body: JSON.stringify({
                    id: sessionId,
                }),
            });

            // Add peers to the session
            await this.makeRequest(`/workspaces/${this.config.workspaceId}/sessions/${sessionId}/peers`, {
                method: 'POST',
                body: JSON.stringify({
                    [this.config.userName!]: {},
                    [this.config.assistantName!]: { observe_me: false }
                }),
            });

            return { sessionId };
        } catch (error) {
            console.error('Failed to start conversation:', error);
            throw error;
        }
    }

    async addTurn(sessionId: string, messages: Message[]): Promise<void> {
        try {
            // Validate messages
            for (let i = 0; i < messages.length; i++) {
                const message = messages[i];
                if (!message || typeof message !== 'object') {
                    throw new Error(`Message at index ${i} must be an object`);
                }
                if (!message.role) {
                    throw new Error(`Message at index ${i} is missing required field 'role'`);
                }
                if (!message.content) {
                    throw new Error(`Message at index ${i} is missing required field 'content'`);
                }
                if (!['user', 'assistant'].includes(message.role)) {
                    throw new Error(`Invalid role '${message.role}' at message index ${i}. Role must be 'user' or 'assistant'`);
                }
            }

            // Prepare messages for the API
            const apiMessages = messages.map(message => ({
                peer_id: message.role === 'user' ? this.config.userName! : this.config.assistantName!,
                content: message.content,
                metadata: message.metadata || {},
            }));

            // Add messages to the session
            await this.makeRequest(`/workspaces/${this.config.workspaceId}/sessions/${sessionId}/messages/`, {
                method: 'POST',
                body: JSON.stringify({
                    messages: apiMessages
                }),
            });
        } catch (error) {
            console.error('Failed to add turn:', error);
            throw error;
        }
    }

    async getPersonalizationInsights(query: string): Promise<string> {
        try {
            const response = await this.makeRequest(`/workspaces/${this.config.workspaceId}/peers/${this.config.userName}/chat`, {
                method: 'POST',
                body: JSON.stringify({
                    queries: query,
                    stream: false,
                }),
            });

            return (response as any).content || 'No personalization insights found.';
        } catch (error) {
            console.error('Error getting personalization insights:', error);
            return 'No personalization insights found.';
        }
    }
}

function parseConfig(request: Request): HonchoConfig | null {
    // Get API key from Authorization header
    const authHeader = request.headers.get('Authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return null;
    }
    const apiKey = authHeader.substring(7);

    if (!apiKey) {
        return null;
    }

    // Get configuration from headers with proper defaults
    const config: HonchoConfig = {
        apiKey,
        baseUrl: request.headers.get('X-Honcho-Base-URL') || 'https://api.honcho.dev',
        workspaceId: request.headers.get('X-Honcho-Workspace-ID') || 'default',
        userName: request.headers.get('X-Honcho-User-Name') || 'User',
        assistantName: request.headers.get('X-Honcho-Assistant-Name') || 'Assistant',
    };

    return config;
}

function createJsonRpcResponse(id: string | number | null, result?: any, error?: { code: number; message: string; data?: any }): JsonRpcResponse {
    const response: JsonRpcResponse = {
        jsonrpc: '2.0',
        id,
    };

    if (error) {
        response.error = error;
    } else {
        response.result = result;
    }

    return response;
}

function createJsonRpcError(code: number, message: string, data?: any): { code: number; message: string; data?: any } {
    return { code, message, data };
}

// Define MCP tools
const tools: Tool[] = [
    {
        name: 'start_conversation',
        description: 'Start a new conversation with a user. Call this when a user starts a new conversation.',
        inputSchema: {
            type: 'object',
            properties: {},
            required: [],
        },
    },
    {
        name: 'add_turn',
        description: 'Add a turn to a conversation. Call this after a user has sent a message and the assistant has responded.',
        inputSchema: {
            type: 'object',
            properties: {
                session_id: {
                    type: 'string',
                    description: 'The ID of the session to add the turn to.',
                },
                messages: {
                    type: 'array',
                    description: 'A list of messages to add to the session.',
                    items: {
                        type: 'object',
                        properties: {
                            role: {
                                type: 'string',
                                enum: ['user', 'assistant'],
                                description: 'The role of the message author.',
                            },
                            content: {
                                type: 'string',
                                description: 'The content of the message.',
                            },
                            metadata: {
                                type: 'object',
                                description: 'Optional metadata about the message.',
                            },
                        },
                        required: ['role', 'content'],
                    },
                },
            },
            required: ['session_id', 'messages'],
        },
    },
    {
        name: 'get_personalization_insights',
        description: 'Get personalization insights about the user, based on the query and the accumulated knowledge of the user across all conversations.',
        inputSchema: {
            type: 'object',
            properties: {
                query: {
                    type: 'string',
                    description: 'The question about the user\'s preferences, habits, etc.',
                },
            },
            required: ['query'],
        },
    },
];

export default {
    async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
        // Handle CORS preflight requests
        if (request.method === 'OPTIONS') {
            return new Response(null, {
                status: 200,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                },
            });
        }

        // Only accept POST requests for JSON-RPC
        if (request.method !== 'POST') {
            return new Response(JSON.stringify(createJsonRpcResponse(null, undefined, createJsonRpcError(-32600, 'Invalid Request'))), {
                status: 400,
                headers: { 'Content-Type': 'application/json' },
            });
        }

        let requestData: JsonRpcRequest;

        try {
            requestData = await request.json() as JsonRpcRequest;
        } catch (error) {
            return new Response(JSON.stringify(createJsonRpcResponse(null, undefined, createJsonRpcError(-32700, 'Parse error'))), {
                status: 400,
                headers: { 'Content-Type': 'application/json' },
            });
        }

        // Validate JSON-RPC format
        if (requestData.jsonrpc !== '2.0') {
            return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32600, 'Invalid Request'))), {
                status: 400,
                headers: { 'Content-Type': 'application/json' },
            });
        }

        if (!requestData.method) {
            return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32600, 'Invalid Request'))), {
                status: 400,
                headers: { 'Content-Type': 'application/json' },
            });
        }

        // Parse configuration
        const config = parseConfig(request);
        if (!config && requestData.method !== 'initialize') {
            return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32602, 'Missing or invalid API key'))), {
                status: 401,
                headers: { 'Content-Type': 'application/json' },
            });
        }

        const honcho = config ? new HonchoWorker(config) : null;

        try {
            switch (requestData.method) {
                case 'initialize':
                    // MCP initialization
                    return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, {
                        protocolVersion: '2024-11-05',
                        capabilities: {
                            tools: {}
                        },
                        serverInfo: {
                            name: 'Honcho MCP Server',
                            version: '1.0.0',
                        },
                    })), {
                        status: 200,
                        headers: {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*',
                        },
                    });

                case 'notifications/initialized':
                    // MCP initialized notification - no response needed
                    return new Response(null, {
                        status: 204,
                        headers: {
                            'Access-Control-Allow-Origin': '*',
                        },
                    });

                case 'tools/list':
                    // Return available tools
                    return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, {
                        tools: tools,
                    })), {
                        status: 200,
                        headers: {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*',
                        },
                    });

                case 'tools/call':
                    if (!honcho) {
                        return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32602, 'Missing API key'))), {
                            status: 401,
                            headers: { 'Content-Type': 'application/json' },
                        });
                    }

                    const toolName = requestData.params?.name;
                    const toolArguments = requestData.params?.arguments || {};

                    switch (toolName) {
                        case 'start_conversation':
                            const sessionResult = await honcho.startConversation();
                            return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, {
                                content: [{
                                    type: 'text',
                                    text: sessionResult.sessionId,
                                }],
                            })), {
                                status: 200,
                                headers: {
                                    'Content-Type': 'application/json',
                                    'Access-Control-Allow-Origin': '*',
                                },
                            });

                        case 'add_turn':
                            if (!toolArguments.session_id) {
                                return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32602, 'session_id is required'))), {
                                    status: 400,
                                    headers: { 'Content-Type': 'application/json' },
                                });
                            }
                            if (!Array.isArray(toolArguments.messages)) {
                                return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32602, 'messages must be an array'))), {
                                    status: 400,
                                    headers: { 'Content-Type': 'application/json' },
                                });
                            }

                            await honcho.addTurn(toolArguments.session_id, toolArguments.messages);
                            return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, {
                                content: [{
                                    type: 'text',
                                    text: 'Turn added successfully',
                                }],
                            })), {
                                status: 200,
                                headers: {
                                    'Content-Type': 'application/json',
                                    'Access-Control-Allow-Origin': '*',
                                },
                            });

                        case 'get_personalization_insights':
                            if (!toolArguments.query) {
                                return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32602, 'query is required'))), {
                                    status: 400,
                                    headers: { 'Content-Type': 'application/json' },
                                });
                            }

                            const insights = await honcho.getPersonalizationInsights(toolArguments.query);
                            return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, {
                                content: [{
                                    type: 'text',
                                    text: insights,
                                }],
                            })), {
                                status: 200,
                                headers: {
                                    'Content-Type': 'application/json',
                                    'Access-Control-Allow-Origin': '*',
                                },
                            });

                        default:
                            return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32601, `Method not found: ${toolName}`))), {
                                status: 404,
                                headers: { 'Content-Type': 'application/json' },
                            });
                    }

                default:
                    return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32601, `Method not found: ${requestData.method}`))), {
                        status: 404,
                        headers: { 'Content-Type': 'application/json' },
                    });
            }
        } catch (error) {
            console.error('Worker error:', error);
            const errorMessage = error instanceof Error ? error.message : 'Internal server error';
            return new Response(JSON.stringify(createJsonRpcResponse(requestData.id ?? null, undefined, createJsonRpcError(-32603, errorMessage))), {
                status: 500,
                headers: { 'Content-Type': 'application/json' },
            });
        }
    },
}; 