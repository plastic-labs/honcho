# Honcho MCP Server

## Quickstart: Use the Hosted MCP Server

Go to https://app.honcho.dev and get an API key. Then go to Claude Desktop and navigate to custom MCP servers.

If you don't have node/bun installed you will need to do that. You can also use npm if you already have that installed. If not, Claude Desktop or Claude Code can help!

Add Honcho to your Claude desktop config. You must provide a username for Honcho to refer to you as -- preferably what you want Claude to actually call you.
```json
{
  "mcpServers": {
    "honcho": {
      "command": "bunx",
      "args": [
        "mcp-remote",
        "https://mcp.honcho.dev",
        "--header",
        "Authorization:${AUTH_HEADER}",
        "--header",
        "X-Honcho-User-Name:${USER_NAME}"
      ],
      "env": {
        "AUTH_HEADER": "Bearer <your-honcho-key>",
        "USER_NAME": "<your-name>"
      }
    }
  }
}
```

You may customize your assistant name and/or workspace ID. Both are optional.

```json
{
  "mcpServers": {
    "honcho": {
      "command": "bunx",
      "args": [
        "mcp-remote",
        "https://mcp.honcho.dev",
        "--header",
        "Authorization:${AUTH_HEADER}",
        "--header",
        "X-Honcho-User-Name:${USER_NAME}",
        "--header",
        "X-Honcho-Assistant-Name:${ASSISTANT_NAME}",
        "--header",
        "X-Honcho-Workspace-ID:${WORKSPACE_ID}"
      ],
      "env": {
        "AUTH_HEADER": "Bearer <your-honcho-key>",
        "USER_NAME": "<your-name>",
        "ASSISTANT_NAME": "<your-assistant-name>",
        "WORKSPACE_ID": "<your-custom-workspace-id>"
      }
    }
  }
}
```

## Available Tools

### start_conversation
Start a new conversation session with Honcho. This initializes a session for tracking conversation history and context.

**Returns:** A session ID that you must store and use for all subsequent interactions in this conversation.

### add_turn
Add a conversation turn (user and assistant messages) to the current session. This stores the conversation in Honcho for context tracking.

**Parameters:**
- `session_id`: The ID of the session to add the turn to
- `messages`: Array of message objects with `role` ("user" or "assistant") and `content`

**Example usage:**
```json
{
  "session_id": "session-uuid",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant", 
      "content": "I'm doing well, thank you!"
    }
  ]
}
```

### get_personalization_insights
Get personalization insights from Honcho based on conversation history. This queries the user's conversation context to provide personalized responses.

**Parameters:**
- `session_id`: The ID of the session for context
- `query`: The question about the user's preferences, habits, etc.

**Example queries:**
- "What does this message reveal about the user's communication preferences?"
- "How formal or casual should I be with the user based on our history?"
- "What emotional state might the user be in right now?"

### search_workspace
Search for messages across the entire workspace.

**Parameters:**
- `query`: The search query to use

### get_workspace_metadata
Get metadata for the current workspace.

**Parameters:** None

### set_workspace_metadata
Set metadata for the current workspace.

**Parameters:**
- `metadata`: A dictionary of metadata to associate with the workspace

### create_peer
Create or get a peer with the specified ID and optional configuration.

**Parameters:**
- `peer_id`: Unique identifier for the peer
- `config`: Optional configuration dictionary for the peer

### get_peer_metadata
Get metadata for a specific peer.

**Parameters:**
- `peer_id`: The ID of the peer to get metadata for

### set_peer_metadata
Set metadata for a specific peer.

**Parameters:**
- `peer_id`: The ID of the peer to set metadata for
- `metadata`: A dictionary of metadata to associate with the peer

### search_peer_messages
Search for messages sent by a peer.

**Parameters:**
- `peer_id`: The ID of the peer to search messages for
- `query`: The search query to use

### chat
Query a peer's representation with natural language questions.

**Parameters:**
- `peer_id`: The ID of the peer to query
- `query`: The natural language question to ask
- `target_peer_id`: Optional target peer ID for local representation queries
- `session_id`: Optional session ID to scope the query to a specific session

### list_peers
Get all peers in the current workspace.

**Parameters:** None

### create_session
Create or get a session with the specified ID and optional configuration.

**Parameters:**
- `session_id`: Unique identifier for the session
- `config`: Optional configuration dictionary for the session

### get_session_metadata
Get metadata for a specific session.

**Parameters:**
- `session_id`: The ID of the session to get metadata for

### set_session_metadata
Set metadata for a specific session.

**Parameters:**
- `session_id`: The ID of the session to set metadata for
- `metadata`: A dictionary of metadata to associate with the session

### add_peers_to_session
Add peers to a session.

**Parameters:**
- `session_id`: The ID of the session to add peers to
- `peer_ids`: List of peer IDs to add to the session

### remove_peers_from_session
Remove peers from a session.

**Parameters:**
- `session_id`: The ID of the session to remove peers from
- `peer_ids`: List of peer IDs to remove from the session

### get_session_peers
Get all peer IDs in a session.

**Parameters:**
- `session_id`: The ID of the session to get peers from

### add_messages_to_session
Add messages to a session.

**Parameters:**
- `session_id`: The ID of the session to add messages to
- `messages`: List of message dictionaries with `peer_id`, `content`, and optional `metadata`

### get_session_messages
Get messages from a session with optional filtering.

**Parameters:**
- `session_id`: The ID of the session to get messages from
- `filters`: Optional dictionary of filter criteria

### get_session_context
Get optimized context for a session within a token limit.

**Parameters:**
- `session_id`: The ID of the session to get context for
- `summary`: Whether to include summary information (default: true)
- `tokens`: Maximum number of tokens to include in the context

### search_session_messages
Search for messages in a specific session.

**Parameters:**
- `session_id`: The ID of the session to search messages in
- `query`: The search query to use

### get_working_representation
Get the current working representation of a peer in a session.

**Parameters:**
- `session_id`: The ID of the session
- `peer_id`: The ID of the peer to get the working representation of
- `target_peer_id`: Optional target peer ID to get the representation of what peer_id knows about target_peer_id

### list_sessions
Get all sessions in the current workspace.

**Parameters:** None

## Contributing or Self Hosting

A Cloudflare Worker that implements the Model Context Protocol (MCP) to provide Honcho functionality as tools for AI assistants like Claude Desktop.

### Deploy MCP Worker

1. **Install dependencies:**
   ```bash
   bun i
   ```

2. **Login to Cloudflare (if not already done):**
   ```bash
   bun wrangler login
   ```

3. **Configure your worker name in `wrangler.toml`:**
   - Update the `name` field to your desired worker name
   - Update the worker names in the `[env.production]` and `[env.staging]` sections

4. **Test locally:**
   ```bash
   bun dev
   ```

5. **Deploy to production:**
   ```bash
   bun run deploy
   ```

### Configuration Options

You can customize the behavior using HTTP headers:

**Available Configuration:**
- `apiKey`: Your Honcho API key
- `baseUrl`: Custom Honcho API base URL (default: https://api.honcho.dev)
- `workspaceId`: Workspace ID (default: "default")
- `userName`: User identifier (default: "User")
- `assistantName`: Assistant identifier (default: "Assistant")

#### Using HTTP Headers:

Pass configuration to mcp-remote via custom headers:

```bash
bunx mcp-remote https://YOUR_WORKER_NAME.YOUR_SUBDOMAIN.workers.dev \
  --header "Authorization:Bearer YOUR_HONCHO_API_KEY" \
  --header "X-Honcho-Workspace-ID:my-workspace" \
  --header "X-Honcho-User-Name:john" \
  --header "X-Honcho-Assistant-Name:Claude" \
  --header "X-Honcho-Base-URL:https://custom.honcho.dev"
```

**Supported Custom Headers:**
- `Authorization: Bearer YOUR_API_KEY` - Your Honcho API key
- `X-Honcho-Base-URL` - Custom Honcho API base URL
- `X-Honcho-Workspace-ID` - Workspace identifier
- `X-Honcho-User-Name` - User identifier
- `X-Honcho-Assistant-Name` - Assistant identifier

### Authentication

The MCP server requires a valid Honcho API key provided via the Authorization header.

### Testing

You can test the MCP server using `mcp-remote` with the local URL:

```bash
bunx mcp-remote http://localhost:8787 --header "Authorization:Bearer your-api-key"
```

### Error Handling

The server provides proper JSON-RPC 2.0 error responses:

- `-32700`: Parse error
- `-32600`: Invalid Request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error

Common issues:
- **Missing API key**: Ensure you provide a valid Honcho API key via header or URL parameter
- **Invalid tool parameters**: Check that required parameters are provided and properly formatted
- **Network errors**: Verify the worker is deployed and accessible