# Honcho MCP Server


## Use the Hosted MCP Server

Go to https://app.honcho.dev and get an API key. Then go to Claude Desktop and navigate to custom MCP servers.

If you don't have node/npm/npx install you will need to do that. Claude Desktop or Claude Code can help!

Add or integrate this into your claude desktop config:
```json
{
  "mcpServers": {
    "honcho": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.honcho.dev",
        "--header",
        "Authorization:${AUTH_HEADER}"
      ],
      "env": {
        "AUTH_HEADER": "Bearer <your-honcho-key>"
      }
    }
  }
}
```

Alternatively you may customize your username, assistant name, and/or workspace ID. All are optional.

```json
{
  "mcpServers": {
    "honcho": {
      "command": "npx",
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

## Run the MCP Server

A Cloudflare Worker that implements the Model Context Protocol (MCP) to provide Honcho functionality as tools for AI assistants like Claude Desktop.

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Login to Cloudflare (if not already done):**
   ```bash
   npx wrangler login
   ```

3. **Configure your worker name in `wrangler.toml`:**
   - Update the `name` field to your desired worker name
   - Update the worker names in the `[env.production]` and `[env.staging]` sections

4. **Test locally:**
   ```bash
   npm run dev
   ```

5. **Deploy to production:**
   ```bash
   npm run deploy
   ```

## Using with Claude Desktop

After deployment, add this MCP server to your Claude Desktop configuration:

1. Open Claude Desktop settings
2. Go to the "Model Context Protocol" section
3. Add a new server with the command:
   ```bash
   npx mcp-remote https://YOUR_WORKER_NAME.YOUR_SUBDOMAIN.workers.dev --header "Authorization:Bearer YOUR_HONCHO_API_KEY"
   ```

   Or with optional configuration using custom headers:
   ```bash
   npx mcp-remote https://YOUR_WORKER_NAME.YOUR_SUBDOMAIN.workers.dev \
     --header "Authorization:Bearer YOUR_HONCHO_API_KEY" \
     --header "X-Honcho-Workspace-ID:my-workspace" \
     --header "X-Honcho-User-Name:john" \
     --header "X-Honcho-Assistant-Name:Claude"
   ```

   **For all available configuration options and custom headers, see the [Configuration Options](#configuration-options) section below.**

### Configuration Options

You can customize the behavior using HTTP headers:

**Available Configuration:**
- `apiKey`: Your Honcho API key
- `baseUrl`: Custom Honcho API base URL (default: https://api.honcho.dev)
- `workspaceId`: Workspace ID (default: "default")
- `userName`: User identifier (default: "User")
- `assistantName`: Assistant identifier (default: "Assistant")

#### Using HTTP Headers:

Pass configuration via custom headers:

```bash
npx mcp-remote https://YOUR_WORKER_NAME.YOUR_SUBDOMAIN.workers.dev \
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

## Available Tools

Once connected, Claude will have access to these Honcho tools:

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
- `query`: The question about the user's preferences, habits, etc.

**Example queries:**
- "What does this message reveal about the user's communication preferences?"
- "How formal or casual should I be with the user based on our history?"
- "What emotional state might the user be in right now?"

## Authentication

The MCP server requires a valid Honcho API key provided via the Authorization header:

```bash
npx mcp-remote https://your-worker.workers.dev --header "Authorization:Bearer YOUR_HONCHO_API_KEY"
```

All configuration options are passed via custom headers - see the Configuration Options section above for full details.

## Protocol Details

This worker implements the MCP (Model Context Protocol) specification using JSON-RPC 2.0. It supports:

- `initialize` - MCP initialization handshake
- `tools/list` - List available Honcho tools
- `tools/call` - Execute Honcho tools

All communication follows the JSON-RPC 2.0 format with proper error handling and response structure.

## Development

### Local Development

1. Install dependencies: `npm install`
2. Start the development server: `npm run dev`
3. The worker will be available at `http://localhost:8787`

### Testing with MCP

You can test the MCP server using `mcp-remote` with the local URL:

```bash
npx mcp-remote http://localhost:8787 --header "Authorization:Bearer your-api-key"
```

## Error Handling

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