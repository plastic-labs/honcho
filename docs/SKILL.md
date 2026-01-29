---
name: honcho-integration
description: Integrate Honcho memory and social cognition into existing Python or TypeScript codebases. Use when adding Honcho SDK, setting up peers, configuring sessions, or implementing the dialectic chat endpoint for AI agents.
allowed-tools: Read, Glob, Grep, Bash(uv:*), Bash(bun:*), Bash(npm:*), Edit, Write, WebFetch, AskUserQuestion
---

# Honcho Integration Guide

This skill helps you integrate Honcho into existing Python or TypeScript applications. Honcho provides AI-native memory for stateful agents—it uses custom reasoning models to learn continually.

## Integration Workflow

Follow these phases in order:

### Phase 1: Codebase Exploration

Before asking the user anything, explore the codebase to understand:

1. **Language & Framework**: Is this Python or TypeScript? What frameworks are used (FastAPI, Express, Next.js, etc.)?
2. **Existing AI/LLM code**: Search for existing LLM integrations (OpenAI, Anthropic, LangChain, etc.)
3. **Entity structure**: Identify users, agents, bots, or other entities that interact
4. **Session/conversation handling**: How does the app currently manage conversations?
5. **Message flow**: Where are messages sent/received? What's the request/response cycle?

Use Glob and Grep to find:

- `**/*.py` or `**/*.ts` files with "openai", "anthropic", "llm", "chat", "message"
- User/session models or types
- API routes handling chat or conversation endpoints

### Phase 2: Interview (REQUIRED)

After exploring the codebase, use the **AskUserQuestion** tool to clarify integration requirements. Ask these questions (adapt based on what you learned in Phase 1):

#### Question Set 1 - Entities & Peers

Ask about which entities should be Honcho peers:

- header: "Peers"
- question: "Which entities should Honcho track and build representations for?"
- options based on what you found (e.g., "End users only", "Users + AI assistant", "Users + multiple AI agents", "All participants including third-party services")
- Include a follow-up if they have multiple AI agents: should any AI peers be observed?

#### Question Set 2 - Integration Pattern

Ask how they want to use Honcho context:

- header: "Pattern"
- question: "How should your AI access Honcho's user context?"
- options:
  - "Tool call (Recommended)" - "Agent queries Honcho on-demand via function calling"
  - "Pre-fetch" - "Fetch user context before each LLM call with predefined queries"
  - "context()" - "Include conversation history and representations in prompt"
  - "Multiple patterns" - "Combine approaches for different use cases"

#### Question Set 3 - Session Structure

Ask about conversation structure:

- header: "Sessions"
- question: "How should conversations map to Honcho sessions?"
- options based on their app (e.g., "One session per chat thread", "One session per user", "Multiple users per session (group chat)", "Custom session logic")

#### Question Set 4 - Specific Queries (if using pre-fetch pattern)

If they chose pre-fetch, ask what context matters:

- header: "Context"
- question: "What user context should be fetched for the AI?"
- multiSelect: true
- options: "Communication style", "Expertise level", "Goals/priorities", "Preferences", "Recent activity summary", "Custom queries"

### Phase 3: Implementation

Based on interview responses, implement the integration:

1. Install the SDK
2. Create Honcho client initialization
3. Set up peer creation for identified entities
4. Implement the chosen integration pattern(s)
5. Add message storage after exchanges
6. Update any existing conversation handlers

### Phase 4: Verification

- Ensure all message exchanges are stored to Honcho
- Verify AI peers have `observe_me=False` (unless user specifically wants AI observation)
- Check that the workspace ID is consistent across the codebase
- Confirm environment variable for API key is documented

---

## Before You Start

1. **Check the latest SDK versions** at <https://docs.honcho.dev/changelog/introduction>
   - Python SDK: `honcho-ai`
   - TypeScript SDK: `@honcho-ai/sdk`

2. **Get an API key** ask the user to get a Honcho API key from <https://app.honcho.dev> and add it to the environment.

## Installation

### Python (use uv)

```bash
uv add honcho-ai
```

### TypeScript (use bun)

```bash
bun add @honcho-ai/sdk
```

## Core Integration Patterns

### 1. Initialize with a Single Workspace

Use ONE workspace for your entire application. The workspace name should reflect your app/product.

**Python:**

```python
from honcho import Honcho
import os

honcho = Honcho(
    workspace_id="your-app-name",
    api_key=os.environ["HONCHO_API_KEY"],
    environment="production"
)
```

**TypeScript:**

```typescript
import { Honcho } from '@honcho-ai/sdk';

const honcho = new Honcho({
    workspaceId: "your-app-name",
    apiKey: process.env.HONCHO_API_KEY,
    environment: "production"
});
```

### 2. Create Peers for ALL Entities

Create peers for **every entity** in your business logic - users AND AI assistants.

**Python:**

```python
# Human users
user = honcho.peer("user-123")

# AI assistants - set observe_me=False so Honcho doesn't model the AI
assistant = honcho.peer("assistant", config={"observe_me": False})
support_bot = honcho.peer("support-bot", config={"observe_me": False})
```

**TypeScript:**

```typescript
// Human users
const user = await honcho.peer("user-123");

// AI assistants - set observe_me=False
const assistant = await honcho.peer("assistant", { config: { observe_me: false } });
const supportBot = await honcho.peer("support-bot", { config: { observe_me: false } });
```

### 3. Multi-Peer Sessions

Sessions can have multiple participants. Configure observation settings per-peer.

**Python:**

```python
from honcho import SessionPeerConfig

session = honcho.session("conversation-123")

# User is observed (Honcho builds a model of them)
user_config = SessionPeerConfig(observe_me=True, observe_others=True)

# AI is NOT observed (no model built of the AI)
ai_config = SessionPeerConfig(observe_me=False, observe_others=True)

session.add_peers([
    (user, user_config),
    (assistant, ai_config)
])
```

**TypeScript:**

```typescript
const session = await honcho.session("conversation-123");

await session.addPeers([
    [user, { observeMe: true, observeOthers: true }],
    [assistant, { observeMe: false, observeOthers: true }]
]);
```

### 4. Add Messages to Sessions

**Python:**

```python
session.add_messages([
    user.message("I'm having trouble with my account"),
    assistant.message("I'd be happy to help. What seems to be the issue?"),
    user.message("I can't reset my password")
])
```

**TypeScript:**

```typescript
await session.addMessages([
    user.message("I'm having trouble with my account"),
    assistant.message("I'd be happy to help. What seems to be the issue?"),
    user.message("I can't reset my password")
]);
```

## Using Honcho for AI Agents

### Pattern A: Dialectic Chat as a Tool Call (Recommended for Agents)

Make Honcho's chat endpoint available as a **tool** for your AI agent. This lets the agent query user context on-demand.

**Python (OpenAI function calling):**

```python
import openai
from honcho import Honcho

honcho = Honcho(workspace_id="my-app", api_key=os.environ["HONCHO_API_KEY"])

# Define the tool for your agent
honcho_tool = {
    "type": "function",
    "function": {
        "name": "query_user_context",
        "description": "Query Honcho to retrieve relevant context about the user based on their history and preferences. Use this when you need to understand the user's background, preferences, past interactions, or goals.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language question about the user, e.g. 'What are this user's main goals?' or 'What communication style does this user prefer?'"
                }
            },
            "required": ["query"]
        }
    }
}

def handle_honcho_tool_call(user_id: str, query: str) -> str:
    """Execute the Honcho chat tool call."""
    peer = honcho.peer(user_id)
    return peer.chat(query)

# Use in your agent loop
def run_agent(user_id: str, user_message: str):
    messages = [{"role": "user", "content": user_message}]

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=[honcho_tool]
    )

    # Handle tool calls
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.function.name == "query_user_context":
                import json
                args = json.loads(tool_call.function.arguments)
                result = handle_honcho_tool_call(user_id, args["query"])
                # Continue conversation with tool result...
```

**TypeScript (OpenAI function calling):**

```typescript
import OpenAI from 'openai';
import { Honcho } from '@honcho-ai/sdk';

const honcho = new Honcho({
    workspaceId: "my-app",
    apiKey: process.env.HONCHO_API_KEY
});

const honchoTool: OpenAI.ChatCompletionTool = {
    type: "function",
    function: {
        name: "query_user_context",
        description: "Query Honcho to retrieve relevant context about the user based on their history and preferences.",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "A natural language question about the user"
                }
            },
            required: ["query"]
        }
    }
};

async function handleHonchoToolCall(userId: string, query: string): Promise<string> {
    const peer = await honcho.peer(userId);
    return await peer.chat(query);
}
```

### Pattern B: Pre-fetch Context with Targeted Queries

For simpler integrations, fetch user context before the LLM call using pre-defined queries.

**Python:**

```python
def get_user_context_for_prompt(user_id: str) -> dict:
    """Fetch key user attributes via targeted Honcho queries."""
    peer = honcho.peer(user_id)

    return {
        "communication_style": peer.chat("What communication style does this user prefer? Be concise."),
        "expertise_level": peer.chat("What is this user's technical expertise level? Be concise."),
        "current_goals": peer.chat("What are this user's current goals or priorities? Be concise."),
        "preferences": peer.chat("What key preferences should I know about this user? Be concise.")
    }

def build_system_prompt(user_context: dict) -> str:
    return f"""You are a helpful assistant. Here's what you know about this user:

Communication style: {user_context['communication_style']}
Expertise level: {user_context['expertise_level']}
Current goals: {user_context['current_goals']}
Key preferences: {user_context['preferences']}

Tailor your responses accordingly."""
```

**TypeScript:**

```typescript
async function getUserContextForPrompt(userId: string): Promise<Record<string, string>> {
    const peer = await honcho.peer(userId);

    const [style, expertise, goals, preferences] = await Promise.all([
        peer.chat("What communication style does this user prefer? Be concise."),
        peer.chat("What is this user's technical expertise level? Be concise."),
        peer.chat("What are this user's current goals or priorities? Be concise."),
        peer.chat("What key preferences should I know about this user? Be concise.")
    ]);

    return {
        communicationStyle: style,
        expertiseLevel: expertise,
        currentGoals: goals,
        preferences: preferences
    };
}
```

### Pattern C: Get Context for LLM Integration

Use `context()` for conversation history with built-in LLM formatting.

**Python:**

```python
import openai

session = honcho.session("conversation-123")
user = honcho.peer("user-123")
assistant = honcho.peer("assistant", config={"observe_me": False})

# Get context formatted for your LLM
context = session.context(
    tokens=2000,
    peer_target=user.id,  # Include representation of this user
    summary=True           # Include conversation summaries
)

# Convert to OpenAI format
messages = context.to_openai(assistant=assistant)

# Or Anthropic format
# messages = context.to_anthropic(assistant=assistant)

# Add the new user message
messages.append({"role": "user", "content": "What should I focus on today?"})

response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages
)

# Store the exchange
session.add_messages([
    user.message("What should I focus on today?"),
    assistant.message(response.choices[0].message.content)
])
```

## Streaming Responses

**Python:**

```python
response_stream = peer.chat("What do we know about this user?", stream=True)

for chunk in response_stream.iter_text():
    print(chunk, end="", flush=True)
```

## Integration Checklist

When integrating Honcho into an existing codebase:

- [ ] Install SDK with `uv add honcho-ai` (Python) or `bun add @honcho-ai/sdk` (TypeScript)
- [ ] Set up `HONCHO_API_KEY` environment variable
- [ ] Initialize Honcho client with a single workspace ID
- [ ] Create peers for all entities (users AND AI assistants)
- [ ] Set `observe_me=False` for AI peers
- [ ] Configure sessions with appropriate peer observation settings
- [ ] Choose integration pattern:
  - [ ] Tool call pattern for agentic systems
  - [ ] Pre-fetch pattern for simpler integrations
  - [ ] context() for conversation history
- [ ] Store messages after each exchange to build user models

## Common Mistakes to Avoid

1. **Multiple workspaces**: Use ONE workspace per application
2. **Forgetting AI peers**: Create peers for AI assistants, not just users
3. **Observing AI peers**: Set `observe_me=False` for AI peers unless you specifically want Honcho to model your AI's behavior
4. **Not storing messages**: Always call `add_messages()` to feed Honcho's reasoning engine
5. **Blocking on processing**: Messages are processed asynchronously; use `get_deriver_status()` if you need to wait

## Resources

- Documentation: <https://docs.honcho.dev>
- Latest SDK versions: <https://docs.honcho.dev/changelog/introduction>
- API Reference: <https://docs.honcho.dev/v3/api-reference/introduction>
