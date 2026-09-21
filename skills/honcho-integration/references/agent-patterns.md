# Agent Access Patterns

How your AI accesses Honcho's user context. Pick based on the interview answer to Question Set 2. These build on the client/peer/session setup in `core-patterns.md`.

- **Pattern A — Dialectic chat as a tool call**: the agent decides when to query context on-demand.
- **Pattern B — Pre-fetch with targeted queries**: fetch a fixed set of attributes before each LLM call.
- **Pattern C — `context()` for LLM integration**: inject conversation history + representation into the prompt.

> **Speed note.** `chat()` runs live dialectic reasoning (a few seconds) — Patterns A and B call it. `context()` (Pattern C) is a near-instant read. Prefer `context()` for per-turn grounding; reach for `chat()` when you genuinely need a reasoned answer.

## Pattern A: Dialectic Chat as a Tool Call

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

## Pattern B: Pre-fetch Context with Targeted Queries

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

## Pattern C: Get Context for LLM Integration

Use `context()` for conversation history with built-in LLM formatting. This is a near-instant read so it's the cheapest way to ground each turn.

**Python:**

```python
import openai

session = honcho.session("conversation-123")
user = honcho.peer("user-123")
assistant = honcho.peer("assistant")

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

**TypeScript:**

```typescript
import OpenAI from 'openai';

const session = await honcho.session("conversation-123");
const user = await honcho.peer("user-123");
const assistant = await honcho.peer("assistant");

// Get context formatted for your LLM
const context = await session.context({
    tokens: 2000,
    peerTarget: user.id,  // Include representation of this user
    summary: true          // Include conversation summaries
});

// Convert to OpenAI format
const messages = context.toOpenAI(assistant);

// Or Anthropic format
// const messages = context.toAnthropic(assistant);

// Add the new user message
messages.push({ role: "user", content: "What should I focus on today?" });

const openai = new OpenAI();
const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages
});

// Store the exchange
await session.addMessages([
    user.message("What should I focus on today?"),
    assistant.message(response.choices[0].message.content!)
]);
```

### What `context()` returns

`session.context()` bundles the session-local view you can drop straight into an LLM call:

- **Recent messages** from the session, trimmed to the `tokens` budget.
- **Conversation summaries** when `summary=True` — the two-tier short/long summaries so older turns still count without spending the full token budget.
- **The target peer's representation** when you pass `peer_target` — Honcho's synthesized understanding of that user, folded in. Omit it and you get session-local context only (no cross-session memory).

The `to_openai()` / `to_anthropic()` helpers format all of that as a `messages` array for the respective provider. Pass your `assistant` peer so its turns are tagged as the assistant role.

## Streaming Responses

```python
stream = peer.chat_stream("What do we know about this user?")

for chunk in stream:
    print(chunk, end="", flush=True)
```

```typescript
const stream = await peer.chatStream("What do we know about this user?");

for await (const chunk of stream) {
    process.stdout.write(chunk);
}
```
